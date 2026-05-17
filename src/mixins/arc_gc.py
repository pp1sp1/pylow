"""Atomic Reference Counting (ARC) and Cycle Collector runtime."""

from __future__ import annotations

import ast
import sys
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

import llvmlite.ir as ir
import llvmlite.binding as llvm

from ..types import (
    Tag, PyType, EXCEPTION_TYPES,
    I8, I32, I64, F64, I1, I8P, VOID,
    GC_HEADER_TY, BOXED_TY, BOXED_PTR,
    LIST_TY, LIST_PTR, ENTRY_TY, ENTRY_PTR,
    DICT_TY, DICT_PTR, STR_TY, STR_PTR,
    CLASS_TY, CLASS_PTR, INSTANCE_TY, INSTANCE_PTR,
    ITER_DATA_TY, ITER_DATA_PTR,
    SZ_GC_HEADER, SZ_BOXED, SZ_ENTRY, SZ_LIST, SZ_DICT, SZ_STR, SZ_ITER,
    pytype_to_llvm,
)
from ..exceptions import CompileError
from ..symbols import VarInfo, SymbolTable
from ..values import Value
from ..type_analyzer import StaticTypeAnalyzer

if TYPE_CHECKING:
    pass


class ArcGcMixin:
    """Atomic Reference Counting (ARC) and Cycle Collector runtime."""

    def _get_or_create_arc_funcs(self):
        if "__py2llvm_incref" in self.functions:
            return

        old_b, old_f = self.builder, self.current_func
        z = ir.Constant(I32, 0)

        # ── forward declarations (allowing cyclic calls) ──
        fty_void_ptr = ir.FunctionType(VOID, [I8P])
        f_incref = ir.Function(self.module, fty_void_ptr, "__py2llvm_incref")
        f_incref.linkage = ""
        f_incref.attributes.add("alwaysinline")
        f_incref.attributes.add("nounwind")

        f_decref = ir.Function(self.module, fty_void_ptr, "__py2llvm_decref")
        f_decref.attributes.add("alwaysinline")
        f_decref.attributes.add("nounwind")

        f_dealloc = ir.Function(self.module, fty_void_ptr, "__py2llvm_dealloc")
        f_dealloc.attributes.add("alwaysinline")
        f_dealloc.attributes.add("nounwind")

        f_cc_suspect = ir.Function(self.module, fty_void_ptr, "__py2llvm_cc_suspect")
        f_cc_suspect.attributes.add("alwaysinline")
        f_cc_suspect.attributes.add("nounwind")

        self.functions["__py2llvm_incref"] = f_incref
        self.functions["__py2llvm_decref"] = f_decref
        self.functions["__py2llvm_dealloc"] = f_dealloc
        self.functions["__py2llvm_cc_suspect"] = f_cc_suspect

        # Build GC functions (traverse, dec_temp, gc_step)
        self._build_gc_functions()

        # ── INCREF: atomic refcnt++ ──
        self.current_func = f_incref
        self.builder = ir.IRBuilder(f_incref.append_basic_block("entry"))
        ptr_inc = f_incref.args[0]
        gc_ptr = self.builder.bitcast(ptr_inc, ir.PointerType(GC_HEADER_TY))
        refcnt_ptr_inc = self.builder.gep(
            gc_ptr, [z, ir.Constant(I32, 0)], inbounds=True
        )
        self.builder.atomic_rmw("add", refcnt_ptr_inc, ir.Constant(I64, 1), "monotonic")
        self.builder.ret_void()

        # ── DECREF: atomic refcnt-- → if 0: dealloc, else: cc_suspect ──
        self.current_func = f_decref
        self.builder = ir.IRBuilder(f_decref.append_basic_block("entry"))
        ptr_dec = f_decref.args[0]

        is_null_dec = self.builder.icmp_signed(
            "==", self.builder.ptrtoint(ptr_dec, I64), ir.Constant(I64, 0)
        )
        do_work_bb = f_decref.append_basic_block("dec.work")
        merge_null = f_decref.append_basic_block("dec.null_merge")
        self.builder.cbranch(is_null_dec, merge_null, do_work_bb)

        self.builder.position_at_end(do_work_bb)
        gc_ptr_dec = self.builder.bitcast(ptr_dec, ir.PointerType(GC_HEADER_TY))
        refcnt_ptr_dec = self.builder.gep(
            gc_ptr_dec, [z, ir.Constant(I32, 0)], inbounds=True
        )
        old_val = self.builder.atomic_rmw(
            "sub", refcnt_ptr_dec, ir.Constant(I64, 1), "acq_rel"
        )
        is_zero = self.builder.icmp_signed("==", old_val, ir.Constant(I64, 1))
        dealloc_bb = f_decref.append_basic_block("dec.dealloc")
        cc_bb = f_decref.append_basic_block("dec.cc")
        self.builder.cbranch(is_zero, dealloc_bb, cc_bb)

        self.builder.position_at_end(dealloc_bb)
        self.builder.call(f_dealloc, [ptr_dec])
        self.builder.branch(merge_null)

        self.builder.position_at_end(cc_bb)
        self.builder.call(f_cc_suspect, [ptr_dec])
        # Call gc_step to process suspect objects
        if "__py2llvm_gc_step" in self.functions:
            gc_step_fn = self.functions["__py2llvm_gc_step"]
            self.builder.call(gc_step_fn, [])
        self.builder.branch(merge_null)

        self.builder.position_at_end(merge_null)
        self.builder.ret_void()

        # ── DEALLOC: free children then free(obj) ──
        # We need to know the type tag to know how to traverse children.
        # We store the tag inside the GC header area: the color field (index 1 of GC_HEADER_TY)
        # can double as a type-tag storage for deallocation purposes.
        # Actually, we'll read it from the object's own type tag field.
        # For BOXED: tag is at GC_HEADER(0..2) + 0 = index 0 of struct → gep index 1
        # For LIST:  tag is implicit = Tag.LIST
        # For DICT:  tag is implicit = Tag.DICT
        # For STR:   tag is implicit = Tag.STR
        #
        # We'll use the simplest approach: store type_tag in gc_next field (index 2 of GC_HEADER_TY).
        # Actually, let's use a simpler approach: the decref caller knows the type.
        # We'll generate typed dealloc helpers.

        # __py2llvm_dealloc_boxed(ptr)
        self._build_dealloc_boxed(f_dealloc)

        self.current_func = old_f
        self.builder = old_b

        # ── CYCLE COLLECTOR: __py2llvm_cc_suspect ──
        self._build_cc_suspect(f_cc_suspect)

    def _build_dealloc_boxed(self, f_dealloc):
        """Deallocates a BOXED object. For other types, typed dealloc are built on-demand."""
        old_b, old_f = self.builder, self.current_func
        z = ir.Constant(I32, 0)

        self.current_func = f_dealloc
        bb = f_dealloc.append_basic_block("entry")
        self.builder = ir.IRBuilder(bb)
        ptr = f_dealloc.args[0]

        # Check if object is static (color == -1) -> skip deallocation
        gc_ptr = self.builder.bitcast(ptr, ir.PointerType(GC_HEADER_TY))
        color_ptr = self.builder.gep(gc_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
        color = self.builder.load(color_ptr)
        is_static = self.builder.icmp_signed("==", color, ir.Constant(I32, -1))
        static_bb = f_dealloc.append_basic_block("dealloc.static")
        nonstatic_bb = f_dealloc.append_basic_block("dealloc.nonstatic")
        self.builder.cbranch(is_static, static_bb, nonstatic_bb)

        self.builder.position_at_end(static_bb)
        self.builder.ret_void()

        self.builder.position_at_end(nonstatic_bb)

        # Read tag from the boxed object: BOXED_TY = { GC_HEADER, tag, payload }
        boxed_ptr = self.builder.bitcast(ptr, BOXED_PTR)
        tag_ptr = self.builder.gep(boxed_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
        tag = self.builder.load(tag_ptr, "d_tag")
        pay_ptr = self.builder.gep(boxed_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
        pay = self.builder.load(pay_ptr, "d_pay")

        # If tag == LIST or DICT, we need to decref children
        is_lst = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST))
        is_dct = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.DICT))
        is_str = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.STR))

        lst_bb = self.current_func.append_basic_block("dealloc.list")
        dct_bb = self.current_func.append_basic_block("dealloc.dict")
        str_bb = self.current_func.append_basic_block("dealloc.str")
        free_bb = self.current_func.append_basic_block("dealloc.free")

        sw = self.builder.switch(tag, free_bb)
        sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)
        sw.add_case(ir.Constant(I64, Tag.TUPLE), lst_bb)  # TUPLE ma tę samą strukturę co LIST
        sw.add_case(ir.Constant(I64, Tag.SET), lst_bb)  # SET ma tę samą strukturę co LIST
        sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
        sw.add_case(ir.Constant(I64, Tag.STR), str_bb)

        # LIST deallocation: decref each element, then free data and list
        self.builder.position_at_end(lst_bb)
        lptr = self.builder.inttoptr(pay, LIST_PTR)
        sp, _, dp = self._list_ptrs(lptr)
        lsize = self.builder.load(sp, "dl_sz")
        ldata = self.builder.load(dp, "dl_data")

        li_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), li_a)
        dl_cond = self.current_func.append_basic_block("dl.cond")
        dl_body = self.current_func.append_basic_block("dl.body")
        dl_done = self.current_func.append_basic_block("dl.done")
        self.builder.branch(dl_cond)

        self.builder.position_at_end(dl_cond)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(li_a), lsize),
            dl_body,
            dl_done,
        )

        self.builder.position_at_end(dl_body)
        slot = self.builder.gep(ldata, [self.builder.load(li_a)], inbounds=True)
        etag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        epay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        # decref if it's a heap object (STR, LIST, TUPLE, SET, DICT)
        is_heap = self.builder.or_(
            self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.STR)),
            self.builder.or_(
                self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.LIST)),
                self.builder.or_(
                    self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.TUPLE)),
                    self.builder.or_(
                        self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.SET)),
                        self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.DICT)),
                    ),
                ),
            ),
        )
        dl_dec = self.current_func.append_basic_block("dl.dec")
        dl_skip = self.current_func.append_basic_block("dl.skip")
        self.builder.cbranch(is_heap, dl_dec, dl_skip)
        self.builder.position_at_end(dl_dec)
        child_ptr = self.builder.inttoptr(epay, I8P)
        f_decref = self.functions["__py2llvm_decref"]
        self.builder.call(f_decref, [child_ptr])
        self.builder.branch(dl_skip)
        self.builder.position_at_end(dl_skip)
        self.builder.store(
            self.builder.add(self.builder.load(li_a), ir.Constant(I64, 1)), li_a
        )
        self.builder.branch(dl_cond)

        self.builder.position_at_end(dl_done)
        # free data array
        self.builder.call(self._free, [self.builder.bitcast(ldata, I8P)])
        # free the LIST object itself
        self.builder.call(self._free, [self.builder.bitcast(lptr, I8P)])
        self.builder.branch(free_bb)

        # DICT deallocation: decref key/value for each entry
        self.builder.position_at_end(dct_bb)
        dptr = self.builder.inttoptr(pay, DICT_PTR)
        d_cap_d = self.builder.load(
            self.builder.gep(dptr, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        d_ents_d = self.builder.load(
            self.builder.gep(dptr, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        di_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), di_a)
        dd_cond = self.current_func.append_basic_block("dd.cond")
        dd_body = self.current_func.append_basic_block("dd.body")
        dd_done = self.current_func.append_basic_block("dd.done")
        dd_skip = self.current_func.append_basic_block("dd.skip")
        dd_dec = self.current_func.append_basic_block("dd.dec")
        self.builder.branch(dd_cond)

        self.builder.position_at_end(dd_cond)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(di_a), d_cap_d),
            dd_body,
            dd_done,
        )

        self.builder.position_at_end(dd_body)
        ent_d = self.builder.gep(d_ents_d, [self.builder.load(di_a)], inbounds=True)
        ktag = self.builder.load(
            self.builder.gep(ent_d, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        empty_slot = self.builder.icmp_signed("==", ktag, ir.Constant(I64, -1))
        self.builder.cbranch(empty_slot, dd_skip, dd_dec)

        self.builder.position_at_end(dd_dec)
        kpay = self.builder.load(
            self.builder.gep(ent_d, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        vtag = self.builder.load(
            self.builder.gep(ent_d, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        vpay = self.builder.load(
            self.builder.gep(ent_d, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        # decref key if heap
        k_is_heap = self.builder.or_(
            self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.STR)),
            self.builder.or_(
                self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.LIST)),
                self.builder.or_(
                    self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.TUPLE)),
                    self.builder.or_(
                        self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.SET)),
                        self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.DICT)),
                    ),
                ),
            ),
        )
        dd_kdec = self.current_func.append_basic_block("dd.kdec")
        dd_kskip = self.current_func.append_basic_block("dd.kskip")
        self.builder.cbranch(k_is_heap, dd_kdec, dd_kskip)
        self.builder.position_at_end(dd_kdec)
        self.builder.call(
            self.functions["__py2llvm_decref"], [self.builder.inttoptr(kpay, I8P)]
        )
        self.builder.branch(dd_kskip)
        self.builder.position_at_end(dd_kskip)
        # decref value if heap
        v_is_heap = self.builder.or_(
            self.builder.icmp_signed("==", vtag, ir.Constant(I64, Tag.STR)),
            self.builder.or_(
                self.builder.icmp_signed("==", vtag, ir.Constant(I64, Tag.LIST)),
                self.builder.or_(
                    self.builder.icmp_signed("==", vtag, ir.Constant(I64, Tag.TUPLE)),
                    self.builder.or_(
                        self.builder.icmp_signed("==", vtag, ir.Constant(I64, Tag.SET)),
                        self.builder.icmp_signed("==", vtag, ir.Constant(I64, Tag.DICT)),
                    ),
                ),
            ),
        )
        dd_vdec = self.current_func.append_basic_block("dd.vdec")
        dd_vskip = self.current_func.append_basic_block("dd.vskip")
        self.builder.cbranch(v_is_heap, dd_vdec, dd_vskip)
        self.builder.position_at_end(dd_vdec)
        self.builder.call(
            self.functions["__py2llvm_decref"], [self.builder.inttoptr(vpay, I8P)]
        )
        self.builder.branch(dd_vskip)
        self.builder.position_at_end(dd_vskip)
        self.builder.branch(dd_skip)

        self.builder.position_at_end(dd_skip)
        self.builder.store(
            self.builder.add(self.builder.load(di_a), ir.Constant(I64, 1)), di_a
        )
        self.builder.branch(dd_cond)

        self.builder.position_at_end(dd_done)
        self.builder.call(self._free, [self.builder.bitcast(d_ents_d, I8P)])
        # free the DICT object itself
        self.builder.call(self._free, [self.builder.bitcast(dptr, I8P)])
        self.builder.branch(free_bb)

        # STR deallocation: only free if NOT static (color != -1)
        self.builder.position_at_end(str_bb)
        sobj = self.builder.inttoptr(pay, STR_PTR)

        # Load color from GC_HEADER (index 1 of GC_HEADER_TY)
        color = self.builder.load(
            self.builder.gep(sobj, [z, z, ir.Constant(I32, 1)], inbounds=True)
        )

        is_static = self.builder.icmp_signed("==", color, ir.Constant(I32, -1))
        free_data_bb = self.current_func.append_basic_block("dealloc.str.free_data")
        skip_all_bb = self.current_func.append_basic_block("dealloc.str.skip_all")

        self.builder.cbranch(is_static, skip_all_bb, free_data_bb)

        self.builder.position_at_end(free_data_bb)
        sdata = self.builder.load(
            self.builder.gep(sobj, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        self.builder.call(self._free, [sdata])
        self.builder.call(self._free, [self.builder.bitcast(sobj, I8P)])
        self.builder.branch(free_bb)

        self.builder.position_at_end(skip_all_bb)
        # Static string object lives in global data segment – do NOT free it.
        # But the boxed wrapper (ptr) was malloc'd and must be freed.
        self.builder.branch(free_bb)

        # Free the object itself
        self.builder.position_at_end(free_bb)
        self.builder.call(self._free, [ptr])
        self.builder.ret_void()

        self.current_func = old_f
        self.builder = old_b

    def _setup_gc_globals(self):
        """Tworzy globalne zmienne GC: stan, roots, worklist, itp."""
        if "__gc_state" not in self.module.globals:
            # 0=off, 1=on (collecting)
            gc_state = ir.GlobalVariable(self.module, I32, "__gc_state")
            gc_state.initializer = ir.Constant(I32, 0)
            gc_state.linkage = "common"
            self.module.globals["__gc_state"] = gc_state

        if "__gc_roots" not in self.module.globals:
            # Prosta tablica roots (w produkcji: TLS lub stack map)
            roots_ty = ir.ArrayType(I8P, 256)
            gc_roots = ir.GlobalVariable(self.module, roots_ty, "__gc_roots")
            gc_roots.initializer = ir.Constant(roots_ty, [ir.Constant(I8P, None)] * 256)
            gc_roots.linkage = "common"
            self.module.globals["__gc_roots"] = gc_roots

        if "__gc_worklist" not in self.module.globals:
            worklist_ty = ir.ArrayType(I8P, 1024)
            gc_worklist = ir.GlobalVariable(self.module, worklist_ty, "__gc_worklist")
            gc_worklist.initializer = ir.Constant(
                worklist_ty, [ir.Constant(I8P, None)] * 1024
            )
            gc_worklist.linkage = "common"
            self.module.globals["__gc_worklist"] = gc_worklist

        if "__gc_worklist_size" not in self.module.globals:
            gc_wl_size = ir.GlobalVariable(self.module, I64, "__gc_worklist_size")
            gc_wl_size.initializer = ir.Constant(I64, 0)
            gc_wl_size.linkage = "common"
            self.module.globals["__gc_worklist_size"] = gc_wl_size

    def _build_cc_suspect(self, f_cc_suspect):
        """Cycle Collector – oznaczanie obiektów jako 'suspect' (PURPLE)."""
        old_b, old_f = self.builder, self.current_func
        z = ir.Constant(I32, 0)

        self.current_func = f_cc_suspect
        entry_bb = f_cc_suspect.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry_bb)
        ptr = f_cc_suspect.args[0]

        # Check for null pointer
        is_null_cc = self.builder.icmp_signed(
            "==", self.builder.ptrtoint(ptr, I64), ir.Constant(I64, 0)
        )
        cc_merge = f_cc_suspect.append_basic_block("cc.merge")
        work_bb = f_cc_suspect.append_basic_block("cc.work")
        self.builder.cbranch(is_null_cc, cc_merge, work_bb)

        # Work block - process the object
        self.builder.position_at_end(work_bb)
        gc_ptr_cc = self.builder.bitcast(ptr, ir.PointerType(GC_HEADER_TY))

        # Check current color
        color_ptr = self.builder.gep(gc_ptr_cc, [z, ir.Constant(I32, 1)], inbounds=True)
        current_color = self.builder.load(color_ptr, "cc_color")

        # If already PURPLE (3), skip to merge
        is_purple = self.builder.icmp_signed("==", current_color, ir.Constant(I32, 3))
        skip_bb = f_cc_suspect.append_basic_block("cc.skip")
        self.builder.cbranch(
            is_purple, skip_bb, skip_bb
        )  # If purple, skip; otherwise fall through

        # Mark as PURPLE (3) - object is suspect
        self.builder.position_at_end(skip_bb)
        self.builder.store(ir.Constant(I32, 3), color_ptr)

        # Add to suspect buffer
        if "__cc_suspect_buf" not in self.module.globals:
            buf_ty = ir.ArrayType(I8P, 1024)
            buf = ir.GlobalVariable(self.module, buf_ty, "__cc_suspect_buf")
            buf.initializer = ir.Constant(buf_ty, [ir.Constant(I8P, None)] * 1024)
            buf.linkage = "common"
            self.module.globals["__cc_suspect_buf"] = buf

        if "__cc_suspect_count" not in self.module.globals:
            cnt = ir.GlobalVariable(self.module, I64, "__cc_suspect_count")
            cnt.initializer = ir.Constant(I64, 0)
            cnt.linkage = "common"
            self.module.globals["__cc_suspect_count"] = cnt

        buf = self.module.globals["__cc_suspect_buf"]
        cnt = self.module.globals["__cc_suspect_count"]

        idx = self.builder.load(cnt, "cc_idx")
        below_limit = self.builder.icmp_signed("<", idx, ir.Constant(I64, 1024))

        add_bb = f_cc_suspect.append_basic_block("cc.add")
        skip_add_bb = f_cc_suspect.append_basic_block("cc.skip_add")
        self.builder.branch(add_bb)  # Always try to add

        self.builder.position_at_end(add_bb)
        elem_ptr = self.builder.gep(buf, [z, idx], inbounds=True)
        self.builder.store(ptr, elem_ptr)
        new_idx = self.builder.add(idx, ir.Constant(I64, 1))
        self.builder.store(new_idx, cnt)

        # If buffer is now full, call gc_step
        is_full = self.builder.icmp_signed("==", new_idx, ir.Constant(I64, 1024))
        gc_bb = f_cc_suspect.append_basic_block("cc.gc_step")
        self.builder.cbranch(is_full, gc_bb, skip_add_bb)

        self.builder.position_at_end(gc_bb)
        gc_step_fn = self.functions["__py2llvm_gc_step"]
        self.builder.call(gc_step_fn, [])
        self.builder.branch(skip_add_bb)

        self.builder.position_at_end(skip_add_bb)
        self.builder.branch(cc_merge)

        # Merge block
        self.builder.position_at_end(cc_merge)
        self.builder.ret_void()

        self.current_func = old_f
        self.builder = old_b

    def _build_gc_functions(self):
        """Buduje pełny zestaw funkcji Cycle Collectora."""
        self._setup_gc_globals()

        # Forward declarations for mutual recursion
        fty_void_ptr = ir.FunctionType(VOID, [I8P])
        if "__py2llvm_traverse" not in self.functions:
            fn_t = ir.Function(self.module, fty_void_ptr, "__py2llvm_traverse")
            self.functions["__py2llvm_traverse"] = fn_t
        if "__py2llvm_dec_temp" not in self.functions:
            fn_d = ir.Function(self.module, fty_void_ptr, "__py2llvm_dec_temp")
            self.functions["__py2llvm_dec_temp"] = fn_d
        if "__py2llvm_gc_step" not in self.functions:
            fn_g = ir.Function(
                self.module, ir.FunctionType(VOID, []), "__py2llvm_gc_step"
            )
            self.functions["__py2llvm_gc_step"] = fn_g

        # Build function bodies
        self._build_traverse()
        self._build_dec_temp()
        self._build_gc_step()

    def _build_traverse(self):
        """Traverse: przejście po obiekcie i dekrementacja temp_refcnt."""
        fn_name = "__py2llvm_traverse"
        if fn_name in self.functions:
            fn = self.functions[fn_name]
        else:
            fty = ir.FunctionType(VOID, [I8P])
            fn = ir.Function(self.module, fty, fn_name)
            self.functions[fn_name] = fn

        old_b, old_f = self.builder, self.current_func
        self.current_func = fn
        self.builder = ir.IRBuilder(fn.append_basic_block("entry"))
        ptr = fn.args[0]
        z = ir.Constant(I32, 0)

        # Check for null
        is_null = self.builder.icmp_signed(
            "==", self.builder.ptrtoint(ptr, I64), ir.Constant(I64, 0)
        )
        null_bb = fn.append_basic_block("traverse.null")
        work_bb = fn.append_basic_block("traverse.work")
        self.builder.cbranch(is_null, null_bb, work_bb)

        self.builder.position_at_end(null_bb)
        self.builder.ret_void()

        self.builder.position_at_end(work_bb)
        # Assume it's a BOXED_PTR - try to read tag
        boxed_ptr = self.builder.bitcast(ptr, BOXED_PTR)
        tag_ptr = self.builder.gep(boxed_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
        tag = self.builder.load(tag_ptr, "traverse_tag")

        # Switch on tag
        end_bb = fn.append_basic_block("traverse.end")
        lst_bb = fn.append_basic_block("traverse.list")
        dct_bb = fn.append_basic_block("traverse.dict")
        str_bb = fn.append_basic_block("traverse.str")
        other_bb = fn.append_basic_block("traverse.other")

        sw = self.builder.switch(tag, other_bb)
        sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)
        sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
        sw.add_case(ir.Constant(I64, Tag.STR), str_bb)

        # Traverse LIST
        self.builder.position_at_end(lst_bb)
        pay_ptr = self.builder.gep(boxed_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
        lst_payload = self.builder.load(pay_ptr, "lst_payload")
        lst_ptr = self.builder.inttoptr(lst_payload, LIST_PTR)
        sp, cp, dp = self._list_ptrs(lst_ptr)
        lsize = self.builder.load(sp, "lst_size")
        ldata = self.builder.load(dp, "lst_data")

        # Loop through list elements and dec_temp
        idx_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), idx_a)
        loop_cond = fn.append_basic_block("t.lst.cond")
        loop_body = fn.append_basic_block("t.lst.body")
        loop_end = fn.append_basic_block("t.lst.end")
        self.builder.branch(loop_cond)

        self.builder.position_at_end(loop_cond)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(idx_a), lsize),
            loop_body,
            loop_end,
        )

        self.builder.position_at_end(loop_body)
        slot = self.builder.gep(ldata, [self.builder.load(idx_a)], inbounds=True)
        etag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        epay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        # If element is heap object, dec_temp it
        is_heap = self.builder.or_(
            self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.LIST)),
            self.builder.or_(
                self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.DICT)),
                self.builder.icmp_signed("==", etag, ir.Constant(I64, Tag.STR)),
            ),
        )
        dec_bb = fn.append_basic_block("t.lst.dec")
        skip_bb = fn.append_basic_block("t.lst.skip")
        self.builder.cbranch(is_heap, dec_bb, skip_bb)

        self.builder.position_at_end(dec_bb)
        child_ptr = self.builder.inttoptr(epay, I8P)
        self.builder.call(self.functions["__py2llvm_dec_temp"], [child_ptr])
        self.builder.branch(skip_bb)

        self.builder.position_at_end(skip_bb)
        self.builder.store(
            self.builder.add(self.builder.load(idx_a), ir.Constant(I64, 1)), idx_a
        )
        self.builder.branch(loop_cond)

        self.builder.position_at_end(loop_end)
        self.builder.branch(end_bb)

        # Traverse DICT (simplified - just go through entries)
        self.builder.position_at_end(dct_bb)
        pay_ptr = self.builder.gep(boxed_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
        dct_payload = self.builder.load(pay_ptr, "dct_payload")
        dct_ptr = self.builder.inttoptr(dct_payload, DICT_PTR)
        dcap = self.builder.load(
            self.builder.gep(dct_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        dents = self.builder.load(
            self.builder.gep(dct_ptr, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        # Simple loop through entries
        di_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), di_a)
        d_cond = fn.append_basic_block("t.dct.cond")
        d_body = fn.append_basic_block("t.dct.body")
        d_end = fn.append_basic_block("t.dct.end")
        self.builder.branch(d_cond)

        self.builder.position_at_end(d_cond)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(di_a), dcap), d_body, d_end
        )

        self.builder.position_at_end(d_body)
        ent = self.builder.gep(dents, [self.builder.load(di_a)], inbounds=True)
        ktag = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        empty_slot = self.builder.icmp_signed("==", ktag, ir.Constant(I64, -1))
        d_skip = fn.append_basic_block("t.dct.skip")
        d_dec = fn.append_basic_block("t.dct.dec")
        self.builder.cbranch(empty_slot, d_skip, d_dec)

        self.builder.position_at_end(d_dec)
        kpay = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        vpay = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        # dec_temp key and value if heap objects
        k_is_heap = self.builder.or_(
            self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.LIST)),
            self.builder.or_(
                self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.DICT)),
                self.builder.icmp_signed("==", ktag, ir.Constant(I64, Tag.STR)),
            ),
        )
        self.builder.store(
            self.builder.add(self.builder.load(di_a), ir.Constant(I64, 1)), di_a
        )
        self.builder.branch(d_cond)

        self.builder.position_at_end(d_skip)
        self.builder.store(
            self.builder.add(self.builder.load(di_a), ir.Constant(I64, 1)), di_a
        )
        self.builder.branch(d_cond)

        self.builder.position_at_end(d_end)
        self.builder.branch(end_bb)

        # STR - nothing to traverse
        self.builder.position_at_end(str_bb)
        self.builder.branch(end_bb)

        # Other - nothing to traverse
        self.builder.position_at_end(other_bb)
        self.builder.branch(end_bb)

        self.builder.position_at_end(end_bb)
        self.builder.ret_void()

        self.current_func = old_f
        self.builder = old_b

    def _build_dec_temp(self):
        """Dec_temp: dekrementacja temp_refcnt i ewentualna deallocacja."""
        fn_name = "__py2llvm_dec_temp"
        if fn_name in self.functions:
            fn = self.functions[fn_name]
        else:
            fty = ir.FunctionType(VOID, [I8P])
            fn = ir.Function(self.module, fty, fn_name)
            self.functions[fn_name] = fn

        old_b, old_f = self.builder, self.current_func
        self.current_func = fn
        self.builder = ir.IRBuilder(fn.append_basic_block("entry"))
        ptr = fn.args[0]
        z = ir.Constant(I32, 0)

        # Check for null
        is_null = self.builder.icmp_signed(
            "==", self.builder.ptrtoint(ptr, I64), ir.Constant(I64, 0)
        )
        null_bb = fn.append_basic_block("dec_temp.null")
        work_bb = fn.append_basic_block("dec_temp.work")
        self.builder.cbranch(is_null, null_bb, work_bb)

        self.builder.position_at_end(null_bb)
        self.builder.ret_void()

        self.builder.position_at_end(work_bb)
        # Get temp_refcnt pointer (index 2 in GC_HEADER)
        gc_ptr = self.builder.bitcast(ptr, ir.PointerType(GC_HEADER_TY))
        temp_refcnt_ptr = self.builder.gep(
            gc_ptr, [z, ir.Constant(I32, 2)], inbounds=True
        )

        # Decrement temp_refcnt
        old_val = self.builder.atomic_rmw(
            "sub", temp_refcnt_ptr, ir.Constant(I64, 1), "acq_rel"
        )

        # Check if it became 0 (old_val was 1)
        is_zero = self.builder.icmp_signed("==", old_val, ir.Constant(I64, 1))
        zero_bb = fn.append_basic_block("dec_temp.zero")
        non_zero_bb = fn.append_basic_block("dec_temp.non_zero")
        self.builder.cbranch(is_zero, zero_bb, non_zero_bb)

        self.builder.position_at_end(zero_bb)
        # Call dealloc
        dealloc_fn = self.functions["__py2llvm_dealloc"]
        self.builder.call(dealloc_fn, [ptr])
        self.builder.branch(non_zero_bb)

        self.builder.position_at_end(non_zero_bb)
        self.builder.ret_void()

        self.current_func = old_f
        self.builder = old_b

    def _build_gc_step(self):
        """gc_step: wykonuje krok Cycle Collectora."""
        fn_name = "__py2llvm_gc_step"
        if fn_name in self.functions:
            fn = self.functions[fn_name]
        else:
            fty = ir.FunctionType(VOID, [])
            fn = ir.Function(self.module, fty, fn_name)
            self.functions[fn_name] = fn

        old_b, old_f = self.builder, self.current_func
        self.current_func = fn
        self.builder = ir.IRBuilder(fn.append_basic_block("entry"))

        # Check if suspect buffer has any objects
        if "__cc_suspect_count" in self.module.globals:
            cnt_ptr = self.module.globals["__cc_suspect_count"]
            cnt = self.builder.load(cnt_ptr, "gc_cnt")
            is_empty = self.builder.icmp_signed("==", cnt, ir.Constant(I64, 0))
            empty_bb = fn.append_basic_block("gc_step.empty")
            work_bb = fn.append_basic_block("gc_step.work")
            self.builder.cbranch(is_empty, empty_bb, work_bb)

            self.builder.position_at_end(empty_bb)
            self.builder.ret_void()

            self.builder.position_at_end(work_bb)
            # Process all suspects in buffer
            buf = self.module.globals["__cc_suspect_buf"]
            z = ir.Constant(I32, 0)

            # Loop through suspects
            idx_a = self.builder.alloca(I64)
            self.builder.store(ir.Constant(I64, 0), idx_a)
            loop_cond = fn.append_basic_block("gc.cond")
            loop_body = fn.append_basic_block("gc.body")
            loop_end = fn.append_basic_block("gc.end")
            self.builder.branch(loop_cond)

            self.builder.position_at_end(loop_cond)
            # Reload count at each iteration (might change)
            cnt_now = self.builder.load(cnt_ptr, "gc_cnt_now")
            idx_val = self.builder.load(idx_a, "idx_val")
            self.builder.cbranch(
                self.builder.icmp_signed("<", idx_val, cnt_now), loop_body, loop_end
            )

            self.builder.position_at_end(loop_body)
            elem_ptr = self.builder.gep(buf, [z, idx_val], inbounds=True)
            obj_ptr = self.builder.load(elem_ptr, "suspect_obj")

            # Call traverse on the object
            traverse_fn = self.functions["__py2llvm_traverse"]
            self.builder.call(traverse_fn, [obj_ptr])

            # Increment index
            self.builder.store(self.builder.add(idx_val, ir.Constant(I64, 1)), idx_a)
            self.builder.branch(loop_cond)

            self.builder.position_at_end(loop_end)
            # Reset suspect count
            self.builder.store(ir.Constant(I64, 0), cnt_ptr)
            self.builder.ret_void()
        else:
            self.builder.ret_void()

        self.current_func = old_f
        self.builder = old_b

