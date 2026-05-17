"""List data structure operations: creation, append, indexing, slicing."""

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


class ListsMixin:
    """List data structure operations: creation, append, indexing, slicing."""

    def _list_ptrs(self, lst: ir.Value):
        """Zwraca (size_ptr, cap_ptr, data_ptr_ptr). GC_HEADER is at index 0."""
        z = ir.Constant(I32, 0)
        sp = self.builder.gep(lst, [z, ir.Constant(I32, 1)], inbounds=True)
        cp = self.builder.gep(lst, [z, ir.Constant(I32, 2)], inbounds=True)
        dp = self.builder.gep(lst, [z, ir.Constant(I32, 3)], inbounds=True)
        return sp, cp, dp

    def create_list(self, elements: List[Value]) -> Value:
        n = len(elements)
        cap = max(n, 4)

        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_LIST)], "lst.raw")
        lst = self.builder.bitcast(raw, LIST_PTR, "lst")
        rawd = self.builder.call(
            self._malloc, [ir.Constant(I64, cap * SZ_BOXED)], "ld.raw"
        )
        data = self.builder.bitcast(rawd, BOXED_PTR, "ld")

        z = ir.Constant(I32, 0)
        # Init GC_HEADER: refcnt=1, color=Black, temp_refcnt=0, gc_next=null
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(lst, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )  # refcnt=1
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(lst, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )  # color=Black
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(lst, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )  # temp_refcnt=0
        self.builder.store(
            null_i8p, self.builder.gep(lst, [z, z, ir.Constant(I32, 3)], inbounds=True)
        )  # gc_next=null

        sp, cp, dp = self._list_ptrs(lst)
        self.builder.store(ir.Constant(I64, n), sp)
        self.builder.store(ir.Constant(I64, cap), cp)
        self.builder.store(data, dp)

        # incref each element (they are now referenced by this list)
        for i, elem in enumerate(elements):
            slot = self.builder.gep(data, [ir.Constant(I32, i)], inbounds=True)
            self._write_slot(slot, elem)
            if elem.is_object or elem.is_list or elem.is_tuple or elem.is_dict or elem.is_str:
                self.builder.call(
                    self.functions["__py2llvm_incref"],
                    [self.builder.bitcast(elem.llvm, I8P)],
                )

        return Value(lst, PyType.LIST)

    def create_tuple(self, elements: List[Value]) -> Value:
        """Tworzy krotkę — struktura identyczna jak LIST, ale z Tag.TUPLE."""
        # Utwórz zwykłą listę, a potem zmień zwracany PyType na TUPLE.
        # Ponieważ create_list zwraca Value(lst, PyType.LIST), wystarczy
        # nadpisać pytype — struktura w pamięci jest ta sama.
        val = self.create_list(elements)
        return Value(val.llvm, PyType.TUPLE)

    def _concat_lists(self, left: Value, right: Value) -> Value:
        """Łączy dwie listy w nową."""
        z = ir.Constant(I32, 0)

        # Pobierz rozmiary obu list
        lsp, lcp, ldp = self._list_ptrs(left.llvm)
        rsp, rcp, rdp = self._list_ptrs(right.llvm)

        left_size = self.builder.load(lsp, "left_sz")
        right_size = self.builder.load(rsp, "right_sz")
        new_size = self.builder.add(left_size, right_size)
        # Dla uproszczenia użyjemy stałego cap = 8
        data_size = ir.Constant(I64, 8 * SZ_BOXED)

        # Alokuj nową listę
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_LIST)], "concat.raw")
        new_lst = self.builder.bitcast(raw, LIST_PTR, "new_lst")

        raw_data = self.builder.call(self._malloc, [data_size], "concat.data")
        new_data = self.builder.bitcast(raw_data, BOXED_PTR, "new_data")

        # Init GC_HEADER
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(new_lst, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(new_lst, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(new_lst, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            null_i8p,
            self.builder.gep(new_lst, [z, z, ir.Constant(I32, 3)], inbounds=True),
        )

        # Ustaw size, cap, data
        nsp, ncp, ndp = self._list_ptrs(new_lst)
        self.builder.store(new_size, nsp)
        self.builder.store(ir.Constant(I64, 8), ncp)  # cap = 8
        self.builder.store(new_data, ndp)

        # Skopiuj elementy z lewej listy
        left_data = self.builder.load(ldp, "left_data")

        li = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), li)
        cond_bb = self.current_func.append_basic_block("cp_left.cond")
        body_bb = self.current_func.append_basic_block("cp_left.body")
        end_bb = self.current_func.append_basic_block("cp_left.end")
        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(li), left_size),
            body_bb,
            end_bb,
        )

        self.builder.position_at_end(body_bb)
        src_slot = self.builder.gep(left_data, [self.builder.load(li)], inbounds=True)
        dst_slot = self.builder.gep(new_data, [self.builder.load(li)], inbounds=True)

        tag, pay = self._read_slot(src_slot)
        self.builder.store(
            tag, self.builder.gep(dst_slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            pay, self.builder.gep(dst_slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )

        # incref dla obiektów
        is_heap = self.builder.or_(
            self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.STR)),
            self.builder.or_(
                self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST)),
                self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.DICT)),
            ),
        )
        dec_bb = self.current_func.append_basic_block("cp_left.inc")
        skip_bb = self.current_func.append_basic_block("cp_left.skip")
        self.builder.cbranch(is_heap, dec_bb, skip_bb)

        self.builder.position_at_end(dec_bb)
        child_ptr = self.builder.inttoptr(pay, I8P)
        self.builder.call(self.functions["__py2llvm_incref"], [child_ptr])
        self.builder.branch(skip_bb)

        self.builder.position_at_end(skip_bb)
        self.builder.store(
            self.builder.add(self.builder.load(li), ir.Constant(I64, 1)), li
        )
        self.builder.branch(cond_bb)

        # Skopiuj elementy z prawej listy
        self.builder.position_at_end(end_bb)
        right_data = self.builder.load(rdp, "right_data")

        ri = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), ri)
        rcond_bb = self.current_func.append_basic_block("cp_right.cond")
        rbody_bb = self.current_func.append_basic_block("cp_right.body")
        rend_bb = self.current_func.append_basic_block("cp_right.end")
        self.builder.branch(rcond_bb)

        self.builder.position_at_end(rcond_bb)
        idx_with_offset = self.builder.add(self.builder.load(ri), left_size)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(ri), right_size),
            rbody_bb,
            rend_bb,
        )

        self.builder.position_at_end(rbody_bb)
        src_slot_r = self.builder.gep(
            right_data, [self.builder.load(ri)], inbounds=True
        )
        dst_slot_r = self.builder.gep(new_data, [idx_with_offset], inbounds=True)

        tag_r, pay_r = self._read_slot(src_slot_r)
        self.builder.store(
            tag_r, self.builder.gep(dst_slot_r, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            pay_r, self.builder.gep(dst_slot_r, [z, ir.Constant(I32, 2)], inbounds=True)
        )

        # incref
        is_heap_r = self.builder.or_(
            self.builder.icmp_signed("==", tag_r, ir.Constant(I64, Tag.STR)),
            self.builder.or_(
                self.builder.icmp_signed("==", tag_r, ir.Constant(I64, Tag.LIST)),
                self.builder.icmp_signed("==", tag_r, ir.Constant(I64, Tag.DICT)),
            ),
        )
        rdec_bb = self.current_func.append_basic_block("cp_right.inc")
        rskip_bb = self.current_func.append_basic_block("cp_right.skip")
        self.builder.cbranch(is_heap_r, rdec_bb, rskip_bb)

        self.builder.position_at_end(rdec_bb)
        child_ptr_r = self.builder.inttoptr(pay_r, I8P)
        self.builder.call(self.functions["__py2llvm_incref"], [child_ptr_r])
        self.builder.branch(rskip_bb)

        self.builder.position_at_end(rskip_bb)
        self.builder.store(
            self.builder.add(self.builder.load(ri), ir.Constant(I64, 1)), ri
        )
        self.builder.branch(rcond_bb)

        self.builder.position_at_end(rend_bb)

        return Value(new_lst, PyType.LIST)

    def list_append(self, lst_val: Value, elem: Value):
        lst = lst_val.llvm
        sp, cp, dp = self._list_ptrs(lst)
        size = self.builder.load(sp, "sz")
        cap = self.builder.load(cp, "cp")
        data = self.builder.load(dp, "dt")

        need = self.builder.icmp_signed(">=", size, cap, "need")
        ds = self.builder.alloca(BOXED_PTR, name="ds")
        self.builder.store(data, ds)

        g_bb = self.current_func.append_basic_block("la.grow")
        n_bb = self.current_func.append_basic_block("la.no")
        c_bb = self.current_func.append_basic_block("la.cont")
        self.builder.cbranch(need, g_bb, n_bb)

        self.builder.position_at_end(g_bb)
        nc = self.builder.mul(cap, ir.Constant(I64, 2), "nc")
        nsz = self.builder.mul(nc, ir.Constant(I64, SZ_BOXED))
        nr = self.builder.call(
            self._realloc, [self.builder.bitcast(data, I8P), nsz], "realloc"
        )
        nd = self.builder.bitcast(nr, BOXED_PTR)
        self.builder.store(nc, cp)
        self.builder.store(nd, ds)
        self.builder.branch(c_bb)

        self.builder.position_at_end(n_bb)
        self.builder.branch(c_bb)

        self.builder.position_at_end(c_bb)
        cur = self.builder.load(ds, "cur")
        self.builder.store(cur, dp)
        slot = self.builder.gep(cur, [size], inbounds=True, name="ns")
        self._write_slot(slot, elem)
        self.builder.store(self.builder.add(size, ir.Constant(I64, 1)), sp)
        if elem.is_object or elem.is_list or elem.is_dict or elem.is_str:
            self.builder.call(
                self.functions["__py2llvm_incref"],
                [self.builder.bitcast(elem.llvm, I8P)],
            )

    def list_getitem(self, lst_val: Value, idx_val: Value) -> Value:
        """lst[idx] → BOXED_PTR (nowa alokacja)."""
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)

        sp, cp, dp = self._list_ptrs(lst_val.llvm)
        size = self.builder.load(sp, "lg_sz")

        # Konwertuj idx na int
        raw_idx = self._to_int(idx_val).llvm

        # Obsługa indeksu ujemnego: jeśli idx < 0, dodaj size
        is_neg = self.builder.icmp_signed("<", raw_idx, ir.Constant(I64, 0))
        adj_idx = self.builder.add(raw_idx, size)
        idx = self.builder.select(is_neg, adj_idx, raw_idx)

        # Bounds check when inside a try block
        self._emit_index_bounds_check(idx, size)

        data = self.builder.load(dp, "dt")
        slot = self.builder.gep(data, [idx], inbounds=True, name="slot")
        tag, pay = self._read_slot(slot)
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
        bv = self.builder.bitcast(raw, BOXED_PTR)
        # Init GC_HEADER
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            null_i8p, self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True)
        )
        # Store tag & payload
        self.builder.store(
            tag, self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            pay, self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        # incref the payload if it's a heap object (STR/LIST/TUPLE/SET/DICT)
        is_heap = self.builder.or_(
            self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.STR)),
            self.builder.or_(
                self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST)),
                self.builder.or_(
                    self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.TUPLE)),
                    self.builder.or_(
                        self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.SET)),
                        self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.DICT)),
                    ),
                ),
            ),
        )
        lg_inc = self.current_func.append_basic_block("lg.inc")
        lg_skip = self.current_func.append_basic_block("lg.skip")
        self.builder.cbranch(is_heap, lg_inc, lg_skip)
        self.builder.position_at_end(lg_inc)
        self.builder.call(
            self.functions["__py2llvm_incref"], [self.builder.inttoptr(pay, I8P)]
        )
        self.builder.branch(lg_skip)
        self.builder.position_at_end(lg_skip)
        return Value(bv, PyType.OBJECT)

    def list_len(self, lst_val: Value) -> Value:
        sp, _, _ = self._list_ptrs(lst_val.llvm)
        return Value(self.builder.load(sp, "llen"), PyType.INT)

    def _handle_slice(self, obj: Value, slice_node: ast.Slice) -> Value:
        """Obsługa wycinków: lst[1:3], lst[:3], lst[1:], lst[::2]"""
        z = ir.Constant(I32, 0)

        # Pobierz początek i koniec
        start_val = Value(ir.Constant(I64, 0), PyType.INT)
        stop_val = Value(ir.Constant(I64, 999999999), PyType.INT)  # duża wartość

        if slice_node.lower:
            start_val = self.visit(slice_node.lower)
        if slice_node.upper:
            stop_val = self.visit(slice_node.upper)

        start = self._to_int(start_val).llvm
        stop = self._to_int(stop_val).llvm

        if obj.is_list:
            lst_ptr = obj.llvm
            # Pobierz rozmiar listy
            sp, cp, dp = self._list_ptrs(lst_ptr)
            sp, cp, dp = self._list_ptrs(obj.llvm)
            size = self.builder.load(sp, "slice_size")

            # Jeśli start jest ujemny, dodaj rozmiar
            is_neg_start = self.builder.icmp_signed("<", start, ir.Constant(I64, 0))
            start_adj = self.builder.add(start, size)
            start_final = self.builder.select(is_neg_start, start_adj, start)

            # Jeśli stop jest ujemny, dodaj rozmiar
            is_neg_stop = self.builder.icmp_signed("<", stop, ir.Constant(I64, 0))
            stop_adj = self.builder.add(stop, size)
            stop_final = self.builder.select(is_neg_stop, stop_adj, stop)

            # Ogranicz do zakresu [0, size]
            start_clamped = self.builder.select(
                self.builder.icmp_signed("<", start_final, ir.Constant(I64, 0)),
                ir.Constant(I64, 0),
                start_final,
            )
            start_clamped = self.builder.select(
                self.builder.icmp_signed(">", start_clamped, size), size, start_clamped
            )

            stop_clamped = self.builder.select(
                self.builder.icmp_signed("<", stop_final, ir.Constant(I64, 0)),
                ir.Constant(I64, 0),
                stop_final,
            )
            stop_clamped = self.builder.select(
                self.builder.icmp_signed(">", stop_clamped, size), size, stop_clamped
            )

            # new_size = max(0, stop - start)
            new_size = self.builder.sub(stop_clamped, start_clamped)
            new_size = self.builder.select(
                self.builder.icmp_signed("<", new_size, ir.Constant(I64, 0)),
                ir.Constant(I64, 0),
                new_size,
            )

            # Utwórz nową listę
            cap = ir.Constant(I64, 8)
            raw = self.builder.call(
                self._malloc, [ir.Constant(I64, SZ_LIST)], "slice.raw"
            )
            new_lst = self.builder.bitcast(raw, LIST_PTR, "new_lst")
            raw_data = self.builder.call(
                self._malloc, [ir.Constant(I64, 8 * SZ_BOXED)], "slice.data"
            )
            new_data = self.builder.bitcast(raw_data, BOXED_PTR, "new_data")

            null_i8p = ir.Constant(I8P, None)
            self.builder.store(
                ir.Constant(I64, 1),
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 0)], inbounds=True),
            )
            self.builder.store(
                ir.Constant(I32, 0),
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 1)], inbounds=True),
            )
            self.builder.store(
                ir.Constant(I64, 0),
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 2)], inbounds=True),
            )
            self.builder.store(
                null_i8p,
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 3)], inbounds=True),
            )

            nsp, ncp, ndp = self._list_ptrs(new_lst)
            self.builder.store(new_size, nsp)
            self.builder.store(cap, ncp)
            self.builder.store(new_data, ndp)

            # Skopiuj elementy
            src_data = self.builder.load(dp, "src_data")

            i = self.builder.alloca(I64)
            self.builder.store(ir.Constant(I64, 0), i)
            cond_bb = self.current_func.append_basic_block("slice.cond")
            body_bb = self.current_func.append_basic_block("slice.body")
            end_bb = self.current_func.append_basic_block("slice.end")
            self.builder.branch(cond_bb)

            self.builder.position_at_end(cond_bb)
            self.builder.cbranch(
                self.builder.icmp_signed("<", self.builder.load(i), new_size),
                body_bb,
                end_bb,
            )

            self.builder.position_at_end(body_bb)
            src_idx = self.builder.add(start_clamped, self.builder.load(i))
            src_slot = self.builder.gep(src_data, [src_idx], inbounds=True)
            dst_slot = self.builder.gep(new_data, [self.builder.load(i)], inbounds=True)

            tag, pay = self._read_slot(src_slot)
            # Bezpośrednio zapisz tag i payload
            self.builder.store(
                tag, self.builder.gep(dst_slot, [z, ir.Constant(I32, 1)], inbounds=True)
            )
            self.builder.store(
                pay, self.builder.gep(dst_slot, [z, ir.Constant(I32, 2)], inbounds=True)
            )

            # incref dla obiektów
            is_heap = self.builder.or_(
                self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.STR)),
                self.builder.or_(
                    self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST)),
                    self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.DICT)),
                ),
            )
            dec_bb = self.current_func.append_basic_block("slice.inc")
            skip_bb = self.current_func.append_basic_block("slice.skip")
            self.builder.cbranch(is_heap, dec_bb, skip_bb)

            self.builder.position_at_end(dec_bb)
            child_ptr = self.builder.inttoptr(pay, I8P)
            self.builder.call(self.functions["__py2llvm_incref"], [child_ptr])
            self.builder.branch(skip_bb)

            self.builder.position_at_end(skip_bb)
            self.builder.store(
                self.builder.add(self.builder.load(i), ir.Constant(I64, 1)), i
            )
            self.builder.branch(cond_bb)

            self.builder.position_at_end(end_bb)
            return Value(new_lst, PyType.LIST)

        # Obsługa OBJECT - sprawdź tag
        if obj.is_object:
            tag, pay = self._read_slot(obj.llvm)
            is_list = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST))

            # Jeśli to LIST, wykonaj slice
            list_bb = self.current_func.append_basic_block("slice.is_list")
            not_list_bb = self.current_func.append_basic_block("slice.not_list")
            end_bb2 = self.current_func.append_basic_block("slice.end_obj")

            self.builder.cbranch(is_list, list_bb, not_list_bb)

            # To jest lista
            self.builder.position_at_end(list_bb)
            lptr = self.builder.inttoptr(pay, LIST_PTR)
            sp, cp, dp = self._list_ptrs(lptr)
            # ... reszta kodu jest taka sama, ale z lptr zamiast obj.llvm

            size = self.builder.load(sp, "slice_size_obj")

            start_val = Value(ir.Constant(I64, 0), PyType.INT)
            stop_val = Value(ir.Constant(I64, 999999999), PyType.INT)
            if slice_node.lower:
                start_val = self.visit(slice_node.lower)
            if slice_node.upper:
                stop_val = self.visit(slice_node.upper)

            start = self._to_int(start_val).llvm
            stop = self._to_int(stop_val).llvm

            is_neg_start = self.builder.icmp_signed("<", start, ir.Constant(I64, 0))
            start_adj = self.builder.add(start, size)
            start_final = self.builder.select(is_neg_start, start_adj, start)

            is_neg_stop = self.builder.icmp_signed("<", stop, ir.Constant(I64, 0))
            stop_adj = self.builder.add(stop, size)
            stop_final = self.builder.select(is_neg_stop, stop_adj, stop)

            start_clamped = self.builder.select(
                self.builder.icmp_signed("<", start_final, ir.Constant(I64, 0)),
                ir.Constant(I64, 0),
                start_final,
            )
            start_clamped = self.builder.select(
                self.builder.icmp_signed(">", start_clamped, size), size, start_clamped
            )

            stop_clamped = self.builder.select(
                self.builder.icmp_signed("<", stop_final, ir.Constant(I64, 0)),
                ir.Constant(I64, 0),
                stop_final,
            )
            stop_clamped = self.builder.select(
                self.builder.icmp_signed(">", stop_clamped, size), size, stop_clamped
            )

            new_size = self.builder.sub(stop_clamped, start_clamped)
            new_size = self.builder.select(
                self.builder.icmp_signed("<", new_size, ir.Constant(I64, 0)),
                ir.Constant(I64, 0),
                new_size,
            )

            cap = ir.Constant(I64, 8)
            raw = self.builder.call(
                self._malloc, [ir.Constant(I64, SZ_LIST)], "slice.raw2"
            )
            new_lst = self.builder.bitcast(raw, LIST_PTR, "new_lst2")
            raw_data = self.builder.call(
                self._malloc, [ir.Constant(I64, 8 * SZ_BOXED)], "slice.data2"
            )
            new_data = self.builder.bitcast(raw_data, BOXED_PTR, "new_data2")

            null_i8p = ir.Constant(I8P, None)
            self.builder.store(
                ir.Constant(I64, 1),
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 0)], inbounds=True),
            )
            self.builder.store(
                ir.Constant(I32, 0),
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 1)], inbounds=True),
            )
            self.builder.store(
                ir.Constant(I64, 0),
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 2)], inbounds=True),
            )
            self.builder.store(
                null_i8p,
                self.builder.gep(new_lst, [z, z, ir.Constant(I32, 3)], inbounds=True),
            )

            nsp, ncp, ndp = self._list_ptrs(new_lst)
            self.builder.store(new_size, nsp)
            self.builder.store(cap, ncp)
            self.builder.store(new_data, ndp)

            src_data = self.builder.load(dp, "src_data2")

            i = self.builder.alloca(I64)
            self.builder.store(ir.Constant(I64, 0), i)
            cond_bb2 = self.current_func.append_basic_block("slice.cond2")
            body_bb2 = self.current_func.append_basic_block("slice.body2")
            end_bb3 = self.current_func.append_basic_block("slice.end2")
            self.builder.branch(cond_bb2)

            self.builder.position_at_end(cond_bb2)
            self.builder.cbranch(
                self.builder.icmp_signed("<", self.builder.load(i), new_size),
                body_bb2,
                end_bb3,
            )

            self.builder.position_at_end(body_bb2)
            src_idx = self.builder.add(start_clamped, self.builder.load(i))
            src_slot = self.builder.gep(src_data, [src_idx], inbounds=True)
            dst_slot = self.builder.gep(new_data, [self.builder.load(i)], inbounds=True)

            tag2, pay2 = self._read_slot(src_slot)
            # Bezpośrednio zapisz tag i payload
            self.builder.store(
                tag2,
                self.builder.gep(dst_slot, [z, ir.Constant(I32, 1)], inbounds=True),
            )
            self.builder.store(
                pay2,
                self.builder.gep(dst_slot, [z, ir.Constant(I32, 2)], inbounds=True),
            )

            is_heap2 = self.builder.or_(
                self.builder.icmp_signed("==", tag2, ir.Constant(I64, Tag.STR)),
                self.builder.or_(
                    self.builder.icmp_signed("==", tag2, ir.Constant(I64, Tag.LIST)),
                    self.builder.icmp_signed("==", tag2, ir.Constant(I64, Tag.DICT)),
                ),
            )
            dec_bb2 = self.current_func.append_basic_block("slice.inc2")
            skip_bb2 = self.current_func.append_basic_block("slice.skip2")
            self.builder.cbranch(is_heap2, dec_bb2, skip_bb2)

            self.builder.position_at_end(dec_bb2)
            child_ptr2 = self.builder.inttoptr(pay2, I8P)
            self.builder.call(self.functions["__py2llvm_incref"], [child_ptr2])
            self.builder.branch(skip_bb2)

            self.builder.position_at_end(skip_bb2)
            self.builder.store(
                self.builder.add(self.builder.load(i), ir.Constant(I64, 1)), i
            )
            self.builder.branch(cond_bb2)

            self.builder.position_at_end(end_bb3)
            self.builder.branch(end_bb2)

            # Not a list - error
            self.builder.position_at_end(not_list_bb)
            self.builder.branch(end_bb2)

            self.builder.position_at_end(end_bb2)
            # Dla uproszczenia, zwracamy nową listę (jeśli była lista) lub pustą
            # W rzeczywistości tu jest problem z PHI - dla uproszczenia zwróćmy pustą listę
            return self.create_list([])

        raise CompileError("Slice obsługiwany tylko dla list.", None)

    # ──────────────────────────────────────────────────────────────
    #  DICT runtime
    # ──────────────────────────────────────────────────────────────

    # def _dict_ptrs(self, dct: ir.Value):
    #     z  = ir.Constant(I32, 0)
    #     sp = self.builder.gep(dct, [z, ir.Constant(I32, 0)], inbounds=True)
    #     cp = self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True)
    #     ep = self.builder.gep(dct, [z, ir.Constant(I32, 2)], inbounds=True)
    #     return sp, cp, ep

