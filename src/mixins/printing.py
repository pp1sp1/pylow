"""Print runtime: value-to-string conversion and output."""

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


class PrintingMixin:
    """Print runtime: value-to-string conversion and output."""

    def _ensure_runtime_func(self, name: str, ftype: ir.FunctionType) -> ir.Function:
        if name in self.functions:
            return self.functions[name]
        for func in self.module.functions:
            if func.name == name:
                self.functions[name] = func
                return func
        fn = ir.Function(self.module, ftype, name=name)
        self.functions[name] = fn
        return fn

    def _ensure_to_str_helpers(self):
        if "__py2llvm_str_from_int" in self.functions:
            return
        old_b, old_f = self.builder, self.current_func
        z = ir.Constant(I32, 0)

        # Generator: int -> str
        fty = ir.FunctionType(STR_PTR, [I64])
        fn_i = ir.Function(self.module, fty, "__py2llvm_str_from_int")
        self.current_func, self.builder = (
            fn_i,
            ir.IRBuilder(fn_i.append_basic_block("entry")),
        )
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        str_obj = self.builder.bitcast(raw_str, STR_PTR)
        data_ptr = self.builder.call(self._malloc, [ir.Constant(I64, 32)])
        sz = self.builder.call(
            self._snprintf,
            [data_ptr, ir.Constant(I64, 32), self._str_ptr("%lld"), fn_i.args[0]],
        )
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(
            ir.Constant(I64, 1), self.builder.gep(str_obj, [z, z, ir.Constant(I32, 0)])
        )
        self.builder.store(
            ir.Constant(I32, 0), self.builder.gep(str_obj, [z, z, ir.Constant(I32, 1)])
        )
        self.builder.store(
            ir.Constant(I64, 0), self.builder.gep(str_obj, [z, z, ir.Constant(I32, 2)])
        )  # temp_refcnt=0
        self.builder.store(
            null_i8p, self.builder.gep(str_obj, [z, z, ir.Constant(I32, 3)])
        )  # gc_next=null
        self.builder.store(
            self.builder.sext(sz, I64),
            self.builder.gep(str_obj, [z, ir.Constant(I32, 1)]),
        )
        self.builder.store(
            ir.Constant(I64, 32), self.builder.gep(str_obj, [z, ir.Constant(I32, 2)])
        )
        self.builder.store(
            data_ptr, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)])
        )
        self.builder.ret(str_obj)
        self.functions["__py2llvm_str_from_int"] = fn_i

        # Generator: float -> str
        fty_f = ir.FunctionType(STR_PTR, [F64])
        fn_f = ir.Function(self.module, fty_f, "__py2llvm_str_from_float")
        self.current_func, self.builder = (
            fn_f,
            ir.IRBuilder(fn_f.append_basic_block("entry")),
        )
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        str_obj = self.builder.bitcast(raw_str, STR_PTR)
        data_ptr = self.builder.call(self._malloc, [ir.Constant(I64, 32)])
        sz = self.builder.call(
            self._snprintf,
            [data_ptr, ir.Constant(I64, 32), self._str_ptr("%.15g"), fn_f.args[0]],
        )
        sz_i64 = self.builder.sext(sz, I64, "sz_i64")

        # NAPRAWA: %.15g wypisuje "8" zamiast "8.0" dla wartości całkowitych.
        # Sprawdzamy czy wynik zawiera kropkę ('.') lub literę ('e'/'E').
        # Jeśli nie — doklejamy ".0" na końcu bufora, tak jak CPython.
        dot_or_e_bb = self.current_func.append_basic_block("flt.has_dot")
        append_dot0_bb = self.current_func.append_basic_block("flt.append_dot0")
        done_bb = self.current_func.append_basic_block("flt.done")

        # Przeskanuj bufor szukając '.' lub 'e' lub 'E'
        # Używamy memchr do znalezienia '.' w wyniku snprintf
        memchr_fn = self.functions.get("memchr")
        if memchr_fn is None:
            memchr_ty = ir.FunctionType(I8P, [I8P, I32, I64])
            memchr_fn = ir.Function(self.module, memchr_ty, name="memchr")
            self.functions["memchr"] = memchr_fn

        # Szukaj '.' w wyniku
        found_dot = self.builder.call(
            memchr_fn,
            [data_ptr, ir.Constant(I32, ord('.')), sz_i64],
            name="found_dot",
        )
        has_dot = self.builder.icmp_signed("!=", found_dot, ir.Constant(I8P, None), "has_dot")

        # Szukaj 'e' lub 'E' w wyniku
        found_e = self.builder.call(
            memchr_fn,
            [data_ptr, ir.Constant(I32, ord('e')), sz_i64],
            name="found_e",
        )
        has_e = self.builder.icmp_signed("!=", found_e, ir.Constant(I8P, None), "has_e")
        found_E = self.builder.call(
            memchr_fn,
            [data_ptr, ir.Constant(I32, ord('E')), sz_i64],
            name="found_E",
        )
        has_E = self.builder.icmp_signed("!=", found_E, ir.Constant(I8P, None), "has_E")

        has_dot_or_e = self.builder.or_(has_dot, self.builder.or_(has_e, has_E), "has_dot_or_e")
        self.builder.cbranch(has_dot_or_e, dot_or_e_bb, append_dot0_bb)

        # Ma kropkę lub 'e' — nic nie doklejaj
        self.builder.position_at_end(dot_or_e_bb)
        self.builder.branch(done_bb)

        # Brak kropki i 'e' — doklej ".0"
        self.builder.position_at_end(append_dot0_bb)
        # Oblicz wskaźnik na koniec stringa: data_ptr + sz
        end_ptr = self.builder.gep(data_ptr, [sz_i64], inbounds=True, name="end_ptr")
        # Zapisz '.0\0' na końcu
        self.builder.store(ir.Constant(I8, ord('.')), end_ptr)
        self.builder.store(ir.Constant(I8, ord('0')), self.builder.gep(end_ptr, [ir.Constant(I64, 1)]))
        self.builder.store(ir.Constant(I8, 0), self.builder.gep(end_ptr, [ir.Constant(I64, 2)]))
        # Powiększ rozmiar o 2
        sz_i64_appended = self.builder.add(sz_i64, ir.Constant(I64, 2), "sz_i64_appended")
        self.builder.branch(done_bb)

        self.builder.position_at_end(done_bb)
        # Phi node: oryginalny rozmiar z dot_or_e_bb, powiększony z append_dot0_bb
        final_sz = self.builder.phi(I64, "final_sz")
        final_sz.add_incoming(sz_i64, dot_or_e_bb)
        final_sz.add_incoming(sz_i64_appended, append_dot0_bb)

        self.builder.store(
            ir.Constant(I64, 1), self.builder.gep(str_obj, [z, z, ir.Constant(I32, 0)])
        )
        self.builder.store(
            ir.Constant(I32, 0), self.builder.gep(str_obj, [z, z, ir.Constant(I32, 1)])
        )
        self.builder.store(
            ir.Constant(I64, 0), self.builder.gep(str_obj, [z, z, ir.Constant(I32, 2)])
        )  # temp_refcnt=0
        self.builder.store(
            null_i8p, self.builder.gep(str_obj, [z, z, ir.Constant(I32, 3)])
        )  # gc_next=null
        self.builder.store(
            final_sz,
            self.builder.gep(str_obj, [z, ir.Constant(I32, 1)]),
        )
        self.builder.store(
            ir.Constant(I64, 32), self.builder.gep(str_obj, [z, ir.Constant(I32, 2)])
        )
        self.builder.store(
            data_ptr, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)])
        )
        self.builder.ret(str_obj)
        self.functions["__py2llvm_str_from_float"] = fn_f

        self.builder, self.current_func = old_b, old_f

    def _get_or_create_to_str_dyn(self) -> ir.Function:
        """Klon __py2llvm_print_dyn, ale zamiast drukować, łączy i zwraca gotowy STRING LLVM"""
        func_name = "__py2llvm_to_str_dyn"
        if func_name in self.functions:
            return self.functions[func_name]

        self._ensure_to_str_helpers()
        concat_fn = self._get_or_create_concat_fn()
        fn_int = self.functions["__py2llvm_str_from_int"]
        fn_flt = self.functions["__py2llvm_str_from_float"]

        fty = ir.FunctionType(STR_PTR, [I64, I64, I1, I8P])
        func = ir.Function(self.module, fty, name=func_name)
        self.functions[func_name] = func

        old_b, old_f = self.builder, self.current_func
        self.current_func, self.builder = (
            func,
            ir.IRBuilder(func.append_basic_block("entry")),
        )

        tag, pay, is_repr, visited_node = (
            func.args[0],
            func.args[1],
            func.args[2],
            func.args[3],
        )
        NODE_TY = ir.LiteralStructType([I64, I8P])
        NODE_PTR = ir.PointerType(NODE_TY)

        curr_str_ptr = self.builder.alloca(STR_PTR)
        self.builder.store(self.create_string("").llvm, curr_str_ptr)

        def append_const(s_val):
            s_obj = self.create_string(s_val).llvm
            cur = self.builder.load(curr_str_ptr)
            self.builder.store(self.builder.call(concat_fn, [cur, s_obj]), curr_str_ptr)

        def append_obj(s_obj):
            cur = self.builder.load(curr_str_ptr)
            self.builder.store(self.builder.call(concat_fn, [cur, s_obj]), curr_str_ptr)

        cont_bb = self.current_func.append_basic_block("sd.cont")
        int_bb = self.current_func.append_basic_block("sd.int")
        flt_bb = self.current_func.append_basic_block("sd.flt")
        bool_bb = self.current_func.append_basic_block("sd.bool")
        str_bb = self.current_func.append_basic_block("sd.str")
        dct_bb = self.current_func.append_basic_block("sd.dct")
        none_bb = self.current_func.append_basic_block("sd.none")

        lst_bb = self.current_func.append_basic_block("sd.lst")
        tup_bb = self.current_func.append_basic_block("sd.tup")
        set_str_bb = self.current_func.append_basic_block("sd.set")
        iter_bb = self.current_func.append_basic_block("sd.iter")

        sw = self.builder.switch(tag, none_bb)
        sw.add_case(ir.Constant(I64, Tag.INT), int_bb)
        sw.add_case(ir.Constant(I64, Tag.FLOAT), flt_bb)
        sw.add_case(ir.Constant(I64, Tag.BOOL), bool_bb)
        sw.add_case(ir.Constant(I64, Tag.STR), str_bb)
        sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
        sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)
        sw.add_case(ir.Constant(I64, Tag.TUPLE), tup_bb)
        sw.add_case(ir.Constant(I64, Tag.SET), set_str_bb)
        sw.add_case(ir.Constant(I64, Tag.ITERATOR), iter_bb)
        inst_bb = self.current_func.append_basic_block("sd.inst")
        sw.add_case(ir.Constant(I64, Tag.INST), inst_bb)

        # -- INT / FLOAT --
        self.builder.position_at_end(int_bb)
        append_obj(self.builder.call(fn_int, [pay]))
        self.builder.branch(cont_bb)

        self.builder.position_at_end(flt_bb)
        append_obj(self.builder.call(fn_flt, [self.builder.bitcast(pay, F64)]))
        self.builder.branch(cont_bb)

        # -- BOOL --
        self.builder.position_at_end(bool_bb)
        t_bb, f_bb = (
            self.current_func.append_basic_block("sdt"),
            self.current_func.append_basic_block("sdf"),
        )
        self.builder.cbranch(
            self.builder.icmp_signed("!=", pay, ir.Constant(I64, 0)), t_bb, f_bb
        )
        self.builder.position_at_end(t_bb)
        append_const("True")
        self.builder.branch(cont_bb)
        self.builder.position_at_end(f_bb)
        append_const("False")
        self.builder.branch(cont_bb)

        # -- STR --
        self.builder.position_at_end(str_bb)
        str_obj = self.builder.inttoptr(pay, STR_PTR)
        str_repr_bb, str_norm_bb = (
            self.current_func.append_basic_block("sdr"),
            self.current_func.append_basic_block("sdn"),
        )
        self.builder.cbranch(is_repr, str_repr_bb, str_norm_bb)
        self.builder.position_at_end(str_repr_bb)
        append_const("'")
        append_obj(str_obj)
        append_const("'")
        self.builder.branch(cont_bb)
        self.builder.position_at_end(str_norm_bb)
        append_obj(str_obj)
        self.builder.branch(cont_bb)

        # -- DICT -- (Szybkie haszowane formatowanie z detekcją pętli stosu)
        self.builder.position_at_end(dct_bb)
        chk_cond, chk_body = (
            self.current_func.append_basic_block("sd.dct.chk"),
            self.current_func.append_basic_block("sd.dct.chkb"),
        )
        curr_ptr = self.builder.alloca(I8P)
        self.builder.store(visited_node, curr_ptr)
        self.builder.branch(chk_cond)
        self.builder.position_at_end(chk_cond)
        curr = self.builder.load(curr_ptr)
        d_real_bb = self.current_func.append_basic_block("sd.dct.real")
        self.builder.cbranch(
            self.builder.icmp_signed(
                "==", self.builder.ptrtoint(curr, I64), ir.Constant(I64, 0)
            ),
            d_real_bb,
            chk_body,
        )
        self.builder.position_at_end(chk_body)
        node = self.builder.bitcast(curr, NODE_PTR)
        z = ir.Constant(I32, 0)
        n_pay, n_nxt = (
            self.builder.load(self.builder.gep(node, [z, z], inbounds=True)),
            self.builder.load(
                self.builder.gep(node, [z, ir.Constant(I32, 1)], inbounds=True)
            ),
        )
        hit_bb, next_bb = (
            self.current_func.append_basic_block("sd.dct.hit"),
            self.current_func.append_basic_block("sd.dct.nxt"),
        )
        self.builder.cbranch(
            self.builder.icmp_signed("==", n_pay, pay), hit_bb, next_bb
        )
        self.builder.position_at_end(hit_bb)
        append_const("{...}")
        self.builder.branch(cont_bb)
        self.builder.position_at_end(next_bb)
        self.builder.store(n_nxt, curr_ptr)
        self.builder.branch(chk_cond)

        self.builder.position_at_end(d_real_bb)
        new_node = self.builder.alloca(NODE_TY)
        self.builder.store(
            pay, self.builder.gep(new_node, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        self.builder.store(
            visited_node,
            self.builder.gep(new_node, [z, ir.Constant(I32, 1)], inbounds=True),
        )
        v_dct = self.builder.bitcast(new_node, I8P)

        dptr = self.builder.inttoptr(pay, DICT_PTR)
        d_cap = self.builder.load(
            self.builder.gep(dptr, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        d_ents = self.builder.load(
            self.builder.gep(dptr, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        append_const("{")
        di_a, printed_items_a = self.builder.alloca(I64), self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), di_a)
        self.builder.store(ir.Constant(I64, 0), printed_items_a)

        d_cond_bb, d_body_bb = (
            self.current_func.append_basic_block("sdd.cond"),
            self.current_func.append_basic_block("sdd.body"),
        )
        d_skip_bb, d_pr_bb = (
            self.current_func.append_basic_block("sdd.skip"),
            self.current_func.append_basic_block("sdd.pr"),
        )
        d_next_bb, d_done_bb = (
            self.current_func.append_basic_block("sdd.next"),
            self.current_func.append_basic_block("sdd.done"),
        )

        self.builder.branch(d_cond_bb)
        self.builder.position_at_end(d_cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(di_a), d_cap),
            d_body_bb,
            d_done_bb,
        )
        self.builder.position_at_end(d_body_bb)
        ent = self.builder.gep(d_ents, [self.builder.load(di_a)], inbounds=True)
        ktag = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True)
        )
        self.builder.cbranch(
            self.builder.icmp_signed("==", ktag, ir.Constant(I64, -1)),
            d_skip_bb,
            d_pr_bb,
        )

        self.builder.position_at_end(d_skip_bb)
        self.builder.branch(d_next_bb)
        self.builder.position_at_end(d_pr_bb)
        d_sep_bb, d_do_pr_bb = (
            self.current_func.append_basic_block("sdd.sep"),
            self.current_func.append_basic_block("sdd.dopr"),
        )
        self.builder.cbranch(
            self.builder.icmp_signed(
                ">", self.builder.load(printed_items_a), ir.Constant(I64, 0)
            ),
            d_sep_bb,
            d_do_pr_bb,
        )
        self.builder.position_at_end(d_sep_bb)
        append_const(", ")
        self.builder.branch(d_do_pr_bb)
        self.builder.position_at_end(d_do_pr_bb)
        kpay = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        vtag = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        vpay = self.builder.load(
            self.builder.gep(ent, [z, ir.Constant(I32, 3)], inbounds=True)
        )
        append_obj(self.builder.call(func, [ktag, kpay, ir.Constant(I1, 1), v_dct]))
        append_const(": ")
        append_obj(self.builder.call(func, [vtag, vpay, ir.Constant(I1, 1), v_dct]))
        self.builder.store(
            self.builder.add(self.builder.load(printed_items_a), ir.Constant(I64, 1)),
            printed_items_a,
        )
        self.builder.branch(d_next_bb)
        self.builder.position_at_end(d_next_bb)
        self.builder.store(
            self.builder.add(self.builder.load(di_a), ir.Constant(I64, 1)), di_a
        )
        self.builder.branch(d_cond_bb)
        self.builder.position_at_end(d_done_bb)
        append_const("}")
        self.builder.branch(cont_bb)

        # -- LIST -- (iteracja po elementach i konwersja na string)
        self.builder.position_at_end(lst_bb)
        lptr = self.builder.inttoptr(pay, LIST_PTR)
        lz = ir.Constant(I32, 0)
        lsz = self.builder.load(self.builder.gep(lptr, [lz, ir.Constant(I32, 1)], inbounds=True))
        ldata = self.builder.load(self.builder.gep(lptr, [lz, ir.Constant(I32, 3)], inbounds=True))
        append_const("[")
        li_a = self.builder.alloca(I64, name="sdl_i")
        self.builder.store(ir.Constant(I64, 0), li_a)
        printed_a = self.builder.alloca(I64, name="sdl_cnt")
        self.builder.store(ir.Constant(I64, 0), printed_a)
        l_cond_bb = self.current_func.append_basic_block("sdl.cond")
        l_body_bb = self.current_func.append_basic_block("sdl.body")
        l_sep_bb = self.current_func.append_basic_block("sdl.sep")
        l_pr_bb = self.current_func.append_basic_block("sdl.pr")
        l_next_bb = self.current_func.append_basic_block("sdl.next")
        l_done_bb = self.current_func.append_basic_block("sdl.done")
        self.builder.branch(l_cond_bb)
        self.builder.position_at_end(l_cond_bb)
        self.builder.cbranch(self.builder.icmp_signed("<", self.builder.load(li_a), lsz), l_body_bb, l_done_bb)
        self.builder.position_at_end(l_body_bb)
        self.builder.cbranch(self.builder.icmp_signed(">", self.builder.load(printed_a), ir.Constant(I64, 0)), l_sep_bb, l_pr_bb)
        self.builder.position_at_end(l_sep_bb)
        append_const(", ")
        self.builder.branch(l_pr_bb)
        self.builder.position_at_end(l_pr_bb)
        lslot = self.builder.gep(ldata, [self.builder.load(li_a)], inbounds=True)
        lslot_boxed = self.builder.bitcast(lslot, BOXED_PTR)
        letag = self.builder.load(self.builder.gep(lslot_boxed, [lz, ir.Constant(I32, 1)], inbounds=True))
        lepay = self.builder.load(self.builder.gep(lslot_boxed, [lz, ir.Constant(I32, 2)], inbounds=True))
        l_vnode = self.builder.inttoptr(self.builder.ptrtoint(lptr, I64), I8P)
        append_obj(self.builder.call(func, [letag, lepay, ir.Constant(I1, 1), l_vnode]))
        self.builder.store(self.builder.add(self.builder.load(printed_a), ir.Constant(I64, 1)), printed_a)
        self.builder.branch(l_next_bb)
        self.builder.position_at_end(l_next_bb)
        self.builder.store(self.builder.add(self.builder.load(li_a), ir.Constant(I64, 1)), li_a)
        self.builder.branch(l_cond_bb)
        self.builder.position_at_end(l_done_bb)
        append_const("]")
        self.builder.branch(cont_bb)

        # -- TUPLE -- (taka sama logika jak LIST, ale z nawiasami okrągłymi)
        self.builder.position_at_end(tup_bb)
        tptr = self.builder.inttoptr(pay, LIST_PTR)
        tsz = self.builder.load(self.builder.gep(tptr, [lz, ir.Constant(I32, 1)], inbounds=True))
        tdata = self.builder.load(self.builder.gep(tptr, [lz, ir.Constant(I32, 3)], inbounds=True))
        append_const("(")
        ti_a = self.builder.alloca(I64, name="sdt_i")
        self.builder.store(ir.Constant(I64, 0), ti_a)
        tprinted_a = self.builder.alloca(I64, name="sdt_cnt")
        self.builder.store(ir.Constant(I64, 0), tprinted_a)
        t_cond_bb = self.current_func.append_basic_block("sdt.cond")
        t_body_bb = self.current_func.append_basic_block("sdt.body")
        t_sep_bb = self.current_func.append_basic_block("sdt.sep")
        t_pr_bb = self.current_func.append_basic_block("sdt.pr")
        t_next_bb = self.current_func.append_basic_block("sdt.next")
        t_done_bb = self.current_func.append_basic_block("sdt.done")
        self.builder.branch(t_cond_bb)
        self.builder.position_at_end(t_cond_bb)
        self.builder.cbranch(self.builder.icmp_signed("<", self.builder.load(ti_a), tsz), t_body_bb, t_done_bb)
        self.builder.position_at_end(t_body_bb)
        self.builder.cbranch(self.builder.icmp_signed(">", self.builder.load(tprinted_a), ir.Constant(I64, 0)), t_sep_bb, t_pr_bb)
        self.builder.position_at_end(t_sep_bb)
        append_const(", ")
        self.builder.branch(t_pr_bb)
        self.builder.position_at_end(t_pr_bb)
        tslot = self.builder.gep(tdata, [self.builder.load(ti_a)], inbounds=True)
        tslot_boxed = self.builder.bitcast(tslot, BOXED_PTR)
        tetag = self.builder.load(self.builder.gep(tslot_boxed, [lz, ir.Constant(I32, 1)], inbounds=True))
        tepay = self.builder.load(self.builder.gep(tslot_boxed, [lz, ir.Constant(I32, 2)], inbounds=True))
        t_vnode = self.builder.inttoptr(self.builder.ptrtoint(tptr, I64), I8P)
        append_obj(self.builder.call(func, [tetag, tepay, ir.Constant(I1, 1), t_vnode]))
        self.builder.store(self.builder.add(self.builder.load(tprinted_a), ir.Constant(I64, 1)), tprinted_a)
        self.builder.branch(t_next_bb)
        self.builder.position_at_end(t_next_bb)
        self.builder.store(self.builder.add(self.builder.load(ti_a), ir.Constant(I64, 1)), ti_a)
        self.builder.branch(t_cond_bb)
        self.builder.position_at_end(t_done_bb)
        append_const(")")
        self.builder.branch(cont_bb)

        # -- SET -- (taka sama logika jak LIST, ale z nawiasami klamrowymi {})
        self.builder.position_at_end(set_str_bb)
        setptr = self.builder.inttoptr(pay, LIST_PTR)
        setsz = self.builder.load(self.builder.gep(setptr, [lz, ir.Constant(I32, 1)], inbounds=True))
        setdata = self.builder.load(self.builder.gep(setptr, [lz, ir.Constant(I32, 3)], inbounds=True))
        append_const("{")
        si_a = self.builder.alloca(I64, name="sds_i")
        self.builder.store(ir.Constant(I64, 0), si_a)
        sprinted_a = self.builder.alloca(I64, name="sds_cnt")
        self.builder.store(ir.Constant(I64, 0), sprinted_a)
        s_cond_bb = self.current_func.append_basic_block("sds.cond")
        s_body_bb = self.current_func.append_basic_block("sds.body")
        s_sep_bb = self.current_func.append_basic_block("sds.sep")
        s_pr_bb = self.current_func.append_basic_block("sds.pr")
        s_next_bb = self.current_func.append_basic_block("sds.next")
        s_done_bb = self.current_func.append_basic_block("sds.done")
        self.builder.branch(s_cond_bb)
        self.builder.position_at_end(s_cond_bb)
        self.builder.cbranch(self.builder.icmp_signed("<", self.builder.load(si_a), setsz), s_body_bb, s_done_bb)
        self.builder.position_at_end(s_body_bb)
        self.builder.cbranch(self.builder.icmp_signed(">", self.builder.load(sprinted_a), ir.Constant(I64, 0)), s_sep_bb, s_pr_bb)
        self.builder.position_at_end(s_sep_bb)
        append_const(", ")
        self.builder.branch(s_pr_bb)
        self.builder.position_at_end(s_pr_bb)
        sslot = self.builder.gep(setdata, [self.builder.load(si_a)], inbounds=True)
        sslot_boxed = self.builder.bitcast(sslot, BOXED_PTR)
        setag = self.builder.load(self.builder.gep(sslot_boxed, [lz, ir.Constant(I32, 1)], inbounds=True))
        sepay = self.builder.load(self.builder.gep(sslot_boxed, [lz, ir.Constant(I32, 2)], inbounds=True))
        s_vnode = self.builder.inttoptr(self.builder.ptrtoint(setptr, I64), I8P)
        append_obj(self.builder.call(func, [setag, sepay, ir.Constant(I1, 1), s_vnode]))
        self.builder.store(self.builder.add(self.builder.load(sprinted_a), ir.Constant(I64, 1)), sprinted_a)
        self.builder.branch(s_next_bb)
        self.builder.position_at_end(s_next_bb)
        self.builder.store(self.builder.add(self.builder.load(si_a), ir.Constant(I64, 1)), si_a)
        self.builder.branch(s_cond_bb)
        self.builder.position_at_end(s_done_bb)
        append_const("}")
        self.builder.branch(cont_bb)

        # -- ITERATOR --
        self.builder.position_at_end(iter_bb)
        append_const("<iterator>")
        self.builder.branch(cont_bb)

        # -- INSTANCE (dataclass-like repr) --
        self.builder.position_at_end(inst_bb)
        inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
        iz = ir.Constant(I32, 0)
        # Read attrs dict pointer from instance (field index 2)
        i_attrs_ptr = self.builder.load(
            self.builder.gep(inst_ptr, [iz, ir.Constant(I32, 2)], inbounds=True),
            "sdi_attrs"
        )
        # Read class name from dict's "__class__" key
        i_get_fn = self.functions["__py2llvm_dict_get_internal"]
        i_cls_key_tag = ir.Constant(I64, Tag.STR)
        i_cls_key_pay = self.builder.ptrtoint(
            self.create_string("__class__").llvm, I64
        )
        i_cls_res_t = self.builder.alloca(I64, name="sdi_cls_rt")
        i_cls_res_p = self.builder.alloca(I64, name="sdi_cls_rp")
        self.builder.store(ir.Constant(I64, 0), i_cls_res_t)
        self.builder.store(ir.Constant(I64, 0), i_cls_res_p)
        self.builder.call(i_get_fn, [i_attrs_ptr, i_cls_key_tag, i_cls_key_pay, i_cls_res_t, i_cls_res_p])
        i_cls_name_tag = self.builder.load(i_cls_res_t, "sdi_cls_nt")
        i_cls_name_pay = self.builder.load(i_cls_res_p, "sdi_cls_np")
        # Print class name if it's a string
        i_is_cls_str = self.builder.icmp_signed("==", i_cls_name_tag, ir.Constant(I64, Tag.STR))
        i_cls_str_bb = self.current_func.append_basic_block("sdi.cls")
        i_nocls_bb = self.current_func.append_basic_block("sdi.nocls")
        self.builder.cbranch(i_is_cls_str, i_cls_str_bb, i_nocls_bb)
        self.builder.position_at_end(i_cls_str_bb)
        i_cls_str_obj = self.builder.inttoptr(i_cls_name_pay, STR_PTR)
        append_obj(i_cls_str_obj)
        append_const("(")
        self.builder.branch(i_nocls_bb)
        self.builder.position_at_end(i_nocls_bb)
        # Iterate over ordered_keys
        i_ordered_list = self.builder.load(
            self.builder.gep(i_attrs_ptr, [iz, ir.Constant(I32, 4)], inbounds=True),
            "sdi_olist"
        )
        i_ol_size = self.builder.load(
            self.builder.gep(i_ordered_list, [iz, ir.Constant(I32, 1)], inbounds=True),
            "sdi_olen"
        )
        i_ol_data = self.builder.load(
            self.builder.gep(i_ordered_list, [iz, ir.Constant(I32, 3)], inbounds=True),
            "sdi_odata"
        )
        i_ii_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_ii_a)
        i_real_idx_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_real_idx_a)
        i_s_cond_bb = self.current_func.append_basic_block("sdi.cond")
        i_s_body_bb = self.current_func.append_basic_block("sdi.body")
        i_s_next_bb = self.current_func.append_basic_block("sdi.next")
        i_s_done_bb = self.current_func.append_basic_block("sdi.done")
        self.builder.branch(i_s_cond_bb)
        self.builder.position_at_end(i_s_cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(i_ii_a), i_ol_size),
            i_s_body_bb, i_s_done_bb,
        )
        self.builder.position_at_end(i_s_body_bb)
        i_idx = self.builder.load(i_ii_a)
        i_kslot = self.builder.gep(i_ol_data, [i_idx], inbounds=True)
        i_ktag = self.builder.load(
            self.builder.gep(i_kslot, [iz, ir.Constant(I32, 1)], inbounds=True)
        )
        i_kpay = self.builder.load(
            self.builder.gep(i_kslot, [iz, ir.Constant(I32, 2)], inbounds=True)
        )
        # Skip __class__ key
        i_is_cls_key = self.builder.icmp_signed("==", i_ktag, ir.Constant(I64, Tag.STR))
        i_skip_cls_bb = self.current_func.append_basic_block("sdi.skipcls")
        i_chk_cls_bb = self.current_func.append_basic_block("sdi.chkcls")
        self.builder.cbranch(i_is_cls_key, i_chk_cls_bb, i_skip_cls_bb)
        self.builder.position_at_end(i_chk_cls_bb)
        i_k_str_obj = self.builder.inttoptr(i_kpay, STR_PTR)
        i_k_str_data = self.builder.load(
            self.builder.gep(i_k_str_obj, [iz, ir.Constant(I32, 3)], inbounds=True)
        )
        i_cls_cmp_str = self.create_string("__class__")
        i_cls_cmp_data = self.builder.load(
            self.builder.gep(i_cls_cmp_str.llvm, [iz, ir.Constant(I32, 3)], inbounds=True)
        )
        i_cmp_res = self.builder.call(self._strcmp, [i_k_str_data, i_cls_cmp_data])
        i_is_cls_match = self.builder.icmp_signed("==", i_cmp_res, ir.Constant(I32, 0))
        # Also check for __frozen__ — skip it too
        i_frozen_cmp_str = self.create_string("__frozen__")
        i_frozen_cmp_data = self.builder.load(
            self.builder.gep(i_frozen_cmp_str.llvm, [iz, ir.Constant(I32, 3)], inbounds=True)
        )
        i_frozen_cmp_res = self.builder.call(self._strcmp, [i_k_str_data, i_frozen_cmp_data])
        i_is_frozen_match = self.builder.icmp_signed("==", i_frozen_cmp_res, ir.Constant(I32, 0))
        i_skip_key = self.builder.or_(i_is_cls_match, i_is_frozen_match)
        self.builder.cbranch(i_skip_key, i_s_next_bb, i_skip_cls_bb)
        self.builder.position_at_end(i_skip_cls_bb)
        i_real_i = self.builder.load(i_real_idx_a)
        i_is_first_attr = self.builder.icmp_signed("==", i_real_i, ir.Constant(I64, 0))
        i_attr_sep_bb = self.current_func.append_basic_block("sdi.sep")
        i_attr_nosep_bb = self.current_func.append_basic_block("sdi.nosep")
        self.builder.cbranch(i_is_first_attr, i_attr_nosep_bb, i_attr_sep_bb)
        self.builder.position_at_end(i_attr_sep_bb)
        append_const(", ")
        self.builder.branch(i_attr_nosep_bb)
        self.builder.position_at_end(i_attr_nosep_bb)
        # NAPRAWA: Nazwa pola bez cudzysłowów (is_repr=0), wartość z cudzysłowami (is_repr=1)
        # Bez tego dataclass repr pokazuje 'x'=1.0 zamiast x=1.0
        append_obj(self.builder.call(func, [i_ktag, i_kpay, ir.Constant(I1, 0), ir.Constant(I8P, None)]))
        append_const("=")
        i_vres_t = self.builder.alloca(I64, name="sdi_vrt")
        i_vres_p = self.builder.alloca(I64, name="sdi_vrp")
        self.builder.call(i_get_fn, [i_attrs_ptr, i_ktag, i_kpay, i_vres_t, i_vres_p])
        i_vtag = self.builder.load(i_vres_t, "sdi_vt")
        i_vpay = self.builder.load(i_vres_p, "sdi_vp")
        append_obj(self.builder.call(func, [i_vtag, i_vpay, ir.Constant(I1, 1), ir.Constant(I8P, None)]))
        self.builder.store(
            self.builder.add(self.builder.load(i_real_idx_a), ir.Constant(I64, 1)),
            i_real_idx_a
        )
        self.builder.branch(i_s_next_bb)
        self.builder.position_at_end(i_s_next_bb)
        self.builder.store(
            self.builder.add(self.builder.load(i_ii_a), ir.Constant(I64, 1)), i_ii_a
        )
        self.builder.branch(i_s_cond_bb)
        self.builder.position_at_end(i_s_done_bb)
        append_const(")")
        self.builder.branch(cont_bb)

        # -- NONE --
        self.builder.position_at_end(none_bb)
        append_const("None")
        self.builder.branch(cont_bb)

        self.builder.position_at_end(cont_bb)
        self.builder.ret(self.builder.load(curr_str_ptr))
        self.current_func, self.builder = old_f, old_b
        return func

    def val_to_str(self, val: Value) -> Value:
        to_str_fn = self._get_or_create_to_str_dyn()
        tag, pay = self._value_to_tag_payload(val)
        null_ptr = self.builder.inttoptr(ir.Constant(I64, 0), I8P)
        res = self.builder.call(to_str_fn, [tag, pay, ir.Constant(I1, 0), null_ptr])
        return Value(res, PyType.STR)

    def print_value(self, v: Value):
        """
        Zunifikowany punkt wejścia. Pobiera tag i payload bezpośrednio
        z rejestrów kompilatora (zero alokacji sterty dla typów statycznych!)
        i przekazuje do uniwersalnego mechanizmu LLVM.
        """
        tag, pay = self._value_to_tag_payload(v)
        func = self._get_or_create_print_dyn()

        # null_ptr to wskaźnik początkowy dla detektora cykli (pusta lista)
        null_ptr = self.builder.inttoptr(ir.Constant(I64, 0), I8P)

        # arg2: is_repr (0 dla top-level print), arg3: visited_node (null)
        self.builder.call(func, [tag, pay, ir.Constant(I1, 0), null_ptr])

    def _get_or_create_print_dyn(self) -> ir.Function:
        """
        Potężny mechanizm runtime. Obsługuje is_repr, rekurencję słowników,
        prawdziwe napisy boolean (True/False) oraz posiada wykrywacz cykli
        zabezpieczający przed Stack Overflow w C.
        """
        func_name = "__py2llvm_print_dyn"
        if func_name in self.functions:
            return self.functions[func_name]

        # Sygnatura: void (i64 tag, i64 payload, i1 is_repr, i8* visited_node)
        fty = ir.FunctionType(VOID, [I64, I64, I1, I8P])
        func = ir.Function(self.module, fty, name=func_name)
        self.functions[func_name] = func

        old_builder = self.builder
        old_func = self.current_func
        self.current_func = func
        entry = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)

        tag = func.args[0]
        pay = func.args[1]
        is_repr = func.args[2]
        visited_node = func.args[3]

        # NODE_TY = { i64 pay, i8* next_node } - alokowany na stosie LLVM!
        NODE_TY = ir.LiteralStructType([I64, I8P])
        NODE_PTR = ir.PointerType(NODE_TY)

        cont_bb = self.current_func.append_basic_block("pd.cont")
        int_bb = self.current_func.append_basic_block("pd.int")
        flt_bb = self.current_func.append_basic_block("pd.flt")
        bool_bb = self.current_func.append_basic_block("pd.bool")
        str_bb = self.current_func.append_basic_block("pd.str")
        lst_bb = self.current_func.append_basic_block("pd.lst")
        tup_bb = self.current_func.append_basic_block("pd.tup")
        set_bb = self.current_func.append_basic_block("pd.set")
        dct_bb = self.current_func.append_basic_block("pd.dct")
        none_bb = self.current_func.append_basic_block("pd.none")
        inst_bb = self.current_func.append_basic_block("pd.inst")

        sw = self.builder.switch(tag, none_bb)
        sw.add_case(ir.Constant(I64, Tag.INT), int_bb)
        sw.add_case(ir.Constant(I64, Tag.FLOAT), flt_bb)
        sw.add_case(ir.Constant(I64, Tag.BOOL), bool_bb)
        sw.add_case(ir.Constant(I64, Tag.STR), str_bb)
        sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)
        sw.add_case(ir.Constant(I64, Tag.TUPLE), tup_bb)
        sw.add_case(ir.Constant(I64, Tag.SET), set_bb)
        sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
        sw.add_case(ir.Constant(I64, Tag.INST), inst_bb)  # INST -> print instance repr
        sw.add_case(ir.Constant(I64, Tag.ITERATOR), none_bb)  # ITERATOR -> drukuj jako <iterator>

        # -- INT / FLOAT --
        self.builder.position_at_end(int_bb)
        self.builder.call(self._printf, [self._str_ptr("%lld"), pay])
        self.builder.branch(cont_bb)

        self.builder.position_at_end(flt_bb)
        # NAPRAWA: Użyto "%#.15g" — flaga '#' wymusza obecność kropki dziesiętną.
        # W C: %#g zawsze wyświetla kropkę, więc 8.0 nie stanie się "8".
        # Ale %#g dodaje też zera końcowe (np. "8.000000000000000"),
        # co nie jest zgodne z Pythonem. Dlatego lepiej: snprintf do bufora,
        # sprawdź kropkę/'e', doklej ".0" jeśli trzeba.
        flt_buf = self.builder.alloca(ir.ArrayType(I8, 48), name="flt_buf")
        flt_buf_ptr = self.builder.bitcast(flt_buf, I8P, "flt_buf_ptr")
        flt_sz = self.builder.call(
            self._snprintf,
            [flt_buf_ptr, ir.Constant(I64, 48), self._str_ptr("%.15g"),
             self.builder.bitcast(pay, F64)],
            name="flt_sz",
        )
        flt_sz_i64 = self.builder.sext(flt_sz, I64, "flt_sz_i64")
        # Szukaj '.' lub 'e'/'E' w wyniku
        memchr_fn2 = self.functions.get("memchr")
        if memchr_fn2 is None:
            memchr_ty2 = ir.FunctionType(I8P, [I8P, I32, I64])
            memchr_fn2 = ir.Function(self.module, memchr_ty2, name="memchr")
            self.functions["memchr"] = memchr_fn2
        fdot = self.builder.call(memchr_fn2, [flt_buf_ptr, ir.Constant(I32, ord('.')), flt_sz_i64], name="pd_fdot")
        has_fd = self.builder.icmp_signed("!=", fdot, ir.Constant(I8P, None))
        fexp = self.builder.call(memchr_fn2, [flt_buf_ptr, ir.Constant(I32, ord('e')), flt_sz_i64], name="pd_fexp")
        has_fe = self.builder.icmp_signed("!=", fexp, ir.Constant(I8P, None))
        fExp = self.builder.call(memchr_fn2, [flt_buf_ptr, ir.Constant(I32, ord('E')), flt_sz_i64], name="pd_fExp")
        has_fE = self.builder.icmp_signed("!=", fExp, ir.Constant(I8P, None))
        has_dot_or_exp = self.builder.or_(has_fd, self.builder.or_(has_fe, has_fE))

        pd_flt_has = self.current_func.append_basic_block("pd.flt.has")
        pd_flt_no = self.current_func.append_basic_block("pd.flt.no")
        pd_flt_done = self.current_func.append_basic_block("pd.flt.done")

        self.builder.cbranch(has_dot_or_exp, pd_flt_has, pd_flt_no)

        # Ma kropkę lub 'e' — drukuj bezpośrednio
        self.builder.position_at_end(pd_flt_has)
        self.builder.call(self._printf, [self._str_ptr("%s"), flt_buf_ptr])
        self.builder.branch(pd_flt_done)

        # Brak kropki — doklej ".0" do bufora i drukuj
        self.builder.position_at_end(pd_flt_no)
        end_p = self.builder.gep(flt_buf_ptr, [flt_sz_i64], inbounds=True)
        self.builder.store(ir.Constant(I8, ord('.')), end_p)
        self.builder.store(ir.Constant(I8, ord('0')), self.builder.gep(end_p, [ir.Constant(I64, 1)]))
        self.builder.store(ir.Constant(I8, 0), self.builder.gep(end_p, [ir.Constant(I64, 2)]))
        self.builder.call(self._printf, [self._str_ptr("%s"), flt_buf_ptr])
        self.builder.branch(pd_flt_done)

        self.builder.position_at_end(pd_flt_done)
        self.builder.branch(cont_bb)

        # -- BOOL (Prawdziwe True / False) --
        self.builder.position_at_end(bool_bb)
        true_bb = self.current_func.append_basic_block("pd.bool.true")
        false_bb = self.current_func.append_basic_block("pd.bool.false")
        cmp_bool = self.builder.icmp_signed("!=", pay, ir.Constant(I64, 0))
        self.builder.cbranch(cmp_bool, true_bb, false_bb)

        self.builder.position_at_end(true_bb)
        self.builder.call(self._printf, [self._str_ptr("True")])
        self.builder.branch(cont_bb)

        self.builder.position_at_end(false_bb)
        self.builder.call(self._printf, [self._str_ptr("False")])
        self.builder.branch(cont_bb)

        # -- STR (Obsługa REPR) --
        self.builder.position_at_end(str_bb)
        str_obj = self.builder.inttoptr(pay, STR_PTR)
        z_str = ir.Constant(I32, 0)
        sptr = self.builder.load(
            self.builder.gep(str_obj, [z_str, ir.Constant(I32, 3)], inbounds=True)
        )
        str_repr_bb = self.current_func.append_basic_block("pd.str.repr")
        str_norm_bb = self.current_func.append_basic_block("pd.str.norm")
        self.builder.cbranch(is_repr, str_repr_bb, str_norm_bb)

        self.builder.position_at_end(str_repr_bb)
        self.builder.call(self._printf, [self._str_ptr("'%s'"), sptr])
        self.builder.branch(cont_bb)

        self.builder.position_at_end(str_norm_bb)
        self.builder.call(self._printf, [self._str_ptr("%s"), sptr])
        self.builder.branch(cont_bb)

        # --- MAKRO: Wykrywanie Cykli (Zero Alokacji Sterty) ---
        def build_cycle_check(tag_name, cycle_str, real_body_bb):
            chk_cond = self.current_func.append_basic_block(f"pd.{tag_name}.chk")
            chk_body = self.current_func.append_basic_block(f"pd.{tag_name}.chk.b")

            curr_ptr = self.builder.alloca(I8P)
            self.builder.store(visited_node, curr_ptr)
            self.builder.branch(chk_cond)

            self.builder.position_at_end(chk_cond)
            curr = self.builder.load(curr_ptr)
            is_null = self.builder.icmp_signed(
                "==", self.builder.ptrtoint(curr, I64), ir.Constant(I64, 0)
            )
            self.builder.cbranch(is_null, real_body_bb, chk_body)

            self.builder.position_at_end(chk_body)
            node = self.builder.bitcast(curr, NODE_PTR)
            z = ir.Constant(I32, 0)
            n_pay = self.builder.load(
                self.builder.gep(node, [z, ir.Constant(I32, 0)], inbounds=True)
            )
            n_nxt = self.builder.load(
                self.builder.gep(node, [z, ir.Constant(I32, 1)], inbounds=True)
            )

            is_match = self.builder.icmp_signed("==", n_pay, pay)
            hit_bb = self.current_func.append_basic_block(f"pd.{tag_name}.hit")
            next_bb = self.current_func.append_basic_block(f"pd.{tag_name}.next")
            self.builder.cbranch(is_match, hit_bb, next_bb)

            self.builder.position_at_end(hit_bb)
            self.builder.call(self._printf, [self._str_ptr(cycle_str)])
            self.builder.branch(cont_bb)

            self.builder.position_at_end(next_bb)
            self.builder.store(n_nxt, curr_ptr)
            self.builder.branch(chk_cond)

            self.builder.position_at_end(real_body_bb)
            new_node = self.builder.alloca(NODE_TY)
            self.builder.store(
                pay, self.builder.gep(new_node, [z, ir.Constant(I32, 0)], inbounds=True)
            )
            self.builder.store(
                visited_node,
                self.builder.gep(new_node, [z, ir.Constant(I32, 1)], inbounds=True),
            )
            return self.builder.bitcast(new_node, I8P)

        # -- LIST --
        self.builder.position_at_end(lst_bb)
        l_real_bb = self.current_func.append_basic_block("pd.lst.real")
        v_lst = build_cycle_check("lst", "[...]", l_real_bb)

        lptr = self.builder.inttoptr(pay, LIST_PTR)
        z = ir.Constant(I32, 0)
        l_size = self.builder.load(
            self.builder.gep(lptr, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        l_data = self.builder.load(
            self.builder.gep(lptr, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        self.builder.call(self._printf, [self._str_ptr("[")])
        li_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), li_a)

        l_cond_bb = self.current_func.append_basic_block("pdl.cond")
        l_body_bb = self.current_func.append_basic_block("pdl.body")
        l_sep_bb = self.current_func.append_basic_block("pdl.sep")
        l_pr_bb = self.current_func.append_basic_block("pdl.pr")
        l_next_bb = self.current_func.append_basic_block("pdl.next")
        l_done_bb = self.current_func.append_basic_block("pdl.done")

        self.builder.branch(l_cond_bb)

        self.builder.position_at_end(l_cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(li_a), l_size),
            l_body_bb,
            l_done_bb,
        )

        self.builder.position_at_end(l_body_bb)
        is_f = self.builder.icmp_signed(
            "==", self.builder.load(li_a), ir.Constant(I64, 0)
        )
        self.builder.cbranch(is_f, l_pr_bb, l_sep_bb)

        self.builder.position_at_end(l_sep_bb)
        self.builder.call(self._printf, [self._str_ptr(", ")])
        self.builder.branch(l_pr_bb)

        self.builder.position_at_end(l_pr_bb)
        slot = self.builder.gep(l_data, [self.builder.load(li_a)], inbounds=True)
        ltag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        lpay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.call(func, [ltag, lpay, ir.Constant(I1, 1), v_lst])
        self.builder.branch(l_next_bb)

        self.builder.position_at_end(l_next_bb)
        self.builder.store(
            self.builder.add(self.builder.load(li_a), ir.Constant(I64, 1)), li_a
        )
        self.builder.branch(l_cond_bb)

        self.builder.position_at_end(l_done_bb)
        self.builder.call(self._printf, [self._str_ptr("]")])
        self.builder.branch(cont_bb)

        # -- TUPLE (identyczna struktura jak LIST, drukowana z () ) --
        self.builder.position_at_end(tup_bb)
        t_real_bb = self.current_func.append_basic_block("pd.tup.real")
        v_tup = build_cycle_check("tup", "(...)", t_real_bb)

        tptr = self.builder.inttoptr(pay, LIST_PTR)
        t_size = self.builder.load(
            self.builder.gep(tptr, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        t_data = self.builder.load(
            self.builder.gep(tptr, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        self.builder.call(self._printf, [self._str_ptr("(")])
        ti_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), ti_a)

        t_cond_bb = self.current_func.append_basic_block("pdt.cond")
        t_body_bb = self.current_func.append_basic_block("pdt.body")
        t_sep_bb = self.current_func.append_basic_block("pdt.sep")
        t_pr_bb = self.current_func.append_basic_block("pdt.pr")
        t_next_bb = self.current_func.append_basic_block("pdt.next")
        t_done_bb = self.current_func.append_basic_block("pdt.done")

        self.builder.branch(t_cond_bb)

        self.builder.position_at_end(t_cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(ti_a), t_size),
            t_body_bb,
            t_done_bb,
        )

        self.builder.position_at_end(t_body_bb)
        is_f_t = self.builder.icmp_signed(
            "==", self.builder.load(ti_a), ir.Constant(I64, 0)
        )
        self.builder.cbranch(is_f_t, t_pr_bb, t_sep_bb)

        self.builder.position_at_end(t_sep_bb)
        self.builder.call(self._printf, [self._str_ptr(", ")])
        self.builder.branch(t_pr_bb)

        self.builder.position_at_end(t_pr_bb)
        t_slot = self.builder.gep(t_data, [self.builder.load(ti_a)], inbounds=True)
        ttag = self.builder.load(
            self.builder.gep(t_slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        tpay = self.builder.load(
            self.builder.gep(t_slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.call(func, [ttag, tpay, ir.Constant(I1, 1), v_tup])
        self.builder.branch(t_next_bb)

        self.builder.position_at_end(t_next_bb)
        self.builder.store(
            self.builder.add(self.builder.load(ti_a), ir.Constant(I64, 1)), ti_a
        )
        self.builder.branch(t_cond_bb)

        self.builder.position_at_end(t_done_bb)
        # Python drukuje 1-el krotki z przecinkiem: (x,)
        is_one_el = self.builder.icmp_signed("==", t_size, ir.Constant(I64, 1))
        one_el_bb = self.current_func.append_basic_block("pdt.one_el")
        close_bb = self.current_func.append_basic_block("pdt.close")
        self.builder.cbranch(is_one_el, one_el_bb, close_bb)
        self.builder.position_at_end(one_el_bb)
        self.builder.call(self._printf, [self._str_ptr(",")])
        self.builder.branch(close_bb)
        self.builder.position_at_end(close_bb)
        self.builder.call(self._printf, [self._str_ptr(")")])
        self.builder.branch(cont_bb)

        # -- SET (identyczna struktura jak LIST, drukowana z {} ) --
        self.builder.position_at_end(set_bb)
        s_real_bb = self.current_func.append_basic_block("pd.set.real")
        v_set = build_cycle_check("set", "{...}", s_real_bb)

        setptr = self.builder.inttoptr(pay, LIST_PTR)
        s_size = self.builder.load(
            self.builder.gep(setptr, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        s_data = self.builder.load(
            self.builder.gep(setptr, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        self.builder.call(self._printf, [self._str_ptr("{")])
        si_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), si_a)

        s_cond_bb = self.current_func.append_basic_block("pds.cond")
        s_body_bb = self.current_func.append_basic_block("pds.body")
        s_sep_bb = self.current_func.append_basic_block("pds.sep")
        s_pr_bb = self.current_func.append_basic_block("pds.pr")
        s_next_bb = self.current_func.append_basic_block("pds.next")
        s_done_bb = self.current_func.append_basic_block("pds.done")

        self.builder.branch(s_cond_bb)

        self.builder.position_at_end(s_cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(si_a), s_size),
            s_body_bb,
            s_done_bb,
        )

        self.builder.position_at_end(s_body_bb)
        is_f_s = self.builder.icmp_signed(
            "==", self.builder.load(si_a), ir.Constant(I64, 0)
        )
        self.builder.cbranch(is_f_s, s_pr_bb, s_sep_bb)

        self.builder.position_at_end(s_sep_bb)
        self.builder.call(self._printf, [self._str_ptr(", ")])
        self.builder.branch(s_pr_bb)

        self.builder.position_at_end(s_pr_bb)
        s_slot = self.builder.gep(s_data, [self.builder.load(si_a)], inbounds=True)
        stag = self.builder.load(
            self.builder.gep(s_slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        spay = self.builder.load(
            self.builder.gep(s_slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.call(func, [stag, spay, ir.Constant(I1, 1), v_set])
        self.builder.branch(s_next_bb)

        self.builder.position_at_end(s_next_bb)
        self.builder.store(
            self.builder.add(self.builder.load(si_a), ir.Constant(I64, 1)), si_a
        )
        self.builder.branch(s_cond_bb)

        self.builder.position_at_end(s_done_bb)
        self.builder.call(self._printf, [self._str_ptr("}")])
        self.builder.branch(cont_bb)

        # -- DICT -- (Iteracja w insertion order via ordered_keys list)
        self.builder.position_at_end(dct_bb)
        d_real_bb = self.current_func.append_basic_block("pd.dct.real")
        v_dct = build_cycle_check("dct", "{...}", d_real_bb)

        dptr = self.builder.inttoptr(pay, DICT_PTR)
        # Get ordered_keys list from dict (index 4)
        ordered_list = self.builder.load(
            self.builder.gep(dptr, [z, ir.Constant(I32, 4)], inbounds=True), "d_ordered"
        )
        # LIST_TY: {GC_HEADER, size, cap, data_ptr}
        d_size = self.builder.load(
            self.builder.gep(ordered_list, [z, ir.Constant(I32, 1)], inbounds=True), "d_olen"
        )
        d_data = self.builder.load(
            self.builder.gep(ordered_list, [z, ir.Constant(I32, 3)], inbounds=True), "d_odata"
        )

        self.builder.call(self._printf, [self._str_ptr("{")])
        di_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), di_a)

        d_cond_bb = self.current_func.append_basic_block("pdd.cond")
        d_body_bb = self.current_func.append_basic_block("pdd.body")
        d_sep_bb = self.current_func.append_basic_block("pdd.sep")
        d_do_pr_bb = self.current_func.append_basic_block("pdd.dopr")
        d_next_bb = self.current_func.append_basic_block("pdd.next")
        d_done_bb = self.current_func.append_basic_block("pdd.done")

        self.builder.branch(d_cond_bb)
        self.builder.position_at_end(d_cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(di_a), d_size),
            d_body_bb,
            d_done_bb,
        )

        self.builder.position_at_end(d_body_bb)
        # Check if first item for comma separation
        is_fd = self.builder.icmp_signed(
            ">", self.builder.load(di_a), ir.Constant(I64, 0)
        )
        self.builder.cbranch(is_fd, d_sep_bb, d_do_pr_bb)

        self.builder.position_at_end(d_sep_bb)
        self.builder.call(self._printf, [self._str_ptr(", ")])
        self.builder.branch(d_do_pr_bb)

        self.builder.position_at_end(d_do_pr_bb)
        # Read the boxed key from ordered_keys list
        idx = self.builder.load(di_a)
        slot = self.builder.gep(d_data, [idx], inbounds=True)
        ktag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        kpay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )

        # Look up the value using dict_get_internal
        res_t = self.builder.alloca(I64, name="dget_rt")
        res_p = self.builder.alloca(I64, name="dget_rp")
        get_fn = self.functions["__py2llvm_dict_get_internal"]
        self.builder.call(get_fn, [dptr, ktag, kpay, res_t, res_p])

        vtag = self.builder.load(res_t, "dval_tag")
        vpay = self.builder.load(res_p, "dval_pay")

        self.builder.call(func, [ktag, kpay, ir.Constant(I1, 1), v_dct])
        self.builder.call(self._printf, [self._str_ptr(": ")])
        self.builder.call(func, [vtag, vpay, ir.Constant(I1, 1), v_dct])

        self.builder.branch(d_next_bb)

        self.builder.position_at_end(d_next_bb)
        self.builder.store(
            self.builder.add(self.builder.load(di_a), ir.Constant(I64, 1)), di_a
        )
        self.builder.branch(d_cond_bb)

        self.builder.position_at_end(d_done_bb)
        self.builder.call(self._printf, [self._str_ptr("}")])
        self.builder.branch(cont_bb)

        # -- INSTANCE (dataclass-like repr) --
        self.builder.position_at_end(inst_bb)
        inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
        z_inst = ir.Constant(I32, 0)
        # Read attrs dict pointer from instance (field index 2)
        attrs_ptr = self.builder.load(
            self.builder.gep(inst_ptr, [z_inst, ir.Constant(I32, 2)], inbounds=True),
            "inst_attrs"
        )
        # The attrs_ptr is a DICT_PTR; read its ordered_keys list (field index 4)
        ordered_list = self.builder.load(
            self.builder.gep(attrs_ptr, [z_inst, ir.Constant(I32, 4)], inbounds=True),
            "inst_olist"
        )
        # Read class name from dict's "__class__" key
        get_fn_inst = self.functions["__py2llvm_dict_get_internal"]
        cls_key_tag = ir.Constant(I64, Tag.STR)
        cls_key_pay = self.builder.ptrtoint(
            self.create_string("__class__").llvm, I64
        )
        cls_res_t = self.builder.alloca(I64, name="cls_rt")
        cls_res_p = self.builder.alloca(I64, name="cls_rp")
        self.builder.store(ir.Constant(I64, 0), cls_res_t)
        self.builder.store(ir.Constant(I64, 0), cls_res_p)
        self.builder.call(get_fn_inst, [attrs_ptr, cls_key_tag, cls_key_pay, cls_res_t, cls_res_p])
        cls_name_tag = self.builder.load(cls_res_t, "cls_nt")
        cls_name_pay = self.builder.load(cls_res_p, "cls_np")
        # Print class name
        is_cls_str = self.builder.icmp_signed("==", cls_name_tag, ir.Constant(I64, Tag.STR))
        inst_cls_bb = self.current_func.append_basic_block("pd.inst.cls")
        inst_nocls_bb = self.current_func.append_basic_block("pd.inst.nocls")
        self.builder.cbranch(is_cls_str, inst_cls_bb, inst_nocls_bb)
        self.builder.position_at_end(inst_cls_bb)
        cls_str_obj = self.builder.inttoptr(cls_name_pay, STR_PTR)
        cls_data = self.builder.load(
            self.builder.gep(cls_str_obj, [z_inst, ir.Constant(I32, 3)], inbounds=True),
            "cls_data"
        )
        self.builder.call(self._printf, [self._str_ptr("%s("), cls_data])
        self.builder.branch(inst_nocls_bb)
        self.builder.position_at_end(inst_nocls_bb)
        # Now iterate over ordered_keys and print key=value
        i_size = self.builder.load(
            self.builder.gep(ordered_list, [z_inst, ir.Constant(I32, 1)], inbounds=True),
            "inst_olen"
        )
        i_data = self.builder.load(
            self.builder.gep(ordered_list, [z_inst, ir.Constant(I32, 3)], inbounds=True),
            "inst_odata"
        )
        ii_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), ii_a)
        real_idx_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), real_idx_a)
        i_cond_bb = self.current_func.append_basic_block("pdi.cond")
        i_body_bb = self.current_func.append_basic_block("pdi.body")
        i_next_bb = self.current_func.append_basic_block("pdi.next")
        i_done_bb = self.current_func.append_basic_block("pdi.done")
        self.builder.branch(i_cond_bb)
        self.builder.position_at_end(i_cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(ii_a), i_size),
            i_body_bb, i_done_bb,
        )
        self.builder.position_at_end(i_body_bb)
        idx_i = self.builder.load(ii_a)
        kslot = self.builder.gep(i_data, [idx_i], inbounds=True)
        ktag_i = self.builder.load(
            self.builder.gep(kslot, [z_inst, ir.Constant(I32, 1)], inbounds=True)
        )
        kpay_i = self.builder.load(
            self.builder.gep(kslot, [z_inst, ir.Constant(I32, 2)], inbounds=True)
        )
        # Check if key is "__class__" — skip it
        is_cls_key = self.builder.icmp_signed("==", ktag_i, ir.Constant(I64, Tag.STR))
        skip_cls_bb = self.current_func.append_basic_block("pdi.skipcls")
        chk_cls_bb = self.current_func.append_basic_block("pdi.chkcls")
        self.builder.cbranch(is_cls_key, chk_cls_bb, skip_cls_bb)
        self.builder.position_at_end(chk_cls_bb)
        k_str_obj = self.builder.inttoptr(kpay_i, STR_PTR)
        k_str_data = self.builder.load(
            self.builder.gep(k_str_obj, [z_inst, ir.Constant(I32, 3)], inbounds=True)
        )
        cls_cmp_str = self.create_string("__class__")
        cls_cmp_data = self.builder.load(
            self.builder.gep(cls_cmp_str.llvm, [z_inst, ir.Constant(I32, 3)], inbounds=True)
        )
        cmp_res = self.builder.call(self._strcmp, [k_str_data, cls_cmp_data])
        is_cls_key_match = self.builder.icmp_signed("==", cmp_res, ir.Constant(I32, 0))
        # Also check for __frozen__ — skip it too
        frozen_cmp_str = self.create_string("__frozen__")
        frozen_cmp_data = self.builder.load(
            self.builder.gep(frozen_cmp_str.llvm, [z_inst, ir.Constant(I32, 3)], inbounds=True)
        )
        frozen_cmp_res = self.builder.call(self._strcmp, [k_str_data, frozen_cmp_data])
        is_frozen_key_match = self.builder.icmp_signed("==", frozen_cmp_res, ir.Constant(I32, 0))
        skip_key = self.builder.or_(is_cls_key_match, is_frozen_key_match)
        self.builder.cbranch(skip_key, i_next_bb, skip_cls_bb)
        self.builder.position_at_end(skip_cls_bb)
        real_i = self.builder.load(real_idx_a)
        is_first_attr = self.builder.icmp_signed("==", real_i, ir.Constant(I64, 0))
        attr_sep_bb = self.current_func.append_basic_block("pdi.sep")
        attr_nosep_bb = self.current_func.append_basic_block("pdi.nosep")
        self.builder.cbranch(is_first_attr, attr_nosep_bb, attr_sep_bb)
        self.builder.position_at_end(attr_sep_bb)
        self.builder.call(self._printf, [self._str_ptr(", ")])
        self.builder.branch(attr_nosep_bb)
        self.builder.position_at_end(attr_nosep_bb)
        # NAPRAWA: Nazwa pola bez cudzysłowów (is_repr=0), wartość z cudzysłowami (is_repr=1)
        self.builder.call(func, [ktag_i, kpay_i, ir.Constant(I1, 0), ir.Constant(I8P, None)])
        self.builder.call(self._printf, [self._str_ptr("=")])
        vres_t = self.builder.alloca(I64, name="pi_rt")
        vres_p = self.builder.alloca(I64, name="pi_rp")
        self.builder.call(get_fn_inst, [attrs_ptr, ktag_i, kpay_i, vres_t, vres_p])
        vtag_i = self.builder.load(vres_t, "pi_vt")
        vpay_i = self.builder.load(vres_p, "pi_vp")
        self.builder.call(func, [vtag_i, vpay_i, ir.Constant(I1, 1), ir.Constant(I8P, None)])
        self.builder.store(
            self.builder.add(self.builder.load(real_idx_a), ir.Constant(I64, 1)),
            real_idx_a
        )
        self.builder.branch(i_next_bb)
        self.builder.position_at_end(i_next_bb)
        self.builder.store(
            self.builder.add(self.builder.load(ii_a), ir.Constant(I64, 1)), ii_a
        )
        self.builder.branch(i_cond_bb)
        self.builder.position_at_end(i_done_bb)
        self.builder.call(self._printf, [self._str_ptr(")")])
        self.builder.branch(cont_bb)

        # -- NONE --
        self.builder.position_at_end(none_bb)
        self.builder.call(self._printf, [self._str_ptr("None")])
        self.builder.branch(cont_bb)

        self.builder.position_at_end(cont_bb)
        self.builder.ret_void()

        self.current_func = old_func
        self.builder = old_builder
        return func

    def _print_list(self, lst_val: Value):
        """Pomocnicza metoda dla typów statycznie rozpoznanych jako listy."""
        func = self._get_or_create_print_dyn()
        tag = ir.Constant(I64, Tag.LIST)
        pay = self.builder.ptrtoint(lst_val.llvm, I64)
        self.builder.call(func, [tag, pay])

    def _print_tag_pay(self, tag: ir.Value, pay: ir.Value):
        """Zastępuje potężne i powtarzalne bloki in-line wywołaniem zewnętrznej funkcji IR."""
        func = self._get_or_create_print_dyn()
        self.builder.call(func, [tag, pay])

    def _emit_print(self, args: List[Value], end_val: Value = None):
        for i, v in enumerate(args):
            if i > 0:
                self.builder.call(self._printf, [self._str_ptr(" ")])
            if v is not None:
                self.print_value(v)
            else:
                self.builder.call(self._printf, [self._str_ptr("None")])
        # FIX: Obsługa end= keyword (domyślnie "\n")
        if end_val is not None:
            if end_val.is_str:
                self._print_str_value(end_val)
            else:
                self._print_str_value(self.val_to_str(end_val))
        else:
            self.builder.call(self._printf, [self._str_ptr("\n")])

    def _print_str_value(self, s_val: Value):
        """Drukuje wartość string bez końcowego newline."""
        z = ir.Constant(I32, 0)
        data_ptr = self.builder.load(self.builder.gep(s_val.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        self.builder.call(self._printf, [data_ptr])

    # ──────────────────────────────────────────────────────────────
    #  Visitor dispatcher
    # ──────────────────────────────────────────────────────────────

