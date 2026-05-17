"""Iterator and generator runtime support."""

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


class IteratorsMixin:
    """Iterator and generator runtime support."""

    def _create_iterator(self, lst_val: Value) -> Value:
        """Tworzy obiekt iteratora z listy: ITER_DATA_TY = {GC_HEADER, list_ptr, idx}."""
        z = ir.Constant(I32, 0)
        
        # Pobierz wskaźnik listy - może być LIST_PTR lub OBJECT (boxed)
        if lst_val.is_object:
            tag, pay = self._read_slot(lst_val.llvm)
            list_ptr = self.builder.inttoptr(pay, LIST_PTR)
        elif lst_val.is_list:
            list_ptr = lst_val.llvm
        else:
            raise CompileError("Nie można utworzyć iteratora z tego typu.", None)
        
        # Alokuj ITER_DATA_TY
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_ITER)], "iter.raw")
        iter_obj = self.builder.bitcast(raw, ITER_DATA_PTR, "iter_obj")
        # Init GC_HEADER
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(iter_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True))  # refcnt
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(iter_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 1)], inbounds=True))  # color
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(iter_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 2)], inbounds=True))  # temp_refcnt
        self.builder.store(null_i8p, self.builder.gep(iter_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 3)], inbounds=True))  # gc_next
        # Zapisz list_ptr i idx=0
        self.builder.store(list_ptr, self.builder.gep(iter_obj, [z, ir.Constant(I32, 1)], inbounds=True))  # list_ptr
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(iter_obj, [z, ir.Constant(I32, 2)], inbounds=True))  # idx=0
        # Zwróć jako boxed OBJECT z tag=ITERATOR
        boxed_raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)], "iter_box.raw")
        bv = self.builder.bitcast(boxed_raw, BOXED_PTR, "iter_bv")
        # Init GC_HEADER boxed
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(bv, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(bv, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(bv, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(null_i8p, self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True))
        # tag = ITERATOR, payload = ptrtoint(iter_obj)
        self.builder.store(ir.Constant(I64, Tag.ITERATOR), self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(self.builder.ptrtoint(iter_obj, I64), self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True))
        return Value(bv, PyType.OBJECT)

    def _builtin_next(self, iter_val: Value, node=None) -> Value:
        """Implementacja next(iterator) – zwraca kolejny element z iteratora."""
        z = ir.Constant(I32, 0)
        # Odczytaj tag i payload z boxed value
        if iter_val.is_object:
            tag, pay = self._read_slot(iter_val.llvm)
        elif iter_val.is_iterator:
            tag = ir.Constant(I64, Tag.ITERATOR)
            pay = self.builder.ptrtoint(iter_val.llvm, I64)
        else:
            # Dla list – utwórz iterator i kontynuuj
            if iter_val.is_list:
                return self._builtin_next(self._create_iterator(iter_val), node)
            raise CompileError("next() wymaga iteratora lub listy.", node)

        # NAPRAWA: Alokacja res_alloca MUSI być w bloku dominującym wszystkie
        # użycia (ok_bb, stop_bb, not_iter_bb, end_bb). Tworzymy ją PRZED
        # rozgałęzieniem, aby spełnić warunek dominacji LLVM.
        res_alloca = self.builder.alloca(BOXED_PTR, name="next_res")

        # Sprawdź tag == ITERATOR
        is_iter = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.ITERATOR))
        not_iter_bb = self.current_func.append_basic_block("next.not_iter")
        iter_bb = self.current_func.append_basic_block("next.iter")
        end_bb = self.current_func.append_basic_block("next.end")
        self.builder.cbranch(is_iter, iter_bb, not_iter_bb)

        # Iterator path
        self.builder.position_at_end(iter_bb)
        iter_data_ptr = self.builder.inttoptr(pay, ITER_DATA_PTR)
        list_ptr = self.builder.load(self.builder.gep(iter_data_ptr, [z, ir.Constant(I32, 1)], inbounds=True), "iter_list")
        idx_val = self.builder.load(self.builder.gep(iter_data_ptr, [z, ir.Constant(I32, 2)], inbounds=True), "iter_idx")

        # Pobierz element z listy pod indeksem idx_val
        sp, _, dp = self._list_ptrs(list_ptr)
        list_size = self.builder.load(sp, "iter_sz")
        list_data = self.builder.load(dp, "iter_data")

        # Sprawdź czy idx < size
        in_bounds = self.builder.icmp_signed("<", idx_val, list_size)
        ok_bb = self.current_func.append_basic_block("next.ok")
        stop_bb = self.current_func.append_basic_block("next.stop")
        self.builder.cbranch(in_bounds, ok_bb, stop_bb)

        # OK: pobierz element i inkrementuj indeks
        self.builder.position_at_end(ok_bb)
        slot = self.builder.gep(list_data, [idx_val], inbounds=True)
        slot_boxed = self.builder.bitcast(slot, BOXED_PTR)
        etag = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
        epay = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
        elem = self._boxed_to_value(etag, epay, node)

        # Inkrementuj indeks w iteratorze
        new_idx = self.builder.add(idx_val, ir.Constant(I64, 1))
        self.builder.store(new_idx, self.builder.gep(iter_data_ptr, [z, ir.Constant(I32, 2)], inbounds=True))

        # Zwróć element (rezultat zapisz w alloca)
        if elem.is_object:
            self.builder.store(elem.llvm, res_alloca)
        else:
            self.builder.store(self._box(elem), res_alloca)
        self.builder.branch(end_bb)

        # StopIteration
        self.builder.position_at_end(stop_bb)
        # Zwróć None
        none_boxed = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
        self.builder.store(none_boxed, res_alloca)
        self.builder.branch(end_bb)

        # Not an iterator path
        self.builder.position_at_end(not_iter_bb)
        # Spróbuj potraktować jako listę (utwórz tymczasowy iterator)
        # Dla uproszczenia, zwróć None
        none_boxed2 = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
        self.builder.store(none_boxed2, res_alloca)
        self.builder.branch(end_bb)

        self.builder.position_at_end(end_bb)
        return Value(self.builder.load(res_alloca), PyType.OBJECT)

    def _builtin_iter(self, iterable_val: Value, node=None) -> Value:
        """Implementacja iter(iterable) – tworzy iterator z iterowalnego obiektu."""
        if iterable_val.is_list:
            return self._create_iterator(iterable_val)
        if iterable_val.is_object:
            # Sprawdź czy to już jest iterator
            tag, _ = self._read_slot(iterable_val.llvm)
            is_iter = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.ITERATOR))
            # Jeśli tak, zwróć jak jest
            iter_bb = self.current_func.append_basic_block("bi.iter")
            list_bb = self.current_func.append_basic_block("bi.list")
            end_bb = self.current_func.append_basic_block("bi.end")
            res_a = self.builder.alloca(BOXED_PTR, name="bi_res")
            self.builder.cbranch(is_iter, iter_bb, list_bb)
            self.builder.position_at_end(iter_bb)
            self.builder.store(iterable_val.llvm, res_a)
            self.builder.branch(end_bb)
            self.builder.position_at_end(list_bb)
            new_iter = self._create_iterator(iterable_val)
            self.builder.store(new_iter.llvm, res_a)
            self.builder.branch(end_bb)
            self.builder.position_at_end(end_bb)
            return Value(self.builder.load(res_a), PyType.OBJECT)
        raise CompileError("iter() wymaga iterowalnego obiektu.", node)

    # ──────────────────────────────────────────────────────────────
    #  Publiczne API
    # ──────────────────────────────────────────────────────────────

