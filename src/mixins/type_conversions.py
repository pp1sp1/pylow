"""Type conversion utilities: casting between Python types."""

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
from ..exceptions import CompileError, PylowError
from ..reporter import ErrorCategory, ErrorLevel
from ..symbols import VarInfo, SymbolTable
from ..values import Value
from ..type_analyzer import StaticTypeAnalyzer

if TYPE_CHECKING:
    pass


class TypeConversionsMixin:
    """Type conversion utilities: casting between Python types."""

    def _to_float(self, v: Value) -> Value:
        if v.is_float:
            return v
        if v.is_bool:
            return Value(
                self.builder.sitofp(self.builder.zext(v.llvm, I64), F64), PyType.FLOAT
            )
        if v.is_int:
            return Value(self.builder.sitofp(v.llvm, F64), PyType.FLOAT)
        if v.is_object:
            tag, pay = self._read_slot(v.llvm)
            res = self.builder.alloca(F64, name="toflt")
            int_bb = self.current_func.append_basic_block("toflt.int")
            bool_bb = self.current_func.append_basic_block("toflt.bool")
            flt_bb = self.current_func.append_basic_block("toflt.flt")
            err_bb = self.current_func.append_basic_block("toflt.err")
            end_bb = self.current_func.append_basic_block("toflt.end")
            sw = self.builder.switch(tag, err_bb)
            sw.add_case(ir.Constant(I64, Tag.INT), int_bb)
            sw.add_case(ir.Constant(I64, Tag.BOOL), bool_bb)
            sw.add_case(ir.Constant(I64, Tag.FLOAT), flt_bb)

            self.builder.position_at_end(int_bb)
            self.builder.store(self.builder.sitofp(pay, F64), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(bool_bb)
            self.builder.store(
                self.builder.sitofp(
                    self.builder.zext(self.builder.trunc(pay, I1), I64), F64
                ),
                res,
            )
            self.builder.branch(end_bb)

            self.builder.position_at_end(flt_bb)
            self.builder.store(self.builder.bitcast(pay, F64), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(err_bb)
            self.builder.store(ir.Constant(F64, 0.0), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(end_bb)
            return Value(self.builder.load(res), PyType.FLOAT)
        raise self._error(
            ErrorCategory.TYPE,
            f"Cannot convert {v.pytype.name} to FLOAT",
            hint=f"Expected a numeric type, got {v.pytype.name}",
            help_text="Use an explicit cast: float(x) converts int or bool to float.",
        )

    def _to_int(self, v: Value) -> Value:
        if v.is_int:
            return v
        if v.is_bool:
            return Value(self.builder.zext(v.llvm, I64), PyType.INT)
        if v.is_float:
            return Value(self.builder.fptosi(v.llvm, I64), PyType.INT)
        if v.is_object:
            tag, pay = self._read_slot(v.llvm)
            res = self.builder.alloca(I64, name="toint")
            int_bb = self.current_func.append_basic_block("toint.int")
            bool_bb = self.current_func.append_basic_block("toint.bool")
            flt_bb = self.current_func.append_basic_block("toint.flt")
            err_bb = self.current_func.append_basic_block("toint.err")
            end_bb = self.current_func.append_basic_block("toint.end")
            sw = self.builder.switch(tag, err_bb)
            sw.add_case(ir.Constant(I64, Tag.INT), int_bb)
            sw.add_case(ir.Constant(I64, Tag.BOOL), bool_bb)
            sw.add_case(ir.Constant(I64, Tag.FLOAT), flt_bb)

            self.builder.position_at_end(int_bb)
            self.builder.store(pay, res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(bool_bb)
            self.builder.store(self.builder.zext(self.builder.trunc(pay, I1), I64), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(flt_bb)
            self.builder.store(
                self.builder.fptosi(self.builder.bitcast(pay, F64), I64), res
            )
            self.builder.branch(end_bb)

            self.builder.position_at_end(err_bb)
            self.builder.store(ir.Constant(I64, 0), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(end_bb)
            return Value(self.builder.load(res), PyType.INT)
        raise self._error(
            ErrorCategory.TYPE,
            f"Cannot convert {v.pytype.name} to INT",
            hint=f"Expected a numeric type, got {v.pytype.name}",
            help_text="Use an explicit cast: int(x) converts float or bool to int.",
        )

    def _to_bool_val(self, v: Value) -> Value:
        if v.is_bool:
            return v
        if v.is_int:
            return Value(
                self.builder.icmp_signed("!=", v.llvm, ir.Constant(I64, 0)), PyType.BOOL
            )
        if v.is_float:
            return Value(
                self.builder.fcmp_ordered("!=", v.llvm, ir.Constant(F64, 0.0)),
                PyType.BOOL,
            )
        if v.is_str:
            z = ir.Constant(I32, 0)
            sz = self.builder.load(self.builder.gep(v.llvm, [z, z], inbounds=True))
            return Value(
                self.builder.icmp_signed("!=", sz, ir.Constant(I64, 0)), PyType.BOOL
            )
        if v.is_object:
            tag, pay = self._read_slot(v.llvm)
            res = self.builder.alloca(I1, name="tobool")
            int_bb = self.current_func.append_basic_block("tobool.int")
            bool_bb = self.current_func.append_basic_block("tobool.bool")
            flt_bb = self.current_func.append_basic_block("tobool.flt")
            ptr_bb = self.current_func.append_basic_block("tobool.ptr")
            none_bb = self.current_func.append_basic_block("tobool.none")
            end_bb = self.current_func.append_basic_block("tobool.end")

            sw = self.builder.switch(tag, none_bb)
            sw.add_case(ir.Constant(I64, Tag.INT), int_bb)
            sw.add_case(ir.Constant(I64, Tag.BOOL), bool_bb)
            sw.add_case(ir.Constant(I64, Tag.FLOAT), flt_bb)
            sw.add_case(ir.Constant(I64, Tag.STR), ptr_bb)
            sw.add_case(ir.Constant(I64, Tag.LIST), ptr_bb)
            sw.add_case(ir.Constant(I64, Tag.DICT), ptr_bb)

            self.builder.position_at_end(int_bb)
            self.builder.store(
                self.builder.icmp_signed("!=", pay, ir.Constant(I64, 0)), res
            )
            self.builder.branch(end_bb)

            self.builder.position_at_end(bool_bb)
            self.builder.store(self.builder.trunc(pay, I1), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(flt_bb)
            self.builder.store(
                self.builder.fcmp_ordered(
                    "!=", self.builder.bitcast(pay, F64), ir.Constant(F64, 0.0)
                ),
                res,
            )
            self.builder.branch(end_bb)

            self.builder.position_at_end(ptr_bb)
            self.builder.store(
                self.builder.icmp_signed("!=", pay, ir.Constant(I64, 0)), res
            )
            self.builder.branch(end_bb)

            self.builder.position_at_end(none_bb)
            self.builder.store(ir.Constant(I1, 0), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(end_bb)
            return Value(self.builder.load(res), PyType.BOOL)
        raise self._error(
            ErrorCategory.TYPE,
            f"Cannot convert {v.pytype.name} to BOOL",
            hint=f"Expected a truthy type, got {v.pytype.name}",
            help_text="Use an explicit cast: bool(x) converts to a boolean value.",
        )

    # ══════════════════════════════════════════════════════════════════
    #  POPRAWKA 4: all() builtin function implementation
    # ══════════════════════════════════════════════════════════════════

    def _coerce(self, a: Value, b: Value) -> Tuple[Value, Value]:
        if a.is_float or b.is_float:
            return self._to_float(a), self._to_float(b)
        if a.is_bool:
            a = self._to_int(a)
        if b.is_bool:
            b = self._to_int(b)
        return a, b

    def _cast_to_llvm(self, v: Value, target: ir.Type, node=None) -> Value:
        if v.llvm.type == target:
            return v
        if target == BOXED_PTR:
            if v.is_object:
                return v
            return Value(self._box(v), PyType.OBJECT)
        if target == F64:
            return self._to_float(v)
        if target == I64:
            return self._to_int(v)
        if target == I1:
            return self._to_bool_val(v)
        if target == I32:
            vi = self._to_int(v)
            return Value(self.builder.trunc(vi.llvm, I32), PyType.INT)
        return v

    def _llvm_type_to_pytype(self, t: ir.Type) -> PyType:
        # NAPRAWA: Helper do MRO (Method Resolution Order) - szukanie klas bazowych
        if not hasattr(self, '_mro_cache'):
            self._mro_cache = {}
        return self._llvm_type_to_pytype_real(t)

    def _get_mro(self, class_name: str, hierarchy: dict) -> list:
        """Zwraca listę klas bazowych w kolejności MRO (C3 linearization uproszczone).

        Dla class C(A, B) gdzie A(Base), B(Base), poprawna kolejność to:
        [A, B, Base] — czyli breadth-first z deduplikacją, co odpowiada
        Pythonowemu C3 linearization dla prostych hierarchii.

        Poprzednia implementacja robiła DFS, co dawało [Base, A, Base, B]
        (z duplikatami i złą kolejnością).
        """
        # NAPRAWA: Defensive initialization — _mro_cache might not exist
        # if _get_mro is called before _llvm_type_to_pytype or if
        # the compiler instance was created without __init__ initialization
        if not hasattr(self, '_mro_cache'):
            self._mro_cache = {}
        if class_name in self._mro_cache:
            return self._mro_cache[class_name]
        result = []
        visited = set()
        # BFS po hierarchii klas — gwarantuje prawidłową kolejność MRO
        queue = list(hierarchy.get(class_name, []))
        # Dodaj bezpośrednie bazy jako kandydatów
        i = 0
        while i < len(queue):
            cls = queue[i]
            i += 1
            if cls in visited:
                continue
            visited.add(cls)
            result.append(cls)
            # Dodaj bazy tej klasy do kolejki (na koniec — BFS)
            for base in hierarchy.get(cls, []):
                if base not in visited:
                    queue.append(base)
        self._mro_cache[class_name] = result
        return result

    def _llvm_type_to_pytype_real(self, t: ir.Type) -> PyType:
        if t == I64:
            return PyType.INT
        if t == F64:
            return PyType.FLOAT
        if t == I1:
            return PyType.BOOL
        if t == I8P:
            return PyType.STR
        if t == LIST_PTR:
            return PyType.LIST
        if t == DICT_PTR:
            return PyType.DICT
        if t == INSTANCE_PTR:
            return PyType.INSTANCE
        if t == BOXED_PTR:
            return PyType.OBJECT
        return PyType.OBJECT

    # ──────────────────────────────────────────────────────────────
    #  Boxing: Value → BoxedValue* (BOXED_PTR)
    # ──────────────────────────────────────────────────────────────

