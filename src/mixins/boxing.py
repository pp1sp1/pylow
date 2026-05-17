"""Boxing/unboxing: converting between typed Values and boxed BOXED_PTR."""

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


class BoxingMixin:
    """Boxing/unboxing: converting between typed Values and boxed BOXED_PTR."""

    def _value_to_tag_payload(self, v: Value) -> Tuple[ir.Value, ir.Value]:
        """Zwraca (tag:i64, payload:i64). Dla OBJECT odczytuje z pamięci."""
        # FFIModuleValue — nie ma reprezentacji runtime, nie można printować
        if getattr(v, 'is_ffi_module', False):
            raise CompileError("Nie można wydrukować referencji do modułu — czy zapomniałeś '()'?")
        if v.is_object:
            # FIX: Jeśli val.llvm nie jest BOXED_PTR (np. wskaźnik funkcji z lambda),
            # konwertuj na tag INST + payload = ptrtoint(func, i64).
            if v.llvm.type != BOXED_PTR:
                return ir.Constant(I64, Tag.INST), self.builder.ptrtoint(v.llvm, I64)
            tag, pay = self._read_slot(v.llvm)
            return tag, pay
        if v.is_instance:
            # Instance: tag = Tag.INST, payload = instance pointer as i64
            return ir.Constant(I64, Tag.INST), self.builder.ptrtoint(v.llvm, I64)
        if v.is_int:
            return ir.Constant(I64, Tag.INT), v.llvm
        if v.is_float:
            return ir.Constant(I64, Tag.FLOAT), self.builder.bitcast(v.llvm, I64)
        if v.is_bool:
            return ir.Constant(I64, Tag.BOOL), self.builder.zext(v.llvm, I64)
        if v.is_str:
            return ir.Constant(I64, Tag.STR), self.builder.ptrtoint(v.llvm, I64)
        if v.is_list:
            return ir.Constant(I64, Tag.LIST), self.builder.ptrtoint(v.llvm, I64)
        if v.is_tuple:
            return ir.Constant(I64, Tag.TUPLE), self.builder.ptrtoint(v.llvm, I64)
        if v.is_set:
            return ir.Constant(I64, Tag.SET), self.builder.ptrtoint(v.llvm, I64)
        if v.is_dict:
            return ir.Constant(I64, Tag.DICT), self.builder.ptrtoint(v.llvm, I64)
        if v.is_iterator:
            return ir.Constant(I64, Tag.ITERATOR), self.builder.ptrtoint(v.llvm, I64)
        if v.is_task:
            return ir.Constant(I64, Tag.TASK), self.builder.ptrtoint(v.llvm, I64)
        if v.is_coroutine:
            return ir.Constant(I64, Tag.COROUTINE), self.builder.ptrtoint(v.llvm, I64)
        if v.is_none:
            return ir.Constant(I64, Tag.NONE), ir.Constant(I64, 0)
        return ir.Constant(I64, Tag.INT), v.llvm

    def _box(self, v: Value) -> ir.Value:
        """Pakuje wartość do BoxedValue* = {GC_HEADER, tag:i64, payload:i64}."""
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)], "box.raw")
        bv = self.builder.bitcast(raw, BOXED_PTR, "bv")
        z = ir.Constant(I32, 0)
        # Init GC_HEADER: refcnt=1, color=Black, temp_refcnt=0, gc_next=null
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )  # refcnt=1
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )  # color=Black
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(bv, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )  # temp_refcnt=0
        self.builder.store(
            null_i8p, self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True)
        )  # gc_next=null
        tp = self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True)
        pp = self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True)
        tag, pay = self._value_to_tag_payload(v)
        self.builder.store(tag, tp)
        self.builder.store(pay, pp)
        # If payload is a heap object (STR/LIST/TUPLE/SET/DICT), incref it - new box holds a reference
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
        inc_bb = self.current_func.append_basic_block("box.inc")
        skip_bb = self.current_func.append_basic_block("box.skip")
        self.builder.cbranch(is_heap, inc_bb, skip_bb)
        self.builder.position_at_end(inc_bb)
        self.builder.call(
            self.functions["__py2llvm_incref"], [self.builder.inttoptr(pay, I8P)]
        )
        self.builder.branch(skip_bb)
        self.builder.position_at_end(skip_bb)
        return bv

    def _unbox(
        self, tag: ir.Value, payload: ir.Value, expected: PyType = PyType.OBJECT
    ) -> Value:
        """Konwertuje (tag:i64, payload:i64) → Value (statycznie)."""
        if expected == PyType.INT:
            return Value(payload, PyType.INT)
        if expected == PyType.FLOAT:
            return Value(self.builder.bitcast(payload, F64), PyType.FLOAT)
        if expected == PyType.BOOL:
            return Value(self.builder.trunc(payload, I1), PyType.BOOL)
        if expected == PyType.STR:
            return Value(self.builder.inttoptr(payload, I8P), PyType.STR)
        if expected == PyType.LIST:
            return Value(self.builder.inttoptr(payload, LIST_PTR), PyType.LIST)
        if expected == PyType.DICT:
            return Value(self.builder.inttoptr(payload, DICT_PTR), PyType.DICT)
        return Value(payload, PyType.OBJECT)

    def _write_slot(self, slot_ptr: ir.Value, v: Value):
        """Zapisuje wartość do slotu BoxedValue (nie allokuje nowej pamięci)."""
        z = ir.Constant(I32, 0)
        tp = self.builder.gep(slot_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
        pp = self.builder.gep(slot_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
        tag, pay = self._value_to_tag_payload(v)
        self.builder.store(tag, tp)
        self.builder.store(pay, pp)

    def _read_slot(self, slot_ptr):
        # NAPRAWA: Jeśli slot_ptr to i64 (surowy adres), musimy rzutować na wskaźnik
        if slot_ptr.type == I64:
            slot_ptr = self.builder.inttoptr(slot_ptr, BOXED_PTR)

        z = ir.Constant(I32, 0)
        tag = self.builder.load(
            self.builder.gep(slot_ptr, [z, ir.Constant(I32, 1)], inbounds=True), "tag"
        )
        pay = self.builder.load(
            self.builder.gep(slot_ptr, [z, ir.Constant(I32, 2)], inbounds=True), "pay"
        )
        return tag, pay

    # ──────────────────────────────────────────────────────────────
    #  LIST runtime
    # ──────────────────────────────────────────────────────────────

    def _boxed_to_value(self, tag: ir.Value, payload: ir.Value, node: ast.AST) -> Value:
        """
        Konwertuje tag i payload z BOXED na Value.
        Zwraca jako OBJECT (BOXED_PTR), z poprawnym tagiem i payloadem
        w alokacji stosowej. Odbiorcy mogą odczytać tag przez _read_slot.
        """
        z = ir.Constant(I32, 0)

        boxed_alloca = self.builder.alloca(BOXED_TY, name="temp_boxed")
        tag_ptr = self.builder.gep(
            boxed_alloca, [z, ir.Constant(I32, 1)], inbounds=True
        )
        pay_ptr = self.builder.gep(
            boxed_alloca, [z, ir.Constant(I32, 2)], inbounds=True
        )
        self.builder.store(tag, tag_ptr)
        self.builder.store(payload, pay_ptr)

        return Value(boxed_alloca, PyType.OBJECT)

    # ══════════════════════════════════════════════════════════════════
    #  POPRAWKA 5: Exception handling as classes
    # ══════════════════════════════════════════════════════════════════

    def _create_exception_object(
        self, exc_type: str, args: list, node: ast.AST
    ) -> Value:
        """
        Tworzy obiekt wyjątku.
        W uproszczonej implementacji wyjątki to struktury zawierające
        typ i komunikat.
        """
        z = ir.Constant(I32, 0)

        # Komunikat błędu - pierwszy argument lub nazwa typu
        if args:
            msg_val = self.val_to_str(args[0])
        else:
            msg_val = self.create_string(exc_type)

        # Utwórz prostą strukturę wyjątku
        # W pełnej implementacji to byłaby instancja klasy wyjątku
        exc_struct_ty = ir.LiteralStructType(
            [
                I64,  # type_tag (identyfikator typu wyjątku)
                STR_PTR,  # message
            ]
        )

        exc_ptr = self.builder.alloca(exc_struct_ty, name=f"exc_{exc_type}")

        # Zapisz tag typu
        type_tag = ir.Constant(I64, hash(exc_type) & 0xFFFFFFFFFFFFFFFF)
        self.builder.store(
            type_tag, self.builder.gep(exc_ptr, [z, ir.Constant(I32, 0)], inbounds=True)
        )

        # Zapisz komunikat
        self.builder.store(
            msg_val.llvm,
            self.builder.gep(exc_ptr, [z, ir.Constant(I32, 1)], inbounds=True),
        )

        # Zwróć jako wskaźnik w BOXED
        boxed = self._box_exception(exc_ptr, exc_type)
        return boxed

    def _box_exception(self, exc_ptr: ir.Value, exc_type: str) -> Value:
        """
        Opakowuje wskaźnik do wyjątku w BOXED.
        Używa Tag.STR z wiadomością jako payload – dzięki temu print(e) drukuje komunikat.
        NAPRAWA: Zamiast odczytywać z tymczasowej struktury na stosie, boxujemy
        msg_val bezpośrednio (msg_val to STR_PTR, bezpieczny na stercie).
        """
        z = ir.Constant(I32, 0)

        # Odczytaj wskaźnik do wiadomości ze struktury wyjątku
        # exc_struct_ty = { i64 type_tag, STR_PTR message }
        msg_ptr = self.builder.gep(exc_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
        msg_str_ptr = self.builder.load(msg_ptr, "exc_msg")

        # NAPRAWA: Użyj standardowego _box zamiast ręcznego malloc+store
        # _box poprawnie ustawia GC header, tag i payload
        msg_value = Value(msg_str_ptr, PyType.STR)
        boxed = self._box(msg_value)

        return Value(boxed, PyType.OBJECT)

    # ══════════════════════════════════════════════════════════════════
    #  POPRAWKA 6: super() function
    # ══════════════════════════════════════════════════════════════════

    def _create_super_object(self, node: ast.AST) -> Value:
        """
        Tworzy obiekt super() do wywoływania metod z klasy bazowej.
        """
        z = ir.Constant(I32, 0)

        if not hasattr(self, "_class_stack") or not self._class_stack:
            raise CompileError("super() musi być użyte wewnątrz metody klasy", node)

        current_class_info = self._class_stack[-1]

        # W uproszczeniu: super zwraca specjalny obiekt, który przy
        # wywołaniu metody szuka w klasie bazowej

        # Struktura super: {class_ptr, instance_ptr}
        super_ty = ir.LiteralStructType([CLASS_PTR, INSTANCE_PTR])
        super_ptr = self.builder.alloca(super_ty, name="super_obj")

        # Pobierz wskaźnik do bieżącej klasy
        class_name = current_class_info.get("name", "")
        if class_name in self._compiled_classes:
            class_ptr = self._compiled_classes[class_name]
        else:
            class_ptr = ir.Constant(CLASS_PTR, None)

        self.builder.store(
            class_ptr,
            self.builder.gep(super_ptr, [z, ir.Constant(I32, 0)], inbounds=True),
        )

        # Pobierz wskaźnik do instancji (self)
        if "self" in [name for name in self.sym._stack[-1] if True]:
            try:
                self_info = self.sym.lookup("self")
                self.builder.store(
                    self.builder.load(self_info.alloca),
                    self.builder.gep(
                        super_ptr, [z, ir.Constant(I32, 1)], inbounds=True
                    ),
                )
            except CompileError:
                self.builder.store(
                    ir.Constant(INSTANCE_PTR, None),
                    self.builder.gep(
                        super_ptr, [z, ir.Constant(I32, 1)], inbounds=True
                    ),
                )
        else:
            self.builder.store(
                ir.Constant(INSTANCE_PTR, None),
                self.builder.gep(super_ptr, [z, ir.Constant(I32, 1)], inbounds=True),
            )

        # Zwróć jako BOXED z tagiem SUPER (np. 50)
        boxed = self.builder.alloca(BOXED_TY, name="boxed_super")
        self.builder.call(
            self._memset,
            [
                self.builder.bitcast(boxed, I8P),
                ir.Constant(I32, 0),
                ir.Constant(I64, SZ_GC_HEADER),
            ],
        )

        tag_ptr = self.builder.gep(boxed, [z, ir.Constant(I32, 1)], inbounds=True)
        self.builder.store(ir.Constant(I64, 50), tag_ptr)  # Tag.SUPER = 50

        pay_ptr = self.builder.gep(boxed, [z, ir.Constant(I32, 2)], inbounds=True)
        self.builder.store(self.builder.ptrtoint(super_ptr, I64), pay_ptr)

        return Value(boxed, PyType.OBJECT)

    # ══════════════════════════════════════════════════════════════════
    #  DODATKOWE WBUDOWANE FUNKCJE - implementacje
    # ══════════════════════════════════════════════════════════════════

