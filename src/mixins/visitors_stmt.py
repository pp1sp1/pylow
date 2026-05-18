"""AST visitor methods for statement nodes."""

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


class VisitorsStmtMixin:
    """AST visitor methods for statement nodes."""

    def _is_class_ref_value(self, val) -> bool:
        """Check if a Value represents a class reference (e.g., device_cls = SmartDevice)
        rather than an instance (e.g., device = SmartDevice("Lamp")).
        A class reference comes from visit_Name when the name matches __is_class_{name}."""
        if not (hasattr(val, 'is_object') and val.is_object):
            return False
        if not (hasattr(val, 'class_name') and val.class_name):
            return False
        # A value is a class reference if it was produced by visit_Name
        # for a name that matches __is_class_{name} — in that case,
        # val.llvm is ir.Constant(CLASS_PTR, None).
        # We check if the LLVM value is a constant (class references are
        # CLASS_PTR constants, while instances are heap-allocated BOXED_PTRs).
        try:
            if isinstance(val.llvm, ir.Constant):
                # Check if it's a CLASS_PTR constant (null pointer = class ref marker)
                if hasattr(val.llvm, 'type') and val.llvm.type == CLASS_PTR:
                    return True
        except Exception:
            pass
        return False

    def visit_Assign(self, node: ast.Assign):
        val = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # FIX: Jeśli wartość to funkcja (lambda), zarejestruj ją pod
                # nazwą zmiennej, aby wywołania add(5,3) mogły ją znaleźć.
                if val.is_object and isinstance(val.llvm, ir.Function):
                    func = val.llvm
                    self.functions[target.id] = func
                    self.functions[f"py_{target.id}"] = func
                    self._function_ast[target.id] = node.value  # AST dla inliningu
                self._assign_name(target.id, val)
            elif isinstance(target, ast.Subscript):
                obj = self.visit(target.value)
                key = self.visit(target.slice)
                if obj.is_dict:
                    self.dict_setitem(obj, key, val)
                elif obj.is_object:
                    tag, pay = self._read_slot(obj.llvm)
                    dct_bb = self.current_func.append_basic_block("asub.dct")
                    err_bb = self.current_func.append_basic_block("asub.err")
                    end_bb = self.current_func.append_basic_block("asub.end")
                    sw = self.builder.switch(tag, err_bb)
                    sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
                    self.builder.position_at_end(dct_bb)
                    dptr = self.builder.inttoptr(pay, DICT_PTR)
                    self.dict_setitem(Value(dptr, PyType.DICT), key, val)
                    self.builder.branch(end_bb)
                    self.builder.position_at_end(err_bb)
                    self.builder.branch(end_bb)
                    self.builder.position_at_end(end_bb)
                else:
                    raise self._error(
                        ErrorCategory.SEMANTIC,
                        "Indexed assignment is only supported for dict",
                        node,
                        help_text="Use dict[key] = value to assign to a dictionary.",
                    )
            elif isinstance(target, ast.Attribute):
                # Handle attribute assignment like self.x = value
                obj = self.visit(target.value)
                attr_name = target.attr

                # NAPRAWA: Check if this attribute is a property with a setter
                inferred_class = obj.class_name if hasattr(obj, 'class_name') else None
                if not inferred_class and hasattr(target, 'value') and isinstance(target.value, ast.Name):
                    var_name = target.value.id
                    try:
                        var_info = self.sym.lookup(var_name)
                        if hasattr(var_info, "class_name") and var_info.class_name:
                            inferred_class = var_info.class_name
                    except: pass

                if inferred_class:
                    class_props = getattr(self, '_class_properties', {}).get(inferred_class, {})
                    if attr_name in class_props and 'setter' in class_props[attr_name]:
                        setter_name = class_props[attr_name]['setter']
                        if setter_name in self.functions:
                            setter_func = self.functions[setter_name]
                            # Prepare self argument based on obj type
                            first_arg_type = setter_func.args[0].type if setter_func.args else BOXED_PTR
                            if obj.is_instance and obj.llvm.type == first_arg_type:
                                self_arg = obj.llvm
                            elif obj.is_instance and first_arg_type == INSTANCE_PTR:
                                self_arg = obj.llvm
                            elif obj.is_object and first_arg_type == INSTANCE_PTR:
                                # obj is boxed, need to extract instance pointer from payload
                                tag, pay = self._read_slot(obj.llvm)
                                inst_tag = ir.Constant(I64, Tag.INST)
                                is_inst = self.builder.icmp_signed("==", tag, inst_tag)
                                inst_bb = self.current_func.append_basic_block("prop_set.inst")
                                err_bb = self.current_func.append_basic_block("prop_set.err")
                                end_bb = self.current_func.append_basic_block("prop_set.end")
                                self.builder.cbranch(is_inst, inst_bb, err_bb)
                                self.builder.position_at_end(inst_bb)
                                inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
                                # Box the value if needed — setter expects BOXED_PTR for value arg
                                if val.is_object:
                                    val_arg = val.llvm
                                else:
                                    val_arg = self._box(val)
                                call_args = [inst_ptr, val_arg]
                                if len(call_args) == len(setter_func.args):
                                    call_args = self._verify_call_args(setter_func, call_args)
                                    self.builder.call(setter_func, call_args)
                                    self._check_exc_after_call()
                                self.builder.branch(end_bb)
                                self.builder.position_at_end(err_bb)
                                self.builder.branch(end_bb)
                                self.builder.position_at_end(end_bb)
                                return  # Property setter handled the assignment
                            elif first_arg_type == DICT_PTR:
                                z = ir.Constant(I32, 0)
                                if obj.is_instance:
                                    self_arg = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 2)], inbounds=True))
                                else:
                                    tag, pay = self._read_slot(obj.llvm)
                                    inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
                                    self_arg = self.builder.load(self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True))
                            else:
                                self_arg = self._box(obj)
                            # Box the value if needed — setter expects BOXED_PTR for value arg
                            if val.is_object:
                                val_arg = val.llvm
                            else:
                                val_arg = self._box(val)
                            call_args = [self_arg, val_arg]
                            if len(call_args) == len(setter_func.args):
                                call_args = self._verify_call_args(setter_func, call_args)
                                self.builder.call(setter_func, call_args)
                                self._check_exc_after_call()
                            return  # Property setter handled the assignment

                if obj.is_instance:
                    # Check if this is a frozen dataclass instance
                    inst_ptr = obj.llvm
                    z = ir.Constant(I32, 0)
                    attrs_ptr = self.builder.load(
                        self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
                    )
                    # Check __frozen__ flag in the instance's attrs dict
                    frozen_key = self.create_string("__frozen__")
                    dct_val = Value(attrs_ptr, PyType.DICT)
                    frozen_check = self.dict_getitem(dct_val, frozen_key)
                    # dict_getitem returns a boxed value; check if it's truthy
                    frozen_tag, frozen_pay = self._read_slot(frozen_check.llvm)
                    is_frozen = self.builder.icmp_signed("==", frozen_tag, ir.Constant(I64, Tag.BOOL))

                    if_frozen_bb = self.current_func.append_basic_block("frozen.check")
                    not_frozen_bb = self.current_func.append_basic_block("frozen.ok")
                    self.builder.cbranch(is_frozen, if_frozen_bb, not_frozen_bb)

                    self.builder.position_at_end(if_frozen_bb)
                    # Raise FrozenInstanceError
                    # NAPRAWA: Dodaj nazwę pola do komunikatu błędu (jak CPython)
                    _frozen_err_msg = f"cannot assign to field '{attr_name}'"
                    if self._exc_handler_stack:
                        self._raise_exception("FrozenInstanceError", _frozen_err_msg)
                    else:
                        # Set global exception state and return
                        err_msg = self.create_string(_frozen_err_msg)
                        self._set_exc_global_state("FrozenInstanceError", err_msg)
                        rt = self.current_func.function_type.return_type
                        if isinstance(rt, ir.VoidType):
                            self.builder.ret_void()
                        elif rt == BOXED_PTR:
                            self.builder.ret(ir.Constant(BOXED_PTR, None))
                        else:
                            self.builder.ret_void()

                    self.builder.position_at_end(not_frozen_bb)
                    # Reload attrs_ptr since we might be in a different block
                    attrs_ptr = self.builder.load(
                        self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
                    )
                    key = self.create_string(attr_name)
                    dct_val = Value(attrs_ptr, PyType.DICT)
                    self.dict_setitem(dct_val, key, val)
                elif obj.is_dict:
                    # For simplified classes, self is a dict
                    key = self.create_string(attr_name)
                    self.dict_setitem(obj, key, val)
                elif obj.is_object:
                    tag, pay = self._read_slot(obj.llvm)
                    inst_bb = self.current_func.append_basic_block("asgn_attr.inst")
                    dct_bb = self.current_func.append_basic_block("asgn_attr.dct")
                    err_bb = self.current_func.append_basic_block("asgn_attr.err")
                    end_bb = self.current_func.append_basic_block("asgn_attr.end")
                    sw = self.builder.switch(tag, err_bb)
                    sw.add_case(ir.Constant(I64, Tag.INST), inst_bb)
                    sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
                    self.builder.position_at_end(inst_bb)
                    inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
                    z = ir.Constant(I32, 0)
                    attrs_ptr = self.builder.load(
                        self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
                    )
                    # Check __frozen__ flag for frozen dataclass enforcement
                    frozen_key = self.create_string("__frozen__")
                    dct_val_frozen = Value(attrs_ptr, PyType.DICT)
                    frozen_check = self.dict_getitem(dct_val_frozen, frozen_key)
                    frozen_tag, frozen_pay = self._read_slot(frozen_check.llvm)
                    is_frozen = self.builder.icmp_signed("==", frozen_tag, ir.Constant(I64, Tag.BOOL))
                    frozen_yes_bb = self.current_func.append_basic_block("asgn_attr.frozen")
                    frozen_no_bb = self.current_func.append_basic_block("asgn_attr.not_frozen")
                    self.builder.cbranch(is_frozen, frozen_yes_bb, frozen_no_bb)

                    self.builder.position_at_end(frozen_yes_bb)
                    # NAPRAWA: Dodaj nazwę pola do komunikatu błędu (jak CPython)
                    _frozen_err_msg = f"cannot assign to field '{attr_name}'"
                    if self._exc_handler_stack:
                        self._raise_exception("FrozenInstanceError", _frozen_err_msg)
                    else:
                        err_msg = self.create_string(_frozen_err_msg)
                        self._set_exc_global_state("FrozenInstanceError", err_msg)
                        rt = self.current_func.function_type.return_type
                        if isinstance(rt, ir.VoidType):
                            self.builder.ret_void()
                        elif rt == BOXED_PTR:
                            self.builder.ret(ir.Constant(BOXED_PTR, None))
                        else:
                            self.builder.ret_void()

                    self.builder.position_at_end(frozen_no_bb)
                    # Reload attrs_ptr
                    attrs_ptr = self.builder.load(
                        self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
                    )
                    key = self.create_string(attr_name)
                    dct_val = Value(attrs_ptr, PyType.DICT)
                    self.dict_setitem(dct_val, key, val)
                    self.builder.branch(end_bb)
                    self.builder.position_at_end(dct_bb)
                    dptr = self.builder.inttoptr(pay, DICT_PTR)
                    key = self.create_string(attr_name)
                    self.dict_setitem(Value(dptr, PyType.DICT), key, val)
                    self.builder.branch(end_bb)
                    self.builder.position_at_end(err_bb)
                    self.builder.branch(end_bb)
                    self.builder.position_at_end(end_bb)
                else:
                    raise CompileError(
                        f"Przypisanie atrybutu nie obsługuje {obj.pytype.name}.", node
                    )
            else:
                raise self._error(
                    ErrorCategory.SEMANTIC,
                    f"Unsupported assignment target: {type(target).__name__}",
                    node,
                    help_text="Only variable names, subscripts, and attributes can be assigned to.",
                )

    def _assign_name(self, name: str, val: Value):
        # NAPRAWA: Save is_class_ref BEFORE boxing, because boxing changes
        # val.llvm from CLASS_PTR constant to BOXED_PTR, which makes
        # _is_class_ref_value return False after boxing.
        is_cls_ref_before_box = self._is_class_ref_value(val)

        # FIX: Lambda/funkcja zwraca Value(func_ptr, PyType.OBJECT), ale func_ptr
        # jest typu wskaźnik funkcyjny, nie BOXED_PTR. Pakujemy go do Box przed
        # przypisaniem, aby typy LLVM się zgadzały.
        if val.is_object and val.llvm.type != BOXED_PTR:
            boxed = self._box(val)
            # NAPRAWA: Preserve class_name when boxing (needed for class reference variables)
            cn = val.class_name if hasattr(val, 'class_name') and val.class_name else None
            val = Value(boxed, PyType.OBJECT, class_name=cn)

        # Sprawdź, czy nazwa jest oznaczona jako globalna
        if name in getattr(self, "_global_vars", set()):
            # Użyj LLVM global variable
            gvar_name = f"__global_{name}"
            if gvar_name in self.module.globals:
                gv = self.module.globals[gvar_name]
                # Box the value
                new_val = val.llvm if val.is_object else self._box(val)
                # Decref old value
                old_val = self.builder.load(gv, name=f"old_{name}")
                is_not_null = self.builder.icmp_signed(
                    "!=", self.builder.ptrtoint(old_val, I64), ir.Constant(I64, 0)
                )
                dec_bb = self.current_func.append_basic_block("gdec")
                gskip_bb = self.current_func.append_basic_block("gskip")
                self.builder.cbranch(is_not_null, dec_bb, gskip_bb)
                self.builder.position_at_end(dec_bb)
                self.builder.call(
                    self.functions["__py2llvm_decref"],
                    [self.builder.bitcast(old_val, I8P)],
                )
                self.builder.branch(gskip_bb)
                self.builder.position_at_end(gskip_bb)
                # Store new value
                self.builder.store(new_val, gv)
                # Incref new value if it's an object
                if val.is_object:
                    self.builder.call(
                        self.functions["__py2llvm_incref"],
                        [self.builder.bitcast(val.llvm, I8P)],
                    )
                return
            # Fall back to symbol table lookup if no LLVM global exists
            try:
                info = self.sym.lookup(name)
                if info.py_type == PyType.OBJECT:
                    old_val = self.builder.load(info.alloca, name=f"old_{name}")
                    new_val = val.llvm if val.is_object else self._box(val)
                    self.builder.store(new_val, info.alloca)
                    if val.is_object:
                        self.builder.call(
                            self.functions["__py2llvm_incref"],
                            [self.builder.bitcast(val.llvm, I8P)],
                        )
                    self.builder.call(
                        self.functions["__py2llvm_decref"],
                        [self.builder.bitcast(old_val, I8P)],
                    )
                else:
                    cast_val = self._cast_to_llvm(val, info.llvm_type)
                    self.builder.store(cast_val.llvm, info.alloca)
                return
            except CompileError:
                pass

        # Sprawdź, czy nazwa jest oznaczona jako nonlocal (zamknięcie)
        if name in getattr(self, "_nonlocal_vars", set()):
            # NAPRAWA: Użyj LLVM global variable dla nonlocal, ponieważ
            # LLVM nie pozwala na dostęp do allocas z innej funkcji.
            nl_gvar_name = f"__nonlocal_{name}"
            if nl_gvar_name not in self.module.globals:
                gv = ir.GlobalVariable(self.module, BOXED_PTR, name=nl_gvar_name)
                gv.initializer = ir.Constant(BOXED_PTR, None)
                gv.linkage = "common"
                self.module.globals[nl_gvar_name] = gv
            gv = self.module.globals[nl_gvar_name]
            new_val = val.llvm if val.is_object else self._box(val)
            # Decref old value if not null
            old_val = self.builder.load(gv, name=f"old_nl_{name}")
            is_not_null = self.builder.icmp_signed(
                "!=", self.builder.ptrtoint(old_val, I64), ir.Constant(I64, 0)
            )
            dec_bb = self.current_func.append_basic_block("nl_dec")
            skip_bb = self.current_func.append_basic_block("nl_skip")
            self.builder.cbranch(is_not_null, dec_bb, skip_bb)
            self.builder.position_at_end(dec_bb)
            self.builder.call(
                self.functions["__py2llvm_decref"],
                [self.builder.bitcast(old_val, I8P)],
            )
            self.builder.branch(skip_bb)
            self.builder.position_at_end(skip_bb)
            # Store new value
            self.builder.store(new_val, gv)
            if val.is_object:
                self.builder.call(
                    self.functions["__py2llvm_incref"],
                    [self.builder.bitcast(val.llvm, I8P)],
                )
            # Zdefiniuj w bieżącym scope jako odniesienie do globalu
            if not self.sym.exists_local(name):
                self.sym.define(name, VarInfo(gv, BOXED_PTR, PyType.OBJECT))
            return

        # NAPRAWA: Jeśli istnieje LLVM global __global_{name} (zadeklarowany
        # na poziomie modułu), zaktualizuj GO RÓWNIEŻ, żeby funkcje z
        # 'global x' widziały zmianę.
        gvar_name = f"__global_{name}"
        sync_global = gvar_name in self.module.globals and name not in getattr(self, "_global_vars", set())

        # 1. Sprawdź, czy zmienna już istnieje w bieżącym zakresie
        if self.sym.exists_local(name):
            info = self.sym.lookup(name)
            if info.py_type == PyType.OBJECT:
                # Zmienna dynamiczna (boxed) - konieczne zarządzanie ARC (incref/decref)
                old_val = self.builder.load(info.alloca, name=f"old_{name}")

                # Lazy Boxing: Jeśli nowa wartość jest unboxed, pakujemy ją "leniwie" przed zapisem
                new_val = val.llvm if val.is_object else self._box(val)
                self.builder.store(new_val, info.alloca)

                if val.is_object:
                    self.builder.call(
                        self.functions["__py2llvm_incref"],
                        [self.builder.bitcast(val.llvm, I8P)],
                    )
                self.builder.call(
                    self.functions["__py2llvm_decref"],
                    [self.builder.bitcast(old_val, I8P)],
                )
            else:
                # Eliminacja ARC: Zmienna jest unboxed (np. typ i64)
                # Omijamy INCREF i DECREF, rzutujemy na wartość natywną
                cast_val = self._cast_to_llvm(val, info.llvm_type)
                self.builder.store(cast_val.llvm, info.alloca)

            # NAPRAWA: Synchronizuj z LLVM global, jeśli istnieje
            if sync_global:
                gv = self.module.globals[gvar_name]
                new_val_for_gv = val.llvm if val.is_object else self._box(val)
                self.builder.store(new_val_for_gv, gv)

            # NAPRAWA: Synchronizuj z nonlocal LLVM global, jeśli istnieje
            nl_gvar_name = f"__nonlocal_{name}"
            if nl_gvar_name in self.module.globals:
                nl_gv = self.module.globals[nl_gvar_name]
                new_val_for_nl = val.llvm if val.is_object else self._box(val)
                self.builder.store(new_val_for_nl, nl_gv)

            # NAPRAWA: Zaktualizuj class_name w VarInfo przy ponownym przypisaniu
            if hasattr(val, 'class_name') and val.class_name:
                info.class_name = val.class_name
            # NAPRAWA: Oznacz jako class_ref jeśli wartość to referencja do klasy
            # (np. device_cls = SmartDevice) — odróżnia od instancji (device = SmartDevice("Lamp"))
            is_cls_ref = is_cls_ref_before_box or self._is_class_ref_value(val)
            if is_cls_ref:
                info.is_class_ref = True
            elif hasattr(info, 'is_class_ref'):
                info.is_class_ref = False
        else:
            # 2. Tworzenie nowej zmiennej
            # NAPRAWA: Sprawdź czy mamy pre-allocas z entry blocku
            # (tworzone przez _compile_body aby uniknąć problemów z dominacją)
            pre_alloca_info = getattr(self, '_pre_allocas', {}).get(name)
            if pre_alloca_info is not None:
                if pre_alloca_info[0] == 'global':
                    # Zmienna z LLVM global — użyj globala jako storage
                    gv = pre_alloca_info[1]
                    new_val = val.llvm if val.is_object else self._box(val)
                    self.builder.store(new_val, gv)
                    if val.is_object:
                        self.builder.call(
                            self.functions["__py2llvm_incref"],
                            [self.builder.bitcast(val.llvm, I8P)],
                        )
                    vi = VarInfo(gv, BOXED_PTR, PyType.OBJECT,
                                 class_name=val.class_name if hasattr(val, 'class_name') and val.class_name else None)
                    vi.is_class_ref = is_cls_ref_before_box or self._is_class_ref_value(val)
                    self.sym.define(name, vi)
                    # NAPRAWA: Synchronizuj z nonlocal LLVM global, jeśli istnieje
                    nl_gvar_name = f"__nonlocal_{name}"
                    if nl_gvar_name in self.module.globals:
                        nl_gv = self.module.globals[nl_gvar_name]
                        self.builder.store(new_val, nl_gv)
                    return
                elif pre_alloca_info[0] == 'alloca':
                    # Zmienna z pre-alloca w entry blocku
                    _, alloca, llvm_type, inferred = pre_alloca_info
                    if inferred == PyType.OBJECT:
                        if val.is_object:
                            self.builder.call(
                                self.functions["__py2llvm_incref"],
                                [self.builder.bitcast(val.llvm, I8P)],
                            )
                            self.builder.store(val.llvm, alloca)
                        else:
                            boxed = self._box(val)
                            self.builder.store(boxed, alloca)
                    else:
                        cast_val = self._cast_to_llvm(val, llvm_type)
                        self.builder.store(cast_val.llvm, alloca)
                    # Zdefiniuj w symbol table
                    self.sym.define(name, VarInfo(alloca, llvm_type, inferred))
                    # Oznacz jako class_ref jeśli trzeba
                    is_cls_ref = is_cls_ref_before_box or self._is_class_ref_value(val)
                    if is_cls_ref and hasattr(self.sym.lookup(name), 'is_class_ref'):
                        self.sym.lookup(name).is_class_ref = True
                    if hasattr(val, 'class_name') and val.class_name:
                        self.sym.lookup(name).class_name = val.class_name
                    # NAPRAWA: Synchronizuj z nonlocal LLVM global, jeśli istnieje
                    nl_gvar_name = f"__nonlocal_{name}"
                    if nl_gvar_name in self.module.globals:
                        nl_gv = self.module.globals[nl_gvar_name]
                        new_val_for_nl = val.llvm if val.is_object else self._box(val)
                        self.builder.store(new_val_for_nl, nl_gv)
                    return

            # Standardowa ścieżka (bez pre-alloca)
            gvar_name = f"__global_{name}"
            has_gvar = gvar_name in self.module.globals

            inferred = PyType.OBJECT
            # If the variable has an LLVM global, always use OBJECT type (BOXED_PTR)
            # so that global accesses use the same storage

            # Wnioskowanie typów: czy zmienna na pewno jest monomorficzna?
            # NAPRAWA: Bezpieczny dostęp do inferred_static_types — atrybut może
            # nie istnieć jeśli _assign_name jest wywołane poza _compile_body.
            _ist = getattr(self, 'inferred_static_types', {})
            if not has_gvar and (
                name in _ist
                and len(_ist[name]) == 1
            ):
                inferred_candidate = list(_ist[name])[0]
                # Optymalizujemy (unboxing) tylko proste typy prymitywne. Obiekty sterty trafiają do Box
                if inferred_candidate in (PyType.INT, PyType.FLOAT, PyType.BOOL):
                    inferred = inferred_candidate

            llvm_target_type = pytype_to_llvm(inferred)

            if has_gvar:
                # Use the LLVM global variable as the storage
                gv = self.module.globals[gvar_name]
                alloca = gv
                llvm_target_type = BOXED_PTR
                inferred = PyType.OBJECT
            else:
                alloca = self.builder.alloca(llvm_target_type, name=name)

            if inferred == PyType.OBJECT or has_gvar:
                # Inicjalizacja BOXED_PTR z zarządzaniem ARC
                if val.is_object:
                    self.builder.call(
                        self.functions["__py2llvm_incref"],
                        [self.builder.bitcast(val.llvm, I8P)],
                    )
                    self.builder.store(val.llvm, alloca)
                else:
                    # Lazy Boxing "w locie" podczas pierwszego przypisania do Box
                    boxed = self._box(val)
                    self.builder.store(boxed, alloca)
            else:
                # Inicjalizacja UNBOXED: zapisujemy surową wartość bez Cycle Collectora i ARC
                cast = self._cast_to_llvm(val, llvm_target_type)
                self.builder.store(cast.llvm, alloca)

            # NAPRAWA: Oznacz class_ref dla nowej zmiennej
            is_cls_ref = is_cls_ref_before_box or self._is_class_ref_value(val)
            var_info = VarInfo(
                alloca,
                llvm_target_type,
                inferred,
                val.class_name
                if hasattr(val, "class_name") and val.class_name
                else None,
            )
            var_info.is_class_ref = is_cls_ref
            self.sym.define(name, var_info)

            # NAPRAWA: Synchronizuj z nonlocal LLVM global, jeśli istnieje
            nl_gvar_name = f"__nonlocal_{name}"
            if nl_gvar_name in self.module.globals:
                nl_gv = self.module.globals[nl_gvar_name]
                new_val_for_nl = val.llvm if val.is_object else self._box(val)
                self.builder.store(new_val_for_nl, nl_gv)

            # No need for separate LLVM global store - the alloca IS the global when has_gvar

    def visit_AnnAssign(self, node: ast.AnnAssign):
        val = self.visit(node.value)
        target = node.target
        if isinstance(target, ast.Name):
            pt = self._ann_to_pytype(node.annotation)
            lt = pytype_to_llvm(pt)
            if self.sym.exists_local(target.id):
                info = self.sym.lookup(target.id)
                cast = self._cast_to_llvm(val, info.llvm_type)
                self.builder.store(cast.llvm, info.alloca)
            else:
                alloca = self.builder.alloca(lt, name=target.id)
                cast = self._cast_to_llvm(val, lt)
                self.builder.store(cast.llvm, alloca)
                self.sym.define(target.id, VarInfo(alloca, lt, pt))

    def visit_AugAssign(self, node: ast.AugAssign):
        name = node.target.id
        # NAPRAWA: Musimy wczytać aktualną wartość zmiennej,
        # aby operacja AugAssign (np. x += 1) bazowała na aktualnym stanie w pamięci.
        current = self.visit(ast.Name(id=name, ctx=ast.Load()))
        rhs = self.visit(node.value)
        result = self._apply_binop(node.op, current, rhs, node)

        # Ensure result is correctly cast to the target variable's type before assignment
        if self.sym.exists_local(name):
            info = self.sym.lookup(name)
            result = self._cast_to_llvm(result, info.llvm_type, node)

        self._assign_name(name, result)

    def _generate_loop(self, iter_val, target_name, l_var, body_action):
        """Pomocnicza metoda do generowania pętli na podstawie typu iteratora."""
        func = self.current_func
        z = ir.Constant(I32, 0)

        if iter_val.is_list:
            sp, _, dp = self._list_ptrs(iter_val.llvm)
            size = self.builder.load(sp)
            data = self.builder.load(dp)
            idx_a = self.builder.alloca(I64, name=f"lc_idx_{target_name}")
            self.builder.store(ir.Constant(I64, 0), idx_a)
            cond_bb = func.append_basic_block("lc.cond")
            body_bb = func.append_basic_block("lc.body")
            end_bb = func.append_basic_block("lc.end")
            self.builder.branch(cond_bb)
            self.builder.position_at_end(cond_bb)
            self.builder.cbranch(self.builder.icmp_signed("<", self.builder.load(idx_a), size), body_bb, end_bb)
            self.builder.position_at_end(body_bb)
            curr_idx = self.builder.load(idx_a)
            slot = self.builder.gep(data, [curr_idx], inbounds=True)
            elem_tag = self.builder.load(self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True))
            elem_pay = self.builder.load(self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True))
            elem_val = self._boxed_to_value(elem_tag, elem_pay, None)
            self.builder.store(elem_val.llvm, l_var)
            body_action()
            self.builder.store(self.builder.add(curr_idx, ir.Constant(I64, 1)), idx_a)
            self.builder.branch(cond_bb)
            self.builder.position_at_end(end_bb)

        elif iter_val.is_object:
            tag, pay = self._read_slot(iter_val.llvm)
            is_list = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST))
            list_bb = func.append_basic_block("lc.obj.list")
            other_bb = func.append_basic_block("lc.obj.other")
            end_lc_bb = func.append_basic_block("lc.obj.end")
            self.builder.cbranch(is_list, list_bb, other_bb)

            # Obsługa jako lista
            self.builder.position_at_end(list_bb)
            lst_ptr = self.builder.inttoptr(pay, LIST_PTR)
            sp, cp, dp = self._list_ptrs(lst_ptr)
            size = self.builder.load(sp)
            data = self.builder.load(dp)
            idx_a = self.builder.alloca(I64, name=f"lc_obj_idx_{target_name}")
            self.builder.store(ir.Constant(I64, 0), idx_a)
            cond_bb = func.append_basic_block("lc.obj.cond")
            body_bb = func.append_basic_block("lc.obj.body")
            self.builder.branch(cond_bb)
            self.builder.position_at_end(cond_bb)
            self.builder.cbranch(self.builder.icmp_signed("<", self.builder.load(idx_a), size), body_bb, end_lc_bb)
            self.builder.position_at_end(body_bb)
            curr_idx = self.builder.load(idx_a)
            slot = self.builder.gep(data, [curr_idx], inbounds=True)
            elem_tag = self.builder.load(self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True))
            elem_pay = self.builder.load(self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True))
            elem_val = self._boxed_to_value(elem_tag, elem_pay, None)
            self.builder.store(elem_val.llvm, l_var)
            body_action()
            self.builder.store(self.builder.add(curr_idx, ir.Constant(I64, 1)), idx_a)
            self.builder.branch(cond_bb)

            # Sprawdz czy to DICT - iteracja po kluczach
            self.builder.position_at_end(other_bb)
            is_dict = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.DICT))
            dict_bb = func.append_basic_block("lc.obj.dict")
            unsup_bb = func.append_basic_block("lc.obj.unsup")
            self.builder.cbranch(is_dict, dict_bb, unsup_bb)

            self.builder.position_at_end(dict_bb)
            dct_ptr = self.builder.inttoptr(pay, DICT_PTR)
            dcap = self.builder.load(
                self.builder.gep(dct_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
            )
            dents = self.builder.load(
                self.builder.gep(dct_ptr, [z, ir.Constant(I32, 3)], inbounds=True)
            )
            di_a = self.builder.alloca(I64, name=f"lc_di_{target_name}")
            self.builder.store(ir.Constant(I64, 0), di_a)
            d_cond_bb = func.append_basic_block("lc.dct.cond")
            d_body_bb = func.append_basic_block("lc.dct.body")
            d_skip_bb = func.append_basic_block("lc.dct.skip")
            self.builder.branch(d_cond_bb)

            self.builder.position_at_end(d_cond_bb)
            di = self.builder.load(di_a)
            self.builder.cbranch(
                self.builder.icmp_signed("<", di, dcap), d_body_bb, end_lc_bb
            )

            self.builder.position_at_end(d_body_bb)
            ent = self.builder.gep(dents, [di], inbounds=True)
            ktag = self.builder.load(
                self.builder.gep(ent, [z, ir.Constant(I32, 0)], inbounds=True)
            )
            # Pomin puste sloty (ktag == -1)
            is_empty_slot = self.builder.icmp_signed("==", ktag, ir.Constant(I64, -1))
            self.builder.cbranch(is_empty_slot, d_skip_bb, func.append_basic_block("lc.dct.use"))
            use_bb = list(func.blocks)[-1]

            self.builder.position_at_end(use_bb)
            kpay = self.builder.load(
                self.builder.gep(ent, [z, ir.Constant(I32, 1)], inbounds=True)
            )
            key_val = self._boxed_to_value(ktag, kpay, None)
            self.builder.store(key_val.llvm, l_var)
            body_action()
            if not self.builder.block.is_terminated:
                self.builder.branch(d_skip_bb)

            self.builder.position_at_end(d_skip_bb)
            self.builder.store(self.builder.add(di, ir.Constant(I64, 1)), di_a)
            self.builder.branch(d_cond_bb)

            self.builder.position_at_end(unsup_bb)
            # Dla nieobslugiwanych typow - po prostu skocz do konca
            self.builder.branch(end_lc_bb)

            self.builder.position_at_end(end_lc_bb)
        else:
            raise CompileError("Unsupported iterator type in comprehension.", None)

    def _visit_comprehensions(self, generators, final_action):
        """Rekurencyjnie odwiedza generatory comprehension."""
        if not generators:
            final_action()
            return

        gen = generators[0]
        iter_val = self.visit(gen.iter)

        if not isinstance(gen.target, ast.Name):
            raise CompileError("Comprehension target must be a name.", gen)

        target_name = gen.target.id
        l_var = self.builder.alloca(BOXED_PTR, name=f"comp_{target_name}")
        self.sym.define(target_name, VarInfo(l_var, BOXED_PTR, PyType.OBJECT))

        def body_action():
            if gen.ifs:
                cond_val = None
                for if_clause in gen.ifs:
                    clause_val = self.visit(if_clause)
                    if cond_val is None:
                        cond_val = clause_val
                    else:
                        cond_val = self._apply_binop(ast.And(), cond_val, clause_val, None)

                truthy = self._eval_truthiness(cond_val)
                add_bb = self.current_func.append_basic_block("comp.if.add")
                skip_bb = self.current_func.append_basic_block("comp.if.skip")
                self.builder.cbranch(truthy.llvm if hasattr(truthy, "llvm") else truthy, add_bb, skip_bb)
                self.builder.position_at_end(add_bb)
                self._visit_comprehensions(generators[1:], final_action)
                self.builder.branch(skip_bb)
                self.builder.position_at_end(skip_bb)
            else:
                self._visit_comprehensions(generators[1:], final_action)

        self._generate_loop(iter_val, target_name, l_var, body_action)

    def visit_ListComp(self, node: ast.ListComp) -> Value:
        res_list = self.create_list([])
        self.sym.push()
        def final_action():
            expr_val = self.visit(node.elt)
            self.list_append(res_list, expr_val)
        self._visit_comprehensions(node.generators, final_action)
        self.sym.pop()
        return res_list

    def visit_DictComp(self, node: ast.DictComp) -> Value:
        res_dict = self.create_dict([])
        self.sym.push()
        def final_action():
            key_val = self.visit(node.key)
            val_val = self.visit(node.value)
            self.dict_setitem(res_dict, key_val, val_val)
        self._visit_comprehensions(node.generators, final_action)
        self.sym.pop()
        return res_dict

    def visit_SetComp(self, node: ast.SetComp) -> Value:
        raise CompileError("Set comprehensions are not yet supported.", node)

    def visit_Expr(self, node: ast.Expr):
        self.visit(node.value)

    def visit_Pass(self, node):
        pass

    # ──────────────────────────────────────────────────────────────
    #  Import - obsługa bibliotek
    # ──────────────────────────────────────────────────────────────

    def visit_If(self, node: ast.If):
        cond = self._to_bool_val(self.visit(node.test)).llvm
        then_bb = self.current_func.append_basic_block("if.then")
        merge_bb = self.current_func.append_basic_block("if.merge")
        else_bb = (
            self.current_func.append_basic_block("if.else") if node.orelse else merge_bb
        )
        self.builder.cbranch(cond, then_bb, else_bb)

        self.builder.position_at_end(then_bb)
        for s in node.body:
            if self.builder.block.is_terminated:
                break
            self.visit(s)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        if node.orelse:
            self.builder.position_at_end(else_bb)
            for s in node.orelse:
                if self.builder.block.is_terminated:
                    break
                self.visit(s)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

        self.builder.position_at_end(merge_bb)

    # ══════════════════════════════════════════════════════════════════
    #  POPRAWKA 7: for/else i while/else (Test 08)
    # ══════════════════════════════════════════════════════════════════

    def visit_While(self, node: ast.While) -> None:
        """
        Kompilacja pętli while z obsługą else.
        Blok else wykonuje się TYLKO jeśli pętla zakończyła się normalnie
        (warunek stał się fałszywy, nie przez break).
        """
        func = self.current_func

        # Alloca dla flagi "break occurred"
        break_flag = self.builder.alloca(I64, name="while_break")
        self.builder.store(ir.Constant(I64, 0), break_flag)

        # Bloki
        cond_bb = func.append_basic_block("while.cond")
        body_bb = func.append_basic_block("while.body")
        else_bb = func.append_basic_block("while.else") if node.orelse else None
        end_bb = func.append_basic_block("while.end")

        self.builder.branch(cond_bb)

        # Stack
        self._loop_exit_stack.append(end_bb)
        self._loop_cond_stack.append(cond_bb)
        self._loop_continue_stack.append(cond_bb)
        self._current_break_flag = break_flag

        # Warunek
        self.builder.position_at_end(cond_bb)
        # KLUCZOWA NAPRAWA: Musimy upewnić się, że zmienne w warunku są wczytywane
        # przy każdym powrocie do cond_bb. visit(node.test) robi to, bo
        # visit_Name wywołuje builder.load(info.alloca).
        cond_val = self.visit(node.test)
        is_truthy = self._eval_truthiness(cond_val)


        self.builder.cbranch(is_truthy.llvm, body_bb, else_bb if else_bb else end_bb)

        # Ciało
        self.builder.position_at_end(body_bb)
        # NAPRAWA: while w Pythonie NIE tworzy nowego zakresu zmiennych.
        # sym.push/pop powodowało, że przy każdym obrocie pętli _assign_name
        # nie znajdowało x w bieżącym zakresie i tworzyło NOWĄ alokację,
        # reinicjalizując zmienną do wartości początkowej.
        for stmt in node.body:
            if self.builder.block.is_terminated:
                break
            self.visit(stmt)
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb)

        # Cześć else
        if else_bb:
            self.builder.position_at_end(else_bb)
            # NAPRAWA: Jeśli pętla skończyła się normalnie (did_break == False),
            # to wykonaj blok else.
            did_break = self.builder.load(break_flag, "did_break")
            else_body_bb = func.append_basic_block("while.else.body")
            self.builder.cbranch(
                self.builder.icmp_signed("!=", did_break, ir.Constant(I64, 0)),
                end_bb, else_body_bb
            )

            self.builder.position_at_end(else_body_bb)
            self.sym.push()
            for stmt in node.orelse:
                if self.builder.block.is_terminated:
                    break
                self.visit(stmt)
            self.sym.pop()
            if not self.builder.block.is_terminated:
                self.builder.branch(end_bb)

        # Oczyść stack
        self._loop_exit_stack.pop()
        self._loop_cond_stack.pop()
        self._loop_continue_stack.pop()
        self._current_break_flag = None

        self.builder.position_at_end(end_bb)

    def visit_For(self, node: ast.For):
        unpack_names = None
        if isinstance(node.target, ast.Tuple):
            unpack_names = []
            for elt in node.target.elts:
                if not isinstance(elt, ast.Name):
                    raise CompileError("Elementy krotki w pętli for muszą być prostymi nazwami.", elt)
                unpack_names.append(elt.id)
            vname = "__tuple_unpack"
        elif isinstance(node.target, ast.Name):
            vname = node.target.id
        else:
            raise CompileError("Zmienna iteratora musi być prostą nazwą lub krotką nazw.", node)

        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            if unpack_names:
                raise CompileError("Rozpakowywanie krotki nie jest obsługiwane z range().", node)
            self._for_range(vname, node.iter, node.body, node.orelse)
        else:
            iter_val = self.visit(node.iter)
            if iter_val.is_list:
                self._for_list(vname, iter_val, node.body, node.orelse, unpack_names=unpack_names)
            elif iter_val.is_dict:
                # Iteracja po słowniku — iteruj po kluczach
                self._for_dict(vname, iter_val, node.body, node.orelse)
            elif iter_val.is_object:

                tag, pay = self._read_slot(iter_val.llvm)
                lst_bb = self.current_func.append_basic_block("for.lst")
                dict_bb = self.current_func.append_basic_block("for.dict")
                err_bb = self.current_func.append_basic_block("for.err")
                end_bb = self.current_func.append_basic_block("for.end")
                sw = self.builder.switch(tag, err_bb)
                sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)
                sw.add_case(ir.Constant(I64, Tag.DICT), dict_bb)
                self.builder.position_at_end(lst_bb)
                lptr = self.builder.inttoptr(pay, LIST_PTR)
                self._for_list(vname, Value(lptr, PyType.LIST), node.body, node.orelse, unpack_names=unpack_names)
                self.builder.branch(end_bb)
                self.builder.position_at_end(dict_bb)
                dptr = self.builder.inttoptr(pay, DICT_PTR)
                self._for_dict(vname, Value(dptr, PyType.DICT), node.body, node.orelse)
                self.builder.branch(end_bb)
                self.builder.position_at_end(err_bb)
                self.builder.branch(end_bb)
                self.builder.position_at_end(end_bb)
            else:
                raise self._error(
                    ErrorCategory.SEMANTIC,
                    "For loop requires range(), a list, or a dict",
                    node,
                    help_text="Pylow for-loops support iterating over range(), lists, and dictionaries.",
                )

    def _for_range(self, vname: str, call: ast.Call, body: list, orelse: list = None):
        args = call.args

        start_val = stop_val = step_val = None
        visited_args = []

        if len(args) == 1:
            start_val = 0
            visited_args.append(self.visit(args[0]))
            step_val = 1
        elif len(args) == 2:
            visited_args.append(self.visit(args[0]))
            visited_args.append(self.visit(args[1]))
            step_val = 1
        elif len(args) == 3:
            visited_args.append(self.visit(args[0]))
            visited_args.append(self.visit(args[1]))
            visited_args.append(self.visit(args[2]))
        else:
            raise self._error(
                ErrorCategory.SEMANTIC,
                "range() requires 1 to 3 arguments",
                help_text="Usage: range(stop), range(start, stop), or range(start, stop, step).",
            )

        if len(args) >= 1:
            start_val = 0 if start_val == 0 else self._get_const_int(visited_args[0])
        if len(args) >= 2:
            stop_val = self._get_const_int(
                visited_args[1] if len(args) == 2 else visited_args[1]
            )
        if len(args) == 3:
            step_val = self._get_const_int(visited_args[2])

        if (
            start_val is not None
            and stop_val is not None
            and step_val is not None
            and step_val != 0
        ):
            if step_val > 0:
                iterations = max(0, (stop_val - start_val + step_val - 1) // step_val)
            else:
                iterations = max(
                    0, (start_val - stop_val + (-step_val) - 1) // (-step_val)
                )

            # Never unroll - always use proper loop to handle break/continue correctly
            pass

        if len(args) == 1:
            start = Value(ir.Constant(I64, 0), PyType.INT)
            stop = self._to_int(visited_args[0])
            step = Value(ir.Constant(I64, 1), PyType.INT)
        elif len(args) == 2:
            start = self._to_int(visited_args[0])
            stop = self._to_int(visited_args[1])
            step = Value(ir.Constant(I64, 1), PyType.INT)
        else:
            start = self._to_int(visited_args[0])
            stop = self._to_int(visited_args[1])
            step = self._to_int(visited_args[2])

        ia = self.builder.alloca(I64, name=vname)
        self.builder.store(start.llvm, ia)
        self.sym.define(vname, VarInfo(ia, I64, PyType.INT))

        # Alloca dla flagi "break occurred"
        break_flag = self.builder.alloca(I64, name="for_range_break")
        self.builder.store(ir.Constant(I64, 0), break_flag)

        # Bloki
        c_bb = self.current_func.append_basic_block("fr.cond")
        b_bb = self.current_func.append_basic_block("fr.body")
        i_bb = self.current_func.append_basic_block("fr.incr")
        e_bb = self.current_func.append_basic_block("fr.end")
        else_bb = self.current_func.append_basic_block("fr.else") if orelse else None

        # NAPRAWA: Break ma skakać do e_bb (pomijać else),
        # nie do else_bb. else_bb jest tylko dla normalnego wyjścia z pętli.
        self._loop_cond_stack.append(c_bb)
        self._loop_continue_stack.append(i_bb)
        self._loop_exit_stack.append(e_bb)  # Break zawsze skacze do e_bb
        saved_break_flag = self._current_break_flag
        self._current_break_flag = break_flag
        self.builder.branch(c_bb)


        self.builder.position_at_end(c_bb)
        cur_int = self.builder.load(ia)
        self.builder.cbranch(
            self.builder.icmp_signed("<", cur_int, stop.llvm), b_bb, else_bb if else_bb else e_bb
        )


        self.builder.position_at_end(b_bb)
        for s in body:
            if self.builder.block.is_terminated:
                break
            self.visit(s)

        if not self.builder.block.is_terminated:
            self.builder.branch(i_bb)

        self.builder.position_at_end(i_bb)
        c2_int = self.builder.load(ia)
        new_int = self.builder.add(c2_int, step.llvm)
        self.builder.store(new_int, ia)
        self.builder.branch(c_bb)

        # Obsługa else dla range - else wykonuje się TYLKO przy normalnym wyjściu
        if else_bb:
            self.builder.position_at_end(else_bb)
            # Nie potrzebujemy sprawdzać break_flag - else_bb jest osiągane
            # tylko przez normalne wyjście z pętli (warunek fałszywy).
            # Break skacze bezpośrednio do e_bb.
            self.sym.push()
            for s in orelse:
                if self.builder.block.is_terminated:
                    break
                self.visit(s)
            self.sym.pop()
            if not self.builder.block.is_terminated:
                self.builder.branch(e_bb)

        self._loop_cond_stack.pop()
        self._loop_continue_stack.pop()
        self._loop_exit_stack.pop()
        self._current_break_flag = saved_break_flag
        self.builder.position_at_end(e_bb)

    def _get_const_int(self, val: Value) -> int:
        """Get constant int value if available, otherwise return None."""
        if val.is_int and isinstance(val.llvm, ir.Constant):
            return val.llvm.constant
        return None

    def _unroll_loop(self, vname: str, start: int, stop: int, step: int, body: list):
        """Unroll small constant loops for better performance."""
        if step == 0:
            return

        current = start
        while current < stop:
            ia = self.builder.alloca(I64, name=f"{vname}_tmp_{current}")
            self.builder.store(ir.Constant(I64, current), ia)
            self.sym.define(vname, VarInfo(ia, I64, PyType.INT))

            for s in body:
                if self.builder.block.is_terminated:
                    break
                self.visit(s)

            current += step

    def _for_dict(self, vname: str, dct_val: Value, body: list, orelse: list = None):
        """Iteracja po kluczach słownika (for key in dict:).

        Wykorzystuje uporządkowaną listę kluczy (index 4 w DictStruct),
        która jest utrzymywana przez dict_setitem.
        """
        z = ir.Constant(I32, 0)

        # Pobierz ordered_keys list z dict struct (index 4)
        ordered_list_ptr = self.builder.load(
            self.builder.gep(dct_val.llvm, [z, ir.Constant(I32, 4)], inbounds=True),
            name="fd_keys_list"
        )

        # Użyj _for_list do iteracji po liście kluczy
        self._for_list(vname, Value(ordered_list_ptr, PyType.LIST), body, orelse)

    def _for_list(self, vname: str, lst_val: Value, body: list, orelse: list = None, unpack_names=None):
        lst = lst_val.llvm
        sp, _, dp = self._list_ptrs(lst)
        size = self.builder.load(sp, "fls")
        data = self.builder.load(dp, "fld")


        ea = self.builder.alloca(BOXED_PTR, name=vname)
        self.sym.define(vname, VarInfo(ea, BOXED_PTR, PyType.OBJECT))

        # Allocas for tuple unpacking (before loop blocks)
        unpack_allocas = {}
        if unpack_names:
            for uname in unpack_names:
                ua = self.builder.alloca(BOXED_PTR, name=uname)
                self.sym.define(uname, VarInfo(ua, BOXED_PTR, PyType.OBJECT))
                unpack_allocas[uname] = ua

        i_a = self.builder.alloca(I64, name="_fli")
        self.builder.store(ir.Constant(I64, 0), i_a)

        # Alloca dla flagi "break occurred"
        break_flag = self.builder.alloca(I64, name="for_list_break")
        self.builder.store(ir.Constant(I64, 0), break_flag)

        # Bloki
        c_bb = self.current_func.append_basic_block("fl.cond")
        b_bb = self.current_func.append_basic_block("fl.body")
        i_bb = self.current_func.append_basic_block("fl.incr")
        e_bb = self.current_func.append_basic_block("fl.end")
        else_bb = self.current_func.append_basic_block("fl.else") if orelse else None

        # NAPRAWA: Break skacze do e_bb (pomija else)
        self._loop_cond_stack.append(c_bb)
        self._loop_continue_stack.append(i_bb)
        self._loop_exit_stack.append(e_bb)
        saved_break_flag = self._current_break_flag
        self._current_break_flag = break_flag
        self.builder.branch(c_bb)


        self.builder.position_at_end(c_bb)
        i = self.builder.load(i_a)
        self.builder.cbranch(
            self.builder.icmp_signed("<", i, size), b_bb, else_bb if else_bb else e_bb
        )


        self.builder.position_at_end(b_bb)
        i2 = self.builder.load(i_a)
        slot = self.builder.gep(data, [i2], inbounds=True, name="fl_sl")
        z = ir.Constant(I32, 0)

        # NAPRAWA: Wypakowywanie wartości z poprawnych indeksów 1 i 2 (a nie 0 i 1)
        tag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        pay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )

        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
        bv = self.builder.bitcast(raw, BOXED_PTR)
        null_i8p = ir.Constant(I8P, None)

        # NAPRAWA: Pełna inicjalizacja GC w Boxie
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

        self.builder.store(
            tag, self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            pay, self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.store(bv, ea)

        # Tuple unpacking: extract elements from inner list
        if unpack_names:
            inner_lst = self.builder.inttoptr(pay, LIST_PTR, "tup_lst")
            _, _, inner_dp = self._list_ptrs(inner_lst)
            inner_data = self.builder.load(inner_dp, "tup_data")

            for idx, uname in enumerate(unpack_names):
                elem_ptr = self.builder.gep(
                    inner_data, [ir.Constant(I64, idx)], inbounds=True, name=f"tup_el_{idx}"
                )
                etag = self.builder.load(
                    self.builder.gep(elem_ptr, [z, ir.Constant(I32, 1)], inbounds=True),
                    f"tup_et_{idx}",
                )
                epay = self.builder.load(
                    self.builder.gep(elem_ptr, [z, ir.Constant(I32, 2)], inbounds=True),
                    f"tup_ep_{idx}",
                )
                raw_elem = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
                elem_bv = self.builder.bitcast(raw_elem, BOXED_PTR)
                # Initialize GC header
                self.builder.store(
                    ir.Constant(I64, 1),
                    self.builder.gep(elem_bv, [z, z, ir.Constant(I32, 0)], inbounds=True),
                )
                self.builder.store(
                    ir.Constant(I32, 0),
                    self.builder.gep(elem_bv, [z, z, ir.Constant(I32, 1)], inbounds=True),
                )
                self.builder.store(
                    ir.Constant(I64, 0),
                    self.builder.gep(elem_bv, [z, z, ir.Constant(I32, 2)], inbounds=True),
                )
                self.builder.store(
                    null_i8p,
                    self.builder.gep(elem_bv, [z, z, ir.Constant(I32, 3)], inbounds=True),
                )
                # Store tag and payload
                self.builder.store(
                    etag,
                    self.builder.gep(elem_bv, [z, ir.Constant(I32, 1)], inbounds=True),
                )
                self.builder.store(
                    epay,
                    self.builder.gep(elem_bv, [z, ir.Constant(I32, 2)], inbounds=True),
                )
                self.builder.store(elem_bv, unpack_allocas[uname])

        for s in body:
            if self.builder.block.is_terminated:
                break
            self.visit(s)

        if not self.builder.block.is_terminated:
            self.builder.branch(i_bb)

        self.builder.position_at_end(i_bb)
        i3 = self.builder.load(i_a)
        self.builder.store(self.builder.add(i3, ir.Constant(I64, 1)), i_a)
        self.builder.branch(c_bb)

        # Obsługa else dla list - else wykonuje się TYLKO przy normalnym wyjściu
        if else_bb:
            self.builder.position_at_end(else_bb)
            # Nie sprawdzamy break_flag - else_bb osiągane tylko przez normalne wyjście
            # Break skacze bezpośrednio do e_bb.
            self.sym.push()
            for s in orelse:
                if self.builder.block.is_terminated:
                    break
                self.visit(s)
            self.sym.pop()
            if not self.builder.block.is_terminated:
                self.builder.branch(e_bb)

        self._loop_cond_stack.pop()
        self._loop_continue_stack.pop()
        self._loop_exit_stack.pop()
        self._current_break_flag = saved_break_flag
        self.builder.position_at_end(e_bb)

    def visit_Break(self, node: ast.Break):
        if not self._loop_exit_stack:
            raise self._error(
                ErrorCategory.SEMANTIC,
                "'break' statement outside of a loop",
                node,
                help_text="The 'break' statement can only be used inside a for or while loop.",
            )

        # NAPRAWA: Ustawiamy flagę break na 1 przed skokiem
        if self._current_break_flag:
            self.builder.store(ir.Constant(I64, 1), self._current_break_flag)

        self.builder.branch(self._loop_exit_stack[-1])

    def visit_Continue(self, node):
        if not self._loop_continue_stack:
            raise self._error(
                ErrorCategory.SEMANTIC,
                "'continue' statement outside of a loop",
                node,
                help_text="The 'continue' statement can only be used inside a for or while loop.",
            )
        self.builder.branch(self._loop_continue_stack[-1])

    # ──────────────────────────────────────────────────────────────
    #  Obsługa wyjątków (Try/Except/Finally/Raise)
    # ──────────────────────────────────────────────────────────────

    def visit_Try(self, node: ast.Try):
        """Obsługa try/except/finally w Pythonie."""
        # Create blocks for try body, except handlers, and finally
        try_bb = self.current_func.append_basic_block("try.body")

        # Create exit block (after try/except/finally)
        exit_bb = self.current_func.append_basic_block("try.exit")

        # Create finally block if we have finally clause
        finally_bb = None
        after_finally_bb = None
        ret_kind_alloca = None
        ret_val_alloca = None
        if node.finalbody:
            finally_bb = self.current_func.append_basic_block("try.finally")
            after_finally_bb = self.current_func.append_basic_block("try.after_finally")
            # Allocate return-kind and return-value for return-through-finally
            # 0=normal, 1=return, 2=reraise (unhandled exception after finally)
            ret_kind_alloca = self.builder.alloca(I32, name="finally_ret_kind")
            self.builder.store(ir.Constant(I32, 0), ret_kind_alloca)
            rt = self.current_func.function_type.return_type
            if isinstance(rt, ir.VoidType):
                ret_val_alloca = None
            else:
                ret_val_alloca = self.builder.alloca(rt, name="finally_ret_val")

        # Push exception handler info
        handler_info = {
            "exit_bb": exit_bb,
            "finally_bb": finally_bb,
            "handlers": [],  # List of (type, block) for except clauses
            "caught_exc": None,  # Will hold caught exception value
            "exc_type_alloca": None,  # Will hold exception type hash for type matching
            "unhandled_bb": None,  # Block for when no handler matches
        }

        # Create exception variable if we have except handlers
        if node.handlers:
            # Allocate storage for caught exception
            exc_var = self.builder.alloca(BOXED_PTR, name="exc_var")
            handler_info["caught_exc"] = exc_var

            # Allocate storage for exception type hash
            exc_type_alloca = self.builder.alloca(I64, name="exc_type_hash")
            self.builder.store(ir.Constant(I64, 0), exc_type_alloca)  # Initialize to 0
            handler_info["exc_type_alloca"] = exc_type_alloca

            # Create blocks for each except handler
            for i, handler in enumerate(node.handlers):
                handler_bb = self.current_func.append_basic_block(f"except.{i}")
                handler_info["handlers"].append((handler, handler_bb))

            # Create unhandled block (no handler matches)
            unhandled_bb = self.current_func.append_basic_block("except.unhandled")
            handler_info["unhandled_bb"] = unhandled_bb

        self._exc_handler_stack.append(handler_info)

        # Push finally info for return-through-finally
        if node.finalbody:
            self._finally_stack.append({
                "finally_bb": finally_bb,
                "ret_kind_alloca": ret_kind_alloca,
                "ret_val_alloca": ret_val_alloca,
                "after_finally_bb": after_finally_bb,
            })

        # Branch to try body
        self.builder.branch(try_bb)

        # Compile try body
        self.builder.position_at_end(try_bb)
        for stmt in node.body:
            if self.builder.block.is_terminated:
                break
            self.visit(stmt)

        # If try body doesn't terminate, branch to finally or exit
        if not self.builder.block.is_terminated:
            if finally_bb:
                self.builder.branch(finally_bb)
            else:
                self.builder.branch(exit_bb)

        # Compile except handlers
        for idx, (hdl, handler_bb) in enumerate(handler_info["handlers"]):
            self.builder.position_at_end(handler_bb)

            # NAPRAWA: Exception type matching
            # If the handler has a specific type (e.g., except ValueError as e),
            # check the stored exception type hash against the expected type.
            # If it doesn't match, branch to the next handler or the unhandled block.
            if hdl.type is not None and handler_info["exc_type_alloca"] is not None:
                # Get the expected type name(s)
                expected_types = []
                if isinstance(hdl.type, ast.Name):
                    expected_types.append(hdl.type.id)
                elif isinstance(hdl.type, ast.Tuple):
                    for elt in hdl.type.elts:
                        if isinstance(elt, ast.Name):
                            expected_types.append(elt.id)

                # Special case: "except Exception" catches ALL known exceptions
                # (since all our exceptions are subclasses of Exception in Python)
                catches_all = "Exception" in expected_types

                if expected_types and not catches_all:
                    # Load the stored exception type hash
                    stored_hash = self.builder.load(handler_info["exc_type_alloca"], "exc_type_hash_val")

                    # Check if any expected type matches
                    match_cond = None
                    for type_name in expected_types:
                        expected_hash = ir.Constant(I64, hash(type_name) & 0xFFFFFFFFFFFFFFFF)
                        this_match = self.builder.icmp_signed("==", stored_hash, expected_hash)
                        if match_cond is None:
                            match_cond = this_match
                        else:
                            match_cond = self.builder.or_(match_cond, this_match)

                    # Determine the "no match" target: next handler or unhandled block
                    no_match_bb = handler_info["unhandled_bb"]
                    if idx + 1 < len(handler_info["handlers"]):
                        _, no_match_bb = handler_info["handlers"][idx + 1]

                    # Create body block for the handler (after type check)
                    handler_body_bb = self.current_func.append_basic_block(f"except.{idx}.body")
                    self.builder.cbranch(match_cond, handler_body_bb, no_match_bb)
                    self.builder.position_at_end(handler_body_bb)

            # If there's an exception variable, assign the caught exception
            if hdl.name and handler_info["caught_exc"]:
                exc_val = self.builder.load(handler_info["caught_exc"], "caught_exc_val")
                exc_alloca = self.builder.alloca(BOXED_PTR, name=f"exc_{hdl.name}")
                self.builder.store(exc_val, exc_alloca)
                self.sym.define(hdl.name, VarInfo(exc_alloca, BOXED_PTR, PyType.OBJECT))

            # Clear the global exception pending flag — we've handled the exception
            self._clear_exc_pending()

            # Compile the handler body
            if hdl.body:
                for stmt in hdl.body:
                    if self.builder.block.is_terminated:
                        break
                    self.visit(stmt)

            # Branch to finally or exit after handler
            if not self.builder.block.is_terminated:
                if finally_bb:
                    self.builder.branch(finally_bb)
                else:
                    self.builder.branch(exit_bb)

        # Compile the unhandled block (no handler matched the exception type)
        if handler_info["unhandled_bb"] is not None:
            self.builder.position_at_end(handler_info["unhandled_bb"])
            # If there's a finally block, set ret_kind=2 (reraise) and go there
            # Otherwise, re-raise as unhandled exception immediately
            if finally_bb:
                self.builder.store(ir.Constant(I32, 2), ret_kind_alloca)  # 2=reraise
                self.builder.branch(finally_bb)
            else:
                # No finally and no matching handler - treat as unhandled exception
                err_msg = self.create_string("Unhandled exception (no matching except clause)")
                self._emit_print([err_msg])
                rt = self.current_func.function_type.return_type
                if isinstance(rt, ir.VoidType):
                    self.builder.ret_void()
                elif rt == I32:
                    self.builder.ret(ir.Constant(I32, 1))
                elif rt == BOXED_PTR:
                    self.builder.ret(ir.Constant(BOXED_PTR, None))
                else:
                    self.builder.ret_void()

        # Compile finally block
        if finally_bb:
            self.builder.position_at_end(finally_bb)
            # Execute finally body
            for stmt in node.finalbody:
                if self.builder.block.is_terminated:
                    break
                self.visit(stmt)

            # After finally, check if there's a pending return
            if not self.builder.block.is_terminated:
                self.builder.branch(after_finally_bb)

            self.builder.position_at_end(after_finally_bb)
            ret_kind = self.builder.load(ret_kind_alloca, "ret_kind")
            normal_bb = self.current_func.append_basic_block("try.normal_exit")
            return_bb = self.current_func.append_basic_block("try.return_exit")
            reraise_bb = self.current_func.append_basic_block("try.reraise_exit")
            # ret_kind: 0=normal, 1=return, 2=reraise
            sw = self.builder.switch(ret_kind, reraise_bb)
            sw.add_case(ir.Constant(I32, 0), normal_bb)
            sw.add_case(ir.Constant(I32, 1), return_bb)

            self.builder.position_at_end(normal_bb)
            self.builder.branch(exit_bb)

            self.builder.position_at_end(return_bb)
            rt = self.current_func.function_type.return_type
            if isinstance(rt, ir.VoidType):
                self.builder.ret_void()
            elif ret_val_alloca is not None:
                rv = self.builder.load(ret_val_alloca, "finally_ret")
                self.builder.ret(rv)
            else:
                self.builder.ret_void()

            self.builder.position_at_end(reraise_bb)
            # Re-raise: treat as unhandled exception
            err_msg = self.create_string("Unhandled exception (no matching except clause)")
            self._emit_print([err_msg])
            rt2 = self.current_func.function_type.return_type
            if isinstance(rt2, ir.VoidType):
                self.builder.ret_void()
            elif rt2 == I32:
                self.builder.ret(ir.Constant(I32, 1))
            elif rt2 == BOXED_PTR:
                self.builder.ret(ir.Constant(BOXED_PTR, None))
            else:
                self.builder.ret_void()

        # Continue at exit block
        self.builder.position_at_end(exit_bb)

        # Pop finally info
        if node.finalbody:
            self._finally_stack.pop()

        # Pop exception handler
        self._exc_handler_stack.pop()

    def visit_Raise(self, node: ast.Raise):
        """Obsługa instrukcji raise."""
        # If we have an exception to raise, get its value
        exc_value = None
        exc_type_name = None  # Name of the exception type (for type matching)

        if node.exc:
            exc_value = self.visit(node.exc)
            # NAPRAWA: Extract exception type name for type matching
            if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                exc_type_name = node.exc.func.id
            elif isinstance(node.exc, ast.Name):
                exc_type_name = node.exc.id

        # If we have active exception handlers, jump to the appropriate handler
        if self._exc_handler_stack:
            handler_info = self._exc_handler_stack[-1]

            # Store the exception value if there's a variable
            if exc_value and handler_info["caught_exc"]:
                self.builder.store(exc_value.llvm, handler_info["caught_exc"])

            # NAPRAWA: Store the exception type hash for type matching
            if exc_type_name and handler_info["exc_type_alloca"] is not None:
                type_hash = ir.Constant(I64, hash(exc_type_name) & 0xFFFFFFFFFFFFFFFF)
                self.builder.store(type_hash, handler_info["exc_type_alloca"])

            # NAPRAWA: Store the exception type name as a string for __exit__
            if exc_type_name and handler_info.get("exc_type_name_alloca") is not None:
                type_name_str = self.create_string(exc_type_name)
                type_name_boxed = self._box(type_name_str)
                self.builder.store(type_name_boxed, handler_info["exc_type_name_alloca"])

            # Jump to the first except clause (type matching is done in the handler)
            if handler_info["handlers"]:
                _, first_handler = handler_info["handlers"][0]
                self.builder.branch(first_handler)
                return
            elif handler_info["finally_bb"]:
                self.builder.branch(handler_info["finally_bb"])
                return

        # No handler in current scope — set global exception state and return
        # so the caller can propagate the exception
        self._set_exc_global_state(exc_type_name, exc_value)

        # Return from current function
        rt = self.current_func.function_type.return_type
        if isinstance(rt, ir.VoidType):
            self.builder.ret_void()
        elif rt == I32:
            self.builder.ret(ir.Constant(I32, 1))
        elif rt == BOXED_PTR:
            self.builder.ret(ir.Constant(BOXED_PTR, None))
        else:
            self.builder.ret_void()

    def visit_With(self, node: ast.With):
        """Obsługa instrukcji with (context managers).

        Generuje kod który:
        1. Ewaluuje wyrażenie context managera
        2. Wywołuje context_manager.__enter__()
        3. Jeśli jest klauzula 'as', przypisuje wynik __enter__ do zmiennej
        4. Wykonuje ciało with (wewnątrz obsługi wyjątków)
        5. Wywołuje context_manager.__exit__(exc_type, exc_val, exc_tb)
           — ZAWSZE, nawet gdy ciało zgłosi wyjątek
        6. Jeśli __exit__ zwróci True (truthy), tłumi wyjątek
        7. Jeśli __exit__ zwróci False, ponownie zgłasza wyjątek

        Strategia obsługi wyjątków:
        - Przed kompilacją ciała with, wpychamy handler na _exc_handler_stack
          z finally_bb ustawionym na blok, który wywołuje __exit__ i ponownie
          zgłasza wyjątek. Dzięki temu visit_Raise widzi nasz handler i
          branch'uje do finally_bb, gwarantując że __exit__ zawsze zostanie
          wywołany — niezależnie czy raise jest wewnątrz try czy nie.
        - Jeśli ciało zakończy się normalnie, wywołujemy __exit__(None, None, None).
        """
        z = ir.Constant(I32, 0)

        # Lista (inferred_class, inst_alloca, has_exit) dla każdego context managera
        cm_info_list = []

        for item in node.items:
            # Evaluate the context manager expression
            context_mgr = self.visit(item.context_expr)

            # Determine class name for method lookup
            inferred_class = context_mgr.class_name if hasattr(context_mgr, 'class_name') else None
            if not inferred_class and isinstance(item.context_expr, ast.Name):
                try:
                    var_info = self.sym.lookup(item.context_expr.id)
                    if hasattr(var_info, 'class_name') and var_info.class_name:
                        inferred_class = var_info.class_name
                except Exception:
                    pass

            # NAPRAWA: Also check for Call expressions (e.g., DatabaseConnection("ProductionDB"))
            if not inferred_class and isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Name):
                    call_name = item.context_expr.func.id
                    if f"__is_class_{call_name}" in self.functions:
                        inferred_class = call_name

            # Extract instance pointer from the context manager value
            inst_ptr = None
            if context_mgr.is_instance and context_mgr.llvm.type == INSTANCE_PTR:
                inst_ptr = context_mgr.llvm
            elif context_mgr.is_object:
                tag, pay = self._read_slot(context_mgr.llvm)
                inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR, "with_self")

            # Store the instance pointer for __exit__ later
            inst_alloca = None
            if inst_ptr is not None:
                inst_alloca = self.builder.alloca(INSTANCE_PTR, name="with_cm")
                self.builder.store(inst_ptr, inst_alloca)

            # Check if __enter__ and __exit__ exist
            has_enter = False
            has_exit = False
            if inferred_class:
                enter_func_name = f"py_{inferred_class}___enter__"
                exit_func_name = f"py_{inferred_class}___exit__"
                has_enter = enter_func_name in self.functions
                has_exit = exit_func_name in self.functions

            cm_info_list.append((inferred_class, inst_alloca, has_exit))

            # Call __enter__() if it exists
            enter_result = None
            if has_enter and inst_ptr is not None:
                enter_func_name = f"py_{inferred_class}___enter__"
                enter_func = self.functions[enter_func_name]
                # Build args: self (INSTANCE_PTR)
                first_arg_type = enter_func.args[0].type if enter_func.args else BOXED_PTR
                if first_arg_type == INSTANCE_PTR:
                    enter_args = [inst_ptr]
                elif first_arg_type == DICT_PTR:
                    enter_args = [self.builder.load(
                        self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
                    )]
                else:
                    enter_args = [self._box(context_mgr)]
                if len(enter_args) == len(enter_func.args):
                    enter_result = self.builder.call(enter_func, enter_args, name="enter_result")
                    self._check_exc_after_call()

            # NAPRAWA: After _check_exc_after_call, block may be terminated
            # if __enter__ raised. Stop processing items — the exception will
            # be caught by our handler below.
            if self.builder.block.is_terminated:
                break

            # If there's an optional variable (as db), assign the __enter__ result
            if item.optional_vars:
                var_name = item.optional_vars.id
                if enter_result is not None:
                    enter_val = Value(enter_result, PyType.OBJECT, class_name=inferred_class)
                    self._assign_name(var_name, enter_val)
                else:
                    # No __enter__ — assign context manager itself
                    self._assign_name(var_name, context_mgr)

        # ══════════════════════════════════════════════════════════════════
        #  Compile body with exception safety for __exit__
        # ══════════════════════════════════════════════════════════════════

        any_has_exit = any(has_exit for _, _, has_exit in cm_info_list)

        if not any_has_exit:
            # Simple path: no __exit__, just compile the body
            for stmt in node.body:
                if self.builder.block.is_terminated:
                    break
                self.visit(stmt)
            return

        # ── Complex path: set up exception interception for __exit__ ──
        # Push a handler onto _exc_handler_stack so that visit_Raise
        # will branch to our exc_handler_bb, where we call __exit__
        # and then re-raise to the outer handler.

        # Allocate exception storage (before branching, in current block)
        exc_val_alloca = self.builder.alloca(BOXED_PTR, name="with_exc_val")
        self.builder.store(ir.Constant(BOXED_PTR, None), exc_val_alloca)
        exc_type_hash_alloca = self.builder.alloca(I64, name="with_exc_type_hash")
        self.builder.store(ir.Constant(I64, 0), exc_type_hash_alloca)
        exc_type_name_alloca = self.builder.alloca(BOXED_PTR, name="with_exc_name")
        self.builder.store(ir.Constant(BOXED_PTR, None), exc_type_name_alloca)

        # Create control flow blocks
        body_bb = self.current_func.append_basic_block("with.body")
        no_exc_bb = self.current_func.append_basic_block("with.no_exc")
        exc_handler_bb = self.current_func.append_basic_block("with.exc_handler")
        end_bb = self.current_func.append_basic_block("with.end")

        # Push exception handler for __exit__
        # This handler has no 'except' clauses — it uses finally_bb to
        # intercept ALL exceptions, call __exit__, and re-raise.
        with_handler_info = {
            "exit_bb": end_bb,
            "finally_bb": exc_handler_bb,
            "handlers": [],  # No except clauses
            "caught_exc": exc_val_alloca,
            "exc_type_alloca": exc_type_hash_alloca,
            "exc_type_name_alloca": exc_type_name_alloca,
            "unhandled_bb": exc_handler_bb,
        }
        self._exc_handler_stack.append(with_handler_info)

        # Branch to body
        if not self.builder.block.is_terminated:
            self.builder.branch(body_bb)

        # ── Compile with body ──
        self.builder.position_at_end(body_bb)
        for stmt in node.body:
            if self.builder.block.is_terminated:
                break
            self.visit(stmt)

        # If body completed normally, branch to no_exc_bb
        if not self.builder.block.is_terminated:
            self.builder.branch(no_exc_bb)

        # Pop the with handler — body is done, no more raise interception needed
        self._exc_handler_stack.pop()

        # ── Normal exit: call __exit__(None, None, None) ──
        self.builder.position_at_end(no_exc_bb)
        self._call_context_manager_exits(cm_info_list, exc_type=None, exc_val=None)
        if not self.builder.block.is_terminated:
            self._clear_exc_pending()
            self.builder.branch(end_bb)

        # ── Exception handler: call __exit__(exc_type, exc_val, exc_tb) and re-raise ──
        # This block is reached when visit_Raise branches to finally_bb
        self.builder.position_at_end(exc_handler_bb)

        # Load saved exception info from the handler's allocas
        # (visit_Raise stored them before branching here)
        saved_hash = self.builder.load(exc_type_hash_alloca, "with_saved_hash")
        saved_val = self.builder.load(exc_val_alloca, "with_saved_val")

        # Create exc_type value using _get_exc_type_value_from_allocas
        # This creates a dict-like object with __name__ attribute
        exc_type_val = self._get_exc_type_value_from_allocas(
            exc_type_hash_alloca, exc_type_name_alloca, None
        )

        # Call __exit__ with exception info
        self._call_context_manager_exits(cm_info_list, exc_type=exc_type_val, exc_val=saved_val)

        if not self.builder.block.is_terminated:
            # Re-raise: restore global exception state
            self.builder.store(ir.Constant(I1, 1), self._exc_pending_global)
            self.builder.store(saved_hash, self._exc_type_hash_global)
            self.builder.store(saved_val, self._exc_value_global)

            # Populate the outer handler's allocas so it can catch the re-raised exception
            if self._exc_handler_stack:
                outer_handler = self._exc_handler_stack[-1]
                if outer_handler.get("exc_type_alloca") is not None:
                    self.builder.store(saved_hash, outer_handler["exc_type_alloca"])
                if outer_handler.get("caught_exc") is not None:
                    self.builder.store(saved_val, outer_handler["caught_exc"])

            # Branch to the outer handler
            if self._exc_handler_stack:
                outer_handler = self._exc_handler_stack[-1]
                if outer_handler["handlers"]:
                    _, first_handler = outer_handler["handlers"][0]
                    self.builder.branch(first_handler)
                elif outer_handler.get("exit_bb"):
                    self.builder.branch(outer_handler["exit_bb"])
                else:
                    self.builder.branch(end_bb)
            else:
                # No outer handler — return with exception state set
                rt = self.current_func.function_type.return_type
                if isinstance(rt, ir.VoidType):
                    self.builder.ret_void()
                elif rt == BOXED_PTR:
                    self.builder.ret(ir.Constant(BOXED_PTR, None))
                else:
                    self.builder.ret(ir.Constant(rt, 0))

        # ── End block: continue after with ──
        self.builder.position_at_end(end_bb)

    def _call_context_manager_exits(self, cm_info_list, exc_type=None, exc_val=None):
        """Call __exit__() on all context managers in reverse order (LIFO).

        Args:
            cm_info_list: List of (inferred_class, inst_alloca, has_exit) tuples
            exc_type: BOXED_PTR value for exception type (or None for normal exit)
            exc_val: BOXED_PTR value for exception value (or None for normal exit)
        """
        z = ir.Constant(I32, 0)

        for inferred_class, inst_alloca, has_exit in reversed(cm_info_list):
            if not (inferred_class and has_exit and inst_alloca is not None):
                continue

            # NAPRAWA: If block was terminated by a previous __exit__ raising,
            # stop calling more __exit__ methods
            if self.builder.block.is_terminated:
                break

            exit_func_name = f"py_{inferred_class}___exit__"
            if exit_func_name not in self.functions:
                continue

            exit_func = self.functions[exit_func_name]
            # Load the stored instance pointer
            cm_inst = self.builder.load(inst_alloca, "with_cm_exit")

            # Build args: self
            first_arg_type = exit_func.args[0].type if exit_func.args else BOXED_PTR
            if first_arg_type == INSTANCE_PTR:
                exit_args = [cm_inst]
            elif first_arg_type == DICT_PTR:
                exit_args = [self.builder.load(
                    self.builder.gep(cm_inst, [z, ir.Constant(I32, 2)], inbounds=True)
                )]
            else:
                # Fallback: box the instance pointer
                raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)], "exit_self_raw")
                bv = self.builder.bitcast(raw, BOXED_PTR, "exit_self_bv")
                # Store GC header + Tag.INST + payload
                self.builder.store(ir.Constant(I64, 1), self.builder.gep(bv, [z, z, ir.Constant(I32, 0)], inbounds=True))
                self.builder.store(ir.Constant(I32, 0), self.builder.gep(bv, [z, z, ir.Constant(I32, 1)], inbounds=True))
                self.builder.store(ir.Constant(I64, 0), self.builder.gep(bv, [z, z, ir.Constant(I32, 2)], inbounds=True))
                self.builder.store(ir.Constant(I8P, None), self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True))
                self.builder.store(ir.Constant(I64, Tag.INST), self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True))
                self.builder.store(self.builder.ptrtoint(cm_inst, I64), self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True))
                exit_args = [bv]

            # Pass exception info for remaining args
            n_extra_args = len(exit_func.args) - 1
            if exc_type is not None and n_extra_args >= 1:
                # We have exception info to pass
                exit_args.append(exc_type)
                if n_extra_args >= 2 and exc_val is not None:
                    exit_args.append(exc_val)
                    # Fill remaining args with None (e.g., exc_tb)
                    for _ in range(n_extra_args - 2):
                        none_val = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
                        exit_args.append(none_val)
                else:
                    # Fill remaining with None
                    for _ in range(n_extra_args - 1):
                        none_val = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
                        exit_args.append(none_val)
            else:
                # No exception info — pass None for all extra args
                for _ in range(n_extra_args):
                    none_val = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
                    exit_args.append(none_val)

            if len(exit_args) == len(exit_func.args):
                exit_result = self.builder.call(exit_func, exit_args, name="exit_result")
                self._check_exc_after_call()
                # NOTE: In a full implementation, we would check the return value
                # of __exit__ here. If it returns True, we suppress the exception.
                # For now, we always propagate the exception.

    def _get_exc_type_value(self) -> ir.Value:
        """Get the current exception type as a BOXED_PTR value.

        Creates a dict-like object with a '__name__' attribute so that
        exc_type.__name__ works in __exit__ methods.
        If no exception, returns None.
        Reads from the current handler's allocas or global state.
        """
        exc_type_alloca = None
        exc_type_name_alloca = None

        if self._exc_handler_stack:
            handler_info = self._exc_handler_stack[-1]
            exc_type_alloca = handler_info.get("exc_type_alloca")
            exc_type_name_alloca = handler_info.get("exc_type_name_alloca")

        return self._get_exc_type_value_from_allocas(exc_type_alloca, exc_type_name_alloca, None)

    def _get_exc_type_value_from_allocas(self, exc_type_alloca, exc_type_name_alloca, caught_exc_alloca) -> ir.Value:
        """Get the current exception type as a BOXED_PTR value from specific allocas.

        Creates a dict-like object with a '__name__' attribute so that
        exc_type.__name__ works in __exit__ methods.
        If no exception, returns None.
        """
        # Determine the source of exception hash
        if exc_type_alloca is not None:
            exc_type_hash = self.builder.load(exc_type_alloca, "with_exc_hash_local")
        else:
            exc_type_hash = self.builder.load(self._exc_type_hash_global, "with_exc_hash_global")

        # Check if there's an exception (exc_type_hash != 0)
        is_exc = self.builder.icmp_signed("!=", exc_type_hash, ir.Constant(I64, 0))

        exc_bb = self.current_func.append_basic_block("with.exc_type.yes")
        no_exc_bb = self.current_func.append_basic_block("with.exc_type.no")
        merge_bb = self.current_func.append_basic_block("with.exc_type.merge")
        self.builder.cbranch(is_exc, exc_bb, no_exc_bb)

        # ── Exception path: create dict with __name__ ──
        self.builder.position_at_end(exc_bb)
        exc_dict = self.create_dict([])
        name_key = self.create_string("__name__")
        if exc_type_name_alloca is not None:
            type_name_val = self.builder.load(exc_type_name_alloca, "with_exc_type_name_val")
            self.dict_setitem(exc_dict, name_key, Value(type_name_val, PyType.OBJECT))
        else:
            generic_name = self.create_string("Exception")
            self.dict_setitem(exc_dict, name_key, generic_name)
        exc_dict_boxed = self._box(exc_dict)
        # Track the actual block we're in after _box (it may create new blocks)
        exc_actual_bb = self.builder.block
        self.builder.branch(merge_bb)

        # ── No exception path: return None ──
        self.builder.position_at_end(no_exc_bb)
        none_val = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
        no_exc_actual_bb = self.builder.block
        self.builder.branch(merge_bb)

        # ── Merge ──
        self.builder.position_at_end(merge_bb)
        phi = self.builder.phi(BOXED_PTR, "exc_type_result")
        phi.add_incoming(exc_dict_boxed, exc_actual_bb)
        phi.add_incoming(none_val, no_exc_actual_bb)
        return phi

    def _get_exc_msg_value(self) -> ir.Value:
        """Get the current exception message as a BOXED_PTR value.

        Reads the exception value from the handler's caught_exc alloca
        or from global state.
        """
        caught_exc = None
        if self._exc_handler_stack:
            handler_info = self._exc_handler_stack[-1]
            caught_exc = handler_info.get("caught_exc")
        return self._get_exc_msg_value_from_alloca(caught_exc)

    def _get_exc_msg_value_from_alloca(self, caught_exc_alloca) -> ir.Value:
        """Get the current exception message as a BOXED_PTR value from a specific alloca."""
        if caught_exc_alloca is not None:
            exc_val = self.builder.load(caught_exc_alloca, "with_exc_caught_val")
            return exc_val
        # Fallback: read from global state
        exc_val = self.builder.load(self._exc_value_global, "with_exc_msg_val")
        return exc_val

    def _set_exc_global_from_handler(self, exc_type_alloca, caught_exc_alloca):
        """Re-set the global exception state from handler allocas so that
        the outer handler can properly catch the re-raised exception."""
        # Set exc_pending = True
        self.builder.store(ir.Constant(I1, 1), self._exc_pending_global)
        # Copy type hash to global
        if exc_type_alloca is not None:
            exc_hash = self.builder.load(exc_type_alloca, "reraise_hash")
            self.builder.store(exc_hash, self._exc_type_hash_global)
        # Copy exception value to global
        if caught_exc_alloca is not None:
            exc_val = self.builder.load(caught_exc_alloca, "reraise_val")
            self.builder.store(exc_val, self._exc_value_global)

    # ──────────────────────────────────────────────────────────────
    #  Generatory (yield, yield from)
    # ──────────────────────────────────────────────────────────────

    def visit_Yield(self, node: ast.Yield) -> Value:
        """Obsługa instrukcji yield w generatorach.
        
        W uproszczonej implementacji, yield dodaje wartość do listy
        generatora (self._generator_list), która jest zwracana jako
        iterator na końcu funkcji.
        """
        if node.value:
            yield_val = self.visit(node.value)
        else:
            yield_val = Value(ir.Constant(I64, 0), PyType.NONE)

        # Jeśli jesteśmy w funkcji generatorowej, dodaj wartość do listy
        if getattr(self, '_is_generator', False):
            if self._generator_list is None:
                self._generator_list = self.create_list([])
            self.list_append(self._generator_list, yield_val)

        return yield_val

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Value:
        """Obsługa instrukcji yield from."""
        if node.value:
            iter_val = self.visit(node.value)
            # Jeśli jesteśmy w funkcji generatorowej, dodaj elementy z iteratora
            if getattr(self, '_is_generator', False):
                if self._generator_list is None:
                    self._generator_list = self.create_list([])
                # Jeśli to lista, rozpakuj i dodaj elementy
                if iter_val.is_list:
                    sp, _, dp = self._list_ptrs(iter_val.llvm)
                    list_size = self.builder.load(sp, "yf_sz")
                    list_data = self.builder.load(dp, "yf_data")
                    z = ir.Constant(I32, 0)
                    idx_a = self.builder.alloca(I64, name="yf_idx")
                    self.builder.store(ir.Constant(I64, 0), idx_a)
                    cond_bb = self.current_func.append_basic_block("yf.cond")
                    body_bb = self.current_func.append_basic_block("yf.body")
                    end_bb = self.current_func.append_basic_block("yf.end")
                    self.builder.branch(cond_bb)
                    self.builder.position_at_end(cond_bb)
                    ci = self.builder.load(idx_a)
                    self.builder.cbranch(self.builder.icmp_signed("<", ci, list_size), body_bb, end_bb)
                    self.builder.position_at_end(body_bb)
                    slot = self.builder.gep(list_data, [ci], inbounds=True)
                    slot_boxed = self.builder.bitcast(slot, BOXED_PTR)
                    etag = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 1)], inbounds=True))
                    epay = self.builder.load(self.builder.gep(slot_boxed, [z, ir.Constant(I32, 2)], inbounds=True))
                    elem = self._boxed_to_value(etag, epay, node)
                    self.list_append(self._generator_list, elem)
                    self.builder.store(self.builder.add(ci, ir.Constant(I64, 1)), idx_a)
                    self.builder.branch(cond_bb)
                    self.builder.position_at_end(end_bb)
            return iter_val

        return Value(ir.Constant(I64, 0), PyType.NONE)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Value:
        """Obsługa async def — kompiluje jak zwykłą funkcję, a następnie
        generuje coroutine entry wrapper dla natywnego asyncio.

        Model egzekucji: każda async funkcja kompiluje się do zwykłej
        funkcji LLVM (py_{name}) plus coroutine entry wrapper
        (py_{name}_coro_entry).  Wewnątrz funkcji, asyncio.sleep()
        wywołuje __async_sleep() z C runtime, który oddaje sterowanie
        do zarządcy zadań (scheduler) za pomocą ucontext swapcontext.
        """
        # Mark that we're compiling an async function
        old_is_async = getattr(self, '_is_async_function', False)
        self._is_async_function = True

        # Compile as a regular function
        result = self.visit_FunctionDef(node)

        # Generate the coroutine entry wrapper
        n_args = len(node.args.args)
        self._generate_coro_entry(node.name, n_args)

        self._is_async_function = old_is_async
        return result

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Obsługa async for — kompiluje jak zwykły for (synchronicznie)."""
        # In our synchronous model, async for is equivalent to regular for
        self.visit_For(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Obsługa async with — kompiluje jak zwykły with (synchronicznie)."""
        # In our synchronous model, async with is equivalent to regular with
        self.visit_With(node)

    # ──────────────────────────────────────────────────────────────
    #  Wyrażenia
    # ──────────────────────────────────────────────────────────────

    def visit_Global(self, node: ast.Global) -> None:
        """
        Obsługa instrukcji 'global x, y, ...'.
        Oznacza zmienne jako odnoszące się do zakresu globalnego.
        W naszej implementacji zmienne globalne są już dostępne w main scope,
        więc ta instrukcja głównie zapobiega tworzeniu lokalnych alokacji.
        """
        # Zapisz info, że te zmienne są globalne w bieżącym scope
        for name in node.names:
            if hasattr(self, "_global_vars"):
                self._global_vars.add(name)
        # Nie generujemy kodu - to tylko informacja dla kompilatora

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            if not hasattr(self, "_nonlocal_vars"):
                self._nonlocal_vars = set()
            self._nonlocal_vars.add(name)

    # ══════════════════════════════════════════════════════════════════
    #  Structural Pattern Matching (Python 3.10+)
    # ══════════════════════════════════════════════════════════════════

    def _collect_match_vars(self, pattern: ast.AST, names: list):
        """Rekurencyjnie zbierz nazwy zmiennych ze wzorca match."""
        if isinstance(pattern, ast.MatchAs):
            if pattern.name and pattern.name != '_':
                names.append(pattern.name)
            if pattern.pattern is not None:
                self._collect_match_vars(pattern.pattern, names)
        elif isinstance(pattern, ast.MatchSequence):
            for sub in pattern.patterns:
                self._collect_match_vars(sub, names)
        elif isinstance(pattern, ast.MatchMapping):
            for sub in pattern.patterns:
                self._collect_match_vars(sub, names)
        elif isinstance(pattern, ast.MatchClass):
            for sub in pattern.patterns:
                self._collect_match_vars(sub, names)
            for sub in pattern.kwd_patterns:
                self._collect_match_vars(sub, names)
        elif isinstance(pattern, ast.MatchOr):
            for sub in pattern.patterns:
                self._collect_match_vars(sub, names)
        elif isinstance(pattern, ast.MatchStar):
            if pattern.name and pattern.name != '_':
                names.append(pattern.name)
        # MatchValue nie wprowadza zmiennych

    def visit_Match(self, node: ast.Match) -> None:
        """Kompiluj match/case do łańcucha if-else w LLVM IR.

        Strategia: oblicz subject raz, zapisz w alloca. Dla każdego case:
        1. Generuj kod sprawdzający pattern (zwraca i1 — match/nomatch)
        2. Jeśli jest guard, dodaj warunek AND
        3. Jeśli match → wykonaj body i skocz do merge_bb
        4. Jeśli nomatch → skocz do następny case

        KLUCZOWE: Alokacje zmiennych match (x, y, name, itd.) MUSZĄ być
        utworzone PRZED rozgałęzieniem, żeby dominowały wszystkie bloki
        używające tych zmiennych (LLVM dominance rule).
        """
        # Oblicz subject raz
        subject_val = self.visit(node.subject)

        # Jeśli typ statyczny (INT/FLOAT/STR/LIST/DICT), boxuj do OBJECT
        if not subject_val.is_object:
            subject_boxed = self._box(subject_val)
            subject_val = Value(subject_boxed, PyType.OBJECT)

        # Zapisz subject do alloca żeby nie przeliczać
        subject_alloca = self.builder.alloca(BOXED_PTR, name="match_subject")
        self.builder.store(subject_val.llvm, subject_alloca)

        # ── KLUCZOWA NAPRAWA: Pre-alokuj zmienne match ──
        # Zbierz wszystkie nazwy zmiennych ze wzorców i utwórz ich
        # allocas PRZED rozgałęzieniem. Bez tego alloca w bloku
        # warunkowym (np. mseq_ok) nie dominuje bloku body,
        # co powoduje błąd LLVM „Instruction does not dominate all uses!"
        # i crash clang (Greedy Register Allocator SIGSEGV).
        all_match_vars = []
        for case in node.cases:
            self._collect_match_vars(case.pattern, all_match_vars)
        # Unikalne nazwy, zachowując kolejność
        seen = set()
        unique_vars = []
        for v in all_match_vars:
            if v not in seen:
                seen.add(v)
                unique_vars.append(v)
        # Pre-alokuj zmienne w bieżącym bloku (który dominuje wszystko)
        # KLUCZOWA NAPRAWA: Inicjalizuj allocas na null! Bez tego
        # _assign_name wczytuje garbage jako old_val i wywołuje
        # __py2llvm_decref na garbage → SIGSEGV.
        for var_name in unique_vars:
            if not self.sym.exists_local(var_name):
                alloca = self.builder.alloca(BOXED_PTR, name=var_name)
                self.builder.store(ir.Constant(BOXED_PTR, None), alloca)
                self.sym.define(var_name, VarInfo(alloca, BOXED_PTR, PyType.OBJECT))

        merge_bb = self.current_func.append_basic_block("match.end")

        for i, case in enumerate(node.cases):
            # Pobierz subject z alloca (dla każdego case)
            subject_loaded = Value(self.builder.load(subject_alloca, name="match_subj"), PyType.OBJECT)

            match_bb = self.current_func.append_basic_block(f"match.case{i}")
            next_bb = self.current_func.append_basic_block(f"match.next{i}")
            body_bb = self.current_func.append_basic_block(f"match.body{i}")

            # Pozycjonuj w bieżącym bloku → skocz do sprawdzania case
            self.builder.branch(match_bb)
            self.builder.position_at_end(match_bb)

            # Sprawdź pattern
            matched = self._compile_match_pattern(subject_loaded, case.pattern)

            # Jeśli jest guard, dodaj warunek AND — ALE tylko gdy pattern
            # pasuje! Jeśli guard wykonuje się bezwarunkowo (nawet gdy
            # pattern nie pasuje), czyta zmienne match (np. age) które
            # nie zostały przypisane → SIGSEGV.
            if case.guard is not None:
                guard_eval_bb = self.current_func.append_basic_block(f"match.guard{i}")
                guard_skip_bb = self.current_func.append_basic_block(f"match.gskip{i}")
                guard_cont_bb = self.current_func.append_basic_block(f"match.gcont{i}")

                # Jeśli pattern pasuje → ewaluuj guard; jeśli nie → pomiń
                self.builder.cbranch(matched, guard_eval_bb, guard_skip_bb)

                # Ewaluuj guard (tylko gdy pattern pasuje — zmienne są ważne)
                self.builder.position_at_end(guard_eval_bb)
                guard_val = self.visit(case.guard)
                if guard_val.is_bool:
                    guard_i1 = guard_val.llvm
                elif guard_val.is_int:
                    guard_i1 = self.builder.icmp_signed("!=", guard_val.llvm, ir.Constant(I64, 0))
                elif guard_val.is_float:
                    guard_i1 = self.builder.fcmp_ordered("!=", guard_val.llvm, ir.Constant(F64, 0.0))
                elif guard_val.is_object:
                    # Boxed value — check tag != NONE and payload != 0
                    g_tag, g_pay = self._read_slot(guard_val.llvm)
                    not_none = self.builder.icmp_signed("!=", g_tag, ir.Constant(I64, Tag.NONE))
                    not_zero = self.builder.icmp_signed("!=", g_pay, ir.Constant(I64, 0))
                    guard_i1 = self.builder.and_(not_none, not_zero, name="guard_truthy")
                else:
                    guard_i1 = self.builder.icmp_signed("!=", guard_val.llvm, ir.Constant(I64, 0))
                # Zapamiętaj rzeczywisty predecessor (guard_eval może tworzyć BB)
                guard_pred_bb = self.builder.block
                self.builder.branch(guard_cont_bb)

                # Pattern nie pasuje → pomiń guard, wynik = false
                self.builder.position_at_end(guard_skip_bb)
                self.builder.branch(guard_cont_bb)

                # Kontynuacja: phi — guard_pred_bb → guard_i1, guard_skip_bb → false
                self.builder.position_at_end(guard_cont_bb)
                matched = self.builder.phi(I1, f"match_guard{i}")
                matched.add_incoming(ir.Constant(I1, 0), guard_skip_bb)
                matched.add_incoming(guard_i1, guard_pred_bb)

            self.builder.cbranch(matched, body_bb, next_bb)

            # Kompiluj body
            self.builder.position_at_end(body_bb)
            for stmt in case.body:
                self.visit(stmt)
            self.builder.branch(merge_bb)

            # Przejdź do następnego case
            self.builder.position_at_end(next_bb)

        # Jeśli żaden case nie pasuje — po prostu skocz do merge
        self.builder.branch(merge_bb)
        self.builder.position_at_end(merge_bb)

    def _compile_match_pattern(self, subject: Value, pattern: ast.AST) -> ir.Value:
        """Kompiluj wzorzec dopasowania → zwraca i1 (matched/not matched).

        Obsługiwane wzorce:
        - MatchAs(pattern=None, name='_') → wildcard (zawsze pasuje)
        - MatchAs(pattern=None, name='x') → przechwyć subject jako zmienną x
        - MatchAs(pattern=sub_pattern, name='x') → dopasuj sub_pattern, przechwyć jako x
        - MatchSequence([p1, p2, ...]) → lista o określonej długości
        - MatchMapping(keys, patterns) → słownik z kluczami
        - MatchClass(cls, patterns) → dopasowanie typu (str, int, itd.)
        - MatchValue(value) → dokładna wartość (literal)
        - MatchOr([p1, p2]) → alternatywa
        - MatchStar(name) → rest of sequence (zapisz jako listę)
        """
        if isinstance(pattern, ast.MatchAs):
            return self._match_as(subject, pattern)
        elif isinstance(pattern, ast.MatchSequence):
            return self._match_sequence(subject, pattern)
        elif isinstance(pattern, ast.MatchMapping):
            return self._match_mapping(subject, pattern)
        elif isinstance(pattern, ast.MatchClass):
            return self._match_class(subject, pattern)
        elif isinstance(pattern, ast.MatchValue):
            return self._match_value(subject, pattern)
        elif isinstance(pattern, ast.MatchOr):
            return self._match_or(subject, pattern)
        elif isinstance(pattern, ast.MatchStar):
            return self._match_star(subject, pattern)
        else:
            raise CompileError(f"Nieobsługiwany wzorzec match: {type(pattern).__name__}", pattern)

    def _match_as(self, subject: Value, pattern: ast.MatchAs) -> ir.Value:
        """MatchAs — wildcard (_) lub przechwycenie zmiennej."""
        # Jeśli jest pod-pattern, dopasuj go najpierw
        if pattern.pattern is not None:
            matched = self._compile_match_pattern(subject, pattern.pattern)
            # Jeśli matched i jest nazwa, przypisz
            if pattern.name and pattern.name != '_':
                # Przypisz subject do zmiennej (tylko gdy match)
                # Używamy alloca + conditional store
                self._match_bind(subject, pattern.name, matched)
            return matched

        # Brak pod-patternu — zawsze pasuje
        if pattern.name and pattern.name != '_':
            # Przypisz subject do zmiennej
            self._assign_name(pattern.name, subject)
        return ir.Constant(I1, 1)  # Zawsze pasuje

    def _match_bind(self, subject: Value, name: str, matched_i1: ir.Value):
        """Przypisz subject do zmiennej warunkowo (tylko gdy matched)."""
        bind_bb = self.current_func.append_basic_block("mbind")
        skip_bb = self.current_func.append_basic_block("mskip")
        cont_bb = self.current_func.append_basic_block("mcont")
        self.builder.cbranch(matched_i1, bind_bb, skip_bb)

        self.builder.position_at_end(bind_bb)
        self._assign_name(name, subject)
        # UWAGA: _assign_name może tworzyć nowe BB (global/nonlocal),
        # więc builder może nie być już w bind_bb.
        self.builder.branch(cont_bb)

        self.builder.position_at_end(skip_bb)
        self.builder.branch(cont_bb)

        self.builder.position_at_end(cont_bb)

    def _match_sequence(self, subject: Value, pattern: ast.MatchSequence) -> ir.Value:
        """MatchSequence — dopasowanie listy o określonej długości."""
        n_patterns = len(pattern.patterns)

        # Odczytaj tag i payload z subject
        tag, pay = self._read_slot(subject.llvm)

        # Sprawdź czy tag to LIST lub TUPLE
        is_list = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST))
        is_tuple = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.TUPLE))
        is_seq = self.builder.or_(is_list, is_tuple, name="match_is_seq")

        ok_bb = self.current_func.append_basic_block("mseq_ok")
        fail_bb = self.current_func.append_basic_block("mseq_fail")
        len_bb = self.current_func.append_basic_block("mseq_len")
        self.builder.cbranch(is_seq, len_bb, fail_bb)

        # Sprawdź długość
        self.builder.position_at_end(len_bb)
        lptr = self.builder.inttoptr(pay, LIST_PTR)
        z = ir.Constant(I32, 0)
        list_len = self.builder.load(
            self.builder.gep(lptr, [z, ir.Constant(I32, 1)], inbounds=True),
            name="mseq_len"
        )
        len_ok = self.builder.icmp_signed("==", list_len, ir.Constant(I64, n_patterns))
        self.builder.cbranch(len_ok, ok_bb, fail_bb)

        # Długość się zgadza — dopasuj każdy sub-pattern
        self.builder.position_at_end(ok_bb)

        # Zapisz list_ptr do alloca żeby nie przeliczać
        list_val = Value(lptr, PyType.LIST)
        all_matched = ir.Constant(I1, 1)

        for i, sub_pattern in enumerate(pattern.patterns):
            # Pobierz element i-tej pozycji z listy
            elem = self.list_getitem(list_val, Value(ir.Constant(I64, i), PyType.INT))
            # Elem jest OBJECT (boxed) — dopasuj sub-pattern
            sub_matched = self._compile_match_pattern(elem, sub_pattern)
            all_matched = self.builder.and_(all_matched, sub_matched, name=f"mseq_and{i}")

        # UWAGA: po wywołaniach list_getitem / _compile_match_pattern builder
        # może być w innym bloku niż ok_bb (metody te tworzą wewnętrzne BB).
        # Musimy zapisać RZECZYWISTY blok poprzedzający result_bb.
        result_bb = self.current_func.append_basic_block("mseq_result")
        ok_pred_bb = self.builder.block  # rzeczywisty predecessor
        self.builder.branch(result_bb)

        # fail_bb → result_bb
        self.builder.position_at_end(fail_bb)
        self.builder.branch(result_bb)

        # Phi: ok_pred_bb → all_matched, fail_bb → false
        self.builder.position_at_end(result_bb)
        phi_matched = self.builder.phi(I1, "mseq_matched")
        phi_matched.add_incoming(ir.Constant(I1, 0), fail_bb)
        phi_matched.add_incoming(all_matched, ok_pred_bb)
        return phi_matched

    def _match_mapping(self, subject: Value, pattern: ast.MatchMapping) -> ir.Value:
        """MatchMapping — dopasowanie słownika z kluczami."""
        # Odczytaj tag i payload
        tag, pay = self._read_slot(subject.llvm)

        # Sprawdź czy tag to DICT
        is_dict = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.DICT))
        ok_bb = self.current_func.append_basic_block("mmap_ok")
        fail_bb = self.current_func.append_basic_block("mmap_fail")
        self.builder.cbranch(is_dict, ok_bb, fail_bb)

        # ok_bb: dopasuj klucze i sub-patterny
        self.builder.position_at_end(ok_bb)
        dptr = self.builder.inttoptr(pay, DICT_PTR)
        dict_val = Value(dptr, PyType.DICT)

        all_matched = ir.Constant(I1, 1)
        for key_node, sub_pattern in zip(pattern.keys, pattern.patterns):
            # Klucz to stała stringowa
            key_val = self.visit(key_node)
            # Pobierz wartość z dict
            elem = self.dict_getitem(dict_val, key_val)
            # Dopasuj sub-pattern
            sub_matched = self._compile_match_pattern(elem, sub_pattern)
            all_matched = self.builder.and_(all_matched, sub_matched, name="mmap_and")

        # UWAGA: dict_getitem / _compile_match_pattern mogą tworzyć nowe BB,
        # więc builder może nie być już w ok_bb.
        result_bb = self.current_func.append_basic_block("mmap_result")
        ok_pred_bb = self.builder.block  # rzeczywisty predecessor
        self.builder.branch(result_bb)

        # fail_bb: tag nie pasuje
        self.builder.position_at_end(fail_bb)
        self.builder.branch(result_bb)

        # result_bb: phi — ok_pred_bb → all_matched, fail_bb → false
        self.builder.position_at_end(result_bb)
        phi_matched = self.builder.phi(I1, "mmap_matched")
        phi_matched.add_incoming(ir.Constant(I1, 0), fail_bb)
        phi_matched.add_incoming(all_matched, ok_pred_bb)
        return phi_matched

    def _match_class(self, subject: Value, pattern: ast.MatchClass) -> ir.Value:
        """MatchClass — dopasowanie typu, np. str(text) lub int(x)."""
        cls_name = None
        if isinstance(pattern.cls, ast.Name):
            cls_name = pattern.cls.id

        # Mapowanie nazwy klasy na Tag
        _TYPE_TAG_MAP = {
            "str": Tag.STR,
            "int": Tag.INT,
            "float": Tag.FLOAT,
            "bool": Tag.BOOL,
            "list": Tag.LIST,
            "dict": Tag.DICT,
            "tuple": Tag.TUPLE,
            "set": Tag.SET,
        }

        if cls_name not in _TYPE_TAG_MAP:
            raise CompileError(f"Match: nieobsługiwany typ '{cls_name}' w pattern", pattern)

        expected_tag = _TYPE_TAG_MAP[cls_name]

        # Odczytaj tag z subject
        tag, pay = self._read_slot(subject.llvm)
        tag_match = self.builder.icmp_signed("==", tag, ir.Constant(I64, expected_tag), name="mclass_tag")

        # Jeśli typ pasuje, dopasuj sub-patterny i przypisz zmienne
        ok_bb = self.current_func.append_basic_block(f"mclass_ok")
        fail_bb = self.current_func.append_basic_block(f"mclass_fail")
        self.builder.cbranch(tag_match, ok_bb, fail_bb)

        # ok_bb: dopasuj sub-patterny
        self.builder.position_at_end(ok_bb)

        # Dla typów prostych (str, int, float) z jednym sub-patternem:
        # przypisz rozpakowaną wartość do zmiennej
        all_matched = ir.Constant(I1, 1)
        for i, sub_pattern in enumerate(pattern.patterns):
            # Rozpakuj wartość z tagged value
            unboxed = self._unbox_for_match(tag, pay, cls_name)
            sub_matched = self._compile_match_pattern(unboxed, sub_pattern)
            all_matched = self.builder.and_(all_matched, sub_matched, name=f"mclass_and{i}")

        # UWAGA: _compile_match_pattern może tworzyć nowe BB,
        # więc builder może nie być już w ok_bb.
        result_bb = self.current_func.append_basic_block(f"mclass_result")
        ok_pred_bb = self.builder.block  # rzeczywisty predecessor
        self.builder.branch(result_bb)

        # fail_bb: typ nie pasuje
        self.builder.position_at_end(fail_bb)
        self.builder.branch(result_bb)

        # result_bb: phi — ok_pred_bb → all_matched, fail_bb → false
        self.builder.position_at_end(result_bb)
        phi_matched = self.builder.phi(I1, "mclass_matched")
        phi_matched.add_incoming(ir.Constant(I1, 0), fail_bb)
        phi_matched.add_incoming(all_matched, ok_pred_bb)
        return phi_matched

    def _unbox_for_match(self, tag: ir.Value, pay: ir.Value, cls_name: str) -> Value:
        """Rozpakuj tagged value do Value konkretnego typu dla MatchClass."""
        if cls_name == "str":
            sptr = self.builder.inttoptr(pay, STR_PTR)
            return Value(sptr, PyType.STR)
        elif cls_name == "int":
            return Value(pay, PyType.INT)
        elif cls_name == "float":
            return Value(self.builder.bitcast(pay, F64), PyType.FLOAT)
        elif cls_name == "bool":
            return Value(self.builder.trunc(pay, I1), PyType.BOOL)
        elif cls_name == "list":
            lptr = self.builder.inttoptr(pay, LIST_PTR)
            return Value(lptr, PyType.LIST)
        elif cls_name == "dict":
            dptr = self.builder.inttoptr(pay, DICT_PTR)
            return Value(dptr, PyType.DICT)
        else:
            return Value(pay, PyType.OBJECT)

    def _match_value(self, subject: Value, pattern: ast.MatchValue) -> ir.Value:
        """MatchValue — dopasowanie dokładnej wartości (literal)."""
        val = self.visit(pattern.value)
        tag_s, pay_s = self._value_to_tag_payload(subject)
        tag_v, pay_v = self._value_to_tag_payload(val)

        # Ten sam tag?
        tag_ok = self.builder.icmp_signed("==", tag_s, tag_v, name="mval_tag")

        # Ten sam payload?
        pay_ok = self.builder.icmp_signed("==", pay_s, pay_v, name="mval_pay")

        return self.builder.and_(tag_ok, pay_ok, name="mval_matched")

    def _match_or(self, subject: Value, pattern: ast.MatchOr) -> ir.Value:
        """MatchOr — alternatywa wzorców (case x | y)."""
        result = ir.Constant(I1, 0)
        for sub_pattern in pattern.patterns:
            sub_matched = self._compile_match_pattern(subject, sub_pattern)
            result = self.builder.or_(result, sub_matched, name="mor")
        return result

    def _match_star(self, subject: Value, pattern: ast.MatchStar) -> ir.Value:
        """MatchStar — rest of sequence (*rest)."""
        # Uproszczenie: przypisz cały subject jako listę
        if pattern.name and pattern.name != '_':
            tag, pay = self._read_slot(subject.llvm)
            is_list = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.LIST))
            ok_bb = self.current_func.append_basic_block("mstar_ok")
            fail_bb = self.current_func.append_basic_block("mstar_fail")
            result_bb = self.current_func.append_basic_block("mstar_result")
            self.builder.cbranch(is_list, ok_bb, fail_bb)

            # ok_bb: to lista — przypisz do zmiennej
            self.builder.position_at_end(ok_bb)
            lptr = self.builder.inttoptr(pay, LIST_PTR)
            self._assign_name(pattern.name, Value(lptr, PyType.LIST))
            self.builder.branch(result_bb)

            # fail_bb: nie jest listą
            self.builder.position_at_end(fail_bb)
            self.builder.branch(result_bb)

            # result_bb: phi — ok_bb → true, fail_bb → false
            self.builder.position_at_end(result_bb)
            phi = self.builder.phi(I1, "mstar_matched")
            phi.add_incoming(ir.Constant(I1, 1), ok_bb)
            phi.add_incoming(ir.Constant(I1, 0), fail_bb)
            return phi
        return ir.Constant(I1, 1)

