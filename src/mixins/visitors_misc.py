"""AST visitor methods for subscript, attribute access, and utility helpers."""

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
from ..values import Value, FFIModuleValue
from ..type_analyzer import StaticTypeAnalyzer

if TYPE_CHECKING:
    pass


class VisitorsMiscMixin:
    """AST visitor methods for subscript, attribute access, and utility helpers."""

    # Mapa stałych modułów wbudowanych: (moduł, attr) → (PyType, wartość)
    # Używane przez visit_Attribute gdy obiekt to FFIModuleValue.
    _BUILTIN_MODULE_CONSTANTS = {
        ("math", "pi"):      (PyType.FLOAT, 3.141592653589793),
        ("math", "e"):       (PyType.FLOAT, 2.718281828459045),
        ("math", "tau"):     (PyType.FLOAT, 6.283185307179586),
        ("math", "inf"):     (PyType.FLOAT, float('inf')),
        ("sys",  "maxsize"): (PyType.INT, 2**63 - 1),
    }

    def visit_Subscript(self, node: ast.Subscript) -> Value:
        obj = self.visit(node.value)

        if isinstance(node.slice, ast.Slice):
            if obj.is_list:
                return self._handle_slice(obj, node.slice)
            if obj.is_str:
                # Statically typed string slicing
                # Check if step is negative at AST level for proper defaults
                step_is_negative = False
                if node.slice.step is not None:
                    step_node = node.slice.step
                    # Check for -N pattern (UnaryOp USub with constant)
                    if isinstance(step_node, ast.UnaryOp) and isinstance(step_node.op, ast.USub):
                        if isinstance(step_node.operand, ast.Constant) and isinstance(step_node.operand.value, int):
                            if step_node.operand.value > 0:
                                step_is_negative = True
                    # Check for negative constant directly
                    elif isinstance(step_node, ast.Constant) and isinstance(step_node.value, int):
                        if step_node.value < 0:
                            step_is_negative = True

                step_val = Value(ir.Constant(I64, 1), PyType.INT)
                if node.slice.step:
                    step_val = self.visit(node.slice.step)

                if step_is_negative:
                    # Negative step: default start = very large (will become len-1), default stop = very negative
                    start_val = Value(ir.Constant(I64, 999999999), PyType.INT) if not node.slice.lower else self.visit(node.slice.lower)
                    stop_val = Value(ir.Constant(I64, -999999999), PyType.INT) if not node.slice.upper else self.visit(node.slice.upper)
                else:
                    start_val = Value(ir.Constant(I64, 0), PyType.INT) if not node.slice.lower else self.visit(node.slice.lower)
                    stop_val = Value(ir.Constant(I64, 999999999), PyType.INT) if not node.slice.upper else self.visit(node.slice.upper)

                return self.string_slice(
                    obj,
                    self._to_int(start_val).llvm,
                    self._to_int(stop_val).llvm,
                    self._to_int(step_val).llvm,
                )
            if obj.is_object:
                tag, pay = self._read_slot(obj.llvm)
                str_bb = self.current_func.append_basic_block("sub_sl.str")
                err_bb = self.current_func.append_basic_block("sub_sl.err")
                end_bb = self.current_func.append_basic_block("sub_sl.end")

                res = self.builder.alloca(BOXED_PTR, name="sub_sl_res")
                sw = self.builder.switch(tag, err_bb)
                sw.add_case(ir.Constant(I64, Tag.STR), str_bb)

                self.builder.position_at_end(str_bb)
                sptr = self.builder.inttoptr(pay, STR_PTR)

                # Check if step is negative at AST level for proper defaults
                step_is_negative = False
                if node.slice.step is not None:
                    step_node = node.slice.step
                    if isinstance(step_node, ast.UnaryOp) and isinstance(step_node.op, ast.USub):
                        if isinstance(step_node.operand, ast.Constant) and isinstance(step_node.operand.value, int):
                            if step_node.operand.value > 0:
                                step_is_negative = True
                    elif isinstance(step_node, ast.Constant) and isinstance(step_node.value, int):
                        if step_node.value < 0:
                            step_is_negative = True

                step_val = Value(ir.Constant(I64, 1), PyType.INT)
                if node.slice.step:
                    step_val = self.visit(node.slice.step)

                if step_is_negative:
                    start_val = Value(ir.Constant(I64, 999999999), PyType.INT) if not node.slice.lower else self.visit(node.slice.lower)
                    stop_val = Value(ir.Constant(I64, -999999999), PyType.INT) if not node.slice.upper else self.visit(node.slice.upper)
                else:
                    start_val = Value(ir.Constant(I64, 0), PyType.INT) if not node.slice.lower else self.visit(node.slice.lower)
                    stop_val = Value(ir.Constant(I64, 999999999), PyType.INT) if not node.slice.upper else self.visit(node.slice.upper)

                v_sl = self.string_slice(
                    Value(sptr, PyType.STR),
                    self._to_int(start_val).llvm,
                    self._to_int(stop_val).llvm,
                    self._to_int(step_val).llvm,
                )
                self.builder.store(self._box(v_sl), res)
                self.builder.branch(end_bb)

                self.builder.position_at_end(err_bb)
                self.builder.store(ir.Constant(BOXED_PTR, None), res)
                self.builder.branch(end_bb)

                self.builder.position_at_end(end_bb)
                return Value(self.builder.load(res), PyType.OBJECT)
            raise self._error(
                ErrorCategory.UNSUPPORTED,
                f"Slicing is not supported for {obj.pytype.name}",
                node,
                help_text="Only lists support slicing in pylow.",
            )


        key = self.visit(node.slice)
        if obj.is_list:
            return self.list_getitem(obj, key)
        if obj.is_dict:
            return self.dict_getitem(obj, key)
        if obj.is_object:
            tag, pay = self._read_slot(obj.llvm)

            lst_bb = self.current_func.append_basic_block("sub.lst")
            dct_bb = self.current_func.append_basic_block("sub.dct")
            str_bb = self.current_func.append_basic_block("sub.str")
            err_bb = self.current_func.append_basic_block("sub.err")
            end_bb = self.current_func.append_basic_block("sub.end")

            res = self.builder.alloca(BOXED_PTR, name="sub_res")

            sw = self.builder.switch(tag, err_bb)
            sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)
            sw.add_case(ir.Constant(I64, Tag.TUPLE), lst_bb)  # TUPLE indeksowanie identyczne jak LIST
            sw.add_case(ir.Constant(I64, Tag.SET), lst_bb)
            sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
            sw.add_case(ir.Constant(I64, Tag.STR), str_bb)

            # For list subscript, we need to convert key to int at runtime
            self.builder.position_at_end(lst_bb)
            lptr = self.builder.inttoptr(pay, LIST_PTR)
            # Convert key to int at runtime
            ktag, kpay = self._value_to_tag_payload(key)
            int_bb_sub = self.current_func.append_basic_block("sub.lst.int")
            err_bb_sub = self.current_func.append_basic_block("sub.lst.err")
            end_bb_sub = self.current_func.append_basic_block("sub.lst.end")
            idx_res = self.builder.alloca(I64, name="sub_lst_idx")
            self.builder.store(ir.Constant(I64, 0), idx_res)
            sw_key = self.builder.switch(ktag, err_bb_sub)
            sw_key.add_case(ir.Constant(I64, Tag.INT), int_bb_sub)
            self.builder.position_at_end(int_bb_sub)
            self.builder.store(kpay, idx_res)
            self.builder.branch(end_bb_sub)
            self.builder.position_at_end(err_bb_sub)
            self.builder.store(ir.Constant(I64, 0), idx_res)
            self.builder.branch(end_bb_sub)
            self.builder.position_at_end(end_bb_sub)
            idx_val = Value(self.builder.load(idx_res), PyType.INT)
            v_lst = self.list_getitem(Value(lptr, PyType.LIST), idx_val)
            self.builder.store(v_lst.llvm, res)
            self.builder.branch(end_bb)

            # For dict subscript, pass key's tag and payload
            self.builder.position_at_end(dct_bb)
            dptr = self.builder.inttoptr(pay, DICT_PTR)
            v_dct = self.dict_getitem(Value(dptr, PyType.DICT), key)
            self.builder.store(v_dct.llvm, res)
            self.builder.branch(end_bb)

            # For string subscript, we need to convert key to int at runtime
            self.builder.position_at_end(str_bb)
            sptr = self.builder.inttoptr(pay, STR_PTR)
            # Convert key to int at runtime
            ktag2, kpay2 = self._value_to_tag_payload(key)
            int_bb_sub2 = self.current_func.append_basic_block("sub.str.int")
            err_bb_sub2 = self.current_func.append_basic_block("sub.str.err")
            end_bb_sub2 = self.current_func.append_basic_block("sub.str.end")
            idx_res2 = self.builder.alloca(I64, name="sub_str_idx")
            self.builder.store(ir.Constant(I64, 0), idx_res2)
            sw_key2 = self.builder.switch(ktag2, err_bb_sub2)
            sw_key2.add_case(ir.Constant(I64, Tag.INT), int_bb_sub2)
            self.builder.position_at_end(int_bb_sub2)
            self.builder.store(kpay2, idx_res2)
            self.builder.branch(end_bb_sub2)
            self.builder.position_at_end(err_bb_sub2)
            self.builder.store(ir.Constant(I64, 0), idx_res2)
            self.builder.branch(end_bb_sub2)
            self.builder.position_at_end(end_bb_sub2)
            idx_val2 = Value(self.builder.load(idx_res2), PyType.INT)
            v_str = self.string_getitem(Value(sptr, PyType.STR), idx_val2)
            self.builder.store(v_str.llvm, res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(err_bb)
            self.builder.store(ir.Constant(BOXED_PTR, None), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(end_bb)
            return Value(self.builder.load(res), PyType.OBJECT)
        raise self._error(
            ErrorCategory.UNSUPPORTED,
            f"Indexing is not supported for {obj.pytype.name}",
            node,
            help_text="Only lists and dicts support indexing in pylow.",
        )

    def visit_Attribute(self, node: ast.Attribute) -> Value:
        """Obsługa atrybutów: obj.attr (dostęp do składowej)."""
        obj = self.visit(node.value)
        attr_name = node.attr

        # ═══════════════════════════════════════════════════════════
        #  FFIModuleValue: attr access on imported module (e.g. time.time)
        #  Return FFIModuleValue so that visit_Call/_method_call
        #  can dispatch it properly. Don't try to _read_slot(None).
        # ═══════════════════════════════════════════════════════════
        if isinstance(obj, FFIModuleValue):
            # Check for module constants first (math.pi, math.e, sys.maxsize, etc.)
            const_key = (obj.module_name, attr_name)
            if const_key in self._BUILTIN_MODULE_CONSTANTS:
                pytype, raw_val = self._BUILTIN_MODULE_CONSTANTS[const_key]
                if pytype == PyType.FLOAT:
                    return Value(ir.Constant(F64, raw_val), PyType.FLOAT)
                elif pytype == PyType.INT:
                    return Value(ir.Constant(I64, raw_val), PyType.INT)
            # Method/attribute access — return FFIModuleValue for _method_call dispatch
            return FFIModuleValue(obj.module_name)

        # Statycznie znany typ STRING – obsługa atrybutów stringa (np. format)
        if obj.is_str:
            # String nie ma dostępnych atrybutów jako wartości –
            # metody są obsługiwane przez _method_call w visit_Call.
            # Jeśli ktoś odwołuje się do atrybutu (nie metody), zgłoś błąd.
            raise self._error(
                ErrorCategory.SEMANTIC,
                f"str has no attribute '{attr_name}'",
                node,
                help_text="String methods are called with dot syntax: s.upper(), s.lower(), etc.",
            )

        # Statycznie znany typ - dla uproszczonej implementacji klas, self jest DICT
        if obj.is_dict:
            # Dostęp do atrybutów (słownik)
            key = self.create_string(attr_name)
            result = self.dict_getitem(obj, key)
            return result

        if obj.is_instance:
            # NAPRAWA: Check if this attribute is a property with a getter
            inferred_class = obj.class_name
            if not inferred_class and hasattr(node, 'value') and isinstance(node.value, ast.Name):
                var_name = node.value.id
                try:
                    var_info = self.sym.lookup(var_name)
                    if hasattr(var_info, "class_name") and var_info.class_name:
                        inferred_class = var_info.class_name
                except: pass

            if inferred_class:
                class_props = getattr(self, '_class_properties', {}).get(inferred_class, {})
                if attr_name in class_props and 'getter' in class_props[attr_name]:
                    getter_name = class_props[attr_name]['getter']
                    if getter_name in self.functions:
                        getter_func = self.functions[getter_name]
                        inst_ptr = obj.llvm
                        first_arg_type = getter_func.args[0].type if getter_func.args else BOXED_PTR
                        if first_arg_type == INSTANCE_PTR:
                            self_arg = inst_ptr
                        elif first_arg_type == DICT_PTR:
                            z = ir.Constant(I32, 0)
                            self_arg = self.builder.load(self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True))
                        else:
                            self_arg = self._box(obj)
                        self_arg_list = self._verify_call_args(getter_func, [self_arg])
                        ret = self.builder.call(getter_func, self_arg_list)
                        self._check_exc_after_call()
                        ret_type = self._llvm_type_to_pytype(getter_func.function_type.return_type)
                        return Value(ret, ret_type)

            # Dostęp do atrybutów instancji
            inst_ptr = obj.llvm
            z = ir.Constant(I32, 0)
            attrs_ptr = self.builder.load(
                self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
            )
            # Look up in instance attrs dict
            key = self.create_string(attr_name)
            # Use dict_getitem
            dct_val = Value(attrs_ptr, PyType.DICT)
            result = self.dict_getitem(dct_val, key)
            return result

        # NAPRAWA: Check for property getter on boxed (OBJECT) instances
        if obj.is_object:
            inferred_class = obj.class_name
            if not inferred_class and hasattr(node, 'value') and isinstance(node.value, ast.Name):
                var_name = node.value.id
                try:
                    var_info = self.sym.lookup(var_name)
                    if hasattr(var_info, "class_name") and var_info.class_name:
                        inferred_class = var_info.class_name
                except: pass

            if inferred_class:
                class_props = getattr(self, '_class_properties', {}).get(inferred_class, {})
                if attr_name in class_props and 'getter' in class_props[attr_name]:
                    getter_name = class_props[attr_name]['getter']
                    if getter_name in self.functions:
                        getter_func = self.functions[getter_name]
                        # obj is boxed, extract instance from payload
                        tag, pay = self._read_slot(obj.llvm)
                        inst_tag = ir.Constant(I64, Tag.INST)
                        is_inst = self.builder.icmp_signed("==", tag, inst_tag)
                        inst_bb = self.current_func.append_basic_block("prop_get.inst")
                        fallback_bb = self.current_func.append_basic_block("prop_get.fallback")
                        end_bb = self.current_func.append_basic_block("prop_get.end")
                        res_alloca = self.builder.alloca(BOXED_PTR, name="prop_get_res")

                        self.builder.cbranch(is_inst, inst_bb, fallback_bb)

                        self.builder.position_at_end(inst_bb)
                        inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
                        first_arg_type = getter_func.args[0].type if getter_func.args else BOXED_PTR
                        if first_arg_type == INSTANCE_PTR:
                            self_arg = inst_ptr
                        elif first_arg_type == DICT_PTR:
                            z = ir.Constant(I32, 0)
                            self_arg = self.builder.load(self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True))
                        else:
                            self_arg = obj.llvm
                        self_arg_list = self._verify_call_args(getter_func, [self_arg])
                        ret = self.builder.call(getter_func, self_arg_list)
                        self._check_exc_after_call()
                        ret_boxed = ret if ret.type == BOXED_PTR else self._box(Value(ret, self._llvm_type_to_pytype(getter_func.function_type.return_type)))
                        self.builder.store(ret_boxed, res_alloca)
                        self.builder.branch(end_bb)

                        self.builder.position_at_end(fallback_bb)
                        # Not an instance — fall through to dict lookup
                        self.builder.store(ir.Constant(BOXED_PTR, None), res_alloca)
                        self.builder.branch(end_bb)

                        self.builder.position_at_end(end_bb)
                        return Value(self.builder.load(res_alloca), PyType.OBJECT)

        # Dynamiczny dispatch po tagu (dla OBJECT i INSTANCE)
        if obj.pytype.name in ("OBJECT", "INSTANCE"):
            tag, pay = self._read_slot(obj.llvm)

            dct_bb = self.current_func.append_basic_block("attr.dct")
            inst_bb = self.current_func.append_basic_block("attr.inst")
            err_bb = self.current_func.append_basic_block("attr.err")
            end_bb = self.current_func.append_basic_block("attr.end")

            res = self.builder.alloca(BOXED_PTR, name="attr_res")

            sw = self.builder.switch(tag, err_bb)
            sw.add_case(ir.Constant(I64, Tag.DICT), dct_bb)
            sw.add_case(ir.Constant(I64, Tag.INST), inst_bb)

            self.builder.position_at_end(dct_bb)
            key = self.create_string(attr_name)
            dct_val = Value(self.builder.inttoptr(pay, DICT_PTR), PyType.DICT)
            v = self.dict_getitem(dct_val, key)
            self.builder.store(v.llvm, res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(inst_bb)
            inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
            z = ir.Constant(I32, 0)
            attrs_ptr = self.builder.load(
                self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True)
            )
            key = self.create_string(attr_name)
            dct_val = Value(attrs_ptr, PyType.DICT)
            v = self.dict_getitem(dct_val, key)
            self.builder.store(v.llvm, res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(err_bb)
            self.builder.store(ir.Constant(BOXED_PTR, None), res)
            self.builder.branch(end_bb)

            self.builder.position_at_end(end_bb)
            return Value(self.builder.load(res), PyType.OBJECT)

        # Dla klas (jesli mamy dostęp do atrybutów klasy)
        raise self._error(
            ErrorCategory.UNSUPPORTED,
            f"Attribute access is not supported for {obj.pytype.name}",
            node,
            help_text="Only instances, dicts, and objects support attribute access.",
        )

    # ──────────────────────────────────────────────────────────────
    #  Exception raising helpers
    # ──────────────────────────────────────────────────────────────

    def string_slice(self, str_val: Value, start: ir.Value, stop: ir.Value, step: ir.Value = None) -> Value:
        """s[start:stop:step] -> nowa STR (BOXED). Obsługuje krok < 0"""
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)

        s_ptr = str_val.llvm
        sp, cp, dp = self._str_ptrs(s_ptr)
        size = self.builder.load(sp, "sl_sz")

        actual_step = step if step else ir.Constant(I64, 1)
        is_rev = self.builder.icmp_signed("<", actual_step, ir.Constant(I64, 0))
        abs_step = self.builder.select(is_rev, self.builder.sub(ir.Constant(I64, 0), actual_step), actual_step)

        # Handle negative start/stop
        is_neg_start = self.builder.icmp_signed("<", start, ir.Constant(I64, 0))
        start_final = self.builder.select(is_neg_start, self.builder.add(start, size), start)

        is_neg_stop = self.builder.icmp_signed("<", stop, ir.Constant(I64, 0))
        stop_final = self.builder.select(is_neg_stop, self.builder.add(stop, size), stop)

        # Clamp for positive step: [0, size]
        # For negative step: start clamp to [0, size-1], stop clamp to [-1, size]
        # Following CPython's PySlice_AdjustIndices
        pos_start_clamped = self.builder.select(
            self.builder.icmp_signed("<", start_final, ir.Constant(I64, 0)), ir.Constant(I64, 0), start_final)
        pos_start_clamped = self.builder.select(
            self.builder.icmp_signed(">", pos_start_clamped, size), size, pos_start_clamped)

        pos_stop_clamped = self.builder.select(
            self.builder.icmp_signed("<", stop_final, ir.Constant(I64, 0)), ir.Constant(I64, 0), stop_final)
        pos_stop_clamped = self.builder.select(
            self.builder.icmp_signed(">", pos_stop_clamped, size), size, pos_stop_clamped)

        # For negative step: start clamped to [-1, size-1], stop clamped to [-1, size-1]
        size_minus_1 = self.builder.sub(size, ir.Constant(I64, 1))
        neg_start_clamped = self.builder.select(
            self.builder.icmp_signed("<", start_final, ir.Constant(I64, 0)), ir.Constant(I64, -1), start_final)
        neg_start_clamped = self.builder.select(
            self.builder.icmp_signed(">", neg_start_clamped, size_minus_1), size_minus_1, neg_start_clamped)

        neg_stop_clamped = self.builder.select(
            self.builder.icmp_signed("<", stop_final, ir.Constant(I64, 0)), ir.Constant(I64, -1), stop_final)
        neg_stop_clamped = self.builder.select(
            self.builder.icmp_signed(">", neg_stop_clamped, size_minus_1), size_minus_1, neg_stop_clamped)

        start_clamped = self.builder.select(is_rev, neg_start_clamped, pos_start_clamped)
        stop_clamped = self.builder.select(is_rev, neg_stop_clamped, pos_stop_clamped)

        # Effective start for iteration
        eff_start = start_clamped

        # Calculate result length
        # Positive step: if start >= stop → 0, else (stop - start + step - 1) / step
        # Negative step: if start <= stop → 0, else (start - stop - step - 1) / (-step)
        # Simplified using CPython formula:
        #   if step > 0: len = max(0, (stop - start - 1) // step + 1) when start < stop
        #   if step < 0: len = max(0, (start - stop - 1) // (-step) + 1) when start > stop
        pos_diff = self.builder.sub(pos_stop_clamped, pos_start_clamped)
        pos_raw_len = self.builder.add(self.builder.sdiv(self.builder.sub(pos_diff, ir.Constant(I64, 1)), actual_step), ir.Constant(I64, 1))
        pos_raw_len = self.builder.select(self.builder.icmp_signed("<=", pos_diff, ir.Constant(I64, 0)), ir.Constant(I64, 0), pos_raw_len)
        pos_raw_len = self.builder.select(self.builder.icmp_signed("<", pos_raw_len, ir.Constant(I64, 0)), ir.Constant(I64, 0), pos_raw_len)

        neg_diff = self.builder.sub(neg_start_clamped, neg_stop_clamped)
        neg_raw_len = self.builder.add(self.builder.sdiv(self.builder.sub(neg_diff, ir.Constant(I64, 1)), abs_step), ir.Constant(I64, 1))
        neg_raw_len = self.builder.select(self.builder.icmp_signed("<=", neg_diff, ir.Constant(I64, 0)), ir.Constant(I64, 0), neg_raw_len)
        neg_raw_len = self.builder.select(self.builder.icmp_signed("<", neg_raw_len, ir.Constant(I64, 0)), ir.Constant(I64, 0), neg_raw_len)

        raw_len = self.builder.select(is_rev, neg_raw_len, pos_raw_len)

        # Alokacja
        raw_s = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)], "sl_raw")
        new_s = self.builder.bitcast(raw_s, STR_PTR, "new_s")
        raw_d = self.builder.call(self._malloc, [self.builder.add(raw_len, ir.Constant(I64, 1))], "sl_data")
        new_d = self.builder.bitcast(raw_d, I8P, "new_d")

        # init gc
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(new_s, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(new_s, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(new_s, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(null_i8p, self.builder.gep(new_s, [z, z, ir.Constant(I32, 3)], inbounds=True))
        self.builder.store(raw_len, self.builder.gep(new_s, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(self.builder.add(raw_len, ir.Constant(I64, 1)), self.builder.gep(new_s, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(new_d, self.builder.gep(new_s, [z, ir.Constant(I32, 3)], inbounds=True))

        src_data = self.builder.load(dp, "sd_slice")
        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)

        cond_bb = self.current_func.append_basic_block("sl.cond")
        body_bb = self.current_func.append_basic_block("sl.body")
        end_bb = self.current_func.append_basic_block("sl.end")
        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("<", self.builder.load(i_a), raw_len), body_bb, end_bb)

        self.builder.position_at_end(body_bb)
        curr_i = self.builder.load(i_a)
        # offset = start + i * step
        offset = self.builder.add(eff_start, self.builder.mul(curr_i, actual_step))
        src_ptr = self.builder.gep(src_data, [offset], inbounds=True)
        dst_ptr = self.builder.gep(new_d, [curr_i], inbounds=True)
        self.builder.store(self.builder.load(src_ptr), dst_ptr)
        self.builder.store(self.builder.add(curr_i, ir.Constant(I64, 1)), i_a)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(end_bb)
        self.builder.store(ir.Constant(I8, 0), self.builder.gep(new_d, [raw_len], inbounds=True))

        boxed_s = self._box(Value(new_s, PyType.STR))
        return Value(boxed_s, PyType.OBJECT)

    def _raise_exception(self, exc_type_name: str, exc_msg: str = ""):
        """Raise an exception by storing type info and branching to the
        first exception handler on the stack.  Only call this when
        self._exc_handler_stack is non-empty."""
        handler_info = self._exc_handler_stack[-1]

        # Store the exception type hash so type-matching in except clauses works
        if handler_info.get("exc_type_alloca"):
            type_hash = hash(exc_type_name) & 0xFFFFFFFFFFFFFFFF
            self.builder.store(
                ir.Constant(I64, type_hash),
                handler_info["exc_type_alloca"],
            )

        # Store a boxed exception value (string message) if caught_exc exists
        if handler_info.get("caught_exc"):
            msg_val = self.create_string(exc_msg if exc_msg else exc_type_name)
            boxed = self._box(msg_val)
            self.builder.store(boxed, handler_info["caught_exc"])

        # Branch to the first handler block
        if handler_info["handlers"]:
            _, first_handler_bb = handler_info["handlers"][0]
            self.builder.branch(first_handler_bb)
        elif handler_info.get("unhandled_bb"):
            self.builder.branch(handler_info["unhandled_bb"])
        else:
            # No handler at all – shouldn't happen, but just return None
            rt = self.current_func.function_type.return_type
            if isinstance(rt, ir.VoidType):
                self.builder.ret_void()
            elif rt == BOXED_PTR:
                self.builder.ret(ir.Constant(BOXED_PTR, None))
            elif rt == I32:
                self.builder.ret(ir.Constant(I32, 1))
            else:
                self.builder.ret_void()

    def _emit_div_zero_check(self, divisor_val: Value, is_float: bool = False):
        """Check for division by zero when inside a try block.
        If divisor is zero, raises ZeroDivisionError.
        After this call the builder is positioned at the 'ok' continuation block."""
        if not self._exc_handler_stack:
            return  # Not inside try – no check needed

        if is_float:
            is_zero = self.builder.fcmp_ordered(
                "==", divisor_val.llvm, ir.Constant(F64, 0.0)
            )
        else:
            is_zero = self.builder.icmp_signed(
                "==", divisor_val.llvm, ir.Constant(I64, 0)
            )

        zero_bb = self.current_func.append_basic_block("div.zero")
        cont_bb = self.current_func.append_basic_block("div.ok")
        self.builder.cbranch(is_zero, zero_bb, cont_bb)

        # zero path → raise ZeroDivisionError
        self.builder.position_at_end(zero_bb)
        self._raise_exception("ZeroDivisionError", "division by zero")

        # continue path
        self.builder.position_at_end(cont_bb)

    def _emit_index_bounds_check(self, idx_i64: ir.Value, size_i64: ir.Value):
        """Check for out-of-bounds index when inside a try block.
        If index >= size or index < 0 (before adjustment), raises IndexError.
        After this call the builder is positioned at the 'ok' continuation block."""
        if not self._exc_handler_stack:
            return  # Not inside try – no check needed

        out_of_bounds = self.builder.or_(
            self.builder.icmp_signed("<", idx_i64, ir.Constant(I64, 0)),
            self.builder.icmp_signed(">=", idx_i64, size_i64),
        )

        oob_bb = self.current_func.append_basic_block("idx.oob")
        cont_bb = self.current_func.append_basic_block("idx.ok")
        self.builder.cbranch(out_of_bounds, oob_bb, cont_bb)

        # out-of-bounds path → raise IndexError
        self.builder.position_at_end(oob_bb)
        self._raise_exception("IndexError", "list index out of range")

        # continue path
        self.builder.position_at_end(cont_bb)

    # ──────────────────────────────────────────────────────────────
    #  Cross-function exception propagation helpers
    # ──────────────────────────────────────────────────────────────

    def _set_exc_global_state(self, exc_type_name: str, exc_value):
        """Set the global exception state so callers can detect and
        propagate the exception. Called from visit_Raise when there's
        no handler in the current function scope."""
        # __py2llvm_exc_pending = 1
        self.builder.store(ir.Constant(I1, 1), self._exc_pending_global)
        # __py2llvm_exc_type_hash = hash(exc_type_name)
        if exc_type_name:
            type_hash = hash(exc_type_name) & 0xFFFFFFFFFFFFFFFF
            self.builder.store(ir.Constant(I64, type_hash), self._exc_type_hash_global)
        else:
            self.builder.store(ir.Constant(I64, 0), self._exc_type_hash_global)
        # __py2llvm_exc_value = exc_value
        if exc_value:
            boxed = exc_value.llvm if exc_value.is_object else self._box(exc_value)
            self.builder.store(boxed, self._exc_value_global)
        else:
            self.builder.store(ir.Constant(BOXED_PTR, None), self._exc_value_global)

    def _check_exc_after_call(self, return_value=None):
        """Check if an exception was raised during a called function.
        If so, propagate it to the current function's exception handler
        (if any) or re-set globals and return. Builder is positioned at
        the continuation block after this call.

        Args:
            return_value: If provided, the LLVM value that was returned by
                          the called function. Not currently used but kept
                          for future extension.

        Returns:
            True if the builder is at a normal continuation block (no exc),
            False if the builder was terminated (exception was propagated).
        """
        pending = self.builder.load(self._exc_pending_global, "exc_pending")

        exc_bb = self.current_func.append_basic_block("exc.propagate")
        cont_bb = self.current_func.append_basic_block("exc.continue")

        self.builder.cbranch(pending, exc_bb, cont_bb)

        # Exception propagation path
        self.builder.position_at_end(exc_bb)

        # Clear the pending flag (we're handling it now)
        self.builder.store(ir.Constant(I1, 0), self._exc_pending_global)

        # Load exception state
        exc_type_hash = self.builder.load(self._exc_type_hash_global, "exc_hash")
        exc_val = self.builder.load(self._exc_value_global, "exc_val")

        if self._exc_handler_stack:
            handler_info = self._exc_handler_stack[-1]
            # Store exception info in handler's allocas
            if handler_info.get("exc_type_alloca"):
                self.builder.store(exc_type_hash, handler_info["exc_type_alloca"])
            if handler_info.get("caught_exc"):
                self.builder.store(exc_val, handler_info["caught_exc"])
            # Branch to first handler
            if handler_info["handlers"]:
                _, first_handler = handler_info["handlers"][0]
                self.builder.branch(first_handler)
            elif handler_info.get("finally_bb"):
                self.builder.branch(handler_info["finally_bb"])
            else:
                # No handler — re-set globals and return
                self.builder.store(ir.Constant(I1, 1), self._exc_pending_global)
                rt = self.current_func.function_type.return_type
                if isinstance(rt, ir.VoidType):
                    self.builder.ret_void()
                elif rt == BOXED_PTR:
                    self.builder.ret(ir.Constant(BOXED_PTR, None))
                elif rt == I32:
                    self.builder.ret(ir.Constant(I32, 1))
                else:
                    self.builder.ret_void()
        else:
            # No handler in caller either — keep globals set and return
            # (propagate further up the call stack)
            self.builder.store(ir.Constant(I1, 1), self._exc_pending_global)
            rt = self.current_func.function_type.return_type
            if isinstance(rt, ir.VoidType):
                self.builder.ret_void()
            elif rt == BOXED_PTR:
                self.builder.ret(ir.Constant(BOXED_PTR, None))
            elif rt == I32:
                self.builder.ret(ir.Constant(I32, 1))
            else:
                self.builder.ret_void()

        # Normal continuation path
        self.builder.position_at_end(cont_bb)
        return True

    def _clear_exc_pending(self):
        """Clear the global exception pending flag. Called at program start
        and after successfully handling an exception."""
        self.builder.store(ir.Constant(I1, 0), self._exc_pending_global)

