################################################################################

"""AST visitor for function/method calls and string method implementations."""

from __future__ import annotations

import ast
import os
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
from ..values import Value, FFIModuleValue
from ..type_analyzer import StaticTypeAnalyzer

from ..ffi.core import FFISignatureDB, RET_PYOBJ, RET_VOID, RET_INT, RET_DOUBLE, RET_CONSTCHAR, RET_VOIDPTR, RET_DATA, ret_type_to_llvm_ir, has_cpython_extensions

if TYPE_CHECKING:
    pass


class VisitorsCallMixin:
    """AST visitor for function/method calls and string method implementations."""

    def visit_Call(self, node: ast.Call) -> Value:
        if isinstance(node.func, ast.Attribute):
            return self._method_call(node)
        if not isinstance(node.func, ast.Name):
            raise CompileError("Obsługiwane są tylko proste wywołania f(args).", node)

        fname = node.func.id

        # ══════════════════════════════════════════════════════════════════
        #  __call__ dispatch: jeśli fname jest zmienną lokalną przechowującą
        #  INSTANCJĘ klasy z metodą __call__, wywołaj __call__ na tej instancji.
        #  KLUCZOWE: Musi być PRZED classref resolution, żeby nie potraktować
        #  instancji jako klasy. Odróżniamy po is_class_ref w VarInfo.
        # ══════════════════════════════════════════════════════════════════
        try:
            var_info_call = self.sym.lookup(fname)
            is_class_ref = getattr(var_info_call, 'is_class_ref', False)
            if (var_info_call is not None
                and hasattr(var_info_call, 'class_name') and var_info_call.class_name
                and not is_class_ref  # NOT a class reference — this is an instance
                and f"py_{var_info_call.class_name}___call__" in self.functions):
                # fname is an instance with __call__ — call it
                call_target = f"py_{var_info_call.class_name}___call__"
                call_func = self.functions[call_target]
                # Determine what type the function expects for its first arg (self)
                expected_self_type = call_func.args[0].type if call_func.args else BOXED_PTR
                # Load the instance from the variable
                obj_val = self.visit(ast.Name(id=fname, ctx=ast.Load()))
                # Build self arg matching the function's expected type
                if expected_self_type == INSTANCE_PTR:
                    # Function expects INSTANCE_PTR — extract instance pointer
                    if obj_val.llvm.type == INSTANCE_PTR:
                        self_arg = obj_val.llvm
                    elif obj_val.is_object and obj_val.llvm.type == BOXED_PTR:
                        tag, pay = self._read_slot(obj_val.llvm)
                        self_arg = self.builder.inttoptr(pay, INSTANCE_PTR, "call_self")
                    else:
                        self_arg = self._box(obj_val)
                else:
                    # Function expects BOXED_PTR — pass boxed value
                    if obj_val.llvm.type == BOXED_PTR:
                        self_arg = obj_val.llvm
                    elif obj_val.llvm.type == INSTANCE_PTR:
                        self_arg = self._box(obj_val)
                    else:
                        self_arg = self._box(obj_val)
                # Build args: self + user args
                call_args = [self_arg]
                for i, arg in enumerate(node.args):
                    v = self.visit(arg)
                    # Determine expected type for this arg position (skip #0 = self)
                    arg_idx = i + 1  # position in call_func.args
                    if arg_idx < len(call_func.args):
                        expected_arg_type = call_func.args[arg_idx].type
                    else:
                        expected_arg_type = BOXED_PTR
                    call_args.append(self._coerce_call_arg(v, expected_arg_type))
                if len(call_args) == len(call_func.args):
                    # Safety: verify all arg types match before calling
                    call_args = self._verify_call_args(call_func, call_args)
                    ret = self.builder.call(call_func, call_args)
                    self._check_exc_after_call()
                    ret_type = call_func.function_type.return_type
                    if ret_type == VOID:
                        return Value(ir.Constant(I64, 0), PyType.NONE)
                    return Value(ret, self._llvm_type_to_pytype(ret_type))
                # Arg count mismatch — fall through to classref check
        except CompileError:
            pass  # Variable not found — not a local variable, fall through

        # NAPRAWA: Resolve alias to actual class name
        # 1) @classmethod parameter (cls, device, etc.)
        # 2) Variable holding a class reference (device_cls = SomeClass) — NOT an instance
        actual_class_name = fname
        cls_map = getattr(self, '_classmethod_cls_map', {})
        if fname in cls_map:
            actual_class_name = cls_map[fname]
        else:
            # Check if fname is a variable holding a class reference (NOT an instance)
            try:
                var_info = self.sym.lookup(fname)
                is_cls_ref = getattr(var_info, 'is_class_ref', False)
                if (var_info is not None
                    and is_cls_ref  # Must be a class reference, not an instance
                    and hasattr(var_info, 'class_name') and var_info.class_name
                    and f"__is_class_{var_info.class_name}" in self.functions):
                    actual_class_name = var_info.class_name
            except CompileError:
                pass

        # Check if this is a class call (found in __classref_ globals)
        # NOTE: When inside a classmethod body, __classref_{class_name} may
        # not be registered yet (it's set at the end of visit_ClassDef).
        # But __classref_{param_name} (e.g., __classref_cls) IS registered
        # by the classmethod handler. So check both keys.
        classref_key = f"__classref_{actual_class_name}"
        if classref_key not in self.functions and actual_class_name != fname:
            classref_key = f"__classref_{fname}"
        if classref_key in self.functions:
            # Create instance
            inst_dict = self.create_dict([])
            class_ptr = self.functions.get(classref_key) or ir.Constant(CLASS_PTR, None)
            inst_obj = self.create_instance(class_ptr, inst_dict)
            inst_boxed = self._box(inst_obj)


            # Store class name in instance for proper method lookup
            # Use actual class name (not 'cls' alias)
            class_name_key = self.create_string("__class__")
            class_name_val = self.create_string(actual_class_name)
            self.dict_setitem(inst_dict, class_name_key, class_name_val)

            # If this is a frozen dataclass, store the __frozen__ flag
            dc_fields = getattr(self, '_dataclass_fields', {}).get(actual_class_name)
            if dc_fields is not None:
                frozen_info = getattr(self, '_dataclass_frozen', {})
                if frozen_info.get(actual_class_name, False):
                    frozen_key = self.create_string("__frozen__")
                    frozen_val = Value(ir.Constant(I64, 1), PyType.BOOL)
                    self.dict_setitem(inst_dict, frozen_key, frozen_val)

            # Call __init__ if it exists (use actual class name)
            init_func_name = f"py_{actual_class_name}___init__"
            dc_fields = getattr(self, '_dataclass_fields', {}).get(actual_class_name)
            if init_func_name in self.functions:
                init_func = self.functions[init_func_name]
                # self is an instance pointer (INSTANCE_PTR)
                call_args = [inst_obj.llvm]  # self as INSTANCE_PTR
                for arg in node.args:
                    v = self.visit(arg)
                    if v.is_dict or v.is_instance or v.is_object:
                        call_args.append(v.llvm)
                    else:
                        call_args.append(self._box(v))
                # NAPRAWA: Only pass as many args as __init__ accepts
                # (extra args from cls() call are silently dropped)
                if len(call_args) >= len(init_func.args):
                    # Truncate to match __init__ signature
                    call_args = call_args[:len(init_func.args)]
                if len(call_args) == len(init_func.args):
                    call_args = self._verify_call_args(init_func, call_args)
                    self.builder.call(init_func, call_args)
                    self._check_exc_after_call()
                else:
                    print(
                        f"WARNING: Argument count mismatch for {init_func_name}: {len(call_args)} vs {len(init_func.args)}"
                    )
            elif dc_fields is not None:
                # Dataclass without explicit __init__: set fields directly on instance dict
                # Build a map of keyword arguments
                kw_args = {}
                for kw in node.keywords:
                    kw_args[kw.arg] = self.visit(kw.value)
                # Positional args map to fields in order
                for i, arg in enumerate(node.args):
                    if i < len(dc_fields):
                        kw_args[dc_fields[i][0]] = self.visit(arg)
                # Set each field on the instance dict
                for field_name, default_val, _ann in dc_fields:
                    key = self.create_string(field_name)
                    if field_name in kw_args:
                        self.dict_setitem(inst_dict, key, kw_args[field_name])
                    elif default_val is not None:
                        # Use default value
                        if isinstance(default_val, ast.AST):
                            # Check for field(default_factory=...) pattern
                            if (isinstance(default_val, ast.Call) and
                                isinstance(default_val.func, ast.Name) and
                                default_val.func.id == 'field'):
                                # Look for default_factory keyword arg
                                factory_arg = None
                                for kw in default_val.keywords:
                                    if kw.arg == 'default_factory':
                                        factory_arg = kw.value
                                        break
                                if factory_arg is not None:
                                    # field(default_factory=list) → create empty list
                                    if (isinstance(factory_arg, ast.Name) and
                                        factory_arg.id == 'list'):
                                        dv = self.create_list([])
                                    elif (isinstance(factory_arg, ast.Name) and
                                        factory_arg.id == 'dict'):
                                        dv = self.create_dict([])
                                    else:
                                        # Try to call the factory
                                        dv = self.create_list([])
                                else:
                                    # field() with no default_factory — look for default=
                                    for kw in default_val.keywords:
                                        if kw.arg == 'default':
                                            dv = self.visit(kw.value)
                                            break
                                    else:
                                        dv = Value(ir.Constant(I64, 0), PyType.NONE)
                            else:
                                # Complex default — try to compile it
                                dv = self.visit(default_val)
                        elif isinstance(default_val, list):
                            dv = self.create_list([self.create_string(str(e)) for e in default_val] if default_val else [])
                        elif isinstance(default_val, str):
                            dv = self.create_string(default_val)
                        elif isinstance(default_val, (int, float)):
                            dv = Value(ir.Constant(I64, int(default_val)), PyType.INT) if isinstance(default_val, int) else Value(ir.Constant(F64, float(default_val)), PyType.FLOAT)
                        else:
                            dv = Value(ir.Constant(I64, 0), PyType.NONE)
                        self.dict_setitem(inst_dict, key, dv)
                    # else: field has no default and wasn't provided — skip (would be runtime error in CPython)

                # Call __post_init__ if it exists (dataclass protocol)
                post_init_func_name = f"py_{actual_class_name}___post_init__"
                if post_init_func_name in self.functions:
                    post_init_func = self.functions[post_init_func_name]
                    pi_call_args = [inst_obj.llvm]
                    if len(pi_call_args) == len(post_init_func.args):
                        pi_call_args = self._verify_call_args(post_init_func, pi_call_args)
                        self.builder.call(post_init_func, pi_call_args)
                        self._check_exc_after_call()

            return Value(inst_boxed, PyType.OBJECT, class_name=actual_class_name)

        if fname == "print":
            args = [self.visit(a) for a in node.args]
            # Obsługa keyword argumentów: end=, sep=
            end_val = None
            for kw in node.keywords:
                if kw.arg == "end":
                    end_val = self.visit(kw.value)
            self._emit_print(args, end_val=end_val)
            return Value(ir.Constant(I64, 0), PyType.NONE)

        if fname == "len":
            if len(node.args) != 1:
                raise CompileError("len() wymaga 1 argumentu.", node)
            return self.builtin_len(self.visit(node.args[0]))

        if fname == "int":
            return self._to_int(self.visit(node.args[0]))
        if fname == "float":
            return self._to_float(self.visit(node.args[0]))
        if fname == "bool":
            return self._to_bool_val(self.visit(node.args[0]))
        if fname == "list":
            args = [self.visit(a) for a in node.args] if node.args else []
            if args:
                iterable = args[0]
                if iterable.is_list or iterable.is_tuple:
                    return iterable
                # For other iterables, convert to list
                res = self.create_list([])
                # Try to iterate and append elements
                z = ir.Constant(I32, 0)
                return res
            return self.create_list([])
        if fname == "dict":
            return self.create_dict([])
        if fname == "tuple":
            if node.args:
                arg = self.visit(node.args[0])
                if arg.is_list or arg.is_tuple:
                    return Value(arg.llvm, PyType.TUPLE)
                return self.create_tuple([arg])
            return self.create_tuple([])
        if fname == "set":
            if node.args:
                arg = self.visit(node.args[0])
                if arg.is_list or arg.is_tuple or arg.is_set:
                    return Value(arg.llvm, PyType.SET)  # Zbiory traktujemy jako listy z Tag.SET
                lst = self.create_list([arg])
                return Value(lst.llvm, PyType.SET)
            lst = self.create_list([])
            return Value(lst.llvm, PyType.SET)
        if fname == "range":
            # range() returns an OBJECT (boxed list) in pyco semantics
            # because it's used in comprehensions and For loops which need iteration
            start = 0
            stop = self._get_const_int(self.visit(node.args[0]))
            step = 1
            if len(node.args) > 1:
                stop = self._get_const_int(self.visit(node.args[1]))
            if len(node.args) > 2:
                step = self._get_const_int(self.visit(node.args[2]))
            elems = []
            current = start
            while current < stop:
                elems.append(Value(ir.Constant(I64, current), PyType.INT, class_name=None))
                current += step
            lst = self.create_list(elems)
            return lst

        if fname == "repr":
            arg = self.visit(node.args[0])
            s = self.val_to_str(arg)
            q = self.create_string("'")
            return self.concat_strings(q, self.concat_strings(s, q))
        if fname == "str":
            arg = self.visit(node.args[0])
            return self.val_to_str(arg)

        if fname == "abs":
            arg = self.visit(node.args[0])
            if arg.is_int:
                return Value(
                    self.builder.select(
                        self.builder.icmp_signed(">", arg.llvm, ir.Constant(I64, 0)),
                        arg.llvm,
                        self.builder.sub(ir.Constant(I64, 0), arg.llvm),
                    ),
                    PyType.INT,
                )
            if arg.is_float:
                return Value(
                    self.builder.select(
                        self.builder.fcmp_ordered(">", arg.llvm, ir.Constant(F64, 0.0)),
                        arg.llvm,
                        self.builder.fsub(ir.Constant(F64, 0.0), arg.llvm),
                    ),
                    PyType.FLOAT,
                )
            raise CompileError("abs() nie obsługuje tego typu.", node)

        if fname == "max" or fname == "min":
            # max(a, b, ...) lub max(list)
            args = [self.visit(a) for a in node.args]
            if len(args) == 1 and args[0].is_list:
                # max(list) - iteruj po liście
                lst = args[0]
                # Zwraca pierwszy element dla uproszczenia
                return self.list_getitem(lst, Value(ir.Constant(I64, 0), PyType.INT))
            if len(args) >= 2:
                # max(a, b) - porównaj dwa
                a, b = args[0], args[1]
                if a.is_int and b.is_int:
                    cmp = (
                        self.builder.icmp_signed(">=", a.llvm, b.llvm)
                        if fname == "max"
                        else self.builder.icmp_signed("<=", a.llvm, b.llvm)
                    )
                    return Value(self.builder.select(cmp, a.llvm, b.llvm), PyType.INT)
                if a.is_float and b.is_float:
                    cmp = self.builder.fcmp_ordered(
                        ">=" if fname == "max" else "<=", a.llvm, b.llvm
                    )
                    return Value(self.builder.select(cmp, a.llvm, b.llvm), PyType.FLOAT)
            raise CompileError(f"{fname}() wymaga 2 argumentów lub listy.", node)

        if fname == "type":
            # type(x) - zwraca string z nazwą typu
            arg = self.visit(node.args[0])
            if arg.is_int:
                return self.create_string("int")
            if arg.is_float:
                return self.create_string("float")
            if arg.is_bool:
                return self.create_string("bool")
            if arg.is_str:
                return self.create_string("str")
            if arg.is_list:
                return self.create_string("list")
            if arg.is_dict:
                return self.create_string("dict")
            if arg.is_object:
                return self.create_string("object")
            return self.create_string("unknown")

        if fname == "round":
            arg = self.visit(node.args[0])
            if arg.is_float:
                # floor(x + 0.5) dla dodatnich
                rounded = self._to_int(arg)
                return rounded
            if arg.is_int:
                return arg
            raise CompileError("round() wymaga liczby.", node)

        # ══════════════════════════════════════════════════════════════════
        #  FIX: Wbudowane funkcje next() i iter() (Test 12 - generatory)
        # ══════════════════════════════════════════════════════════════════

        if fname == "next":
            """next(iterator) -> nastepna wartosc z iteratora."""
            if len(node.args) < 1:
                raise CompileError("next() wymaga co najmniej 1 argumentu.", node)
            return self._builtin_next(self.visit(node.args[0]), node)

        if fname == "iter":
            """iter(iterable) -> iterator."""
            if len(node.args) != 1:
                raise CompileError("iter() wymaga 1 argumentu.", node)
            return self._builtin_iter(self.visit(node.args[0]), node)

        if fname == "all":
            """
            all(iterable) -> bool
            Zwraca True jeśli wszystkie elementy są truthy.
            """
            if len(node.args) != 1:
                raise CompileError("all() wymaga dokładnie jednego argumentu", node)

            iterable = self.visit(node.args[0])

            if not (iterable.is_list or iterable.is_object):
                raise CompileError("all() wymaga argumentu typu list lub object", node)

            return self._handle_builtin_all([iterable], node)

        # ══════════════════════════════════════════════════════════════════
        #  DODATKOWE WBUDOWANE FUNKCJE
        # ══════════════════════════════════════════════════════════════════

        if fname == "any":
            """any(iterable) -> bool"""
            if len(node.args) != 1:
                raise CompileError("any() wymaga dokładnie jednego argumentu", node)

            iterable = self.visit(node.args[0])
            if not (iterable.is_list or iterable.is_object):
                raise CompileError("any() wymaga argumentu typu list lub object", node)

            return self._handle_builtin_any([iterable], node)

        if fname == "sum":
            """sum(iterable[, start]) -> number"""
            if not node.args:
                raise CompileError("sum() wymaga argumentu", node)

            args = [self.visit(a) for a in node.args]
            return self._handle_builtin_sum(args, node)

        if fname == "sorted":
            """sorted(iterable) -> list - uproszczenie: zwróć kopię"""
            if not node.args:
                raise CompileError("sorted() wymaga argumentu", node)

            args = [self.visit(a) for a in node.args]
            return self._handle_builtin_sorted(args, node)

        if fname == "isinstance":
            """isinstance(obj, class_or_tuple) -> bool"""
            args = [self.visit(a) for a in node.args]
            return self._handle_builtin_isinstance(args, node)

        if fname == "chr":
            """chr(i) -> str"""
            if len(node.args) != 1:
                raise CompileError("chr() wymaga jednego argumentu", node)
            args = [self.visit(a) for a in node.args]
            return self._handle_builtin_chr(args, node)

        if fname == "ord":
            """ord(c) -> int"""
            if len(node.args) != 1:
                raise CompileError("ord() wymaga jednego argumentu", node)
            args = [self.visit(a) for a in node.args]
            return self._handle_builtin_ord(args, node)

        if fname == "id":
            """id(obj) -> int (adres pamięci)"""
            if not node.args:
                raise CompileError("id() wymaga argumentu", node)
            args = [self.visit(a) for a in node.args]
            return self._handle_builtin_id(args, node)

        if fname == "input":
            """input([prompt]) -> str"""
            args = [self.visit(a) for a in node.args] if node.args else []
            return self._handle_builtin_input(args, node)

        if fname == "zip":
            return self._handle_builtin_zip(node)

        if fname == "enumerate":
            return self._handle_builtin_enumerate(node)

        if fname == "map":
            return self._handle_builtin_map(node)

        if fname == "filter":
            return self._handle_builtin_filter(node)

        # ══════════════════════════════════════════════════════════════════
        #  POPRAWKA 5: Obsługa wyjątków jako klas (Test 07)
        # ══════════════════════════════════════════════════════════════════

        if fname in EXCEPTION_TYPES:
            args = [self.visit(a) for a in node.args]
            return self._create_exception_object(fname, args, node)

        # ══════════════════════════════════════════════════════════════════
        #  POPRAWKA 6: Funkcja super() (Test 06)
        # ══════════════════════════════════════════════════════════════════

        if fname == "super":
            return self._create_super_object(node)

        # ══════════════════════════════════════════════════════════════════
        #  FFI: Direct call to native .so symbol
        # ══════════════════════════════════════════════════════════════════
        if hasattr(self, '_ffi_symbols') and fname in self._ffi_symbols:
            return self._ffi_call(fname, node)

        # ══════════════════════════════════════════════════════════════════
        #  Built-in module function imported directly (from time import time)
        #  Sprawdź czy fname jest zmapowane do built-in handlera
        # ══════════════════════════════════════════════════════════════════
        builtin_direct = self._try_builtin_direct_call(fname, node)
        if builtin_direct is not None:
            return builtin_direct

        # ── Async: if we're inside an async function and the callee is
        # also async, spawn it as a coroutine task instead of calling directly.
        # This is the core of true async: calling fetch_data(...) inside
        # main() creates a Task, not a blocking function call.
        # NOTE: This MUST come BEFORE the generic "fname in self.functions"
        # check, because async functions ARE in self.functions but we want
        # to spawn them instead of calling directly.
        # ──────────────────────────────────────────────────────────────
        is_in_async = getattr(self, '_is_async_function', False)
        async_fns = getattr(self, '_async_functions', set())
        if is_in_async and fname in async_fns:
            # Spawn the async function as a coroutine task
            call_args_llvm = []
            for an in node.args:
                v = self.visit(an)
                call_args_llvm.append(v)
            return self._spawn_async_call(fname, call_args_llvm)

        # Check built-in modules (math, os, sys, time)
        if fname in self.functions:
            # Direct function from built-in module
            func = self.functions[fname]
            return self._call_llvm_function(func, node)

        # User-defined functions have py_ prefix
        llvm_fname = f"py_{fname}"
        if llvm_fname not in self.functions:
            raise CompileError(f"Nieznana funkcja: '{fname}'.", node)

        # Check if we can inline this function
        if fname in self._function_ast and self._can_inline(node):
            return self._inline_function(fname, node)

        func = self.functions[llvm_fname]
        exp = list(func.function_type.args)
        if len(node.args) != len(exp):
            raise CompileError(
                f"'{fname}' wymaga {len(exp)} arg, podano {len(node.args)}.", node
            )

        call_args = []
        for an, et in zip(node.args, exp):
            v = self.visit(an)
            if et == BOXED_PTR:
                if v.is_object:
                    call_args.append(v.llvm)
                else:
                    call_args.append(self._box(v))
            else:
                v = self._cast_to_llvm(v, et, an)
                call_args.append(v.llvm)

        call_args = self._verify_call_args(func, call_args)
        ret = self.builder.call(func, call_args, name=f"{fname}_ret")
        self._check_exc_after_call()
        ret_type = func.function_type.return_type
        if ret_type == BOXED_PTR:
            return Value(ret, PyType.OBJECT)
        return Value(ret, self._llvm_type_to_pytype(ret_type))

    # ══════════════════════════════════════════════════════════════════
    #  Built-in module dispatch — hardcodowane moduły standardowe
    #  (time, os, sys, math, random, json, re, ...)
    #  Każdy moduł mapuje swoje funkcje na wywołania libc / LLVM IR.
    # ══════════════════════════════════════════════════════════════════

    # Mapa: (moduł, metoda) → handler
    _BUILTIN_MODULE_DISPATCH = None  # lazy init

    def _get_builtin_dispatch(self):
        """Zwraca mapę dispatchu modułów built-in (lazy singleton)."""
        if self._BUILTIN_MODULE_DISPATCH is not None:
            return self._BUILTIN_MODULE_DISPATCH
        d = {
            ("time", "time"):       "_builtin_time_time",
            ("time", "sleep"):      "_builtin_time_sleep",
            ("time", "time_ns"):    "_builtin_time_time_ns",
            ("os", "getcwd"):       "_builtin_os_getcwd",
            ("os", "exit"):         "_builtin_os_exit",
            ("os", "getenv"):       "_builtin_os_getenv",
            ("os", "system"):       "_builtin_os_system",
            ("sys", "exit"):        "_builtin_sys_exit",
            ("math", "sqrt"):       "_builtin_math_func",
            ("math", "sin"):        "_builtin_math_func",
            ("math", "cos"):        "_builtin_math_func",
            ("math", "exp"):        "_builtin_math_func",
            ("math", "log"):        "_builtin_math_func",
            ("math", "pow"):        "_builtin_math_func",
            ("math", "floor"):      "_builtin_math_func",
            ("math", "ceil"):       "_builtin_math_func",
            ("math", "fabs"):       "_builtin_math_func",
            ("random", "random"):   "_builtin_random_random",
            ("random", "randint"):  "_builtin_random_randint",
            ("random", "choice"):   "_builtin_random_choice",
            # asyncio module
            ("asyncio", "sleep"):       "_builtin_asyncio_sleep",
            ("asyncio", "run"):         "_builtin_asyncio_run",
            ("asyncio", "gather"):      "_builtin_asyncio_gather",
            ("asyncio", "create_task"): "_builtin_asyncio_create_task",
            ("asyncio", "wait"):        "_builtin_asyncio_wait",
        }
        self.__class__._BUILTIN_MODULE_DISPATCH = d
        return d

    def _builtin_module_call(self, mod_name: str, mname: str, node) -> Optional[Value]:
        """Dispatch built-in module method call. Returns None if not a built-in."""
        dispatch = self._get_builtin_dispatch()
        key = (mod_name, mname)
        if key not in dispatch:
            return None
        handler_name = dispatch[key]
        handler = getattr(self, handler_name, None)
        if handler is None:
            return None
        return handler(mod_name, mname, node)

    # ── Helper: bezpieczna deklaracja libc function ─────────────────

    def _declare_libc(self, name: str, fty: ir.FunctionType) -> ir.Function:
        """Zadeklaruj funkcję libc, sprawdzając self.functions ORAZ
        self.module.globals żeby uniknąć DuplicatedNameError."""
        fn = self.functions.get(name) or self.module.globals.get(name)
        if fn is not None:
            return fn
        fn = ir.Function(self.module, fty, name=name)
        self.functions[name] = fn
        return fn

    # ── time module ──────────────────────────────────────────────────

    def _builtin_time_time(self, mod, mname, node):
        """time.time() → double (sekundy od epoch). Używa clock_gettime."""
        if len(node.args) != 0:
            raise CompileError("time.time() nie przyjmuje argumentów.", node)
        # Zadeklaruj clock_gettime jeśli potrzeba: int clock_gettime(clockid_t, struct timespec*)
        clock_gettime = self._declare_libc("clock_gettime", ir.FunctionType(I32, [I32, I8P]))
        # timespec na stack: { i64 tv_sec, i64 tv_nsec } = 16 bajtów
        ts = self.builder.alloca(ir.ArrayType(I8, 16), name="timespec")
        ts_ptr = self.builder.bitcast(ts, I8P)
        # CLOCK_REALTIME = 0
        ret = self.builder.call(clock_gettime, [ir.Constant(I32, 0), ts_ptr], name="cg_ret")
        # Wyciągnij tv_sec (offset 0, i64) i tv_nsec (offset 8, i64)
        ts_i64 = self.builder.bitcast(ts, ir.PointerType(I64, 0))
        tv_sec = self.builder.load(self.builder.gep(ts_i64, [ir.Constant(I32, 0)], inbounds=True), name="tv_sec")
        tv_nsec_ptr = self.builder.gep(ts_i64, [ir.Constant(I32, 1)], inbounds=True)
        tv_nsec = self.builder.load(tv_nsec_ptr, name="tv_nsec")
        # Wynik: tv_sec + tv_nsec / 1e9
        sec_f = self.builder.sitofp(tv_sec, F64)
        nsec_f = self.builder.sitofp(tv_nsec, F64)
        nsec_frac = self.builder.fmul(nsec_f, ir.Constant(F64, 1e-9), name="nsec_frac")
        result = self.builder.fadd(sec_f, nsec_frac, name="time_time")
        return Value(result, PyType.FLOAT)

    def _builtin_time_sleep(self, mod, mname, node):
        """time.sleep(secs) → nanosleep. Zwraca None."""
        if len(node.args) != 1:
            raise CompileError("time.sleep() wymaga 1 argumentu.", node)
        secs_val = self.visit(node.args[0])
        secs_f = self._to_float(secs_val)
        # Zadeklaruj nanosleep: int nanosleep(const struct timespec*, struct timespec*)
        nanosleep = self._declare_libc("nanosleep", ir.FunctionType(I32, [I8P, I8P]))
        # struct timespec na stack
        req_ts = self.builder.alloca(ir.ArrayType(I8, 16), name="req_ts")
        req_i64 = self.builder.bitcast(req_ts, ir.PointerType(I64, 0))
        # tv_sec = floor(secs)
        sec_int = self.builder.fptosi(secs_f.llvm, I64, name="sleep_sec")
        self.builder.store(sec_int, self.builder.gep(req_i64, [ir.Constant(I32, 0)], inbounds=True))
        # tv_nsec = (secs - floor(secs)) * 1e9
        sec_back = self.builder.sitofp(sec_int, F64)
        frac = self.builder.fsub(secs_f.llvm, sec_back, name="sleep_frac")
        nsec = self.builder.fmul(frac, ir.Constant(F64, 1e9), name="sleep_nsec")
        nsec_int = self.builder.fptosi(nsec, I64)
        self.builder.store(nsec_int, self.builder.gep(req_i64, [ir.Constant(I32, 1)], inbounds=True))
        # Wywołaj nanosleep(req, NULL)
        req_ptr = self.builder.bitcast(req_ts, I8P)
        null_ptr = ir.Constant(I8P, None)
        self.builder.call(nanosleep, [req_ptr, null_ptr])
        return Value(ir.Constant(I64, 0), PyType.NONE)

    def _builtin_time_time_ns(self, mod, mname, node):
        """time.time_ns() → int (nanosekundy od epoch)."""
        if len(node.args) != 0:
            raise CompileError("time.time_ns() nie przyjmuje argumentów.", node)
        clock_gettime = self._declare_libc("clock_gettime", ir.FunctionType(I32, [I32, I8P]))
        ts = self.builder.alloca(ir.ArrayType(I8, 16), name="timespec_ns")
        ts_ptr = self.builder.bitcast(ts, I8P)
        self.builder.call(clock_gettime, [ir.Constant(I32, 0), ts_ptr], name="cg_ns_ret")
        ts_i64 = self.builder.bitcast(ts, ir.PointerType(I64, 0))
        tv_sec = self.builder.load(self.builder.gep(ts_i64, [ir.Constant(I32, 0)], inbounds=True), name="tv_sec_ns")
        tv_nsec = self.builder.load(self.builder.gep(ts_i64, [ir.Constant(I32, 1)], inbounds=True), name="tv_nsec_ns")
        # Wynik: tv_sec * 1_000_000_000 + tv_nsec
        ns = self.builder.add(
            self.builder.mul(tv_sec, ir.Constant(I64, 1_000_000_000)),
            tv_nsec,
            name="time_ns"
        )
        return Value(ns, PyType.INT)

    # ── os module ────────────────────────────────────────────────────

    def _builtin_os_getcwd(self, mod, mname, node):
        """os.getcwd() → string. Używa libc getcwd(buf, size)."""
        if len(node.args) != 0:
            raise CompileError("os.getcwd() nie przyjmuje argumentów.", node)
        getcwd = self._declare_libc("getcwd", ir.FunctionType(I8P, [I8P, I64]))
        # Bufor na stack (PATH_MAX = 4096)
        buf = self.builder.alloca(ir.ArrayType(I8, 4096), name="cwd_buf")
        buf_ptr = self.builder.bitcast(buf, I8P)
        result_ptr = self.builder.call(getcwd, [buf_ptr, ir.Constant(I64, 4096)], name="cwd")
        return self._cstr_to_pylow_str(buf_ptr)

    def _builtin_os_exit(self, mod, mname, node):
        """os.exit(code) → exit()."""
        if len(node.args) != 1:
            raise CompileError("os.exit() wymaga 1 argumentu.", node)
        code_val = self.visit(node.args[0])
        code_int = self._to_int(code_val)
        exit_fn = self._declare_libc("exit", ir.FunctionType(VOID, [I32]))
        self.builder.call(exit_fn, [self.builder.trunc(code_int.llvm, I32)])
        return Value(ir.Constant(I64, 0), PyType.NONE)

    def _builtin_os_getenv(self, mod, mname, node):
        """os.getenv(name) → string lub None. Używa libc getenv()."""
        if len(node.args) != 1:
            raise CompileError("os.getenv() wymaga 1 argumentu.", node)
        name_val = self.visit(node.args[0])
        cstr = self._pyval_to_cstr(name_val)
        getenv = self._declare_libc("getenv", ir.FunctionType(I8P, [I8P]))
        result_ptr = self.builder.call(getenv, [cstr], name="env_val")
        # Sprawdź NULL → zwróć None
        is_null = self.builder.icmp_signed("==", result_ptr, ir.Constant(I8P, None))
        none_bb = self.current_func.append_basic_block("getenv.null")
        str_bb = self.current_func.append_basic_block("getenv.str")
        merge_bb = self.current_func.append_basic_block("getenv.merge")
        self.builder.cbranch(is_null, none_bb, str_bb)
        self.builder.position_at_end(none_bb)
        none_val = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
        self.builder.branch(merge_bb)
        self.builder.position_at_end(str_bb)
        str_val = self._box(self._cstr_to_pylow_str(result_ptr))
        self.builder.branch(merge_bb)
        self.builder.position_at_end(merge_bb)
        phi = self.builder.phi(BOXED_PTR, "getenv_result")
        phi.add_incoming(none_val, none_bb)
        phi.add_incoming(str_val, str_bb)
        return Value(phi, PyType.OBJECT)

    def _builtin_os_system(self, mod, mname, node):
        """os.system(cmd) → int. Używa libc system()."""
        if len(node.args) != 1:
            raise CompileError("os.system() wymaga 1 argumentu.", node)
        cmd_val = self.visit(node.args[0])
        cstr = self._pyval_to_cstr(cmd_val)
        system_fn = self._declare_libc("system", ir.FunctionType(I32, [I8P]))
        ret = self.builder.call(system_fn, [cstr], name="system_ret")
        return Value(self.builder.sext(ret, I64), PyType.INT)

    # ── sys module ───────────────────────────────────────────────────

    def _builtin_sys_exit(self, mod, mname, node):
        """sys.exit(code) → exit()."""
        if len(node.args) != 1:
            raise CompileError("sys.exit() wymaga 1 argumentu.", node)
        code_val = self.visit(node.args[0])
        code_int = self._to_int(code_val)
        exit_fn = self._declare_libc("exit", ir.FunctionType(VOID, [I32]))
        self.builder.call(exit_fn, [self.builder.trunc(code_int.llvm, I32)])
        return Value(ir.Constant(I64, 0), PyType.NONE)

    # ── math module ──────────────────────────────────────────────────

    def _builtin_math_func(self, mod, mname, node):
        """math.sqrt/sin/cos/... → LLVM intrinsic."""
        intrinsic_name = f"llvm.{mname}.f64"
        fn = self.functions.get(intrinsic_name)
        if fn is None:
            # Fallback: zadeklaruj jako extern C
            fn = self.functions.get(f"math.{mname}")
        if fn is None:
            fty = ir.FunctionType(F64, [F64] if mname != "pow" else [F64, F64])
            fn = ir.Function(self.module, fty, name=mname)
            self.functions[f"math.{mname}"] = fn
        args = [self._to_float(self.visit(a)) for a in node.args]
        if mname == "pow" and len(args) != 2:
            raise CompileError("math.pow() wymaga 2 argumentów.", node)
        elif mname != "pow" and len(args) != 1:
            raise CompileError(f"math.{mname}() wymaga 1 argumentu.", node)
        result = self.builder.call(fn, [a.llvm for a in args], name=f"math_{mname}")
        return Value(result, PyType.FLOAT)

    # ── random module ────────────────────────────────────────────────

    _random_seeded = False

    def _ensure_random_seed(self):
        """Wywołaj srand(clock_gettime) raz — inicjalizacja generatora RNG."""
        if self._random_seeded:
            return
        self._random_seeded = True
        # Użyj clock_gettime (nie time() — time może mieć złą sygnaturę w externals)
        clock_gettime = self._declare_libc("clock_gettime", ir.FunctionType(I32, [I32, I8P]))
        ts = self.builder.alloca(ir.ArrayType(I8, 16), name="seed_ts")
        ts_ptr = self.builder.bitcast(ts, I8P)
        self.builder.call(clock_gettime, [ir.Constant(I32, 0), ts_ptr], name="seed_cg")
        ts_i64 = self.builder.bitcast(ts, ir.PointerType(I64, 0))
        tv_sec = self.builder.load(self.builder.gep(ts_i64, [ir.Constant(I32, 0)], inbounds=True), name="seed_sec")
        tv_nsec = self.builder.load(self.builder.gep(ts_i64, [ir.Constant(I32, 1)], inbounds=True), name="seed_nsec")
        # seed = tv_sec ^ tv_nsec (xor daje lepsze rozproszenie)
        seed_val = self.builder.xor(tv_sec, tv_nsec, name="seed_val")
        srand_fn = self._declare_libc("srand", ir.FunctionType(ir.VoidType(), [I32]))
        self.builder.call(srand_fn, [self.builder.trunc(seed_val, I32)])

    def _builtin_random_random(self, mod, mname, node):
        """random.random() → float [0.0, 1.0). Używa drand48()."""
        if len(node.args) != 0:
            raise CompileError("random.random() nie przyjmuje argumentów.", node)
        self._ensure_random_seed()
        drand48 = self._declare_libc("drand48", ir.FunctionType(F64, []))
        result = self.builder.call(drand48, [], name="rand_val")
        return Value(result, PyType.FLOAT)

    def _builtin_random_randint(self, mod, mname, node):
        """random.randint(a, b) → int [a, b]. Używa rand()."""
        if len(node.args) != 2:
            raise CompileError("random.randint() wymaga 2 argumentów.", node)
        self._ensure_random_seed()
        a_val = self._to_int(self.visit(node.args[0]))
        b_val = self._to_int(self.visit(node.args[1]))
        rand_fn = self._declare_libc("rand", ir.FunctionType(I32, []))
        r = self.builder.call(rand_fn, [], name="rand_int")
        r_i64 = self.builder.sext(r, I64, name="rand_i64")
        # range = b - a + 1;  result = a + r % range
        one = ir.Constant(I64, 1)
        zero = ir.Constant(I64, 0)
        rng = self.builder.add(self.builder.sub(b_val.llvm, a_val.llvm), one, name="rand_range")
        # Zabezpieczenie: range > 0
        is_pos = self.builder.icmp_signed(">", rng, zero)
        safe_rng = self.builder.select(is_pos, rng, one, name="safe_range")
        mod_val = self.builder.srem(r_i64, safe_rng, name="rand_mod")
        result = self.builder.add(a_val.llvm, mod_val, name="randint")
        return Value(result, PyType.INT)

    def _builtin_random_choice(self, mod, mname, node):
        """random.choice(seq) → losowy element z sekwencji (lista)."""
        if len(node.args) != 1:
            raise CompileError("random.choice() wymaga 1 argumentu.", node)
        seq_val = self.visit(node.args[0])

        # Jeśli seq_val to OBJECT (boxed), unboxuj do LIST_PTR
        if seq_val.is_object and not seq_val.is_list:
            tag, pay = self._read_slot(seq_val.llvm)
            lptr = self.builder.inttoptr(pay, LIST_PTR)
            seq_val = Value(lptr, PyType.LIST)

        if not seq_val.is_list:
            raise CompileError("random.choice() wymaga listy jako argumentu.", node)
        self._ensure_random_seed()
        # Pobierz długość listy
        z = ir.Constant(I32, 0)
        list_len = self.builder.load(
            self.builder.gep(seq_val.llvm, [z, ir.Constant(I32, 1)], inbounds=True),
            name="choice_len"
        )
        # Zabezpieczenie: len > 0
        is_empty = self.builder.icmp_signed("==", list_len, ir.Constant(I64, 0))
        safe_len = self.builder.select(is_empty, ir.Constant(I64, 1), list_len, name="choice_safe_len")
        # Wygeneruj losowy indeks
        rand_fn = self._declare_libc("rand", ir.FunctionType(I32, []))
        r = self.builder.call(rand_fn, [], name="choice_rand")
        r_i64 = self.builder.sext(r, I64, name="choice_rand_i64")
        # index = abs(r) % len
        abs_r = self.builder.select(
            self.builder.icmp_signed("<", r_i64, ir.Constant(I64, 0)),
            self.builder.sub(ir.Constant(I64, 0), r_i64),
            r_i64,
            name="choice_abs_r"
        )
        idx = self.builder.srem(abs_r, safe_len, name="choice_idx")
        # Pobierz element pod indeksem
        return self.list_getitem(seq_val, Value(idx, PyType.INT))

    # ══════════════════════════════════════════════════════════════════
    #  asyncio module — cooperative event loop for async/await
    #
    #  Implementation strategy:
    #  Since pylow compiles to native code, we implement a simplified
    #  synchronous cooperative model:
    #    - async def functions compile as regular functions returning
    #      a boxed coroutine result (Tag.COROUTINE)
    #    - await just extracts the result (coroutine completes synchronously)
    #    - asyncio.sleep() uses nanosleep (like time.sleep) but returns
    #      a coroutine-tagged None
    #    - asyncio.run() calls the coroutine function directly
    #    - asyncio.gather() runs coroutines sequentially (cooperative)
    #    - asyncio.create_task() wraps a coroutine call in a task object
    # ══════════════════════════════════════════════════════════════════

    def _builtin_asyncio_sleep(self, mod, mname, node):
        """asyncio.sleep(secs) → None.

        Wywołuje __async_sleep(secs) z C runtime, która oddaje
        sterowanie do zarządcy zadań (scheduler) za pomocą ucontext.
        Jeśli nie jesteśmy wewnątrz schedulera, fallback do blokującego
        nanosleep.
        """
        if len(node.args) > 1:
            raise CompileError("asyncio.sleep() wymaga 0 lub 1 argumentu.", node)

        if node.args:
            secs_val = self.visit(node.args[0])
            secs_f = self._to_float(secs_val)
            # Call __async_sleep(secs) — non-blocking if inside scheduler
            self.builder.call(
                self.functions["__async_sleep"],
                [secs_f.llvm]
            )
        else:
            # asyncio.sleep() = asyncio.sleep(0) = yield
            self.builder.call(
                self.functions["__async_sleep"],
                [ir.Constant(F64, 0.0)]
            )

        return Value(ir.Constant(I64, 0), PyType.NONE)

    def _builtin_asyncio_run(self, mod, mname, node):
        """asyncio.run(coro) → result.

        Spawnuje główną korutynę jako zadanie i uruchamia pętlę zdarzeń
        (scheduler) za pomocą __async_run().  Blokuje (z perspektywy kodu
        wywołującego) aż główna korutyna się zakończy.
        """
        if len(node.args) != 1:
            raise CompileError("asyncio.run() wymaga 1 argumentu (korutyny).", node)

        # The argument should be a call to an async function like main()
        # We need to spawn it as a coroutine and run the scheduler
        arg = node.args[0]

        # Check if the argument is a call to an async function
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
            fname = arg.func.id
            async_fns = getattr(self, '_async_functions', set())

            if fname in async_fns:
                # Spawn the async function as a coroutine and run the scheduler
                # Pack args into struct, call __async_run(py_{name}_coro_entry, args_ptr)
                n_args = len(arg.args)
                call_args_llvm = [self.visit(a) for a in arg.args]

                # Build args struct
                args_struct_fields = [BOXED_PTR] * n_args
                args_struct_ty = ir.LiteralStructType(args_struct_fields)
                args_struct_ptr_ty = ir.PointerType(args_struct_ty)
                sz = ir.Constant(I64, 8 * max(n_args, 1))
                raw = self.builder.call(self._malloc, [sz], "run_args_raw")
                args_ptr = self.builder.bitcast(raw, args_struct_ptr_ty, "run_args_ptr")

                # Store each argument
                z = ir.Constant(I32, 0)
                for i, arg_val in enumerate(call_args_llvm):
                    if isinstance(arg_val, Value):
                        if arg_val.is_object:
                            boxed_arg = arg_val.llvm
                        else:
                            boxed_arg = self._box(arg_val)
                    else:
                        boxed_arg = arg_val
                    field_ptr = self.builder.gep(
                        args_ptr, [z, ir.Constant(I32, i)], inbounds=True
                    )
                    self.builder.store(boxed_arg, field_ptr)

                # Get coro entry function
                entry_name = f"py_{fname}_coro_entry"
                coro_entry_fn = self.functions.get(entry_name)
                if coro_entry_fn is None:
                    # Generate it
                    self._generate_coro_entry(fname, n_args)
                    coro_entry_fn = self.functions.get(entry_name)

                if coro_entry_fn is None:
                    raise CompileError(f"Brak coroutine entry dla: {fname}", node)

                # Cast function pointer to void(*)(void*)
                fn_ptr_ty = ir.PointerType(ir.FunctionType(VOID, [I8P]))
                fn_ptr = self.builder.bitcast(coro_entry_fn, fn_ptr_ty, "run_fn_ptr")

                # Cast args_ptr to i8*
                args_i8p = self.builder.bitcast(args_ptr, I8P, "run_args_i8p")

                # Call __async_run(fn, arg) → BOXED_PTR
                result = self.builder.call(
                    self.functions["__async_run"],
                    [fn_ptr, args_i8p],
                    name="asyncio_run_result"
                )
                return Value(result, PyType.OBJECT)

        # Fallback: if arg is not a direct async call, just evaluate it
        coro_val = self.visit(arg)
        if coro_val.is_object:
            return coro_val
        else:
            return Value(self._box(coro_val), PyType.OBJECT)

    def _builtin_asyncio_gather(self, mod, mname, node):
        """asyncio.gather(*coros) → list of results.

        Każdy argument jest ewaluowany (co spawnuje zadanie), a następnie
        czekamy na zakończenie każdego zadania po kolei.  Ponieważ
        wszystkie zadania są w kolejce gotowych, scheduler uruchomi je
        współbieżnie podczas pierwszego await.
        """
        if not node.args:
            return self.create_list([])

        # Phase 1: Evaluate each argument → spawns tasks
        task_handles = []
        for arg in node.args:
            val = self.visit(arg)
            task_handles.append(val)

        # Phase 2: Await each task and collect results
        # Task handles have PyType.OBJECT with Tag.TASK at runtime.
        # We need runtime tag checks to determine how to handle each value.
        results = []
        for i, val in enumerate(task_handles):
            if val.is_object and val.llvm.type == BOXED_PTR:
                # Runtime check: is this a Task (Tag.TASK)?
                tag, pay = self._read_slot(val.llvm)
                is_task_tag = self.builder.icmp_signed(
                    "==", tag, ir.Constant(I64, Tag.TASK)
                )

                task_bb = self.current_func.append_basic_block(f"gather{i}.task")
                val_bb = self.current_func.append_basic_block(f"gather{i}.val")
                merge_bb = self.current_func.append_basic_block(f"gather{i}.merge")

                self.builder.cbranch(is_task_tag, task_bb, val_bb)

                # Task path: await and get result
                self.builder.position_at_end(task_bb)
                task_ptr = self.builder.inttoptr(pay, I8P, f"gather{i}_task_ptr")
                self.builder.call(
                    self.functions["__async_await_task"],
                    [task_ptr]
                )
                task_result = self.builder.call(
                    self.functions["__async_task_result"],
                    [task_ptr],
                    name=f"gather{i}_result"
                )
                self.builder.branch(merge_bb)

                # Value path: just pass through
                self.builder.position_at_end(val_bb)
                val_result = val.llvm
                self.builder.branch(merge_bb)

                # Merge
                self.builder.position_at_end(merge_bb)
                phi = self.builder.phi(BOXED_PTR, f"gather{i}_phi")
                phi.add_incoming(task_result, task_bb)
                phi.add_incoming(val_result, val_bb)
                results.append(Value(phi, PyType.OBJECT))
            else:
                # Not a boxed value — just box it
                if val.is_object:
                    results.append(val)
                else:
                    results.append(Value(self._box(val), PyType.OBJECT))

        return self.create_list(results)

    def _builtin_asyncio_create_task(self, mod, mname, node):
        """asyncio.create_task(coro) → task handle.

        Ewaluuje argument (który spawnuje zadanie jeśli to wywołanie
        async funkcji) i zwraca uchwyt zadania (Tag.TASK).
        """
        if len(node.args) != 1:
            raise CompileError("asyncio.create_task() wymaga 1 argumentu.", node)

        coro_val = self.visit(node.args[0])
        # The argument should already be a Task (from spawning an async function)
        return coro_val

    def _builtin_asyncio_wait(self, mod, mname, node):
        """asyncio.wait(coros) → (done, pending).
        
        In our synchronous model, all coroutines complete immediately,
        so pending is always empty.
        """
        if len(node.args) != 1:
            raise CompileError("asyncio.wait() wymaga 1 argumentu.", node)

        # In sync model, all tasks are done, none pending
        done_list = self.create_list([])
        pending_list = self.create_list([])

        # Return a tuple (done, pending) as a list
        return self.create_list([done_list, pending_list])

    # ── Helper: konwersja pylow str → C string ──────────────────────

    def _pyval_to_cstr(self, val: Value) -> ir.Instruction:
        """Konwertuj wartość pylow (str/int/float) do i8* (C string)."""
        if val.is_str:
            # STR_TY: pole 3 to i8* data pointer
            z = ir.Constant(I32, 0)
            data_ptr = self.builder.load(
                self.builder.gep(val.llvm, [z, ir.Constant(I32, 3)], inbounds=True),
                name="str_data"
            )
            return data_ptr
        # Dla innych typów — box i wywołaj __repr__ lub cast
        raise CompileError("Nie można przekonwertować do C string.")

    # ── Helper: direct call for `from module import func` ───────────

    # Mapa: nazwa zaimportowanej funkcji → (moduł, metoda)
    _BUILTIN_DIRECT_CALL_MAP = {
        # from time import ...
        "time":    ("time", "time"),
        "sleep":   ("time", "sleep"),
        "time_ns": ("time", "time_ns"),
        # from os import ...
        "getcwd":  ("os", "getcwd"),
        "getenv":  ("os", "getenv"),
        "system":  ("os", "system"),
        # from random import ...
        "random":  ("random", "random"),
        "randint": ("random", "randint"),
        "choice":  ("random", "choice"),
        # from asyncio import ...
        # Note: "sleep" also maps to time.sleep, so we check the module context
        # via the VarInfo's class_name to disambiguate
        "async_sleep":     ("asyncio", "sleep"),
        "async_run":       ("asyncio", "run"),
        "async_gather":    ("asyncio", "gather"),
        "create_task":     ("asyncio", "create_task"),
        "async_wait":      ("asyncio", "wait"),
    }

    def _try_builtin_direct_call(self, fname: str, node) -> Optional[Value]:
        """Obsługa `from time import time; time()` — bezpośrednie wywołanie.
        
        Also handles `from asyncio import sleep, run, gather` by checking
        the VarInfo's class_name to determine the source module.
        """
        # First check the standard direct call map
        mapping = self._BUILTIN_DIRECT_CALL_MAP.get(fname)
        # Sprawdź czy ta nazwa jest w symbol table jako built-in import
        try:
            info = self.sym.lookup(fname)
            if not getattr(info, 'is_ffi_module', False):
                return None
            cn = getattr(info, 'class_name', '')
            if not cn or not cn.startswith('__builtin_'):
                return None
            # NAPRAWA: If the standard map doesn't have this name,
            # try to resolve from the class_name (e.g., __builtin_asyncio.sleep → asyncio, sleep)
            if mapping is None:
                # Parse class_name: __builtin_{module}.{func}
                if cn.startswith('__builtin_'):
                    rest = cn[len('__builtin_'):]
                    if '.' in rest:
                        mod_name, func_name = rest.split('.', 1)
                        return self._builtin_module_call(mod_name, func_name, node)
                return None
        except CompileError:
            return None
        mod_name, mname = mapping
        return self._builtin_module_call(mod_name, mname, node)

    def _dict_method_call(self, obj: Value, mname: str, node) -> Value:
        """Obsługa metod słownika: clear, keys, values, items, get, pop, update."""
        z = ir.Constant(I32, 0)
        if mname == "clear":
            # dict.clear() — ustaw długość na 0 i wyczyść ordered_keys
            len_ptr = self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
            self.builder.store(ir.Constant(I64, 0), len_ptr)
            # Wyczyść ordered_keys listę
            ordered_list_ptr = self.builder.load(
                self.builder.gep(obj.llvm, [z, ir.Constant(I32, 4)], inbounds=True),
                name="dclear_keys"
            )
            keys_len_ptr = self.builder.gep(ordered_list_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
            self.builder.store(ir.Constant(I64, 0), keys_len_ptr)
            return Value(ir.Constant(I64, 0), PyType.NONE)

        if mname == "keys":
            # dict.keys() — zwraca listę kluczy (użyj ordered_keys)
            return self._dict_keys(obj)

        if mname == "values":
            # dict.values() — zwraca listę wartości
            return self._dict_values(obj)

        if mname == "items":
            # dict.items() — zwraca listę krotek (key, value)
            return self._dict_items(obj)

        if mname == "get":
            # dict.get(key[, default]) — zwraca wartość lub default
            if not node.args:
                raise CompileError("dict.get() wymaga co najmniej 1 argumentu.", node)
            key = self.visit(node.args[0])
            default = self.visit(node.args[1]) if len(node.args) > 1 else Value(ir.Constant(I64, 0), PyType.NONE)
            return self._dict_get(obj, key, default)

        raise CompileError(
            f"Metoda '{mname}' na słowniku nie jest obsługiwana.", node
        )

    def _dict_keys(self, dct: Value) -> Value:
        """Zwraca listę kluczy słownika (używa ordered_keys list)."""
        z = ir.Constant(I32, 0)
        ordered_list_ptr = self.builder.load(
            self.builder.gep(dct.llvm, [z, ir.Constant(I32, 4)], inbounds=True),
            name="dkeys_list"
        )
        # Zwróć kopię ordered_keys jako nową listę
        src_list = Value(ordered_list_ptr, PyType.LIST)
        return self._list_copy(src_list)

    def _dict_values(self, dct: Value) -> Value:
        """Zwraca listę wartości słownika."""
        z = ir.Constant(I32, 0)
        # Pobierz ordered_keys i dla każdego klucza zrób dict_getitem
        ordered_list_ptr = self.builder.load(
            self.builder.gep(dct.llvm, [z, ir.Constant(I32, 4)], inbounds=True),
            name="dvals_list"
        )
        src_list = Value(ordered_list_ptr, PyType.LIST)
        # Użyj _for_list do zebrania wartości
        result = self.create_list([])
        sp, _, dp = self._list_ptrs(ordered_list_ptr)
        size = self.builder.load(sp, "dvals_size")
        data = self.builder.load(dp, "dvals_data")

        val_alloca = self.builder.alloca(LIST_PTR, name="dvals_result")
        self.builder.store(result.llvm, val_alloca)

        i_a = self.builder.alloca(I64, name="_dvals_i")
        self.builder.store(ir.Constant(I64, 0), i_a)

        loop_bb = self.current_func.append_basic_block("dvals.loop")
        body_bb = self.current_func.append_basic_block("dvals.body")
        done_bb = self.current_func.append_basic_block("dvals.done")

        self.builder.branch(loop_bb)
        self.builder.position_at_end(loop_bb)
        i = self.builder.load(i_a, name="dvals_i")
        self.builder.cbranch(
            self.builder.icmp_signed("<", i, size), body_bb, done_bb
        )

        self.builder.position_at_end(body_bb)
        i2 = self.builder.load(i_a)
        slot = self.builder.gep(data, [i2], inbounds=True)
        tag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        pay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        # Klucz jako boxed Value
        key_val = Value(self._make_boxed(tag, pay), PyType.OBJECT)
        # Pobierz wartość z dict
        dict_val = self.dict_getitem(dct, key_val)
        result_current = self.builder.load(val_alloca, name="dvals_res_cur")
        self.list_append(Value(result_current, PyType.LIST), dict_val)
        # Inkrementuj
        i_next = self.builder.add(i2, ir.Constant(I64, 1))
        self.builder.store(i_next, i_a)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(done_bb)
        return Value(self.builder.load(val_alloca, name="dvals_final"), PyType.LIST)

    def _dict_items(self, dct: Value) -> Value:
        """Zwraca listę krotek (key, value) ze słownika."""
        z = ir.Constant(I32, 0)
        ordered_list_ptr = self.builder.load(
            self.builder.gep(dct.llvm, [z, ir.Constant(I32, 4)], inbounds=True),
            name="ditems_list"
        )
        sp, _, dp = self._list_ptrs(ordered_list_ptr)
        size = self.builder.load(sp, "ditems_size")
        data = self.builder.load(dp, "ditems_data")

        result = self.create_list([])
        val_alloca = self.builder.alloca(LIST_PTR, name="ditems_result")
        self.builder.store(result.llvm, val_alloca)

        i_a = self.builder.alloca(I64, name="_ditems_i")
        self.builder.store(ir.Constant(I64, 0), i_a)

        loop_bb = self.current_func.append_basic_block("ditems.loop")
        body_bb = self.current_func.append_basic_block("ditems.body")
        done_bb = self.current_func.append_basic_block("ditems.done")

        self.builder.branch(loop_bb)
        self.builder.position_at_end(loop_bb)
        i = self.builder.load(i_a, name="ditems_i")
        self.builder.cbranch(
            self.builder.icmp_signed("<", i, size), body_bb, done_bb
        )

        self.builder.position_at_end(body_bb)
        i2 = self.builder.load(i_a)
        slot = self.builder.gep(data, [i2], inbounds=True)
        tag = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        pay = self.builder.load(
            self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        key_val = Value(self._make_boxed(tag, pay), PyType.OBJECT)
        dict_val = self.dict_getitem(dct, key_val)
        pair = self.create_tuple([key_val, dict_val])
        result_current = self.builder.load(val_alloca, name="ditems_res_cur")
        self.list_append(Value(result_current, PyType.LIST), pair)
        i_next = self.builder.add(i2, ir.Constant(I64, 1))
        self.builder.store(i_next, i_a)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(done_bb)
        return Value(self.builder.load(val_alloca, name="ditems_final"), PyType.LIST)

    def _dict_get(self, dct: Value, key: Value, default: Value) -> Value:
        """dict.get(key, default) — wyszukaj klucz, zwróć wartość lub default."""
        # Sprawdź czy klucz istnieje używając __py2llvm_dict_contains
        self._ensure_dict_funcs()
        dict_cont_fn = self.functions["__py2llvm_dict_contains"]
        kt, kp = self._value_to_tag_payload(key)
        exists = self.builder.call(dict_cont_fn, [dct.llvm, kt, kp], name="dget_exists")

        found_bb = self.current_func.append_basic_block("dget.found")
        not_found_bb = self.current_func.append_basic_block("dget.not_found")
        merge_bb = self.current_func.append_basic_block("dget.merge")

        self.builder.cbranch(exists, found_bb, not_found_bb)

        self.builder.position_at_end(found_bb)
        found = self.dict_getitem(dct, key)
        found_boxed = found.llvm if found.is_object else self._box(found)
        self.builder.branch(merge_bb)

        self.builder.position_at_end(not_found_bb)
        default_boxed = default.llvm if default.is_object else self._box(default)
        self.builder.branch(merge_bb)

        self.builder.position_at_end(merge_bb)
        phi = self.builder.phi(BOXED_PTR, "dget_result")
        phi.add_incoming(found_boxed, found_bb)
        phi.add_incoming(default_boxed, not_found_bb)
        return Value(phi, PyType.OBJECT)

    def _make_boxed(self, tag_val: ir.Value, pay_val: ir.Value) -> ir.Value:
        """Utwórz boxed value z tag i payload."""
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
        bv = self.builder.bitcast(raw, BOXED_PTR)
        z = ir.Constant(I32, 0)
        null_i8p = ir.Constant(I8P, None)
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
            null_i8p,
            self.builder.gep(bv, [z, z, ir.Constant(I32, 3)], inbounds=True),
        )
        self.builder.store(tag_val, self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(pay_val, self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True))
        return bv

    def _list_copy(self, lst: Value) -> Value:
        """Zwraca kopię listy."""
        z = ir.Constant(I32, 0)
        sp, _, dp = self._list_ptrs(lst.llvm)
        size = self.builder.load(sp, "dcpy_size")
        data = self.builder.load(dp, "dcpy_data")
        result = self.create_list([])
        i_a = self.builder.alloca(I64, name="_dcpy_i")
        self.builder.store(ir.Constant(I64, 0), i_a)
        loop_bb = self.current_func.append_basic_block("dcpy.loop")
        body_bb = self.current_func.append_basic_block("dcpy.body")
        done_bb = self.current_func.append_basic_block("dcpy.done")
        self.builder.branch(loop_bb)
        self.builder.position_at_end(loop_bb)
        i = self.builder.load(i_a)
        self.builder.cbranch(self.builder.icmp_signed("<", i, size), body_bb, done_bb)
        self.builder.position_at_end(body_bb)
        i2 = self.builder.load(i_a)
        slot = self.builder.gep(data, [i2], inbounds=True)
        tag = self.builder.load(self.builder.gep(slot, [z, ir.Constant(I32, 1)], inbounds=True))
        pay = self.builder.load(self.builder.gep(slot, [z, ir.Constant(I32, 2)], inbounds=True))
        elem = Value(self._make_boxed(tag, pay), PyType.OBJECT)
        self.list_append(result, elem)
        self.builder.store(self.builder.add(i2, ir.Constant(I64, 1)), i_a)
        self.builder.branch(loop_bb)
        self.builder.position_at_end(done_bb)
        return result

    def _str_method_call(self, obj: Value, mname: str, node) -> Value:
        """Generuje kod dla metod stringa: strip, upper, lower, itp."""
        z = ir.Constant(I32, 0)

        if mname in ("strip", "lstrip", "rstrip"):
            # Zaimplementuj strip: stwórz nowy string z obciętymi spacjami
            # Prostsze podejście: wywołaj runtime helper lub generuj inline
            return self._str_strip(obj, mname)

        if mname == "upper":
            return self._str_upper(obj)

        if mname == "lower":
            return self._str_lower(obj)

        if mname == "split":
            delimiter = None
            if node.args:
                delimiter = self.visit(node.args[0])
            return self._str_split_impl(obj, delimiter, node)

        # ══════════════════════════════════════════════════════════════════
        #  POPRAWKA 3: str.join() (Test 02)
        # ══════════════════════════════════════════════════════════════════

        if mname == "join":
            """
            separator.join(iterable) -> str
            Łączy elementy iterowalne używając separatora.
            """
            if len(node.args) != 1:
                raise CompileError(
                    "str.join() wymaga dokładnie jednego argumentu", node
                )

            iterable = self.visit(node.args[0])

            if not iterable.is_list:
                raise CompileError("str.join() wymaga argumentu typu list", node)

            return self._str_join_impl(obj, iterable, node)

        # Implement replace(old, new[, count])
        if mname == "replace":
            if len(node.args) < 2:
                raise CompileError("str.replace() wymaga co najmniej 2 argumentów.", node)
            old_val = self.visit(node.args[0])
            new_val = self.visit(node.args[1])
            if not old_val.is_str:
                raise CompileError("str.replace() wymaga string argumentu", node)
            if not new_val.is_str:
                raise CompileError("str.replace() wymaga string argumentu", node)
            return self._str_replace_impl(obj, old_val, new_val, node)

        # NAPRAWA: Dodatkowe metody stringa (Test 02)
        if mname == "title":
            return self._str_title(obj)
        if mname == "capitalize":
            return self._str_capitalize(obj)
        if mname == "center":
            if len(node.args) < 1:
                raise CompileError("str.center() wymaga co najmniej 1 argumentu.", node)
            width = self.visit(node.args[0])
            fillchar = self.visit(node.args[1]) if len(node.args) > 1 else self.create_string(" ")
            return self._str_center(obj, width, fillchar, node)
        if mname == "startswith" or mname == "endswith":
            if len(node.args) < 1:
                raise CompileError(f"str.{mname}() wymaga co najmniej 1 argumentu.", node)
            prefix = self.visit(node.args[0])
            return self._str_startsend(obj, prefix, mname == "startswith", node)
        if mname == "find":
            if len(node.args) < 1:
                raise CompileError("str.find() wymaga co najmniej 1 argumentu.", node)
            substr = self.visit(node.args[0])
            return self._str_find(obj, substr, node)
        if mname == "count":
            if len(node.args) < 1:
                raise CompileError("str.count() wymaga co najmniej 1 argumentu.", node)
            substr = self.visit(node.args[0])
            return self._str_count(obj, substr, node)
        if mname == "zfill":
            if len(node.args) < 1:
                raise CompileError("str.zfill() wymaga 1 argumentu.", node)
            width = self.visit(node.args[0])
            return self._str_zfill(obj, width, node)
        if mname in ("isalpha", "isdigit", "isnumeric", "isalnum", "isspace", "isupper", "islower"):
            return self._str_ischeck(obj, mname, node)
        if mname in ("rjust", "ljust"):
            if len(node.args) < 1:
                raise CompileError(f"str.{mname}() wymaga co najmniej 1 argumentu.", node)
            width = self.visit(node.args[0])
            fillchar = self.visit(node.args[1]) if len(node.args) > 1 else self.create_string(" ")
            return self._str_justify(obj, width, fillchar, mname == "rjust", node)

        # FIX: Obsługa str.format() (Test 13)
        if mname == "format":
            return self._str_format_method(obj, node)

        raise CompileError(f"Metoda str.{mname}() nieobsługiwana.", node)

    # ──────────────────────────────────────────────────────────────
    #  NAPRAWA: Implementacje dodatkowych metod string (Test 02)
    # ──────────────────────────────────────────────────────────────

    def _str_title(self, obj: Value) -> Value:
        """str.title() – kapitalizuj pierwszą literę każdego słowa."""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        new_data = self.builder.call(self._malloc, [src_len])
        toupper_fn = self._get_or_declare("toupper", ir.FunctionType(I32, [I32]))
        tolower_fn = self._get_or_declare("tolower", ir.FunctionType(I32, [I32]))
        isspace_fn = self._get_or_declare("isspace", ir.FunctionType(I32, [I32]))

        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)
        prev_space_a = self.builder.alloca(I1)
        self.builder.store(ir.Constant(I1, 1), prev_space_a)  # Start as if after space

        loop_bb = self.current_func.append_basic_block("title.loop")
        body_bb = self.current_func.append_basic_block("title.body")
        done_bb = self.current_func.append_basic_block("title.done")

        self.builder.branch(loop_bb)
        self.builder.position_at_end(loop_bb)
        i = self.builder.load(i_a)
        self.builder.cbranch(self.builder.icmp_signed("<", i, src_len), body_bb, done_bb)

        self.builder.position_at_end(body_bb)
        ch = self.builder.load(self.builder.gep(src_data, [i], inbounds=True))
        ch_i32 = self.builder.zext(ch, I32)
        was_space = self.builder.load(prev_space_a)

        # If after space -> toupper, else -> tolower
        upper_ch = self.builder.call(toupper_fn, [ch_i32])
        lower_ch = self.builder.call(tolower_fn, [ch_i32])
        result_ch = self.builder.select(was_space, upper_ch, lower_ch)
        self.builder.store(self.builder.trunc(result_ch, I8), self.builder.gep(new_data, [i], inbounds=True))

        # Check if current char is space (for next iteration)
        sp = self.builder.call(isspace_fn, [ch_i32])
        is_sp = self.builder.icmp_signed("!=", sp, ir.Constant(I32, 0))
        self.builder.store(is_sp, prev_space_a)

        self.builder.store(self.builder.add(i, ir.Constant(I64, 1)), i_a)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(done_bb)
        null_ptr = self.builder.gep(new_data, [src_len], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_ptr)
        return self._make_str_object(src_len, src_len, new_data)

    def _str_capitalize(self, obj: Value) -> Value:
        """str.capitalize() – kapitalizuj pierwszą literę, reszta lowercase."""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        new_data = self.builder.call(self._malloc, [src_len])
        toupper_fn = self._get_or_declare("toupper", ir.FunctionType(I32, [I32]))
        tolower_fn = self._get_or_declare("tolower", ir.FunctionType(I32, [I32]))

        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)

        loop_bb = self.current_func.append_basic_block("cap.loop")
        body_bb = self.current_func.append_basic_block("cap.body")
        done_bb = self.current_func.append_basic_block("cap.done")

        self.builder.branch(loop_bb)
        self.builder.position_at_end(loop_bb)
        i = self.builder.load(i_a)
        self.builder.cbranch(self.builder.icmp_signed("<", i, src_len), body_bb, done_bb)

        self.builder.position_at_end(body_bb)
        ch = self.builder.load(self.builder.gep(src_data, [i], inbounds=True))
        ch_i32 = self.builder.zext(ch, I32)
        is_first = self.builder.icmp_signed("==", i, ir.Constant(I64, 0))
        upper_ch = self.builder.call(toupper_fn, [ch_i32])
        lower_ch = self.builder.call(tolower_fn, [ch_i32])
        result_ch = self.builder.select(is_first, upper_ch, lower_ch)
        self.builder.store(self.builder.trunc(result_ch, I8), self.builder.gep(new_data, [i], inbounds=True))
        self.builder.store(self.builder.add(i, ir.Constant(I64, 1)), i_a)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(done_bb)
        null_ptr = self.builder.gep(new_data, [src_len], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_ptr)
        return self._make_str_object(src_len, src_len, new_data)

    def _str_center(self, obj: Value, width_val: Value, fillchar_val: Value, node) -> Value:
        """str.center(width[, fillchar]) - proste generowanie LLVM IR."""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        w = self._to_int(width_val).llvm

        # total = max(src_len, w)
        need_pad = self.builder.icmp_signed("<", src_len, w)
        total = self.builder.select(need_pad, w, src_len)
        pad = self.builder.sub(total, src_len)
        left_pad = self.builder.sdiv(pad, ir.Constant(I64, 2))

        new_data = self.builder.call(self._malloc, [self.builder.add(total, ir.Constant(I64, 1))])
        # Fill with fillchar (assume ' ')
        fill_byte = ir.Constant(I8, 32)  # space

        # memset left pad
        self.builder.call(self._memset, [new_data, self.builder.zext(fill_byte, I32), left_pad])

        # memcpy source
        dst_offset = self.builder.gep(new_data, [left_pad], inbounds=True)
        self.builder.call(self._memcpy_decl, [dst_offset, src_data, src_len])

        # memset right pad
        right_start = self.builder.add(left_pad, src_len)
        right_pad = self.builder.sub(total, right_start)
        right_dst = self.builder.gep(new_data, [right_start], inbounds=True)
        self.builder.call(self._memset, [right_dst, self.builder.zext(fill_byte, I32), right_pad])

        # null terminate
        null_dst = self.builder.gep(new_data, [total], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_dst)

        return self._make_str_object(total, self.builder.add(total, ir.Constant(I64, 1)), new_data)

    def _str_startsend(self, obj: Value, prefix: Value, is_start: bool, node) -> Value:
        """str.startswith(prefix) / str.endswith(prefix) -> bool"""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        pfx_len = self.builder.load(self.builder.gep(prefix.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        pfx_data = self.builder.load(self.builder.gep(prefix.llvm, [z, ir.Constant(I32, 3)], inbounds=True))

        # Check prefix fits
        fits = self.builder.icmp_signed("<=", pfx_len, src_len)
        fits_i1 = fits

        # If doesn't fit, return False
        check_bb = self.current_func.append_basic_block("ss.check")
        no_bb = self.current_func.append_basic_block("ss.no")
        after_bb = self.current_func.append_basic_block("ss.after")
        self.builder.cbranch(fits_i1, check_bb, no_bb)

        self.builder.position_at_end(check_bb)
        if is_start:
            start_offset = ir.Constant(I64, 0)
        else:
            start_offset = self.builder.sub(src_len, pfx_len)

        # memcmp
        memcmp_fn = self._get_or_declare("memcmp", ir.FunctionType(I32, [I8P, I8P, I64]))
        src_start = self.builder.gep(src_data, [start_offset], inbounds=True)
        cmp_res = self.builder.call(memcmp_fn, [src_start, pfx_data, pfx_len])
        is_match = self.builder.icmp_signed("==", cmp_res, ir.Constant(I32, 0))
        self.builder.branch(after_bb)

        self.builder.position_at_end(no_bb)
        self.builder.branch(after_bb)

        self.builder.position_at_end(after_bb)
        result = self.builder.phi(I1, "ss_res")
        result.add_incoming(is_match, check_bb)
        result.add_incoming(ir.Constant(I1, 0), no_bb)
        return Value(result, PyType.BOOL)

    def _str_find(self, obj: Value, substr: Value, node) -> Value:
        """str.find(sub) -> int (-1 if not found)"""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        sub_len = self.builder.load(self.builder.gep(substr.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        sub_data = self.builder.load(self.builder.gep(substr.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        memcmp_fn = self._get_or_declare("memcmp", ir.FunctionType(I32, [I8P, I8P, I64]))

        max_i = self.builder.sub(src_len, sub_len)
        result_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, -1), result_a)
        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)

        cond_bb = self.current_func.append_basic_block("find.cond")
        body_bb = self.current_func.append_basic_block("find.body")
        found_bb = self.current_func.append_basic_block("find.found")
        next_bb = self.current_func.append_basic_block("find.next")
        done_bb = self.current_func.append_basic_block("find.done")

        # Check if sub is longer than src -> -1
        can_find = self.builder.icmp_signed("<=", sub_len, src_len)
        start_bb = self.current_func.append_basic_block("find.start")
        self.builder.cbranch(can_find, start_bb, done_bb)

        self.builder.position_at_end(start_bb)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        i = self.builder.load(i_a)
        self.builder.cbranch(self.builder.icmp_signed("<=", i, max_i), body_bb, done_bb)

        self.builder.position_at_end(body_bb)
        src_at_i = self.builder.gep(src_data, [i], inbounds=True)
        cmp_res = self.builder.call(memcmp_fn, [src_at_i, sub_data, sub_len])
        is_match = self.builder.icmp_signed("==", cmp_res, ir.Constant(I32, 0))
        self.builder.cbranch(is_match, found_bb, next_bb)

        self.builder.position_at_end(found_bb)
        self.builder.store(i, result_a)
        self.builder.branch(done_bb)

        self.builder.position_at_end(next_bb)
        self.builder.store(self.builder.add(i, ir.Constant(I64, 1)), i_a)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(done_bb)
        return Value(self.builder.load(result_a), PyType.INT)

    def _str_count(self, obj: Value, substr: Value, node) -> Value:
        """str.count(sub) -> int"""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        sub_len = self.builder.load(self.builder.gep(substr.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        sub_data = self.builder.load(self.builder.gep(substr.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        memcmp_fn = self._get_or_declare("memcmp", ir.FunctionType(I32, [I8P, I8P, I64]))

        max_i = self.builder.sub(src_len, sub_len)
        count_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), count_a)
        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)

        cond_bb = self.current_func.append_basic_block("cnt.cond")
        body_bb = self.current_func.append_basic_block("cnt.body")
        inc_bb = self.current_func.append_basic_block("cnt.inc")
        next_bb = self.current_func.append_basic_block("cnt.next")
        done_bb = self.current_func.append_basic_block("cnt.done")

        can_find = self.builder.icmp_signed("<=", sub_len, src_len)
        start_bb = self.current_func.append_basic_block("cnt.start")
        self.builder.cbranch(can_find, start_bb, done_bb)

        self.builder.position_at_end(start_bb)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        i = self.builder.load(i_a)
        self.builder.cbranch(self.builder.icmp_signed("<=", i, max_i), body_bb, done_bb)

        self.builder.position_at_end(body_bb)
        src_at_i = self.builder.gep(src_data, [i], inbounds=True)
        cmp_res = self.builder.call(memcmp_fn, [src_at_i, sub_data, sub_len])
        is_match = self.builder.icmp_signed("==", cmp_res, ir.Constant(I32, 0))
        self.builder.cbranch(is_match, inc_bb, next_bb)

        self.builder.position_at_end(inc_bb)
        c = self.builder.load(count_a)
        self.builder.store(self.builder.add(c, ir.Constant(I64, 1)), count_a)
        # Skip past this match (non-overlapping)
        self.builder.store(self.builder.add(i, sub_len), i_a)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(next_bb)
        self.builder.store(self.builder.add(i, ir.Constant(I64, 1)), i_a)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(done_bb)
        return Value(self.builder.load(count_a), PyType.INT)

    def _str_zfill(self, obj: Value, width_val: Value, node) -> Value:
        """str.zfill(width) -> str"""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        w = self._to_int(width_val).llvm

        need_pad = self.builder.icmp_signed("<", src_len, w)
        total = self.builder.select(need_pad, w, src_len)
        pad_len = self.builder.sub(total, src_len)

        new_data = self.builder.call(self._malloc, [self.builder.add(total, ir.Constant(I64, 1))])
        # Fill with '0'
        self.builder.call(self._memset, [new_data, ir.Constant(I32, 48), pad_len])  # '0' = 48
        # Copy source after zeros
        dst_at_pad = self.builder.gep(new_data, [pad_len], inbounds=True)
        self.builder.call(self._memcpy_decl, [dst_at_pad, src_data, src_len])
        # null terminate
        null_dst = self.builder.gep(new_data, [total], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_dst)

        return self._make_str_object(total, self.builder.add(total, ir.Constant(I64, 1)), new_data)

    def _str_ischeck(self, obj: Value, mname: str, node) -> Value:
        """str.isalpha/isdigit/isspace/etc. -> bool"""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))

        # Map method name to C check function
        check_map = {
            "isalpha": "isalpha", "isdigit": "isdigit", "isnumeric": "isdigit",
            "isalnum": "isalnum", "isspace": "isspace",
            "isupper": "isupper", "islower": "islower",
        }
        c_fn_name = check_map.get(mname, "isalpha")
        check_fn = self._get_or_declare(c_fn_name, ir.FunctionType(I32, [I32]))

        result_a = self.builder.alloca(I1)
        self.builder.store(ir.Constant(I1, 1), result_a)  # Assume True

        i_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_a)

        # Check empty string -> False for is* methods
        is_empty = self.builder.icmp_signed("==", src_len, ir.Constant(I64, 0))
        empty_bb = self.current_func.append_basic_block("ischeck.empty")
        loop_entry_bb = self.current_func.append_basic_block("ischeck.entry")
        done_bb = self.current_func.append_basic_block("ischeck.done")
        self.builder.cbranch(is_empty, empty_bb, loop_entry_bb)

        self.builder.position_at_end(empty_bb)
        self.builder.store(ir.Constant(I1, 0), result_a)
        self.builder.branch(done_bb)

        self.builder.position_at_end(loop_entry_bb)
        cond_bb = self.current_func.append_basic_block("ischeck.cond")
        body_bb = self.current_func.append_basic_block("ischeck.body")
        fail_bb = self.current_func.append_basic_block("ischeck.fail")
        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        i = self.builder.load(i_a)
        self.builder.cbranch(self.builder.icmp_signed("<", i, src_len), body_bb, done_bb)

        self.builder.position_at_end(body_bb)
        ch = self.builder.load(self.builder.gep(src_data, [i], inbounds=True))
        ch_i32 = self.builder.zext(ch, I32)
        res = self.builder.call(check_fn, [ch_i32])
        is_ok = self.builder.icmp_signed("!=", res, ir.Constant(I32, 0))
        cont_bb = self.current_func.append_basic_block("ischeck.cont")
        self.builder.cbranch(is_ok, cont_bb, fail_bb)

        self.builder.position_at_end(cont_bb)
        self.builder.store(self.builder.add(i, ir.Constant(I64, 1)), i_a)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(fail_bb)
        self.builder.store(ir.Constant(I1, 0), result_a)
        self.builder.branch(done_bb)

        self.builder.position_at_end(done_bb)
        return Value(self.builder.load(result_a), PyType.BOOL)

    def _str_justify(self, obj: Value, width_val: Value, fillchar_val: Value, is_right: bool, node) -> Value:
        """str.rjust/ljust(width[, fillchar]) -> str"""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True))
        src_data = self.builder.load(self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True))
        w = self._to_int(width_val).llvm

        need_pad = self.builder.icmp_signed("<", src_len, w)
        total = self.builder.select(need_pad, w, src_len)
        pad_len = self.builder.sub(total, src_len)

        new_data = self.builder.call(self._malloc, [self.builder.add(total, ir.Constant(I64, 1))])
        fill_byte = ir.Constant(I8, 32)  # space default

        if is_right:
            # rjust: pad on left
            self.builder.call(self._memset, [new_data, self.builder.zext(fill_byte, I32), pad_len])
            dst_at = self.builder.gep(new_data, [pad_len], inbounds=True)
            self.builder.call(self._memcpy_decl, [dst_at, src_data, src_len])
        else:
            # ljust: source first, pad on right
            self.builder.call(self._memcpy_decl, [new_data, src_data, src_len])
            right_dst = self.builder.gep(new_data, [src_len], inbounds=True)
            self.builder.call(self._memset, [right_dst, self.builder.zext(fill_byte, I32), pad_len])

        null_dst = self.builder.gep(new_data, [total], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_dst)
        return self._make_str_object(total, self.builder.add(total, ir.Constant(I64, 1)), new_data)

    def _str_format_method(self, obj: Value, node) -> Value:
        """Obsługa str.format() – np. '{0} version {1:.1f}'.format(name, version)."""
        import re as _re

        # Pobierz łańcuch formatujący z AST (musi być stałą)
        fmt_str = None
        if hasattr(node, 'func') and hasattr(node.func, 'value'):
            if isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
                fmt_str = node.func.value.value
        if fmt_str is None:
            # Fallback: zwróć sam string
            return obj

        # Parsuj specyfikatory: {0}, {1:.1f}, {} (auto-numbered)
        parts = []
        last_end = 0
        auto_idx = 0
        for m in _re.finditer(r'\{([^}]*)\}', fmt_str):
            # Tekst przed specyfikatorem
            if m.start() > last_end:
                parts.append(('literal', fmt_str[last_end:m.start()]))
            spec = m.group(1)
            # Rozdziel indeks od formatu: {0:.1f} -> idx=0, fmt=.1f
            if ':' in spec:
                idx_str, fmt_spec = spec.split(':', 1)
                idx = int(idx_str) if idx_str.strip() else auto_idx
            else:
                if spec.strip() == '':
                    idx = auto_idx
                    fmt_spec = ''
                else:
                    idx = int(spec)
                    fmt_spec = ''
            auto_idx = idx + 1
            parts.append(('spec', (idx, fmt_spec)))
            last_end = m.end()
        # Tekst po ostatnim specyfikatorze
        if last_end < len(fmt_str):
            parts.append(('literal', fmt_str[last_end:]))

        # Zbierz argumenty
        fmt_args = [self.visit(a) for a in node.args] if node.args else []

        # Generuj wynik: start z pustym stringiem, potem sklejaj
        result = self.create_string("")
        for kind, value in parts:
            if kind == 'literal':
                part_str = self.create_string(value)
                result = self.concat_strings(result, part_str)
            elif kind == 'spec':
                idx, fmt_spec = value
                if idx >= len(fmt_args):
                    part_str = self.create_string("")
                else:
                    arg = fmt_args[idx]
                    if fmt_spec == '':
                        part_str = self.val_to_str(arg)
                    elif fmt_spec == '.1f':
                        part_str = self._format_float(self._to_float(arg), 1)
                    elif fmt_spec == '.2f':
                        part_str = self._format_float(self._to_float(arg), 2)
                    elif fmt_spec == '.3f':
                        part_str = self._format_float(self._to_float(arg), 3)
                    elif fmt_spec == 'd':
                        part_str = self.val_to_str(self._to_int(arg))
                    elif fmt_spec == 's':
                        part_str = self.val_to_str(arg)
                    else:
                        # Próba parsowania precyzji z .Nf
                        if fmt_spec.startswith('.') and fmt_spec.endswith('f'):
                            try:
                                precision = int(fmt_spec[1:-1])
                                part_str = self._format_float(self._to_float(arg), precision)
                            except ValueError:
                                part_str = self.val_to_str(arg)
                        else:
                            part_str = self.val_to_str(arg)
                result = self.concat_strings(result, part_str)
        return result

    def _get_or_declare(self, name: str, fty: ir.FunctionType) -> ir.Function:
        if name in self.functions:
            return self.functions[name]
        fn = ir.Function(self.module, fty, name=name)
        self.functions[name] = fn
        return fn

    def _str_upper(self, obj: Value) -> Value:
        return self._str_case_transform(obj, upper=True)

    def _str_lower(self, obj: Value) -> Value:
        return self._str_case_transform(obj, upper=False)

    def _str_case_transform(self, obj: Value, upper: bool) -> Value:
        """upper=True → toupper, upper=False → tolower dla każdego znaku."""
        z = ir.Constant(I32, 0)

        # Pobierz długość i wskaźnik danych ze źródłowego STR_TY
        src_len = self.builder.load(
            self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        src_cap = self.builder.load(
            self.builder.gep(obj.llvm, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        src_data = self.builder.load(
            self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        # Alokuj nowy bufor danych
        new_data = self.builder.call(self._malloc, [src_cap])

        # Pętla: kopiuj + transform każdy bajt
        i_alloca = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), i_alloca)

        loop_bb = self.current_func.append_basic_block("case.loop")
        body_bb = self.current_func.append_basic_block("case.body")
        done_bb = self.current_func.append_basic_block("case.done")

        self.builder.branch(loop_bb)
        self.builder.position_at_end(loop_bb)
        i = self.builder.load(i_alloca)
        cond = self.builder.icmp_signed("<", i, src_len)
        self.builder.cbranch(cond, body_bb, done_bb)

        self.builder.position_at_end(body_bb)
        src_ch_ptr = self.builder.gep(src_data, [i], inbounds=True)
        ch = self.builder.load(src_ch_ptr)
        ch_i32 = self.builder.zext(ch, I32)

        # Declare toupper/tolower if needed
        if upper:
            fn = self._get_or_declare("toupper", ir.FunctionType(I32, [I32]))
        else:
            fn = self._get_or_declare("tolower", ir.FunctionType(I32, [I32]))

        transformed = self.builder.call(fn, [ch_i32])
        transformed_i8 = self.builder.trunc(transformed, I8)
        dst_ch_ptr = self.builder.gep(new_data, [i], inbounds=True)
        self.builder.store(transformed_i8, dst_ch_ptr)

        i_next = self.builder.add(i, ir.Constant(I64, 1))
        self.builder.store(i_next, i_alloca)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(done_bb)
        # null-terminate
        null_ptr = self.builder.gep(new_data, [src_len], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_ptr)

        # Stwórz nowy obiekt STR_TY
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        new_s = self.builder.bitcast(raw_str, STR_PTR)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 0)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I8P, None),
            self.builder.gep(new_s, [z, z, ir.Constant(I32, 3)], inbounds=True),
        )
        self.builder.store(
            src_len, self.builder.gep(new_s, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            src_cap, self.builder.gep(new_s, [z, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.store(
            new_data, self.builder.gep(new_s, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        return Value(new_s, PyType.STR)

    def _str_strip(self, obj: Value, mname: str) -> Value:
        """strip/lstrip/rstrip — usuwa białe znaki."""
        z = ir.Constant(I32, 0)
        src_len = self.builder.load(
            self.builder.gep(obj.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
        )
        src_data = self.builder.load(
            self.builder.gep(obj.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
        )

        # Znajdź start (lstrip lub strip)
        isspace_fn = self._get_or_declare("isspace", ir.FunctionType(I32, [I32]))

        start_a = self.builder.alloca(I64)
        end_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), start_a)
        self.builder.store(src_len, end_a)

        if mname in ("strip", "lstrip"):
            ls_bb = self.current_func.append_basic_block("strip.ls")
            ls_body = self.current_func.append_basic_block("strip.ls.body")
            ls_done = self.current_func.append_basic_block("strip.ls.done")
            self.builder.branch(ls_bb)
            self.builder.position_at_end(ls_bb)
            s = self.builder.load(start_a)
            cond1 = self.builder.icmp_signed("<", s, src_len)
            self.builder.cbranch(cond1, ls_body, ls_done)
            self.builder.position_at_end(ls_body)
            ch = self.builder.load(self.builder.gep(src_data, [s], inbounds=True))
            ch_i32 = self.builder.zext(ch, I32)
            sp = self.builder.call(isspace_fn, [ch_i32])
            is_sp = self.builder.icmp_signed("!=", sp, ir.Constant(I32, 0))
            inc_bb = self.current_func.append_basic_block("strip.ls.inc")
            self.builder.cbranch(is_sp, inc_bb, ls_done)
            self.builder.position_at_end(inc_bb)
            self.builder.store(self.builder.add(s, ir.Constant(I64, 1)), start_a)
            self.builder.branch(ls_bb)
            self.builder.position_at_end(ls_done)

        if mname in ("strip", "rstrip"):
            rs_bb = self.current_func.append_basic_block("strip.rs")
            rs_body = self.current_func.append_basic_block("strip.rs.body")
            rs_done = self.current_func.append_basic_block("strip.rs.done")
            self.builder.branch(rs_bb)
            self.builder.position_at_end(rs_bb)
            e = self.builder.load(end_a)
            s2 = self.builder.load(start_a)
            cond2 = self.builder.icmp_signed(">", e, s2)
            self.builder.cbranch(cond2, rs_body, rs_done)
            self.builder.position_at_end(rs_body)
            em1 = self.builder.sub(e, ir.Constant(I64, 1))
            ch = self.builder.load(self.builder.gep(src_data, [em1], inbounds=True))
            ch_i32 = self.builder.zext(ch, I32)
            sp = self.builder.call(isspace_fn, [ch_i32])
            is_sp = self.builder.icmp_signed("!=", sp, ir.Constant(I32, 0))
            dec_bb = self.current_func.append_basic_block("strip.rs.dec")
            self.builder.cbranch(is_sp, dec_bb, rs_done)
            self.builder.position_at_end(dec_bb)
            self.builder.store(em1, end_a)
            self.builder.branch(rs_bb)
            self.builder.position_at_end(rs_done)

        # new_len = end - start
        start_v = self.builder.load(start_a)
        end_v = self.builder.load(end_a)
        new_len = self.builder.sub(end_v, start_v)
        new_cap = self.builder.add(new_len, ir.Constant(I64, 1))

        # Alokuj bufor i skopiuj
        new_data = self.builder.call(self._malloc, [new_cap])
        src_start_ptr = self.builder.gep(src_data, [start_v], inbounds=True)
        memcpy_fn = self._get_or_declare(
            "memcpy", ir.FunctionType(I8P, [I8P, I8P, I64])
        )
        self.builder.call(memcpy_fn, [new_data, src_start_ptr, new_len])
        null_ptr = self.builder.gep(new_data, [new_len], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_ptr)

        # Nowy STR_TY
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        new_s = self.builder.bitcast(raw_str, STR_PTR)
        z_s = ir.Constant(I32, 0)
        self.builder.store(
            ir.Constant(I64, 1),
            self.builder.gep(new_s, [z_s, z_s, ir.Constant(I32, 0)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I32, 0),
            self.builder.gep(new_s, [z_s, z_s, ir.Constant(I32, 1)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I64, 0),
            self.builder.gep(new_s, [z_s, z_s, ir.Constant(I32, 2)], inbounds=True),
        )
        self.builder.store(
            ir.Constant(I8P, None),
            self.builder.gep(new_s, [z_s, z_s, ir.Constant(I32, 3)], inbounds=True),
        )
        self.builder.store(
            new_len, self.builder.gep(new_s, [z_s, ir.Constant(I32, 1)], inbounds=True)
        )
        self.builder.store(
            new_cap, self.builder.gep(new_s, [z_s, ir.Constant(I32, 2)], inbounds=True)
        )
        self.builder.store(
            new_data, self.builder.gep(new_s, [z_s, ir.Constant(I32, 3)], inbounds=True)
        )

        return Value(new_s, PyType.STR)

    # ══════════════════════════════════════════════════════════════════
    #  POPRAWKA 3: str.join() implementation (Test 02)
    # ══════════════════════════════════════════════════════════════════

    def _str_join_impl(
        self, separator: Value, items_list: Value, node: ast.AST
    ) -> Value:
        """
        Implementacja str.join(list) -> str
        """
        z = ir.Constant(I32, 0)
        func = self.current_func

        sep_ptr = separator.llvm  # STR_PTR
        list_ptr = items_list.llvm  # LIST_PTR

        # Pobierz długość separatora
        sep_len_ptr = self.builder.gep(sep_ptr, [z, ir.Constant(I32, 1)], inbounds=True)
        sep_len = self.builder.load(sep_len_ptr, "sep_len")

        sep_data_ptr = self.builder.gep(
            sep_ptr, [z, ir.Constant(I32, 3)], inbounds=True
        )
        sep_data = self.builder.load(sep_data_ptr, "sep_data")

        # Pobierz informacje o liście
        list_size_ptr = self.builder.gep(
            list_ptr, [z, ir.Constant(I32, 1)], inbounds=True
        )
        list_size = self.builder.load(list_size_ptr, "list_size")

        list_data_ptr = self.builder.gep(
            list_ptr, [z, ir.Constant(I32, 3)], inbounds=True
        )
        list_data = self.builder.load(list_data_ptr, "list_data")

        # Oblicz całkowity rozmiar wynikowego stringa
        # total_len = sum(len(item) for item in list) + sep_len * (list_size - 1)

        # Alloca dla accumulatora
        total_len_alloca = self.builder.alloca(I64, name="total_len")
        self.builder.store(ir.Constant(I64, 0), total_len_alloca)

        result_alloca = self.builder.alloca(I8P, name="result_ptr")
        self.builder.store(ir.Constant(I8P, None), result_alloca)

        # Pierwsza pętla: oblicz całkowitą długość
        idx_alloca = self.builder.alloca(I64, name="join_idx")
        self.builder.store(ir.Constant(I64, 0), idx_alloca)

        calc_cond = func.append_basic_block("join.calc.cond")
        calc_body = func.append_basic_block("join.calc.body")
        calc_done = func.append_basic_block("join.calc.done")

        self.builder.branch(calc_cond)

        self.builder.position_at_end(calc_cond)
        idx = self.builder.load(idx_alloca, "j_idx")
        self.builder.cbranch(
            self.builder.icmp_signed("<", idx, list_size), calc_body, calc_done
        )

        self.builder.position_at_end(calc_body)
        # Pobierz element listy (jest BOXED)
        elem_ptr = self.builder.gep(list_data, [idx], inbounds=True)
        elem_tag_ptr = self.builder.gep(
            elem_ptr, [z, ir.Constant(I32, 1)], inbounds=True
        )
        elem_pay_ptr = self.builder.gep(
            elem_ptr, [z, ir.Constant(I32, 2)], inbounds=True
        )

        elem_tag = self.builder.load(elem_tag_ptr, "elem_tag")
        elem_pay = self.builder.load(elem_pay_ptr, "elem_pay")

        # Sprawdź czy to STR
        is_str = self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.STR))

        str_path = func.append_basic_block("join.elem.str")
        non_str_path = func.append_basic_block("join.elem.nonstr")
        next_elem = func.append_basic_block("join.calc.next")

        self.builder.cbranch(is_str, str_path, non_str_path)

        self.builder.position_at_end(str_path)
        elem_str_ptr = self.builder.inttoptr(elem_pay, STR_PTR)
        elem_len_ptr = self.builder.gep(
            elem_str_ptr, [z, ir.Constant(I32, 1)], inbounds=True
        )
        elem_len = self.builder.load(elem_len_ptr, "elem_len")

        # Dodaj do total_len
        curr_len = self.builder.load(total_len_alloca, "curr_len")
        elem_len_64 = self.builder.zext(elem_len, I64)
        new_len = self.builder.add(curr_len, elem_len_64)
        self.builder.store(new_len, total_len_alloca)
        self.builder.branch(next_elem)

        self.builder.position_at_end(non_str_path)
        # Dla nie-stringów, konwertuj na string i dodaj długość
        # Uproszczenie: wymaga konwersji przez snprintf
        # TODO: pełna implementacja konwersji
        self.builder.branch(next_elem)

        self.builder.position_at_end(next_elem)
        # Dodaj separator (jeśli nie pierwszy element)
        is_first = self.builder.icmp_signed("==", idx, ir.Constant(I64, 0))
        add_sep = func.append_basic_block("join.add_sep")
        skip_sep = func.append_basic_block("join.skip_sep")
        self.builder.cbranch(is_first, skip_sep, add_sep)

        self.builder.position_at_end(add_sep)
        curr_len2 = self.builder.load(total_len_alloca, "curr_len2")
        with_sep = self.builder.add(curr_len2, sep_len)
        self.builder.store(with_sep, total_len_alloca)
        self.builder.branch(skip_sep)

        self.builder.position_at_end(skip_sep)
        self.builder.store(self.builder.add(idx, ir.Constant(I64, 1)), idx_alloca)
        self.builder.branch(calc_cond)

        self.builder.position_at_end(calc_done)

        # Alokuj wynikowy string - separate header and data
        total_len = self.builder.load(total_len_alloca, "total_len")
        # +1 dla null terminator
        buf_size = self.builder.add(total_len, ir.Constant(I64, 1))
        data_buf = self.builder.call(self._malloc, [buf_size], "join_data")

        # Allocate STR object header
        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)], "join_str_raw")
        result_str_obj = self.builder.bitcast(raw_str, STR_PTR, "join_str_obj")

        # Init GC header
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(result_str_obj, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(result_str_obj, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(result_str_obj, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(null_i8p, self.builder.gep(result_str_obj, [z, z, ir.Constant(I32, 3)], inbounds=True))

        # Set len, cap, data
        self.builder.store(total_len, self.builder.gep(result_str_obj, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(buf_size, self.builder.gep(result_str_obj, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(data_buf, self.builder.gep(result_str_obj, [z, ir.Constant(I32, 3)], inbounds=True))

        # Druga pętla: kopiuj elementy
        write_pos_alloca = self.builder.alloca(I64, name="write_pos")
        self.builder.store(ir.Constant(I64, 0), write_pos_alloca)

        self.builder.store(ir.Constant(I64, 0), idx_alloca)

        copy_cond = func.append_basic_block("join.copy.cond")
        copy_body = func.append_basic_block("join.copy.body")
        copy_done = func.append_basic_block("join.copy.done")

        self.builder.branch(copy_cond)

        self.builder.position_at_end(copy_cond)
        idx2 = self.builder.load(idx_alloca, "j_idx2")
        self.builder.cbranch(
            self.builder.icmp_signed("<", idx2, list_size), copy_body, copy_done
        )

        self.builder.position_at_end(copy_body)
        # Dodaj separator jeśli nie pierwszy
        is_first2 = self.builder.icmp_signed("==", idx2, ir.Constant(I64, 0))
        copy_sep_bb = func.append_basic_block("join.copy.sep")
        skip_sep2_bb = func.append_basic_block("join.skip_sep2")
        self.builder.cbranch(is_first2, skip_sep2_bb, copy_sep_bb)

        self.builder.position_at_end(copy_sep_bb)
        wpos = self.builder.load(write_pos_alloca, "wpos")
        dest = self.builder.gep(data_buf, [wpos], inbounds=True)
        self.builder.call(self._memcpy_decl, [dest, sep_data, sep_len])
        self.builder.store(self.builder.add(wpos, sep_len), write_pos_alloca)
        self.builder.branch(skip_sep2_bb)

        self.builder.position_at_end(skip_sep2_bb)

        # Kopiuj element
        elem_ptr2 = self.builder.gep(list_data, [idx2], inbounds=True)
        elem_tag2 = self.builder.load(
            self.builder.gep(elem_ptr2, [z, ir.Constant(I32, 1)], inbounds=True), "et2"
        )
        elem_pay2 = self.builder.load(
            self.builder.gep(elem_ptr2, [z, ir.Constant(I32, 2)], inbounds=True), "ep2"
        )

        is_str2 = self.builder.icmp_signed("==", elem_tag2, ir.Constant(I64, Tag.STR))
        copy_str_bb = func.append_basic_block("join.copy.str")
        copy_conv_bb = func.append_basic_block("join.copy.conv")
        copy_next_bb = func.append_basic_block("join.copy.next")

        self.builder.cbranch(is_str2, copy_str_bb, copy_conv_bb)

        self.builder.position_at_end(copy_str_bb)
        elem_str2 = self.builder.inttoptr(elem_pay2, STR_PTR)
        elem_data2 = self.builder.load(
            self.builder.gep(elem_str2, [z, ir.Constant(I32, 3)], inbounds=True),
            "ed2",
        )
        elem_len2 = self.builder.load(
            self.builder.gep(elem_str2, [z, ir.Constant(I32, 1)], inbounds=True),
            "el2",
        )

        wpos2 = self.builder.load(write_pos_alloca, "wpos2")
        dest2 = self.builder.gep(data_buf, [wpos2], inbounds=True)
        elem_len2_64 = self.builder.zext(elem_len2, I64)
        self.builder.call(self._memcpy_decl, [dest2, elem_data2, elem_len2_64])
        self.builder.store(self.builder.add(wpos2, elem_len2_64), write_pos_alloca)
        self.builder.branch(copy_next_bb)

        self.builder.position_at_end(copy_conv_bb)
        # Konwersja nie-string elementu (np. int) na string
        # Uproszczenie: pomijamy w tej wersji
        self.builder.branch(copy_next_bb)

        self.builder.position_at_end(copy_next_bb)
        self.builder.store(self.builder.add(idx2, ir.Constant(I64, 1)), idx_alloca)
        self.builder.branch(copy_cond)

        self.builder.position_at_end(copy_done)
        # Null terminator
        wpos_final = self.builder.load(write_pos_alloca, "wpos_final")
        null_pos = self.builder.gep(data_buf, [wpos_final], inbounds=True)
        self.builder.store(ir.Constant(I8, 0), null_pos)

        return Value(result_str_obj, PyType.STR)

    def _str_split_impl(self, obj: Value, delimiter: Value, node: ast.AST) -> Value:
        """
        Implementacja str.split([delimiter]) -> list
        """
        z = ir.Constant(I32, 0)
        func = self.current_func

        s_ptr = obj.llvm
        s_len = self.builder.load(self.builder.gep(s_ptr, [z, ir.Constant(I32, 1)], inbounds=True))
        s_data = self.builder.load(self.builder.gep(s_ptr, [z, ir.Constant(I32, 3)], inbounds=True))

        res_list = self.create_list([])

        if delimiter is None or delimiter.pytype == PyType.NONE:
            # Simple split by whitespace (simplified: split by ' ')
            delim_ptr = self.create_string(" ").llvm
            delim_len = ir.Constant(I64, 1)
            delim_data = self.builder.load(self.builder.gep(delim_ptr, [z, ir.Constant(I32, 3)], inbounds=True))
        else:
            delim_ptr = delimiter.llvm
            delim_len = self.builder.load(self.builder.gep(delim_ptr, [z, ir.Constant(I32, 1)], inbounds=True))
            delim_data = self.builder.load(self.builder.gep(delim_ptr, [z, ir.Constant(I32, 3)], inbounds=True))

        curr_pos_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), curr_pos_a)

        split_cond = func.append_basic_block("split.cond")
        split_body = func.append_basic_block("split.body")
        split_end = func.append_basic_block("split.end")
        self.builder.branch(split_cond)

        self.builder.position_at_end(split_cond)
        curr_pos = self.builder.load(curr_pos_a)
        self.builder.cbranch(self.builder.icmp_signed("<", curr_pos, s_len), split_body, split_end)

        self.builder.position_at_end(split_body)
        # Find next delimiter
        found_pos_a = self.builder.alloca(I64)
        self.builder.store(curr_pos, found_pos_a)

        search_cond = func.append_basic_block("split.search.cond")
        search_body = func.append_basic_block("split.search.body")
        search_no_more = func.append_basic_block("split.search.no_more")
        self.builder.branch(search_cond)

        self.builder.position_at_end(search_cond)
        s_idx = self.builder.load(found_pos_a)
        # Check if s_idx + delim_len <= s_len
        can_search = self.builder.icmp_signed("<=", self.builder.add(s_idx, delim_len), s_len)
        self.builder.cbranch(can_search, search_body, search_no_more)

        self.builder.position_at_end(search_body)
        # Compare delim_data with s_data[s_idx...s_idx+delim_len]
        # Use a loop to check all bytes of the delimiter
        match_a = self.builder.alloca(I1)
        self.builder.store(ir.Constant(I1, True), match_a)
        k_a = self.builder.alloca(I64)
        self.builder.store(ir.Constant(I64, 0), k_a)

        delim_check_cond = func.append_basic_block("split.delim.chk.cond")
        delim_check_body = func.append_basic_block("split.delim.chk.body")
        delim_check_done = func.append_basic_block("split.delim.chk.done")
        self.builder.branch(delim_check_cond)

        self.builder.position_at_end(delim_check_cond)
        k = self.builder.load(k_a)
        all_bytes_checked = self.builder.icmp_signed(">=", k, delim_len)
        self.builder.cbranch(all_bytes_checked, delim_check_done, delim_check_body)

        self.builder.position_at_end(delim_check_body)
        s_ch = self.builder.load(self.builder.gep(s_data, [self.builder.add(s_idx, k)], inbounds=True))
        d_ch = self.builder.load(self.builder.gep(delim_data, [k], inbounds=True))
        eq = self.builder.icmp_signed("==", s_ch, d_ch)
        cur_match = self.builder.load(match_a)
        self.builder.store(self.builder.and_(cur_match, eq), match_a)
        self.builder.store(self.builder.add(k, ir.Constant(I64, 1)), k_a)
        self.builder.branch(delim_check_cond)

        self.builder.position_at_end(delim_check_done)
        match = self.builder.load(match_a)
        match_bb = func.append_basic_block("split.match")
        no_match_bb = func.append_basic_block("split.no_match")
        self.builder.cbranch(match, match_bb, no_match_bb)

        self.builder.position_at_end(no_match_bb)
        self.builder.store(self.builder.add(s_idx, ir.Constant(I64, 1)), found_pos_a)
        self.builder.branch(search_cond)

        self.builder.position_at_end(match_bb)
        # Match found! Extract substring from curr_pos to found_pos
        found_pos = self.builder.load(found_pos_a)
        sub_len = self.builder.sub(found_pos, curr_pos)
        sub_raw = self.builder.call(self._malloc, [self.builder.add(sub_len, ir.Constant(I64, 1))])
        sub_raw_i8p = self.builder.bitcast(sub_raw, I8P)
        src_offset = self.builder.gep(s_data, [curr_pos])
        self.builder.call(self.functions["memcpy"], [sub_raw_i8p, src_offset, sub_len])
        self.builder.store(ir.Constant(I8, 0), self.builder.gep(sub_raw_i8p, [sub_len]))
        # Create STR_TY for the substring
        sub_s_raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        sub_s = self.builder.bitcast(sub_s_raw, STR_PTR)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(sub_s, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(sub_s, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(sub_s, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(ir.Constant(I8P, None), self.builder.gep(sub_s, [z, z, ir.Constant(I32, 3)], inbounds=True))
        self.builder.store(sub_len, self.builder.gep(sub_s, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(sub_len, self.builder.gep(sub_s, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(sub_raw_i8p, self.builder.gep(sub_s, [z, ir.Constant(I32, 3)], inbounds=True))
        self.list_append(res_list, Value(sub_s, PyType.STR))

        # Update curr_pos = found_pos + delim_len
        self.builder.store(self.builder.add(found_pos, delim_len), curr_pos_a)
        self.builder.branch(split_cond)

        # No more delimiter found - go to split_end which will add the remaining text
        self.builder.position_at_end(search_no_more)
        self.builder.branch(split_end)

        self.builder.position_at_end(split_end)
        # Add last part - copy remaining text after the last delimiter
        # Only add if there is remaining text (curr_pos < s_len)
        last_sub_start = self.builder.load(curr_pos_a)
        has_remaining = self.builder.icmp_signed("<", last_sub_start, s_len)
        add_last_bb = func.append_basic_block("split.add_last")
        skip_last_bb = func.append_basic_block("split.skip_last")
        self.builder.cbranch(has_remaining, add_last_bb, skip_last_bb)

        self.builder.position_at_end(add_last_bb)
        last_sub_len = self.builder.sub(s_len, last_sub_start)
        last_raw = self.builder.call(self._malloc, [self.builder.add(last_sub_len, ir.Constant(I64, 1))])
        last_raw_i8 = self.builder.bitcast(last_raw, I8P)
        src_off_ptr = self.builder.gep(s_data, [last_sub_start])
        self.builder.call(
            self.functions["memcpy"],
            [last_raw_i8, src_off_ptr, last_sub_len],
        )
        # null-terminate
        self.builder.store(
            ir.Constant(I8, 0),
            self.builder.gep(last_raw_i8, [last_sub_len]),
        )
        # Create proper STR_TY for the last part
        raw_s = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        last_str = self.builder.bitcast(raw_s, STR_PTR)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(last_str, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(last_str, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(last_str, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(ir.Constant(I8P, None), self.builder.gep(last_str, [z, z, ir.Constant(I32, 3)], inbounds=True))
        self.builder.store(last_sub_len, self.builder.gep(last_str, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(last_sub_len, self.builder.gep(last_str, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(last_raw_i8, self.builder.gep(last_str, [z, ir.Constant(I32, 3)], inbounds=True))
        self.list_append(res_list, Value(last_str, PyType.STR))
        self.builder.branch(skip_last_bb)

        self.builder.position_at_end(skip_last_bb)
        return res_list

    def _str_replace_impl(self, obj: Value, old_val: Value, new_val: Value, node: ast.AST) -> Value:
        z = ir.Constant(I32, 0)
        func = self.current_func

        s_ptr = obj.llvm
        s_len = self.builder.load(self.builder.gep(s_ptr, [z, ir.Constant(I32, 1)], inbounds=True), "s_len")
        s_data = self.builder.load(self.builder.gep(s_ptr, [z, ir.Constant(I32, 3)], inbounds=True), "s_data")

        old_ptr = old_val.llvm
        old_len = self.builder.load(self.builder.gep(old_ptr, [z, ir.Constant(I32, 1)], inbounds=True), "old_len")
        old_data = self.builder.load(self.builder.gep(old_ptr, [z, ir.Constant(I32, 3)], inbounds=True), "old_data")

        new_ptr = new_val.llvm
        new_len = self.builder.load(self.builder.gep(new_ptr, [z, ir.Constant(I32, 1)], inbounds=True), "new_len")
        new_data = self.builder.load(self.builder.gep(new_ptr, [z, ir.Constant(I32, 3)], inbounds=True), "new_data")

        # First pass: count occurrences
        count_a = self.builder.alloca(I64, name="repl_count")
        self.builder.store(ir.Constant(I64, 0), count_a)
        idx_a = self.builder.alloca(I64, name="repl_idx")
        self.builder.store(ir.Constant(I64, 0), idx_a)

        cnt_cond = func.append_basic_block("repl.cnt.cond")
        cnt_body = func.append_basic_block("repl.cnt.body")
        cnt_check = func.append_basic_block("repl.cnt.check")
        cnt_skip = func.append_basic_block("repl.cnt.skip")
        cnt_end = func.append_basic_block("repl.cnt.end")

        self.builder.branch(cnt_cond)

        self.builder.position_at_end(cnt_cond)
        idx = self.builder.load(idx_a, "r_idx")
        can_check = self.builder.icmp_signed("<=", self.builder.add(idx, old_len), s_len)
        self.builder.cbranch(can_check, cnt_body, cnt_end)

        self.builder.position_at_end(cnt_body)
        match_a = self.builder.alloca(I1, name="repl_match")
        self.builder.store(ir.Constant(I1, True), match_a)
        k_a = self.builder.alloca(I64, name="repl_k")
        self.builder.store(ir.Constant(I64, 0), k_a)
        self.builder.branch(cnt_check)

        self.builder.position_at_end(cnt_check)
        k = self.builder.load(k_a)
        all_checked = self.builder.icmp_signed(">=", k, old_len)
        self.builder.cbranch(all_checked, cnt_skip, func.append_basic_block("repl.cnt.cmp"))

        cmp_bb = list(func.blocks)[-1]
        self.builder.position_at_end(cmp_bb)
        s_ch = self.builder.load(self.builder.gep(s_data, [self.builder.add(idx, k)], inbounds=True))
        o_ch = self.builder.load(self.builder.gep(old_data, [k], inbounds=True))
        eq = self.builder.icmp_signed("==", s_ch, o_ch)
        cur_match = self.builder.load(match_a)
        self.builder.store(self.builder.and_(cur_match, eq), match_a)
        self.builder.store(self.builder.add(k, ir.Constant(I64, 1)), k_a)
        self.builder.branch(cnt_check)

        self.builder.position_at_end(cnt_skip)
        is_match = self.builder.load(match_a)
        cnt_match_bb = func.append_basic_block("repl.cnt.match")
        cnt_nomatch_bb = func.append_basic_block("repl.cnt.nomatch")
        self.builder.cbranch(is_match, cnt_match_bb, cnt_nomatch_bb)

        self.builder.position_at_end(cnt_match_bb)
        c = self.builder.load(count_a)
        self.builder.store(self.builder.add(c, ir.Constant(I64, 1)), count_a)
        self.builder.store(self.builder.add(idx, old_len), idx_a)
        self.builder.branch(cnt_cond)

        self.builder.position_at_end(cnt_nomatch_bb)
        self.builder.store(self.builder.add(idx, ir.Constant(I64, 1)), idx_a)
        self.builder.branch(cnt_cond)

        self.builder.position_at_end(cnt_end)
        count = self.builder.load(count_a, "repl_total")
        old_total = self.builder.mul(count, old_len)
        new_total = self.builder.mul(count, new_len)
        result_len = self.builder.add(self.builder.sub(s_len, old_total), new_total)

        result_buf = self.builder.call(self._malloc, [self.builder.add(result_len, ir.Constant(I64, 1))])
        result_i8p = self.builder.bitcast(result_buf, I8P)

        # Second pass: copy with replacement
        src_idx_a2 = self.builder.alloca(I64, name="repl_src2")
        dst_idx_a2 = self.builder.alloca(I64, name="repl_dst2")
        self.builder.store(ir.Constant(I64, 0), src_idx_a2)
        self.builder.store(ir.Constant(I64, 0), dst_idx_a2)

        cp_cond = func.append_basic_block("repl.cp.cond")
        cp_body = func.append_basic_block("repl.cp.body")
        cp_check = func.append_basic_block("repl.cp.check")
        cp_skip = func.append_basic_block("repl.cp.skip")
        cp_nomatch = func.append_basic_block("repl.cp.nomatch")
        cp_end = func.append_basic_block("repl.cp.end")

        self.builder.branch(cp_cond)

        self.builder.position_at_end(cp_cond)
        src_idx = self.builder.load(src_idx_a2)
        can_cp = self.builder.icmp_signed("<=", self.builder.add(src_idx, old_len), s_len)
        self.builder.cbranch(can_cp, cp_body, cp_end)

        self.builder.position_at_end(cp_body)
        cp_match_a = self.builder.alloca(I1, name="cp_match")
        self.builder.store(ir.Constant(I1, True), cp_match_a)
        cp_k_a = self.builder.alloca(I64, name="cp_k")
        self.builder.store(ir.Constant(I64, 0), cp_k_a)
        self.builder.branch(cp_check)

        self.builder.position_at_end(cp_check)
        cp_k = self.builder.load(cp_k_a)
        cp_all = self.builder.icmp_signed(">=", cp_k, old_len)
        self.builder.cbranch(cp_all, cp_skip, func.append_basic_block("repl.cp.cmp"))

        cp_cmp_bb = list(func.blocks)[-1]
        self.builder.position_at_end(cp_cmp_bb)
        s_ch2 = self.builder.load(self.builder.gep(s_data, [self.builder.add(src_idx, cp_k)], inbounds=True))
        o_ch2 = self.builder.load(self.builder.gep(old_data, [cp_k], inbounds=True))
        eq2 = self.builder.icmp_signed("==", s_ch2, o_ch2)
        cur_m2 = self.builder.load(cp_match_a)
        self.builder.store(self.builder.and_(cur_m2, eq2), cp_match_a)
        self.builder.store(self.builder.add(cp_k, ir.Constant(I64, 1)), cp_k_a)
        self.builder.branch(cp_check)

        self.builder.position_at_end(cp_skip)
        cp_is_match = self.builder.load(cp_match_a)
        self.builder.cbranch(cp_is_match, func.append_basic_block("repl.cp.domatch"), cp_nomatch)

        cp_match_bb = list(func.blocks)[-1]
        self.builder.position_at_end(cp_match_bb)
        dst_i = self.builder.load(dst_idx_a2)
        self.builder.call(self.functions["memcpy"], [
            self.builder.gep(result_i8p, [dst_i]),
            new_data,
            new_len,
        ])
        self.builder.store(self.builder.add(dst_i, new_len), dst_idx_a2)
        self.builder.store(self.builder.add(src_idx, old_len), src_idx_a2)
        self.builder.branch(cp_cond)

        self.builder.position_at_end(cp_nomatch)
        src_i = self.builder.load(src_idx_a2)
        dst_j = self.builder.load(dst_idx_a2)
        src_ch = self.builder.load(self.builder.gep(s_data, [src_i], inbounds=True))
        self.builder.store(src_ch, self.builder.gep(result_i8p, [dst_j]))
        self.builder.store(self.builder.add(src_i, ir.Constant(I64, 1)), src_idx_a2)
        self.builder.store(self.builder.add(dst_j, ir.Constant(I64, 1)), dst_idx_a2)
        self.builder.branch(cp_cond)

        self.builder.position_at_end(cp_end)
        src_rem = self.builder.load(src_idx_a2)
        dst_rem = self.builder.load(dst_idx_a2)
        remaining = self.builder.sub(s_len, src_rem)
        self.builder.call(self.functions["memcpy"], [
            self.builder.gep(result_i8p, [dst_rem]),
            self.builder.gep(s_data, [src_rem]),
            remaining,
        ])
        final_dst = self.builder.add(dst_rem, remaining)
        self.builder.store(ir.Constant(I8, 0), self.builder.gep(result_i8p, [final_dst]))

        raw_str = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)])
        result_str = self.builder.bitcast(raw_str, STR_PTR)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(result_str, [z, z, ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, 0), self.builder.gep(result_str, [z, z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(result_str, [z, z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(ir.Constant(I8P, None), self.builder.gep(result_str, [z, z, ir.Constant(I32, 3)], inbounds=True))
        total_len = self.builder.add(dst_rem, remaining)
        self.builder.store(total_len, self.builder.gep(result_str, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(total_len, self.builder.gep(result_str, [z, ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(result_i8p, self.builder.gep(result_str, [z, ir.Constant(I32, 3)], inbounds=True))

        return Value(result_str, PyType.STR)

    def _method_call(self, node: ast.Call) -> Value:
        mname = node.func.attr

        # ══════════════════════════════════════════════════════════════════
        #  Pure-Python module call (module.func()) — checked BEFORE FFI
        #  so that pure-Python imports are NOT routed through _ffi_call.
        #  Pure-Python module functions use pylow's internal calling
        #  convention (BOXED_PTR args/returns), not C-style FFI.
        # ══════════════════════════════════════════════════════════════════
        if hasattr(self, '_pure_python_module_symbols') and isinstance(node.func.value, ast.Name):
            mod_name = node.func.value.id
            if mod_name in self._pure_python_module_symbols:
                return self._pure_python_module_call(mod_name, mname, node)

        # ══════════════════════════════════════════════════════════════════
        #  FFI: Module-qualified call (module.func()) — MUST be checked
        #  BEFORE visit(node.func.value) to prevent visit_Name from
        #  throwing "Niezdefiniowana zmienna" for FFI module references.
        # ══════════════════════════════════════════════════════════════════
        if hasattr(self, '_ffi_module_symbols') and isinstance(node.func.value, ast.Name):
            mod_name = node.func.value.id
            if mod_name in self._ffi_module_symbols:
                # CRITICAL: For CPython extension modules (pyinit_symbol != None),
                # we MUST route through _ffi_wrapper_call because the .so functions
                # expect CPython PyObject* arguments, not pylow's internal representation.
                # _ffi_call would call the function directly, passing pylow objects
                # as i8*, which causes segfaults.
                if hasattr(self, '_ffi_modules'):
                    ffi_mod = self._ffi_modules.get(mod_name)
                    if ffi_mod and ffi_mod.pyinit_symbol is not None:
                        return self._ffi_wrapper_call(mname, node, ffi_fn=None, module_name=mod_name)
                # Non-CPython FFI: resolve symbol and call directly
                ffi_fn = self.resolve_ffi_symbol(mod_name, mname)
                if ffi_fn is not None:
                    return self._ffi_call(mname, node, ffi_fn=ffi_fn, module_name=mod_name)
                # Symbol not in ELF exports — use dlsym runtime resolution
                return self._ffi_dlsym_call(mod_name, mname, node)

        # ══════════════════════════════════════════════════════════════════
        #  super().method() — compile-time MRO resolution
        #  Kiedy node.func.value to super(), resolve'ujemy metodę
        #  na podstawie MRO klasy, w której jesteśmy, omijając
        #  bieżącą klasę i szukając w kolejnych klasach MRO.
        # ══════════════════════════════════════════════════════════════════
        if (isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"):
            return self._super_method_call(mname, node)

        obj = self.visit(node.func.value)

        # ══════════════════════════════════════════════════════════════════
        #  FFI: obj is an FFIModuleValue (returned by visit_Name for
        #  FFI module references) — dispatch to FFI call.
        # ══════════════════════════════════════════════════════════════════
        if isinstance(obj, FFIModuleValue):
            mod_name = obj.module_name

            # ════════════════════════════════════════════════════════════
            #  Built-in module dispatch (time, os, sys, math, etc.)
            #  These are NOT FFI .so modules — they have native LLVM
            #  implementations declared in _init_builtin_modules().
            # ════════════════════════════════════════════════════════════
            builtin_result = self._builtin_module_call(mod_name, mname, node)
            if builtin_result is not None:
                return builtin_result

            if hasattr(self, '_ffi_module_symbols') and mod_name in self._ffi_module_symbols:
                # CRITICAL: For CPython extension modules, always use wrapper path
                # because .so functions expect CPython PyObject* arguments.
                if hasattr(self, '_ffi_modules'):
                    ffi_mod = self._ffi_modules.get(mod_name)
                    if ffi_mod and ffi_mod.pyinit_symbol is not None:
                        return self._ffi_wrapper_call(mname, node, ffi_fn=None, module_name=mod_name)
                # Non-CPython FFI: resolve symbol and call directly
                ffi_fn = self.resolve_ffi_symbol(mod_name, mname)
                if ffi_fn is not None:
                    return self._ffi_call(mname, node, ffi_fn=ffi_fn, module_name=mod_name)
                # Symbol not in ELF exports — use dlsym runtime resolution
                return self._ffi_dlsym_call(mod_name, mname, node)
            # Module registered but no symbols found — try dlsym
            return self._ffi_dlsym_call(mod_name, mname, node)

        # ══════════════════════════════════════════════════════════════════
        #  Pure-Python module call via FFIModuleValue — checked after
        #  built-in dispatch but before falling through to dlsym.
        # ══════════════════════════════════════════════════════════════════
        if hasattr(self, '_pure_python_module_symbols') and isinstance(obj, FFIModuleValue):
            mod_name = obj.module_name
            if mod_name in self._pure_python_module_symbols:
                return self._pure_python_module_call(mod_name, mname, node)

        # 1. Statycznie znany typ
        if obj.is_str:
            return self._str_method_call(obj, mname, node)
        if obj.is_list:
            if mname == "append":
                if len(node.args) != 1:
                    raise CompileError("append() wymaga 1 argumentu.", node)
                self.list_append(obj, self.visit(node.args[0]))
                return Value(ir.Constant(I64, 0), PyType.NONE)
        if obj.is_dict:
            return self._dict_method_call(obj, mname, node)
        # Dict method call on boxed object — only if the method name
        # is a known dict method. Otherwise, let it fall through.
        _DICT_METHODS = {"clear", "keys", "values", "items", "get", "pop", "update", "setdefault"}
        if obj.is_object and obj.llvm.type == BOXED_PTR and mname in _DICT_METHODS:
            # Check if it's a dict at runtime
            tag, pay = self._read_slot(obj.llvm)
            is_dict_tag = self.builder.icmp_signed("==", tag, ir.Constant(I64, Tag.DICT))
            dict_meth_bb = self.current_func.append_basic_block("meth.dict")
            not_dict_bb = self.current_func.append_basic_block("meth.not_dict")
            merge_bb = self.current_func.append_basic_block("meth.dict_merge")
            # Alloca for the result
            result_alloca = self.builder.alloca(BOXED_PTR, name="dict_meth_result")
            self.builder.cbranch(is_dict_tag, dict_meth_bb, not_dict_bb)

            self.builder.position_at_end(dict_meth_bb)
            dptr = self.builder.inttoptr(pay, DICT_PTR, name="meth_dict_ptr")
            dict_val = Value(dptr, PyType.DICT)
            dict_result = self._dict_method_call(dict_val, mname, node)
            dict_result_boxed = dict_result.llvm if dict_result.is_object else self._box(dict_result)
            self.builder.store(dict_result_boxed, result_alloca)
            self.builder.branch(merge_bb)

            self.builder.position_at_end(not_dict_bb)
            # Not a dict — error
            none_boxed = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
            self.builder.store(none_boxed, result_alloca)
            self.builder.branch(merge_bb)

            self.builder.position_at_end(merge_bb)
            return Value(self.builder.load(result_alloca, name="dict_meth_res"), PyType.OBJECT)

        # 1b. Statycznie znana instancja klasy (PyType.INSTANCE) - bezpośrednie wywołanie
        if obj.is_instance:
            inferred_class = obj.class_name
            # Sprawdź VarInfo jako fallback
            if not inferred_class and hasattr(node.func, "value") and hasattr(node.func.value, "id"):
                var_name = node.func.value.id
                try:
                    var_info = self.sym.lookup(var_name)
                    if hasattr(var_info, "class_name") and var_info.class_name:
                        inferred_class = var_info.class_name
                except: pass
            if inferred_class:
                target = f"py_{inferred_class}_{mname}"
                found_func = None
                if target in self.functions and isinstance(self.functions[target], ir.Function):
                    found_func = self.functions[target]
                # MRO - szukaj w klasach bazowych
                if found_func is None:
                    hierarchy = getattr(self, '_class_hierarchy', {})
                    mro_classes = self._get_mro(inferred_class, hierarchy)
                    for cls_name in mro_classes:
                        t = f"py_{cls_name}_{mname}"
                        if t in self.functions and isinstance(self.functions[t], ir.Function):
                            found_func = self.functions[t]
                            break
                if found_func and isinstance(found_func, ir.Function):
                    inst_ptr = obj.llvm
                    first_arg_type = found_func.args[0].type if found_func.args else BOXED_PTR
                    if first_arg_type == INSTANCE_PTR:
                        self_arg = inst_ptr
                    elif first_arg_type == DICT_PTR:
                        z = ir.Constant(I32, 0)
                        self_arg = self.builder.load(self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True))
                    else:
                        self_arg = self._box(obj)
                    call_args = [self_arg]
                    for arg in node.args:
                        v = self.visit(arg)
                        call_args.append(v.llvm if (v.is_object or v.is_dict or v.is_instance) else self._box(v))
                    if len(call_args) == len(found_func.args):
                        call_args = self._verify_call_args(found_func, call_args)
                        ret = self.builder.call(found_func, call_args)
                        self._check_exc_after_call()
                        return Value(ret, self._llvm_type_to_pytype(found_func.function_type.return_type))

        # 1c. Class reference (ClassName.method()) — for classmethod/staticmethod calls
        # CRITICAL: Only match if this is truly a class reference (e.g., Dog.speak()),
        # NOT an instance (e.g., d.speak()). Instances have class_name too but
        # should be dispatched via the instance method path below.
        if obj.is_object and obj.class_name and self._is_class_ref_value(obj):
            is_class_key = f"__is_class_{obj.class_name}"
            if is_class_key in self.functions:
                inferred_class = obj.class_name
                target = f"py_{inferred_class}_{mname}"
                found_func = None
                if target in self.functions and isinstance(self.functions[target], ir.Function):
                    found_func = self.functions[target]
                if found_func is None:
                    hierarchy = getattr(self, '_class_hierarchy', {})
                    mro_classes = self._get_mro(inferred_class, hierarchy)
                    for cls_name in mro_classes:
                        t = f"py_{cls_name}_{mname}"
                        if t in self.functions and isinstance(self.functions[t], ir.Function):
                            found_func = self.functions[t]
                            break
                if found_func and isinstance(found_func, ir.Function):
                    # Check if this is a static method (no self arg)
                    static_key = f"{inferred_class}_{mname}"
                    is_static = static_key in getattr(self, '_static_methods', {})
                    if is_static:
                        # Static method: no self arg, just pass the args
                        call_args = []
                        for arg in node.args:
                            v = self.visit(arg)
                            call_args.append(v.llvm if (v.is_object or v.is_dict or v.is_instance) else self._box(v))
                    else:
                        # Classmethod: first arg is self (INSTANCE_PTR), pass null placeholder
                        # The classmethod doesn't use self for anything meaningful
                        call_args = [ir.Constant(INSTANCE_PTR, None)]
                        for arg in node.args:
                            v = self.visit(arg)
                            call_args.append(v.llvm if (v.is_object or v.is_dict or v.is_instance) else self._box(v))
                    if len(call_args) == len(found_func.args):
                        call_args = self._verify_call_args(found_func, call_args)
                        ret = self.builder.call(found_func, call_args)
                        self._check_exc_after_call()
                        return Value(ret, self._llvm_type_to_pytype(found_func.function_type.return_type))

        # 2. Obsługa instancji i obiektów dynamicznych
        if obj.pytype.name in ("OBJECT", "INSTANCE", "DICT", "LIST"):
            z = ir.Constant(I32, 0)
            tag, pay = self._read_slot(obj.llvm)

            # --- A. Metody Stringa (STR_METHODS) ---
            STR_METHODS = ("strip", "lstrip", "rstrip", "upper", "lower", "split", "join", "replace", "startswith", "endswith", "find", "count", "title", "capitalize", "center", "index", "zfill", "isalpha", "isdigit", "isnumeric", "isalnum", "isspace", "isupper", "islower", "format", "rjust", "ljust", "rstrip")
            if mname in STR_METHODS:
                str_bb = self.current_func.append_basic_block("mc.str")
                err_bb = self.current_func.append_basic_block("mc.str.err")
                end_bb = self.current_func.append_basic_block("mc.str.end")
                res_alloca = self.builder.alloca(BOXED_PTR, name="mc_res_str")

                sw = self.builder.switch(tag, err_bb)
                sw.add_case(ir.Constant(I64, Tag.STR), str_bb)

                self.builder.position_at_end(str_bb)
                sptr = self.builder.inttoptr(pay, STR_PTR)
                str_val = Value(sptr, PyType.STR)
                result_str = self._str_method_call(str_val, mname, node)
                self.builder.store(self._box(result_str), res_alloca)
                self.builder.branch(end_bb)

                self.builder.position_at_end(err_bb)
                self.builder.store(ir.Constant(BOXED_PTR, None), res_alloca)
                self.builder.branch(end_bb)

                self.builder.position_at_end(end_bb)
                return Value(self.builder.load(res_alloca), PyType.OBJECT)

            # --- B. Metody Klasy/Instancji ---
            found_func = None
            inferred_class = None
            # Najpierw sprawdź class_name na obiekcie Value (z visit_Name)
            if obj.class_name:
                inferred_class = obj.class_name
            # Potem sprawdź VarInfo w symbol table
            if not inferred_class and hasattr(node.func, "value") and hasattr(node.func.value, "id"):
                var_name = node.func.value.id
                try:
                    var_info = self.sym.lookup(var_name)
                    if hasattr(var_info, "class_name") and var_info.class_name:
                        inferred_class = var_info.class_name
                except: pass

            if inferred_class:
                target = f"py_{inferred_class}_{mname}"
                if target in self.functions and isinstance(self.functions[target], ir.Function):
                    found_func = self.functions[target]

            # NAPRAWA: MRO (Method Resolution Order) - szukaj w klasach bazowych
            if found_func is None and inferred_class:
                hierarchy = getattr(self, '_class_hierarchy', {})
                mro_classes = self._get_mro(inferred_class, hierarchy)
                for cls_name in mro_classes:
                    target = f"py_{cls_name}_{mname}"
                    if target in self.functions and isinstance(self.functions[target], ir.Function):
                        found_func = self.functions[target]
                        break

            if found_func is None:
                for fname in self.functions:
                    if fname.startswith("py_") and fname.endswith(f"_{mname}"):
                        found_func = self.functions[fname]
                        break

            if found_func and isinstance(found_func, ir.Function):
                inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR)
                first_arg_type = found_func.args[0].type if found_func.args else BOXED_PTR
                if first_arg_type == INSTANCE_PTR:
                    self_arg = inst_ptr
                elif first_arg_type == DICT_PTR:
                    self_arg = self.builder.load(self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True))
                else:
                    self_arg = obj.llvm

                call_args = [self_arg]
                for arg in node.args:
                    v = self.visit(arg)
                    call_args.append(v.llvm if (v.is_object or v.is_dict or v.is_instance) else self._box(v))

                if len(call_args) == len(found_func.args):
                    call_args = self._verify_call_args(found_func, call_args)
                    ret = self.builder.call(found_func, call_args)
                    self._check_exc_after_call()
                    return Value(ret, self._llvm_type_to_pytype(found_func.function_type.return_type))

            # --- C. Inne metody wbudowane (np. append) ---
            if mname == "append":
                lst_bb = self.current_func.append_basic_block("mc.lst")
                err_bb_append = self.current_func.append_basic_block("mc.lst.err")
                end_bb_append = self.current_func.append_basic_block("mc.lst.end")
                res_append = self.builder.alloca(BOXED_PTR, name="mc_res_app")

                sw = self.builder.switch(tag, err_bb_append)
                sw.add_case(ir.Constant(I64, Tag.LIST), lst_bb)

                self.builder.position_at_end(lst_bb)
                lptr = self.builder.inttoptr(pay, LIST_PTR)
                self.list_append(Value(lptr, PyType.LIST), self.visit(node.args[0]))
                self.builder.store(self._box(Value(ir.Constant(I64, 0), PyType.NONE)), res_append)
                self.builder.branch(end_bb_append)

                self.builder.position_at_end(err_bb_append)
                self.builder.store(ir.Constant(BOXED_PTR, None), res_append)
                self.builder.branch(end_bb_append)

                self.builder.position_at_end(end_bb_append)
                return Value(self.builder.load(res_append), PyType.OBJECT)

        raise CompileError(f"Metoda '{mname}' na {obj.pytype} nieobsługiwana.", node)

    # ══════════════════════════════════════════════════════════════════
    #  Call-arg type coercion helpers
    #  These ensure LLVM arg types match the callee's signature,
    #  preventing TypeError from llvmlite at builder.call().
    # ══════════════════════════════════════════════════════════════════

    def _coerce_call_arg(self, v: Value, expected_type) -> ir.Value:
        """Coerce a Value's LLVM value to match the expected LLVM parameter type.

        Handles the common conversions:
        - INSTANCE_PTR → BOXED_PTR  (via _box)
        - BOXED_PTR → INSTANCE_PTR  (via _read_slot + inttoptr)
        - Raw primitive → BOXED_PTR  (via _box)
        - Same type → pass through
        """
        actual_type = v.llvm.type

        # Types already match — fast path
        if actual_type == expected_type:
            return v.llvm

        # INSTANCE_PTR → BOXED_PTR
        if expected_type == BOXED_PTR and actual_type == INSTANCE_PTR:
            return self._box(v)

        # BOXED_PTR → INSTANCE_PTR (unbox an instance)
        if expected_type == INSTANCE_PTR and actual_type == BOXED_PTR:
            tag, pay = self._read_slot(v.llvm)
            return self.builder.inttoptr(pay, INSTANCE_PTR, "unbox_inst")

        # BOXED_PTR → DICT_PTR (extract dict from boxed instance)
        if expected_type == DICT_PTR and actual_type == BOXED_PTR:
            tag, pay = self._read_slot(v.llvm)
            inst_ptr = self.builder.inttoptr(pay, INSTANCE_PTR, "unbox_inst4dict")
            z = ir.Constant(I32, 0)
            return self.builder.load(
                self.builder.gep(inst_ptr, [z, ir.Constant(I32, 2)], inbounds=True),
                "inst_dict"
            )

        # INSTANCE_PTR → DICT_PTR (GEP field 2)
        if expected_type == DICT_PTR and actual_type == INSTANCE_PTR:
            z = ir.Constant(I32, 0)
            return self.builder.load(
                self.builder.gep(v.llvm, [z, ir.Constant(I32, 2)], inbounds=True),
                "inst_dict"
            )

        # Raw value → BOXED_PTR (box it)
        if expected_type == BOXED_PTR:
            return self._box(v)

        # BOXED_PTR → raw (unbox to expected type)
        if actual_type == BOXED_PTR:
            tag, pay = self._read_slot(v.llvm)
            return self._unbox_raw(pay, expected_type)

        # Pointer-to-pointer: use bitcast
        if isinstance(expected_type, ir.PointerType) and isinstance(actual_type, ir.PointerType):
            return self.builder.bitcast(v.llvm, expected_type, "coerce_bitcast")

        # Fallback: try direct cast
        return v.llvm

    def _unbox_raw(self, payload: ir.Value, target_type) -> ir.Value:
        """Convert a raw i64 payload to the target LLVM type."""
        if target_type == I64:
            return payload
        if target_type == F64:
            return self.builder.bitcast(payload, F64, "unbox_f64")
        if target_type == I32:
            return self.builder.trunc(payload, I32, "unbox_i32")
        if target_type == I1:
            return self.builder.trunc(payload, I1, "unbox_i1")
        if isinstance(target_type, ir.PointerType):
            return self.builder.inttoptr(payload, target_type, "unbox_ptr")
        return payload

    def _verify_call_args(self, func, call_args: list) -> list:
        """Verify and fix arg types to match the function signature.

        This is a safety net: if any arg type doesn't match the expected
        parameter type, we apply coercion to fix it. This prevents
        TypeError from llvmlite's builder.call().
        """
        fixed_args = []
        for i, arg in enumerate(call_args):
            if i < len(func.args):
                expected = func.args[i].type
                actual = arg.type
                if actual != expected:
                    # Need to coerce — wrap in a temporary Value
                    # and use _coerce_call_arg
                    pytype = self._llvm_type_to_pytype(actual)
                    tmp_val = Value(arg, pytype)
                    fixed_args.append(self._coerce_call_arg(tmp_val, expected))
                else:
                    fixed_args.append(arg)
            else:
                fixed_args.append(arg)
        return fixed_args

    # ══════════════════════════════════════════════════════════════════
    #  super().method() — compile-time MRO resolution
    # ══════════════════════════════════════════════════════════════════

    def _super_method_call(self, mname: str, node: ast.Call) -> Value:
        """Obsługa super().method() z MRO resolution.

        Strategia:
        1. Pobierz bieżącą klasę z _class_stack
        2. Znajdź wszystkie klasy pochodne (które mają tę klasę w swoim MRO)
        3. Jeśli istnieją klasy pochodne z tą metodą, generuj runtime
           dispatch na podstawie self.__class__ — wybierz właściwą metodę
           z MRO klasy pochodnej
        4. W przeciwnym razie, użyj prostego compile-time MRO z klas bazowych
        """
        z = ir.Constant(I32, 0)

        # Pobierz informację o bieżącej klasie ze stacku
        if not hasattr(self, "_class_stack") or not self._class_stack:
            raise CompileError("super() musi być użyte wewnątrz metody klasy", node)

        current_class_info = self._class_stack[-1]
        current_class_name = current_class_info.get("name", "")

        hierarchy = getattr(self, '_class_hierarchy', {})

        # Krok 1: Znajdź wszystkie klasy, w których MRO current_class występuje
        # To pozwala nam obsłużyć wielodziedziczenie
        derived_classes = []
        for cls_name, bases in hierarchy.items():
            mro = self._get_mro(cls_name, hierarchy)
            if current_class_name in mro:
                derived_classes.append((cls_name, mro))

        # Krok 2: Dla każdej klasy pochodnej, znajdź następną klasę w MRO
        # po current_class_name, która ma metodę mname
        # Mapa: derived_class_name -> (next_class_name, llvm_function)
        dispatch_map = {}
        for derived_cls, mro in derived_classes:
            # Znajdź pozycję current_class w MRO
            try:
                idx = mro.index(current_class_name)
            except ValueError:
                continue
            # Szukaj następnej klasy w MRO z tą metodą
            for next_cls in mro[idx + 1:]:
                target = f"py_{next_cls}_{mname}"
                if target in self.functions and isinstance(self.functions[target], ir.Function):
                    dispatch_map[derived_cls] = (next_cls, self.functions[target])
                    break

        # Also add the simple case: current_class's own MRO
        current_mro = self._get_mro(current_class_name, hierarchy)
        simple_next = None
        for cls_name in current_mro:
            target = f"py_{cls_name}_{mname}"
            if target in self.functions and isinstance(self.functions[target], ir.Function):
                simple_next = (cls_name, self.functions[target])
                break

        # Krok 3: Jeśli nie ma klas pochodnych lub dispatch_map jest pusta,
        # użyj prostego compile-time resolve
        if not dispatch_map and simple_next:
            return self._super_call_direct(simple_next[0], simple_next[1], node)

        # Krok 4: Jeśli jest tylko jedna klasa pochodna, użyj jej MRO
        if len(dispatch_map) == 1:
            derived_cls = list(dispatch_map.keys())[0]
            next_cls, func = dispatch_map[derived_cls]
            return self._super_call_direct(next_cls, func, node)

        # Krok 5: Wiele klas pochodnych — generuj runtime dispatch
        # na podstawie self.__class__ z instancji
        return self._super_call_runtime_dispatch(
            current_class_name, mname, dispatch_map, simple_next, node
        )

    def _super_call_direct(self, next_class_name: str, found_func,
                           node: ast.Call) -> Value:
        """Bezpośrednie wywołanie metody z super() — compile-time resolve."""
        z = ir.Constant(I32, 0)

        # Pobierz self z symbol table
        self_val = None
        try:
            self_info = self.sym.lookup("self")
            self_val = self.builder.load(self_info.alloca, "super_self")
        except CompileError:
            raise CompileError("super() wymaga zmiennej 'self'", node)

        # Buduj argumenty: self + argumenty użytkownika
        first_arg_type = found_func.args[0].type if found_func.args else BOXED_PTR
        call_args = self._prepare_super_self_args(self_val, first_arg_type, node)

        if len(call_args) == len(found_func.args):
            call_args = self._verify_call_args(found_func, call_args)
            ret = self.builder.call(found_func, call_args,
                                    name=f"super_{next_class_name}_{node.func.attr}")
            self._check_exc_after_call()
            return Value(ret, self._llvm_type_to_pytype(found_func.function_type.return_type))
        else:
            raise CompileError(
                f"super().{node.func.attr}() — zła liczba argumentów: "
                f"{len(call_args)} vs {len(found_func.args)}", node
            )

    def _super_call_runtime_dispatch(self, current_class_name: str, mname: str,
                                      dispatch_map: dict, simple_next,
                                      node: ast.Call) -> Value:
        """Runtime dispatch dla super() z wielodziedziczeniem.

        Generuje kod który:
        1. Czyta self.__class__ z dict instancji
        2. Sprawdza po kolei każdy case w dispatch_map
        3. Wywołuje odpowiednią metodę
        """
        z = ir.Constant(I32, 0)

        # Pobierz self
        self_val = None
        try:
            self_info = self.sym.lookup("self")
            self_val = self.builder.load(self_info.alloca, "super_self")
        except CompileError:
            raise CompileError("super() wymaga zmiennej 'self'", node)

        # Wyciągnij dict z instancji (jeśli to INSTANCE_PTR)
        if self_val.type == INSTANCE_PTR:
            attrs_ptr = self.builder.load(
                self.builder.gep(self_val, [z, ir.Constant(I32, 2)], inbounds=True),
                "super_attrs"
            )
        else:
            attrs_ptr = self_val

        # Czytaj self.__class__ z dict
        class_key = self.create_string("__class__")
        class_val = self.dict_getitem(Value(attrs_ptr, PyType.DICT), class_key)
        # class_val to boxed string — wyciągnij payload
        cls_tag, cls_pay = self._read_slot(class_val.llvm)

        # Porównaj z każdą klasą pochodną
        # Generuj łańcuch if-else
        result_alloca = self.builder.alloca(BOXED_PTR, name="super_result")

        # Klasa bazowa — fallback
        if simple_next:
            fallback_class, fallback_func = simple_next
        else:
            fallback_func = None

        # Dla każdej klasy pochodnej, porównaj self.__class__ z jej nazwą
        # Używamy hash comparison: hash(class_name) == hash(stored_name)
        # To jest uproszczenie — prawdziwe porównanie stringów byłoby lepsze
        # ale hash wystarcza dla unikalnych nazw klas

        # Pobierz hash przechowanego __class__ — zrób to przez runtime
        # Na razie, dla uproszczenia, generuj prosty if-else na hashach

        # Utwórz bloki dla każdego case + fallback + end
        case_blocks = []
        for derived_cls in dispatch_map:
            case_bb = self.current_func.append_basic_block(f"super.{derived_cls}")
            case_blocks.append((derived_cls, case_bb))
        fallback_bb = self.current_func.append_basic_block("super.fallback")
        end_bb = self.current_func.append_basic_block("super.end")

        # Porównuj hash self.__class__ z hashem każdej klasy pochodnej
        # Dla uproszczenia, konwertujemy class_val (string) do hash
        # za pomocą runtime function
        # NAPRAWA: Prostsze podejście — porównaj payload (string pointer)
        # z pre-zapisanym stringiem klasy

        # Jeszcze prościej: użyj static string comparison
        # Zrób switch na podstawie hash(name)
        class_name_hash = self._string_hash_runtime(class_val)

        # Sprawdź każdy case
        first_cond = None
        for i, (derived_cls, case_bb) in enumerate(case_blocks):
            expected_hash = ir.Constant(I64, self._djb2_hash(derived_cls))
            this_cond = self.builder.icmp_signed("==", class_name_hash, expected_hash,
                                                  name=f"super_is_{derived_cls}")

            if i == 0:
                # Pierwszy check
                next_check_bb = self.current_func.append_basic_block(f"super.check_{i+1}")
                self.builder.cbranch(this_cond, case_bb, next_check_bb)
                self.builder.position_at_end(next_check_bb)
            elif i < len(case_blocks) - 1:
                # Środkowy check
                next_check_bb = self.current_func.append_basic_block(f"super.check_{i+1}")
                self.builder.cbranch(this_cond, case_bb, next_check_bb)
                self.builder.position_at_end(next_check_bb)
            else:
                # Ostatni check — jak nie match'uje, idź do fallback
                self.builder.cbranch(this_cond, case_bb, fallback_bb)

        # Compile each case block
        for derived_cls, case_bb in case_blocks:
            self.builder.position_at_end(case_bb)
            next_cls, func = dispatch_map[derived_cls]
            first_arg_type = func.args[0].type if func.args else BOXED_PTR
            call_args = self._prepare_super_self_args(self_val, first_arg_type, node)
            if len(call_args) == len(func.args):
                ret = self.builder.call(func, call_args, name=f"super_{next_cls}_{mname}")
                self._check_exc_after_call()
                ret_type = func.function_type.return_type
                if ret_type == VOID:
                    self.builder.store(self._box(Value(ir.Constant(I64, 0), PyType.NONE)), result_alloca)
                else:
                    self.builder.store(self._box(Value(ret, self._llvm_type_to_pytype(ret_type))), result_alloca)
            self.builder.branch(end_bb)

        # Fallback block
        self.builder.position_at_end(fallback_bb)
        if fallback_func:
            first_arg_type = fallback_func.args[0].type if fallback_func.args else BOXED_PTR
            call_args = self._prepare_super_self_args(self_val, first_arg_type, node)
            if len(call_args) == len(fallback_func.args):
                ret = self.builder.call(fallback_func, call_args, name=f"super_fallback_{mname}")
                self._check_exc_after_call()
                ret_type = fallback_func.function_type.return_type
                if ret_type == VOID:
                    self.builder.store(self._box(Value(ir.Constant(I64, 0), PyType.NONE)), result_alloca)
                else:
                    self.builder.store(self._box(Value(ret, self._llvm_type_to_pytype(ret_type))), result_alloca)
        else:
            self.builder.store(ir.Constant(BOXED_PTR, None), result_alloca)
        self.builder.branch(end_bb)

        self.builder.position_at_end(end_bb)
        return Value(self.builder.load(result_alloca, "super_res"), PyType.OBJECT)

    def _prepare_super_self_args(self, self_val, first_arg_type, node: ast.Call) -> list:
        """Przygotuj argumenty dla wywołania super().method()."""
        z = ir.Constant(I32, 0)
        if first_arg_type == INSTANCE_PTR:
            if self_val.type == INSTANCE_PTR:
                call_args = [self_val]
            elif self_val.type == DICT_PTR:
                try:
                    self_info = self.sym.lookup("self")
                    if self_info.llvm_type == INSTANCE_PTR:
                        call_args = [self.builder.load(self_info.alloca, "super_self_inst")]
                    else:
                        call_args = [self_val]
                except Exception:
                    call_args = [self_val]
            else:
                call_args = [self_val]
        elif first_arg_type == DICT_PTR:
            if self_val.type == DICT_PTR:
                call_args = [self_val]
            elif self_val.type == INSTANCE_PTR:
                call_args = [self.builder.load(
                    self.builder.gep(self_val, [z, ir.Constant(I32, 2)], inbounds=True)
                )]
            else:
                call_args = [self_val]
        else:
            call_args = [self_val]

        for arg in node.args:
            v = self.visit(arg)
            call_args.append(v.llvm if (v.is_object or v.is_dict or v.is_instance) else self._box(v))

        return call_args

    def _string_hash_runtime(self, str_val: Value) -> ir.Value:
        """Oblicz hash stringa w runtime (dla porównania nazw klas).

        Czyta dane stringa z STR_TY i oblicza prosty hash djb2 na
        podstawie bajtów. Zwraca i64 hash do porównania z compile-time
        hashem.
        """
        tag, pay = self._read_slot(str_val.llvm)
        # pay = ptrtoint(STR_PTR, i64) — wskaźnik do struktury stringa
        str_ptr = self.builder.inttoptr(pay, STR_PTR, "class_str_ptr")
        z = ir.Constant(I32, 0)

        # STR_TY: { GC_HEADER, len:i64, cap:i64, data:i8* }
        # Odczytaj długość i wskaźnik danych
        str_len = self.builder.load(
            self.builder.gep(str_ptr, [z, ir.Constant(I32, 1)], inbounds=True),
            name="class_str_len"
        )
        str_data = self.builder.load(
            self.builder.gep(str_ptr, [z, ir.Constant(I32, 3)], inbounds=True),
            name="class_str_data"
        )

        # Oblicz djb2 hash: hash = hash * 33 + byte
        # To musi odpowiadać compile-time hash obliczanemu przez
        # self._djb2_hash(class_name)
        hash_val = ir.Constant(I64, 5381)
        # Prosta pętla hashująca — inline w LLVM IR
        loop_bb = self.current_func.append_basic_block("hash.loop")
        body_bb = self.current_func.append_basic_block("hash.body")
        done_bb = self.current_func.append_basic_block("hash.done")

        idx_a = self.builder.alloca(I64, name="hash_idx")
        hash_a = self.builder.alloca(I64, name="hash_val")
        self.builder.store(ir.Constant(I64, 0), idx_a)
        self.builder.store(ir.Constant(I64, 5381), hash_a)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(loop_bb)
        ci = self.builder.load(idx_a, "hi")
        cv = self.builder.load(hash_a, "hv")
        self.builder.cbranch(
            self.builder.icmp_signed("<", ci, str_len),
            body_bb, done_bb
        )

        self.builder.position_at_end(body_bb)
        byte_ptr = self.builder.gep(str_data, [ci], inbounds=True)
        byte_val = self.builder.load(byte_ptr, "hb")
        byte_i64 = self.builder.zext(byte_val, I64)
        new_hash = self.builder.add(
            self.builder.mul(cv, ir.Constant(I64, 33)),
            byte_i64,
            name="nh"
        )
        self.builder.store(new_hash, hash_a)
        self.builder.store(self.builder.add(ci, ir.Constant(I64, 1)), idx_a)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(done_bb)
        return self.builder.load(hash_a, "class_hash")

    @staticmethod
    def _djb2_hash(s: str) -> int:
        """DJB2 hash algorithm — musi odpowiadać _string_hash_runtime."""
        h = 5381
        for c in s:
            h = h * 33 + ord(c)
        return h & 0xFFFFFFFFFFFFFFFF

    # ──────────────────────────────────────────────────────────────
    #  Pure-Python module function calls
    # ──────────────────────────────────────────────────────────────

    def _pure_python_module_call(
        self, module_name: str, method_name: str, node: ast.Call,
    ) -> Value:
        """Call a function exported by a pure-Python module.

        Unlike FFI calls, pure-Python module functions use pylow's
        internal calling convention (``BoxedValue*`` args and return
        values), so no C-style type marshaling is needed.  Arguments
        are simply boxed and passed directly.

        The function is looked up in three ways, in order:
          1. Mangled name  (``{module}_{method}``)  — for COMPILED_UNIT
          2. Python name   (``method``)             — for INLINE
          3. Full mangled  (``{module}__{method}``) — alternative mangling

        Args:
            module_name: The pure-Python module name (e.g. ``"parse"``).
            method_name: The function name within the module.
            node: The AST Call node.

        Returns:
            A Value representing the call result.
        """
        # ── Resolve the function ──
        safe_mod = module_name.replace(".", "_")
        mangled_alt1 = f"{safe_mod}_{method_name}"   # e.g. parse_func
        mangled_alt2 = f"{safe_mod}__{method_name}"  # e.g. parse__func

        fn = (
            self.functions.get(mangled_alt2)
            or self.functions.get(mangled_alt1)
            or self.functions.get(method_name)
        )
        if fn is None:
            raise CompileError(
                f"Moduł pure Python '{module_name}' nie eksportuje "
                f"funkcji '{method_name}'.", node
            )

        # ── Build call arguments ──
        fty = fn.function_type
        expected_args = list(fty.args)

        call_args = []
        for i, arg_node in enumerate(node.args):
            v = self.visit(arg_node)

            if i < len(expected_args):
                target_type = expected_args[i]
            else:
                # Extra args default to BOXED_PTR
                target_type = BOXED_PTR

            call_args.append(self._coerce_call_arg(v, target_type))

        # Pad with boxed None if the function expects more args than provided
        while len(call_args) < len(expected_args):
            none_boxed = self._box(Value(ir.Constant(I64, 0), PyType.NONE))
            call_args.append(none_boxed)

        # Truncate if we have too many args (shouldn't happen, but safety)
        if len(call_args) > len(expected_args):
            call_args = call_args[:len(expected_args)]

        # ── Emit the call ──
        ret = self.builder.call(
            fn, call_args,
            name=f"purepy_{safe_mod}_{method_name}_ret",
        )

        # ── Determine return type ──
        ret_type = fty.return_type
        if ret_type == VOID:
            return Value(ir.Constant(I64, 0), PyType.NONE)
        elif ret_type == BOXED_PTR:
            return Value(ret, PyType.OBJECT)
        elif ret_type == I64 or ret_type == I32:
            return Value(ret, PyType.INT)
        elif ret_type == F64:
            return Value(ret, PyType.FLOAT)
        elif ret_type == I8P:
            return Value(ret, PyType.STR)
        elif ret_type == STR_PTR:
            return Value(ret, PyType.STR)
        elif ret_type == LIST_PTR:
            return Value(ret, PyType.LIST)
        elif ret_type == DICT_PTR:
            return Value(ret, PyType.DICT)
        elif ret_type == INSTANCE_PTR:
            return Value(ret, PyType.INSTANCE)
        elif ret_type == I1:
            return Value(self.builder.zext(ret, I64), PyType.BOOL)
        else:
            # Fallback: try to infer from LLVM type name
            ret_type_str = str(ret_type)
            if 'str' in ret_type_str.lower():
                return Value(ret, PyType.STR)
            elif 'list' in ret_type_str.lower():
                return Value(ret, PyType.LIST)
            elif 'dict' in ret_type_str.lower():
                return Value(ret, PyType.DICT)
            # Default: treat as boxed object — but box it correctly
            # by converting to BOXED_PTR first if needed
            if ret_type != BOXED_PTR:
                boxed = self._box(Value(ret, PyType.OBJECT))
                return Value(boxed, PyType.OBJECT)
            return Value(ret, PyType.OBJECT)

    # ──────────────────────────────────────────────────────────────
    #  FFI: Zero-overhead native .so function calls
    # ──────────────────────────────────────────────────────────────

    def _ffi_call(self, symbol_name: str, node: ast.Call, ffi_fn: "ir.Function" = None, module_name: str = None) -> Value:
        """Generate a direct builder.call() to an FFI symbol.

        This method performs zero-overhead FFI calls: the symbol is
        declared as ``declare external`` in LLVM IR and called directly
        via builder.call().  No C++ wrapper, no runtime conversion layer.

        Type marshaling is performed at compile time based on the
        signature database: BoxedValue* is bitcast to the appropriate
        C type (i8* for PyObject*, i64 for integers, f64 for doubles).

        **CPython Extension Handling**: When the target function belongs
        to a CPython extension module (detected via module_name), this
        method routes the call through the CPython bridge instead of
        calling directly.  CPython extension functions expect real
        PyObject* values with the correct memory layout — pylow's
        lightweight internal types (BoxedValue, STR_TY etc.) are
        incompatible.  The bridge creates real CPython objects,
        calls the function, and converts the result back.

        Args:
            symbol_name: The FFI symbol name.
            node: The AST Call node.
            ffi_fn: Optional pre-resolved LLVM IR function.
            module_name: Optional FFI module name (for CPython ext detection).

        Returns:
            A Value representing the call result.
        """
        if ffi_fn is None:
            ffi_fn = self._ffi_symbols.get(symbol_name)
        if ffi_fn is None:
            raise CompileError(f"FFI symbol '{symbol_name}' nie jest zadeklarowany.", node)

        # ── CPython extension detection ──
        # If the function belongs to a CPython extension module, we MUST
        # route through the CPython bridge because the C code expects
        # real PyObject* values with proper memory layout.  Passing
        # pylow's internal types directly causes segfaults when the C
        # code dereferences ob_type (which is 0x0 in pylow's structs).
        is_cpython_ext = False
        if module_name and hasattr(self, '_ffi_modules'):
            ffi_mod = self._ffi_modules.get(module_name)
            if ffi_mod and ffi_mod.pyinit_symbol is not None:
                is_cpython_ext = True

        if is_cpython_ext:
            return self._ffi_wrapper_call(symbol_name, node, ffi_fn, module_name)

        # Look up signature from the FFI signature database
        sigdb = getattr(self, '_ffi_sigdb', None)
        if sigdb is None:
            sigdb = FFISignatureDB()

        ret_cat, param_count = sigdb.lookup(symbol_name)

        # Get the function type
        fty = ffi_fn.function_type
        expected_args = list(fty.args)

        # Build call arguments with type marshaling
        call_args = []
        for i, arg_node in enumerate(node.args):
            v = self.visit(arg_node)

            if i < len(expected_args):
                target_type = expected_args[i]
            else:
                # Variadic args default to i8*
                target_type = I8P

            # Marshal: BoxedValue* -> C type
            if target_type == I8P:
                # Opaque pointer — bitcast from BoxedValue* or pass raw
                if v.is_object:
                    # BoxedValue* -> i8*: bitcast the boxed pointer
                    call_args.append(self.builder.bitcast(v.llvm, I8P))
                elif v.is_str:
                    # STR_PTR -> i8*: pass the string data pointer
                    z = ir.Constant(I32, 0)
                    sptr = v.llvm
                    data_ptr = self.builder.load(
                        self.builder.gep(sptr, [z, ir.Constant(I32, 3)], inbounds=True)
                    )
                    call_args.append(data_ptr)
                elif v.is_int:
                    # i64 -> i8*: inttoptr (for pointer-sized values)
                    call_args.append(self.builder.inttoptr(v.llvm, I8P))
                elif v.is_float:
                    # f64 -> i8*: bitcast via alloca
                    alloca = self.builder.alloca(F64)
                    self.builder.store(v.llvm, alloca)
                    call_args.append(self.builder.bitcast(alloca, I8P))
                else:
                    # Generic: box and bitcast
                    boxed = self._box(v)
                    call_args.append(self.builder.bitcast(boxed, I8P))

            elif target_type == I64:
                # Integer parameter
                if v.is_int:
                    call_args.append(v.llvm)
                elif v.is_object:
                    # Unbox: read tag+payload, extract integer payload
                    tag, pay = self._read_slot(v.llvm)
                    call_args.append(pay)
                else:
                    int_val = self._to_int(v)
                    call_args.append(int_val.llvm)

            elif target_type == F64:
                # Double parameter
                if v.is_float:
                    call_args.append(v.llvm)
                elif v.is_object:
                    # Unbox float from boxed
                    tag, pay = self._read_slot(v.llvm)
                    call_args.append(self.builder.bitcast(pay, F64))
                else:
                    float_val = self._to_float(v)
                    call_args.append(float_val.llvm)

            elif target_type == I32:
                # int32 parameter
                if v.is_int:
                    call_args.append(self.builder.trunc(v.llvm, I32))
                elif v.is_bool:
                    call_args.append(self.builder.zext(v.llvm, I32))
                else:
                    int_val = self._to_int(v)
                    call_args.append(self.builder.trunc(int_val.llvm, I32))

            elif target_type == ir.PointerType(BOXED_TY):
                # BoxedValue* parameter — box if not already
                if v.is_object:
                    call_args.append(v.llvm)
                else:
                    call_args.append(self._box(v))

            else:
                # Default: try to cast
                try:
                    v = self._cast_to_llvm(v, target_type, arg_node)
                    call_args.append(v.llvm)
                except Exception:
                    # Last resort: box and bitcast
                    boxed = self._box(v)
                    call_args.append(self.builder.bitcast(boxed, target_type))

        # Emit the direct call
        ret = self.builder.call(ffi_fn, call_args, name=f"ffi_{symbol_name}_ret")

        # Determine return type and wrap in Value
        ret_pytype = sigdb.get_return_pytype(symbol_name)

        if fty.return_type == VOID:
            return Value(ir.Constant(I64, 0), PyType.NONE)
        elif fty.return_type == I8P:
            # Opaque pointer return — could be PyObject*, const char*, void*
            if ret_cat in (RET_PYOBJ, RET_DATA):
                # PyObject* return — treat as BoxedValue* (OBJECT)
                boxed_ptr = self.builder.bitcast(ret, BOXED_PTR)
                return Value(boxed_ptr, PyType.OBJECT)
            elif ret_cat in (RET_CONSTCHAR, RET_CHARPTR):
                # const char* return — create a pylow string from C string
                return self._cstr_to_pylow_str(ret)
            else:
                # void* return — opaque
                return Value(ret, PyType.OBJECT)
        elif fty.return_type == I64 or fty.return_type == I32:
            return Value(ret, PyType.INT)
        elif fty.return_type == F64:
            return Value(ret, PyType.FLOAT)
        else:
            return Value(ret, ret_pytype)

    # ──────────────────────────────────────────────────────────────
    #  FFI: AOT C++ wrapper call — Zero-Python Mode
    # ──────────────────────────────────────────────────────────────

    def _ffi_wrapper_call(self, symbol_name: str, node: ast.Call,
                          ffi_fn: "ir.Function", module_name: str) -> Value:
        """Call a CPython extension function through an AOT-generated C++ wrapper.

        Instead of routing through a runtime CPython bridge (which requires
        libpython), this method calls the AOT-generated wrapper function
        directly.  The wrapper (e.g., pylow_ffi_markupsafe_escape) handles
        PyObject* conversion internally using a mini-runtime that provides
        just enough Py* symbols for the .so to work.

        The LLVM IR emits:
          declare @pylow_ffi_<module>_<function>(<native_arg_types>)

        And at the call site:
          call @pylow_ffi_markupsafe_escape(str_data, str_len, &out_len)

        The wrapper returns a C string (char*) which we convert to a pylow
        string.  The caller is responsible for freeing the C string via
        pylow_ffi_free().

        Args:
            symbol_name: The FFI symbol name (Python method name, e.g., "escape").
            node: The AST Call node.
            ffi_fn: The original LLVM IR Function declaration (unused — we call the wrapper instead).
            module_name: The FFI module name (e.g., "markupsafe").

        Returns:
            A Value representing the call result.
        """
        from src.ffi.generator import FFIManager, WrapperSignature

        # Look up the FFIManager and find the wrapper signature for this symbol
        ffi_manager = getattr(self, '_ffi_manager', None)
        if ffi_manager is None:
            # Create FFIManager from registered modules.
            # Pass the user-facing module name as alias so the FFIManager
            # can map "markupsafe" → "_speedups" (internal PyInit_ name).
            ffi_manager = FFIManager()
            self._ffi_manager = ffi_manager
        # ALWAYS re-register all FFI modules because lazy registration
        # means some modules may not have been registered yet when earlier
        # calls created the FFIManager. This ensures all module aliases
        # and wrapper signatures are available for all modules.
        if hasattr(self, '_ffi_modules'):
            for _mn, _fm in self._ffi_modules.items():
                if _fm.pyinit_symbol is not None and _mn not in ffi_manager._modules:
                    ffi_manager.register_module(_fm, alias=_mn)

        # Find the wrapper signature for this symbol.
        # IMPORTANT: The FFIManager stores signatures under internal module names
        # (e.g., "pylow_ffi__speedups__escape_inner") but the compiler uses
        # user-facing names (e.g., module_name="markupsafe", symbol_name="escape").
        # We use _method_aliases to resolve (module_name, symbol_name) → internal key.
        # If no alias exists, fall back to constructing the key directly.
        sig = None
        method_key = (module_name, symbol_name)
        if method_key in ffi_manager._method_aliases:
            internal_key = ffi_manager._method_aliases[method_key]
            sig = ffi_manager._wrapper_signatures.get(internal_key)

        if sig is None:
            # Try direct construction with both user-facing and internal names
            resolved_name = ffi_manager._resolve_name(module_name)
            for try_name in [module_name, resolved_name]:
                expected_key = f"pylow_ffi_{try_name}_{symbol_name}"
                sig = ffi_manager._wrapper_signatures.get(expected_key)
                if sig:
                    break

        if sig is None:
            # No AOT wrapper for this symbol — fall back to dlsym-based call.
            # We pass use_wrapper=False to avoid infinite loop back to _ffi_wrapper_call.
            return self._ffi_dlsym_call(module_name, symbol_name, node, use_wrapper=False)

        # ── Declare the wrapper function in LLVM IR ──
        wrapper_name = sig.symbol_name  # e.g., "pylow_ffi_markupsafe_escape"
        wrapper_fn = self._ffi_symbols.get(wrapper_name)
        if wrapper_fn is None:
            # Build LLVM IR function type from the wrapper signature
            # Signatures: char* pylow_ffi_<mod>_<func>(const char* data, int64_t len, int64_t* out_len)
            #           char* pylow_ffi_<mod>_<func>(int64_t* out_len)  -- METH_NOARGS
            #           long  pylow_ffi_<mod>_<func>(int64_t arg)       -- int-returning
            param_types = []
            for pt in sig.param_types:
                if pt in ("const char*", "char*"):
                    param_types.append(I8P)
                elif pt == "int64_t":
                    param_types.append(I64)
                elif pt == "int64_t*":
                    param_types.append(I8P)  # i8* for pointer args
                elif pt == "int32_t":
                    param_types.append(I32)
                else:
                    param_types.append(I8P)  # default: opaque pointer

            if sig.return_type in ("char*", "const char*"):
                ret_type = I8P
            elif sig.return_type == "int64_t":
                ret_type = I64
            elif sig.return_type == "int32_t":
                ret_type = I32
            elif sig.return_type == "void":
                ret_type = VOID
            else:
                ret_type = I8P  # default: returns pointer

            wrapper_ty = ir.FunctionType(ret_type, param_types)
            wrapper_fn = ir.Function(self.module, wrapper_ty, name=wrapper_name)
            self._ffi_symbols[wrapper_name] = wrapper_fn

        # ── Build call arguments based on wrapper signature ──
        z = ir.Constant(I32, 0)
        call_args = []

        # METH_NOARGS: no user args, just out_len pointer
        if sig.method_flags == 4:  # METH_NOARGS
            # Signature: char* pylow_ffi_<mod>_<func>(int64_t* out_len)
            out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
            call_args.append(self.builder.bitcast(out_len_alloca, I8P))

        elif sig.method_flags == 8:  # METH_O
            # Signature: char* pylow_ffi_<mod>_<func>(const char* data, int64_t len, int64_t* out_len)
            if node.args:
                arg_val = self.visit(node.args[0])
            else:
                arg_val = self.create_string("")

            if arg_val.is_str:
                # Extract string data pointer and length from pylow's STR_TY
                str_data_ptr = self.builder.load(
                    self.builder.gep(arg_val.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                )
                str_len_val = self.builder.load(
                    self.builder.gep(arg_val.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                )
                out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
            elif arg_val.is_int:
                # Convert int to string, then pass as string arg
                int_as_str = self.val_to_str(arg_val)
                str_data_ptr = self.builder.load(
                    self.builder.gep(int_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                )
                str_len_val = self.builder.load(
                    self.builder.gep(int_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                )
                out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
            else:
                # Dict/List/Object argument — use _pyobj wrapper variant
                if arg_val.is_dict:
                    pyobj_ptr = self._ffi_create_pyobj_dict(arg_val)
                    result = self._ffi_call_pyobj_wrapper(symbol_name, module_name, pyobj_ptr)
                    if result is not None:
                        decref_fn = self.functions.get("pylow_ffi_pyobject_decref")
                        if decref_fn is None:
                            decref_ty = ir.FunctionType(ir.VoidType(), [I8P])
                            decref_fn = ir.Function(self.module, decref_ty, name="Py_DecRef")
                            self.functions["pylow_ffi_pyobject_decref"] = decref_fn
                        self.builder.call(decref_fn, [pyobj_ptr])
                        return result
                    # Fallback: convert to string
                    obj_as_str = self.val_to_str(arg_val)
                    str_data_ptr = self.builder.load(
                        self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                    )
                    str_len_val = self.builder.load(
                        self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                    )
                    out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                    call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
                elif arg_val.is_list:
                    pyobj_ptr = self._ffi_create_pyobj_list(arg_val)
                    result = self._ffi_call_pyobj_wrapper(symbol_name, module_name, pyobj_ptr)
                    if result is not None:
                        decref_fn = self.functions.get("pylow_ffi_pyobject_decref")
                        if decref_fn is None:
                            decref_ty = ir.FunctionType(ir.VoidType(), [I8P])
                            decref_fn = ir.Function(self.module, decref_ty, name="Py_DecRef")
                            self.functions["pylow_ffi_pyobject_decref"] = decref_fn
                        self.builder.call(decref_fn, [pyobj_ptr])
                        return result
                    # Fallback: convert to string
                    obj_as_str = self.val_to_str(arg_val)
                    str_data_ptr = self.builder.load(
                        self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                    )
                    str_len_val = self.builder.load(
                        self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                    )
                    out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                    call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
                else:
                    obj_as_str = self.val_to_str(arg_val)
                    str_data_ptr = self.builder.load(
                        self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                    )
                    str_len_val = self.builder.load(
                        self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                    )
                    out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                    call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]

        else:
            # METH_VARARGS or unknown — fall back to dlsym-based call
            # NOTE: METH_VARARGS (0x0001) and METH_VARARGS|METH_KEYWORDS (0x0003)
            # wrappers now handle tuple creation internally, so we CAN call them
            # with the same signature as METH_O: (data, len, out_len_ptr).
            if sig.method_flags & 0x0001:  # METH_VARARGS bit set
                # Same calling convention as METH_O for the wrapper:
                # char* pylow_ffi_<mod>_<func>(const char* data, int64_t len, int64_t* out_len)
                if node.args:
                    arg_val = self.visit(node.args[0])
                else:
                    arg_val = self.create_string("")

                if arg_val.is_str:
                    str_data_ptr = self.builder.load(
                        self.builder.gep(arg_val.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                    )
                    str_len_val = self.builder.load(
                        self.builder.gep(arg_val.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                    )
                    out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                    call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
                elif arg_val.is_int:
                    int_as_str = self.val_to_str(arg_val)
                    str_data_ptr = self.builder.load(
                        self.builder.gep(int_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                    )
                    str_len_val = self.builder.load(
                        self.builder.gep(int_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                    )
                    out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                    call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
                else:
                    # Dict/List/Object argument — use _pyobj wrapper variant
                    # to create a proper CPython PyObject* instead of converting
                    # to string (which would lose type information and cause
                    # incorrect behavior for ujson.dumps etc.)
                    if arg_val.is_dict:
                        pyobj_ptr = self._ffi_create_pyobj_dict(arg_val)
                        result = self._ffi_call_pyobj_wrapper(symbol_name, module_name, pyobj_ptr)
                        if result is not None:
                            # Also call Py_DecRef on the dict PyObject* to avoid leak
                            decref_fn = self.functions.get("pylow_ffi_pyobject_decref")
                            if decref_fn is None:
                                decref_ty = ir.FunctionType(ir.VoidType(), [I8P])
                                decref_fn = ir.Function(self.module, decref_ty, name="Py_DecRef")
                                self.functions["pylow_ffi_pyobject_decref"] = decref_fn
                            self.builder.call(decref_fn, [pyobj_ptr])
                            return result
                        # Fallback: convert to string
                        obj_as_str = self.val_to_str(arg_val)
                        str_data_ptr = self.builder.load(
                            self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                        )
                        str_len_val = self.builder.load(
                            self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                        )
                        out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                        call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
                    elif arg_val.is_list:
                        pyobj_ptr = self._ffi_create_pyobj_list(arg_val)
                        result = self._ffi_call_pyobj_wrapper(symbol_name, module_name, pyobj_ptr)
                        if result is not None:
                            decref_fn = self.functions.get("pylow_ffi_pyobject_decref")
                            if decref_fn is None:
                                decref_ty = ir.FunctionType(ir.VoidType(), [I8P])
                                decref_fn = ir.Function(self.module, decref_ty, name="Py_DecRef")
                                self.functions["pylow_ffi_pyobject_decref"] = decref_fn
                            self.builder.call(decref_fn, [pyobj_ptr])
                            return result
                        # Fallback: convert to string
                        obj_as_str = self.val_to_str(arg_val)
                        str_data_ptr = self.builder.load(
                            self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                        )
                        str_len_val = self.builder.load(
                            self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                        )
                        out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                        call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
                    else:
                        obj_as_str = self.val_to_str(arg_val)
                        str_data_ptr = self.builder.load(
                            self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                        )
                        str_len_val = self.builder.load(
                            self.builder.gep(obj_as_str.llvm, [z, ir.Constant(I32, 1)], inbounds=True)
                        )
                        out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_out_len")
                        call_args = [str_data_ptr, str_len_val, self.builder.bitcast(out_len_alloca, I8P)]
            else:
                return self._ffi_dlsym_call(module_name, symbol_name, node)

        # ── Call the wrapper function ──
        ret = self.builder.call(wrapper_fn, call_args, name=f"ffi_{symbol_name}_wrapper_ret")

        # ── Process the return value ──
        if sig.return_type in ("char*", "const char*"):
            # String return — convert C string to pylow string
            # Check for NULL return (error)
            ret_ok = self.builder.icmp_unsigned(
                "!=", ret, ir.Constant(I8P, None),
                name=f"ffi_{symbol_name}_wrapper_ok"
            )
            fail_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_w_fail")
            ok_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_w_ok")
            merge_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_w_merge")
            self.builder.cbranch(ret_ok, ok_bb, fail_bb)

            # Failed — return empty string
            self.builder.position_at_start(fail_bb)
            empty_str = self.create_string("")
            self.builder.branch(merge_bb)

            # Success — convert C string to pylow string
            self.builder.position_at_start(ok_bb)
            if 'out_len_alloca' in locals():
                out_len = self.builder.load(out_len_alloca, name=f"ffi_{symbol_name}_result_len")
                result_str = self._cstr_and_len_to_pylow_str(ret, out_len)
            else:
                result_str = self._cstr_to_pylow_str(ret)

            # Free the C string allocated by the wrapper
            free_fn = self.functions.get("pylow_ffi_free")
            if free_fn is None:
                free_ty = ir.FunctionType(ir.VoidType(), [I8P])
                free_fn = ir.Function(self.module, free_ty, name="pylow_ffi_free")
                self.functions["pylow_ffi_free"] = free_fn
            self.builder.call(free_fn, [ret])
            self.builder.branch(merge_bb)

            # Merge
            self.builder.position_at_start(merge_bb)
            result_ptr = self.builder.phi(STR_PTR, f"ffi_{symbol_name}_result")
            result_ptr.add_incoming(empty_str.llvm, fail_bb)
            result_ptr.add_incoming(result_str.llvm, ok_bb)
            return Value(result_ptr, PyType.STR)

        elif sig.return_type == "int64_t":
            return Value(ret, PyType.INT)
        elif sig.return_type == "int32_t":
            return Value(self.builder.sext(ret, I64), PyType.INT)
        elif sig.return_type == "void":
            return Value(ir.Constant(I64, 0), PyType.NONE)
        else:
            # Default: treat as opaque pointer / object
            return Value(ret, PyType.OBJECT)

    # ──────────────────────────────────────────────────────────────
    #  FFI: Object Bridge — pylow→CPython PyObject* conversion
    # ──────────────────────────────────────────────────────────────

    def _ffi_create_pyobj_dict(self, dict_val: Value) -> ir.Value:
        """Create a CPython PyObject* dict from a pylow dict value.

        Uses pylow_ffi_create_dict() from the mini-runtime, which accepts
        null-separated key and value strings plus type codes for each value.

        Type codes: 0=string, 1=int, 2=float, 3=bool, 4=None

        Returns an i8* (PyObject*) that the _pyobj wrapper variant can use.
        """
        # Declare pylow_ffi_create_dict if not already declared
        create_dict_fn = self.functions.get("pylow_ffi_create_dict")
        if create_dict_fn is None:
            create_dict_ty = ir.FunctionType(I8P, [I8P, I64, I8P, I64, I8P, I64])
            create_dict_fn = ir.Function(self.module, create_dict_ty, name="pylow_ffi_create_dict")
            self.functions["pylow_ffi_create_dict"] = create_dict_fn

        # Ensure __py2llvm_dict_get_internal is available for value lookups
        self._ensure_dict_funcs()
        dict_get_fn = self.functions["__py2llvm_dict_get_internal"]

        # Declare helper functions we'll need
        strlen_fn = self.functions.get("strlen")
        if strlen_fn is None:
            strlen_ty = ir.FunctionType(I64, [I8P])
            strlen_fn = ir.Function(self.module, strlen_ty, name="strlen")
            self.functions["strlen"] = strlen_fn

        memcpy_fn = self.functions.get("llvm.memcpy.p0.p0.i64")
        if memcpy_fn is None:
            memcpy_ty = ir.FunctionType(ir.VoidType(), [I8P, I8P, I64, I1])
            memcpy_fn = ir.Function(self.module, memcpy_ty, name="llvm.memcpy.p0.p0.i64")
            self.functions["llvm.memcpy.p0.p0.i64"] = memcpy_fn

        strcpy_fn = self.functions.get("strcpy")
        if strcpy_fn is None:
            strcpy_ty = ir.FunctionType(I8P, [I8P, I8P])
            strcpy_fn = ir.Function(self.module, strcpy_ty, name="strcpy")
            self.functions["strcpy"] = strcpy_fn

        # Read the pylow dict's structure: DICT_TY = {GC_HEADER, size, cap, entries, ordered_keys}
        dct = dict_val.llvm
        z = ir.Constant(I32, 0)

        # Get dict size
        dict_size = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 1)], inbounds=True),
            name="pyobj_dict_size"
        )

        # Get ordered_keys list: LIST_TY = {GC_HEADER, size, cap, data_ptr}
        ordered_keys_ptr = self.builder.load(
            self.builder.gep(dct, [z, ir.Constant(I32, 4)], inbounds=True),
            name="pyobj_dict_ordered_keys"
        )
        # Load the data pointer from ordered_keys list (field 3 of LIST_TY)
        # This gives BOXED_PTR = BOXED_TY* pointing to array of inline BOXED_TY entries
        list_data = self.builder.load(
            self.builder.gep(ordered_keys_ptr, [z, ir.Constant(I32, 3)], inbounds=True),
            name="pyobj_dict_list_data"
        )

        # Allocate buffers for keys, vals, and types
        buf_size = self.builder.mul(dict_size, ir.Constant(I64, 256), name="pyobj_buf_size")

        keys_buf = self.builder.call(self._malloc, [buf_size], name="pyobj_keys_buf")
        vals_buf = self.builder.call(self._malloc, [buf_size], name="pyobj_vals_buf")
        types_buf = self.builder.call(
            self._malloc,
            [self.builder.mul(dict_size, ir.Constant(I64, 8))],
            name="pyobj_types_buf"
        )

        # Key and value write position trackers
        keys_pos = self.builder.alloca(I64, name="pyobj_keys_pos")
        vals_pos = self.builder.alloca(I64, name="pyobj_vals_pos")
        self.builder.store(ir.Constant(I64, 0), keys_pos)
        self.builder.store(ir.Constant(I64, 0), vals_pos)

        # Alloca for dict_get results (tag + payload of the value)
        val_tag_alloca = self.builder.alloca(I64, name="pyobj_val_tag")
        val_pay_alloca = self.builder.alloca(I64, name="pyobj_val_pay")

        # Loop: for i in range(dict_size)
        i_var = self.builder.alloca(I64, name="pyobj_dict_i")
        self.builder.store(ir.Constant(I64, 0), i_var)

        loop_cond = self.current_func.append_basic_block("pyobj_dict_loop_cond")
        loop_body = self.current_func.append_basic_block("pyobj_dict_loop_body")
        loop_end = self.current_func.append_basic_block("pyobj_dict_loop_end")

        self.builder.branch(loop_cond)
        self.builder.position_at_start(loop_cond)

        i_val = self.builder.load(i_var, name="pyobj_i")
        cond = self.builder.icmp_signed("<", i_val, dict_size, name="pyobj_loop_cond")
        self.builder.cbranch(cond, loop_body, loop_end)
        self.builder.position_at_start(loop_body)

        # ── Read the boxed key entry from ordered_keys list ──
        # BOXED_TY = {GC_HEADER_TY, tag:i64, payload:i64}
        # Field 0 = GC_HEADER_TY, Field 1 = tag, Field 2 = payload
        # The list stores BOXED_TY inline, so GEP with index gives BOXED_TY*
        boxed_entry_ptr = self.builder.gep(list_data, [i_val], inbounds=True, name="pyobj_boxed_entry")

        # FIX (Bug 1): Use field index 1 for tag, index 2 for payload
        # (NOT 4 and 5 which were out-of-bounds for the 3-field struct)
        key_tag = self.builder.load(
            self.builder.gep(boxed_entry_ptr, [z, ir.Constant(I32, 1)], inbounds=True),
            name="pyobj_entry_tag"
        )
        key_pay = self.builder.load(
            self.builder.gep(boxed_entry_ptr, [z, ir.Constant(I32, 2)], inbounds=True),
            name="pyobj_entry_payload"
        )

        # ── Serialize key (always a string in Python dicts) ──
        # key_pay = ptrtoint(STR_PTR, i64) → inttoptr back to STR_PTR
        key_str_ptr = self.builder.inttoptr(key_pay, STR_PTR, name="pyobj_key_str_ptr")
        # STR_TY = {GC_HEADER_TY, len:i64, cap:i64, data:i8*}
        # Field 3 is the data pointer (i8*)
        key_data = self.builder.load(
            self.builder.gep(key_str_ptr, [z, ir.Constant(I32, 3)], inbounds=True),
            name="pyobj_key_data"
        )

        # Write key to keys_buf at current position
        current_keys_pos = self.builder.load(keys_pos, name="pyobj_cur_keys_pos")
        key_dest = self.builder.gep(
            keys_buf,
            [current_keys_pos],
            inbounds=True,
            name="pyobj_key_dest"
        )

        key_len = self.builder.call(strlen_fn, [key_data], name="pyobj_key_len")
        # Copy key_len + 1 bytes (including null terminator)
        key_copy_len = self.builder.add(key_len, ir.Constant(I64, 1), name="pyobj_key_copy_len")
        self.builder.call(memcpy_fn, [key_dest, key_data, key_copy_len, ir.Constant(I1, 0)])
        # Advance position: key_len + 1
        new_keys_pos = self.builder.add(key_len, ir.Constant(I64, 1), name="pyobj_new_keys_pos")
        self.builder.store(new_keys_pos, keys_pos)

        # ── Look up value using __py2llvm_dict_get_internal ──
        # void __py2llvm_dict_get_internal(DICT_PTR dct, i64 kt, i64 kp, i64* res_t, i64* res_p)
        self.builder.store(ir.Constant(I64, Tag.NONE), val_tag_alloca)
        self.builder.store(ir.Constant(I64, 0), val_pay_alloca)
        self.builder.call(dict_get_fn, [dct, key_tag, key_pay, val_tag_alloca, val_pay_alloca])

        val_tag = self.builder.load(val_tag_alloca, name="pyobj_val_tag_val")
        val_pay = self.builder.load(val_pay_alloca, name="pyobj_val_pay_val")

        # ── Serialize value based on its tag type ──
        # Type codes for pylow_ffi_create_dict: 0=string, 1=int, 2=float, 3=bool, 4=None
        current_vals_pos = self.builder.load(vals_pos, name="pyobj_cur_vals_pos")
        val_dest = self.builder.gep(
            vals_buf,
            [current_vals_pos],
            inbounds=True,
            name="pyobj_val_dest"
        )

        # Create basic blocks for type dispatch
        # Each type has a check block and a handler block to avoid self-loops
        none_handler_bb = self.current_func.append_basic_block("pyobj_val_none_h")
        check_int_bb = self.current_func.append_basic_block("pyobj_val_chk_int")
        int_handler_bb = self.current_func.append_basic_block("pyobj_val_int_h")
        check_float_bb = self.current_func.append_basic_block("pyobj_val_chk_float")
        float_handler_bb = self.current_func.append_basic_block("pyobj_val_float_h")
        check_bool_bb = self.current_func.append_basic_block("pyobj_val_chk_bool")
        bool_handler_bb = self.current_func.append_basic_block("pyobj_val_bool_h")
        bool_true_bb = self.current_func.append_basic_block("pyobj_bool_true")
        bool_false_bb = self.current_func.append_basic_block("pyobj_bool_false")
        bool_done_bb = self.current_func.append_basic_block("pyobj_bool_done")
        check_str_bb = self.current_func.append_basic_block("pyobj_val_chk_str")
        str_handler_bb = self.current_func.append_basic_block("pyobj_val_str_h")
        val_default_bb = self.current_func.append_basic_block("pyobj_val_default")
        val_merge_bb = self.current_func.append_basic_block("pyobj_val_merge")

        # Chain: none? → int? → float? → bool? → str? → default
        self.builder.cbranch(
            self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.NONE)),
            none_handler_bb, check_int_bb
        )

        # ── NONE ──
        self.builder.position_at_start(none_handler_bb)
        self.builder.call(strcpy_fn, [val_dest, self._str_ptr("None")])
        self.builder.branch(val_merge_bb)

        # ── check INT ──
        self.builder.position_at_start(check_int_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.INT)),
            int_handler_bb, check_float_bb
        )

        # ── INT handler ──
        self.builder.position_at_start(int_handler_bb)
        fmt_int = self._str_ptr("%lld")
        self.builder.call(self._snprintf, [val_dest, ir.Constant(I64, 32), fmt_int, val_pay])
        self.builder.branch(val_merge_bb)

        # ── check FLOAT ──
        self.builder.position_at_start(check_float_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.FLOAT)),
            float_handler_bb, check_bool_bb
        )

        # ── FLOAT handler ──
        self.builder.position_at_start(float_handler_bb)
        fval = self.builder.bitcast(val_pay, F64, name="pyobj_fval")
        fmt_float = self._str_ptr("%.17g")
        self.builder.call(self._snprintf, [val_dest, ir.Constant(I64, 64), fmt_float, fval])
        self.builder.branch(val_merge_bb)

        # ── check BOOL ──
        self.builder.position_at_start(check_bool_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.BOOL)),
            bool_handler_bb, check_str_bb
        )

        # ── BOOL handler ──
        self.builder.position_at_start(bool_handler_bb)
        is_true = self.builder.icmp_signed("!=", val_pay, ir.Constant(I64, 0), name="pyobj_bool_is_true")
        self.builder.cbranch(is_true, bool_true_bb, bool_false_bb)

        self.builder.position_at_start(bool_true_bb)
        self.builder.call(strcpy_fn, [val_dest, self._str_ptr("True")])
        self.builder.branch(bool_done_bb)

        self.builder.position_at_start(bool_false_bb)
        self.builder.call(strcpy_fn, [val_dest, self._str_ptr("False")])
        self.builder.branch(bool_done_bb)

        self.builder.position_at_start(bool_done_bb)
        self.builder.branch(val_merge_bb)

        # ── check STR ──
        self.builder.position_at_start(check_str_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.STR)),
            str_handler_bb, val_default_bb
        )

        # ── STR handler ──
        self.builder.position_at_start(str_handler_bb)
        val_str_ptr = self.builder.inttoptr(val_pay, STR_PTR, name="pyobj_val_str_ptr")
        val_str_data = self.builder.load(
            self.builder.gep(val_str_ptr, [z, ir.Constant(I32, 3)], inbounds=True),
            name="pyobj_val_str_data"
        )
        val_str_len = self.builder.call(strlen_fn, [val_str_data], name="pyobj_val_str_len")
        val_str_copy_len = self.builder.add(val_str_len, ir.Constant(I64, 1))
        self.builder.call(memcpy_fn, [val_dest, val_str_data, val_str_copy_len, ir.Constant(I1, 0)])
        self.builder.branch(val_merge_bb)

        # ── DEFAULT (LIST, DICT, TUPLE, SET, etc.) ──
        self.builder.position_at_start(val_default_bb)
        self.builder.call(strcpy_fn, [val_dest, self._str_ptr("[complex]")])
        self.builder.branch(val_merge_bb)

        # ── MERGE: compute val string length and type code ──
        self.builder.position_at_start(val_merge_bb)
        # The value string is already written at val_dest. We need:
        # 1. The length of what was written (to advance vals_pos)
        # 2. The type code (to store in types_buf)
        val_written_len = self.builder.call(strlen_fn, [val_dest], name="pyobj_val_written_len")
        new_vals_pos = self.builder.add(
            self.builder.add(current_vals_pos, val_written_len),
            ir.Constant(I64, 1),  # +1 for null terminator
            name="pyobj_new_vals_pos"
        )
        self.builder.store(new_vals_pos, vals_pos)

        # Compute type code based on val_tag
        # Map: NONE→4, INT→1, FLOAT→2, BOOL→3, STR→0, default→0
        type_code = self.builder.select(
            self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.NONE)),
            ir.Constant(I64, 4),
            self.builder.select(
                self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.INT)),
                ir.Constant(I64, 1),
                self.builder.select(
                    self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.FLOAT)),
                    ir.Constant(I64, 2),
                    self.builder.select(
                        self.builder.icmp_signed("==", val_tag, ir.Constant(I64, Tag.BOOL)),
                        ir.Constant(I64, 3),
                        ir.Constant(I64, 0),  # STR or default → string
                    )
                )
            )
        )

        # Store type code in types array
        types_entry = self.builder.gep(
            self.builder.bitcast(types_buf, ir.PointerType(I64)),
            [i_val],
            inbounds=True,
            name="pyobj_type_entry"
        )
        self.builder.store(type_code, types_entry)

        # Increment loop counter
        self.builder.store(
            self.builder.add(i_val, ir.Constant(I64, 1)),
            i_var
        )
        self.builder.branch(loop_cond)
        self.builder.position_at_start(loop_end)

        # Get final buffer sizes
        keys_len = self.builder.load(keys_pos, name="pyobj_keys_final_len")
        vals_len = self.builder.load(vals_pos, name="pyobj_vals_final_len")

        # Call pylow_ffi_create_dict
        pyobj_dict = self.builder.call(
            create_dict_fn,
            [keys_buf, keys_len, vals_buf, vals_len, types_buf, dict_size],
            name="pyobj_dict_result"
        )

        # Free temporary buffers (the dict now owns the PyObjects)
        free_fn = self.functions.get("free")
        if free_fn is None:
            free_ty = ir.FunctionType(ir.VoidType(), [I8P])
            free_fn = ir.Function(self.module, free_ty, name="free")
            self.functions["free"] = free_fn
        self.builder.call(free_fn, [keys_buf])
        self.builder.call(free_fn, [vals_buf])
        self.builder.call(free_fn, [types_buf])

        return pyobj_dict

    def _ffi_create_pyobj_list(self, list_val: Value) -> ir.Value:
        """Create a CPython PyObject* list from a pylow list value.

        Uses pylow_ffi_create_list() from the mini-runtime.
        Iterates over the pylow list's BOXED_TY entries and serializes
        each element as a null-separated string with type codes.

        Type codes: 0=string, 1=int, 2=float, 3=bool, 4=None

        Returns an i8* (PyObject*) that the _pyobj wrapper variant can use.
        """
        # Declare pylow_ffi_create_list if not already declared
        create_list_fn = self.functions.get("pylow_ffi_create_list")
        if create_list_fn is None:
            create_list_ty = ir.FunctionType(I8P, [I8P, I64, I8P, I64])
            create_list_fn = ir.Function(self.module, create_list_ty, name="pylow_ffi_create_list")
            self.functions["pylow_ffi_create_list"] = create_list_fn

        # Helper functions
        strlen_fn = self.functions.get("strlen")
        if strlen_fn is None:
            strlen_ty = ir.FunctionType(I64, [I8P])
            strlen_fn = ir.Function(self.module, strlen_ty, name="strlen")
            self.functions["strlen"] = strlen_fn

        memcpy_fn = self.functions.get("llvm.memcpy.p0.p0.i64")
        if memcpy_fn is None:
            memcpy_ty = ir.FunctionType(ir.VoidType(), [I8P, I8P, I64, I1])
            memcpy_fn = ir.Function(self.module, memcpy_ty, name="llvm.memcpy.p0.p0.i64")
            self.functions["llvm.memcpy.p0.p0.i64"] = memcpy_fn

        strcpy_fn = self.functions.get("strcpy")
        if strcpy_fn is None:
            strcpy_ty = ir.FunctionType(I8P, [I8P, I8P])
            strcpy_fn = ir.Function(self.module, strcpy_ty, name="strcpy")
            self.functions["strcpy"] = strcpy_fn

        lst = list_val.llvm
        z = ir.Constant(I32, 0)

        # Read list size and data pointer
        list_size = self.builder.load(
            self.builder.gep(lst, [z, ir.Constant(I32, 1)], inbounds=True),
            name="pyobj_list_size"
        )
        list_data = self.builder.load(
            self.builder.gep(lst, [z, ir.Constant(I32, 3)], inbounds=True),
            name="pyobj_list_data"
        )

        # Allocate buffers
        buf_size = self.builder.mul(list_size, ir.Constant(I64, 256), name="pyobj_list_buf_size")
        items_buf = self.builder.call(self._malloc, [buf_size], name="pyobj_items_buf")
        types_buf = self.builder.call(
            self._malloc,
            [self.builder.mul(list_size, ir.Constant(I64, 8))],
            name="pyobj_list_types_buf"
        )

        # Position tracker for items buffer
        items_pos = self.builder.alloca(I64, name="pyobj_items_pos")
        self.builder.store(ir.Constant(I64, 0), items_pos)

        # Loop: for i in range(list_size)
        i_var = self.builder.alloca(I64, name="pyobj_list_i")
        self.builder.store(ir.Constant(I64, 0), i_var)

        loop_cond = self.current_func.append_basic_block("pyobj_list_loop_cond")
        loop_body = self.current_func.append_basic_block("pyobj_list_loop_body")
        loop_end = self.current_func.append_basic_block("pyobj_list_loop_end")

        self.builder.branch(loop_cond)
        self.builder.position_at_start(loop_cond)

        i_val = self.builder.load(i_var, name="pyobj_list_i_val")
        cond = self.builder.icmp_signed("<", i_val, list_size, name="pyobj_list_loop_cond")
        self.builder.cbranch(cond, loop_body, loop_end)
        self.builder.position_at_start(loop_body)

        # Read BOXED_TY entry: {GC_HEADER_TY, tag:i64, payload:i64}
        # Field 1 = tag, Field 2 = payload
        boxed_entry_ptr = self.builder.gep(list_data, [i_val], inbounds=True, name="pyobj_list_boxed")
        elem_tag = self.builder.load(
            self.builder.gep(boxed_entry_ptr, [z, ir.Constant(I32, 1)], inbounds=True),
            name="pyobj_list_elem_tag"
        )
        elem_pay = self.builder.load(
            self.builder.gep(boxed_entry_ptr, [z, ir.Constant(I32, 2)], inbounds=True),
            name="pyobj_list_elem_pay"
        )

        # Serialize element based on tag type
        current_items_pos = self.builder.load(items_pos, name="pyobj_cur_items_pos")
        item_dest = self.builder.gep(
            items_buf,
            [current_items_pos],
            inbounds=True,
            name="pyobj_item_dest"
        )

        # Type dispatch blocks
        # Each type has a check block and a handler block to avoid self-loops
        none_handler_bb = self.current_func.append_basic_block("pyobj_list_none_h")
        check_int_bb = self.current_func.append_basic_block("pyobj_list_chk_int")
        int_handler_bb = self.current_func.append_basic_block("pyobj_list_int_h")
        check_float_bb = self.current_func.append_basic_block("pyobj_list_chk_float")
        float_handler_bb = self.current_func.append_basic_block("pyobj_list_float_h")
        check_bool_bb = self.current_func.append_basic_block("pyobj_list_chk_bool")
        bool_handler_bb = self.current_func.append_basic_block("pyobj_list_bool_h")
        bool_true_bb = self.current_func.append_basic_block("pyobj_list_bool_true")
        bool_false_bb = self.current_func.append_basic_block("pyobj_list_bool_false")
        bool_done_bb = self.current_func.append_basic_block("pyobj_list_bool_done")
        check_str_bb = self.current_func.append_basic_block("pyobj_list_chk_str")
        str_handler_bb = self.current_func.append_basic_block("pyobj_list_str_h")
        list_default_bb = self.current_func.append_basic_block("pyobj_list_default")
        list_merge_bb = self.current_func.append_basic_block("pyobj_list_merge")

        # Chain: none? → int? → float? → bool? → str? → default
        self.builder.cbranch(
            self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.NONE)),
            none_handler_bb, check_int_bb
        )

        # NONE
        self.builder.position_at_start(none_handler_bb)
        self.builder.call(strcpy_fn, [item_dest, self._str_ptr("None")])
        self.builder.branch(list_merge_bb)

        # check INT
        self.builder.position_at_start(check_int_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.INT)),
            int_handler_bb, check_float_bb
        )

        # INT handler
        self.builder.position_at_start(int_handler_bb)
        fmt_int = self._str_ptr("%lld")
        self.builder.call(self._snprintf, [item_dest, ir.Constant(I64, 32), fmt_int, elem_pay])
        self.builder.branch(list_merge_bb)

        # check FLOAT
        self.builder.position_at_start(check_float_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.FLOAT)),
            float_handler_bb, check_bool_bb
        )

        # FLOAT handler
        self.builder.position_at_start(float_handler_bb)
        fval = self.builder.bitcast(elem_pay, F64, name="pyobj_list_fval")
        fmt_float = self._str_ptr("%.17g")
        self.builder.call(self._snprintf, [item_dest, ir.Constant(I64, 64), fmt_float, fval])
        self.builder.branch(list_merge_bb)

        # check BOOL
        self.builder.position_at_start(check_bool_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.BOOL)),
            bool_handler_bb, check_str_bb
        )

        # BOOL handler
        self.builder.position_at_start(bool_handler_bb)
        is_true = self.builder.icmp_signed("!=", elem_pay, ir.Constant(I64, 0))
        self.builder.cbranch(is_true, bool_true_bb, bool_false_bb)

        self.builder.position_at_start(bool_true_bb)
        self.builder.call(strcpy_fn, [item_dest, self._str_ptr("True")])
        self.builder.branch(bool_done_bb)

        self.builder.position_at_start(bool_false_bb)
        self.builder.call(strcpy_fn, [item_dest, self._str_ptr("False")])
        self.builder.branch(bool_done_bb)

        self.builder.position_at_start(bool_done_bb)
        self.builder.branch(list_merge_bb)

        # check STR
        self.builder.position_at_start(check_str_bb)
        self.builder.cbranch(
            self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.STR)),
            str_handler_bb, list_default_bb
        )

        # STR handler
        self.builder.position_at_start(str_handler_bb)
        val_str_ptr = self.builder.inttoptr(elem_pay, STR_PTR, name="pyobj_list_str_ptr")
        val_str_data = self.builder.load(
            self.builder.gep(val_str_ptr, [z, ir.Constant(I32, 3)], inbounds=True),
            name="pyobj_list_str_data"
        )
        val_str_len = self.builder.call(strlen_fn, [val_str_data], name="pyobj_list_str_len")
        val_str_copy_len = self.builder.add(val_str_len, ir.Constant(I64, 1))
        self.builder.call(memcpy_fn, [item_dest, val_str_data, val_str_copy_len, ir.Constant(I1, 0)])
        self.builder.branch(list_merge_bb)

        # DEFAULT (LIST, DICT, TUPLE, SET)
        self.builder.position_at_start(list_default_bb)
        self.builder.call(strcpy_fn, [item_dest, self._str_ptr("[complex]")])
        self.builder.branch(list_merge_bb)

        # MERGE
        self.builder.position_at_start(list_merge_bb)
        item_written_len = self.builder.call(strlen_fn, [item_dest], name="pyobj_item_written_len")
        new_items_pos = self.builder.add(
            self.builder.add(current_items_pos, item_written_len),
            ir.Constant(I64, 1),  # +1 for null terminator
            name="pyobj_new_items_pos"
        )
        self.builder.store(new_items_pos, items_pos)

        # Type code: NONE→4, INT→1, FLOAT→2, BOOL→3, STR→0, default→0
        elem_type_code = self.builder.select(
            self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.NONE)),
            ir.Constant(I64, 4),
            self.builder.select(
                self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.INT)),
                ir.Constant(I64, 1),
                self.builder.select(
                    self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.FLOAT)),
                    ir.Constant(I64, 2),
                    self.builder.select(
                        self.builder.icmp_signed("==", elem_tag, ir.Constant(I64, Tag.BOOL)),
                        ir.Constant(I64, 3),
                        ir.Constant(I64, 0),
                    )
                )
            )
        )

        types_entry = self.builder.gep(
            self.builder.bitcast(types_buf, ir.PointerType(I64)),
            [i_val],
            inbounds=True,
            name="pyobj_list_type_entry"
        )
        self.builder.store(elem_type_code, types_entry)

        # Increment loop counter
        self.builder.store(
            self.builder.add(i_val, ir.Constant(I64, 1)),
            i_var
        )
        self.builder.branch(loop_cond)
        self.builder.position_at_start(loop_end)

        # Get final buffer size
        items_len = self.builder.load(items_pos, name="pyobj_items_final_len")

        # Call pylow_ffi_create_list
        pyobj_list = self.builder.call(
            create_list_fn,
            [items_buf, items_len, types_buf, list_size],
            name="pyobj_list_result"
        )

        free_fn = self.functions.get("free")
        if free_fn is None:
            free_ty = ir.FunctionType(ir.VoidType(), [I8P])
            free_fn = ir.Function(self.module, free_ty, name="free")
            self.functions["free"] = free_fn
        self.builder.call(free_fn, [items_buf])
        self.builder.call(free_fn, [types_buf])

        return pyobj_list

    def _ffi_call_pyobj_wrapper(self, symbol_name: str, module_name: str,
                                 pyobj_ptr: ir.Value) -> Value:
        """Call the _pyobj wrapper variant with a pre-built PyObject*.

        Used by _ffi_wrapper_call when the argument is a dict/list/object
        that needs to be a proper CPython PyObject*.

        Args:
            symbol_name: The Python method name (e.g., "dumps").
            module_name: The FFI module name (e.g., "ujson").
            pyobj_ptr: An i8* pointing to a valid PyObject*.

        Returns:
            A Value representing the call result (string).
        """
        from src.ffi.generator import FFIManager, WrapperSignature

        ffi_manager = getattr(self, '_ffi_manager', None)
        if ffi_manager is None:
            ffi_manager = FFIManager()
            self._ffi_manager = ffi_manager

        # Find the _pyobj variant signature
        sig = None
        method_key = (module_name, f"{symbol_name}_pyobj")
        if method_key in ffi_manager._method_aliases:
            internal_key = ffi_manager._method_aliases[method_key]
            sig = ffi_manager._wrapper_signatures.get(internal_key)

        if sig is None:
            resolved_name = ffi_manager._resolve_name(module_name)
            for try_name in [module_name, resolved_name]:
                expected_key = f"pylow_ffi_{try_name}_{symbol_name}_pyobj"
                sig = ffi_manager._wrapper_signatures.get(expected_key)
                if sig:
                    break

        if sig is None:
            # No _pyobj variant — fall back to string conversion
            return None

        # Declare the _pyobj wrapper in LLVM IR
        wrapper_name = sig.symbol_name
        wrapper_fn = self._ffi_symbols.get(wrapper_name)
        if wrapper_fn is None:
            # char* pylow_ffi_<mod>_<func>_pyobj(void* pyobj_ptr, int64_t* out_len)
            wrapper_ty = ir.FunctionType(I8P, [I8P, I8P])
            wrapper_fn = ir.Function(self.module, wrapper_ty, name=wrapper_name)
            self._ffi_symbols[wrapper_name] = wrapper_fn

        # Build call args: (pyobj_ptr, &out_len)
        out_len_alloca = self.builder.alloca(I64, name=f"ffi_{symbol_name}_pyobj_out_len")
        call_args = [pyobj_ptr, self.builder.bitcast(out_len_alloca, I8P)]

        # Call the wrapper
        ret = self.builder.call(wrapper_fn, call_args, name=f"ffi_{symbol_name}_pyobj_ret")

        # Process return value (same as the string return path in _ffi_wrapper_call)
        ret_ok = self.builder.icmp_unsigned(
            "!=", ret, ir.Constant(I8P, None),
            name=f"ffi_{symbol_name}_pyobj_ok"
        )
        fail_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_pyobj_fail")
        ok_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_pyobj_ok")
        merge_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_pyobj_merge")
        self.builder.cbranch(ret_ok, ok_bb, fail_bb)

        # Failed — return empty string
        self.builder.position_at_start(fail_bb)
        empty_str = self.create_string("")
        self.builder.branch(merge_bb)

        # Success — convert C string to pylow string
        self.builder.position_at_start(ok_bb)
        out_len = self.builder.load(out_len_alloca, name=f"ffi_{symbol_name}_pyobj_result_len")
        result_str = self._cstr_and_len_to_pylow_str(ret, out_len)

        # Free the C string allocated by the wrapper
        free_fn = self.functions.get("pylow_ffi_free")
        if free_fn is None:
            free_ty = ir.FunctionType(ir.VoidType(), [I8P])
            free_fn = ir.Function(self.module, free_ty, name="pylow_ffi_free")
            self.functions["pylow_ffi_free"] = free_fn
        self.builder.call(free_fn, [ret])
        self.builder.branch(merge_bb)

        # Merge
        self.builder.position_at_start(merge_bb)
        result_ptr = self.builder.phi(STR_PTR, f"ffi_{symbol_name}_pyobj_result")
        result_ptr.add_incoming(empty_str.llvm, fail_bb)
        result_ptr.add_incoming(result_str.llvm, ok_bb)
        return Value(result_ptr, PyType.STR)

    # ──────────────────────────────────────────────────────────────
    #  FFI: dlsym-based runtime symbol resolution for CPython
    #  extension modules where the symbol is not in ELF exports
    # ──────────────────────────────────────────────────────────────

    def _ffi_dlsym_call(self, module_name: str, symbol_name: str, node: ast.Call, use_wrapper: bool = True) -> Value:
        """Generate a runtime dlsym call for an FFI symbol not found in ELF exports.

        This handles CPython extension modules where the function (e.g., escape)
        is not exported as a dynamic symbol but is accessible via dlsym after
        calling PyInit_* on the .so file.

        The generated code:
        1. Calls dlopen() on the .so file (if not already open)
        2. Calls dlsym() to find the symbol
        3. Calls the symbol as a function pointer

        For CPython extensions, the function signature is assumed to be
        PyObject* func(PyObject* self, PyObject* args) or PyObject* func(PyObject* self, PyObject* arg)
        depending on the METH_* flag.  We default to the most common pattern:
        PyObject* func(PyObject*, PyObject*) which matches METH_O (single arg).

        Args:
            module_name: The FFI module name (e.g., "markupsafe").
            symbol_name: The function name within the module (e.g., "escape").
            node: The AST Call node.

        Returns:
            A Value representing the call result.
        """
        # Find the .so file path for this module
        so_paths = getattr(self, '_ffi_so_paths', [])
        pkg_dir = None
        so_file = None
        for sp in so_paths:
            if os.path.isdir(sp) and (sp.endswith(module_name) or os.path.basename(sp) == module_name):
                pkg_dir = sp
                break
            elif os.path.isfile(sp):
                so_file = sp
                break

        # If it's a package directory, find the first .so inside
        if pkg_dir and not so_file:
            import os as _os
            for root, dirs, files in _os.walk(pkg_dir):
                for f in files:
                    if f.endswith('.so') and not f.endswith('.pyd'):
                        so_file = _os.path.join(root, f)
                        break
                if so_file:
                    break

        if not so_file:
            # Fallback: try to find via _find_ffi_so
            so_path = getattr(self, '_find_ffi_so', lambda x: None)
            if callable(so_path):
                found = so_path(module_name)
                if found:
                    if os.path.isdir(found):
                        for root, dirs, files in os.walk(found):
                            for f in files:
                                if f.endswith('.so'):
                                    so_file = os.path.join(root, f)
                                    break
                            if so_file:
                                break
                    else:
                        so_file = found

        if not so_file:
            raise CompileError(
                f"FFI: Nie można znaleźć pliku .so dla modułu '{module_name}' "
                f"(symbol: '{symbol_name}').", node
            )

        # ── Ensure dlopen/dlsym declarations exist ──
        dlopen_fn = self.functions.get("dlopen")
        if dlopen_fn is None:
            dlopen_fn = self.functions.get("__py2llvm_dlopen")
        dlsym_fn = self.functions.get("dlsym")
        if dlsym_fn is None:
            dlsym_fn = self.functions.get("__py2llvm_dlsym")

        if dlopen_fn is None or dlsym_fn is None:
            raise CompileError(
                f"FFI: dlopen/dlsym nie są zadeklarowane — nie można rozwiązać "
                f"symbolu '{symbol_name}' z '{module_name}' w czasie wykonania.", node
            )

        # ── Create global constants for the .so path and symbol name ──
        so_path_str = so_file + '\0'
        so_path_const = ir.GlobalVariable(
            self.module, ir.ArrayType(I8, len(so_path_str)),
            name=f"__ffi_so_{module_name}"
        )
        so_path_const.global_constant = True
        so_path_const.linkage = "private"
        so_path_const.initializer = ir.Constant(
            ir.ArrayType(I8, len(so_path_str)),
            [ir.Constant(I8, ord(c)) for c in so_path_str]
        )

        # ── Generate dlopen call ──
        # RTLD_NOW (2) | RTLD_GLOBAL (256) = 258
        # RTLD_NOW: resolve all relocations immediately so stub symbols
        #           are available for the .so's Py* imports.
        # RTLD_GLOBAL: make the .so's symbols available for future dlsym
        #              calls and for other shared libraries.
        z = ir.Constant(I32, 0)
        so_path_ptr = self.builder.gep(
            so_path_const, [z, z], inbounds=True
        )
        rtld_flags = ir.Constant(I32, 258)  # RTLD_NOW | RTLD_GLOBAL
        handle = self.builder.call(dlopen_fn, [so_path_ptr, rtld_flags], name=f"ffi_{module_name}_handle")

        # ── NULL check for dlopen ──
        null_ptr = ir.Constant(I8P, None)
        dlopen_ok = self.builder.icmp_unsigned("!=", handle, null_ptr, name=f"ffi_{module_name}_dlopen_ok")

        dlopen_fail_bb = self.current_func.append_basic_block(f"ffi_{module_name}_dlopen_fail")
        dlopen_ok_bb = self.current_func.append_basic_block(f"ffi_{module_name}_dlopen_ok")
        self.builder.cbranch(dlopen_ok, dlopen_ok_bb, dlopen_fail_bb)

        # dlopen failed: print error and exit
        self.builder.position_at_start(dlopen_fail_bb)
        err_msg_str = f"[pylow] FFI: dlopen failed for module '{module_name}'\n\0"
        err_msg_const = ir.GlobalVariable(
            self.module, ir.ArrayType(I8, len(err_msg_str)),
            name=f"__ffi_dlopen_err_{module_name}"
        )
        err_msg_const.global_constant = True
        err_msg_const.linkage = "private"
        err_msg_const.initializer = ir.Constant(
            ir.ArrayType(I8, len(err_msg_str)),
            [ir.Constant(I8, ord(c)) for c in err_msg_str]
        )
        err_msg_ptr = self.builder.gep(err_msg_const, [z, z], inbounds=True)
        self.builder.call(self._printf, [err_msg_ptr])
        exit_fn = self.functions.get("exit") or self.functions.get("sys.exit")
        if exit_fn:
            self.builder.call(exit_fn, [ir.Constant(I32, 1)])
        # exit() never returns — mark as unreachable so LLVM
        # knows this block has a valid terminator.
        self.builder.unreachable()

        self.builder.position_at_start(dlopen_ok_bb)

        # ── Generate dlsym call ──
        # Try multiple symbol name patterns because CPython extension methods
        # are often exported under their C function name, not the Python name.
        # Common patterns: "escape", "markupsafe_escape", "_escape", etc.
        sym_names_to_try = [symbol_name]
        # Add common C naming patterns
        sym_names_to_try.append(f"{module_name}_{symbol_name}")
        sym_names_to_try.append(f"_{module_name}_{symbol_name}")
        sym_names_to_try.append(f"_{symbol_name}")
        # Also try the C symbol names from method_defs if available
        # Search not just for exact name match but also for naming variations
        # (e.g., "escape" → "_escape_inner", "_escape_inner" → "escape_unicode")
        ffi_mod = self._ffi_modules.get(module_name)
        if ffi_mod:
            # Build search names for method_defs matching
            mdef_search_names = [symbol_name]
            if not symbol_name.startswith("_"):
                mdef_search_names.append(f"_{symbol_name}")
                mdef_search_names.append(f"_{symbol_name}_inner")
            elif symbol_name.startswith("_") and not symbol_name.startswith("__"):
                mdef_search_names.append(symbol_name[1:])
                if symbol_name.endswith("_inner"):
                    mdef_search_names.append(symbol_name[:-6])

            for mdef in ffi_mod.method_defs:
                mdef_name = mdef.get("name", "")
                if mdef_name in mdef_search_names:
                    c_sym = mdef.get("func_symbol", "")
                    if c_sym and c_sym not in sym_names_to_try:
                        sym_names_to_try.append(c_sym)
                    # Also try the Python method name from the method_def
                    if mdef_name and mdef_name not in sym_names_to_try:
                        sym_names_to_try.append(mdef_name)

        # Chain dlsym calls: try each name, keep the first non-NULL result.
        # Use select: sym_addr = (new != NULL) ? new : old
        sym_addr = ir.Constant(I8P, None)
        null_ptr = ir.Constant(I8P, None)

        for idx, sname in enumerate(sym_names_to_try):
            sname_str = sname + '\0'
            sname_const = ir.GlobalVariable(
                self.module, ir.ArrayType(I8, len(sname_str)),
                name=f"__ffi_sym_{module_name}_{symbol_name}_v{idx}"
            )
            sname_const.global_constant = True
            sname_const.linkage = "private"
            sname_const.initializer = ir.Constant(
                ir.ArrayType(I8, len(sname_str)),
                [ir.Constant(I8, ord(c)) for c in sname_str]
            )
            sname_ptr = self.builder.gep(sname_const, [z, z], inbounds=True)
            new_addr = self.builder.call(dlsym_fn, [handle, sname_ptr],
                                         name=f"ffi_{symbol_name}_addr_v{idx}")
            # If we already have a result, keep it; otherwise use the new one
            already_found = self.builder.icmp_unsigned("!=", sym_addr, null_ptr,
                                                        name=f"ffi_{symbol_name}_has_v{idx}")
            sym_addr = self.builder.select(already_found, sym_addr, new_addr,
                                            name=f"ffi_{symbol_name}_best_v{idx}")

        # ── Fallback: vaddr-based resolution for LOCAL (static) C symbols ──
        # When dlsym fails because the C function is static (LOCAL symbol in
        # the ELF file, not in the dynamic symbol table), we can compute its
        # runtime address using:
        #   target_addr = pyinit_runtime_addr + (func_vaddr - pyinit_vaddr)
        #
        # The delta (func_vaddr - pyinit_vaddr) is a compile-time constant
        # from the ELF analysis.  pyinit_runtime_addr is obtained via
        # dlsym("PyInit_*") since PyInit_ IS always a GLOBAL exported symbol.
        # This works because both functions reside in the same .so, so their
        # relative offset is fixed regardless of where the .so is loaded.
        ffi_mod = self._ffi_modules.get(module_name)
        vaddr_delta = None  # Will be set if we find a matching method_def
        vaddr_pyinit_name = None

        if ffi_mod:
            # Search method_defs for the matching Python method name.
            # Try exact match first, then common Python naming conventions:
            # - "escape" may also be registered as "_escape_inner" in C extensions
            # - "_escape_inner" may be searched as "escape" or "_escape_inner"
            search_names = [symbol_name]
            # Add common Python→C naming variations
            if not symbol_name.startswith("_"):
                search_names.append(f"_{symbol_name}")
                search_names.append(f"_{symbol_name}_inner")
            elif symbol_name.startswith("_") and not symbol_name.startswith("__"):
                search_names.append(symbol_name[1:])  # Remove leading _
                if symbol_name.endswith("_inner"):
                    search_names.append(symbol_name[:-6])  # Remove _inner suffix

            matched_mdef = None
            for mdef in ffi_mod.method_defs:
                mdef_name = mdef.get("name", "")
                if mdef_name in search_names:
                    matched_mdef = mdef
                    break

            if matched_mdef:
                func_vaddr_str = matched_mdef.get("func_vaddr", "0x0")
                func_vaddr = int(func_vaddr_str, 16) if isinstance(func_vaddr_str, str) else int(func_vaddr_str)
                # Find the PyInit_ symbol for the same .so file
                # For from_file: use the module's pyinit_symbol
                # For from_package: use _so_pyinit_map matched by _so_path
                so_path_hint = matched_mdef.get("_so_path", "")
                pyinit_vaddr = 0
                pyinit_sym_name = None

                if so_path_hint and hasattr(ffi_mod, '_so_pyinit_map'):
                    pyinit_info = ffi_mod._so_pyinit_map.get(so_path_hint)
                    if pyinit_info:
                        pyinit_vaddr = pyinit_info["vaddr"]
                        pyinit_sym_name = pyinit_info["name"]

                if not pyinit_vaddr and ffi_mod.pyinit_symbol:
                    pyinit_vaddr = ffi_mod.pyinit_symbol.address
                    pyinit_sym_name = ffi_mod.pyinit_symbol.name

                if func_vaddr and pyinit_vaddr and pyinit_sym_name:
                    vaddr_delta = func_vaddr - pyinit_vaddr
                    vaddr_pyinit_name = pyinit_sym_name

        if vaddr_delta is not None and vaddr_pyinit_name is not None:
            # Generate code to:
            # 1. dlsym(handle, "PyInit_*") → get runtime address of PyInit_
            # 2. target = pyinit_addr + delta
            # 3. Use target if dlsym didn't find the symbol by name
            pyinit_name_str = vaddr_pyinit_name + '\0'
            pyinit_name_const = ir.GlobalVariable(
                self.module, ir.ArrayType(I8, len(pyinit_name_str)),
                name=f"__ffi_pyinit_{module_name}_{symbol_name}"
            )
            pyinit_name_const.global_constant = True
            pyinit_name_const.linkage = "private"
            pyinit_name_const.initializer = ir.Constant(
                ir.ArrayType(I8, len(pyinit_name_str)),
                [ir.Constant(I8, ord(c)) for c in pyinit_name_str]
            )
            pyinit_name_ptr = self.builder.gep(pyinit_name_const, [z, z], inbounds=True)
            pyinit_addr = self.builder.call(dlsym_fn, [handle, pyinit_name_ptr],
                                            name=f"ffi_{module_name}_pyinit_addr")

            # Calculate: target_addr = pyinit_addr + delta
            # delta is a compile-time constant, so this is very efficient
            pyinit_int = self.builder.ptrtoint(pyinit_addr, I64, name=f"ffi_{symbol_name}_pyinit_int")
            target_int = self.builder.add(pyinit_int, ir.Constant(I64, vaddr_delta),
                                          name=f"ffi_{symbol_name}_target_int")
            target_ptr = self.builder.inttoptr(target_int, I8P, name=f"ffi_{symbol_name}_vaddr_ptr")

            # Use vaddr-resolved address only if dlsym by name failed
            # AND PyInit was found (which it should always be since it's GLOBAL)
            dlsym_found = self.builder.icmp_unsigned("!=", sym_addr, null_ptr,
                                                      name=f"ffi_{symbol_name}_dlsym_found")
            pyinit_found = self.builder.icmp_unsigned("!=", pyinit_addr, null_ptr,
                                                       name=f"ffi_{symbol_name}_pyinit_found")
            use_vaddr = self.builder.and_(
                self.builder.not_(dlsym_found, name=f"ffi_{symbol_name}_not_dlsym"),
                pyinit_found,
                name=f"ffi_{symbol_name}_use_vaddr"
            )
            sym_addr = self.builder.select(use_vaddr, target_ptr, sym_addr,
                                            name=f"ffi_{symbol_name}_final_addr")

        # ── NULL check for final dlsym result ──
        dlsym_ok = self.builder.icmp_unsigned("!=", sym_addr, null_ptr, name=f"ffi_{symbol_name}_dlsym_ok")

        dlsym_fail_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_dlsym_fail")
        dlsym_ok_bb = self.current_func.append_basic_block(f"ffi_{symbol_name}_dlsym_ok")
        self.builder.cbranch(dlsym_ok, dlsym_ok_bb, dlsym_fail_bb)

        # dlsym failed: print error and exit
        self.builder.position_at_start(dlsym_fail_bb)
        sym_err_msg = f"[pylow] FFI: symbol '{symbol_name}' not found in module '{module_name}'\n\0"
        sym_err_const = ir.GlobalVariable(
            self.module, ir.ArrayType(I8, len(sym_err_msg)),
            name=f"__ffi_dlsym_err_{module_name}_{symbol_name}"
        )
        sym_err_const.global_constant = True
        sym_err_const.linkage = "private"
        sym_err_const.initializer = ir.Constant(
            ir.ArrayType(I8, len(sym_err_msg)),
            [ir.Constant(I8, ord(c)) for c in sym_err_msg]
        )
        sym_err_ptr = self.builder.gep(sym_err_const, [z, z], inbounds=True)
        self.builder.call(self._printf, [sym_err_ptr])
        if exit_fn:
            self.builder.call(exit_fn, [ir.Constant(I32, 1)])
        # exit() never returns — mark as unreachable so LLVM
        # knows this block has a valid terminator.
        self.builder.unreachable()

        self.builder.position_at_start(dlsym_ok_bb)

        # ── Check if this is a CPython extension module ──
        # CPython extension functions use inline macros (PyUnicode_Check,
        # Py_INCREF, etc.) that directly access PyObject memory layout.
        # We CANNOT call them directly with pylow's internal values —
        # we must use the CPython bridge to convert arguments to real
        # PyObject* and results back.
        is_cpython_ext = (
            hasattr(self, '_ffi_modules')
            and has_cpython_extensions(self._ffi_modules)
        )

        # Build the argument value (needed for both paths)
        arg_val = None
        if node.args:
            arg_val = self.visit(node.args[0])

        if is_cpython_ext and use_wrapper:
            # ── AOT C++ wrapper path: use _ffi_wrapper_call ──
            # For CPython extension modules, we route through AOT-generated
            # wrapper functions instead of calling dlsym-resolved functions directly.
            # The wrapper handles PyObject* conversion internally.
            # Find the module name for this symbol
            _mod_name = module_name
            # Try to get the FFI function declaration
            _ffi_fn = self._ffi_symbols.get(symbol_name)
            if _ffi_fn is None:
                # Create a dummy declaration for the wrapper call routing
                _ffi_fn = ir.Function(self.module,
                    ir.FunctionType(I8P, [I8P, I8P]),
                    name=symbol_name)
            return self._ffi_wrapper_call(symbol_name, node, _ffi_fn, _mod_name)

        else:
            # ── Direct call path: for non-CPython FFI functions ──
            # Default signature: i8* func(i8*, i8*)
            func_ptr_type = ir.PointerType(ir.FunctionType(I8P, [I8P, I8P]))
            func_ptr = self.builder.bitcast(sym_addr, func_ptr_type, name=f"ffi_{symbol_name}_fptr")

            call_args = []
            if arg_val is not None:
                # Marshal to i8* (PyObject* equivalent)
                if arg_val.is_str:
                    data_ptr = self.builder.load(
                        self.builder.gep(arg_val.llvm, [z, ir.Constant(I32, 3)], inbounds=True)
                    )
                    call_args.append(data_ptr)
                elif arg_val.is_int:
                    boxed = self._box(arg_val)
                    call_args.append(self.builder.bitcast(boxed, I8P))
                elif arg_val.is_object:
                    call_args.append(self.builder.bitcast(arg_val.llvm, I8P))
                else:
                    boxed = self._box(arg_val)
                    call_args.append(self.builder.bitcast(boxed, I8P))
            else:
                call_args.append(ir.Constant(I8P, None))

            self_arg = ir.Constant(I8P, None)
            ret_i8p = self.builder.call(func_ptr, [self_arg, call_args[0]], name=f"ffi_{symbol_name}_ret")

            ret_boxed = self.builder.bitcast(ret_i8p, BOXED_PTR)
            return Value(ret_boxed, PyType.OBJECT)

    def _cstr_to_pylow_str(self, cstr_ptr: "ir.Value") -> Value:
        """Convert a C const char* (i8*) to a pylow string Value.

        Creates a new STR object by measuring the C string length with
        strlen and copying the data into a pylow-managed buffer.

        Args:
            cstr_ptr: An i8* LLVM value pointing to a null-terminated C string.

        Returns:
            A Value of PyType.STR containing the copied string.
        """
        # Declare strlen if not already declared
        if "strlen" not in self.functions:
            fty = ir.FunctionType(I64, [I8P])
            fn = ir.Function(self.module, fty, name="strlen")
            self.functions["strlen"] = fn

        strlen_fn = self.functions["strlen"]
        length = self.builder.call(strlen_fn, [cstr_ptr], name="cstr_len")

        # Create a new pylow string with the measured length
        z = ir.Constant(I32, 0)
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)], name="ffi_str_raw")
        str_obj = self.builder.bitcast(raw, STR_PTR, name="ffi_str_obj")

        # Init GC header
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, -1), self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 1)], inbounds=True))  # static
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(null_i8p, self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 3)], inbounds=True))

        # Store length and capacity
        self.builder.store(length, self.builder.gep(str_obj, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(length, self.builder.gep(str_obj, [z, ir.Constant(I32, 2)], inbounds=True))

        # Allocate data buffer and copy
        buf_size = self.builder.add(length, ir.Constant(I64, 1))
        data_buf = self.builder.call(self._malloc, [buf_size], name="ffi_str_buf")

        # Copy the C string data
        if "memcpy" not in self.functions:
            fty = ir.FunctionType(I8P, [I8P, I8P, I64])
            fn = ir.Function(self.module, fty, name="memcpy")
            self.functions["memcpy"] = fn
        self.builder.call(self.functions["memcpy"], [data_buf, cstr_ptr, buf_size])

        # Store data pointer
        self.builder.store(data_buf, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True))

        return Value(str_obj, PyType.STR)

    def _cstr_and_len_to_pylow_str(self, cstr_ptr: "ir.Value", length: "ir.Value") -> Value:
        """Convert a C const char* + known length to a pylow string Value.

        This is similar to _cstr_to_pylow_str but takes the length as an
        explicit parameter instead of calling strlen.  This is used by the
        CPython bridge path where the length is already known.

        Args:
            cstr_ptr: An i8* LLVM value pointing to a null-terminated C string.
            length: An i64 LLVM value with the string length.

        Returns:
            A Value of PyType.STR containing the copied string.
        """
        z = ir.Constant(I32, 0)
        raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_STR)], name="ffi_str_raw")
        str_obj = self.builder.bitcast(raw, STR_PTR, name="ffi_str_obj")

        # Init GC header
        null_i8p = ir.Constant(I8P, None)
        self.builder.store(ir.Constant(I64, 1), self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True))
        self.builder.store(ir.Constant(I32, -1), self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 1)], inbounds=True))  # static
        self.builder.store(ir.Constant(I64, 0), self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 2)], inbounds=True))
        self.builder.store(null_i8p, self.builder.gep(str_obj, [z, ir.Constant(I32, 0), ir.Constant(I32, 3)], inbounds=True))

        # Store length and capacity
        self.builder.store(length, self.builder.gep(str_obj, [z, ir.Constant(I32, 1)], inbounds=True))
        self.builder.store(length, self.builder.gep(str_obj, [z, ir.Constant(I32, 2)], inbounds=True))

        # Allocate data buffer (length + 1 for null terminator) and copy
        buf_size = self.builder.add(length, ir.Constant(I64, 1))
        data_buf = self.builder.call(self._malloc, [buf_size], name="ffi_str_buf")

        # Copy the C string data
        if "memcpy" not in self.functions:
            fty = ir.FunctionType(I8P, [I8P, I8P, I64])
            fn = ir.Function(self.module, fty, name="memcpy")
            self.functions["memcpy"] = fn
        self.builder.call(self.functions["memcpy"], [data_buf, cstr_ptr, buf_size])

        # Store data pointer
        self.builder.store(data_buf, self.builder.gep(str_obj, [z, ir.Constant(I32, 3)], inbounds=True))

        return Value(str_obj, PyType.STR)