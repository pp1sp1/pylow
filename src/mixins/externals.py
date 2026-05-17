################################################################################

"""External C runtime function declarations and built-in module initialization."""

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


class ExternalDeclarationsMixin:
    """External C runtime function declarations and built-in module initialization."""

    def _declare_externals(self):
        def decl(name, ret, *args, var_arg=False):
            fty = ir.FunctionType(ret, list(args), var_arg=var_arg)
            fn = ir.Function(self.module, fty, name=name)
            self.functions[name] = fn
            return fn

        self._printf = decl("printf", I32, I8P, var_arg=True)
        self._malloc = decl("malloc", I8P, I64)
        self._realloc = decl("realloc", I8P, I8P, I64)
        self._free = decl("free", VOID, I8P)
        self._strcpy = decl("strcpy", I8P, I8P, I8P)
        self._strcat = decl("strcat", I8P, I8P, I8P)
        self._strcmp = decl("strcmp", I32, I8P, I8P)
        self._strstr = decl("strstr", I8P, I8P, I8P)
        self._snprintf = decl("snprintf", I32, I8P, I64, I8P, var_arg=True)
        self._memset = decl("memset", I8P, I8P, I32, I64)

        # memcpy declaration
        if "memcpy" not in self.functions:
            fty = ir.FunctionType(I8P, [I8P, I8P, I64], var_arg=False)
            fn = ir.Function(self.module, fty, name="memcpy")
            self.functions["memcpy"] = fn
            self._memcpy_decl = fn

        # ── FFI: dlopen/dlsym for dynamic .so loading ──
        if "dlopen" not in self.functions:
            fty_dlopen = ir.FunctionType(I8P, [I8P, I32])
            fn_dlopen = ir.Function(self.module, fty_dlopen, name="dlopen")
            self.functions["dlopen"] = fn_dlopen

        if "dlsym" not in self.functions:
            fty_dlsym = ir.FunctionType(I8P, [I8P, I8P])
            fn_dlsym = ir.Function(self.module, fty_dlsym, name="dlsym")
            self.functions["dlsym"] = fn_dlsym

        if "dlclose" not in self.functions:
            fty_dlclose = ir.FunctionType(I32, [I8P])
            fn_dlclose = ir.Function(self.module, fty_dlclose, name="dlclose")
            self.functions["dlclose"] = fn_dlclose

        if "strlen" not in self.functions:
            fty_strlen = ir.FunctionType(I64, [I8P])
            fn_strlen = ir.Function(self.module, fty_strlen, name="strlen")
            self.functions["strlen"] = fn_strlen

        # ═══════════════════════════════════════════════════════════
        #  Cross-function exception propagation globals
        # ═══════════════════════════════════════════════════════════
        exc_pending = ir.GlobalVariable(self.module, I1, name="__py2llvm_exc_pending")
        exc_pending.initializer = ir.Constant(I1, 0)
        exc_pending.linkage = "common"
        self._exc_pending_global = exc_pending

        exc_type_hash = ir.GlobalVariable(self.module, I64, name="__py2llvm_exc_type_hash")
        exc_type_hash.initializer = ir.Constant(I64, 0)
        exc_type_hash.linkage = "common"
        self._exc_type_hash_global = exc_type_hash

        exc_value = ir.GlobalVariable(self.module, BOXED_PTR, name="__py2llvm_exc_value")
        exc_value.initializer = ir.Constant(BOXED_PTR, None)
        exc_value.linkage = "common"
        self._exc_value_global = exc_value

    def _init_builtin_modules(self):
        """Initialize built-in module wrappers."""
        # Define module functions based on libs_mode and dynamic_libs

        # Math module functions (only if math is not dynamic)
        if "math" not in self.dynamic_libs:
            self._init_math_module()

        # Time module
        if "time" not in self.dynamic_libs:
            self._init_time_module()

        # OS module (basic functions)
        if "os" not in self.dynamic_libs:
            self._init_os_module()

        # Sys module
        if "sys" not in self.dynamic_libs:
            self._init_sys_module()

    def _init_math_module(self):
        """Initialize math module functions using LLVM intrinsics."""
        # Common math functions using LLVM intrinsics
        math_funcs = [
            ("sqrt", F64, [F64], "llvm.sqrt.f64"),
            ("sin", F64, [F64], "llvm.sin.f64"),
            ("cos", F64, [F64], "llvm.cos.f64"),
            ("exp", F64, [F64], "llvm.exp.f64"),
            ("log", F64, [F64], "llvm.log.f64"),
            ("pow", F64, [F64, F64], "llvm.pow.f64"),
            ("floor", F64, [F64], "llvm.floor.f64"),
            ("ceil", F64, [F64], "llvm.ceil.f64"),
            ("fabs", F64, [F64], "llvm.fabs.f64"),
        ]

        for name, ret_type, arg_types, intrinsic_name in math_funcs:
            # Check if intrinsic already declared
            if intrinsic_name in self.functions:
                continue
            # Declare the intrinsic
            fty = ir.FunctionType(ret_type, arg_types)
            fn = ir.Function(self.module, fty, name=intrinsic_name)
            self.functions[intrinsic_name] = fn
            # Also store under short name for import handling
            self.functions[f"math.{name}"] = fn
            self.functions[name] = fn  # Allow direct sqrt() call

    def _init_time_module(self):
        """Initialize time module functions — clock_gettime/nanosleep."""
        # clock_gettime — deklarowana lazy w _builtin_time_time() bo
        # sygnatura timespec* jest specyficzna.
        # Nie deklarujemy libc time() bo ma złą sygnaturę (time_t*).
        # Zamiast tego _builtin_time_time() używa clock_gettime inline.
        pass

    def _init_os_module(self):
        """Initialize os module functions."""
        # getcwd() — deklarowana lazy w _builtin_os_getcwd() bo
        # prawdziwa sygnatura libc to getcwd(char*, size_t).
        # getenv/system — deklarowane lazy w ich handlerach.
        pass

    def _init_sys_module(self):
        """Initialize sys module functions."""
        # exit() function
        fty = ir.FunctionType(VOID, [I32])
        fn = ir.Function(self.module, fty, name="exit")
        self.functions["sys.exit"] = fn

    # ══════════════════════════════════════════════════════════════════
    #  ARC (Atomic Ref Count) + Cycle Collector Runtime
    # ══════════════════════════════════════════════════════════════════
