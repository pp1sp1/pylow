################################################################################

"""AST visitor methods for function/class definitions and module structure."""

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
from ..values import Value
from ..type_analyzer import StaticTypeAnalyzer
from ..ffi.core import FFIModule, FFISignatureDB

if TYPE_CHECKING:
    pass


class VisitorsFuncMixin:
    """AST visitor methods for function/class definitions and module structure."""

    def visit_Module(self, node: ast.Module):
        func_defs = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        class_defs = [n for n in node.body if isinstance(n, ast.ClassDef)]
        other_stmts = [
            n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        # Pre-scan top-level assignments to create LLVM global variables
        # This allows functions with 'global' declarations to access them
        top_level_vars = set()
        for stmt in other_stmts:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        top_level_vars.add(tgt.id)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    top_level_vars.add(stmt.target.id)
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    top_level_vars.add(stmt.target.id)

        # Scan all function bodies (including class methods) for 'global' declarations
        # so that LLVM module-level globals are created for those variables too
        global_decls = set()
        def _scan_global(node):
            if isinstance(node, ast.Global):
                global_decls.update(node.names)
            for child in ast.iter_child_nodes(node):
                # Don't descend into nested function defs – their 'global'
                # refers to module-level, not the enclosing function's locals.
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _scan_global(child)
        for fd in func_defs:
            for child in fd.body:
                _scan_global(child)
        for cd in class_defs:
            for item in cd.body:
                if isinstance(item, ast.FunctionDef):
                    for child in item.body:
                        _scan_global(child)

        # NAPRAWA: Scan for 'nonlocal' declarations in nested functions
        # so that LLVM globals are created for those variables too.
        # LLVM nie pozwala na dostęp do allocas z innej funkcji, więc
        # nonlocal variables muszą być przechowywane w LLVM globals.
        nonlocal_decls = set()
        def _scan_nonlocal(node):
            if isinstance(node, ast.Nonlocal):
                nonlocal_decls.update(node.names)
            for child in ast.iter_child_nodes(node):
                # Descend into nested function defs to find all nonlocal declarations
                _scan_nonlocal(child)
        for fd in func_defs:
            for child in fd.body:
                _scan_nonlocal(child)
        for cd in class_defs:
            for item in cd.body:
                if isinstance(item, ast.FunctionDef):
                    for child in item.body:
                        _scan_nonlocal(child)

        # Merge: any variable declared 'global' inside a function must also
        # have an LLVM global, even if there's no top-level assignment yet.
        all_global_vars = top_level_vars | global_decls

        # Create LLVM global variables for all identified global vars
        for var_name in all_global_vars:
            gvar_name = f"__global_{var_name}"
            if gvar_name not in self.module.globals:
                gv = ir.GlobalVariable(self.module, BOXED_PTR, name=gvar_name)
                gv.initializer = ir.Constant(BOXED_PTR, None)
                gv.linkage = "common"
                self.module.globals[gvar_name] = gv

        # NAPRAWA: Create LLVM global variables for nonlocal vars
        for var_name in nonlocal_decls:
            nl_gvar_name = f"__nonlocal_{var_name}"
            if nl_gvar_name not in self.module.globals:
                gv = ir.GlobalVariable(self.module, BOXED_PTR, name=nl_gvar_name)
                gv.initializer = ir.Constant(BOXED_PTR, None)
                gv.linkage = "common"
                self.module.globals[nl_gvar_name] = gv

        # Pre-declare functions
        for fd in func_defs:
            self._pre_declare(fd)

        # NAPRAWA: Pre-process import statements BEFORE compiling function bodies
        # so that imported modules (like asyncio) are available inside functions.
        # We only process the import registration (symbol table entries), not
        # the actual LLVM code generation.
        import_stmts = [s for s in other_stmts if isinstance(s, (ast.Import, ast.ImportFrom))]
        for imp in import_stmts:
            self.visit(imp)

        # NAPRAWA: Pre-rejestruj hierarchię klas i deklaracje metod ZANIM
        # skompilujesz ciała metod. Bez tego super() w A.speak() nie widzi
        # klasy C w hierarchii, więc MRO nie znajduje B po A (bug: brak "B speaking").
        for cd in class_defs:
            base_names = []
            for base in cd.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
            if not hasattr(self, '_class_hierarchy'):
                self._class_hierarchy = {}
            self._class_hierarchy[cd.name] = base_names

            # Pre-declare class methods so _super_method_call can find them
            for item in cd.body:
                if isinstance(item, ast.FunctionDef):
                    # Detect decorator type
                    prop_kind = None
                    prop_name = None
                    is_staticmethod = False
                    for dec in item.decorator_list:
                        if isinstance(dec, ast.Name):
                            if dec.id == 'property':
                                prop_kind = 'getter'
                                prop_name = item.name
                                break
                            elif dec.id == 'staticmethod':
                                is_staticmethod = True
                                break
                        elif isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                            if dec.attr == 'setter':
                                prop_kind = 'setter'
                                prop_name = dec.value.id
                                break
                            elif dec.attr == 'deleter':
                                prop_kind = 'deleter'
                                prop_name = dec.value.id
                                break

                    if prop_kind:
                        method_llvm_name = f"py_{cd.name}_{prop_name}_{prop_kind}"
                    else:
                        method_llvm_name = f"py_{cd.name}_{item.name}"

                    if method_llvm_name not in self.functions:
                        if is_staticmethod:
                            arg_types = []
                            for arg in item.args.args:
                                if arg.annotation:
                                    arg_types.append(pytype_to_llvm(self._ann_to_pytype(arg.annotation)))
                                else:
                                    arg_types.append(BOXED_PTR)
                        else:
                            arg_types = [INSTANCE_PTR]
                            for arg in item.args.args[1:]:
                                if arg.annotation:
                                    arg_types.append(pytype_to_llvm(self._ann_to_pytype(arg.annotation)))
                                else:
                                    arg_types.append(BOXED_PTR)

                        if item.name == "__init__":
                            ret_type = VOID
                        elif item.returns is not None:
                            ret_type = pytype_to_llvm(self._ann_to_pytype(item.returns))
                        else:
                            ret_type = BOXED_PTR

                        fty = ir.FunctionType(ret_type, arg_types)
                        func = ir.Function(self.module, fty, name=method_llvm_name)
                        self.functions[method_llvm_name] = func

        # Clear MRO cache since hierarchy is now complete
        if hasattr(self, '_mro_cache'):
            self._mro_cache.clear()

        # Compile class definitions first (so they're available to functions)
        for cd in class_defs:
            self.visit(cd)

        # Compile function definitions
        for fd in func_defs:
            self.visit(fd)

        # Compile other statements (including class instances, etc.)
        # Filter out already-processed imports to avoid double-processing
        remaining_stmts = [s for s in other_stmts if not isinstance(s, (ast.Import, ast.ImportFrom))]
        if remaining_stmts or import_stmts:
            # Re-include imports so their LLVM code (if any) gets generated too
            # But for built-in modules, there's no LLVM code, just symbol table entries
            if remaining_stmts:
                self._compile_top_level(remaining_stmts)

        # Always create C-compatible main
        self._create_c_main()

    def _pre_declare(self, node):
        """Pre-declare a function (FunctionDef or AsyncFunctionDef) in the LLVM module."""
        n_args = len(node.args.args)
        arg_types = []
        for arg in node.args.args:
            if arg.annotation:
                arg_types.append(pytype_to_llvm(self._ann_to_pytype(arg.annotation)))
            else:
                arg_types.append(BOXED_PTR)
        # All user functions get py_ prefix to avoid collisions with C main
        llvm_name = f"py_{node.name}"
        if node.returns:
            ret_type = pytype_to_llvm(self._ann_to_pytype(node.returns))
        else:
            ret_type = BOXED_PTR
        fty = ir.FunctionType(ret_type, arg_types)
        func = ir.Function(self.module, fty, name=llvm_name)
        for la, pa in zip(func.args, node.args.args):
            la.name = pa.arg
        self.functions[node.name] = func  # Map Python name → LLVM function
        self.functions[llvm_name] = func  # Also map llvm name

        # ── Async: also pre-declare the coroutine entry wrapper ──
        # py_{name}_coro_entry(i8* arg_ptr) -> void
        # This is generated later by _generate_coro_entry in async_runtime.py,
        # but we pre-declare it here so it's available for forward references.
        is_async = isinstance(node, ast.AsyncFunctionDef)
        if is_async:
            coro_entry_name = f"py_{node.name}_coro_entry"
            if coro_entry_name not in self.functions:
                coro_fty = ir.FunctionType(VOID, [I8P])
                coro_fn = ir.Function(self.module, coro_fty, name=coro_entry_name)
                self.functions[coro_entry_name] = coro_fn

        # Store AST for potential inlining
        self._function_ast[node.name] = node

        # Track if this is an async function
        if isinstance(node, ast.AsyncFunctionDef):
            if not hasattr(self, '_async_functions'):
                self._async_functions = set()
            self._async_functions.add(node.name)

    def _ann_to_pytype(self, ann: ast.expr) -> PyType:
        if isinstance(ann, ast.Name):
            return {
                "int": PyType.INT,
                "float": PyType.FLOAT,
                "bool": PyType.BOOL,
                "None": PyType.NONE,
                "str": PyType.STR,
                "list": PyType.LIST,
                "dict": PyType.DICT,
            }.get(ann.id, PyType.OBJECT)
        return PyType.OBJECT

    def _compile_body(
        self,
        func: ir.Function,
        stmts: list,
        args_info: Optional[List[Tuple[str, ir.Type]]] = None,
    ):
        self.current_func = func
        self.sym.push()
        entry = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        analyzer = StaticTypeAnalyzer()
        for stmt in stmts:
            analyzer.visit(stmt)
        self.inferred_static_types = analyzer.var_types

        # Zapisz i zresetuj deklaracje global/nonlocal dla tego scope'u
        saved_global_vars = self._global_vars
        saved_nonlocal_vars = self._nonlocal_vars
        self._global_vars = set()
        self._nonlocal_vars = set()

        if args_info:
            for la in func.args:
                alloca = self.builder.alloca(la.type, name=la.name)
                self.builder.store(la, alloca)
                if la.type == BOXED_PTR:
                    self.sym.define(la.name, VarInfo(alloca, BOXED_PTR, PyType.OBJECT))
                else:
                    pt = self._llvm_type_to_pytype(la.type)
                    self.sym.define(la.name, VarInfo(alloca, la.type, pt))

        # FIX: Wykryj funkcje generatorowe (zawierające yield) i skompiluj
        # je tak, by zwracały iterator z listą zebranych wartości yield.
        def _has_yield(node):
            if isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                return True
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if _has_yield(child):
                        return True
            return False

        is_generator = any(_has_yield(stmt) for stmt in stmts)
        self._is_generator = is_generator
        self._generator_list = None  # zostanie utworzona przy pierwszym yield

        # NAPRAWA: Pre-skanuj ciało funkcji aby znaleść wszystkie przypisania
        # zmiennych lokalnych i utwórz ich allocas w entry blocku.
        # Bez tego allocas tworzone wewnątrz pętli/warunków nie dominują
        # bloków, które ich używają, co powoduje błąd "Instruction does
        # not dominate all uses" przy weryfikacji IR.
        _pre_assigned_names = set()
        _global_names_in_func = set()
        _nonlocal_names_in_func = set()

        def _scan_assigns(node):
            """Skanuj AST aby znaleść wszystkie przypisania zmiennych."""
            if isinstance(node, ast.Global):
                _global_names_in_func.update(node.names)
                return  # Nie skanuj głębiej — global deklaruje zmienne
            if isinstance(node, ast.Nonlocal):
                _nonlocal_names_in_func.update(node.names)
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return  # Nie skanuj zagnieżdżonych funkcji
            if isinstance(node, ast.ClassDef):
                return  # Nie skanuj klas
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        _pre_assigned_names.add(target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    _pre_assigned_names.add(node.target.id)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    _pre_assigned_names.add(node.target.id)
            for child in ast.iter_child_nodes(node):
                _scan_assigns(child)

        for stmt in stmts:
            _scan_assigns(stmt)

        # Utwórz allocas dla wszystkich zmiennych lokalnych w entry blocku
        # (pomijając zmienne global/nonlocal oraz argumenty funkcji)
        self._pre_allocas = {}
        arg_names = set()
        if args_info:
            for la in func.args:
                arg_names.add(la.name)

        for var_name in _pre_assigned_names:
            if var_name in _global_names_in_func or var_name in _nonlocal_names_in_func:
                continue
            if var_name in arg_names:
                continue
            # Sprawdź czy LLVM global istnieje (zmienna modułowa)
            gvar_name = f"__global_{var_name}"
            has_gvar = gvar_name in self.module.globals
            if has_gvar:
                # Dla zmiennych z LLVM global, używamy globala jako storage
                gv = self.module.globals[gvar_name]
                self._pre_allocas[var_name] = ('global', gv)
            else:
                # Określ typ na podstawie type inference
                inferred = PyType.OBJECT
                _ist = self.inferred_static_types
                if var_name in _ist and len(_ist[var_name]) == 1:
                    candidate = list(_ist[var_name])[0]
                    if candidate in (PyType.INT, PyType.FLOAT, PyType.BOOL):
                        inferred = candidate
                llvm_type = pytype_to_llvm(inferred)
                alloca = self.builder.alloca(llvm_type, name=var_name)
                # NAPRAWA: Zainicjalizuj pre-alloca na null/zero, żeby decref
                # na niezainicjalizowanej zmiennej nie powodował crashu.
                # Bez tego, przy ponownym przypisaniu w pętli (np. doc = _db[doc_id]),
                # old_val z niezainicjalizowanej alloca jest garbage i decref crashuje.
                if llvm_type == BOXED_PTR:
                    self.builder.store(ir.Constant(BOXED_PTR, None), alloca)
                elif llvm_type == I64:
                    self.builder.store(ir.Constant(I64, 0), alloca)
                elif llvm_type == F64:
                    self.builder.store(ir.Constant(F64, 0.0), alloca)
                elif llvm_type == I1:
                    self.builder.store(ir.Constant(I1, 0), alloca)
                self._pre_allocas[var_name] = ('alloca', alloca, llvm_type, inferred)

        # Rejestruj _global_vars i _nonlocal_vars zeskanowane w tej funkcji
        for gn in _global_names_in_func:
            self._global_vars.add(gn)
        for nn in _nonlocal_names_in_func:
            self._nonlocal_vars.add(nn)

        for stmt in stmts:
            if self.builder is None or self.builder.block.is_terminated:
                break
            self.visit(stmt)

        if self.builder is not None and not self.builder.block.is_terminated:
            self._cleanup_scope_vars()
            rt = func.function_type.return_type
            if isinstance(rt, ir.VoidType):
                self.builder.ret_void()
            elif rt == I32:
                self.builder.ret(ir.Constant(I32, 0))
            elif rt == BOXED_PTR:
                # FIX: Jeśli to generator, zwróć iterator z zebraną listą yield
                if is_generator and self._generator_list is not None:
                    iter_val = self._create_iterator(self._generator_list)
                    self.builder.ret(iter_val.llvm)
                else:
                    # NAPRAWA: Zamiast "pass", musimy bezwzględnie zwrócić None,
                    # inaczej LLVM wygeneruje błąd w trakcie egzekucji (cięcie outputu)
                    raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
                    bv = self.builder.bitcast(raw, BOXED_PTR)
                    z = ir.Constant(I32, 0)
                    null_i8p = ir.Constant(I8P, None)

                    gc_hdr = self.builder.gep(bv, [z, z], inbounds=True)
                    self.builder.store(
                        ir.Constant(I64, 1),
                        self.builder.gep(gc_hdr, [z, z], inbounds=True),
                    )
                    self.builder.store(
                        ir.Constant(I32, 0),
                        self.builder.gep(gc_hdr, [z, ir.Constant(I32, 1)], inbounds=True),
                    )
                    self.builder.store(
                        ir.Constant(I64, 0),
                        self.builder.gep(gc_hdr, [z, ir.Constant(I32, 2)], inbounds=True),
                    )
                    self.builder.store(
                        null_i8p,
                        self.builder.gep(gc_hdr, [z, ir.Constant(I32, 3)], inbounds=True),
                    )
                    self.builder.store(
                        ir.Constant(I64, Tag.NONE),
                        self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True),
                    )
                    self.builder.store(
                        ir.Constant(I64, 0),
                        self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True),
                    )
                    self.builder.ret(bv)
            else:
                self.builder.ret(ir.Constant(rt, 0))

        self.sym.pop()
        # Przywróć deklaracje global/nonlocal z zewnętrznego scope'u
        self._global_vars = saved_global_vars
        self._nonlocal_vars = saved_nonlocal_vars
        self.current_func = None
        self.builder = None

    def _cleanup_scope_vars(self):
        """No-op - ARC handles cleanup via decref at reassignment."""
        pass

    def _cleanup_scope_vars_except(self, except_val: ir.Value):
        """No-op - ARC handles cleanup via decref at reassignment."""
        pass

    # ══════════════════════════════════════════════════════════════════
    #  POPRAWKA 1: Obsługa instrukcji 'global' (Test 04)
    # ══════════════════════════════════════════════════════════════════

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Save current context (may be inside another function or top-level)
        saved_func = self.current_func
        saved_builder = self.builder
        saved_sym = self.sym

        # Map Python name to LLVM function (with py_ prefix)
        func = self.functions.get(f"py_{node.name}") or self.functions.get(node.name)
        if func is None:
            self._pre_declare(node)
            func = self.functions.get(f"py_{node.name}") or self.functions.get(
                node.name
            )
            if func is None:
                raise CompileError(f"Niezdefiniowana funkcja: '{node.name}'", node)

        self._compile_body(func, node.body, args_info=True)

        # Restore context after compiling this function's body
        self.current_func = saved_func
        self.builder = saved_builder
        self.sym = saved_sym

    def visit_ClassDef(self, node: ast.ClassDef):
        """Kompiluje definicję klasy - minimalna wersja."""
        class_name = node.name

        # Detect @dataclass decorator and collect field info
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == 'dataclass') or
            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass')
            for d in node.decorator_list
        )
        is_frozen = False
        dataclass_fields = []  # [(field_name, default_value_or_None, annotation)]
        if is_dataclass:
            # Check for frozen=True in @dataclass(frozen=True)
            for d in node.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass':
                    for kw in d.keywords:
                        if kw.arg == 'frozen':
                            if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                is_frozen = True
                            elif isinstance(kw.value, ast.NameConstant) and kw.value.value is True:
                                is_frozen = True

            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_name = item.target.id
                    default = None
                    if item.value is not None:
                        # Has a default value
                        try:
                            default = ast.literal_eval(item.value)
                        except (ValueError, TypeError):
                            # Complex default (e.g. field(default_factory=...)), try to compile it
                            default = item.value  # Keep as AST node
                    dataclass_fields.append((field_name, default, item.annotation))
            # Store dataclass field info for use in visit_Call
            if not hasattr(self, '_dataclass_fields'):
                self._dataclass_fields = {}
            self._dataclass_fields[class_name] = dataclass_fields

            # Store frozen flag
            if not hasattr(self, '_dataclass_frozen'):
                self._dataclass_frozen = {}
            self._dataclass_frozen[class_name] = is_frozen

        # NAPRAWA: Wykryj property (getter/setter/deleter) i wygeneruj unikalne nazwy LLVM
        # @property def status(self): ... -> py_Class_status_getter
        # @status.setter def status(self, val): ... -> py_Class_status_setter
        # @status.deleter def status(self): ... -> py_Class_status_deleter
        if not hasattr(self, '_class_properties'):
            self._class_properties = {}
        class_props = {}  # prop_name -> {'getter': llvm_name, 'setter': llvm_name, 'deleter': llvm_name}
        self._class_properties[class_name] = class_props

        # Dla każdej metody, utwórz funkcję LLVM
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Detect decorator type
                prop_kind = None  # None = regular method, 'getter', 'setter', 'deleter'
                prop_name = None  # The property name (for setter/deleter, the name of the prop)
                is_classmethod = False
                is_staticmethod = False

                for dec in item.decorator_list:
                    if isinstance(dec, ast.Name):
                        if dec.id == 'property':
                            prop_kind = 'getter'
                            prop_name = item.name
                            break
                        elif dec.id == 'classmethod':
                            is_classmethod = True
                            break
                        elif dec.id == 'staticmethod':
                            is_staticmethod = True
                            break
                    elif isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                        if dec.attr == 'setter':
                            prop_kind = 'setter'
                            prop_name = dec.value.id
                            break
                        elif dec.attr == 'deleter':
                            prop_kind = 'deleter'
                            prop_name = dec.value.id
                            break

                if prop_kind:
                    # Property method — generate unique LLVM name
                    method_llvm_name = f"py_{class_name}_{prop_name}_{prop_kind}"
                    # Track in class_properties
                    if prop_name not in class_props:
                        class_props[prop_name] = {}
                    class_props[prop_name][prop_kind] = method_llvm_name
                else:
                    method_llvm_name = f"py_{class_name}_{item.name}"

                # Determine arg types based on decorator
                if is_staticmethod:
                    # Static methods have no self/cls arg
                    arg_types = []
                    for arg in item.args.args:
                        if arg.annotation:
                            arg_types.append(
                                pytype_to_llvm(self._ann_to_pytype(arg.annotation))
                            )
                        else:
                            arg_types.append(BOXED_PTR)
                else:
                    # Regular method or classmethod — first arg is self (INSTANCE_PTR)
                    arg_types = [INSTANCE_PTR]
                    for arg in item.args.args[1:]:  # Skip self/cls
                        if arg.annotation:
                            arg_types.append(
                                pytype_to_llvm(self._ann_to_pytype(arg.annotation))
                            )
                        else:
                            arg_types.append(BOXED_PTR)
                # __init__ should return void (nothing), not a boxed object
                if item.name == "__init__":
                    ret_type = VOID
                elif item.returns is not None:
                    ret_type = pytype_to_llvm(self._ann_to_pytype(item.returns))
                else:
                    ret_type = BOXED_PTR

                if method_llvm_name in self.functions:
                    func = self.functions[method_llvm_name]
                else:
                    fty = ir.FunctionType(ret_type, arg_types)
                    func = ir.Function(self.module, fty, name=method_llvm_name)
                    self.functions[method_llvm_name] = func

        # NAPRAWA: Zapisz informacje o dziedziczeniu (bases) dla MRO
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
        self._class_stack.append({"name": class_name, "bases": base_names})

        # NAPRAWA: Zapisz mapowanie class_name -> base_names dla wyszukiwania metod
        if not hasattr(self, '_class_hierarchy'):
            self._class_hierarchy = {}
        self._class_hierarchy[class_name] = base_names
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Use same decorator detection logic as the first loop
                prop_kind = None
                prop_name = None
                is_classmethod = False
                is_staticmethod = False
                for dec in item.decorator_list:
                    if isinstance(dec, ast.Name):
                        if dec.id == 'property':
                            prop_kind = 'getter'
                            prop_name = item.name
                            break
                        elif dec.id == 'classmethod':
                            is_classmethod = True
                            break
                        elif dec.id == 'staticmethod':
                            is_staticmethod = True
                            break
                    elif isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                        if dec.attr == 'setter':
                            prop_kind = 'setter'
                            prop_name = dec.value.id
                            break
                        elif dec.attr == 'deleter':
                            prop_kind = 'deleter'
                            prop_name = dec.value.id
                            break

                if prop_kind:
                    method_llvm_name = f"py_{class_name}_{prop_name}_{prop_kind}"
                else:
                    method_llvm_name = f"py_{class_name}_{item.name}"

                if method_llvm_name not in self.functions:
                    continue
                func = self.functions[method_llvm_name]

                # Zapisz stan
                old_func = self.current_func
                old_builder = self.builder
                old_sym = self.sym

                self.current_func = func
                self.sym = SymbolTable()
                entry = func.append_basic_block("entry")
                self.builder = ir.IRBuilder(entry)

                # NAPRAWA: Skopiuj zaimportowane moduły ze starego symbol table
                # do nowego. Bez tego wewnątrz metod klasowych 'parse', 'math'
                # itp. są niedostępne — CompileError: Niezdefiniowana zmienna.
                for mod_name, mod_info in self._imported_modules.items():
                    asname = mod_name  # Domyślnie alias = nazwa modułu
                    var_info = VarInfo(
                        None, I8P, PyType.OBJECT,
                        class_name=f"__builtin_module__{mod_name}",
                        is_ffi_module=True,
                        ffi_module_name=mod_name,
                    )
                    self.sym.define(asname, var_info)

                # NAPRAWA: Uruchom StaticTypeAnalyzer na ciele metody przed
                # wizytacją, aby inferred_static_types istniał dla _assign_name.
                # Bez tego AttributeError (CRASH w test_21_import_pylibs.py).
                analyzer = StaticTypeAnalyzer()
                for stmt in item.body:
                    analyzer.visit(stmt)
                self.inferred_static_types = analyzer.var_types

                # Zdefiniuj parametry
                if is_staticmethod:
                    # Static method: all args are regular params (no self/cls)
                    # NAPRAWA: Track static methods so _method_call doesn't add self arg
                    if not hasattr(self, '_static_methods'):
                        self._static_methods = {}
                    self._static_methods[f"{class_name}_{item.name}"] = True
                    for i, arg in enumerate(func.args):
                        if i < len(item.args.args):
                            arg_name = item.args.args[i].arg
                        else:
                            arg_name = f"arg{i}"
                        alloca = self.builder.alloca(arg.type, name=arg_name)
                        self.builder.store(arg, alloca)
                        if arg.type == BOXED_PTR:
                            self.sym.define(
                                arg_name, VarInfo(alloca, BOXED_PTR, PyType.OBJECT)
                            )
                        else:
                            pt = self._llvm_type_to_pytype(arg.type)
                            self.sym.define(arg_name, VarInfo(alloca, arg.type, pt))
                else:
                    # Regular method or classmethod: first arg is self/cls (INSTANCE_PTR)
                    first_arg_name = item.args.args[0].arg if item.args.args else "self"
                    py_args = item.args.args[1:]  # pomiń pierwszy arg (self/cls)
                    for i, arg in enumerate(func.args):
                        if i == 0:
                            arg_name = first_arg_name
                        elif i - 1 < len(py_args):
                            arg_name = py_args[i - 1].arg
                        else:
                            arg_name = f"arg{i}"
                        alloca = self.builder.alloca(arg.type, name=arg_name)
                        self.builder.store(arg, alloca)
                        if arg.type == INSTANCE_PTR:
                            self.sym.define(
                                arg_name, VarInfo(alloca, INSTANCE_PTR, PyType.INSTANCE,
                                                  class_name=class_name)
                            )
                        elif arg.type == BOXED_PTR:
                            self.sym.define(
                                arg_name, VarInfo(alloca, BOXED_PTR, PyType.OBJECT)
                            )
                        else:
                            pt = self._llvm_type_to_pytype(arg.type)
                            self.sym.define(arg_name, VarInfo(alloca, arg.type, pt))

                    # NAPRAWA: For @classmethod, register cls as a class reference
                    # so that cls(...) calls the class constructor
                    if is_classmethod:
                        cls_var_name = first_arg_name  # usually "cls"
                        # Register cls in symbol table as a class reference
                        # This allows visit_Call to recognize cls() as class instantiation
                        self.functions[f"__classref_{cls_var_name}"] = ir.Constant(CLASS_PTR, None)
                        self.functions[f"__is_class_{cls_var_name}"] = True
                        # Also store mapping so we know which class cls refers to
                        if not hasattr(self, '_classmethod_cls_map'):
                            self._classmethod_cls_map = {}
                        self._classmethod_cls_map[cls_var_name] = class_name
                        # NAPRAWA: Mark cls VarInfo as class_ref so __call__ dispatch
                        # in visit_Call does NOT treat cls as an instance with __call__.
                        # Without this, cls("DefaultDevice") inside a classmethod would
                        # dispatch to py_Class___call__(null_instance, arg) causing segfault.
                        try:
                            cls_info = self.sym.lookup(cls_var_name)
                            cls_info.is_class_ref = True
                        except CompileError:
                            pass

                # Odwiedź ciało
                # Jeśli self to BOXED, wydobądź DICT_PTR z payload
                if func.args and func.args[0].type == BOXED_PTR:
                    z = ir.Constant(I32, 0)
                    self_ptr = func.args[0]
                    payload_ptr = self.builder.gep(
                        self_ptr, [z, ir.Constant(I32, 2)], inbounds=True
                    )
                    payload_int = self.builder.load(payload_ptr, "self_payload_int")
                    dict_ptr = self.builder.inttoptr(payload_int, DICT_PTR, "self_dict")
                    # Zaktualizuj symbol żeby self był DICT
                    if "self" in self.sym._stack[-1]:
                        alloca = self.builder.alloca(DICT_PTR, name="self_dict_alloca")
                        self.builder.store(dict_ptr, alloca)
                        self.sym._stack[-1]["self"] = VarInfo(
                            alloca, DICT_PTR, PyType.DICT
                        )
                for stmt in item.body:
                    if self.builder.block.is_terminated:
                        break
                    self.visit(stmt)

                # Domyślny powrót - nie generuj dla BOXED bo funkcja może już zwracać wartość
                # __init__ should return void, not a boxed object
                if not self.builder.block.is_terminated:
                    if func.function_type.return_type == VOID:
                        self.builder.ret_void()
                    elif func.function_type.return_type == BOXED_PTR:
                        # NAPRAWA: Zwracanie poprawnego instancjonowania None, aby blok był skończony!
                        raw = self.builder.call(
                            self._malloc, [ir.Constant(I64, SZ_BOXED)]
                        )
                        bv = self.builder.bitcast(raw, BOXED_PTR)
                        z = ir.Constant(I32, 0)
                        null_i8p = ir.Constant(I8P, None)
                        self.builder.store(
                            ir.Constant(I64, 1),
                            self.builder.gep(
                                bv, [z, z, ir.Constant(I32, 0)], inbounds=True
                            ),
                        )
                        self.builder.store(
                            ir.Constant(I32, 0),
                            self.builder.gep(
                                bv, [z, z, ir.Constant(I32, 1)], inbounds=True
                            ),
                        )
                        self.builder.store(
                            ir.Constant(I64, 0),
                            self.builder.gep(
                                bv, [z, z, ir.Constant(I32, 2)], inbounds=True
                            ),
                        )
                        self.builder.store(
                            null_i8p,
                            self.builder.gep(
                                bv, [z, z, ir.Constant(I32, 3)], inbounds=True
                            ),
                        )
                        self.builder.store(
                            ir.Constant(I64, Tag.NONE),
                            self.builder.gep(
                                bv, [z, ir.Constant(I32, 1)], inbounds=True
                            ),
                        )
                        self.builder.store(
                            ir.Constant(I64, 0),
                            self.builder.gep(
                                bv, [z, ir.Constant(I32, 2)], inbounds=True
                            ),
                        )
                        self.builder.ret(bv)
                    else:
                        self.builder.ret_void()

                # Przywróć stan
                self.current_func = old_func
                self.builder = old_builder
                self.sym = old_sym
        self._class_stack.pop()

        # Oznacz jako klasę - użyj klucza, którego szuka visit_Call
        self.functions[f"__classref_{class_name}"] = ir.Constant(CLASS_PTR, None)
        self.functions[f"__is_class_{class_name}"] = True

    def _compile_top_level(self, stmts: list):
        """Compiles top-level statements into __py2llvm_top_level function."""
        fty = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, fty, name="__py2llvm_top_level")
        self.functions["__py2llvm_top_level"] = func
        self._compile_body(func, stmts)

    def _create_c_main(self):
        """Creates C-compatible main that calls __py2llvm_top_level."""
        if "main" in self.module.globals:
            return  # Already created by user (shouldn't happen with py_ prefix)
        fty = ir.FunctionType(I32, [])
        main_func = ir.Function(self.module, fty, name="main")
        bb = main_func.append_basic_block("entry")
        builder = ir.IRBuilder(bb)
        # Clear exception state at program start
        builder.store(ir.Constant(I1, 0), self._exc_pending_global)
        if "__py2llvm_top_level" in self.functions:
            builder.call(self.functions["__py2llvm_top_level"], [])
        # Check for unhandled exceptions at program end
        pending = builder.load(self._exc_pending_global, "main_exc_pending")
        normal_bb = main_func.append_basic_block("main.normal")
        exc_bb = main_func.append_basic_block("main.unhandled_exc")
        builder.cbranch(pending, exc_bb, normal_bb)

        builder.position_at_end(exc_bb)
        # Print unhandled exception using fputs to stderr
        # Create a global string for the error prefix
        err_str = "Unhandled exception\n\0"
        enc = err_str.encode("utf-8")
        arr_ty = ir.ArrayType(I8, len(enc))
        err_gv = ir.GlobalVariable(self.module, arr_ty, name=".main_err_str")
        err_gv.global_constant = True
        err_gv.initializer = ir.Constant(arr_ty, bytearray(enc))
        err_gv.linkage = "private"
        z = ir.Constant(I32, 0)
        err_ptr = builder.gep(err_gv, [z, z], inbounds=True)
        # Use printf to print the message
        if "printf" in self.functions:
            builder.call(self.functions["printf"], [err_ptr])
        builder.ret(ir.Constant(I32, 1))

        builder.position_at_end(normal_bb)
        builder.ret(ir.Constant(I32, 0))

    # ──────────────────────────────────────────────────────────────
    #  Instrukcje
    # ──────────────────────────────────────────────────────────────

    def visit_Return(self, node: ast.Return):
        # Compute return value and determine what to do
        ret_val = None  # The LLVM value to return (or None for void)
        rt = self.current_func.function_type.return_type

        if node.value is None:
            self._cleanup_scope_vars()
            if rt == I32:
                ret_val = ir.Constant(I32, 0)
            elif rt == BOXED_PTR:
                raw = self.builder.call(self._malloc, [ir.Constant(I64, SZ_BOXED)])
                bv = self.builder.bitcast(raw, BOXED_PTR)
                z = ir.Constant(I32, 0)
                null_i8p = ir.Constant(I8P, None)

                gc_hdr = self.builder.gep(bv, [z, z], inbounds=True)
                self.builder.store(
                    ir.Constant(I64, 1),
                    self.builder.gep(gc_hdr, [z, z], inbounds=True),
                )
                self.builder.store(
                    ir.Constant(I32, 0),
                    self.builder.gep(gc_hdr, [z, ir.Constant(I32, 1)], inbounds=True),
                )
                self.builder.store(
                    ir.Constant(I64, 0),
                    self.builder.gep(gc_hdr, [z, ir.Constant(I32, 2)], inbounds=True),
                )
                self.builder.store(
                    null_i8p,
                    self.builder.gep(gc_hdr, [z, ir.Constant(I32, 3)], inbounds=True),
                )
                self.builder.store(
                    ir.Constant(I64, Tag.NONE),
                    self.builder.gep(bv, [z, ir.Constant(I32, 1)], inbounds=True),
                )
                self.builder.store(
                    ir.Constant(I64, 0),
                    self.builder.gep(bv, [z, ir.Constant(I32, 2)], inbounds=True),
                )
                ret_val = bv
            # For void returns, ret_val stays None
        else:
            v = self.visit(node.value)
            exp = self.current_func.function_type.return_type

            if exp == I32:
                if v.is_object:
                    tag, pay = self._read_slot(v.llvm)
                    ret_val = self.builder.trunc(pay, I32)
                    self._cleanup_scope_vars_except(v.llvm)
                else:
                    v = self._cast_to_llvm(v, I32)
                    self._cleanup_scope_vars()
                    ret_val = v.llvm
            elif exp == BOXED_PTR:
                if v.is_object:
                    ret_val = v.llvm
                    self.builder.call(
                        self.functions["__py2llvm_incref"],
                        [self.builder.bitcast(v.llvm, I8P)],
                    )
                else:
                    ret_val = self._box(v)
                self._cleanup_scope_vars_except(ret_val)
            else:
                if exp == DICT_PTR and v.is_object:
                    ret_val = v.llvm
                    self.builder.call(
                        self.functions["__py2llvm_incref"],
                        [self.builder.bitcast(v.llvm, I8P)],
                    )
                    self._cleanup_scope_vars_except(ret_val)
                else:
                    v = self._cast_to_llvm(v, exp)
                    self._cleanup_scope_vars()
                    ret_val = v.llvm

        # NAPRAWA: Async functions in our synchronous model just return
        # their value normally — no COROUTINE wrapping needed.
        # The async/await semantics are handled at the call site:
        # - asyncio.run(coro) just calls the function and returns the result
        # - await expr just evaluates the expression and returns it
        # - asyncio.sleep() does the actual nanosleep and returns None
        # This approach is correct for our synchronous execution model.

        # Now handle the actual return: redirect through finally if active
        if self._finally_stack:
            finfo = self._finally_stack[-1]
            # Store return kind = 1 (return pending)
            self.builder.store(ir.Constant(I32, 1), finfo["ret_kind_alloca"])
            # Store return value if applicable
            if ret_val is not None and finfo["ret_val_alloca"] is not None:
                self.builder.store(ret_val, finfo["ret_val_alloca"])
            # Branch to finally block
            self.builder.branch(finfo["finally_bb"])
        else:
            # Normal return
            if ret_val is None:
                self.builder.ret_void()
            else:
                self.builder.ret(ret_val)

    def visit_Lambda(self, node: ast.Lambda) -> Value:
        if not hasattr(self, '_lambda_counter'):
            self._lambda_counter = 0
        self._lambda_counter += 1
        lam_name = f"__lambda_{self._lambda_counter}"
        n_params = len(node.args.args)
        arg_types = [BOXED_PTR] * n_params
        fty = ir.FunctionType(BOXED_PTR, arg_types)
        func = ir.Function(self.module, fty, name=f"py_{lam_name}")
        self.functions[f"py_{lam_name}"] = func
        self._function_ast[lam_name] = node
        old_func = self.current_func
        old_builder = self.builder
        old_sym = self.sym
        self.current_func = func
        self.sym = SymbolTable()
        entry = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        for i, (param, arg) in enumerate(zip(node.args.args, func.args)):
            alloca = self.builder.alloca(BOXED_PTR, name=param.arg)
            self.builder.store(arg, alloca)
            self.sym.define(param.arg, VarInfo(alloca, BOXED_PTR, PyType.OBJECT))
        body_val = self.visit(node.body)
        ret_boxed = self._box(body_val)
        self.builder.ret(ret_boxed)
        self.current_func = old_func
        self.builder = old_builder
        self.sym = old_sym
        return Value(func, PyType.OBJECT)

    def visit_Import(self, node: ast.Import):
        """Obsługa instrukcji import — wbudowane moduły i FFI .so."""
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname if alias.asname else module_name

            # Register module as available
            self._imported_modules[module_name] = {}

            # ═══════════════════════════════════════════════════════════
            #  FFI: Check if this is a native .so module
            # ═══════════════════════════════════════════════════════════
            if hasattr(self, '_ffi_modules'):
                # Try to find the .so file on the filesystem
                so_path = self._find_ffi_so(module_name)
                if so_path:
                    if os.path.isdir(so_path):
                        self.register_ffi_package(module_name, so_path)
                    else:
                        self.register_ffi_module(module_name, so_path)
                    # Register the module name in the symbol table as an FFI module reference
                    # This allows visit_Name to resolve 'markupsafe' without error.
                    # Use is_ffi_module=True and ffi_module_name so that visit_Name
                    # returns FFIModuleValue instead of attempting builder.load(None).
                    var_info = VarInfo(
                        None, I8P, PyType.OBJECT,
                        class_name=f"__ffi_module__{module_name}",
                        is_ffi_module=True,
                        ffi_module_name=module_name,
                    )
                    self.sym.define(asname, var_info)
                    continue  # FFI module registered, no need for built-in handling

            # ═══════════════════════════════════════════════════════════
            #  Built-in module handling — register module name in symbol
            #  table so that visit_Name can resolve 'time', 'os', etc.
            #  without "Niezdefiniowana zmienna" error.
            #
            #  We use is_ffi_module=True + ffi_module_name so that
            #  visit_Name returns FFIModuleValue, and _method_call
            #  dispatches to the built-in handler.
            # ═══════════════════════════════════════════════════════════
            if module_name == "os":
                self._imported_modules["os"]["getcwd"] = self.functions.get("os.getcwd")
            elif module_name == "sys":
                self._imported_modules["sys"]["exit"] = self.functions.get("sys.exit")
            elif module_name == "math":
                math_funcs = [
                    "sqrt", "sin", "cos", "exp", "log",
                    "pow", "floor", "ceil", "fabs",
                ]
                for fn in math_funcs:
                    full_name = f"math.{fn}"
                    if full_name in self.functions:
                        self.functions[fn] = self.functions[full_name]

            # Register EVERY imported module name in the symbol table
            # as a module reference (FFIModuleValue). This allows
            # visit_Name to resolve module names like 'time', 'json', etc.
            # _method_call will then dispatch based on the module name.
            if not self.sym.exists_local(asname):
                var_info = VarInfo(
                    None, I8P, PyType.OBJECT,
                    class_name=f"__builtin_module__{module_name}",
                    is_ffi_module=True,
                    ffi_module_name=module_name,
                )
                self.sym.define(asname, var_info)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Obsługa 'from module import ...' — wbudowane moduły i FFI .so."""
        module = node.module if node.module else ""

        # ═══════════════════════════════════════════════════════════
        #  FFI: Check if importing from a native .so module
        # ═══════════════════════════════════════════════════════════
        if hasattr(self, '_ffi_modules'):
            # FFI: Try auto-discovery if module not yet registered
            if module not in self._ffi_modules:
                so_path = self._find_ffi_so(module)
                if so_path:
                    if os.path.isdir(so_path):
                        self.register_ffi_package(module, so_path)
                    else:
                        self.register_ffi_module(module, so_path)

            if module in self._ffi_modules:
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname if alias.asname else name

                    # Resolve the symbol in the FFI module
                    ffi_fn = self.resolve_ffi_symbol(module, name)
                    if ffi_fn is not None:
                        # Register under short name for direct lookup
                        self.functions[asname] = ffi_fn
                        var_info = VarInfo(
                            None, ffi_fn.type, PyType.OBJECT,
                            is_ffi_module=True, ffi_module_name=module,
                        )
                        self.sym.define(asname, var_info)
                return

        # Handle specific imports from built-in modules
        # Map built-in module functions that can be imported directly
        _builtin_from_imports = {
            "time":     ["time", "sleep", "time_ns"],
            "os":       ["getcwd", "exit", "getenv", "system"],
            "sys":      ["exit"],
            "math":     ["sqrt", "sin", "cos", "exp", "log", "pow", "floor", "ceil", "fabs"],
            "random":   ["random", "randint", "choice"],
            "asyncio":  ["sleep", "run", "gather", "create_task", "wait"],
        }

        for alias in node.names:
            name = alias.name
            asname = alias.asname if alias.asname else name

            if module in _builtin_from_imports and name in _builtin_from_imports[module]:
                # Zarejestruj jako funkcję built-in w symbol table
                # Używamy special name "__builtin_{module}.{name}" żeby
                # visit_Call mogło to rozpoznać i dispatchować
                full_name = f"__builtin_{module}.{name}"
                var_info = VarInfo(
                    None, I8P, PyType.OBJECT,
                    class_name=full_name,
                    is_ffi_module=True,
                    ffi_module_name=module,
                )
                self.sym.define(asname, var_info)
                # Zapisz mapowanie do dispatchu w visit_Call
                self._imported_modules.setdefault(module, {})[name] = full_name

            elif module == "math":
                # Legacy math import — użyj LLVM intrinsics
                func_name = f"math.{name}"
                if func_name in self.functions:
                    fn = self.functions[func_name]
                    self.functions[asname] = fn
                    var_info = VarInfo(None, fn.type, PyType.OBJECT)
                    self.sym.define(asname, var_info)

            elif module == "ctypes":
                if name in ["c_int", "c_float", "c_double", "c_char"]:
                    pass

            elif module in self._imported_modules:
                pass

    def _find_ffi_so(self, module_name: str) -> Optional[str]:
        """Locate a .so file for a given module name.

        Search strategy:
        1. Check _ffi_so_search_paths (user-configured paths).
        2. Check the current working directory.
        3. Check Python's site-packages via importlib.

        Args:
            module_name: The module name to find (e.g., "markupsafe").

        Returns:
            Absolute path to the .so file, or None if not found.
        """
        import importlib.util

        # User-configured search paths
        search_paths = getattr(self, '_ffi_so_search_paths', [])

        # Always include cwd
        search_paths = list(search_paths) + [os.getcwd()]

        # Strategy 1: Direct file paths
        for base in search_paths:
            # Try exact name
            candidate = os.path.join(base, f"{module_name}.so")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

            # Try as package directory
            pkg_dir = os.path.join(base, module_name)
            if os.path.isdir(pkg_dir):
                # Check if it contains .so files
                so_files = [
                    os.path.join(root, f)
                    for root, dirs, files in os.walk(pkg_dir)
                    for f in files if f.endswith('.so')
                ]
                if so_files:
                    return os.path.abspath(pkg_dir)

        # Strategy 1.5: Search in .venv site-packages
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else os.getcwd()
        for venv_pattern in [
            os.path.join(script_dir, ".venv", "lib"),
            os.path.join(os.getcwd(), ".venv", "lib"),
        ]:
            if not os.path.isdir(venv_pattern):
                continue
            for py_dir in os.listdir(venv_pattern):
                if py_dir.startswith("python"):
                    sp_dir = os.path.join(venv_pattern, py_dir, "site-packages")
                    if not os.path.isdir(sp_dir):
                        continue
                    # Try exact .so file
                    candidate = os.path.join(sp_dir, f"{module_name}.so")
                    if os.path.isfile(candidate):
                        return os.path.abspath(candidate)
                    # Try as package directory
                    pkg_dir = os.path.join(sp_dir, module_name)
                    if os.path.isdir(pkg_dir):
                        so_files = [
                            os.path.join(root, f)
                            for root, dirs, files in os.walk(pkg_dir)
                            for f in files if f.endswith('.so')
                        ]
                        if so_files:
                            return os.path.abspath(pkg_dir)

        # Strategy 2: Use importlib to find the module
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin and spec.origin.endswith('.so'):
                return os.path.abspath(spec.origin)
            if spec and spec.submodule_search_locations:
                # It's a package
                pkg_dir = list(spec.submodule_search_locations)[0]
                so_files = [
                    os.path.join(root, f)
                    for root, dirs, files in os.walk(pkg_dir)
                    for f in files if f.endswith('.so')
                ]
                if so_files:
                    return os.path.abspath(pkg_dir)
        except (ModuleNotFoundError, ValueError):
            pass

        # Strategy 3: Search in Python's sys.path
        for sp in sys.path:
            if not sp:
                continue
            candidate = os.path.join(sp, f"{module_name}.so")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            pkg_dir = os.path.join(sp, module_name)
            if os.path.isdir(pkg_dir):
                so_files = [
                    os.path.join(root, f)
                    for root, dirs, files in os.walk(pkg_dir)
                    for f in files if f.endswith('.so')
                ]
                if so_files:
                    return os.path.abspath(pkg_dir)

        return None

    # ──────────────────────────────────────────────────────────────
    #  Sterowanie przepływem
    # ──────────────────────────────────────────────────────────────


