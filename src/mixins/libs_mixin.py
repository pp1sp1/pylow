"""Compiler mixin integrating the library management subsystem.

LibsMixin extends the PythonToLLVMCompiler with library discovery,
registration, and linking capabilities.  It overrides visit_Import
and visit_ImportFrom to intercept ``import`` statements and route
pure-Python modules through the compilation pipeline, while leaving
FFI (.so) and built-in modules to the existing handlers.

Critical design: LibsMixin is listed BEFORE VisitorsFuncMixin in the
compiler's MRO, so its visit_Import / visit_ImportFrom take
precedence.  Modules that the libs subsystem cannot handle are
delegated back to VisitorsFuncMixin via direct class-method call.
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import llvmlite.ir as ir

from ..types import PyType, I8, I32, I64, F64, I1, I8P, VOID, BOXED_PTR
from ..exceptions import CompileError
from ..symbols import VarInfo
from ..values import Value, FFIModuleValue
from ..libs.config import LibraryConfig, LinkMode, PurePythonLinkStrategy
from ..libs.registry import LibraryEntry, LibraryKind, LibraryRegistry
from ..libs.pure_python import PurePythonHandler, PurePythonModuleInfo
from ..libs.linker import LinkManager, LinkPlan

if TYPE_CHECKING:
    from ..compiler import PythonToLLVMCompiler
    from .visitors_func import VisitorsFuncMixin


# Built-in modules that are provided by the compiler runtime
_BUILTIN_MODULES = {"math", "time", "os", "sys", "random", "asyncio"}


class LibsMixin:
    """Mixin that integrates the library management subsystem into the compiler.

    This mixin **overrides** ``visit_Import`` and ``visit_ImportFrom`` so
    that pure-Python modules (.py files) are discovered and compiled
    before the FFI / built-in fallback in VisitorsFuncMixin kicks in.

    Modules the libs subsystem cannot handle (no .py found, compilation
    fails) are passed through to VisitorsFuncMixin unchanged.
    """

    # ──────────────────────────────────────────────────────────────
    #  Initialization (called from compiler __init__)
    # ──────────────────────────────────────────────────────────────

    def _init_libs_subsystem(self) -> None:
        """Initialize the library management subsystem."""
        from .visitors_func import VisitorsFuncMixin as _VFM

        self._libs_config: LibraryConfig = LibraryConfig(
            libs_mode=LinkMode.from_string(
                getattr(self, "libs_mode", "static")
            ),
            dynamic_libs=getattr(self, "dynamic_libs", set()),
        )

        self._libs_registry: LibraryRegistry = LibraryRegistry(self._libs_config)
        self._pure_python_handler: PurePythonHandler = PurePythonHandler()
        self._link_manager: LinkManager = LinkManager(
            self._libs_registry, self._libs_config
        )

        self._libs_search_paths: List[str] = list(
            self._libs_config.effective_search_paths()
        )

        # Pure-Python modules whose functions have been registered in
        # self._ffi_module_symbols so _method_call can resolve them.
        self._pure_python_module_symbols: Dict[str, Set[str]] = {}

        # Keep a reference to the parent-class visitor so we can
        # delegate modules that we don't handle.
        self._parent_visit_Import = _VFM.visit_Import
        self._parent_visit_ImportFrom = _VFM.visit_ImportFrom

        self._register_builtin_modules()

    def _register_builtin_modules(self) -> None:
        """Register all built-in modules in the library registry."""
        math_symbols = {
            "sqrt", "sin", "cos", "exp", "log",
            "pow", "floor", "ceil", "fabs",
        }
        self._libs_registry.register_builtin("math", exported_symbols=math_symbols)
        self._libs_registry.register_builtin("time")
        self._libs_registry.register_builtin("os")
        self._libs_registry.register_builtin("sys")
        self._libs_registry.register_builtin("random")
        self._libs_registry.register_builtin("asyncio")

    # ──────────────────────────────────────────────────────────────
    #  Configuration API
    # ──────────────────────────────────────────────────────────────

    def set_libs_config(self, config: LibraryConfig) -> None:
        self._libs_config = config
        self._libs_registry._config = config
        self._link_manager._config = config
        self._libs_search_paths = list(config.effective_search_paths())

    def add_lib_search_path(self, path: str) -> None:
        abspath = os.path.abspath(path)
        if abspath not in self._libs_search_paths:
            self._libs_search_paths.append(abspath)

    def set_module_link_mode(self, module_name: str, mode: LinkMode) -> None:
        if mode == LinkMode.DYNAMIC:
            self._libs_config.dynamic_libs.add(module_name)
            self._libs_config.static_libs.discard(module_name)
        elif mode == LinkMode.STATIC:
            self._libs_config.static_libs.add(module_name)
            self._libs_config.dynamic_libs.discard(module_name)
        entry = self._libs_registry.get(module_name)
        if entry is not None:
            entry.link_mode = mode

    def set_module_pure_python_strategy(
        self, module_name: str, strategy: PurePythonLinkStrategy
    ) -> None:
        self._libs_config.per_module_strategy[module_name] = strategy
        entry = self._libs_registry.get(module_name)
        if entry is not None and entry.kind == LibraryKind.PURE_PYTHON:
            entry.pure_python_strategy = strategy

    # ──────────────────────────────────────────────────────────────
    #  Overridden import visitors (take precedence via MRO)
    # ──────────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        """Override of VisitorsFuncMixin.visit_Import.

        For each imported module name, try the libs subsystem first:
          1. If the module is already in _ffi_modules (FFI .so) or is
             a known built-in, delegate to the parent visitor.
          2. If a .py file is found, compile it as pure-Python and
             register its symbols so _method_call can reach them.
          3. Otherwise, delegate to the parent visitor.
        """
        handled: list = []
        remaining: list = []

        for alias in node.names:
            module_name = alias.name
            asname = alias.asname if alias.asname else module_name

            # ── Check if already handled by FFI or built-in ──
            if self._is_ffi_or_builtin(module_name):
                remaining.append(alias)
                continue

            # ── Try pure-Python (.py) ──
            if self._try_import_pure_python(module_name, asname):
                handled.append(alias)
                continue

            # ── Not handled by libs → pass to parent ──
            remaining.append(alias)

        # Delegate remaining aliases to VisitorsFuncMixin.visit_Import
        if remaining:
            remaining_node = ast.Import(names=remaining)
            ast.copy_location(remaining_node, node)
            self._parent_visit_Import(self, remaining_node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Override of VisitorsFuncMixin.visit_ImportFrom.

        If the source module is a pure-Python module whose symbols we
        already know, resolve them directly.  Otherwise delegate to the
        parent visitor.
        """
        module_name = node.module if node.module else ""
        if not module_name:
            self._parent_visit_ImportFrom(self, node)
            return

        # If we already compiled this as pure-Python, resolve symbols
        if module_name in self._pure_python_module_symbols:
            self._handle_pure_python_from_import(module_name, node)
            return

        # If it's FFI or built-in, delegate
        if self._is_ffi_or_builtin(module_name):
            self._parent_visit_ImportFrom(self, node)
            return

        # Try to discover & compile as pure-Python first
        py_path = self._find_pure_python(module_name)
        if py_path is not None:
            asname_root = module_name.split(".")[0]
            if self._try_import_pure_python(module_name, asname_root):
                if module_name in self._pure_python_module_symbols:
                    self._handle_pure_python_from_import(module_name, node)
                    return

        # Fall through to parent
        self._parent_visit_ImportFrom(self, node)

    # ──────────────────────────────────────────────────────────────
    #  Import helpers
    # ──────────────────────────────────────────────────────────────

    def _is_ffi_or_builtin(self, module_name: str) -> bool:
        """Check if a module is already handled by FFI or is a known built-in.

        Pure-Python modules that were successfully compiled are NOT
        considered FFI/builtin — they have their own dispatch path
        in _method_call via _pure_python_module_symbols.
        """
        # Pure-Python module — not FFI
        if hasattr(self, '_pure_python_module_symbols') and module_name in self._pure_python_module_symbols:
            return True  # Already handled by libs
        if hasattr(self, "_ffi_modules") and module_name in self._ffi_modules:
            return True
        if module_name in _BUILTIN_MODULES:
            return True
        # Also check if already registered as FFI module symbol
        if hasattr(self, "_ffi_module_symbols") and module_name in self._ffi_module_symbols:
            return True
        return False

    def _try_import_pure_python(
        self, module_name: str, asname: str
    ) -> bool:
        """Try to discover and compile a pure-Python module.

        Returns True if the module was found (even if some functions
        couldn't be compiled and were replaced with stubs).  Returns
        False only if the module is not a .py file at all.
        """
        py_path = self._find_pure_python(module_name)
        if py_path is None:
            return False

        try:
            entry = self._register_pure_python_module(module_name, py_path)
        except (CompileError, FileNotFoundError, SyntaxError) as exc:
            # Compilation failed completely — warn and return False so
            # the parent visitor can try its own handling.
            import warnings
            warnings.warn(
                f"[pylow-libs] Nie udało się skompilować modułu pure Python "
                f"'{module_name}': {exc}.  Delegating to default import handler."
            )
            return False

        # ── Register the module name in the symbol table ──
        self._imported_modules[module_name] = {}

        # Register as module reference (is_ffi_module=True so visit_Name
        # returns FFIModuleValue, which _method_call can dispatch on).
        if not self.sym.exists_local(asname):
            var_info = VarInfo(
                None, I8P, PyType.OBJECT,
                class_name=f"__pure_python_module__{module_name}",
                is_ffi_module=True,
                ffi_module_name=module_name,
            )
            self.sym.define(asname, var_info)

        # ── Register exported symbols ──
        # IMPORTANT: We register in _pure_python_module_symbols FIRST,
        # so that _method_call's pure-Python check intercepts calls
        # BEFORE the FFI path.  We also add to _ffi_module_symbols as
        # a fallback for resolve_ffi_symbol().
        info = self._pure_python_handler.get_module_info(module_name)
        if info is not None:
            symbol_names: Set[str] = set()

            for py_name, mangled in info.exported_functions.items():
                fn = self.functions.get(mangled)
                if fn is not None:
                    self._ffi_symbols[mangled] = fn
                    self._ffi_symbols[py_name] = fn
                    self._imported_modules.setdefault(
                        module_name, {}
                    )[py_name] = fn
                    self.functions[py_name] = fn
                    symbol_names.add(py_name)
                    symbol_names.add(mangled)

            for py_name, mangled in info.exported_globals.items():
                symbol_names.add(py_name)
                symbol_names.add(mangled)

            # Register in _pure_python_module_symbols so that
            # _method_call's pure-Python check intercepts these.
            self._pure_python_module_symbols[module_name] = symbol_names

            # Also register in _ffi_module_symbols as fallback for
            # resolve_ffi_symbol(), but ONLY for the symbol names
            # (not the module name itself, to avoid FFI intercept).
            self._ffi_module_symbols.setdefault(module_name, set()).update(
                symbol_names
            )

        return True

    def _handle_pure_python_from_import(
        self, module_name: str, node: ast.ImportFrom
    ) -> None:
        """Resolve ``from module import name1, name2, ...`` for a
        pure-Python module whose symbols are already registered."""
        info = self._pure_python_handler.get_module_info(module_name)
        if info is None:
            return

        for alias in node.names:
            name = alias.name
            asname = alias.asname if alias.asname else name

            # Try function
            mangled = info.exported_functions.get(name)
            if mangled is not None:
                fn = self.functions.get(mangled)
                if fn is not None:
                    self.functions[asname] = fn
                    var_info = VarInfo(None, fn.type, PyType.OBJECT)
                    self.sym.define(asname, var_info)
                    continue

            # Try global
            mangled_g = info.exported_globals.get(name)
            if mangled_g is not None:
                var_info = VarInfo(
                    None, I8P, PyType.OBJECT,
                    class_name=mangled_g,
                )
                self.sym.define(asname, var_info)
                continue

            # Symbol not found in this module's exports — leave for
            # parent handler (might be a submodule or runtime attr).
            # Create a placeholder so visit_Name doesn't crash.
            full_name = f"__builtin_{module_name}.{name}"
            var_info = VarInfo(
                None, I8P, PyType.OBJECT,
                class_name=full_name,
                is_ffi_module=True,
                ffi_module_name=module_name,
            )
            self.sym.define(asname, var_info)
            self._imported_modules.setdefault(module_name, {})[name] = full_name

    # ──────────────────────────────────────────────────────────────
    #  Module discovery
    # ──────────────────────────────────────────────────────────────

    def _find_native_so(self, module_name: str) -> Optional[str]:
        if hasattr(self, "_find_ffi_so"):
            result = self._find_ffi_so(module_name)
            if result is not None:
                return result
        for base in self._libs_search_paths:
            candidate = os.path.join(base, f"{module_name}.so")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            pkg_dir = os.path.join(base, module_name)
            if os.path.isdir(pkg_dir):
                has_so = any(
                    f.endswith(".so")
                    for root, dirs, files in os.walk(pkg_dir)
                    for f in files
                )
                if has_so:
                    return os.path.abspath(pkg_dir)
        return None

    def _find_pure_python(self, module_name: str) -> Optional[str]:
        return PurePythonHandler.find_module(
            module_name, self._libs_search_paths
        )

    # ──────────────────────────────────────────────────────────────
    #  Module registration & compilation
    # ──────────────────────────────────────────────────────────────

    def _register_native_module(
        self, module_name: str, so_path: str
    ) -> LibraryEntry:
        if os.path.isdir(so_path):
            self.register_ffi_package(module_name, so_path)
        else:
            self.register_ffi_module(module_name, so_path)

        ffi_mod = self.get_ffi_module(module_name)
        exported = set(ffi_mod.exported_symbols.keys()) if ffi_mod else set()
        imported = set(ffi_mod.imported_py_symbols) if ffi_mod else set()

        link_mode = self._libs_config.resolve_link_mode(module_name)
        return self._libs_registry.register_native_so(
            name=module_name,
            so_path=so_path,
            link_mode=link_mode,
            exported_symbols=exported,
            imported_symbols=imported,
        )

    def _register_pure_python_module(
        self, module_name: str, py_path: str
    ) -> LibraryEntry:
        link_mode = self._libs_config.resolve_link_mode(module_name)
        strategy = self._libs_config.resolve_pure_python_strategy(module_name)

        entry = self._libs_registry.register_pure_python(
            name=module_name,
            source_path=py_path,
            link_mode=link_mode,
            strategy=strategy,
        )

        info = self._pure_python_handler.parse_module(module_name, py_path)
        info.strategy = strategy
        info.link_mode = link_mode

        self._pure_python_handler.analyze_exports(info)

        all_exports = set(info.exported_functions.keys()) | set(
            info.exported_globals.keys()
        )
        entry.exported_symbols.update(all_exports)

        # Compile according to strategy
        if strategy == PurePythonLinkStrategy.INLINE:
            self._pure_python_handler.compile_inline(self, info)
        elif strategy == PurePythonLinkStrategy.COMPILED_UNIT:
            self._pure_python_handler.compile_as_unit(self, info)
        elif strategy == PurePythonLinkStrategy.STUB_ONLY:
            self._pure_python_handler.compile_stub_only(self, info)

        if link_mode == LinkMode.STATIC:
            self._libs_registry.mark_linked(module_name)

        return entry

    # ──────────────────────────────────────────────────────────────
    #  Linking API
    # ──────────────────────────────────────────────────────────────

    def compile_and_link(
        self, source: str, output_path: str = "a.out"
    ) -> str:
        ir_text = self.compile(source)
        plan = self._link_manager.create_plan()
        self._link_manager.emit_link_ir(self, plan)
        return self._link_manager.execute_plan(self, plan, output_path)

    def get_link_plan(self) -> LinkPlan:
        return self._link_manager.create_plan()

    def libs_summary(self) -> str:
        lines = [
            "═══ Konfiguracja bibliotek ═══",
            self._libs_config.summary(),
            "",
            "═══ Rejestr bibliotek ═══",
            self._libs_registry.summary(),
        ]
        return "\n".join(lines)
