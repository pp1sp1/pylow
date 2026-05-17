"""Main compiler class composing all functional mixins.

PythonToLLVMCompiler inherits from all mixin classes to provide
a complete Python -> LLVM IR compilation pipeline, including FFI
support for native .so libraries.
"""

from __future__ import annotations

import ast
import sys
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple, Set

import llvmlite.ir as ir
import llvmlite.binding as llvm

from .types import (
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
from .exceptions import CompileError
from .symbols import VarInfo, SymbolTable
from .values import Value
from .type_analyzer import StaticTypeAnalyzer
from .ffi import FFIModule, FFISignatureDB
from .mixins import (
    ExternalDeclarationsMixin,
    ArcGcMixin,
    StringsMixin,
    ListsMixin,
    DictsMixin,
    IteratorsMixin,
    BuiltinsMixin,
    TypeConversionsMixin,
    BoxingMixin,
    PrintingMixin,
    DynamicOpsMixin,
    VisitorsExprMixin,
    VisitorsStmtMixin,
    VisitorsFuncMixin,
    VisitorsCallMixin,
    VisitorsMiscMixin,
    AsyncRuntimeMixin,
)


class PythonToLLVMCompiler(
    ExternalDeclarationsMixin,
    ArcGcMixin,
    StringsMixin,
    ListsMixin,
    DictsMixin,
    IteratorsMixin,
    BuiltinsMixin,
    TypeConversionsMixin,
    BoxingMixin,
    PrintingMixin,
    DynamicOpsMixin,
    VisitorsExprMixin,
    VisitorsStmtMixin,
    VisitorsFuncMixin,
    VisitorsCallMixin,
    VisitorsMiscMixin,
    AsyncRuntimeMixin,
    ast.NodeVisitor,
):
    """Python to LLVM IR compiler with dynamic typing support (v4) + FFI.

    Compiles a subset of Python 3 to LLVM IR. Local variables without
    type annotations are stored as BOXED_PTR, allowing type changes
    during their lifetime (x = 5; x = "ok").

    Features:
    - Dynamic typing via boxed values with runtime tag dispatch
    - ARC (Atomic Reference Counting) with cycle collector
    - Full string, list, dict, set, and tuple runtime support
    - Class/OOP with method resolution
    - Exception handling
    - Generator/iterator support
    - String method implementations
    - FFI: Zero-overhead native .so library integration
      - Direct ``declare external`` in LLVM IR for .so symbols
      - Compile-time type marshaling based on signature database
      - Minimal C stub generation for CPython Py* dependencies

    Args:
        module_name: Name for the LLVM IR module.
        libs_mode: Library linking mode ("static" or "dynamic").
        dynamic_libs: Set of library names to link dynamically.
    """

    def __init__(
        self,
        module_name: str = "py_module",
        libs_mode: str = "static",
        dynamic_libs: Optional[set] = None,
    ) -> None:
        self.module = ir.Module(name=module_name)
        self.module.triple = "x86_64-pc-linux-gnu"

        # Library configuration
        self.libs_mode: str = libs_mode  # "static" or "dynamic"
        self.dynamic_libs: set = dynamic_libs or set()  # set of library names

        self.builder: Optional[ir.IRBuilder] = None
        self.current_func: Optional[ir.Function] = None
        self.sym: SymbolTable = SymbolTable()
        self.functions: Dict[str, ir.Function] = {}
        self._loop_exit_stack: List[ir.Block] = []
        self._loop_cond_stack: List[ir.Block] = []
        self._loop_continue_stack: List[ir.Block] = []
        self._str_cache: Dict[str, ir.GlobalVariable] = {}
        self._str_cache_objs: Dict[str, ir.GlobalVariable] = {}  # String object interning
        # Exception handling
        self._exc_handler_stack: List[Dict] = []  # Stack of exception handlers
        # Module/import handling
        self._imported_modules: Dict[str, Dict] = {}  # module_name -> {func_name: llvm_function}

        # Function inlining support
        self._function_ast: Dict[str, ast.FunctionDef] = {}  # name -> AST node
        self._compiled_funcs: Dict[str, ir.Function] = {}  # name -> LLVM function
        self._inline_threshold: int = 10  # Max statements for inlining

        # Class stack (for super())
        self._class_stack: List[Dict] = []

        # Compiled classes dictionary
        self._compiled_classes: Dict[str, ir.Value] = {}

        # Global/nonlocal variables in current scope
        self._global_vars: set = set()
        self._nonlocal_vars: set = set()

        # Break flag for loops with else clause
        self._current_break_flag: Optional[ir.AllocaInstr] = None

        # Finally stack for return-through-finally
        self._finally_stack: List[Dict] = []

        # Generator support
        self._is_generator: bool = False
        self._generator_list: Optional[Value] = None

        # NAPRAWA: Inicjalizacja inferred_static_types w __init__
        # Bez tego atrybut nie istnieje przy wizytacji ciał metod klasowych,
        # co powoduje AttributeError (CRASH w test_21_import_pylibs.py).
        self.inferred_static_types: Dict[str, Set[PyType]] = {}

        # ═══════════════════════════════════════════════════════════
        #  FFI Subsystem
        # ═══════════════════════════════════════════════════════════
        # Maps module_name -> FFIModule for all loaded native libraries.
        self._ffi_modules: Dict[str, FFIModule] = {}
        # Maps symbol_name -> ir.Function for all FFI-declared symbols.
        self._ffi_symbols: Dict[str, ir.Function] = {}
        # Signature database for type marshaling at call sites.
        self._ffi_sigdb: FFISignatureDB = FFISignatureDB()
        # Maps module_name -> set of symbol names (for module-qualified access).
        self._ffi_module_symbols: Dict[str, Set[str]] = {}
        # Tracks .so files that need stub generation and linking.
        self._ffi_so_paths: List[str] = []
        # C stub source code to compile and link (populated during compile()).
        self._ffi_stub_sources: Dict[str, str] = {}  # module_name -> C source

        # Initialize all runtime components
        self._declare_externals()
        self._get_or_create_arc_funcs()
        self._init_builtin_modules()
        self._ensure_dict_funcs()
        self._declare_async_runtime()

    # ──────────────────────────────────────────────────────────────
    #  Visitor dispatcher
    # ──────────────────────────────────────────────────────────────

    def visit(self, node: ast.AST) -> Optional[Value]:
        """Dispatch to the appropriate visitor method for the given AST node.

        Args:
            node: The AST node to visit.

        Returns:
            A Value for expression nodes, or None for statement nodes.
        """
        method = "visit_" + type(node).__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Handle unsupported AST node types.

        Args:
            node: The unsupported AST node.

        Raises:
            CompileError: Always, with the node type name.
        """
        raise CompileError(
            f"Nieobsługiwany węzeł AST: {type(node).__name__}",
            node if hasattr(node, "lineno") else None,
        )

    # ──────────────────────────────────────────────────────────────
    #  FFI Public API
    # ──────────────────────────────────────────────────────────────

    def register_ffi_module(self, module_name: str, so_path: str) -> None:
        """Register a native .so library as an FFI module.

        Analyzes the .so file, generates LLVM IR external declarations
        for its exported symbols, and records it for C stub generation
        and linking.

        Args:
            module_name: Python module name (e.g., "markupsafe").
            so_path: Filesystem path to the .so file.
        """
        if module_name in self._ffi_modules:
            return  # Already registered

        # Analyze the .so
        ffi_mod = FFIModule.from_file(so_path, link_mode=self.libs_mode)
        self._ffi_modules[module_name] = ffi_mod
        self._ffi_so_paths.append(so_path)

        # Generate LLVM IR external declarations for exported symbols
        declarations = ffi_mod.get_llvm_declarations(self.module)
        symbol_names = set()
        for sym_name, fn in declarations.items():
            self._ffi_symbols[sym_name] = fn
            self.functions[sym_name] = fn  # Also register in global functions dict
            symbol_names.add(sym_name)

        self._ffi_module_symbols[module_name] = symbol_names

        # Generate C stubs if the .so imports Py* symbols
        if ffi_mod.imported_py_symbols:
            stub_c = ffi_mod.generate_c_stubs()
            self._ffi_stub_sources[module_name] = stub_c

    def register_ffi_package(self, module_name: str, pkg_dir: str) -> None:
        """Register a package directory containing multiple .so files.

        Analyzes all .so files in the package directory and aggregates
        their exported symbols and Py* imports.

        Args:
            module_name: Python package name (e.g., "numpy").
            pkg_dir: Filesystem path to the package directory.
        """
        if module_name in self._ffi_modules:
            return

        ffi_mod = FFIModule.from_package(pkg_dir, link_mode=self.libs_mode)
        self._ffi_modules[module_name] = ffi_mod
        self._ffi_so_paths.append(pkg_dir)

        # Generate declarations
        declarations = ffi_mod.get_llvm_declarations(self.module)
        symbol_names = set()
        for sym_name, fn in declarations.items():
            self._ffi_symbols[sym_name] = fn
            self.functions[sym_name] = fn
            symbol_names.add(sym_name)

        self._ffi_module_symbols[module_name] = symbol_names

        if ffi_mod.imported_py_symbols:
            stub_c = ffi_mod.generate_c_stubs()
            self._ffi_stub_sources[module_name] = stub_c

    def get_ffi_module(self, module_name: str) -> Optional[FFIModule]:
        """Get the FFIModule for a registered native module."""
        return self._ffi_modules.get(module_name)

    def is_ffi_module(self, module_name: str) -> bool:
        """Check if a module name corresponds to a registered FFI module."""
        return module_name in self._ffi_modules

    def resolve_ffi_symbol(self, module_name: str, symbol_name: str) -> Optional[ir.Function]:
        """Resolve a module-qualified FFI symbol.

        Args:
            module_name: The FFI module name.
            symbol_name: The symbol name within the module.

        Returns:
            The LLVM IR Function for the symbol, or None if not found.
        """
        # Try module-qualified name first (e.g., "escape" in markupsafe)
        if module_name in self._ffi_module_symbols:
            if symbol_name in self._ffi_module_symbols[module_name]:
                return self._ffi_symbols.get(symbol_name)

        # Try as a global FFI symbol
        return self._ffi_symbols.get(symbol_name)

    def get_ffi_stub_sources(self) -> Dict[str, str]:
        """Get all generated C stub sources.

        Returns:
            Dict mapping module_name -> C source code string.
        """
        return dict(self._ffi_stub_sources)

    def get_ffi_so_paths(self) -> List[str]:
        """Get all .so paths that need to be linked."""
        return list(self._ffi_so_paths)

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def compile(self, source: str) -> str:
        """Compile Python source code to LLVM IR.

        Args:
            source: Python source code string.

        Returns:
            The generated LLVM IR as a string.
        """
        tree = ast.parse(source)
        self.visit(tree)
        return str(self.module)

    def save_ir(self, filename: str) -> None:
        """Save the generated LLVM IR to a file.

        Args:
            filename: Path to the output file.
        """
        with open(filename, "w") as f:
            f.write(str(self.module))

    def verify(self, ir_text: Optional[str] = None) -> bool:
        """Verify the generated LLVM IR for correctness.

        Args:
            ir_text: Optional IR text to verify. If None, uses the current module.

        Returns:
            True if the IR is valid, False otherwise.
        """
        if ir_text is None:
            ir_text = str(self.module)
        try:
            mod = llvm.parse_assembly(ir_text)
            mod.verify()
            return True
        except Exception:
            return False
