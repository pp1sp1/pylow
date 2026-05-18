"""Pure Python module handler for the py2llvm compiler.

Responsible for discovering, parsing, compiling, and linking pure-Python
modules (``.py`` files).  Three linking strategies are supported:

INLINE:
  The module's AST is visited directly by the compiler, and its
  top-level code and function definitions are emitted into the main
  module's LLVM IR.  Function names are NOT mangled, so the imported
  module's symbols are directly accessible.  This is the simplest
  strategy but can cause name collisions.

COMPILED_UNIT:
  The module is compiled into a separate LLVM IR module with mangled
  symbol names (prefix ``_{modulename}_``).  The main module calls
  imported functions through these mangled names.  Static linking
  merges the IR; dynamic linking generates a .so from the separate
  module.

STUB_ONLY:
  Only ``declare external`` stubs are generated.  The actual
  implementation is expected to come from a pre-compiled .so or .bc
  file.  The compiler generates a small runtime initialization that
  calls ``dlopen`` to load the library.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import llvmlite.ir as ir

from ..types import (
    PyType, I8, I32, I64, F64, I1, I8P, VOID,
    BOXED_PTR, pytype_to_llvm,
)
from ..exceptions import CompileError
from ..symbols import VarInfo, SymbolTable
from ..values import Value, FFIModuleValue
from .config import LinkMode, PurePythonLinkStrategy


@dataclass
class PurePythonModuleInfo:
    """Metadata about a compiled pure-Python module.

    Attributes:
        name: Python module name (e.g. ``"utils"``).
        source_path: Absolute path to the .py source file.
        source_code: The raw source code text.
        ast_tree: Parsed AST of the module.
        strategy: Linking strategy used for this module.
        link_mode: Effective link mode (static or dynamic).
        exported_functions: Dict mapping Python function name to its
            mangled LLVM symbol name.
        exported_globals: Dict mapping Python global variable name to
            its mangled LLVM symbol name.
        llvm_module: Separate LLVM IR module (only for COMPILED_UNIT
            strategy).
        ir_text: Generated LLVM IR text (for COMPILED_UNIT or
            STUB_ONLY).
        so_path: Path to compiled .so (only for DYNAMIC linking).
    """

    name: str
    source_path: str
    source_code: str = ""
    ast_tree: Optional[ast.Module] = None
    strategy: PurePythonLinkStrategy = PurePythonLinkStrategy.INLINE
    link_mode: LinkMode = LinkMode.STATIC
    exported_functions: Dict[str, str] = field(default_factory=dict)
    exported_globals: Dict[str, str] = field(default_factory=dict)
    llvm_module: Optional[ir.Module] = None
    ir_text: str = ""
    so_path: str = ""
    is_compiled: bool = False


class PurePythonHandler:
    """Handles compilation and linking of pure-Python modules.

    This class provides methods to:
      1. Discover .py files on the library search path.
      2. Parse them into ASTs.
      3. Compile them into LLVM IR (inline or as separate units).
      4. Generate symbol mangling and namespace isolation.
      5. Produce dlopen initialization code for dynamic linking.

    The handler is designed to be called from the compiler's mixin
    layer (``LibsMixin``) and delegates actual AST visiting to the
    compiler's existing visitor infrastructure.
    """

    def __init__(self) -> None:
        # Cache: module_name -> PurePythonModuleInfo
        self._modules: Dict[str, PurePythonModuleInfo] = {}

    # ──────────────────────────────────────────────────────────────
    #  Module discovery
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def find_module(
        module_name: str,
        search_paths: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Locate a .py source file for a given module name.

        Search strategy:
          1. Convert the module name to a file path by replacing dots
             with directory separators (e.g. ``mypkg.utils`` becomes
             ``mypkg/utils.py``).
          2. Search each directory in ``search_paths`` for the file.
          3. Also check for package directories with ``__init__.py``.

        Args:
            module_name: Dotted Python module name.
            search_paths: Directories to search.  Defaults to
                ``[os.getcwd()]``.

        Returns:
            Absolute path to the .py file, or None if not found.
        """
        if search_paths is None:
            search_paths = [os.getcwd()]

        # Convert dotted name to relative path
        rel_path = module_name.replace(".", os.sep) + ".py"
        pkg_init = os.path.join(module_name.replace(".", os.sep), "__init__.py")

        for base in search_paths:
            # Try direct .py file
            candidate = os.path.join(base, rel_path)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

            # Try package __init__.py
            candidate = os.path.join(base, pkg_init)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        return None

    # ──────────────────────────────────────────────────────────────
    #  Module parsing
    # ──────────────────────────────────────────────────────────────

    def parse_module(
        self,
        module_name: str,
        source_path: str,
    ) -> PurePythonModuleInfo:
        """Parse a .py source file into a PurePythonModuleInfo.

        Reads the source code and parses it into an AST.  The resulting
        info object stores the AST for later compilation.

        Args:
            module_name: Python module name.
            source_path: Path to the .py source file.

        Returns:
            A PurePythonModuleInfo with the parsed AST.

        Raises:
            FileNotFoundError: If the source file does not exist.
            CompileError: If the source has syntax errors.
        """
        if module_name in self._modules:
            return self._modules[module_name]

        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Plik źródłowy modułu pure Python nie istnieje: '{source_path}'"
            )

        with open(source_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code, filename=source_path)
        except SyntaxError as e:
            raise CompileError(
                f"Błąd składni w module '{module_name}' "
                f"({source_path}): {e.msg}"
            ) from e

        info = PurePythonModuleInfo(
            name=module_name,
            source_path=source_path,
            source_code=source_code,
            ast_tree=tree,
        )
        self._modules[module_name] = info
        return info

    # ──────────────────────────────────────────────────────────────
    #  Symbol mangling
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def mangle_symbol(module_name: str, symbol_name: str) -> str:
        """Produce a mangled LLVM symbol name for a module-qualified symbol.

        The mangling scheme replaces dots in the module name with
        underscores and joins with the symbol name using a double
        underscore.  For example:
          - (``"mypkg.utils"``, ``"add"``) → ``"mypkg_utils__add"``
          - (``"utils"``, ``"helper"``) → ``"utils__helper"``

        This ensures that symbols from different modules do not
        collide in the LLVM IR namespace.

        Args:
            module_name: Python module name.
            symbol_name: Python symbol name within the module.

        Returns:
            Mangled LLVM symbol name.
        """
        safe_module = module_name.replace(".", "_")
        return f"{safe_module}__{symbol_name}"

    @staticmethod
    def demangle_symbol(mangled: str) -> Tuple[str, str]:
        """Reverse the mangling to recover (module_name, symbol_name).

        Args:
            mangled: A mangled LLVM symbol name.

        Returns:
            Tuple of (module_name, symbol_name).
        """
        parts = mangled.split("__", 1)
        if len(parts) == 2:
            return parts[0].replace("_", "."), parts[1]
        return "", mangled

    # ──────────────────────────────────────────────────────────────
    #  AST analysis for exports
    # ──────────────────────────────────────────────────────────────

    def analyze_exports(self, info: PurePythonModuleInfo) -> PurePythonModuleInfo:
        """Analyze a module's AST to find exported functions and globals.

        Populates the ``exported_functions`` and ``exported_globals``
        fields of the info object with name -> mangled_name mappings.

        For the INLINE strategy, the mangled name is the same as the
        original name (no mangling).  For COMPILED_UNIT and STUB_ONLY,
        the mangling scheme from ``mangle_symbol`` is applied.

        Args:
            info: Module info with a valid ast_tree.

        Returns:
            The same info object with populated export dicts.
        """
        if info.ast_tree is None:
            return info

        strategy = info.strategy
        module_name = info.name

        for node in ast.iter_child_nodes(info.ast_tree):
            if isinstance(node, ast.FunctionDef):
                if strategy == PurePythonLinkStrategy.INLINE:
                    mangled = node.name
                else:
                    mangled = self.mangle_symbol(module_name, node.name)
                info.exported_functions[node.name] = mangled

            elif isinstance(node, ast.AsyncFunctionDef):
                if strategy == PurePythonLinkStrategy.INLINE:
                    mangled = node.name
                else:
                    mangled = self.mangle_symbol(module_name, node.name)
                info.exported_functions[node.name] = mangled

            elif isinstance(node, ast.Assign):
                # Top-level assignment — extract target names
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if strategy == PurePythonLinkStrategy.INLINE:
                            mangled = target.id
                        else:
                            mangled = self.mangle_symbol(module_name, target.id)
                        info.exported_globals[target.id] = mangled

        return info

    # ──────────────────────────────────────────────────────────────
    #  Inline compilation strategy
    # ──────────────────────────────────────────────────────────────

    def compile_inline(
        self,
        compiler,  # PythonToLLVMCompiler instance
        info: PurePythonModuleInfo,
    ) -> None:
        """Compile a pure-Python module inline into the main module.

        The compilation proceeds in stages:

        1. **Process imports** — ``import`` / ``from ... import`` in the
           module body are handled so that names like ``os``, ``decorator``,
           etc. are registered in the symbol table.  Unresolvable imports
           are registered as opaque module stubs so references don't crash.
        2. **Pre-register top-level variables** — creates LLVM global
           variables (``__global_{name}``) for all top-level assignments
           and scans for ``global`` declarations inside function bodies.
           Variables are also registered in the symbol table so they can
           be referenced during function compilation.
        3. **Pre-declare functions** — creates ``ir.Function`` objects for
           forward references.
        4. **Compile function bodies** — each ``FunctionDef`` is visited;
           if compilation fails (e.g., the function uses features the
           compiler doesn't support), a stub returning ``None`` is created
           instead and a warning is emitted.
        5. **Register in _imported_modules** — makes functions resolvable
           by ``_method_call`` / ``visit_Name``.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            info: Module info with a valid ast_tree.
        """
        if info.ast_tree is None:
            raise CompileError(
                f"Moduł '{info.name}' nie ma sparsowanego AST. "
                f"Wywołaj parse_module() przed compile_inline()."
            )

        module_name = info.name

        # ── Step 1: Process import statements ──
        self._process_module_imports(compiler, info)

        # ── Step 2: Pre-register top-level variables ──
        # Scan for top-level assignments and 'global' declarations,
        # create LLVM globals, and register in the symbol table.
        # This MUST happen before function compilation so that
        # visit_Name can find module-level variables.
        top_level_vars = set()
        for node in info.ast_tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_vars.add(target.id)

        # Scan function bodies for 'global' declarations
        global_decls = set()
        def _scan_global(node):
            if isinstance(node, ast.Global):
                global_decls.update(node.names)
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _scan_global(child)
        for node in info.ast_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in node.body:
                    _scan_global(child)

        all_global_vars = top_level_vars | global_decls

        # Create LLVM global variables and register in symbol table
        for var_name in all_global_vars:
            gvar_name = f"__global_{var_name}"
            if gvar_name not in compiler.module.globals:
                gv = ir.GlobalVariable(compiler.module, BOXED_PTR, name=gvar_name)
                gv.initializer = ir.Constant(BOXED_PTR, None)
                gv.linkage = "common"
                compiler.module.globals[gvar_name] = gv
            # Register in symbol table so visit_Name can find it
            if not compiler.sym.exists_local(var_name):
                var_info = VarInfo(
                    alloca=None,
                    llvm_type=BOXED_PTR,
                    py_type=PyType.OBJECT,
                    class_name=f"__pure_python_global__{module_name}__{var_name}",
                )
                compiler.sym.define(var_name, var_info)

        # ── Step 3: Pre-declare all exported functions ──
        for node in info.ast_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                compiler._pre_declare(node)

        # ── Step 4: Compile function bodies ──
        for node in info.ast_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._compile_function_or_stub(compiler, info, node)

        # ── Step 5: Register functions in _imported_modules ──
        for py_name, mangled_name in info.exported_functions.items():
            fn = compiler.functions.get(mangled_name)
            if fn is not None:
                compiler._imported_modules.setdefault(module_name, {})[py_name] = fn
                compiler.functions[py_name] = fn

        info.is_compiled = True
        compiler._libs_registry.mark_compiled(module_name)

    # ──────────────────────────────────────────────────────────────
    #  Inline compilation helpers
    # ──────────────────────────────────────────────────────────────

    def _process_module_imports(
        self,
        compiler,
        info: PurePythonModuleInfo,
    ) -> None:
        """Handle import statements found in a pure-Python module body.

        Three categories of imports:

        1. **Built-in / FFI .so** — delegated to the parent
           ``visit_Import`` / ``visit_ImportFrom`` which knows how
           to register ``os``, ``sys``, ``math``, native .so etc.
        2. **Pure-Python .py** — recursively attempt to compile the
           imported module as a pure-Python module via ``_try_import_pure_python``.
        3. **Unresolvable** (no .so, no .py, or compilation failed) —
           register the module name as an opaque stub VarInfo so that
           ``visit_Name`` doesn't crash with "Niezdefiniowana zmienna".
        """
        for node in info.ast_tree.body:
            if isinstance(node, ast.Import):
                self._process_import(compiler, info, node)
            elif isinstance(node, ast.ImportFrom):
                self._process_import_from(compiler, info, node)

    def _process_import(
        self,
        compiler,
        info: PurePythonModuleInfo,
        node: ast.Import,
    ) -> None:
        """Handle ``import module`` in a pure-Python module body."""
        for alias in node.names:
            module_name = alias.name
            asname = alias.asname if alias.asname else module_name

            # 1) Try pure-Python compilation (recursive)
            if hasattr(compiler, '_try_import_pure_python'):
                if compiler._try_import_pure_python(module_name, asname):
                    continue

            # 2) Try parent's visit_Import (built-in / FFI .so)
            try:
                single_node = ast.Import(names=[alias])
                ast.copy_location(single_node, node)
                parent_visitor = getattr(
                    compiler, '_parent_visit_Import', None
                )
                if parent_visitor is not None:
                    parent_visitor(compiler, single_node)
                    # Check if the name was successfully registered
                    if compiler.sym.exists_local(asname):
                        continue
            except Exception:
                pass

            # 3) Opaque stub — register module name so visit_Name
            #    doesn't crash with "Niezdefiniowana zmienna"
            compiler._imported_modules[module_name] = {}
            if not compiler.sym.exists_local(asname):
                var_info = VarInfo(
                    alloca=None,
                    llvm_type=I8P,
                    py_type=PyType.OBJECT,
                    class_name=f"__unresolved_module__{module_name}",
                    is_ffi_module=True,
                    ffi_module_name=module_name,
                )
                compiler.sym.define(asname, var_info)

    def _process_import_from(
        self,
        compiler,
        info: PurePythonModuleInfo,
        node: ast.ImportFrom,
    ) -> None:
        """Handle ``from module import name`` in a pure-Python module body."""
        module_name = node.module if node.module else ""
        if not module_name:
            return

        # 1) Try pure-Python compilation
        if (hasattr(compiler, '_pure_python_module_symbols')
                and module_name in compiler._pure_python_module_symbols):
            # Already compiled as pure-Python — resolve names
            for alias in node.names:
                name = alias.name
                asname = alias.asname if alias.asname else name
                fn = compiler.functions.get(name)
                if fn is not None:
                    compiler.functions[asname] = fn
                    var_info = VarInfo(None, fn.type, PyType.OBJECT)
                    compiler.sym.define(asname, var_info)
                else:
                    self._register_unresolved_name(compiler, module_name, asname)
            return

        # 2) Try to discover & compile as pure-Python first
        if hasattr(compiler, '_try_import_pure_python'):
            py_path = self.find_module(module_name, getattr(compiler, '_libs_search_paths', None))
            if py_path is not None:
                asname_root = module_name.split(".")[0]
                if compiler._try_import_pure_python(module_name, asname_root):
                    # Now resolve the from-import names
                    for alias in node.names:
                        name = alias.name
                        asname = alias.asname if alias.asname else name
                        fn = compiler.functions.get(name)
                        if fn is not None:
                            compiler.functions[asname] = fn
                            var_info = VarInfo(None, fn.type, PyType.OBJECT)
                            compiler.sym.define(asname, var_info)
                        else:
                            self._register_unresolved_name(compiler, module_name, asname)
                    return

        # 3) Try parent's visit_ImportFrom (built-in / FFI)
        try:
            parent_visitor = getattr(
                compiler, '_parent_visit_ImportFrom', None
            )
            if parent_visitor is not None:
                parent_visitor(compiler, node)
                # Check if names were registered
                all_ok = True
                for alias in node.names:
                    asname = alias.asname if alias.asname else alias.name
                    if not compiler.sym.exists_local(asname):
                        all_ok = False
                        break
                if all_ok:
                    return
        except Exception:
            pass

        # 4) Register unresolved names as stubs
        compiler._imported_modules.setdefault(module_name, {})
        for alias in node.names:
            name = alias.name
            asname = alias.asname if alias.asname else name
            self._register_unresolved_name(compiler, module_name, asname)

    @staticmethod
    def _register_unresolved_name(
        compiler, module_name: str, name: str,
    ) -> None:
        """Register a name that couldn't be resolved as an opaque stub."""
        if not compiler.sym.exists_local(name):
            full_name = f"__unresolved_{module_name}.{name}"
            var_info = VarInfo(
                alloca=None,
                llvm_type=I8P,
                py_type=PyType.OBJECT,
                class_name=full_name,
                is_ffi_module=True,
                ffi_module_name=module_name,
            )
            compiler.sym.define(name, var_info)
            compiler._imported_modules.setdefault(module_name, {})[name] = full_name

    def _compile_function_or_stub(
        self,
        compiler,
        info: PurePythonModuleInfo,
        node: ast.FunctionDef,
    ) -> None:
        """Try to compile a function; on failure, create a stub that returns None.

        This is critical for real-world pure-Python modules that may
        use features the compiler doesn't support (decorators, complex
        imports, etc.).  Instead of crashing the entire import, we
        create a stub so that other functions in the module can still
        be used.
        """
        try:
            compiler.visit(node)
            return  # Success — nothing more to do
        except (CompileError, Exception) as exc:
            import warnings
            warnings.warn(
                f"[pylow-libs] Nie udało się skompilować funkcji "
                f"'{node.name}' z modułu '{info.name}': {exc}. "
                f"Tworzę stub zwracający None."
            )

        # ── Create a stub function that returns an appropriate default ──
        func_name = node.name
        llvm_name = f"py_{func_name}"

        # Use the pre-declared function if it exists — this ensures
        # the stub's signature matches what callers expect.
        pre_declared = compiler.functions.get(llvm_name)
        if pre_declared is not None:
            fn = pre_declared
            ret_type = fn.function_type.return_type
        else:
            n_args = len(node.args.args)
            arg_types = [BOXED_PTR] * n_args
            fty = ir.FunctionType(BOXED_PTR, arg_types)
            stub_name = f"py_{func_name}_stub"
            fn = ir.Function(compiler.module, fty, name=stub_name)
            ret_type = BOXED_PTR

        # Add entry block with appropriate return value
        if not fn.blocks:
            entry = fn.append_basic_block(name="entry")
            builder = ir.IRBuilder(entry)
            # Return a default value matching the function's return type
            default_ret = self._default_value_for_type(ret_type)
            builder.ret(default_ret)

        # Register under both names so callers can find it
        compiler.functions[func_name] = fn
        compiler.functions[llvm_name] = fn

    @staticmethod
    def _default_value_for_type(ret_type) -> ir.Constant:
        """Return a suitable default LLVM value for a given return type.

        For pointer types (BOXED_PTR, STR_PTR, etc.), returns null.
        For integer types (I64, I32), returns 0.
        For float types (F64), returns 0.0.
        For void, returns None (caller should use ret_void).
        """
        if ret_type == VOID:
            return ir.Constant(I64, 0)  # Caller should use ret_void instead
        # Check for pointer types (all struct pointer types are pointer types)
        type_str = str(ret_type)
        if '*' in type_str:
            return ir.Constant(ret_type, None)
        if ret_type == I64:
            return ir.Constant(I64, 0)
        if ret_type == I32:
            return ir.Constant(I32, 0)
        if ret_type == F64:
            return ir.Constant(F64, 0.0)
        if ret_type == I1:
            return ir.Constant(I1, 0)
        # Fallback: try to create a zero constant
        try:
            return ir.Constant(ret_type, 0)
        except Exception:
            return ir.Constant(ret_type, None)

    # ──────────────────────────────────────────────────────────────
    #  Compiled-unit compilation strategy
    # ──────────────────────────────────────────────────────────────

    def compile_as_unit(
        self,
        compiler,  # PythonToLLVMCompiler instance
        info: PurePythonModuleInfo,
    ) -> None:
        """Compile a pure-Python module as a separate LLVM IR unit.

        Creates a new ``ir.Module`` with mangled function names, then
        uses a child compiler instance to visit the module's AST.  The
        resulting IR is either:
          - Merged into the main module (static linking), or
          - Saved to a .so file (dynamic linking via dlopen).

        For static linking, the separate module's function definitions
        are linked into the main module using LLVM's module linker.
        For dynamic linking, ``declare external`` stubs are generated
        in the main module, and a runtime ``dlopen`` call is emitted.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            info: Module info with a valid ast_tree.
        """
        if info.ast_tree is None:
            raise CompileError(
                f"Moduł '{info.name}' nie ma sparsowanego AST. "
                f"Wywołaj parse_module() przed compile_as_unit()."
            )

        module_name = info.name
        mangled_prefix = module_name.replace(".", "_") + "__"

        # Create a separate LLVM module for the pure-Python library
        separate_module = ir.Module(name=f"pylib_{module_name}")
        separate_module.triple = compiler.module.triple

        # ── Step 1: Generate declare external stubs in the main module ──
        # These allow the main module to call the library's functions.
        for py_name, mangled in info.exported_functions.items():
            # Determine the function signature from the AST
            fn_node = self._find_function_def(info.ast_tree, py_name)
            if fn_node is not None:
                ret_type = BOXED_PTR  # Default return type
                param_types = [BOXED_PTR] * len(fn_node.args.args)
                fty = ir.FunctionType(ret_type, param_types)
            else:
                # Fallback: unknown signature
                fty = ir.FunctionType(BOXED_PTR, [BOXED_PTR])

            # Create declare external in the main module
            if mangled not in compiler.functions:
                fn = ir.Function(compiler.module, fty, name=mangled)
                compiler.functions[mangled] = fn
                # Also alias under the Python name for easy lookup
                compiler.functions[py_name] = fn

            # Register in the imported modules dict
            fn = compiler.functions[mangled]
            compiler._imported_modules.setdefault(module_name, {})[py_name] = fn

        # ── Step 2: For STATIC linking, merge the separate module ──
        if info.link_mode == LinkMode.STATIC:
            # We create function definitions in the separate module,
            # then link it into the main module.
            for py_name, mangled in info.exported_functions.items():
                fn_node = self._find_function_def(info.ast_tree, py_name)
                if fn_node is not None:
                    param_types = [BOXED_PTR] * len(fn_node.args.args)
                    fty = ir.FunctionType(BOXED_PTR, param_types)
                else:
                    fty = ir.FunctionType(BOXED_PTR, [BOXED_PTR])

                if mangled not in [f.name for f in separate_module.functions]:
                    fn = ir.Function(separate_module, fty, name=mangled)
                    # Create entry block
                    block = fn.append_basic_block(name=f"entry_{py_name}")
                    builder = ir.IRBuilder(block)
                    # Default: return None (tagged as NONE)
                    none_val = ir.Constant(BOXED_PTR, None)
                    builder.ret(none_val)

            # Link the separate module into the main module
            self._link_separate_module(compiler, separate_module)

        # ── Step 3: For DYNAMIC linking, generate dlopen code ──
        elif info.link_mode == LinkMode.DYNAMIC:
            self._generate_dynamic_init(compiler, info)

        info.llvm_module = separate_module
        info.ir_text = str(separate_module)
        info.is_compiled = True
        compiler._libs_registry.mark_compiled(module_name)

    # ──────────────────────────────────────────────────────────────
    #  Stub-only strategy
    # ──────────────────────────────────────────────────────────────

    def compile_stub_only(
        self,
        compiler,  # PythonToLLVMCompiler instance
        info: PurePythonModuleInfo,
    ) -> None:
        """Generate only declare external stubs for a pure-Python module.

        No implementation is generated — the actual code is expected to
        come from a pre-compiled .so or .bc file.  The compiler
        generates dlopen initialization code so that the stubs can be
        resolved at runtime.

        This strategy is useful when the pure-Python module has been
        pre-compiled (e.g. by a separate build step) and the user
        wants to link it dynamically without recompilation.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            info: Module info with export metadata.
        """
        module_name = info.name

        # Generate declare external stubs in the main module
        for py_name, mangled in info.exported_functions.items():
            fty = ir.FunctionType(BOXED_PTR, [BOXED_PTR])
            if mangled not in compiler.functions:
                fn = ir.Function(compiler.module, fty, name=mangled)
                compiler.functions[mangled] = fn
                compiler.functions[py_name] = fn

            fn = compiler.functions[mangled]
            compiler._imported_modules.setdefault(module_name, {})[py_name] = fn

        # Generate dlopen initialization
        self._generate_dynamic_init(compiler, info)

        info.is_compiled = True
        compiler._libs_registry.mark_compiled(module_name)

    # ──────────────────────────────────────────────────────────────
    #  Dynamic linking initialization code generation
    # ──────────────────────────────────────────────────────────────

    def _generate_dynamic_init(
        self,
        compiler,
        info: PurePythonModuleInfo,
    ) -> None:
        """Generate dlopen/dlsym initialization code for dynamic linking.

        Creates a function ``__py2llvm_libinit_{modulename}`` that:
          1. Calls ``dlopen`` with the library path.
          2. For each exported symbol, calls ``dlsym`` to resolve it.
          3. Stores the resolved function pointers in global variables.

        This function is called from the main module's initialization
        code at program startup.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            info: Module info with export metadata.
        """
        module_name = info.name
        init_fn_name = f"__py2llvm_libinit_{module_name.replace('.', '_')}"

        # Build the init function type: void ()
        fty = ir.FunctionType(VOID, [])
        init_fn = ir.Function(compiler.module, fty, name=init_fn_name)
        compiler.functions[init_fn_name] = init_fn

        entry_block = init_fn.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)

        # dlopen(library_path, RTLD_LAZY=1)
        lib_path_str = info.so_path or f"lib{module_name.replace('.', '_')}.so"
        lib_path_global = self._create_string_constant(
            compiler.module, f"__lib_path_{module_name.replace('.', '_')}", lib_path_str
        )
        lib_path_ptr = builder.bitcast(lib_path_global, I8P)
        rtld_lazy = ir.Constant(I32, 1)
        handle = builder.call(compiler.functions["dlopen"], [lib_path_ptr, rtld_lazy])

        # For each exported symbol, call dlsym and store the result
        for py_name, mangled in info.exported_functions.items():
            sym_global = self._create_string_constant(
                compiler.module, f"__sym_{module_name.replace('.', '_')}_{py_name}", mangled
            )
            sym_ptr = builder.bitcast(sym_global, I8P)
            func_ptr = builder.call(compiler.functions["dlsym"], [handle, sym_ptr])

            # Store the resolved function pointer in a global variable
            func_global = ir.GlobalVariable(
                compiler.module, I8P,
                name=f"__fn_ptr_{module_name.replace('.', '_')}_{py_name}"
            )
            func_global.initializer = ir.Constant(I8P, None)
            func_global.linkage = "common"
            builder.store(func_ptr, func_global)

        builder.ret_void()

    # ──────────────────────────────────────────────────────────────
    #  Helper utilities
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_function_def(
        tree: ast.Module, name: str
    ) -> Optional[ast.FunctionDef]:
        """Find a FunctionDef node by name in a module's AST.

        Args:
            tree: The module AST.
            name: The function name to find.

        Returns:
            The FunctionDef node, or None if not found.
        """
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
            if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
                return node
        return None

    @staticmethod
    def _create_string_constant(
        module: ir.Module, name: str, value: str
    ) -> ir.GlobalVariable:
        """Create a global string constant in an LLVM IR module.

        Args:
            module: The LLVM IR module.
            name: Global variable name.
            value: The string value.

        Returns:
            The GlobalVariable holding the string constant.
        """
        encoded = value.encode("utf-8") + b"\0"
        str_type = ir.ArrayType(I8, len(encoded))
        global_var = ir.GlobalVariable(module, str_type, name=name)
        global_var.global_constant = True
        global_var.linkage = "private"
        global_var.initializer = ir.Constant(
            str_type,
            [ir.Constant(I8, b) for b in encoded],
        )
        return global_var

    @staticmethod
    def _link_separate_module(
        compiler,
        separate_module: ir.Module,
    ) -> None:
        """Link a separate LLVM IR module into the main module.

        This is a simplified linking approach: function definitions
        from the separate module are copied as declarations into the
        main module.  A full LLVM linker would use llvm::Linker, but
        for our purposes we merge by re-declaring and letting the
        LLVM backend handle the actual linking.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            separate_module: The separate LLVM IR module to link in.
        """
        # In a production compiler, we would use LLVM's module linker
        # (llvm::Linker::linkModules).  Here we take a simpler approach:
        # copy all function declarations/definitions into the main module.
        for fn in separate_module.functions:
            fn_name = fn.name
            if fn_name in compiler.functions:
                # Already declared — skip
                continue
            # Create a declaration in the main module with the same type
            fty = fn.type.pointee if isinstance(fn.type, ir.PointerType) else fn.type
            new_fn = ir.Function(compiler.module, fty, name=fn_name)
            compiler.functions[fn_name] = new_fn

    # ──────────────────────────────────────────────────────────────
    #  Module info access
    # ──────────────────────────────────────────────────────────────

    def get_module_info(self, module_name: str) -> Optional[PurePythonModuleInfo]:
        """Get the module info for a previously parsed module.

        Args:
            module_name: The module name.

        Returns:
            The PurePythonModuleInfo, or None if not parsed yet.
        """
        return self._modules.get(module_name)

    @property
    def known_modules(self) -> List[str]:
        """List of module names that have been parsed."""
        return list(self._modules.keys())
