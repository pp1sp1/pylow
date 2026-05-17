"""Link manager for the py2llvm compiler.

Orchestrates the linking of all registered libraries into the final
output binary.  The link manager uses the library registry to
determine what needs to be linked, the library config to determine
how (static vs. dynamic), and the topological ordering to determine
the correct link sequence.

Key responsibilities:
  1. Determine the link plan (ordered list of link actions).
  2. Emit LLVM IR for static linking (merge modules).
  3. Emit LLVM IR for dynamic linking (dlopen/dlsym init code).
  4. Generate the final link command for the system linker (ld/gcc).
  5. Report unresolved symbols.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import llvmlite.ir as ir
import llvmlite.binding as llvm

from .config import LibraryConfig, LinkMode, PurePythonLinkStrategy
from .registry import LibraryEntry, LibraryKind, LibraryRegistry


class LinkAction(Enum):
    """Type of link action for a single library.

    MERGE_IR:
      Merge the library's LLVM IR into the main module.  Used for
      static linking of pure-Python modules and LLVM bitcode.

    DECLARE_EXTERNAL:
      Emit ``declare external`` stubs and dlopen/dlsym init code.
      Used for dynamic linking of all library types.

    LINK_SO:
      Add the .so path to the system linker command.  Used for
      static linking of native .so libraries (the linker resolves
      symbols at link time).

    LINK_BITCODE:
      Add the .bc path to the LLVM linker.  Used for static linking
      of LLVM bitcode files.

    SKIP:
      The library is already linked or does not need any action.
    """

    MERGE_IR = auto()
    DECLARE_EXTERNAL = auto()
    LINK_SO = auto()
    LINK_BITCODE = auto()
    SKIP = auto()


@dataclass
class LinkStep:
    """A single step in the link plan.

    Attributes:
        library_name: The module name being linked.
        action: The type of link action.
        source_path: Path to the source file (if applicable).
        symbols: Symbols provided by this library.
    """

    library_name: str
    action: LinkAction
    source_path: str = ""
    symbols: Set[str] = field(default_factory=set)


@dataclass
class LinkPlan:
    """Complete plan for linking all registered libraries.

    Attributes:
        steps: Ordered list of link steps.
        so_paths: .so files to pass to the system linker.
        bc_paths: .bc files to pass to the LLVM linker.
        c_stub_sources: C source code for FFI stubs.
        init_functions: Initialization functions to call at startup.
        unresolved_symbols: Symbols that could not be resolved.
    """

    steps: List[LinkStep] = field(default_factory=list)
    so_paths: List[str] = field(default_factory=list)
    bc_paths: List[str] = field(default_factory=list)
    c_stub_sources: Dict[str, str] = field(default_factory=dict)
    init_functions: List[str] = field(default_factory=list)
    unresolved_symbols: Set[str] = field(default_factory=set)


class LinkManager:
    """Orchestrates the linking of all registered libraries.

    The link manager is responsible for producing a complete link plan
    from the library registry and executing that plan to produce a
    working binary.  It supports both static and dynamic linking and
    handles the interaction between pure-Python modules, native .so
    libraries, and LLVM bitcode.
    """

    def __init__(
        self,
        registry: LibraryRegistry,
        config: Optional[LibraryConfig] = None,
    ) -> None:
        self._registry = registry
        self._config = config or registry._config

    # ──────────────────────────────────────────────────────────────
    #  Plan generation
    # ──────────────────────────────────────────────────────────────

    def create_plan(self) -> LinkPlan:
        """Generate a complete link plan from the library registry.

        The plan is ordered topologically (dependencies first) and
        specifies the correct link action for each library based on
        its kind and link mode.

        Returns:
            A LinkPlan with all steps, paths, and diagnostics.
        """
        plan = LinkPlan()

        # Get topological order (dependencies first)
        try:
            order = self._registry.topological_order()
        except RuntimeError as e:
            # Cycle detected — still produce a plan, but warn
            order = list(self._registry._entries.keys())
            plan.unresolved_symbols.add(f"CYCLE_WARNING: {e}")

        for lib_name in order:
            entry = self._registry.get(lib_name)
            if entry is None:
                continue

            if entry.is_linked:
                plan.steps.append(LinkStep(
                    library_name=lib_name,
                    action=LinkAction.SKIP,
                    source_path=entry.source_path,
                    symbols=entry.exported_symbols,
                ))
                continue

            action = self._determine_action(entry)
            step = LinkStep(
                library_name=lib_name,
                action=action,
                source_path=entry.source_path,
                symbols=entry.exported_symbols,
            )
            plan.steps.append(step)

            # Collect paths and init functions
            if action == LinkAction.LINK_SO:
                plan.so_paths.append(entry.source_path)
            elif action == LinkAction.LINK_BITCODE:
                plan.bc_paths.append(entry.source_path)
            elif action == LinkAction.DECLARE_EXTERNAL:
                init_fn = f"__py2llvm_libinit_{lib_name.replace('.', '_')}"
                plan.init_functions.append(init_fn)
                if entry.source_path:
                    plan.so_paths.append(entry.source_path)

        # Check for unresolved symbols
        plan.unresolved_symbols.update(self._registry.unresolved_symbols())

        return plan

    def _determine_action(self, entry: LibraryEntry) -> LinkAction:
        """Determine the link action for a single library entry.

        The action depends on the library kind and link mode:
          - BUILTIN: always SKIP (already in the main module).
          - PURE_PYTHON + STATIC: MERGE_IR.
          - PURE_PYTHON + DYNAMIC: DECLARE_EXTERNAL.
          - NATIVE_SO + STATIC: LINK_SO.
          - NATIVE_SO + DYNAMIC: DECLARE_EXTERNAL.
          - LLVM_BITCODE + STATIC: LINK_BITCODE.
          - LLVM_BITCODE + DYNAMIC: DECLARE_EXTERNAL.

        Args:
            entry: The library entry.

        Returns:
            The appropriate LinkAction.
        """
        if entry.kind == LibraryKind.BUILTIN:
            return LinkAction.SKIP

        if entry.kind == LibraryKind.PURE_PYTHON:
            if entry.link_mode == LinkMode.STATIC:
                return LinkAction.MERGE_IR
            return LinkAction.DECLARE_EXTERNAL

        if entry.kind == LibraryKind.NATIVE_SO:
            if entry.link_mode == LinkMode.STATIC:
                return LinkAction.LINK_SO
            return LinkAction.DECLARE_EXTERNAL

        if entry.kind == LibraryKind.LLVM_BITCODE:
            if entry.link_mode == LinkMode.STATIC:
                return LinkAction.LINK_BITCODE
            return LinkAction.DECLARE_EXTERNAL

        return LinkAction.SKIP

    # ──────────────────────────────────────────────────────────────
    #  Link plan execution
    # ──────────────────────────────────────────────────────────────

    def execute_plan(
        self,
        compiler,
        plan: LinkPlan,
        output_path: str,
    ) -> str:
        """Execute a link plan to produce a final binary.

        This method:
          1. Generates the link command from the plan.
          2. Executes the system linker (gcc/clang) to produce the
             final binary.
          3. Returns the path to the produced binary.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            plan: The link plan to execute.
            output_path: Path for the output binary.

        Returns:
            Absolute path to the produced binary.

        Raises:
            RuntimeError: If the linker fails.
        """
        # Save the LLVM IR to a temporary .ll file
        ir_path = output_path + ".ll"
        compiler.save_ir(ir_path)

        # Build the linker command
        cmd = self._build_link_command(ir_path, output_path, plan)

        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Linker zwrócił błąd (kod {result.returncode}):\n"
                    f"Komenda: {' '.join(cmd)}\n"
                    f"stderr: {result.stderr}"
                )
        except FileNotFoundError:
            raise RuntimeError(
                f"Nie znaleziono linkera. Upewnij się, że gcc lub clang "
                f"jest zainstalowany."
            )

        return os.path.abspath(output_path)

    def _build_link_command(
        self,
        ir_path: str,
        output_path: str,
        plan: LinkPlan,
    ) -> List[str]:
        """Build the system linker command from a link plan.

        Uses gcc or clang as the linker driver.  The command:
          1. Compiles the LLVM IR (.ll) to an object file (.o).
          2. Links the object file with all .so and .bc files.
          3. Produces the final executable.

        Args:
            ir_path: Path to the LLVM IR file.
            output_path: Path for the output binary.
            plan: The link plan.

        Returns:
            The linker command as a list of strings.
        """
        # Choose compiler driver
        cc = os.environ.get("CC", "gcc")

        # For static .so linking, we need to compile the .ll first
        obj_path = output_path + ".o"

        cmd = [cc, "-o", output_path]

        # Add the LLVM IR file (gcc can handle .ll with -x ir)
        # Actually, we need to compile it to .o first using clang/llc
        # Simplified: use clang which can handle .ll directly
        if cc == "gcc":
            # GCC needs object files, not .ll
            # Use llc to compile .ll -> .o
            llc_cmd = ["llc", "-filetype=obj", "-o", obj_path, ir_path]
            try:
                subprocess.run(llc_cmd, capture_output=True, check=True, timeout=30)
            except (FileNotFoundError, subprocess.CalledProcessError):
                # Fallback: try clang
                cc = "clang"

        if cc == "clang":
            cmd = ["clang", "-o", output_path, ir_path]
        else:
            cmd = [cc, "-o", output_path, obj_path]

        # Add .so paths for static linking
        for so_path in plan.so_paths:
            if os.path.isfile(so_path):
                cmd.append(so_path)

        # Add system libraries
        cmd.extend(["-lm", "-ldl", "-lpthread"])

        return cmd

    # ──────────────────────────────────────────────────────────────
    #  LLVM IR generation for linking
    # ──────────────────────────────────────────────────────────────

    def emit_link_ir(
        self,
        compiler,
        plan: LinkPlan,
    ) -> None:
        """Emit LLVM IR for all link steps into the compiler's module.

        This method is called after the main module's IR has been
        generated and before the final output.  It ensures that:
          - All ``declare external`` stubs are present for dynamic libs.
          - All function definitions are merged for static libs.
          - Initialization functions are registered.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            plan: The link plan to execute.
        """
        for step in plan.steps:
            if step.action == LinkAction.SKIP:
                continue

            elif step.action == LinkAction.DECLARE_EXTERNAL:
                # Ensure all exported symbols have declare external stubs
                for sym_name in step.symbols:
                    if sym_name not in compiler.functions:
                        fty = ir.FunctionType(
                            ir.PointerType(ir.IntType(8)),
                            [ir.PointerType(ir.IntType(8))]
                        )
                        fn = ir.Function(compiler.module, fty, name=sym_name)
                        compiler.functions[sym_name] = fn

            elif step.action == LinkAction.MERGE_IR:
                # For pure-Python modules compiled as separate units,
                # the merge has already been done by PurePythonHandler.
                # Mark as linked.
                self._registry.mark_linked(step.library_name)

            elif step.action == LinkAction.LINK_SO:
                # Native .so — already handled by FFI subsystem.
                self._registry.mark_linked(step.library_name)

        # Generate the master initialization function that calls all
        # per-library init functions in topological order.
        if plan.init_functions:
            self._emit_master_init(compiler, plan.init_functions)

    def _emit_master_init(
        self,
        compiler,
        init_fn_names: List[str],
    ) -> None:
        """Generate a master initialization function.

        Creates ``__py2llvm_libs_init`` that calls all per-library
        initialization functions in order.  This function should be
        called at program startup before any library code is invoked.

        Args:
            compiler: The PythonToLLVMCompiler instance.
            init_fn_names: List of init function names to call.
        """
        fty = ir.FunctionType(ir.VoidType(), [])
        master_init = ir.Function(
            compiler.module, fty, name="__py2llvm_libs_init"
        )
        compiler.functions["__py2llvm_libs_init"] = master_init

        entry = master_init.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry)

        for init_name in init_fn_names:
            init_fn = compiler.functions.get(init_name)
            if init_fn is not None:
                builder.call(init_fn, [])

        builder.ret_void()

    # ──────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────

    def link_report(self, plan: LinkPlan) -> str:
        """Generate a human-readable link report.

        Useful for diagnostic output and debugging.

        Args:
            plan: The link plan to report on.

        Returns:
            Multi-line string describing the link plan.
        """
        lines = ["Raport linkowania:"]
        lines.append(f"  Kroki: {len(plan.steps)}")
        lines.append(f"  Ścieżki .so: {len(plan.so_paths)}")
        lines.append(f"  Ścieżki .bc: {len(plan.bc_paths)}")
        lines.append(f"  Funkcje init: {len(plan.init_functions)}")

        for step in plan.steps:
            action_str = step.action.name
            lines.append(
                f"    {step.library_name:20s} → {action_str:20s} "
                f"({len(step.symbols)} symboli)"
            )

        if plan.so_paths:
            lines.append("  Biblioteki .so:")
            for p in plan.so_paths:
                lines.append(f"    {p}")

        if plan.unresolved_symbols:
            lines.append(f"  Nierozwiązane symbole: {len(plan.unresolved_symbols)}")
            for sym in sorted(plan.unresolved_symbols)[:20]:
                lines.append(f"    {sym}")
            if len(plan.unresolved_symbols) > 20:
                lines.append(
                    f"    ... i {len(plan.unresolved_symbols) - 20} więcej"
                )

        return "\n".join(lines)
