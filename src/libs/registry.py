"""Library registry for the py2llvm compiler.

Maintains a central catalog of all known libraries (pure Python, native
.so, and LLVM bitcode) and their metadata.  The registry is populated
during compilation as ``import`` statements are resolved, and it drives
the linking phase by providing the linker with a complete bill of
materials.

Design principles:
  - A library is identified by its Python module name (e.g. ``"utils"``).
  - Each entry stores the library type, source path, effective link
    mode, and any module-specific metadata.
  - The registry is intentionally separate from the compiler so that it
    can be inspected, serialized, and reused across compilations.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .config import LibraryConfig, LinkMode, PurePythonLinkStrategy


class LibraryKind(enum.Enum):
    """Classification of a library by its source language / format.

    PURE_PYTHON:
      A module written entirely in Python (.py file).  The compiler
      parses the source and generates LLVM IR from it.

    NATIVE_SO:
      A pre-compiled shared object (.so / .dll / .dylib).  The compiler
      uses the FFI subsystem to analyze it and generate ``declare
      external`` stubs.

    LLVM_BITCODE:
      A pre-compiled LLVM bitcode file (.bc).  The compiler links it
      directly using LLVM's linker.

    BUILTIN:
      A built-in module provided by the compiler runtime (e.g. math,
      time, os, sys).  These modules are always statically linked
      because their implementations are LLVM intrinsics or hard-coded
      LLVM IR.
    """

    PURE_PYTHON = "pure_python"
    NATIVE_SO = "native_so"
    LLVM_BITCODE = "llvm_bitcode"
    BUILTIN = "builtin"


@dataclass
class LibraryEntry:
    """Metadata for a single registered library.

    Attributes:
        name: Python module name (e.g. ``"utils"``, ``"markupsafe"``).
        kind: The type of library.
        source_path: Filesystem path to the source file (.py, .so, .bc).
            Empty string for built-in modules.
        link_mode: Effective linking mode (static or dynamic).
        pure_python_strategy: Linking strategy for pure-Python modules.
            None for non-pure-Python libraries.
        exported_symbols: Set of symbol names this library provides.
        imported_symbols: Set of symbol names this library requires.
        is_compiled: Whether the library's LLVM IR has been generated.
        is_linked: Whether the library has been linked into the output.
        extra: Additional metadata (e.g. CPython version, architecture).
    """

    name: str
    kind: LibraryKind
    source_path: str = ""
    link_mode: LinkMode = LinkMode.STATIC
    pure_python_strategy: Optional[PurePythonLinkStrategy] = None
    exported_symbols: Set[str] = field(default_factory=set)
    imported_symbols: Set[str] = field(default_factory=set)
    is_compiled: bool = False
    is_linked: bool = False
    extra: Dict = field(default_factory=dict)

    @property
    def is_pure_python(self) -> bool:
        """Check if this library is a pure Python module."""
        return self.kind == LibraryKind.PURE_PYTHON

    @property
    def is_native(self) -> bool:
        """Check if this library is a native shared object."""
        return self.kind == LibraryKind.NATIVE_SO

    @property
    def is_builtin(self) -> bool:
        """Check if this library is a built-in module."""
        return self.kind == LibraryKind.BUILTIN


class LibraryRegistry:
    """Central catalog of all libraries known to the compiler.

    The registry is populated incrementally as imports are resolved.
    It provides lookup, iteration, and dependency-ordering services
    that are used by the linker to produce a correctly ordered link
    sequence.

    Thread safety:
      The registry is NOT thread-safe.  If parallel compilation is
      desired, external synchronization is required.
    """

    def __init__(self, config: Optional[LibraryConfig] = None) -> None:
        self._entries: Dict[str, LibraryEntry] = {}
        self._config: LibraryConfig = config or LibraryConfig()
        # Dependency graph: module_name -> set of module_names it depends on
        self._dependencies: Dict[str, Set[str]] = {}

    # ──────────────────────────────────────────────────────────────
    #  Registration
    # ──────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        kind: LibraryKind,
        source_path: str = "",
        link_mode: Optional[LinkMode] = None,
        pure_python_strategy: Optional[PurePythonLinkStrategy] = None,
        exported_symbols: Optional[Set[str]] = None,
        imported_symbols: Optional[Set[str]] = None,
        extra: Optional[Dict] = None,
    ) -> LibraryEntry:
        """Register a new library or update an existing entry.

        If a library with the same name already exists, its metadata
        is updated (merged) rather than replaced.  This allows
        incremental discovery: the first registration may come from
        an ``import`` statement, and subsequent registrations from
        the FFI analyzer or pure-Python compiler can enrich the entry.

        Args:
            name: Python module name.
            kind: Library type.
            source_path: Path to the source file.
            link_mode: Override linking mode.  If None, resolved from config.
            pure_python_strategy: Override pure-Python strategy.
            exported_symbols: Symbols this library provides.
            imported_symbols: Symbols this library requires.
            extra: Additional metadata.

        Returns:
            The registered LibraryEntry.

        Raises:
            ValueError: If a library with the same name but a different
                kind is already registered.
        """
        if name in self._entries:
            existing = self._entries[name]
            if existing.kind != kind:
                raise ValueError(
                    f"Konflikt typów biblioteki '{name}': "
                    f"istnieje jako {existing.kind.value}, "
                    f"rejestrowano jako {kind.value}."
                )
            # Merge metadata
            if source_path:
                existing.source_path = source_path
            if link_mode is not None:
                existing.link_mode = link_mode
            if pure_python_strategy is not None:
                existing.pure_python_strategy = pure_python_strategy
            if exported_symbols:
                existing.exported_symbols.update(exported_symbols)
            if imported_symbols:
                existing.imported_symbols.update(imported_symbols)
            if extra:
                existing.extra.update(extra)
            return existing

        # Resolve effective link mode from config if not explicitly set
        effective_mode = link_mode or self._config.resolve_link_mode(name)

        # Resolve pure-Python strategy from config if not explicitly set
        effective_strategy = pure_python_strategy
        if effective_strategy is None and kind == LibraryKind.PURE_PYTHON:
            effective_strategy = self._config.resolve_pure_python_strategy(name)

        entry = LibraryEntry(
            name=name,
            kind=kind,
            source_path=source_path,
            link_mode=effective_mode,
            pure_python_strategy=effective_strategy,
            exported_symbols=exported_symbols or set(),
            imported_symbols=imported_symbols or set(),
            extra=extra or {},
        )
        self._entries[name] = entry
        return entry

    def register_builtin(
        self,
        name: str,
        exported_symbols: Optional[Set[str]] = None,
    ) -> LibraryEntry:
        """Register a built-in module (e.g. math, time, os, sys).

        Built-in modules are always statically linked because their
        implementations are LLVM intrinsics or hard-coded LLVM IR
        generated by the compiler's runtime mixin.

        Args:
            name: Module name (e.g. ``"math"``).
            exported_symbols: Symbols the built-in module provides.

        Returns:
            The registered LibraryEntry.
        """
        return self.register(
            name=name,
            kind=LibraryKind.BUILTIN,
            link_mode=LinkMode.STATIC,
            exported_symbols=exported_symbols,
        )

    def register_pure_python(
        self,
        name: str,
        source_path: str,
        link_mode: Optional[LinkMode] = None,
        strategy: Optional[PurePythonLinkStrategy] = None,
    ) -> LibraryEntry:
        """Register a pure-Python module.

        The source path must point to an existing .py file.  The
        effective linking strategy is resolved from the configuration
        if not explicitly provided.

        Args:
            name: Module name.
            source_path: Path to the .py source file.
            link_mode: Override linking mode.
            strategy: Override pure-Python compilation strategy.

        Returns:
            The registered LibraryEntry.

        Raises:
            FileNotFoundError: If source_path does not exist.
        """
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Plik źródłowy modułu pure Python nie istnieje: '{source_path}'"
            )
        return self.register(
            name=name,
            kind=LibraryKind.PURE_PYTHON,
            source_path=source_path,
            link_mode=link_mode,
            pure_python_strategy=strategy,
        )

    def register_native_so(
        self,
        name: str,
        so_path: str,
        link_mode: Optional[LinkMode] = None,
        exported_symbols: Optional[Set[str]] = None,
        imported_symbols: Optional[Set[str]] = None,
        extra: Optional[Dict] = None,
    ) -> LibraryEntry:
        """Register a native .so library.

        The path must point to an existing shared object file.  The
        FFI subsystem will analyze it to determine exported symbols
        and Py* imports.

        Args:
            name: Module name.
            so_path: Path to the .so file.
            link_mode: Override linking mode.
            exported_symbols: Symbols the .so exports.
            imported_symbols: Symbols the .so requires.
            extra: Additional metadata.

        Returns:
            The registered LibraryEntry.

        Raises:
            FileNotFoundError: If so_path does not exist.
        """
        if not os.path.isfile(so_path):
            raise FileNotFoundError(
                f"Plik biblioteki natywnej nie istnieje: '{so_path}'"
            )
        return self.register(
            name=name,
            kind=LibraryKind.NATIVE_SO,
            source_path=so_path,
            link_mode=link_mode,
            exported_symbols=exported_symbols,
            imported_symbols=imported_symbols,
            extra=extra,
        )

    # ──────────────────────────────────────────────────────────────
    #  Lookup
    # ──────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[LibraryEntry]:
        """Look up a library by name.

        Args:
            name: Module name.

        Returns:
            The LibraryEntry, or None if not registered.
        """
        return self._entries.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> LibraryEntry:
        if name not in self._entries:
            raise KeyError(f"Biblioteka '{name}' nie jest zarejestrowana.")
        return self._entries[name]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    # ──────────────────────────────────────────────────────────────
    #  Filtering & queries
    # ──────────────────────────────────────────────────────────────

    def entries_by_kind(self, kind: LibraryKind) -> List[LibraryEntry]:
        """Return all entries of a given kind.

        Args:
            kind: The library kind to filter by.

        Returns:
            List of matching LibraryEntry objects.
        """
        return [e for e in self._entries.values() if e.kind == kind]

    def entries_by_link_mode(self, mode: LinkMode) -> List[LibraryEntry]:
        """Return all entries with a given link mode.

        Args:
            mode: The link mode to filter by.

        Returns:
            List of matching LibraryEntry objects.
        """
        return [e for e in self._entries.values() if e.link_mode == mode]

    @property
    def pure_python_modules(self) -> List[LibraryEntry]:
        """All registered pure-Python modules."""
        return self.entries_by_kind(LibraryKind.PURE_PYTHON)

    @property
    def native_so_modules(self) -> List[LibraryEntry]:
        """All registered native .so modules."""
        return self.entries_by_kind(LibraryKind.NATIVE_SO)

    @property
    def builtin_modules(self) -> List[LibraryEntry]:
        """All registered built-in modules."""
        return self.entries_by_kind(LibraryKind.BUILTIN)

    @property
    def static_modules(self) -> List[LibraryEntry]:
        """All modules configured for static linking."""
        return self.entries_by_link_mode(LinkMode.STATIC)

    @property
    def dynamic_modules(self) -> List[LibraryEntry]:
        """All modules configured for dynamic linking."""
        return self.entries_by_link_mode(LinkMode.DYNAMIC)

    # ──────────────────────────────────────────────────────────────
    #  Dependency tracking
    # ──────────────────────────────────────────────────────────────

    def add_dependency(self, module_name: str, depends_on: str) -> None:
        """Record that one module depends on another.

        Dependencies are used by the linker to determine the correct
        link order: if module A depends on module B, then B must be
        linked before A.

        Args:
            module_name: The importing module.
            depends_on: The imported module.
        """
        self._dependencies.setdefault(module_name, set()).add(depends_on)

    def get_dependencies(self, module_name: str) -> Set[str]:
        """Get the set of modules that a given module depends on.

        Args:
            module_name: The module to query.

        Returns:
            Set of module names (possibly empty).
        """
        return set(self._dependencies.get(module_name, set()))

    def topological_order(self) -> List[str]:
        """Return module names in dependency-safe (topological) order.

        Modules with no dependencies come first, followed by modules
        that depend on them, and so on.  This is the order in which
        libraries should be linked to ensure all symbols are available
        when needed.

        Returns:
            List of module names in topological order.

        Raises:
            RuntimeError: If a dependency cycle is detected.
        """
        # Kahn's algorithm
        in_degree: Dict[str, int] = {name: 0 for name in self._entries}
        graph: Dict[str, Set[str]] = {name: set() for name in self._entries}

        for mod, deps in self._dependencies.items():
            if mod not in in_degree:
                continue
            for dep in deps:
                if dep in in_degree:
                    graph[dep].add(mod)
                    in_degree[mod] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._entries):
            remaining = set(self._entries) - set(result)
            raise RuntimeError(
                f"Wykryto cykl zależności między modułami: {remaining}"
            )

        return result

    # ──────────────────────────────────────────────────────────────
    #  Marking compilation / linking status
    # ──────────────────────────────────────────────────────────────

    def mark_compiled(self, name: str) -> None:
        """Mark a library as having its LLVM IR generated.

        Args:
            name: Module name.

        Raises:
            KeyError: If the module is not registered.
        """
        if name in self._entries:
            self._entries[name].is_compiled = True

    def mark_linked(self, name: str) -> None:
        """Mark a library as having been linked into the output.

        Args:
            name: Module name.

        Raises:
            KeyError: If the module is not registered.
        """
        if name in self._entries:
            self._entries[name].is_linked = True

    # ──────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the registry contents.

        Returns:
            Multi-line string with one line per library.
        """
        lines = [f"Rejestr bibliotek ({len(self._entries)} wpisów):"]
        for entry in self._entries.values():
            status = "✓" if entry.is_linked else ("⟳" if entry.is_compiled else "○")
            lines.append(
                f"  {status} {entry.name:20s} [{entry.kind.value:12s}] "
                f"link={entry.link_mode} "
                f"symbols={len(entry.exported_symbols)}"
            )
        return "\n".join(lines)

    def unresolved_symbols(self) -> Set[str]:
        """Find symbols that are required but not provided by any library.

        This is useful for diagnostic output before linking: if there
        are unresolved symbols, the linker will fail.

        Returns:
            Set of symbol names that have no provider.
        """
        all_exported: Set[str] = set()
        all_imported: Set[str] = set()
        for entry in self._entries.values():
            all_exported.update(entry.exported_symbols)
            all_imported.update(entry.imported_symbols)
        return all_imported - all_exported
