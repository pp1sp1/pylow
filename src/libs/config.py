"""Library linking configuration for the py2llvm compiler.

Provides dataclasses and enums for configuring how libraries (both pure Python
and native .so) are linked into the compiled output.  The user can choose
between static linking (symbols embedded directly in the output binary) and
dynamic linking (symbols resolved at runtime via dlopen/dlsym).

The configuration supports:
  - A global default linking mode (``libs_mode``)
  - Per-library overrides via ``dynamic_libs`` / ``static_libs`` sets
  - Per-module linking strategies for pure Python modules
  - Search paths for library discovery
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set


class LinkMode(Enum):
    """Determines how a library's symbols are resolved at link/run time.

    STATIC:
      The library's compiled LLVM IR is merged into the main module at
      compile time.  All function definitions are available directly
      without any runtime indirection.  This produces a larger binary
      but avoids dlopen/dlsym overhead and ensures all symbols are
      resolved at link time.

    DYNAMIC:
      The library is loaded at runtime via ``dlopen`` and its symbols
      are resolved with ``dlsym``.  The compiled output contains only
      ``declare external`` stubs plus a small initialization snippet
      that calls ``dlopen`` at program startup.  This yields a smaller
      binary but adds runtime overhead and a dependency on the .so
      file being present at execution time.
    """

    STATIC = auto()
    DYNAMIC = auto()

    @classmethod
    def from_string(cls, mode: str) -> "LinkMode":
        """Parse a string into a LinkMode value.

        Args:
            mode: One of ``"static"`` or ``"dynamic"`` (case-insensitive).

        Returns:
            The corresponding LinkMode enum member.

        Raises:
            ValueError: If the string does not match any known mode.
        """
        normalized = mode.strip().lower()
        if normalized == "static":
            return cls.STATIC
        if normalized == "dynamic":
            return cls.DYNAMIC
        raise ValueError(
            f"Nieznany tryb linkowania: '{mode}'. "
            f"Oczekiwano 'static' lub 'dynamic'."
        )

    def __str__(self) -> str:
        return self.name.lower()


class PurePythonLinkStrategy(Enum):
    """How a pure-Python module is compiled and linked.

    INLINE:
      The module's Python source is parsed and its top-level statements
      and function definitions are compiled directly into the main
      module's LLVM IR.  Function calls have zero overhead because they
      are direct calls.  However, namespace collisions are possible if
      the imported module defines symbols with the same name as the
      importing module.

    COMPILED_UNIT:
      The module is compiled into a separate LLVM IR unit with mangled
      symbol names (e.g. ``mypkg_utils_add`` instead of ``add``).  At
      link time the unit is either merged (static) or loaded at runtime
      (dynamic).  This preserves namespace boundaries and enables
      separate compilation, but adds a thin dispatch layer for
      cross-module calls.

    STUB_ONLY:
      Only LLVM IR ``declare external`` stubs are generated.  The actual
      implementation is expected to be provided by an already-compiled
      .so or .bc file on the library search path.  This is useful when
      the pure-Python module has been pre-compiled separately.
    """

    INLINE = auto()
    COMPILED_UNIT = auto()
    STUB_ONLY = auto()

    @classmethod
    def from_string(cls, strategy: str) -> "PurePythonLinkStrategy":
        """Parse a string into a PurePythonLinkStrategy.

        Args:
            strategy: One of ``"inline"``, ``"compiled_unit"``, or ``"stub_only"``.

        Returns:
            The corresponding enum member.

        Raises:
            ValueError: If the string does not match any known strategy.
        """
        normalized = strategy.strip().lower().replace("-", "_")
        mapping = {
            "inline": cls.INLINE,
            "compiled_unit": cls.COMPILED_UNIT,
            "compiledunit": cls.COMPILED_UNIT,
            "stub_only": cls.STUB_ONLY,
            "stubonly": cls.STUB_ONLY,
        }
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(
            f"Nieznana strategia linkowania pure Python: '{strategy}'. "
            f"Oczekiwano 'inline', 'compiled_unit' lub 'stub_only'."
        )

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class LibraryConfig:
    """Central configuration object for library linking behaviour.

    Collects all user-configurable options that control how the compiler
    discovers, compiles, and links libraries — both native .so (FFI)
    and pure-Python modules.

    Attributes:
        libs_mode: Global default linking mode.  If a library is not
            explicitly listed in ``dynamic_libs`` or ``static_libs``,
            this mode is used.
        dynamic_libs: Set of library names that MUST be linked
            dynamically, regardless of the global default.
        static_libs: Set of library names that MUST be linked
            statically, regardless of the global default.
        pure_python_strategy: Default compilation/linking strategy for
            pure-Python modules.
        per_module_strategy: Per-module overrides for pure-Python
            linking strategy.  Keys are module names, values are the
            strategy to use for that module.
        search_paths: Ordered list of directories to search when
            locating library files (.py, .so, .bc).
        auto_discover: If True, the compiler will automatically search
            for .so / .py files on ``search_paths`` when an ``import``
            statement is encountered.
        allow_fallback: If True, when the requested linking mode is not
            possible (e.g., no .so file found for dynamic linking), the
            compiler falls back to the other mode with a warning.
    """

    libs_mode: LinkMode = LinkMode.STATIC
    dynamic_libs: Set[str] = field(default_factory=set)
    static_libs: Set[str] = field(default_factory=set)
    pure_python_strategy: PurePythonLinkStrategy = PurePythonLinkStrategy.INLINE
    per_module_strategy: Dict[str, PurePythonLinkStrategy] = field(default_factory=dict)
    search_paths: List[str] = field(default_factory=list)
    auto_discover: bool = True
    allow_fallback: bool = True

    # ──────────────────────────────────────────────────────────────
    #  Factory constructors
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, cfg: dict) -> "LibraryConfig":
        """Create a LibraryConfig from a plain dictionary.

        Accepted keys (all optional):
          - ``libs_mode``: ``"static"`` or ``"dynamic"``
          - ``dynamic_libs``: list or set of library names
          - ``static_libs``: list or set of library names
          - ``pure_python_strategy``: ``"inline"``, ``"compiled_unit"``, or ``"stub_only"``
          - ``per_module_strategy``: dict of module_name -> strategy string
          - ``search_paths``: list of directory paths
          - ``auto_discover``: bool
          - ``allow_fallback``: bool

        Args:
            cfg: Configuration dictionary.

        Returns:
            A fully populated LibraryConfig.
        """
        libs_mode = LinkMode.from_string(cfg["libs_mode"]) if "libs_mode" in cfg else LinkMode.STATIC
        dynamic_libs = set(cfg.get("dynamic_libs", []))
        static_libs = set(cfg.get("static_libs", []))
        pps = PurePythonLinkStrategy.from_string(cfg["pure_python_strategy"]) if "pure_python_strategy" in cfg else PurePythonLinkStrategy.INLINE
        per_module = {
            k: PurePythonLinkStrategy.from_string(v) if isinstance(v, str) else v
            for k, v in cfg.get("per_module_strategy", {}).items()
        }
        search_paths = cfg.get("search_paths", [])

        return cls(
            libs_mode=libs_mode,
            dynamic_libs=dynamic_libs,
            static_libs=static_libs,
            pure_python_strategy=pps,
            per_module_strategy=per_module,
            search_paths=search_paths,
            auto_discover=cfg.get("auto_discover", True),
            allow_fallback=cfg.get("allow_fallback", True),
        )

    # ──────────────────────────────────────────────────────────────
    #  Resolution helpers
    # ──────────────────────────────────────────────────────────────

    def resolve_link_mode(self, library_name: str) -> LinkMode:
        """Determine the effective linking mode for a given library.

        The resolution order is:
          1. If the library is in ``static_libs``, return STATIC.
          2. If the library is in ``dynamic_libs``, return DYNAMIC.
          3. Otherwise, return the global default ``libs_mode``.

        Args:
            library_name: The name of the library to resolve.

        Returns:
            The effective LinkMode for the library.
        """
        if library_name in self.static_libs:
            return LinkMode.STATIC
        if library_name in self.dynamic_libs:
            return LinkMode.DYNAMIC
        return self.libs_mode

    def resolve_pure_python_strategy(self, module_name: str) -> PurePythonLinkStrategy:
        """Determine the compilation/linking strategy for a pure-Python module.

        The resolution order is:
          1. If the module has an entry in ``per_module_strategy``, use it.
          2. Otherwise, return the global default ``pure_python_strategy``.

        Args:
            module_name: The name of the pure-Python module.

        Returns:
            The effective PurePythonLinkStrategy for the module.
        """
        return self.per_module_strategy.get(module_name, self.pure_python_strategy)

    def effective_search_paths(self) -> List[str]:
        """Return the complete ordered list of library search paths.

        Combines user-configured paths with sensible defaults (cwd,
        site-packages).  Duplicates are removed while preserving order.

        Returns:
            Ordered list of absolute directory paths.
        """
        paths: List[str] = []
        seen: set = set()

        # User-configured paths first
        for p in self.search_paths:
            absp = os.path.abspath(p)
            if absp not in seen:
                paths.append(absp)
                seen.add(absp)

        # Current working directory
        cwd = os.path.abspath(os.getcwd())
        if cwd not in seen:
            paths.append(cwd)
            seen.add(cwd)

        # Python site-packages
        try:
            import site
            for sp in site.getsitepackages():
                absp = os.path.abspath(sp)
                if absp not in seen:
                    paths.append(absp)
                    seen.add(absp)
        except (AttributeError, ImportError):
            pass

        # Virtual environment site-packages
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            for sub in ["lib", "lib64"]:
                base = os.path.join(venv, sub)
                if os.path.isdir(base):
                    for entry in os.listdir(base):
                        candidate = os.path.join(base, entry, "site-packages")
                        if os.path.isdir(candidate):
                            absp = os.path.abspath(candidate)
                            if absp not in seen:
                                paths.append(absp)
                                seen.add(absp)

        return paths

    def summary(self) -> str:
        """Return a human-readable summary of the configuration.

        Useful for diagnostic output and logging.

        Returns:
            Multi-line string describing the current configuration.
        """
        lines = [
            f"Tryb domyślny linkowania: {self.libs_mode}",
            f"Strategia pure Python: {self.pure_python_strategy}",
            f"Biblioteki dynamiczne: {sorted(self.dynamic_libs) or '(brak)'}",
            f"Biblioteki statyczne: {sorted(self.static_libs) or '(brak)'}",
            f"Auto-discovery: {'włączony' if self.auto_discover else 'wyłączony'}",
            f"Fallback: {'włączony' if self.allow_fallback else 'wyłączony'}",
        ]
        if self.per_module_strategy:
            lines.append("Strategie per-moduł:")
            for mod, strat in sorted(self.per_module_strategy.items()):
                lines.append(f"  {mod}: {strat}")
        if self.search_paths:
            lines.append("Ścieżki wyszukiwania:")
            for p in self.search_paths:
                lines.append(f"  {p}")
        return "\n".join(lines)
