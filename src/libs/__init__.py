"""py2llvm Library Management Subsystem.

Provides a unified interface for discovering, compiling, and linking
both pure-Python modules and native .so libraries.  The user can
choose between static and dynamic linking on a per-library basis.

Architecture overview:

  LibraryConfig
    Central configuration: global defaults, per-library overrides,
    search paths, and pure-Python linking strategies.

  LibraryRegistry
    Catalog of all known libraries with metadata, dependency tracking,
    and topological ordering for correct link sequences.

  PurePythonHandler
    Discovers .py files, parses them, compiles them to LLVM IR using
    one of three strategies (inline, compiled_unit, stub_only), and
    generates the appropriate link code.

  LinkManager
    Orchestrates the final linking phase: generates a link plan from
    the registry, emits LLVM IR for static/dynamic linking, and
    invokes the system linker to produce the output binary.

Usage example::

    from src.libs import LibraryConfig, LibraryRegistry, LinkManager

    # Configure: default static, but link markupsafe dynamically
    config = LibraryConfig(
        libs_mode=LinkMode.STATIC,
        dynamic_libs={"markupsafe"},
    )

    registry = LibraryRegistry(config)
    registry.register_pure_python("utils", "/path/to/utils.py")
    registry.register_native_so("markupsafe", "/path/to/markupsafe.so")

    linker = LinkManager(registry, config)
    plan = linker.create_plan()
    linker.execute_plan(compiler, plan, "output_binary")
"""

from .config import (
    LinkMode,
    PurePythonLinkStrategy,
    LibraryConfig,
)
from .registry import (
    LibraryKind,
    LibraryEntry,
    LibraryRegistry,
)
from .pure_python import (
    PurePythonModuleInfo,
    PurePythonHandler,
)
from .linker import (
    LinkAction,
    LinkStep,
    LinkPlan,
    LinkManager,
)

__all__ = [
    # Config
    "LinkMode",
    "PurePythonLinkStrategy",
    "LibraryConfig",
    # Registry
    "LibraryKind",
    "LibraryEntry",
    "LibraryRegistry",
    # Pure Python
    "PurePythonModuleInfo",
    "PurePythonHandler",
    # Linker
    "LinkAction",
    "LinkStep",
    "LinkPlan",
    "LinkManager",
]
