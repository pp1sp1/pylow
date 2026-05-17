################################################################################
# PLIK: src/symbols.py

"""Symbol table implementation for variable tracking during compilation.

Provides VarInfo (variable metadata) and SymbolTable (scoped name resolution).
VarInfo supports FFI module markers for native .so library references.
"""

from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING

import llvmlite.ir as ir

from .types import PyType
from .exceptions import CompileError, PylowError
from .reporter import ErrorCategory

if TYPE_CHECKING:
    from .values import Value


class VarInfo:
    """Metadata about a compiled variable.

    Tracks the LLVM alloca instruction, LLVM type, Python type,
    and optionally the class name for instance variables.

    For FFI module references (imported native .so libraries),
    ``alloca`` is ``None`` and ``is_ffi_module`` is ``True``.
    The ``ffi_module_name`` field stores the original Python module
    name (e.g., "markupsafe") so that attribute access like
    ``markupsafe.escape`` can be resolved at compile time.

    Attributes:
        alloca: LLVM alloca instruction (stack slot) for this variable.
            None for FFI module references.
        llvm_type: The LLVM IR type stored in the alloca.
        py_type: The static Python type of the variable.
        class_name: Optional class name for instance variables.
        is_ffi_module: True if this VarInfo represents an imported FFI module.
        ffi_module_name: The Python module name for FFI module references.
    """
    __slots__ = ("alloca", "llvm_type", "py_type", "class_name", "is_ffi_module", "ffi_module_name", "is_class_ref")

    def __init__(
        self,
        alloca: Optional[ir.AllocaInstr],
        llvm_type: ir.Type,
        py_type: PyType,
        class_name: Optional[str] = None,
        is_ffi_module: bool = False,
        ffi_module_name: Optional[str] = None,
    ) -> None:
        self.alloca = alloca
        self.llvm_type = llvm_type
        self.py_type = py_type
        self.class_name = class_name  # Track class name for instances
        self.is_ffi_module = is_ffi_module
        self.ffi_module_name = ffi_module_name
        self.is_class_ref = False  # True if this variable holds a class reference (e.g., device_cls = SmartDevice)


class SymbolTable:
    """Scoped symbol table supporting nested scopes.

    Uses a stack of dictionaries where each frame represents a lexical scope.
    Variable lookup searches from the innermost scope outward.

    Attributes:
        parent: Optional parent symbol table for closure support.
    """

    def __init__(self, parent: Optional[SymbolTable] = None) -> None:
        self._stack: List[Dict[str, VarInfo]] = [{}]
        self.parent: Optional[SymbolTable] = parent

    def push(self) -> None:
        """Enter a new inner scope."""
        self._stack.append({})

    def pop(self) -> None:
        """Leave the current inner scope."""
        self._stack.pop()

    def define(self, name: str, info: VarInfo) -> None:
        """Define a variable in the current (innermost) scope.

        Args:
            name: Variable name.
            info: Variable metadata (alloca, type, etc.).
        """
        self._stack[-1][name] = info

    def lookup(self, name: str, node: object = None) -> VarInfo:
        """Look up a variable by name, searching from innermost scope outward.

        Args:
            name: Variable name to find.
            node: Optional AST node providing line/column info for errors.

        Returns:
            The VarInfo for the variable.

        Raises:
            CompileError: If the variable is not defined in any scope.
        """
        for frame in reversed(self._stack):
            if name in frame:
                return frame[name]
        exc = CompileError(f"Undefined name: '{name}'")
        exc.node = node  # Attach node for line/column info in reporter
        raise exc

    def exists_local(self, name: str) -> bool:
        """Check if a variable exists in the current (innermost) scope only.

        Args:
            name: Variable name to check.

        Returns:
            True if the variable is defined in the current scope.
        """
        return name in self._stack[-1]

    def lookup_nonlocal(self, name: str) -> Optional[VarInfo]:
        """Look up a variable in outer scopes (for nonlocal declarations).

        Searches all scopes except the innermost one.

        Args:
            name: Variable name to find.

        Returns:
            The VarInfo for the variable, or None if not found.
        """
        for frame in reversed(self._stack[:-1]):
            if name in frame:
                return frame[name]
        return None
