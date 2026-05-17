################################################################################
# PLIK: src/values.py

"""Value wrapper for compiler expression results.

Every expression in the compiler produces a Value — a pair of
(LLVM IR value, static Python type) with optional class name tracking.

FFIModuleValue is a special Value subclass representing a reference to
an imported native .so module. It does not have an underlying LLVM IR
value because module references are purely compile-time constructs used
for attribute resolution (e.g., ``markupsafe.escape`` -> FFI call).
"""

from __future__ import annotations

import llvmlite.ir as ir

from .types import PyType


class Value:
    """Pair of (ir.Value, PyType) — the result of every expression.

    Wraps an LLVM IR value together with its static Python type information.
    This enables the compiler to make type-driven decisions while
    still supporting dynamic typing through PyType.OBJECT.

    Attributes:
        llvm: The underlying LLVM IR value.
        pytype: The static Python type of this value.
        class_name: Optional class name for instance values.
    """

    __slots__ = ("llvm", "pytype", "class_name")

    def __init__(
        self, llvm: ir.Value, pytype: PyType, class_name: str = None
    ) -> None:
        self.llvm = llvm
        self.pytype = pytype
        self.class_name = class_name  # Track class name for instances

    @property
    def is_int(self) -> bool:
        """Check if this value has static type INT."""
        return self.pytype == PyType.INT

    @property
    def is_float(self) -> bool:
        """Check if this value has static type FLOAT."""
        return self.pytype == PyType.FLOAT

    @property
    def is_bool(self) -> bool:
        """Check if this value has static type BOOL."""
        return self.pytype == PyType.BOOL

    @property
    def is_str(self) -> bool:
        """Check if this value has static type STR."""
        return self.pytype == PyType.STR

    @property
    def is_list(self) -> bool:
        """Check if this value has static type LIST."""
        return self.pytype == PyType.LIST

    @property
    def is_dict(self) -> bool:
        """Check if this value has static type DICT."""
        return self.pytype == PyType.DICT

    @property
    def is_none(self) -> bool:
        """Check if this value has static type NONE."""
        return self.pytype == PyType.NONE

    @property
    def is_instance(self) -> bool:
        """Check if this value has static type INSTANCE."""
        return self.pytype == PyType.INSTANCE

    @property
    def is_object(self) -> bool:
        """Check if this value has static type OBJECT (dynamically typed)."""
        return self.pytype == PyType.OBJECT

    @property
    def is_tuple(self) -> bool:
        """Check if this value has static type TUPLE."""
        return self.pytype == PyType.TUPLE

    @property
    def is_iterator(self) -> bool:
        """Check if this value has static type ITERATOR."""
        return self.pytype == PyType.ITERATOR

    @property
    def is_set(self) -> bool:
        """Check if this value has static type SET."""
        return self.pytype == PyType.SET

    @property
    def is_coroutine(self) -> bool:
        """Check if this value has static type COROUTINE."""
        return self.pytype == PyType.COROUTINE

    @property
    def is_task(self) -> bool:
        """Check if this value has static type TASK."""
        return self.pytype == PyType.TASK

    @property
    def is_numeric(self) -> bool:
        """Check if this value is a numeric type (INT, FLOAT, or BOOL)."""
        return self.pytype in (PyType.INT, PyType.FLOAT, PyType.BOOL)


class FFIModuleValue(Value):
    """Compile-time reference to an imported FFI (native .so) module.

    This is NOT a runtime value — it has no LLVM IR representation.
    Instead, it serves as a marker that tells the compiler's attribute
    access and method call visitors that the "object" is a foreign
    module, and that attribute lookups (e.g., ``markupsafe.escape``)
    should be resolved as FFI symbol calls.

    When ``visit_Name`` encounters a variable whose VarInfo has
    ``is_ffi_module=True``, it returns an FFIModuleValue instead of
    attempting to ``builder.load`` from a (nonexistent) alloca.

    When ``_method_call`` receives an FFIModuleValue as the object,
    it dispatches to ``_ffi_call()`` with the resolved symbol.

    Attributes:
        module_name: The Python module name (e.g., "markupsafe").
    """

    __slots__ = ("module_name",)

    def __init__(self, module_name: str) -> None:
        # llvm=None and pytype=OBJECT — this is a compile-time marker
        super().__init__(llvm=None, pytype=PyType.OBJECT)
        self.module_name = module_name

    @property
    def is_ffi_module(self) -> bool:
        """Always True for FFIModuleValue."""
        return True

    # Override all type-check properties to return False
    # (FFI module references are not any of the standard types)
    @property
    def is_int(self) -> bool:
        return False

    @property
    def is_float(self) -> bool:
        return False

    @property
    def is_bool(self) -> bool:
        return False

    @property
    def is_str(self) -> bool:
        return False

    @property
    def is_list(self) -> bool:
        return False

    @property
    def is_dict(self) -> bool:
        return False

    @property
    def is_none(self) -> bool:
        return False

    @property
    def is_instance(self) -> bool:
        return False

    @property
    def is_object(self) -> bool:
        return False

    @property
    def is_tuple(self) -> bool:
        return False

    @property
    def is_iterator(self) -> bool:
        return False

    @property
    def is_set(self) -> bool:
        return False

    @property
    def is_numeric(self) -> bool:
        return False
