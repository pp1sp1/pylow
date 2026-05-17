"""Runtime type tags, static type system, and LLVM IR type definitions.

This module defines the core type system used throughout the compiler:
- Tag: Runtime type tags stored in boxed values for dynamic dispatch.
- PyType: Static type enumeration used during compilation.
- LLVM IR type constants and struct layouts.
- pytype_to_llvm: Mapping from static PyType to LLVM IR types.
"""

from enum import IntEnum, auto
from typing import TYPE_CHECKING

import llvmlite.ir as ir


# ══════════════════════════════════════════════════════════════════
#  RUNTIME TAGS
# ══════════════════════════════════════════════════════════════════


class Tag(IntEnum):
    """Runtime type tag stored inside BoxedValue for dynamic dispatch.
    
    Each boxed value carries a Tag to identify its payload type at runtime,
    enabling dynamic typing support (e.g., variables that change type).
    """
    NONE = 0
    INT = 1
    FLOAT = 2
    BOOL = 3
    STR = 4       # payload = i64(ptr i8*)
    LIST = 5      # payload = i64(ptr ListObject*)
    DICT = 6      # payload = i64(ptr DictObject*)
    INST = 7      # payload = i64(ptr InstanceObject*)
    TUPLE = 8     # payload = i64(ptr ListObject*) — identical structure to LIST, printed with ()
    ITERATOR = 9  # payload = i64(ptr IterData*) — iterator over list (generator)
    SET = 10      # payload = i64(ptr ListObject*) — identical structure to LIST, printed with {}
    COROUTINE = 11  # payload = i64(ptr BoxedValue*) — coroutine result (boxed value)
    TASK = 12       # payload = i64(ptr ListObject*) — task list of coroutine results


# ══════════════════════════════════════════════════════════════════
#  STATIC TYPE SYSTEM
# ══════════════════════════════════════════════════════════════════


class PyType(IntEnum):
    """Static type enumeration used during compilation.
    
    PyType.OBJECT represents a dynamically-typed value whose actual type
    is only known at runtime (stored as BOXED_PTR with a Tag).
    """
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    NONE = auto()
    STR = auto()
    LIST = auto()
    DICT = auto()
    INSTANCE = auto()   # Class instance
    OBJECT = auto()     # Dynamic — type known only at runtime
    TUPLE = auto()      # Tuple — structure like LIST, printed with ()
    ITERATOR = auto()   # Iterator (generator)
    SET = auto()        # Set — structure like LIST, printed with {}
    COROUTINE = auto()  # Coroutine — async function result
    TASK = auto()        # Task — scheduled coroutine


# Set of known exception type names used in the compiler.
EXCEPTION_TYPES: set[str] = {
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "NameError",
    "ZeroDivisionError",
    "NotImplementedError",
    "StopIteration",
    "IOError",
    "OSError",
    "FileNotFoundError",
    "PermissionError",
    "FrozenInstanceError",
}


# ══════════════════════════════════════════════════════════════════
#  LLVM IR TYPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# --- Primitive types ---
I8 = ir.IntType(8)
I32 = ir.IntType(32)
I64 = ir.IntType(64)
F64 = ir.DoubleType()
I1 = ir.IntType(1)
I8P = ir.PointerType(I8)
VOID = ir.VoidType()

# --- GC header type ---
# GC_HEADER_TY = { refcnt: i64, color: i32, temp_refcnt: i64, gc_next: i8* }
# color: 0=Black (In use), 1=Gray (Possible cycle), 2=White (Garbage), 3=Purple (Suspect)
GC_HEADER_TY = ir.LiteralStructType([I64, I32, I64, I8P])

# --- Boxed value type ---
BOXED_TY = ir.LiteralStructType([GC_HEADER_TY, I64, I64])
BOXED_PTR = ir.PointerType(BOXED_TY)

# --- List type ---
LIST_TY = ir.LiteralStructType([GC_HEADER_TY, I64, I64, BOXED_PTR])
LIST_PTR = ir.PointerType(LIST_TY)

# --- Hash table entry type ---
ENTRY_TY = ir.LiteralStructType([I64, I64, I64, I64])
ENTRY_PTR = ir.PointerType(ENTRY_TY)

# --- Dictionary type ---
DICT_TY = ir.LiteralStructType([GC_HEADER_TY, I64, I64, ENTRY_PTR, LIST_PTR])
DICT_PTR = ir.PointerType(DICT_TY)

# --- String type ---
STR_TY = ir.LiteralStructType([GC_HEADER_TY, I64, I64, I8P])
STR_PTR = ir.PointerType(STR_TY)

# --- Class type ---
CLASS_TY = ir.LiteralStructType([GC_HEADER_TY, I8P, DICT_PTR, DICT_PTR])
CLASS_PTR = ir.PointerType(CLASS_TY)

# --- Instance type ---
INSTANCE_TY = ir.LiteralStructType([GC_HEADER_TY, CLASS_PTR, DICT_PTR])
INSTANCE_PTR = ir.PointerType(INSTANCE_TY)

# --- Type sizes (including alignment) ---
SZ_GC_HEADER = 32   # {i64(8), i32(4), padding(4), i64(8), i8*(8)}
SZ_BOXED = 48       # GC_HEADER(32) + tag(8) + payload(8)
SZ_ENTRY = 32       # 4 * i64 = 32 bytes
SZ_LIST = 56        # GC_HEADER(32) + size(8) + cap(8) + data_ptr(8)
SZ_DICT = 64        # GC_HEADER(32) + size(8) + cap(8) + entries_ptr(8) + ordered_keys_ptr(8)
SZ_STR = 56         # GC_HEADER(32) + len(8) + cap(8) + data_ptr(8)

# --- Iterator (generator) data structure: {GC_HEADER, list_ptr, index} ---
ITER_DATA_TY = ir.LiteralStructType([GC_HEADER_TY, LIST_PTR, I64])
ITER_DATA_PTR = ir.PointerType(ITER_DATA_TY)
SZ_ITER = 48  # GC_HEADER(32) + list_ptr(8) + index(8)


def pytype_to_llvm(pt: PyType) -> ir.Type:
    """Map a static PyType to its corresponding LLVM IR type.
    
    Args:
        pt: The static Python type to convert.
        
    Returns:
        The LLVM IR type that represents values of this PyType.
        For PyType.OBJECT and unknown types, returns BOXED_PTR.
    """
    if pt == PyType.INT:
        return I64
    if pt == PyType.FLOAT:
        return F64
    if pt == PyType.BOOL:
        return I1
    if pt == PyType.STR:
        return STR_PTR
    if pt == PyType.LIST:
        return LIST_PTR
    if pt == PyType.TUPLE:
        return LIST_PTR   # TUPLE has the same structure as LIST
    if pt == PyType.SET:
        return LIST_PTR   # SET has the same structure as LIST, printed with {}
    if pt == PyType.DICT:
        return DICT_PTR
    if pt == PyType.INSTANCE:
        return INSTANCE_PTR
    return BOXED_PTR
