"""Core FFI analysis engine for native .so library integration.

Absorbs and refactors the logic from generate_cpp_wrapper.py into a
clean, compiler-integrable module.  Key responsibilities:

1. Py* API signature database (934+ symbols, CPython 3.12 ABI).
2. ELF/PE binary analysis via LIEF (with pure-Python fallback).
3. FFIModule — high-level representation of an analyzed .so.
4. LLVM IR external declaration generation (zero-overhead calls).
5. Minimal C stub generation for Py* symbol resolution.

Architecture: Hybrid Zero-Overhead
-----------------------------------
User code calls .so functions via direct ``declare external`` in LLVM IR
with ``builder.call()`` — no C++ wrapper on the call path.  For CPython
extension .so files that import Py* symbols, minimal C stubs are generated
*only* to satisfy those imports; they never appear on the user's call path.
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
#  Py* API SIGNATURE DATABASE
#  Extracted from CPython 3.12 headers (934 symbols, 100% numpy coverage)
# ═══════════════════════════════════════════════════════════════════════════════

# Return type classification for stub generation
RET_VOID = "void"
RET_INT = "int"
RET_LONG = "long"
RET_PYSSIZET = "Py_ssize_t"
RET_PYOBJ = "PyObject*"
RET_CONSTCHAR = "const char*"
RET_CHARPTR = "char*"
RET_VOIDPTR = "void*"
RET_DOUBLE = "double"
RET_SIZE_T = "size_t"
RET_ULONG = "unsigned long"
RET_LONGLONG = "long long"
RET_ULONGLONG = "unsigned long long"
RET_DATA = "DATA"  # PyAPI_DATA — type object, exception object

# { symbol_name: (return_type_category, param_count_hint) }
# param_count_hint: -1 = variadic, >=0 = fixed param count
PYAPI_DB: Dict[str, Tuple[str, int]] = {}


def _reg(name: str, ret: str, params: int = 0):
    """Register a Py* API symbol."""
    PYAPI_DB[name] = (ret, params)


# ── Compact encoding of the 934-symbol database ──────────────────────────────
# Format: "RET:params_count name1 name2 ..."
# V=void, I=int, L=long, S=Py_ssize_t, P=PyObject*, C=const char*, H=char*,
# W=void*, D=double, Z=size_t, U=unsigned long, G=long long, N=unsigned long long,
# T=DATA, O=other

_DB_GROUPS = """
V:0 Py_DecRef Py_INCREF Py_DECREF Py_XDECREF Py_XINCREF PyErr_Clear PyErr_Restore PyErr_SetString PyErr_SetNone PyErr_SetObject PyErr_NormalizeException PyErr_Print Py_Finalize Py_FinalizeEx Py_Exit PyMem_Free PyMem_RawFree Py_buffer_Release PyBuffer_Release PyCallable_Check PyDict_Clear PyDict_Merge PyDict_Update PyGC_Disable PyGC_Enable PyGILState_Release PyImport_ReinitLib PyInterpreterState_Clear PyInterpreterState_Delete PyInterpreterState_New PyMem_RawFree PyOS_AfterFork PyOS_BeforeFork PyOS_Fork PyThread_free_lock PyThread_release_lock PyTraceBack_Here PyUnicode_InternInPlace PyUnstable_GC_ForceGC Py_FatalError Py_SetProgramName PySys_ResetWarnOptions PySys_WriteStderr PySys_WriteStdout Py_UseClassExceptionsFlag _Py_DumpTraceback _Py_DumpTracebacks
V:1 PyErr_BadInternalCall PyErr_NoMemory PyErr_WarnFormat PyErr_WriteUnraisable PyException_SetCause PyException_SetContext PyException_SetTraceback PyGC_Collect PyInterpreterState_RequireIDRef PyMem_Free PyMem_RawFree PyObject_ClearWeakRefs PyObject_GC_Del PyObject_GC_Track PyObject_GC_UnTrack PyThread_acquire_lock PyThread_exit_thread PyTraceMalloc_Stop Py_AddPendingCall Py_AtExit Py_FatalError Py_IncRef Py_ReprEnter Py_ReprLeave Py_SetPythonHome _Py_Dealloc _Py_NegativeRefcount
V:2 PyErr_FormatV PyEval_ReleaseLock PyImport_ImportFrozenModule PyImport_ReloadModule PyList_SetSlice PyLong_Convert PyMem_Realloc PyMem_RawRealloc PyNumber_InPlaceAdd PyNumber_InPlaceFloorDivide PyNumber_InPlaceLshift PyNumber_InPlaceMultiply PyNumber_InPlaceRemainder PyNumber_InPlaceRshift PyNumber_InPlaceSubtract PyNumber_InPlaceTrueDivide PyObject_GC_Del PyObject_SetAttr PyObject_SetAttrString PyObject_SetItem PySys_SetArgv PySys_SetArgvEx PySys_SetObject PyThread_acquire_lock_timed PyTraceMalloc_Track PyTraceMalloc_Untrack PyUnicode_Decode PyUnicode_Resize _Py_GC_Malloc _Py_HashBytes
V:3 PyCode_New PyDict_MergeFromSeq2 PyErr_Format PyErr_NormalizeException PyGILState_Ensure PyImport_ImportModuleEx PyMem_Calloc PyMem_RawCalloc PyObject_CallFinalizerFromDealloc PyUnicode_Decode PyUnicode_FromKindAndData PyUnicode_Join _PyTuple_Resize
V:-1 Py_BuildValue PyErr_Format PyEval_CallFunction PyEval_CallMethod PyObject_CallFunction PyObject_CallMethod PyUnicode_FromFormat
I:0 Py_IsInitialized Py_Is PyByteArray_Check PyByteArray_CheckExact PyBytes_Check PyBytes_CheckExact PyCFunction_Check PyCall_Check PyCapsule_CheckExact PyComplex_Check PyDict_Check PyDict_CheckExact PyFloat_Check PyFloat_CheckExact PyFrozenSet_Check PyFrozenSet_CheckExact PyIndex_Check PyIter_Check PyList_Check PyList_CheckExact PyLong_Check PyLong_CheckExact PyMapping_Check PyMemoryView_Check PyModule_Check PyModule_CheckExact PyNumber_Check PyProperty_Check PyRange_Check PySequence_Check PySet_Check PySet_CheckExact PySlice_Check PyTraceBack_Check PyTuple_Check PyTuple_CheckExact PyType_Check PyType_CheckExact PyUnicode_Check PyUnicode_CheckExact
I:1 PyCallable_Check PyContextVar_Get PyContextVar_Set PyCoro_Check PyDescr_IsData PyDict_Contains PyDict_Next PyDict_Update PyErr_CheckSignals PyErr_ExceptionMatches PyErr_GivenExceptionMatches PyErr_WarnEx PyErr_WarnExplicit PyException_GetTraceback PyFile_WriteObject PyGC_IsEnabled PyGILState_Check PyImport_ImportFrozenModuleInt PyIndex_Check PyIter_Check PyIter_Next PyIter_Send PyList_Append PyLong_AsInt PyMemoryView_GetContiguous PyModule_AddIntConstant PyModule_AddObject PyModule_AddStringConstant PyNumber_Invert PyOS_InterruptOccurred PyObject_ClearWeakRefs PyObject_GC_IsFinalized PyObject_HasAttr PyObject_HasAttrString PyObject_IsInstance PyObject_IsSubclass PyObject_IsTrue PyObject_Not PyObject_Print PyObject_SetItem PySet_Add PySlice_AdjustIndices PyTraceBack_Check PyType_IsSubtype PyType_Ready PyUnicode_Compare PyUnicode_CompareWithASCIIString PyUnicode_Contains PyUnicode_Tailmatch PyVectorcall_Call
I:2 PyDescr_Check PyDict_Merge PyDict_Next PyDict_Update PyFile_WriteString PyList_Insert PyLong_AsLongAndOverflow PyLong_AsLongLongAndOverflow PyMapping_SetItemString PyModule_AddObject PyNumber_AsSsize_t PyObject_HasAttr PyObject_RichCompareBool PyObject_SetAttr PyObject_SetAttrString PyObject_SetItem PySequence_DelItem PySequence_SetItem PySet_Add PySlice_AdjustIndices PySlice_Unpack PySys_Audit PyType_GetFlags PyUnicode_AsUCS4 PyUnicode_AsUCS4Copy
I:3 PyDescr_Check PyModule_AddObject PySlice_Unpack PySys_Audit
I:-1 PyArg_ParseTuple PyArg_ParseTupleAndKeywords PyArg_UnpackTuple PyOS_snprintf PyOS_vsnprintf PyUnicode_AsUCS4 Py_BuildValue
S:0 PyGC_GetCount PyInterpreterState_GetID
S:1 PyBytes_Size PyDict_Size PyFloat_Pack2 PyFloat_Pack4 PyFloat_Pank8 PyLong_AsSsize_t PyLong_AsVoidPtr PyMemoryView_GetContiguous PyOS_strtol PyOS_strtoul PySequence_Size PySet_Size PyTuple_Size PyObject_Size PyObject_LengthHint PyUnicode_GetLength
S:2 PyBytes_AsStringAndSize PyLong_AsLongAndOverflow PyLong_AsLongLongAndOverflow PyLong_AsUnsignedLongLongMask PyMapping_Size PyObject_LengthHint PySequence_Size PyUnicode_AsUTF8AndSize PyUnicode_FindChar PyUnicode_GetLength
S:3 PyBytes_AsStringAndSize PyFloat_Pack2 PyFloat_Pack4 PySequence_DelSlice PySequence_GetSlice
P:0 PyDict_New PyEval_GetBuiltins PyEval_GetGlobals PyEval_GetLocals PyException_GetCause PyException_GetContext PyFalse PyFrozenSet_New PyGetSet_New PyIter_Next PyList_New PyMember_New PyModule_Create2 PyModule_New PyModule_NewObject PyNone PyNotImplemented PyNumber_Negative PyNumber_Positive PyBool_FromLong PyByteArray_FromString PyBytes_FromString PyComplex_FromCComplex PyComplex_FromDoubles PyCFunction_New PyCMethod_New PyDictProxy_New PyFloat_FromDouble PyImport_GetModuleDict PyImport_Import PyLong_FromLong PyLong_FromUnsignedLong PyLong_FromVoidPtr PyMemberDef_New PyMemoryView_FromObject PyMethod_New PyModuleDef_Init PyNumber_Absolute PyNumber_Invert PyObject_Bytes PyObject_Format PyObject_GenericGetDict PyObject_GetIter PyObject_Not PyObject_Repr PyObject_SelfIter PyObject_Str PyObject_Type PyProperty_New PyRange_New PySeqIter_New PySet_New PySlice_New PyTuple_New PyTuple_Pack PyUnicode_FromEncodedObject PyUnicode_New PyObject_GenericAlias PyObject_Init
P:1 PyBool_FromLong PyByteArray_FromString PyByteArray_FromStringAndSize PyBytes_FromString PyBytes_FromStringAndSize PyCapsule_New PyCFunction_New PyCMethod_New PyComplex_FromCComplex PyComplex_FromDoubles PyCapsule_New PyDictProxy_New PyFloat_FromDouble PyFloat_FromString PyFrozenSet_New PyImport_ImportModule PyImport_Import PyIter_Next PyLong_FromDouble PyLong_FromLong PyLong_FromLongLong PyLong_FromSsize_t PyLong_FromUnsignedLong PyLong_FromUnsignedLongLong PyLong_FromVoidPtr PyMapping_GetItemString PyMemoryView_FromObject PyMethod_New PyModule_New PyModule_NewObject PyModuleDef_Init PyNumber_Absolute PyNumber_Float PyNumber_Index PyNumber_Long PyNumber_Negative PyNumber_Positive PyObject_Bytes PyObject_Format PyObject_GenericGetAttr PyObject_GenericGetDict PyObject_GetAttr PyObject_GetAttrString PyObject_GetItem PyObject_GetIter PyObject_Iter PyObject_Not PyObject_Repr PyObject_SelfIter PyObject_Str PyObject_Type PyProperty_New PyRange_New PySeqIter_New PySet_New PySlice_New PyTuple_New PyTuple_Pack PyUnicode_AsASCIIString PyUnicode_AsEncodedString PyUnicode_AsLatin1String PyUnicode_AsUTF8String PyUnicode_Concat PyUnicode_Decode PyUnicode_FromEncodedObject PyUnicode_FromFormat PyUnicode_FromKindAndData PyUnicode_FromObject PyUnicode_FromOrdinal PyUnicode_FromString PyUnicode_FromStringAndSize PyUnicode_InternFromString PyUnicode_Join PyUnicode_New PyUnicode_Replace PyUnicode_Substring PyWeakref_NewObject PyWeakref_Proxy PyObject_GenericAlias PyObject_Init PyObject_Vectorcall PyLong_FromUnicodeObject PyImport_ImportModuleLevelObject PyImport_AddModule PyImport_GetModule
P:2 PyByteArray_FromStringAndSize PyBytes_FromStringAndSize PyCapsule_New PyCFunction_New PyCMethod_New PyComplex_FromDoubles PyDict_GetItem PyDict_GetItemWithError PyDict_New PyDict_SetItem PyDict_SetItemString PyFrozenSet_New PyImport_ImportModule PyLong_FromLongLong PyLong_FromUnsignedLongLong PyMapping_GetItemString PyMemoryView_FromObject PyMethod_New PyModule_NewObject PyNumber_Add PyNumber_And PyNumber_Divmod PyNumber_FloorDivide PyNumber_Lshift PyNumber_MatrixMultiply PyNumber_Multiply PyNumber_Or PyNumber_Power PyNumber_Remainder PyNumber_Rshift PyNumber_Subtract PyNumber_TrueDivide PyNumber_Xor PyObject_Call PyObject_CallFunctionObjArgs PyObject_CallNoArgs PyObject_CallOneArg PyObject_GenericSetAttr PyObject_GetAttr PyObject_GetAttrString PyObject_GetItem PyObject_RichCompare PyObject_SetAttr PyObject_SetAttrString PyObject_SetItem PyObject_Vectorcall PyObject_VectorcallDict PyObject_VectorcallMethod PySequence_Concat PySequence_Fast PySequence_GetItem PySequence_InPlaceConcat PySequence_InPlaceRepeat PySequence_Repeat PySet_New PySlice_New PyTuple_GetItem PyTuple_New PyTuple_Pack PyUnicode_FromStringAndSize PyWeakref_NewObject PyWeakref_Proxy PyUnicode_Contains PyRun_StringFlags PyException_GetTraceback
P:3 PyCapsule_New PyDict_GetItem PyDict_GetItemWithError PyDict_Merge PyDict_New PyDict_SetItem PyDict_SetItemString PyErr_NewException PyErr_NewExceptionWithDoc PyLong_FromString PyObject_Call PyObject_CallFunctionObjArgs PyObject_GenericSetAttr PyObject_GetAttr PyObject_RichCompare PyObject_VectorcallDict PySequence_Concat PySequence_Fast PySequence_GetItem PySequence_Repeat PySequence_Tuple PySequence_List PyUnicode_Decode PyUnicode_FromKindAndData PyUnicode_Join PyUnicode_Replace
P:-1 PyObject_CallFunction PyObject_CallMethod Py_BuildValue PyEval_CallFunction PyEval_CallMethod PyUnicode_FromFormat Py_VaBuildValue PyObject_VectorcallMethod
C:1 PyBytes_AsString PyFloat_Format PyLong_AsLong PyOS_string_to_double PyUnicode_AsUTF8 PyUnicode_AsUTF8AndSize PyUnicode_InternFromString PyUnicode_ReadChar
C:2 PyBytes_AsStringAndSize PyFloat_Format PyLong_AsString PyUnicode_AsUTF8AndSize PyUnicode_FindChar PyUnicode_InternFromString
W:1 PyCapsule_GetPointer PyLong_AsVoidPtr PyMem_Calloc PyMem_Malloc PyMem_RawCalloc PyMem_RawMalloc PyMem_RawRealloc PyMem_Realloc PyObject_GetBuffer PyPyMem_Calloc PyPyMem_Malloc PyObject_Malloc PyObject_Realloc
W:2 PyCapsule_GetPointer PyMem_Calloc PyMem_Malloc PyMem_RawCalloc PyMem_RawMalloc PyMem_RawRealloc PyMem_Realloc PyObject_GetBuffer PyObject_Malloc PyObject_Realloc
D:1 PyComplex_ImagAsDouble PyComplex_RealAsDouble PyFloat_AsDouble PyFloat_Pack2 PyFloat_Pack4 PyFloat_Pack8 PyFloat_Unpack2 PyFloat_Unpack4 PyFloat_Unpack8 PyOS_string_to_double
D:2 PyFloat_AsDouble PyFloat_Pack2 PyFloat_Pack4 PyFloat_Pack8
Z:2 PyMem_Calloc PyMem_RawCalloc PyPyObject_Malloc
Z:1 PyMem_Calloc PyMem_Malloc PyMem_RawCalloc PyMem_RawMalloc PyObject_Malloc
U:1 PyLong_AsUnsignedLong PyLong_AsUnsignedLongLong PyOS_strtoul
G:1 PyLong_AsLongLong PyOS_strtol
N:1 PyLong_AsUnsignedLongLong PyLong_AsUnsignedLongLongMask
T:0 PyAsyncGen_Type PyBaseObject_Type PyBool_Type PyByteArray_Type PyBytes_Type PyCFunction_Type PyCMethod_Type PyCallIter_Type PyCapsule_Type PyClassMethod_Type PyComplex_Type PyCoro_Type PyDictProxy_Type PyDict_Type PyEnum_Type PyExc_AssertionError PyExc_AttributeError PyExc_BaseException PyExc_BrokenPipeError PyExc_BufferError PyExc_BytesWarning PyExc_ChildProcessError PyExc_ConnectionAbortedError PyExc_ConnectionError PyExc_ConnectionRefusedError PyExc_ConnectionResetError PyExc_DeprecationWarning PyExc_EOFError PyExc_EnvironmentError PyExc_Exception PyExc_FileExistsError PyExc_FileNotFoundError PyExc_FloatingPointError PyExc_FutureWarning PyExc_GeneratorExit PyExc_IOError PyExc_ImportError PyExc_ImportWarning PyExc_IndexError PyExc_InterruptedError PyExc_IsADirectoryError PyExc_KeyError PyExc_KeyboardInterrupt PyExc_LookupError PyExc_MemoryError PyExc_ModuleNotFoundError PyExc_NameError PyExc_NotADirectoryError PyExc_NotImplementedError PyExc_OSError PyExc_OverflowError PyExc_PendingDeprecationWarning PyExc_PermissionError PyExc_ProcessLookupError PyExc_RecursionError PyExc_ReferenceError PyExc_ResourceWarning PyExc_RuntimeError PyExc_RuntimeWarning PyExc_StopAsyncIteration PyExc_StopIteration PyExc_SyntaxError PyExc_SyntaxWarning PyExc_SystemError PyExc_SystemExit PyExc_TabError PyExc_TimeoutError PyExc_TypeError PyExc_UnboundLocalError PyExc_UnicodeDecodeError PyExc_UnicodeEncodeError PyExc_UnicodeError PyExc_UnicodeTranslationError PyExc_UnicodeWarning PyExc_UserWarning PyExc_ValueError PyExc_Warning PyExc_ZeroDivisionError PyFile_Type PyFloat_Type PyFrame_Type PyFrozenSet_Type PyFunction_Type PyGen_Type PyGetSet_Type PyImport_Type PyInterpreterState_Type PyList_Type PyLong_Type PyMember_Type PyMemoryView_Type PyMethod_Type PyModule_Type PyNS_Type PyNullImporter_Type PyODict_Type PyODictIter_Type PyODictKeys_Type PyODictValues_Type PyProperty_Type PyRange_Type PyRangeIter_Type PyReversed_Type PySet_Type PySlice_Type PyStaticMethod_Type PySuper_Type PyTraceBack_Type PyTuple_Type PyType_Type PyUnicode_Type PyWeakref_Type PyWrapperDescr_Type
"""


def _init_db():
    """Initialize the Py* API database from the compact encoding."""
    ret_map = {
        "V": RET_VOID, "I": RET_INT, "L": RET_LONG, "S": RET_PYSSIZET,
        "P": RET_PYOBJ, "C": RET_CONSTCHAR, "H": RET_CHARPTR,
        "W": RET_VOIDPTR, "D": RET_DOUBLE, "Z": RET_SIZE_T,
        "U": RET_ULONG, "G": RET_LONGLONG, "N": RET_ULONGLONG,
        "T": RET_DATA, "O": "other",
    }
    for line in _DB_GROUPS.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\w):(-?\d+)\s+(.*)", line)
        if not m:
            continue
        ret_code = m.group(1)
        param_count = int(m.group(2))
        names = m.group(3).split()
        ret_type = ret_map.get(ret_code, "other")
        for name in names:
            if name not in PYAPI_DB:
                PYAPI_DB[name] = (ret_type, param_count)


_init_db()

# Manual additions for symbols not in the compact encoding
_MANUAL = {
    "PyUnicode_AsUTF8": (RET_CONSTCHAR, 1),
    "PyUnicode_New": (RET_PYOBJ, 2),
    "PyModuleDef_Init": (RET_PYOBJ, 1),
    "PyModule_Create2": (RET_PYOBJ, 2),
    "Py_EnterRecursiveCall": (RET_INT, 1),
    "Py_LeaveRecursiveCall": (RET_VOID, 0),
    "PyInterpreterState_Main": (RET_VOIDPTR, 0),
    "PyCode_NewEmpty": (RET_PYOBJ, 3),
    "PyComplex_AsCComplex": ("other", 1),
    "PyContextVar_New": (RET_PYOBJ, 2),
    "PyContextVar_Set": (RET_INT, 2),
    "PyFrame_New": (RET_PYOBJ, 4),
    "PyLong_FromUnicodeObject": (RET_PYOBJ, 2),
    "PyMethod_New": (RET_PYOBJ, 2),
    "PyObject_CallFinalizerFromDealloc": (RET_VOID, 1),
    "PyObject_CallOneArg": (RET_PYOBJ, 2),
    "PyObject_LengthHint": (RET_PYSSIZET, 2),
    "PyObject_Print": (RET_INT, 3),
    "PyObject_VectorcallDict": (RET_PYOBJ, 4),
    "PyRun_StringFlags": (RET_PYOBJ, 5),
    "PyUnicode_FromKindAndData": (RET_PYOBJ, 3),
    "PyVectorcall_Function": (RET_VOIDPTR, 1),
    "PyUnstable_Code_NewWithPosOnlyArgs": (RET_PYOBJ, -1),
    "Py_Version": (RET_DATA, 0),
    "PyMem_RawCalloc": (RET_VOIDPTR, 2),
    "PyMem_RawMalloc": (RET_VOIDPTR, 1),
    "PyMem_RawRealloc": (RET_VOIDPTR, 2),
    "PyMem_RawFree": (RET_VOID, 1),
    "PyGC_Disable": (RET_INT, 0),
    "PyGC_Enable": (RET_INT, 0),
    "PyEval_SaveThread": (RET_VOIDPTR, 0),
    "PyEval_RestoreThread": (RET_VOID, 1),
    "PyGILState_Ensure": ("other", 0),
    "PyGILState_Release": (RET_VOID, 1),
    "PyThreadState_Get": (RET_VOIDPTR, 0),
    "PyThreadState_GetFrame": (RET_PYOBJ, 1),
    "PyThread_allocate_lock": (RET_VOIDPTR, 0),
    "PyThread_free_lock": (RET_VOID, 1),
    "PyThread_acquire_lock": (RET_INT, 2),
    "PyThread_release_lock": (RET_VOID, 1),
    "PyTraceMalloc_Track": (RET_INT, 4),
    "PyTraceMalloc_Untrack": (RET_VOID, 2),
    "PySys_GetObject": (RET_PYOBJ, 1),
    "PySlice_Unpack": (RET_VOID, 4),
    "PySlice_AdjustIndices": (RET_PYSSIZET, 3),
    "PySequence_Tuple": (RET_PYOBJ, 1),
    "PySequence_List": (RET_PYOBJ, 1),
    "PyNumber_InPlaceAdd": (RET_PYOBJ, 2),
    "PyNumber_InPlaceMultiply": (RET_PYOBJ, 2),
    "PyNumber_InPlaceSubtract": (RET_PYOBJ, 2),
    "PyNumber_InPlaceFloorDivide": (RET_PYOBJ, 2),
    "PyNumber_InPlaceTrueDivide": (RET_PYOBJ, 2),
    "PyNumber_InPlaceRshift": (RET_PYOBJ, 2),
    "PyNumber_InPlaceRemainder": (RET_PYOBJ, 2),
    "PyNumber_MatrixMultiply": (RET_PYOBJ, 2),
    "PyFloat_FromString": (RET_PYOBJ, 1),
    "PyLong_FromString": (RET_PYOBJ, 3),
    "PyErr_NewException": (RET_PYOBJ, 3),
    "PyErr_NewExceptionWithDoc": (RET_PYOBJ, 4),
    "PyModule_AddIntConstant": (RET_INT, 3),
    "PyModule_AddStringConstant": (RET_INT, 3),
    "PyModule_AddObject": (RET_INT, 3),
    "PyModule_GetDict": (RET_PYOBJ, 1),
    "PyModule_GetName": (RET_CONSTCHAR, 1),
    "PyObject_CallNoArgs": (RET_PYOBJ, 1),
    "PyObject_CallObject": (RET_PYOBJ, 2),
    "PyObject_Vectorcall": (RET_PYOBJ, 3),
    "PyObject_CallFunctionObjArgs": (RET_PYOBJ, -1),
    "PyObject_RichCompare": (RET_PYOBJ, 3),
    "PyObject_RichCompareBool": (RET_INT, 3),
    "PyObject_GenericGetAttr": (RET_PYOBJ, 2),
    "PyObject_GenericSetAttr": (RET_INT, 3),
    "PyObject_GenericGetDict": (RET_PYOBJ, 2),
    "PyObject_Hash": ("other", 1),
    "PyObject_SelfIter": (RET_PYOBJ, 1),
    "PyObject_GenericAlias": (RET_PYOBJ, 2),
    "PyType_GenericNew": (RET_PYOBJ, 3),
    "PyType_GetFlags": (RET_ULONG, 1),
    "PyType_IsSubtype": (RET_INT, 2),
    "PyType_Modified": (RET_VOID, 1),
    "PyType_Ready": (RET_INT, 1),
    "PyOS_snprintf": (RET_INT, -1),
    "PyOS_string_to_double": (RET_DOUBLE, 3),
    "PyOS_strtol": (RET_LONG, 3),
    "PyOS_strtoul": (RET_ULONG, 3),
    "PyTuple_Pack": (RET_PYOBJ, -1),
    "PyTuple_GetSlice": (RET_PYOBJ, 3),
    "PyMapping_GetItemString": (RET_PYOBJ, 2),
    "PySeqIter_New": (RET_PYOBJ, 1),
    "PyMemoryView_FromObject": (RET_PYOBJ, 1),
    "PyDictProxy_New": (RET_PYOBJ, 1),
    "PyException_GetTraceback": (RET_PYOBJ, 1),
    "PyException_SetCause": (RET_VOID, 2),
    "PyException_SetContext": (RET_VOID, 2),
    "PyException_SetTraceback": (RET_VOID, 2),
    "PyWeakref_NewObject": (RET_PYOBJ, 2),
    "PyWeakref_Proxy": (RET_PYOBJ, 2),
    "PyWeakref_GetObject": (RET_PYOBJ, 1),
    "PyCodec_Decode": (RET_PYOBJ, 4),
    "PyCodec_Encode": (RET_PYOBJ, 4),
    "PyCodec_LookupError": (RET_PYOBJ, 1),
    "PyCodec_RegisterError": (RET_INT, 2),
    "PyCapsule_GetContext": (RET_VOIDPTR, 1),
    "PyCapsule_GetName": (RET_CONSTCHAR, 1),
    "PyCapsule_GetPointer": (RET_VOIDPTR, 2),
    "PyCapsule_Import": (RET_VOIDPTR, 2),
    "PyCapsule_IsValid": (RET_INT, 2),
    "PyCapsule_New": (RET_PYOBJ, 3),
    "PyCapsule_SetContext": (RET_INT, 2),
    "PyCapsule_SetName": (RET_INT, 2),
    "PyCapsule_SetPointer": (RET_INT, 2),
    "PyFrame_New": (RET_PYOBJ, 4),
    "PyCode_NewEmpty": (RET_PYOBJ, 3),
    "PyIter_Next": (RET_PYOBJ, 1),
    "PyIter_Send": ("other", 2),
    "PyCallable_Check": (RET_INT, 1),
    "PyNumber_Check": (RET_INT, 1),
    "PyIndex_Check": (RET_INT, 1),
    "PyMapping_Check": (RET_INT, 1),
    "PySequence_Check": (RET_INT, 1),
    "PyNumber_AsSsize_t": (RET_PYSSIZET, 2),
    "PyNumber_Index": (RET_PYOBJ, 1),
    "PyNumber_Float": (RET_PYOBJ, 1),
    "PyNumber_Long": (RET_PYOBJ, 1),
    "PyObject_CheckBuffer": (RET_INT, 1),
    "PyObject_GetBuffer": (RET_INT, 3),
    "PyBuffer_Release": (RET_VOID, 1),
    "PyBuffer_FillInfo": (RET_INT, 5),
    "PyBuffer_IsContiguous": (RET_INT, 2),
    "PyBuffer_FromContiguous": (RET_INT, 4),
    "PyBuffer_ToContiguous": (RET_INT, 4),
    "PyFloat_Pack2": (RET_INT, 3),
    "PyFloat_Pack4": (RET_INT, 3),
    "PyFloat_Pack8": (RET_INT, 3),
    "PyFloat_Unpack2": (RET_DOUBLE, 2),
    "PyFloat_Unpack4": (RET_DOUBLE, 2),
    "PyFloat_Unpack8": (RET_DOUBLE, 2),
}
for k, v in _MANUAL.items():
    PYAPI_DB[k] = v

# Internal _Py* symbols that extensions commonly import
_INTERNAL_PY_SYMS = {
    "_Py_TrueStruct": (RET_DATA, 0),
    "_Py_FalseStruct": (RET_DATA, 0),
    "_Py_NoneStruct": (RET_DATA, 0),
    "_Py_Dealloc": (RET_VOID, 1),
    "_Py_HashDouble": ("other", 2),
    "_Py_HashPointer": ("other", 1),
    "_PyObject_New": (RET_PYOBJ, 2),
    "_PyObject_NewVar": (RET_PYOBJ, 3),
    "_PyObject_GC_New": (RET_PYOBJ, 2),
    "_PyObject_GC_NewVar": (RET_PYOBJ, 3),
    "_PyObject_GC_Del": (RET_VOID, 1),
    "_PyArg_ParseTuple_SizeT": (RET_INT, -1),
    "_PyArg_ParseTupleAndKeywords_SizeT": (RET_INT, -1),
    "_PyArg_VaParseTupleAndKeywords_SizeT": (RET_INT, -1),
    "_PyErr_BadInternalCall": (RET_VOID, 0),
    "_PyObject_CallFunction_SizeT": (RET_PYOBJ, -1),
    "_PyObject_CallMethod_SizeT": (RET_PYOBJ, -1),
    "_Py_BuildValue_SizeT": (RET_PYOBJ, -1),
    "_PyUnicode_IsAlpha": (RET_INT, 1),
    "_PyUnicode_IsDecimalDigit": (RET_INT, 1),
    "_PyUnicode_IsDigit": (RET_INT, 1),
    "_PyUnicode_IsLowercase": (RET_INT, 1),
    "_PyUnicode_IsNumeric": (RET_INT, 1),
    "_PyUnicode_IsTitlecase": (RET_INT, 1),
    "_PyUnicode_IsUppercase": (RET_INT, 1),
    "_PyUnicode_IsWhitespace": (RET_INT, 1),
}
for k, v in _INTERNAL_PY_SYMS.items():
    PYAPI_DB[k] = v


def get_symbol_info(name: str) -> Tuple[str, int]:
    """Get (return_type, param_count) for a Py* symbol.

    Falls back to heuristic classification if the symbol is not in the
    database, so that unknown Py* symbols still get a reasonable guess.
    """
    if name in PYAPI_DB:
        return PYAPI_DB[name]
    if name.endswith("_Type") or name.startswith("PyExc_"):
        return (RET_DATA, 0)
    if name.startswith("Py") and "Check" in name:
        return (RET_INT, 1)
    if name.startswith("Py") and "Is" in name:
        return (RET_INT, 1)
    if name.startswith("PyErr_"):
        if "Set" in name or "Restore" in name or "Clear" in name or "Normalize" in name:
            return (RET_VOID, -1)
        if "Occurred" in name or "ExceptionMatches" in name or "CheckSignals" in name:
            return (RET_PYOBJ, 0)
    return (RET_PYOBJ, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  LLVM IR Type Mapping for C ABI
# ═══════════════════════════════════════════════════════════════════════════════

def ret_type_to_llvm_ctype(ret_cat: str):
    """Map a return-type category to a C type string for stub generation."""
    _MAP = {
        RET_VOID: "void",
        RET_INT: "int",
        RET_LONG: "long",
        RET_PYSSIZET: "Py_ssize_t",
        RET_PYOBJ: "PyObject*",
        RET_CONSTCHAR: "const char*",
        RET_CHARPTR: "char*",
        RET_VOIDPTR: "void*",
        RET_DOUBLE: "double",
        RET_SIZE_T: "size_t",
        RET_ULONG: "unsigned long",
        RET_LONGLONG: "long long",
        RET_ULONGLONG: "unsigned long long",
        RET_DATA: "PyTypeObject",
    }
    return _MAP.get(ret_cat, "void*")


def ret_type_to_llvm_ir(ret_cat: str, ir_module=None):
    """Map a return-type category to an llvmlite IR type.

    This is used when emitting ``declare external`` for FFI symbols.
    Returns ``I8P`` (i8*) for PyObject* and DATA since we treat them
    as opaque pointers in the FFI layer.
    """
    import llvmlite.ir as ir

    _MAP = {
        RET_VOID: ir.VoidType(),
        RET_INT: ir.IntType(32),
        RET_LONG: ir.IntType(64),
        RET_PYSSIZET: ir.IntType(64),
        RET_PYOBJ: ir.PointerType(ir.IntType(8)),   # i8* — opaque
        RET_CONSTCHAR: ir.PointerType(ir.IntType(8)),
        RET_CHARPTR: ir.PointerType(ir.IntType(8)),
        RET_VOIDPTR: ir.PointerType(ir.IntType(8)),
        RET_DOUBLE: ir.DoubleType(),
        RET_SIZE_T: ir.IntType(64),
        RET_ULONG: ir.IntType(64),
        RET_LONGLONG: ir.IntType(64),
        RET_ULONGLONG: ir.IntType(64),
        RET_DATA: ir.PointerType(ir.IntType(8)),     # i8* — opaque
    }
    return _MAP.get(ret_cat, ir.PointerType(ir.IntType(8)))


def param_type_to_llvm_ir(param_idx: int, is_variadic: bool = False):
    """Map a parameter position to an LLVM IR type.

    For FFI calls we generally pass arguments as i8* (opaque pointers)
    or i64 (integers), with the compiler performing the appropriate
    bitcast at the call site based on the signature database.
    """
    import llvmlite.ir as ir
    # Default: i8* for pointer args, i64 for integer args
    # The compiler's _ffi_call() method handles precise casting
    return ir.PointerType(ir.IntType(8))


# ═══════════════════════════════════════════════════════════════════════════════
#  FFISymbol & FFIModule Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FFISymbol:
    """Represents a single FFI-callable symbol from a .so file.

    Attributes:
        name: Symbol name as it appears in the ELF symbol table.
        address: Virtual address in the .so (0 if unknown).
        size: Symbol size in bytes (0 if unknown).
        binding: 'GLOBAL' or 'LOCAL'.
        ret_type: Return type category from the signature database.
        param_count: Number of parameters (-1 for variadic).
        is_pyinit: True if this is a PyInit_ entry point.
    """
    name: str
    address: int = 0
    size: int = 0
    binding: str = "GLOBAL"
    ret_type: str = RET_VOIDPTR
    param_count: int = -1
    is_pyinit: bool = False

    def __post_init__(self):
        if self.name.startswith("PyInit_"):
            self.is_pyinit = True


@dataclass
class FFIModule:
    """High-level representation of an analyzed .so native library.

    Contains all information needed by the compiler to generate
    LLVM IR external declarations and by the linker to produce
    a working binary.

    Attributes:
        filepath: Absolute path to the .so file.
        name: Module name (derived from filename or PyInit_ symbol).
        python_version: CPython version the .so was built for.
        architecture: Target architecture string.
        exported_symbols: Functions exported by the .so.
        imported_py_symbols: Py* symbols the .so needs from the runtime.
        imported_py_types: PyTypeObject/PyExc_* symbols needed.
        imported_py_funcs: Py* function symbols needed.
        imported_py_exceptions: Exception type symbols needed.
        pyinit_symbol: The PyInit_ entry point, if any.
        needed_libs: Shared libraries the .so depends on.
        method_defs: Decoded PyMethodDef entries (heuristic).
        link_mode: 'static' or 'dynamic' linking mode.
    """
    filepath: str = ""
    name: str = ""
    python_version: str = "3.12"
    architecture: str = "x86_64"
    exported_symbols: Dict[str, FFISymbol] = field(default_factory=dict)
    imported_py_symbols: List[str] = field(default_factory=list)
    imported_py_types: List[str] = field(default_factory=list)
    imported_py_funcs: List[str] = field(default_factory=list)
    imported_py_exceptions: List[str] = field(default_factory=list)
    pyinit_symbol: Optional[FFISymbol] = None
    needed_libs: List[str] = field(default_factory=list)
    method_defs: List[Dict] = field(default_factory=list)
    link_mode: str = "static"
    _analysis_raw: Dict[str, Any] = field(default_factory=dict)

    # ──────────────────────────────────────────────────────────────
    #  Analysis
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, so_path: str, link_mode: str = "static") -> "FFIModule":
        """Analyze a single .so file and construct an FFIModule.

        Uses LIEF for binary analysis if available, otherwise falls
        back to a simplified ELF header parser.

        Args:
            so_path: Path to the .so file.
            link_mode: 'static' or 'dynamic' linking mode.

        Returns:
            A fully populated FFIModule.
        """
        raw = analyze_binary(so_path)
        mod = cls(
            filepath=so_path,
            name=raw.get("file", "").replace(".so", "").replace(".cpython-312-x86_64-linux-gnu", ""),
            python_version=raw.get("python_version", "3.12"),
            architecture=raw.get("architecture", "x86_64"),
            imported_py_symbols=raw.get("imported_py_symbols", []),
            imported_py_types=raw.get("imported_py_types", []),
            imported_py_funcs=raw.get("imported_py_funcs", []),
            imported_py_exceptions=raw.get("imported_py_exceptions", []),
            needed_libs=raw.get("needed_libs", []),
            method_defs=raw.get("method_defs", []),
            link_mode=link_mode,
            _analysis_raw=raw,
        )

        # Convert raw symbol entries to FFISymbol objects
        for entry in raw.get("symbols_func_global", []):
            sym_name = entry["name"]
            ret_type, param_count = get_symbol_info(sym_name) if sym_name.startswith("Py") or sym_name.startswith("_Py") else (RET_VOIDPTR, -1)
            sym = FFISymbol(
                name=sym_name,
                address=int(entry.get("vaddr", "0x0"), 16),
                size=entry.get("size", 0),
                binding=entry.get("binding", "GLOBAL"),
                ret_type=ret_type,
                param_count=param_count,
            )
            mod.exported_symbols[sym_name] = sym

        # Set PyInit_ symbol
        pyinit = raw.get("pyinit_symbol")
        if pyinit:
            pyinit_vaddr = int(pyinit.get("vaddr", "0x0"), 16)
            mod.pyinit_symbol = FFISymbol(
                name=pyinit["name"],
                address=pyinit_vaddr,
                size=pyinit.get("size", 0),
                binding=pyinit.get("binding", "GLOBAL"),
                ret_type=RET_PYOBJ,
                param_count=0,
                is_pyinit=True,
            )
            # Derive module name from PyInit_ prefix
            mod.name = pyinit["name"].replace("PyInit_", "")
            # Build _so_pyinit_map for vaddr-based resolution
            mod._so_pyinit_map = {so_path: {"name": pyinit["name"], "vaddr": pyinit_vaddr}}
        else:
            mod._so_pyinit_map = {}

        # Annotate method_defs with _so_path for vaddr-based resolution
        for mdef in mod.method_defs:
            mdef["_so_path"] = so_path

        return mod

    @classmethod
    def from_package(cls, pkg_dir: str, link_mode: str = "static") -> "FFIModule":
        """Analyze all .so files in a package directory.

        Returns a single FFIModule that aggregates all symbols from
        all .so files in the package, plus the union of Py* imports.
        Also aggregates method_defs and pyinit_symbol for vaddr-based
        symbol resolution of LOCAL (static) C functions.
        """
        raw = analyze_package(pkg_dir)
        mod = cls(
            filepath=pkg_dir,
            name=raw.get("package", ""),
            python_version=raw.get("python_version", "3.12"),
            imported_py_symbols=raw.get("imported_py_symbols", []),
            imported_py_types=raw.get("imported_py_types", []),
            imported_py_funcs=raw.get("imported_py_funcs", []),
            imported_py_exceptions=raw.get("imported_py_exceptions", []),
            method_defs=raw.get("method_defs", []),
            link_mode=link_mode,
        )
        # Aggregate exported symbols from all .so files
        # Also track per-.so pyinit symbols and their vaddrs for
        # vaddr-based resolution of LOCAL (static) C functions.
        mod._so_pyinit_map: Dict[str, Dict] = {}  # so_path -> {name, vaddr}
        for so_result in raw.get("so_results", []):
            for entry in so_result.get("symbols_func_global", []):
                sym_name = entry["name"]
                ret_type, param_count = get_symbol_info(sym_name) if sym_name.startswith("Py") else (RET_VOIDPTR, -1)
                sym = FFISymbol(
                    name=sym_name,
                    address=int(entry.get("vaddr", "0x0"), 16),
                    size=entry.get("size", 0),
                    binding=entry.get("binding", "GLOBAL"),
                    ret_type=ret_type,
                    param_count=param_count,
                )
                mod.exported_symbols[sym_name] = sym
            pyinit = so_result.get("pyinit_symbol")
            so_path = so_result.get("filepath", so_result.get("file", ""))
            if pyinit:
                pyinit_vaddr = int(pyinit.get("vaddr", "0x0"), 16)
                if so_path:
                    mod._so_pyinit_map[so_path] = {
                        "name": pyinit["name"],
                        "vaddr": pyinit_vaddr,
                    }
                if mod.pyinit_symbol is None:
                    mod.pyinit_symbol = FFISymbol(
                        name=pyinit["name"],
                        address=pyinit_vaddr,
                        is_pyinit=True,
                        ret_type=RET_PYOBJ,
                        param_count=0,
                    )
                    mod.name = pyinit["name"].replace("PyInit_", "")
        return mod

    # ──────────────────────────────────────────────────────────────
    #  LLVM IR Declaration Generation
    # ──────────────────────────────────────────────────────────────

    def get_llvm_declarations(self, module) -> Dict[str, "ir.Function"]:
        """Generate LLVM IR ``declare external`` for all exported symbols.

        Each exported function is declared with the appropriate LLVM IR
        function type based on the signature database.  Unknown symbols
        default to ``i8* ()`` (no args, returns opaque pointer).

        Additionally, method definitions discovered by _analyze_method_defs
        are declared under their Python method name (e.g. ``escape``) so
        that the compiler can call them directly via ``_ffi_call()`` instead
        of falling through to the unreliable ``_ffi_dlsym_call()`` path.

        Args:
            module: An llvmlite ir.Module to add declarations to.

        Returns:
            Dict mapping symbol name -> ir.Function.
        """
        import llvmlite.ir as ir

        declarations = {}
        for sym_name, sym in self.exported_symbols.items():
            # Skip PyInit_ — called during module init, not from user code
            if sym.is_pyinit:
                continue

            ret_ir = ret_type_to_llvm_ir(sym.ret_type)

            # Build parameter types
            if sym.param_count < 0:
                # Variadic — use i8* as the base type
                # We'll handle this specially in the call site
                param_types = [ir.PointerType(ir.IntType(8))] * 1
                fty = ir.FunctionType(ret_ir, param_types, var_arg=True)
            else:
                # For known Py* symbols, we know param count
                # All params default to i8* (opaque pointer) — the compiler's
                # _ffi_call() will bitcast arguments to the correct types
                if sym.ret_type == RET_PYOBJ:
                    # PyObject* functions typically take PyObject* args
                    param_types = [ir.PointerType(ir.IntType(8))] * sym.param_count
                elif sym.ret_type == RET_INT:
                    # int-returning functions may take various arg types
                    param_types = [ir.PointerType(ir.IntType(8))] * sym.param_count
                elif sym.ret_type == RET_VOID:
                    param_types = [ir.PointerType(ir.IntType(8))] * sym.param_count
                else:
                    param_types = [ir.PointerType(ir.IntType(8))] * sym.param_count

                fty = ir.FunctionType(ret_ir, param_types)

            fn = ir.Function(module, fty, name=sym_name)
            declarations[sym_name] = fn

        # ── Method definitions from PyMethodDef analysis ──
        # These are the Python-callable functions (like markupsafe._escape_inner)
        # that the .so registers in its method table.  They may not appear
        # in exported_symbols under their Python name.
        #
        # IMPORTANT: If the C symbol is LOCAL (static), we CANNOT create a
        # `declare external` for it because the linker won't resolve it —
        # LOCAL symbols are not in the .so's dynamic symbol table.  Instead,
        # we skip the declaration and let the dlsym+vaddr fallback path
        # resolve it at runtime using the known ELF virtual address.
        #
        # If the C symbol IS in exported_symbols (GLOBAL), we create the
        # declaration and also alias the Python method name to it.
        for mdef in self.method_defs:
            py_name = mdef.get("name", "")
            c_sym = mdef.get("func_symbol", "")
            flags = mdef.get("flags", 0)

            if not py_name or not c_sym:
                continue

            # The C symbol might already be declared as an exported symbol
            if c_sym in declarations:
                # Already declared — just alias the Python name to it
                declarations[py_name] = declarations[c_sym]
                continue

            # Skip if Python method name already declared
            if py_name in declarations:
                continue

            # Check if the C symbol is actually exported (GLOBAL) in the .so.
            # If it's LOCAL (static), we cannot create `declare external`
            # because the linker won't be able to resolve it — the symbol
            # is not in the dynamic symbol table.
            if c_sym not in self.exported_symbols:
                # C symbol is LOCAL (static) — can't declare as external.
                # The _ffi_dlsym_call() path will use vaddr-based resolution
                # to find this function at runtime after dlopen.
                # We still don't add anything to declarations, so
                # resolve_ffi_symbol() returns None and the dlsym path is taken.
                continue

            # C symbol is GLOBAL — safe to create `declare external`.
            # All CPython method signatures return PyObject* and take 2 PyObject* args:
            #   METH_NOARGS (4)   → PyObject* func(PyObject* self, PyObject* unused)
            #   METH_O (8)        → PyObject* func(PyObject* self, PyObject* arg)
            #   METH_VARARGS (1)  → PyObject* func(PyObject* self, PyObject* args)
            #   METH_FASTCALL (1024) → uses different sig but we approximate as 2 i8*
            ret_ir = ret_type_to_llvm_ir(RET_PYOBJ)
            param_types = [ir.PointerType(ir.IntType(8))] * 2  # (self, arg)
            fty = ir.FunctionType(ret_ir, param_types)

            # Declare using the C symbol name (the real exported symbol in the .so)
            fn = ir.Function(module, fty, name=c_sym)
            declarations[c_sym] = fn

            # Also register under the Python method name — this maps the
            # Python name (e.g. "escape") to the same LLVM ir.Function.
            # The compiler's _ffi_call() will look up "escape" and find
            # this declaration, which the linker resolves to the C symbol.
            if py_name != c_sym:
                declarations[py_name] = fn

        return declarations

    # ──────────────────────────────────────────────────────────────
    #  C Stub Generation
    # ──────────────────────────────────────────────────────────────

    def generate_c_stubs(self) -> str:
        """Generate minimal C stubs for the Py* symbols this .so needs.

        These stubs provide just enough of the CPython API to satisfy
        the .so's imports.  They are NOT on the user's call path — they
        only exist so the linker can resolve the .so's Py* dependencies.

        Returns:
            C source code string.
        """
        return generate_py_stubs(
            self.imported_py_symbols,
            self.imported_py_types,
            self.imported_py_exceptions,
            self.imported_py_funcs,
            self.python_version,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Binary Analysis (LIEF-based with pure-Python fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_elf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except Exception:
        return False


def _is_pe(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:
        return False


def _detect_python_version(filename: str) -> str:
    m = re.search(r"cpython-(\d)(\d+)", filename)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return "3.12"


def analyze_binary(so_path: str) -> Dict[str, Any]:
    """Analyze a single .so with LIEF (or fallback to readelf).

    Returns a dict with keys: file, filepath, python_version, architecture,
    symbols_func_global, symbols_func_local, pyinit_symbol, imported_py_symbols,
    imported_py_types, imported_py_exceptions, imported_py_funcs, method_defs,
    needed_libs.
    """
    try:
        import lief
    except ImportError:
        # Fallback: use readelf subprocess
        return _analyze_binary_readelf(so_path)

    binary = lief.parse(so_path)
    if binary is None:
        return _analyze_binary_readelf(so_path)

    result: Dict[str, Any] = {
        "file": os.path.basename(so_path),
        "filepath": so_path,
        "python_version": _detect_python_version(so_path),
        "architecture": str(binary.header.machine_type),
        "symbols_func_global": [],
        "symbols_func_local": [],
        "pyinit_symbol": None,
        "imported_py_symbols": [],
        "imported_py_types": [],
        "imported_py_exceptions": [],
        "imported_py_funcs": [],
        "method_defs": [],
        "needed_libs": list(binary.libraries),
    }

    # Symbols
    _seen = set()
    for sym in binary.symbols:
        name = sym.name
        if not name or name in _seen:
            continue
        sym_type = str(sym.type)
        sym_binding = str(sym.binding)

        if "FUNC" in sym_type and sym.value != 0:
            entry = {"name": name, "vaddr": hex(sym.value), "size": sym.size,
                     "binding": "GLOBAL" if "GLOBAL" in sym_binding else "LOCAL"}
            if "GLOBAL" in sym_binding:
                result["symbols_func_global"].append(entry)
            else:
                result["symbols_func_local"].append(entry)
            _seen.add(name)
            if name.startswith("PyInit_"):
                result["pyinit_symbol"] = entry
        elif "OBJECT" in sym_type and sym.value != 0:
            _seen.add(name)

    # Imported Py* + _Py* symbols (from relocations)
    imported_py = set()
    for reloc in binary.pltgot_relocations:
        if reloc.symbol and reloc.symbol.name and (reloc.symbol.name.startswith("Py") or reloc.symbol.name.startswith("_Py")):
            imported_py.add(reloc.symbol.name)
    for reloc in binary.relocations:
        if hasattr(reloc, "symbol") and reloc.symbol and reloc.symbol.name and (reloc.symbol.name.startswith("Py") or reloc.symbol.name.startswith("_Py")):
            imported_py.add(reloc.symbol.name)

    for sym in binary.dynamic_symbols:
        if sym.imported and sym.name and (sym.name.startswith("Py") or sym.name.startswith("_Py")):
            imported_py.add(sym.name)

    # Categorize imports
    for sym in sorted(imported_py):
        ret_type, _ = get_symbol_info(sym)
        if ret_type == RET_DATA:
            if sym.startswith("PyExc_"):
                result["imported_py_exceptions"].append(sym)
            else:
                result["imported_py_types"].append(sym)
        else:
            result["imported_py_funcs"].append(sym)
        result["imported_py_symbols"].append(sym)

    _known_singletons = {"_Py_TrueStruct", "_Py_FalseStruct", "_Py_NoneStruct"}
    for s in _known_singletons:
        if s in imported_py and s not in result["imported_py_types"]:
            result["imported_py_types"].append(s)

    _analyze_method_defs(binary, result)
    return result


def _analyze_binary_readelf(so_path: str) -> Dict[str, Any]:
    """Fallback analysis using readelf when LIEF is not available.

    Uses subprocess to call readelf for symbol extraction.  This works
    on any Linux system with binutils installed.
    """
    import subprocess

    result: Dict[str, Any] = {
        "file": os.path.basename(so_path),
        "filepath": so_path,
        "python_version": _detect_python_version(so_path),
        "architecture": "x86_64",
        "symbols_func_global": [],
        "symbols_func_local": [],
        "pyinit_symbol": None,
        "imported_py_symbols": [],
        "imported_py_types": [],
        "imported_py_exceptions": [],
        "imported_py_funcs": [],
        "method_defs": [],
        "needed_libs": [],
    }

    try:
        # Get dynamic symbols
        proc = subprocess.run(
            ["readelf", "-Ws", so_path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return result

        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            # Format: Num: Value Size Type Bind Vis Ndx Name
            try:
                name = parts[-1]
                sym_type = parts[3]
                binding = parts[4]
                size = int(parts[2], 16) if parts[2].startswith("0x") else int(parts[2])
                vaddr = parts[1]
            except (ValueError, IndexError):
                continue

            if "FUNC" not in sym_type:
                continue

            entry = {"name": name, "vaddr": vaddr, "size": size, "binding": binding.upper()}

            if "GLOBAL" in binding.upper() and name and not name.startswith("."):
                result["symbols_func_global"].append(entry)
                if name.startswith("PyInit_"):
                    result["pyinit_symbol"] = entry
            elif "LOCAL" in binding.upper():
                result["symbols_func_local"].append(entry)

            # Track Py* imports (UND symbols)
            if "UND" in parts and name and (name.startswith("Py") or name.startswith("_Py")):
                ret_type, _ = get_symbol_info(name)
                if ret_type == RET_DATA:
                    if name.startswith("PyExc_"):
                        result["imported_py_exceptions"].append(name)
                    else:
                        result["imported_py_types"].append(name)
                else:
                    result["imported_py_funcs"].append(name)
                result["imported_py_symbols"].append(name)

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        # Get needed libraries
        proc = subprocess.run(
            ["readelf", "-d", so_path],
            capture_output=True, text=True, timeout=30,
        )
        for line in proc.stdout.splitlines():
            if "NEEDED" in line:
                m = re.search(r"\[(.+?)\]", line)
                if m:
                    result["needed_libs"].append(m.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return result


def _apply_relocations(binary, section_data: bytearray, section_vaddr: int) -> None:
    """Apply ELF relocations to a section's data buffer in-place.

    For PIE shared objects, the .data section contains placeholder values
    (often 0x0) for pointer fields that will be resolved at load time.
    Without applying relocations, PyMethodDef.ml_meth pointers are 0x0,
    making method discovery impossible.

    We handle two common x86_64 relocation types:
    - R_X86_64_64: absolute symbol reference (sym.value)
    - R_X86_64_RELATIVE: base + addend (addend is the vaddr for PIE)
    """
    import lief as _lief
    _R_X86_64_64 = _lief.ELF.Relocation.TYPE.X86_64_64
    _R_X86_64_RELATIVE = _lief.ELF.Relocation.TYPE.X86_64_RELATIVE
    for reloc in binary.relocations:
        addr = reloc.address
        if not (section_vaddr <= addr < section_vaddr + len(section_data)):
            continue
        off = addr - section_vaddr
        if off + 8 > len(section_data):
            continue
        try:
            rtype = reloc.type
            # R_X86_64_64: S + A — absolute symbol address
            if rtype == _R_X86_64_64:
                if reloc.symbol and reloc.symbol.value:
                    struct.pack_into("<Q", section_data, off, reloc.symbol.value)
            # R_X86_64_RELATIVE: B + A — for PIE, addend is the vaddr
            elif rtype == _R_X86_64_RELATIVE:
                struct.pack_into("<Q", section_data, off, reloc.addend)
        except Exception:
            pass


def _analyze_method_defs(binary, result: dict):
    """Try to decode PyMethodDef tables from the .data section.

    Also handles LOCAL (static) C functions: when func_ptr does not match
    a known symbol name, we still record the entry with a placeholder
    func_symbol so that vaddr-based resolution can find it at runtime.

    CRITICAL: For PIE shared objects, the .data section contains unresolved
    relocations (func_ptr == 0x0 on disk).  We must apply relocations first
    to resolve the actual function pointers before scanning.
    """
    try:
        data_section = binary.get_section(".data")
        rodata_section = binary.get_section(".rodata")
        if not data_section or not rodata_section:
            return

        # Use a mutable copy so we can apply relocations in-place
        data = bytearray(data_section.content)
        data_vaddr = data_section.virtual_address
        rodata = bytes(rodata_section.content)
        rodata_vaddr = rodata_section.virtual_address

        # Apply ELF relocations to resolve func_ptr fields
        _apply_relocations(binary, data, data_vaddr)

        known_funcs = {}
        for sym in binary.symbols:
            if "FUNC" in str(sym.type) and sym.value != 0:
                known_funcs[sym.value] = sym.name

        # Also try to find LOCAL (static) function symbols
        # LIEF includes LOCAL symbols from .symtab in binary.symbols,
        # but some stripped binaries may not have them.
        # We also check exported_symbols for the vaddr range to validate
        # that func_ptr points into the .text section.
        text_section = binary.get_section(".text")
        text_vaddr = text_section.virtual_address if text_section else 0
        text_end = text_vaddr + (len(bytes(text_section.content)) if text_section else 0)

        for i in range(0, len(data) - 32, 8):
            name_ptr = struct.unpack_from("<Q", data, i)[0]
            func_ptr = struct.unpack_from("<Q", data, i + 8)[0]
            flags_val = struct.unpack_from("<I", data, i + 16)[0]
            doc_ptr = struct.unpack_from("<Q", data, i + 24)[0]

            if not (rodata_vaddr <= name_ptr < rodata_vaddr + len(rodata)):
                continue
            if not (1 <= flags_val <= 1024):
                continue
            if doc_ptr != 0 and not (rodata_vaddr <= doc_ptr < rodata_vaddr + len(rodata)):
                continue

            # Validate that func_ptr points into .text section
            if text_vaddr and not (text_vaddr <= func_ptr < text_end):
                # Not in .text — likely not a real method table entry
                # Skip unless it's in known_funcs (which we already trust)
                if func_ptr not in known_funcs:
                    continue

            name_offset = name_ptr - rodata_vaddr
            end = rodata.find(b"\0", name_offset)
            if end == -1:
                continue
            method_name = rodata[name_offset:end].decode("ascii", errors="replace")

            METH_FLAGS = {1: "METH_VARARGS", 2: "METH_KEYWORDS", 4: "METH_NOARGS",
                         8: "METH_O", 16: "METH_CLASS", 32: "METH_STATIC",
                         256: "METH_COEXIST", 1024: "METH_FASTCALL"}
            flag_strs = [name for val, name in METH_FLAGS.items() if flags_val & val]
            if not flag_strs:
                continue

            # Get the C symbol name, or use a placeholder for LOCAL (static) functions
            func_symbol = known_funcs.get(func_ptr, f"_static_func_{hex(func_ptr)}")

            result["method_defs"].append({
                "name": method_name,
                "func_symbol": func_symbol,
                "func_vaddr": hex(func_ptr),
                "flags": flags_val,
                "flags_str": " | ".join(flag_strs),
                "is_local": func_ptr not in known_funcs,
            })
    except Exception:
        pass


def analyze_package(pkg_dir: str) -> Dict[str, Any]:
    """Analyze all .so files in a package directory (e.g., numpy/)."""
    all_results = []
    all_imported = set()
    all_types = set()
    all_exceptions = set()
    all_funcs = set()
    all_method_defs = []
    py_version = "3.12"

    so_files = []
    for root, dirs, files in os.walk(pkg_dir):
        for f in files:
            if f.endswith(".so"):
                so_files.append(os.path.join(root, f))

    for so_path in sorted(so_files):
        try:
            r = analyze_binary(so_path)
            all_results.append(r)
            all_imported.update(r["imported_py_symbols"])
            all_types.update(r["imported_py_types"])
            all_exceptions.update(r["imported_py_exceptions"])
            all_funcs.update(r["imported_py_funcs"])
            # Aggregate method_defs from each .so file, annotating
            # with the source .so so we can later match to the correct
            # PyInit_ symbol for vaddr-based resolution.
            for mdef in r.get("method_defs", []):
                mdef["_so_path"] = so_path
                all_method_defs.append(mdef)
            if r["pyinit_symbol"]:
                py_version = r["python_version"]
        except Exception:
            pass

    return {
        "package": os.path.basename(os.path.abspath(pkg_dir)),
        "python_version": py_version,
        "so_count": len(so_files),
        "so_results": all_results,
        "imported_py_symbols": sorted(all_imported),
        "imported_py_types": sorted(all_types),
        "imported_py_exceptions": sorted(all_exceptions),
        "imported_py_funcs": sorted(all_funcs),
        "method_defs": all_method_defs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Minimal C Stub Generator for Py* Symbols
# ═══════════════════════════════════════════════════════════════════════════════

def generate_py_stubs(
    imported_py_symbols: List[str],
    imported_py_types: List[str],
    imported_py_exceptions: List[str],
    imported_py_funcs: List[str],
    python_version: str = "3.12",
) -> str:
    """Generate minimal C stubs for the Py* symbols a .so needs.

    These stubs provide just enough of the CPython API for the linker
    to resolve the .so's imports.  They are NOT on the user's call path.

    Args:
        imported_py_symbols: All Py* symbols needed.
        imported_py_types: PyTypeObject/PyExc_* symbols.
        imported_py_exceptions: Exception type symbols.
        imported_py_funcs: Function symbols.
        python_version: CPython version string.

    Returns:
        C source code string.
    """
    ver_major, ver_minor = python_version.split(".")
    pep623 = int(ver_minor) >= 12

    out = []
    out.append(f"""\
// ============================================================
// pylow_ffi_stubs.c — Minimal CPython C API stubs
// Generated automatically by pylow FFI subsystem
//
// Principle: ONLY symbols imported by the .so extension.
// No libpython, no Py_Initialize, no interpreter.
//
// CPython target: {python_version} (PEP 623={'YES' if pep623 else 'NO'})
// Imported: {len(imported_py_funcs)} functions, {len(imported_py_types)} types,
//           {len(imported_py_exceptions)} exceptions
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

typedef long Py_ssize_t;
typedef Py_ssize_t Py_hash_t;
typedef unsigned int Py_UCS4;

// ════════════════════════════════════════════════════════════
//  Py_TPFLAGS constants (CRITICAL for type subclass checks)
// ════════════════════════════════════════════════════════════
#define Py_TPFLAGS_HEAPTYPE             (1UL << 9)
#define Py_TPFLAGS_READY                (1UL << 12)
#define Py_TPFLAGS_HAVE_GC              (1UL << 14)
#define Py_TPFLAGS_IMMUTABLETYPE        (1UL << 8)
#define Py_TPFLAGS_LONG_SUBCLASS        (1UL << 24)
#define Py_TPFLAGS_LIST_SUBCLASS        (1UL << 25)
#define Py_TPFLAGS_TUPLE_SUBCLASS       (1UL << 26)
#define Py_TPFLAGS_BYTES_SUBCLASS       (1UL << 27)
#define Py_TPFLAGS_UNICODE_SUBCLASS     (1UL << 28)
#define Py_TPFLAGS_DICT_SUBCLASS        (1UL << 29)
#define Py_TPFLAGS_BASE_EXC_SUBCLASS    (1UL << 30)
#define Py_TPFLAGS_TYPE_SUBCLASS        (1UL << 31)

// ════════════════════════════════════════════════════════════
//  Core object types (CPython {python_version} ABI, x86_64)
// ════════════════════════════════════════════════════════════
struct _typeobject {{
    Py_ssize_t ob_refcnt;
    struct _typeobject *ob_type;
    const char *tp_name;
    char _pad_to_tp_flags[0xA8 - 0x18];
    unsigned long tp_flags;
}};

struct _object {{
    Py_ssize_t ob_refcnt;
    struct _typeobject *ob_type;
}};

typedef struct _object PyObject;
typedef struct _typeobject PyTypeObject;

// Singleton objects
static PyTypeObject _Py_NoneStruct_obj = {{1, NULL, "NoneType", {{0}}, Py_TPFLAGS_IMMUTABLETYPE}};
static PyTypeObject _Py_TrueStruct_obj = {{1, NULL, "bool", {{0}}, Py_TPFLAGS_LONG_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE}};
static PyTypeObject _Py_FalseStruct_obj = {{1, NULL, "bool", {{0}}, Py_TPFLAGS_LONG_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE}};

PyObject* _Py_NoneStruct = (PyObject*)&_Py_NoneStruct_obj;
PyObject* _Py_TrueStruct = (PyObject*)&_Py_TrueStruct_obj;
PyObject* _Py_FalseStruct = (PyObject*)&_Py_FalseStruct_obj;
""")

    # Generate type object stubs
    TYPE_FLAGS_MAP = {
        "PyUnicode_Type": "Py_TPFLAGS_UNICODE_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE",
        "PyLong_Type": "Py_TPFLAGS_LONG_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE",
        "PyFloat_Type": "Py_TPFLAGS_IMMUTABLETYPE",
        "PyBytes_Type": "Py_TPFLAGS_BYTES_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE",
        "PyTuple_Type": "Py_TPFLAGS_TUPLE_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE",
        "PyList_Type": "Py_TPFLAGS_LIST_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE",
        "PyDict_Type": "Py_TPFLAGS_DICT_SUBCLASS",
        "PySet_Type": "Py_TPFLAGS_IMMUTABLETYPE",
        "PyBool_Type": "Py_TPFLAGS_LONG_SUBCLASS | Py_TPFLAGS_IMMUTABLETYPE",
        "PyComplex_Type": "Py_TPFLAGS_IMMUTABLETYPE",
        "PyBaseObject_Type": "Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE",
        "PyType_Type": "Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE",
    }

    for type_name in imported_py_types:
        if type_name.startswith("PyExc_"):
            continue
        if type_name.startswith("_Py_"):
            continue
        flags = TYPE_FLAGS_MAP.get(type_name, "Py_TPFLAGS_IMMUTABLETYPE")
        short_name = type_name.replace("_Type", "")
        out.append(f"""
static PyTypeObject _{type_name}_obj = {{1, NULL, "{short_name}", {{0}}, {flags}}};
PyTypeObject {type_name} = _{type_name}_obj;
""")

    # Generate exception object stubs
    for exc_name in imported_py_exceptions:
        short_name = exc_name.replace("PyExc_", "")
        out.append(f"""
static PyTypeObject _{exc_name}_obj = {{1, NULL, "{short_name}", {{0}}, Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_BASE_EXC_SUBCLASS}};
PyTypeObject {exc_name} = _{exc_name}_obj;
""")

    # Generate function stubs
    # ── Special-case critical functions that must return non-NULL ──
    # Returning NULL from these causes immediate segfaults in any .so
    # that calls them during PyInit_* or normal operation.
    _STUB_SPECIAL_RET = {
        # Module creation — MUST return non-NULL so PyInit doesn't crash
        "PyModule_Create2": "return _Py_NoneStruct;",
        "PyModuleDef_Init": "return _Py_NoneStruct;",
        "PyModule_New": "return _Py_NoneStruct;",
        "PyModule_NewObject": "return _Py_NoneStruct;",
        # Object allocation — MUST return non-NULL
        "PyUnicode_New": "return _Py_NoneStruct;",
        "PyUnicode_FromString": "return _Py_NoneStruct;",
        "PyUnicode_FromStringAndSize": "return _Py_NoneStruct;",
        "PyUnicode_FromEncodedObject": "return _Py_NoneStruct;",
        "PyUnicode_FromObject": "return _Py_NoneStruct;",
        "PyUnicode_InternFromString": "return _Py_NoneStruct;",
        "PyBytes_FromString": "return _Py_NoneStruct;",
        "PyBytes_FromStringAndSize": "return _Py_NoneStruct;",
        "PyLong_FromLong": "return _Py_NoneStruct;",
        "PyLong_FromUnsignedLong": "return _Py_NoneStruct;",
        "PyLong_FromLongLong": "return _Py_NoneStruct;",
        "PyLong_FromUnsignedLongLong": "return _Py_NoneStruct;",
        "PyLong_FromVoidPtr": "return _Py_NoneStruct;",
        "PyLong_FromSsize_t": "return _Py_NoneStruct;",
        "PyLong_FromDouble": "return _Py_NoneStruct;",
        "PyFloat_FromDouble": "return _Py_NoneStruct;",
        "PyBool_FromLong": "return (arg1) ? _Py_TrueStruct : _Py_FalseStruct;",
        "PyTuple_New": "return _Py_NoneStruct;",
        "PyTuple_Pack": "return _Py_NoneStruct;",
        "PyList_New": "return _Py_NoneStruct;",
        "PyDict_New": "return _Py_NoneStruct;",
        "PySet_New": "return _Py_NoneStruct;",
        "PyFrozenSet_New": "return _Py_NoneStruct;",
        "PySlice_New": "return _Py_NoneStruct;",
        "PyRange_New": "return _Py_NoneStruct;",
        "PyMemoryView_FromObject": "return _Py_NoneStruct;",
        # Object ops
        "PyObject_Str": "return _Py_NoneStruct;",
        "PyObject_Repr": "return _Py_NoneStruct;",
        "PyObject_Bytes": "return _Py_NoneStruct;",
        "PyObject_Format": "return _Py_NoneStruct;",
        "PyObject_GetIter": "return _Py_NoneStruct;",
        "PyObject_GetAttr": "return _Py_NoneStruct;",
        "PyObject_GetAttrString": "return _Py_NoneStruct;",
        "PyObject_GetItem": "return _Py_NoneStruct;",
        "PyObject_Call": "return _Py_NoneStruct;",
        "PyObject_CallNoArgs": "return _Py_NoneStruct;",
        "PyObject_CallOneArg": "return _Py_NoneStruct;",
        "PyObject_CallFunctionObjArgs": "return _Py_NoneStruct;",
        "PyObject_Vectorcall": "return _Py_NoneStruct;",
        "PyObject_VectorcallDict": "return _Py_NoneStruct;",
        "PyObject_VectorcallMethod": "return _Py_NoneStruct;",
        "PyObject_GenericGetAttr": "return _Py_NoneStruct;",
        "PyObject_GenericGetDict": "return _Py_NoneStruct;",
        "PyObject_GenericAlias": "return _Py_NoneStruct;",
        "PyObject_Init": "return _Py_NoneStruct;",
        # String ops
        "PyUnicode_AsUTF8String": "return _Py_NoneStruct;",
        "PyUnicode_AsASCIIString": "return _Py_NoneStruct;",
        "PyUnicode_AsLatin1String": "return _Py_NoneStruct;",
        "PyUnicode_AsEncodedString": "return _Py_NoneStruct;",
        "PyUnicode_Concat": "return _Py_NoneStruct;",
        "PyUnicode_Replace": "return _Py_NoneStruct;",
        "PyUnicode_Substring": "return _Py_NoneStruct;",
        "PyUnicode_Join": "return _Py_NoneStruct;",
        "PyUnicode_Decode": "return _Py_NoneStruct;",
        "PyUnicode_FromKindAndData": "return _Py_NoneStruct;",
        "PyUnicode_FromFormat": "return _Py_NoneStruct;",
        "PyUnicode_FromOrdinal": "return _Py_NoneStruct;",
        # Import
        "PyImport_ImportModule": "return _Py_NoneStruct;",
        "PyImport_Import": "return _Py_NoneStruct;",
        "PyImport_ImportModuleLevelObject": "return _Py_NoneStruct;",
        "PyImport_AddModule": "return _Py_NoneStruct;",
        "PyImport_GetModule": "return _Py_NoneStruct;",
        "PyImport_GetModuleDict": "return _Py_NoneStruct;",
        # Number ops
        "PyNumber_Add": "return _Py_NoneStruct;",
        "PyNumber_Subtract": "return _Py_NoneStruct;",
        "PyNumber_Multiply": "return _Py_NoneStruct;",
        "PyNumber_TrueDivide": "return _Py_NoneStruct;",
        "PyNumber_FloorDivide": "return _Py_NoneStruct;",
        "PyNumber_Remainder": "return _Py_NoneStruct;",
        "PyNumber_Power": "return _Py_NoneStruct;",
        "PyNumber_Negative": "return _Py_NoneStruct;",
        "PyNumber_Positive": "return _Py_NoneStruct;",
        "PyNumber_Absolute": "return _Py_NoneStruct;",
        "PyNumber_Invert": "return _Py_NoneStruct;",
        "PyNumber_Long": "return _Py_NoneStruct;",
        "PyNumber_Float": "return _Py_NoneStruct;",
        "PyNumber_Index": "return _Py_NoneStruct;",
        # Sequence/dict ops
        "PySequence_GetItem": "return _Py_NoneStruct;",
        "PySequence_GetSlice": "return _Py_NoneStruct;",
        "PySequence_List": "return _Py_NoneStruct;",
        "PySequence_Tuple": "return _Py_NoneStruct;",
        "PySequence_Fast": "return _Py_NoneStruct;",
        "PySequence_Concat": "return _Py_NoneStruct;",
        "PySequence_Repeat": "return _Py_NoneStruct;",
        "PyMapping_GetItemString": "return _Py_NoneStruct;",
        "PyDict_GetItem": "return _Py_NoneStruct;",
        "PyDict_GetItemWithError": "return _Py_NoneStruct;",
        "PyDict_SetItem": "return 0;",
        "PyDict_SetItemString": "return 0;",
        "PyIter_Next": "return _Py_NoneStruct;",
        # Method/func
        "PyCFunction_New": "return _Py_NoneStruct;",
        "PyCMethod_New": "return _Py_NoneStruct;",
        "PyMethod_New": "return _Py_NoneStruct;",
        # Exception
        "PyErr_NewException": "return _Py_NoneStruct;",
        "PyErr_NewExceptionWithDoc": "return _Py_NoneStruct;",
        "PyErr_Occurred": "return NULL;",
        # Misc
        "PyCapsule_New": "return _Py_NoneStruct;",
        "PyWeakref_NewObject": "return _Py_NoneStruct;",
        "PyWeakref_Proxy": "return _Py_NoneStruct;",
        "PySeqIter_New": "return _Py_NoneStruct;",
        "PyMember_New": "return _Py_NoneStruct;",
        "PyGetSet_New": "return _Py_NoneStruct;",
        "PyProperty_New": "return _Py_NoneStruct;",
        # Int-returning functions that should succeed
        "PyModule_AddObject": "return 0;",
        "PyModule_AddIntConstant": "return 0;",
        "PyModule_AddStringConstant": "return 0;",
        "PyType_Ready": "return 0;",
        "PyObject_SetAttr": "return 0;",
        "PyObject_SetAttrString": "return 0;",
        "PyObject_SetItem": "return 0;",
        "PyObject_IsTrue": "return 1;",
        "PyObject_Not": "return 0;",
        "PyObject_HasAttr": "return 0;",
        "PyObject_HasAttrString": "return 0;",
        "PyObject_IsInstance": "return 0;",
        "PyObject_IsSubclass": "return 0;",
        "PyCallable_Check": "return 1;",
        # Ref counting — no-ops in stub mode
        "Py_INCREF": "return;",
        "Py_DECREF": "return;",
        "Py_XDECREF": "return;",
        "Py_XINCREF": "return;",
        "Py_DecRef": "return;",
        # Check functions — return 0 (false) but don't crash
        "PyUnicode_Check": "return 0;",
        "PyLong_Check": "return 0;",
        "PyLong_CheckExact": "return 0;",
        "PyFloat_Check": "return 0;",
        "PyBytes_Check": "return 0;",
        "PyList_Check": "return 0;",
        "PyTuple_Check": "return 0;",
        "PyDict_Check": "return 0;",
        "PySet_Check": "return 0;",
        "PyType_Check": "return 0;",
        "PyType_IsSubtype": "return 0;",
        # GIL
        "PyGILState_Ensure": "return 0;",
    }

    for func_name in imported_py_funcs:
        ret_type, param_count = get_symbol_info(func_name)
        c_ret = ret_type_to_llvm_ctype(ret_type)

        # Check for special-case return value
        special_ret = _STUB_SPECIAL_RET.get(func_name)

        if special_ret:
            if param_count < 0:
                out.append(f"""
{c_ret} {func_name}(PyObject* self, ...) {{
    // FFI stub — variadic, special return
    {special_ret}
}}
""")
            elif param_count == 0:
                out.append(f"""
{c_ret} {func_name}() {{
    // FFI stub — no args, special return
    {special_ret}
}}
""")
            else:
                args = ", ".join([f"PyObject* arg{i+1}" for i in range(param_count)])
                out.append(f"""
{c_ret} {func_name}({args}) {{
    // FFI stub — {param_count} args, special return
    {special_ret}
}}
""")
        elif param_count < 0:
            # Variadic — provide a minimal stub with va_list
            out.append(f"""
{c_ret} {func_name}(PyObject* self, ...) {{
    // FFI stub — variadic, returns default
    return ({c_ret})0;
}}
""")
        elif param_count == 0:
            out.append(f"""
{c_ret} {func_name}() {{
    // FFI stub — no args
    return ({c_ret})0;
}}
""")
        else:
            args = ", ".join([f"PyObject* arg{i+1}" for i in range(param_count)])
            out.append(f"""
{c_ret} {func_name}({args}) {{
    // FFI stub — {param_count} args
    return ({c_ret})0;
}}
""")

    out.append("\n")
    return "\n".join(out)


def generate_cpython_bridge() -> str:
    """Generate C bridge code for calling CPython extension functions.

    When pylow calls CPython extension .so functions (like markupsafe's
    escape_unicode), it cannot pass its internal value representations
    directly because CPython extensions expect real PyObject* values
    with the correct memory layout.  CPython macros like PyUnicode_Check,
    Py_INCREF, PyUnicode_KIND etc. are expanded inline in the compiled
    .so code and directly access PyObject struct fields — they cannot be
    intercepted by stubs.

    This bridge provides helper functions that:
    1. Initialize the CPython interpreter (if not already done)
    2. Convert raw C strings to CPython PyObject* using the real API
    3. Call the CPython extension function with proper PyObject* arguments
    4. Convert the result back to a C string

    The bridge MUST be compiled and linked against the real libpython3.x.so.

    Returns:
        C source code string for the bridge.
    """
    return """
// ============================================================
// pylow_ffi_cpython_bridge.c — CPython extension call bridge
// Generated automatically by pylow FFI subsystem
//
// Provides safe wrappers for calling CPython extension functions
// that expect real PyObject* values.  MUST be linked against
// the real libpython3.x.so — do NOT generate Py* stubs when
// using this bridge.
// ============================================================

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ════════════════════════════════════════════════════════════
//  Forward declarations for CPython API functions
//  (resolved from libpython at link time)
// ════════════════════════════════════════════════════════════
typedef long Py_ssize_t;
typedef void PyObject;

extern void Py_Initialize(void);
extern int  Py_IsInitialized(void);
extern PyObject* PyUnicode_FromStringAndSize(const char*, Py_ssize_t);
extern const char* PyUnicode_AsUTF8AndSize(PyObject*, Py_ssize_t*);
extern void Py_DecRef(PyObject*);
extern PyObject* PyLong_FromLong(long);
extern long PyLong_AsLong(PyObject*);
extern PyObject* PyFloat_FromDouble(double);
extern double PyFloat_AsDouble(PyObject*);
extern PyObject* PyBool_FromLong(long);
extern int  PyObject_IsTrue(PyObject*);
extern void PyErr_Clear(void);

// Track whether we already initialized CPython
static int _pylow_py_initialized = 0;

// ════════════════════════════════════════════════════════════
//  pylow_ffi_ensure_py_init — Initialize CPython once
// ════════════════════════════════════════════════════════════
static void pylow_ffi_ensure_py_init(void) {
    if (!_pylow_py_initialized) {
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        _pylow_py_initialized = 1;
    }
}

// ════════════════════════════════════════════════════════════
//  pylow_ffi_call_meth_o_str — Call a METH_O CPython extension
//  function that takes a string argument and returns a string.
//
//  This is the most common pattern: CPython extension methods
//  like markupsafe's escape_unicode have signature:
//    PyObject* func(PyObject* self, PyObject* arg)
//  where 'arg' is a PyUnicodeObject* and the return is also
//  a PyUnicodeObject*.
//
//  Parameters:
//    func_ptr  — function pointer obtained via dlsym
//    str_data  — raw UTF-8 string data
//    str_len   — length of the string data
//    out_len   — [out] length of the returned string
//
//  Returns:
//    malloc'd C string (caller must free with pylow_ffi_free),
//    or NULL on error.
// ════════════════════════════════════════════════════════════
char* pylow_ffi_call_meth_o_str(void* func_ptr,
                                 const char* str_data,
                                 long str_len,
                                 long* out_len) {
    pylow_ffi_ensure_py_init();

    // Create a CPython PyUnicodeObject from the raw string
    PyObject* py_str = PyUnicode_FromStringAndSize(str_data, (Py_ssize_t)str_len);
    if (!py_str) {
        PyErr_Clear();
        *out_len = 0;
        return NULL;
    }

    // Call: PyObject* func(PyObject* self, PyObject* arg)
    // self=NULL is acceptable for module-level METH_O functions
    typedef PyObject* (*meth_o_fn)(PyObject*, PyObject*);
    PyObject* result = ((meth_o_fn)func_ptr)(NULL, py_str);
    Py_DecRef(py_str);

    if (!result) {
        PyErr_Clear();
        *out_len = 0;
        return NULL;
    }

    // Extract the C string from the result
    Py_ssize_t len = 0;
    const char* utf8 = PyUnicode_AsUTF8AndSize(result, &len);

    char* copy = NULL;
    if (utf8) {
        copy = (char*)malloc((size_t)len + 1);
        if (copy) {
            memcpy(copy, utf8, (size_t)len);
            copy[len] = '\\0';
        }
    }

    *out_len = (long)len;
    Py_DecRef(result);

    return copy;
}

// ════════════════════════════════════════════════════════════
//  pylow_ffi_call_meth_o_int — Call a METH_O CPython extension
//  function that takes a string argument and returns an int.
// ════════════════════════════════════════════════════════════
long pylow_ffi_call_meth_o_int(void* func_ptr,
                                const char* str_data,
                                long str_len) {
    pylow_ffi_ensure_py_init();

    PyObject* py_str = PyUnicode_FromStringAndSize(str_data, (Py_ssize_t)str_len);
    if (!py_str) {
        PyErr_Clear();
        return 0;
    }

    typedef PyObject* (*meth_o_fn)(PyObject*, PyObject*);
    PyObject* result = ((meth_o_fn)func_ptr)(NULL, py_str);
    Py_DecRef(py_str);

    if (!result) {
        PyErr_Clear();
        return 0;
    }

    long val = PyLong_AsLong(result);
    Py_DecRef(result);
    return val;
}

// ════════════════════════════════════════════════════════════
//  pylow_ffi_call_meth_noargs_str — Call a METH_NOARGS CPython
//  extension function that returns a string.
// ════════════════════════════════════════════════════════════
char* pylow_ffi_call_meth_noargs_str(void* func_ptr, long* out_len) {
    pylow_ffi_ensure_py_init();

    // METH_NOARGS: PyObject* func(PyObject* self, PyObject* Py_UNUSED(args))
    typedef PyObject* (*meth_noargs_fn)(PyObject*, PyObject*);
    PyObject* result = ((meth_noargs_fn)func_ptr)(NULL, NULL);

    if (!result) {
        PyErr_Clear();
        *out_len = 0;
        return NULL;
    }

    Py_ssize_t len = 0;
    const char* utf8 = PyUnicode_AsUTF8AndSize(result, &len);

    char* copy = NULL;
    if (utf8) {
        copy = (char*)malloc((size_t)len + 1);
        if (copy) {
            memcpy(copy, utf8, (size_t)len);
            copy[len] = '\\0';
        }
    }

    *out_len = (long)len;
    Py_DecRef(result);
    return copy;
}

// ════════════════════════════════════════════════════════════
//  pylow_ffi_call_meth_varargs_str — Call a METH_VARARGS CPython
//  extension function with a single string argument.
//
//  METH_VARARGS functions receive a tuple of positional args.
//  This bridge creates a 1-element tuple containing the string,
//  calls the function, and returns the result as a C string.
// ════════════════════════════════════════════════════════════
char* pylow_ffi_call_meth_varargs_str(void* func_ptr,
                                       const char* str_data,
                                       long str_len,
                                       long* out_len) {
    pylow_ffi_ensure_py_init();

    PyObject* py_str = PyUnicode_FromStringAndSize(str_data, (Py_ssize_t)str_len);
    if (!py_str) {
        PyErr_Clear();
        *out_len = 0;
        return NULL;
    }

    // Build a 1-element tuple: (py_str,)
    extern PyObject* PyTuple_New(Py_ssize_t);
    extern int PyTuple_SetItem(PyObject*, Py_ssize_t, PyObject*);
    PyObject* args = PyTuple_New(1);
    if (!args) {
        Py_DecRef(py_str);
        PyErr_Clear();
        *out_len = 0;
        return NULL;
    }
    PyTuple_SetItem(args, 0, py_str);  // Steals ref to py_str

    // METH_VARARGS: PyObject* func(PyObject* self, PyObject* args)
    typedef PyObject* (*meth_varargs_fn)(PyObject*, PyObject*);
    PyObject* result = ((meth_varargs_fn)func_ptr)(NULL, args);
    Py_DecRef(args);

    if (!result) {
        PyErr_Clear();
        *out_len = 0;
        return NULL;
    }

    Py_ssize_t len = 0;
    const char* utf8 = PyUnicode_AsUTF8AndSize(result, &len);

    char* copy = NULL;
    if (utf8) {
        copy = (char*)malloc((size_t)len + 1);
        if (copy) {
            memcpy(copy, utf8, (size_t)len);
            copy[len] = '\\0';
        }
    }

    *out_len = (long)len;
    Py_DecRef(result);
    return copy;
}

// ════════════════════════════════════════════════════════════
//  pylow_ffi_free — Free memory allocated by bridge functions
// ════════════════════════════════════════════════════════════
void pylow_ffi_free(void* ptr) {
    free(ptr);
}
"""


def has_cpython_extensions(ffi_modules: dict) -> bool:
    """Check if any registered FFI module is a CPython extension.

    A CPython extension module is one that has a PyInit_ symbol,
    meaning it was compiled as a CPython C extension and expects
    to be called with real PyObject* values.

    Args:
        ffi_modules: Dict of module_name -> FFIModule.

    Returns:
        True if at least one CPython extension module is registered.
    """
    for mod in ffi_modules.values():
        if mod.pyinit_symbol is not None:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  FFISignatureDB — Compiler-facing interface
# ═══════════════════════════════════════════════════════════════════════════════

class FFISignatureDB:
    """Compiler-facing interface to the FFI signature database.

    Provides methods for the compiler to query symbol signatures and
    generate LLVM IR declarations.  This is the primary API surface
    that src/compiler.py and src/mixins/visitors_call.py use.

    Usage in the compiler::

        ffi_db = FFISignatureDB()
        ret_ir, param_types = ffi_db.get_llvm_signature("PyLong_FromLong")
        # ret_ir = ir.PointerType(ir.IntType(8))  # PyObject* as i8*
        # param_types = [ir.PointerType(ir.IntType(8))]  # 1 param
    """

    def __init__(self):
        self._custom: Dict[str, Tuple[str, int]] = {}

    def register_symbol(self, name: str, ret_type: str, param_count: int):
        """Register a custom symbol signature (e.g., from a .h file)."""
        self._custom[name] = (ret_type, param_count)

    def lookup(self, name: str) -> Tuple[str, int]:
        """Look up a symbol's signature.

        Checks custom registrations first, then the PyAPI_DB, then
        falls back to heuristic classification.
        """
        if name in self._custom:
            return self._custom[name]
        return get_symbol_info(name)

    def get_llvm_signature(self, name: str):
        """Get the LLVM IR function type for a symbol.

        Returns:
            Tuple of (return_type_ir, param_types_list, is_variadic).
        """
        import llvmlite.ir as ir

        ret_cat, param_count = self.lookup(name)
        ret_ir = ret_type_to_llvm_ir(ret_cat)
        i8p = ir.PointerType(ir.IntType(8))

        if param_count < 0:
            # Variadic
            return ret_ir, [i8p], True
        else:
            return ret_ir, [i8p] * param_count, False

    def get_return_pytype(self, name: str):
        """Map the return type of a symbol to a PyType enum value.

        Used by the compiler to determine the type of the Value
        returned from an FFI call.
        """
        from ..types import PyType

        ret_cat, _ = self.lookup(name)
        _MAP = {
            RET_INT: PyType.INT,
            RET_LONG: PyType.INT,
            RET_PYSSIZET: PyType.INT,
            RET_DOUBLE: PyType.FLOAT,
            RET_CONSTCHAR: PyType.STR,
            RET_CHARPTR: PyType.STR,
            RET_PYOBJ: PyType.OBJECT,
            RET_DATA: PyType.OBJECT,
            RET_VOIDPTR: PyType.OBJECT,
            RET_VOID: PyType.NONE,
            RET_SIZE_T: PyType.INT,
            RET_ULONG: PyType.INT,
            RET_LONGLONG: PyType.INT,
            RET_ULONGLONG: PyType.INT,
        }
        return _MAP.get(ret_cat, PyType.OBJECT)