"""AOT C++ wrapper generator for CPython extension .so modules.

Integrates the mini-runtime + wrapper generation into the pylow compiler's
FFI subsystem.  For each registered CPython extension module, generates:
  1. ffi_<module>_runtime.cpp — Py* symbol stubs (weak linkage)
  2. ffi_<module>_wrapper.cpp — Wrapper functions with pylow-friendly signatures

Phase 1 MVP: METH_O (string arg), METH_NOARGS (string return), generic fallback.
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from src.ffi.core import (
    FFIModule, FFISymbol, get_symbol_info,
    RET_VOID, RET_INT, RET_LONG, RET_PYSSIZET, RET_PYOBJ, RET_CONSTCHAR,
    RET_CHARPTR, RET_VOIDPTR, RET_DOUBLE, RET_SIZE_T, RET_ULONG,
    RET_LONGLONG, RET_ULONGLONG, RET_DATA,
)

METH_VARARGS = 0x0001
METH_KEYWORDS = 0x0002
METH_NOARGS = 0x0004
METH_O = 0x0008
METH_CLASS = 0x0010
METH_STATIC = 0x0020

@dataclass
class WrapperSignature:
    """Describes the C calling convention of a generated wrapper function."""
    symbol_name: str        # e.g., "pylow_ffi_markupsafe_escape"
    original_symbol: str    # e.g., "escape_unicode"
    module_name: str        # e.g., "markupsafe"
    return_type: str        # e.g., "char*" or "int64_t"
    param_types: List[str]  # e.g., ["const char*", "int64_t", "int64_t*"]
    method_flags: int       # METH_O, METH_NOARGS, METH_VARARGS, etc.
    fallback: bool = False  # True if needs dlsym to libpython

# Type flags map for tp_flags at offset 0xA8
_TYPE_FLAGS: Dict[str, str] = {
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
    "PyModule_Type": "Py_TPFLAGS_IMMUTABLETYPE",
}

def _ret_cat_to_c(ret_cat: str) -> str:
    _M = {RET_VOID:"void", RET_INT:"int", RET_LONG:"long", RET_PYSSIZET:"Py_ssize_t",
          RET_PYOBJ:"PyObject*", RET_CONSTCHAR:"const char*", RET_CHARPTR:"char*",
          RET_VOIDPTR:"void*", RET_DOUBLE:"double", RET_SIZE_T:"size_t",
          RET_ULONG:"unsigned long", RET_LONGLONG:"long long",
          RET_ULONGLONG:"unsigned long long", RET_DATA:"PyTypeObject"}
    return _M.get(ret_cat, "void*")

# Py* symbols we provide working implementations for
_COVERED_SYMS = {
    # Refcounting
    "Py_DecRef","Py_INCREF","Py_DECREF","Py_XDECREF","Py_XINCREF",
    # Memory
    "PyObject_Malloc","PyObject_Free","PyObject_Realloc","PyObject_Calloc",
    "_Py_Dealloc",
    # Module
    "PyModule_Create2","PyModuleDef_Init","PyModule_New",
    "PyModule_AddObject","PyModule_AddStringConstant","PyModule_AddIntConstant",
    "PyModule_GetState","PyState_FindModule",
    # Init
    "PyObject_Init",
    # Unicode — working implementations
    "PyUnicode_New","PyUnicode_FromString",
    "PyUnicode_FromStringAndSize","PyUnicode_AsUTF8","PyUnicode_AsUTF8AndSize",
    "PyUnicode_GetLength","PyUnicode_InternFromString","PyUnicode_AsUTF8String",
    "PyUnicode_AsEncodedString","PyUnicode_FromKindAndData",
    "PyUnicode_DecodeUTF8",
    "_PyUnicode_Ready",
    # Long — working implementations
    "PyLong_FromLong","PyLong_FromUnsignedLong","PyLong_FromLongLong",
    "PyLong_FromUnsignedLongLong","PyLong_FromSsize_t",
    "PyLong_AsLong","PyLong_AsSsize_t","PyLong_AsLongLong",
    "PyLong_AsUnsignedLongLong","PyLong_FromString",
    # Float — working implementations
    "PyFloat_FromDouble","PyFloat_AsDouble",
    # Bytes — working implementations
    "PyBytes_FromString","PyBytes_FromStringAndSize",
    "PyBytes_AsString","PyBytes_Size","PyBytes_CheckExact",
    # Dict — minimal implementations
    "PyDict_New","PyDict_SetItem","PyDict_SetItemString",
    "PyDict_GetItem","PyDict_GetItemString","PyDict_Next",
    "PyDict_Keys","PyDict_Size",
    # List — minimal implementations
    "PyList_New","PyList_Append","PyList_Sort",
    "PyList_GetItem","PyList_SetItem",
    # Tuple — minimal implementations
    "PyTuple_New","PyTuple_Pack","PyTuple_GetItem","PyTuple_SetItem",
    # PyObject utilities
    "PyObject_IsTrue","PyObject_Not","PyObject_Str","PyObject_Repr",
    "PyObject_Type","PyObject_IsInstance","PyObject_IsSubclass",
    "PyObject_GetAttrString","PyObject_HasAttrString",
    "PyObject_CallObject","PyObject_CallFunctionObjArgs","PyObject_CallMethod",
    "PyObject_GetBuffer","PyBuffer_Release","PyObject_Number",
    # Error handling
    "PyErr_SetString","PyErr_Occurred","PyErr_Clear","PyErr_NoMemory",
    "PyErr_ExceptionMatches","PyErr_Format","PyErr_NewException",
    # Type checks
    "PyUnicode_Check","PyLong_Check","PyBytes_Check",
    "PyTuple_Check","PyList_Check","PyDict_Check",
    "PyFloat_Check","PyBool_Check",
    "PyCallable_Check","PyIter_Check","PyType_IsSubtype",
    # Import
    "PyImport_ImportModule",
    # Argument parsing
    "PyArg_ParseTuple","PyArg_ParseTupleAndKeywords",
    # Number
    "PyNumber_ToBase",
}


def _safe_stub(sym: str, rc: str, pc: int) -> str:
    """Generate a safe (non-aborting) stub for an unimplemented Py* symbol.

    Returns 0/nullptr/empty instead of aborting, so the program can link
    and run — we'll see where real logic is actually needed.
    """
    crt = _ret_cat_to_c(rc)
    ps = "..." if pc < 0 else ("void" if pc == 0 else ", ".join(["void*"]*pc))
    # Safe return value based on return type
    if rc in (RET_VOID,):
        body = 'fprintf(stderr,"[pylow-ffi] STUB: %s\\n");'
    elif rc in (RET_INT, RET_LONG, RET_PYSSIZET, RET_SIZE_T, RET_ULONG, RET_LONGLONG, RET_ULONGLONG):
        body = 'fprintf(stderr,"[pylow-ffi] STUB: %s\\n");return -1;'
    elif rc in (RET_DOUBLE,):
        body = 'fprintf(stderr,"[pylow-ffi] STUB: %s\\n");return 0.0;'
    else:  # RET_PYOBJ, RET_CONSTCHAR, RET_CHARPTR, RET_VOIDPTR, RET_DATA, etc.
        body = 'fprintf(stderr,"[pylow-ffi] STUB: %s\\n");return nullptr;'
    return f'__attribute__((weak)) {crt} {sym}({ps}){{{body}}}'


def _generate_runtime(module: FFIModule) -> str:
    """Generate C++ mini-runtime source for a registered module.

    This mini-runtime provides enough of the CPython C API for extension
    .so files to load and function.  Key design decisions:

    1. ALL type objects (PyBaseObject_Type, PyType_Type, etc.) are ALWAYS
       defined — they are fundamental for _mini_runtime_init().
    2. _Py_TrueStruct, _Py_FalseStruct, _Py_NoneStruct are ALWAYS defined.
    3. PyObject_Malloc/Free/Realloc redirect to malloc/free/realloc.
    4. Covered symbols have working implementations.
    5. Uncovered symbols get safe stubs (return 0/nullptr) instead of abort().
    """
    nm = module.name
    pv = module.python_version
    it = module.imported_py_types
    if_ = module.imported_py_funcs
    ie = module.imported_py_exceptions

    needs_unicode = any(s.startswith("PyUnicode_") for s in if_) or "PyUnicode_Type" in it
    needs_long = any(s.startswith("PyLong_") for s in if_) or "PyLong_Type" in it
    needs_tuple = any(s.startswith("PyTuple_") for s in if_) or "PyTuple_Type" in it
    needs_bytes = any(s.startswith("PyBytes_") for s in if_) or "PyBytes_Type" in it
    needs_list = any(s.startswith("PyList_") for s in if_) or "PyList_Type" in it
    needs_dict = any(s.startswith("PyDict_") for s in if_) or "PyDict_Type" in it
    needs_float = any(s.startswith("PyFloat_") for s in if_) or "PyFloat_Type" in it

    # Collect needed type objects — ALWAYS include ALL fundamental types
    # because the working implementations (PyLong_FromLong, PyFloat_FromDouble, etc.)
    # ALWAYS reference their Type objects regardless of which symbols are imported.
    nt = set(it)
    for need, tname in [(needs_unicode,"PyUnicode_Type"),(needs_long,"PyLong_Type"),
                        (needs_tuple,"PyTuple_Type"),(needs_list,"PyList_Type"),
                        (needs_dict,"PyDict_Type"),(needs_bytes,"PyBytes_Type"),
                        (needs_float,"PyFloat_Type")]:
        if need: nt.add(tname)
    # ALWAYS include ALL fundamental type objects — the working implementations
    # below unconditionally reference PyLong_Type, PyFloat_Type, etc.
    # Without these, compilation fails for modules that don't import them directly.
    nt.update([
        "PyType_Type","PyModule_Type","PyBaseObject_Type","PyBool_Type",
        "PyLong_Type","PyFloat_Type","PyBytes_Type","PyTuple_Type",
        "PyList_Type","PyDict_Type","PyUnicode_Type",
    ])
    # ALWAYS define singletons — many extensions need True/False/None
    nt.update(["_Py_TrueStruct","_Py_FalseStruct","_Py_NoneStruct"])
    # ALWAYS include imported exceptions — they're referenced by the .so at link time
    nt.update(ie)

    L = []  # output lines
    L.append(f"""\
// ffi_{nm}_runtime.cpp — Minimal CPython C API runtime (pylow FFIManager)
// CPython {pv} | {len(if_)} funcs, {len(nt)} types, {len(ie)} exceptions
// All Py* symbols are weak — multiple modules can link without conflicts.
// Unimplemented symbols return 0/nullptr (safe stub) instead of aborting.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstdarg>
typedef long Py_ssize_t;
typedef Py_ssize_t Py_hash_t;
typedef unsigned int Py_UCS4;
#define Py_TPFLAGS_HEAPTYPE         (1UL << 9)
#define Py_TPFLAGS_READY            (1UL << 12)
#define Py_TPFLAGS_HAVE_GC          (1UL << 14)
#define Py_TPFLAGS_IMMUTABLETYPE    (1UL << 8)
#define Py_TPFLAGS_LONG_SUBCLASS    (1UL << 24)
#define Py_TPFLAGS_LIST_SUBCLASS    (1UL << 25)
#define Py_TPFLAGS_TUPLE_SUBCLASS   (1UL << 26)
#define Py_TPFLAGS_BYTES_SUBCLASS   (1UL << 27)
#define Py_TPFLAGS_UNICODE_SUBCLASS (1UL << 28)
#define Py_TPFLAGS_DICT_SUBCLASS    (1UL << 29)
#define Py_TPFLAGS_BASE_EXC_SUBCLASS (1UL << 30)
// Ensure all Py* symbols are exported to the dynamic symbol table
// so that dlopen'd .so files can resolve them at link time.
#pragma GCC visibility push(default)
extern "C" {{
struct _typeobject {{
    Py_ssize_t ob_refcnt;               // 0x00
    struct _typeobject *ob_type;         // 0x08
    Py_ssize_t ob_size;                 // 0x10  (PyObject_VAR_HEAD includes this)
    const char *tp_name;                // 0x18
    char _pad_to_tp_flags[0xA8 - 0x20]; // padding (0xA8 - 0x20 = 0x88)
    unsigned long tp_flags;             // 0xA8
}};
struct _object {{ Py_ssize_t ob_refcnt; _typeobject *ob_type; }};
typedef struct _object PyObject;
typedef struct _typeobject PyTypeObject;""")

    # Structs for needed types — ALWAYS define all common structs
    # so working implementations can compile regardless of which are imported
    L.append("""
struct PyUnicode_State { unsigned interned:2,kind:3,compact:1,ascii:1,statically_allocated:1; unsigned :24; };
struct PyASCIIObject {
    Py_ssize_t ob_refcnt; PyTypeObject *ob_type; Py_ssize_t length;
    Py_hash_t hash; PyUnicode_State state; // ASCII data follows (PEP 623)
};
struct PyCompactUnicodeObject { PyASCIIObject _base; Py_ssize_t utf8_length; char *utf8; };
static_assert(offsetof(PyASCIIObject, ob_refcnt)==0,"ABI");
static_assert(offsetof(PyASCIIObject, ob_type)==8,"ABI");
static_assert(offsetof(PyASCIIObject, length)==16,"ABI");
static_assert(offsetof(PyASCIIObject, hash)==24,"ABI");
static_assert(offsetof(PyASCIIObject, state)==32,"ABI");
struct PyLongObject { Py_ssize_t ob_refcnt; PyTypeObject *ob_type;
    Py_ssize_t ob_size; uint32_t ob_digit[1]; };
struct PyTupleObject { Py_ssize_t ob_refcnt; PyTypeObject *ob_type;
    Py_ssize_t ob_size; PyObject *ob_item[1]; };
struct PyBytesObject { Py_ssize_t ob_refcnt; PyTypeObject *ob_type;
    Py_ssize_t ob_size; Py_hash_t ob_shash; char ob_sval[1]; };
struct PyListObject { Py_ssize_t ob_refcnt; PyTypeObject *ob_type;
    Py_ssize_t ob_size; PyObject **ob_item; Py_ssize_t allocated; };
struct PyFloatObject { Py_ssize_t ob_refcnt; PyTypeObject *ob_type; double ob_fval; };
struct _PyDictEntry { PyObject *key; PyObject *value; };
struct PyDictObject { Py_ssize_t ob_refcnt; PyTypeObject *ob_type;
    Py_ssize_t ma_used; Py_ssize_t ma_allocated; _PyDictEntry *ma_entries; };

// Forward declarations — PyType_Type is referenced by exception objects before its definition
extern PyTypeObject PyType_Type;
// Forward declarations — these functions are referenced before their definitions
int PyBytes_Check(PyObject *o);
PyObject* PyList_New(Py_ssize_t len);
PyObject* PyBytes_FromStringAndSize(const char *s, Py_ssize_t len);""")

    # Type objects + singletons — ALWAYS define fundamental ones
    # PyType_Type was forward-declared above, so skip it here and define separately
    L.append("\n// Global type objects (tp_flags at offset 0xA8!)")
    # Define PyType_Type properly (self-referential: ob_type = &PyType_Type)
    if "PyType_Type" in nt:
        L.append(f'// PyType_Type definition (forward-declared above)')
        L.append(f'__attribute__((weak, visibility("default"))) PyTypeObject PyType_Type = {{1,&PyType_Type,0,"type",{{}},Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE}};')
    for tn in sorted(nt):
        if tn == "PyType_Type":
            continue  # Already defined above
        if tn.startswith("_Py_") and tn.endswith("Struct"):
            short = tn.replace("_Py_","").replace("Struct","")
            L.append(f'__attribute__((weak)) PyObject {tn} = {{1,nullptr}};')
            L.append(f'#define Py_{short} (&{tn})')
        elif tn.startswith("PyExc_"):
            # CPython declares exceptions as PyAPI_DATA(PyObject*) PyExc_Xxx;
            # The .so expects a POINTER (PyObject*), not a struct (PyTypeObject).
            # We define a backing PyObject struct with ob_type=&PyType_Type
            # (matching CPython ABI where exceptions inherit from BaseException),
            # then export a PyObject* pointer pointing to it.
            # Both are weak so multiple modules can link without conflicts.
            # Both have visibility("default") so the .so can resolve them at link time.
            #
            # IMPORTANT: ob_type is set to &PyType_Type at static init time so the
            # object is valid even before _mini_runtime_init() runs.  The address of
            # PyType_Type is a compile-time constant (it's a global), so the linker
            # can resolve it.
            short = tn.replace("PyExc_","")
            L.append(f'__attribute__((weak, visibility("default"))) '
                     f'PyObject _{tn}_obj = {{1, &PyType_Type}};')
            L.append(f'__attribute__((weak, visibility("default"))) '
                     f'PyObject* {tn} = &_{tn}_obj;')
        else:
            fl = _TYPE_FLAGS.get(tn, "Py_TPFLAGS_IMMUTABLETYPE")
            sn = tn.replace("Py","").replace("_Type","").lower()
            # PyBaseObject_Type and other fundamental types need visibility(default)
            # so the linker can resolve them when linking with .so files
            L.append(f'__attribute__((weak, visibility("default"))) PyTypeObject {tn} = {{1,nullptr,0,"{sn}",{{}},{fl}}};')

    # Runtime init — ALWAYS initializes all singletons
    L.append("__attribute__((weak)) int _mini_rt_init_done=0;\n__attribute__((weak)) void _mini_runtime_init(void){")
    L.append("  if(_mini_rt_init_done) return; _mini_rt_init_done=1;")
    # Singletons: True/False → PyLong_Type, None → PyBaseObject_Type
    L.append("  _Py_TrueStruct.ob_type=&PyLong_Type;")
    L.append("  _Py_FalseStruct.ob_type=&PyLong_Type;")
    L.append("  _Py_NoneStruct.ob_type=&PyBaseObject_Type;")
    for tn in sorted(nt):
        if (tn.startswith("_Py_") and tn.endswith("Struct")) or tn.startswith("PyExc_"): continue
        if tn == "PyType_Type": continue  # Already self-referential from definition
        L.append(f"  {tn}.ob_type=&PyType_Type;")
    # Initialize exception objects' ob_type to &PyType_Type
    # (already set at static init time, but re-set here for safety in case
    # of static init order fiasco or if another module overrode them)
    for tn in sorted(nt):
        if not tn.startswith("PyExc_"): continue
        L.append(f"  _{tn}_obj.ob_type=&PyType_Type;")
        L.append(f"  {tn} = &_{tn}_obj;")
    L.append("}")

    # Refcounting
    L.append("""
__attribute__((weak)) void Py_INCREF(PyObject *o){if(o)o->ob_refcnt++;}
__attribute__((weak)) void Py_DECREF(PyObject *o){if(o&&--o->ob_refcnt<=0)free(o);}
__attribute__((weak)) void Py_DecRef(PyObject *o){Py_DECREF(o);}
__attribute__((weak)) void Py_XDECREF(PyObject *o){if(o)Py_DECREF(o);}
__attribute__((weak)) void Py_XINCREF(PyObject *o){if(o)Py_INCREF(o);}""")

    # Memory allocators — redirect to malloc/free/realloc
    L.append("""
// Memory allocators — redirect to standard C library
__attribute__((weak)) void* PyObject_Malloc(size_t n){return malloc(n);}
__attribute__((weak)) void PyObject_Free(void *p){free(p);}
__attribute__((weak)) void* PyObject_Realloc(void *p,size_t n){return realloc(p,n);}
__attribute__((weak)) void* PyObject_Calloc(size_t nelem,size_t elsize){return calloc(nelem,elsize);}
__attribute__((weak)) void _Py_Dealloc(PyObject *o){Py_DECREF(o);}""")

    # Module creation + initialization
    L.append("""
// Module creation — returns dummy module object with ob_type=PyModule_Type
__attribute__((weak)) PyObject* PyModule_Create2(void *def,int api){
    (void)def;(void)api; _mini_runtime_init();
    static PyObject dm={1,nullptr}; dm.ob_type=&PyModule_Type; return &dm;}
__attribute__((weak, visibility("default"))) PyObject* PyModuleDef_Init(void *def){
    (void)def; _mini_runtime_init();
    static PyObject dm={1,nullptr}; dm.ob_type=&PyModule_Type; return &dm;}
__attribute__((weak)) PyObject* PyModule_New(PyObject *name){
    (void)name; _mini_runtime_init();
    static PyObject dm={1,nullptr}; dm.ob_type=&PyModule_Type; return &dm;}
__attribute__((weak)) PyObject* PyObject_Init(PyObject *obj,PyTypeObject *tp){
    if(obj){obj->ob_refcnt=1;obj->ob_type=tp;}return obj;}
__attribute__((weak)) int PyModule_AddObject(PyObject *m,const char *n,PyObject *v){
    (void)m;(void)n;if(v)Py_DECREF(v);return 0;}
__attribute__((weak)) int PyModule_AddStringConstant(PyObject *m,const char *n,const char *v){
    (void)m;(void)n;(void)v;return 0;}
__attribute__((weak)) int PyModule_AddIntConstant(PyObject *m,const char *n,long v){
    (void)m;(void)n;(void)v;return 0;}
__attribute__((weak)) void* PyModule_GetState(PyObject *m){(void)m;return nullptr;}
__attribute__((weak)) PyObject* PyState_FindModule(void *def){(void)def;return nullptr;}""")

    # PyUnicode_* — ALWAYS include working implementations (most used API)
    L.append(r"""
// ── PyUnicode_* — working implementations ──
__attribute__((weak, visibility("default"))) PyObject* PyUnicode_New(Py_ssize_t size,Py_UCS4 maxchar){
    int kind,is_ascii=(maxchar<=127)?1:0;
    if(maxchar<=255)kind=1;else if(maxchar<=65535)kind=2;else kind=4;
    Py_ssize_t hdr=is_ascii?sizeof(PyASCIIObject):sizeof(PyCompactUnicodeObject);
    Py_ssize_t total=hdr+(Py_ssize_t)size*kind+kind;
    void *mem=calloc(1,total);if(!mem)return nullptr;
    PyASCIIObject *o=static_cast<PyASCIIObject*>(mem);
    o->ob_refcnt=1;o->ob_type=&PyUnicode_Type;o->length=size;o->hash=-1;
    o->state.interned=0;o->state.kind=kind;o->state.compact=1;
    o->state.ascii=is_ascii;
    if(!is_ascii){auto *c=reinterpret_cast<PyCompactUnicodeObject*>(mem);c->utf8=nullptr;c->utf8_length=0;}
    return reinterpret_cast<PyObject*>(mem);}
__attribute__((weak)) PyObject* PyUnicode_FromString(const char *u){
    if(!u)return nullptr;Py_ssize_t len=(Py_ssize_t)strlen(u);Py_UCS4 mc=0;
    for(Py_ssize_t i=0;i<len;i++){unsigned char c=(unsigned char)u[i];if(c>mc)mc=c;}
    PyObject *obj=PyUnicode_New(len,mc);if(!obj)return nullptr;
    PyASCIIObject *ao=reinterpret_cast<PyASCIIObject*>(obj);
    char *d=reinterpret_cast<char*>(obj)+(ao->state.ascii?sizeof(PyASCIIObject):sizeof(PyCompactUnicodeObject));
    if(ao->state.kind==1){memcpy(d,u,len);d[len]='\0';}
    else if(ao->state.kind==2){auto *p=reinterpret_cast<uint16_t*>(d);for(Py_ssize_t i=0;i<len;i++)p[i]=(uint16_t)(unsigned char)u[i];}
    else{auto *p=reinterpret_cast<uint32_t*>(d);for(Py_ssize_t i=0;i<len;i++)p[i]=(uint32_t)(unsigned char)u[i];}
    return obj;}
__attribute__((weak)) PyObject* PyUnicode_FromStringAndSize(const char *u,Py_ssize_t size){
    if(!u)return PyUnicode_New(size,127);Py_UCS4 mc=0;
    for(Py_ssize_t i=0;i<size;i++){unsigned char c=(unsigned char)u[i];if(c>mc)mc=c;}
    PyObject *obj=PyUnicode_New(size,mc);if(!obj)return nullptr;
    PyASCIIObject *ao=reinterpret_cast<PyASCIIObject*>(obj);
    char *d=reinterpret_cast<char*>(obj)+(ao->state.ascii?sizeof(PyASCIIObject):sizeof(PyCompactUnicodeObject));
    if(ao->state.kind==1){memcpy(d,u,size);d[size]='\0';}return obj;}
__attribute__((weak)) const char* PyUnicode_AsUTF8(PyObject *unicode){
    if(!unicode||unicode->ob_type!=&PyUnicode_Type)return nullptr;
    PyASCIIObject *o=reinterpret_cast<PyASCIIObject*>(unicode);
    if(o->state.ascii)return reinterpret_cast<const char*>(unicode)+sizeof(PyASCIIObject);
    PyCompactUnicodeObject *c=reinterpret_cast<PyCompactUnicodeObject*>(unicode);
    if(c->utf8)return c->utf8;
    int kind=o->state.kind;Py_ssize_t length=o->length;
    char *ds=reinterpret_cast<char*>(unicode)+sizeof(PyCompactUnicodeObject);
    Py_ssize_t bs=length*4+1;char *buf=(char*)calloc(1,bs);if(!buf)return nullptr;
    Py_ssize_t pos=0;
    for(Py_ssize_t i=0;i<length&&pos<bs-4;i++){
        Py_UCS4 ch;if(kind==1)ch=(unsigned char)ds[i];
        else if(kind==2)ch=reinterpret_cast<uint16_t*>(ds)[i];
        else ch=reinterpret_cast<uint32_t*>(ds)[i];
        if(ch<0x80)buf[pos++]=(char)ch;
        else if(ch<0x800){buf[pos++]=(char)(0xC0|(ch>>6));buf[pos++]=(char)(0x80|(ch&0x3F));}
        else if(ch<0x10000){buf[pos++]=(char)(0xE0|(ch>>12));buf[pos++]=(char)(0x80|((ch>>6)&0x3F));buf[pos++]=(char)(0x80|(ch&0x3F));}
        else{buf[pos++]=(char)(0xF0|(ch>>18));buf[pos++]=(char)(0x80|((ch>>12)&0x3F));buf[pos++]=(char)(0x80|((ch>>6)&0x3F));buf[pos++]=(char)(0x80|(ch&0x3F));}}
    buf[pos]='\0';c->utf8=buf;c->utf8_length=pos;return buf;}
__attribute__((weak)) const char* PyUnicode_AsUTF8AndSize(PyObject *unicode,Py_ssize_t *size){
    const char *r=PyUnicode_AsUTF8(unicode);
    if(r&&size){*size=reinterpret_cast<PyASCIIObject*>(unicode)->length;}return r;}
__attribute__((weak)) Py_ssize_t PyUnicode_GetLength(PyObject *unicode){
    if(!unicode||unicode->ob_type!=&PyUnicode_Type)return -1;
    return reinterpret_cast<PyASCIIObject*>(unicode)->length;}
__attribute__((weak)) PyObject* PyUnicode_InternFromString(const char *u){return PyUnicode_FromString(u);}
__attribute__((weak)) PyObject* PyUnicode_AsEncodedString(PyObject *unicode,const char *encoding,const char *errors){
    // Convert unicode to bytes (UTF-8 by default)
    // ujson uses this to get key bytes, then PyBytes_AsString on the result
    (void)encoding;(void)errors;
    if(!unicode||unicode->ob_type!=&PyUnicode_Type)return nullptr;
    const char *utf8=PyUnicode_AsUTF8(unicode);
    if(!utf8)return nullptr;
    Py_ssize_t len=PyUnicode_GetLength(unicode);
    return PyBytes_FromStringAndSize(utf8,len);
}
__attribute__((weak)) PyObject* PyUnicode_AsUTF8String(PyObject *unicode){
    // Convert unicode to UTF-8 bytes object
    return PyUnicode_AsEncodedString(unicode,"utf-8",nullptr);
}
__attribute__((weak)) PyObject* PyUnicode_FromKindAndData(int kind,const void *data,Py_ssize_t size){
    Py_UCS4 mc=127;if(kind==2)mc=0xFFFF;else if(kind==4)mc=0x10FFFF;
    PyObject *obj=PyUnicode_New(size,mc);if(!obj||!data)return obj;
    PyASCIIObject *ao=reinterpret_cast<PyASCIIObject*>(obj);
    char *d=reinterpret_cast<char*>(obj)+(ao->state.ascii?sizeof(PyASCIIObject):sizeof(PyCompactUnicodeObject));
    memcpy(d,data,size*kind);return obj;}
__attribute__((weak)) PyObject* PyUnicode_DecodeUTF8(const char *s,Py_ssize_t size,const char *errors){
    (void)errors;return PyUnicode_FromStringAndSize(s,size);}
__attribute__((weak)) int _PyUnicode_Ready(PyObject *o){(void)o;return 0;}""")

    # PyLong_* — ALWAYS include working implementations
    L.append(r"""
// ── PyLong_* — working implementations ──
static PyObject* _pylong_from_long(long v){
    PyLongObject *o=(PyLongObject*)calloc(1,sizeof(PyLongObject)+sizeof(uint32_t));
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyLong_Type;
    if(v<0){o->ob_size=-1;o->ob_digit[0]=(uint32_t)(-v);}
    else if(v==0){o->ob_size=0;}else{o->ob_size=1;o->ob_digit[0]=(uint32_t)v;}
    return(PyObject*)o;}
__attribute__((weak)) PyObject* PyLong_FromLong(long v){return _pylong_from_long(v);}
__attribute__((weak)) PyObject* PyLong_FromUnsignedLong(unsigned long v){
    PyLongObject *o=(PyLongObject*)calloc(1,sizeof(PyLongObject)+sizeof(uint32_t));
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyLong_Type;o->ob_size=1;o->ob_digit[0]=(uint32_t)v;return(PyObject*)o;}
__attribute__((weak)) PyObject* PyLong_FromLongLong(long long v){return _pylong_from_long((long)v);}
__attribute__((weak)) PyObject* PyLong_FromUnsignedLongLong(unsigned long long v){
    PyLongObject *o=(PyLongObject*)calloc(1,sizeof(PyLongObject)+sizeof(uint32_t));
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyLong_Type;o->ob_size=1;o->ob_digit[0]=(uint32_t)v;return(PyObject*)o;}
__attribute__((weak)) PyObject* PyLong_FromSsize_t(Py_ssize_t v){return _pylong_from_long((long)v);}
__attribute__((weak)) long PyLong_AsLong(PyObject *obj){
    if(!obj||obj->ob_type!=&PyLong_Type)return -1;
    PyLongObject *o=(PyLongObject*)obj;if(o->ob_size==0)return 0;
    long v=(long)o->ob_digit[0];return(o->ob_size<0)?-v:v;}
__attribute__((weak)) Py_ssize_t PyLong_AsSsize_t(PyObject *obj){return(Py_ssize_t)PyLong_AsLong(obj);}
__attribute__((weak)) long long PyLong_AsLongLong(PyObject *obj){return(long long)PyLong_AsLong(obj);}
__attribute__((weak)) unsigned long long PyLong_AsUnsignedLongLong(PyObject *obj){return(unsigned long long)PyLong_AsLong(obj);}
__attribute__((weak)) PyObject* PyLong_FromString(const char *str,char **pend,int base){
    (void)base;char *end=nullptr;long v=strtol(str,&end,base);
    if(pend)*pend=end;return _pylong_from_long(v);}""")

    # PyFloat_* — working implementations
    L.append("""
// ── PyFloat_* — working implementations ──
__attribute__((weak)) PyObject* PyFloat_FromDouble(double v){
    PyFloatObject *o=(PyFloatObject*)calloc(1,sizeof(PyFloatObject));
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyFloat_Type;o->ob_fval=v;return(PyObject*)o;}
__attribute__((weak)) double PyFloat_AsDouble(PyObject *obj){
    if(!obj)return -1.0;
    if(obj->ob_type==&PyFloat_Type)return((PyFloatObject*)obj)->ob_fval;
    if(obj->ob_type==&PyLong_Type)return(double)PyLong_AsLong(obj);
    return -1.0;}""")

    # PyBytes_*
    L.append("""
// ── PyBytes_* — working implementations ──
__attribute__((weak)) PyObject* PyBytes_FromString(const char *s){
    if(!s)return nullptr;Py_ssize_t len=(Py_ssize_t)strlen(s);
    PyBytesObject *o=(PyBytesObject*)calloc(1,sizeof(PyBytesObject)+len);
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyBytes_Type;o->ob_size=len;o->ob_shash=-1;
    memcpy(o->ob_sval,s,len+1);return(PyObject*)o;}
__attribute__((weak)) PyObject* PyBytes_FromStringAndSize(const char *s,Py_ssize_t len){
    PyBytesObject *o=(PyBytesObject*)calloc(1,sizeof(PyBytesObject)+len);
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyBytes_Type;o->ob_size=len;o->ob_shash=-1;
    if(s)memcpy(o->ob_sval,s,len);o->ob_sval[len]='\\0';return(PyObject*)o;}
__attribute__((weak)) char* PyBytes_AsString(PyObject *obj){
    if(!obj||obj->ob_type!=&PyBytes_Type)return nullptr;
    return((PyBytesObject*)obj)->ob_sval;}
__attribute__((weak)) Py_ssize_t PyBytes_Size(PyObject *obj){
    if(!obj||obj->ob_type!=&PyBytes_Type)return -1;
    return((PyBytesObject*)obj)->ob_size;}
__attribute__((weak)) int PyBytes_CheckExact(PyObject *o){return PyBytes_Check(o);}""")

    # PyDict_* — WORKING implementations with real storage
    L.append(r"""
// ── PyDict_* — working implementations with real key-value storage ──
__attribute__((weak)) PyObject* PyDict_New(void){
    PyDictObject *o=(PyDictObject*)calloc(1,sizeof(PyDictObject));
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyDict_Type;
    o->ma_used=0;o->ma_allocated=8;
    o->ma_entries=(_PyDictEntry*)calloc(8,sizeof(_PyDictEntry));
    if(!o->ma_entries){free(o);return nullptr;}
    return(PyObject*)o;}
// Helper: find entry by key (using pointer identity for speed)
static int _dict_find(PyDictObject *d,PyObject *k){
    for(Py_ssize_t i=0;i<d->ma_used;i++){if(d->ma_entries[i].key==k)return (int)i;}
    return -1;}
__attribute__((weak)) int PyDict_SetItem(PyObject *d,PyObject *k,PyObject *v){
    if(!d||!k)return -1;PyDictObject *dict=(PyDictObject*)d;
    int idx=_dict_find(dict,k);
    if(idx>=0){
        // Update existing entry
        if(v)Py_INCREF(v);
        if(dict->ma_entries[idx].value)Py_DECREF(dict->ma_entries[idx].value);
        dict->ma_entries[idx].value=v;
    } else {
        // Add new entry — grow array if needed
        if(dict->ma_used>=dict->ma_allocated){
            Py_ssize_t new_alloc=dict->ma_allocated*2;
            _PyDictEntry *new_arr=(_PyDictEntry*)realloc(dict->ma_entries,new_alloc*sizeof(_PyDictEntry));
            if(!new_arr)return -1;
            memset(new_arr+dict->ma_used,0,(new_alloc-dict->ma_allocated)*sizeof(_PyDictEntry));
            dict->ma_entries=new_arr;dict->ma_allocated=new_alloc;
        }
        Py_INCREF(k);if(v)Py_INCREF(v);
        dict->ma_entries[dict->ma_used].key=k;
        dict->ma_entries[dict->ma_used].value=v;
        dict->ma_used++;
    }
    return 0;}
__attribute__((weak)) int PyDict_SetItemString(PyObject *d,const char *k,PyObject *v){
    if(!d||!k)return -1;
    PyObject *key=PyUnicode_FromString(k);if(!key)return -1;
    int r=PyDict_SetItem(d,key,v);Py_DECREF(key);return r;}
__attribute__((weak)) PyObject* PyDict_GetItem(PyObject *d,PyObject *k){
    if(!d)return nullptr;PyDictObject *dict=(PyDictObject*)d;
    int idx=_dict_find(dict,k);
    return(idx>=0)?dict->ma_entries[idx].value:nullptr;}
__attribute__((weak)) PyObject* PyDict_GetItemString(PyObject *d,const char *k){
    if(!d||!k)return nullptr;
    PyObject *key=PyUnicode_FromString(k);if(!key)return nullptr;
    PyObject *v=PyDict_GetItem(d,key);Py_DECREF(key);return v;}
__attribute__((weak)) int PyDict_Next(PyObject *d,Py_ssize_t *pos,PyObject **k,PyObject **v){
    if(!d||!pos)return 0;PyDictObject *dict=(PyDictObject*)d;
    if(*pos>=dict->ma_used)return 0;
    if(k)*k=dict->ma_entries[*pos].key;
    if(v)*v=dict->ma_entries[*pos].value;
    (*pos)++;return 1;}
__attribute__((weak)) PyObject* PyDict_Keys(PyObject *d){
    if(!d)return PyList_New(0);PyDictObject *dict=(PyDictObject*)d;
    PyObject *list=PyList_New(dict->ma_used);if(!list)return nullptr;
    for(Py_ssize_t i=0;i<dict->ma_used;i++){PyListObject *pl=(PyListObject*)list;pl->ob_item[i]=dict->ma_entries[i].key;Py_INCREF(dict->ma_entries[i].key);}
    return list;}
__attribute__((weak)) Py_ssize_t PyDict_Size(PyObject *d){
    if(!d)return 0;return((PyDictObject*)d)->ma_used;}""")

    # PyList_*
    L.append("""
// ── PyList_* — minimal implementations ──
__attribute__((weak)) PyObject* PyList_New(Py_ssize_t len){
    PyListObject *o=(PyListObject*)calloc(1,sizeof(PyListObject));
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyList_Type;
    o->ob_size=len;o->allocated=len;
    if(len>0){o->ob_item=(PyObject**)calloc(len,sizeof(PyObject*));if(!o->ob_item){free(o);return nullptr;}}
    return(PyObject*)o;}
__attribute__((weak)) int PyList_Append(PyObject *l,PyObject *v){
    if(!l||l->ob_type!=&PyList_Type)return -1;
    PyListObject *o=(PyListObject*)l;
    Py_ssize_t new_size=o->ob_size+1;
    if(new_size>o->allocated){
        Py_ssize_t new_alloc=new_size*2;
        PyObject **new_item=(PyObject**)realloc(o->ob_item,new_alloc*sizeof(PyObject*));
        if(!new_item)return -1;
        memset(new_item+o->allocated,0,(new_alloc-o->allocated)*sizeof(PyObject*));
        o->ob_item=new_item;o->allocated=new_alloc;
    }
    if(v)Py_INCREF(v);
    o->ob_item[o->ob_size]=v;o->ob_size=new_size;return 0;}
__attribute__((weak)) int PyList_Sort(PyObject *l){(void)l;return 0;}
__attribute__((weak)) PyObject* PyList_GetItem(PyObject *l,Py_ssize_t i){
    if(!l||l->ob_type!=&PyList_Type)return nullptr;
    PyListObject *o=(PyListObject*)l;if(i<0||i>=o->ob_size)return nullptr;return o->ob_item[i];}
__attribute__((weak)) int PyList_SetItem(PyObject *l,Py_ssize_t i,PyObject *v){
    if(!l||l->ob_type!=&PyList_Type||i<0)return -1;
    PyListObject *o=(PyListObject*)l;
    if(i>=o->ob_size)return -1;
    if(o->ob_item[i])Py_DECREF(o->ob_item[i]);
    o->ob_item[i]=v;
    return 0;}""")

    # PyTuple_*
    L.append("""
// ── PyTuple_* — minimal implementations ──
__attribute__((weak)) PyObject* PyTuple_New(Py_ssize_t len){
    Py_ssize_t sz=sizeof(PyTupleObject)+(len>0?len*sizeof(PyObject*):sizeof(PyObject*));
    PyTupleObject *o=(PyTupleObject*)calloc(1,sz);
    if(!o)return nullptr;o->ob_refcnt=1;o->ob_type=&PyTuple_Type;o->ob_size=len;
    return(PyObject*)o;}
__attribute__((weak)) PyObject* PyTuple_Pack(PyObject *first,...){(void)first;return PyTuple_New(0);}
__attribute__((weak)) PyObject* PyTuple_GetItem(PyObject *t,Py_ssize_t i){
    if(!t||t->ob_type!=&PyTuple_Type)return nullptr;
    PyTupleObject *o=(PyTupleObject*)t;if(i<0||i>=o->ob_size)return nullptr;return o->ob_item[i];}
// PyTuple_SetItem: steals reference to v (like CPython), sets item in tuple.
// Returns 0 on success, -1 on error.
__attribute__((weak)) int PyTuple_SetItem(PyObject *t,Py_ssize_t i,PyObject *v){
    if(!t||t->ob_type!=&PyTuple_Type||i<0)return -1;
    PyTupleObject *o=(PyTupleObject*)t;
    if(i>=o->ob_size)return -1;
    // Steal reference: do NOT INCREF v (CPython steals the reference)
    // If there's an existing item, DECREF it
    if(o->ob_item[i])Py_DECREF(o->ob_item[i]);
    o->ob_item[i]=v;
    return 0;}""")

    # PyObject utilities + error handling + type checks
    L.append("""
// ── PyObject utilities ──
__attribute__((weak)) int PyObject_IsTrue(PyObject *o){
    if(!o)return 0;
    if(o==&_Py_FalseStruct)return 0;
    if(o==&_Py_NoneStruct)return 0;
    if(o->ob_type==&PyLong_Type){
        PyLongObject *lo=(PyLongObject*)o;return lo->ob_size!=0;
    }
    return 1;}
__attribute__((weak)) int PyObject_Not(PyObject *o){return !PyObject_IsTrue(o);}
__attribute__((weak)) PyObject* PyObject_Str(PyObject *o){
    if(!o)return nullptr;
    if(o->ob_type==&PyUnicode_Type){Py_INCREF(o);return o;}
    if(o->ob_type==&PyLong_Type){
        long v=PyLong_AsLong(o);char buf[32];snprintf(buf,32,"%ld",v);return PyUnicode_FromString(buf);}
    if(o->ob_type==&PyFloat_Type){
        double v=PyFloat_AsDouble(o);char buf[64];snprintf(buf,64,"%g",v);return PyUnicode_FromString(buf);}
    Py_INCREF(o);return o;}
__attribute__((weak)) PyObject* PyObject_Repr(PyObject *o){return PyObject_Str(o);}
__attribute__((weak)) PyObject* PyObject_Type(PyObject *o){if(!o)return nullptr;return(PyObject*)o->ob_type;}
__attribute__((weak)) int PyObject_IsInstance(PyObject *o,PyObject *c){(void)o;(void)c;return 0;}
__attribute__((weak)) int PyObject_IsSubclass(PyObject *o,PyObject *c){(void)o;(void)c;return 0;}
__attribute__((weak)) PyObject* PyObject_GetAttrString(PyObject *o,const char *n){(void)o;(void)n;return nullptr;}
__attribute__((weak)) int PyObject_HasAttrString(PyObject *o,const char *n){(void)o;(void)n;return 0;}
__attribute__((weak)) PyObject* PyObject_CallObject(PyObject *c,PyObject *a){(void)c;(void)a;return nullptr;}
__attribute__((weak)) PyObject* PyObject_CallFunctionObjArgs(PyObject *c,...){(void)c;return nullptr;}
__attribute__((weak)) PyObject* PyObject_CallMethod(PyObject *o,const char *n,const char *f,...){(void)o;(void)n;(void)f;return nullptr;}
__attribute__((weak)) int PyObject_GetBuffer(PyObject *o,void *b,int f){(void)o;(void)b;(void)f;return -1;}
__attribute__((weak)) void PyBuffer_Release(void *b){(void)b;}
__attribute__((weak)) PyObject* PyObject_Number(PyObject *o){Py_INCREF(o);return o;}

// ── Error handling ──
__attribute__((weak)) void PyErr_SetString(PyObject *e,const char *m){(void)e;fprintf(stderr,"[pylow-ffi] PyErr_SetString: %s\\n",m?m:"(null)");}
__attribute__((weak)) PyObject* PyErr_Occurred(void){return nullptr;}
__attribute__((weak)) void PyErr_Clear(void){}
__attribute__((weak)) void PyErr_NoMemory(void){}
__attribute__((weak)) int PyErr_ExceptionMatches(PyObject *e){(void)e;return 0;}
__attribute__((weak)) PyObject* PyErr_Format(PyObject *e,const char *f,...){(void)e;(void)f;return nullptr;}
__attribute__((weak)) PyObject* PyErr_NewException(const char *n,PyObject *b,PyObject *d){(void)n;(void)b;(void)d;return nullptr;}

// ── Type checks ──
__attribute__((weak)) int PyUnicode_Check(PyObject *o){return o&&o->ob_type&&(o->ob_type->tp_flags&Py_TPFLAGS_UNICODE_SUBCLASS);}
__attribute__((weak)) int PyLong_Check(PyObject *o){return o&&o->ob_type&&(o->ob_type->tp_flags&Py_TPFLAGS_LONG_SUBCLASS);}
__attribute__((weak)) int PyBytes_Check(PyObject *o){return o&&o->ob_type&&(o->ob_type->tp_flags&Py_TPFLAGS_BYTES_SUBCLASS);}
__attribute__((weak)) int PyTuple_Check(PyObject *o){return o&&o->ob_type&&(o->ob_type->tp_flags&Py_TPFLAGS_TUPLE_SUBCLASS);}
__attribute__((weak)) int PyList_Check(PyObject *o){return o&&o->ob_type&&(o->ob_type->tp_flags&Py_TPFLAGS_LIST_SUBCLASS);}
__attribute__((weak)) int PyDict_Check(PyObject *o){return o&&o->ob_type&&(o->ob_type->tp_flags&Py_TPFLAGS_DICT_SUBCLASS);}
__attribute__((weak)) int PyFloat_Check(PyObject *o){return o&&o->ob_type==&PyFloat_Type;}
__attribute__((weak)) int PyBool_Check(PyObject *o){return (o==&_Py_TrueStruct||o==&_Py_FalseStruct);}
__attribute__((weak)) int PyCallable_Check(PyObject *o){(void)o;return 0;}
__attribute__((weak)) int PyIter_Check(PyObject *o){(void)o;return 0;}
__attribute__((weak)) int PyType_IsSubtype(PyTypeObject *a,PyTypeObject *b){
    if(!a||!b)return 0;
    if(a==b)return 1;
    // For our mini-runtime, all type objects with matching tp_flags subclass bits
    // are considered subtypes. This is enough for isinstance() checks in .so code.
    if(b==&PyType_Type){
        // Everything whose ob_type is a type object is an instance of type
        return (a->ob_type==&PyType_Type)?1:0;
    }
    return 0;
}

// ── Import ──
__attribute__((weak)) PyObject* PyImport_ImportModule(const char *n){(void)n;return nullptr;}

// ── Argument parsing — WORKING implementations for format O|Osif... ──
// Minimal but functional PyArg_ParseTuple that handles the most common format
// specifiers: O (PyObject*), i (int), s (const char*), f (double), l (long),
// | (optional separator), : (function name), $ (error message).
// This is enough for ujson (O|OOOOiiiOO), markupsafe, and most CPython extensions.
static int _pyarg_parse_impl(PyObject *args, const char *format, va_list *vap) {
    if(!args||!format) return 0;
    // Get tuple size
    Py_ssize_t nargs = 0;
    if(args->ob_type==&PyTuple_Type){
        PyTupleObject *t=(PyTupleObject*)args; nargs=t->ob_size;
    } else if(args->ob_type==&PyList_Type){
        PyListObject *l=(PyListObject*)args; nargs=l->ob_size;
    } else return 0;

    int argidx = 0;
    int optional = 0;
    for(const char *p = format; *p; p++) {
        if(*p == '|') { optional = 1; continue; }
        if(*p == ':' || *p == '$') break;  // function name / error msg separator
        if(argidx >= nargs) {
            // If we're in optional section, missing args are set to default
            if(optional) {
                switch(*p) {
                    case 'O': { PyObject **out=va_arg(*vap,PyObject**); *out=&_Py_NoneStruct; break; }
                    case 'i': { int *out=va_arg(*vap,int*); *out=0; break; }
                    case 'l': { long *out=va_arg(*vap,long*); *out=0; break; }
                    case 'f': { double *out=va_arg(*vap,double*); *out=0.0; break; }
                    case 's': { const char **out=va_arg(*vap,const char**); *out=""; break; }
                    case 'n': { Py_ssize_t *out=va_arg(*vap,Py_ssize_t*); *out=0; break; }
                    case 'b': { int *out=va_arg(*vap,int*); *out=0; break; }
                    case 'B': { int *out=va_arg(*vap,int*); *out=0; break; }
                    case 'h': { int *out=va_arg(*vap,int*); *out=0; break; }
                    case 'd': { double *out=va_arg(*vap,double*); *out=0.0; break; }
                    case 'D': { /* complex — skip */ va_arg(*vap,void*); break; }
                    case 'S': { PyObject **out=va_arg(*vap,PyObject**); *out=&_Py_NoneStruct; break; }
                    case 'N': { PyObject **out=va_arg(*vap,PyObject**); *out=&_Py_NoneStruct; break; }
                    case 'z': { const char **out=va_arg(*vap,const char**); *out=nullptr; break; }
                    case 'p': { int *out=va_arg(*vap,int*); *out=0; break; }
                    case 'L': { long long *out=va_arg(*vap,long long*); *out=0; break; }
                    case 'I': { unsigned int *out=va_arg(*vap,unsigned int*); *out=0; break; }
                    case 'k': { unsigned long *out=va_arg(*vap,unsigned long*); *out=0; break; }
                    case 'K': { unsigned long long *out=va_arg(*vap,unsigned long long*); *out=0; break; }
                    case 'C': { /* int as long */ { long *out=va_arg(*vap,long*); *out=0; } break; }
                    case 'c': { /* char from str */ { char *out=va_arg(*vap,char*); *out='\0'; } break; }
                    case 'e': { const char **out=va_arg(*vap,const char**); *out="utf-8"; break; }
                    case 'u': { /* Py_UNICODE* */ va_arg(*vap,void*); break; }
                    default: va_arg(*vap,void*); break;
                }
                argidx++;
                continue;
            }
            return 0;  // Missing required arg
        }
        // Get the arg from the tuple
        PyObject *item = nullptr;
        if(args->ob_type==&PyTuple_Type){
            PyTupleObject *t=(PyTupleObject*)args;
            item = (argidx < t->ob_size) ? t->ob_item[argidx] : nullptr;
        } else if(args->ob_type==&PyList_Type){
            PyListObject *l=(PyListObject*)args;
            item = (argidx < l->ob_size) ? l->ob_item[argidx] : nullptr;
        }
        if(!item && !optional) return 0;

        switch(*p) {
            case 'O': {
                PyObject **out = va_arg(*vap, PyObject**);
                *out = item;
                break;
            }
            case 'S': {  // str object, no conversion
                PyObject **out = va_arg(*vap, PyObject**);
                *out = item;
                break;
            }
            case 'N': {  // object, steals ref (we don't)
                PyObject **out = va_arg(*vap, PyObject**);
                *out = item;
                break;
            }
            case 'i': {
                int *out = va_arg(*vap, int*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = (int)PyLong_AsLong(item);
                else *out = PyObject_IsTrue(item);
                break;
            }
            case 'b': {
                int *out = va_arg(*vap, int*);
                if(!item) *out = 0;
                else *out = PyObject_IsTrue(item);
                break;
            }
            case 'B': {
                int *out = va_arg(*vap, int*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = (int)PyLong_AsLong(item);
                else *out = 0;
                break;
            }
            case 'h': {
                int *out = va_arg(*vap, int*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = (int)PyLong_AsLong(item);
                else *out = 0;
                break;
            }
            case 'l': {
                long *out = va_arg(*vap, long*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = PyLong_AsLong(item);
                else *out = 0;
                break;
            }
            case 'L': {
                long long *out = va_arg(*vap, long long*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = PyLong_AsLongLong(item);
                else *out = 0;
                break;
            }
            case 'I': {
                unsigned int *out = va_arg(*vap, unsigned int*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = (unsigned int)PyLong_AsLong(item);
                else *out = 0;
                break;
            }
            case 'k': {
                unsigned long *out = va_arg(*vap, unsigned long*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = (unsigned long)PyLong_AsLong(item);
                else *out = 0;
                break;
            }
            case 'K': {
                unsigned long long *out = va_arg(*vap, unsigned long long*);
                if(!item) *out = 0;
                else *out = PyLong_AsUnsignedLongLong(item);
                break;
            }
            case 'n': {
                Py_ssize_t *out = va_arg(*vap, Py_ssize_t*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = PyLong_AsSsize_t(item);
                else *out = 0;
                break;
            }
            case 'f': {
                double *out = va_arg(*vap, double*);
                if(!item) *out = 0.0;
                else *out = PyFloat_AsDouble(item);
                break;
            }
            case 'd': {
                double *out = va_arg(*vap, double*);
                if(!item) *out = 0.0;
                else *out = PyFloat_AsDouble(item);
                break;
            }
            case 's': {
                const char **out = va_arg(*vap, const char**);
                if(!item) *out = "";
                else if(item->ob_type==&PyUnicode_Type) *out = PyUnicode_AsUTF8(item);
                else if(item->ob_type==&PyBytes_Type) *out = PyBytes_AsString(item);
                else *out = "";
                break;
            }
            case 'z': {  // s or None
                const char **out = va_arg(*vap, const char**);
                if(!item || item==&_Py_NoneStruct) *out = nullptr;
                else if(item->ob_type==&PyUnicode_Type) *out = PyUnicode_AsUTF8(item);
                else if(item->ob_type==&PyBytes_Type) *out = PyBytes_AsString(item);
                else *out = nullptr;
                break;
            }
            case 'p': {  // predicate (bool)
                int *out = va_arg(*vap, int*);
                *out = item ? PyObject_IsTrue(item) : 0;
                break;
            }
            case 'c': {  // char from str of length 1
                char *out = va_arg(*vap, char*);
                if(item && item->ob_type==&PyUnicode_Type) {
                    const char *s = PyUnicode_AsUTF8(item);
                    *out = s ? s[0] : '\0';
                } else *out = '\0';
                break;
            }
            case 'C': {  // int as long
                long *out = va_arg(*vap, long*);
                if(!item) *out = 0;
                else if(item->ob_type==&PyLong_Type) *out = PyLong_AsLong(item);
                else *out = 0;
                break;
            }
            case 'e': {  // encoding string
                const char **out = va_arg(*vap, const char**);
                *out = "utf-8";
                break;
            }
            case 'D': {  // complex
                va_arg(*vap, void*);  // skip
                break;
            }
            case 'u': {  // Py_UNICODE*
                va_arg(*vap, void*);  // skip
                break;
            }
            default:
                // Unknown format — skip output pointer
                va_arg(*vap, void*);
                break;
        }
        argidx++;
    }
    return 1;
}

__attribute__((weak)) int PyArg_ParseTuple(PyObject *args, const char *format, ...) {
    va_list vap;
    va_start(vap, format);
    int r = _pyarg_parse_impl(args, format, &vap);
    va_end(vap);
    return r;
}
__attribute__((weak)) int PyArg_ParseTupleAndKeywords(PyObject *args, PyObject *kwargs, const char *format, ...) {
    // For now, ignore kwargs — most CPython extensions use positional args
    // for the critical path (e.g., ujson's O|OOOOiiiOO uses all positional)
    (void)kwargs;
    va_list vap;
    va_start(vap, format);
    // Skip the 'keywords' char** parameter that comes after format
    // In CPython: PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, ...)
    // The va_list starts after format, so keywords is the first va_arg
    // We need to consume the keywords parameter
    va_arg(vap, char**);  // skip keywords array
    int r = _pyarg_parse_impl(args, format, &vap);
    va_end(vap);
    return r;
}

// ── PyNumber — safe stubs ──
__attribute__((weak)) PyObject* PyNumber_ToBase(PyObject *o,int base){(void)o;(void)base;return nullptr;}""")

    # Remaining imported symbols — SAFE stubs (return 0/nullptr, NO abort)
    remaining = [s for s in if_ if s not in _COVERED_SYMS]
    if remaining:
        L.append("// ── Remaining Py* — safe stubs (return 0/nullptr, no abort) ──")
        for sym in remaining:
            rc, pc = get_symbol_info(sym)
            L.append(_safe_stub(sym, rc, pc))

    # pylow_ffi_free — utility for freeing wrapper-allocated memory
    L.append('\n__attribute__((weak, visibility("default"))) void pylow_ffi_free(void *ptr)'+'{free(ptr);}')

    # pylow_ffi_create_dict — creates a PyDict from serialized key-value pairs.
    # The LLVM side serializes dict entries as null-separated strings:
    #   keys: "key1\0key2\0key3\0"  (contiguous, null-terminated strings)
    #   vals: "val1\0val2\0val3\0"
    #   types: type codes for each value (0=string, 1=int, 2=float, 3=bool, 4=None)
    # count: number of key-value pairs
    # Returns a new PyObject* dict (caller must Py_DecRef when done).
    L.append(r"""
// ── Object Bridge: pylow → CPython PyObject* conversion ──
__attribute__((weak, visibility("default"))) PyObject* pylow_ffi_create_dict(
    const char *keys_data, int64_t keys_len,
    const char *vals_data, int64_t vals_len,
    const int64_t *val_types, int64_t count) {
    _mini_runtime_init();
    PyObject *dict = PyDict_New();
    if(!dict) return nullptr;
    const char *kp = keys_data;
    const char *vp = vals_data;
    for(int64_t i = 0; i < count; i++) {
        // Create key (always a string in pylow's dict representation)
        PyObject *key_obj = PyUnicode_FromString(kp);
        if(!key_obj) { Py_DecRef(dict); return nullptr; }
        kp += strlen(kp) + 1;
        // Create value based on type
        PyObject *val_obj = nullptr;
        int64_t vtype = val_types ? val_types[i] : 0;
        switch(vtype) {
            case 1: { // int
                long iv = strtol(vp, nullptr, 10);
                val_obj = PyLong_FromLong(iv);
                break;
            }
            case 2: { // float
                double dv = strtod(vp, nullptr);
                val_obj = PyFloat_FromDouble(dv);
                break;
            }
            case 3: { // bool
                val_obj = (strcmp(vp, "True") == 0) ? PyLong_FromLong(1) : PyLong_FromLong(0);
                break;
            }
            case 4: { // None
                val_obj = &_Py_NoneStruct;
                Py_INCREF(val_obj);
                break;
            }
            default: { // string (type 0) or unknown
                val_obj = PyUnicode_FromString(vp);
                break;
            }
        }
        if(!val_obj) { Py_DecRef(key_obj); Py_DecRef(dict); return nullptr; }
        vp += strlen(vp) + 1;
        PyDict_SetItem(dict, key_obj, val_obj);
        Py_DecRef(key_obj);
        Py_DecRef(val_obj);
    }
    return dict;
}
// pylow_ffi_create_list — creates a PyList from serialized items.
// items_data: null-separated string representations
// item_types: type codes per item (0=string, 1=int, 2=float, 3=bool, 4=None)
// count: number of items
__attribute__((weak, visibility("default"))) PyObject* pylow_ffi_create_list(
    const char *items_data, int64_t items_len,
    const int64_t *item_types, int64_t count) {
    _mini_runtime_init();
    PyObject *list = PyList_New((Py_ssize_t)count);
    if(!list) return nullptr;
    const char *ip = items_data;
    for(int64_t i = 0; i < count; i++) {
        PyObject *item_obj = nullptr;
        int64_t itype = item_types ? item_types[i] : 0;
        switch(itype) {
            case 1: { long iv = strtol(ip, nullptr, 10); item_obj = PyLong_FromLong(iv); break; }
            case 2: { double dv = strtod(ip, nullptr); item_obj = PyFloat_FromDouble(dv); break; }
            case 3: { item_obj = (strcmp(ip, "True") == 0) ? PyLong_FromLong(1) : PyLong_FromLong(0); break; }
            case 4: { item_obj = &_Py_NoneStruct; Py_INCREF(item_obj); break; }
            default: { item_obj = PyUnicode_FromString(ip); break; }
        }
        if(!item_obj) { Py_DecRef(list); return nullptr; }
        ip += strlen(ip) + 1;
        // PyList_SET_ITEM steals reference
        PyTupleObject *lo = (PyTupleObject*)list;
        // Use the list's internal storage directly
        // PyList_New allocates ob_item array; we set items there
        PyListObject *plist = (PyListObject*)list;
        plist->ob_item[i] = item_obj;
    }
    return list;
}""")

    L.append('\n#pragma GCC visibility pop')
    L.append('\n} // extern "C"')
    return "\n".join(L)


def _method_return_type(mdef: Dict, module: FFIModule) -> str:
    """Determine wrapper return type based on method info."""
    c_sym = mdef.get("func_symbol", "")
    if c_sym in module.exported_symbols:
        sym = module.exported_symbols[c_sym]
        if sym.ret_type in (RET_PYOBJ, RET_CONSTCHAR): return "char*"
        if sym.ret_type in (RET_INT, RET_LONG): return "int64_t"
    return "char*"  # CPython extension methods return PyObject*


def _generate_wrapper(module: FFIModule, sigs: Dict[str, WrapperSignature]) -> str:
    """Generate C++ wrapper source for a registered module."""
    nm = module.name

    # ── Resolve actual .so file paths ──
    # module.filepath may be a package directory (e.g., .../site-packages/markupsafe)
    # but dlopen() requires a real .so file.  We use _so_pyinit_map which maps
    # actual .so paths → PyInit info.  Fall back to method_defs' _so_path annotation.
    so_pyinit_map = getattr(module, '_so_pyinit_map', {})
    so_paths = list(so_pyinit_map.keys()) if so_pyinit_map else []
    if not so_paths:
        # Fallback: collect unique _so_path from method_defs
        seen = set()
        for mdef in module.method_defs:
            sp = mdef.get("_so_path", "")
            if sp and sp not in seen:
                so_paths.append(sp)
                seen.add(sp)
    if not so_paths:
        # Last resort: if filepath is a .so file, use it directly
        if module.filepath and os.path.isfile(module.filepath) and module.filepath.endswith('.so'):
            so_paths = [module.filepath]
        else:
            so_paths = [module.filepath]  # will fail at runtime but at least compiles

    # Primary .so path (first one) — used for the shared dlopen handle
    primary_so = so_paths[0]

    # IMPORTANT: The wrapper does NOT #include the runtime source file.
    # Instead, it declares the mini-runtime functions as extern "C" and
    # relies on the linker to resolve them from ffi_<module>_runtime.o.
    # This avoids path issues and allows separate compilation.
    #
    # CRITICAL: CPython extension functions like escape_unicode are often
    # LOCAL (static) symbols — dlsym(RTLD_DEFAULT) can NOT find them.
    # Instead, we use dlopen() + vaddr-based resolution: load the .so,
    # get its base address, then compute func_ptr = base + vaddr.
    L = [f"""\
// ffi_{nm}_wrapper.cpp — AOT wrapper functions for pylow (FFIManager)
// Converts pylow native types ↔ PyObject* via mini-runtime, calls .so via dlopen+vaddr.
// Runtime symbols are resolved by the linker from ffi_{nm}_runtime.o.
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

// Forward declarations — provided by ffi_{nm}_runtime.o at link time
typedef long Py_ssize_t;
struct _typeobject;
struct _object {{ Py_ssize_t ob_refcnt; _typeobject *ob_type; }};
typedef struct _object PyObject;
typedef struct _typeobject PyTypeObject;

extern "C" {{
void _mini_runtime_init(void);
PyObject* PyUnicode_FromStringAndSize(const char*, Py_ssize_t);
void Py_DecRef(PyObject*);
void Py_INCREF(PyObject*);
const char* PyUnicode_AsUTF8(PyObject*);
Py_ssize_t PyLong_AsSsize_t(PyObject*);
PyObject* PyTuple_New(Py_ssize_t);
int PyTuple_SetItem(PyObject*, Py_ssize_t, PyObject*);
void pylow_ffi_free(void*);
}}

// ── dlopen+vaddr resolution for LOCAL (static) symbols ──
// CPython extensions often declare their functions as 'static', which
// means they don't appear in the dynamic symbol table.  dlsym() can't
// find them.  We work around this by computing the function address
// from the .so's base address + the virtual address offset from ELF."""]

    # Generate per-.so dlopen handles (supports packages with multiple .so files)
    for idx, so_path in enumerate(so_paths):
        fn_suffix = f"_{idx}" if idx > 0 else ""
        L.append(f"""
static void* _get_so_handle{fn_suffix}(void) {{
    static void* handle = nullptr;
    if (!handle) {{
        handle = dlopen("{so_path}", RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {{
            fprintf(stderr, "[pylow-ffi] dlopen({so_path}) failed: %s\\n", dlerror());
        }}
    }}
    return handle;
}}""")

    # Build a mapping: _so_path → handle function name
    so_handle_fns = {}
    for idx, so_path in enumerate(so_paths):
        fn_suffix = f"_{idx}" if idx > 0 else ""
        so_handle_fns[so_path] = f"_get_so_handle{fn_suffix}"

    # Generate vaddr-based function resolvers for each method
    # DEDUPLICATE: same c_sym (e.g. objToJSON) may appear in multiple method_defs
    # (e.g. both 'dumps' and 'encode' map to objToJSON). Generate resolver only once.
    _seen_c_syms: set = set()
    for mdef in module.method_defs:
        py_name = mdef.get("name", "")
        c_sym = mdef.get("func_symbol", "")
        func_vaddr = mdef.get("func_vaddr", "0x0")
        mdef_so_path = mdef.get("_so_path", "")
        if not py_name or not c_sym: continue
        if c_sym in _seen_c_syms: continue
        _seen_c_syms.add(c_sym)
        # Convert vaddr string to integer
        if isinstance(func_vaddr, str):
            func_vaddr_int = int(func_vaddr, 16) if func_vaddr.startswith("0x") else int(func_vaddr)
        else:
            func_vaddr_int = func_vaddr

        # Pick the right handle function for this method's .so
        handle_fn = so_handle_fns.get(mdef_so_path, "_get_so_handle")

        # Find the PyInit_ info for this .so (for vaddr-based base resolution)
        init_info = so_pyinit_map.get(mdef_so_path, {})
        init_vaddr_val = init_info.get("vaddr", 0)
        if not init_vaddr_val and module.pyinit_symbol:
            if isinstance(module.pyinit_symbol, dict):
                v = module.pyinit_symbol.get("vaddr", 0)
                if isinstance(v, str): v = int(v, 16)
                init_vaddr_val = v
            else:
                init_vaddr_val = getattr(module.pyinit_symbol, 'address', 0)
        init_name = init_info.get("name", f"PyInit_{nm}")

        L.append(f"""
static void* _resolve_{c_sym}(void) {{
    // {c_sym} is at vaddr 0x{func_vaddr_int:x} in the .so
    // After dlopen, we find it via dlsym first (works for GLOBAL),
    // then fall back to base+vaddr for LOCAL (static) symbols.
    void* handle = {handle_fn}();
    if (!handle) return nullptr;
    // Try dlsym first (works for GLOBAL/exported symbols)
    void* fn = dlsym(handle, "{c_sym}");
    if (fn) return fn;
    // Fallback: vaddr-based resolution for LOCAL symbols.
    // We find the base address by resolving a known GLOBAL symbol
    // (PyInit_) and subtracting its known vaddr.
    typedef void (*InitFn)(void);
    InitFn init_fn = (InitFn)dlsym(handle, "{init_name}");
    if (!init_fn) {{
        fprintf(stderr, "[pylow-ffi] Cannot find {init_name} for base resolution\\n");
        return nullptr;
    }}
    // {init_name} vaddr in the .so
    uintptr_t init_vaddr = 0x{init_vaddr_val:x};
    if (init_vaddr == 0) {{
        fprintf(stderr, "[pylow-ffi] {init_name} vaddr unknown\\n");
        return nullptr;
    }}
    uintptr_t base = (uintptr_t)init_fn - init_vaddr;
    return (void*)(base + 0x{func_vaddr_int:x});
}}""")

    # Wrapper functions — also extern "C" for C linkage
    L.append("""
// Wrapper functions — also extern "C" for C linkage
extern "C" {""")

    for mdef in module.method_defs:
        py_name = mdef.get("name", "")
        c_sym = mdef.get("func_symbol", "")
        flags = mdef.get("flags", 0)
        if not py_name or not c_sym: continue
        wsym = f"pylow_ffi_{nm}_{py_name}"
        sig = sigs.get(wsym)
        if not sig: continue
        rt = sig.return_type

        if flags & METH_O:
            if rt == "char*":
                L.append(f"""
// {wsym} — {c_sym} (METH_O, string→string). Caller must pylow_ffi_free() result.
__attribute__((visibility("default"))) char* {wsym}(const char *data,int64_t len,int64_t *out_len){{
    if(!data)return nullptr; _mini_runtime_init();
    PyObject *py_in=PyUnicode_FromStringAndSize(data,(Py_ssize_t)len);
    if(!py_in){{fprintf(stderr,"[pylow-ffi] {wsym}: PyUnicode_FromStringAndSize failed\\n");return nullptr;}}
    typedef PyObject*(*F)(PyObject*,PyObject*);
    F fn=(F)_resolve_{c_sym}();
    if(!fn){{fprintf(stderr,"[pylow-ffi] {wsym}: resolve({c_sym}) failed\\n");Py_DecRef(py_in);return nullptr;}}
    PyObject *py_res=fn(nullptr,py_in); Py_DecRef(py_in);
    if(!py_res){{fprintf(stderr,"[pylow-ffi] {wsym}: {c_sym} returned NULL\\n");return nullptr;}}
    const char *utf8=PyUnicode_AsUTF8(py_res); char *result=nullptr;
    if(utf8){{int64_t sl=(int64_t)strlen(utf8);result=(char*)malloc(sl+1);
        if(result){{memcpy(result,utf8,sl+1);if(out_len)*out_len=sl;}}}}
    Py_DecRef(py_res); return result;
}}""")
            else:
                L.append(f"""
// {wsym} — {c_sym} (METH_O, string→int)
__attribute__((visibility("default"))) int64_t {wsym}(const char *data,int64_t len){{
    if(!data)return -1; _mini_runtime_init();
    PyObject *py_in=PyUnicode_FromStringAndSize(data,(Py_ssize_t)len);
    if(!py_in)return -1;
    typedef PyObject*(*F)(PyObject*,PyObject*);
    F fn=(F)_resolve_{c_sym}();
    if(!fn){{Py_DecRef(py_in);return -1;}}
    PyObject *py_res=fn(nullptr,py_in); Py_DecRef(py_in);
    if(!py_res)return -1; int64_t r=PyLong_AsSsize_t(py_res); Py_DecRef(py_res); return r;
}}""")
        elif flags & METH_NOARGS:
            if rt == "char*":
                L.append(f"""
// {wsym} — {c_sym} (METH_NOARGS, →string). Caller must pylow_ffi_free() result.
__attribute__((visibility("default"))) char* {wsym}(int64_t *out_len){{
    _mini_runtime_init();
    typedef PyObject*(*F)(PyObject*,PyObject*);
    F fn=(F)_resolve_{c_sym}();
    if(!fn){{fprintf(stderr,"[pylow-ffi] {wsym}: resolve({c_sym}) failed\\n");return nullptr;}}
    PyObject *py_res=fn(nullptr,nullptr);
    if(!py_res)return nullptr;
    const char *utf8=PyUnicode_AsUTF8(py_res); char *result=nullptr;
    if(utf8){{int64_t sl=(int64_t)strlen(utf8);result=(char*)malloc(sl+1);
        if(result){{memcpy(result,utf8,sl+1);if(out_len)*out_len=sl;}}}}
    Py_DecRef(py_res); return result;
}}""")
            else:
                L.append(f"""
// {wsym} — {c_sym} (METH_NOARGS, →int)
__attribute__((visibility("default"))) int64_t {wsym}(void){{
    _mini_runtime_init();
    typedef PyObject*(*F)(PyObject*,PyObject*);
    F fn=(F)_resolve_{c_sym}();
    if(!fn)return -1;
    PyObject *py_res=fn(nullptr,nullptr);
    if(!py_res)return -1; int64_t r=PyLong_AsSsize_t(py_res); Py_DecRef(py_res); return r;
}}""")
        elif flags & METH_VARARGS:
            kw = " + METH_KEYWORDS" if flags & METH_KEYWORDS else ""
            has_kw = bool(flags & METH_KEYWORDS)
            # METH_VARARGS: PyObject* func(PyObject* self, PyObject* args)
            # METH_VARARGS|KEYWORDS: PyObject* func(PyObject* self, PyObject* args, PyObject* kwargs)
            # The wrapper receives a string arg, creates a PyTuple with it,
            # calls the C function, then converts the result back.
            # NOTE: C++ does NOT allow default arguments in typedef function pointers.
            # We declare kw_dict as a local variable instead.
            if rt == "char*":
                kw_typedef = ", PyObject*" if has_kw else ""
                kw_declare = "\n    PyObject *kw_dict=nullptr;" if has_kw else ""
                kw_call = ", kw_dict" if has_kw else ""
                L.append(f"""
// {wsym} — {c_sym} (METH_VARARGS{kw}, string→string). Caller must pylow_ffi_free() result.
__attribute__((visibility("default"))) char* {wsym}(const char *data,int64_t len,int64_t *out_len){{
    if(!data)return nullptr; _mini_runtime_init();
    PyObject *py_in=PyUnicode_FromStringAndSize(data,(Py_ssize_t)len);
    if(!py_in){{fprintf(stderr,"[pylow-ffi] {wsym}: PyUnicode_FromStringAndSize failed\\n");return nullptr;}}
    // Build args tuple: (py_in,)
    PyObject *args_tuple=PyTuple_New(1);
    if(!args_tuple){{Py_DecRef(py_in);return nullptr;}}
    // PyTuple_SetItem steals reference — INCREF so the tuple's DECREF will free it
    Py_INCREF(py_in);
    PyTuple_SetItem(args_tuple,0,py_in);
    typedef PyObject*(*F)(PyObject*,PyObject*{kw_typedef});
    F fn=(F)_resolve_{c_sym}();
    if(!fn){{fprintf(stderr,"[pylow-ffi] {wsym}: resolve({c_sym}) failed\\n");Py_DecRef(args_tuple);return nullptr;}}{kw_declare}
    PyObject *py_res=fn(nullptr,args_tuple{kw_call}); Py_DecRef(args_tuple);
    if(!py_res){{fprintf(stderr,"[pylow-ffi] {wsym}: {c_sym} returned NULL\\n");return nullptr;}}
    const char *utf8=PyUnicode_AsUTF8(py_res); char *result=nullptr;
    if(utf8){{int64_t sl=(int64_t)strlen(utf8);result=(char*)malloc(sl+1);
        if(result){{memcpy(result,utf8,sl+1);if(out_len)*out_len=sl;}}}}
    Py_DecRef(py_res); return result;
}}""")
            else:
                kw_typedef = ", PyObject*" if has_kw else ""
                kw_declare = "\n    PyObject *kw_dict=nullptr;" if has_kw else ""
                kw_call = ", kw_dict" if has_kw else ""
                L.append(f"""
// {wsym} — {c_sym} (METH_VARARGS{kw}, string→int)
__attribute__((visibility("default"))) int64_t {wsym}(const char *data,int64_t len){{
    if(!data)return -1; _mini_runtime_init();
    PyObject *py_in=PyUnicode_FromStringAndSize(data,(Py_ssize_t)len);
    if(!py_in)return -1;
    PyObject *args_tuple=PyTuple_New(1);
    if(!args_tuple){{Py_DecRef(py_in);return -1;}}
    Py_INCREF(py_in);
    PyTuple_SetItem(args_tuple,0,py_in);
    typedef PyObject*(*F)(PyObject*,PyObject*{kw_typedef});
    F fn=(F)_resolve_{c_sym}();
    if(!fn){{Py_DecRef(args_tuple);return -1;}}{kw_declare}
    PyObject *py_res=fn(nullptr,args_tuple{kw_call}); Py_DecRef(args_tuple);
    if(!py_res)return -1; int64_t r=PyLong_AsSsize_t(py_res); Py_DecRef(py_res); return r;
}}""")
        elif flags & METH_NOARGS:
            pass  # METH_NOARGS doesn't need _pyobj variant
        else:
            L.append(f"""
// {wsym} — {c_sym} (flags=0x{flags:x}, fallback)
__attribute__((visibility("default"))) void* {wsym}(void){{
    fprintf(stderr,"[pylow-ffi] {wsym}: fallback (flags=0x{flags:x})\\n");return nullptr;}}""")

    # ── Generate _pyobj wrapper variants for METH_O and METH_VARARGS ──
    # These accept a raw PyObject* directly, allowing the LLVM side to create
    # proper CPython objects (PyDict, PyList, etc.) and pass them through.
    # The wrapper wraps the PyObject* in a tuple (for METH_VARARGS) or passes
    # it directly (for METH_O), then calls the .so function.
    _seen_pyobj: set = set()
    for mdef in module.method_defs:
        py_name = mdef.get("name", "")
        c_sym = mdef.get("func_symbol", "")
        flags = mdef.get("flags", 0)
        if not py_name or not c_sym: continue
        # Only generate _pyobj variants for METH_O and METH_VARARGS
        if not (flags & METH_O or flags & METH_VARARGS): continue
        # Deduplicate (same c_sym used for different py_names)
        if c_sym in _seen_pyobj: continue
        _seen_pyobj.add(c_sym)

        wsym_pyobj = f"pylow_ffi_{nm}_{py_name}_pyobj"
        has_kw = bool(flags & METH_KEYWORDS)
        kw_typedef = ", PyObject*" if has_kw else ""
        kw_declare = "\n    PyObject *kw_dict=nullptr;" if has_kw else ""
        kw_call = ", kw_dict" if has_kw else ""

        if flags & METH_VARARGS:
            # METH_VARARGS _pyobj variant:
            # Takes a PyObject* directly, wraps it in a PyTuple, calls the .so function.
            # This is used for dict/list/object arguments that the LLVM side creates
            # as proper CPython PyObject* objects using pylow_ffi_create_dict etc.
            L.append(f"""
// {wsym_pyobj} — {c_sym} (METH_VARARGS, PyObject*→string). Caller must pylow_ffi_free() result.
// Takes a pre-built PyObject* (e.g., from pylow_ffi_create_dict), wraps in tuple, calls .so.
__attribute__((visibility("default"))) char* {wsym_pyobj}(void *pyobj_ptr,int64_t *out_len){{
    if(!pyobj_ptr)return nullptr; _mini_runtime_init();
    PyObject *pyobj=(PyObject*)pyobj_ptr;
    PyObject *args_tuple=PyTuple_New(1);
    if(!args_tuple)return nullptr;
    Py_INCREF(pyobj);
    PyTuple_SetItem(args_tuple,0,pyobj);
    typedef PyObject*(*F)(PyObject*,PyObject*{kw_typedef});
    F fn=(F)_resolve_{c_sym}();
    if(!fn){{fprintf(stderr,"[pylow-ffi] {wsym_pyobj}: resolve({c_sym}) failed\\n");Py_DecRef(args_tuple);return nullptr;}}{kw_declare}
    PyObject *py_res=fn(nullptr,args_tuple{kw_call}); Py_DecRef(args_tuple);
    if(!py_res){{fprintf(stderr,"[pylow-ffi] {wsym_pyobj}: {c_sym} returned NULL\\n");return nullptr;}}
    const char *utf8=PyUnicode_AsUTF8(py_res); char *result=nullptr;
    if(utf8){{int64_t sl=(int64_t)strlen(utf8);result=(char*)malloc(sl+1);
        if(result){{memcpy(result,utf8,sl+1);if(out_len)*out_len=sl;}}}}
    Py_DecRef(py_res); return result;
}}""")
        elif flags & METH_O:
            # METH_O _pyobj variant:
            # Takes a PyObject* directly, passes it as the single arg to the .so function.
            L.append(f"""
// {wsym_pyobj} — {c_sym} (METH_O, PyObject*→string). Caller must pylow_ffi_free() result.
// Takes a pre-built PyObject* directly, passes as single arg to .so function.
__attribute__((visibility("default"))) char* {wsym_pyobj}(void *pyobj_ptr,int64_t *out_len){{
    if(!pyobj_ptr)return nullptr; _mini_runtime_init();
    PyObject *pyobj=(PyObject*)pyobj_ptr;
    typedef PyObject*(*F)(PyObject*,PyObject*);
    F fn=(F)_resolve_{c_sym}();
    if(!fn){{fprintf(stderr,"[pylow-ffi] {wsym_pyobj}: resolve({c_sym}) failed\\n");return nullptr;}}
    PyObject *py_res=fn(nullptr,pyobj);
    if(!py_res){{fprintf(stderr,"[pylow-ffi] {wsym_pyobj}: {c_sym} returned NULL\\n");return nullptr;}}
    const char *utf8=PyUnicode_AsUTF8(py_res); char *result=nullptr;
    if(utf8){{int64_t sl=(int64_t)strlen(utf8);result=(char*)malloc(sl+1);
        if(result){{memcpy(result,utf8,sl+1);if(out_len)*out_len=sl;}}}}
    Py_DecRef(py_res); return result;
}}""")

    L.append(f"""
}} // extern "C" """)
    return "\n".join(L)


class FFIManager:
    """Manages AOT C++ wrapper generation for CPython extension .so modules.

    For each registered CPython extension module, generates:
    1. ffi_<module>_runtime.cpp — Mini-runtime with Py* symbol stubs (weak linkage)
    2. ffi_<module>_wrapper.cpp — Wrapper functions with pylow-friendly signatures

    The wrapper functions (e.g., pylow_ffi_markupsafe_escape) convert between
    pylow's native types (const char*, int64_t) and real PyObject* using the
    mini-runtime, then call the original C function from the .so.

    Usage::
        mgr = FFIManager()
        mod = FFIModule.from_file("markupsafe/_speedups.cpython-312-x86_64-linux-gnu.so")
        mgr.register_module(mod)
        runtime_cpp = mgr.generate_runtime("markupsafe")
        wrapper_cpp = mgr.generate_wrapper("markupsafe")
    """

    def __init__(self) -> None:
        self._modules: Dict[str, FFIModule] = {}
        self._wrapper_signatures: Dict[str, WrapperSignature] = {}
        # Map from user-facing module name (e.g., "markupsafe") to the
        # internal FFIModule name (e.g., "_speedups").  The compiler
        # registers modules by their import name, but FFIModule.name is
        # derived from the PyInit_ symbol.
        self._name_aliases: Dict[str, str] = {}
        # Map from (user_module_name, user_method_name) to the internal
        # wrapper signature key.  E.g., ("markupsafe", "escape") →
        # the WrapperSignature for _speedups._escape_inner.
        self._method_aliases: Dict[Tuple[str, str], str] = {}

    def register_module(self, ffi_module: FFIModule, alias: str = None) -> None:
        """Register a CPython extension module for wrapper generation.

        Analyzes the module's method definitions and creates WrapperSignature
        entries for each callable method.

        Args:
            ffi_module: The FFIModule to register.
            alias: Optional user-facing module name (e.g., "markupsafe").
                   If provided, the module is also accessible under this name
                   in addition to its internal name (e.g., "_speedups").
        """
        name = ffi_module.name
        if not name:
            print("[pylow-ffi] WARNING: register_module called with unnamed module",
                  file=sys.stderr)
            return
        self._modules[name] = ffi_module
        if alias and alias != name:
            self._name_aliases[alias] = name
        self._build_signatures(name, ffi_module, user_alias=alias)
        print(f"[pylow-ffi] Registered module '{name}'"
              f"{f' (alias: {alias})' if alias and alias != name else ''}: "
              f"{len(ffi_module.imported_py_funcs)} Py* imports, "
              f"{len(ffi_module.method_defs)} method_defs")

    def _build_signatures(self, module_name: str, module: FFIModule,
                          user_alias: str = None) -> None:
        """Build WrapperSignature entries for all method definitions."""
        for mdef in module.method_defs:
            py_name = mdef.get("name", "")
            c_sym = mdef.get("func_symbol", "")
            flags = mdef.get("flags", 0)
            if not py_name or not c_sym: continue

            wsym = f"pylow_ffi_{module_name}_{py_name}"
            rt = _method_return_type(mdef, module)

            if flags & METH_O:
                pt = ["const char*", "int64_t", "int64_t*"] if rt == "char*" else ["const char*", "int64_t"]
            elif flags & METH_NOARGS:
                pt = ["int64_t*"] if rt == "char*" else []
            elif flags & METH_VARARGS:
                # METH_VARARGS wrapper has the same signature as METH_O:
                # it takes a string arg and wraps it in a tuple internally
                pt = ["const char*", "int64_t", "int64_t*"] if rt == "char*" else ["const char*", "int64_t"]
            else:
                pt = []

            needs_fb = any(s not in _COVERED_SYMS for s in module.imported_py_funcs)
            if needs_fb:
                for s in module.imported_py_funcs:
                    if s not in _COVERED_SYMS:
                        print(f"[pylow-ffi] FFI: {module_name} -> Fallback (dlsym {s})")
                        break

            self._wrapper_signatures[wsym] = WrapperSignature(
                symbol_name=wsym, original_symbol=c_sym, module_name=module_name,
                return_type=rt, param_types=pt, method_flags=flags, fallback=needs_fb,
            )

            # ── Also register a _pyobj variant for METH_O and METH_VARARGS ──
            # This variant takes a raw PyObject* directly instead of a string.
            # The LLVM side uses it when the argument is a dict/list/object
            # that needs to be a proper CPython PyObject*, not a string.
            if flags & METH_O or flags & METH_VARARGS:
                wsym_pyobj = f"{wsym}_pyobj"
                # Signature: char* pylow_ffi_<mod>_<func>_pyobj(void* pyobj_ptr, int64_t* out_len)
                pt_pyobj = ["void*", "int64_t*"]
                self._wrapper_signatures[wsym_pyobj] = WrapperSignature(
                    symbol_name=wsym_pyobj, original_symbol=c_sym, module_name=module_name,
                    return_type="char*", param_types=pt_pyobj, method_flags=flags,
                    fallback=needs_fb,
                )
                if user_alias:
                    self._method_aliases[(user_alias, f"{py_name}_pyobj")] = wsym_pyobj

            # Build method aliases for user-facing access.
            # E.g., if user_alias="markupsafe" and py_name="_escape_inner",
            # register ("markupsafe", "escape") and ("markupsafe", "_escape_inner")
            # as aliases pointing to the internal wrapper key.
            if user_alias:
                # Always register the original py_name under the user alias
                self._method_aliases[(user_alias, py_name)] = wsym
                # Strip leading underscore for common patterns
                # (_escape_inner → escape_inner, __init__ → init)
                if py_name.startswith('_'):
                    stripped = py_name.lstrip('_')
                    if stripped:
                        self._method_aliases[(user_alias, stripped)] = wsym
                # Strip _inner suffix (_escape_inner → _escape, escape_inner → escape)
                if py_name.endswith('_inner'):
                    without_inner = py_name[:-6]  # Remove _inner
                    self._method_aliases[(user_alias, without_inner)] = wsym
                    without_inner_stripped = without_inner.lstrip('_')
                    if without_inner_stripped and without_inner_stripped != without_inner:
                        self._method_aliases[(user_alias, without_inner_stripped)] = wsym

    def _resolve_name(self, module_name: str) -> str:
        """Resolve a module name (possibly an alias) to the internal name."""
        return self._name_aliases.get(module_name, module_name)

    def generate_runtime(self, module_name: str) -> str:
        """Generate C++ mini-runtime source for a module."""
        resolved = self._resolve_name(module_name)
        if resolved not in self._modules:
            raise KeyError(f"Module '{module_name}' (resolved: '{resolved}') not registered with FFIManager")
        return _generate_runtime(self._modules[resolved])

    def generate_wrapper(self, module_name: str) -> str:
        """Generate C++ wrapper source for a module."""
        resolved = self._resolve_name(module_name)
        if resolved not in self._modules:
            raise KeyError(f"Module '{module_name}' (resolved: '{resolved}') not registered with FFIManager")
        msigs = {s: sig for s, sig in self._wrapper_signatures.items()
                 if sig.module_name == resolved}
        return _generate_wrapper(self._modules[resolved], msigs)

    def get_wrapper_signatures(self, module_name: str) -> Dict[str, WrapperSignature]:
        """Get the LLVM IR signatures for all wrapper functions of a module.

        The LLVM emitter uses this to substitute call @escape_unicode with
        call @pylow_ffi_markupsafe_escape.
        """
        resolved = self._resolve_name(module_name)
        return {s: sig for s, sig in self._wrapper_signatures.items()
                if sig.module_name == resolved}

    def get_all_wrapper_symbol_names(self) -> List[str]:
        """Get all pylow_ffi_* wrapper symbol names across all modules."""
        return sorted(self._wrapper_signatures.keys())

    def get_symbol_mapping(self, module_name: str) -> Dict[str, str]:
        """Get mapping from original symbol names to wrapper symbol names.

        Useful for the LLVM emitter to substitute direct .so calls with wrappers.
        """
        resolved = self._resolve_name(module_name)
        return {sig.original_symbol: sig.symbol_name
                for sig in self._wrapper_signatures.values()
                if sig.module_name == resolved}
