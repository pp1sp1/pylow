"""py2llvm - Python to LLVM IR compiler package.

A compiler that translates a subset of Python 3 into LLVM IR,
with support for dynamic typing via boxed values, ARC memory
management with cycle collection, a comprehensive runtime,
and zero-overhead FFI for native .so libraries.

Library Management (v4.2+):
  The compiler now includes a unified library management subsystem
  that handles both pure-Python modules and native .so libraries,
  with user-selectable static or dynamic linking per library.
"""

from .compiler import PythonToLLVMCompiler
from .types import Tag, PyType, EXCEPTION_TYPES
from .exceptions import CompileError, PylowError
from .reporter import (
    ErrorLevel,
    ErrorCategory,
    PylowDiagnostic,
    ErrorReporter,
    diagnostic_from_syntax_error,
    diagnostic_from_compile_error,
    diagnostic_from_exception,
)
from .values import Value, FFIModuleValue
from .symbols import VarInfo, SymbolTable
from .type_analyzer import StaticTypeAnalyzer
from .ffi import FFIModule, FFISignatureDB, PYAPI_DB
from .libs import (
    LinkMode,
    PurePythonLinkStrategy,
    LibraryConfig,
    LibraryKind,
    LibraryEntry,
    LibraryRegistry,
    PurePythonModuleInfo,
    PurePythonHandler,
    LinkAction,
    LinkStep,
    LinkPlan,
    LinkManager,
)

__version__ = "4.2.0"

__all__ = [
    "PythonToLLVMCompiler",
    "Tag",
    "PyType",
    "EXCEPTION_TYPES",
    "CompileError",
    "PylowError",
    # Error reporting
    "ErrorLevel",
    "ErrorCategory",
    "PylowDiagnostic",
    "ErrorReporter",
    "diagnostic_from_syntax_error",
    "diagnostic_from_compile_error",
    "diagnostic_from_exception",
    "Value",
    "FFIModuleValue",
    "VarInfo",
    "SymbolTable",
    "StaticTypeAnalyzer",
    "FFIModule",
    "FFISignatureDB",
    "PYAPI_DB",
    # Library management
    "LinkMode",
    "PurePythonLinkStrategy",
    "LibraryConfig",
    "LibraryKind",
    "LibraryEntry",
    "LibraryRegistry",
    "PurePythonModuleInfo",
    "PurePythonHandler",
    "LinkAction",
    "LinkStep",
    "LinkPlan",
    "LinkManager",
]
