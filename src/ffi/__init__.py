"""py2llvm FFI (Foreign Function Interface) subsystem.

Provides zero-overhead native .so library integration:
- ELF/PE binary analysis with LIEF
- CPython Py* API signature database (934+ symbols)
- Automatic LLVM IR external declarations for .so symbols
- Minimal C stub generation for Py* symbol resolution
- Direct builder.call() code generation without runtime wrappers
- AOT C++ wrapper generation with mini-runtime (Zero-Python Mode)
"""

from .core import (
    FFISymbol,
    FFIModule,
    FFISignatureDB,
    get_symbol_info,
    analyze_binary,
    analyze_package,
    generate_py_stubs,
    PYAPI_DB,
)

from .generator import (
    FFIManager,
    WrapperSignature,
)

__all__ = [
    "FFISymbol",
    "FFIModule",
    "FFISignatureDB",
    "get_symbol_info",
    "analyze_binary",
    "analyze_package",
    "generate_py_stubs",
    "PYAPI_DB",
    "FFIManager",
    "WrapperSignature",
]