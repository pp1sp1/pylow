#!/usr/bin/env python3
"""
pyco - Python do LLVM IR Kompilator
Użycie: pyco <plik.py> [opcje]
"""

import sys
import os

build_dir = "build"
# Add venv to path if available
venv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv",
    "lib",
    "python3.12",
    "site-packages",
)
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

import subprocess
import sysconfig
from pathlib import Path


# ──────────────────────────────────────────────────────────────
#  ANSI color helpers for CLI output
# ──────────────────────────────────────────────────────────────

class _C:
    """ANSI escape codes for terminal coloring."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    GREEN       = "\033[32m"
    BRIGHT_GREEN = "\033[92m"
    RED         = "\033[31m"
    BRIGHT_RED  = "\033[91m"
    YELLOW      = "\033[33m"
    BRIGHT_YELLOW = "\033[93m"
    BLUE        = "\033[34m"
    BRIGHT_BLUE = "\033[94m"
    CYAN        = "\033[36m"
    BRIGHT_CYAN = "\033[96m"
    WHITE       = "\033[37m"
    MAGENTA     = "\033[35m"

    # Composites
    SUCCESS = BOLD + BRIGHT_GREEN     # Bold bright green for success
    ERROR   = BOLD + BRIGHT_RED       # Bold bright red for errors
    WARN    = BOLD + BRIGHT_YELLOW    # Bold bright yellow for warnings
    FILE    = BRIGHT_BLUE             # Bright blue for file paths
    INFO    = BRIGHT_CYAN             # Bright cyan for info messages
    DIM_ERR = DIM + RED               # Dimmed red for error details


def _supports_color():
    """Check if stderr supports ANSI colors.

    Decision order:
    1. NO_COLOR env-var → forced off
    2. FORCE_COLOR / PYCO_COLOR / COLORTERM env-var → forced on
    3. TERM=dumb → forced off
    4. stderr.isatty() → on if TTY
    5. Otherwise → off
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if os.environ.get("PYCO_COLOR") is not None:
        return True
    if os.environ.get("COLORTERM") is not None:
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def _c(code, text):
    """Wrap text in ANSI color code if colors are supported."""
    if _supports_color():
        return f"{code}{text}{_C.RESET}"
    return text


def _tag():
    """Color the [pyco] tag."""
    return _c(_C.BOLD + _C.BRIGHT_CYAN, "[pyco]")


def _file(path):
    """Color a file path in blue."""
    return _c(_C.FILE, str(path))


def _success(msg):
    """Print a success message (green)."""
    print(f"{_tag()} {_c(_C.SUCCESS, msg)}", file=sys.stderr)


def _error(msg):
    """Print an error message (red)."""
    print(f"{_tag()} {_c(_C.ERROR, msg)}", file=sys.stderr)


def _warn(msg):
    """Print a warning message (yellow)."""
    print(f"{_tag()} {_c(_C.WARN, msg)}", file=sys.stderr)


def _info(msg):
    """Print an info message (cyan)."""
    print(f"{_tag()} {_c(_C.INFO, msg)}", file=sys.stderr)


def parse_args():
    """Parse command line arguments."""
    args = {
        "input_file": None,
        "run": False,
        "save_ir": False,
        "libs_mode": "static",  # 'static' or 'dynamic'
        "libs": set(),  # set of library names
        "verbose": False,  # show Python tracebacks
    }

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--run":
            args["run"] = True
        elif arg == "--save-ir":
            args["save_ir"] = True
        elif arg == "--dynamic":
            args["libs_mode"] = "dynamic"
        elif arg == "--static":
            args["libs_mode"] = "static"
        elif arg == "--verbose" or arg == "-v":
            args["verbose"] = True
        elif arg.startswith("--libs="):
            # --libs=math,os, sys - specify which libraries to use dynamically
            libs_str = arg[7:]  # Remove --libs=
            args["libs"] = set(libs_str.split(","))
        elif arg.startswith("-"):
            # Unknown option
            _error(f"Unknown option: {_c(_C.DIM_ERR, arg)}")
            sys.exit(1)
        else:
            # Must be input file
            if args["input_file"] is None:
                args["input_file"] = arg
            else:
                _error(f"Unexpected argument: {_c(_C.DIM_ERR, arg)}")
                sys.exit(1)
        i += 1

    return args


def _handle_compilation_error(exc, source, source_file, verbose):
    """Render a compilation error using the modern ErrorReporter.

    For user errors (CompileError, PylowError) the Python traceback is
    suppressed unless ``--verbose`` is active.  For internal compiler
    errors (ICE / unexpected exceptions) the traceback is always shown.
    """
    from src.reporter import (
        ErrorReporter,
        diagnostic_from_compile_error,
        diagnostic_from_syntax_error,
        diagnostic_from_exception,
    )
    from src.exceptions import CompileError, PylowError

    reporter = ErrorReporter()

    if isinstance(exc, SyntaxError):
        # Syntax errors from ast.parse() — always user errors
        diag = diagnostic_from_syntax_error(exc, source, source_file)
        reporter.render(diag)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if isinstance(exc, PylowError):
        # Structured error with full diagnostic info
        diag = exc.diagnostic
        # Patch source line if missing
        if diag.source_line is None and diag.line is not None and source:
            source_lines = source.splitlines()
            if 0 < diag.line <= len(source_lines):
                diag.source_line = source_lines[diag.line - 1]
        if diag.source_file is None and source_file:
            diag.source_file = source_file
        reporter.render(diag)
        if verbose or diag.level.value == "ice":
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if isinstance(exc, CompileError):
        # Legacy CompileError — best-effort diagnostic
        diag = diagnostic_from_compile_error(exc, source, source_file)
        reporter.render(diag)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Unexpected exception → ICE
    diag = diagnostic_from_exception(exc, source, source_file)
    reporter.render(diag)
    # Always show full traceback for ICE
    import traceback
    traceback.print_exc()
    sys.exit(1)


def _compile_ffi_stubs(compiler, build_dir: str) -> list:
    """Compile FFI C stubs to object files and return the .o paths.

    For each registered FFI module that needs CPython Py* stubs,
    writes the C source to build/pylow_ffi_stubs_<module>.c and
    compiles it with clang to a .o file.

    Returns:
        List of .o file paths that were successfully compiled.
    """
    stub_sources = compiler.get_ffi_stub_sources()
    if not stub_sources:
        return []

    obj_files = []
    for mod_name, c_source in stub_sources.items():
        c_path = os.path.join(build_dir, f"pylow_ffi_stubs_{mod_name}.c")
        o_path = os.path.join(build_dir, f"pylow_ffi_stubs_{mod_name}.o")

        # Write C source
        with open(c_path, "w") as f:
            f.write(c_source)

        # Compile stub C -> .o  (pure C, no C++ flags)
        try:
            result = subprocess.run(
                ["clang", "-c", "-O2", "-fPIC", c_path, "-o", o_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                _error(f"FFI stubs compile error ({_c(_C.FILE, mod_name)}): {_c(_C.DIM_ERR, result.stderr.strip())}")
                # Retry without optimizations in case O2 causes issues
                result2 = subprocess.run(
                    ["clang", "-c", "-fPIC", c_path, "-o", o_path],
                    capture_output=True, text=True,
                )
                if result2.returncode != 0:
                    _error(f"FFI stubs compile error (C-only): {_c(_C.DIM_ERR, result2.stderr.strip())}")
                    continue
            obj_files.append(o_path)
            _success(f"FFI stubs compiled: {_c(_C.FILE, mod_name)}")
        except FileNotFoundError:
            _error("clang not available for stub compilation")

    return obj_files


def _find_libpython() -> tuple:
    """Find the CPython shared library on the system.

    Returns:
        Tuple of (lib_dir, lib_name) where lib_dir is the directory containing
        libpython and lib_name is the link flag (e.g., "python3.12").
        Returns (None, None) if not found.
    """
    # Strategy 1: Use sysconfig to find the library
    libdir = sysconfig.get_config_var('LIBDIR')
    ldlibrary = sysconfig.get_config_var('LDLIBRARY')

    if libdir and ldlibrary:
        # ldlibrary is like "libpython3.12.so.1.0" or "libpython3.12d.so"
        # Extract the link name: remove "lib" prefix and ".so*" suffix
        if ldlibrary.startswith('lib') and '.so' in ldlibrary:
            link_name = ldlibrary[3:ldlibrary.index('.so')]
            # Verify the library actually exists
            lib_path = os.path.join(libdir, ldlibrary)
            if os.path.exists(lib_path) or os.path.exists(lib_path + '.1.0'):
                return libdir, link_name

    # Strategy 2: Search common paths
    for search_dir in [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
        '/usr/lib',
        '/lib/x86_64-linux-gnu',
        '/lib64',
    ]:
        if not os.path.isdir(search_dir):
            continue
        for fname in os.listdir(search_dir):
            if fname.startswith('libpython3.') and fname.endswith('.so'):
                link_name = fname[3:fname.index('.so')]
                return search_dir, link_name
            # Also check versioned .so files
            if fname.startswith('libpython3.') and '.so.' in fname:
                link_name = fname[3:fname.index('.so')]
                # Check if the unversioned symlink exists
                if os.path.exists(os.path.join(search_dir, link_name + '.so')):
                    return search_dir, link_name

    # Strategy 3: Check the venv's lib directory
    for py_ver in ['python3.12', 'python3.13', 'python3.11']:
        venv_lib = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '.venv', 'lib', py_ver
        )
        if not os.path.isdir(venv_lib):
            continue
        # Look for libpython in the venv prefix
        venv_prefix = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '.venv'
        )
        # Check if the host python has a shared library
        venv_libdir = os.path.join(venv_prefix, 'lib')
        if os.path.isdir(venv_libdir):
            for root, dirs, files in os.walk(venv_libdir):
                for f in files:
                    if f.startswith('libpython3.') and '.so' in f:
                        link_name = f[3:f.index('.so')]
                        return root, link_name

    return None, None


def _compile_cpython_bridge(build_dir: str) -> tuple:
    """Generate and compile the CPython bridge C file.

    DEPRECATED: This function is kept for fallback but is no longer the
    primary approach.  The AOT C++ wrapper system (FFIManager) is preferred.
    """
    from src.ffi.core import generate_cpython_bridge

    bridge_c = generate_cpython_bridge()
    c_path = os.path.join(build_dir, "pylow_ffi_cpython_bridge.c")
    o_path = os.path.join(build_dir, "pylow_ffi_cpython_bridge.o")

    with open(c_path, "w") as f:
        f.write(bridge_c)

    # Find Python include path for compilation
    include_dir = sysconfig.get_path('include')
    include_flags = [f"-I{include_dir}"] if include_dir and os.path.isdir(include_dir) else []

    try:
        result = subprocess.run(
            ["clang", "-c", "-O2", "-fPIC"] + include_flags + [c_path, "-o", o_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            _error(f"CPython bridge compile error: {_c(_C.DIM_ERR, result.stderr.strip())}")
            result2 = subprocess.run(
                ["clang", "-c", "-O2", "-fPIC", c_path, "-o", o_path],
                capture_output=True, text=True,
            )
            if result2.returncode != 0:
                _error(f"CPython bridge compile error (retry): {_c(_C.DIM_ERR, result2.stderr.strip())}")
                return None, False
        _success("CPython bridge compiled (fallback)")
        return o_path, True
    except FileNotFoundError:
        _error("clang not available")
        return None, False


def _compile_ffi_wrappers(compiler, build_dir: str) -> list:
    """Generate and compile AOT C++ wrappers for CPython extension modules.

    Uses FFIManager to generate:
    1. ffi_<module>_runtime.cpp — Mini-runtime with Py* symbol stubs (weak linkage)
    2. ffi_<module>_wrapper.cpp — Wrapper functions with pylow-friendly signatures

    Then compiles both with g++ -O3 -fPIC -c and returns .o file paths.

    This is the Zero-Python Mode approach: no libpython needed at link time.
    The mini-runtime provides just enough Py* symbols for the .so to work.

    Returns:
        List of .o file paths that were successfully compiled.
    """
    from src.ffi.generator import FFIManager

    _ffi_modules = getattr(compiler, '_ffi_modules', {})
    ffi_manager = FFIManager()

    # Register all CPython extension modules with user-facing name as alias
    for mod_name, ffi_mod in _ffi_modules.items():
        if ffi_mod.pyinit_symbol is not None:  # CPython extension
            ffi_manager.register_module(ffi_mod, alias=mod_name)

    if not ffi_manager._modules:
        return []

    obj_files = []
    for mod_name in ffi_manager._modules:
        # Generate runtime source
        try:
            runtime_cpp = ffi_manager.generate_runtime(mod_name)
        except Exception as e:
            _error(f"Runtime generation error ({_c(_C.FILE, mod_name)}): {_c(_C.DIM_ERR, str(e))}")
            continue

        # Generate wrapper source
        try:
            wrapper_cpp = ffi_manager.generate_wrapper(mod_name)
        except Exception as e:
            _error(f"Wrapper generation error ({_c(_C.FILE, mod_name)}): {_c(_C.DIM_ERR, str(e))}")
            continue

        # Write source files
        runtime_path = os.path.join(build_dir, f"ffi_{mod_name}_runtime.cpp")
        wrapper_path = os.path.join(build_dir, f"ffi_{mod_name}_wrapper.cpp")

        with open(runtime_path, "w") as f:
            f.write(runtime_cpp)
        with open(wrapper_path, "w") as f:
            f.write(wrapper_cpp)

        # Check if any wrapper symbols need fallback (dlsym to libpython)
        sigs = ffi_manager.get_wrapper_signatures(mod_name)
        has_fallback = any(s.fallback for s in sigs.values())

        if has_fallback:
            _warn(f"FFI: {_c(_C.FILE, mod_name)} -> Fallback (dlsym to libpython)")
        else:
            _info(f"FFI: {_c(_C.FILE, mod_name)} -> Wrapper (Zero-Python Mode)")

        # Compile with g++ -O3 -fPIC
        runtime_o = os.path.join(build_dir, f"ffi_{mod_name}_runtime.o")
        wrapper_o = os.path.join(build_dir, f"ffi_{mod_name}_wrapper.o")

        try:
            # Compile runtime
            result = subprocess.run(
                ["g++", "-O3", "-fPIC", "-c", runtime_path, "-o", runtime_o],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                _error(f"g++ runtime error ({_c(_C.FILE, mod_name)}): {_c(_C.DIM_ERR, result.stderr.strip())}")
                # Retry with -O0
                result2 = subprocess.run(
                    ["g++", "-fPIC", "-c", runtime_path, "-o", runtime_o],
                    capture_output=True, text=True,
                )
                if result2.returncode != 0:
                    _error(f"g++ runtime error ({_c(_C.FILE, mod_name)}) no -O: {_c(_C.DIM_ERR, result2.stderr.strip())}")
                    continue
            obj_files.append(runtime_o)

            # Compile wrapper (includes runtime via #include)
            result = subprocess.run(
                ["g++", "-O3", "-fPIC", "-c", wrapper_path, "-o", wrapper_o],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                _error(f"g++ wrapper error ({_c(_C.FILE, mod_name)}): {_c(_C.DIM_ERR, result.stderr.strip())}")
                result2 = subprocess.run(
                    ["g++", "-fPIC", "-c", wrapper_path, "-o", wrapper_o],
                    capture_output=True, text=True,
                )
                if result2.returncode != 0:
                    _error(f"g++ wrapper error ({_c(_C.FILE, mod_name)}) no -O: {_c(_C.DIM_ERR, result2.stderr.strip())}")
                    # Remove runtime .o since wrapper failed
                    if runtime_o in obj_files:
                        obj_files.remove(runtime_o)
                    continue
            obj_files.append(wrapper_o)

            _success(f"FFI wrappers compiled: {_c(_C.FILE, mod_name)} ({len(sigs)} functions)")

        except FileNotFoundError:
            _error("g++ not available for wrapper compilation")
            # Fallback: try clang++ as alternative
            try:
                result = subprocess.run(
                    ["clang++", "-O3", "-fPIC", "-c", runtime_path, "-o", runtime_o],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    _error(f"clang++ runtime error ({_c(_C.FILE, mod_name)}): {_c(_C.DIM_ERR, result.stderr.strip())}")
                    continue
                obj_files.append(runtime_o)

                result = subprocess.run(
                    ["clang++", "-O3", "-fPIC", "-c", wrapper_path, "-o", wrapper_o],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    _error(f"clang++ wrapper error ({_c(_C.FILE, mod_name)}): {_c(_C.DIM_ERR, result.stderr.strip())}")
                    continue
                obj_files.append(wrapper_o)
                _success(f"FFI wrappers compiled (clang++): {_c(_C.FILE, mod_name)}")
            except FileNotFoundError:
                _error("Neither g++ nor clang++ available")

    return obj_files


def _build_ffi_link_flags(compiler) -> tuple:
    """Build linker flags for FFI .so libraries.

    Returns:
        Tuple of (extra_link_flags, rpath_dirs) where:
        - extra_link_flags: list of flags like -L<dir>, -l:<filename>, -rdynamic, -ldl
        - rpath_dirs: set of directories for -Wl,-rpath
    """
    so_paths = compiler.get_ffi_so_paths()
    if not so_paths:
        return [], set()

    link_flags = []
    rpath_dirs = set()

    # Always add -rdynamic so that symbols from stubs are visible to .so
    link_flags.append("-rdynamic")

    # Always add -ldl for dlopen/dlsym
    link_flags.append("-ldl")

    # Expand directories into actual .so file paths
    actual_so_files = []
    for so_path in so_paths:
        so_abs = os.path.abspath(so_path)
        if os.path.isdir(so_abs):
            # Package directory — find all .so files inside
            for root, dirs, files in os.walk(so_abs):
                for f in files:
                    if f.endswith('.so'):
                        actual_so_files.append(os.path.join(root, f))
        elif os.path.isfile(so_abs):
            actual_so_files.append(so_abs)
        # If path doesn't exist, skip it silently

    # For each .so, add -L<dir> and -l:<filename.so>
    seen_dirs = set()
    for so_abs in actual_so_files:
        so_dir = os.path.dirname(so_abs)
        so_filename = os.path.basename(so_abs)

        # Add library search path
        if so_dir not in seen_dirs:
            link_flags.append(f"-L{so_dir}")
            seen_dirs.add(so_dir)

        # Add the .so as a library to link (using -l: which accepts full filenames)
        link_flags.append(f"-l:{so_filename}")

        # Track rpath directories for runtime resolution
        rpath_dirs.add(so_dir)

    return link_flags, rpath_dirs


def main():
    args = parse_args()

    if args["input_file"] is None:
        print(f"{_c(_C.BOLD, 'Usage:')} pyco {_c(_C.FILE, '<file.py>')} [options]")
        print(f"  {_c(_C.BRIGHT_GREEN, '--run')}         - run the compiled program")
        print(f"  {_c(_C.BRIGHT_GREEN, '--save-ir')}     - save LLVM IR to .ll file")
        print(f"  {_c(_C.BRIGHT_GREEN, '--static')}      - static libraries (default)")
        print(f"  {_c(_C.BRIGHT_GREEN, '--dynamic')}     - dynamic libraries")
        print(f"  {_c(_C.BRIGHT_GREEN, '--libs=m,os')}   - select libraries: math, os, sys, time, random")
        print(f"  {_c(_C.BRIGHT_GREEN, '--verbose,-v')}  - show full Python tracebacks")
        sys.exit(1)

    input_file = args["input_file"]

    # Check if file exists
    if not os.path.exists(input_file):
        _error(f"File {_file(input_file)} does not exist")
        sys.exit(1)

    # Read source code
    try:
        with open(input_file, "r") as f:
            source = f.read()
    except IOError as e:
        _error(f"Error reading file: {_c(_C.DIM_ERR, str(e))}")
        sys.exit(1)

    # Import compiler
    try:
        from src.main import PythonToLLVMCompiler
    except ImportError:
        _error("Cannot import compiler module")
        sys.exit(1)

    # Compile
    _info(f"Compiling {_file(input_file)}...")
    _info(f"Library mode: {_c(_C.WHITE, args['libs_mode'])}")
    _info(f"Dynamic libraries: {_c(_C.WHITE, str(args['libs']) if args['libs'] else 'none')}")

    try:
        compiler = PythonToLLVMCompiler(
            module_name=Path(input_file).stem,
            libs_mode=args["libs_mode"],
            dynamic_libs=args["libs"],
        )
        ir_text = compiler.compile(source, source_file=input_file)
        _success("Compilation finished successfully")
    except Exception as e:
        _handle_compilation_error(e, source, input_file, args["verbose"])

    # Save IR
    ll_file = os.path.join(build_dir, Path(Path(input_file).stem).with_suffix(".ll"))

    with open(ll_file, "w") as f:
        f.write(ir_text)
    _success(f"IR saved: {_file(ll_file)}")

    # Compile to executable
    output_file = os.path.join(build_dir, Path(input_file).stem)
    _info(f"Compiling to {_file(output_file)}...")

    # Check if clang or gcc is available
    cc_bin = None
    for cc in ["clang", "gcc", "cc"]:
        try:
            subprocess.run([cc, "--version"], capture_output=True, check=True)
            cc_bin = cc
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    cc_available = cc_bin is not None
    if not cc_available:
        _warn("No C compiler (clang/gcc) found — skipping native compilation")

    if cc_available:
        # Build linker flags based on libs
        libs_to_link = []

        # Default static libraries
        if args["libs_mode"] == "static":
            libs_to_link = ["-lm", "-lrt"]  # math + clock_gettime

        # Add dynamic library linking if specified
        if "math" in args["libs"]:
            libs_to_link.append("-lm")
        if "time" in args["libs"]:
            libs_to_link.append("-lrt")

        # ──────────────────────────────────────────────────────
        #  FFI: Compile stubs and build link flags for .so
        # ──────────────────────────────────────────────────────

        # Check if any CPython extension modules were detected
        from src.ffi.core import has_cpython_extensions
        _ffi_modules = getattr(compiler, '_ffi_modules', {})
        use_cpython_wrappers = has_cpython_extensions(_ffi_modules)

        ffi_obj_files = []
        ffi_link_flags, ffi_rpath_dirs = _build_ffi_link_flags(compiler)

        if use_cpython_wrappers:
            # ── CPython extension path: use AOT C++ wrappers (Zero-Python Mode) ──
            _info("CPython extension module detected — generating AOT C++ wrappers")

            # Generate and compile C++ wrappers with mini-runtime
            wrapper_obj_files = _compile_ffi_wrappers(compiler, build_dir)
            ffi_obj_files.extend(wrapper_obj_files)

            # Check if any wrappers need fallback to libpython
            # If so, find and link against libpython for dlsym-based Py* resolution
            from src.ffi.generator import FFIManager
            _ffi_manager = FFIManager()
            for mod_name, ffi_mod in _ffi_modules.items():
                if ffi_mod.pyinit_symbol is not None:
                    _ffi_manager.register_module(ffi_mod, alias=mod_name)

            needs_libpython = False
            for mod_name in _ffi_manager._modules:
                sigs = _ffi_manager.get_wrapper_signatures(mod_name)
                if any(s.fallback for s in sigs.values()):
                    needs_libpython = True
                    break

            if needs_libpython:
                py_lib_dir, py_lib_name = _find_libpython()
                if py_lib_dir and py_lib_name:
                    ffi_link_flags.append(f"-L{py_lib_dir}")
                    ffi_link_flags.append(f"-l{py_lib_name}")
                    ffi_rpath_dirs.add(py_lib_dir)
                    _info(f"Fallback: linking with libpython: {_file(f'-L{py_lib_dir} -l{py_lib_name}')}" )
                else:
                    _warn("libpython not found — fallback wrappers may not work")

            # The mini-runtime provides Py* symbols, so we DON'T link libpython
            # unless fallback mode is needed.  The .so's Py* imports are resolved
            # from the mini-runtime's __attribute__((weak)) symbols.
        else:
            # ── Non-CPython FFI path: use stubs as before ──
            ffi_obj_files.extend(_compile_ffi_stubs(compiler, build_dir))

        # ──────────────────────────────────────────────────────
        #  Async runtime: compile C scheduler if async functions exist
        # ──────────────────────────────────────────────────────
        async_obj_files = []
        has_async = getattr(compiler, '_async_functions', set())
        if has_async:
            from src.mixins.async_runtime import AsyncRuntimeMixin
            async_c_source = AsyncRuntimeMixin.get_async_runtime_c()
            async_c_path = os.path.join(build_dir, "pylow_async_runtime.c")
            async_o_path = os.path.join(build_dir, "pylow_async_runtime.o")

            with open(async_c_path, "w") as f:
                f.write(async_c_source)

            try:
                result = subprocess.run(
                    ["clang", "-c", "-O2", "-fPIC", async_c_path, "-o", async_o_path],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    # Retry without optimizations
                    result2 = subprocess.run(
                        ["clang", "-c", "-fPIC", async_c_path, "-o", async_o_path],
                        capture_output=True, text=True,
                    )
                    if result2.returncode != 0:
                        _error(f"Async runtime compile error: {_c(_C.DIM_ERR, result2.stderr.strip())}")
                    else:
                        async_obj_files.append(async_o_path)
                else:
                    async_obj_files.append(async_o_path)
                    _success("Async runtime compiled")
            except FileNotFoundError:
                # Try gcc as fallback
                try:
                    result = subprocess.run(
                        ["gcc", "-c", "-O2", "-fPIC", async_c_path, "-o", async_o_path],
                        capture_output=True, text=True,
                    )
                    if result.returncode != 0:
                        _error(f"gcc async runtime error: {_c(_C.DIM_ERR, result.stderr.strip())}")
                    else:
                        async_obj_files.append(async_o_path)
                        _success("Async runtime compiled (gcc)")
                except FileNotFoundError:
                    _error("Neither clang nor gcc available for async runtime compilation")

        # Build the input files list: LLVM IR + stub object files
        # .so files are NOT passed as input — they are linked via -L/-l flags
        if cc_bin == "clang":
            # clang can consume LLVM IR directly via "-x ir"
            # IMPORTANT: "-x none" resets the language after "-x ir" so that
            # .o files are recognised as object files, not parsed as IR source.
            input_files = ["-x", "ir", str(ll_file), "-x", "none"]
        else:
            # gcc cannot consume LLVM IR — use llvmlite to compile IR → .o first
            import llvmlite.binding as _llvm
            # llvmlite >= 0.44 handles init automatically
            try:
                _llvm.initialize()
                _llvm.initialize_all_targets()
                _llvm.initialize_all_asmprinters()
            except RuntimeError:
                pass  # Already initialized or auto-initialized
            with open(ll_file) as _f:
                _ir_text = _f.read()
            _mod = _llvm.parse_assembly(_ir_text)
            _mod.verify()
            _target = _llvm.Target.from_default_triple()
            _tm = _target.create_target_machine(opt=3, reloc="pic")
            _obj_path = os.path.join(build_dir, Path(input_file).stem + ".o")
            with open(_obj_path, "wb") as _f:
                _f.write(_tm.emit_object(_mod))
            input_files = [_obj_path]
        for o_file in ffi_obj_files:
            input_files.append(o_file)
        for o_file in async_obj_files:
            input_files.append(o_file)

        # Build rpath flags for runtime .so resolution
        rpath_flags = []
        for rpath_dir in sorted(ffi_rpath_dirs):
            rpath_flags.append(f"-Wl,-rpath,{rpath_dir}")

        # Final clang command
        cmd = (
            [cc_bin, "-O3"]
            + input_files
            + ["-o", output_file]
            + libs_to_link
            + ffi_link_flags
            + rpath_flags
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                _error(f"{_c(_C.BOLD, cc_bin)} error: {_c(_C.DIM_ERR, result.stderr.strip())}")
                # Retry without LTO if it failed
                if "-flto" in cmd:
                    cmd_no_lto = [a for a in cmd if a != "-flto"]
                    result2 = subprocess.run(cmd_no_lto, capture_output=True, text=True)
                    if result2.returncode != 0:
                        _error(f"{_c(_C.BOLD, cc_bin)} error (without LTO): {_c(_C.DIM_ERR, result2.stderr.strip())}")
                        sys.exit(1)
                    else:
                        _success(f"Executable (without LTO): {_file('./' + output_file)}")
                else:
                    sys.exit(1)
            else:
                _success(f"Executable: {_file('./' + output_file)}")
        except Exception as e:
            _error(f"Compilation error: {_c(_C.DIM_ERR, str(e))}")
            sys.exit(1)

    # Run if requested
    if args["run"] and cc_available:
        _info(f"Running {_file('./' + output_file)}...")
        print(file=sys.stderr)  # blank line before program output
        os.system(f"./{output_file}")


if __name__ == "__main__":
    main()
