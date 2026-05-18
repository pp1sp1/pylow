#!/usr/bin/env python3
"""Compatibility entry point for the py2llvm compiler.

Usage:
    python main.py <file.py>                          # Compile and print LLVM IR
    python main.py <file.py> -o out.ll                # Compile and save to file
    python main.py <file.py> --libs-mode static       # Static linking (default)
    python main.py <file.py> --libs-mode dynamic       # Dynamic linking
    python main.py <file.py> --dynamic-lib markupsafe  # Link specific lib dynamically
    python main.py <file.py> --static-lib math         # Force static link for math
    python main.py <file.py> --link-bin output         # Compile and link to binary
"""

import argparse
import sys

from src import PythonToLLVMCompiler, LibraryConfig, LinkMode


def main() -> None:
    """Main entry point for command-line compilation."""
    parser = argparse.ArgumentParser(
        description="py2llvm — Python to LLVM IR compiler"
    )
    parser.add_argument("source_file", help="Python source file to compile")
    parser.add_argument("-o", "--output", help="Output LLVM IR file")
    parser.add_argument(
        "--libs-mode",
        choices=["static", "dynamic"],
        default="static",
        help="Default library linking mode (default: static)",
    )
    parser.add_argument(
        "--dynamic-lib",
        action="append",
        default=[],
        help="Link a specific library dynamically (can be repeated)",
    )
    parser.add_argument(
        "--static-lib",
        action="append",
        default=[],
        help="Force static linking for a specific library (can be repeated)",
    )
    parser.add_argument(
        "--lib-path",
        action="append",
        default=[],
        help="Add a directory to the library search path",
    )
    parser.add_argument(
        "--link-bin",
        metavar="OUTPUT",
        help="Compile and link to an executable binary",
    )

    args = parser.parse_args()

    # Build library configuration
    config = LibraryConfig(
        libs_mode=LinkMode.from_string(args.libs_mode),
        dynamic_libs=set(args.dynamic_lib),
        static_libs=set(args.static_lib),
        search_paths=args.lib_path,
    )

    # Read source code
    with open(args.source_file, "r") as f:
        source = f.read()

    # Create compiler with library config
    compiler = PythonToLLVMCompiler(
        libs_mode=args.libs_mode,
        dynamic_libs=set(args.dynamic_lib),
        libs_config=config,
    )

    # Add library search paths
    for path in args.lib_path:
        compiler.add_lib_search_path(path)

    # Compile
    ir_text = compiler.compile(source)

    if args.link_bin:
        # Compile and link to binary
        try:
            binary_path = compiler.compile_and_link(source, args.link_bin)
            print(f"Binary written to {binary_path}", file=sys.stderr)
        except RuntimeError as e:
            print(f"Link error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.output:
        # Save LLVM IR
        with open(args.output, "w") as f:
            f.write(ir_text)
        print(f"LLVM IR written to {args.output}", file=sys.stderr)
    else:
        # Print LLVM IR
        print(ir_text)


if __name__ == "__main__":
    main()
