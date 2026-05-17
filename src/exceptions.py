"""Custom exception classes for the pylow compiler.

Hierarchy::

    Exception
      └─ CompileError          — legacy, still used by ~80 raise sites
           └─ PylowError       — structured error carrying a PylowDiagnostic

``PylowError`` is a subclass of ``CompileError`` so that existing
``except CompileError: pass`` patterns continue to work without
modification.

Migration path
--------------
* Old code: ``raise CompileError("msg", node)``
* New code: ``raise PylowError.from_node(category, "msg", node, help_text=...)``

Both are caught by ``except CompileError``.  The difference is that
``PylowError`` carries structured data (category, column, token length,
help text) that the ``ErrorReporter`` can render beautifully.
"""

from __future__ import annotations

from typing import Optional

from .reporter import ErrorCategory, ErrorLevel, PylowDiagnostic


class CompileError(Exception):
    """Exception raised when a compilation error is encountered.

    Automatically appends the source line number (if available)
    to the error message for easier debugging.

    Args:
        msg: Human-readable error description.
        node: Optional AST node providing line number context.
    """

    def __init__(self, msg: str, node: object = None) -> None:
        loc = f" (linia {node.lineno})" if node and hasattr(node, "lineno") else ""
        super().__init__(f"CompileError{loc}: {msg}")
        # Store for potential later extraction by the reporter
        self.node = node
        self.raw_msg = msg


class PylowError(CompileError):
    """Structured compilation error that carries a ``PylowDiagnostic``.

    The ``.diagnostic`` attribute holds all the structured information
    needed by ``ErrorReporter`` to render a beautiful, Gleam-style error
    message.

    Because ``PylowError`` extends ``CompileError``, existing
    ``except CompileError`` handlers will catch it transparently.

    Args:
        diagnostic: A fully populated ``PylowDiagnostic``.
    """

    def __init__(self, diagnostic: PylowDiagnostic) -> None:
        self.diagnostic: PylowDiagnostic = diagnostic
        # Build a backward-compatible message for CompileError.__init__
        loc = f" (linia {diagnostic.line})" if diagnostic.line else ""
        super().__init__(diagnostic.message, node=None)
        # Override the CompileError-formatted message with our own
        self.args = (f"{diagnostic.category.value}{loc}: {diagnostic.message}",)

    # ── factory methods ────────────────────────────────────────────

    @classmethod
    def from_node(
        cls,
        category: ErrorCategory,
        message: str,
        node: object = None,
        *,
        source_file: Optional[str] = None,
        hint: Optional[str] = None,
        help_text: Optional[str] = None,
        level: ErrorLevel = ErrorLevel.ERROR,
        source: Optional[str] = None,
    ) -> "PylowError":
        """Create a ``PylowError`` from an AST node.

        This is the recommended way to raise structured errors inside
        visitor methods::

            raise PylowError.from_node(
                ErrorCategory.TYPE,
                f"Cannot assign '{value_type}' to variable of type '{target_type}'",
                node,
                hint=f"Expected type: {target_type}",
                help_text="Change the value or add an explicit type cast.",
            )

        Args:
            category:   Error category for the header banner.
            message:    Primary one-line explanation.
            node:       AST node providing line/column info.
            source_file: Path to the source file.
            hint:       Secondary hint shown after the pointer.
            help_text:  Multi-line help/suggestion text.
            level:      Severity (default: ERROR).
            source:     Full source text (used to extract the source line).
        """
        line = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)

        # Auto-compute token length for names/constants
        length = None
        if end_col is None and col is not None:
            name = getattr(node, "id", None) or getattr(node, "attr", None)
            if name:
                length = len(name)
            elif hasattr(node, "s") and isinstance(node.s, str):
                length = len(repr(node.s))
            elif hasattr(node, "value") and isinstance(getattr(node, "value", None), (int, float)):
                length = len(str(node.value))

        # Extract source line
        source_line = None
        if line and source:
            source_lines = source.splitlines()
            if 0 < line <= len(source_lines):
                source_line = source_lines[line - 1]

        diag = PylowDiagnostic(
            category=category,
            level=level,
            message=message,
            source_file=source_file,
            line=line,
            col=col,
            end_col=end_col,
            length=length,
            source_line=source_line,
            hint=hint,
            help_text=help_text,
        )
        return cls(diag)

    @classmethod
    def syntax_error(
        cls,
        message: str,
        source_file: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
        end_col: Optional[int] = None,
        source: Optional[str] = None,
        help_text: Optional[str] = None,
    ) -> "PylowError":
        """Create a ``PylowError`` for syntax errors (no AST node available)."""
        source_line = None
        if line and source:
            source_lines = source.splitlines()
            if 0 < line <= len(source_lines):
                source_line = source_lines[line - 1]

        diag = PylowDiagnostic(
            category=ErrorCategory.SYNTAX,
            level=ErrorLevel.ERROR,
            message=message,
            source_file=source_file,
            line=line,
            col=col,
            end_col=end_col,
            source_line=source_line,
            help_text=help_text or "Fix the syntax error and try compiling again.",
        )
        return cls(diag)

    @classmethod
    def ice(
        cls,
        message: str,
        source_file: Optional[str] = None,
        help_text: Optional[str] = None,
    ) -> "PylowError":
        """Create an Internal Compiler Error (ICE).

        ICEs always use ``ErrorLevel.ICE`` and ``ErrorCategory.INTERNAL``.
        The Python traceback is shown alongside the pretty diagnostic.
        """
        import traceback
        tb = traceback.format_exc()
        diag = PylowDiagnostic(
            category=ErrorCategory.INTERNAL,
            level=ErrorLevel.ICE,
            message=message,
            source_file=source_file,
            help_text=help_text or (
                "This is an internal compiler error (ICE).  Please report it\n"
                "with the full traceback and the source file that triggered it.\n\n"
                f"{tb}"
            ),
        )
        return cls(diag)
