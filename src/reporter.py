"""Modern error reporting system for the pylow compiler.

Inspired by Gleam and Elm compilers — clean layout, Unicode box-drawing,
breathable whitespace, and pastel ANSI colours.  All user-facing messages
are in English.

Architecture
------------
* ``ErrorLevel``   — severity (error / warning / ice)
* ``ErrorCategory``— phase of compilation that produced the error
* ``PylowDiagnostic`` — structured data for a single diagnostic
* ``ErrorReporter``   — renders one or more diagnostics to the terminal

Rendering contract
------------------
Every diagnostic is rendered as a self-contained block that follows this
layout (80-char wide):

 ── TYPE MISMATCH ─────────────────────────────────────────────────────── src/main.py:14

    An expression has a type that doesn't match the variable type definition.

14 │     total_power: int = "high"
   │                        ^^^^── This is a 'str'
   │
   └── Expected type: int

    HELP: Change the value to an integer literal or explicitly cast it
          using int(). Verified during pylow semantic analysis phase.

────────────────────────────────────────────────────────────────────────────────

Integration notes
-----------------
* ``CompileError`` (and its subclass ``PylowError``) carry a
  ``PylowDiagnostic`` instance on the ``.diagnostic`` attribute.
* The CLI entry-point (``pylow.py``) catches these exceptions and hands
  them to ``ErrorReporter.render()``.
* When ``--verbose`` is *not* active, only the pretty diagnostic is
  shown — the raw Python traceback is suppressed for user errors.
  ICEs always show the traceback.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────
#  ANSI helpers
# ──────────────────────────────────────────────────────────────────────

class _A:
    """Namespace for ANSI escape-code constants.

    Only the subset that works reliably on modern terminals (xterm-256,
    true-colour) is used.  No aggressive / dark colours — everything is
    *bright* or *pastel* for readability on both light and dark
    backgrounds.
    """
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"

    # Foreground
    RED         = "\033[31m"
    BRIGHT_RED  = "\033[91m"
    YELLOW      = "\033[33m"
    BRIGHT_YELLOW = "\033[93m"
    GREEN       = "\033[32m"
    BRIGHT_GREEN  = "\033[92m"
    BLUE        = "\033[34m"
    BRIGHT_BLUE = "\033[94m"
    CYAN        = "\033[36m"
    BRIGHT_CYAN = "\033[96m"
    WHITE       = "\033[37m"
    BRIGHT_WHITE  = "\033[97m"
    MAGENTA     = "\033[35m"
    BRIGHT_MAGENTA = "\033[95m"

    # Convenience composites used in the mockup
    ERROR_HEADER   = BOLD + BRIGHT_RED      # Bold Bright Red for error titles
    WARNING_HEADER = BOLD + BRIGHT_YELLOW   # Bold Bright Yellow for warning titles
    ICE_HEADER     = BOLD + BRIGHT_MAGENTA  # Bold Bright Magenta for ICE titles
    CODE_LINE      = BRIGHT_BLUE            # Bright Blue for │ line numbers etc.
    CODE_TEXT      = BRIGHT_WHITE           # Bright White for source text
    POINTER       = BRIGHT_RED              # Bright Red for ^^^^ pointers
    HELP_LABEL     = BOLD                   # Bold for "HELP:"
    HELP_TEXT      = BRIGHT_CYAN            # Bright Cyan for help body
    SUBTITLE       = BRIGHT_WHITE           # Bright White for subtitle text
    HINT_TEXT      = DIM + BRIGHT_WHITE     # Dimmed for └── hints


def _supports_color(stream: Optional[int] = None) -> bool:
    """Return True if the terminal likely supports ANSI colours.

    Decision order:
    1. ``NO_COLOR`` env-var → forced off
    2. ``FORCE_COLOR`` / ``PYCO_COLOR`` / ``COLORTERM`` env-var → forced on
    3. ``TERM=dumb`` → forced off
    4. ``sys.stderr.isatty()`` → on if TTY
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
        if sys.stderr.isatty():
            return True
    except Exception:
        pass
    return False


# ──────────────────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────────────────

class ErrorLevel(Enum):
    """Severity of the diagnostic."""
    ERROR   = "error"
    WARNING = "warning"
    ICE     = "ice"       # Internal Compiler Error


class ErrorCategory(Enum):
    """Phase of compilation that produced the diagnostic.

    Maps directly to the three integration points requested:
    1. LEXER / PARSER   → syntax errors from ``ast.parse()``
    2. SEMANTIC         → type / name errors from visitors
    3. INTERNAL         → ICE: LLVM IR generation failures
    """
    SYNTAX   = "SYNTAX ERROR"
    SEMANTIC = "SEMANTIC ERROR"
    TYPE     = "TYPE MISMATCH"
    NAME     = "UNDEFINED NAME"
    UNSUPPORTED = "UNSUPPORTED FEATURE"
    FFI      = "FFI ERROR"
    INTERNAL = "INTERNAL COMPILER ERROR"


# ──────────────────────────────────────────────────────────────────────
#  Diagnostic data
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PylowDiagnostic:
    """Structured representation of a single compiler diagnostic.

    Attributes:
        category:   The error category (determines header text).
        level:      Severity — error, warning, or ICE.
        message:    Primary one-line explanation shown under the header.
        source_file: Path to the source file (shown top-right).
        line:       1-based line number in the source file.
        col:        0-based column offset in the source line.
        end_col:    0-based end column (exclusive).  Used to compute the
                    number of ``^`` characters.  If *None*, ``length`` is
                    used instead.
        length:     Number of characters to underline.  Ignored when
                    ``end_col`` is set.
        source_line: The actual text of the offending line (pre-fetched
                    from the file so we don't re-read inside render).
        hint:       Secondary hint text shown after the pointer line
                    (e.g. ``└── Expected type: int``).
        help_text:  Multi-line help / suggestion shown in the HELP block.
    """

    category: ErrorCategory
    level: ErrorLevel = ErrorLevel.ERROR
    message: str = ""
    source_file: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    end_col: Optional[int] = None
    length: Optional[int] = None
    source_line: Optional[str] = None
    hint: Optional[str] = None
    help_text: Optional[str] = None

    # ── derived helpers ────────────────────────────────────────────

    @property
    def underline_len(self) -> int:
        """How many ``^`` to draw."""
        if self.end_col is not None and self.col is not None:
            return max(1, self.end_col - self.col)
        if self.length is not None:
            return max(1, self.length)
        return 1

    @property
    def col_offset(self) -> int:
        """0-based column where the underline starts."""
        return self.col if self.col is not None else 0

    @property
    def header_title(self) -> str:
        """Text displayed in the top banner, e.g. ``TYPE MISMATCH``."""
        return self.category.value


# ──────────────────────────────────────────────────────────────────────
#  Reporter
# ──────────────────────────────────────────────────────────────────────

_TERMINAL_WIDTH = 80


class ErrorReporter:
    """Renders ``PylowDiagnostic`` objects to the terminal.

    Usage::

        reporter = ErrorReporter(use_color=True)
        reporter.render(diagnostic, file=sys.stderr)
    """

    def __init__(self, use_color: Optional[bool] = None) -> None:
        if use_color is None:
            use_color = _supports_color()
        self._color: bool = use_color

    # ── low-level helpers ──────────────────────────────────────────

    def _c(self, code: str, text: str) -> str:
        """Wrap *text* in *code* only when colour is enabled."""
        if self._color:
            return f"{code}{text}{_A.RESET}"
        return text

    @staticmethod
    def _rpad(text: str, width: int, fill: str = " ") -> str:
        """Right-pad *text* to *width* using *fill*."""
        if len(text) >= width:
            return text
        return text + fill * (width - len(text))

    # ── public API ─────────────────────────────────────────────────

    def render(self, diag: PylowDiagnostic, file=None) -> None:
        """Render a single diagnostic to *file* (default: stderr)."""
        if file is None:
            file = sys.stderr
        output = self.format(diag)
        print(output, file=file, end="")

    def render_many(self, diags: Sequence[PylowDiagnostic], file=None) -> None:
        """Render multiple diagnostics, separated by blank lines."""
        if file is None:
            file = sys.stderr
        for i, d in enumerate(diags):
            if i > 0:
                print(file=file)
            self.render(d, file=file)

    def format(self, diag: PylowDiagnostic) -> str:
        """Return the fully formatted diagnostic as a string."""
        parts: List[str] = []
        self._fmt_header(parts, diag)
        self._fmt_subtitle(parts, diag)
        self._fmt_code_block(parts, diag)
        self._fmt_help(parts, diag)
        self._fmt_footer(parts, diag)
        return "".join(parts) + "\n"

    # ── header ─────────────────────────────────────────────────────

    def _fmt_header(self, parts: List[str], diag: PylowDiagnostic) -> None:
        """ ── TYPE MISMATCH ──────────────────────────────── src/main.py:14 """
        title = diag.header_title
        # Choose colour by level
        if diag.level == ErrorLevel.WARNING:
            header_style = _A.WARNING_HEADER
        elif diag.level == ErrorLevel.ICE:
            header_style = _A.ICE_HEADER
        else:
            header_style = _A.ERROR_HEADER

        coloured_title = self._c(header_style, f"── {title} ──")

        # Right-side location
        loc_parts: List[str] = []
        if diag.source_file:
            loc_parts.append(diag.source_file)
        if diag.line is not None:
            loc_parts.append(str(diag.line))
        loc_str = ":".join(loc_parts)

        # Measure visible (non-ANSI) width of title
        vis_title = f"── {title} ──"
        vis_loc = f" {loc_str}" if loc_str else ""
        # Minimum gap between title and location
        min_gap = 1
        target_width = _TERMINAL_WIDTH

        # "── TITLE ──" + gap + loc_str
        # We fill the gap with ──
        content_width = len(vis_title) + min_gap + len(vis_loc)
        if content_width > target_width:
            # Too wide — just print without filler
            filler = ""
        else:
            fill_len = target_width - len(vis_title) - len(vis_loc)
            filler = "─" * fill_len

        coloured_loc = self._c(header_style, loc_str) if loc_str else ""
        coloured_filler = self._c(header_style, filler)

        parts.append(coloured_title)
        parts.append(coloured_filler)
        if coloured_loc:
            parts.append(" ")
            parts.append(coloured_loc)
        parts.append("\n")

    # ── subtitle ───────────────────────────────────────────────────

    def _fmt_subtitle(self, parts: List[str], diag: PylowDiagnostic) -> None:
        """
            An expression has a type that doesn't match ...
        """
        if not diag.message:
            parts.append("\n")
            return
        indent = "    "
        msg = self._c(_A.SUBTITLE, diag.message)
        parts.append(f"\n{indent}{msg}\n")

    # ── code block ─────────────────────────────────────────────────

    def _fmt_code_block(self, parts: List[str], diag: PylowDiagnostic) -> None:
        """
        14 │     total_power: int = "high"
           │                        ^^^^── This is a 'str'
           │
           └── Expected type: int
        """
        if diag.line is None and diag.source_line is None:
            return

        parts.append("\n")

        line_no = diag.line or 1
        source = diag.source_line or ""

        # ── primary source line ──
        no_str = str(line_no)
        no_coloured = self._c(_A.CODE_LINE, no_str)
        gutter = self._c(_A.CODE_LINE, " │ ")
        # Indent the gutter to align with the line number width
        gutter_pad = " " * len(no_str)

        parts.append(f"{no_coloured}{gutter}{self._c(_A.CODE_TEXT, source.rstrip())}\n")

        # ── pointer line ──
        col_off = diag.col_offset
        ulen = diag.underline_len

        # Leading spaces to align with the ^^^^
        leading = " " * col_off
        carets = "^" * ulen

        ptr_line = (
            self._c(_A.CODE_LINE, f"{gutter_pad} │ ")
            + leading
            + self._c(_A.POINTER, carets)
        )

        # └── hint after carets
        if diag.hint:
            ptr_line += self._c(_A.POINTER, "── ")
            ptr_line += self._c(_A.POINTER, diag.hint)

        parts.append(ptr_line + "\n")

        # Empty gutter line for visual separation before └──
        parts.append(self._c(_A.CODE_LINE, f"{gutter_pad} │\n"))

        # └── secondary hint line — show the category if hint was
        # already shown inline, or show the hint if it wasn't inline
        if diag.hint:
            # Hint already shown on the ^^^^ line, show category as └──
            secondary = diag.category.value
        else:
            secondary = diag.category.value

        parts.append(
            self._c(_A.CODE_LINE, f"{gutter_pad} ")
            + self._c(_A.CODE_LINE, "└── ")
            + self._c(_A.HINT_TEXT, secondary)
            + "\n"
        )

    # ── help block ─────────────────────────────────────────────────

    def _fmt_help(self, parts: List[str], diag: PylowDiagnostic) -> None:
        """
            HELP: Change the value to an integer literal ...
                  Verified during pylow semantic analysis phase.
        """
        if not diag.help_text:
            return

        indent = "    "
        label = self._c(_A.HELP_LABEL, "HELP:")
        # Wrap help text — first line after label, subsequent lines aligned
        lines = diag.help_text.strip().splitlines()
        if not lines:
            return

        parts.append(f"\n{indent}{label} {self._c(_A.HELP_TEXT, lines[0])}\n")
        continuation_pad = indent + "      "  # align with text after "HELP: "
        for ln in lines[1:]:
            parts.append(f"{continuation_pad}{self._c(_A.HELP_TEXT, ln)}\n")

    # ── footer ─────────────────────────────────────────────────────

    def _fmt_footer(self, parts: List[str], diag: PylowDiagnostic) -> None:
        """──────────────────────────────────────────────────────────────"""
        line = "─" * _TERMINAL_WIDTH
        # Match footer colour to error level
        if diag.level == ErrorLevel.WARNING:
            style = _A.WARNING_HEADER
        elif diag.level == ErrorLevel.ICE:
            style = _A.ICE_HEADER
        else:
            style = _A.ERROR_HEADER
        parts.append(f"\n{self._c(style, line)}\n")


# ──────────────────────────────────────────────────────────────────────
#  Convenience: build a diagnostic from various exception types
# ──────────────────────────────────────────────────────────────────────

def diagnostic_from_syntax_error(
    exc: SyntaxError,
    source: str,
    source_file: Optional[str] = None,
) -> PylowDiagnostic:
    """Convert a Python ``SyntaxError`` into a ``PylowDiagnostic``."""
    source_lines = source.splitlines()
    line_no = exc.lineno or 1
    col = (exc.offset or 1) - 1  # Python uses 1-based offset
    end_col = (exc.end_offset or (exc.offset or 1)) - 1 if exc.end_offset else None

    src_line = ""
    if 0 < line_no <= len(source_lines):
        src_line = source_lines[line_no - 1]

    return PylowDiagnostic(
        category=ErrorCategory.SYNTAX,
        level=ErrorLevel.ERROR,
        message=str(exc.msg) if exc.msg else "Invalid syntax",
        source_file=source_file or exc.filename,
        line=line_no,
        col=col,
        end_col=end_col,
        source_line=src_line,
        help_text="Fix the syntax error and try compiling again.",
    )


def diagnostic_from_compile_error(
    exc: "CompileError",  # type: ignore[name-defined]  # noqa: F821
    source: str,
    source_file: Optional[str] = None,
) -> PylowDiagnostic:
    """Convert a ``CompileError`` (or ``PylowError``) into a ``PylowDiagnostic``.

    If the exception already has a ``.diagnostic`` attribute (i.e. it is a
    ``PylowError``), that diagnostic is returned (with source-line patched
    in if missing).  Otherwise a best-effort diagnostic is built from the
    exception message and the AST node it carries.
    """
    # If the exception already carries a structured diagnostic, use it
    existing = getattr(exc, "diagnostic", None)
    if existing is not None:
        diag = existing
        # Patch source_line if not already set
        if diag.source_line is None and diag.line is not None and source:
            source_lines = source.splitlines()
            if 0 < diag.line <= len(source_lines):
                diag.source_line = source_lines[diag.line - 1]
        if diag.source_file is None and source_file:
            diag.source_file = source_file
        return diag

    # ── Fallback: extract info from the plain CompileError ──────────
    msg = str(exc)
    # Strip the "CompileError (linia N): " prefix
    if msg.startswith("CompileError"):
        # Format: "CompileError (linia N): actual message"
        # or:     "CompileError: actual message"
        colon_idx = msg.find(": ")
        if colon_idx != -1:
            msg = msg[colon_idx + 2:]

    node = getattr(exc, "node", None)
    line_no = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)

    src_line = None
    source_lines = source.splitlines() if source else []
    if line_no and 0 < line_no <= len(source_lines):
        src_line = source_lines[line_no - 1]

    # Infer category from message content (best-effort)
    category = ErrorCategory.SEMANTIC
    msg_lower = msg.lower()
    if "niezdefiniowana zmienna" in msg_lower or "undefined" in msg_lower:
        category = ErrorCategory.NAME
    elif "typ" in msg_lower or "type" in msg_lower or "konwertować" in msg_lower or "convert" in msg_lower:
        category = ErrorCategory.TYPE
    elif "nieobsługiw" in msg_lower or "unsupported" in msg_lower:
        category = ErrorCategory.UNSUPPORTED

    return PylowDiagnostic(
        category=category,
        level=ErrorLevel.ERROR,
        message=msg,
        source_file=source_file,
        line=line_no,
        col=col,
        end_col=end_col,
        source_line=src_line,
        help_text=None,
    )


def diagnostic_from_exception(
    exc: Exception,
    source: str = "",
    source_file: Optional[str] = None,
) -> PylowDiagnostic:
    """Convert any unexpected exception into an ICE diagnostic."""
    import traceback
    tb = traceback.format_exc()
    return PylowDiagnostic(
        category=ErrorCategory.INTERNAL,
        level=ErrorLevel.ICE,
        message=f"Unexpected error: {type(exc).__name__}: {exc}",
        source_file=source_file,
        source_line=None,
        help_text=(
            "This is an internal compiler error (ICE). Please report it\n"
            "with the full traceback and the source file that triggered it.\n\n"
            f"{tb}"
        ),
    )
