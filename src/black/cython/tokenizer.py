"""
Token pass: extract comments and layout information from source using Python's
tokenize module (stable public API, works correctly on .pyx/.pxd/.pxi files).
"""

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass, field


@dataclass
class Comment:
    line: int        # 1-indexed source line
    col: int         # 0-indexed column
    text: str        # "# ..." including the leading hash
    is_inline: bool  # True when non-whitespace code precedes the comment on the same line


@dataclass
class LayoutInfo:
    comments: list[Comment] = field(default_factory=list)  # sorted by line
    keyword_lines: dict[int, str] = field(default_factory=dict)  # line → 'else'|'finally'
    code_lines: frozenset[int] = field(default_factory=frozenset)  # lines with non-comment tokens


def extract_layout(source: str) -> LayoutInfo:
    """Extract all comments from Cython source using Python's tokenize module.

    Cython keywords (cdef, cpdef, nogil, …) are NAME tokens to the Python
    tokenizer, so this works on .pyx/.pxd/.pxi source without modification.
    """
    # lines_with_code[line] = True when a substantive token was seen on that line
    # before any COMMENT token on that line.
    lines_with_code: dict[int, bool] = {}
    comments: list[Comment] = []
    keyword_lines: dict[int, str] = {}

    _IGNORED = frozenset({
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.ENCODING,
    })

    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        ttype, tstring, tstart, _tend, _line = tok
        line, col = tstart

        if ttype == tokenize.COMMENT:
            is_inline = lines_with_code.get(line, False)
            comments.append(Comment(line=line, col=col, text=tstring, is_inline=is_inline))
        elif ttype == tokenize.NAME and tstring in ('else', 'finally'):
            keyword_lines[line] = tstring
        elif ttype not in _IGNORED:
            lines_with_code[line] = True

    comments.sort(key=lambda c: c.line)
    return LayoutInfo(
        comments=comments,
        keyword_lines=keyword_lines,
        code_lines=frozenset(lines_with_code),
    )
