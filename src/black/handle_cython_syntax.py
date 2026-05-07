"""Cython syntax masking and restoration.

The masker replaces Cython-only spans with __cy_* placeholder identifiers so
that Black's Python formatter can process the surrounding Python skeleton.
The restorer walks the __cy_* anchors in left-to-right order and substitutes
the original Cython text back in.

Placeholder contract (see extend-black-cython.md for full spec):
  - Every placeholder is a bare Python identifier prefixed __cy_.
  - masked_text always ends at the anchor token; the restorer checks
    result[anchor_end - len(masked_text) : anchor_end] == masked_text.
  - Restoration is positional, not by name lookup.
"""

from __future__ import annotations

import io
import re
import tokenize as _tokenize

# A substitution record: (masked_text, original_text).
# masked_text is the contiguous span Black saw; its __cy_* anchor is at the
# END of the string (both identifier-only and skeleton-plus-anchor shapes).
Replacement = tuple[str, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholder(*parts: str) -> str:
    """Build an identifier-safe __cy_* token from one or more text fragments."""
    combined = "_".join(p for p in parts if p)
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", combined).strip("_")
    return f"__cy_{sanitized}"


def _line_offsets(src: str) -> list[int]:
    """Return list where result[i] is the absolute char offset of line i+1."""
    result = [0]
    for line in src.splitlines(keepends=True):
        result.append(result[-1] + len(line))
    return result


def _abs_start(tok: _tokenize.TokenInfo, offsets: list[int]) -> int:
    row, col = tok.start
    return offsets[row - 1] + col


def _abs_end(tok: _tokenize.TokenInfo, offsets: list[int]) -> int:
    row, col = tok.end
    return offsets[row - 1] + col


def _tokenize_src(src: str) -> list[_tokenize.TokenInfo]:
    toks: list[_tokenize.TokenInfo] = []
    try:
        for tok in _tokenize.generate_tokens(io.StringIO(src).readline):
            toks.append(tok)
    except _tokenize.TokenError:
        pass  # Cython parse gate already verified the source; partial is OK
    return toks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Cython-specific NAME tokens not yet handled by the masker.
# Updated as each Phase 2 step adds a new construct.
_UNHANDLED_KEYWORDS: frozenset[str] = frozenset({"cpdef", "ctypedef"})

# Cython keywords that must not appear in the masked source after masking.
# Any remaining occurrence means the masker encountered an unhandled construct.
_POST_MASK_KEYWORDS: frozenset[str] = frozenset(
    {"cdef", "cpdef", "ctypedef", "cimport"}
)


def validate_cython_subset(src: str) -> None:
    """Raise if src contains Cython constructs the masker cannot handle yet,
    or any identifier starting with __cy_ (reserved namespace).
    """
    for tok in _tokenize_src(src):
        if tok.type != _tokenize.NAME:
            continue
        if tok.string.startswith("__cy_"):
            raise ValueError(
                f"Source contains reserved identifier {tok.string!r} at line "
                f"{tok.start[0]}; the '__cy_' prefix is reserved for the Cython "
                f"formatter"
            )
        if tok.string in _UNHANDLED_KEYWORDS:
            raise NotImplementedError(
                f"Cython construct {tok.string!r} at line {tok.start[0]} is not "
                f"yet handled by the masker"
            )


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class _Edit:
    __slots__ = ("start", "end", "masked_text", "original_text")

    def __init__(
        self, start: int, end: int, masked_text: str, original_text: str
    ) -> None:
        self.start = start
        self.end = end
        self.masked_text = masked_text
        self.original_text = original_text


def mask_cython(src: str) -> tuple[str, list[Replacement]]:
    """Replace Cython-only spans with __cy_* placeholders.

    Returns (masked_source, replacements) where replacements is an ordered
    list of (masked_text, original_text) pairs in left-to-right source order.
    """
    toks = _tokenize_src(src)
    offsets = _line_offsets(src)
    edits: list[_Edit] = []
    n = len(toks)
    i = 0

    while i < n:
        tok = toks[i]

        # ---- from X.Y cimport a, b, c ----
        if tok.type == _tokenize.NAME and tok.string == "from":
            cimport_idx = -1
            saw_import = False
            j = i + 1
            while j < n:
                t = toks[j]
                if t.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                    break
                if t.type == _tokenize.NAME and t.string == "import":
                    saw_import = True
                if (
                    not saw_import
                    and t.type == _tokenize.NAME
                    and t.string == "cimport"
                ):
                    cimport_idx = j
                    break
                j += 1

            if cimport_idx >= 0:
                cimport_tok = toks[cimport_idx]
                name_parts: list[str] = []
                last_tok = cimport_tok
                k = cimport_idx + 1
                while k < n:
                    t = toks[k]
                    if t.type in (
                        _tokenize.NEWLINE,
                        _tokenize.ENDMARKER,
                        _tokenize.COMMENT,
                    ):
                        break
                    if t.type == _tokenize.NAME:
                        name_parts.append(t.string)
                        last_tok = t
                    elif t.type == _tokenize.OP and t.string == "*":
                        name_parts.append("star")
                        last_tok = t
                    k += 1

                span_start = _abs_start(cimport_tok, offsets)
                span_end = _abs_end(last_tok, offsets)
                original_text = src[span_start:span_end]
                ph = _placeholder("cimport", *name_parts)
                edits.append(
                    _Edit(span_start, span_end, f"import {ph}", original_text)
                )
                i = k
                continue

        # ---- cdef TYPE VAR [= EXPR] variable declaration ----
        if tok.type == _tokenize.NAME and tok.string == "cdef":
            # Classify by scanning ahead for (, :, =, 'class', or NEWLINE.
            decl_type = "variable"
            j = i + 1
            while j < n:
                t = toks[j]
                if t.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER, _tokenize.COMMENT):
                    break
                if t.type == _tokenize.OP and t.string in ("(", ":"):
                    decl_type = "function_or_block"
                    break
                if t.type == _tokenize.OP and t.string == ",":
                    decl_type = "multi_variable"
                    break
                if t.type == _tokenize.OP and t.string == "=":
                    break  # = before ( or : → simple variable with initializer
                if t.type == _tokenize.NAME and t.string == "class":
                    decl_type = "cdef_class"
                    break
                j += 1

            if decl_type != "variable":
                # Not a simple variable declaration; leave unmasked.
                # The post-masking check will catch this as unhandled.
                i += 1
                continue

            # Collect all NAME tokens between cdef and = (or NEWLINE).
            name_toks: list[str] = []
            last_name_idx = -1
            k = i + 1
            while k < n:
                t = toks[k]
                if t.type in (
                    _tokenize.NEWLINE,
                    _tokenize.ENDMARKER,
                    _tokenize.COMMENT,
                ):
                    break
                if t.type == _tokenize.OP and t.string == "=":
                    break
                if t.type == _tokenize.NAME:
                    name_toks.append(t.string)
                    last_name_idx = k
                k += 1

            if last_name_idx < 0 or len(name_toks) < 1:
                # Degenerate cdef with no names; leave unmasked.
                i += 1
                continue

            # Span: from 'cdef' through the last NAME (= var name).
            # The initializer (= EXPR) and trailing comment stay literal.
            span_start = _abs_start(tok, offsets)
            span_end = _abs_end(toks[last_name_idx], offsets)
            original_text = src[span_start:span_end]
            ph = _placeholder(*name_toks)
            edits.append(_Edit(span_start, span_end, ph, original_text))
            i = k
            continue

        # ---- standalone cimport X.Y [as alias] ----
        if tok.type == _tokenize.NAME and tok.string == "cimport":
            j = i + 1
            module_parts: list[str] = []
            last_tok = tok
            while j < n:
                t = toks[j]
                if t.type == _tokenize.NAME:
                    module_parts.append(t.string)
                    last_tok = t
                    j += 1
                elif t.type == _tokenize.OP and t.string == ".":
                    j += 1
                else:
                    break
            span_start = _abs_start(tok, offsets)
            span_end = _abs_end(last_tok, offsets)
            original_text = src[span_start:span_end]
            ph = _placeholder(*module_parts) if module_parts else _placeholder("module")
            edits.append(_Edit(span_start, span_end, f"import {ph}", original_text))
            i = j
            continue

        i += 1

    # Apply edits right-to-left so leftward positions are unaffected
    result = src
    for edit in reversed(edits):
        result = result[: edit.start] + edit.masked_text + result[edit.end :]

    # Sanity check: no Cython-specific keyword should remain after masking.
    # Any hit means the masker encountered a construct it doesn't yet handle.
    # This check runs even when there were no edits, so unhandled source is caught.
    for t in _tokenize_src(result):
        if t.type == _tokenize.NAME and t.string in _POST_MASK_KEYWORDS:
            raise NotImplementedError(
                f"mask_cython: '{t.string}' at line {t.start[0]} was not masked — "
                f"construct not yet supported by the masker"
            )

    if not edits:
        return src, []

    replacements: list[Replacement] = [
        (e.masked_text, e.original_text) for e in edits
    ]
    return result, replacements


# ---------------------------------------------------------------------------
# Unmasking
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(r"__cy_\w+")


def unmask_cython(src: str, replacements: list[Replacement]) -> str:
    """Restore original Cython spans from the placeholder-substituted Black output.

    Walks __cy_* anchors left-to-right; for each, verifies the surrounding
    masked_text context, then replaces that span with original_text.
    Processes right-to-left so earlier positions are not shifted.
    """
    if not replacements:
        return src

    anchors = list(_ANCHOR_RE.finditer(src))

    if len(anchors) != len(replacements):
        raise AssertionError(
            f"unmask_cython: expected {len(replacements)} __cy_* anchor(s), "
            f"found {len(anchors)}"
        )

    result = src
    for anchor, (masked_text, original_text) in zip(
        reversed(anchors), reversed(replacements)
    ):
        anchor_end = anchor.end()
        masked_start = anchor_end - len(masked_text)
        if masked_start < 0 or result[masked_start:anchor_end] != masked_text:
            actual = result[max(0, masked_start) : anchor_end]
            raise AssertionError(
                f"unmask_cython: invariant violated at anchor {anchor.group()!r}: "
                f"expected {masked_text!r}, found {actual!r}"
            )
        result = result[:masked_start] + original_text + result[anchor_end:]

    return result
