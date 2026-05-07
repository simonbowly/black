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
_UNHANDLED_KEYWORDS: frozenset[str] = frozenset()

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

        # ---- cdef / cpdef: variable or function header ----
        if tok.type == _tokenize.NAME and tok.string in ("cdef", "cpdef"):
            keyword = tok.string

            # Classify by scanning ahead for (, :, =, 'class', or NEWLINE.
            # Track [ ] depth so colons inside memoryview slices are ignored.
            open_paren_idx = -1
            decl_type = "variable"  # default for cdef; will be overridden for cpdef
            bracket_depth = 0
            j = i + 1
            while j < n:
                t = toks[j]
                if t.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER, _tokenize.COMMENT):
                    break
                if t.type == _tokenize.OP and t.string == "[":
                    bracket_depth += 1
                    j += 1
                    continue
                if t.type == _tokenize.OP and t.string == "]":
                    bracket_depth -= 1
                    j += 1
                    continue
                if bracket_depth == 0:
                    if t.type == _tokenize.OP and t.string == "(":
                        open_paren_idx = j
                        decl_type = "function"
                        break
                    if keyword == "cdef":
                        if t.type == _tokenize.OP and t.string == ":":
                            decl_type = "block_header"
                            break
                        if t.type == _tokenize.OP and t.string == ",":
                            decl_type = "multi_variable"
                            break
                        if t.type == _tokenize.OP and t.string == "=":
                            break  # variable with initializer
                        if t.type == _tokenize.NAME and t.string == "class":
                            decl_type = "cdef_class"
                            break
                j += 1

            # ---- function header ----
            if decl_type == "function" and open_paren_idx >= 0:
                # Collect return-type + function-name NAMEs (keyword → open paren).
                hdr_name_toks: list[str] = []
                last_hdr_idx = -1
                for k in range(i + 1, open_paren_idx):
                    if toks[k].type == _tokenize.NAME:
                        hdr_name_toks.append(toks[k].string)
                        last_hdr_idx = k

                if last_hdr_idx < 0:
                    i += 1  # no name before ( — leave unmasked
                    continue

                # Span 1: keyword … function-name → 'def __cy_...'
                s1_start = _abs_start(tok, offsets)
                s1_end = _abs_end(toks[last_hdr_idx], offsets)
                ph1 = _placeholder(*hdr_name_toks)
                edits.append(_Edit(s1_start, s1_end, f"def {ph1}", src[s1_start:s1_end]))

                # Typed-argument spans: scan between ( and )
                j = open_paren_idx + 1
                while j < n:
                    t = toks[j]
                    if t.type == _tokenize.OP and t.string == ")":
                        j += 1
                        break
                    if t.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                        break

                    # Collect one argument slot (up to , / ) / = at bracket depth 0)
                    arg_name_toks: list[str] = []
                    arg_first_idx = -1
                    arg_last_idx = -1
                    arg_brk_depth = 0
                    k = j
                    while k < n:
                        t2 = toks[k]
                        if t2.type == _tokenize.OP and t2.string == "[":
                            arg_brk_depth += 1
                            k += 1
                            continue
                        if t2.type == _tokenize.OP and t2.string == "]":
                            arg_brk_depth -= 1
                            k += 1
                            continue
                        if arg_brk_depth > 0:
                            k += 1
                            continue
                        if t2.type == _tokenize.OP and t2.string in (",", ")"):
                            break
                        if t2.type == _tokenize.OP and t2.string == "=":
                            break  # default value
                        if t2.type in (_tokenize.NL, _tokenize.COMMENT):
                            k += 1
                            continue
                        if t2.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                            break
                        if t2.type == _tokenize.NAME:
                            arg_name_toks.append(t2.string)
                            if arg_first_idx < 0:
                                arg_first_idx = k
                            arg_last_idx = k
                        k += 1

                    # Skip past any default value, then past the , to the next arg.
                    depth = 0
                    sk = k
                    while sk < n:
                        t3 = toks[sk]
                        if t3.type == _tokenize.OP and t3.string == "(":
                            depth += 1
                        elif t3.type == _tokenize.OP and t3.string == ")":
                            if depth == 0:
                                j = sk  # point at ) so outer loop breaks on it
                                break
                            depth -= 1
                        elif t3.type == _tokenize.OP and t3.string == "," and depth == 0:
                            j = sk + 1  # advance past ,
                            break
                        sk += 1
                    else:
                        j = sk

                    # Mask if typed (2+ NAMEs before =)
                    if len(arg_name_toks) >= 2 and arg_first_idx >= 0:
                        sa_start = _abs_start(toks[arg_first_idx], offsets)
                        sa_end = _abs_end(toks[arg_last_idx], offsets)
                        ph_arg = _placeholder(*arg_name_toks)
                        edits.append(_Edit(sa_start, sa_end, ph_arg, src[sa_start:sa_end]))

                # Advance i past ) and : to the start of the next logical line
                while j < n:
                    if toks[j].type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                        j += 1
                        break
                    j += 1
                i = j
                continue

            # ---- cdef class header ----
            if decl_type == "cdef_class" and keyword == "cdef":
                # j is at the 'class' keyword; find the class name (first NAME after it)
                class_name_idx = -1
                k = j + 1
                while k < n:
                    t = toks[k]
                    if t.type == _tokenize.NAME:
                        class_name_idx = k
                        break
                    k += 1

                if class_name_idx >= 0:
                    s_start = _abs_start(tok, offsets)
                    s_end = _abs_end(toks[class_name_idx], offsets)
                    ph = _placeholder("class", toks[class_name_idx].string)
                    edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                    i = class_name_idx + 1
                    continue
                i += 1
                continue

            # ---- variable declaration (cdef only) ----
            if decl_type == "variable" and keyword == "cdef":
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

                if last_name_idx >= 0 and len(name_toks) >= 1:
                    span_start = _abs_start(tok, offsets)
                    span_end = _abs_end(toks[last_name_idx], offsets)
                    ph = _placeholder(*name_toks)
                    edits.append(_Edit(span_start, span_end, ph, src[span_start:span_end]))
                    i = k
                    continue

            # Unhandled: block_header, multi_variable, cdef_class, cpdef non-function, etc.
            i += 1
            continue

        # ---- ctypedef TYPE ALIAS (simple alias only; struct/enum/fused deferred) ----
        if tok.type == _tokenize.NAME and tok.string == "ctypedef":
            name_toks: list[str] = []
            last_name_idx = -1
            saw_colon = False
            k = i + 1
            while k < n:
                t = toks[k]
                if t.type in (
                    _tokenize.NEWLINE,
                    _tokenize.ENDMARKER,
                    _tokenize.COMMENT,
                ):
                    break
                if t.type == _tokenize.OP and t.string in (":", "("):
                    saw_colon = True  # struct/enum/fptr block — leave unmasked
                    break
                if t.type == _tokenize.NAME:
                    name_toks.append(t.string)
                    last_name_idx = k
                k += 1

            if not saw_colon and last_name_idx >= 0 and len(name_toks) >= 2:
                span_start = _abs_start(tok, offsets)
                span_end = _abs_end(toks[last_name_idx], offsets)
                ph = _placeholder("ctypedef", *name_toks)
                edits.append(_Edit(span_start, span_end, ph, src[span_start:span_end]))
                i = k
                continue

            i += 1
            continue

        # ---- plain def with Cython-typed arguments ----
        if tok.type == _tokenize.NAME and tok.string == "def":
            # Find the opening paren of the parameter list.
            open_paren_idx = -1
            j = i + 1
            while j < n:
                t = toks[j]
                if t.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                    break
                if t.type == _tokenize.OP and t.string == "(":
                    open_paren_idx = j
                    break
                j += 1

            if open_paren_idx >= 0:
                # Scan arguments for typed spans (TYPE NAME pairs).
                # Break on ':' to avoid masking Python annotation-style args.
                j = open_paren_idx + 1
                while j < n:
                    t = toks[j]
                    if t.type == _tokenize.OP and t.string == ")":
                        break
                    if t.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                        break

                    def_arg_toks: list[str] = []
                    def_arg_first: int = -1
                    def_arg_last: int = -1
                    def_brk_depth = 0
                    k = j
                    while k < n:
                        t2 = toks[k]
                        if t2.type == _tokenize.OP and t2.string == "[":
                            def_brk_depth += 1
                            k += 1
                            continue
                        if t2.type == _tokenize.OP and t2.string == "]":
                            def_brk_depth -= 1
                            k += 1
                            continue
                        if def_brk_depth > 0:
                            k += 1
                            continue
                        if t2.type == _tokenize.OP and t2.string in (",", ")", "="):
                            break
                        if t2.type == _tokenize.OP and t2.string == ":":
                            break  # annotation separator (a: int), not memoryview slice
                        if t2.type in (_tokenize.NL, _tokenize.COMMENT):
                            k += 1
                            continue
                        if t2.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                            break
                        if t2.type == _tokenize.NAME:
                            def_arg_toks.append(t2.string)
                            if def_arg_first < 0:
                                def_arg_first = k
                            def_arg_last = k
                        k += 1

                    depth = 0
                    sk = k
                    while sk < n:
                        t3 = toks[sk]
                        if t3.type == _tokenize.OP and t3.string == "(":
                            depth += 1
                        elif t3.type == _tokenize.OP and t3.string == ")":
                            if depth == 0:
                                j = sk
                                break
                            depth -= 1
                        elif (
                            t3.type == _tokenize.OP
                            and t3.string == ","
                            and depth == 0
                        ):
                            j = sk + 1
                            break
                        sk += 1
                    else:
                        j = sk

                    if len(def_arg_toks) >= 2 and def_arg_first >= 0:
                        sa_start = _abs_start(toks[def_arg_first], offsets)
                        sa_end = _abs_end(toks[def_arg_last], offsets)
                        ph_arg = _placeholder(*def_arg_toks)
                        edits.append(
                            _Edit(sa_start, sa_end, ph_arg, src[sa_start:sa_end])
                        )

            i += 1
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
