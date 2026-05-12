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
import keyword as _keyword
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


def _prev_significant(
    toks: list[_tokenize.TokenInfo], i: int
) -> _tokenize.TokenInfo | None:
    """Return the last non-whitespace token before index i, or None.

    NEWLINE is intentionally NOT skipped: it marks the start of a new
    logical line, and _is_unary_position needs to see it.
    """
    for k in range(i - 1, -1, -1):
        t = toks[k]
        if t.type not in (
            _tokenize.NL,
            _tokenize.INDENT,
            _tokenize.DEDENT,
            _tokenize.COMMENT,
            _tokenize.ENCODING,
        ):
            return t
    return None


def _is_unary_position(prev_sig: _tokenize.TokenInfo | None) -> bool:
    """Return True if a unary prefix operator (C cast <type> or &) is valid here."""
    if prev_sig is None:
        return True
    t = prev_sig
    # Start of a new logical line: always unary context.
    if t.type == _tokenize.NEWLINE:
        return True
    # After a closing bracket: 'x[0]<int>' is comparison, 'x()&y' is bitwise-and
    if t.type == _tokenize.OP and t.string in (")", "]", "}"):
        return False
    # After a literal: 'x<int>' is comparison, '1&y' is bitwise-and
    if t.type in (_tokenize.NUMBER, _tokenize.STRING):
        return False
    if t.type == _tokenize.NAME:
        if t.string in ("True", "False", "None"):
            return False
        # Keywords like 'return', 'and', 'or', 'not', 'yield', 'if', 'else'
        # all signal an expression-start (unary) context.
        if _keyword.iskeyword(t.string):
            return True
        # Regular identifier (variable, function name) — binary context.
        return False
    # Any OP that is not a closing bracket means unary context
    # (opening brackets, arithmetic operators, comparison ops, assignment, etc.)
    return True


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
    # Tokens excluded from the bare NAME NAME handler (handled by their own branches
    # later in the loop, or meaningless as type names).
    _bare_excl = frozenset({"cimport", "nogil", "gil", "extern"})

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
                paren_depth = 0
                while k < n:
                    t = toks[k]
                    if t.type == _tokenize.OP and t.string == "(":
                        paren_depth += 1
                        k += 1
                        continue
                    if t.type == _tokenize.OP and t.string == ")":
                        paren_depth -= 1
                        last_tok = t
                        k += 1
                        if paren_depth == 0:
                            break
                        continue
                    if t.type in (
                        _tokenize.NEWLINE,
                        _tokenize.ENDMARKER,
                        _tokenize.COMMENT,
                    ):
                        if paren_depth == 0:
                            break
                        k += 1
                        continue
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

            # ---- cdef extern from "h"/"*": block header ----
            if (
                keyword == "cdef"
                and i + 3 < n
                and toks[i + 1].type == _tokenize.NAME
                and toks[i + 1].string == "extern"
                and toks[i + 2].type == _tokenize.NAME
                and toks[i + 2].string == "from"
            ):
                tok3 = toks[i + 3]
                if tok3.type == _tokenize.STRING:
                    hdr_bare = tok3.string.strip("'\"")
                    ph = _placeholder("extern", hdr_bare)
                elif tok3.type == _tokenize.OP and tok3.string == "*":
                    ph = _placeholder("extern", "wildcard")
                else:
                    ph = None
                if ph is not None:
                    s_start = _abs_start(tok, offsets)
                    s_end = _abs_end(tok3, offsets)
                    next_i = i + 4  # normally points at ':'
                    # Absorb optional 'namespace "..."' clause before ':'
                    if (
                        next_i < n
                        and toks[next_i].type == _tokenize.NAME
                        and toks[next_i].string == "namespace"
                        and next_i + 1 < n
                        and toks[next_i + 1].type == _tokenize.STRING
                    ):
                        s_end = _abs_end(toks[next_i + 1], offsets)
                        next_i += 2
                    # Absorb optional 'nogil' qualifier before ':'
                    if (
                        next_i < n
                        and toks[next_i].type == _tokenize.NAME
                        and toks[next_i].string == "nogil"
                    ):
                        s_end = _abs_end(toks[next_i], offsets)
                        next_i += 1
                    edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                    i = next_i  # points at ':' — processed normally by main loop
                    continue

            # Classify by scanning ahead for (, :, =, 'class', struct/union/enum,
            # or NEWLINE.  Track [ ] depth so colons inside memoryview are ignored.
            open_paren_idx = -1
            decl_type = "variable"  # default for cdef; will be overridden for cpdef
            bracket_depth = 0
            struct_kw = ""       # "struct" / "union" / "enum" if seen
            struct_name_idx = -1  # index of the block-name NAME token
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
                        if j == i + 1:
                            # Parenthesized return type immediately after keyword
                            # (e.g. 'cdef (int, double)' or 'cpdef (char*)').
                            # Skip to the matching ')' and continue classifying.
                            paren_depth_rt = 1
                            j += 1
                            while j < n and paren_depth_rt > 0:
                                if toks[j].type == _tokenize.OP:
                                    if toks[j].string == "(":
                                        paren_depth_rt += 1
                                    elif toks[j].string == ")":
                                        paren_depth_rt -= 1
                                j += 1
                            continue
                        open_paren_idx = j
                        decl_type = "function"
                        break
                    # struct/union/enum/fused keyword for cdef; enum only for cpdef
                    if t.type == _tokenize.NAME and t.string in (
                        "struct",
                        "union",
                        "enum",
                        "fused",
                    ):
                        if keyword == "cdef" or t.string == "enum":
                            struct_kw = t.string
                            j += 1
                            continue
                    if struct_kw and t.type == _tokenize.NAME:
                        struct_name_idx = j
                        j += 1
                        continue
                    if t.type == _tokenize.OP and t.string == ":":
                        if keyword == "cdef":
                            decl_type = "struct_block" if struct_kw else "block_header"
                            break
                        if keyword == "cpdef" and struct_kw:
                            decl_type = "struct_block"
                            break
                    if keyword == "cdef":
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

                # Remember current edit count so we can roll back speculative
                # function-skeleton edits if this turns out to be a forward
                # declaration (no body colon, e.g. inside cdef extern from blocks).
                pre_func_edits = len(edits)

                # Span 1: keyword … function-name [→ C-name alias string] → 'def __cy_...'
                s1_start = _abs_start(tok, offsets)
                s1_end = _abs_end(toks[last_hdr_idx], offsets)
                # Absorb optional C-name alias string between name and '(': foo "bar"(...)
                maybe_alias = last_hdr_idx + 1
                while maybe_alias < open_paren_idx and toks[maybe_alias].type in (
                    _tokenize.NL, _tokenize.INDENT, _tokenize.DEDENT, _tokenize.COMMENT,
                ):
                    maybe_alias += 1
                if (
                    maybe_alias < open_paren_idx
                    and toks[maybe_alias].type == _tokenize.STRING
                    and toks[maybe_alias].start[0] == toks[last_hdr_idx].start[0]
                ):
                    s1_end = _abs_end(toks[maybe_alias], offsets)
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

                    # Variadic '...' argument: mask as __cy_varargs identifier
                    if t.type == _tokenize.OP and t.string == "...":
                        va_start = _abs_start(t, offsets)
                        va_end = _abs_end(t, offsets)
                        edits.append(
                            _Edit(va_start, va_end, _placeholder("varargs"), src[va_start:va_end])
                        )
                        j += 1
                        continue

                    # Collect one argument slot (up to , / ) / = at bracket depth 0)
                    arg_name_toks: list[str] = []
                    arg_first_idx = -1
                    arg_last_idx = -1
                    arg_brk_depth = 0
                    arg_paren_depth = 0
                    arg_slot_start = j  # first token index of this arg slot
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
                        # Track ( ) depth for tuple-typed args like '(int, double) x'
                        if t2.type == _tokenize.OP and t2.string == "(":
                            arg_paren_depth += 1
                            k += 1
                            continue
                        if t2.type == _tokenize.OP and t2.string == ")":
                            if arg_paren_depth > 0:
                                arg_paren_depth -= 1
                                k += 1
                                continue
                            break  # end of arg list at paren depth 0
                        if arg_brk_depth > 0:
                            k += 1
                            continue
                        if t2.type == _tokenize.OP and t2.string == ",":
                            if arg_paren_depth > 0:
                                k += 1
                                continue  # comma inside tuple type, not an arg separator
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
                        # If the slot starts with '(' (tuple type like '(int, double) x'),
                        # include the opening '(' in the replacement span.
                        if (
                            toks[arg_slot_start].type == _tokenize.OP
                            and toks[arg_slot_start].string == "("
                        ):
                            sa_start = _abs_start(toks[arg_slot_start], offsets)
                        else:
                            sa_start = _abs_start(toks[arg_first_idx], offsets)
                        # Absorb trailing [] suffixes (e.g. 'int x[2][2]' → one placeholder)
                        arr_end = arg_last_idx
                        while (
                            arr_end + 1 < n
                            and toks[arr_end + 1].type == _tokenize.OP
                            and toks[arr_end + 1].string == "["
                        ):
                            depth = 1
                            arr_k = arr_end + 2
                            while arr_k < n and depth > 0:
                                if toks[arr_k].type == _tokenize.OP:
                                    if toks[arr_k].string == "[":
                                        depth += 1
                                    elif toks[arr_k].string == "]":
                                        depth -= 1
                                arr_k += 1
                            arr_end = arr_k - 1  # index of ']'
                        sa_end = _abs_end(toks[arr_end], offsets)
                        ph_arg = _placeholder(*arg_name_toks)
                        edits.append(_Edit(sa_start, sa_end, ph_arg, src[sa_start:sa_end]))

                # Check for postfix tokens between ) and : (nogil, with gil,
                # except VAL / except * / except ?VAL, and combinations).
                # The outer arg-scan loop did j += 1 before breaking on ')', so
                # j is now one past ')'.  The ')' token is at j - 1.
                # If any non-whitespace token (other than 'except +', which is
                # a C++ pattern out of scope) appears before ':', replace
                # ') postfix:' with '):  # __cy_postfix_... (comment anchor).
                paren_idx = j - 1  # index of the closing ')'
                has_postfix = False
                has_except_plus = False  # C++ exception propagation — skip
                pk = j  # j is already one past ')', scan from there
                colon_idx = -1
                while pk < n:
                    pt = toks[pk]
                    if pt.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                        break
                    if pt.type == _tokenize.OP and pt.string == ":":
                        colon_idx = pk
                        break
                    if pt.type == _tokenize.NAME and pt.string == "except":
                        # peek ahead to distinguish except+VALUE (C++) from
                        # except VALUE / except * / except ?VALUE (Cython)
                        pk2 = pk + 1
                        while pk2 < n and toks[pk2].type in (
                            _tokenize.NL,
                            _tokenize.COMMENT,
                        ):
                            pk2 += 1
                        if (
                            pk2 < n
                            and toks[pk2].type == _tokenize.OP
                            and toks[pk2].string == "+"
                        ):
                            has_except_plus = True
                            break
                        has_postfix = True
                    elif pt.type not in (_tokenize.NL, _tokenize.COMMENT):
                        has_postfix = True
                    pk += 1

                if has_postfix and not has_except_plus and colon_idx >= 0:
                    pf_start = _abs_start(toks[paren_idx], offsets)
                    pf_end = _abs_end(toks[colon_idx], offsets)
                    postfix_raw = src[pf_start + 1 : pf_end - 1].strip()
                    ph_pf = _placeholder("postfix", postfix_raw)
                    masked_pf = f"):  # {ph_pf}"
                    edits.append(_Edit(pf_start, pf_end, masked_pf, src[pf_start:pf_end]))
                    j = colon_idx + 1

                # Forward declaration: no colon found after ')' (e.g. a function
                # declaration inside a cdef extern from block).  Roll back the
                # speculative function-skeleton edits and emit a single identifier
                # placeholder covering the whole declaration.
                if colon_idx < 0 and not has_except_plus:
                    del edits[pre_func_edits:]
                    # Find the last significant token before NEWLINE.
                    # pk is at NEWLINE or ENDMARKER after the postfix scan.
                    end_k = pk - 1
                    while end_k > i and toks[end_k].type in (
                        _tokenize.NL,
                        _tokenize.COMMENT,
                        _tokenize.INDENT,
                        _tokenize.DEDENT,
                        _tokenize.NEWLINE,
                        _tokenize.ENDMARKER,
                    ):
                        end_k -= 1
                    fd_start = _abs_start(tok, offsets)
                    fd_end = _abs_end(toks[end_k], offsets)
                    ph_fd = _placeholder(*hdr_name_toks)
                    edits.append(_Edit(fd_start, fd_end, ph_fd, src[fd_start:fd_end]))

                # Advance i past ) and : to the start of the next logical line
                while j < n:
                    if toks[j].type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                        j += 1
                        break
                    j += 1
                i = j
                continue

            # ---- cdef: compound block header (bare or qualified) ----
            # Handles 'cdef:', 'cdef readonly:', 'cdef public:' etc.
            # j points at ':'; toks[j-1] is the last qualifier (or 'cdef' itself).
            if (
                decl_type == "block_header"
                and keyword == "cdef"
                and not struct_kw
            ):
                # Replace 'cdef [qualifier...]' → 'if __cy_cdef_block'; ':' stays.
                s_start = _abs_start(tok, offsets)
                s_end = _abs_end(toks[j - 1], offsets)
                ph = _placeholder("cdef", "block")
                edits.append(_Edit(s_start, s_end, f"if {ph}", src[s_start:s_end]))
                i = j  # advance past qualifier tokens to ':'
                continue

            # ---- cdef struct / union / enum / fused block header ----
            if decl_type == "struct_block" and keyword == "cdef":
                if struct_name_idx >= 0:
                    s_start = _abs_start(tok, offsets)
                    s_end = _abs_end(toks[struct_name_idx], offsets)
                    ph = _placeholder(struct_kw, toks[struct_name_idx].string)
                    edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                    i = struct_name_idx + 1
                    continue
                # anonymous: cdef enum: (no name) — span from 'cdef' to struct keyword
                last_kw_idx = -1
                for kk in range(i, j):
                    if toks[kk].type == _tokenize.NAME:
                        last_kw_idx = kk
                if last_kw_idx >= 0:
                    s_start = _abs_start(tok, offsets)
                    s_end = _abs_end(toks[last_kw_idx], offsets)
                    ph = _placeholder("cdef", "anon", struct_kw)
                    edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                    i = last_kw_idx + 1
                    continue
                i += 1
                continue

            # ---- cpdef enum [Name]: block header ----
            if decl_type == "struct_block" and keyword == "cpdef":
                if struct_name_idx >= 0:
                    s_start = _abs_start(tok, offsets)
                    s_end = _abs_end(toks[struct_name_idx], offsets)
                    ph = _placeholder("cpdef", struct_kw, toks[struct_name_idx].string)
                    edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                    i = struct_name_idx + 1
                    continue
                # anonymous cpdef enum:
                last_kw_idx = -1
                for kk in range(i, j):
                    if toks[kk].type == _tokenize.NAME:
                        last_kw_idx = kk
                if last_kw_idx >= 0:
                    s_start = _abs_start(tok, offsets)
                    s_end = _abs_end(toks[last_kw_idx], offsets)
                    ph = _placeholder("cpdef", "anon", struct_kw)
                    edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                    i = last_kw_idx + 1
                    continue
                i += 1
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
                    # Check if a ':' follows (body) or end-of-line (forward declaration)
                    has_colon = False
                    ck = class_name_idx + 1
                    while ck < n and toks[ck].type not in (
                        _tokenize.NEWLINE, _tokenize.ENDMARKER,
                    ):
                        if toks[ck].type == _tokenize.OP and toks[ck].string == ":":
                            has_colon = True
                            break
                        ck += 1
                    if has_colon:
                        edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                    else:
                        # Forward declaration: emit as plain identifier, not class statement
                        edits.append(_Edit(s_start, s_end, ph, src[s_start:s_end]))
                    i = class_name_idx + 1
                    continue
                i += 1
                continue

            # ---- variable declaration (cdef only) ----
            if decl_type == "variable" and keyword == "cdef":
                name_toks: list[str] = []
                last_name_idx = -1
                last_tok_idx = -1
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
                    if t.type not in (
                        _tokenize.NL, _tokenize.INDENT, _tokenize.DEDENT,
                    ):
                        last_tok_idx = k
                    if t.type == _tokenize.NAME:
                        name_toks.append(t.string)
                        last_name_idx = k
                    k += 1

                if last_name_idx >= 0 and len(name_toks) >= 1:
                    span_start = _abs_start(tok, offsets)
                    # Use last_tok_idx so trailing ']' from e.g. 'cdef int a[N]' is included
                    span_end = _abs_end(toks[last_tok_idx], offsets)
                    ph = _placeholder(*name_toks)
                    edits.append(_Edit(span_start, span_end, ph, src[span_start:span_end]))
                    i = k
                    continue

            # ---- multi-variable cdef: cdef TYPE var1, var2 [= expr], var3 ----
            # Replace the entire declaration with a single placeholder so the
            # original text (including initializers) is preserved verbatim.
            # Splitting per-variable would leave 'a = 1, b = 2' which is not
            # valid Python.
            if decl_type == "multi_variable" and keyword == "cdef":
                name_toks: list[str] = []
                last_tok_idx = -1
                k = i + 1
                while k < n:
                    t = toks[k]
                    if t.type in (
                        _tokenize.NEWLINE,
                        _tokenize.ENDMARKER,
                        _tokenize.COMMENT,
                    ):
                        break
                    if t.type == _tokenize.NAME:
                        name_toks.append(t.string)
                    last_tok_idx = k
                    k += 1
                if last_tok_idx >= 0 and name_toks:
                    span_start = _abs_start(tok, offsets)
                    span_end = _abs_end(toks[last_tok_idx], offsets)
                    ph = _placeholder(*name_toks)
                    edits.append(
                        _Edit(span_start, span_end, ph, src[span_start:span_end])
                    )
                i = k
                continue

            # Unhandled: block_header, cdef_class, cpdef non-function, etc.
            i += 1
            continue

        # ---- ctypedef: simple alias, struct/union/enum/class block header, or forward decl ----
        if tok.type == _tokenize.NAME and tok.string == "ctypedef":
            name_toks: list[str] = []
            last_name_idx = -1
            saw_colon = False
            ct_struct_kw = ""       # "struct" / "union" / "enum" / "class" / "fused" if present
            ct_struct_name_idx = -1  # last NAME before '[' or ':' after struct kw
            ct_bracket_end = -1     # index of ']' in [object ...] group, if any
            handled_func_ptr = False
            k = i + 1
            while k < n:
                t = toks[k]
                if t.type in (
                    _tokenize.NEWLINE,
                    _tokenize.ENDMARKER,
                    _tokenize.COMMENT,
                ):
                    break
                if t.type == _tokenize.OP and t.string == ":":
                    saw_colon = True
                    break
                if t.type == _tokenize.OP and t.string == "(":
                    # Function pointer: ctypedef RETURN_TYPE (*name)(args) [postfixes]
                    # Pattern: ( * NAME )
                    if (
                        k + 1 < n
                        and toks[k + 1].type == _tokenize.OP
                        and toks[k + 1].string == "*"
                        and k + 2 < n
                        and toks[k + 2].type == _tokenize.NAME
                    ):
                        func_name = toks[k + 2].string
                        # Absorb entire declaration to end of logical line
                        end_k = k
                        while end_k < n and toks[end_k].type not in (
                            _tokenize.NEWLINE,
                            _tokenize.ENDMARKER,
                            _tokenize.COMMENT,
                        ):
                            end_k += 1
                        last_k = end_k - 1
                        span_start = _abs_start(tok, offsets)  # tok = 'ctypedef'
                        span_end = _abs_end(toks[last_k], offsets)
                        ph = _placeholder("ctypedef", func_name)
                        edits.append(
                            _Edit(span_start, span_end, ph, src[span_start:span_end])
                        )
                        i = end_k
                        handled_func_ptr = True
                    break  # exit inner loop (handled or unrecognised)
                # Skip [...] bracket groups (memoryview slices in plain aliases,
                # or [object PyType] in ctypedef class declarations).
                if t.type == _tokenize.OP and t.string == "[":
                    depth = 1
                    k += 1
                    while k < n and depth > 0:
                        if toks[k].type == _tokenize.OP:
                            if toks[k].string == "[":
                                depth += 1
                            elif toks[k].string == "]":
                                depth -= 1
                        k += 1
                    if ct_struct_kw:
                        ct_bracket_end = k - 1  # index of ']' for class block
                    continue
                # Skip "extern" in "ctypedef extern class"
                if t.type == _tokenize.NAME and t.string == "extern" and not ct_struct_kw:
                    k += 1
                    continue
                if t.type == _tokenize.NAME and t.string in (
                    "struct",
                    "union",
                    "enum",
                    "fused",
                    "class",  # ctypedef class (Cython extension type declaration)
                ) and not ct_struct_kw:
                    ct_struct_kw = t.string
                    k += 1
                    continue
                if t.type == _tokenize.NAME:
                    name_toks.append(t.string)
                    last_name_idx = k
                    # Track last NAME before '[' as the block name
                    if ct_struct_kw and ct_bracket_end < 0:
                        ct_struct_name_idx = k
                k += 1

            # Function pointer form already handled inside the inner loop
            if handled_func_ptr:
                continue

            # ctypedef struct/union/enum/class Name: block header
            if saw_colon and ct_struct_kw and ct_struct_name_idx >= 0:
                s_start = _abs_start(tok, offsets)
                if ct_bracket_end >= 0:
                    s_end = _abs_end(toks[ct_bracket_end], offsets)
                    next_i = ct_bracket_end + 1
                else:
                    s_end = _abs_end(toks[ct_struct_name_idx], offsets)
                    next_i = ct_struct_name_idx + 1
                ph = _placeholder("ctypedef", ct_struct_kw, toks[ct_struct_name_idx].string)
                edits.append(_Edit(s_start, s_end, f"class {ph}", src[s_start:s_end]))
                i = next_i
                continue

            # ctypedef struct/union/enum TypeName (forward declaration, no body)
            if not saw_colon and ct_struct_kw and ct_struct_name_idx >= 0:
                span_start = _abs_start(tok, offsets)
                span_end = _abs_end(toks[ct_struct_name_idx], offsets)
                ph = _placeholder("ctypedef", ct_struct_kw, toks[ct_struct_name_idx].string)
                edits.append(_Edit(span_start, span_end, ph, src[span_start:span_end]))
                i = ct_struct_name_idx + 1
                continue

            # ctypedef TYPE ALIAS simple alias
            if not saw_colon and last_name_idx >= 0 and len(name_toks) >= 2:
                span_start = _abs_start(tok, offsets)
                span_end = _abs_end(toks[last_name_idx], offsets)
                # Absorb optional C-name alias string: ctypedef long long foo "__int128_t"
                alias_k = last_name_idx + 1
                while alias_k < n and toks[alias_k].type in (
                    _tokenize.NL, _tokenize.INDENT, _tokenize.DEDENT,
                    _tokenize.COMMENT,
                ):
                    alias_k += 1
                if (
                    alias_k < n
                    and toks[alias_k].type == _tokenize.STRING
                    and toks[alias_k].start[0] == toks[last_name_idx].start[0]
                ):
                    span_end = _abs_end(toks[alias_k], offsets)
                    k = alias_k + 1
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
                        # Absorb trailing [] suffix (e.g. 'const char a[]' → one placeholder)
                        def_arr_end = def_arg_last
                        if (
                            def_arg_last + 1 < n
                            and toks[def_arg_last + 1].type == _tokenize.OP
                            and toks[def_arg_last + 1].string == "["
                        ):
                            depth = 1
                            arr_k = def_arg_last + 2
                            while arr_k < n and depth > 0:
                                if toks[arr_k].type == _tokenize.OP:
                                    if toks[arr_k].string == "[":
                                        depth += 1
                                    elif toks[arr_k].string == "]":
                                        depth -= 1
                                arr_k += 1
                            def_arr_end = arr_k - 1  # index of ']'
                        sa_end = _abs_end(toks[def_arr_end], offsets)
                        ph_arg = _placeholder(*def_arg_toks)
                        edits.append(
                            _Edit(sa_start, sa_end, ph_arg, src[sa_start:sa_end])
                        )

            # Advance i past the entire header line so the bare NAME NAME handler
            # does not re-scan the typed args inside the parentheses.
            if open_paren_idx >= 0:
                while j < n:
                    if toks[j].type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                        j += 1
                        break
                    j += 1
                i = j
            else:
                i += 1
            continue

        # ---- Cython for-from loop: for VAR from BOUNDS [by STEP]: ----
        # 'for VAR from ...' is invalid Python; replace 'from BOUNDS [by STEP]'
        # with 'in __cy_for_from_VAR' so the loop header becomes a valid Python
        # for-in statement and the indented body is left intact.
        if (
            tok.type == _tokenize.NAME
            and tok.string == "for"
            and i + 2 < n
            and toks[i + 1].type == _tokenize.NAME
            and toks[i + 2].type == _tokenize.NAME
            and toks[i + 2].string == "from"
        ):
            loop_var = toks[i + 1].string
            from_tok = toks[i + 2]
            # Find the ':' ending the for-from header (track bracket depth)
            colon_idx = -1
            j = i + 3
            bdepth = 0
            while j < n:
                t = toks[j]
                if t.type in (_tokenize.NEWLINE, _tokenize.ENDMARKER):
                    break
                if t.type == _tokenize.OP and t.string in ("(", "[", "{"):
                    bdepth += 1
                elif t.type == _tokenize.OP and t.string in (")", "]", "}"):
                    bdepth -= 1
                elif t.type == _tokenize.OP and t.string == ":" and bdepth == 0:
                    colon_idx = j
                    break
                j += 1
            if colon_idx >= 0:
                # span: from 'from' to the token just before ':'
                s_start = _abs_start(from_tok, offsets)
                s_end = _abs_end(toks[colon_idx - 1], offsets)
                ph = _placeholder("for", "from", loop_var)
                edits.append(_Edit(s_start, s_end, f"in {ph}", src[s_start:s_end]))
                i = colon_idx + 1
                continue
            i += 1
            continue

        # ---- C cast: <TYPE>EXPR ----
        # Detected when '<' appears in unary/expression-start position.
        # Forms:
        #   <int>x        → absorb NAME   → __cy_cast_int_x
        #   <double>3.14  → absorb NUMBER → __cy_cast_double_3_14
        #   <char>'>'     → absorb STRING → __cy_cast_char
        #   <int>(expr)   → func-call form → __cy_cast_int  (leaves (expr) intact)
        #   <int>-1       → mask only     → __cy_cast_int  (leaves -1 as subtraction)
        if tok.type == _tokenize.OP and tok.string == "<":
            prev_sig = _prev_significant(toks, i)
            if _is_unary_position(prev_sig):
                # Scan forward: accept NAME and '*' tokens until '>'
                cast_type_parts: list[str] = []
                j = i + 1
                is_cast = False
                while j < n:
                    ct = toks[j]
                    if ct.type == _tokenize.NAME:
                        cast_type_parts.append(ct.string)
                        j += 1
                        continue
                    if ct.type == _tokenize.OP and ct.string == "*":
                        cast_type_parts.append("star")
                        j += 1
                        continue
                    if ct.type == _tokenize.OP and ct.string == "**":
                        # double-pointer type like <void**>
                        cast_type_parts.append("starstar")
                        j += 1
                        continue
                    if ct.type == _tokenize.OP and ct.string == "?":
                        # Cython checked cast: <Foo?>x
                        cast_type_parts.append("q")
                        j += 1
                        continue
                    if ct.type == _tokenize.OP and ct.string == "[":
                        # Memoryview slice type: <int[:n]> or <long[:1:1]>
                        # Skip the balanced [...] group.
                        cast_type_parts.append("slice")
                        bracket_d = 1
                        j += 1
                        while j < n and bracket_d > 0:
                            if toks[j].type == _tokenize.OP:
                                if toks[j].string == "[":
                                    bracket_d += 1
                                elif toks[j].string == "]":
                                    bracket_d -= 1
                            j += 1
                        continue
                    if ct.type == _tokenize.OP and ct.string == ">" and cast_type_parts:
                        is_cast = True
                        break
                    break  # any other token: not a C cast

                if is_cast:
                    # j is the index of '>'
                    ph_cast = _placeholder("cast", *cast_type_parts)
                    nj = j + 1  # token immediately after '>'
                    cast_start = _abs_start(tok, offsets)
                    if nj < n and toks[nj].type == _tokenize.NAME:
                        # Absorb trailing NAME: <int>x → __cy_cast_int_x
                        ph_cast = _placeholder("cast", *cast_type_parts, toks[nj].string)
                        cast_end = _abs_end(toks[nj], offsets)
                        edits.append(_Edit(cast_start, cast_end, ph_cast, src[cast_start:cast_end]))
                        i = nj + 1
                        continue
                    elif nj < n and toks[nj].type == _tokenize.NUMBER:
                        # Absorb trailing NUMBER: <double>3.14 → __cy_cast_double_3_14
                        ph_cast = _placeholder("cast", *cast_type_parts, toks[nj].string)
                        cast_end = _abs_end(toks[nj], offsets)
                        edits.append(_Edit(cast_start, cast_end, ph_cast, src[cast_start:cast_end]))
                        i = nj + 1
                        continue
                    elif nj < n and toks[nj].type == _tokenize.STRING:
                        # Absorb trailing STRING: <char>'>' → __cy_cast_char
                        cast_end = _abs_end(toks[nj], offsets)
                        edits.append(_Edit(cast_start, cast_end, ph_cast, src[cast_start:cast_end]))
                        i = nj + 1
                        continue
                    elif nj < n and toks[nj].type == _tokenize.OP and toks[nj].string == "(":
                        # Function-call form: <int>(expr) → __cy_cast_int(expr)
                        cast_end = _abs_end(toks[j], offsets)
                        edits.append(_Edit(cast_start, cast_end, ph_cast, src[cast_start:cast_end]))
                        i = j + 1
                        continue
                    elif (
                        nj < n
                        and toks[nj].type == _tokenize.OP
                        and toks[nj].string in ("-", "+", "~")
                        and nj + 1 < n
                        and toks[nj + 1].type in (_tokenize.NAME, _tokenize.NUMBER)
                    ):
                        # Absorb unary op + NAME/NUMBER: <int>-1 → __cy_cast_int_neg_1
                        # Avoids leaving '-1' as Python subtraction (which would cause
                        # Black to insert spaces, corrupting the restored text).
                        _unary_sfx = {"-": "neg", "+": "pos", "~": "inv"}
                        ph_cast = _placeholder(
                            "cast", *cast_type_parts,
                            _unary_sfx[toks[nj].string],
                            toks[nj + 1].string,
                        )
                        cast_end = _abs_end(toks[nj + 1], offsets)
                        edits.append(_Edit(cast_start, cast_end, ph_cast, src[cast_start:cast_end]))
                        i = nj + 2
                        continue
                    elif (
                        nj < n
                        and toks[nj].type == _tokenize.OP
                        and toks[nj].string == "&"
                        and nj + 1 < n
                        and toks[nj + 1].type == _tokenize.NAME
                    ):
                        # <type>&name → absorb both into one placeholder so the
                        # masked source stays valid Python (two adjacent identifiers
                        # would be a syntax error).
                        ao_name = toks[nj + 1].string
                        ph_cast = _placeholder("cast", *cast_type_parts, "addrof", ao_name)
                        cast_end = _abs_end(toks[nj + 1], offsets)
                        edits.append(_Edit(cast_start, cast_end, ph_cast, src[cast_start:cast_end]))
                        i = nj + 2
                        continue
                    else:
                        # Last resort: mask only <type>, leaving the operand as-is.
                        # Used for patterns we cannot fully absorb inline,
                        # e.g. <int>-(expr) where the operand is parenthesised.
                        cast_end = _abs_end(toks[j], offsets)
                        edits.append(_Edit(cast_start, cast_end, ph_cast, src[cast_start:cast_end]))
                        i = j + 1
                        continue

        # ---- address-of: &VAR or &(EXPR) ----
        # Detected when '&' appears in unary/expression-start position.
        # Forms:
        #   &x        → absorb NAME → __cy_addrof_x  (attr access .attr follows naturally)
        #   &(expr)   → func-call form → __cy_addrof(expr)
        if tok.type == _tokenize.OP and tok.string == "&":
            prev_sig = _prev_significant(toks, i)
            if _is_unary_position(prev_sig):
                nj = i + 1
                ao_start = _abs_start(tok, offsets)
                if nj < n and toks[nj].type == _tokenize.NAME:
                    # Absorb: &x → __cy_addrof_x
                    ph_ao = _placeholder("addrof", toks[nj].string)
                    ao_end = _abs_end(toks[nj], offsets)
                    edits.append(_Edit(ao_start, ao_end, ph_ao, src[ao_start:ao_end]))
                    i = nj + 1
                    continue
                elif nj < n and toks[nj].type == _tokenize.OP and toks[nj].string == "(":
                    # Function-call form: &(expr) → __cy_addrof(expr)
                    ph_ao = _placeholder("addrof")
                    ao_end = _abs_end(tok, offsets)
                    edits.append(_Edit(ao_start, ao_end, ph_ao, src[ao_start:ao_end]))
                    i = nj
                    continue
                # else: & in some unrecognised unary context — leave unmasked

        # ---- C pointer type: NAME* / NAME** in sizeof/typeof argument position ----
        # 'sizeof(void*)' tokenizes as NAME('void') OP('*') OP(')').
        # The '*' has no right operand so lib2to3 rejects the expression.
        # Note: '**' is a single token; count stars from its length.
        # Mask the preceding NAME together with the pointer token(s) when they are
        # followed by ')' or ',' — i.e. when '*' cannot be binary multiplication.
        if (
            tok.type == _tokenize.OP
            and all(c == "*" for c in tok.string)
            and len(tok.string) >= 1
            and i + 1 < n
            and toks[i + 1].type == _tokenize.OP
            and toks[i + 1].string in (")", ",", "*", "**", "[")
        ):
            prev_sig = _prev_significant(toks, i)
            if (
                prev_sig is not None
                and prev_sig.type == _tokenize.NAME
                and not _keyword.iskeyword(prev_sig.string)
            ):
                # Absorb consecutive pointer tokens (*, **, *, **…) after the NAME
                star_count = 0
                sk = i
                while sk < n and toks[sk].type == _tokenize.OP and all(
                    c == "*" for c in toks[sk].string
                ):
                    star_count += len(toks[sk].string)
                    sk += 1
                ptr_start = _abs_start(prev_sig, offsets)
                ptr_end = _abs_end(toks[sk - 1], offsets)
                ptr_parts = ["ptr"] * star_count + [prev_sig.string]
                ph_ptr = _placeholder(*ptr_parts)
                edits.append(_Edit(ptr_start, ptr_end, ph_ptr, src[ptr_start:ptr_end]))
                i = sk
                continue

        # ---- Cython C character literal: c'X' ----
        # In Cython, c'x' is a C-level character literal.  It tokenizes as NAME 'c'
        # immediately adjacent to STRING (no whitespace), e.g. c'\0', c'A', c'\x10'.
        # NAME STRING adjacency without an operator is not valid Python.
        if (
            tok.type == _tokenize.NAME
            and tok.string == "c"
            and i + 1 < n
            and toks[i + 1].type == _tokenize.STRING
            and toks[i + 1].string[0] in ("'", '"')
            and toks[i + 1].start == (tok.start[0], tok.end[1])  # immediately adjacent
        ):
            str_tok = toks[i + 1]
            raw_content = str_tok.string.strip("'\"")
            ph = _placeholder("char", raw_content)
            cl_start = _abs_start(tok, offsets)
            cl_end = _abs_end(str_tok, offsets)
            edits.append(_Edit(cl_start, cl_end, ph, src[cl_start:cl_end]))
            i += 2
            continue

        # ---- NAME "C-rename" alias (enum value rename, e.g. ONE "1") ----
        # In Cython enum bodies, 'ONE "1"' renames the C constant.  NAME STRING (same
        # line, not immediately adjacent) is not valid Python, so mask the pair.
        if (
            tok.type == _tokenize.NAME
            and not _keyword.iskeyword(tok.string)
            and tok.string not in _bare_excl
            and i + 1 < n
            and toks[i + 1].type == _tokenize.STRING
            and toks[i + 1].start[0] == tok.start[0]
            and toks[i + 1].start != (tok.start[0], tok.end[1])  # not immediately adjacent
        ):
            str_tok = toks[i + 1]
            raw_content = str_tok.string.strip("'\"")
            ph = _placeholder("rename", tok.string, raw_content)
            r_start = _abs_start(tok, offsets)
            r_end = _abs_end(str_tok, offsets)
            edits.append(_Edit(r_start, r_end, ph, src[r_start:r_end]))
            i += 2
            continue

        # ---- bare TYPE[N] NAME declaration (array member in struct/extern body) ----
        # 'MyStruct[2] b' is not valid Python (subscript-expr followed by NAME).
        # Detect NAME '[' EXPR ']' NAME on the same line and mask as one placeholder.
        if (
            tok.type == _tokenize.NAME
            and not _keyword.iskeyword(tok.string)
            and tok.string not in _bare_excl
            and i + 1 < n
            and toks[i + 1].type == _tokenize.OP
            and toks[i + 1].string == "["
            and toks[i + 1].start[0] == tok.start[0]
        ):
            # Find matching ']', then check if a NAME follows on the same line.
            brk_depth = 1
            arr_k = i + 2
            while arr_k < n and brk_depth > 0:
                if toks[arr_k].type == _tokenize.OP:
                    if toks[arr_k].string == "[":
                        brk_depth += 1
                    elif toks[arr_k].string == "]":
                        brk_depth -= 1
                arr_k += 1
            # arr_k is now one past ']'; check for trailing NAME on same line
            if (
                arr_k < n
                and toks[arr_k].type == _tokenize.NAME
                and not _keyword.iskeyword(toks[arr_k].string)
                and toks[arr_k].start[0] == tok.start[0]
            ):
                arr_name_tok = toks[arr_k]
                arrm_start = _abs_start(tok, offsets)
                arrm_end = _abs_end(arr_name_tok, offsets)
                ph = _placeholder(tok.string, arr_name_tok.string)
                edits.append(_Edit(arrm_start, arrm_end, ph, src[arrm_start:arrm_end]))
                i = arr_k + 1
                continue

        # ---- bare TYPE NAME declaration (struct/union/enum body, no cdef prefix) ----
        # Two adjacent non-keyword NAME tokens on the same source line cannot be valid
        # Python — they must be a Cython bare type declaration (struct/enum body).
        # Cython-specific keywords (cimport, nogil, gil) are also excluded so they are
        # handled by their own branches below / on the next iteration.
        if (
            tok.type == _tokenize.NAME
            and not _keyword.iskeyword(tok.string)
            and tok.string not in _bare_excl
            and i + 1 < n
            and toks[i + 1].type == _tokenize.NAME
            and not _keyword.iskeyword(toks[i + 1].string)
            and toks[i + 1].string not in _bare_excl
            and toks[i + 1].start[0] == tok.start[0]
        ):
            # Collect all consecutive non-keyword NAMEs on the same line
            # (e.g. 'unsigned int flags' → '__cy_unsigned_int_flags').
            bare_toks_list: list[str] = [tok.string]
            bare_last_k = i
            k = i + 1
            src_line = tok.start[0]
            while (
                k < n
                and toks[k].type == _tokenize.NAME
                and not _keyword.iskeyword(toks[k].string)
                and toks[k].string not in _bare_excl
                and toks[k].start[0] == src_line
            ):
                bare_toks_list.append(toks[k].string)
                bare_last_k = k
                k += 1
            sp_start = _abs_start(tok, offsets)
            sp_end = _abs_end(toks[bare_last_k], offsets)
            # Absorb optional C-name alias string: e.g. int foo "bar"(...)
            if (
                k < n
                and toks[k].type == _tokenize.STRING
                and toks[k].start[0] == src_line
            ):
                sp_end = _abs_end(toks[k], offsets)
                k += 1
            # If '(' follows on the same line, this is a bare function forward
            # declaration (allowed in cdef extern from bodies without cdef keyword).
            # Absorb the whole line — args, C-name alias, except postfix — into one
            # placeholder so no unparseable fragments remain in the masked output.
            skip_k = k
            while skip_k < n and toks[skip_k].type in (
                _tokenize.NL, _tokenize.INDENT, _tokenize.DEDENT, _tokenize.COMMENT,
            ):
                skip_k += 1
            if (
                skip_k < n
                and toks[skip_k].type == _tokenize.OP
                and toks[skip_k].string == "("
                and toks[skip_k].start[0] == src_line
            ):
                # Extend span to end of line.
                end_k = skip_k
                while end_k < n and toks[end_k].type not in (
                    _tokenize.NEWLINE, _tokenize.ENDMARKER,
                ):
                    if toks[end_k].type not in (
                        _tokenize.NL, _tokenize.INDENT, _tokenize.DEDENT,
                        _tokenize.COMMENT,
                    ):
                        bare_last_k = end_k
                    end_k += 1
                sp_end = _abs_end(toks[bare_last_k], offsets)
                k = end_k
            ph = _placeholder(*bare_toks_list)
            edits.append(_Edit(sp_start, sp_end, ph, src[sp_start:sp_end]))
            i = k
            continue

        # ---- include "file.pyx" directive ----
        # 'include NAME_TOKEN' is not valid Python (NAME STRING is rejected by
        # the parser), so we mask the whole statement as an identifier.
        if tok.type == _tokenize.NAME and tok.string == "include":
            j = i + 1
            if j < n and toks[j].type == _tokenize.STRING and toks[j].start[0] == tok.start[0]:
                file_raw = toks[j].string.strip("'\"\n")
                s_start = _abs_start(tok, offsets)
                s_end = _abs_end(toks[j], offsets)
                ph = _placeholder("include", file_raw)
                edits.append(_Edit(s_start, s_end, ph, src[s_start:s_end]))
                i = j + 1
                continue

        # ---- bare 'enum:' statement (anonymous enum in extern/struct body) ----
        # 'enum:' without cdef/cpdef prefix appears in cdef extern from bodies.
        # Mask 'enum' → 'if __cy_anon_enum' so the body parses as a Python if-block.
        if tok.type == _tokenize.NAME and tok.string == "enum":
            # Only mask at statement-start: previous meaningful token is ':', NEWLINE, etc.
            prev_sig = None
            for pk in range(i - 1, -1, -1):
                if toks[pk].type not in (
                    _tokenize.NL,
                    _tokenize.NEWLINE,
                    _tokenize.INDENT,
                    _tokenize.DEDENT,
                    _tokenize.COMMENT,
                    _tokenize.ENCODING,
                ):
                    prev_sig = toks[pk]
                    break
            at_stmt_start = prev_sig is None or (
                prev_sig.type == _tokenize.OP and prev_sig.string == ":"
            )
            if at_stmt_start:
                j = i + 1
                if j < n and toks[j].type == _tokenize.OP and toks[j].string == ":" and toks[j].start[0] == tok.start[0]:
                    s_start = _abs_start(tok, offsets)
                    s_end = _abs_end(tok, offsets)
                    ph = _placeholder("anon", "enum")
                    edits.append(_Edit(s_start, s_end, f"if {ph}", src[s_start:s_end]))
                    i = j
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
        if masked_start >= 0 and result[masked_start:anchor_end] == masked_text:
            result = result[:masked_start] + original_text + result[anchor_end:]
            continue

        # Fallback: Black may have wrapped 'import __cy_X' as
        # 'import (\n    __cy_X,\n)' when the masked line exceeded 88 chars.
        # The trailing comma is added by Black's magic trailing comma logic.
        # Limit search to result[:anchor_end+5] — the closing ')' is at most
        # a few chars past anchor_end, and anchor positions are stable
        # (processing right-to-left keeps leftward positions unchanged).
        if masked_text.startswith("import "):
            anchor_text = anchor.group()
            wrapped = re.search(
                r"import \(\n\s*" + re.escape(anchor_text) + r",?\s*\n\s*\)",
                result[: anchor_end + 5],
            )
            if wrapped:
                result = result[:wrapped.start()] + original_text + result[wrapped.end():]
                continue

        actual = result[max(0, masked_start) : anchor_end]
        raise AssertionError(
            f"unmask_cython: invariant violated at anchor {anchor.group()!r}: "
            f"expected {masked_text!r}, found {actual!r}"
        )

    return result
