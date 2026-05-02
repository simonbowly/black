"""
Cyblack: opinionated Cython code formatter.
"""
from black.cython.codegen import CodeFormatter, render
from black.cython.comments import build_comment_map, CommentMap
from black.cython.safety import _parse, assert_equivalent, ASTDifference, preprocess_all
from black.cython.tokenizer import extract_layout, LayoutInfo


class FormatError(Exception):
    """Raised when the formatter refuses to process the source."""


class EquivalenceError(Exception):
    """Raised when formatted output is not semantically equivalent to input."""


def _check_comments_preserved(source: str, result: str) -> None:
    """Verify that formatted result contains exactly the same comments as source.

    Checks:
    1. Same comment texts in the same order (comments appear in document order
       in both source and result since the AST walk order is deterministic).
    2. Each comment is attached to the same node type and has the same role
       (leading block comment vs trailing line comment).
    """
    # Preprocess both sides so sentinel assignment lines don't appear in
    # layout.comments (they are AST nodes, not comments).
    src_preprocessed = preprocess_all(source)[0]
    res_preprocessed = preprocess_all(result)[0]

    src_layout = extract_layout(src_preprocessed)
    res_layout = extract_layout(res_preprocessed)

    # Build comment maps to get node_type + role for each comment.
    src_tree = _parse(source)
    res_tree = _parse(result)
    src_cmap = build_comment_map(src_tree, src_layout)
    res_cmap = build_comment_map(res_tree, res_layout)

    def annotated(layout: LayoutInfo, cmap: CommentMap) -> list[tuple[str, str, str]]:
        """Return [(text, node_type, role), ...] in document (line) order."""
        line_to_annotation: dict[int, tuple[str, str]] = {}
        for nc in cmap.values():
            for c in nc.leading:
                line_to_annotation[c.line] = (nc.node_type, "leading")
            for c in nc.clause_header_leading:
                line_to_annotation[c.line] = (nc.node_type, "clause_header_leading")
            if nc.trailing is not None:
                line_to_annotation[nc.trailing.line] = (nc.node_type, "trailing")
            for c in nc.dangling:
                line_to_annotation[c.line] = (nc.node_type, "dangling")

        result = []
        for c in layout.comments:  # already sorted by line
            node_type, role = line_to_annotation.get(c.line, ("", ""))
            result.append((c.text, node_type, role))
        return result

    src_annotated = annotated(src_layout, src_cmap)
    res_annotated = annotated(res_layout, res_cmap)

    if src_annotated != res_annotated:
        lost = [t for t in src_annotated if t not in res_annotated]
        added = [t for t in res_annotated if t not in src_annotated]
        parts = []
        if lost:
            parts.append(f"lost {lost}")
        if added:
            parts.append(f"added {added}")
        raise EquivalenceError(f"Comment mismatch — {'; '.join(parts)}")


def format_source(source: str, *, check: bool = True) -> str:
    """Format Cython source code.

    Args:
        source: Cython source code as a string.
        check: If True, verify parse-tree equivalence and idempotency after
               formatting.  Set to False only when calling from inside a
               safety check to avoid infinite recursion.

    Returns:
        Formatted source code.

    Raises:
        FormatError: If source contains tabs or cannot be parsed by Cython.
        EquivalenceError: If formatted code is not semantically equivalent to
                          the original, or if formatting is not idempotent.
    """
    if "\t" in source:
        raise FormatError("Source contains tabs — refusing to format")

    # Preprocess source so include/directive/future lines are sentinel assignments.
    # The preprocessed form is what the tokenizer and comment correlator see.
    preprocessed, include_map, directive_map, future_map = preprocess_all(source)
    layout = extract_layout(preprocessed)
    tree = _parse(source, restore_futures=False)
    comment_map = build_comment_map(tree, layout)
    formatter = CodeFormatter(
        comment_map,
        include_map=include_map,
        directive_map=directive_map,
        future_map=future_map,
    )
    fragments = formatter.format(tree)
    formatter._assert_all_consumed()
    result = render(fragments)

    if check:
        try:
            assert_equivalent(source, result)
        except ASTDifference as exc:
            raise EquivalenceError(
                f"Formatted output is not equivalent to input:\n{exc}"
            ) from exc

        _check_comments_preserved(source, result)

        # Idempotency: formatting the result again must produce identical output
        result2 = format_source(result, check=False)
        if result != result2:
            raise EquivalenceError(
                f"Formatting is not idempotent:\n"
                f"First pass:\n{result!r}\n"
                f"Second pass:\n{result2!r}"
            )

    return result
