"""
Safety checks for cyblack — three layers of correctness verification.

Level 1  assert_equivalent()    Cython parse-tree equivalence (always on)
Level 2  assert_stable()        Idempotency check            (always on)
Level 3  assert_c_equivalent()  Generated C equivalence      (CYBLACK_CHECK_C=1)
"""

import contextlib
import difflib
import inspect
import io
import math
import os
import re
from itertools import zip_longest
from pathlib import Path
from typing import Callable

from Cython.Build import cythonize
from Cython.Compiler import Errors
from Cython.Compiler import Options
from Cython.Compiler import PyrexTypes
from Cython.Compiler import TreeFragment
from Cython.Compiler.Nodes import Node, FromCImportStatNode, FromImportStatNode
from Cython.Compiler.ExprNodes import NameNode, ProxyNode
from Cython.Compiler.Symtab import Scope
from Cython.Compiler.Scanning import StringSourceDescriptor
from Cython.Compiler.TreeFragment import StringParseContext

_INCLUDE_LINE_RE = re.compile(
    r'^(?P<indent>[ \t]*)include\s+(?P<quote>["\'])(?P<filename>.+?)(?P=quote)(?P<trailing>\s*(?:#.*)?)$'
)
_LANGUAGE_LEVEL_DIRECTIVE_RE = re.compile(r"(^|\n)\s*#\s*cython:\s*.*\blanguage_level\s*=")
_LANGUAGE_LEVEL_VALUE_RE = re.compile(r"#\s*cython\s*:.*\blanguage_level\s*=\s*(\w+)")
_DIRECTIVE_COMMENT_RE = re.compile(r"#\s*cython\s*:.*=")
_DIRECTIVE_KV_RE = re.compile(r"#\s*cython\s*:\s*(.+?)\s*=\s*(.+)")
_FUTURE_IMPORT_RE = re.compile(r"^from\s+__future__\s+import\b")
_FUTURE_SENTINEL_RE = re.compile(
    r'(?m)^__cython_future_(\d+)__\s*=\s*"((?:[^"\\]|\\.)*)"'
)


def preprocess_all(
    source: str,
) -> tuple[str, dict[str, str], dict[str, str], dict[str, str]]:
    """Preprocess Cython source for string-based parsing.

    Performs three transformations so that a plain ``StringParseContext`` can
    parse the result without needing file I/O or AST patching:

    1. **Include statements** (anywhere in the file) — each
       ``include "path.pxi"`` line is replaced with the assignment statement::

           __cython_include_{i}__ = "path.pxi"

    2. **Top-of-file directive comments** — each ``#cython: key = val`` line
       that appears before the first real (non-comment, non-blank) source line
       is replaced with the assignment statement::

           __cython_directive_{j}__ = {"key": "val"}

    3. **Future imports** — ``from __future__ import ...`` lines are replaced
       with the assignment statement::

           __cython_future_{k}__ = "from __future__ import ..."

       These are restored to their original text before parsing (in
       ``_parse``) so that the parse context sees the actual future directive.

    Both sentinels use a per-type counter (``i`` for includes, ``j`` for
    directives, ``k`` for future imports) in the order they appear in the
    file.  Any trailing inline comment on the original line is preserved
    after the assignment.

    Returns ``(preprocessed_source, include_map, directive_map, future_map)``
    where each map maps sentinel name → original text.
    """
    include_map: dict[str, str] = {}
    directive_map: dict[str, str] = {}
    future_map: dict[str, str] = {}
    out_lines: list[str] = []
    include_idx = 0
    directive_idx = 0
    future_idx = 0
    header_done = False

    for line in source.splitlines(keepends=True):
        body = line.rstrip('\r\n')
        nl = line[len(body):]
        stripped = body.strip()

        inc_match = _INCLUDE_LINE_RE.match(body)
        if inc_match:
            sentinel = f"__cython_include_{include_idx}__"
            filename = inc_match.group('filename')
            include_map[sentinel] = filename
            out_lines.append(
                f"{inc_match.group('indent')}{sentinel} = "
                f'"{filename}"{inc_match.group("trailing")}{nl}'
            )
            include_idx += 1
            if not header_done and stripped and not stripped.startswith('#'):
                header_done = True
            continue

        # from __future__ import ... — anywhere in file; must be restored before parsing
        if _FUTURE_IMPORT_RE.match(stripped):
            sentinel = f"__cython_future_{future_idx}__"
            future_map[sentinel] = stripped
            escaped = stripped.replace("\\", "\\\\").replace('"', '\\"')
            out_lines.append(f'{sentinel} = "{escaped}"{nl}')
            future_idx += 1
            if not header_done:
                header_done = True
            continue

        if not header_done:
            if not stripped:
                out_lines.append(line)
            elif stripped.startswith('#'):
                if _DIRECTIVE_COMMENT_RE.match(stripped):
                    kv_match = _DIRECTIVE_KV_RE.match(stripped)
                    if kv_match:
                        key = kv_match.group(1).strip()
                        val = kv_match.group(2).strip()
                        sentinel = f"__cython_directive_{directive_idx}__"
                        directive_map[sentinel] = stripped
                        out_lines.append(
                            f'{sentinel} = {{"{key}": "{val}"}}{nl}'
                        )
                        directive_idx += 1
                    else:
                        out_lines.append(line)
                else:
                    out_lines.append(line)
            else:
                header_done = True
                out_lines.append(line)
        else:
            out_lines.append(line)

    return "".join(out_lines), include_map, directive_map, future_map


def _restore_future_sentinels(preprocessed: str, future_map: dict[str, str]) -> str:
    """Replace ``__cython_future_N__ = "text"`` lines with the original text.

    This is called inside ``_parse()`` so that Cython's parser sees the real
    ``from __future__ import ...`` statement and updates its parse context
    accordingly (e.g. setting ``is_absolute`` on subsequent cimport nodes).
    """
    if not future_map:
        return preprocessed

    def repl(m: re.Match) -> str:
        idx = m.group(1)
        sentinel = f"__cython_future_{idx}__"
        return future_map.get(sentinel, m.group(0))

    return _FUTURE_SENTINEL_RE.sub(repl, preprocessed)


def _parse(source: str, *, restore_futures: bool = True):
    """Parse Cython source into an AST via TreeFragment.

    Calls ``preprocess_all`` first to replace ``include`` statements,
    top-of-file ``#cython:`` directive comments, and ``from __future__``
    imports with sentinel assignment statements.

    When ``restore_futures=True`` (the default, used for equivalence
    checking), ``from __future__`` sentinels are restored to the original
    text before parsing so Cython's parser processes them correctly (e.g.
    setting ``is_absolute`` on subsequent cimport nodes).

    When ``restore_futures=False`` (used by the formatter), the preprocessed
    text is parsed as-is so that ``from __future__`` sentinels appear as
    ``SingleAssignmentNode``s in the tree.  The formatter then emits the
    original ``from __future__ import ...`` text when it encounters these
    sentinels, ensuring the formatted output is semantically correct.

    If the explicit language level is set via a directive comment, it is
    extracted from the original source and applied to the parse context so
    Cython uses the correct level even though the directive line has been
    replaced.  On ``CompileError`` (e.g. Python 2 ``print`` statements), the
    parse is retried with ``language_level=2`` unless an explicit
    ``language_level`` directive was present (which would mean the original
    source itself is broken).

    Calls ``Errors.init_thread()`` before each attempt to fully initialise
    Cython's thread-local error state.
    """
    preprocessed, _include_map, _directive_map, future_map = preprocess_all(source)
    if restore_futures:
        to_parse = _restore_future_sentinels(preprocessed, future_map)
    else:
        to_parse = preprocessed
    ll_match = _LANGUAGE_LEVEL_VALUE_RE.search(source)
    explicit_language_level = ll_match.group(1) if ll_match else None

    Errors.init_thread()
    try:
        ctx = StringParseContext("source")
        if explicit_language_level is not None:
            ctx.set_language_level(explicit_language_level)
        return TreeFragment.parse_from_strings("source", to_parse, context=ctx)
    except Errors.CompileError:
        if _LANGUAGE_LEVEL_DIRECTIVE_RE.search(source):
            raise

    Errors.init_thread()
    ctx = StringParseContext("source")
    ctx.set_language_level(2)
    return TreeFragment.parse_from_strings("source", to_parse, context=ctx)


# ---------------------------------------------------------------------------
# Tree traversal and node comparison
# ---------------------------------------------------------------------------

# Attribute names that never contribute to comparison, regardless of which
# node they appear on:
#   pos    source position tuple (descriptor, line, col); shifts when the
#          formatter moves code.  Still read from .pos for error reporting.
#   scope  Symtab.Scope object; non-comparable (no __eq__), new identity on
#          every parse.
_SKIP_ATTRS = frozenset({"pos"})


# Per-class enumeration of Node-valued attributes that Cython does NOT list in
# child_attrs but that compare_nodes must still traverse structurally.  Keep
# this explicit: we follow what Cython's own traversal protocol (child_attrs)
# says, plus exactly these documented exceptions.  Adding a new entry is a
# deliberate decision; auto-discovery hid drift between parser versions.
#
#   CClassDefNode.bases     TupleNode holding base-class expressions.
#                           Documented in Nodes.py since 2017 but never added
#                           to child_attrs (an upstream Cython oversight).
#   PyClassDefNode.classobj Py3ClassNode (Py3) or ClassNode (Py2) constructed
#                           internally by PyClassDefNode.__init__.  Holds
#                           metaclass-relevant leaf attributes that must be
#                           compared.
_EXTRA_CHILDREN: dict[str, tuple[str, ...]] = {
    "CClassDefNode":   ("bases", "decorators"),
    "PyClassDefNode":  ("classobj",),
    "LambdaNode":  ("result_expr", "args", "star_arg", "starstar_arg"),
    "NameNode":  ("annotation",),
    "AttributeNode":  ("annotation",),
    "IndexNode":  ("annotation",),
    "PyArgDeclNode":  ("annotation",),
    "AnnotationNode":  ("expr", "string"),
    "Py3ClassNode": ("doc",),
    "GeneratorExpressionNode": ("loop",),
    "ComprehensionNode": ("append",),
    "MatchAndAssignPatternNode": ("target",),
    "FusedTypeNode": ("types",),
    "TypecastNode": ("base_type", "declarator"),
    "NewExprNode": ("cppclass",),
    "SizeofTypeNode": ("base_type", "declarator"),
    "CVarDefNode": ("decorators",),
    "CppClassNode": ("base_classes",),
    "CythonArrayNode": ("base_type_node",),
    "GILStatNode": ("state_temp",),
    "GILExitNode": ("state_temp",),
}


# Note: this is pretty laborious, and we could just identify 'is it a node', but
# the point is that we want an expicit expectation of which attributes are nodes
# in the AST, and which are properties which we should compare for equality. If
# we use this defintion consistently in walk_tree and compare_leaf then we can be
# sure not to have missed any important attributes in AST comparison.
#
# OTOH it's brittle: the node set may change between cython versions. But we can
# validate and extend by running a test over all of cython's test codes.
#
# Process if there's an AST comparison 'fail':
# - Add to EXTRA_CHILDREN if it's a mis-identified node
# - Add to BACKREF_ATTRS if it's a refernce to parent (unlikely)
# - Add normalisation if it's sometimes a node and sometimes not, but it remains
#   very simple (e.g. NameNode -> string)


# Per-class back-reference attributes.  These are Node-valued attributes that
# point UP the tree (to a node the main walk has already visited) rather than
# down.  Following them would cycle, and comparing them as leaves would fail
# (Nodes have no __eq__).  The parent they point at is already covered by the
# main walk, so dropping the pointer loses no comparison.
#
# All four helper nodes listed here are constructed inside
# PyClassDefNode.__init__ (Cython Nodes.py:5196-5222) with class_def_node=self.
# CClassDefNode does not carry this attribute at all, so the skip never
# affects cdef-class comparisons.
_BACKREF_ATTRS: dict[str, frozenset[str]] = {
    "Py3ClassNode":            frozenset({"class_def_node"}),
    "PyClassNamespaceNode":    frozenset({"class_def_node"}),
    "PyClassMetaclassNode":    frozenset({"class_def_node"}),
    # No examples ... maybe picked out based on older cython versions?
    # Or python2 compatibility layer?
    #"ClassNode":               frozenset({"class_def_node"}),
    #"ClassCellInjectorNode":   frozenset({"class_def_node"}),
}


def _extra_children(node: Node) -> tuple[str, ...]:
    return _EXTRA_CHILDREN.get(type(node).__name__, ())


def _backref_attrs(node: Node) -> frozenset[str]:
    return _BACKREF_ATTRS.get(type(node).__name__, frozenset())


def _flatten_statlists(node: Node) -> None:
    """Recursively flatten ``StatListNode``s that are directly nested inside
    another ``StatListNode.stats`` list.

    Cython generates nested ``StatListNode``s as an implementation detail for:

    - Semicolon-separated statements:  ``i = 1; j = 2``  →  inner
      ``StatListNode([i=1, j=2])`` inside the module-level ``StatListNode``.
    - ``cdef:`` suites:  ``cdef:\\n    int x\\n    double y``  →  inner
      ``StatListNode([CVarDef(x), CVarDef(y)])`` inside the class body.
    - Multi-name imports:  ``import inspect, types``  →  same.

    Black-style formatting expands all of these to separate lines, so the
    re-parsed tree is flat where the original was nested.  Normalising both
    trees before comparison removes this cosmetic difference.
    """
    if type(node).__name__ == "StatListNode":
        # Recurse first (post-order) so nested statlists are already flattened
        for child in (node.stats or []):
            _flatten_statlists(child)
        new_stats: list = []
        for child in (node.stats or []):
            if type(child).__name__ == "StatListNode":
                new_stats.extend(child.stats or [])
            else:
                new_stats.append(child)
        node.stats = new_stats
    else:
        for attr_name in (node.child_attrs or ()):
            child = getattr(node, attr_name, None)
            if child is None:
                continue
            if isinstance(child, list):
                for item in child:
                    if item is not None and hasattr(item, "child_attrs"):
                        _flatten_statlists(item)
            elif hasattr(child, "child_attrs"):
                _flatten_statlists(child)


def _walk_tree(root: Node):
    """Iterative depth-first yield of (node, path) for every Node in a tree.

    Follows `child_attrs` plus `_extra_children`.  Path is a tuple of attribute
    names and list indices suitable for debugging.
    """
    stack = [(root, ())]
    while stack:
        node, path = stack.pop()
        yield node, path
        children = []
        attrs = tuple(node.child_attrs or ()) + _extra_children(node)
        for attr_name in attrs:
            child = getattr(node, attr_name, None)
            if child is None:  # FIXME
                continue
            if isinstance(child, list):
                for i, item in enumerate(child):
                    if item is not None:  # FIXME
                        children.append((item, path + (attr_name, i)))
            else:
                children.append((child, path + (attr_name,)))

        for child, child_path in reversed(children):
            for backref_attr in _backref_attrs(child):
                if getattr(child, backref_attr) is not node:
                    raise ValueError(
                        f"Back-reference {type(child).__name__}.{backref_attr} "
                        f"does not point to parent {type(node).__name__}"
                    )
            if not isinstance(child, Node):
                raise TypeError(
                    f"Child of {type(node).__name__} at {child_path} is not a Node: "
                    f"{child!r} ({type(child).__name__})"
                )
            if child is node:
                raise ValueError(
                    f"Node {type(node).__name__} at {child_path} is its own child"
                )
            stack.append((child, child_path))


class _NodeMismatch(Exception):
    """Raised by compare_nodes; caught by assert_equivalent to build a
    source-line-anchored ASTDifference.  Not part of the public API."""

    def __init__(self, n1: Node, n2: Node, reason: str) -> None:
        super().__init__(reason)
        self.n1 = n1
        self.n2 = n2
        self.reason = reason


class _NodeCompareFailed(Exception):
    """Raised by compare_nodes; caught by assert_equivalent to build a
    source-line-anchored ASTDifference.  Not part of the public API."""

    def __init__(self, n1: Node, n2: Node, reason: str) -> None:
        super().__init__(reason)
        self.n1 = n1
        self.n2 = n2
        self.reason = reason


class ASTDifference(Exception):
    pass


class ASTCompareFailed(Exception):
    pass


# FIXMEs
# Return bool is bad, lose context. Just bubble up exceptions with more detail


def _leaf_equal(v1: object, v2: object) -> bool:
    """Strict equality for non-child leaf attribute values.

    Every type that may appear as a non-child attribute value in a freshly
    parsed Cython AST is enumerated explicitly.  An unexpected type raises
    TypeError — matching the fail-fast contract of the rest of the module.

    Requires `type(v1) is type(v2)` for str/bytes to catch EncodedString
    drifting to plain str (or vice versa).  Pyrex types are compared via
    repr() since the base class has no __eq__ but every subclass __repr__
    embeds its specific type discriminator.
    """

    # Require exact matches (subtype and value) for simple types
    if v1 is None and v2 is None:
        return True
    if isinstance(v1, float) and isinstance(v2, float) and math.isnan(v1) and math.isnan(v2):
        return True
    if isinstance(v1, (int, float, bool, bytes, str)) and isinstance(v2, (int, float, bool, bytes, str)):
        return type(v1) is type(v2) and v1 == v2
    if isinstance(v1, PyrexTypes.BaseType) and isinstance(v2, PyrexTypes.BaseType):
        return repr(v1) == repr(v2)

    # Recurse into iterables (identical entries)
    if isinstance(v1, list) and isinstance(v2, list):
        return all(_leaf_equal(a, b) for a, b in zip(v1, v2))
    if isinstance(v1, tuple) and isinstance(v2, tuple):
        return all(_leaf_equal(a, b) for a, b in zip(v1, v2))

    # Recurse into mappings (identical keys and values)
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1.keys() != v2.keys():
            return False
        return all(_leaf_equal(v, v2[k]) for k, v in v1.items())

    if isinstance(v1, Scope) and isinstance(v2, Scope):
        # Attributes are static to each Scope subclass, it only matters that
        # the types are identical (ModuleScope, ClosureScope, etc)
        return type(v1) is type(v2)

    raise TypeError(
        f"Unhandled leaf type in comparison: {type(v1)!r} / {type(v2)!r}  "
        f"values: {v1!r} / {v2!r}"
    )


def _normalise_docstring(docstring):
    """ Docstrings may have had some formatting applied """
    return inspect.cleandoc(docstring)


def _normalise_imported_names(v1):
    """ Ignore the first part of FromCImportStatNode.imported_names tuples.
    These are position-dependent and expected to change with formatting. """
    normalised = []
    for (ssd, line, column), name, as_name in v1:
        if not isinstance(ssd, StringSourceDescriptor):
            raise TypeError(
                f"Expected StringSourceDescriptor in imported_names position, "
                f"got {type(ssd).__name__}"
            )
        if not isinstance(line, int) or not isinstance(column, int):
            raise TypeError(
                f"Expected int line/column in imported_names position, "
                f"got {type(line).__name__}/{type(column).__name__}"
            )
        normalised.append((name, as_name))
    return normalised


def _normalise_import_items(v1):
    # Better alternative: turn NameNode into name when inside a FromImportStatNode
    # items() list. Should be a tuple of identical elements.
    normalised = []
    for name, name_node in v1:
        if not isinstance(name_node, NameNode):
            raise TypeError(f"Normalisation failed {name_node}")
        normalised.append((name, name_node.name))
    return normalised


def _normalise_proxy_node_constant_result(v1):
    if type(v1) is not object:
        raise TypeError(f"Normalisation failed, should be plain object {v1}")
    return None


def compare_nodes(n1: Node, n2: Node) -> None:
    """Assert n1 and n2 have the same class and attributes. Does not recurse
    into child nodes; that should have already been handled by the calling
    traversal routine.

    Raises _NodeMismatch on the first difference.
    """
    if type(n1) is not type(n2):
        raise _NodeMismatch(
            n1, n2,
            f"type differs: {type(n1).__name__} vs {type(n2).__name__}"
        )

    node_type = type(n1)

    skip = (
        frozenset(n1.child_attrs or ())
        | frozenset(_extra_children(n1))
        | _backref_attrs(n1)
        | _SKIP_ATTRS
    )
    keys1 = set(n1.__dict__) - skip
    keys2 = set(n2.__dict__) - skip
    if keys1 != keys2:
        raise _NodeMismatch(
            n1, n2,
            f"attribute set differs: {sorted(keys1 ^ keys2)}"
        )

    for key in sorted(keys1):
        v1 = n1.__dict__[key]
        v2 = n2.__dict__[key]

        try:
            if key == "doc" and isinstance(v1, str) and isinstance(v2, str):
                v1 = _normalise_docstring(v1)
                v2 = _normalise_docstring(v2)

            if key == "imported_names" and isinstance(n1, FromCImportStatNode):
                v1 = _normalise_imported_names(v1)
                v2 = _normalise_imported_names(v2)

            if key == "items" and node_type is FromImportStatNode:
                v1 = _normalise_import_items(v1)
                v2 = _normalise_import_items(v2)

            if key == "constant_result" and node_type is ProxyNode:
                v1 = _normalise_proxy_node_constant_result(v1)
                v2 = _normalise_proxy_node_constant_result(v2)

            compare_equal = _leaf_equal(v1, v2)
        except Exception as exc:
            raise _NodeCompareFailed(n1, n2, f"Comparison failed attribute={key} {node_type}") from exc

        if not compare_equal:
            raise _NodeMismatch(
                n1, n2,
                f"attribute {key!r} of {type(n1).__name__} differs: {v1!r} vs {v2!r}"
            )


def _format_mismatch(src: str, dst: str, exc: _NodeMismatch) -> str:
    """ Add cython source lines to exception message originating from a
    difference found in node comparison. """
    line1 = exc.n1.pos[1] if getattr(exc.n1, "pos", None) else None
    line2 = exc.n2.pos[1] if getattr(exc.n2, "pos", None) else None
    src_lines = src.splitlines()
    dst_lines = dst.splitlines()
    src_line = src_lines[line1 - 1] if line1 and 1 <= line1 <= len(src_lines) else "<unknown>"
    dst_line = dst_lines[line2 - 1] if line2 and 1 <= line2 <= len(dst_lines) else "<unknown>"
    return (
        f"Parse trees differ after formatting.\n"
        f"  Reason: {exc.reason}\n"
        f"  Source   line {line1}: {src_line}\n"
        f"  Formatted line {line2}: {dst_line}"
    )


def assert_equivalent(src: str, dst: str) -> None:
    """Verify that src and dst parse to equivalent Cython ASTs.

    Walks both parse trees depth-first in lock-step, comparing each pair of
    nodes via compare_nodes.  On the first mismatch, raises an ASTDifference
    naming the reason and the offending line numbers in both the source and
    the formatted output (via node.pos).

    Raises:
        ASTDifference: If the trees differ.
    """
    tree_src = _parse(src)
    tree_dst = _parse(dst)

    _flatten_statlists(tree_src)
    _flatten_statlists(tree_dst)

    sentinel = object()
    for pair_src, pair_dst in zip_longest(
        _walk_tree(tree_src), _walk_tree(tree_dst),
        fillvalue=(sentinel, ()),
    ):
        n1, path1 = pair_src
        n2, path2 = pair_dst
        if n1 is sentinel or n2 is sentinel:
            longer = "source" if n2 is sentinel else "formatted"
            raise ASTDifference(
                f"Parse trees differ after formatting: node count mismatch "
                f"({longer} tree has extra nodes)."
            )
        try:
            compare_nodes(n1, n2)
        except _NodeMismatch as exc:
            raise ASTDifference(_format_mismatch(src, dst, exc)) from exc
        except _NodeCompareFailed as exc:
            raise ASTCompareFailed(_format_mismatch(src, dst, exc)) from exc
