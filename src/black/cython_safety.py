"""Cython AST equivalence check — vendored from cyblack.safety.

Only the assert_equivalent() entry point and its supporting machinery are
included.  assert_stable(), assert_c_equivalent(), and the comment/generated-C
checkers from cyblack are not vendored; Black's own stability check handles
idempotency, and generated-C equivalence is out of scope.

Source: cyblack/cyblack/safety.py (snapshot taken from the cyblack workspace).
"""

import inspect
import math
import re
from itertools import zip_longest

from Cython.Compiler import Errors
from Cython.Compiler import PyrexTypes
from Cython.Compiler import TreeFragment
from Cython.Compiler.ExprNodes import NameNode, ProxyNode
from Cython.Compiler.Nodes import Node, FromCImportStatNode, FromImportStatNode
from Cython.Compiler.Scanning import StringSourceDescriptor
from Cython.Compiler.Symtab import Scope
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
    if not future_map:
        return preprocessed

    def repl(m: re.Match) -> str:
        idx = m.group(1)
        sentinel = f"__cython_future_{idx}__"
        return future_map.get(sentinel, m.group(0))

    return _FUTURE_SENTINEL_RE.sub(repl, preprocessed)


def _parse(source: str, *, restore_futures: bool = True):
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

_SKIP_ATTRS = frozenset({"pos"})

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

_BACKREF_ATTRS: dict[str, frozenset[str]] = {
    "Py3ClassNode":            frozenset({"class_def_node"}),
    "PyClassNamespaceNode":    frozenset({"class_def_node"}),
    "PyClassMetaclassNode":    frozenset({"class_def_node"}),
}


def _extra_children(node: Node) -> tuple[str, ...]:
    return _EXTRA_CHILDREN.get(type(node).__name__, ())


def _backref_attrs(node: Node) -> frozenset[str]:
    return _BACKREF_ATTRS.get(type(node).__name__, frozenset())


def _flatten_statlists(node: Node) -> None:
    if type(node).__name__ == "StatListNode":
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
    stack = [(root, ())]
    while stack:
        node, path = stack.pop()
        yield node, path
        children = []
        attrs = tuple(node.child_attrs or ()) + _extra_children(node)
        for attr_name in attrs:
            child = getattr(node, attr_name, None)
            if child is None:
                continue
            if isinstance(child, list):
                for i, item in enumerate(child):
                    if item is not None:
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
    def __init__(self, n1: Node, n2: Node, reason: str) -> None:
        super().__init__(reason)
        self.n1 = n1
        self.n2 = n2
        self.reason = reason


class _NodeCompareFailed(Exception):
    def __init__(self, n1: Node, n2: Node, reason: str) -> None:
        super().__init__(reason)
        self.n1 = n1
        self.n2 = n2
        self.reason = reason


class ASTDifference(Exception):
    pass


class ASTCompareFailed(Exception):
    pass


def _leaf_equal(v1: object, v2: object) -> bool:
    if v1 is None and v2 is None:
        return True
    if isinstance(v1, float) and isinstance(v2, float) and math.isnan(v1) and math.isnan(v2):
        return True
    if isinstance(v1, (int, float, bool, bytes, str)) and isinstance(v2, (int, float, bool, bytes, str)):
        return type(v1) is type(v2) and v1 == v2
    if isinstance(v1, PyrexTypes.BaseType) and isinstance(v2, PyrexTypes.BaseType):
        return repr(v1) == repr(v2)
    if isinstance(v1, list) and isinstance(v2, list):
        return all(_leaf_equal(a, b) for a, b in zip(v1, v2))
    if isinstance(v1, tuple) and isinstance(v2, tuple):
        return all(_leaf_equal(a, b) for a, b in zip(v1, v2))
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1.keys() != v2.keys():
            return False
        return all(_leaf_equal(v, v2[k]) for k, v in v1.items())
    if isinstance(v1, Scope) and isinstance(v2, Scope):
        return type(v1) is type(v2)
    raise TypeError(
        f"Unhandled leaf type in comparison: {type(v1)!r} / {type(v2)!r}  "
        f"values: {v1!r} / {v2!r}"
    )


def _normalise_docstring(docstring: str) -> str:
    return inspect.cleandoc(docstring)


def _normalise_imported_names(v1):
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
    nodes via compare_nodes.  Raises ASTDifference on the first mismatch.
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
