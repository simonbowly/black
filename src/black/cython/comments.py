"""
Comment correlation pass: attach extracted Comment objects to AST nodes so
the formatter can emit them in the right place.

Scope (MVP): leading and trailing comments on statement-level nodes — i.e.,
nodes that appear directly inside a StatListNode, plus single-statement bodies
of compound constructs (def, class, if, for, while, try, with, etc.).

Not yet handled: comments on `else:` / `finally:` keyword lines (the AST does
not record those keyword line numbers); comments between clauses in try/except
are partially supported via ExceptClauseNode traversal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from Cython.Compiler.Nodes import Node, TryFinallyStatNode

from black.cython.tokenizer import Comment, LayoutInfo


@dataclass
class NodeComments:
    leading: list[Comment] = field(default_factory=list)
    trailing: Comment | None = None
    clause_header_leading: list[Comment] = field(default_factory=list)  # comments before clause keyword
    dangling: list[Comment] = field(default_factory=list)
    node_type: str = ""  # typename of the AST node this entry belongs to


CommentMap = dict[int, NodeComments]

# `body` is the only attr we treat as a statement container when its value is
# a single Node rather than a StatListNode.  `else_clause` / `finally_clause`
# are omitted because we don't have the `else:` / `finally:` keyword's line
# number easily available to bound the leading-comment range.
_SINGLE_BODY_ATTRS = frozenset({"body"})


class CommentMapBuilder:
    """Correlates comments to AST nodes by walking the parse tree."""

    def __init__(self, layout: LayoutInfo) -> None:
        self._by_line: dict[int, Comment] = {c.line: c for c in layout.comments}
        self._keyword_lines: dict[int, str] = layout.keyword_lines
        self._code_lines: frozenset[int] = layout.code_lines
        self._claimed: set[int] = set()
        self._result: defaultdict[int, NodeComments] = defaultdict(NodeComments)
        # Track which lines are indented (have comments at col > 0)
        self._indented_lines: set[int] = {c.line for c in layout.comments if c.col > 0}

    def _nc(self, node: Node) -> NodeComments:
        """Return (creating if needed) the NodeComments entry for node."""
        nid = id(node)
        entry = self._result[nid]
        if not entry.node_type:
            entry.node_type = type(node).__name__
        return entry

    def _claim_leading(self, start_line: int, end_line: int, node: Node) -> None:
        """Assign standalone comments in [start_line, end_line] as leading of node."""
        for ln in range(start_line, end_line + 1):
            c = self._by_line.get(ln)
            if c is not None and ln not in self._claimed and not c.is_inline:
                self._nc(node).leading.append(c)
                self._claimed.add(ln)

    def _claim_clause_header_leading(self, start_line: int, end_line: int, node: Node) -> None:
        """Assign standalone comments in [start_line, end_line] as clause header leading of node.

        These are comments before a clause keyword (else:, finally:) that should be emitted
        by the parent codegen before the keyword, not by the body's _apply_comments.
        """
        for ln in range(start_line, end_line + 1):
            c = self._by_line.get(ln)
            if c is not None and ln not in self._claimed and not c.is_inline:
                self._nc(node).clause_header_leading.append(c)
                self._claimed.add(ln)

    def _claim_dangling(self, start_line: int, end_line: int, node: Node) -> None:
        """Assign standalone comments in [start_line, end_line] as dangling of node.

        Dangling comments are those after the last statement in a block.
        Only claim indented comments (col > 0) — unindented comments belong to outer scopes.
        """
        for ln in range(start_line, end_line + 1):
            c = self._by_line.get(ln)
            if c is not None and ln not in self._claimed and not c.is_inline and c.col > 0:
                self._nc(node).dangling.append(c)
                self._claimed.add(ln)

    def _claim_trailing(self, line: int, node: Node) -> None:
        """Assign the inline comment on `line` as trailing of node (at most one).

        The trailing comment is attached to whichever node will emit the source
        line it sits on.  For IfStatNode that is the first IfClauseNode (the
        outer IfStatNode emits nothing on its own — each clause emits its own
        header line).  Redirecting here keeps the attachment-point ≡ emission-site
        invariant that codegen relies on.
        """
        if type(node).__name__ == "IfStatNode" and node.if_clauses:
            node = node.if_clauses[0]
        c = self._by_line.get(line)
        if c is not None and line not in self._claimed and c.is_inline:
            self._nc(node).trailing = c
            self._claimed.add(line)

    @staticmethod
    def _end_line(node: Node) -> int:
        try:
            return node.end_pos()[1]
        except Exception:
            return node.pos[1]

    def _find_keyword_line(self, start: int, end: int, keyword: str) -> int | None:
        """Find the line number of a keyword in the range [start, end]."""
        for ln in range(start, end + 1):
            if self._keyword_lines.get(ln) == keyword:
                return ln
        return None

    _DECORATED_TYPES: frozenset[str] = frozenset({"DefNode", "CClassDefNode", "CFuncDefNode", "CVarDefNode"})

    def _claim_decorated_def_keyword_trailing(self, stat: Node) -> None:
        """For decorated def/class nodes, also claim the inline comment on the
        actual def/class keyword line (which is NOT stat.pos[1] when decorators
        are present — pos[1] points to the first decorator line instead).
        """
        if type(stat).__name__ not in self._DECORATED_TYPES:
            return
        decs = getattr(stat, "decorators", None) or []
        if not decs:
            return
        last_dec_line = decs[-1].pos[1]
        # The keyword line is the first code line after the last decorator.
        for ln in range(last_dec_line + 1, last_dec_line + 10):
            if ln in self._code_lines:
                self._claim_trailing(ln, stat)
                break

    def _visit_stmt(self, stat: Node, container_start: int, container_end: int | None = None) -> None:
        """Assign leading/trailing comments to a single statement and recurse.

        If container_end is provided, claims comments after the statement as dangling.
        """
        stat_start = stat.pos[1]
        self._claim_leading(container_start + 1, stat_start - 1, stat)
        self._claim_trailing(stat_start, stat)
        self._claim_decorated_def_keyword_trailing(stat)

        # Claim comments after the statement as dangling (only relevant for bodies)
        if container_end is not None:
            stat_end = self._end_line(stat)
            self._claim_dangling(stat_end + 1, container_end, stat)

        self._visit_node(stat, container_end)

    def _visit_statlist(
        self, statlist: Node, container_start: int, container_end: int | None = None
    ) -> None:
        """Walk a StatListNode, assigning comments to each child statement.

        If container_end is provided, claims comments after the last statement as dangling.
        """
        stats = statlist.stats if statlist.stats else []
        if not stats:
            # If no statements but container_end is provided, claim dangling comments
            if container_end is not None:
                self._claim_dangling(container_start + 1, container_end, statlist)
            return

        prev_end = container_start

        for i, stat in enumerate(stats):
            stat_start = stat.pos[1]

            # Standalone comments between prev_end+1 and stat_start-1 → leading
            self._claim_leading(prev_end + 1, stat_start - 1, stat)

            # Inline comment on the statement's opening line → trailing.
            # Special case 1: Cython wraps `import X` as
            #   StatListNode(SingleAssignmentNode(lhs=NameNode, rhs=ImportNode))
            # The wrapper StatListNode sits at the same line as the inner node.
            # Redirect trailing to the inner child so that the comment is not
            # claimed on the wrapper (which has no formatter output of its own).
            trailing_target = stat
            if (
                type(stat).__name__ == "StatListNode"
                and stat.stats
                and len(stat.stats) == 1
                and stat.stats[0].pos[1] == stat_start
            ):
                trailing_target = stat.stats[0]
            self._claim_trailing(stat_start, trailing_target)
            self._claim_decorated_def_keyword_trailing(stat)

            # Recurse into nested blocks - pass container_end for dangling comment handling
            # For compound statements, the body might have dangling comments extending
            # beyond the AST end pos, so pass the next statement's line as container_end
            if i < len(stats) - 1:
                next_stat_start = stats[i + 1].pos[1]
                self._visit_node(stat, next_stat_start - 1)
            else:
                # Last statement - use the provided container_end
                self._visit_node(stat, container_end)

            prev_end = self._end_line(stat)

        # Claim comments after the last statement as dangling
        if container_end is not None:
            self._claim_dangling(prev_end + 1, container_end, statlist)

    def _visit_clause_body(
        self, body: Node, keyword_line: int | None, prev_end: int, next_keyword_line: int | None = None,
        parent_container_end: int | None = None
    ) -> None:
        """Visit a clause body (else/finally) and claim trailing/leading comments on the keyword line.

        next_keyword_line: line number of the next clause keyword (e.g., finally: after else:),
                           or None if this is the last clause.
        parent_container_end: container_end from the parent node (used when there's no next clause).
        """
        if keyword_line is not None:
            # Claim inline comment on the keyword line (e.g., "else:  # comment")
            # Attach it to the body node itself so codegen can emit it on the keyword line
            self._claim_trailing(keyword_line, body)

            # Claim standalone comments between preceding block and keyword line as "clause header"
            # comments. These are comments before the clause keyword (e.g., "# else branch").
            # The codegen will emit them before the keyword; _apply_comments will skip them.
            self._claim_clause_header_leading(prev_end + 1, keyword_line - 1, body)

            # Visit the body with the keyword line as container_start for proper leading comment bounds
            if type(body).__name__ == "StatListNode":
                # For StatListNode, pass container_end to claim dangling comments
                # Dangling comments go up to (but not including) the next keyword line
                # or the parent's container_end if there's no next clause
                if next_keyword_line:
                    container_end = next_keyword_line - 1
                elif parent_container_end:
                    container_end = parent_container_end
                else:
                    container_end = self._end_line(body)
                self._visit_statlist(body, keyword_line, container_end)
            else:
                # Single-statement body; also pass container_end for dangling comments
                if next_keyword_line:
                    container_end = next_keyword_line - 1
                elif parent_container_end:
                    container_end = parent_container_end
                else:
                    container_end = self._end_line(body)
                self._visit_stmt(body, keyword_line, container_end)
        else:
            # No keyword line found; visit normally
            self._visit_node(body)

    def _visit_node(self, node: Node, container_end: int | None = None) -> None:
        node_type = type(node).__name__

        # Module body
        if node_type == "ModuleNode":
            body = node.body
            self._visit_node(body)
            return

        # Statement list
        if node_type == "StatListNode":
            self._visit_statlist(node, node.pos[1], container_end)
            return

        # If / elif / else chain
        if node_type == "IfStatNode":
            prev_end = node.pos[1]
            if_clauses = node.if_clauses or []
            else_clause = getattr(node, "else_clause", None)

            # Process if/elif clauses
            for i, clause in enumerate(if_clauses):
                clause_start = clause.pos[1]
                self._claim_trailing(clause_start, clause)

                # Comments between previous clause end and this clause start are leading
                if i > 0:
                    self._claim_leading(prev_end + 1, clause_start - 1, clause)

                # Determine end of this clause's container
                if i < len(if_clauses) - 1:
                    next_clause_line = if_clauses[i + 1].pos[1]
                    self._visit_node(clause, next_clause_line - 1)
                elif else_clause is not None:
                    else_keyword_line = self._find_keyword_line(
                        self._end_line(clause) + 1,
                        else_clause.pos[1],
                        "else",
                    )
                    if else_keyword_line is not None:
                        self._visit_node(clause, else_keyword_line - 1)
                    else:
                        self._visit_node(clause, else_clause.pos[1] - 1)
                else:
                    self._visit_node(clause, container_end)

                prev_end = self._end_line(clause)

            # Process else clause
            if else_clause is not None:
                else_keyword_line = self._find_keyword_line(
                    prev_end + 1,
                    else_clause.pos[1],
                    "else",
                )
                self._visit_clause_body(
                    else_clause,
                    else_keyword_line,
                    prev_end,
                    parent_container_end=container_end,
                )
            return

        # Try / except / else / finally
        if node_type == "TryExceptStatNode":
            body = node.body
            except_clauses = node.except_clauses or []
            else_clause = getattr(node, "else_clause", None)

            # Visit try body
            if except_clauses:
                self._visit_node(body, except_clauses[0].pos[1] - 1)
            elif else_clause is not None:
                else_keyword_line = self._find_keyword_line(
                    self._end_line(body) + 1,
                    else_clause.pos[1],
                    "else",
                )
                if else_keyword_line is not None:
                    self._visit_node(body, else_keyword_line - 1)
                else:
                    self._visit_node(body, else_clause.pos[1] - 1)
            else:
                self._visit_node(body, container_end)

            prev_end = self._end_line(body)

            # Visit except clauses
            for i, clause in enumerate(except_clauses):
                clause_start = clause.pos[1]
                self._claim_trailing(clause_start, clause)
                self._claim_leading(prev_end + 1, clause_start - 1, clause)

                if i < len(except_clauses) - 1:
                    self._visit_node(clause, except_clauses[i + 1].pos[1] - 1)
                elif else_clause is not None:
                    else_keyword_line = self._find_keyword_line(
                        self._end_line(clause) + 1,
                        else_clause.pos[1],
                        "else",
                    )
                    if else_keyword_line is not None:
                        self._visit_node(clause, else_keyword_line - 1)
                    else:
                        self._visit_node(clause, else_clause.pos[1] - 1)
                else:
                    self._visit_node(clause, container_end)

                prev_end = self._end_line(clause)

            # Visit else clause
            if else_clause is not None:
                else_keyword_line = self._find_keyword_line(
                    prev_end + 1,
                    else_clause.pos[1],
                    "else",
                )
                self._visit_clause_body(
                    else_clause,
                    else_keyword_line,
                    prev_end,
                    parent_container_end=container_end,
                )
            return

        if node_type == "TryFinallyStatNode":
            body = node.body
            finally_clause = node.finally_clause

            # Special case: body might be a TryExceptStatNode wrapping the actual try body
            if type(body).__name__ == "TryExceptStatNode":
                finally_keyword_line = self._find_keyword_line(
                    self._end_line(body) + 1,
                    finally_clause.pos[1],
                    "finally",
                )
                if finally_keyword_line is not None:
                    self._visit_node(body, finally_keyword_line - 1)
                else:
                    self._visit_node(body, finally_clause.pos[1] - 1)
                prev_end = self._end_line(body)
            else:
                finally_keyword_line = self._find_keyword_line(
                    self._end_line(body) + 1,
                    finally_clause.pos[1],
                    "finally",
                )
                if finally_keyword_line is not None:
                    self._visit_node(body, finally_keyword_line - 1)
                else:
                    self._visit_node(body, finally_clause.pos[1] - 1)
                prev_end = self._end_line(body)

            # Visit finally clause
            self._visit_clause_body(
                finally_clause,
                finally_keyword_line,
                prev_end,
                parent_container_end=container_end,
            )
            return

        # Generic handling for nodes with statement bodies
        for attr_name in _SINGLE_BODY_ATTRS:
            child = getattr(node, attr_name, None)
            if child is not None and isinstance(child, Node):
                self._visit_node(child, container_end)

        # Visit other Node-valued children
        for attr_name in getattr(node, "child_attrs", ()):
            if attr_name in _SINGLE_BODY_ATTRS:
                continue
            child = getattr(node, attr_name, None)
            if child is None:
                continue
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, Node):
                        self._visit_node(item)
            elif isinstance(child, Node):
                self._visit_node(child)

    def build(self, tree: Node) -> CommentMap:
        self._visit_node(tree)
        return dict(self._result)


def build_comment_map(tree: Node, layout: LayoutInfo) -> CommentMap:
    """Return a mapping of node-id -> NodeComments for `tree`.

    The mapping keys are `id(node)` because Cython nodes are not hashable.
    """
    return CommentMapBuilder(layout).build(tree)
