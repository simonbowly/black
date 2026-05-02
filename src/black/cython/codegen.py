"""
Code generation for the Cython formatter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import chain
from typing import Final

from Cython.Compiler.Nodes import Node
from Cython.Compiler.ExprNodes import ExprNode, NameNode
from Cython.Compiler.PyrexTypes import CType, BaseType

from black.cython.comments import CommentMap
from black.cython.safety import preprocess_all


@dataclass(frozen=True)
class Fragment:
    text: str


@dataclass(frozen=True)
class HardNewline(Fragment):
    text: str = "\n"


@dataclass(frozen=True)
class SoftNewline(Fragment):
    text: str = "\n"


@dataclass(frozen=True)
class Indent(Fragment):
    text: str = "    "


@dataclass(frozen=True)
class Dedent(Fragment):
    text: str = ""


@dataclass(frozen=True)
class CommentFragment(Fragment):
    pass


@dataclass(frozen=True)
class _HeaderEnd(Fragment):
    """Sentinel marking the end of a compound-statement header line."""
    text: str = ""


def render(fragments: list[Fragment]) -> str:
    pieces: list[str] = []
    indent = 0
    start_of_line = True
    header_end_count = 0
    for frag in fragments:
        if isinstance(frag, Indent):
            indent += 1
            continue
        if isinstance(frag, Dedent):
            indent -= 1
            if indent < 0:
                raise ValueError("Negative indentation during render")
            continue
        if isinstance(frag, _HeaderEnd):
            header_end_count += 1
            continue
        if start_of_line and not isinstance(frag, (HardNewline, SoftNewline)):
            pieces.append("    " * indent)
        pieces.append(frag.text)
        start_of_line = isinstance(frag, (HardNewline, SoftNewline))
    if indent != 0:
        raise ValueError(f"Unbalanced indentation at render end: {indent}")
    if header_end_count != 0:
        raise ValueError(f"Unconsumed _HeaderEnd sentinels at render end: {header_end_count}")
    out = "".join(pieces)
    if out and not out.endswith("\n"):
        out += "\n"
    return out


class CodeFormatter:
    def __init__(
        self,
        comment_map: CommentMap,
        *,
        include_map: dict[str, str],
        directive_map: dict[str, str],
        future_map: dict[str, str],
    ) -> None:
        self.comment_map = comment_map
        self.include_map = include_map
        self.directive_map = directive_map
        self.future_map = future_map
        self._consumed: set[int] = set()

    def _assert_all_consumed(self) -> None:
        unconsumed = sorted(k for k in self.comment_map if k not in self._consumed)
        if unconsumed:
            raise ValueError(f"Unconsumed comment attachments: {unconsumed}")

    def _comments_for(self, node: Node):
        return self.comment_map.get(id(node))

    def _mark_consumed(self, node: Node) -> None:
        self._consumed.add(id(node))

    def _emit_comment(self, text: str) -> list[Fragment]:
        return [CommentFragment(text), HardNewline()]

    def _apply_comments(self, node: Node, fragments: list[Fragment]) -> list[Fragment]:
        nc = self._comments_for(node)
        clean = list(fragments)
        if nc is None:
            return [frag for frag in clean if not isinstance(frag, _HeaderEnd)]

        self._mark_consumed(node)

        leading = list(chain(
            (CommentFragment(c.text), HardNewline()) for c in nc.leading
        ))
        leading = list(chain.from_iterable(leading))

        clause_header_leading = list(chain(
            (CommentFragment(c.text), HardNewline()) for c in nc.clause_header_leading
        ))
        clause_header_leading = list(chain.from_iterable(clause_header_leading))

        dangling = list(chain(
            (CommentFragment(c.text), HardNewline()) for c in nc.dangling
        ))
        dangling = list(chain.from_iterable(dangling))

        if nc.trailing is not None:
            trailer = [Fragment("  "), CommentFragment(nc.trailing.text)]
            try:
                first_header_idx = next(
                    i for i, frag in enumerate(clean) if isinstance(frag, _HeaderEnd)
                )
            except StopIteration:
                try:
                    first_header_idx = next(
                        i for i, frag in enumerate(clean) if isinstance(frag, HardNewline)
                    )
                except StopIteration as exc:
                    raise ValueError(
                        f"Cannot place trailing comment for {type(node).__name__}"
                    ) from exc
            clean[first_header_idx:first_header_idx] = trailer

        clean = [frag for frag in clean if not isinstance(frag, _HeaderEnd)]
        return [*clause_header_leading, *leading, *clean, *dangling]

    def format(self, node: Node) -> list[Fragment]:
        method = getattr(self, f"_fmt_{type(node).__name__}", None)
        if method is None:
            raise TypeError(f"Unsupported node type: {type(node).__name__}")
        fragments = method(node)
        return self._apply_comments(node, fragments)

    def _fmt_ModuleNode(self, node: Node) -> list[Fragment]:
        if set(node.child_attrs or ()) != {"body"}:
            raise TypeError(f"Unexpected ModuleNode children: {node.child_attrs}")
        return self.format(node.body)

    def _fmt_StatListNode(self, node: Node) -> list[Fragment]:
        if set(node.child_attrs or ()) != {"stats"}:
            raise TypeError(f"Unexpected StatListNode children: {node.child_attrs}")
        fragments: list[Fragment] = []
        for stat in node.stats or []:
            fragments.extend(self.format(stat))
        return fragments

    def _fmt_PassStatNode(self, node: Node) -> list[Fragment]:
        return [Fragment("pass"), HardNewline()]

    def _fmt_ExprStatNode(self, node: Node) -> list[Fragment]:
        if set(node.child_attrs or ()) != {"expr"}:
            raise TypeError(f"Unexpected ExprStatNode children: {node.child_attrs}")
        return [*self._fmt_expr(node.expr), HardNewline()]

    def _fmt_NameNode(self, node: NameNode) -> list[Fragment]:
        return [Fragment(node.name)]

    def _fmt_IntNode(self, node: Node) -> list[Fragment]:
        return [Fragment(str(node.value))]

    def _fmt_FloatNode(self, node: Node) -> list[Fragment]:
        return [Fragment(str(node.value))]

    def _fmt_BoolNode(self, node: Node) -> list[Fragment]:
        return [Fragment("True" if node.value else "False")]

    def _fmt_NoneNode(self, node: Node) -> list[Fragment]:
        return [Fragment("None")]

    def _fmt_StringNode(self, node: Node) -> list[Fragment]:
        value = getattr(node, "value", None)
        if not isinstance(value, str):
            raise TypeError(f"Unexpected StringNode value: {value!r}")
        return [Fragment(repr(value))]

    def _fmt_AttributeNode(self, node: Node) -> list[Fragment]:
        return [*self._fmt_expr(node.obj), Fragment("."), Fragment(node.attribute)]

    def _fmt_IndexNode(self, node: Node) -> list[Fragment]:
        return [
            *self._fmt_expr(node.base),
            Fragment("["),
            *self._fmt_expr(node.index),
            Fragment("]"),
        ]

    def _fmt_PrimaryCmpNode(self, node: Node) -> list[Fragment]:
        return [
            *self._fmt_expr(node.operand1),
            Fragment(f" {node.operator} "),
            *self._fmt_expr(node.operand2),
        ]

    def _fmt_AddNode(self, node: Node) -> list[Fragment]:
        return self._fmt_binary_infix(node, " + ")

    def _fmt_SubNode(self, node: Node) -> list[Fragment]:
        return self._fmt_binary_infix(node, " - ")

    def _fmt_MulNode(self, node: Node) -> list[Fragment]:
        return self._fmt_binary_infix(node, " * ")

    def _fmt_DivNode(self, node: Node) -> list[Fragment]:
        return self._fmt_binary_infix(node, " / ")

    def _fmt_ModNode(self, node: Node) -> list[Fragment]:
        return self._fmt_binary_infix(node, " % ")

    def _fmt_PowNode(self, node: Node) -> list[Fragment]:
        return self._fmt_binary_infix(node, " ** ")

    def _fmt_SingleAssignmentNode(self, node: Node) -> list[Fragment]:
        return [
            *self._fmt_expr(node.lhs),
            Fragment(" = "),
            *self._fmt_expr(node.rhs),
            HardNewline(),
        ]

    def _fmt_ReturnStatNode(self, node: Node) -> list[Fragment]:
        if node.value is None:
            return [Fragment("return"), HardNewline()]
        return [Fragment("return "), *self._fmt_expr(node.value), HardNewline()]

    def _fmt_AssertStatNode(self, node: Node) -> list[Fragment]:
        if node.value is None:
            return [Fragment("assert "), *self._fmt_expr(node.cond), HardNewline()]
        return [
            Fragment("assert "),
            *self._fmt_expr(node.cond),
            Fragment(", "),
            *self._fmt_expr(node.value),
            HardNewline(),
        ]

    def _fmt_IfClauseNode(self, node: Node) -> list[Fragment]:
        return [
            Fragment("if "),
            *self._fmt_expr(node.condition),
            Fragment(":"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ]

    def _fmt_IfStatNode(self, node: Node) -> list[Fragment]:
        fragments: list[Fragment] = []
        clauses = node.if_clauses or []
        for index, clause in enumerate(clauses):
            clause_fragments = self._fmt_IfClauseNode(clause)
            if index > 0:
                clause_fragments[0] = Fragment("elif ")
            fragments.extend(clause_fragments)
        if node.else_clause is not None:
            fragments.extend([
                Fragment("else:"),
                _HeaderEnd(),
                HardNewline(),
                Indent(),
                *self.format(node.else_clause),
                Dedent(),
            ])
        return fragments

    def _fmt_ForInStatNode(self, node: Node) -> list[Fragment]:
        fragments = [
            Fragment("for "),
            *self._fmt_expr(node.target),
            Fragment(" in "),
            *self._fmt_expr(node.iterator.sequence),
            Fragment(":"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ]
        if node.else_clause is not None:
            fragments.extend([
                Fragment("else:"),
                _HeaderEnd(),
                HardNewline(),
                Indent(),
                *self.format(node.else_clause),
                Dedent(),
            ])
        return fragments

    def _fmt_WhileStatNode(self, node: Node) -> list[Fragment]:
        fragments = [
            Fragment("while "),
            *self._fmt_expr(node.condition),
            Fragment(":"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ]
        if node.else_clause is not None:
            fragments.extend([
                Fragment("else:"),
                _HeaderEnd(),
                HardNewline(),
                Indent(),
                *self.format(node.else_clause),
                Dedent(),
            ])
        return fragments

    def _fmt_DefNode(self, node: Node) -> list[Fragment]:
        return [
            Fragment("def "),
            Fragment(node.name),
            Fragment("("),
            *self._fmt_py_args(node.args),
            Fragment("):"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ]

    def _fmt_CFuncDefNode(self, node: Node) -> list[Fragment]:
        prefix = "cpdef " if getattr(node, "overridable", False) else "cdef "
        return_type = self._format_c_base_type(getattr(node, "base_type", None))
        parts = [Fragment(prefix)]
        if return_type:
            parts.extend([Fragment(return_type), Fragment(" ")])
        parts.extend([
            Fragment(node.declarator.base.name),
            Fragment("("),
            *self._fmt_c_func_args(node.declarator.args),
            Fragment("):"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ])
        return parts

    def _fmt_CClassDefNode(self, node: Node) -> list[Fragment]:
        fragments = [Fragment("cdef class "), Fragment(node.class_name)]
        if getattr(node, "bases", None) is not None:
            base_fragments = self._fmt_bases(node.bases.args)
            if base_fragments:
                fragments.extend([Fragment("("), *base_fragments, Fragment(")")])
        fragments.extend([
            Fragment(":"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ])
        return fragments

    def _fmt_PyClassDefNode(self, node: Node) -> list[Fragment]:
        fragments = [Fragment("class "), Fragment(node.name)]
        bases = getattr(node.classobj, "bases", None)
        if bases is not None and getattr(bases, "args", None):
            fragments.extend([Fragment("("), *self._fmt_bases(bases.args), Fragment(")")])
        fragments.extend([
            Fragment(":"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ])
        return fragments

    def _fmt_CVarDefNode(self, node: Node) -> list[Fragment]:
        base_type = self._format_c_base_type(getattr(node, "base_type", None))
        declarators = getattr(node, "declarators", None) or []
        if not base_type or not declarators:
            raise TypeError(f"Unsupported CVarDefNode: {node!r}")
        fragments: list[Fragment] = [Fragment("cdef "), Fragment(base_type), Fragment(" ")]
        items: list[str] = []
        for declarator, default in declarators:
            name = self._format_c_declarator(declarator)
            if default is None:
                items.append(name)
            else:
                items.append(f"{name} = {self._fragments_to_text(self._fmt_expr(default))}")
        fragments.append(Fragment(", ".join(items)))
        fragments.append(HardNewline())
        return fragments

    def _fmt_CImportStatNode(self, node: Node) -> list[Fragment]:
        if not getattr(node, "module_name", None):
            raise TypeError("Unsupported CImportStatNode without module_name")
        return [Fragment(f"cimport {node.module_name}"), HardNewline()]

    def _fmt_FromCImportStatNode(self, node: Node) -> list[Fragment]:
        module_name = getattr(node, "module_name", None)
        imported = []
        for _pos, name, as_name in node.imported_names:
            imported.append(name if as_name is None else f"{name} as {as_name}")
        return [
            Fragment(f"from {module_name} cimport {', '.join(imported)}"),
            HardNewline(),
        ]

    def _fmt_FromImportStatNode(self, node: Node) -> list[Fragment]:
        module = node.module.module_name.value if getattr(node, "module", None) else ""
        items = []
        for name, name_node in node.items:
            if name == name_node.name:
                items.append(name)
            else:
                items.append(f"{name} as {name_node.name}")
        return [Fragment(f"from {module} import {', '.join(items)}"), HardNewline()]

    def _fmt_ImportNode(self, node: Node) -> list[Fragment]:
        module_name = getattr(node, "module_name", None)
        if module_name is None:
            raise TypeError(f"Unsupported ImportNode: {node!r}")
        return [Fragment(f"import {module_name}")]

    def _fmt_GILStatNode(self, node: Node) -> list[Fragment]:
        keyword = "with nogil" if getattr(node, "state", "") == "nogil" else "with gil"
        return [
            Fragment(f"{keyword}:"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ]

    def _fmt_WithStatNode(self, node: Node) -> list[Fragment]:
        items = []
        for manager, target in node.manager or []:
            text = self._fragments_to_text(self._fmt_expr(manager))
            if target is not None:
                text = f"{text} as {self._fragments_to_text(self._fmt_expr(target))}"
            items.append(text)
        return [
            Fragment(f"with {', '.join(items)}:"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ]

    def _fmt_TryExceptStatNode(self, node: Node) -> list[Fragment]:
        fragments = [
            Fragment("try:"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ]
        for clause in node.except_clauses or []:
            fragments.extend(self.format(clause))
        if node.else_clause is not None:
            fragments.extend([
                Fragment("else:"),
                _HeaderEnd(),
                HardNewline(),
                Indent(),
                *self.format(node.else_clause),
                Dedent(),
            ])
        return fragments

    def _fmt_ExceptClauseNode(self, node: Node) -> list[Fragment]:
        fragments = [Fragment("except")]
        if node.pattern is not None:
            fragments.extend([Fragment(" "), *self._fmt_expr(node.pattern)])
        if node.target is not None:
            fragments.extend([Fragment(" as "), *self._fmt_expr(node.target)])
        fragments.extend([
            Fragment(":"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.body),
            Dedent(),
        ])
        return fragments

    def _fmt_TryFinallyStatNode(self, node: Node) -> list[Fragment]:
        return [
            *self.format(node.body),
            Fragment("finally:"),
            _HeaderEnd(),
            HardNewline(),
            Indent(),
            *self.format(node.finally_clause),
            Dedent(),
        ]

    def _fmt_RaiseStatNode(self, node: Node) -> list[Fragment]:
        fragments = [Fragment("raise")]
        if node.exc_type is not None:
            fragments.extend([Fragment(" "), *self._fmt_expr(node.exc_type)])
        if node.exc_value is not None:
            fragments.extend([Fragment(" from "), *self._fmt_expr(node.exc_value)])
        fragments.append(HardNewline())
        return fragments

    def _fmt_IncludeStatNode(self, node: Node) -> list[Fragment]:
        filename = getattr(node, "filename", None)
        if not isinstance(filename, str):
            raise TypeError(f"Unsupported IncludeStatNode: {node!r}")
        return [Fragment(f'include "{filename}"'), HardNewline()]

    def _fmt_expr(self, node: ExprNode | Node) -> list[Fragment]:
        method = getattr(self, f"_fmt_{type(node).__name__}", None)
        if method is None:
            raise TypeError(f"Unsupported expression node type: {type(node).__name__}")
        return method(node)

    def _fmt_binary_infix(self, node: Node, operator: str) -> list[Fragment]:
        return [
            *self._fmt_expr(node.operand1),
            Fragment(operator),
            *self._fmt_expr(node.operand2),
        ]

    def _fmt_py_args(self, args: Node | None) -> list[Fragment]:
        if args is None:
            return []
        items: list[str] = []
        for arg in getattr(args, "args", None) or []:
            items.append(self._format_py_arg(arg))
        return [Fragment(", ".join(items))]

    def _format_py_arg(self, arg: Node) -> str:
        name = getattr(arg, "name", None)
        if not isinstance(name, str):
            raise TypeError(f"Unsupported argument node: {arg!r}")
        annotation = getattr(arg, "annotation", None)
        default = getattr(arg, "default", None)
        text = name
        if annotation is not None:
            text = f"{text}: {self._fragments_to_text(self._fmt_expr(annotation))}"
        if default is not None:
            text = f"{text} = {self._fragments_to_text(self._fmt_expr(default))}"
        if getattr(arg, "is_star_arg", False):
            text = f"*{text}"
        if getattr(arg, "is_starstar_arg", False):
            text = f"**{text}"
        return text

    def _fmt_c_func_args(self, args: list[Node] | None) -> list[Fragment]:
        if not args:
            return []
        return [Fragment(", ".join(self._format_c_arg(arg) for arg in args))]

    def _format_c_arg(self, arg: Node) -> str:
        base_type = self._format_c_base_type(getattr(arg, "base_type", None))
        declarator = self._format_c_declarator(getattr(arg, "declarator", None))
        default = getattr(arg, "default", None)
        pieces = []
        if base_type:
            pieces.append(base_type)
        if declarator:
            pieces.append(declarator)
        text = " ".join(pieces)
        if default is not None:
            text = f"{text}={self._fragments_to_text(self._fmt_expr(default))}"
        return text

    def _format_c_base_type(self, base_type: BaseType | None) -> str:
        if base_type is None:
            return ""
        name = getattr(base_type, "name", None)
        if isinstance(name, str) and name:
            return name
        declaration_code = getattr(base_type, "declaration_code", None)
        if callable(declaration_code):
            rendered = declaration_code("", for_display=1)
            return rendered.strip()
        return str(base_type).strip()

    def _format_c_declarator(self, declarator: object) -> str:
        if declarator is None:
            return ""
        name = getattr(declarator, "name", None)
        if isinstance(name, str):
            return name
        base = getattr(declarator, "base", None)
        if base is not None and base is not declarator:
            base_name = self._format_c_declarator(base)
            if base_name:
                return base_name
        return str(declarator).strip()

    def _fmt_bases(self, args: list[Node] | None) -> list[Fragment]:
        if not args:
            return []
        return [Fragment(", ".join(self._fragments_to_text(self._fmt_expr(arg)) for arg in args))]

    @staticmethod
    def _fragments_to_text(fragments: list[Fragment]) -> str:
        return "".join(
            frag.text
            for frag in fragments
            if not isinstance(frag, (HardNewline, SoftNewline, Indent, Dedent, _HeaderEnd))
        )
