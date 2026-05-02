from __future__ import annotations

import keyword
import re
from dataclasses import dataclass

from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.Nodes import Node
from Cython.Compiler.PyrexTypes import BaseType

from black.cython.safety import _parse

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INDENT = "    "


class ProjectionError(Exception):
    """Raised when a Cython construct cannot yet be projected into Python."""


@dataclass(frozen=True)
class Replacement:
    token: str
    value: str


@dataclass(frozen=True)
class ProjectedSource:
    surrogate: str
    replacements: tuple[Replacement, ...]


class _IdentifierFactory:
    def __init__(self, source: str) -> None:
        self._used = set(_IDENTIFIER_RE.findall(source))
        self._counter = 0

    def make(self, length: int) -> str:
        if length < 1:
            raise ProjectionError("Cannot create an empty projection token.")
        while True:
            candidate = self._candidate(length, self._counter)
            self._counter += 1
            if keyword.iskeyword(candidate) or candidate in self._used:
                continue
            self._used.add(candidate)
            return candidate

    @staticmethod
    def _candidate(length: int, counter: int) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        if length == 1:
            return alphabet[counter % len(alphabet)]
        encoded = _IdentifierFactory._encode(counter, alphabet)
        width = length - 1
        if len(encoded) > width:
            encoded = encoded[-width:]
        return "_" + encoded.rjust(width, "x")

    @staticmethod
    def _encode(counter: int, alphabet: str) -> str:
        if counter == 0:
            return alphabet[0]
        base = len(alphabet)
        chars: list[str] = []
        value = counter
        while value:
            value, remainder = divmod(value, base)
            chars.append(alphabet[remainder])
        return "".join(reversed(chars))


class SurrogateProjector:
    def __init__(self, source: str) -> None:
        self._source = source
        self._factory = _IdentifierFactory(source)
        self._replacements: list[Replacement] = []

    def project(self) -> ProjectedSource:
        if not self._source.strip():
            return ProjectedSource(self._source, ())

        tree = _parse(self._source, restore_futures=False)
        surrogate = self._project_stmt(tree.body, indent=0)
        if surrogate and not surrogate.endswith("\n"):
            surrogate += "\n"
        return ProjectedSource(surrogate, tuple(self._replacements))

    def _project_stmt(self, node: Node, *, indent: int) -> str:
        method = getattr(self, f"_project_{type(node).__name__}", None)
        if method is None:
            raise ProjectionError(
                f"Unsupported Cython statement node: {type(node).__name__}"
            )
        return method(node, indent=indent)

    def _project_block(self, node: Node, *, indent: int) -> str:
        return self._project_stmt(node, indent=indent)

    def _project_ModuleNode(self, node: Node, *, indent: int) -> str:
        return self._project_stmt(node.body, indent=indent)

    def _project_StatListNode(self, node: Node, *, indent: int) -> str:
        return "".join(self._project_stmt(stat, indent=indent) for stat in node.stats or [])

    def _project_PassStatNode(self, node: Node, *, indent: int) -> str:
        return f"{_INDENT * indent}pass\n"

    def _project_CImportStatNode(self, node: Node, *, indent: int) -> str:
        module_name = getattr(node, "module_name", None)
        if not module_name:
            raise ProjectionError("Unsupported cimport statement without a module name.")
        as_name = getattr(node, "as_name", None)
        surrogate = f"import {module_name}"
        original = f"cimport {module_name}"
        if as_name:
            surrogate += f" as {as_name}"
            original += f" as {as_name}"
        self._add_exact_replacement(surrogate, original)
        return f"{_INDENT * indent}{surrogate}\n"

    def _project_FromCImportStatNode(self, node: Node, *, indent: int) -> str:
        module_name = getattr(node, "module_name", None)
        if not module_name:
            raise ProjectionError("Unsupported from-cimport statement without a module name.")
        imported_items: list[str] = []
        for _pos, name, as_name in getattr(node, "imported_names", None) or []:
            imported_items.append(name if as_name is None else f"{name} as {as_name}")
        if not imported_items:
            raise ProjectionError("Unsupported empty from-cimport statement.")
        imported = ", ".join(imported_items)
        surrogate = f"from {module_name} import {imported}"
        original = f"from {module_name} cimport {imported}"
        self._add_exact_replacement(surrogate, original)
        return f"{_INDENT * indent}{surrogate}\n"

    def _project_CVarDefNode(self, node: Node, *, indent: int) -> str:
        declarators = getattr(node, "declarators", None) or []
        if len(declarators) != 1:
            raise ProjectionError(
                "Only single-declarator cdef variable statements are supported yet."
            )
        declarator = declarators[0]
        decl_token = self._make_value_token(self._project_c_var_header(node, declarator))
        default = getattr(declarator, "default", None)
        if default is None:
            return f"{_INDENT * indent}{decl_token}\n"
        return (
            f"{_INDENT * indent}{decl_token} = "
            f"{self._project_expr(default)}\n"
        )

    def _project_ExprStatNode(self, node: Node, *, indent: int) -> str:
        return f"{_INDENT * indent}{self._project_expr(node.expr)}\n"

    def _project_SingleAssignmentNode(self, node: Node, *, indent: int) -> str:
        return (
            f"{_INDENT * indent}{self._project_expr(node.lhs)} = "
            f"{self._project_expr(node.rhs)}\n"
        )

    def _project_ReturnStatNode(self, node: Node, *, indent: int) -> str:
        if node.value is None:
            return f"{_INDENT * indent}return\n"
        return f"{_INDENT * indent}return {self._project_expr(node.value)}\n"

    def _project_InPlaceAssignmentNode(self, node: Node, *, indent: int) -> str:
        operator = getattr(node, "operator", "")
        op_text = operator if str(operator).endswith("=") else f"{operator}="
        return (
            f"{_INDENT * indent}{self._project_expr(node.lhs)} {op_text} "
            f"{self._project_expr(node.rhs)}\n"
        )

    def _project_IfStatNode(self, node: Node, *, indent: int) -> str:
        parts: list[str] = []
        for index, clause in enumerate(node.if_clauses or []):
            keyword_text = "if" if index == 0 else "elif"
            parts.append(
                f"{_INDENT * indent}{keyword_text} "
                f"{self._project_expr(clause.condition)}:\n"
            )
            parts.append(self._project_block(clause.body, indent=indent + 1))
        if node.else_clause is not None:
            parts.append(f"{_INDENT * indent}else:\n")
            parts.append(self._project_block(node.else_clause, indent=indent + 1))
        return "".join(parts)

    def _project_ForInStatNode(self, node: Node, *, indent: int) -> str:
        if getattr(node, "is_async", False):
            raise ProjectionError("Async for loops are not supported yet.")
        iterator = getattr(node, "iterator", None)
        sequence = getattr(iterator, "sequence", None)
        if sequence is None:
            raise ProjectionError("Unsupported Cython for-loop iterator.")
        parts = [
            f"{_INDENT * indent}for {self._project_expr(node.target)} in "
            f"{self._project_expr(sequence)}:\n",
            self._project_block(node.body, indent=indent + 1),
        ]
        if node.else_clause is not None:
            parts.append(f"{_INDENT * indent}else:\n")
            parts.append(self._project_block(node.else_clause, indent=indent + 1))
        return "".join(parts)

    def _project_DefNode(self, node: Node, *, indent: int) -> str:
        args = self._project_py_args(getattr(node, "args", None))
        return self._emit_compound_statement(
            f"{_INDENT * indent}def {node.name}({args}):\n"
            f"{self._project_docstring(getattr(node, 'doc', None), indent=indent + 1)}"
            f"{self._project_block(node.body, indent=indent + 1)}"
        )

    def _project_CFuncDefNode(self, node: Node, *, indent: int) -> str:
        declarator = getattr(node, "declarator", None)
        if declarator is None:
            raise ProjectionError("Cython function is missing a declarator.")
        if getattr(declarator, "nogil", False):
            raise ProjectionError("`nogil` functions are not supported by the adapter yet.")
        if getattr(declarator, "with_gil", False):
            raise ProjectionError("`with gil` functions are not supported by the adapter yet.")
        if getattr(declarator, "exception_value", None) is not None:
            raise ProjectionError(
                "Exception clauses are not supported by the adapter yet."
            )
        if getattr(node, "decorators", None):
            raise ProjectionError("Decorated Cython functions are not supported yet.")
        if getattr(node, "py_func_stat", None) is not None:
            raise ProjectionError("cpdef helper nodes are not supported yet.")

        header = self._project_c_func_header(node)
        header_token = self._make_header_token(header)
        args = ", ".join(
            self._project_c_arg(arg) for arg in getattr(declarator, "args", None) or []
        )
        return self._emit_compound_statement(
            f"{_INDENT * indent}def {header_token}({args}):\n"
            f"{self._project_docstring(getattr(node, 'doc', None), indent=indent + 1)}"
            f"{self._project_block(node.body, indent=indent + 1)}"
        )

    def _project_c_func_header(self, node: Node) -> str:
        parts: list[str] = ["cpdef" if getattr(node, "overridable", False) else "cdef"]
        visibility = getattr(node, "visibility", None)
        if visibility and visibility != "private":
            parts.append(visibility)
        if getattr(node, "api", False):
            parts.append("api")
        base_type = self._format_c_base_type(getattr(node, "base_type", None))
        if base_type:
            parts.append(base_type)
        name = self._format_c_declarator(getattr(node, "declarator", None))
        if not name:
            raise ProjectionError("Unable to determine the Cython function name.")
        parts.append(name)
        return " ".join(parts)

    def _project_CClassDefNode(self, node: Node, *, indent: int) -> str:
        bases = getattr(node, "bases", None)
        base_args = getattr(bases, "args", None) if bases is not None else None
        if base_args:
            raise ProjectionError("cdef class bases are not supported yet.")
        header = self._project_c_class_header(node)
        header_token = self._make_class_header_token(header)
        return self._emit_compound_statement(
            f"{_INDENT * indent}class {header_token}:\n"
            f"{self._project_docstring(getattr(node, 'doc', None), indent=indent + 1)}"
            f"{self._project_block(node.body, indent=indent + 1)}"
        )

    def _project_c_class_header(self, node: Node) -> str:
        parts = ["cdef class", getattr(node, "class_name", "")]
        if not parts[-1]:
            raise ProjectionError("Unable to determine the Cython class name.")
        return " ".join(parts)

    def _project_py_args(self, args: Node | list[Node] | None) -> str:
        if args is None:
            return ""
        if isinstance(args, list):
            arg_nodes = args
        else:
            arg_nodes = getattr(args, "args", None) or []
        items: list[str] = []
        for arg in arg_nodes:
            items.append(self._project_py_arg(arg))
        return ", ".join(items)

    def _project_py_arg(self, arg: Node) -> str:
        name = getattr(arg, "name", None)
        if not isinstance(name, str):
            declarator = getattr(arg, "declarator", None)
            name = self._format_c_declarator(declarator)
        if not isinstance(name, str) or not name:
            raise ProjectionError(f"Unsupported Python argument node: {arg!r}")
        text = self._make_value_token(
            " ".join(
                part
                for part in (
                    self._format_c_base_type(getattr(arg, "base_type", None)),
                    name,
                )
                if part
            )
        ) if getattr(arg, "base_type", None) is not None else name
        annotation = getattr(arg, "annotation", None)
        if annotation is not None:
            text = f"{text}: {self._project_expr(annotation)}"
        default = getattr(arg, "default", None)
        if default is not None:
            text = f"{text}={self._project_expr(default)}"
        if getattr(arg, "is_star_arg", False):
            text = f"*{text}"
        if getattr(arg, "is_starstar_arg", False):
            text = f"**{text}"
        return text

    def _project_c_arg(self, arg: Node) -> str:
        if getattr(arg, "annotation", None) is not None:
            raise ProjectionError("Annotated Cython arguments are not supported yet.")
        if getattr(arg, "kw_only", False):
            raise ProjectionError("Keyword-only Cython arguments are not supported yet.")
        if getattr(arg, "not_none", False) or getattr(arg, "or_none", False):
            raise ProjectionError("Cython noneability markers are not supported yet.")

        base_type = self._format_c_base_type(getattr(arg, "base_type", None))
        declarator = self._format_c_declarator(getattr(arg, "declarator", None))
        raw = " ".join(part for part in (base_type, declarator) if part)
        if not raw:
            raise ProjectionError(f"Unsupported Cython argument node: {arg!r}")

        token = self._make_value_token(raw)
        default = getattr(arg, "default", None)
        if default is not None:
            return f"{token}={self._project_expr(default)}"
        return token

    def _make_header_token(self, value: str) -> str:
        prefix = "def "
        return self._make_prefixed_token(prefix, value)

    def _make_class_header_token(self, value: str) -> str:
        prefix = "class "
        return self._make_prefixed_token(prefix, value)

    def _make_prefixed_token(self, prefix: str, value: str) -> str:
        token_length = len(value) - len(prefix)
        if token_length < 1:
            raise ProjectionError(f"Cannot project Cython header: {value!r}")
        token = self._factory.make(token_length)
        self._replacements.append(Replacement(f"{prefix}{token}", value))
        return token

    def _make_value_token(self, value: str) -> str:
        token = self._factory.make(len(value))
        self._replacements.append(Replacement(token, value))
        return token

    def _add_exact_replacement(self, surrogate: str, original: str) -> None:
        self._replacements.append(Replacement(surrogate, original))

    def _emit_compound_statement(self, text: str) -> str:
        return text

    def _project_docstring(self, doc: str | None, *, indent: int) -> str:
        if not doc:
            return ""
        return f"{_INDENT * indent}{doc!r}\n"

    def _project_c_var_header(self, node: Node, declarator: object) -> str:
        parts = ["cdef"]
        visibility = getattr(node, "visibility", None)
        if visibility and visibility != "private":
            parts.append(visibility)
        if getattr(node, "api", False):
            parts.append("api")
        base_type = self._format_c_base_type(getattr(node, "base_type", None))
        if base_type:
            parts.append(base_type)
        name = self._format_c_declarator(declarator)
        if not name:
            raise ProjectionError("Unable to determine the Cython variable name.")
        parts.append(name)
        return " ".join(parts)

    def _project_expr(self, node: Node) -> str:
        method = getattr(self, f"_expr_{type(node).__name__}", None)
        if method is None:
            raise ProjectionError(
                f"Unsupported Cython expression node: {type(node).__name__}"
            )
        return method(node)

    def _expr_NameNode(self, node: NameNode) -> str:
        return node.name

    def _expr_IntNode(self, node: Node) -> str:
        return str(node.value)

    def _expr_FloatNode(self, node: Node) -> str:
        return str(node.value)

    def _expr_BoolNode(self, node: Node) -> str:
        return "True" if node.value else "False"

    def _expr_NoneNode(self, node: Node) -> str:
        return "None"

    def _expr_StringNode(self, node: Node) -> str:
        value = getattr(node, "value", None)
        if not isinstance(value, str):
            raise ProjectionError(f"Unexpected StringNode value: {value!r}")
        return repr(value)

    def _expr_TupleNode(self, node: Node) -> str:
        items = [self._project_expr(arg) for arg in getattr(node, "args", None) or []]
        if len(items) == 1:
            return f"({items[0]},)"
        return f"({', '.join(items)})"

    def _expr_AttributeNode(self, node: Node) -> str:
        return f"{self._project_expr(node.obj)}.{node.attribute}"

    def _expr_IndexNode(self, node: Node) -> str:
        return f"{self._project_expr(node.base)}[{self._project_expr(node.index)}]"

    def _expr_SimpleCallNode(self, node: Node) -> str:
        args = ", ".join(self._project_expr(arg) for arg in getattr(node, "args", None) or [])
        return f"{self._project_expr(node.function)}({args})"

    def _expr_PrimaryCmpNode(self, node: Node) -> str:
        return (
            f"{self._project_expr(node.operand1)} {node.operator} "
            f"{self._project_expr(node.operand2)}"
        )

    def _expr_AddNode(self, node: Node) -> str:
        return self._project_binary(node, "+")

    def _expr_SubNode(self, node: Node) -> str:
        return self._project_binary(node, "-")

    def _expr_MulNode(self, node: Node) -> str:
        return self._project_binary(node, "*")

    def _expr_DivNode(self, node: Node) -> str:
        return self._project_binary(node, "/")

    def _expr_ModNode(self, node: Node) -> str:
        return self._project_binary(node, "%")

    def _expr_PowNode(self, node: Node) -> str:
        return self._project_binary(node, "**")

    def _project_binary(self, node: Node, operator: str) -> str:
        return (
            f"{self._project_expr(node.operand1)} {operator} "
            f"{self._project_expr(node.operand2)}"
        )

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


def project_source(source: str) -> ProjectedSource:
    return SurrogateProjector(source).project()
