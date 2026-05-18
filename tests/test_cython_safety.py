"""Tests for the vendored Cython AST equivalence checker (cython_safety.py).

Ported from cyblack/tests/test_safety.py; requires Cython to be installed.
"""

import pytest

pytest.importorskip("Cython", reason="Cython is required for cython_safety tests")

from black.cython_safety import (
    ASTDifference,
    assert_equivalent,
)


class TestAssertEquivalent:

    def test_identical_code_passes(self):
        code = "x = 1\n"
        assert_equivalent(code, code)

    def test_same_semantics_different_whitespace_passes(self):
        assert_equivalent("x = 1\n", "x = 1\n\n")

    def test_changed_name_raises(self):
        with pytest.raises(ASTDifference):
            assert_equivalent("x = 1\n", "y = 1\n")

    def test_changed_value_raises(self):
        with pytest.raises(ASTDifference):
            assert_equivalent("x = 1\n", "x = 2\n")

    def test_changed_operator_raises(self):
        with pytest.raises(ASTDifference):
            assert_equivalent("a < b\n", "a > b\n")

    def test_added_statement_raises(self):
        with pytest.raises(ASTDifference):
            assert_equivalent("x = 1\n", "x = 1\ny = 2\n")

    def test_removed_statement_raises(self):
        with pytest.raises(ASTDifference):
            assert_equivalent("x = 1\ny = 2\n", "x = 1\n")

    def test_multiline_code_passes(self):
        code = "def foo():\n    return 1\n"
        assert_equivalent(code, code)

    def test_cython_code_passes(self):
        code = "cdef int x = 0\n"
        assert_equivalent(code, code)

    def test_docstring_indent_normalised(self):
        a = 'def foo():\n    """Hello.\n\n    World.\n    """\n    pass\n'
        b = 'def foo():\n    """Hello.\n\n        World.\n        """\n    pass\n'
        assert_equivalent(a, b)

    def test_class_docstring_indent_normalised(self):
        # Black re-indents a class with 2-space indentation to 4-space, which
        # changes the raw string content of the docstring UnicodeNode.
        a = 'class Foo:\n  """\n  Body.\n  """\n  pass\n'
        b = 'class Foo:\n    """\n    Body.\n    """\n    pass\n'
        assert_equivalent(a, b)

    def test_class_docstring_content_change_raises(self):
        a = 'class Foo:\n    """\n    Body.\n    """\n    pass\n'
        b = 'class Foo:\n    """\n    Different body.\n    """\n    pass\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_string_literal_content_change_raises(self):
        # Normalisation must not apply to regular string literals, only docstrings.
        with pytest.raises(ASTDifference):
            assert_equivalent('x = "  hello  "\n', 'x = "    hello    "\n')

    def test_error_message_contains_diff(self):
        with pytest.raises(ASTDifference, match=r"(?i)(source|differ)"):
            assert_equivalent("x = 1\n", "y = 1\n")

    def test_cimport_spacing_passes(self):
        code1 = "from libc.math cimport  sqrt,  cos\n"
        code2 = "from libc.math cimport sqrt, cos\n"
        assert_equivalent(code1, code2)

    def test_cimport_name_change_raises(self):
        code1 = "from libc.math cimport sqrt, cos\n"
        code2 = "from libc.math cimport sqrt, sin\n"
        with pytest.raises(ASTDifference):
            assert_equivalent(code1, code2)

    def test_lambda_same(self):
        code1 = "lambda x:   x**2"
        code2 = "lambda x: x ** 2"
        assert_equivalent(code1, code2)

    def test_lambda_diff(self):
        code1 = "lambda x:   x**2"
        code2 = "lambda y: y ** 2"
        with pytest.raises(ASTDifference):
            assert_equivalent(code1, code2)

    def test_annotation_same(self):
        assert_equivalent("A:int=5", "A: int = 5")

    def test_annotation_diff(self):
        with pytest.raises(ASTDifference):
            assert_equivalent("A: float = 5", "A: int = 5")


class TestAssertEquivalentIncludes:

    def test_same_include_passes(self):
        code = 'include "defs.pxi"\nx = 1\n'
        assert_equivalent(code, code)

    def test_include_whitespace_difference_passes(self):
        a = 'include "defs.pxi"\nx = 1\n'
        b = 'include "defs.pxi"\nx  =  1\n'
        assert_equivalent(a, b)

    def test_different_include_filename_raises(self):
        a = 'include "defs.pxi"\nx = 1\n'
        b = 'include "defz.pxi"\nx = 1\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_include_present_vs_absent_raises(self):
        a = 'include "defs.pxi"\nx = 1\n'
        b = 'x = 1\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)
