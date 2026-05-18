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


class TestDocstringIndentNormalisation:
    """Docstring re-indent passes; content change and plain strings still raise.

    Covers every position where a docstring can appear in Cython source.
    Each *_indent_normalised test uses 2-space vs 4-space indentation to mirror
    what Black produces when re-indenting a whole file.
    Each *_content_change_raises test confirms genuine edits are still caught.
    """

    def test_method_docstring_indent_normalised(self):
        a = 'class Foo:\n  def method(self):\n    """\n    A method.\n    """\n    pass\n'
        b = 'class Foo:\n    def method(self):\n        """\n        A method.\n        """\n        pass\n'
        assert_equivalent(a, b)

    def test_method_docstring_content_change_raises(self):
        a = 'class Foo:\n    def method(self):\n        """\n        A method.\n        """\n        pass\n'
        b = 'class Foo:\n    def method(self):\n        """\n        Different.\n        """\n        pass\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_cdef_func_docstring_indent_normalised(self):
        a = 'cdef int add(int a, int b):\n  """\n  A cdef function.\n  """\n  return a + b\n'
        b = 'cdef int add(int a, int b):\n    """\n    A cdef function.\n    """\n    return a + b\n'
        assert_equivalent(a, b)

    def test_cdef_func_docstring_content_change_raises(self):
        a = 'cdef int add(int a, int b):\n    """\n    A cdef function.\n    """\n    return a + b\n'
        b = 'cdef int add(int a, int b):\n    """\n    Different.\n    """\n    return a + b\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_cpdef_func_docstring_indent_normalised(self):
        a = 'cpdef int add(int a, int b):\n  """\n  A cpdef function.\n  """\n  return a + b\n'
        b = 'cpdef int add(int a, int b):\n    """\n    A cpdef function.\n    """\n    return a + b\n'
        assert_equivalent(a, b)

    def test_cpdef_func_docstring_content_change_raises(self):
        a = 'cpdef int add(int a, int b):\n    """\n    A cpdef function.\n    """\n    return a + b\n'
        b = 'cpdef int add(int a, int b):\n    """\n    Different.\n    """\n    return a + b\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_cdef_class_docstring_indent_normalised(self):
        a = 'cdef class Foo:\n  """\n  A cdef class.\n  """\n  pass\n'
        b = 'cdef class Foo:\n    """\n    A cdef class.\n    """\n    pass\n'
        assert_equivalent(a, b)

    def test_cdef_class_docstring_content_change_raises(self):
        a = 'cdef class Foo:\n    """\n    A cdef class.\n    """\n    pass\n'
        b = 'cdef class Foo:\n    """\n    Different.\n    """\n    pass\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_cython_property_docstring_indent_normalised(self):
        a = (
            'cdef class Foo:\n'
            '  property bar:\n'
            '    """\n'
            '    A property.\n'
            '    """\n'
            '    def __get__(self):\n'
            '      return 1\n'
        )
        b = (
            'cdef class Foo:\n'
            '    property bar:\n'
            '        """\n'
            '        A property.\n'
            '        """\n'
            '        def __get__(self):\n'
            '            return 1\n'
        )
        assert_equivalent(a, b)

    def test_cython_property_docstring_content_change_raises(self):
        a = (
            'cdef class Foo:\n'
            '    property bar:\n'
            '        """\n'
            '        A property.\n'
            '        """\n'
            '        def __get__(self):\n'
            '            return 1\n'
        )
        b = (
            'cdef class Foo:\n'
            '    property bar:\n'
            '        """\n'
            '        Different.\n'
            '        """\n'
            '        def __get__(self):\n'
            '            return 1\n'
        )
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_nested_class_docstring_indent_normalised(self):
        a = (
            'class Outer:\n'
            '  """\n'
            '  Outer doc.\n'
            '  """\n'
            '  class Inner:\n'
            '    """\n'
            '    Inner doc.\n'
            '    """\n'
            '    pass\n'
        )
        b = (
            'class Outer:\n'
            '    """\n'
            '    Outer doc.\n'
            '    """\n'
            '    class Inner:\n'
            '        """\n'
            '        Inner doc.\n'
            '        """\n'
            '        pass\n'
        )
        assert_equivalent(a, b)

    def test_nested_class_docstring_content_change_raises(self):
        a = 'class Outer:\n    """\n    Outer.\n    """\n    pass\n'
        b = 'class Outer:\n    """\n    Different.\n    """\n    pass\n'
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_python_property_docstring_indent_normalised(self):
        a = (
            'class Foo:\n'
            '  @property\n'
            '  def myprop(self):\n'
            '    """\n'
            '    A property.\n'
            '    """\n'
            '    return 1\n'
        )
        b = (
            'class Foo:\n'
            '    @property\n'
            '    def myprop(self):\n'
            '        """\n'
            '        A property.\n'
            '        """\n'
            '        return 1\n'
        )
        assert_equivalent(a, b)

    def test_python_property_docstring_content_change_raises(self):
        a = (
            'class Foo:\n'
            '    @property\n'
            '    def myprop(self):\n'
            '        """\n'
            '        A property.\n'
            '        """\n'
            '        return 1\n'
        )
        b = (
            'class Foo:\n'
            '    @property\n'
            '    def myprop(self):\n'
            '        """\n'
            '        Different.\n'
            '        """\n'
            '        return 1\n'
        )
        with pytest.raises(ASTDifference):
            assert_equivalent(a, b)

    def test_string_literal_in_class_body_not_normalised(self):
        # A plain string assignment inside a class is not a docstring and
        # must be compared byte-for-byte.
        with pytest.raises(ASTDifference):
            assert_equivalent(
                'class Foo:\n  x = "  hello  "\n',
                'class Foo:\n  x = "    hello    "\n',
            )

    def test_string_literal_in_method_body_not_normalised(self):
        with pytest.raises(ASTDifference):
            assert_equivalent(
                'def foo():\n  x = "  hello  "\n',
                'def foo():\n  x = "    hello    "\n',
            )

    def test_string_literal_in_cdef_func_body_not_normalised(self):
        with pytest.raises(ASTDifference):
            assert_equivalent(
                'cdef int foo():\n  x = "  hello  "\n  return 0\n',
                'cdef int foo():\n  x = "    hello    "\n  return 0\n',
            )


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
