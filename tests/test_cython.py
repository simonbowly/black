"""Tests for Cython-mode formatting behaviour.

Mirrors the pattern of test_ipynb.py.  Module-level mark means these tests
only run when --run-optional=cython is passed.  Cython must be installed.
"""

import pathlib
import textwrap
from dataclasses import replace
from unittest.mock import patch

import pytest

import black
from black import Mode, NothingChanged, format_cython_string, format_str
from black.handle_cython import cython_dependencies_are_installed

pytestmark = pytest.mark.cython
pytest.importorskip("Cython", reason="Cython is required for cython tests")

CYTHON_MODE = Mode(is_cython=True)


# ---------------------------------------------------------------------------
# Acceptance criterion: pure-Python source routes through try-Python-first
# ---------------------------------------------------------------------------

def test_format_cython_string_pure_python_matches_format_str() -> None:
    """format_cython_string on valid-Python source must produce the same output
    as format_str.  This is the Phase 1 acceptance criterion: the try-Python-first
    path handles pure-Python .pyx without ever touching the masker.
    """
    src = "x = {'a':1,'b':2}\n"
    mode = CYTHON_MODE
    assert format_cython_string(src, fast=True, mode=mode) == format_str(
        src, mode=replace(mode, is_cython=False)
    )


def test_format_cython_string_already_formatted_raises_nothing_changed() -> None:
    """format_cython_string raises NothingChanged for already-formatted Python source."""
    src = 'x = {"a": 1, "b": 2}\n'
    with pytest.raises(NothingChanged):
        format_cython_string(src, fast=True, mode=CYTHON_MODE)


# ---------------------------------------------------------------------------
# Directory walk still ignores .pyx files even with Cython installed
# ---------------------------------------------------------------------------

def test_directory_walk_still_ignores_pyx_with_cython_installed(
    tmp_path: pathlib.Path,
) -> None:
    """gen_python_files with default include still yields only .py files, even
    when Cython is installed.  The Cython pipeline must not change discovery.
    """
    import re
    from black import gen_python_files
    from black.const import DEFAULT_EXCLUDES, DEFAULT_INCLUDES
    from black.report import Report

    (tmp_path / "foo.pyx").write_text("cdef int x = 1\n")
    (tmp_path / "bar.py").write_text("x = 1\n")
    sources = list(
        gen_python_files(
            tmp_path.iterdir(),
            tmp_path,
            re.compile(DEFAULT_INCLUDES),
            re.compile(DEFAULT_EXCLUDES),
            None,
            None,
            Report(),
            None,
            verbose=False,
            quiet=False,
        )
    )
    assert sources == [tmp_path / "bar.py"]


# ---------------------------------------------------------------------------
# Phase 2: fixture-based formatting tests (mask → format → restore)
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(__file__).parent / "data" / "cython"


def _format_fixture(name: str) -> str:
    """Format tests/data/cython/<name>.pyx and assert Cython AST equivalence.

    Mirrors Black's _assert_format_inner pattern: call the core formatting
    function directly, then run the safety check as an explicit separate step.
    """
    from black.cython_safety import assert_equivalent

    src = (DATA_DIR / f"{name}.pyx").read_text()
    dst = format_cython_string(src, fast=False, mode=CYTHON_MODE)
    if src != dst:
        assert_equivalent(src, dst)
    return dst


def _expected(name: str) -> str:
    return (DATA_DIR / f"{name}.expected.pyx").read_text()


def test_cimport_formatting() -> None:
    """from X cimport Y and standalone cimport X are masked, formatted, and restored.

    Phase 2 Step 1: cimport constructs only.
    """
    assert _format_fixture("cimport") == _expected("cimport")


def test_cdef_variable_formatting() -> None:
    """cdef TYPE VAR [= EXPR] declarations are masked, formatted, and restored.

    Phase 2 Step 2: simple cdef variable declarations (no function headers,
    no cdef class, no multi-variable declarations).
    """
    assert _format_fixture("cdef_vars") == _expected("cdef_vars")


def test_cdef_function_formatting() -> None:
    """cdef/cpdef function headers with typed arguments are masked, formatted,
    and restored.  Phase 2 Step 3.
    """
    assert _format_fixture("cdef_functions") == _expected("cdef_functions")


def test_cdef_class_formatting() -> None:
    """cdef class headers are masked as 'class __cy_class_*', formatted, and
    restored.  Phase 2 Step 4.
    """
    assert _format_fixture("cdef_class") == _expected("cdef_class")


def test_typed_def_formatting() -> None:
    """Typed arguments in plain def functions are masked, formatted, and restored.
    Phase 2 Step 5.
    """
    assert _format_fixture("typed_def") == _expected("typed_def")


def test_memoryview_formatting() -> None:
    """Memoryview type syntax (double[:], int[:, :]) in def/cdef args and cdef
    variable declarations is masked, formatted, and restored.  Phase 2 Step 6.
    """
    assert _format_fixture("memoryview") == _expected("memoryview")


def test_ctypedef_formatting() -> None:
    """Simple ctypedef TYPE ALIAS declarations are masked, formatted, and
    restored.  Phase 2 Step 7.
    """
    assert _format_fixture("ctypedef") == _expected("ctypedef")


def test_nogil_formatting() -> None:
    """nogil and with gil function-header postfixes are masked via a comment
    anchor, formatted, and restored.  Phase 2 Step 8.
    """
    assert _format_fixture("nogil") == _expected("nogil")


def test_cdef_struct_formatting() -> None:
    """cdef struct/union/enum and ctypedef struct/union/enum block headers are
    masked as 'class __cy_KEYWORD_Name:', formatted, and restored.
    Phase 2 Step 9.
    """
    assert _format_fixture("cdef_struct") == _expected("cdef_struct")


def test_cdef_block_formatting() -> None:
    """cdef: compound block is masked as 'if __cy_cdef_block:', formatted, and
    restored.  Phase 2 Step 10.
    """
    assert _format_fixture("cdef_block") == _expected("cdef_block")


def test_multi_cdef_formatting() -> None:
    """cdef TYPE var1, var2, var3 [= expr] declarations are masked as a single
    placeholder spanning the whole declaration, formatted, and restored.
    Phase 2 Step 12.
    """
    assert _format_fixture("multi_cdef") == _expected("multi_cdef")


def test_cdef_extern_formatting() -> None:
    """cdef extern from "h": header is masked as 'class __cy_extern_NAME', body
    declarations are handled by the bare NAME NAME handler, formatted, and
    restored.  Phase 2 Step 11.
    """
    assert _format_fixture("cdef_extern") == _expected("cdef_extern")


def test_cdef_extern_wildcard_formatting() -> None:
    """cdef extern from *: header is masked as 'class __cy_extern_wildcard', body
    declarations are handled by existing handlers, formatted, and restored.
    Phase 2 Step 13.
    """
    assert _format_fixture("cdef_extern_wildcard") == _expected("cdef_extern_wildcard")


def test_except_postfix_formatting() -> None:
    """except -1 / except * / except ?-1 and combinations with nogil/with gil
    in function headers are masked via a comment anchor and restored.
    Phase 2 Step 14.
    """
    assert _format_fixture("except_postfix") == _expected("except_postfix")


def test_anon_enum_formatting() -> None:
    """Anonymous cdef enum: blocks (no name) are masked as 'class __cy_cdef_anon_enum:',
    formatted, and restored.  Phase 2 Step 16.
    """
    assert _format_fixture("anon_enum") == _expected("anon_enum")


def test_cpdef_enum_formatting() -> None:
    """cpdef enum [Name]: blocks are masked as 'class __cy_cpdef_enum_*:', formatted,
    and restored.  Phase 2 Step 16.
    """
    assert _format_fixture("cpdef_enum") == _expected("cpdef_enum")


def test_fused_types_formatting() -> None:
    """ctypedef fused Name: and cdef fused Name: block headers are masked as
    'class __cy_*_fused_Name:', formatted, and restored.  Phase 2 Step 16.
    """
    assert _format_fixture("fused_types") == _expected("fused_types")


def test_include_directive_formatting() -> None:
    """include "file.pyx" statements are masked as an identifier placeholder,
    formatted, and restored.  Phase 2 Step 18.
    """
    assert _format_fixture("include_directive") == _expected("include_directive")


def test_extern_anon_enum_formatting() -> None:
    """Bare enum: blocks inside cdef extern bodies (no cdef/cpdef prefix) are
    masked as 'if __cy_anon_enum:', formatted, and restored.  Phase 2 Step 18.
    """
    assert _format_fixture("extern_anon_enum") == _expected("extern_anon_enum")


def test_ctypedef_class_formatting() -> None:
    """ctypedef class DOTTED_NAME [object PyType]: headers and ctypedef struct/union/enum
    TypeName (forward declarations without body) are masked, formatted, and restored.
    Phase 2 Step 19.
    """
    assert _format_fixture("ctypedef_class") == _expected("ctypedef_class")


def test_cdef_extern_namespace_formatting() -> None:
    """cdef extern from "h" namespace "ns": and cdef extern from * namespace "ns":
    headers absorb the optional namespace clause into the placeholder span.
    Phase 2 Step 20.
    """
    assert _format_fixture("cdef_extern_namespace") == _expected("cdef_extern_namespace")


def test_array_typed_arg_formatting() -> None:
    """Array-typed args like 'char msg[]' and 'const char a[]' must absorb the
    trailing [] into the placeholder span so the masked source is valid Python.
    Phase 2 Step 21.
    """
    assert _format_fixture("array_arg") == _expected("array_arg")


def test_vararg_formatting() -> None:
    """Variadic '...' args in cdef/def parameter lists must be masked as
    __cy_varargs so the masked source is valid Python (bare '...' in a parameter
    list is invalid Python).  Phase 2 Step 22.
    """
    assert _format_fixture("vararg") == _expected("vararg")


def test_ctypedef_funcptr_formatting() -> None:
    """ctypedef RETURN_TYPE (*name)(args) function-pointer typedefs are masked
    as a single identifier-only placeholder spanning the whole declaration.
    Phase 2 Step 23.
    """
    assert _format_fixture("ctypedef_funcptr") == _expected("ctypedef_funcptr")


def test_for_from_formatting() -> None:
    """Cython 'for VAR from BOUNDS [by STEP]:' loops are masked by replacing
    'from BOUNDS [by STEP]' with 'in __cy_for_from_VAR' so Black sees a valid
    Python for-in loop.  Phase 2 Step 24.
    """
    assert _format_fixture("for_from") == _expected("for_from")


def test_cdef_qualified_block_formatting() -> None:
    """cdef readonly: and cdef public: block headers (qualified cdef blocks)
    are masked as 'if __cy_cdef_block:', formatted, and restored.
    Phase 2 Step 25.
    """
    assert _format_fixture("cdef_qualified_block") == _expected("cdef_qualified_block")


def test_ctypedef_memview_formatting() -> None:
    """ctypedef TYPE[:, ::1] ALIAS memoryview typedefs are masked as a single
    identifier-only placeholder spanning the full declaration.
    Phase 2 Step 26.
    """
    assert _format_fixture("ctypedef_memview") == _expected("ctypedef_memview")


def test_paren_return_type_formatting() -> None:
    """cdef/cpdef with a parenthesized return type (tuple types like
    '(int, double)' or pointer types like '(char*)') are masked by treating
    the '(...)' group as part of the return-type expression and absorbing it
    into the function-header placeholder span.  Phase 2 Step 27.
    """
    assert _format_fixture("paren_return_type") == _expected("paren_return_type")


def test_c_cast_formatting() -> None:
    """C-style cast expressions <TYPE>EXPR are masked, formatted, and restored.

    NAME absorption: <int>x → __cy_cast_int_x.
    Function-call form: <double>(expr) → __cy_cast_double(expr).
    Minus form: <unsigned char>-1 → __cy_cast_unsigned_char -1 (valid subtraction).
    String absorption: <char>'A' → __cy_cast_char.
    Phase 2 Step 28.
    """
    assert _format_fixture("c_cast") == _expected("c_cast")


def test_addrof_formatting() -> None:
    """Address-of operator &VAR and &(EXPR) are masked, formatted, and restored.

    Name absorption: &x → __cy_addrof_x (attribute access .attr follows naturally).
    Paren form: &(expr) → __cy_addrof(expr).
    Phase 2 Step 28.
    """
    assert _format_fixture("addrof") == _expected("addrof")


def test_cdef_extern_forward_decl_formatting() -> None:
    """cdef/cpdef function declarations inside cdef extern from blocks (no colon,
    no body) are masked as a single identifier-only placeholder rather than a
    def-skeleton, so the masked source does not contain a bare 'def func(args)'
    without a trailing colon.  Phase 2 Step 31.
    """
    assert _format_fixture("cdef_extern_forward_decl") == _expected(
        "cdef_extern_forward_decl"
    )


def test_c_char_literal_formatting() -> None:
    """Cython C character literals c'X' are masked as __cy_char_X identifiers
    so Black can process the surrounding Python expression.  Phase 2 Step 30.
    """
    assert _format_fixture("c_char") == _expected("c_char")


def test_cdef_extern_nogil_formatting() -> None:
    """cdef extern from "h" nogil: and cdef extern from * namespace "ns" nogil:
    headers absorb the trailing nogil qualifier into the placeholder span so it
    does not leak into the masked Python as 'class __cy_extern_X nogil:'.
    Phase 2 Step 29.
    """
    assert _format_fixture("cdef_extern_nogil") == _expected("cdef_extern_nogil")


def test_cimport_multiline_formatting() -> None:
    """Multi-line 'from X cimport (\\n    a, b,\\n)' forms are handled by tracking
    parenthesis depth in the cimport name-collection loop so the closing ')' is
    included in the masked span.  Phase 2 Step 32.
    """
    assert _format_fixture("cimport_multiline") == _expected("cimport_multiline")


def test_ctypedef_c_alias_formatting() -> None:
    """ctypedef TYPE ALIAS "C-name" and enum body 'VALUE "rename"' aliases absorb
    the trailing C-name string into the placeholder span.  Phase 2 Step 33.
    """
    assert _format_fixture("ctypedef_c_alias") == _expected("ctypedef_c_alias")


def test_func_c_name_alias_formatting() -> None:
    """cdef int foo "C_name"(args) forward declarations absorb the C-name alias
    string between the function name and the opening '('.  Phase 2 Step 34.
    """
    assert _format_fixture("func_c_name_alias") == _expected("func_c_name_alias")


def test_unmask_cython_handles_black_wrapped_import() -> None:
    """When a masked 'from X import __cy_cimport_...' line exceeds 88 chars,
    Black wraps it as 'import (\\n    __cy_...,\\n)'.  The unmasker must detect
    this form and restore the original cimport statement.  Phase 2 Step 17.
    """
    from black.handle_cython_syntax import mask_cython, unmask_cython

    src = (
        "from libcpp.algorithm cimport "
        "is_sorted, sort, stable_sort, nth_element, all_of, count, copy\n"
    )
    masked, replacements = mask_cython(src)
    formatted = format_str(masked, mode=replace(CYTHON_MODE, is_cython=False))
    result = unmask_cython(formatted, replacements)
    assert result == src


# ---------------------------------------------------------------------------
# Phase 2: validate_cython_subset rejects unhandled constructs and __cy_ ids
# ---------------------------------------------------------------------------

def test_validate_accepts_cdef_variable() -> None:
    """validate_cython_subset passes for simple cdef variable declarations.

    Phase 2 Step 2: cdef variables are handled, so validate no longer raises.
    """
    from black.handle_cython_syntax import validate_cython_subset

    validate_cython_subset("cdef int x = 1\ncdef double y\n")


def test_validate_accepts_cpdef_function() -> None:
    """validate_cython_subset passes for cpdef function headers.

    Phase 2 Step 3: cpdef functions are now handled.
    """
    from black.handle_cython_syntax import validate_cython_subset

    validate_cython_subset("cpdef int foo(int x):\n    return x\n")


def test_validate_accepts_cdef_class() -> None:
    """validate_cython_subset passes for cdef class declarations.

    Phase 2 Step 4: cdef class headers are handled by the masker.
    """
    from black.handle_cython_syntax import validate_cython_subset

    validate_cython_subset("cdef class Foo:\n    pass\n")


def test_validate_accepts_ctypedef_alias() -> None:
    """validate_cython_subset passes for simple ctypedef alias declarations.

    Phase 2 Step 7: ctypedef TYPE ALIAS is now handled.
    """
    from black.handle_cython_syntax import validate_cython_subset

    validate_cython_subset("ctypedef int MyInt\nctypedef double Scalar\n")


def test_validate_rejects_cy_prefix() -> None:
    """validate_cython_subset raises if source contains any __cy_* identifier."""
    from black.handle_cython_syntax import validate_cython_subset

    with pytest.raises(ValueError, match="__cy_"):
        validate_cython_subset("__cy_reserved = 1\n")


def test_validate_accepts_pure_cimport() -> None:
    """validate_cython_subset passes for cimport-only Cython constructs."""
    from black.handle_cython_syntax import validate_cython_subset

    validate_cython_subset(
        textwrap.dedent("""\
        from libc.math cimport sin, cos
        cimport numpy
        x = 1
        """)
    )


# ---------------------------------------------------------------------------
# Cython parse gate: unparseable Cython source fails before masking
# ---------------------------------------------------------------------------

def test_format_cython_string_bad_cython_parse_fails_at_gate(
    tmp_path: pathlib.Path,
) -> None:
    """format_file_in_place on source that is invalid both as Python and as
    Cython should fail cleanly at the Cython parse gate (before any masking).
    """
    pyx_file = tmp_path / "bad.pyx"
    # This is invalid Python (cdef) and invalid Cython (cdef with broken syntax)
    pyx_file.write_text("cdef int x =\n")
    with pytest.raises(Exception):
        black.format_file_in_place(
            pyx_file,
            fast=True,
            mode=black.Mode(),
            write_back=black.WriteBack.NO,
        )


# ---------------------------------------------------------------------------
# CLI path: format_file_in_place with fast=False exercises the full pipeline
# ---------------------------------------------------------------------------

def test_format_file_in_place_pyx_runs_safety_checks(
    tmp_path: pathlib.Path,
) -> None:
    """format_file_in_place with fast=False (the default CLI path) formats a
    Cython .pyx file and runs the safety checks without crashing.
    """
    pyx = tmp_path / "test.pyx"
    pyx.write_text("def add(int a, int b):\n    return a+b\n")
    changed = black.format_file_in_place(
        pyx, fast=False, mode=black.Mode(), write_back=black.WriteBack.YES
    )
    assert changed is True
    assert pyx.read_text() == "def add(int a, int b):\n    return a + b\n"


# ---------------------------------------------------------------------------
# Safety check wiring: assert_equivalent is called and its errors propagate
# ---------------------------------------------------------------------------

@pytest.mark.incompatible_with_mypyc
def test_cython_safety_check_is_called_by_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    """format_file_in_place propagates errors from cython_safety.assert_equivalent.

    Mirrors test_black.py::test_code_option_safe: patches assert_equivalent to
    raise so we can confirm it is wired into the pipeline for .pyx files, not
    silently skipped.
    """
    import black.cython_safety as cython_safety

    pyx = tmp_path / "test.pyx"
    pyx.write_text("def add(int a, int b):\n    return a+b\n")
    with patch.object(
        cython_safety, "assert_equivalent", side_effect=AssertionError("mocked check")
    ):
        with pytest.raises(AssertionError, match="mocked check"):
            black.format_file_in_place(
                pyx, fast=False, mode=black.Mode(), write_back=black.WriteBack.NO
            )


@pytest.mark.incompatible_with_mypyc
def test_cython_safety_check_catches_ast_corruption(
    tmp_path: pathlib.Path,
) -> None:
    """The safety check catches a real AST change introduced by a buggy unmask.

    Patches unmask_cython to rename a function in its output, simulating a
    masking/unmasking bug.  Verifies that cython_safety.assert_equivalent
    raises, confirming the check would catch real corruption in practice.
    """
    pyx = tmp_path / "test.pyx"
    pyx.write_text("cdef int foo(int x):\n    return x\n")

    real_unmask = black.unmask_cython

    def corrupting_unmask(src: str, replacements: list) -> str:
        return real_unmask(src, replacements).replace("foo", "CORRUPTED")

    from black.cython_safety import ASTDifference

    with patch.object(black, "unmask_cython", side_effect=corrupting_unmask):
        with pytest.raises(ASTDifference, match="foo.*CORRUPTED"):
            black.format_file_in_place(
                pyx, fast=False, mode=black.Mode(), write_back=black.WriteBack.NO
            )
