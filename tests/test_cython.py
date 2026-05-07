"""Tests for Cython-mode formatting behaviour.

Mirrors the pattern of test_ipynb.py.  Module-level mark means these tests
only run when --run-optional=cython is passed.  Cython must be installed.
"""

import pathlib
import textwrap
from dataclasses import replace

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
    """Run format_cython_string on tests/data/cython/<name>.pyx and return result."""
    src = (DATA_DIR / f"{name}.pyx").read_text()
    return format_cython_string(src, fast=False, mode=CYTHON_MODE)


def _expected(name: str) -> str:
    return (DATA_DIR / f"{name}.expected.pyx").read_text()


def test_cimport_formatting() -> None:
    """from X cimport Y and standalone cimport X are masked, formatted, and restored.

    Phase 2 Step 1: cimport constructs only.
    """
    assert _format_fixture("cimport") == _expected("cimport")


# ---------------------------------------------------------------------------
# Phase 2: validate_cython_subset rejects unhandled constructs and __cy_ ids
# ---------------------------------------------------------------------------

def test_validate_rejects_cdef() -> None:
    """validate_cython_subset raises for cdef (not yet handled)."""
    from black.handle_cython_syntax import validate_cython_subset

    with pytest.raises(NotImplementedError, match="cdef"):
        validate_cython_subset("cdef int x = 1\n")


def test_validate_rejects_cpdef() -> None:
    """validate_cython_subset raises for cpdef (not yet handled)."""
    from black.handle_cython_syntax import validate_cython_subset

    with pytest.raises(NotImplementedError, match="cpdef"):
        validate_cython_subset("cpdef int foo(int x):\n    return x\n")


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
