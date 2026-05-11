"""Phase 0 characterization tests for .pyx/.pxd/.pxi file handling.

These tests lock in Black's *existing* behaviour for Cython-suffixed files before
any code changes are made.  They are green against unmodified Black 26.3.1 and exist
as regression sentinels.  Any later change that flips one of these tests must call it
out explicitly in extend-black-cython-changelog.md.

Test 5 (cython_syntax_pyx_raises_invalid_input) is expected to be updated by Phase 1
when the Cython pipeline is wired up: the error shape will change from InvalidInput
to either a successful format (Cython installed) or a missing-dep error.
"""

import re
from pathlib import Path

import pytest

import black
from black.const import DEFAULT_EXCLUDES, DEFAULT_INCLUDES
from black.report import Report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_include():
    return black.re_compile_maybe_verbose(DEFAULT_INCLUDES)


def _default_exclude():
    return black.re_compile_maybe_verbose(DEFAULT_EXCLUDES)


# ---------------------------------------------------------------------------
# Test 1 — directory walk skips Cython suffixes with default include regex
# ---------------------------------------------------------------------------

def test_directory_walk_skips_cython_suffixes(tmp_path: Path) -> None:
    """gen_python_files with the default include regex yields only .py, not .pyx/.pxd/.pxi."""
    (tmp_path / "foo.pyx").write_text("cdef int x = 1\n")
    (tmp_path / "bar.pxd").write_text("cdef int y\n")
    (tmp_path / "baz.pxi").write_text("# include file\n")
    (tmp_path / "quux.py").write_text("x = 1\n")

    sources = list(
        black.gen_python_files(
            tmp_path.iterdir(),
            tmp_path,
            _default_include(),
            _default_exclude(),
            None,
            None,
            Report(),
            None,
            verbose=False,
            quiet=False,
        )
    )
    assert sources == [tmp_path / "quux.py"]


# ---------------------------------------------------------------------------
# Test 2 — --include opt-in works for .pyx
# ---------------------------------------------------------------------------

def test_include_optin_works_for_pyx(tmp_path: Path) -> None:
    """gen_python_files with include=r'\\.pyx$' yields only the .pyx file."""
    (tmp_path / "foo.pyx").write_text("cdef int x = 1\n")
    (tmp_path / "bar.pxd").write_text("cdef int y\n")
    (tmp_path / "baz.pxi").write_text("# include file\n")
    (tmp_path / "quux.py").write_text("x = 1\n")

    sources = list(
        black.gen_python_files(
            tmp_path.iterdir(),
            tmp_path,
            re.compile(r"\.pyx$"),
            _default_exclude(),
            None,
            None,
            Report(),
            None,
            verbose=False,
            quiet=False,
        )
    )
    assert sources == [tmp_path / "foo.pyx"]


# ---------------------------------------------------------------------------
# Test 3 — direct CLI accepts a .pyx path regardless of include regex
# ---------------------------------------------------------------------------

def test_direct_cli_accepts_pyx_path(tmp_path: Path) -> None:
    """get_sources with a direct .pyx file path includes it, ignoring the include regex."""
    pyx_file = tmp_path / "foo.pyx"
    pyx_file.write_text("cdef int x = 1\n")

    sources = black.get_sources(
        root=tmp_path,
        src=(str(pyx_file),),
        quiet=False,
        verbose=False,
        include=_default_include(),
        exclude=None,
        extend_exclude=None,
        force_exclude=None,
        report=Report(),
        stdin_filename=None,
    )
    assert pyx_file in sources


# ---------------------------------------------------------------------------
# Test 4 — pure-Python .pyx reformats successfully
# ---------------------------------------------------------------------------

def test_pure_python_pyx_reformats(tmp_path: Path) -> None:
    """format_file_in_place on a .pyx containing valid Python does not raise."""
    pyx_file = tmp_path / "pure.pyx"
    # Deliberately unformatted Python so Black has something to do.
    pyx_file.write_text("x = {'a':1,'b':2}\n")

    result = black.format_file_in_place(
        pyx_file,
        fast=True,
        mode=black.Mode(),
        write_back=black.WriteBack.NO,
    )
    # result is True (changed) or False (unchanged); either is fine — no exception.
    assert result in (True, False)


# ---------------------------------------------------------------------------
# Test 5 — Cython-syntax .pyx raises an error (shape evolves across phases)
# ---------------------------------------------------------------------------

# History:
#   Phase 0 (baseline): raised InvalidInput — Black's Python parser rejected cdef.
#   Phase 1: still raised InvalidInput — stub masker was a no-op.
#   Phase 2 Step 1: raises NotImplementedError from validate_cython_subset
#     because 'cdef' was not yet handled by the masker.
#   Phase 2 Step 2: cdef variable declarations handled; test moved to cdef function.
#   Phase 2 Step 3: cdef/cpdef functions handled; test moved to ctypedef.
#   Phase 2 Step 7: simple ctypedef aliases handled; test moved to ctypedef struct.
#   Phase 2 Step 9: ctypedef/cdef struct blocks handled; test moved to cdef extern.
#   Phase 2 Step 11: cdef extern from blocks handled; test moved to ctypedef fused.
#   Phase 2 Step 16: ctypedef fused blocks handled; test moved to ctypedef func-ptr.
#   Phase 2 Step 23: ctypedef func-ptr handled; test moved to Cython for-from loop.
#   Phase 2 Step 24: for-from loop handled; test moved to body-level C cast.
#   Phase 2 Step 28: C cast expressions handled; test moved to forward declaration.
#   Phase 2 Step 31: forward declarations handled; test moved to sizeof(void*).
def test_cython_syntax_pyx_raises_for_unhandled_construct(tmp_path: Path) -> None:
    """format_file_in_place on a .pyx with an unhandled cdef construct raises."""
    pyx_file = tmp_path / "cython_syntax.pyx"
    # sizeof() with a C pointer type: 'sizeof(void*)' tokenizes as NAME '(' NAME '*'
    # ')' — the '*' without a right operand in binary position is rejected by
    # lib2to3.  Will flip when body-level sizeof / C-pointer-type masking lands.
    pyx_file.write_text("cdef void foo():\n    x = sizeof(void*)\n")

    # Parser rejects 'void*' (binary * without right operand) as invalid Python.
    with pytest.raises(black.parsing.InvalidInput):
        black.format_file_in_place(
            pyx_file,
            fast=True,
            mode=black.Mode(),
            write_back=black.WriteBack.NO,
        )
