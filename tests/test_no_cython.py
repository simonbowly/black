"""Tests for Cython-mode behaviour when Cython is NOT installed.

Mirrors the pattern of test_no_ipynb.py.  The module-level mark means these
tests run only when the `cython` optional-test is NOT selected.  When
--run-optional=cython is passed the tests are skipped automatically.
"""

import pathlib
from dataclasses import replace
from unittest.mock import patch

import pytest

import black
from black.handle_cython import cython_dependencies_are_installed

pytestmark = pytest.mark.no_cython


def _clear_cache() -> None:
    cython_dependencies_are_installed.cache_clear()


# ---------------------------------------------------------------------------
# Cython-syntax .pyx with Cython missing → clear missing-dep error
# ---------------------------------------------------------------------------

def test_cython_syntax_pyx_missing_dep_not_invalid_input(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format_file_in_place on Cython-syntax .pyx with Cython absent raises
    a descriptive error about the missing dependency, not InvalidInput.

    This is the key behaviour change from Phase 1: the error message now tells
    the user how to fix the problem, rather than silently failing with an
    unrelated parse error.
    """
    _clear_cache()
    monkeypatch.setattr("black.cython_dependencies_are_installed", lambda warn: False)
    pyx_file = tmp_path / "cython_syntax.pyx"
    pyx_file.write_text("cdef int x = 1\n")
    with pytest.raises(ValueError, match="Cython is not installed"):
        black.format_file_in_place(
            pyx_file,
            fast=True,
            mode=black.Mode(),
            write_back=black.WriteBack.NO,
        )


# ---------------------------------------------------------------------------
# Pure-Python .pyx still formats fine without Cython installed
# ---------------------------------------------------------------------------

def test_pure_python_pyx_still_formats_without_cython(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """format_file_in_place on pure-Python .pyx must succeed even when Cython
    is absent (try-Python-first path; never reaches the dep check).
    """
    _clear_cache()
    monkeypatch.setattr("black.cython_dependencies_are_installed", lambda warn: False)
    pyx_file = tmp_path / "pure.pyx"
    pyx_file.write_text("x = {'a':1,'b':2}\n")
    result = black.format_file_in_place(
        pyx_file,
        fast=True,
        mode=black.Mode(),
        write_back=black.WriteBack.NO,
    )
    assert result in (True, False)  # no exception raised
