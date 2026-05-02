from dataclasses import replace
from pathlib import Path

from click.testing import CliRunner

import black
import black.handle_cython as handle_cython
from tests.util import DEFAULT_MODE

EMPTY_CONFIG = Path(__file__).parent / "data" / "empty_pyproject.toml"
CYTHON_MODE = replace(DEFAULT_MODE, is_cython=True)
RUNNER = CliRunner()


def test_cython_mode_uses_distinct_cache_key() -> None:
    assert DEFAULT_MODE.get_cache_key() != CYTHON_MODE.get_cache_key()


def test_format_str_cython_basic() -> None:
    source = "cdef int func(int x,int y=1):\n    return x+y\n"
    expected = "cdef int func(int x, int y=1):\n    return x + y\n"

    actual = black.format_str(source, mode=CYTHON_MODE)

    assert actual == expected
    black.assert_cython_equivalent(source, actual)
    black.assert_stable(source, actual, CYTHON_MODE)


def test_single_file_force_cython(tmp_path: Path) -> None:
    path = tmp_path / "file.py"
    path.write_text("cdef int func(int x,int y=1):\n    return x+y\n", encoding="utf-8")

    result = RUNNER.invoke(
        black.main,
        [str(path), "--cython", f"--config={EMPTY_CONFIG}"],
    )

    assert result.exit_code == 0, result.output
    assert (
        path.read_text(encoding="utf-8")
        == "cdef int func(int x, int y=1):\n    return x + y\n"
    )


def test_single_file_auto_detect_cython_suffix(tmp_path: Path) -> None:
    path = tmp_path / "file.pyx"
    path.write_text(
        "cdef int func(int x):\n    if x>0:\n        return x\n    else:\n        return 0\n",
        encoding="utf-8",
    )

    result = RUNNER.invoke(black.main, [str(path), f"--config={EMPTY_CONFIG}"])

    assert result.exit_code == 0, result.output
    assert (
        path.read_text(encoding="utf-8")
        == "cdef int func(int x):\n    if x > 0:\n        return x\n    else:\n        return 0\n"
    )


def test_cython_and_pyi_flags() -> None:
    result = RUNNER.invoke(
        black.main,
        ["-", "--pyi", "--cython"],
        input=b"cdef int func(int x):\n    return x\n",
    )

    assert result.exit_code == 1
    assert result.output == "Cannot pass both `pyi` and `cython` flags!\n"


def test_cython_and_ipynb_flags() -> None:
    result = RUNNER.invoke(
        black.main,
        ["-", "--ipynb", "--cython"],
        input=b"{}\n",
    )

    assert result.exit_code == 1
    assert result.output == "Cannot pass both `ipynb` and `cython` flags!\n"


def test_missing_cython_dependency_message(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "file.pyx"
    path.write_text("cdef int func(int x):\n    return x\n", encoding="utf-8")

    handle_cython.cython_dependencies_are_installed.cache_clear()
    monkeypatch.setattr(handle_cython, "find_spec", lambda _: None)

    result = RUNNER.invoke(black.main, [str(path), f"--config={EMPTY_CONFIG}"])

    expected = (
        "Cython dependencies are not installed.\n"
        'You can fix this by running ``pip install "black[cython]"``\n'
    )
    assert expected in result.output
