from dataclasses import replace
from pathlib import Path

from click.testing import CliRunner

import black
import black.handle_cython as handle_cython
from black.cython.inflate import inflate_source
from black.cython.project import project_source
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


def test_cython_mode_reuses_black_formatter() -> None:
    source = (
        "cdef int some_long_function_name("
        "int some_long_argument_name,int another_long_argument_name=1):\n"
        "    return some_long_argument_name+another_long_argument_name\n"
    )
    cython_mode = replace(CYTHON_MODE, line_length=40)

    projected = project_source(source)
    formatted_surrogate = black.format_str(
        projected.surrogate,
        mode=replace(DEFAULT_MODE, line_length=40),
    )

    actual = black.format_str(source, mode=cython_mode)

    assert actual == inflate_source(formatted_surrogate, projected.replacements)


def test_formats_cyblack_base_subset() -> None:
    source = """\
cdef double pi=3.14159265358979

cdef double circle_area(double r):
    \"\"\"Return the area of a circle with radius r.\"\"\"
    cdef double area
    area=pi*r*r
    return area

cdef int factorial(int n):
    \"\"\"Return n! iteratively.\"\"\"
    cdef int result=1
    cdef int i
    for i in range(2,n+1):
        result*=i
        i=i
    return result

def summarise(double r,int n):
    \"\"\"Print a short summary.\"\"\"
    cdef double a
    a=circle_area(r)
    return (a,factorial(n))
"""
    expected = """\
cdef double pi = 3.14159265358979


cdef double circle_area(double r):
    "Return the area of a circle with radius r."
    cdef double area
    area = pi * r * r
    return area


cdef int factorial(int n):
    "Return n! iteratively."
    cdef int result = 1
    cdef int i
    for i in range(2, n + 1):
        result *= i
        i = i
    return result


def summarise(double r, int n):
    "Print a short summary."
    cdef double a
    a = circle_area(r)
    return (a, factorial(n))
"""

    actual = black.format_str(source, mode=CYTHON_MODE)

    assert actual == expected
    black.assert_cython_equivalent(source, actual)


def test_formats_cdef_class_sample() -> None:
    source = """\
cpdef int calculate_sum(int a,int b):
    return a+b

cpdef double calculate_average(list numbers):
    return sum(numbers)/len(numbers)

cpdef bint is_even(int number):
    return number%2==0

cdef class Calculator:
    cpdef int multiply(self,int a,int b):
        return a*b
"""
    expected = """\
cpdef int calculate_sum(int a, int b):
    return a + b


cpdef double calculate_average(list numbers):
    return sum(numbers) / len(numbers)


cpdef bint is_even(int number):
    return number % 2 == 0


cdef class Calculator:
    cpdef int multiply(self, int a, int b):
        return a * b
"""

    actual = black.format_str(source, mode=CYTHON_MODE)

    assert actual == expected
    black.assert_cython_equivalent(source, actual)


def test_formats_simple_cimport_call() -> None:
    source = """\
from libc.math cimport sqrt

cpdef double norm(double x):
    return sqrt(x*x+1.0)
"""
    expected = """\
from libc.math cimport sqrt


cpdef double norm(double x):
    return sqrt(x * x + 1.0)
"""

    actual = black.format_str(source, mode=CYTHON_MODE)

    assert actual == expected
    black.assert_cython_equivalent(source, actual)


def test_formats_decorator_and_memoryview_sample() -> None:
    source = """\
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_sum(double[:] arr):
    cdef double total=0.0
    cdef int i
    for i in range(arr.shape[0]):
        total+=arr[i]
    return total

@cython.cdivision(True)
def divide_ints(int a,int b):
    return a/b

@cython.inline
cdef int square(int x):
    return x*x
"""
    expected = """\
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_sum(double[:] arr):
    cdef double total = 0.0
    cdef int i
    for i in range(arr.shape[0]):
        total += arr[i]
    return total


@cython.cdivision(True)
def divide_ints(int a, int b):
    return a / b


@cython.inline
cdef int square(int x):
    return x * x
"""

    actual = black.format_str(source, mode=CYTHON_MODE)

    assert actual == expected
    black.assert_cython_equivalent(source, actual)


def test_formats_cdef_class_fields_sample() -> None:
    source = """\
cdef class Point:
    cdef public double x,y

    def __init__(self,double x,double y):
        self.x=x
        self.y=y

    cdef double distance_from_origin(self):
        return (self.x**2+self.y**2)**0.5
"""
    expected = """\
cdef class Point:
    cdef public double x, y

    def __init__(self, double x, double y):
        self.x = x
        self.y = y

    cdef double distance_from_origin(self):
        return (self.x**2 + self.y**2) ** 0.5
"""

    actual = black.format_str(source, mode=CYTHON_MODE)

    assert actual == expected
    black.assert_cython_equivalent(source, actual)


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
