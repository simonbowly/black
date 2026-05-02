from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

from black.output import out

CYTHON_SUFFIXES = frozenset({".pxd", ".pxi", ".pyx"})


def is_cython_path(path: Path) -> bool:
    return path.suffix in CYTHON_SUFFIXES


def cython_dependency_error_message() -> str:
    return (
        "Cython dependencies are not installed.\n"
        'You can fix this by running ``pip install "black[cython]"``'
    )


@lru_cache
def cython_dependencies_are_installed(*, warn: bool) -> bool:
    installed = find_spec("Cython") is not None
    if not installed and warn:
        out(cython_dependency_error_message())
    return installed
