"""Cython suffix detection and optional-dependency check."""

from functools import lru_cache
from importlib.util import find_spec

from black.output import out

CYTHON_SUFFIXES = frozenset({".pyx", ".pxd", ".pxi"})

MISSING_DEP_MESSAGE = (
    "Cannot format Cython syntax: Cython is not installed.\n"
    'You can fix this by running ``pip install "black[cython]"``'
)


@lru_cache
def cython_dependencies_are_installed(*, warn: bool) -> bool:
    installed = find_spec("Cython") is not None
    if not installed and warn:
        out(MISSING_DEP_MESSAGE)
    return installed
