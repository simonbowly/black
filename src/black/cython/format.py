from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from black.mode import Mode

from black.cython.inflate import inflate_source
from black.cython.project import ProjectionError, project_source


class FormatError(Exception):
    """Raised when the adapter cannot safely format the source."""


class EquivalenceError(Exception):
    """Compatibility placeholder for the previous formatter API."""


PythonFormatter = Callable[[str, Mode], str]


def format_source(source: str, *, mode: Mode, python_formatter: PythonFormatter) -> str:
    if "\t" in source:
        raise FormatError("Source contains tabs — refusing to format")

    try:
        projected = project_source(source)
    except ProjectionError as exc:
        raise FormatError(str(exc)) from exc

    if not projected.surrogate:
        return source

    surrogate_mode = replace(mode, is_cython=False)
    formatted_surrogate = python_formatter(projected.surrogate, surrogate_mode)
    return inflate_source(formatted_surrogate, projected.replacements)
