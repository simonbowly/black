from __future__ import annotations

from collections.abc import Sequence

from black.cython.project import Replacement


def inflate_source(source: str, replacements: Sequence[Replacement]) -> str:
    result = source
    for replacement in sorted(replacements, key=lambda item: len(item.token), reverse=True):
        result = result.replace(replacement.token, replacement.value)
    return result
