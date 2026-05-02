from __future__ import annotations

import re
from collections.abc import Sequence

from black.cython.project import Replacement

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def inflate_source(source: str, replacements: Sequence[Replacement]) -> str:
    result = source
    for replacement in sorted(replacements, key=lambda item: len(item.token), reverse=True):
        if _IDENTIFIER_RE.fullmatch(replacement.token):
            result = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(replacement.token)}(?![A-Za-z0-9_])",
                replacement.value,
                result,
            )
        else:
            result = result.replace(replacement.token, replacement.value)
    return result
