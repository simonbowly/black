"""Cython syntax masking and restoration.

Phase 1: stub implementations.  All functions are no-ops that pass source
through unchanged, allowing the pipeline plumbing to be tested end-to-end
before real masking rules are added in Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# A substitution record: (masked_text, original_text).
# masked_text is the span Black saw; original_text is what gets restored.
Replacement = tuple[str, str]


def validate_cython_subset(src: str) -> bool:
    """Return True if every Cython-only construct in src is handled by the masker.

    Phase 1 stub: always returns True.
    """
    return True


def mask_cython(src: str) -> tuple[str, list[Replacement]]:
    """Replace Cython-only spans with __cy_* placeholders.

    Phase 1 stub: returns source unchanged with an empty substitution list.
    """
    return src, []


def unmask_cython(src: str, replacements: list[Replacement]) -> str:
    """Restore original Cython spans from the placeholder-substituted Black output.

    Phase 1 stub: returns source unchanged (replacements is always empty from stub).
    """
    return src
