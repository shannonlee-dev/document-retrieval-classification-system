"""Privacy-safe text boundary for bundled documents."""

from __future__ import annotations

import re


SAFE_TERMS = frozenset(
    """
    algorithm animation astronaut ball baseball bat batting bitmap catcher color
    computer data dimensional display earth field file flight format galaxy
    game graphics gravity hit hitter image inning launch league loss lunar
    mission model monitor moon object orbit orbital pitcher pixel planet
    player program render resolution rocket run satellite score screen
    season shuttle software solar space spacecraft stadium star team telescope
    throw tracing universe vector video visual win window
    """.split()
)


def sanitize_text(text: str) -> str:
    """Retain only generic allowlisted terms from text."""

    return " ".join(token for token in re.findall(r"[a-z]+", text.lower()) if token in SAFE_TERMS)


def is_safe_text(text: str) -> bool:
    """Return whether text is nonblank and entirely composed of safe terms."""

    tokens = text.split()
    return bool(tokens) and all(token in SAFE_TERMS for token in tokens)
