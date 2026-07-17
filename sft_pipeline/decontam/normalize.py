"""
Match-time normalization for decontamination.

This is deliberately MORE aggressive than the Stage 1 ``prompt_id`` normalizer
(which keeps case and punctuation). For contamination matching we want a prompt
like ``"What is 2+2?"`` to collide with ``"what is 2 2"``, so we lowercase and
strip all punctuation before tokenizing. It is Unicode-aware: ``[\\W_]`` under
``re.UNICODE`` drops punctuation/underscore but keeps letters and digits of any
script, so multilingual prompts (and CJK) are preserved rather than erased.
"""
from __future__ import annotations

import re
import unicodedata

# Any run of non-word characters (Unicode) or underscores becomes a separator.
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Normalize and split *text* into match tokens.

    NFKC → lowercase → non-word chars → spaces → split on whitespace.
    Returns an empty list for empty/whitespace-only input.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_WORD.sub(" ", text)
    return text.split()
