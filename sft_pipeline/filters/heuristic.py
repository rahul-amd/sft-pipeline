"""
Heuristic quality filters.

Checks for:
  - Low information density (type-token ratio)
  - Self-contradiction between reasoning and answer
  - Extremely generic / boilerplate responses
"""
from __future__ import annotations

import re

from sft_pipeline.config import HeuristicFilterConfig
from sft_pipeline.filters.structural import FilterResult

# Negation words used for basic contradiction detection
_NEGATION_WORDS = frozenset([
    "not", "no", "never", "cannot", "can't", "won't", "isn't", "aren't",
    "wasn't", "weren't", "doesn't", "don't", "didn't", "false", "incorrect",
    "wrong", "impossible",
])

# Boilerplate phrases that indicate a near-empty response
_BOILERPLATE_PATTERNS = re.compile(
    r"^(i (don't|do not) know|i am (not sure|unsure)|i cannot (answer|help)|"
    r"as an ai|i'm (just )?an ai|i apologize|i'm sorry, but i|"
    r"this (question|topic) is (too|very) (complex|broad|vague))[\.,!]?$",
    re.IGNORECASE,
)


def check_heuristic(record: dict, cfg: HeuristicFilterConfig) -> FilterResult:
    """
    Apply heuristic filters to a response record.
    """
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    # 1. Information density: type-token ratio on the full response
    full_text = (reasoning + " " + answer).lower()
    tokens = full_text.split()
    if tokens:
        ttr = len(set(tokens)) / len(tokens)
        if ttr < cfg.min_info_density:
            return FilterResult(False, f"low_info_density:{ttr:.2f}")

    # 2. Boilerplate / refusal response
    answer_stripped = answer.strip()
    if _BOILERPLATE_PATTERNS.match(answer_stripped):
        return FilterResult(False, "boilerplate_answer")

    # 3. Self-contradiction (light heuristic — not a full NLI check)
    if cfg.flag_self_contradiction and _has_contradiction(reasoning, answer):
        return FilterResult(False, "self_contradiction")

    return FilterResult(True)


def _has_contradiction(reasoning: str, answer: str) -> bool:
    """
    Very simple heuristic: check whether the answer negates a key claim
    that appeared unnegated in the reasoning (or vice versa).

    This catches obvious cases like reasoning saying "X is true" and
    the answer saying "X is false". It is NOT a semantic NLI check.
    """
    r_tokens = set(reasoning.lower().split())
    a_tokens = set(answer.lower().split())

    r_negated = bool(r_tokens & _NEGATION_WORDS)
    a_negated = bool(a_tokens & _NEGATION_WORDS)

    # Heuristic: if reasoning is affirmative but answer is strongly negated
    # (or vice versa), flag it. Very conservative — only flag obvious cases.
    if r_negated != a_negated:
        # Extract the subject tokens of the answer
        a_content = {t for t in a_tokens if len(t) > 3} - _NEGATION_WORDS
        r_content = {t for t in r_tokens if len(t) > 3} - _NEGATION_WORDS
        overlap = a_content & r_content
        # Only flag if there's significant content overlap but different polarity
        if len(overlap) >= 3:
            return True
    return False
