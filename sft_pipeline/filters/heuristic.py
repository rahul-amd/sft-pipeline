"""
Heuristic quality filters.

Checks for:
  - Low information density (Mean Segmental TTR, length-independent)
  - Self-contradiction between reasoning and answer
  - Generic / boilerplate refusal responses
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

# Boilerplate phrases that indicate a non-answer.
# Anchored ^ … $ so we only reject answers that are *entirely* a refusal,
# not ones that open with a caveat before answering.
_BOILERPLATE_PATTERNS = re.compile(
    r"^("
    r"i (don'?t|do not) know"
    r"|i am (not sure|unsure)"
    r"|i('m| am) (not sure|unsure)"
    r"|i cannot (answer|help|assist|provide|respond)"
    r"|i('m| am) (just )?an ai"
    r"|as an ai( (language model|assistant|system))?"
    r"|i apologize"
    r"|i'?m sorry,? but i"
    r"|this (question|topic|task|problem) is (too|very) (complex|broad|vague|difficult|sensitive|controversial)"
    r"|i (am not able|am unable) to"
    r")[\s\.,!?]*$",
    re.IGNORECASE,
)


def check_heuristic(record: dict, cfg: HeuristicFilterConfig) -> FilterResult:
    """
    Apply heuristic filters to a response record.
    """
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    # 1. Information density: Mean Segmental TTR on the full response.
    #    Raw TTR collapses for long texts (< 0.30 for any response > ~700 words)
    #    because common words repeat.  MSTTR computes TTR in fixed-size windows
    #    and averages, making it length-independent.
    full_text = (reasoning + " " + answer).lower()
    tokens = full_text.split()
    if tokens:
        msttr = _compute_msttr(tokens, cfg.msttr_segment_size)
        if msttr < cfg.min_info_density:
            return FilterResult(False, f"low_info_density:{msttr:.2f}")

    # 2. Boilerplate / refusal response
    answer_stripped = answer.strip()
    if _BOILERPLATE_PATTERNS.match(answer_stripped):
        return FilterResult(False, "boilerplate_answer")

    # 3. Self-contradiction (light heuristic — not a full NLI check)
    if cfg.flag_self_contradiction and _has_contradiction(reasoning, answer):
        return FilterResult(False, "self_contradiction")

    return FilterResult(True)


def _compute_msttr(tokens: list[str], segment_size: int) -> float:
    """
    Mean Segmental Type-Token Ratio.

    Splits `tokens` into non-overlapping windows of `segment_size`, computes
    TTR (unique/total) for each complete window, then returns the mean.

    For texts shorter than one window, falls back to plain TTR.
    This makes the metric length-independent: a 5 000-word detailed proof
    gets the same score as a 200-word answer with equivalent lexical variety.
    """
    if len(tokens) < segment_size:
        return len(set(tokens)) / len(tokens)

    segments = [
        tokens[i : i + segment_size]
        for i in range(0, len(tokens) - segment_size + 1, segment_size)
    ]
    return sum(len(set(seg)) / len(seg) for seg in segments) / len(segments)


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

    # Only flag when polarity differs (one side negated, other not)
    if r_negated == a_negated:
        return False

    # Require ≥3 shared content words (len > 3, not a negation word itself)
    a_content = {t for t in a_tokens if len(t) > 3} - _NEGATION_WORDS
    r_content = {t for t in r_tokens if len(t) > 3} - _NEGATION_WORDS
    return len(a_content & r_content) >= 3
