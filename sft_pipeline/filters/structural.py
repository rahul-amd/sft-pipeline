"""
Structural quality filters.

Checks that a response record has all required fields, is within length
bounds, and doesn't contain repetition loops.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from sft_pipeline.config import StructuralFilterConfig


@dataclass
class FilterResult:
    passed: bool
    reason: str = ""


def check_structural(record: dict, cfg: StructuralFilterConfig) -> FilterResult:
    """
    Apply all structural filters to a response record.
    Returns FilterResult(passed=True) if the record passes all checks.
    """
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")
    prompt = record.get("prompt", "")

    # 1. Required fields present and non-empty
    if not prompt or not prompt.strip():
        return FilterResult(False, "missing_prompt")
    if not reasoning or not reasoning.strip():
        return FilterResult(False, "missing_reasoning")
    if not answer or not answer.strip():
        return FilterResult(False, "missing_answer")

    # 2. Response token length (approximate via whitespace split)
    full_response = reasoning + " " + answer
    n_tokens = len(full_response.split())
    if n_tokens < cfg.min_response_tokens:
        return FilterResult(False, f"too_short:{n_tokens}")
    if n_tokens > cfg.max_response_tokens:
        return FilterResult(False, f"too_long:{n_tokens}")

    # 3. Repetition loop detection — check for repeated n-grams
    if _has_repetition(reasoning, cfg.max_repetition_ngram, cfg.max_repetition_count):
        return FilterResult(False, "repetition_loop")

    return FilterResult(True)


def _has_repetition(text: str, ngram_size: int, max_count: int) -> bool:
    """
    Return True if any ngram of size `ngram_size` appears more than
    `max_count` times in the text (indicates a repetition loop).
    """
    tokens = text.lower().split()
    if len(tokens) < ngram_size * (max_count + 1):
        return False
    ngrams = [
        " ".join(tokens[i : i + ngram_size])
        for i in range(len(tokens) - ngram_size + 1)
    ]
    counts = Counter(ngrams)
    return any(c > max_count for c in counts.values())
