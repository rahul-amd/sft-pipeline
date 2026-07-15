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

    Supports two record formats:
      - New (Stage 5 raw output): uses 'response' for length + repetition checks.
      - Legacy / parsed: uses 'reasoning' + 'answer' fields as before.
    """
    prompt = record.get("prompt", "")
    response = record.get("response", "")
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    # 1. Prompt must always be present
    if not prompt or not prompt.strip():
        return FilterResult(False, "missing_prompt")

    # 2. Determine response text to check.
    #    Use response path when no separately-parsed fields are present.
    #    The emptiness of response is checked *after* we decide the path,
    #    so an empty response yields "missing_response" rather than
    #    falling through to the parsed path (which would give "missing_reasoning").
    if not (reasoning or answer):
        # Unparsed format: work directly on response
        if not response or not response.strip():
            return FilterResult(False, "missing_response")
        response_text = response
        repetition_text = response
    else:
        # Parsed format: require both reasoning and answer
        if not reasoning or not reasoning.strip():
            return FilterResult(False, "missing_reasoning")
        if not answer or not answer.strip():
            return FilterResult(False, "missing_answer")
        response_text = reasoning + " " + answer
        repetition_text = reasoning

    # 3. Response token length (approximate via whitespace split)
    n_tokens = len(response_text.split())
    if n_tokens < cfg.min_response_tokens:
        return FilterResult(False, f"too_short:{n_tokens}")
    if n_tokens > cfg.max_response_tokens:
        return FilterResult(False, f"too_long:{n_tokens}")

    # 4. Repetition loop detection — check for repeated n-grams
    if _has_repetition(repetition_text, cfg.max_repetition_ngram, cfg.max_repetition_count):
        return FilterResult(False, "repetition_loop")

    return FilterResult(True)


def _has_repetition(text: str, ngram_size: int, max_count: int) -> bool:
    """
    Return True if any ngram of size `ngram_size` appears more than the
    allowed count (indicates a repetition loop).

    The allowed count is length-aware: long reasoning traces naturally repeat
    5-grams like "the answer is 0" while re-checking work, so the threshold
    scales with text length (1 extra allowed repeat per 150 tokens).
    Calibrated against LLM-judge labels: genuinely-stuck responses show max
    5-gram counts several times higher than the busiest healthy trace of the
    same length.
    """
    tokens = text.lower().split()
    threshold = max(max_count, len(tokens) // 150)
    # Minimum tokens for threshold+1 (possibly overlapping) occurrences.
    if len(tokens) < threshold + ngram_size:
        return False
    ngrams = [
        " ".join(tokens[i : i + ngram_size])
        for i in range(len(tokens) - ngram_size + 1)
    ]
    counts = Counter(ngrams)
    return any(c > threshold for c in counts.values())
