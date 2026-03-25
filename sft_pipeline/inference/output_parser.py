"""
Parse teacher model outputs into reasoning trace + final answer.

Expected format (configurable delimiters):
  <think>
  ... step-by-step reasoning ...
  </think>
  <answer>
  ... concise final answer ...
  </answer>

Falls back to heuristic splitting when delimiters are absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sft_pipeline.config import ReasoningDelimiters


@dataclass
class ParsedOutput:
    reasoning: str
    answer: str
    used_fallback: bool = False
    valid: bool = True


_FALLBACK_SPLIT_PATTERNS = re.compile(
    r"\b(therefore|thus|in conclusion|hence|so,|the answer is|final answer:?|"
    r"in summary|to summarize)\b",
    re.IGNORECASE,
)


def parse_output(text: str, delimiters: ReasoningDelimiters) -> ParsedOutput:
    """
    Extract reasoning trace and answer from a model output string.
    Returns a ParsedOutput. Sets valid=False if both fields are empty.
    """
    text = text.strip()
    ts, te = delimiters.think_start, delimiters.think_end
    as_, ae = delimiters.answer_start, delimiters.answer_end

    # Primary: extract delimited sections
    think_match = re.search(
        re.escape(ts) + r"(.*?)" + re.escape(te), text, re.DOTALL
    )
    answer_match = re.search(
        re.escape(as_) + r"(.*?)" + re.escape(ae), text, re.DOTALL
    )

    if think_match and answer_match:
        return ParsedOutput(
            reasoning=think_match.group(1).strip(),
            answer=answer_match.group(1).strip(),
            used_fallback=False,
            valid=True,
        )

    # Fallback: try splitting at common conclusion markers
    fb_match = _FALLBACK_SPLIT_PATTERNS.search(text)
    if fb_match:
        split_pos = fb_match.start()
        reasoning = text[:split_pos].strip()
        answer = text[split_pos:].strip()
        if reasoning and answer:
            return ParsedOutput(
                reasoning=reasoning,
                answer=answer,
                used_fallback=True,
                valid=True,
            )

    # Last resort: treat full text as reasoning, answer is last sentence
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= 2:
        return ParsedOutput(
            reasoning=" ".join(sentences[:-1]),
            answer=sentences[-1],
            used_fallback=True,
            valid=True,
        )

    # Cannot parse
    return ParsedOutput(reasoning=text, answer="", used_fallback=True, valid=False)


def select_best_candidate(
    candidates: list[str], delimiters: ReasoningDelimiters
) -> tuple[ParsedOutput, int]:
    """
    From a list of raw candidate strings, select the best one.
    Prefer: valid parse > longest reasoning trace.
    Returns (ParsedOutput, candidate_index).
    """
    parsed = [parse_output(c, delimiters) for c in candidates]

    # Filter to valid parses
    valid = [(p, i) for i, p in enumerate(parsed) if p.valid and p.reasoning and p.answer]
    if not valid:
        # All invalid — return the one with most text
        best_idx = max(range(len(candidates)), key=lambda i: len(candidates[i]))
        return parsed[best_idx], best_idx

    # Among valid, prefer: non-fallback first, then longest reasoning
    non_fallback = [(p, i) for p, i in valid if not p.used_fallback]
    pool = non_fallback if non_fallback else valid
    best_p, best_i = max(pool, key=lambda x: len(x[0].reasoning))
    return best_p, best_i
