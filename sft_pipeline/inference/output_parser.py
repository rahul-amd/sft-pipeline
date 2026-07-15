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

    # Think delimiters only (no answer tags): everything after the think
    # block's closing delimiter is the answer. Common for teacher models that
    # emit a thought section followed directly by the final response.
    if think_match:
        answer_tail = text[think_match.end():].strip()
        if answer_tail:
            return ParsedOutput(
                reasoning=think_match.group(1).strip(),
                answer=answer_tail,
                used_fallback=False,
                valid=True,
            )

    # Fallback: split at the LAST conclusion marker. Long reasoning traces
    # use "therefore"/"thus" many times mid-derivation; only the final one
    # introduces the actual conclusion.
    fb_match = None
    for fb_match in _FALLBACK_SPLIT_PATTERNS.finditer(text):
        pass
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

    # Last resort: treat full text as reasoning, answer is the last sentence.
    # Slice the original text rather than re-joining split sentences —
    # re-joining collapses newlines, which silently corrupts any code block
    # or formatted math that spans the split.
    last_boundary = None
    for last_boundary in re.finditer(r"(?<=[.!?])\s+", text):
        pass
    if last_boundary:
        reasoning = text[: last_boundary.start()].strip()
        answer = text[last_boundary.end():].strip()
        if reasoning and answer:
            return ParsedOutput(
                reasoning=reasoning,
                answer=answer,
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
