"""
Frozen snapshot of the PRE-IMPROVEMENT-LOOP verifier chain (2026-07-14).

This is a faithful copy of the filter logic as it stood before the
sanity-check improvement loop, kept so the comparison viz can re-score any
sample with the old behaviour. Do NOT import this from pipeline code — the
live filters are in sft_pipeline/filters/.

Baseline behaviours reproduced here:
  - parse: strip leading channel marker, <think>/<answer> delimiters (absent
    in the data), FIRST-conclusion-marker fallback, sentence-join last resort
    (the join collapses newlines — a real baseline bug).
  - structural: min 50 tokens, 5-gram repeated > 3 times = loop.
  - heuristic: MSTTR < 0.30, boilerplate, self-contradiction ON (no gates).
  - math: '=N' numbers on both sides; early 'uncertain' exit on unparseable
    LaTeX (before the consistency check).
  - code: language tag OPTIONAL (any fenced block executes as Python), every
    non-zero exit rejects, stdin left open (input() blocks until timeout).
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    passed: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Baseline parse (channel-prefix strip + default delimiters + join fallback)
# ---------------------------------------------------------------------------

_CHANNEL_PREFIX = re.compile(r"^\s*<\|[^\n>]*>?(?:thought|analysis|final)?\s*", re.IGNORECASE)

_FALLBACK_SPLIT_PATTERNS = re.compile(
    r"\b(therefore|thus|in conclusion|hence|so,|the answer is|final answer:?|"
    r"in summary|to summarize)\b",
    re.IGNORECASE,
)


def parse_reasoning_answer(response: str) -> tuple[str, str, bool]:
    text = _CHANNEL_PREFIX.sub("", (response or "").lstrip(), count=1).strip()

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if think_match and answer_match:
        return think_match.group(1).strip(), answer_match.group(1).strip(), True

    # FIRST conclusion marker (baseline bug: long traces say "therefore" early)
    fb_match = _FALLBACK_SPLIT_PATTERNS.search(text)
    if fb_match:
        reasoning = text[: fb_match.start()].strip()
        answer = text[fb_match.start():].strip()
        if reasoning and answer:
            return reasoning, answer, True

    # Sentence-join last resort (baseline bug: collapses newlines → corrupts code)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) >= 2:
        return " ".join(sentences[:-1]), sentences[-1], True

    return text, "", False


# ---------------------------------------------------------------------------
# Baseline structural
# ---------------------------------------------------------------------------

MIN_RESPONSE_TOKENS = 50
MAX_RESPONSE_TOKENS = 8_000
MAX_REPETITION_NGRAM = 5
MAX_REPETITION_COUNT = 3


def check_structural(record: dict) -> FilterResult:
    prompt = record.get("prompt", "")
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    if not prompt or not prompt.strip():
        return FilterResult(False, "missing_prompt")
    if not reasoning or not reasoning.strip():
        return FilterResult(False, "missing_reasoning")
    if not answer or not answer.strip():
        return FilterResult(False, "missing_answer")

    response_text = reasoning + " " + answer
    n_tokens = len(response_text.split())
    if n_tokens < MIN_RESPONSE_TOKENS:
        return FilterResult(False, f"too_short:{n_tokens}")
    if n_tokens > MAX_RESPONSE_TOKENS:
        return FilterResult(False, f"too_long:{n_tokens}")

    if _has_repetition(reasoning):
        return FilterResult(False, "repetition_loop")
    return FilterResult(True)


def _has_repetition(text: str) -> bool:
    tokens = text.lower().split()
    if len(tokens) < MAX_REPETITION_NGRAM * (MAX_REPETITION_COUNT + 1):
        return False
    ngrams = [
        " ".join(tokens[i : i + MAX_REPETITION_NGRAM])
        for i in range(len(tokens) - MAX_REPETITION_NGRAM + 1)
    ]
    return any(c > MAX_REPETITION_COUNT for c in Counter(ngrams).values())


# ---------------------------------------------------------------------------
# Baseline heuristic
# ---------------------------------------------------------------------------

MIN_INFO_DENSITY = 0.3
MSTTR_SEGMENT_SIZE = 100

_NEGATION_WORDS = frozenset([
    "not", "no", "never", "cannot", "can't", "won't", "isn't", "aren't",
    "wasn't", "weren't", "doesn't", "don't", "didn't", "false", "incorrect",
    "wrong", "impossible",
])

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


def check_heuristic(record: dict) -> FilterResult:
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")
    full_text = (reasoning + " " + answer).lower()

    tokens = full_text.split()
    if tokens:
        msttr = _compute_msttr(tokens)
        if msttr < MIN_INFO_DENSITY:
            return FilterResult(False, f"low_info_density:{msttr:.2f}")

    if _BOILERPLATE_PATTERNS.match(answer.strip()):
        return FilterResult(False, "boilerplate_answer")

    # Baseline: contradiction check unconditional (no answer-length gate)
    if _has_contradiction(reasoning, answer):
        return FilterResult(False, "self_contradiction")

    return FilterResult(True)


def _compute_msttr(tokens: list[str]) -> float:
    seg = MSTTR_SEGMENT_SIZE
    if len(tokens) < seg:
        return len(set(tokens)) / len(tokens)
    segments = [tokens[i : i + seg] for i in range(0, len(tokens) - seg + 1, seg)]
    return sum(len(set(s)) / len(s) for s in segments) / len(segments)


def _has_contradiction(reasoning: str, answer: str) -> bool:
    r_tokens = set(reasoning.lower().split())
    a_tokens = set(answer.lower().split())
    if bool(r_tokens & _NEGATION_WORDS) == bool(a_tokens & _NEGATION_WORDS):
        return False
    a_content = {t for t in a_tokens if len(t) > 3} - _NEGATION_WORDS
    r_content = {t for t in r_tokens if len(t) > 3} - _NEGATION_WORDS
    return len(a_content & r_content) >= 3


# ---------------------------------------------------------------------------
# Baseline math verifier
# ---------------------------------------------------------------------------

_LATEX_INLINE = re.compile(r"\$([^$]+)\$|\\\((.+?)\\\)")
_LATEX_DISPLAY = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_NUMERIC_RESULT = re.compile(r"=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:$|[^\w])", re.MULTILINE)


def check_math(record: dict) -> FilterResult:
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    answer_exprs = _extract_math_strings(answer)
    if not answer_exprs:
        return FilterResult(True)

    parse_failures = sum(1 for e in answer_exprs if _try_parse_sympy(e) == "error")
    if parse_failures == len(answer_exprs):
        # Baseline: early exit BEFORE the consistency check
        return FilterResult(True, reason="math_uncertain")

    r_numbers = set(m.group(1) for m in _NUMERIC_RESULT.finditer(reasoning))
    a_numbers = set(m.group(1) for m in _NUMERIC_RESULT.finditer(answer))
    if r_numbers and a_numbers:
        if not (a_numbers & r_numbers) and not _is_subset_derivable(r_numbers, a_numbers):
            return FilterResult(False, "math_answer_not_in_reasoning")
    return FilterResult(True)


def _extract_math_strings(text: str) -> list[str]:
    results = []
    for m in _LATEX_INLINE.finditer(text):
        results.append(m.group(1) or m.group(2))
    for m in _LATEX_DISPLAY.finditer(text):
        results.append(m.group(1) or m.group(2))
    return [s.strip() for s in results if s and s.strip()]


def _try_parse_sympy(expr_str: str) -> str:
    try:
        from sympy.parsing.latex import parse_latex
        parse_latex(expr_str)
        return "ok"
    except Exception:
        return "error"


def _is_subset_derivable(r_numbers: set[str], a_numbers: set[str]) -> bool:
    try:
        a_vals = {float(n) for n in a_numbers if "/" not in n}
        r_vals = {float(n) for n in r_numbers if "/" not in n}
        if not a_vals or not r_vals:
            return True
        for av in a_vals:
            if any(abs(rv - av) < 1e-6 for rv in r_vals):
                return True
        return False
    except (ValueError, ZeroDivisionError):
        return True


# ---------------------------------------------------------------------------
# Baseline code verifier (optional lang tag — the headline bug)
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def check_code(record: dict, timeout: int = 10) -> FilterResult:
    full_text = record.get("reasoning", "") + "\n" + record.get("answer", "")
    code_blocks = _CODE_BLOCK.findall(full_text)
    if not code_blocks:
        return FilterResult(True)
    code = code_blocks[-1].strip()
    if not code:
        return FilterResult(True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return FilterResult(False, f"code_error:{(result.stderr or '')[:200]}")
        return FilterResult(True)
    except subprocess.TimeoutExpired:
        return FilterResult(False, "code_timeout")
    except Exception as exc:
        return FilterResult(False, f"code_error:{exc}")
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Baseline chain (structural → heuristic → math/code)
# ---------------------------------------------------------------------------

def run_baseline_chain(record: dict, domain: str, code_timeout: int = 10) -> tuple[bool, str]:
    """Parse with baseline parser, run the baseline filter chain. Mutates record."""
    reasoning, answer, _ = parse_reasoning_answer(record.get("response", ""))
    rec = dict(record)
    rec["reasoning"] = reasoning
    rec["answer"] = answer

    r = check_structural(rec)
    if not r.passed:
        return False, f"structural:{r.reason}"
    r = check_heuristic(rec)
    if not r.passed:
        return False, f"heuristic:{r.reason}"
    if domain == "math":
        r = check_math(rec)
        if not r.passed:
            return False, f"math:{r.reason}"
    elif domain == "code":
        r = check_code(rec, timeout=code_timeout)
        if not r.passed:
            return False, f"code:{r.reason}"
    return True, ""
