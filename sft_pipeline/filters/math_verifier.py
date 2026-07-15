"""
Math domain quality filter.

Checks internal numeric consistency: numbers the answer concludes with
(boxed / "= N") are compared against the numeric literals in the reasoning.

This filter is deliberately non-rejecting (informational only) — see
``check_math``. It previously ran SymPy's LaTeX parser for a "parse sanity"
signal that the caller discarded; that dead ANTLR cost has been removed.
"""
from __future__ import annotations

import re

from sft_pipeline.filters.structural import FilterResult

# LaTeX inline math: $...$  or  \(...\)
_LATEX_INLINE = re.compile(r"\$([^$]+)\$|\\\((.+?)\\\)")
# Display math: $$...$$  or  \[...\]
_LATEX_DISPLAY = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
# Numeric result pattern: "= 42", "= -3.14", "= 1/2"
_NUMERIC_RESULT = re.compile(
    r"=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:$|[^\w])", re.MULTILINE
)
# Final boxed answer: \boxed{42}, \boxed{\frac{1}{2}} ...
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
# Any numeric literal (used generously on the reasoning side)
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def check_math(record: dict) -> FilterResult:
    """
    Apply math verification to a response record.

    Numeric consistency — the concluded numbers (boxed / "= N" in the answer)
    are compared against ALL numeric literals in the reasoning (generous by
    design: false rejection is the costly error).

    NOTE: this filter never rejects. Every path returns ``passed=True`` — the
    checks are informational only (measured against an LLM judge on 995 labeled
    Stage 5 records, every rejection was a false positive). A prior version also
    ran SymPy's ``parse_latex`` on the answer's LaTeX spans to set a
    ``math_uncertain`` reason, but that reason is discarded by the caller (which
    only reads ``.passed``), so the ANTLR parse — tens of ms per expression on
    every math/science answer — was pure wasted CPU and has been removed. If you
    ever make this filter actually reject, re-introduce SymPy verification here.
    """
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    answer_exprs = _extract_math_strings(answer)
    a_numbers = _extract_answer_numbers(answer)

    if not answer_exprs and not a_numbers:
        # No math to verify — pass through
        return FilterResult(True)

    # Numeric consistency: concluded numbers should occur in the reasoning.
    r_numbers = set(_ANY_NUMBER.findall(reasoning))
    if a_numbers and r_numbers:
        if not (a_numbers & r_numbers) and not _is_subset_derivable(r_numbers, a_numbers):
            return FilterResult(True, "math_numbers_disjoint")

    return FilterResult(True)


def _extract_answer_numbers(answer: str) -> set[str]:
    """
    Numbers the answer *concludes with*: contents of \\boxed{...} plus
    "= N" result patterns. These are the claims worth checking against
    the reasoning.
    """
    nums: set[str] = set()
    for m in _BOXED.finditer(answer):
        nums.update(_ANY_NUMBER.findall(m.group(1)))
    nums.update(m.group(1) for m in _NUMERIC_RESULT.finditer(answer))
    return nums


def _extract_math_strings(text: str) -> list[str]:
    results = []
    for m in _LATEX_INLINE.finditer(text):
        results.append(m.group(1) or m.group(2))
    for m in _LATEX_DISPLAY.finditer(text):
        results.append(m.group(1) or m.group(2))
    return [s.strip() for s in results if s and s.strip()]


def _is_subset_derivable(r_numbers: set[str], a_numbers: set[str]) -> bool:
    """
    Very rough check: if the answer number can be obtained by
    simple arithmetic from reasoning numbers, don't flag.
    We only handle the trivial case where the answer is a sum or product.
    """
    try:
        a_vals = {float(n.replace("/", "/")) for n in a_numbers if "/" not in n}
        r_vals = {float(n) for n in r_numbers if "/" not in n}
        if not a_vals or not r_vals:
            return True  # Can't verify — don't flag
        # Pass if any answer value equals sum of any subset of reasoning values
        # (simplified: just check if it's close to any reasoning value)
        for av in a_vals:
            if any(abs(rv - av) < 1e-6 for rv in r_vals):
                return True
        return False
    except (ValueError, ZeroDivisionError):
        return True  # Uncertain — don't flag
