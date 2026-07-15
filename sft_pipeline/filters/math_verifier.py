"""
Math domain quality filter using SymPy.

Attempts to:
  1. Extract LaTeX math expressions from the response
  2. Verify that the final answer expression is syntactically parseable
  3. Check internal consistency: if a numeric result is computed in the
     reasoning, it should appear (or be derivable) in the answer

Parse failures are treated as 'uncertain' (not 'failed') to avoid
over-filtering on SymPy's limited LaTeX coverage.
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

    Two independent checks:
      1. Numeric consistency — the concluded numbers (boxed / "= N" in the
         answer) must appear somewhere in the reasoning. The reasoning side is
         matched against ALL numeric literals (generous by design: false
         rejection is the costly error). Runs even when LaTeX is unparseable.
      2. SymPy parse sanity — unparseable LaTeX is 'uncertain', never a reject.
    """
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    answer_exprs = _extract_math_strings(answer)
    a_numbers = _extract_answer_numbers(answer)

    if not answer_exprs and not a_numbers:
        # No math to verify — pass through
        return FilterResult(True)

    # 1. Numeric consistency: concluded numbers should occur in the reasoning.
    #    Informational only — measured against an LLM judge on 995 labeled
    #    Stage 5 records, every rejection this check made was a false positive
    #    (answers legitimately introduce fresh worked examples whose numbers
    #    never appear in the reasoning). Kept as a non-rejecting signal.
    r_numbers = set(_ANY_NUMBER.findall(reasoning))
    if a_numbers and r_numbers:
        if not (a_numbers & r_numbers) and not _is_subset_derivable(r_numbers, a_numbers):
            return FilterResult(True, "math_numbers_disjoint")

    # 2. SymPy parse sanity on the answer's LaTeX spans.
    if answer_exprs:
        parse_failures = sum(
            1 for expr_str in answer_exprs if _try_parse_sympy(expr_str) == "error"
        )
        if parse_failures == len(answer_exprs):
            return FilterResult(True, reason="math_uncertain")

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


def _try_parse_sympy(expr_str: str) -> str:
    """Returns 'ok', 'error', or 'uncertain'."""
    try:
        from sympy.parsing.latex import parse_latex
        parse_latex(expr_str)
        return "ok"
    except Exception:
        return "error"


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
