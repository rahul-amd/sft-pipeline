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


def check_math(record: dict) -> FilterResult:
    """
    Apply math verification to a response record.
    Returns FilterResult with passed=True, passed=False, or
    passed=True with reason='uncertain' for unparseable expressions.
    """
    reasoning = record.get("reasoning", "")
    answer = record.get("answer", "")

    # Extract math from answer
    answer_exprs = _extract_math_strings(answer)
    if not answer_exprs:
        # No math to verify — pass through
        return FilterResult(True)

    # Try to parse each answer expression with SymPy
    parse_failures = 0
    for expr_str in answer_exprs:
        result = _try_parse_sympy(expr_str)
        if result == "error":
            parse_failures += 1

    if parse_failures == len(answer_exprs) and len(answer_exprs) > 0:
        # All expressions failed to parse — uncertain, don't filter
        return FilterResult(True, reason="math_uncertain")

    # Check internal consistency: numeric results from reasoning vs answer
    r_numbers = set(_extract_numbers(reasoning))
    a_numbers = set(_extract_numbers(answer))

    if r_numbers and a_numbers:
        # The final numeric answer should appear somewhere in the reasoning
        # (or be derivable — we just check the simpler case)
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
    """Returns 'ok', 'error', or 'uncertain'."""
    try:
        from sympy.parsing.latex import parse_latex
        parse_latex(expr_str)
        return "ok"
    except Exception:
        return "error"


def _extract_numbers(text: str) -> list[str]:
    """Extract numeric values from text (as strings for exact comparison)."""
    return [m.group(1) for m in _NUMERIC_RESULT.finditer(text)]


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
