"""
Unit tests for heuristic.py quality filters.

Covers:
  - MSTTR info-density check (passes good short/long responses, fails repetitive ones)
  - Boilerplate / refusal detection
  - Self-contradiction detection
  - Edge cases (empty text, disabled flags, custom thresholds)
"""
import pytest

from sft_pipeline.config import HeuristicFilterConfig
from sft_pipeline.filters.heuristic import (
    _compute_msttr,
    _has_contradiction,
    check_heuristic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(reasoning: str = "", answer: str = "") -> dict:
    return {"reasoning": reasoning, "answer": answer}


# Good reasoning used in several tests — diverse vocab, ~150 tokens
_GOOD_REASONING = (
    "We need to find the eigenvalues of the matrix A. First, compute the characteristic "
    "polynomial det(A - lambda*I) = 0. Expanding along the first row gives a cubic equation "
    "in lambda. The diagonal entries are 3, 1, and 2. The characteristic polynomial "
    "simplifies to (lambda-1)(lambda-2)(lambda-3) = 0. Therefore the eigenvalues are "
    "lambda = 1, lambda = 2, and lambda = 3. Verification: substituting each root back "
    "into the polynomial confirms all three satisfy the equation."
)
_GOOD_ANSWER = "The eigenvalues of the matrix are 1, 2, and 3."


@pytest.fixture
def cfg():
    return HeuristicFilterConfig()


# ===========================================================================
# 1. MSTTR / info-density
# ===========================================================================

class TestInfoDensity:
    def test_good_short_response_passes(self, cfg):
        result = check_heuristic(_rec(_GOOD_REASONING, _GOOD_ANSWER), cfg)
        assert result.passed

    def test_repetitive_short_response_fails(self, cfg):
        # "the answer is four" repeated 20× → TTR ≈ 0.017
        rep = "the answer is four " * 20
        result = check_heuristic(_rec(rep, rep), cfg)
        assert not result.passed
        assert result.reason.startswith("low_info_density")

    def test_long_good_response_passes(self, cfg):
        """
        A 2 000-word response with varied vocabulary must pass.
        Raw TTR would be ~0.06 for this length (fail), but MSTTR stays high.
        """
        # Build a long response from a diverse word pool
        words = (
            "matrix eigenvalue determinant polynomial characteristic equation "
            "diagonal coefficient variable parameter iterate converge diverge "
            "approximate estimate theorem proof lemma corollary hypothesis axiom "
            "definition property example counterexample step process method "
            "technique approach framework model algorithm function result compute "
            "calculate derive factor solution quadratic linear cubic exponential "
            "logarithm trigonometric geometric arithmetic sequence series power "
        ).split()
        sentence = " ".join(words)  # 70 unique words
        # Repeat so we get ~2 000 tokens but keep each 100-word segment diverse
        long_text = (sentence + " ") * 30   # ~2 100 tokens
        result = check_heuristic(_rec(long_text, _GOOD_ANSWER), cfg)
        assert result.passed, (
            f"Long good response wrongly rejected: {result.reason}"
        )

    def test_long_repetitive_response_fails(self, cfg):
        """A long response that repeats the same 4-word phrase must fail."""
        rep = "the answer is wrong " * 400   # 1 600 tokens, TTR per 100-word segment ≈ 0.04
        result = check_heuristic(_rec(rep, "yes"), cfg)
        assert not result.passed
        assert result.reason.startswith("low_info_density")

    def test_empty_response_passes(self, cfg):
        """Empty text has no tokens — the TTR check is skipped (no div-by-zero)."""
        result = check_heuristic(_rec("", ""), cfg)
        # May fail for missing_reasoning etc. in structural, but heuristic should not crash
        # and specifically should not raise an error for empty text
        assert "low_info_density" not in result.reason

    def test_custom_low_threshold_passes_mediocre(self):
        """Lowering the threshold below the text's MSTTR allows it through."""
        # "the answer is four" × 20 + same answer → ~160 tokens, 4 unique words
        # MSTTR (segment=100): 4/100 = 0.04  →  threshold must be < 0.04 to pass
        cfg_lax = HeuristicFilterConfig(min_info_density=0.03)
        rep = "the answer is four " * 20
        result = check_heuristic(_rec(rep, rep), cfg_lax)
        assert result.passed

    def test_custom_high_threshold_rejects_normal(self):
        """Raising the threshold rejects even normal responses."""
        cfg_strict = HeuristicFilterConfig(min_info_density=0.99)
        result = check_heuristic(_rec(_GOOD_REASONING, _GOOD_ANSWER), cfg_strict)
        assert not result.passed
        assert result.reason.startswith("low_info_density")


# ===========================================================================
# 2. MSTTR helper directly
# ===========================================================================

class TestMSTTR:
    def test_short_text_uses_plain_ttr(self):
        # Fewer tokens than segment_size → plain TTR
        tokens = "the cat sat on the mat".split()  # 6 tokens, 5 unique → TTR=5/6
        assert abs(_compute_msttr(tokens, segment_size=100) - 5 / 6) < 1e-6

    def test_repetitive_long_text_low_score(self):
        tokens = ("the answer is four " * 50).split()  # 200 tokens
        score = _compute_msttr(tokens, segment_size=100)
        assert score < 0.10

    def test_diverse_long_text_high_score(self):
        # Build text with high per-segment diversity
        words = list("abcdefghijklmnopqrstuvwxyz") * 4  # 104 unique-ish chars as tokens
        tokens = words * 2  # 208 tokens; first 100 → 26 unique = TTR=0.26, but words are single chars
        # Use actual words for a realistic test
        vocab = [f"word{i}" for i in range(100)]
        tokens = vocab * 3  # 300 tokens; every 100-token segment = all 100 unique → TTR=1.0
        score = _compute_msttr(tokens, segment_size=100)
        assert score == 1.0

    def test_single_full_segment_equals_ttr(self):
        tokens = "alpha beta gamma delta epsilon zeta eta theta".split()
        score = _compute_msttr(tokens, segment_size=4)
        # Segments: [alpha beta gamma delta], [epsilon zeta eta theta]
        # Each segment: 4 unique / 4 total = 1.0
        assert abs(score - 1.0) < 1e-6

    def test_partial_trailing_segment_ignored(self):
        # 250 tokens: 2 full segments of 100 + 50 leftover; leftover is ignored
        vocab = [f"w{i}" for i in range(50)]
        tokens = vocab * 5  # 250 tokens; each 100-segment has 50 unique → TTR=0.5
        score = _compute_msttr(tokens, segment_size=100)
        assert abs(score - 0.5) < 1e-6


# ===========================================================================
# 3. Boilerplate / refusal
# ===========================================================================

class TestBoilerplate:
    @pytest.mark.parametrize("answer", [
        "I don't know",
        "I don't know.",
        "I do not know",
        "I am not sure",
        "I'm not sure",
        "I cannot answer",
        "I cannot help",
        "I cannot assist",
        "I'm an AI",
        "I am an AI",
        "I'm just an AI",
        "As an AI",
        "As an AI language model",
        "As an AI assistant",
        "I apologize",
        "I'm sorry, but I",
        "This question is too complex",
        "This topic is very vague.",
        "This problem is too difficult",
        "I am not able to",
        "I am unable to",
    ])
    def test_boilerplate_rejected(self, cfg, answer):
        result = check_heuristic(_rec(_GOOD_REASONING, answer), cfg)
        assert not result.passed, f"Expected rejection for: '{answer}'"
        assert result.reason == "boilerplate_answer"

    @pytest.mark.parametrize("answer", [
        # Genuine answers that happen to contain qualifying words
        "I don't know for certain, but based on the evidence the answer is 42.",
        "As an AI system, I can compute this: the result is 7.",
        "I apologize for the confusion in step 2; the correct answer is 5.",
        "I'm sorry, but I believe the correct interpretation leads to answer B.",
        # Normal substantive answers
        "The eigenvalues are 1, 2, and 3.",
        "The time complexity is O(n log n).",
        "Based on the analysis above, this is the final answer to the question.",
    ])
    def test_genuine_answer_not_rejected(self, cfg, answer):
        result = check_heuristic(_rec(_GOOD_REASONING, answer), cfg)
        assert "boilerplate_answer" not in result.reason, (
            f"Wrongly rejected as boilerplate: '{answer}'"
        )


# ===========================================================================
# 4. Self-contradiction
# ===========================================================================

class TestSelfContradiction:
    def test_contradiction_flagged(self, cfg):
        # Reasoning is affirmative; answer negates the same claim
        r = "This algorithm correctly sorts duplicate values using stable comparison."
        a = "This algorithm does not sort duplicate values correctly."
        result = check_heuristic(_rec(r, a), cfg)
        assert not result.passed
        assert result.reason == "self_contradiction"

    def test_same_polarity_not_flagged(self, cfg):
        # Both sides are negated — same polarity, no contradiction
        r = "The function cannot converge because the sequence does not satisfy Cauchy."
        a = "Therefore the series does not converge and cannot be summed."
        result = check_heuristic(_rec(r, a), cfg)
        assert result.passed or result.reason != "self_contradiction"

    def test_different_polarity_few_shared_words_not_flagged(self, cfg):
        # Polarity differs but < 3 shared content words → conservative, no flag
        r = "Not applicable here."
        a = "The result is positive."
        result = check_heuristic(_rec(r, a), cfg)
        assert "self_contradiction" not in result.reason

    def test_flipped_direction_also_flagged(self, cfg):
        # Reasoning is negated; answer is affirmative — contradiction is symmetric
        r = "The quadratic formula cannot produce complex roots for this equation with real coefficients."
        a = "The quadratic formula produces complex roots for this equation with real coefficients."
        result = check_heuristic(_rec(r, a), cfg)
        assert not result.passed
        assert result.reason == "self_contradiction"

    def test_contradiction_check_disabled(self):
        cfg_no_contra = HeuristicFilterConfig(flag_self_contradiction=False)
        r = "This algorithm correctly sorts duplicate values using stable comparison."
        a = "This algorithm does not sort duplicate values correctly."
        result = check_heuristic(_rec(r, a), cfg_no_contra)
        assert "self_contradiction" not in result.reason

    def test_contradiction_helper_directly(self):
        assert _has_contradiction(
            "This solution uses matrix operations with standard approach",
            "This solution does not work with standard matrix operations",
        )
        assert not _has_contradiction(
            "The function is not continuous",
            "Therefore it is not differentiable",
        )
        assert not _has_contradiction(
            "not working",   # < 3 shared content words
            "working fine",
        )


# ===========================================================================
# 5. Filter short-circuits: first failure wins, cheaper checks run first
# ===========================================================================

class TestFilterOrdering:
    def test_boilerplate_checked_after_ttr(self, cfg):
        """
        An answer that is both low-TTR and boilerplate should be rejected
        for low_info_density (TTR check runs first).
        """
        rep = "i dont know " * 30
        result = check_heuristic(_rec(rep, "I don't know"), cfg)
        assert not result.passed
        assert result.reason.startswith("low_info_density")

    def test_contradiction_checked_last(self, cfg):
        """
        If a record fails the boilerplate check, the contradiction check
        is never reached.
        """
        r = "This algorithm correctly sorts duplicate values using stable comparison."
        a = "I don't know"   # boilerplate — must trigger before contradiction
        result = check_heuristic(_rec(r, a), cfg)
        assert result.reason == "boilerplate_answer"


# ===========================================================================
# 6. raw_response path (Stage 5 format — no separate reasoning/answer fields)
# ===========================================================================

def _raw_rec(raw_response: str) -> dict:
    return {"prompt": "What is 2 + 2?", "raw_response": raw_response}


_GOOD_RAW = (
    "<think>\n"
    + _GOOD_REASONING
    + "\n</think>\n<answer>\n"
    + _GOOD_ANSWER
    + "\n</answer>"
)


class TestRawResponsePath:
    def test_good_raw_response_passes(self, cfg):
        result = check_heuristic(_raw_rec(_GOOD_RAW), cfg)
        assert result.passed

    def test_repetitive_raw_response_fails(self, cfg):
        rep_raw = "the answer is four " * 200   # high repetition, 800 tokens
        result = check_heuristic(_raw_rec(rep_raw), cfg)
        assert not result.passed
        assert result.reason.startswith("low_info_density")

    def test_empty_raw_response_no_crash(self, cfg):
        """Empty raw_response has no tokens — TTR check skipped, no error."""
        result = check_heuristic(_raw_rec(""), cfg)
        assert "low_info_density" not in result.reason

    def test_boilerplate_skipped_for_raw(self, cfg):
        """
        Boilerplate check requires parsed answer field.
        A raw_response that looks like a boilerplate answer should NOT be
        rejected for boilerplate (we can't isolate just the answer portion).
        """
        raw = "I don't know the answer to your question. " * 20  # repetitive but not parsed
        result = check_heuristic(_raw_rec(raw), cfg)
        # Will fail low_info_density, but NOT boilerplate_answer
        assert result.reason != "boilerplate_answer"

    def test_contradiction_skipped_for_raw(self, cfg):
        """Contradiction check requires separate reasoning/answer — skipped for raw."""
        # Build a raw response that would look contradictory if parsed,
        # but since it's raw it should only be evaluated on TTR
        raw = (
            _GOOD_REASONING + " "
            + "This algorithm does not sort duplicate values correctly. "
        ) * 2  # diverse enough to pass TTR; contradiction should be ignored
        result = check_heuristic(_raw_rec(raw), cfg)
        assert "self_contradiction" not in result.reason

    def test_raw_takes_priority_over_empty_parsed(self, cfg):
        """
        When raw_response is present and reasoning/answer are absent,
        the raw_response path is used (not the parsed path).
        """
        rec = {
            "prompt": "What is 2+2?",
            "raw_response": _GOOD_RAW,
            # No reasoning or answer keys at all
        }
        result = check_heuristic(rec, cfg)
        assert result.passed

    def test_parsed_path_used_when_both_present(self, cfg):
        """
        When both raw_response and reasoning/answer are present, parsed path is used.
        Boilerplate check should run (and fire) on the answer field.
        """
        rec = {
            "prompt": "What is 2+2?",
            "raw_response": _GOOD_RAW,
            "reasoning": _GOOD_REASONING,
            "answer": "I don't know",  # boilerplate answer
        }
        result = check_heuristic(rec, cfg)
        assert result.reason == "boilerplate_answer"
