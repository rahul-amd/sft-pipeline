"""Tests for structural filter."""
from __future__ import annotations

import pytest

from sft_pipeline.config import StructuralFilterConfig
from sft_pipeline.filters.structural import check_structural


@pytest.fixture
def cfg():
    return StructuralFilterConfig()


def _record(**kwargs):
    base = {
        "prompt": "What is 2 + 2?",
        "reasoning": (
            "We need to add 2 and 2 together. "
            "Addition is a fundamental arithmetic operation where we combine two numbers. "
            "Two plus two equals four because when you count two objects and then count "
            "two more objects you end up with a total of four objects. "
            "This is one of the most basic facts in mathematics and forms the foundation "
            "of more complex arithmetic and algebraic reasoning throughout all of math."
        ),
        "answer": "4",
    }
    base.update(kwargs)
    return base


def test_valid_record_passes(cfg):
    result = check_structural(_record(), cfg)
    assert result.passed


def test_missing_reasoning(cfg):
    result = check_structural(_record(reasoning=""), cfg)
    assert not result.passed
    assert "missing_reasoning" in result.reason


def test_missing_answer(cfg):
    result = check_structural(_record(answer=""), cfg)
    assert not result.passed
    assert "missing_answer" in result.reason


def test_missing_prompt(cfg):
    result = check_structural(_record(prompt=""), cfg)
    assert not result.passed
    assert "missing_prompt" in result.reason


def test_too_short(cfg):
    result = check_structural(_record(reasoning="Short.", answer="x"), cfg)
    assert not result.passed
    assert "too_short" in result.reason


def test_too_long(cfg):
    long_text = " ".join(["word"] * (cfg.max_response_tokens + 100))
    result = check_structural(_record(reasoning=long_text), cfg)
    assert not result.passed
    assert "too_long" in result.reason


def test_repetition_loop(cfg):
    loop_text = "the quick brown fox " * 50  # 5-gram repeated many times
    result = check_structural(_record(reasoning=loop_text), cfg)
    assert not result.passed
    assert "repetition_loop" in result.reason


# ---------------------------------------------------------------------------
# raw_response path (Stage 5 output format: no separate reasoning/answer)
# ---------------------------------------------------------------------------

def _raw_record(**kwargs):
    """Record in Stage 5 raw output format — only raw_response, no parsed fields."""
    base = {
        "prompt": "What is 2 + 2?",
        "raw_response": (
            "<think>\n"
            "We need to add 2 and 2 together. "
            "Addition is a fundamental arithmetic operation where we combine two numbers. "
            "Two plus two equals four because when you count two objects and then count "
            "two more objects you end up with a total of four objects. "
            "This is one of the most basic facts in mathematics and forms the foundation "
            "of more complex arithmetic and algebraic reasoning throughout all of math.\n"
            "</think>\n"
            "<answer>\n4\n</answer>"
        ),
    }
    base.update(kwargs)
    return base


def test_raw_response_valid_passes(cfg):
    result = check_structural(_raw_record(), cfg)
    assert result.passed


def test_raw_response_missing_prompt(cfg):
    result = check_structural(_raw_record(prompt=""), cfg)
    assert not result.passed
    assert "missing_prompt" in result.reason


def test_raw_response_empty_response(cfg):
    result = check_structural(_raw_record(raw_response=""), cfg)
    assert not result.passed
    assert "missing_response" in result.reason


def test_raw_response_whitespace_only(cfg):
    result = check_structural(_raw_record(raw_response="   \n\t  "), cfg)
    assert not result.passed
    assert "missing_response" in result.reason


def test_raw_response_too_short(cfg):
    result = check_structural(_raw_record(raw_response="Short answer."), cfg)
    assert not result.passed
    assert "too_short" in result.reason


def test_raw_response_too_long(cfg):
    long_text = " ".join(["word"] * (cfg.max_response_tokens + 100))
    result = check_structural(_raw_record(raw_response=long_text), cfg)
    assert not result.passed
    assert "too_long" in result.reason


def test_raw_response_repetition_loop(cfg):
    loop_text = "the quick brown fox " * 50
    result = check_structural(_raw_record(raw_response=loop_text), cfg)
    assert not result.passed
    assert "repetition_loop" in result.reason


def test_raw_response_takes_priority_over_parsed(cfg):
    """When raw_response present but reasoning/answer absent, raw_response path is used."""
    rec = {
        "prompt": "What is 2 + 2?",
        "raw_response": (
            "Let me think through this arithmetic problem step by step. "
            "Addition combines two quantities into a single total value. "
            "Starting from two objects and then counting two additional objects "
            "produces a combined total of four objects altogether. "
            "Therefore the result of adding two and two together equals four. "
            "This follows directly from the foundational definition of addition "
            "over the natural numbers in elementary mathematics."
        ),
        # No reasoning or answer fields at all
    }
    result = check_structural(rec, cfg)
    assert result.passed


def test_parsed_fields_take_priority_when_both_present(cfg):
    """When both raw_response AND reasoning/answer present, parsed path is used."""
    rec = {
        "prompt": "What is 2 + 2?",
        "raw_response": "Some raw text that is long enough " * 10,
        "reasoning": "",   # empty → should fail with missing_reasoning
        "answer": "4",
    }
    result = check_structural(rec, cfg)
    assert not result.passed
    assert "missing_reasoning" in result.reason
