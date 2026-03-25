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
