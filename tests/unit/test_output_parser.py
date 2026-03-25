"""Tests for the inference output parser."""
from __future__ import annotations

import pytest

from sft_pipeline.config import ReasoningDelimiters
from sft_pipeline.inference.output_parser import ParsedOutput, parse_output, select_best_candidate


@pytest.fixture
def delimiters():
    return ReasoningDelimiters()


def test_parse_with_delimiters(delimiters):
    text = (
        "<think>\nStep 1: add 2 and 2.\nStep 2: result is 4.\n</think>\n"
        "<answer>\n4\n</answer>"
    )
    result = parse_output(text, delimiters)
    assert result.valid
    assert not result.used_fallback
    assert "Step 1" in result.reasoning
    assert result.answer == "4"


def test_parse_fallback_therefore(delimiters):
    text = "We add 2 and 2. Therefore, the answer is 4."
    result = parse_output(text, delimiters)
    assert result.valid
    assert result.used_fallback
    assert result.answer


def test_parse_fallback_last_sentence(delimiters):
    text = "This is a long explanation. The calculation is straightforward. The answer is 42."
    result = parse_output(text, delimiters)
    assert result.valid
    assert result.used_fallback


def test_parse_empty_gives_invalid(delimiters):
    result = parse_output("", delimiters)
    # Empty string — reasoning and answer both empty
    assert not result.valid or result.answer == ""


def test_select_best_prefers_delimited(delimiters):
    candidates = [
        "Some vague text without any structure.",
        "<think>\nDetailed reasoning here.\n</think>\n<answer>\nCorrect answer.\n</answer>",
    ]
    best, idx = select_best_candidate(candidates, delimiters)
    assert idx == 1
    assert not best.used_fallback
    assert best.reasoning == "Detailed reasoning here."
    assert best.answer == "Correct answer."


def test_select_best_prefers_longer_reasoning(delimiters):
    candidates = [
        "<think>\nShort.\n</think>\n<answer>\nA.\n</answer>",
        "<think>\nVery long and detailed reasoning with many steps and explanations.\n</think>\n<answer>\nB.\n</answer>",
    ]
    best, idx = select_best_candidate(candidates, delimiters)
    assert idx == 1
