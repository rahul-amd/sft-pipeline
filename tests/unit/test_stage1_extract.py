"""
Unit tests for _extract_prompt and its helpers in stage1_collect.py.

Covers:
  - Plain string passthrough
  - OpenAI chat format (with/without system prompt)
  - ShareGPT format (with/without system prompt)
  - JSON-encoded string versions of both formats
  - Vision API multi-part content (list of parts)
  - Edge cases: empty conversations, missing user turn, malformed JSON,
    single-message dict, non-string val
"""
import json

import pytest

from sft_pipeline.stages.stage1_collect import (
    _extract_from_openai_messages,
    _extract_from_sharegpt_messages,
    _extract_prompt,
)


# ---------------------------------------------------------------------------
# Plain string
# ---------------------------------------------------------------------------

def test_plain_string_returned_as_is():
    assert _extract_prompt("What is 2 + 2?") == "What is 2 + 2?"


def test_plain_string_stripped():
    assert _extract_prompt("  hello world  ") == "hello world"


def test_empty_string_returns_none():
    assert _extract_prompt("") is None
    assert _extract_prompt("   ") is None


# ---------------------------------------------------------------------------
# OpenAI format — Python list
# ---------------------------------------------------------------------------

def test_openai_user_only():
    messages = [{"role": "user", "content": "Explain gravity."}]
    assert _extract_prompt(messages) == "Explain gravity."


def test_openai_system_plus_user():
    messages = [
        {"role": "system", "content": "You are a helpful tutor."},
        {"role": "user", "content": "What is a derivative?"},
    ]
    result = _extract_prompt(messages)
    assert result == "You are a helpful tutor.\n\nWhat is a derivative?"


def test_openai_system_user_assistant_takes_first_user():
    """Only the first user turn is extracted; assistant turns are ignored."""
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Follow-up question."},
    ]
    result = _extract_prompt(messages)
    assert result == "Be concise.\n\nFirst question."


def test_openai_no_user_turn_returns_none():
    messages = [
        {"role": "system", "content": "System only."},
        {"role": "assistant", "content": "No user here."},
    ]
    assert _extract_prompt(messages) is None


def test_openai_system_only_returns_none():
    messages = [{"role": "system", "content": "Just a system prompt."}]
    assert _extract_prompt(messages) is None


def test_openai_empty_user_content_skipped():
    """A user message with empty content must not be returned."""
    messages = [
        {"role": "user", "content": "   "},
        {"role": "user", "content": "Valid question."},
    ]
    assert _extract_prompt(messages) == "Valid question."


def test_openai_multi_part_content():
    """Vision API sends content as a list of parts; text parts should be joined."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }
    ]
    assert _extract_prompt(messages) == "Describe this image:"


def test_openai_no_system_prefix_when_absent():
    """When there is no system message, result must not start with 'None'."""
    messages = [{"role": "user", "content": "Hello."}]
    result = _extract_prompt(messages)
    assert result == "Hello."
    assert not result.startswith("None")


# ---------------------------------------------------------------------------
# ShareGPT format — Python list
# ---------------------------------------------------------------------------

def test_sharegpt_human_only():
    messages = [{"from": "human", "value": "What is photosynthesis?"}]
    assert _extract_prompt(messages) == "What is photosynthesis?"


def test_sharegpt_system_plus_human():
    messages = [
        {"from": "system", "value": "You are a biology teacher."},
        {"from": "human", "value": "Explain mitosis."},
    ]
    result = _extract_prompt(messages)
    assert result == "You are a biology teacher.\n\nExplain mitosis."


def test_sharegpt_human_gpt_human_takes_first():
    messages = [
        {"from": "human", "value": "First turn."},
        {"from": "gpt", "value": "Reply."},
        {"from": "human", "value": "Second turn."},
    ]
    assert _extract_prompt(messages) == "First turn."


def test_sharegpt_gpt_only_returns_none():
    messages = [{"from": "gpt", "value": "No human here."}]
    assert _extract_prompt(messages) is None


def test_sharegpt_user_alias_accepted():
    """Some datasets use 'user' instead of 'human' in the from field."""
    messages = [{"from": "user", "value": "Question with user alias."}]
    assert _extract_prompt(messages) == "Question with user alias."


# ---------------------------------------------------------------------------
# JSON-encoded string versions
# ---------------------------------------------------------------------------

def test_json_string_openai_format():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Name a planet."},
    ]
    val = json.dumps(messages)
    assert _extract_prompt(val) == "Be brief.\n\nName a planet."


def test_json_string_sharegpt_format():
    messages = [
        {"from": "system", "value": "Context."},
        {"from": "human", "value": "Tell me more."},
    ]
    val = json.dumps(messages)
    assert _extract_prompt(val) == "Context.\n\nTell me more."


def test_plain_string_starting_with_bracket_but_invalid_json():
    """A string that looks like JSON but is malformed should be returned as-is."""
    val = "[not valid JSON"
    assert _extract_prompt(val) == "[not valid JSON"


# ---------------------------------------------------------------------------
# Single-message dict
# ---------------------------------------------------------------------------

def test_single_dict_with_content_key():
    assert _extract_prompt({"content": "A single message."}) == "A single message."


def test_single_dict_with_value_key():
    assert _extract_prompt({"value": "ShareGPT-style single."}) == "ShareGPT-style single."


def test_single_dict_with_text_key():
    assert _extract_prompt({"text": "Text key."}) == "Text key."


def test_single_dict_empty_content_returns_none():
    assert _extract_prompt({"content": "  "}) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_none_val_returns_none():
    assert _extract_prompt(None) is None


def test_empty_list_returns_none():
    assert _extract_prompt([]) is None


def test_list_of_non_dicts_returns_none():
    assert _extract_prompt(["a", "b", "c"]) is None


def test_unknown_list_format_returns_none():
    """A list of dicts without 'role' or 'from' keys is unrecognised."""
    messages = [{"speaker": "alice", "text": "Hello."}]
    assert _extract_prompt(messages) is None
