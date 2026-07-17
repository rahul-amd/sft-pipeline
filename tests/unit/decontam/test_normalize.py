"""Tests for the decontamination match-time tokenizer."""
from __future__ import annotations

from sft_pipeline.decontam.normalize import tokenize


def test_lowercase_and_punctuation_stripped():
    assert tokenize("What is 2+2?") == ["what", "is", "2", "2"]


def test_whitespace_collapsed():
    assert tokenize("  a\t b\n c ") == ["a", "b", "c"]


def test_unicode_letters_preserved():
    # Accented + CJK characters are word chars → kept, not erased. Punctuation dropped.
    assert tokenize("Café 汉字!") == ["café", "汉字"]


def test_underscore_is_a_separator():
    assert tokenize("foo_bar baz") == ["foo", "bar", "baz"]


def test_empty_and_blank():
    assert tokenize("") == []
    assert tokenize("   \t\n ") == []
