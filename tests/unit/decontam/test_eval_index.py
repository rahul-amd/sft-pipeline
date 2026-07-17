"""Tests for the eval n-gram containment index + field extraction."""
from __future__ import annotations

from sft_pipeline.config import EvalDatasetSource
from sft_pipeline.decontam.eval_index import (
    EvalNGramIndex,
    build_index,
    extract_field_text,
    iter_eval_texts,
)
from sft_pipeline.decontam.normalize import tokenize

_LONG = "the quick brown fox jumps over the lazy dog while the cat sleeps quietly"  # 14 tokens


def _match(idx: EvalNGramIndex, text: str):
    return idx.match(tokenize(text))


def test_13gram_containment_hit():
    idx = EvalNGramIndex(ngram_size=13)
    idx.add_text(_LONG, "eval_a")
    idx.finalize()
    m = _match(idx, "Note: " + _LONG + " -- please discuss.")
    assert m is not None
    eid, span = m
    assert idx.eval_names[eid] == "eval_a"
    assert len(span.split()) == 13


def test_no_false_positive():
    idx = EvalNGramIndex(13)
    idx.add_text(_LONG, "e")
    idx.finalize()
    assert _match(idx, "completely unrelated text about gardening and soil chemistry today") is None


def test_short_item_exact_fallback():
    idx = EvalNGramIndex(13)
    idx.add_text("what is 2+2", "short_eval")  # tokenizes to [what, is, 2, 2] → 4 tokens < 13
    idx.finalize()
    assert idx.gram_lens == (4,)  # only the short-length gram is indexed
    assert _match(idx, "hey, what is 2+2 exactly?") is not None
    assert _match(idx, "what is 3 plus 3") is None


def test_short_item_requires_full_containment():
    idx = EvalNGramIndex(13)
    idx.add_text("alpha beta gamma delta", "e")  # 4 tokens
    idx.finalize()
    assert _match(idx, "alpha beta gamma omega") is None          # only 3 of 4 → no
    assert _match(idx, "x alpha beta gamma delta y") is not None  # full span present → yes


def test_attribution_picks_owning_eval():
    idx = EvalNGramIndex(13)
    idx.add_text("one two three four five six seven eight nine ten eleven twelve thirteen", "eval_long")
    idx.add_text("secret phrase", "eval_short")
    idx.finalize()
    m = _match(idx, "prefix secret phrase suffix")
    assert m is not None and idx.eval_names[m[0]] == "eval_short"


def test_extract_field_text_variants():
    assert extract_field_text("hello world") == "hello world"
    assert extract_field_text(["A", "B", "C"]) == "A B C"          # choices list → joined
    conv = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    assert extract_field_text(conv) == "sys\n\nu"                   # conversation → first user turn
    assert extract_field_text(None) is None
    assert extract_field_text([]) is None


def test_iter_eval_texts_local_multifield(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text('{"question": "q one", "choices": ["c1", "c2"]}\n')
    src = EvalDatasetSource(
        name="e", source="local_jsonl", path=str(p), match_fields=["question", "choices"]
    )
    texts = list(iter_eval_texts(src))
    assert "q one" in texts
    assert "c1 c2" in texts  # both match_fields registered independently


def test_build_index_reports_item_counts(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text('{"question": "alpha beta gamma delta epsilon"}\n')
    src = EvalDatasetSource(name="e", source="local_jsonl", path=str(p), match_fields=["question"])
    idx, per_eval = build_index([src], ngram_size=13)
    assert per_eval["e"] == 1
    assert idx.total_grams >= 1
