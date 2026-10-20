"""Grounding and judge regressions drawn from the medium-run audit."""
import json
import random
from pathlib import Path

import pytest

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import LLMJudgeConfig, PipelineConfig, Stage2Config, Stage6Config
from sft_pipeline.filters.llm_judge import _call_judge, check_llm_judge
from sft_pipeline.stages.stage2_generate import (
    _call_generator, _check_grounding, _check_question_context, _check_question_context_one, run_stage2,
)
from sft_pipeline.stages.stage6_filter import _process_shard
from sft_pipeline.storage import iter_jsonl_dir


def scores(**updates):
    return {"accuracy": 9, "reasoning": 9, "format": 9, "overall": 9,
            "verdict": "correct", "prompt_status": "clear", "reference_status": "consistent",
            "reason": "The candidate satisfies the actual request.", "reference_reason": "", **updates}


def completion(value, finish="stop"):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"finish_reason": finish,
                                  "message": {"content": json.dumps(value)}}]}
    return Response()


def test_judge_sees_full_text_and_separate_evidence(monkeypatch):
    seen = []
    def post(*args, **kwargs):
        seen.append(kwargs["json"])
        return completion(scores(reference_status="questionable", reference_reason="Reference has a bug."))
    monkeypatch.setattr("httpx.post", post)
    record = {"prompt": "question " * 90 + "Require a set, not a list.",
              "reasoning": "explanation " * 200 + "This returns unique integers.",
              "answer": "details " * 90 + "return set(random.sample(range(20), 4))",
              "source_context": "A set has unique members.",
              "reference_answer": "return [random.randint(0, 19) for _ in range(4)]"}
    assert check_llm_judge(record, LLMJudgeConfig(sample_rate=1)).passed
    payload = json.loads(seen[0]["messages"][1]["content"])
    assert payload["question"] == record["prompt"]
    assert payload["candidate"] == {k: record[k] for k in ("reasoning", "answer")}
    assert payload["source_context"] == record["source_context"]
    assert payload["reference_answer"] == record["reference_answer"]
    assert record["judge"]["scores"]["reference_status"] == "questionable"


def test_oversized_judge_input_is_never_silently_clipped(monkeypatch):
    monkeypatch.setattr(_call_judge.retry, "sleep", lambda _: None)
    monkeypatch.setattr("httpx.post", lambda *a, **kw: pytest.fail("oversized request sent"))
    with pytest.raises(ValueError, match="no content was clipped"):
        check_llm_judge({"prompt": "x" * 1000}, LLMJudgeConfig(sample_rate=1, max_input_chars=500))


@pytest.mark.parametrize("changes,reason", [
    ({"verdict": "incorrect"}, "judge_incorrect"),
    ({"verdict": "uncertain"}, "judge_uncertain"),
    ({"prompt_status": "ambiguous"}, "judge_ambiguous_prompt"),
    ({"overall": 5}, "judge_score"),
])
def test_judge_does_not_accept_wrong_or_ambiguous_answers_on_style_score(monkeypatch, changes, reason):
    monkeypatch.setattr("sft_pipeline.filters.llm_judge._call_judge", lambda **_: scores(**changes))
    record = {"prompt": "How high is the net?", "answer": "A polished guess."}
    result = check_llm_judge(record, LLMJudgeConfig(sample_rate=1))
    assert not result.passed and result.reason.startswith(reason)
    assert record["judge"]["passed"] is False


@pytest.mark.parametrize("bad", [{"overall": True}, {"verdict": "yes"}, {"prompt_status": None}, {"reason": ""}])
def test_malformed_judge_verdict_is_an_error(monkeypatch, bad):
    monkeypatch.setattr("sft_pipeline.filters.llm_judge._call_judge", lambda **_: scores(**bad))
    record = {}
    with pytest.raises(ValueError):
        check_llm_judge(record, LLMJudgeConfig(sample_rate=1))
    assert record["judge"]["status"] == "error"


def test_unsampled_and_disabled_are_not_recorded_as_quality_verdicts():
    class ZeroRandom:
        def random(self):
            return 0.0
    record = {}
    assert check_llm_judge(record, LLMJudgeConfig(sample_rate=0), ZeroRandom()).passed
    assert record["judge"]["status"] == "not_sampled"
    assert check_llm_judge(record, LLMJudgeConfig(enabled=False)).passed
    assert record["judge"]["status"] == "disabled"


def verdict(i=0, **changes):
    return {"index": i, "self_contained": True, "answerable": True,
            "reference_answer": "0.19 metres", "evidence": "2.43 m and 2.24 m",
            "reason": "Both heights are given for volleyball.", **changes}


def test_generator_explicit_abstention_is_distinct_from_empty_response(monkeypatch):
    monkeypatch.setattr(_call_generator.retry, "sleep", lambda _: None)
    monkeypatch.setattr("httpx.post", lambda *a, **kw: completion([]))
    assert _call_generator("passage", 2, "http://mock", "mock", 0, 512) == []
    class EmptyResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    monkeypatch.setattr("httpx.post", lambda *a, **kw: EmptyResponse())
    from tenacity import RetryError
    with pytest.raises(RetryError):
        _call_generator("passage", 2, "http://mock", "mock", 0, 512)


def test_grounding_rejects_missing_context_and_invented_evidence(monkeypatch):
    values = [verdict(2, evidence="a fabricated quote"), verdict(1, self_contained=False), verdict(0)]
    monkeypatch.setattr("sft_pipeline.stages.stage2_generate._check_question_context",
                        lambda *a: [verdict(0), verdict(1, self_contained=False), verdict(2)])
    monkeypatch.setattr("httpx.post", lambda *a, **kw: completion(values))
    result = _check_grounding("Volleyball heights are 2.43 m and 2.24 m.",
                             ["Named sport?", "Which net?", "Invented claim?"], Stage2Config())
    assert [r["accepted"] for r in result] == [True, False, False]
    assert not result[2]["evidence_matches_source"]


@pytest.mark.parametrize("values", [[], [verdict(), verdict()], [verdict(answerable="true")],
                                     [verdict(index=4)], [verdict(reference_answer=None)]])
def test_malformed_or_incomplete_grounding_is_not_a_quality_rejection(monkeypatch, values):
    monkeypatch.setattr(_check_grounding.retry, "sleep", lambda _: None)
    monkeypatch.setattr("sft_pipeline.stages.stage2_generate._check_question_context", lambda *a: [verdict()])
    monkeypatch.setattr("httpx.post", lambda *a, **kw: completion(values))
    with pytest.raises(ValueError):
        _check_grounding("2.43 m and 2.24 m", ["A question?"], Stage2Config())


def test_source_cannot_override_failed_source_blind_context_check(monkeypatch):
    requests = []
    def post(*args, **kw):
        data = json.loads(kw["json"]["messages"][1]["content"])
        requests.append(data)
        if "source_passage" not in data:
            return completion([{"index": 0, "self_contained": False, "reason": "Which parish?"}])
        return completion([verdict(evidence="Fabrique", reference_answer="Fabrique")])
    monkeypatch.setattr("httpx.post", post)
    result = _check_grounding("The unnamed parish was governed by the Fabrique.",
                             ["What was the parish's governing body called?"], Stage2Config())
    assert len(requests) == 2 and "source_passage" not in requests[0]
    assert not result[0]["accepted"] and not result[0]["self_contained"]
    assert result[0]["evidence_matches_source"]


def test_context_checks_never_expose_neighboring_questions(monkeypatch):
    requests = []
    def post(*args, **kw):
        data = json.loads(kw["json"]["messages"][1]["content"])
        requests.append(data)
        return completion([{"index": 0, "reason": "Named subject", "self_contained": True}])
    monkeypatch.setattr("httpx.post", post)
    result = _check_question_context(["When was institute A founded?", "When was it founded?"], Stage2Config())
    assert len(requests) == 2 and all(len(r["questions"]) == 1 for r in requests)
    assert requests[1]["questions"][0]["question"] == "When was it founded?"
    assert [r["index"] for r in result] == [0, 1]


@pytest.mark.parametrize("values", [[], [{"index": 1, "self_contained": True, "reason": "x"}],
    [{"index": 0, "self_contained": "true", "reason": "x"}],
    [{"index": 0, "self_contained": True, "reason": None}]])
def test_bad_context_check_is_an_operational_error(monkeypatch, values):
    monkeypatch.setattr(_check_question_context_one.retry, "sleep", lambda _: None)
    monkeypatch.setattr("httpx.post", lambda *a, **kw: completion(values))
    with pytest.raises(ValueError):
        _check_question_context(["A question?"], Stage2Config())


def test_stage2_retains_source_and_checkpoints_rejected_chunks(tmp_path, monkeypatch):
    passage = "Volleyball uses net heights of 2.43 m and 2.24 m for men's and women's competitions. " * 2
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(passage)
    cfg = PipelineConfig.model_validate({"stage2_generate": {
        "corpora": [{"source": "local", "path": str(corpus)}], "output_dir": str(tmp_path / "stage2")}})
    questions = ["How much higher is the men's volleyball net than the women's net?",
                 "How much higher is the men's net than the women's net?"]
    monkeypatch.setattr("sft_pipeline.stages.stage2_generate._call_generator", lambda *a, **k: questions)
    monkeypatch.setattr("sft_pipeline.stages.stage2_generate._check_question_context",
                        lambda *a: [verdict(0), verdict(1, self_contained=False)])
    monkeypatch.setattr("httpx.post", lambda *a, **kw: completion([verdict(0), verdict(1, self_contained=False)]))
    with CheckpointManager(tmp_path / "check.duckdb") as cm:
        run_stage2(cfg, cm)
        monkeypatch.setattr("sft_pipeline.stages.stage2_generate._call_generator", lambda *a, **k: pytest.fail("regenerated chunk"))
        run_stage2(cfg, cm)
    rows = list(iter_jsonl_dir(cfg.stage2_generate.output_dir))
    assert len(rows) == 1 and rows[0]["prompt"] == questions[0]
    assert rows[0]["source_context"].strip() == passage.strip()
    assert rows[0]["reference_answer"] == "0.19 metres"
    audit = list((Path(cfg.stage2_generate.output_dir) / "_state/grounding").glob("*.json"))
    assert len(audit) == 1
    assert [v["accepted"] for v in json.loads(audit[0].read_text())["questions"]] == [True, False]


def test_all_rejected_chunk_is_checkpointed_and_audited(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("An unnamed parish's governing body was called a Fabrique. The location is omitted. " * 2)
    cfg = PipelineConfig.model_validate({"stage2_generate": {
        "corpora": [{"source": "local", "path": str(corpus)}], "output_dir": str(tmp_path / "stage2")}})
    calls = []
    def generate(*a, **kw):
        calls.append(1)
        return ["What was the original name of the parish's governing body?"]
    monkeypatch.setattr("sft_pipeline.stages.stage2_generate._call_generator", generate)
    monkeypatch.setattr("httpx.post", lambda *a, **kw: completion([verdict(self_contained=False)]))
    with CheckpointManager(tmp_path / "check.duckdb") as cm:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="0 failed chunks, 0 prompts"):
                run_stage2(cfg, cm)
    assert calls == [1]
    assert list(iter_jsonl_dir(cfg.stage2_generate.output_dir)) == []


def test_filter_audit_contains_rejected_and_accepted_verdicts_without_polluting_data(tmp_path, monkeypatch):
    rows = [{"prompt_id": str(i), "prompt": "A complete task", "reasoning": "Explanation of the task.",
             "answer": "The answer", "judge": {"status": "stale"}} for i in range(3)]
    rows[2]["finish_reason"] = "length"
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(r)+"\n" for r in rows))
    output = tmp_path / "filtered"
    output.mkdir()
    responses = iter([scores(), scores(verdict="incorrect", overall=2)])
    monkeypatch.setattr("sft_pipeline.filters.llm_judge._call_judge", lambda **_: next(responses))
    cfg = Stage6Config(structural={"min_response_tokens": 1}, llm_judge={"sample_rate": 1}, code={"enabled": False})
    stats = _process_shard(str(source), "part-000000", str(output), cfg, None, 42)
    assert stats["input"] == 3 and stats["passed"] == 1
    assert len(list(iter_jsonl_dir(output))) == 1
    decisions = list(iter_jsonl_dir(output / "_decisions"))
    assert len(decisions) == 3
    assert decisions[1]["judge"]["scores"]["overall"] == 2
    assert decisions[2]["judge"]["status"] == "not_reached"
    assert decisions[2]["rejection_reason"] == "structural:truncated_response"
