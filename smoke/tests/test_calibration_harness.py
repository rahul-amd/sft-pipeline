"""Protocol/measurement checks; mocked scores do not validate model quality."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest


def harness():
    spec = importlib.util.spec_from_file_location("calibrate_judge", Path("smoke/calibrate_judge.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_uses_training_split_and_resumes(tmp_path, monkeypatch):
    module = harness()
    cases = []
    for split in ("calibration", "holdout"):
        for positive in (True, False):
            name = f"{split}-{positive}"
            cases.append(dict(case_id=name, group=split, split=split, expected_accept=positive,
                              prompt=name, reasoning="", answer="answer"))
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(c)+"\n" for c in cases))
    monkeypatch.setenv("SFT_API_KEY", "test-token")
    calls = []

    def post(url, **kw):
        calls.append(kw)
        prompt = kw["json"]["messages"][1]["content"]
        # Train needs threshold 8; holdout would favor threshold 2.
        value = (8 if "True" in prompt else 7) if "calibration" in prompt else (2 if "True" in prompt else 1)
        result = dict(accuracy=value, reasoning=value, format=9, overall=value,
                      verdict="correct", prompt_status="clear", reference_status="not_provided", reason="Mock verdict.")
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}
        return Response()

    monkeypatch.setattr("httpx.post", post)
    args = argparse.Namespace(api_base="http://mock/v1", model="mock", api_key_env="SFT_API_KEY",
                              max_tokens=512, baseline_ref="854656d", cases=str(path),
                              output=str(tmp_path / "results"), concurrency=2)
    module.main(args)
    assert len(calls) == 8
    report = json.loads((tmp_path / "results/report.json").read_text())
    for variant in ("baseline", "current"):
        result = report["variants"][variant]
        assert result["suggested_threshold"] == 8
        assert result["calibration_at_suggested"]["balanced_accuracy"] == 1
        assert result["holdout_at_suggested"]["false_rejects"] == 1
    module.main(args)
    assert len(calls) == 8  # Completed requests are reused.
    args.model = "different-model"
    with pytest.raises(ValueError, match="inputs changed"):
        module.main(args)


def test_current_metrics_apply_verdict_gate():
    rows = [{"scores": {"overall": 10, "verdict": "incorrect", "prompt_status": "clear"},
             "expected_accept": False}]
    assert harness().confusion(rows, 6, "current")["true_rejects"] == 1
    assert harness().confusion(rows, 6, "baseline")["false_accepts"] == 1


def test_separate_review_preserves_missing_final_answer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("review_run", Path("smoke/review_grounding_run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "stage5").mkdir()
    (tmp_path / "stage6/_decisions").mkdir(parents=True)
    row = {"prompt_id": "p", "prompt": "What is 2 + 2?", "reasoning": "Unfinished calculation",
           "answer": "", "response": "Unfinished calculation", "finish_reason": "length"}
    (tmp_path / "stage5/part-000000.jsonl").write_text(json.dumps(row)+"\n")
    (tmp_path / "stage6/_decisions/part-000000.jsonl").write_text(json.dumps({
        "prompt_id": "p", "passed": False, "rejection_reason": "structural:truncated_response"})+"\n")
    seen = []
    def judge(prompt, reasoning, answer, *args, **kwargs):
        seen.append((reasoning, answer))
        return {"overall": 1, "verdict": "incorrect", "prompt_status": "clear"}
    monkeypatch.setattr(module, "_call_judge", judge)
    monkeypatch.setenv("SFT_API_KEY", "test-token")
    module.main(argparse.Namespace(run=str(tmp_path), output=str(tmp_path / "review"),
                                   api_base="http://mock", model="mock"))
    assert seen == [("Unfinished calculation", "")]
    result = json.loads((tmp_path / "review/verdicts.jsonl").read_text())
    assert result["answer"] == "" and not result["reviewer_accept"]
