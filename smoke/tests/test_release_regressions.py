import json
from pathlib import Path

import pytest

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import CorpusSource, LLMJudgeConfig, Stage2Config, load_config
from sft_pipeline.filters.llm_judge import check_llm_judge
from sft_pipeline.stages.stage2_generate import _iter_local_corpus


def test_all_shipped_configs_load(monkeypatch):
    for key, value in {"SFT_API_BASE": "http://localhost:8000/v1", "SFT_API_KEY": "test",
                       "SFT_MODEL": "test", "SFT_RUN_ID": "test"}.items():
        monkeypatch.setenv(key, value)
    root = Path(__file__).resolve().parents[2]
    for path in [*(root / "config").glob("*.yaml"), root / "smoke/config.yaml"]:
        cfg = load_config(path)
        assert "${" not in cfg.stage5_inference.output_dir
        assert "${" not in cfg.global_.checkpoint_db


def test_missing_environment_variable_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("SFT_MISSING_TEST_KEY", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text('stage5_inference: {api_key: "${SFT_MISSING_TEST_KEY}"}\n')
    with pytest.raises(ValueError, match="SFT_MISSING_TEST_KEY"):
        load_config(path)


def test_error_checkpoint_accepts_arbitrary_exception_text(tmp_path):
    with CheckpointManager(tmp_path / "checkpoint.duckdb") as cm:
        cm.mark_stage_failed("stage2_generate", 'bad "source"\nC:\\data\\file.txt')
        assert cm.all_stage_statuses()[0]["status"] == "failed"


@pytest.mark.parametrize("suffix,format", [(".json", "array"), (".json", "object"),
                                          (".jsonl", "lines"), (".txt", "text")])
def test_custom_corpus_file_formats(tmp_path, suffix, format):
    text = "A document about science, mathematics, and programming. " * 5
    path = tmp_path / ("source" + suffix)
    value = {"text": text}
    if format == "array":
        path.write_text(json.dumps([value], indent=2))
    elif format in ("object", "lines"):
        path.write_text(json.dumps(value) + "\n")
    else:
        path.write_text(text)
    rows = list(_iter_local_corpus(CorpusSource(source="local", path=str(path))))
    assert len(rows) == 1
    assert rows[0][1] == text


def test_corpus_identity_includes_subdirectory_and_full_text(tmp_path):
    common = "identical document prefix " * 10
    for dirname in ("a", "b"):
        d = tmp_path / dirname
        d.mkdir()
        (d / "source.json").write_text(json.dumps([{"text": common + dirname}]))
    rows = list(_iter_local_corpus(CorpusSource(source="local", path=str(tmp_path))))
    assert len({r[0] for r in rows}) == 2


def test_invalid_chunk_overlap_is_rejected():
    with pytest.raises(ValueError, match="chunk_overlap_tokens"):
        Stage2Config(chunk_size_tokens=10, chunk_overlap_tokens=10)


@pytest.mark.parametrize("result", [None, [], {}, {"overall": "bad"}, {"overall": 20},
                                   {"overall": float("nan")}])
def test_judge_malformed_verdict_does_not_pass_silently(monkeypatch, result):
    monkeypatch.setattr("sft_pipeline.filters.llm_judge._call_judge", lambda **_: result)
    with pytest.raises(ValueError):
        check_llm_judge({}, LLMJudgeConfig(sample_rate=1.0))


def test_judge_explicit_fail_open_is_available(monkeypatch):
    def error(**kwargs):
        raise ConnectionError("unavailable")
    monkeypatch.setattr("sft_pipeline.filters.llm_judge._call_judge", error)
    cfg = LLMJudgeConfig(sample_rate=1.0, on_error="pass")
    assert check_llm_judge({}, cfg).passed
    cfg.on_error = "raise"
    with pytest.raises(RuntimeError, match="no quality verdict"):
        check_llm_judge({}, cfg)


def test_truncated_teacher_response_is_rejected():
    from sft_pipeline.config import StructuralFilterConfig
    from sft_pipeline.filters.structural import check_structural
    result = check_structural({"prompt": "Solve this.", "response": "unfinished " * 100,
                               "finish_reason": "length"}, StructuralFilterConfig())
    assert not result.passed
    assert result.reason == "truncated_response"
