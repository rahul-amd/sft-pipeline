"""Contract-only tests for external services; never contact E2B or publish data."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from sft_pipeline.config import load_config
from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.filters.code_verifier import _run_e2b


@pytest.mark.parametrize("error", [None, "SyntaxError: bad code"])
def test_e2b_protocol(monkeypatch, error):
    calls = []

    class Sandbox:
        @classmethod
        def create(cls, **kwargs):
            calls.append(kwargs)
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def run_code(self, code):
            assert code == "print(42)"
            return SimpleNamespace(error=error, text="42" if error is None else "")

    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", SimpleNamespace(Sandbox=Sandbox))
    result = _run_e2b("print(42)", 7)
    assert result.success == (error is None)
    assert calls == [{"timeout": 7, "allow_internet_access": False}, "closed"]


def test_e2b_service_failure_fails_closed(monkeypatch):
    def fail(**kwargs):
        raise OSError("fixture service unavailable")
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter",
                        SimpleNamespace(Sandbox=SimpleNamespace(create=fail)))
    with pytest.raises(RuntimeError, match="no execution verdict"):
        _run_e2b("print(42)", 3)


@pytest.mark.parametrize("fails", [False, True])
def test_hub_export_protocol_without_publication(tmp_path, monkeypatch, fails):
    from sft_pipeline.export.jsonl_writer import _push_to_hub
    calls = []

    class Dataset:
        def push_to_hub(self, repo_id, private):
            calls.append((repo_id, private))
            if fails:
                raise OSError("fixture publication failed")

    def load(*args, **kwargs):
        assert args == ("json",)
        assert kwargs == {"data_files": str(tmp_path / "*.jsonl"), "split": "train"}
        return Dataset()

    monkeypatch.setattr("datasets.load_dataset", load)
    if fails:
        with pytest.raises(OSError, match="publication failed"):
            _push_to_hub(tmp_path, "test/never-published")
    else:
        _push_to_hub(tmp_path, "test/never-published")
    assert calls == [("test/never-published", True)]


def test_filter_debug_collects_reasons_without_normal_output(tmp_path):
    import yaml
    from sft_pipeline.stages.stage6_filter import run_stage6
    raw = {"global": {"base_path": str(tmp_path / "run")},
           "stage6_filter": {"debug_rejections": True, "debug_rejection_limit": 1}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    inp = Path(cfg.stage5_inference.output_dir)
    inp.mkdir(parents=True)
    (inp / "part-000000.jsonl").write_text(json.dumps({"prompt_id": "bad", "response": ""}) + "\n")
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage6(cfg, cm)
        assert not cm.is_stage_complete("stage6_filter")
    rows = [json.loads(line) for line in Path(cfg.stage6_filter.debug_rejection_path).read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["_rejection_reason"]
    assert not list(Path(cfg.stage6_filter.output_dir).glob("part-*.jsonl"))
