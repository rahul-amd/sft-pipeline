"""
Ray (distributed) path for the decontamination stage.

Uses a real local Ray cluster (ray.init with actual worker processes, NOT
local_mode) so the same serialization + object-store + task path as the real
cluster is exercised. Asserts the Ray path produces identical decisions to the
single-node path and that shard-level resume works. Skipped when Ray isn't
installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_pipeline.checkpoint import CheckpointManager, prompt_id
from sft_pipeline.config import load_config
from sft_pipeline.stages.decontaminate import run_decontaminate
from sft_pipeline.storage import iter_jsonl_dir

ray = pytest.importorskip("ray", reason="ray not installed")

# 14-token eval sentence → exercises the 13-gram path.
_EVAL = "the industrial revolution began in britain during the late eighteenth century and transformed manufacturing"
_CONTAM = f"Discuss how {_EVAL} across europe."
_CLEAN = [
    "Write a short story about a lighthouse keeper.",
    "Explain the rules of chess to a beginner.",
    "What ingredients go into a classic margherita pizza?",
]


@pytest.fixture(scope="module", autouse=True)
def _local_ray():
    ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False,
             configure_logging=False, log_to_driver=False)
    yield
    ray.shutdown()


def _setup(tmp_path: Path, distributed: bool) -> "PipelineConfig":  # noqa: F821
    tmp_path.mkdir(parents=True, exist_ok=True)
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"question": _EVAL}) + "\n")

    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    # Three shards so Ray actually fans out; contamination in shards 0 and 2.
    shards = {
        "part-000000.jsonl": [_CONTAM, _CLEAN[0]],
        "part-000001.jsonl": [_CLEAN[1]],
        "part-000002.jsonl": [_CLEAN[2], _CONTAM.replace("across europe", "worldwide")],
    }
    for fname, prompts in shards.items():
        with (stage1 / fname).open("w") as f:
            for p in prompts:
                f.write(json.dumps({"prompt_id": prompt_id(p), "prompt": p, "source": "web"}) + "\n")

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "global: {run_id: r, base_path: \"%s\", checkpoint_db: \"%s\", device: cpu}\n"
        "stage1_collect: {output_dir: \"%s\"}\n"
        "decontaminate:\n"
        "  enabled: true\n"
        "  distributed: %s\n"
        "  n_workers: 1\n"
        "  evals:\n"
        "    - {name: hist, source: local_jsonl, path: \"%s\", match_fields: [question]}\n"
        % (
            tmp_path.as_posix(),
            (tmp_path / "ckpt.duckdb").as_posix(),
            stage1.as_posix(),
            "true" if distributed else "false",
            eval_path.as_posix(),
        )
    )
    return load_config(cfg_yaml)


def _run(cfg):
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)
    survivors = {r["prompt_id"] for r in iter_jsonl_dir(Path(cfg.decontaminate.output_dir))}
    report = json.loads(Path(cfg.decontaminate.report_path).read_text())
    return survivors, report


def test_ray_matches_single_node(tmp_path):
    serial_survivors, serial_report = _run(_setup(tmp_path / "serial", distributed=False))
    ray_survivors, ray_report = _run(_setup(tmp_path / "ray", distributed=True))

    assert ray_survivors == serial_survivors
    assert ray_report["total_input"] == serial_report["total_input"] == 5
    assert ray_report["total_removed"] == serial_report["total_removed"] == 2
    # The two contaminated prompts are gone; the three clean ones remain.
    assert prompt_id(_CONTAM) not in ray_survivors
    for c in _CLEAN:
        assert prompt_id(c) in ray_survivors


def test_ray_resume_skips_done_shards(tmp_path):
    cfg = _setup(tmp_path, distributed=True)
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)
        first = json.loads(Path(cfg.decontaminate.report_path).read_text())
        run_decontaminate(cfg, cm)  # all shards already in the ledger → skipped
        second = json.loads(Path(cfg.decontaminate.report_path).read_text())
    assert second["total_input"] == first["total_input"]
    assert second["total_removed"] == first["total_removed"]
