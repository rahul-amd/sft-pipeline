"""
Ray (distributed) path for Stage 6 filtering.

Real local Ray workers (not local_mode). Asserts the Ray path produces the same
survivors as the single-node path and that shard-level resume works. Skipped
when Ray isn't installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import load_config
from sft_pipeline.stages.stage6_filter import run_stage6
from sft_pipeline.storage import iter_jsonl_dir

ray = pytest.importorskip("ray", reason="ray not installed")

_REASONING = (
    "Let me reason about this carefully and in enough detail to clear the "
    "structural length floor. I identify the quantities, apply the relevant "
    "rule step by step, check the intermediate results, and then conclude."
)


def _rec(pid: str, answer: str, reasoning: str = _REASONING) -> dict:
    return {
        "prompt_id": pid,
        "prompt": f"Q {pid}",
        "domain": "general",
        "response": f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>",
    }


@pytest.fixture(scope="module", autouse=True)
def _local_ray():
    ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False,
             configure_logging=False, log_to_driver=False)
    yield
    ray.shutdown()


def _setup(tmp_path: Path, distributed: bool):
    tmp_path.mkdir(parents=True, exist_ok=True)
    stage5 = tmp_path / "stage5"
    stage5.mkdir()
    # 3 shards; each has a good record and a too-short (structural-reject) one.
    for s in range(3):
        with (stage5 / f"part-00000{s}.jsonl").open("w") as f:
            f.write(json.dumps(_rec(f"s{s}-ok", "The final answer is 42.")) + "\n")
            f.write(json.dumps(_rec(f"s{s}-bad", "No.", reasoning="tiny")) + "\n")

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "global: {run_id: r, base_path: \"%s\", checkpoint_db: \"%s\", device: cpu}\n"
        "stage5_inference: {output_dir: \"%s\"}\n"
        "stage6_filter:\n"
        "  enabled: true\n"
        "  distributed: %s\n"
        "  n_workers: 1\n"
        "  output_dir: \"%s\"\n"
        "  report_path: \"%s\"\n"
        "  llm_judge: {enabled: false}\n"
        % (
            tmp_path.as_posix(), (tmp_path / "ckpt.duckdb").as_posix(), stage5.as_posix(),
            "true" if distributed else "false",
            (tmp_path / "stage6").as_posix(), (tmp_path / "stage6" / "report.json").as_posix(),
        )
    )
    return load_config(cfg_yaml)


def _run(cfg):
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage6(cfg, cm)
    survivors = {r["prompt_id"] for r in iter_jsonl_dir(Path(cfg.stage6_filter.output_dir))}
    report = json.loads(Path(cfg.stage6_filter.report_path).read_text())
    return survivors, report


def test_ray_matches_single_node(tmp_path):
    serial_survivors, serial_report = _run(_setup(tmp_path / "serial", distributed=False))
    ray_survivors, ray_report = _run(_setup(tmp_path / "ray", distributed=True))

    assert ray_survivors == serial_survivors
    assert ray_report["total_input"] == serial_report["total_input"] == 6
    assert ray_report["total_passed"] == serial_report["total_passed"] == 3  # 3 good, 3 too-short
    assert all(pid.endswith("-ok") for pid in ray_survivors)


def test_ray_resume_skips_done_shards(tmp_path):
    cfg = _setup(tmp_path, distributed=True)
    ledger = Path(cfg.stage6_filter.output_dir) / "_state" / "shard_stats.jsonl"
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage6(cfg, cm)
        first = json.loads(Path(cfg.stage6_filter.report_path).read_text())
        assert sum(1 for _ in ledger.read_text().splitlines() if _.strip()) == 3
        run_stage6(cfg, cm)  # all shards already recorded → skipped
        second = json.loads(Path(cfg.stage6_filter.report_path).read_text())
    assert sum(1 for _ in ledger.read_text().splitlines() if _.strip()) == 3
    assert second["total_passed"] == first["total_passed"]
