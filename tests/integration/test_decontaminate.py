"""
Integration test for the decontamination stage.

Plants a prompt that embeds an eval question among clean prompts, runs
run_decontaminate, and verifies: the contaminated prompt is dropped from the
clean pool, the removed record is logged with the right eval, the report is
written, Stage 3 resolves its input to the decontaminated dir, and a re-run
resumes without changing the result. Also checks serial == parallel decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

from sft_pipeline.checkpoint import CheckpointManager, prompt_id
from sft_pipeline.config import load_config
from sft_pipeline.stages.decontaminate import run_decontaminate
from sft_pipeline.stages.stage3_cluster import _resolve_input_dirs
from sft_pipeline.storage import iter_jsonl_dir

# 14-token eval question → exercises the 13-gram path (not the short fallback).
_EVAL_Q = "the mitochondria is the powerhouse of the cell and produces energy for the body"
_CONTAMINATED = f"Explain why {_EVAL_Q} in simple terms."
_CLEAN = [
    "What is the capital city of France and roughly how many people live there",
    "Write a haiku about the ocean at dawn",
]


def _setup(tmp_path: Path, n_workers: int) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps({"question": _EVAL_Q}) + "\n")

    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    recs = [{"prompt_id": prompt_id(_CONTAMINATED), "prompt": _CONTAMINATED, "source": "webset"}]
    recs += [{"prompt_id": prompt_id(p), "prompt": p, "source": "webset"} for p in _CLEAN]
    with (stage1 / "part-000000.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    cfg_text = f"""
global:
  seed: 42
  run_id: dtest
  base_path: "{tmp_path.as_posix()}"
  checkpoint_db: "{(tmp_path / 'ckpt.duckdb').as_posix()}"
  device: cpu
stage1_collect:
  output_dir: "{stage1.as_posix()}"
stage2_generate:
  output_dir: "{(tmp_path / 'stage2').as_posix()}"
decontaminate:
  enabled: true
  ngram_size: 13
  n_workers: {n_workers}
  output_dir: "{(tmp_path / 'stage_decontam').as_posix()}"
  report_path: "{(tmp_path / 'stage_decontam' / 'decontam_report.json').as_posix()}"
  removed_dir: "{(tmp_path / 'stage_decontam' / 'removed').as_posix()}"
  evals:
    - name: my_eval
      source: local_jsonl
      path: "{eval_path.as_posix()}"
      match_fields: [question]
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    return cfg_file


def test_decontaminate_removes_contaminated_prompt(tmp_path):
    cfg = load_config(_setup(tmp_path, n_workers=1))
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)

    out_dir = Path(cfg.decontaminate.output_dir)
    survivors = {r["prompt_id"] for r in iter_jsonl_dir(out_dir)}
    assert prompt_id(_CONTAMINATED) not in survivors
    assert prompt_id(_CLEAN[0]) in survivors
    assert prompt_id(_CLEAN[1]) in survivors

    report = json.loads(Path(cfg.decontaminate.report_path).read_text())
    assert report["total_input"] == 3
    assert report["total_removed"] == 1
    assert report["total_survivors"] == 2
    assert report["removed_per_eval"]["my_eval"] == 1
    assert report["removed_per_source"]["webset"] == 1

    removed = list(iter_jsonl_dir(Path(cfg.decontaminate.removed_dir)))
    assert len(removed) == 1
    assert removed[0]["prompt_id"] == prompt_id(_CONTAMINATED)
    assert removed[0]["matched_eval"] == "my_eval"

    # Stage 3 now reads the decontaminated pool, not stage1/stage2.
    assert _resolve_input_dirs(cfg) == [out_dir]


def test_state_dir_not_read_by_stage3(tmp_path):
    # _shard_stats.jsonl and the removed/ dir must not leak into Stage 3's input.
    cfg = load_config(_setup(tmp_path, n_workers=1))
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)
    out_dir = Path(cfg.decontaminate.output_dir)
    for rec in iter_jsonl_dir(out_dir):
        assert "prompt" in rec  # only real prompt records, no stats/removed rows


def test_decontaminate_resume_is_stable(tmp_path):
    cfg = load_config(_setup(tmp_path, n_workers=1))
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_decontaminate(cfg, cm)
        first = json.loads(Path(cfg.decontaminate.report_path).read_text())
        run_decontaminate(cfg, cm)  # second run: shard already done → skipped
        second = json.loads(Path(cfg.decontaminate.report_path).read_text())
    assert second["total_input"] == first["total_input"]
    assert second["total_removed"] == first["total_removed"]


def test_parallel_matches_serial(tmp_path):
    serial_cfg = load_config(_setup(tmp_path / "s", n_workers=1))
    parallel_cfg = load_config(_setup(tmp_path / "p", n_workers=2))
    with CheckpointManager(serial_cfg.global_.checkpoint_db) as cm:
        run_decontaminate(serial_cfg, cm)
    with CheckpointManager(parallel_cfg.global_.checkpoint_db) as cm:
        run_decontaminate(parallel_cfg, cm)
    s = {r["prompt_id"] for r in iter_jsonl_dir(Path(serial_cfg.decontaminate.output_dir))}
    p = {r["prompt_id"] for r in iter_jsonl_dir(Path(parallel_cfg.decontaminate.output_dir))}
    assert s == p
