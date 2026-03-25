"""
End-to-end smoke test: Stages 1, 4, 6 (with mocked vLLM).

Runs the full pipeline on 20 synthetic prompts using:
  - Real Stage 1 (in-memory HF datasets via datasets.Dataset.from_dict)
  - Skipped Stage 2 (no corpora)
  - Skipped Stage 3 (pre-populated prompt JSONL, no GPU embedding)
  - Real Stage 4 (quota sampling from pre-populated data)
  - Mocked Stage 5 (no vLLM — returns pre-canned responses)
  - Real Stage 6 (CPU filters, LLM judge disabled)

Assertions:
  - Final JSONL has expected schema
  - Checkpoint resume skips already-processed items
  - Filter report is written
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import (
    ALL_SAMPLE_PROMPTS,
    make_prompt_record,
    make_response_record,
)


# ---------------------------------------------------------------------------
# Mock vLLM inference
# ---------------------------------------------------------------------------

def _mock_run_stage5(cfg, cm):
    """
    Replacement for run_stage5 that writes pre-canned responses
    directly to the Stage 5 output directory, bypassing vLLM.
    """
    from sft_pipeline.checkpoint import ItemStatus
    from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl_dir

    stage4_dir = Path(cfg.stage4_sample.output_path).parent
    out_dir = Path(cfg.stage5_inference.output_path).parent
    ensure_dir(out_dir)

    cm.mark_stage_started("stage5_inference")
    total = 0
    with ShardedJSONLWriter(out_dir, shard_size_mb=100) as writer:
        for rec in iter_jsonl_dir(stage4_dir):
            pid = rec["prompt_id"]
            if cm.is_processed(pid, "stage5_inference"):
                continue
            response = make_response_record(rec["prompt"], domain=rec.get("domain", "general"))
            response["prompt_id"] = pid
            writer.write(response)
            cm.mark_processed(pid, "stage5_inference", status=ItemStatus.SUCCESS)
            total += 1

    cm.mark_stage_complete("stage5_inference", output_count=total)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_smoke_pipeline(tmp_path):
    """Full pipeline smoke test on 20 synthetic prompts."""
    import orjson
    from sft_pipeline.checkpoint import CheckpointManager
    from sft_pipeline.config import PipelineConfig, load_config

    # Use forward slashes to avoid YAML escape issues on Windows
    tp = tmp_path.as_posix()

    # Write config
    cfg_text = f"""
global:
  seed: 42
  run_id: smoke_test
  base_path: "{tp}"
  checkpoint_db: "{tp}/checkpoints.duckdb"
  device: cpu

stage1_collect:
  enabled: false
  output_path: "{tp}/stage1/prompts.jsonl"

stage2_generate:
  enabled: false
  output_path: "{tp}/stage2/prompts.jsonl"

stage3_cluster:
  enabled: false
  embeddings_dir: "{tp}/stage3/embeddings"
  faiss_index_path: "{tp}/stage3/faiss.index"
  output_path: "{tp}/stage3/clustered_prompts.jsonl"

stage4_sample:
  enabled: true
  total_prompts: 15
  domain_quotas:
    math: 0.30
    code: 0.30
    science: 0.20
    general: 0.20
    language: 0.0
  difficulty_quotas:
    easy: 0.33
    medium: 0.34
    hard: 0.33
  dedup_cosine_threshold: 0.95
  output_path: "{tp}/stage4/sampled_prompts.jsonl"

stage5_inference:
  enabled: true
  model: "test-model"
  n_replicas: 1
  output_path: "{tp}/stage5/responses.jsonl"

stage6_filter:
  enabled: true
  output_path: "{tp}/stage6/filtered.jsonl"
  report_path: "{tp}/stage6/filter_report.json"
  llm_judge:
    enabled: false
  code:
    enabled: false
  math:
    enabled: false

export:
  final_jsonl_path: "{tp}/final/dataset.jsonl"
  push_to_hub: false
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(cfg_file)

    # Pre-populate Stage 3 output (normally produced by Stages 1–3)
    stage3_dir = Path(cfg.stage3_cluster.output_path).parent
    stage3_dir.mkdir(parents=True, exist_ok=True)
    domains = ["math"] * 5 + ["code"] * 5 + ["science"] * 5 + ["general"] * 5
    difficulties = ["easy", "medium", "hard"] * 6 + ["easy", "medium"]
    records = [
        {
            **make_prompt_record(
                ALL_SAMPLE_PROMPTS[i],
                domain=domains[i],
                difficulty=difficulties[i],
            ),
            "cluster_id": i % 5,
        }
        for i in range(20)
    ]
    shard = stage3_dir / "part-000000.jsonl"
    with shard.open("wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec) + b"\n")

    # Run pipeline with mocked Stage 5
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        from sft_pipeline.stages.stage4_sample import run_stage4
        from sft_pipeline.stages.stage6_filter import run_stage6

        run_stage4(cfg, cm)
        _mock_run_stage5(cfg, cm)
        run_stage6(cfg, cm)

    # Assertions
    stage6_dir = Path(cfg.stage6_filter.output_path).parent
    output_records = list(
        rec
        for shard in sorted(stage6_dir.glob("*.jsonl"))
        for rec in (orjson.loads(line) for line in shard.read_bytes().splitlines() if line)
    )

    assert len(output_records) > 0, "No records in Stage 6 output"

    required_fields = {"prompt_id", "prompt", "reasoning", "answer", "domain", "difficulty"}
    for rec in output_records:
        missing = required_fields - set(rec.keys())
        assert not missing, f"Record missing fields: {missing}"
        assert rec["prompt"], "Empty prompt"
        assert rec["reasoning"], "Empty reasoning"
        assert rec["answer"], "Empty answer"

    # Filter report exists
    report_path = Path(cfg.stage6_filter.report_path)
    assert report_path.exists(), "Filter report not written"
    report = json.loads(report_path.read_text())
    assert "total_input" in report
    assert "pass_rate" in report
    assert report["total_input"] > 0


def test_checkpoint_resume(tmp_path):
    """Verify that re-running Stage 6 skips already-processed items."""
    import orjson
    from sft_pipeline.checkpoint import CheckpointManager, ItemStatus
    from sft_pipeline.config import load_config
    from sft_pipeline.stages.stage6_filter import run_stage6
    from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir

    tp = tmp_path.as_posix()

    cfg_text = f"""
global:
  seed: 42
  run_id: resume_test
  base_path: "{tp}"
  checkpoint_db: "{tp}/checkpoints.duckdb"
  device: cpu
stage6_filter:
  enabled: true
  output_path: "{tp}/stage6/filtered.jsonl"
  report_path: "{tp}/stage6/filter_report.json"
  llm_judge:
    enabled: false
  code:
    enabled: false
  math:
    enabled: false
stage5_inference:
  enabled: false
  model: test
  n_replicas: 1
  output_path: "{tp}/stage5/responses.jsonl"
stage1_collect:
  enabled: false
  output_path: "{tp}/stage1/prompts.jsonl"
stage2_generate:
  enabled: false
  output_path: "{tp}/stage2/prompts.jsonl"
stage3_cluster:
  enabled: false
  embeddings_dir: "{tp}/stage3/embeddings"
  faiss_index_path: "{tp}/stage3/faiss.index"
  output_path: "{tp}/stage3/clustered_prompts.jsonl"
stage4_sample:
  enabled: false
  total_prompts: 10
  output_path: "{tp}/stage4/sampled_prompts.jsonl"
export:
  final_jsonl_path: "{tp}/final/dataset.jsonl"
  push_to_hub: false
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(cfg_file)

    # Pre-populate Stage 5 output with 10 records
    stage5_dir = Path(cfg.stage5_inference.output_path).parent
    stage5_dir.mkdir(parents=True, exist_ok=True)
    responses = [make_response_record(p) for p in ALL_SAMPLE_PROMPTS[:10]]
    shard = stage5_dir / "part-000000.jsonl"
    with shard.open("wb") as f:
        for rec in responses:
            f.write(orjson.dumps(rec) + b"\n")

    # First run
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage6(cfg, cm)
        count_after_first = cm.processed_count("stage6_filter")

    assert count_after_first == 10

    # Second run — should skip all (resume)
    with CheckpointManager(cfg.global_.checkpoint_db) as cm:
        run_stage6(cfg, cm)
        count_after_second = cm.processed_count("stage6_filter")

    assert count_after_second == count_after_first, (
        "Stage 6 re-processed items on second run — checkpoint resume broken"
    )
