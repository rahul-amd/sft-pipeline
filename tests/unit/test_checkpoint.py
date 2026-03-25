"""Tests for the DuckDB checkpoint manager."""
from __future__ import annotations

import pytest

from sft_pipeline.checkpoint import CheckpointManager, ItemStatus, prompt_id


def test_prompt_id_is_stable():
    pid = prompt_id("hello world")
    assert pid == prompt_id("hello world")
    assert pid != prompt_id("hello world!")
    assert pid.startswith("sha256:")


def test_open_and_close(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    cm.open()
    cm.close()


def test_mark_and_check_processed(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        assert not cm.is_processed("abc", "stage1")
        cm.mark_processed("abc", "stage1", status=ItemStatus.SUCCESS)
        assert cm.is_processed("abc", "stage1")
        assert not cm.is_processed("abc", "stage2")  # different stage


def test_preload_cache(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        cm.mark_processed("id1", "stage5", status=ItemStatus.SUCCESS)
        cm.mark_processed("id2", "stage5", status=ItemStatus.SUCCESS)
        cm.mark_processed("id3", "stage5", status=ItemStatus.FAILED)

        cm.preload_processed("stage5")
        # id1, id2 successful — in cache
        assert cm.is_processed("id1", "stage5")
        assert cm.is_processed("id2", "stage5")
        # id3 failed — not in cache
        assert not cm.is_processed("id3", "stage5")


def test_stage_lifecycle(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        assert not cm.is_stage_complete("stage1")
        cm.mark_stage_started("stage1", input_count=100)
        assert not cm.is_stage_complete("stage1")
        cm.mark_stage_complete("stage1", output_count=90)
        assert cm.is_stage_complete("stage1")


def test_batch_mark_processed(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        items = [(f"id_{i}", ItemStatus.SUCCESS, f"shard_{i // 10}.jsonl") for i in range(50)]
        cm.mark_processed_batch(items, "stage5")
        assert cm.processed_count("stage5") == 50
        assert cm.is_processed("id_0", "stage5")
        assert cm.is_processed("id_49", "stage5")


def test_shard_manifest(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        cm.register_shard("shard_001", "stage5", "/data/part-000001.jsonl", 1000, 2048000)
        cm.register_shard("shard_002", "stage5", "/data/part-000002.jsonl", 1200, 2500000)
        shards = cm.get_shards("stage5")
        assert len(shards) == 2
        assert "/data/part-000001.jsonl" in shards


def test_all_stage_statuses(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        cm.mark_stage_started("stage1")
        cm.mark_stage_complete("stage1", output_count=100)
        cm.mark_stage_started("stage2")
        statuses = cm.all_stage_statuses()
        names = [s["stage"] for s in statuses]
        assert "stage1" in names
        assert "stage2" in names
