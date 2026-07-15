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


def test_batch_mark_processed_empty_is_noop(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        cm.mark_processed_batch([], "stage5")  # must not raise
        assert cm.processed_count("stage5") == 0


def test_batch_upsert_dedups_within_batch(tmp_path):
    # The bulk upsert path must not choke on the same item_id appearing twice
    # in one batch — the last write wins, exactly as the old row-by-row path.
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        items = [
            ("dup", ItemStatus.SKIPPED, "old.jsonl"),
            ("dup", ItemStatus.SUCCESS, "new.jsonl"),
            ("other", ItemStatus.SUCCESS, "a.jsonl"),
        ]
        cm.mark_processed_batch(items, "stage6")
        # One physical row for "dup", reflecting the last write (SUCCESS).
        assert cm.processed_count("stage6") == 2  # dup + other, both success
        assert cm.is_processed("dup", "stage6")


def test_batch_upsert_replaces_existing_row(tmp_path):
    # A later batch re-marking the same id (e.g. across a resume) must replace,
    # not duplicate or error on the primary key.
    cm = CheckpointManager(tmp_path / "ckpt.duckdb")
    with cm:
        cm.mark_processed_batch([("x", ItemStatus.SKIPPED, "s0.jsonl")], "stage6")
        cm.mark_processed_batch([("x", ItemStatus.SUCCESS, "s1.jsonl")], "stage6")
        assert cm.processed_count("stage6") == 1  # replaced, not duplicated
        assert cm.is_processed("x", "stage6")


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
