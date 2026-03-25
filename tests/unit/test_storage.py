"""Tests for storage utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_pipeline.storage import (
    ShardedJSONLWriter,
    count_jsonl_lines,
    iter_jsonl,
    iter_jsonl_dir,
)


def test_iter_jsonl(tmp_path):
    f = tmp_path / "test.jsonl"
    records = [{"id": i, "text": f"hello {i}"} for i in range(10)]
    f.write_text("\n".join(json.dumps(r) for r in records))
    result = list(iter_jsonl(f))
    assert len(result) == 10
    assert result[0] == {"id": 0, "text": "hello 0"}


def test_iter_jsonl_missing_file(tmp_path):
    result = list(iter_jsonl(tmp_path / "nonexistent.jsonl"))
    assert result == []


def test_iter_jsonl_dir(tmp_path):
    for i in range(3):
        shard = tmp_path / f"part-{i:06d}.jsonl"
        shard.write_text(json.dumps({"shard": i, "n": 0}) + "\n"
                         + json.dumps({"shard": i, "n": 1}) + "\n")
    result = list(iter_jsonl_dir(tmp_path))
    assert len(result) == 6


def test_count_jsonl_lines(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text("\n".join(json.dumps({"x": i}) for i in range(7)))
    assert count_jsonl_lines(f) == 7


def test_sharded_writer_basic(tmp_path):
    out = tmp_path / "output"
    records = [{"id": i, "val": "x" * 10} for i in range(100)]
    with ShardedJSONLWriter(out, shard_size_mb=100) as writer:
        for r in records:
            writer.write(r)
    assert writer.total_records == 100
    read_back = list(iter_jsonl_dir(out))
    assert len(read_back) == 100


def test_sharded_writer_creates_multiple_shards(tmp_path):
    out = tmp_path / "output"
    # Force sharding at very small size
    with ShardedJSONLWriter(out, shard_size_mb=1) as writer:
        # Each record is ~100 bytes; 1MB / 100 bytes ≈ 10,000 records per shard
        # Write 50,000 to get ~5 shards
        for i in range(50_000):
            writer.write({"id": i, "text": "a" * 50})
    assert len(writer.written_shards) > 1
    assert writer.total_records == 50_000


def test_sharded_writer_no_empty_shards(tmp_path):
    out = tmp_path / "output"
    with ShardedJSONLWriter(out, shard_size_mb=1) as writer:
        pass  # write nothing
    assert len(writer.written_shards) == 0
    assert len(list(out.glob("*.jsonl"))) == 0
