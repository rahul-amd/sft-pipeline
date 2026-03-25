"""
Local filesystem storage utilities.

Provides:
  - Path resolution helpers
  - Streaming JSONL reader (line-by-line, memory-safe for large files)
  - ShardedJSONLWriter — auto-splits output into ≤ N MB shard files
  - Parquet read/write helpers for embeddings
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

import orjson

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_dir(base_path: str, stage: str) -> Path:
    return ensure_dir(Path(base_path) / stage)


# ---------------------------------------------------------------------------
# JSONL reading
# ---------------------------------------------------------------------------

def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield parsed dicts from a JSONL file, one line at a time."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def iter_jsonl_dir(directory: str | Path) -> Iterator[dict]:
    """Yield records from all .jsonl files in a directory, sorted by filename."""
    d = Path(directory)
    if not d.exists():
        return
    for shard in sorted(d.glob("*.jsonl")):
        yield from iter_jsonl(shard)


def count_jsonl_lines(path: str | Path) -> int:
    """Count lines in a JSONL file without loading all into memory."""
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# JSONL writing — sharded
# ---------------------------------------------------------------------------

class ShardedJSONLWriter:
    """
    Writes JSONL records to a series of shard files under ``output_dir``.
    Opens a new shard when the current one exceeds ``shard_size_mb`` MB.

    Shard files are named ``part-000000.jsonl``, ``part-000001.jsonl``, …

    Usage::

        with ShardedJSONLWriter("/data/stage5", shard_size_mb=500) as writer:
            writer.write(record)
        shards = writer.written_shards   # list of paths written
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size_mb: int = 500,
        on_shard_complete: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """
        Args:
            output_dir: Directory where shard files are written.
            shard_size_mb: Maximum shard size in MiB before opening a new file.
            on_shard_complete: Optional callback(path, record_count, size_bytes)
                called each time a shard is finalized.
        """
        self._dir = ensure_dir(output_dir)
        self._shard_max_bytes = shard_size_mb * 1024 * 1024
        self._on_shard_complete = on_shard_complete

        self._shard_idx = 0
        self._current_file = None
        self._current_path: Path | None = None
        self._current_bytes = 0
        self._current_records = 0
        self._total_records = 0
        self.written_shards: list[str] = []

    def __enter__(self) -> ShardedJSONLWriter:
        self._open_shard()
        return self

    def __exit__(self, *_: object) -> None:
        self._close_shard()

    def _open_shard(self) -> None:
        name = f"part-{self._shard_idx:06d}.jsonl"
        self._current_path = self._dir / name
        self._current_file = self._current_path.open("wb")
        self._current_bytes = 0
        self._current_records = 0
        logger.debug("Opened shard %s", self._current_path)

    def _close_shard(self) -> None:
        if self._current_file and not self._current_file.closed:
            self._current_file.close()
            size = self._current_path.stat().st_size if self._current_path.exists() else 0
            if self._current_records > 0:
                self.written_shards.append(str(self._current_path))
                logger.info(
                    "Closed shard %s (%d records, %.1f MB)",
                    self._current_path.name,
                    self._current_records,
                    size / 1024 / 1024,
                )
                if self._on_shard_complete:
                    self._on_shard_complete(
                        str(self._current_path), self._current_records, size
                    )
            elif self._current_path and self._current_path.exists():
                self._current_path.unlink()  # remove empty shard

    def write(self, record: dict) -> None:
        if self._current_file is None:
            raise RuntimeError("Writer is not open. Use as context manager.")
        data = orjson.dumps(record) + b"\n"
        self._current_file.write(data)
        self._current_bytes += len(data)
        self._current_records += 1
        self._total_records += 1

        if self._current_bytes >= self._shard_max_bytes:
            self._close_shard()
            self._shard_idx += 1
            self._open_shard()

    def write_batch(self, records: list[dict]) -> None:
        for r in records:
            self.write(r)

    @property
    def total_records(self) -> int:
        return self._total_records


# ---------------------------------------------------------------------------
# Parquet helpers (for embeddings)
# ---------------------------------------------------------------------------

def write_parquet(path: str | Path, data: dict[str, Any]) -> None:
    """Write a dict of column arrays to Parquet. Requires pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(data)
    pq.write_table(table, str(path), compression="zstd")


def read_parquet_column(path: str | Path, column: str) -> Any:
    """Read a single column from a Parquet file as a numpy array."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=[column])
    return table[column].to_pylist()


def read_parquet(path: str | Path) -> dict[str, list]:
    """Read all columns from a Parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    return {name: table[name].to_pylist() for name in table.schema.names}
