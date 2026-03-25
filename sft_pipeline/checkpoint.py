"""
DuckDB-based checkpoint tracker.

Every pipeline stage calls this module to:
  - Check whether an item has already been processed (skip on resume)
  - Mark items as processed after successful output
  - Mark whole stages as complete
  - Query overall pipeline status

The DuckDB file lives at config.global_.checkpoint_db (local disk).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Iterator

import duckdb

logger = logging.getLogger(__name__)


class ItemStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stage_status (
    stage_name      VARCHAR PRIMARY KEY,
    status          VARCHAR NOT NULL DEFAULT 'pending',
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    input_count     BIGINT,
    output_count    BIGINT,
    metadata        JSON
);

CREATE TABLE IF NOT EXISTS processed_items (
    item_id         VARCHAR NOT NULL,
    stage_name      VARCHAR NOT NULL,
    status          VARCHAR NOT NULL,
    output_shard    VARCHAR,
    processed_at    TIMESTAMP DEFAULT now(),
    error_msg       VARCHAR,
    PRIMARY KEY (item_id, stage_name)
);

CREATE TABLE IF NOT EXISTS shard_manifest (
    shard_id        VARCHAR PRIMARY KEY,
    stage_name      VARCHAR NOT NULL,
    path            VARCHAR NOT NULL,
    record_count    BIGINT,
    size_bytes      BIGINT,
    written_at      TIMESTAMP DEFAULT now()
);
"""


def prompt_id(text: str) -> str:
    """Stable, collision-resistant ID for a prompt string."""
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:32]


class CheckpointManager:
    """
    Thread-safe DuckDB checkpoint manager.

    Usage::

        cm = CheckpointManager("/data/run_001/checkpoints.duckdb")
        with cm:
            if not cm.is_processed("sha256:abc", "stage5_inference"):
                # ... do work ...
                cm.mark_processed("sha256:abc", "stage5_inference", shard="part-000.jsonl")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: duckdb.DuckDBPyConnection | None = None
        # In-memory set of already-processed item IDs per stage for fast lookup
        self._processed_cache: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> CheckpointManager:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self._conn is None:
            self._conn = duckdb.connect(str(self._path))
            self._conn.execute(_SCHEMA)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Stage-level operations
    # ------------------------------------------------------------------

    def mark_stage_started(self, stage: str, input_count: int | None = None) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO stage_status (stage_name, status, started_at, input_count)
                VALUES (?, 'running', now(), ?)
                ON CONFLICT (stage_name) DO UPDATE
                SET status = 'running', started_at = now(), input_count = ?
                """,
                [stage, input_count, input_count],
            )

    def mark_stage_complete(
        self, stage: str, output_count: int | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO stage_status (stage_name, status, completed_at, output_count)
                VALUES (?, 'completed', now(), ?)
                ON CONFLICT (stage_name) DO UPDATE
                SET status = 'completed', completed_at = now(), output_count = ?
                """,
                [stage, output_count, output_count],
            )

    def mark_stage_failed(self, stage: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO stage_status (stage_name, status, metadata)
                VALUES (?, 'failed', ?)
                ON CONFLICT (stage_name) DO UPDATE
                SET status = 'failed', metadata = ?
                """,
                [stage, f'{{"error": "{error}"}}', f'{{"error": "{error}"}}'],
            )

    def is_stage_complete(self, stage: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM stage_status WHERE stage_name = ?", [stage]
            ).fetchone()
        return row is not None and row[0] == "completed"

    def all_stage_statuses(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT stage_name, status, started_at, completed_at, input_count, output_count "
                "FROM stage_status ORDER BY started_at NULLS LAST"
            ).fetchall()
        return [
            {
                "stage": r[0], "status": r[1], "started_at": r[2],
                "completed_at": r[3], "input_count": r[4], "output_count": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Item-level operations
    # ------------------------------------------------------------------

    def preload_processed(self, stage: str) -> None:
        """
        Cache all processed item IDs for a stage into memory.
        Call once at the start of each stage for O(1) is_processed lookups.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT item_id FROM processed_items WHERE stage_name = ? AND status = 'success'",
                [stage],
            ).fetchall()
        self._processed_cache[stage] = {r[0] for r in rows}
        logger.info(
            "Preloaded %d already-processed items for stage %s",
            len(self._processed_cache[stage]), stage,
        )

    def is_processed(self, item_id: str, stage: str) -> bool:
        if stage in self._processed_cache:
            return item_id in self._processed_cache[stage]
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM processed_items WHERE item_id = ? AND stage_name = ? AND status = 'success'",
                [item_id, stage],
            ).fetchone()
        return row is not None

    def mark_processed(
        self,
        item_id: str,
        stage: str,
        status: ItemStatus = ItemStatus.SUCCESS,
        shard: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO processed_items (item_id, stage_name, status, output_shard, error_msg)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (item_id, stage_name) DO UPDATE
                SET status = ?, output_shard = ?, error_msg = ?, processed_at = now()
                """,
                [item_id, stage, status, shard, error_msg,
                 status, shard, error_msg],
            )
        if status == ItemStatus.SUCCESS and stage in self._processed_cache:
            self._processed_cache[stage].add(item_id)

    def mark_processed_batch(
        self,
        items: list[tuple[str, ItemStatus, str | None]],  # (item_id, status, shard)
        stage: str,
    ) -> None:
        """Batch-insert processed item records."""
        rows = [
            (item_id, stage, status, shard, None)
            for item_id, status, shard in items
        ]
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO processed_items (item_id, stage_name, status, output_shard, error_msg)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (item_id, stage_name) DO UPDATE
                SET status = ?, output_shard = ?, processed_at = now()
                """,
                [(r[0], r[1], r[2], r[3], r[4], r[2], r[3]) for r in rows],
            )
        if stage in self._processed_cache:
            for item_id, status, _ in items:
                if status == ItemStatus.SUCCESS:
                    self._processed_cache[stage].add(item_id)

    # ------------------------------------------------------------------
    # Shard manifest
    # ------------------------------------------------------------------

    def register_shard(
        self,
        shard_id: str,
        stage: str,
        path: str,
        record_count: int,
        size_bytes: int,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO shard_manifest (shard_id, stage_name, path, record_count, size_bytes)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (shard_id) DO NOTHING
                """,
                [shard_id, stage, path, record_count, size_bytes],
            )

    def get_shards(self, stage: str) -> list[str]:
        """Return paths of all written shards for a stage, ordered by shard_id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path FROM shard_manifest WHERE stage_name = ? ORDER BY shard_id",
                [stage],
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def processed_count(self, stage: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*) FROM processed_items WHERE stage_name = ? AND status = 'success'",
                [stage],
            ).fetchone()
        return row[0] if row else 0
