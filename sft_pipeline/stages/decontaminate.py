"""
Decontamination — remove prompts that overlap downstream eval benchmarks.

Runs between Stage 2 and Stage 3. Builds an n-gram containment index over the
configured eval datasets, scans every collected/generated prompt, and writes a
CLEAN survivor pool to ``decontaminate.output_dir`` (which Stage 3 then reads
instead of stage1/stage2). Also writes an uncapped record of every removed
prompt and a JSON report with per-eval / per-source removal counts.

Design notes
------------
- Matching is deterministic (no RNG), so the multiprocessing path yields exactly
  the same decisions as the serial path.
- The eval index can be large; to avoid pickling it to every worker on Linux we
  set it as a module global and let ``fork`` share it copy-on-write. On spawn/
  forkserver platforms (e.g. macOS dev) it is shipped via the pool initializer.
- Resume is shard-level: one output shard per input shard, and a completed shard
  is recorded in ``_shard_stats.jsonl`` (written only after atomic rename). A
  re-run skips shards already listed there and redoes at most one partial shard.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterator

import orjson

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import PipelineConfig
from sft_pipeline.decontam.eval_index import EvalNGramIndex, build_index
from sft_pipeline.decontam.normalize import tokenize
from sft_pipeline.storage import ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

STAGE = "decontaminate"

# Populated in workers (via fork inheritance or the pool initializer).
_WORKER_INDEX: EvalNGramIndex | None = None


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------

def _worker_init_with_index(index: EvalNGramIndex) -> None:
    global _WORKER_INDEX
    _WORKER_INDEX = index


def _scan_chunk(records: list[dict]) -> list[tuple[dict, tuple[int, str] | None]]:
    idx = _WORKER_INDEX
    return [(rec, idx.match(tokenize(rec.get("prompt", "")))) for rec in records]


def _chunked(it: Iterator[dict], size: int) -> Iterator[list[dict]]:
    chunk: list[dict] = []
    for rec in it:
        chunk.append(rec)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _make_executor(n_workers: int, index: EvalNGramIndex) -> ProcessPoolExecutor:
    """Create the worker pool, sharing the index cheaply where possible.

    fork  → set the module global and let workers inherit it copy-on-write
            (no pickling of the potentially large index).
    other → ship the index once per worker via the initializer.
    """
    if "fork" in mp.get_all_start_methods():
        global _WORKER_INDEX
        _WORKER_INDEX = index
        return ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork"))
    return ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init_with_index,
        initargs=(index,),
    )


def _scan_records(
    records: Iterator[dict],
    index: EvalNGramIndex,
    executor: ProcessPoolExecutor | None,
    chunk_size: int,
    n_workers: int,
) -> Iterator[tuple[dict, tuple[int, str] | None]]:
    """Yield (record, match) for each record. match = (eval_id, span) or None."""
    if executor is None:
        for rec in records:
            yield rec, index.match(tokenize(rec.get("prompt", "")))
        return

    chunks = _chunked(records, chunk_size)
    max_pending = n_workers * 2
    pending = set()
    # Prime the window.
    for _ in range(max_pending):
        try:
            pending.add(executor.submit(_scan_chunk, next(chunks)))
        except StopIteration:
            break
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            yield from fut.result()
            try:
                pending.add(executor.submit(_scan_chunk, next(chunks)))
            except StopIteration:
                pass


# ---------------------------------------------------------------------------
# Input shards + resume bookkeeping
# ---------------------------------------------------------------------------

def _collect_input_shards(cfg: PipelineConfig) -> list[tuple[str, Path]]:
    """(source_tag, shard_path) for every stage1/stage2 input shard, in order."""
    shards: list[tuple[str, Path]] = []
    for tag, dir_str in (
        ("stage1", cfg.stage1_collect.output_dir),
        ("stage2", cfg.stage2_generate.output_dir),
    ):
        d = Path(dir_str)
        if d.exists():
            shards.extend((tag, p) for p in sorted(d.glob("part-*.jsonl")))
    return shards


def _load_shard_stats(stats_path: Path) -> tuple[set[str], dict]:
    """Load completed-shard stats for resume. Returns (done_keys, aggregate)."""
    agg = {"input": 0, "removed": 0, "per_eval": Counter(), "per_source": Counter()}
    done: set[str] = set()
    if not stats_path.exists():
        return done, agg
    for row in iter_jsonl(stats_path):
        key = row.get("shard")
        if not key or key in done:
            continue
        done.add(key)
        agg["input"] += row.get("input", 0)
        agg["removed"] += row.get("removed", 0)
        agg["per_eval"].update(row.get("per_eval", {}))
        agg["per_source"].update(row.get("per_source", {}))
    logger.info("Decontaminate: resuming — %d shard(s) already done", len(done))
    return done, agg


def _process_shard(
    tag: str,
    shard: Path,
    key: str,
    index: EvalNGramIndex,
    executor: ProcessPoolExecutor | None,
    cfg_decontam,
    out_dir: Path,
    removed_dir: Path,
    stats_path: Path,
    agg: dict,
    n_workers: int,
) -> None:
    out_tmp = out_dir / f"{key}.jsonl.tmp"
    rem_tmp = removed_dir / f"{key}.jsonl.tmp"
    n_in = n_removed = 0
    per_eval: Counter = Counter()
    per_source: Counter = Counter()

    with out_tmp.open("wb") as out_f, rem_tmp.open("wb") as rem_f:
        for rec, match in _scan_records(
            iter_jsonl(shard), index, executor, cfg_decontam.worker_chunk_size, n_workers
        ):
            n_in += 1
            if match is None:
                out_f.write(orjson.dumps(rec) + b"\n")
            else:
                eid, span = match
                eval_name = index.eval_names[eid]
                n_removed += 1
                per_eval[eval_name] += 1
                per_source[rec.get("source", "")] += 1
                rem_f.write(orjson.dumps({
                    "prompt_id": rec.get("prompt_id", ""),
                    "source": rec.get("source", ""),
                    "matched_eval": eval_name,
                    "matched_ngram": span,
                }) + b"\n")

    # Atomic publish, then record the shard as done (order matters for resume).
    out_tmp.replace(out_dir / f"{key}.jsonl")
    rem_tmp.replace(removed_dir / f"{key}.jsonl")
    with stats_path.open("ab") as sf:
        sf.write(orjson.dumps({
            "shard": key, "input": n_in, "removed": n_removed,
            "per_eval": dict(per_eval), "per_source": dict(per_source),
        }) + b"\n")

    agg["input"] += n_in
    agg["removed"] += n_removed
    agg["per_eval"].update(per_eval)
    agg["per_source"].update(per_source)
    logger.info(
        "Decontaminate: %s — %d in, %d removed (%.2f%%)",
        key, n_in, n_removed, 100.0 * n_removed / n_in if n_in else 0.0,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_decontaminate(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s = cfg.decontaminate

    if not s.enabled or not s.evals:
        logger.info(
            "Decontaminate: %s — skipping; Stage 3 will read the raw stage1/stage2 pool.",
            "disabled" if not s.enabled else "no evals configured",
        )
        return

    cm.mark_stage_started(STAGE)
    t0 = time.time()

    logger.info(
        "Decontaminate: building eval n-gram index (ngram_size=%d) from %d eval source(s)...",
        s.ngram_size, len(s.evals),
    )
    index, per_eval_items = build_index(s.evals, s.ngram_size)
    if index.total_grams == 0:
        logger.warning(
            "Decontaminate: eval index is empty (no eval text loaded) — skipping. "
            "Stage 3 will read the raw pool. Check match_fields / splits."
        )
        return
    logger.info(
        "Decontaminate: index ready — %d grams, gram lengths=%s, evals=%s",
        index.total_grams, list(index.gram_lens), index.eval_names,
    )

    out_dir = ensure_dir(s.output_dir)
    removed_dir = ensure_dir(s.removed_dir)

    n_workers = s.n_workers if s.n_workers is not None else (os.cpu_count() or 1)
    n_workers = max(1, n_workers)

    input_shards = _collect_input_shards(cfg)
    if not input_shards:
        raise FileNotFoundError(
            f"Decontaminate: no part-*.jsonl shards found in "
            f"{cfg.stage1_collect.output_dir} (or {cfg.stage2_generate.output_dir}). "
            "Run Stage 1 first."
        )

    # State lives in a subdir so Stage 3's top-level *.jsonl glob never reads it.
    stats_path = ensure_dir(out_dir / "_state") / "shard_stats.jsonl"
    done, agg = _load_shard_stats(stats_path)

    logger.info(
        "Decontaminate: %d input shard(s), %d to process, n_workers=%d",
        len(input_shards), sum(1 for t, p in input_shards if f"{t}-{p.stem}" not in done),
        n_workers,
    )

    executor = _make_executor(n_workers, index) if n_workers > 1 else None
    try:
        for tag, shard in input_shards:
            key = f"{tag}-{shard.stem}"
            if key in done:
                continue
            _process_shard(
                tag, shard, key, index, executor, s,
                out_dir, removed_dir, stats_path, agg, n_workers,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    survivors = agg["input"] - agg["removed"]
    report = {
        "total_input": agg["input"],
        "total_removed": agg["removed"],
        "total_survivors": survivors,
        "removal_rate": round(agg["removed"] / max(agg["input"], 1), 6),
        "ngram_size": s.ngram_size,
        "gram_lengths": list(index.gram_lens),
        "total_eval_ngrams": index.total_grams,
        "eval_items_loaded": per_eval_items,
        "removed_per_eval": dict(agg["per_eval"]),
        "removed_per_source": dict(agg["per_source"]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    report_path = Path(s.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    cm.mark_stage_complete(STAGE, output_count=survivors)
    logger.info(
        "Decontaminate complete: input=%d  removed=%d (%.2f%%)  survivors=%d  elapsed=%.0fs",
        agg["input"], agg["removed"], 100.0 * report["removal_rate"], survivors,
        report["elapsed_seconds"],
    )
    logger.info("Decontaminate: report → %s  clean pool → %s", report_path, out_dir)
    if agg["per_eval"]:
        logger.info(
            "Decontaminate: removals by eval: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(agg["per_eval"].items())),
        )
