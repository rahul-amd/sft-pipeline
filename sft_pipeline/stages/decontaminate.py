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
- Parallelism is PER SHARD, not per record: each worker reads, scans, and writes
  one whole input shard and returns only a small stats dict. Per-record fan-out
  was measured to lose to serial — the per-record work (tokenize + dict lookups)
  is microseconds, so pickling every record to a worker and back dominated.
  Per-shard fan-out has zero per-record IPC and no cross-shard barriers.
- The eval index can be large; to avoid pickling it to every worker on Linux we
  set it as a module global and let ``fork`` share it copy-on-write. On spawn/
  forkserver platforms (e.g. Windows/macOS dev) it is shipped once per worker
  via the pool initializer.
- Resume is shard-level: one output shard per input shard, and a completed shard
  is recorded in ``_shard_stats.jsonl`` (written only after the worker's atomic
  rename). A re-run skips shards already listed there and redoes at most the
  shards that were in flight.
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

import orjson

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import PipelineConfig
from sft_pipeline.decontam.eval_index import EvalNGramIndex, build_index
from sft_pipeline.decontam.normalize import tokenize
from sft_pipeline.storage import ensure_dir, iter_jsonl

logger = logging.getLogger(__name__)

STAGE = "decontaminate"

# Populated in workers (via fork inheritance or the pool initializer) and in
# the parent for the serial path.
_WORKER_INDEX: EvalNGramIndex | None = None


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------

def _worker_init_with_index(index: EvalNGramIndex) -> None:
    global _WORKER_INDEX
    _WORKER_INDEX = index


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


def _process_shard(
    shard_path: str,
    key: str,
    out_dir: str,
    removed_dir: str,
    index: EvalNGramIndex,
) -> dict:
    """Scan ONE input shard against *index*.

    Runs inline (serial), in a worker process (ProcessPool), or as a Ray task.
    Writes the clean and removed shards via tmp-file + atomic rename, and returns
    the stats row for ``_shard_stats.jsonl``. Only these small stats cross the
    process/node boundary — records never do.
    """
    out_tmp = Path(out_dir) / f"{key}.jsonl.tmp"
    rem_tmp = Path(removed_dir) / f"{key}.jsonl.tmp"
    n_in = n_removed = 0
    per_eval: Counter = Counter()
    per_source: Counter = Counter()

    with out_tmp.open("wb") as out_f, rem_tmp.open("wb") as rem_f:
        for rec in iter_jsonl(shard_path):
            n_in += 1
            match = index.match(tokenize(rec.get("prompt", "")))
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

    # Atomic publish; the parent records the shard as done only after receiving
    # this result (publish-before-record, required for safe resume).
    out_tmp.replace(Path(out_dir) / f"{key}.jsonl")
    rem_tmp.replace(Path(removed_dir) / f"{key}.jsonl")
    return {
        "shard": key, "input": n_in, "removed": n_removed,
        "per_eval": dict(per_eval), "per_source": dict(per_source),
    }


def _process_shard_pooled(shard_path: str, key: str, out_dir: str, removed_dir: str) -> dict:
    """ProcessPool adapter: read the fork/initializer-shared global index.

    Kept so the single-node pool doesn't pickle the (possibly large) index with
    every task. The Ray path instead passes the index explicitly via ray.put.
    """
    assert _WORKER_INDEX is not None, "worker index not initialized"
    return _process_shard(shard_path, key, out_dir, removed_dir, _WORKER_INDEX)


# ---------------------------------------------------------------------------
# Input shards + resume bookkeeping
# ---------------------------------------------------------------------------

# Stage key → (filename tag, output-dir accessor). The tag prefixes output
# shard names so stage1/stage2 shards of the same index don't collide.
_STAGE_SPECS: dict[str, tuple[str, str]] = {
    "stage1_collect": ("stage1", "stage1_collect"),
    "stage2_generate": ("stage2", "stage2_generate"),
}


def _collect_input_shards(cfg: PipelineConfig) -> list[tuple[str, Path]]:
    """(source_tag, shard_path) for input shards of the configured input_stages."""
    shards: list[tuple[str, Path]] = []
    for stage_key in cfg.decontaminate.input_stages:
        tag, cfg_attr = _STAGE_SPECS[stage_key]
        d = Path(getattr(cfg, cfg_attr).output_dir)
        if d.exists():
            shards.extend((tag, p) for p in sorted(d.glob("part-*.jsonl")))
        else:
            logger.warning(
                "Decontaminate: input stage '%s' output dir does not exist (%s) — skipping",
                stage_key, d,
            )
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


def _record_stats(stats: dict, stats_path: Path, agg: dict) -> None:
    """Append a completed shard's stats row and fold it into the aggregate."""
    with stats_path.open("ab") as sf:
        sf.write(orjson.dumps(stats) + b"\n")
    agg["input"] += stats["input"]
    agg["removed"] += stats["removed"]
    agg["per_eval"].update(stats["per_eval"])
    agg["per_source"].update(stats["per_source"])
    n_in, n_removed = stats["input"], stats["removed"]
    logger.info(
        "Decontaminate: %s — %d in, %d removed (%.2f%%)",
        stats["shard"], n_in, n_removed, 100.0 * n_removed / n_in if n_in else 0.0,
    )


def _run_shards_ray(
    todo: list[tuple[str, Path]],
    index: EvalNGramIndex,
    out_dir: str,
    removed_dir: str,
    stats_path: Path,
    agg: dict,
    cfg: PipelineConfig,
) -> None:
    """Fan the shard scan out across a Ray cluster.

    Workers write their own clean/removed shards to the shared filesystem and
    return only a stats dict; the driver (here) is the sole writer of the ledger.
    The index is broadcast once via ``ray.put`` — Ray materializes it in each
    worker rather than pickling it per task. A failed shard is left unrecorded
    so a re-run reprocesses it.
    """
    import ray

    from sft_pipeline import ray_utils

    ray_utils.ensure_ray(cfg)
    index_ref = ray.put(index)
    remote = ray.remote(num_cpus=1)(_process_shard)

    future_to_key = {
        remote.remote(str(shard), key, out_dir, removed_dir, index_ref): key
        for key, shard in todo
    }
    for _done, _total, key, result, err in ray_utils.as_completed(
        future_to_key, desc="Decontaminate shard"
    ):
        if err is not None:
            logger.error("Decontaminate: shard %s failed; will retry on resume.", key)
            continue
        _record_stats(result, stats_path, agg)


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
        "Decontaminate: building eval n-gram index (ngram_size=%d, min_gram_size=%d) "
        "from %d eval source(s)...",
        s.ngram_size, s.min_gram_size, len(s.evals),
    )
    index, per_eval_items = build_index(s.evals, s.ngram_size, s.min_gram_size)
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

    # State lives OUTSIDE the clean-pool dir (next to the report), so out_dir
    # stays a leaf containing only survivor shards — Stage 3 can glob it
    # (recursively or not) without ever ingesting bookkeeping files.
    report_path = Path(s.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = report_path.parent / "_shard_stats.jsonl"
    done, agg = _load_shard_stats(stats_path)

    todo = [
        (f"{tag}-{shard.stem}", shard)
        for tag, shard in input_shards
        if f"{tag}-{shard.stem}" not in done
    ]
    logger.info(
        "Decontaminate: %d input shard(s), %d to process, n_workers=%d",
        len(input_shards), len(todo), n_workers,
    )

    if s.distributed and todo:
        _run_shards_ray(todo, index, str(out_dir), str(removed_dir), stats_path, agg, cfg)
    elif n_workers <= 1 or len(todo) <= 1:
        # Serial: run shards inline.
        for key, shard in todo:
            stats = _process_shard(str(shard), key, str(out_dir), str(removed_dir), index)
            _record_stats(stats, stats_path, agg)
    else:
        # Single-node parallel: one whole shard per task, bounded submission
        # window so a huge shard list doesn't pile up futures. No barrier between
        # shards — a slow shard only occupies its own worker.
        executor = _make_executor(min(n_workers, len(todo)), index)
        try:
            shard_iter = iter(todo)
            pending = set()

            def _submit_next() -> bool:
                try:
                    key, shard = next(shard_iter)
                except StopIteration:
                    return False
                pending.add(executor.submit(
                    _process_shard_pooled, str(shard), key, str(out_dir), str(removed_dir)
                ))
                return True

            for _ in range(n_workers * 2):
                if not _submit_next():
                    break
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    _record_stats(fut.result(), stats_path, agg)
                    _submit_next()
        finally:
            executor.shutdown()

    survivors = agg["input"] - agg["removed"]
    report = {
        "total_input": agg["input"],
        "total_removed": agg["removed"],
        "total_survivors": survivors,
        "removal_rate": round(agg["removed"] / max(agg["input"], 1), 6),
        "ngram_size": s.ngram_size,
        "min_gram_size": s.min_gram_size,
        "gram_lengths": list(index.gram_lens),
        "total_eval_ngrams": index.total_grams,
        "eval_items_loaded": per_eval_items,
        # Eval items too short to index (below min_gram_size) — a large count
        # here means a match_field is mostly noise (e.g. single-word answers).
        "eval_items_dropped_short": dict(index.dropped_short),
        "removed_per_eval": dict(agg["per_eval"]),
        "removed_per_source": dict(agg["per_source"]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
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
