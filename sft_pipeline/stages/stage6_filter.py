"""
Stage 6 — Automated Quality Filtering.

Applies filters in cost-order (cheapest first, most expensive last):
  1. Structural  — field presence, length bounds, repetition
  2. Heuristic   — info density, contradiction markers
  3. Math        — SymPy expression verification (math/science domains)
  4. Code        — sandboxed code execution (code domain)
  5. LLM Judge   — sampled quality scoring (all domains)

Writes a filtered JSONL and a JSON filter report.

Debug mode (``stage6_filter.debug_rejections: true``):
  Scans the input, collects the first N rejected records with their reasons,
  writes them to ``debug_rejection_path``, then stops.  The checkpoint DB is
  not touched, so the run is completely safe to repeat after inspection.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import orjson

from sft_pipeline.checkpoint import CheckpointManager
from sft_pipeline.config import PipelineConfig
from sft_pipeline.filters.code_verifier import check_code
from sft_pipeline.filters.heuristic import check_heuristic
from sft_pipeline.filters.llm_judge import check_llm_judge
from sft_pipeline.filters.math_verifier import check_math
from sft_pipeline.filters.structural import check_structural
from sft_pipeline.storage import ensure_dir, iter_jsonl, iter_jsonl_dir

logger = logging.getLogger(__name__)

STAGE = "stage6_filter"


def _parse_record(record: dict, delimiters) -> None:
    """
    Populate ``reasoning`` / ``answer`` from the raw ``response`` in place.

    Stage 5 writes raw responses only; the math/code filters (and the export
    schema) need the parsed fields. Uses the configured reasoning delimiters —
    set ``stage5_inference.delimiters`` to whatever the teacher actually
    emitted (e.g. ``<|channel>thought`` / ``<channel|>``).
    """
    from sft_pipeline.inference.output_parser import parse_output

    if record.get("reasoning") or record.get("answer"):
        return  # already parsed
    response = record.get("response")
    if not response:
        return
    parsed = parse_output(response, delimiters)
    record["reasoning"] = parsed.reasoning
    record["answer"] = parsed.answer


def _apply_filters(record: dict, s6, rng: random.Random, delimiters=None) -> str | None:
    """
    Run the full filter chain on *record*.

    Returns the rejection reason string (``"filter_name:sub_reason"``) if the
    record was rejected, or ``None`` if it passed all filters.
    """
    domain = record.get("domain", "general")

    if s6.parse_responses and delimiters is not None:
        _parse_record(record, delimiters)

    # 1. Structural
    r = check_structural(record, s6.structural)
    if not r.passed:
        return f"structural:{r.reason}"

    # 2. Heuristic
    r = check_heuristic(record, s6.heuristic)
    if not r.passed:
        return f"heuristic:{r.reason}"

    # 3. Math (only for relevant domains)
    if s6.math.enabled and domain in s6.math.domains:
        r = check_math(record)
        if not r.passed:
            return f"math:{r.reason}"

    # 4. Code (only for code domain)
    if s6.code.enabled and domain in s6.code.domains:
        r = check_code(record, s6.code)
        if not r.passed:
            return f"code:{r.reason}"

    # 5. LLM Judge (sampled)
    r = check_llm_judge(record, s6.llm_judge, rng=rng)
    if not r.passed:
        return f"llm_judge:{r.reason}"

    return None


# ---------------------------------------------------------------------------
# Per-shard filtering (shard-map model)
# ---------------------------------------------------------------------------
#
# The filter chain is CPU/subprocess-bound (code sandbox dominates) and every
# record is independent, so the unit of parallelism is a whole INPUT SHARD:
# each task filters one shard in input order, writes its surviving records to
# its own output shard (tmp + atomic rename), and returns only a small stats
# dict. Records never cross a process/node boundary. This one function backs the
# serial, single-node-pool, and Ray paths alike.


def _process_shard(
    shard_path: str,
    key: str,
    out_dir: str,
    s6,
    delimiters,
    seed: int,
) -> dict:
    """Filter one input shard; write survivors to ``out_dir/{key}.jsonl``.

    Returns the stats row for ``_shard_stats.jsonl``. The llm_judge sample RNG is
    seeded deterministically from ``(seed, key)`` so the sampled subset is stable
    per run and independent of task scheduling.
    """
    rng = random.Random(f"{seed}:{key}")
    out_tmp = Path(out_dir) / f"{key}.jsonl.tmp"
    n_in = n_pass = 0
    rejection_counts: Counter = Counter()
    domain_pass: Counter = Counter()
    domain_fail: Counter = Counter()

    with out_tmp.open("wb") as out_f:
        for rec in iter_jsonl(shard_path):
            n_in += 1
            domain = rec.get("domain", "general")
            reason = _apply_filters(rec, s6, rng, delimiters=delimiters)
            if reason:
                rejection_counts[reason.split(":")[0]] += 1
                domain_fail[domain] += 1
            else:
                out_f.write(orjson.dumps(rec) + b"\n")  # _parse_record enrichment persists
                n_pass += 1
                domain_pass[domain] += 1

    out_tmp.replace(Path(out_dir) / f"{key}.jsonl")
    return {
        "shard": key, "input": n_in, "passed": n_pass,
        "rejection_counts": dict(rejection_counts),
        "domain_pass": dict(domain_pass), "domain_fail": dict(domain_fail),
    }


def _load_shard_stats(stats_path: Path) -> tuple[set[str], dict]:
    """Load completed-shard stats for resume. Returns (done_keys, aggregate)."""
    agg = {
        "input": 0, "passed": 0,
        "rejection_counts": Counter(), "domain_pass": Counter(), "domain_fail": Counter(),
    }
    done: set[str] = set()
    if not stats_path.exists():
        return done, agg
    for row in iter_jsonl(stats_path):
        key = row.get("shard")
        if not key or key in done:
            continue
        done.add(key)
        agg["input"] += row.get("input", 0)
        agg["passed"] += row.get("passed", 0)
        agg["rejection_counts"].update(row.get("rejection_counts", {}))
        agg["domain_pass"].update(row.get("domain_pass", {}))
        agg["domain_fail"].update(row.get("domain_fail", {}))
    logger.info("Stage6: resuming — %d shard(s) already done", len(done))
    return done, agg


def _record_stats(stats: dict, stats_path: Path, agg: dict) -> None:
    """Append a completed shard's stats row and fold it into the aggregate."""
    with stats_path.open("ab") as sf:
        sf.write(orjson.dumps(stats) + b"\n")
    agg["input"] += stats["input"]
    agg["passed"] += stats["passed"]
    agg["rejection_counts"].update(stats["rejection_counts"])
    agg["domain_pass"].update(stats["domain_pass"])
    agg["domain_fail"].update(stats["domain_fail"])
    n_in, n_pass = stats["input"], stats["passed"]
    logger.info(
        "Stage6: %s — %d in, %d passed (%.1f%%)",
        stats["shard"], n_in, n_pass, 100.0 * n_pass / n_in if n_in else 0.0,
    )


def _run_shards_ray(todo, s6, delimiters, seed, out_dir, stats_path, agg, cfg) -> None:
    """Fan the shard filter out across a Ray cluster.

    Workers write their own output shards to the shared filesystem and return
    only stats; the driver is the sole writer of the ledger. A failed shard is
    left unrecorded so a re-run reprocesses it.
    """
    import ray

    from sft_pipeline import ray_utils

    ray_utils.ensure_ray(cfg)
    remote = ray.remote(num_cpus=1)(_process_shard)
    future_to_key = {
        remote.remote(str(shard), key, out_dir, s6, delimiters, seed): key
        for key, shard in todo
    }
    for _done, _total, key, result, err in ray_utils.as_completed(
        future_to_key, desc="Stage6 shard"
    ):
        if err is not None:
            logger.error("Stage6: shard %s failed; will retry on resume.", key)
            continue
        _record_stats(result, stats_path, agg)


def _run_debug(cfg: PipelineConfig) -> None:
    """
    Debug mode: scan input, collect the first N rejected records, write them
    with their ``_rejection_reason`` field to a JSONL file, then exit.

    Does NOT write normal stage output and does NOT modify the checkpoint DB.
    """
    s5 = cfg.stage5_inference
    s6 = cfg.stage6_filter

    stage5_dir = Path(s5.output_dir)
    limit = s6.debug_rejection_limit
    debug_path = Path(s6.debug_rejection_path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.global_.seed)

    logger.info(
        "Stage6 DEBUG MODE: scanning %s for the first %d rejected records → %s",
        stage5_dir, limit, debug_path,
    )

    if not s6.parse_responses and (s6.math.enabled or s6.llm_judge.enabled):
        logger.warning(
            "Stage6: parse_responses is off but math/llm_judge filters are enabled. "
            "They need 'reasoning'/'answer' fields; on raw Stage 5 'response' "
            "records those checks silently pass everything."
        )

    collected: list[dict] = []
    total_scanned = 0

    for record in iter_jsonl_dir(stage5_dir):
        total_scanned += 1
        reason = _apply_filters(record, s6, rng, delimiters=cfg.stage5_inference.delimiters)
        if reason is not None:
            entry = dict(record)  # shallow copy — don't mutate the original
            entry["_rejection_reason"] = reason
            collected.append(entry)
            logger.debug(
                "DEBUG rejected #%d  pid=%s  reason=%s",
                len(collected), record.get("prompt_id", "?"), reason,
            )
            if len(collected) >= limit:
                break

    # Write debug file
    with debug_path.open("wb") as fh:
        for entry in collected:
            fh.write(orjson.dumps(entry) + b"\n")

    logger.info(
        "Stage6 DEBUG: scanned %d records, collected %d rejections → %s",
        total_scanned, len(collected), debug_path,
    )


def run_stage6(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s5 = cfg.stage5_inference
    s6 = cfg.stage6_filter

    # ── Debug mode: collect sample rejections and quit without touching the DB ──
    if s6.debug_rejections:
        _run_debug(cfg)
        return

    stage5_dir = Path(s5.output_dir)
    out_dir = ensure_dir(s6.output_dir)

    cm.mark_stage_started(STAGE)

    if not s6.parse_responses and (s6.math.enabled or s6.llm_judge.enabled):
        logger.warning(
            "Stage6: parse_responses is off but math/llm_judge filters are enabled. "
            "They need 'reasoning'/'answer' fields; on raw Stage 5 'response' "
            "records those checks silently pass everything. "
            "Enable stage6_filter.parse_responses or disable these filters."
        )

    n_workers = s6.n_workers if s6.n_workers is not None else (os.cpu_count() or 1)
    n_workers = max(1, n_workers)

    input_shards = sorted(stage5_dir.glob("*.jsonl"))
    if not input_shards:
        raise FileNotFoundError(
            f"Stage6: no *.jsonl shards found in {stage5_dir}. Run Stage 5 first."
        )

    # Resume ledger lives in a subdir so it isn't globbed as stage output.
    stats_path = ensure_dir(out_dir / "_state") / "shard_stats.jsonl"
    done, agg = _load_shard_stats(stats_path)
    todo = [(shard.stem, shard) for shard in input_shards if shard.stem not in done]

    logger.info(
        "Stage6: starting — input=%s  output=%s  %d shard(s), %d to process  "
        "distributed=%s  n_workers=%d",
        stage5_dir, out_dir, len(input_shards), len(todo), s6.distributed, n_workers,
    )

    delimiters = cfg.stage5_inference.delimiters
    seed = cfg.global_.seed
    t0 = time.time()

    if s6.distributed and todo:
        _run_shards_ray(todo, s6, delimiters, seed, str(out_dir), stats_path, agg, cfg)
    elif n_workers <= 1 or len(todo) <= 1:
        for key, shard in todo:
            _record_stats(
                _process_shard(str(shard), key, str(out_dir), s6, delimiters, seed),
                stats_path, agg,
            )
    else:
        # Single-node parallel: one whole shard per task, bounded submission
        # window. No barrier — a slow shard occupies only its own worker.
        executor = ProcessPoolExecutor(max_workers=min(n_workers, len(todo)))
        try:
            shard_iter = iter(todo)
            pending = set()

            def _submit_next() -> bool:
                try:
                    key, shard = next(shard_iter)
                except StopIteration:
                    return False
                pending.add(executor.submit(
                    _process_shard, str(shard), key, str(out_dir), s6, delimiters, seed
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

    # Write filter report
    total_input = agg["input"]
    total_passed = agg["passed"]
    pass_rate = total_passed / max(total_input, 1)
    report = {
        "total_input": total_input,
        "total_passed": total_passed,
        "pass_rate": round(pass_rate, 4),
        "rejection_counts": dict(agg["rejection_counts"]),
        "domain_pass_counts": dict(agg["domain_pass"]),
        "domain_fail_counts": dict(agg["domain_fail"]),
    }
    report_path = Path(s6.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Filter report written to %s", report_path)

    elapsed = time.time() - t0
    rate = total_input / elapsed if elapsed > 0 else 0
    cm.mark_stage_complete(STAGE, output_count=total_passed)
    logger.info(
        "Stage6 complete: input=%d  passed=%d (%.1f%%)  elapsed=%.0fs  rate=%.0f/s",
        total_input, total_passed, 100.0 * pass_rate, elapsed, rate,
    )
    if agg["rejection_counts"]:
        logger.info(
            "Stage6 rejection breakdown: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(agg["rejection_counts"].items())),
        )
