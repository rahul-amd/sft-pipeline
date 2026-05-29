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
import random
import time
from collections import defaultdict
from pathlib import Path

import orjson

from sft_pipeline.checkpoint import CheckpointManager, ItemStatus
from sft_pipeline.config import PipelineConfig
from sft_pipeline.filters.code_verifier import check_code
from sft_pipeline.filters.heuristic import check_heuristic
from sft_pipeline.filters.llm_judge import check_llm_judge
from sft_pipeline.filters.math_verifier import check_math
from sft_pipeline.filters.structural import check_structural
from sft_pipeline.storage import ShardedJSONLWriter, ensure_dir, iter_jsonl_dir

logger = logging.getLogger(__name__)

STAGE = "stage6_filter"


def _apply_filters(record: dict, s6, rng: random.Random) -> str | None:
    """
    Run the full filter chain on *record*.

    Returns the rejection reason string (``"filter_name:sub_reason"``) if the
    record was rejected, or ``None`` if it passed all filters.
    """
    domain = record.get("domain", "general")

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

    # Warn about expensive filters requiring parsed fields (same as normal mode)
    if s6.math.enabled or s6.llm_judge.enabled:
        logger.warning(
            "Stage6: math or llm_judge filters are enabled.  These filters require "
            "'reasoning' and 'answer' fields.  If Stage 5 output only contains "
            "'response', those checks will be skipped per-record."
        )

    collected: list[dict] = []
    total_scanned = 0

    for record in iter_jsonl_dir(stage5_dir):
        total_scanned += 1
        reason = _apply_filters(record, s6, rng)
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
    out_dir = Path(s6.output_dir)
    ensure_dir(out_dir)

    cm.mark_stage_started(STAGE)

    checkpoint_items = s6.checkpoint_items
    if checkpoint_items:
        already_done = cm.processed_count(STAGE)
        cm.preload_processed(STAGE)
    else:
        already_done = 0
        logger.info("Stage6: per-item checkpointing disabled — no resume support for this run")

    logger.info(
        "Stage6: starting — input=%s  output=%s  already_done=%d  checkpoint_items=%s",
        stage5_dir, out_dir, already_done, checkpoint_items,
    )

    rng = random.Random(cfg.global_.seed)

    # Per-filter rejection counters
    rejection_counts: dict[str, int] = defaultdict(int)
    domain_pass: dict[str, int] = defaultdict(int)
    domain_fail: dict[str, int] = defaultdict(int)

    total_input = 0
    total_passed = 0

    if s6.math.enabled or s6.llm_judge.enabled:
        logger.warning(
            "Stage6: math or llm_judge filters are enabled.  These filters require "
            "'reasoning' and 'answer' fields.  If Stage 5 output only contains "
            "'response', those checks will be skipped per-record.  "
            "Parse the responses first or disable these filters."
        )

    t0 = time.time()
    LOG_EVERY = 10_000
    CHECKPOINT_EVERY = 100_000
    checkpoint_buffer: list[tuple[str, ItemStatus, str | None]] = []

    with ShardedJSONLWriter(out_dir, shard_size_mb=500) as writer:
        for record in iter_jsonl_dir(stage5_dir):
            pid = record.get("prompt_id", "")
            if checkpoint_items and cm.is_processed(pid, STAGE):
                continue

            total_input += 1
            domain = record.get("domain", "general")

            rejection_reason = _apply_filters(record, s6, rng)

            if rejection_reason:
                rejection_counts[rejection_reason.split(":")[0]] += 1
                domain_fail[domain] += 1
                if checkpoint_items:
                    checkpoint_buffer.append((pid, ItemStatus.SKIPPED, rejection_reason))
            else:
                writer.write(record)
                total_passed += 1
                domain_pass[domain] += 1
                if checkpoint_items:
                    checkpoint_buffer.append((pid, ItemStatus.SUCCESS, None))

            if checkpoint_items and total_input % CHECKPOINT_EVERY == 0:
                cm.mark_processed_batch(checkpoint_buffer, STAGE)
                checkpoint_buffer.clear()

            if total_input % LOG_EVERY == 0:
                elapsed = time.time() - t0
                rate = total_input / elapsed if elapsed > 0 else 0
                pass_rate = 100.0 * total_passed / total_input
                rejection_summary = ", ".join(
                    f"{k}={v}" for k, v in sorted(rejection_counts.items())
                ) or "none"
                logger.info(
                    "Stage6: %d processed  passed=%d (%.1f%%)  rate=%.0f/s"
                    "  rejections=[%s]",
                    total_input, total_passed, pass_rate, rate, rejection_summary,
                )

        if checkpoint_items and checkpoint_buffer:
            cm.mark_processed_batch(checkpoint_buffer, STAGE)
            checkpoint_buffer.clear()

    # Write filter report
    pass_rate = total_passed / max(total_input, 1)
    report = {
        "total_input": total_input,
        "total_passed": total_passed,
        "pass_rate": round(pass_rate, 4),
        "rejection_counts": dict(rejection_counts),
        "domain_pass_counts": dict(domain_pass),
        "domain_fail_counts": dict(domain_fail),
    }
    report_path = Path(s6.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Filter report written to %s", report_path)

    elapsed = time.time() - t0
    rate = total_input / elapsed if elapsed > 0 else 0
    cm.mark_stage_complete(STAGE, output_count=already_done + total_passed)
    logger.info(
        "Stage6 complete: input=%d  passed=%d (%.1f%%)  elapsed=%.0fs  rate=%.0f/s",
        total_input, total_passed, 100.0 * pass_rate, elapsed, rate,
    )
    if rejection_counts:
        logger.info(
            "Stage6 rejection breakdown: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(rejection_counts.items())),
        )
