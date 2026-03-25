"""
Stage 6 — Automated Quality Filtering.

Applies filters in cost-order (cheapest first, most expensive last):
  1. Structural  — field presence, length bounds, repetition
  2. Heuristic   — info density, contradiction markers
  3. Math        — SymPy expression verification (math/science domains)
  4. Code        — sandboxed code execution (code domain)
  5. LLM Judge   — sampled quality scoring (all domains)

Writes a filtered JSONL and a JSON filter report.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

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


def run_stage6(cfg: PipelineConfig, cm: CheckpointManager) -> None:
    s5 = cfg.stage5_inference
    s6 = cfg.stage6_filter

    stage5_dir = Path(s5.output_path).parent
    out_dir = Path(s6.output_path).parent
    ensure_dir(out_dir)

    cm.mark_stage_started(STAGE)
    cm.preload_processed(STAGE)

    rng = random.Random(cfg.global_.seed)

    # Per-filter rejection counters
    rejection_counts: dict[str, int] = defaultdict(int)
    domain_pass: dict[str, int] = defaultdict(int)
    domain_fail: dict[str, int] = defaultdict(int)

    total_input = 0
    total_passed = 0

    with ShardedJSONLWriter(out_dir, shard_size_mb=500) as writer:
        for record in iter_jsonl_dir(stage5_dir):
            pid = record.get("prompt_id", "")
            if cm.is_processed(pid, STAGE):
                total_passed += 1
                continue

            total_input += 1
            domain = record.get("domain", "general")
            rejection_reason: str | None = None

            # 1. Structural
            r = check_structural(record, s6.structural)
            if not r.passed:
                rejection_reason = f"structural:{r.reason}"

            # 2. Heuristic
            if rejection_reason is None:
                r = check_heuristic(record, s6.heuristic)
                if not r.passed:
                    rejection_reason = f"heuristic:{r.reason}"

            # 3. Math (only for relevant domains)
            if rejection_reason is None and s6.math.enabled:
                if domain in s6.math.domains:
                    r = check_math(record)
                    if not r.passed:
                        rejection_reason = f"math:{r.reason}"

            # 4. Code (only for code domain)
            if rejection_reason is None and s6.code.enabled:
                if domain in s6.code.domains:
                    r = check_code(record, s6.code)
                    if not r.passed:
                        rejection_reason = f"code:{r.reason}"

            # 5. LLM Judge (sampled)
            if rejection_reason is None:
                r = check_llm_judge(record, s6.llm_judge, rng=rng)
                if not r.passed:
                    rejection_reason = f"llm_judge:{r.reason}"

            if rejection_reason:
                rejection_counts[rejection_reason.split(":")[0]] += 1
                domain_fail[domain] += 1
                cm.mark_processed(pid, STAGE, status=ItemStatus.SKIPPED, error_msg=rejection_reason)
            else:
                writer.write(record)
                total_passed += 1
                domain_pass[domain] += 1
                cm.mark_processed(pid, STAGE, status=ItemStatus.SUCCESS)

            if total_input % 100_000 == 0:
                pass_rate = 100.0 * total_passed / max(total_input, 1)
                logger.info(
                    "Stage6: processed %d, passed %d (%.1f%%)",
                    total_input, total_passed, pass_rate,
                )

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

    cm.mark_stage_complete(STAGE, output_count=total_passed)
    logger.info(
        "Stage6 complete: input=%d, passed=%d (%.1f%%)",
        total_input, total_passed, 100.0 * pass_rate,
    )
