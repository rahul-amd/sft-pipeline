#!/usr/bin/env python3
"""
Sanity check for Stage 6 verifiers (math + code) against the Stage 5 dataset.

It measures how well the deterministic verifiers agree with a strong LLM judge
that acts as ground truth.

Pipeline (each step is resumable; skips if its output already exists):
  1. peek    — print the dataset schema + a couple of sample rows
  2. sample  — stream `rahular/sft-pipeline-stage5`, take N records per domain
  3. verify  — parse each response on </think>, run the matching verifier
  4. judge   — score each response with the reference LLM judge (ground truth),
               by shelling out to scripts/llm_judge_eval.py
  5. metrics — confusion matrix (verifier decision vs judge label) per domain

Ground-truth convention
------------------------
The verifier is the classifier under test.  Positive class = the verifier
REJECTS a response (flags it as low quality).

    "bad"  := judge answer_correctness < --good-threshold   (default 4, on 1–5)
    "good" := judge answer_correctness >= --good-threshold

    TP: verifier rejects & response is bad   (correct rejection)
    FP: verifier rejects & response is good   (over-filtering — the costly error)
    FN: verifier passes  & response is bad    (missed bad sample)
    TN: verifier passes  & response is good

Response parsing
----------------
The Stage 5 dataset stores one raw `response` string in a malformed "harmony"
format: it opens with `<|channel>thought` and has NO `<think>`/`</think>` or
`<answer>` delimiters and no explicit reasoning↔answer separator.  We strip the
leading channel marker and then reuse the pipeline's own parser
(`sft_pipeline.inference.output_parser.parse_output`), which — finding none of
its configured delimiters — falls back to heuristic splitting on conclusion
markers ("therefore", "final answer", ...).  This is exactly what Stage 5 does
in production, so the verifier is exercised on the same reasoning/answer split.

Usage
-----
    # Configure everything (HF token, judge API/token, sample sizes) in
    # sanity-checks/config.yaml (copy from config.example.yaml). It is loaded
    # automatically; any CLI flag overrides the matching config value.
    python sanity-checks/sanity_check.py

    # Override the judge on the CLI (e.g. Z.AI glm-5.2, concurrency limit 10):
    python sanity-checks/sanity_check.py \
        --judge-provider openai \
        --judge-model glm-5.2 \
        --judge-api-base https://api.z.ai/api/paas/v4/ \
        --judge-api-key "$ZAI_API_KEY" \
        --judge-concurrency 10

    # run a single step
    python sanity-checks/sanity_check.py --steps peek
    python sanity-checks/sanity_check.py --steps metrics
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import orjson

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"

# Make the repo importable when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sanity_check")

JUDGE_SCRIPT = REPO_ROOT / "scripts" / "llm_judge_eval.py"

# Candidate field names, in priority order, for tolerating schema drift.
_RESPONSE_KEYS = ["response", "raw_response", "answer", "output", "completion"]
_PROMPT_KEYS = ["prompt", "question", "instruction"]
_DOMAIN_KEYS = ["domain", "domain_hint"]
_ID_KEYS = ["prompt_id", "id"]


# ---------------------------------------------------------------------------
# Small JSONL helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(orjson.loads(line))
    return out


def _first_present(record: dict, keys: list[str]) -> str:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def get_response(record: dict) -> str:
    v = _first_present(record, _RESPONSE_KEYS)
    if v:
        return v
    # Fall back to the last assistant turn in a messages array.
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c
    return ""


def get_prompt(record: dict) -> str:
    v = _first_present(record, _PROMPT_KEYS)
    if v:
        return v
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c
    return ""


def get_domain(record: dict) -> str:
    return _first_present(record, _DOMAIN_KEYS) or "unknown"


def get_id(record: dict) -> str:
    v = _first_present(record, _ID_KEYS)
    if v:
        return v
    import hashlib
    return "h:" + hashlib.sha256(get_prompt(record).encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Response parsing (mirror of what the real pipeline does at Stage 5→6)
# ---------------------------------------------------------------------------
#
# The Stage 5 dataset stores a single raw `response` string in a mangled
# "harmony" format.  Every response has the structure:
#
#     <|channel>thought
#     ... reasoning trace ...
#     <channel|>
#     ... final answer ...
#
# So `<|channel>thought` / `<channel|>` act as the thinking delimiters.  We
# feed them to the pipeline's own parser (output_parser.parse_output), which
# extracts reasoning = the think block, answer = everything after it.

THINK_OPEN = "<|channel>thought"
THINK_CLOSE = "<channel|>"


def parse_reasoning_answer(response: str) -> tuple[str, str, bool]:
    """
    Split a raw Stage 5 response into (reasoning, answer, valid) using the
    pipeline's own output parser with the dataset's actual delimiters.
    """
    from sft_pipeline.config import ReasoningDelimiters
    from sft_pipeline.inference.output_parser import parse_output

    delims = ReasoningDelimiters(think_start=THINK_OPEN, think_end=THINK_CLOSE)
    parsed = parse_output(response or "", delims)
    return parsed.reasoning, parsed.answer, parsed.valid


# ---------------------------------------------------------------------------
# Step 1: peek
# ---------------------------------------------------------------------------

def step_peek(args) -> None:
    from datasets import load_dataset

    logger.info("Peeking at %s (split=%s, streaming) ...", args.dataset, args.split)
    ds = load_dataset(
        args.dataset, split=args.split, streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    features = getattr(ds, "features", None)
    if features:
        print("\nFeatures:")
        for name, feat in features.items():
            print(f"  {name}: {feat}")

    print("\nSample rows:")
    for i, row in enumerate(ds):
        if i >= args.peek_rows:
            break
        keys = list(row.keys())
        print(f"\n--- row {i} — keys: {keys}")
        for k in keys:
            v = row[k]
            s = v if isinstance(v, str) else repr(v)
            if len(s) > 300:
                s = s[:300] + f"… (+{len(s) - 300} chars)"
            print(f"  {k}: {s}")


# ---------------------------------------------------------------------------
# Step 2: sample
# ---------------------------------------------------------------------------

def step_sample(args) -> None:
    """
    The dataset is arranged domain-contiguously (all math first, then code, ...),
    so each domain gets its own stream with a skip offset (`domain_offsets`)
    instead of scanning millions of foreign-domain rows from the head.
    """
    from datasets import load_dataset

    for d in args.domains:
        path = args.data_dir / f"{d}.jsonl"
        if path.exists() and not args.force:
            existing = sum(1 for _ in path.open("rb"))
            if existing >= args.n:
                logger.info("sample[%s]: %s has %d rows — skipping (use --force)",
                            d, path, existing)
                continue
            logger.info("sample[%s]: %s has only %d/%d rows — re-sampling",
                        d, path, existing, args.n)

        offset = int(args.domain_offsets.get(d, 0))
        logger.info("sample[%s]: streaming %s from row %d for %d records ...",
                    d, args.dataset, offset, args.n)
        ds = load_dataset(
            args.dataset, split=args.split, streaming=True,
            token=os.environ.get("HF_TOKEN"),
        )
        if offset:
            ds = ds.skip(offset)

        bucket: list[dict] = []
        seen = 0
        for row in ds:
            seen += 1
            if get_domain(row) == d:
                bucket.append(row)
                if len(bucket) >= args.n:
                    break
            if seen % 20_000 == 0:
                logger.info("sample[%s]: scanned %d  collected=%d", d, seen, len(bucket))
            if seen >= args.max_scan:
                logger.warning("sample[%s]: hit --max-scan=%d with only %d collected "
                               "(is the domain offset right?)", d, args.max_scan, len(bucket))
                break

        write_jsonl(path, bucket)
        logger.info("sample[%s]: wrote %d records (scanned %d rows) → %s",
                    d, len(bucket), seen, path)


# ---------------------------------------------------------------------------
# Step 3: verify
# ---------------------------------------------------------------------------

def step_verify(args) -> None:
    """
    Run the production Stage 6 deterministic chain (structural → heuristic →
    math/code, short-circuit) on each record — the same order and configs as
    stage6_filter._apply_filters, minus the LLM judge (that's our ground truth).
    """
    from sft_pipeline.config import (
        CodeFilterConfig,
        HeuristicFilterConfig,
        StructuralFilterConfig,
    )
    from sft_pipeline.filters.code_verifier import check_code
    from sft_pipeline.filters.heuristic import check_heuristic
    from sft_pipeline.filters.math_verifier import check_math
    from sft_pipeline.filters.structural import check_structural

    structural_cfg = StructuralFilterConfig()
    heuristic_cfg = HeuristicFilterConfig()
    code_cfg = CodeFilterConfig(
        enabled=True, sandbox="subprocess",
        timeout_seconds=args.code_timeout, domains=["code"],
    )

    def run_chain(record: dict, domain: str) -> tuple[bool, str]:
        r = check_structural(record, structural_cfg)
        if not r.passed:
            return False, f"structural:{r.reason}"
        r = check_heuristic(record, heuristic_cfg)
        if not r.passed:
            return False, f"heuristic:{r.reason}"
        if domain == "math":
            r = check_math(record)
            if not r.passed:
                return False, f"math:{r.reason}"
            return True, f"math:{r.reason}" if r.reason else ""
        if domain == "code":
            r = check_code(record, code_cfg)
            if not r.passed:
                return False, f"code:{r.reason}"
            return True, f"code:{r.reason}" if r.reason else ""
        return True, "no_verifier"

    for d in args.domains:
        src = args.data_dir / f"{d}.jsonl"
        dst = args.data_dir / f"{d}_verified.jsonl"
        if not src.exists():
            logger.warning("verify: %s missing — run the sample step first", src)
            continue
        if (dst.exists() and not args.force
                and dst.stat().st_mtime >= src.stat().st_mtime):
            logger.info("verify: %s up to date — skipping (use --force)", dst)
            continue

        records = read_jsonl(src)

        def verify_one(r: dict) -> dict:
            response = get_response(r)
            reasoning, answer, valid = parse_reasoning_answer(response)
            r["reasoning"] = reasoning
            r["answer"] = answer
            r["_parse_valid"] = valid
            passed, reason = run_chain(r, d)
            r["verifier_passed"] = passed
            r["verifier_reason"] = reason
            return r

        # Code verification is subprocess-bound (one python process per
        # record, up to timeout seconds each) — parallelize with threads.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.verify_workers) as pool:
            out = list(pool.map(verify_one, records))

        reject_hist: Counter = Counter()
        for r in out:
            if not r["verifier_passed"]:
                reason = r["verifier_reason"]
                reject_hist[reason.split(":")[0] + ":" + reason.split(":")[1][:30]] += 1

        write_jsonl(dst, out)
        n_pass = sum(1 for r in out if r["verifier_passed"])
        logger.info(
            "verify[%s]: %d records  passed=%d  rejected=%d → %s",
            d, len(out), n_pass, len(out) - n_pass, dst,
        )
        if reject_hist:
            logger.info("verify[%s]: rejection buckets: %s", d, dict(reject_hist.most_common()))


# ---------------------------------------------------------------------------
# Step 4: judge (shell out to scripts/llm_judge_eval.py)
# ---------------------------------------------------------------------------

def step_judge(args) -> None:
    if not args.judge_api_base or not args.judge_model:
        logger.error(
            "judge: --judge-api-base and --judge-model are required for this step."
        )
        sys.exit(2)

    for d in args.domains:
        src = args.data_dir / f"{d}_verified.jsonl"
        if not src.exists():
            logger.warning("judge: %s missing — run the verify step first", src)
            continue

        judge_in = args.data_dir / f"{d}_judge_input.jsonl"
        judge_new = args.data_dir / f"{d}_judge_new.jsonl"
        judge_out = args.data_dir / f"{d}_judge_scores.jsonl"

        records = read_jsonl(src)

        # Incremental: keep cached scores (judge verdicts never change per
        # record), only send unjudged prompt_ids to the API.
        cached: dict[str, dict] = {}
        if not args.force:
            # judge_new may hold partial scores from a crashed previous run
            # (the judge script streams results as they complete) — recover
            # them before deciding what still needs judging.
            for path in (judge_out, judge_new):
                if path.exists():
                    for s in read_jsonl(path):
                        if isinstance(s.get("judge_answer_correctness"), (int, float)):
                            cached[s.get("prompt_id")] = s

        todo = [r for r in records if get_id(r) not in cached]
        if not todo:
            logger.info("judge[%s]: all %d records already scored — skipping (use --force)",
                        d, len(records))
            continue
        logger.info("judge[%s]: %d records total, %d cached, %d to judge",
                    d, len(records), len(cached), len(todo))

        payload = [
            {
                "prompt_id": get_id(r),
                "domain": d,
                "messages": [
                    {"role": "user", "content": get_prompt(r)},
                    {"role": "assistant", "content": get_response(r)},
                ],
            }
            for r in todo
        ]
        write_jsonl(judge_in, payload)

        cmd = [
            sys.executable, str(JUDGE_SCRIPT),
            str(judge_in), str(judge_new),
            "--model", args.judge_model,
            "--api-base", args.judge_api_base,
            "--api-key", args.judge_api_key,
            "--provider", args.judge_provider,
            "--concurrency", str(args.judge_concurrency),
            "--max-tokens", str(args.judge_max_tokens),
            "--think-open", THINK_OPEN,
            "--think-close", THINK_CLOSE,
        ]
        logger.info("judge[%s]: %s", d, " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Merge cached + fresh scores, in the order of the verified file.
        for s in read_jsonl(judge_new):
            if isinstance(s.get("judge_answer_correctness"), (int, float)):
                cached[s.get("prompt_id")] = s
        merged = [cached[get_id(r)] for r in records if get_id(r) in cached]
        write_jsonl(judge_out, merged)
        judge_new.unlink(missing_ok=True)  # merged — remove so stale partials can't linger
        logger.info("judge[%s]: merged %d scores → %s", d, len(merged), judge_out)


# ---------------------------------------------------------------------------
# Step 5: metrics
# ---------------------------------------------------------------------------

def _safe_div(a: int, b: int) -> float | None:
    return (a / b) if b else None


def step_metrics(args) -> None:
    summary: dict = {"good_threshold": args.good_threshold, "domains": {}}

    for d in args.domains:
        verified_path = args.data_dir / f"{d}_verified.jsonl"
        scores_path = args.data_dir / f"{d}_judge_scores.jsonl"
        if not verified_path.exists() or not scores_path.exists():
            logger.warning("metrics: missing inputs for %s — skipping", d)
            continue

        verified = {get_id(r): r for r in read_jsonl(verified_path)}
        scores = read_jsonl(scores_path)

        TP = FP = TN = FN = 0
        unlabeled = 0
        reason_positive: Counter = Counter()
        reason_all: Counter = Counter()
        fp_examples: list[dict] = []
        fn_examples: list[dict] = []

        for s in scores:
            pid = s.get("prompt_id")
            r = verified.get(pid)
            if r is None:
                continue

            reason_all[r.get("verifier_reason", "")] += 1
            corr = s.get("judge_answer_correctness")
            if not isinstance(corr, (int, float)):
                unlabeled += 1
                continue

            bad = corr < args.good_threshold
            positive = not r["verifier_passed"]  # verifier rejected it
            if positive:
                reason_positive[r.get("verifier_reason", "")] += 1

            if positive and bad:
                TP += 1
            elif positive and not bad:
                FP += 1
                if len(fp_examples) < args.n_examples:
                    fp_examples.append(_example(r, s, corr))
            elif (not positive) and bad:
                FN += 1
                if len(fn_examples) < args.n_examples:
                    fn_examples.append(_example(r, s, corr))
            else:
                TN += 1

        labeled = TP + FP + TN + FN
        report = {
            "domain": d,
            "n_scored": len(scores),
            "n_labeled": labeled,
            "n_unlabeled_by_judge": unlabeled,
            "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
            "verifier_rejected": TP + FP,
            "verifier_passed": TN + FN,
            "judge_bad": TP + FN,
            "judge_good": FP + TN,
            "precision": _safe_div(TP, TP + FP),
            "recall": _safe_div(TP, TP + FN),
            "specificity": _safe_div(TN, TN + FP),
            "accuracy": _safe_div(TP + TN, labeled),
            "f1": _f1(TP, FP, FN),
            "rejection_reasons": dict(reason_positive.most_common()),
            "all_reasons": dict(reason_all.most_common()),
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
        }
        out_path = args.reports_dir / f"{d}_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
        summary["domains"][d] = {k: report[k] for k in
                                 ("confusion", "precision", "recall",
                                  "specificity", "accuracy", "f1",
                                  "n_labeled", "n_unlabeled_by_judge")}
        _print_domain_report(report)

    summary_path = args.reports_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    logger.info("metrics: summary written → %s", summary_path)


def _example(r: dict, s: dict, corr) -> dict:
    return {
        "prompt_id": get_id(r),
        "verifier_reason": r.get("verifier_reason", ""),
        "judge_answer_correctness": corr,
        "judge_thinking_coherence": s.get("judge_thinking_coherence"),
        "judge_reasoning": (s.get("judge_reasoning") or "")[:300],
        "prompt": get_prompt(r)[:300],
        "answer": (r.get("answer") or "")[:300],
    }


def _f1(tp: int, fp: int, fn: int) -> float | None:
    p = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    if not p or not rec or (p + rec) == 0:
        return None
    return 2 * p * rec / (p + rec)


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


def _print_domain_report(rep: dict) -> None:
    c = rep["confusion"]
    print(f"\n{'='*60}")
    print(f"  {rep['domain'].upper()}  —  verifier vs LLM judge")
    print(f"{'='*60}")
    print(f"  scored={rep['n_scored']}  labeled={rep['n_labeled']}  "
          f"unlabeled(judge errors)={rep['n_unlabeled_by_judge']}")
    print(f"  verifier: rejected={rep['verifier_rejected']}  passed={rep['verifier_passed']}")
    print(f"  judge   : bad={rep['judge_bad']}  good={rep['judge_good']}")
    print(f"\n  Confusion matrix (positive = verifier REJECTS):")
    print(f"                       judge bad   judge good")
    print(f"    verifier reject   {c['TP']:>9}   {c['FP']:>10}   (TP / FP)")
    print(f"    verifier pass     {c['FN']:>9}   {c['TN']:>10}   (FN / TN)")
    print(f"\n  precision={_fmt(rep['precision'])}  recall={_fmt(rep['recall'])}  "
          f"specificity={_fmt(rep['specificity'])}  accuracy={_fmt(rep['accuracy'])}  "
          f"f1={_fmt(rep['f1'])}")
    if rep["rejection_reasons"]:
        print("\n  Rejection reasons:")
        for reason, n in rep["rejection_reasons"].items():
            print(f"    {reason:<40} {n}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STEPS = ["peek", "sample", "verify", "judge", "metrics"]
DEFAULT_STEPS = ["sample", "verify", "judge", "metrics"]


def load_config(path: Path) -> dict:
    """
    Load a YAML config and flatten it into a flat dict of argparse defaults.

    The `judge:` sub-block is flattened to judge_<key> keys.  Anything not
    recognised is ignored.  Missing file → empty dict (all CLI defaults apply).
    """
    if not path or not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    flat: dict = {}
    for k in ("hf_token", "dataset", "split", "domains", "n", "max_scan",
              "code_timeout", "good_threshold", "n_examples", "domain_offsets"):
        if k in raw and raw[k] not in (None, ""):
            flat[k] = raw[k]

    judge = raw.get("judge") or {}
    judge_map = {
        "provider": "judge_provider",
        "model": "judge_model",
        "api_base": "judge_api_base",
        "api_key": "judge_api_key",
        "concurrency": "judge_concurrency",
        "max_tokens": "judge_max_tokens",
    }
    for src, dst in judge_map.items():
        if src in judge and judge[src] not in (None, ""):
            flat[dst] = judge[src]

    return flat


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sanity-check Stage 6 math/code verifiers against an LLM judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="YAML config; its values become defaults (CLI overrides them)")
    p.add_argument("--hf-token", default="",
                   help="HuggingFace token (else config / `hf auth login` cache / HF_TOKEN env)")
    p.add_argument("--dataset", default="rahular/sft-pipeline-stage5")
    p.add_argument("--split", default="train")
    p.add_argument("--domains", nargs="+", default=["math", "code"])
    p.add_argument("--n", type=int, default=1000, help="Records to sample per domain")
    p.add_argument("--max-scan", type=int, default=2_000_000,
                   help="Give up filling buckets after scanning this many rows")
    p.add_argument("--domain-offsets", type=orjson.loads, default={},
                   help='JSON dict of per-domain stream skip offsets, e.g. {"code": 1399000} '
                        "(the dataset is domain-contiguous)")
    p.add_argument("--data-dir", type=Path, default=here / "data")
    p.add_argument("--reports-dir", type=Path, default=here / "reports")
    p.add_argument("--peek-rows", type=int, default=3)
    p.add_argument("--code-timeout", type=int, default=10)
    p.add_argument("--verify-workers", type=int, default=8,
                   help="Thread pool size for the verify step (code execution is subprocess-bound)")
    p.add_argument("--good-threshold", type=float, default=4.0,
                   help="judge answer_correctness >= this counts as 'good'")
    p.add_argument("--n-examples", type=int, default=15,
                   help="How many FP/FN examples to save per domain for inspection")
    # judge (reference labeler)
    p.add_argument("--judge-provider", default=os.environ.get("JUDGE_PROVIDER", "auto"),
                   choices=["auto", "openai", "openai-responses", "anthropic"])
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    p.add_argument("--judge-api-base", default=os.environ.get("JUDGE_API_BASE", ""))
    p.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", "none"))
    p.add_argument("--judge-concurrency", type=int, default=32)
    p.add_argument("--judge-max-tokens", type=int, default=8000,
                   help="Max tokens per judge call (keep high for reasoning models)")
    # control
    p.add_argument("--steps", default=",".join(DEFAULT_STEPS),
                   help=f"Comma-separated subset of {ALL_STEPS}")
    p.add_argument("--force", action="store_true", help="Recompute steps even if outputs exist")

    # Load config first so its values become defaults; explicit CLI flags still win.
    cfg_path = p.parse_known_args()[0].config
    cfg_defaults = load_config(cfg_path)
    if cfg_defaults:
        logger.info("loaded config defaults from %s: %s",
                    cfg_path, sorted(cfg_defaults))
        p.set_defaults(**cfg_defaults)

    args = p.parse_args()

    # Propagate HF token to the environment for `datasets` streaming.
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        p.error(f"unknown step(s): {unknown}; valid: {ALL_STEPS}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "peek": step_peek,
        "sample": step_sample,
        "verify": step_verify,
        "judge": step_judge,
        "metrics": step_metrics,
    }
    for s in steps:
        logger.info("=== step: %s ===", s)
        dispatch[s](args)


if __name__ == "__main__":
    main()
