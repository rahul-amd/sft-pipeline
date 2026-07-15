#!/usr/bin/env python3
"""
Build the baseline-vs-current comparison dataset for the Gradio viz.

Re-scores every sampled record with the frozen baseline chain
(baseline_verifiers.py), joins the current verdicts (data/{domain}_verified.jsonl,
produced by sanity_check.py with the live filters) and the LLM-judge labels,
and writes data/comparison.jsonl.

Usage:
    python sanity-checks/prep_comparison.py [--workers 8] [--code-timeout 10]
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson

import baseline_verifiers as bl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("prep_comparison")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domains", nargs="+", default=["math", "code"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--code-timeout", type=int, default=10)
    p.add_argument("--out", type=Path, default=DATA / "comparison.jsonl")
    args = p.parse_args()

    rows: list[dict] = []
    for d in args.domains:
        raw = read_jsonl(DATA / f"{d}.jsonl")
        current = {r["prompt_id"]: r for r in read_jsonl(DATA / f"{d}_verified.jsonl")}
        scores = {s["prompt_id"]: s for s in read_jsonl(DATA / f"{d}_judge_scores.jsonl")}
        logger.info("[%s] %d raw, %d current verdicts, %d judge scores",
                    d, len(raw), len(current), len(scores))

        def score_one(r: dict, d=d) -> dict:
            pid = r["prompt_id"]
            b_passed, b_reason = bl.run_baseline_chain(r, d, code_timeout=args.code_timeout)
            cur = current.get(pid, {})
            s = scores.get(pid, {})
            corr = s.get("judge_answer_correctness")
            return {
                "prompt_id": pid,
                "domain": d,
                "prompt": (r.get("prompt") or "")[:800],
                "response": (r.get("response") or "")[:6000],
                "source": r.get("source", ""),
                "difficulty": r.get("difficulty", ""),
                "baseline_passed": b_passed,
                "baseline_reason": b_reason,
                "current_passed": bool(cur.get("verifier_passed", True)),
                "current_reason": cur.get("verifier_reason", ""),
                "judge_correctness": corr if isinstance(corr, (int, float)) else None,
                "judge_reasoning": (s.get("judge_reasoning") or "")[:600],
            }

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            out = list(pool.map(score_one, raw))
        rows.extend(out)
        n_rej = sum(1 for r in out if not r["baseline_passed"])
        logger.info("[%s] baseline rejected %d / %d", d, n_rej, len(out))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r) + b"\n")
    logger.info("wrote %d rows → %s", len(rows), args.out)


if __name__ == "__main__":
    main()
