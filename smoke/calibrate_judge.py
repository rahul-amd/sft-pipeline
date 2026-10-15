"""Compare the pre-grounding judge with the current judge on labeled cases.

Run on a reachable OpenAI-compatible endpoint. Thresholds are selected only
on the calibration split, then evaluated on untouched held-out groups.
This script proposes thresholds; it never changes pipeline configuration.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import httpx
from json_repair import repair_json
from tenacity import retry, stop_after_attempt, wait_fixed

from sft_pipeline.filters.llm_judge import (
    JUDGE_PROMPT_VERSION, _JUDGE_SYSTEM, _call_judge, judge_rejection_reason,
)


def confusion(results, threshold, variant):
    counts = {"true_accepts": 0, "true_rejects": 0, "false_accepts": 0, "false_rejects": 0}
    for row in results:
        accepted = (row["scores"]["overall"] >= threshold if variant == "baseline"
                    else judge_rejection_reason(row["scores"], threshold) is None)
        expected = row["expected_accept"]
        key = ("true_accepts" if expected else "false_accepts") if accepted else (
            "false_rejects" if expected else "true_rejects")
        counts[key] += 1
    positive = counts["true_accepts"] + counts["false_rejects"]
    negative = counts["true_rejects"] + counts["false_accepts"]
    counts["accept_recall"] = counts["true_accepts"] / positive if positive else None
    counts["reject_recall"] = counts["true_rejects"] / negative if negative else None
    counts["balanced_accuracy"] = ((counts["accept_recall"] + counts["reject_recall"]) / 2
                                    if positive and negative else None)
    return counts


def main(args):
    source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:sft_pipeline/filters/llm_judge.py"], text=True)
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(source).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ("_JUDGE_SYSTEM", "_JUDGE_USER")}
    cases_path = Path(args.cases)
    with cases_path.open() as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert all(type(c["expected_accept"]) is bool for c in cases)
    assert all(c["split"] in ("calibration", "holdout") for c in cases)
    assert all({c["expected_accept"] for c in cases if c["split"] == split} == {True, False}
               for split in ("calibration", "holdout")), "Both splits need positive and negative labels"
    groups = [{c["group"] for c in cases if c["split"] == split} for split in ("calibration", "holdout")]
    assert all(groups) and not groups[0] & groups[1], "Calibration and holdout groups overlap"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_commit = subprocess.check_output(["git", "rev-parse", args.baseline_ref], text=True).strip()
    manifest = {"model": args.model, "api_base": args.api_base, "max_tokens": args.max_tokens,
                "baseline_ref": baseline_commit, "current_prompt_version": JUDGE_PROMPT_VERSION,
                "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                "current_prompt_sha256": hashlib.sha256(_JUDGE_SYSTEM.encode()).hexdigest()}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Calibration inputs changed; use a fresh output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    (output / "report.json").unlink(missing_ok=True)
    (output / "prompts.json").write_text(json.dumps({"baseline": constants, "current": _JUDGE_SYSTEM}, indent=2)+"\n")
    (output / "cases.used.jsonl").write_bytes(cases_path.read_bytes())
    result_path = output / "scores.jsonl"
    existing = [json.loads(line) for line in result_path.open()] if result_path.exists() else []
    done = {(r["variant"], r["case_id"]) for r in existing}
    expected_keys = {(variant, case["case_id"]) for case in cases for variant in ("baseline", "current")}
    assert len(done) == len(existing) and done <= expected_keys, "Invalid saved score rows"
    key = os.environ[args.api_key_env]

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
    def baseline(case):
        response = httpx.post(args.api_base.rstrip("/")+"/chat/completions", timeout=180,
            headers={"Authorization": f"Bearer {key}"}, json={"model": args.model,
                "temperature": 0, "max_tokens": args.max_tokens,
                "messages": [{"role": "system", "content": constants["_JUDGE_SYSTEM"]},
                    {"role": "user", "content": constants["_JUDGE_USER"].format(
                        prompt=case["prompt"][:500], reasoning=case["reasoning"][:1500], answer=case["answer"][:500])}]})
        response.raise_for_status()
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "stop", "Truncated baseline verdict"
        content = re.sub(r"<think>.*?</think>", "", choice["message"]["content"], flags=re.DOTALL).strip()
        value = json.loads(repair_json(content))
        assert type(value.get("overall")) in (int, float) and 1 <= value["overall"] <= 10
        return value

    def evaluate(variant, case):
        start = time.monotonic()
        scores = baseline(case) if variant == "baseline" else _call_judge(
            prompt=case["prompt"], reasoning=case["reasoning"], answer=case["answer"],
            endpoint=args.api_base, model=args.model, api_key=key, max_tokens=args.max_tokens,
            request_timeout=180, source_context=case.get("source_context"),
            reference_answer=case.get("reference_answer"))
        return {"variant": variant, "case_id": case["case_id"], "group": case["group"],
                "split": case["split"], "expected_accept": case["expected_accept"],
                "scores": scores, "elapsed_seconds": time.monotonic()-start}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, result_path.open("a") as stream:
        tasks = [pool.submit(evaluate, variant, case) for case in cases for variant in ("baseline", "current")
                 if (variant, case["case_id"]) not in done]
        for task in as_completed(tasks):
            result = task.result()
            stream.write(json.dumps(result)+"\n")
            stream.flush()
            existing.append(result)
            print(f"Completed {len(existing)}/{len(cases)*2}", flush=True)
    assert len(existing) == len(cases)*2
    report = {"status": "completed", "manifest": manifest,
              "limitations": "Assistant-curated paired regressions, not human gold labels or a representative production-quality estimate. No config is automatically changed.",
              "variants": {}}
    for variant in ("baseline", "current"):
        train = [r for r in existing if r["variant"] == variant and r["split"] == "calibration"]
        test = [r for r in existing if r["variant"] == variant and r["split"] == "holdout"]
        sweep = {threshold: confusion(train, threshold, variant) for threshold in range(1, 11)}
        best = max(sweep, key=lambda t: (sweep[t]["balanced_accuracy"], -sweep[t]["false_accepts"], -abs(t-6)))
        report["variants"][variant] = {"calibration_sweep": sweep, "suggested_threshold": best,
            "calibration_at_suggested": sweep[best], "holdout_at_suggested": confusion(test, best, variant),
            "calibration_at_default_6": sweep[6], "holdout_at_default_6": confusion(test, 6, variant)}
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="SFT_API_KEY")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", default="smoke/fixtures/judge-calibration-v2.jsonl")
    parser.add_argument("--baseline-ref", default="854656d")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    main(parser.parse_args())
