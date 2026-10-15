"""Review every inferred example in a small run using a separate judge model.

Model agreement is diagnostic evidence, not an estimate of true correctness.
The reviewer never receives the pipeline's scores or acceptance decision.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from sft_pipeline.config import ReasoningDelimiters
from sft_pipeline.filters.llm_judge import _call_judge, judge_rejection_reason
from sft_pipeline.inference.output_parser import parse_output
from sft_pipeline.storage import iter_jsonl_dir


def main(args):
    root = Path(args.run)
    records = list(iter_jsonl_dir(root / "stage5"))
    if not records or len(records) > 250:
        raise ValueError("This whole-run review expects 1–250 inferred records")
    decisions = {r["prompt_id"]: r for r in iter_jsonl_dir(root / "stage6/_decisions")}
    if set(decisions) != {r["prompt_id"] for r in records}:
        raise ValueError("Incomplete pipeline decision audit")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "api_base": args.api_base, "review_format_version": 2,
                "input_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Review inputs changed; choose a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    (output / "report.json").unlink(missing_ok=True)
    path = output / "verdicts.jsonl"
    results = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {r["prompt_id"] for r in results}

    def review(row):
        parsed = parse_output(row.get("response", ""), ReasoningDelimiters())
        # Explicit empty fields matter: a token-limited thinking response may
        # have no final answer. Never substitute its reasoning as the answer.
        reasoning = row["reasoning"] if "reasoning" in row else parsed.reasoning
        answer = row["answer"] if "answer" in row else parsed.answer
        scores = _call_judge(row["prompt"], reasoning, answer, args.api_base, args.model,
            api_key=os.environ["SFT_API_KEY"], max_tokens=1536, request_timeout=180,
            source_context=row.get("source_context"), reference_answer=row.get("reference_answer"))
        return {"prompt_id": row["prompt_id"], "prompt": row["prompt"], "reasoning": reasoning,
                "answer": answer, "source_context": row.get("source_context"),
                "reference_answer": row.get("reference_answer"), "scores": scores,
                "reviewer_accept": judge_rejection_reason(scores, 6) is None,
                "pipeline": decisions[row["prompt_id"]]}

    with ThreadPoolExecutor(max_workers=4) as pool, path.open("a") as stream:
        tasks = [pool.submit(review, row) for row in records if row["prompt_id"] not in done]
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            stream.write(json.dumps(result)+"\n")
            stream.flush()
    if len(results) != len(records):
        raise ValueError("Duplicate or missing review results")
    counts = Counter((r["pipeline"]["passed"], r["reviewer_accept"]) for r in results)
    report = {"status": "completed", "examples": len(results), "manifest": manifest,
              "both_accept": counts[True, True], "both_reject": counts[False, False],
              "pipeline_accept_reviewer_reject": counts[True, False],
              "pipeline_reject_reviewer_accept": counts[False, True],
              "limitations": "Separate model, shared Qwen family and rubric; model agreement is not correctness ground truth. Structural rejections can disagree with an otherwise correct answer."}
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    main(parser.parse_args())
