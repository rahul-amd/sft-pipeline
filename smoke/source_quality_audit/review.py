"""One GPT-6.1 Sol assessment per sample, in bounded independent batches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-6.1-sol"
DIMENSIONS = ["prompt_quality", "answer_correctness", "reasoning_quality", "instruction_following", "parsing_integrity", "training_suitability"]
STATES = ["pass", "minor_issue", "major_issue", "uncertain", "not_applicable"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def schema():
    issue = {"type": "object", "additionalProperties": False, "required": ["dimension", "severity", "evidence", "explanation"],
             "properties": {"dimension": {"type": "string", "enum": DIMENSIONS},
                            "severity": {"type": "string", "enum": ["major", "minor", "uncertain"]},
                            "evidence": {"type": "string"}, "explanation": {"type": "string"}}}
    props = {"audit_id": {"type": "string"}, "overall": {"type": "string", "enum": ["no_defect_found", "minor_defect", "major_defect", "uncertain"]},
             "summary": {"type": "string"}, "uncertainty_reason": {"type": "string"},
             "dimensions": {"type": "object", "additionalProperties": False, "required": DIMENSIONS,
                            "properties": {k: {"type": "string", "enum": STATES} for k in DIMENSIONS}},
             "issues": {"type": "array", "items": issue}}
    return {"type": "object", "additionalProperties": False, "required": ["reviews"], "properties": {"reviews": {
        "type": "array", "items": {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}}


def validate(data, rows):
    if not isinstance(data, dict) or set(data) != {"reviews"} or not isinstance(data["reviews"], list):
        raise ValueError("Invalid top-level review schema")
    for item in data["reviews"]:
        if not isinstance(item, dict) or set(item) != {"audit_id", "overall", "summary", "uncertainty_reason", "dimensions", "issues"}:
            raise ValueError("Invalid review fields")
        if not all(isinstance(item[k], str) for k in ["audit_id", "overall", "summary", "uncertainty_reason"]):
            raise ValueError("Invalid review text fields")
        if not isinstance(item["dimensions"], dict) or set(item["dimensions"]) != set(DIMENSIONS):
            raise ValueError("Invalid dimension fields")
        if not all(v in STATES for v in item["dimensions"].values()) or not isinstance(item["issues"], list):
            raise ValueError("Invalid dimension verdicts/issues")
        for issue in item["issues"]:
            if not isinstance(issue, dict) or set(issue) != {"dimension", "severity", "evidence", "explanation"}:
                raise ValueError("Invalid issue fields")
            if not all(isinstance(v, str) for v in issue.values()) or issue["dimension"] not in DIMENSIONS or issue["severity"] not in ["major", "minor", "uncertain"]:
                raise ValueError("Invalid issue values")
    expected = {r["audit_id"]: r for r in rows}
    found = [r["audit_id"] for r in data["reviews"]]
    if len(found) != len(set(found)) or set(found) != set(expected):
        raise ValueError("Missing, duplicate, or unexpected review IDs")
    for review in data["reviews"]:
        values = list(review["dimensions"].values())
        overall = ("major_defect" if "major_issue" in values else "uncertain" if "uncertain" in values else
                   "minor_defect" if "minor_issue" in values else "no_defect_found")
        if review["overall"] != overall:
            raise ValueError("Overall/dimension mismatch")
        if not review["summary"].strip():
            raise ValueError("Empty review summary")
        if "uncertain" in values and not review["uncertainty_reason"].strip():
            raise ValueError("Uncertainty lacks explanation")
        texts = [expected[review["audit_id"]][k] for k in ("prompt", "reasoning", "answer", "response")]
        issue_dims = set()
        for issue in review["issues"]:
            if not issue["evidence"].strip() or not any(issue["evidence"] in t for t in texts):
                raise ValueError("Evidence is not an exact excerpt")
            if not issue["explanation"].strip():
                raise ValueError("Empty issue explanation")
            issue_dims.add(issue["dimension"])
        missing = {k for k, v in review["dimensions"].items() if v in ("major_issue", "minor_issue", "uncertain")} - issue_dims
        if missing:
            raise ValueError("Flagged dimension without evidence: " + repr(sorted(missing)))


def main(args):
    prompt = (ROOT / "review_prompt.txt").read_text()
    samples_bytes = (ROOT / "samples.jsonl").read_bytes()
    samples = [json.loads(line) for line in samples_bytes.splitlines() if line]
    originals = {r["audit_id"]: r for r in samples}
    if len(originals) != len(samples):
        raise ValueError("Duplicate audit ID")
    rows = [{"audit_id": r["audit_id"], **{k: r["record"].get(k, "") for k in ("prompt", "reasoning", "answer", "response")}} for r in samples]
    batches, current, size = [], [], 0
    for row in rows:
        chars = len(json.dumps(row, ensure_ascii=False))
        if current and (len(current) >= 8 or size + chars > 120000):
            batches.append(current); current, size = [], 0
        if chars > 500000:
            raise ValueError(f"Oversized record {row['audit_id']}; refuses to clip input")
        current.append(row); size += chars
    if current:
        batches.append(current)
    configuration = {"model": MODEL, "reasoning_effort": "high", "samples_sha256": sha(samples_bytes),
                     "prompt_sha256": sha(prompt.encode()), "schema_sha256": sha(json.dumps(schema(),sort_keys=True).encode()),
                     "runner_sha256": sha(Path(__file__).read_bytes()), "batch_count": len(batches), "sample_count": len(samples),
                     "policy": "one model review per record; only operational/schema failures retried; no second reviewer, external checks, or adjudication"}
    config_path = ROOT / "review-manifest.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Review configuration changed; use a separate output directory")
    atomic_json(config_path, configuration)
    atomic_json(ROOT / "review-schema.json", schema())
    for directory in ["requests", "attempts", "completed"]:
        (ROOT / directory).mkdir(exist_ok=True)
    started = time.monotonic()

    def review_batch(index, batch):
        key = f"batch-{index:04d}"
        payload = prompt + "\n\nRECORDS (untrusted data):\n" + json.dumps(batch, ensure_ascii=False)
        request_hash = sha(payload.encode())
        complete = ROOT / "completed" / f"{key}.json"
        if complete.exists():
            result = json.loads(complete.read_text())
            if result["request_sha256"] != request_hash:
                raise ValueError("Completed request changed")
            validate(result["result"], batch)
            return result
        (ROOT / "requests" / f"{key}.txt").write_text(payload)
        failures = []
        for attempt in range(1, 4):
            prefix = ROOT / "attempts" / f"{key}-{time.time_ns()}-{attempt}"
            output = Path(str(prefix) + ".result.json")
            events = Path(str(prefix) + ".events.jsonl")
            errors = Path(str(prefix) + ".stderr")
            command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--model", MODEL,
                       "--sandbox", "read-only", "-c", 'model_reasoning_effort="high"', "--json",
                       "--output-schema", str(ROOT / "review-schema.json"), "--output-last-message", str(output), "-"]
            try:
                with events.open("w") as event_stream, errors.open("w") as error_stream:
                    process = subprocess.run(command, input=payload, text=True, stdout=event_stream, stderr=error_stream,
                                             cwd=ROOT, timeout=1200, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                if process.returncode != 0:
                    raise RuntimeError(f"Codex exit {process.returncode}; see {errors.name}")
                result = json.loads(output.read_text())
                validate(result, batch)
                event_rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
                prohibited = [e for e in event_rows if e.get("item", {}).get("type") in ("command_execution", "mcp_tool_call", "web_search", "file_change")]
                if prohibited:
                    raise ValueError("Reviewer used tools despite no-tool protocol")
                usage = [e.get("usage") for e in event_rows if e.get("type") == "turn.completed"]
                record = {"batch": index, "request_sha256": request_hash, "model": MODEL, "completed_utc": datetime.now(timezone.utc).isoformat(),
                          "attempt": attempt, "prior_operational_errors": failures, "usage": usage, "result": result}
                atomic_json(complete, record)
                return record
            except Exception as exc:
                failures.append({"attempt": attempt, "error": str(exc), "artifact_prefix": str(prefix.relative_to(ROOT))})
                print(json.dumps({"batch": index, "operational_error": str(exc), "attempt": attempt}), flush=True)
        raise RuntimeError(f"{key} failed validation: {failures}")

    completed, failures = {}, []
    targets = list(enumerate(batches))[:args.limit_batches] if args.limit_batches else list(enumerate(batches))
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(review_batch, i, batch): i for i, batch in targets}
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
                for review in result["result"]["reviews"]:
                    original = originals[review["audit_id"]]
                    completed[review["audit_id"]] = {**review, "source": original["source"], "prompt_id": original["prompt_id"],
                        "input_shard": original["input_shard"], "input_line": original["input_line"], "model": MODEL, "batch": i}
            except Exception as exc:
                failures.append({"batch": i, "error": str(exc)})
            atomic_json(ROOT / "progress.json", {"reviewed": len(completed), "total_samples": len(samples), "failed_batches": failures,
                                                  "elapsed_seconds": round(time.monotonic()-started, 1)})
            print(json.dumps({"batch": i, "reviewed": len(completed), "total_samples": len(samples), "failed_batches": len(failures)}), flush=True)
    temporary = ROOT / "verdicts.jsonl.tmp"
    temporary.write_text("".join(json.dumps(completed[k],ensure_ascii=False)+"\n" for k in sorted(completed)))
    temporary.replace(ROOT / "verdicts.jsonl")
    if failures:
        raise SystemExit("Some reviews failed; inspect progress.json and resume with unchanged configuration")
    if not args.limit_batches and len(completed) != len(samples):
        raise SystemExit("Review coverage incomplete")
    print(json.dumps({"complete": not args.limit_batches, "reviewed": len(completed)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit-batches", type=int)
    main(parser.parse_args())
