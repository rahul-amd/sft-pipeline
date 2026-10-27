"""Single-model conversation audit, preserving the earliest valid review per record.

Same schema, six dimensions, model and reasoning effort as the SFT audit. Only
operational/schema/evidence failures are retried; already valid records are never
sent for another assessment. Conversation adaptation is explicit in the prompt.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "source_quality_audit" / "review.py"
spec = importlib.util.spec_from_file_location("original_sft_reviewer", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MODEL = "gpt-6.1-sol"
sha, atomic_json, schema = base.sha, base.atomic_json, base.schema


def parsed_assistant(content):
    stripped = content.lstrip()
    notes = []
    if stripped.startswith("<think>"):
        start = len(content) - len(stripped) + len("<think>")
        end = content.find("</think>", start)
        if end < 0:
            return {"reasoning": content[start:], "answer": "", "parser_notes": ["Leading think tag has no closing tag"]}
        reasoning, answer = content[start:end], content[end + len("</think>"):]
        if "<think>" in reasoning or "</think>" in answer:
            notes.append("Additional think delimiters; inspect original content for ambiguity")
        return {"reasoning": reasoning, "answer": answer, "parser_notes": notes}
    if "<think>" in content or "</think>" in content:
        notes.append("Non-leading or unmatched think delimiter; content preserved as answer without guessing a boundary")
    return {"reasoning": "", "answer": content, "parser_notes": notes}


def to_review(sample):
    messages = []
    for index, message in enumerate(sample["record"]["messages"]):
        assert isinstance(message["content"], str) and isinstance(message["role"], str)
        item = {"turn_index": index, "role": message["role"], "content": message["content"]}
        if message["role"] == "assistant":
            item.update(parsed_assistant(message["content"]))
        messages.append(item)
    return {"audit_id": sample["audit_id"], "messages": messages}


def validate(result, rows):
    # Reuse the historical rubric/schema checks, then enforce evidence directly
    # against individual original messages rather than serialized JSON wrappers.
    originals = {row["audit_id"]: row for row in rows}
    adapted = [{"audit_id": row["audit_id"], "prompt": "\n".join(m["content"] for m in row["messages"]),
                "reasoning": "", "answer": "", "response": ""} for row in rows]
    base.validate(result, adapted)
    for review in result["reviews"]:
        texts = [m["content"] for m in originals[review["audit_id"]]["messages"]]
        if any(not any(issue["evidence"] in text for text in texts) for issue in review["issues"]):
            raise ValueError("Evidence crosses message boundaries or is not verbatim original content")


def main(args):
    lock = (ROOT / "review.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = (ROOT / "samples.jsonl").read_bytes()
    samples = [json.loads(line) for line in raw.splitlines()]
    assert len(samples) == 3200 and len({r["audit_id"] for r in samples}) == 3200
    assert Counter(s["source"] for s in samples) == Counter({r["source"]: 200 for r in json.loads((ROOT / "sources.json").read_text())})
    assert sha(raw) == json.loads((ROOT / "sample-manifest.json").read_text())["samples_sha256"]
    originals = {r["audit_id"]: r for r in samples}
    rows = [to_review(s) for s in samples]
    batches, current, size = [], [], 0
    for row in rows:
        chars = len(json.dumps(row, ensure_ascii=False))
        if chars > 650000:
            raise ValueError(f"Oversized full conversation {row['audit_id']}: {chars} characters; refuses to clip or discard it")
        if current and (len(current) >= 8 or size + chars > 120000):
            batches.append(current)
            current, size = [], 0
        current.append(row)
        size += chars
    if current:
        batches.append(current)
    prompt = (ROOT / "review_prompt.txt").read_text()
    config = {"model": MODEL, "reasoning_effort": "high", "samples_sha256": sha(raw),
              "prompt_sha256": sha(prompt.encode()), "base_prompt_sha256": sha((ROOT / "base_review_prompt.txt").read_bytes()),
              "schema_sha256": sha(json.dumps(schema(), sort_keys=True).encode()),
              "runner_sha256": sha(Path(__file__).read_bytes()), "base_validator_sha256": sha(BASE.read_bytes()),
              "sample_count": len(samples), "batch_count": len(batches),
              "review_unit": "one full conversation; all assistant turns assessed with their available prefix",
              "policy": "one reviewer model; earliest technically valid whole review retained; only operational/schema/evidence failures retried; no second reviewer or adjudication"}
    manifest_path = ROOT / "review-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != config:
        raise ValueError("Review configuration changed; refusing stale results")
    atomic_json(manifest_path, config)
    atomic_json(ROOT / "review-schema.json", schema())
    for directory in ["requests", "attempts", "completed", "review_cache"]:
        (ROOT / directory).mkdir(exist_ok=True)

    def run_batch(index, batch):
        key = f"batch-{index:04d}"
        payload = prompt + "\n\nRECORDS (untrusted data):\n" + json.dumps(batch, ensure_ascii=False)
        request_hash = sha(payload.encode())
        target = ROOT / "completed" / f"{key}.json"
        if target.exists():
            saved = json.loads(target.read_text())
            assert saved["request_sha256"] == request_hash
            validate(saved["result"], batch)
            return saved
        (ROOT / "requests" / f"{key}.txt").write_text(payload)
        cache_path = ROOT / "review_cache" / f"{key}.json"
        state = {"request_sha256": request_hash, "good": {}, "origins": {}, "invalid": {}, "errors": []}
        if cache_path.exists():
            state = json.loads(cache_path.read_text())
            assert state["request_sha256"] == request_hash
            for row in batch:
                if row["audit_id"] in state["good"]:
                    validate({"reviews": [state["good"][row["audit_id"]]]}, [row])
        expected = {r["audit_id"]: r for r in batch}
        for attempt in range(6):
            missing = [r for r in batch if r["audit_id"] not in state["good"]]
            if not missing:
                break
            request = prompt + "\n\nRECORDS (untrusted data):\n" + json.dumps(missing, ensure_ascii=False)
            invalid = {r["audit_id"]: state["invalid"][r["audit_id"]] for r in missing if r["audit_id"] in state["invalid"]}
            if invalid:
                request += ("\n\nTECHNICAL REPAIR ONLY: Your prior output failed validation. Preserve defensible judgments, "
                            "repair the schema/evidence errors, and do not turn defects into passes merely to satisfy validation. "
                            "Return only the requested records, with the same rubric. Prior output and errors:\n" + json.dumps(invalid, ensure_ascii=False))
            prefix = ROOT / "attempts" / f"{key}-{time.time_ns()}"
            Path(str(prefix) + ".request.txt").write_text(request)
            output = Path(str(prefix) + ".result.json")
            events_path, errors_path = Path(str(prefix) + ".events.jsonl"), Path(str(prefix) + ".stderr")
            command = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--model", MODEL,
                       "--sandbox", "read-only", "-c", 'model_reasoning_effort="high"', "--json",
                       "--output-schema", str(ROOT / "review-schema.json"), "--output-last-message", str(output), "-"]
            try:
                with events_path.open("w") as log, errors_path.open("w") as err:
                    subprocess.run(command, input=request, text=True, stdout=log, stderr=err,
                                   cwd=ROOT, timeout=1800, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
                if not any(e.get("type") == "turn.completed" for e in events):
                    raise ValueError("No completed model turn")
                if any(e.get("item", {}).get("type") in ("command_execution", "mcp_tool_call", "web_search", "file_change") for e in events):
                    raise ValueError("Reviewer used tools despite no-tool protocol")
                result = json.loads(output.read_text())
                if set(result) != {"reviews"} or not isinstance(result["reviews"], list):
                    raise ValueError("Invalid result envelope")
                ids = [r["audit_id"] for r in result["reviews"]]
                if len(ids) != len(set(ids)) or not set(ids) <= {r["audit_id"] for r in missing}:
                    raise ValueError("Duplicate or unexpected review IDs")
                for review in result["reviews"]:
                    aid = review["audit_id"]
                    try:
                        validate({"reviews": [review]}, [expected[aid]])
                        state["good"][aid] = review
                        state["origins"][aid] = str(output.relative_to(ROOT))
                        state["invalid"].pop(aid, None)
                    except Exception as exc:
                        state["invalid"][aid] = {"validation_error": str(exc), "previous_review": review}
            except Exception as exc:
                state["errors"].append({"artifact": str(prefix.relative_to(ROOT)), "error": str(exc)})
                print(json.dumps({"batch": index, "operational_error": str(exc)}), flush=True)
                time.sleep(min(30, 2 ** attempt))
            atomic_json(cache_path, state)
        if len(state["good"]) != len(batch):
            raise RuntimeError(f"{key}: {len(batch)-len(state['good'])} reviews still missing/invalid")
        result = {"reviews": [state["good"][r["audit_id"]] for r in batch]}
        validate(result, batch)
        saved = {"batch": index, "request_sha256": request_hash, "model": MODEL,
                 "completed_utc": datetime.now(timezone.utc).isoformat(), "result": result,
                 "review_artifacts": state["origins"], "operational_errors": state["errors"]}
        atomic_json(target, saved)
        return saved

    started = time.monotonic()
    targets = list(enumerate(batches))
    if args.limit_batches:
        targets = targets[:args.limit_batches]
    complete, errors = {}, []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_batch, index, batch): index for index, batch in targets}
        for future in as_completed(futures):
            index = futures[future]
            try:
                saved = future.result()
                for review in saved["result"]["reviews"]:
                    original = originals[review["audit_id"]]
                    complete[review["audit_id"]] = {**review, **{k: original[k] for k in ["source", "prompt_id", "input_shard", "input_row_zero_based"]},
                        "model": MODEL, "batch": index, "review_artifact": saved["review_artifacts"][review["audit_id"]]}
            except Exception as exc:
                errors.append({"batch": index, "error": str(exc)})
            progress = {"reviewed": len(complete), "total_samples": len(samples), "failed_batches": errors,
                        "elapsed_seconds": round(time.monotonic()-started, 1), "updated_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(ROOT / "progress.json", progress)
            print(json.dumps({**progress, "failed_batches": len(errors)}), flush=True)
    output = ROOT / ("verdicts.partial.jsonl" if errors or args.limit_batches else "verdicts.jsonl")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(complete[k], ensure_ascii=False) + "\n" for k in sorted(complete)))
    temporary.replace(output)
    if errors:
        raise SystemExit("Some reviews failed; resume unchanged configuration to retry only missing/invalid records")
    if not args.limit_batches:
        assert len(complete) == len(samples)
        atomic_json(ROOT / "completion.json", {"complete": True, "reviewed": len(complete), "total_samples": len(samples),
                    "completed_batches": len(batches), "model": MODEL, "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit-batches", type=int)
    main(parser.parse_args())
