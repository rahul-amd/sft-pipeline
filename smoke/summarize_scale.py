"""Collect medium-run evidence after export, resume, recovery and audit finish."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    paths = sorted(path.glob("part-*.jsonl")) if path.is_dir() else [path]
    for file in paths:
        with file.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def main(run_id):
    run = Path("smoke/runs") / run_id
    evidence = Path("smoke/runs/freeze_validation")
    smoke = read_json(run / "smoke_report.json")
    assert smoke["status"] == "passed" and smoke["resume_unchanged"]
    assert smoke["counts"]["stage1_collect"] >= 9000
    assert smoke["counts"]["stage2_generate"] >= 900
    assert 3000 <= smoke["counts"]["stage5_inference"] <= 5000
    final = list(rows(run / "final"))
    inferred = list(rows(run / "stage5"))
    assert len(inferred) == smoke["counts"]["stage4_sample"]
    assert len({r["prompt_id"] for r in inferred}) == len(inferred)
    final_ids = {r["id"] for r in final}
    truncated_ids = {r["prompt_id"] for r in inferred if r.get("finish_reason") == "length"}
    assert not (final_ids & truncated_ids), "Truncated completions reached export"
    recovered = read_json(evidence / "recovery-stages34.json")
    recovered["stage5_inference"] = read_json(evidence / "recovery-stage5.json")
    assert all(r["status"] == "passed" for r in recovered.values())
    audit = read_json(evidence / f"audit-{run_id}" / "summary.json")
    assert audit["status"] == "completed"
    suites = ET.parse(evidence / "full-tests-final-v2.xml").getroot().findall("testsuite")
    tests = {k: sum(int(s.attrib[k]) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    assert tests["tests"] >= 259 and not any(tests[k] for k in ("failures", "errors", "skipped"))
    stage_times = {}
    stage = None
    for line in (run / "pipeline.log").read_text().splitlines():
        header = re.fullmatch(r"─+ (\w+) ─+", line)
        if header:
            stage = header[1]
        stamp = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[", line)
        if stage and stamp:
            timestamp = datetime.fromisoformat(stamp[1])
            stage_times.setdefault(stage, [timestamp, timestamp])[1] = timestamp
    timings = {name: {"first_log_utc": a.isoformat(), "last_log_utc": b.isoformat(),
                      "log_span_seconds": (b-a).total_seconds()} for name, (a, b) in stage_times.items()}
    container_actions = {}
    filter_start, filter_end = [t.replace(tzinfo=timezone.utc).timestamp()
                                for t in stage_times["stage6_filter"]]
    for line in (evidence / "docker-events.log").read_text().splitlines():
        fields = line.split()
        if (len(fields) == 3 and fields[2].startswith("sft-sandbox-")
                and filter_start <= int(fields[0]) < filter_end + 1):
            container_actions.setdefault(fields[1], set()).add(fields[2])
    assert container_actions.get("create"), "No live Docker sandbox creation observed"
    assert container_actions["create"] == container_actions.get("destroy", set()), "Sandbox cleanup incomplete"
    metrics = list(rows(run / "resources.jsonl"))
    peaks = {}
    rates = []
    previous = None
    for sample in metrics:
        for card, data in sample.get("gpu", {}).items():
            peaks[card] = max(peaks.get(card, 0), int(data["VRAM Total Used Memory (B)"]))
        lines = sample.get("teacher", [])
        if not isinstance(lines, list):
            continue
        token_line = next((line for line in lines if line.startswith("vllm:generation_tokens_total")), None)
        active_line = next((line for line in lines if line.startswith("vllm:num_requests_running")), None)
        if token_line and active_line:
            current = (sample["elapsed_seconds"], float(token_line.split()[-1]), float(active_line.split()[-1]))
            if previous and current[2] >= 32 and previous[2] >= 32 and current[0] > previous[0]:
                rates.append((current[1]-previous[1])/(current[0]-previous[0]))
            previous = current
    result = {
        "status": "functional_validation_passed", "run_id": run_id,
        "scope": "Unattended built-in HTTP flow, GPU embeddings/clustering, Docker filtering, export and unchanged-output resume",
        "slurm": {"job_id": 54437, "partition": "amd-tw-verification", "gpus": 4},
        "models": {"teacher": "Qwen/Qwen3-4B", "helper": "Qwen/Qwen3-8B", "auditor": "Qwen/Qwen3-32B",
                   "embedder": "sentence-transformers/all-MiniLM-L6-v2"},
        "tests": tests, "smoke": smoke, "recovery": recovered,
        "annotation_import_live": read_json(evidence / "import-fallback.json"),
        "environment": read_json(evidence / "environment.json"),
        "docker_container_actions": {action: len(names) for action, names in container_actions.items()},
        "sampling_comparison": read_json(evidence / "sampling-comparison.json"),
        "audit": audit, "resources": read_json(run / "resource_summary.json"),
        "audit_followup": {
            "report": "smoke/audit-findings.md",
            "conclusion": "Functional HTTP release candidate; improve Stage 2 source grounding and judge calibration before a production-sized run or verified-quality claims.",
            "non_judge_filter_replay": read_json(evidence / f"audit-{run_id}" / "non-judge-replay.json"),
            "note": "Direct assistant inspection found real content defects and reviewer mistakes; raw verdict counts are not error-rate estimates."},
        "cleanup": {
            "sandbox_containers_remaining": len((evidence / "docker-final-containers.txt").read_text().splitlines()),
            "user_jobs_in_final_queue_snapshot": max(0, len((evidence / "queue-after-cancel.txt").read_text().splitlines())-1)},
        "stage_timing_from_logs": timings,
        "teacher_generation_tokens_per_second_median_while_at_least_32_running": statistics.median(rates) if rates else None,
        "gpu_peak_vram_bytes_by_rocm_smi_card": peaks,
        "performance_caveats": ["Resource samples are approximately 10 seconds apart.",
            "Serving counters include the concurrent Stage 5 recovery probe; rates are operational observations, not an isolated benchmark.",
            "rocm-smi card numbering is not asserted to match the process ROCR visibility order.",
            "Summed process-tree RSS can double-count shared pages and excludes separately launched model servers."],
        "inference_finish_reasons": dict(Counter(r.get("finish_reason") for r in inferred)),
        "truncated_completions_in_export": len(final_ids & truncated_ids),
        "final_domains": dict(Counter(r["domain"] for r in final)),
        "final_difficulties": dict(Counter(r["difficulty"] for r in final)),
        "sources": read_json(evidence / "sources/manifest.json"),
        "source_file_sha256_at_summary": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("sft_pipeline").rglob("*.py"))},
        "limitations": ["The audit is a model review, not verified ground truth or a population-weighted accuracy estimate.",
            "Use a fresh run ID after changing inputs, model or config; artifact invalidation remains deferred.",
            "Multi-file publication is recoverable by rerun but is not one atomic transaction.",
            "No math filter; successful code execution does not prove task correctness.",
            "Offline/distributed inference, clean-install validation and publication hygiene remain deferred.",
            "The tested ROCm image can mask failing process exit codes; checkpoint/artifact validation is required."]}
    output = Path("smoke/results/freeze-validation.json")
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "tests", "inference_finish_reasons", "final_domains", "resources")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="freeze_scale_v2")
    main(parser.parse_args().run_id)
