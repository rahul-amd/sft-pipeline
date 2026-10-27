"""Durable audit driver: sample, validate a pilot, review all records, summarize."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def status(stage, **extra):
    value = {"stage": stage, "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = ROOT / "run-status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(ROOT / "run-status.json")
    print(json.dumps(value), flush=True)


def run(script, log, *args):
    with (ROOT / log).open("a") as output:
        return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT,
            stdout=output, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}).returncode


def main():
    lock = (ROOT / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status("sampling", total_samples=3200)
    if run("sample.py", "sampling.log", "--workers", "2"):
        raise RuntimeError("Sampling failed; inspect sampling.log. Completed shard caches are resumable.")
    status("pilot_review", total_samples=3200)
    if run("review.py", "review-pilot.log", "--concurrency", "2", "--limit-batches", "2"):
        raise RuntimeError("Pilot failed; inspect review-pilot.log before scaling up")
    for attempt in range(1, 4):
        status("reviewing", total_samples=3200, resume_attempt=attempt)
        if run("review.py", "review.log", "--concurrency", "8") == 0:
            break
        if attempt == 3:
            raise RuntimeError("Review incomplete after retries; inspect progress.json. Valid reviews are preserved.")
        time.sleep(30)
    status("validating_and_summarizing", total_samples=3200)
    if run("summarize.py", "summary.log"):
        raise RuntimeError("Final validation/aggregation failed; inspect summary.log")
    status("complete", total_samples=3200, complete=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc), complete=False)
        raise
