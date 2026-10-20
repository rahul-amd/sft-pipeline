"""Run the full validation while recording process memory and server metrics."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

import psutil

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="smoke/config-scale.yaml")
parser.add_argument("--timeout", type=int, default=21600)
args = parser.parse_args()
root = Path("smoke/runs") / os.environ["SFT_RUN_ID"]
root.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
process = subprocess.Popen([sys.executable, "smoke/run.py", "--config", args.config,
                            "--timeout", str(args.timeout)])
peak_rss = 0
samples = 0
try:
    with (root / "resources.jsonl").open("w") as stream:
        while process.poll() is None:
            rss = 0
            try:
                parent = psutil.Process(process.pid)
                for child in [parent, *parent.children(recursive=True)]:
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            peak_rss = max(peak_rss, rss)
            row = {"elapsed_seconds": round(time.monotonic()-start, 2), "process_tree_rss_bytes": rss}
            for label, variable in (("teacher", "SFT_API_BASE"), ("helper", "SFT_JUDGE_API_BASE")):
                endpoint = os.environ[variable].removesuffix("/v1") + "/metrics"
                try:
                    request = urllib.request.Request(endpoint, headers={
                        "Authorization": "Bearer " + os.environ["SFT_API_KEY"]})
                    with urllib.request.urlopen(request, timeout=5) as response:
                        text = response.read().decode()
                    row[label] = [line for line in text.splitlines() if line.startswith((
                        "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:request_success_total",
                        "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"))]
                except Exception as exc:
                    row[label] = {"error": str(exc)}
            tool = shutil.which("rocm-smi")
            if tool:
                check = subprocess.run([tool, "-d", "0", "1", "2", "3", "--showmeminfo", "vram", "--showuse", "--json"],
                                       capture_output=True, text=True, timeout=10)
                try:
                    row["gpu"] = json.loads(check.stdout)
                except ValueError:
                    row["gpu_probe_error"] = check.stderr[-300:]
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            samples += 1
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
finally:
    if process.poll() is None:
        process.terminate()
    report = {"elapsed_seconds": round(time.monotonic()-start, 2), "samples": samples,
              "peak_process_tree_rss_bytes": peak_rss, "exit_code": process.wait(),
              "memory_scope": "pipeline driver and descendants; summed RSS can double-count shared pages; excludes serving processes"}
    (root / "resource_summary.json").write_text(json.dumps(report, indent=2) + "\n")

success_path = root / "smoke_report.json"
if not success_path.exists() or json.loads(success_path.read_text()).get("status") != "passed":
    raise RuntimeError("No successful artifact validation report; child exit code alone is insufficient")
if report["exit_code"]:
    raise SystemExit(report["exit_code"])
