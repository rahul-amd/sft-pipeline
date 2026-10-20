"""Archive compact evidence from the completed GPU validation runs."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
matrix = ROOT / "smoke/runs/gpu_matrix_20261008"
run_names = ["gpu_full_v3_20261008", "gpu_distributed_v3_20261008", "gpu_offline_full_20261008"]
result = {"date": "2026-10-08", "slurm_job": "54411", "allocation_released": True,
          "partition_constraint": "Do not use hissu for future jobs", "full_runs": {}}
for name in run_names:
    result["full_runs"][name] = json.loads((ROOT / "smoke/runs" / name / "smoke_report.json").read_text())
result["tests"] = {}
for name in ("tests-v2", "additional-tests"):
    suites = ET.parse(matrix / f"{name}.xml").getroot()
    suite = suites.find("testsuite")
    result["tests"][name] = {k: suite.attrib[k] for k in ("tests", "errors", "failures", "skipped", "time")}
result["coverage"] = json.loads((matrix / "coverage.json").read_text())["totals"]
result["probes"] = {name: json.loads((matrix / f"{name}.json").read_text()) for name in (
    "components", "ray_probe", "corpus_hf", "annotations", "cli_paths", "offline_single", "offline_ray",
    "cached-flat-observations", "exit-codes", "exit-codes-uncovered-gpu0")}
result["environment"] = json.loads((matrix / "preflight-installed.json").read_text())
result["live_exclusions"] = ["E2B (no SDK/credential)", "Hugging Face publication (protocol tests only)",
    "FAISS GPU kernels (CPU wheel; fallback tested)", "NVIDIA/CUDA runtime",
    "Tensor/pipeline parallel degree greater than one", "Unreachable Stage 3 import fallback block"]
out = ROOT / "smoke/results/gpu-validation.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
