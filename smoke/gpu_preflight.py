"""Report capabilities without exposing credential values."""
import importlib
import importlib.util
import json
import os
import platform
import socket
import subprocess

packages = ["torch", "vllm", "transformers", "sentence_transformers", "faiss", "hdbscan",
            "polars", "pyarrow", "ray", "duckdb", "datasets", "json_repair", "openai",
            "orjson", "pytest", "flash_kmeans", "coverage", "e2b_code_interpreter"]
result = {"host": socket.gethostname(), "python": platform.python_version(), "packages": {}}
for name in packages:
    try:
        module = importlib.import_module(name)
        result["packages"][name] = getattr(module, "__version__", "installed")
    except Exception as exc:
        result["packages"][name] = {"error": str(exc)}
import torch
result["gpu"] = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(),
                 "hip": torch.version.hip,
                 "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
result["e2b_key_available"] = bool(os.environ.get("E2B_API_KEY"))
try:
    check = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=15)
    result["docker"] = {"exit_code": check.returncode, "version": check.stdout.strip(),
                        "error": check.stderr[-500:]}
except Exception as exc:
    result["docker"] = {"error": str(exc)}
print(json.dumps(result, indent=2))
