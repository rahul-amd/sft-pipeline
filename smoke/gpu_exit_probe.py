"""Check real process exit codes after initializing optional GPU libraries."""
import json
from pathlib import Path
import subprocess
import sys

root = Path("smoke/runs/gpu_matrix_20261008")
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
cases = {
    "python": "pass",
    "torch_gpu": "import torch; torch.ones(1, device='cuda')",
    "embedding_gpu": "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda'); m.encode(['exit code probe'])",
}
results = {}
for name, setup in cases.items():
    with (root / f"exit-{name}{suffix}.log").open("w") as log:
        result = subprocess.run([sys.executable, "-c", setup + "; import sys; sys.exit(17)"],
                                stdout=log, stderr=subprocess.STDOUT, timeout=120)
    results[name] = {"expected": 17, "actual": result.returncode}
(root / f"exit-codes{suffix}.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results))
