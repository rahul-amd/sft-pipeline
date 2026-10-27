#!/usr/bin/env bash
set -euo pipefail
quest_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$quest_dir/container_exec.sh" /work/serve.py serve Cloudflare/clef-flash \
  --revision f0b7d3cd65296abfe459e6c228b618f5d8b26d2f \
  --served-model-name Cloudflare/clef-flash \
  --host 0.0.0.0 --port 18971 \
  --dtype bfloat16 --max-model-len 98304 --max-num-batched-tokens 98304 \
  --max-num-seqs 4 --gpu-memory-utilization 0.85 --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}'
