#!/usr/bin/env bash
set -euo pipefail
quest_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$quest_dir/container_exec.sh" /work/serve.py serve Cloudflare/clef \
 --revision a30162b99d68a481d8bd75d38df069a0cf83df40 \
 --served-model-name Cloudflare/clef --host 0.0.0.0 --port "${CLEF_PORT:-18972}" \
 --dtype bfloat16 --max-model-len 98304 --max-num-batched-tokens 98304 \
 --max-num-seqs 2 --gpu-memory-utilization 0.85 --enforce-eager \
 --hf-overrides '{"architectures":["ClefFlashForJointSchema"],"joint_head_config":{"hidden_size":5120,"width":1024,"routing_layers":2,"layers":4,"heads":16,"feedforward":4096}}' \
 --limit-mm-per-prompt '{"image":0,"video":0}'
