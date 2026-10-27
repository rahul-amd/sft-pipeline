#!/usr/bin/env bash
set -euo pipefail
quest_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$quest_dir/../.." && pwd)"
exec docker run --rm --cpus 2 --memory 12g --shm-size 1g --user "$(id -u):$(id -g)" \
 -v "$repo_dir:/repo:ro" -v "$quest_dir:/work" \
 -e HF_HOME=/work/cache/hf -e HF_XET_CACHE=/work/cache/xet \
 -e XDG_CACHE_HOME=/work/cache/xdg -e VLLM_CACHE_ROOT=/work/cache/vllm \
 -e AITER_JIT_DIR=/work/cache/aiter -e AITER_ROOT_DIR=/work/cache/aiter-root \
 -e NUMBA_CACHE_DIR=/work/cache/numba -e TRITON_CACHE_DIR=/work/cache/triton \
 -e TVM_FFI_CACHE_DIR=/work/cache/tvm -e TMPDIR=/work/tmp \
 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_NO_USAGE_STATS=1 \
 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e TOKENIZERS_PARALLELISM=false \
 --entrypoint python vllm/vllm-openai-rocm:nightly-rocm72-5d35ce3934a7804b298559d2d99eec33792e1eb9 "$@"
