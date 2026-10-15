#!/usr/bin/env bash
# Run from a Slurm compute step. All writable validation assets stay in smoke/.
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
container_path="${SFT_GPU_CONTAINER:-/shared_silo/scratch/containers/vllm-openai-rocm-gemma4.sif}"
runtime_dir="${SFT_RUNTIME_DIR:-$root_dir/smoke/runs/gpu_matrix_20261008}"
export APPTAINERENV_OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export APPTAINERENV_HF_HOME="${HF_HOME:-/shared_silo/scratch/hf_cache}"
export APPTAINERENV_PYTHONPATH="$root_dir"
# ROCR selects physical devices; HIP indexes that already-filtered list.
export APPTAINERENV_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra assigned_gpus <<< "$APPTAINERENV_ROCR_VISIBLE_DEVICES"
logical_gpus=0
for ((i=1; i<${#assigned_gpus[@]}; i++)); do logical_gpus+=",$i"; done
export APPTAINERENV_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$logical_gpus}"
export APPTAINERENV_TMPDIR="/tmp"
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
export APPTAINERENV_VLLM_CACHE_ROOT="$runtime_dir/cache/vllm"
export APPTAINERENV_TRITON_CACHE_DIR="$runtime_dir/cache/triton"
if [[ -f "$runtime_dir/ray-token" ]]; then
    export APPTAINERENV_RAY_AUTH_MODE=token
    export APPTAINERENV_RAY_AUTH_TOKEN_PATH="$runtime_dir/ray-token"
fi
for variable in SFT_API_BASE SFT_API_KEY SFT_MODEL SFT_RUN_ID SFT_RAY_ADDRESS \
                SFT_JUDGE_API_BASE SFT_JUDGE_MODEL \
                SFT_TEST_DOCKER SFT_SKIP_NETWORK_TESTS COVERAGE_PROCESS_START \
                COVERAGE_FILE RAY_ADDRESS RAY_TMPDIR; do
    if [[ -v "$variable" ]]; then
        export "APPTAINERENV_$variable=${!variable}"
    fi
done
exec singularity exec --rocm --cleanenv \
    --bind /shared_silo/scratch,/usr/bin/docker:/usr/bin/docker,/var/run/docker.sock:/var/run/docker.sock \
    "$container_path" "$@"
