#!/bin/bash
# =============================================================================
# Job array: one vLLM worker per task, 1 node × 2 GPUs each.
#
# Submit with:
#   sbatch vllm/slurm_serve_array.sh                  # 16 workers (default)
#   sbatch --array=0-7 vllm/slurm_serve_array.sh      # 8 workers
#
# Then submit the nginx coordinator (see slurm_nginx.sh):
#   sbatch --dependency=after:<array_job_id> \
#          --export=ALL,ARRAY_JOB_ID=<array_job_id>,N_WORKERS=<N> \
#          vllm/slurm_nginx.sh
#
# Or use the one-liner helper at the bottom of this file's comments.
#
# Environment overrides:
#   MODEL         HF model id          (default: Qwen/Qwen3-30B-A3B-Thinking-2507)
#   TP            tensor-parallel size (default: 2)
#   MAX_MODEL_LEN max sequence length  (default: unset → model default)
#   PORT          vLLM HTTP port       (default: 8000)
#   GPU_MEM_UTIL  GPU memory fraction  (default: 0.92)
#   SCRATCH       base scratch path    (default: /scratch/project_462000963)
#
# One-liner to submit array + nginx in one go:
#   JID=$(sbatch --parsable vllm/slurm_serve_array.sh) && \
#   echo "Workers: $JID" && \
#   sbatch --dependency=after:$JID \
#          --export=ALL,ARRAY_JOB_ID=$JID,N_WORKERS=$(( $(scontrol show job $JID | grep -o 'ArrayTaskId=[0-9-]*' | grep -o '[0-9]*$') + 1 )) \
#          vllm/slurm_nginx.sh
# =============================================================================
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --job-name=vllm_worker
#SBATCH --output=logs/vllm_worker_%A_%a.log

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/project_462000963}"
PORT="${PORT:-8000}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Rendezvous: advertise this worker's address ───────────────────────────────
# The nginx coordinator (slurm_nginx.sh) reads these files to build its upstream
# config.  One file per array task: content is "hostname:port".
RENDEZVOUS_DIR="${SCRATCH}/users/aralikatte/vllm_rendezvous/${SLURM_ARRAY_JOB_ID}"
mkdir -p "${RENDEZVOUS_DIR}"
echo "$(hostname):${PORT}" > "${RENDEZVOUS_DIR}/${SLURM_ARRAY_TASK_ID}.txt"
log "Registered as $(hostname):${PORT} → ${RENDEZVOUS_DIR}/${SLURM_ARRAY_TASK_ID}.txt"

# ── Hand off to serve.sh ──────────────────────────────────────────────────────
# serve.sh inherits MODEL, TP, MAX_MODEL_LEN, PORT, GPU_MEM_UTIL, SCRATCH, etc.
# ROCM_COMPAT is left at default (0) — correct for sbatch jobs.
exec bash vllm/serve.sh
