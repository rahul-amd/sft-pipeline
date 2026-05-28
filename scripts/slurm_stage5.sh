#!/bin/bash
# =============================================================================
# Slurm job: Stage 5 inference — async HTTP calls to an external vLLM server.
#
# This stage needs NO GPU on the pipeline node; all model compute runs on
# the vLLM server (started separately via vllm/slurm_serve_array.sh or
# vllm/serve.sh).  A single CPU node with ~32 GB RAM is sufficient.
#
# Usage:
#   sbatch scripts/slurm_stage5.sh                          # default config
#   sbatch scripts/slurm_stage5.sh config/stage5_inference.yaml
#
# Prerequisites:
#   1. Stage 4 complete: runs/research_stage3_001/stage4/part-*.jsonl exists
#   2. vLLM server is already running and reachable at stage5_inference.api_base
#      Check with:  curl http://<node>:<port>/v1/models
#   3. Update api_base in config/stage5_inference.yaml to the actual server URL
#
# Resume:
#   Re-submit this job at any time — already-written responses are skipped
#   automatically via DuckDB checkpointing.
#
# Time estimates (7M prompts, 8192 max_tokens):
#   4 replicas, concurrency=256  → ~10 days
#  16 replicas, concurrency=1024 → ~2.5 days
#  64 replicas, concurrency=4096 → ~15 hours
#  Adjust concurrency in config/stage5_inference.yaml before submitting.
# =============================================================================

#SBATCH --job-name=sft-stage5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g          # CPU-only partition — no GPU allocation needed
#SBATCH --output=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-stage5.out
#SBATCH --error=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-stage5.err

set -euo pipefail

# ── Guard: must be submitted via sbatch ──────────────────────────────────────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: This script must be submitted via sbatch, not run directly."
    echo "       Usage: sbatch scripts/slurm_stage5.sh [config/path.yaml]"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────
PIPELINE_CONFIG="${1:-config/stage5_inference.yaml}"

PROJECT_DIR="/scratch/project_462000963/users/aralikatte/sft-pipeline"
SCRATCH="/scratch/project_462000963"
SIF="${SCRATCH}/users/aralikatte/sincons/python_latest.sif"
OVERLAY="${SCRATCH}/users/aralikatte/sincons/python_latest_overlay.img"

# ── Logging helper ────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "${PROJECT_DIR}/logs"

log "Job ID    : $SLURM_JOB_ID"
log "Node      : $(hostname)"
log "Config    : $PIPELINE_CONFIG"
log "Project   : $PROJECT_DIR"

# ── Singularity wrapper (no --rocm — no GPUs requested) ──────────────────────
SING="singularity exec \
    --bind ${SCRATCH} \
    --bind /users/aralikatte \
    --overlay ${OVERLAY}:ro ${SIF} \
    bash ${PROJECT_DIR}/scripts/run_in_env.sh"

# ── Sanity check: vLLM server reachable? ─────────────────────────────────────
# Extract api_base from the config YAML and hit /v1/models.
API_BASE=$(grep 'api_base:' "${PROJECT_DIR}/${PIPELINE_CONFIG}" \
    | grep -v '#' | head -1 | awk '{print $2}' | tr -d '"')
MODELS_URL="${API_BASE}/models"

log "Checking vLLM server at $MODELS_URL ..."
if curl --silent --fail --max-time 15 "$MODELS_URL" > /dev/null 2>&1; then
    log "vLLM server is reachable."
else
    log "WARNING: vLLM server health check failed ($MODELS_URL)."
    log "         Ensure the vLLM job is running before submitting stage 5."
    log "         Continuing anyway — inference tasks will fail immediately if unreachable."
fi

# ── Raise file descriptor limit ──────────────────────────────────────────────
# concurrency=4096 opens ~4096 simultaneous TCP connections.
# Default ulimit -n (1024) causes "Too many open files".
ulimit -n 65536

# ── Run Stage 5 ───────────────────────────────────────────────────────────────
log "Starting Stage 5 inference ..."
cd "$PROJECT_DIR"

$SING sft-pipeline run-stage stage5_inference --config "$PIPELINE_CONFIG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "Stage 5 completed successfully."
else
    log "ERROR: Stage 5 exited with code $EXIT_CODE."
fi

log "Job finished."
exit $EXIT_CODE
