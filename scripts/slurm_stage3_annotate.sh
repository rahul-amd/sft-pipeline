#!/bin/bash
# =============================================================================
# Slurm job: LLM annotation of prompts via an external vLLM server.
#
# This runs ONLY the annotation step — no GPUs, no Ray, no embedding.
# Requires a single CPU node with enough RAM to hold the full prompt list
# in memory (~1–2 GB for 50M prompts).
#
# Usage:
#   sbatch scripts/slurm_annotate.sh
#   sbatch scripts/slurm_annotate.sh config/stage3_annotate.yaml
#
# Prerequisites:
#   - Stage 1 output exists at stage1_collect.output_dir in the config
#   - vLLM server is already running at stage3_cluster.annotation_api_base
#     (start it with: sbatch vllm/slurm_serve.sh  or  bash vllm/serve.sh)
#
# Resume:
#   Re-submit this job at any time — already-annotated prompts are skipped
#   automatically from the checkpoint in {stage3.output_dir}/annotations.parquet.
#
# Time estimate (rough):
#   50M prompts ÷ 64 concurrency × ~1 s/prompt = ~220 hours single-replica.
#   With 4 vLLM replicas behind nginx (concurrency=256): ~55 hours.
#   With 16 replicas (concurrency=1024): ~14 hours.
#   With 64 replicas (concurrency=4096): ~3.4 hours best-case; --time=12:00:00 for safety.
#   Adjust annotation_concurrency in the config to match your setup.
# =============================================================================

#SBATCH --job-name=sft-annotate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g          # CPU-only partition — no GPU allocation needed
#SBATCH --output=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-annotate.out
#SBATCH --error=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-annotate.err

set -euo pipefail

# ── Guard: must be submitted via sbatch ──────────────────────────────────────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: This script must be submitted via sbatch, not run directly."
    echo "       Usage: sbatch scripts/slurm_annotate.sh [config/path.yaml]"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────
PIPELINE_CONFIG="${1:-config/stage3_annotate.yaml}"

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
# CPU-only container: just bind scratch + home, no ROCm binds needed.
SING="singularity exec \
    --bind ${SCRATCH} \
    --bind /users/aralikatte \
    --overlay ${OVERLAY}:ro ${SIF} \
    ${PROJECT_DIR}/scripts/run_in_env.sh"

# ── Sanity check: vLLM server reachable? ─────────────────────────────────────
# Extract api_base from config and do a quick /health check.
API_BASE=$(grep 'annotation_api_base:' "${PROJECT_DIR}/${PIPELINE_CONFIG}" \
    | awk '{print $2}' | tr -d '"' | head -1)
HEALTH_URL="${API_BASE%/v1}/health"

log "Checking vLLM server at $HEALTH_URL ..."
if curl --silent --fail --max-time 10 "$HEALTH_URL" > /dev/null 2>&1; then
    log "vLLM server is reachable."
else
    log "WARNING: vLLM server health check failed ($HEALTH_URL)."
    log "         Make sure the server is running before submitting this job."
    log "         Continuing anyway — annotation will fail immediately if unreachable."
fi

# ── Raise file descriptor limit ──────────────────────────────────────────────
# annotation_concurrency=4096 opens ~4096 simultaneous TCP connections.
# Default ulimit -n is 1024 on most clusters → EMFILE / "Too many open files".
ulimit -n 16384

# ── Run annotation ────────────────────────────────────────────────────────────
log "Starting annotation ..."
cd "$PROJECT_DIR"

$SING sft-pipeline annotate --config "$PIPELINE_CONFIG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "Annotation completed successfully."
else
    log "ERROR: annotation exited with code $EXIT_CODE."
fi

log "Job finished."
exit $EXIT_CODE
