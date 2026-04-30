#!/bin/bash
# =============================================================================
# Slurm job: Stage 4 — quota sampling + near-dedup from Stage 3 output.
#
# Usage:
#   sbatch scripts/slurm_stage4.sh
#   sbatch scripts/slurm_stage4.sh config/stage4_sample.yaml
#
# What this does:
#   - Loads all Stage 3 JSONL (~15M records) into a Polars DataFrame
#   - Loads sentence embeddings (~30 GB float16) for cosine near-dedup
#   - Enforces per-domain × per-difficulty quotas to reach ~7M prompts
#   - Removes near-duplicates via FAISS flat index (GPU-accelerated)
#   - Sorts by embedding similarity for KV-cache efficiency in Stage 5
#   - Writes deduplicated, sorted JSONL to stage4/
#
# Prerequisites:
#   - Stage 3 complete: runs/research_stage3_001/stage3/part-*.jsonl
#   - Embeddings present: runs/research_stage3_001/stage3/embeddings/
#
# Resume:
#   Re-submit at any time — Stage 4 is fast enough to re-run from scratch
#   (~10-30 min total). DuckDB checkpoint skips the stage if already complete.
#
# Memory notes:
#   15M embeddings × 1024-dim × float16 = ~30 GB just for embeddings.
#   Polars DataFrame for 15M records adds ~5-10 GB. Request ≥128G.
#
# Time estimate:
#   Sampling + quota enforcement: < 5 min
#   Loading embeddings:           ~5 min
#   FAISS dedup (7M vectors, GPU): ~10-20 min
#   Total: ~30 min; --time=2:00:00 for safety.
# =============================================================================

#SBATCH --job-name=sft-stage4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --output=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-stage4.out
#SBATCH --error=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-stage4.err

set -euo pipefail

# ── Guard: must be submitted via sbatch ──────────────────────────────────────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: This script must be submitted via sbatch, not run directly."
    echo "       Usage: sbatch scripts/slurm_stage4.sh [config/path.yaml]"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────
PIPELINE_CONFIG="${1:-config/stage4_sample.yaml}"

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

# ── Singularity wrapper ───────────────────────────────────────────────────────
# --rocm needed for GPU cosine dedup (FAISS flat index).
# /opt/rocm bound for libamdhip64.so (run_in_env.sh sets LD_LIBRARY_PATH).
SING="singularity exec --rocm \
    --bind ${SCRATCH} \
    --bind /users/aralikatte \
    --bind /opt/rocm \
    --overlay ${OVERLAY}:ro ${SIF} \
    bash ${PROJECT_DIR}/scripts/run_in_env.sh"

# ── Run stage 4 ───────────────────────────────────────────────────────────────
log "Starting Stage 4 (sampling + dedup) ..."
cd "$PROJECT_DIR"

$SING sft-pipeline run-stage stage4_sample --config "$PIPELINE_CONFIG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "Stage 4 completed successfully."
else
    log "ERROR: Stage 4 exited with code $EXIT_CODE."
fi

log "Job finished."
exit $EXIT_CODE
