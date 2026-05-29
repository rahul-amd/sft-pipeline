#!/bin/bash
# =============================================================================
# Slurm job: Stage 6 quality filtering.
#
# For the structural+heuristic-only config (config/stage6_filter.yaml) this
# needs no GPU and no external server.  A single CPU node with ~32 GB RAM is
# sufficient for 7M records.
#
# Usage:
#   sbatch scripts/slurm_stage6.sh                        # default config
#   sbatch scripts/slurm_stage6.sh config/stage6_filter.yaml
#
# Prerequisites:
#   - Stage 5 complete: runs/research_stage3_001/stage5/part-*.jsonl exists
#
# Resume:
#   Re-submit at any time — already-processed records are skipped automatically.
# =============================================================================

#SBATCH --job-name=sft-stage6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=hissu
#SBATCH --output=./logs/slurm-%j-stage6.out
#SBATCH --error=./logs/slurm-%j-stage6.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: This script must be submitted via sbatch, not run directly."
    exit 1
fi

PIPELINE_CONFIG="${1:-config/stage6_filter.yaml}"

PROJECT_DIR="/scratch/project_462000963/users/aralikatte/sft-pipeline"
SCRATCH="/scratch/project_462000963"
SIF="${SCRATCH}/users/aralikatte/sincons/python_latest.sif"
OVERLAY="${SCRATCH}/users/aralikatte/sincons/python_latest_overlay.img"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "${PROJECT_DIR}/logs"

log "Job ID    : $SLURM_JOB_ID"
log "Node      : $(hostname)"
log "Config    : $PIPELINE_CONFIG"

SING="singularity exec \
    --bind ${SCRATCH} \
    --bind /users/aralikatte \
    --overlay ${OVERLAY}:ro ${SIF} \
    bash ${PROJECT_DIR}/scripts/run_in_env.sh"

log "Starting Stage 6 ..."
cd "$PROJECT_DIR"

$SING sft-pipeline run-stage stage6_filter --config "$PIPELINE_CONFIG"

EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && log "Stage 6 completed successfully." || log "ERROR: Stage 6 exited with code $EXIT_CODE."
log "Job finished."
exit $EXIT_CODE
