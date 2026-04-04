#!/bin/bash
# =============================================================================
# Slurm job: Stage 1 distributed prompt collection across 32 nodes.
#
# Starts a Ray cluster (head + workers), then runs stage1_collect in
# distributed mode. All 32 nodes read from the shared filesystem and
# write phase1 outputs to the shared scratch directory.
#
# Usage:
#   sbatch scripts/slurm_stage1.sh
#   sbatch scripts/slurm_stage1.sh config/stage1_research.yaml   # custom config
#
# Requirements:
#   - stage1_collect.distributed must be true in the config
#   - sft-pipeline installed via:
#       singularity exec $SIF pip install --user -e /scratch/project_462000963/users/aralikatte/sft-pipeline/
#   - Shared filesystem mounted at /scratch/project_462000963 on all nodes
# =============================================================================

#SBATCH --job-name=sft-stage1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --output=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-stage1.out
#SBATCH --error=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-stage1.err

set -euo pipefail

# ── Guard: must be submitted via sbatch, not run directly ────────────────────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: This script must be submitted via sbatch, not run directly."
    echo "       Usage: sbatch scripts/slurm_stage1.sh [config/path.yaml]"
    exit 1
fi

# ── Config ───────────────────────────────────────────────────────────────────
PIPELINE_CONFIG="${1:-config/stage1_research.yaml}"

PROJECT_DIR="/scratch/project_462000963/users/aralikatte/sft-pipeline"
SCRATCH="/scratch/project_462000963"
SIF="${SCRATCH}/users/aralikatte/sincons/python_latest.sif"
OVERLAY="${SCRATCH}/users/aralikatte/sincons/python_latest_overlay.img"

RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

# ── Singularity wrapper ───────────────────────────────────────────────────────
# run_in_env.sh activates the sft-pipeline conda env inside the container
# before running the command. The overlay is mounted :ro so multiple nodes
# can open it simultaneously.
SING="singularity exec --bind ${SCRATCH} --bind /users/aralikatte --bind /opt/rocm \
    --overlay ${OVERLAY}:ro ${SIF} \
    ${PROJECT_DIR}/scripts/run_in_env.sh"

# ── Logging helpers ───────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Ensure logs dir exists ────────────────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/logs"

# ── Node discovery ────────────────────────────────────────────────────────────
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${ALL_NODES[0]}"
WORKER_NODES=("${ALL_NODES[@]:1}")
N_WORKERS="${#WORKER_NODES[@]}"

# Resolve head IP — pick the first non-loopback IPv4 address
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    hostname --ip-address | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | grep -v '^127\.' | head -1)

log "Head node : $HEAD_NODE ($HEAD_IP)"
log "Workers   : $N_WORKERS nodes"
log "Config    : $PIPELINE_CONFIG"
log "Project   : $PROJECT_DIR"
log "Container : $SIF"

# ── Verify config has distributed: true ───────────────────────────────────────
if ! grep -q "distributed: true" "${PROJECT_DIR}/${PIPELINE_CONFIG}"; then
    log "WARNING: '${PIPELINE_CONFIG}' does not contain 'distributed: true'."
    log "         Stage 1 will run single-node on the head. Add the following"
    log "         to your config under stage1_collect:"
    log "           stage1_collect:"
    log "             distributed: true"
fi

# ── Clean up stale /tmp/ray from any previous job ────────────────────────────
# /tmp is node-local; a previous job may have left a stale /tmp/ray dir which
# causes "FileExistsError: [Errno 17] File exists: '/tmp/ray'" on ray start.
log "Cleaning up stale /tmp/ray on all nodes ..."
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" rm -rf /tmp/ray 2>/dev/null || true

# ── Start Ray head ────────────────────────────────────────────────────────────
log "Starting Ray head on $HEAD_NODE ..."

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    $SING ray start \
        --head \
        --node-ip-address="$HEAD_IP" \
        --port=$RAY_PORT \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --num-cpus="$SLURM_CPUS_PER_TASK" \
        --block &

RAY_HEAD_PID=$!
log "Waiting for head to initialise (20s) ..."
sleep 20

# Quick sanity check that the head is up
# --overlap lets this srun share the head node's task slot with the --block ray start
if ! srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        $SING ray status --address="${HEAD_IP}:${RAY_PORT}" &>/dev/null; then
    log "ERROR: Ray head did not start. Check the .err log for details."
    exit 1
fi
log "Ray head is up."

# ── Start Ray workers ─────────────────────────────────────────────────────────
if [ "$N_WORKERS" -gt 0 ]; then
    WORKER_LIST=$(IFS=','; echo "${WORKER_NODES[*]}")
    log "Starting $N_WORKERS Ray workers ..."

    srun --nodes="$N_WORKERS" --ntasks="$N_WORKERS" -w "$WORKER_LIST" \
        $SING ray start \
            --address="${HEAD_IP}:${RAY_PORT}" \
            --num-cpus="$SLURM_CPUS_PER_TASK" \
            --block &

    log "Waiting for workers to join (30s) ..."
    sleep 30

    # Log how many nodes Ray sees
    CLUSTER_INFO=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
        $SING ray status --address="${HEAD_IP}:${RAY_PORT}" 2>/dev/null || echo "(status unavailable)")
    log "Ray cluster status:"
    echo "$CLUSTER_INFO"
fi

# ── Run Stage 1 ───────────────────────────────────────────────────────────────
log "Launching Stage 1 (distributed) ..."
cd "$PROJECT_DIR"

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    $SING sft-pipeline run-stage stage1_collect \
        --config "$PIPELINE_CONFIG"

PIPELINE_EXIT=$?

if [ $PIPELINE_EXIT -eq 0 ]; then
    log "Stage 1 completed successfully."
else
    log "ERROR: Stage 1 exited with code $PIPELINE_EXIT."
fi

# ── Tear down Ray ─────────────────────────────────────────────────────────────
log "Stopping Ray cluster ..."
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    $SING ray stop --force || true

# The --block srun jobs will exit once ray stop is called
wait $RAY_HEAD_PID 2>/dev/null || true

log "Job finished."
exit $PIPELINE_EXIT
