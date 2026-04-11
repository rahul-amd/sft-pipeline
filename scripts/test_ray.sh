#!/bin/bash
# =============================================================================
# Minimal Ray smoke test — 1 node, no pipeline code.
#
# Submits a single-node job that starts Ray, runs a trivial remote function,
# checks GPU visibility, and prints a pass/fail summary.
#
# Usage:
#   sbatch scripts/test_ray.sh
#
# Check results:
#   tail -f logs/slurm-<jobid>-ray-test.out
# =============================================================================

#SBATCH --job-name=ray-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=64G
#SBATCH --time=0:10:00
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --output=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-ray-test.out
#SBATCH --error=/scratch/project_462000963/users/aralikatte/sft-pipeline/logs/slurm-%j-ray-test.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: submit via sbatch, not run directly."
    exit 1
fi

PROJECT_DIR="/scratch/project_462000963/users/aralikatte/sft-pipeline"
SCRATCH="/scratch/project_462000963"
SIF="${SCRATCH}/users/aralikatte/sincons/python_latest.sif"
OVERLAY="${SCRATCH}/users/aralikatte/sincons/python_latest_overlay.img"

# /tmp is local to the compute node — short path, no quota, no AF_UNIX limit issues
RAY_TEMP_DIR="/tmp/ray_${SLURM_JOB_ID}"
RAY_PORT=6379

SING="singularity exec --rocm \
    --bind ${SCRATCH} --bind /users/aralikatte \
    --bind /opt/rocm \
    --overlay ${OVERLAY}:ro ${SIF} \
    ${PROJECT_DIR}/scripts/run_in_env.sh"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${RAY_TEMP_DIR}"

HEAD_IP=$(hostname --ip-address | tr ' ' '\n' \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | grep -v '^127\.' | head -1)

log "Node     : $(hostname) (${HEAD_IP})"
log "Job ID   : ${SLURM_JOB_ID}"
log "Ray tmp  : ${RAY_TEMP_DIR}"

# ── Start Ray head ────────────────────────────────────────────────────────────
log "Starting Ray head ..."
$SING ray start \
    --head \
    --node-ip-address="${HEAD_IP}" \
    --port=${RAY_PORT} \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus=8 \
    --temp-dir="${RAY_TEMP_DIR}" \
    --block &

RAY_PID=$!
log "Waiting 45s for Ray to initialise ..."
sleep 45

# ── Verify head is up ─────────────────────────────────────────────────────────
if ! $SING ray status --address="${HEAD_IP}:${RAY_PORT}" 2>/dev/null; then
    log "FAIL: Ray head did not start within 45s."
    log "Check the .err log for details."
    kill $RAY_PID 2>/dev/null || true
    exit 1
fi
log "Ray head is up."

# ── Run smoke test ────────────────────────────────────────────────────────────
log "Running smoke test ..."
$SING python3 - <<'PYEOF'
import ray, torch, os

ray.init(address="auto")
print(f"  Ray version  : {ray.__version__}")
print(f"  Cluster resources: {ray.cluster_resources()}")

@ray.remote(num_gpus=1)
def gpu_check():
    avail = torch.cuda.is_available()
    n     = torch.cuda.device_count() if avail else 0
    name  = torch.cuda.get_device_name(0) if avail else "N/A"
    pid   = os.getpid()
    ver   = torch.__version__
    rocr  = os.environ.get("ROCR_VISIBLE_DEVICES", "(not set)")
    hip   = os.environ.get("HIP_VISIBLE_DEVICES", "(not set)")
    return {
        "cuda_available": avail,
        "device_count": n,
        "device_name": name,
        "torch_version": ver,
        "pid": pid,
        "ROCR_VISIBLE_DEVICES": rocr,
        "HIP_VISIBLE_DEVICES": hip,
    }

futures = [gpu_check.remote() for _ in range(8)]
results = ray.get(futures)

all_ok = True
for i, r in enumerate(results):
    status = "OK" if r["cuda_available"] else "NO GPU"
    print(f"  GPU task {i}: [{status}] device={r['device_name']!r} "
          f"torch={r['torch_version']} "
          f"ROCR={r['ROCR_VISIBLE_DEVICES']} HIP={r['HIP_VISIBLE_DEVICES']}")
    if not r["cuda_available"]:
        all_ok = False

ray.shutdown()
if not all_ok:
    raise SystemExit("FAIL: one or more GPU tasks could not see a GPU")
print("PASS: all 8 GPU tasks succeeded")
PYEOF

SMOKE_EXIT=$?

# ── Tear down ─────────────────────────────────────────────────────────────────
log "Stopping Ray ..."
$SING ray stop --force 2>/dev/null || true
wait $RAY_PID 2>/dev/null || true
rm -rf "${RAY_TEMP_DIR}" || true

if [ $SMOKE_EXIT -eq 0 ]; then
    log "PASS: Ray is working correctly on this node."
else
    log "FAIL: Smoke test exited with code ${SMOKE_EXIT}."
fi
exit $SMOKE_EXIT
