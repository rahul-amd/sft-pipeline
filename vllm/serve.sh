#!/bin/bash
# =============================================================================
# Start the vLLM OpenAI-compatible HTTP server from the ROCm SIF.
#
# Run on a GPU compute node (interactive or via Slurm — see slurm_serve.sh).
#
# Usage:
#   bash vllm/serve.sh --model <HF-model-id>  [--tensor-parallel-size N]
#
#   # Examples
#   bash vllm/serve.sh --model Qwen/Qwen2.5-7B-Instruct
#   bash vllm/serve.sh --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 8
#   MODEL=Qwen/Qwen3.5-122B-A10B TP=16 bash vllm/serve.sh
#
# Environment overrides:
#   MODEL        HuggingFace model id (required unless --model is passed)
#   TP           tensor-parallel size  (default: 8, one per MI250X GCD pair)
#   PORT         server port           (default: 8000)
#   HOST         bind address          (default: 0.0.0.0)
#   SIF          path to vllm_rocm.sif
#   HF_HOME      HF cache directory    (default: ${SCRATCH}/hf_cache)
#   GPU_MEM_UTIL fraction of GPU memory to use (default: 0.92)
#   MAX_MODEL_LEN max context length   (default: unset → model default)
#   SCRATCH      base scratch path     (default: /scratch/project_462000963)
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRATCH="${SCRATCH:-/scratch/project_462000963}"
SINCONS_DIR="${SINCONS_DIR:-${SCRATCH}/users/aralikatte/sincons}"
SIF="${SIF:-${SINCONS_DIR}/vllm_rocm.sif}"

MODEL="${MODEL:-}"
TP="${TP:-8}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"

# ── Parse --model / --tensor-parallel-size flags ─────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL="$2"; shift 2 ;;
        --tensor-parallel-size) TP="$2";    shift 2 ;;
        --port)                 PORT="$2";  shift 2 ;;
        --help|-h)
            grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "${MODEL}" ]; then
    err "No model specified.  Pass --model <HF-model-id> or set MODEL= env var."
    exit 1
fi

if [ ! -f "${SIF}" ]; then
    err "SIF not found: ${SIF}"
    err "Build it first:  bash vllm/build_sif.sh"
    exit 1
fi

# ── Detect singularity vs apptainer ──────────────────────────────────────────
if   command -v apptainer  >/dev/null 2>&1; then SING_CMD=apptainer
elif command -v singularity >/dev/null 2>&1; then SING_CMD=singularity
else
    err "Neither 'singularity' nor 'apptainer' found in PATH."
    exit 1
fi

# ── Build vllm serve argument list ───────────────────────────────────────────
VLLM_ARGS=(
    python -m vllm.entrypoints.openai.api_server
    --model              "${MODEL}"
    --tensor-parallel-size "${TP}"
    --dtype              float16
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --host               "${HOST}"
    --port               "${PORT}"
    --trust-remote-code
)

if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

# ── Print plan ────────────────────────────────────────────────────────────────
echo
log "SIF              : ${SIF}"
log "Model            : ${MODEL}"
log "Tensor parallel  : ${TP}"
log "Port             : ${PORT}"
log "GPU mem util     : ${GPU_MEM_UTIL}"
log "HF_HOME          : ${HF_HOME}"
echo

# ── Verify GPU is visible ─────────────────────────────────────────────────────
if [ ! -e /dev/kfd ]; then
    err "/dev/kfd not found — are you on a GPU compute node?"
    err "Request one first:"
    echo "  srun --account=project_462000963 --partition=standard-g \\"
    echo "       --nodes=1 --ntasks=1 --gpus-per-node=${TP} --mem=128G \\"
    echo "       --time=4:00:00 --pty bash"
    exit 1
fi

# ── ROCm sanity check on host ─────────────────────────────────────────────────
if command -v /opt/rocm/bin/rocm-smi >/dev/null 2>&1; then
    log "GPUs visible on host (rocm-smi):"
    /opt/rocm/bin/rocm-smi --showproductname 2>/dev/null || true
    echo
fi

# ── Create HF cache dir ───────────────────────────────────────────────────────
mkdir -p "${HF_HOME}"

# ── Launch ────────────────────────────────────────────────────────────────────
log "Starting vLLM server ..."
log "Endpoint will be: http://$(hostname -s):${PORT}/v1"
echo

# Key flags:
#   --rocm          fixes cgroup device delegation (needed when inside srun --pty)
#   --bind /opt/rocm  exposes ROCm libraries (libamdhip64.so etc.)
#   --bind SCRATCH    model weights + HF cache land here
#   --env             propagate HF_HOME so the model downloads to scratch
"${SING_CMD}" exec --rocm \
    --bind /opt/rocm \
    --bind "${SCRATCH}" \
    --env HF_HOME="${HF_HOME}" \
    --env TRITON_CACHE_DIR="${SCRATCH}/users/aralikatte/triton_cache" \
    --env ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
    "${SIF}" \
    "${VLLM_ARGS[@]}"
