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
#   BIND_HOST    bind address          (default: 0.0.0.0)
#                NOTE: do NOT use HOST — it is a reserved shell variable on
#                Cray/LUMI nodes (set to the node hostname) and will cause
#                "Errno 99: Cannot assign requested address" in the API server.
#   SIF          path to vllm_rocm.sif
#   HF_HOME      HF cache directory    (default: ${SCRATCH}/hf_cache)
#   GPU_MEM_UTIL fraction of GPU memory to use (default: 0.92)
#   MAX_MODEL_LEN max context length   (default: unset → model default)
#   SCRATCH      base scratch path     (default: /scratch/project_462000963)
#   NIC          override auto-detected NIC name (e.g. eth0, hsn0, ib0)
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRATCH="${SCRATCH:-/scratch/project_462000963}"
SINCONS_DIR="${SINCONS_DIR:-${SCRATCH}/users/aralikatte/sincons}"
SIF="${SIF:-${SINCONS_DIR}/vllm_rocm.sif}"

MODEL="${MODEL:-}"
TP="${TP:-8}"
PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"   # NOT HOST — that env var is set by the OS on Cray/LUMI nodes
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
NIC="${NIC:-}"   # auto-detected below if not set

# ── Parse --model / --tensor-parallel-size flags ─────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL="$2"; shift 2 ;;
        --tensor-parallel-size) TP="$2";    shift 2 ;;
        --port)                 PORT="$2";      shift 2 ;;
        --host)                 BIND_HOST="$2"; shift 2 ;;
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
    --host               "${BIND_HOST}"
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
log "Bind host        : ${BIND_HOST}"
log "Port             : ${PORT}"
log "GPU mem util     : ${GPU_MEM_UTIL}"
log "HF_HOME          : ${HF_HOME}"
log "NIC override     : ${NIC:-(auto-detect)}"
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

# ── Network interface detection ───────────────────────────────────────────────
# HPC nodes have multiple NICs (InfiniBand ib0/hsn0, management eth0, etc.).
# Ray's GCS, RCCL, and vLLM's distributed coordinator all need to bind to a
# routable IP.  If they pick the wrong interface the OS returns EADDRNOTAVAIL
# (Errno 99: "Cannot assign requested address") before any GPU work starts.
#
# Strategy: use the interface that owns the default route — it is always the
# one reachable from other compute nodes.  Users can override with NIC=<name>.
if [ -z "${NIC}" ]; then
    # ip route show default → "default via <gw> dev <iface> ..."
    NIC="$(ip route show default 2>/dev/null | awk '/dev/ {print $5; exit}')"
fi

if [ -n "${NIC}" ]; then
    NODE_IP="$(ip -4 addr show dev "${NIC}" 2>/dev/null \
               | awk '/inet / {split($2,a,"/"); print a[1]; exit}')"
else
    NODE_IP=""
fi

if [ -n "${NIC}" ] && [ -n "${NODE_IP}" ]; then
    log "Primary NIC      : ${NIC} (${NODE_IP})"
    log "  → NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME, VLLM_HOST_IP set to ${NIC} / ${NODE_IP}"
else
    log "Warning: could not auto-detect primary NIC — using OS defaults."
    log "  If you see Errno 99, set NIC=<iface> (e.g. NIC=eth0 or NIC=hsn0)."
    NIC=""
    NODE_IP=""
fi
echo

# ── Create HF cache dir ───────────────────────────────────────────────────────
mkdir -p "${HF_HOME}"
mkdir -p "${SCRATCH}/users/aralikatte/triton_cache"

# ── Launch ────────────────────────────────────────────────────────────────────
log "Starting vLLM server ..."
log "Endpoint will be: http://$(hostname -s):${PORT}/v1"
echo

# Key Singularity flags:
#   --rocm          fixes cgroup device delegation (needed when inside srun --pty)
#   --bind /opt/rocm  exposes ROCm libraries (libamdhip64.so etc.)
#   --bind SCRATCH    model weights + HF cache land here
#
# Key env vars passed into the container:
#   VLLM_HOST_IP          — tells vLLM which IP to advertise for worker coordination
#   NCCL_SOCKET_IFNAME    — pins RCCL's NIC so it doesn't bind a non-routable interface
#   GLOO_SOCKET_IFNAME    — same for the Gloo fallback backend
#   RAY_DISABLE_DASHBOARD — stops Ray from binding its dashboard on a random port
#   RAY_GCS_SERVER_ADDRESS — pins Ray's GCS to the same IP as vLLM
"${SING_CMD}" exec --rocm \
    --bind /opt/rocm \
    --bind "${SCRATCH}" \
    --env HF_HOME="${HF_HOME}" \
    --env TRITON_CACHE_DIR="${SCRATCH}/users/aralikatte/triton_cache" \
    --env ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
    --env VLLM_HOST_IP="${NODE_IP}" \
    --env NCCL_SOCKET_IFNAME="${NIC}" \
    --env GLOO_SOCKET_IFNAME="${NIC}" \
    --env RAY_DISABLE_DASHBOARD=1 \
    "${SIF}" \
    "${VLLM_ARGS[@]}"
