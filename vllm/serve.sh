#!/bin/bash
# =============================================================================
# Start the vLLM OpenAI-compatible HTTP server from the ROCm SIF.
#
# Run on a GPU compute node (interactive or via Slurm — see slurm_serve.sh).
#
# Usage:
#   bash vllm/serve.sh --model <HF-model-id>  [--tensor-parallel-size N]
#                                              [--data-parallel-size N]
#
#   # Examples
#   bash vllm/serve.sh --model Qwen/Qwen2.5-7B-Instruct
#   bash vllm/serve.sh --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 8
#   MODEL=Qwen/Qwen3.5-122B-A10B TP=16 bash vllm/serve.sh
#   # MoE with expert parallelism: TP=2, DP=4 → EP=8, 8 GPUs per worker
#   MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507 TP=2 DP=4 bash vllm/serve.sh
#
# Environment overrides:
#   MODEL        HuggingFace model id (required unless --model is passed)
#   TP           tensor-parallel size  (default: 8, one per MI250X GCD pair)
#   DP           data-parallel size    (default: 1, i.e. disabled)
#                MoE models: set DP > 1 to enable expert parallelism.
#                vLLM computes EP = TP × DP automatically.
#                Total GPUs per worker = TP × DP.
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
#   ROCM_COMPAT  1 (default): use --rocm for cgroup device delegation.  Strips
#                /.singularity.d/libs from LD_LIBRARY_PATH to avoid GLIBC_2.38
#                mismatch (host libs injected by --rocm need glibc 2.38+ but the
#                container is Ubuntu 22.04 / glibc 2.35).  Required on LUMI for
#                both interactive srun sessions AND sbatch jobs.
#                0: use --bind /dev/kfd --bind /dev/dri instead.  Only works on
#                clusters where Slurm propagates device cgroups to Singularity.
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRATCH="${SCRATCH:-/scratch/project_462000963}"
SINCONS_DIR="${SINCONS_DIR:-${SCRATCH}/users/aralikatte/sincons}"
SIF="${SIF:-${SINCONS_DIR}/vllm_rocm.sif}"

MODEL="${MODEL:-}"
TP="${TP:-8}"
DP="${DP:-1}"
PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"   # NOT HOST — that env var is set by the OS on Cray/LUMI nodes
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
NIC="${NIC:-}"   # auto-detected below if not set
ROCM_COMPAT="${ROCM_COMPAT:-1}"

# ── Parse --model / --tensor-parallel-size flags ─────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL="$2";         shift 2 ;;
        --tensor-parallel-size) TP="$2";            shift 2 ;;
        --data-parallel-size)   DP="$2";            shift 2 ;;
        --max-model-len)        MAX_MODEL_LEN="$2"; shift 2 ;;
        --port)                 PORT="$2";          shift 2 ;;
        --host)                 BIND_HOST="$2";     shift 2 ;;
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

if [ "${DP}" -gt 1 ]; then
    VLLM_ARGS+=(--data-parallel-size "${DP}" --enable-expert-parallel)
fi

if [ -n "${MAX_MODEL_LEN}" ]; then
    VLLM_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

# ── Print plan ────────────────────────────────────────────────────────────────
echo
log "SIF              : ${SIF}"
log "Model            : ${MODEL}"
log "Tensor parallel  : ${TP}"
log "Data parallel    : ${DP}$( [ "${DP}" -gt 1 ] && echo " (EP = TP × DP = $(( TP * DP )))" || echo " (EP disabled)" )"
log "Max seq len      : ${MAX_MODEL_LEN:-(model default)}"
log "Bind host        : ${BIND_HOST}"
log "Port             : ${PORT}"
log "GPU mem util     : ${GPU_MEM_UTIL}"
log "HF_HOME          : ${HF_HOME}"
log "NIC override     : ${NIC:-(auto-detect)}"
echo

# ── Verify GPU is visible (advisory only — cgroups can make stat() return EPERM
#    even when the device is accessible via the container) ─────────────────────
if [ ! -e /dev/kfd ]; then
    log "Warning: /dev/kfd not found via stat() — may be a cgroup device restriction."
    log "  If the container fails to see the GPU, request a GPU allocation first:"
    log "  srun --account=project_462000963 --partition=standard-g --gpus-per-node=${TP} --pty bash"
fi

# ── GFX architecture guard ────────────────────────────────────────────────────
# hipErrorInvalidImage means the SIF kernels were compiled for a different GPU
# arch. Detect the host arch from /opt/rocm (always present on the host) and
# check it against the explicit gfxNNN in the image tag if present.
# The check only fires when the tag contains a literal "gfxNNN" string that
# differs from the host — tags like "mi300" (no explicit gfx) pass through.
HOST_GFX=""
if [ -x /opt/rocm/bin/rocminfo ]; then
    HOST_GFX="$(/opt/rocm/bin/rocminfo 2>/dev/null \
                 | awk '/^\s+Name:/ && /gfx/ {gsub(/^[[:space:]]+Name:[[:space:]]+/,""); print $0; exit}')" \
              || true
fi
log "Host GPU arch    : ${HOST_GFX:-unknown (rocminfo not found at /opt/rocm)}"

SIF_TAG="${VLLM_TAG:-}"
if [ -n "${HOST_GFX}" ] && echo "${SIF_TAG}" | grep -q 'gfx'; then
    TAG_GFX="$(echo "${SIF_TAG}" | grep -o 'gfx[0-9a-z]*' | head -1)"
    if [ "${TAG_GFX}" != "${HOST_GFX}" ]; then
        err "──────────────────────────────────────────────────────────────────"
        err " GFX ARCHITECTURE MISMATCH — this will cause hipErrorInvalidImage"
        err "──────────────────────────────────────────────────────────────────"
        err " Host GPU   : ${HOST_GFX}"
        err " SIF tag    : ${TAG_GFX}  (from VLLM_TAG=${SIF_TAG})"
        err ""
        err " Rebuild the SIF with a tag that matches ${HOST_GFX}:"
        err "   VLLM_TAG=<tag_containing_${HOST_GFX}> bash vllm/build_sif.sh"
        err ""
        err " Browse tags: https://hub.docker.com/r/rocm/vllm/tags"
        exit 1
    fi
    log "GFX arch check   : ✓ tag (${TAG_GFX}) matches host (${HOST_GFX})"
fi
echo

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
    NIC="$(ip route show default 2>/dev/null | awk '/dev/ {print $5; exit}')" || true
fi

if [ -n "${NIC}" ]; then
    NODE_IP="$(ip -4 addr show dev "${NIC}" 2>/dev/null \
               | awk '/inet / {split($2,a,"/"); print a[1]; exit}')" || true
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

# Singularity flags:
#   --bind /dev/kfd   AMD KFD compute device (required for torch.cuda)
#   --bind /dev/dri   DRI render devices     (required for ROCm)
#   --bind SCRATCH    model weights + HF cache
#
# Two modes (controlled by ROCM_COMPAT):
#
# ROCM_COMPAT=1 (default) — required on LUMI for both interactive and sbatch:
#   --rocm handles AMD KFD/DRI device delegation regardless of cgroup context.
#   --rocm also injects host ROCm libs into /.singularity.d/libs/; those libs
#   need glibc 2.38+ but the vLLM container is Ubuntu 22.04 (glibc 2.35), so
#   we strip /.singularity.d/ from LD_LIBRARY_PATH via a bash wrapper before
#   exec-ing the actual vLLM server.  The container's own ROCm copy is used.
#
# ROCM_COMPAT=0 — only for clusters where Slurm propagates device cgroups
#   to Singularity natively.  Uses --bind /dev/kfd --bind /dev/dri; no lib
#   injection.  Does NOT work on LUMI (cgroups v2 blocks /dev/kfd open()).
#
# Key env vars forwarded to container:
#   VLLM_HOST_IP          — which IP vLLM advertises for worker coordination
#   NCCL_SOCKET_IFNAME    — pins RCCL to the right NIC
#   GLOO_SOCKET_IFNAME    — same for Gloo fallback
#   RAY_DISABLE_DASHBOARD — stops Ray binding a random dashboard port

# Build the env-var args shared by both modes
SING_ENV_ARGS=(
    --env HF_HOME="${HF_HOME}"
    --env TRITON_CACHE_DIR="${SCRATCH}/users/aralikatte/triton_cache"
    --env ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"
    --env VLLM_HOST_IP="${NODE_IP}"
    --env NCCL_SOCKET_IFNAME="${NIC}"
    --env GLOO_SOCKET_IFNAME="${NIC}"
    --env RAY_DISABLE_DASHBOARD=1
)

if [ "${ROCM_COMPAT}" = "1" ]; then
    log "Mode: ROCM_COMPAT=1 (--rocm + LD_LIBRARY_PATH strip)"
    # Wrap the vLLM command in a bash one-liner that strips /.singularity.d/libs
    # from LD_LIBRARY_PATH before exec-ing vLLM (avoids GLIBC_2.38 mismatch).
    VLLM_CMD_STR="${VLLM_ARGS[*]}"
    "${SING_CMD}" exec \
        --rocm \
        --bind "${SCRATCH}" \
        "${SING_ENV_ARGS[@]}" \
        "${SIF}" \
        bash -c '
            export LD_LIBRARY_PATH="$(
                printf "%s" "${LD_LIBRARY_PATH:-}" \
                | tr ":" "\n" | grep -v "^/.singularity.d" \
                | tr "\n" ":" | sed "s/:$//"
            )"
            exec '"${VLLM_CMD_STR}"'
        '
else
    log "Mode: batch (--bind /dev/kfd /dev/dri)"
    "${SING_CMD}" exec \
        --bind /dev/kfd \
        --bind /dev/dri \
        --bind "${SCRATCH}" \
        "${SING_ENV_ARGS[@]}" \
        "${SIF}" \
        "${VLLM_ARGS[@]}"
fi
