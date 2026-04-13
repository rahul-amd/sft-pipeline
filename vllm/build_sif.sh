#!/bin/bash
# =============================================================================
# Build a Singularity SIF from the official vLLM ROCm Docker image.
#
# Run on a login node or any node with singularity/apptainer and internet access.
# The resulting SIF is self-contained — no overlay needed for vLLM itself.
#
# Usage:
#   bash vllm/build_sif.sh                   # uses defaults
#   VLLM_TAG=v0.8.5 bash vllm/build_sif.sh  # pin a specific version
#
# The build pulls ~15–25 GB from Docker Hub; takes 10–30 min depending on
# network speed.  Run inside a tmux/screen session.
#
# After building, use vllm/serve.sh to start the inference server.
# =============================================================================

set -euo pipefail

# ── Configuration — override via environment variables ────────────────────────

# vLLM ROCm image tag.
# Browse available tags at: https://hub.docker.com/r/rocm/vllm/tags
# Match the tag's ROCm version to your cluster's ROCm (check: cat /opt/rocm/.info/version).
# This cluster: ROCm 6.3.4  →  use a rocm6.3 or rocm6.x tag.
VLLM_TAG="${VLLM_TAG:-rocm7.12.0_gfx950-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0}"
DOCKER_IMAGE="docker://rocm/vllm:${VLLM_TAG}"

# Output paths
SCRATCH="${SCRATCH:-/scratch/project_462000963}"
SINCONS_DIR="${SINCONS_DIR:-${SCRATCH}/users/aralikatte/sincons}"
SIF="${SIF:-${SINCONS_DIR}/vllm_rocm.sif}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
ok()      { echo "[$(date '+%H:%M:%S')] ✓ $*"; }
err()     { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

# ── Detect singularity vs apptainer ──────────────────────────────────────────
if   command -v apptainer  >/dev/null 2>&1; then SING_CMD=apptainer
elif command -v singularity >/dev/null 2>&1; then SING_CMD=singularity
else
    err "Neither 'singularity' nor 'apptainer' found in PATH."
    err "Load the module first:  module load singularity"
    exit 1
fi
log "Using: $(command -v "${SING_CMD}") — $("${SING_CMD}" --version 2>&1 | head -1)"

# ── Print plan ────────────────────────────────────────────────────────────────
echo
log "Docker image : ${DOCKER_IMAGE}"
log "Output SIF   : ${SIF}"
echo

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir -p "${SINCONS_DIR}"

if [ -f "${SIF}" ]; then
    ok "SIF already exists: ${SIF}"
    log "Delete it and re-run to rebuild:  rm ${SIF}"
    exit 0
fi

log "Building SIF (this pulls ~15–25 GB; takes 10–30 min) ..."
log "Tip: run this inside tmux/screen so a disconnect doesn't interrupt it."
echo

# --no-https is not needed for Docker Hub; remove if it causes issues.
# Some clusters require --fakeroot for unprivileged builds.
"${SING_CMD}" build "${SIF}" "${DOCKER_IMAGE}"

ok "SIF created: ${SIF}"
echo
log "SIF size: $(du -sh "${SIF}" | cut -f1)"
echo
log "Next step — start the vLLM server:"
echo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "    bash ${SCRIPT_DIR}/serve.sh --model <HF-model-id>"
echo
log "Or submit a Slurm job with vllm/slurm_serve.sh."
