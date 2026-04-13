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
#
# CRITICAL — match BOTH the ROCm version AND the GFX (GPU) architecture:
#
#   Cluster ROCm version:  cat /opt/rocm/.info/version   → 6.3.4
#   Cluster GPU arch:      /opt/rocm/bin/rocminfo | grep 'Name:.*gfx'
#                          MI250X = gfx90a
#                          MI300X = gfx942
#                          MI350X = gfx950   ← DO NOT use this on MI250X
#
# Find valid tags:
#   curl -s "https://hub.docker.com/v2/repositories/rocm/vllm/tags?page_size=100" \
#        | python3 -c "import sys,json; [print(t['name']) for t in json.load(sys.stdin)['results']]" \
#        | grep gfx90a
#
# For this cluster (ROCm 6.3, MI250X/gfx90a) a known-good tag is:
#   rocm6.3_mi300_ubuntu22.04_py3.11_vllm_v0.8.5.post1
# (despite saying "mi300" AMD ships these with gfx90a kernels for MI250X too)
#
# Browse all tags: https://hub.docker.com/r/rocm/vllm/tags
VLLM_TAG="${VLLM_TAG:-rocm6.3_mi300_ubuntu22.04_py3.11_vllm_v0.8.5.post1}"
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
