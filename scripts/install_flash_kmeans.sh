#!/bin/bash
# =============================================================================
# Install flash-kmeans with ROCm/Triton support inside the Singularity
# container.
#
# Run this ONCE from inside the container with the sft-pipeline conda env
# active (uses pip --user so it writes to ~/.local, not the read-only overlay):
#
#   singularity exec \
#       --bind /scratch/project_462000963 \
#       --bind /users/aralikatte \
#       --overlay /scratch/project_462000963/users/aralikatte/sincons/python_latest_overlay.img:ro \
#       /scratch/project_462000963/users/aralikatte/sincons/python_latest.sif \
#       /scratch/project_462000963/users/aralikatte/sft-pipeline/scripts/run_in_env.sh \
#       bash /scratch/project_462000963/users/aralikatte/sft-pipeline/scripts/install_flash_kmeans.sh
#
# Or equivalently from a LUMI login node:
#   singularity exec ... scripts/run_in_env.sh bash scripts/install_flash_kmeans.sh
#
# =============================================================================

set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== flash-kmeans installation for ROCm ==="

# ── Step 1: Check PyTorch ROCm is available ──────────────────────────────────
log "Checking PyTorch + ROCm ..."
python -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device name     : {torch.cuda.get_device_name(0)}')
    print(f'  Device count    : {torch.cuda.device_count()}')
else:
    print('  WARNING: No CUDA/ROCm device found. flash-kmeans requires a GPU.')
"

# ── Step 2: Check/install Triton ─────────────────────────────────────────────
# On ROCm PyTorch >= 2.1, Triton (AMD/HIP backend) is bundled.
# If not available, install the standalone package which ships ROCm wheels
# for Triton >= 3.x.
log "Checking Triton ..."
if python -c "import triton; print(f'  Triton version: {triton.__version__}')" 2>/dev/null; then
    log "Triton already available."
else
    log "Triton not found — installing standalone triton package ..."
    log "(This provides ROCm/HIP backend for Triton >= 3.x)"
    pip install --user "triton>=3.0"
    python -c "import triton; print(f'  Triton version: {triton.__version__}')"
    log "Triton installed."
fi

# ── Step 3: Install flash-kmeans ─────────────────────────────────────────────
log "Installing flash-kmeans ..."
pip install --user flash-kmeans

# ── Step 4: Verify the install ───────────────────────────────────────────────
log "Verifying flash-kmeans import ..."
python -c "
from flash_kmeans import batch_kmeans_Euclid
import torch, numpy as np

print('  flash-kmeans import: OK')

if torch.cuda.is_available():
    # Quick smoke test: 1000 random points, 10 clusters
    x = torch.randn(1, 1000, 64, dtype=torch.float16).cuda()
    ids, centers, _ = batch_kmeans_Euclid(x, n_clusters=10, tol=1e-4, verbose=False)
    print(f'  Smoke test: {ids.shape} cluster assignments, {centers.shape} centroids — OK')
else:
    print('  Smoke test skipped (no GPU on this node)')
"

log "=== Installation complete ==="
log ""
log "To use flash_kmeans in Stage 3, set in your config:"
log "  stage3_cluster:"
log "    clustering_algorithm: \"flash_kmeans\""
log "    n_clusters: 100000"
