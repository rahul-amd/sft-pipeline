#!/bin/bash
# =============================================================================
# One-time environment setup: install GPU-dependent packages into the overlay.
#
# Run this ONCE from a compute node (needs GPUs visible) with the overlay
# mounted read-write.  After this, all Slurm jobs use the overlay read-only.
#
# Usage (from a login or compute node):
#
#   # 1. Get an interactive GPU node
#   srun --account=project_462000963 --partition=standard-g \
#        --nodes=1 --gpus-per-node=1 --time=0:30:00 --pty bash
#
#   # 2. Run this script inside the container with the overlay writable
#   singularity exec \
#       --bind /scratch/project_462000963 \
#       --bind /users/aralikatte \
#       --bind /opt/rocm \
#       --overlay /scratch/project_462000963/users/aralikatte/sincons/python_latest_overlay.img:rw \
#       /scratch/project_462000963/users/aralikatte/sincons/python_latest.sif \
#       bash /scratch/project_462000963/users/aralikatte/sft-pipeline/scripts/setup_env.sh
#
# What this does:
#   1. Installs ROCm PyTorch (replaces any existing CUDA build)
#   2. Installs sentence-transformers
#   3. Installs flash-kmeans + Triton
#   4. Runs smoke tests to verify each component works on the GPU
#
# ROCm version on this cluster: 6.3.4  (verify: cat /opt/rocm/.info/version)
# PyTorch ROCm wheel: rocm6.3 (or rocm6.2 if 6.3 not yet published)
# =============================================================================

set -euo pipefail

ROCM_VERSION="6.3"
TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
ok()  { echo "[$(date '+%H:%M:%S')] ✓ $*"; }
err() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

log "=== SFT Pipeline — environment setup for ROCm ${ROCM_VERSION} ==="
log "Overlay is writable; packages installed to ~/.local via pip --user"
echo

# ── Activate conda env ────────────────────────────────────────────────────────
source /share/miniconda3/etc/profile.d/conda.sh
conda activate sft-pipeline
log "Conda env: $(conda info --base)/envs/sft-pipeline"

# ── Show host ROCm version ────────────────────────────────────────────────────
if [ -f /opt/rocm/.info/version ]; then
    log "Host ROCm version: $(cat /opt/rocm/.info/version)"
else
    log "Warning: /opt/rocm/.info/version not found — check bind mount"
fi
echo

# ── Step 1: ROCm PyTorch ──────────────────────────────────────────────────────
log "Step 1: Installing ROCm PyTorch from ${TORCH_INDEX_URL} ..."
log "(This replaces any existing CUDA build — safe to re-run)"

pip install --user torch torchvision torchaudio \
    --index-url "${TORCH_INDEX_URL}"

# Verify
python - <<'EOF'
import torch
print(f"  torch version  : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device count   : {torch.cuda.device_count()}")
    print(f"  Device[0] name : {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("ERROR: torch.cuda.is_available() is False after install.\n"
                     "Check that /opt/rocm is bound and the wheel URL is correct.")
EOF
ok "ROCm PyTorch OK"
echo

# ── Step 2: sentence-transformers ────────────────────────────────────────────
log "Step 2: Installing / upgrading sentence-transformers ..."
pip install --user "sentence-transformers>=3.0"

python - <<'EOF'
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
# Use a tiny model for the smoke test to avoid a large download
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
vecs = model.encode(["hello world", "smoke test"])
print(f"  sentence-transformers OK — device={device}, embed shape={vecs.shape}")
EOF
ok "sentence-transformers OK"
echo

# ── Step 3: Triton (needed by flash-kmeans) ───────────────────────────────────
log "Step 3: Checking Triton ..."
if python -c "import triton; print(f'  Triton {triton.__version__} already installed')" 2>/dev/null; then
    ok "Triton already present"
else
    log "Installing triton ..."
    pip install --user "triton>=3.0"
    python -c "import triton; print(f'  triton {triton.__version__}')"
    ok "Triton installed"
fi
echo

# ── Step 4: flash-kmeans ──────────────────────────────────────────────────────
log "Step 4: Installing flash-kmeans ..."
pip install --user flash-kmeans

python - <<'EOF'
import torch
from flash_kmeans import batch_kmeans_Euclid

print(f"  flash-kmeans imported OK")
if torch.cuda.is_available():
    x = torch.randn(1, 2000, 64, dtype=torch.float16).cuda()
    ids, centers, _ = batch_kmeans_Euclid(x, n_clusters=10, tol=1e-4, verbose=False)
    print(f"  smoke test: {ids.shape} assignments, {centers.shape} centroids — OK")
else:
    print("  smoke test skipped (no GPU)")
EOF
ok "flash-kmeans OK"
echo

# ── Summary ───────────────────────────────────────────────────────────────────
log "=== Setup complete ==="
log ""
log "All packages are installed in ~/.local (inside the overlay)."
log "The overlay can now be mounted :ro in Slurm jobs."
log ""
log "Next steps:"
log "  sbatch scripts/slurm_stage3.sh config/stage3_cluster.yaml"
