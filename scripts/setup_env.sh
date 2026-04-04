#!/bin/bash
# =============================================================================
# Complete one-time environment setup for the SFT pipeline.
#
# Without --initial-setup (default — GPU packages only):
#   Re-installs ROCm PyTorch, sentence-transformers, Triton, and flash-kmeans
#   into the overlay.  Run INSIDE the container on a GPU compute node with
#   /opt/rocm, /dev/kfd, and /dev/dri all bound in.
#
#   singularity exec \
#       --bind /scratch/project_462000963 --bind /users/aralikatte \
#       --bind /opt/rocm --bind /dev/kfd --bind /dev/dri \
#       --overlay /scratch/.../python_latest_overlay.img:rw \
#       /scratch/.../python_latest.sif \
#       bash /scratch/.../sft-pipeline/scripts/setup_env.sh
#
# With --initial-setup (full first-time bootstrap):
#   The script auto-detects its execution context and runs the matching phase:
#
#   Phase 1 — HOST (outside the container, login or compute node):
#     Converts the Docker image to a SIF and creates the Singularity overlay.
#     Prints the singularity exec command to run for Phase 2.
#
#       bash /path/to/sft-pipeline/scripts/setup_env.sh --initial-setup
#
#   Phase 2 — CONTAINER, no GPU needed (overlay :rw, no /dev/kfd):
#     Creates /share, downloads Miniconda, creates the sft-pipeline conda env,
#     and installs all base Python packages from pyproject.toml.
#     Falls through automatically to Phase 3 if a GPU is also present.
#
#       singularity exec \
#           --bind /scratch/project_462000963 --bind /users/aralikatte \
#           --overlay /scratch/.../python_latest_overlay.img:rw \
#           /scratch/.../python_latest.sif \
#           bash /scratch/.../sft-pipeline/scripts/setup_env.sh --initial-setup
#
#   Phase 3 — CONTAINER, GPU required (overlay :rw, /dev/kfd + /dev/dri bound):
#     Installs ROCm PyTorch, sentence-transformers, Triton, and flash-kmeans.
#     Runs automatically after Phase 2 if GPU binds are present; otherwise
#     run separately (without --initial-setup) on a GPU compute node.
#
# Three binds are required for ROCm GPU access inside Singularity:
#   --bind /opt/rocm   ROCm runtime libraries (libamdhip64.so, etc.)
#   --bind /dev/kfd    AMD KFD kernel driver — torch.cuda.is_available() needs this
#   --bind /dev/dri    DRI render devices (renderD128, etc.)
#
# ROCm version on this cluster: 6.3.4  (verify: cat /opt/rocm/.info/version)
# =============================================================================

set -euo pipefail

# ── Configuration — override via environment variables ────────────────────────

# Base Docker image for the SIF.  Use a plain Python image; GPU libraries come
# from --bind /opt/rocm at runtime, not baked into the SIF.
DOCKER_IMAGE="${DOCKER_IMAGE:-docker://python:3.11-slim}"

# Cluster paths
SCRATCH="${SCRATCH:-/scratch/project_462000963}"
USER_DIR="${USER_DIR:-/users/aralikatte}"
SINCONS_DIR="${SINCONS_DIR:-${SCRATCH}/users/aralikatte/sincons}"
SIF="${SIF:-${SINCONS_DIR}/python_latest.sif}"
OVERLAY="${OVERLAY:-${SINCONS_DIR}/python_latest_overlay.img}"
OVERLAY_SIZE_MB="${OVERLAY_SIZE_MB:-30720}"   # 30 GB

# Conda / Python
CONDA_ROOT="${CONDA_ROOT:-/share/miniconda3}"
CONDA_ENV="sft-pipeline"
PYTHON_VERSION="3.11"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# ROCm / PyTorch
ROCM_VERSION="6.3"
TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

# Project root — derived from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
ok()      { echo "[$(date '+%H:%M:%S')] ✓ $*"; }
err()     { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }
section() {
    echo
    echo "══════════════════════════════════════════════════════════════"
    log "$*"
    echo "══════════════════════════════════════════════════════════════"
}

# Returns 0 (true) when running inside a Singularity / Apptainer container.
_in_container() {
    [ -n "${SINGULARITY_NAME:-}" ] || [ -n "${APPTAINER_NAME:-}" ]
}

# Detect singularity vs apptainer
if   command -v apptainer  >/dev/null 2>&1; then SING_CMD=apptainer
elif command -v singularity >/dev/null 2>&1; then SING_CMD=singularity
else SING_CMD=""
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
INITIAL_SETUP=false
for arg in "$@"; do
    case "$arg" in
        --initial-setup) INITIAL_SETUP=true ;;
        --help|-h)
            grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
            exit 0
            ;;
        *) err "Unknown argument: $arg"; exit 1 ;;
    esac
done

log "=== SFT Pipeline — environment setup ==="
log "  --initial-setup : ${INITIAL_SETUP}"
log "  in container    : $( _in_container && echo yes || echo no )"
log "  GPU (/dev/kfd)  : $( [ -e /dev/kfd ] && echo found || echo absent )"
log "  project dir     : ${PROJECT_DIR}"
echo

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — HOST
# Build the SIF from Docker and create the Singularity overlay image.
# Runs outside the container.  Skip if already done (idempotent).
# ══════════════════════════════════════════════════════════════════════════════
if $INITIAL_SETUP && ! _in_container; then

    section "Phase 1 — Host setup (SIF + overlay)"

    if [ -z "${SING_CMD}" ]; then
        err "Neither 'singularity' nor 'apptainer' found in PATH."
        err "Load the module first, e.g.:  module load singularity"
        exit 1
    fi
    log "Singularity/Apptainer: $(command -v "${SING_CMD}") — $("${SING_CMD}" --version 2>&1 | head -1)"
    echo

    mkdir -p "${SINCONS_DIR}"
    log "Container directory: ${SINCONS_DIR}"
    echo

    # ── Build SIF ─────────────────────────────────────────────────────────────
    if [ -f "${SIF}" ]; then
        ok "SIF already exists: ${SIF} — skipping build"
        log "(Delete and re-run to force rebuild)"
    else
        log "Building SIF from ${DOCKER_IMAGE} ..."
        log "This pulls ~500 MB–1 GB from Docker Hub; takes a few minutes."
        log "Note: some clusters require --fakeroot for unprivileged builds."
        "${SING_CMD}" build "${SIF}" "${DOCKER_IMAGE}"
        ok "SIF created: ${SIF}"
    fi
    echo

    # ── Create overlay ────────────────────────────────────────────────────────
    if [ -f "${OVERLAY}" ]; then
        ok "Overlay already exists: ${OVERLAY} — skipping creation"
        log "(Delete and re-run to force recreation)"
    else
        log "Creating ${OVERLAY_SIZE_MB} MB overlay at ${OVERLAY} ..."
        log "(${OVERLAY_SIZE_MB} MB = $((OVERLAY_SIZE_MB / 1024)) GB — holds Miniconda + conda env + pip packages)"
        "${SING_CMD}" overlay create --size "${OVERLAY_SIZE_MB}" "${OVERLAY}"
        ok "Overlay created: ${OVERLAY}"
    fi
    echo

    ok "Phase 1 complete."
    echo
    log "Next — run Phase 2 inside the container (no GPU needed):"
    echo
    echo "    ${SING_CMD} exec \\"
    echo "        --bind ${SCRATCH} \\"
    echo "        --bind ${USER_DIR} \\"
    echo "        --overlay ${OVERLAY}:rw \\"
    echo "        ${SIF} \\"
    echo "        bash ${PROJECT_DIR}/scripts/setup_env.sh --initial-setup"
    echo
    log "If you are already on a GPU compute node, add the GPU binds to run"
    log "Phases 2 and 3 back-to-back:"
    echo
    echo "    ${SING_CMD} exec \\"
    echo "        --bind ${SCRATCH} \\"
    echo "        --bind ${USER_DIR} \\"
    echo "        --bind /opt/rocm --bind /dev/kfd --bind /dev/dri \\"
    echo "        --overlay ${OVERLAY}:rw \\"
    echo "        ${SIF} \\"
    echo "        bash ${PROJECT_DIR}/scripts/setup_env.sh --initial-setup"
    echo
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CONTAINER, NO GPU REQUIRED
# Create /share, install Miniconda, create conda env, install base packages.
# Runs inside the container with overlay :rw.  GPU binds are not needed here.
# ══════════════════════════════════════════════════════════════════════════════
if $INITIAL_SETUP && _in_container; then

    section "Phase 2 — Base environment (Miniconda + conda env + base packages)"

    # ── Create /share ─────────────────────────────────────────────────────────
    log "Creating $(dirname "${CONDA_ROOT}") ..."
    mkdir -p "$(dirname "${CONDA_ROOT}")"
    ok "$(dirname "${CONDA_ROOT}") ready"
    echo

    # ── Download and install Miniconda ────────────────────────────────────────
    if [ -d "${CONDA_ROOT}" ]; then
        ok "Miniconda already installed at ${CONDA_ROOT} — skipping download"
    else
        log "Downloading Miniconda installer ..."
        curl -fsSL "${MINICONDA_URL}" -o /tmp/miniconda_installer.sh
        log "Installing Miniconda to ${CONDA_ROOT} (batch, non-interactive) ..."
        bash /tmp/miniconda_installer.sh -b -p "${CONDA_ROOT}"
        rm -f /tmp/miniconda_installer.sh
        ok "Miniconda installed: ${CONDA_ROOT}"
    fi
    echo

    # ── Initialise conda ──────────────────────────────────────────────────────
    # shellcheck disable=SC1090
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    log "Conda: $(conda --version)"
    echo

    # ── Create sft-pipeline conda env ────────────────────────────────────────
    if [ -d "${CONDA_ROOT}/envs/${CONDA_ENV}" ]; then
        ok "Conda env '${CONDA_ENV}' already exists — skipping creation"
    else
        log "Creating conda env '${CONDA_ENV}' (Python ${PYTHON_VERSION}) ..."
        conda create -n "${CONDA_ENV}" python="${PYTHON_VERSION}" -y
        ok "Conda env '${CONDA_ENV}' created"
    fi
    conda activate "${CONDA_ENV}"
    log "Active env: $(conda info --base)/envs/${CONDA_ENV}"
    echo

    # ── Install base Python packages ──────────────────────────────────────────
    section "Phase 2 — pip installs"

    log "Upgrading pip / setuptools / wheel ..."
    pip install --user --upgrade pip setuptools wheel
    echo

    # Install the sft-pipeline package and all its declared dependencies.
    # pyproject.toml is the source of truth; requirements.txt (if present) is
    # used for any additional pinned overrides.
    if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
        log "Found requirements.txt — installing pinned overrides ..."
        pip install --user -r "${PROJECT_DIR}/requirements.txt"
        ok "requirements.txt done"
        echo
    fi

    if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
        log "Installing sft-pipeline (editable) and all declared dependencies ..."
        # --no-build-isolation ensures the already-installed build tools are reused
        pip install --user -e "${PROJECT_DIR}"
        ok "sft-pipeline installed (editable mode)"
    else
        log "Warning: no pyproject.toml found at ${PROJECT_DIR} — skipping package install"
    fi
    echo

    ok "Phase 2 complete."
    echo

    # ── Decide whether to continue to Phase 3 ────────────────────────────────
    if [ ! -e /dev/kfd ]; then
        log "No GPU detected (/dev/kfd absent) — Phase 3 (GPU packages) will run separately."
        echo
        log "To complete setup, get a GPU compute node and re-run without --initial-setup:"
        echo
        log "  Step 1 — Get a GPU compute node:"
        echo
        echo "    srun --account=project_462000963 --partition=standard-g \\"
        echo "         --nodes=1 --ntasks=1 --cpus-per-task=8 \\"
        echo "         --gpus-per-node=1 --mem=32G --time=0:30:00 --pty bash"
        echo
        log "  Step 2 — Verify GPUs on the host:"
        echo
        echo "    /opt/rocm/bin/rocm-smi"
        echo
        log "  Step 3 — Run GPU setup inside the container:"
        echo
        echo "    singularity exec \\"
        echo "        --bind ${SCRATCH} \\"
        echo "        --bind ${USER_DIR} \\"
        echo "        --bind /opt/rocm --bind /dev/kfd --bind /dev/dri \\"
        echo "        --overlay ${OVERLAY}:rw \\"
        echo "        ${SIF} \\"
        echo "        bash ${PROJECT_DIR}/scripts/setup_env.sh"
        echo
        exit 0
    fi

    log "/dev/kfd found — GPU is available, continuing directly to Phase 3"
    echo
    # Falls through into Phase 3 below.
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CONTAINER + GPU REQUIRED
# Install ROCm PyTorch, sentence-transformers, Triton, and flash-kmeans.
# Requires: inside container, /dev/kfd + /dev/dri bound, overlay :rw.
# ══════════════════════════════════════════════════════════════════════════════

# ── Guard: must be inside the container ──────────────────────────────────────
if ! _in_container; then
    err "Phase 3 must be run inside a Singularity container."
    err "Run with --initial-setup on the host first to build the SIF and overlay."
    exit 1
fi

# ── Guard: must be on a GPU compute node ─────────────────────────────────────
# /dev/kfd is the AMD KFD device.  It only exists on nodes where GPUs are
# physically present AND allocated to the current job.  Absent on login nodes
# or compute nodes where --gpus-per-node was not specified.
if [ ! -e /dev/kfd ]; then
    err "=== WRONG NODE ==="
    err "/dev/kfd not found — this node has no GPU allocated."
    err ""
    err "Phase 3 (GPU package installs) must run inside a Singularity container"
    err "on a GPU compute node.  Steps:"
    err ""
    err "  1. Get a compute node with 1 GPU:"
    err "     srun --account=project_462000963 --partition=standard-g \\"
    err "          --nodes=1 --ntasks=1 --cpus-per-task=8 \\"
    err "          --gpus-per-node=1 --mem=32G --time=0:30:00 --pty bash"
    err ""
    err "  2. From that shell, verify GPUs are visible on the host:"
    err "     /opt/rocm/bin/rocm-smi"
    err ""
    err "  3. Then run this script inside the container:"
    err "     singularity exec \\"
    err "         --bind /scratch/project_462000963 \\"
    err "         --bind /users/aralikatte \\"
    err "         --bind /opt/rocm --bind /dev/kfd --bind /dev/dri \\"
    err "         --overlay .../python_latest_overlay.img:rw \\"
    err "         .../python_latest.sif \\"
    err "         bash .../sft-pipeline/scripts/setup_env.sh"
    exit 1
fi
ok "/dev/kfd found — running on a GPU compute node"
echo

# ── ROCm runtime paths ────────────────────────────────────────────────────────
# Singularity bind-mounts /opt/rocm but does NOT add it to PATH or
# LD_LIBRARY_PATH.  Set them now so rocm-smi works and torch can load
# libamdhip64.so (without this, torch.cuda.is_available() returns False).
export PATH="/opt/rocm/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# MI250X (gfx90a) — prevents "No supported GPU" from the HSA runtime if
# the PyTorch wheel targets a slightly different gfx revision.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.0.0}"

log "PATH prefix      : /opt/rocm/bin"
log "LD_LIBRARY_PATH  : /opt/rocm/lib (prepended)"
log "HSA_OVERRIDE_GFX : ${HSA_OVERRIDE_GFX_VERSION}"
echo

# ── Activate conda env ────────────────────────────────────────────────────────
# shellcheck disable=SC1090
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
log "Conda env : $(conda info --base)/envs/${CONDA_ENV}"

# ── Show host ROCm version ────────────────────────────────────────────────────
if [ -f /opt/rocm/.info/version ]; then
    log "Host ROCm : $(cat /opt/rocm/.info/version)"
else
    log "Warning: /opt/rocm/.info/version not found — check --bind /opt/rocm"
fi
echo

# ── Verify ROCm libraries are actually loadable ───────────────────────────────
log "Checking libamdhip64.so is reachable ..."
python3 - <<'PYEOF'
import ctypes, sys
try:
    ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    print("  libamdhip64.so : OK")
except OSError as e:
    print(f"  libamdhip64.so : FAILED — {e}")
    print()
    print("  /opt/rocm/lib is either not bound or not in LD_LIBRARY_PATH.")
    print("  Re-run singularity exec with  --bind /opt/rocm  and make sure")
    print("  LD_LIBRARY_PATH=/opt/rocm/lib is set before running this script.")
    sys.exit(1)
PYEOF
echo

section "Phase 3 — GPU packages (ROCm ${ROCM_VERSION})"

# ── Step 1: ROCm PyTorch ──────────────────────────────────────────────────────
log "Step 1: Installing ROCm PyTorch from ${TORCH_INDEX_URL} ..."
log "(Replaces any existing CUDA build — safe to re-run)"

pip install --user torch torchvision torchaudio \
    --index-url "${TORCH_INDEX_URL}"

# Verify wheel, then check GPU device access separately so we can give a
# precise error message distinguishing "wrong wheel" from "missing bind".
python - <<'EOF'
import torch
ver   = torch.__version__
avail = torch.cuda.is_available()
print(f"  torch version  : {ver}")
print(f"  CUDA available : {avail}")
if "+rocm" not in ver:
    raise SystemExit(
        f"ERROR: torch version {ver!r} does not contain '+rocm'.\n"
        "The CUDA wheel was installed instead of the ROCm wheel.\n"
        "Check that --index-url points to the correct ROCm index."
    )
if avail:
    print(f"  Device count   : {torch.cuda.device_count()}")
    print(f"  Device[0] name : {torch.cuda.get_device_name(0)}")
else:
    print()
    print("  NOTE: torch.cuda.is_available() is False.")
    print("  The ROCm wheel is correctly installed but /dev/kfd or /dev/dri")
    print("  is not bound into the container.  Re-run singularity exec with:")
    print("    --bind /dev/kfd --bind /dev/dri")
    print("  The pip installs below will still complete successfully.")
EOF
ok "ROCm PyTorch installed"
echo

# ── Step 2: sentence-transformers ────────────────────────────────────────────
log "Step 2: Installing / upgrading sentence-transformers ..."
pip install --user "sentence-transformers>=3.0"

python - <<'EOF'
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
# Tiny model for smoke test — avoids a large model download
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
vecs  = model.encode(["hello world", "smoke test"])
print(f"  sentence-transformers OK — device={device}, embed shape={vecs.shape}")
EOF
ok "sentence-transformers OK"
echo

# ── Step 3: Triton (required by flash-kmeans) ─────────────────────────────────
log "Step 3: Checking Triton ..."
if python -c "import triton; print(f'  Triton {triton.__version__} already installed')" 2>/dev/null; then
    ok "Triton already present"
else
    log "Triton not found — installing ..."
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

print("  flash-kmeans imported OK")
if torch.cuda.is_available():
    x = torch.randn(1, 2000, 64, dtype=torch.float16).cuda()
    ids, centers, _ = batch_kmeans_Euclid(x, n_clusters=10, tol=1e-4, verbose=False)
    print(f"  smoke test: {ids.shape} assignments, {centers.shape} centroids — OK")
else:
    print("  smoke test skipped (no GPU visible)")
EOF
ok "flash-kmeans OK"
echo

# ── Summary ───────────────────────────────────────────────════════════════════
section "Setup complete"
log ""
log "All packages are installed in the conda env $(CONDA_ENV) (persisted inside the overlay)."
log "The overlay can now be mounted :ro for all Slurm jobs."
