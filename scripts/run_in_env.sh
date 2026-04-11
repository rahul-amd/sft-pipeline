#!/bin/bash
# Entrypoint for singularity exec: activates the sft-pipeline conda env, then
# runs whatever command is passed as arguments.
#
# Usage (inside Singularity):
#   singularity exec ... scripts/run_in_env.sh ray start --head ...
#   singularity exec ... scripts/run_in_env.sh sft-pipeline run-stage stage1_collect ...

source /share/miniconda3/etc/profile.d/conda.sh
conda activate sft-pipeline

# ── GPU device visibility ─────────────────────────────────────────────────────
# Slurm sets ROCR_VISIBLE_DEVICES to restrict GPU access per task.
# Ray's AMD GPU manager (ray/_private/accelerators/amd_gpu.py) raises RuntimeError
# if ROCR_VISIBLE_DEVICES is set without HIP_VISIBLE_DEVICES.  Translate here so
# both ROCm runtime and Ray agree on which GPUs are visible.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi

# ── ROCm runtime ──────────────────────────────────────────────────────────────
# /opt/rocm is bind-mounted from the host but Singularity does not automatically
# add it to PATH or LD_LIBRARY_PATH.  Without these, torch cannot find
# libamdhip64.so and torch.cuda.is_available() silently returns False.
export PATH="/opt/rocm/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# HSA_OVERRIDE_GFX_VERSION is intentionally NOT set here.
# ROCm 6.3 natively supports gfx90a (MI250X) — the override is unnecessary and
# harmful: forcing 9.0.0 (gfx900/Vega10) loads wrong kernels for gfx90a, causing
# GPU memory access faults (e.g. hipRAND crashes) on the first real kernel launch.

# ── Triton kernel cache ───────────────────────────────────────────────────────
# Default ~/.triton/cache sits on the home filesystem which has a strict quota.
# Redirect to the scratch filesystem so compiled kernels don't fill the quota.
# SCRATCH is set by Slurm; fall back to the known project scratch path.
_SCRATCH="${SCRATCH:-/scratch/project_462000963}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_SCRATCH}/users/aralikatte/triton_cache}"
mkdir -p "${TRITON_CACHE_DIR}"
unset _SCRATCH

# ── pip --user packages ───────────────────────────────────────────────────────
# Packages installed with --user land in ~/.local/bin (CLIs) and
# ~/.local/lib/pythonX.Y/site-packages (imports).  Derive the version from the
# active interpreter so this stays correct across Python upgrades.
_PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PATH="/users/aralikatte/.local/bin${PATH:+:${PATH}}"
export PYTHONPATH="/users/aralikatte/.local/lib/python${_PY_VER}/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
unset _PY_VER

exec "$@"
