#!/bin/bash
# Entrypoint for singularity exec: activates the sft-pipeline conda env, then
# runs whatever command is passed as arguments.
#
# Usage (inside Singularity):
#   singularity exec ... scripts/run_in_env.sh ray start --head ...
#   singularity exec ... scripts/run_in_env.sh sft-pipeline run-stage stage1_collect ...

source /share/miniconda3/etc/profile.d/conda.sh
conda activate sft-pipeline

# ── ROCm runtime ──────────────────────────────────────────────────────────────
# /opt/rocm is bind-mounted from the host but Singularity does not automatically
# add it to PATH or LD_LIBRARY_PATH.  Without these, torch cannot find
# libamdhip64.so and torch.cuda.is_available() silently returns False.
export PATH="/opt/rocm/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# MI250X ships as gfx90a.  Some PyTorch ROCm builds probe gfx version via HSA;
# setting this explicitly prevents "No supported GPU" errors on that arch.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-9.0.0}"

# ── pip --user packages ───────────────────────────────────────────────────────
# Packages installed with --user land in ~/.local/bin (CLIs) and
# ~/.local/lib/pythonX.Y/site-packages (imports).  Derive the version from the
# active interpreter so this stays correct across Python upgrades.
_PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PATH="/users/aralikatte/.local/bin${PATH:+:${PATH}}"
export PYTHONPATH="/users/aralikatte/.local/lib/python${_PY_VER}/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
unset _PY_VER

exec "$@"
