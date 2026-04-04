#!/bin/bash
# Entrypoint for singularity exec: activates the sft-pipeline conda env, then
# runs whatever command is passed as arguments.
#
# Usage (inside Singularity):
#   singularity exec ... scripts/run_in_env.sh ray start --head ...
#   singularity exec ... scripts/run_in_env.sh sft-pipeline run-stage stage1_collect ...

source /share/miniconda3/etc/profile.d/conda.sh
conda activate sft-pipeline

# Packages installed via pip --user (ray, sft-pipeline CLIs) land in ~/.local/bin
export PATH="/users/aralikatte/.local/bin:$PATH"
export PYTHONPATH="/users/aralikatte/.local/lib/python3.12/site-packages:$PYTHONPATH"

exec "$@"
