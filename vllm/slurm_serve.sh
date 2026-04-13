#!/bin/bash
# =============================================================================
# Slurm batch job: run the vLLM ROCm server on a single node.
#
# Submit:
#   sbatch vllm/slurm_serve.sh
#   sbatch --export=MODEL=Qwen/Qwen2.5-72B-Instruct,TP=8 vllm/slurm_serve.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/vllm_serve_<jobid>.log
#
# The server prints its endpoint URL to the log once it is ready:
#   grep "Uvicorn running" logs/vllm_serve_*.log
#
# Interactive GPU check:
#   srun --account=project_462000963 --partition=standard-g \
#        --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=32G --time=0:30:00 \
#        singularity exec \
#            --bind /dev/kfd --bind /dev/dri \
#            --bind /scratch/project_462000963 \
#            /scratch/project_462000963/users/aralikatte/sincons/vllm_rocm.sif \
#            python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#
# NOTE: do NOT use --rocm — the vLLM image has ROCm baked in and --rocm
# injects host libs that need glibc 2.38+ (container is Ubuntu 22.04 / glibc
# 2.35), breaking torch import with "GLIBC_2.38 not found".
# =============================================================================

#SBATCH --job-name=vllm-serve
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g

# ── Resource request ─────────────────────────────────────────────────────────
# Adjust --gpus-per-node to match your TP value and model size.
#   7B  model  →  1–2 GCDs  (--gpus-per-node=2,  TP=2)
#   72B model  →  8  GCDs  (--gpus-per-node=8,  TP=8)
#   122B model → 16  GCDs  (--gpus-per-node=16, TP=16)
# Each MI250X chip has 2 GCDs; a full node has 8 chips = 16 GCDs.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00

#SBATCH --output=logs/vllm_serve_%j.log
#SBATCH --error=logs/vllm_serve_%j.log

set -euo pipefail

mkdir -p logs

# ── Model / parallelism — override at submit time via --export ────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TP="${TP:-8}"
PORT="${PORT:-8000}"

SCRATCH="${SCRATCH:-/scratch/project_462000963}"

echo "========================================================"
echo " vLLM ROCm server"
echo " Job ID     : ${SLURM_JOB_ID}"
echo " Node       : $(hostname)"
echo " Model      : ${MODEL}"
echo " TP         : ${TP}"
echo " Port       : ${PORT}"
echo " Started    : $(date)"
echo "========================================================"
echo

# Print the endpoint so it's easy to grep from the log
echo "Endpoint: http://$(hostname -s):${PORT}/v1"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL}" TP="${TP}" PORT="${PORT}" SCRATCH="${SCRATCH}" \
    bash "${SCRIPT_DIR}/serve.sh"
