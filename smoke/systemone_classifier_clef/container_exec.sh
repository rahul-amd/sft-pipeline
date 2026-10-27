#!/usr/bin/env bash
set -euo pipefail
quest_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$quest_dir/../.." && pwd)"
image="vllm/vllm-openai-rocm:nightly-rocm72-5d35ce3934a7804b298559d2d99eec33792e1eb9"
case "${SLURM_JOB_PARTITION:-}" in
  amd-tw-backlog|amd-tw-verification) ;;
  hissu)
    cutoff=$(date -u -d '2026-10-28 00:28:30 UTC' +%s)
    (( $(date -u +%s) < cutoff )) || { echo 'Hissu authorization has expired' >&2; exit 2; }
    ;;
  *) echo "Refusing GPU execution outside approved non-hissu partitions" >&2; exit 2 ;;
esac
allocated_devices="${SLURM_STEP_GPUS:?Run within the approved Slurm allocation}"
gpu_devices="${CLEF_CARD_INDEX:-$allocated_devices}"
IFS=',' read -r -a selected_cards <<< "$gpu_devices"
for selected in "${selected_cards[@]}"; do
  [[ ",$allocated_devices," == *",$selected,"* ]] || { echo 'Selected GPU is outside this allocation' >&2; exit 2; }
done
# This cluster assigns GRES via renderD[128,136,144,152,160,168,176,184].
# Resolve that device file to a UUID; DRM card and ROCr ordinal orders differ.
IFS=',' read -r -a allocated_cards <<< "$gpu_devices"
gpu_uuids=()
for card in "${allocated_cards[@]}"; do
  render_node=$((128 + 8 * card))
  gpu_uid="$(cat "/sys/class/drm/renderD${render_node}/device/unique_id")"
  gpu_uuids+=("GPU-${gpu_uid#0x}")
done
gpu_visibility="$(IFS=,; echo "${gpu_uuids[*]}")"
extra_args=(--label "clef27.job=$SLURM_JOB_ID")
if [[ "${CLEF_KEEP_CONTAINER:-0}" != 1 ]]; then
  extra_args+=(--rm)
fi
if [[ "${CLEF_DETACH:-0}" == 1 ]]; then
  extra_args+=(--detach)
fi
if [[ -n "${CLEF_CONTAINER_NAME:-}" ]]; then
  extra_args+=(--name "$CLEF_CONTAINER_NAME")
fi
exec docker run "${extra_args[@]}" --user "$(id -u):$(id -g)" \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --shm-size 16g --network host \
  -v "$repo_dir:/repo:ro" -v "$quest_dir:/work" \
  -e "ROCR_VISIBLE_DEVICES=$gpu_visibility" \
  -e HF_HOME=/work/cache/hf -e HF_XET_CACHE=/work/cache/xet -e XDG_CACHE_HOME=/work/cache/xdg \
  -e VLLM_CACHE_ROOT=/work/cache/vllm -e TRITON_CACHE_DIR=/work/cache/triton \
  -e AITER_JIT_DIR=/work/cache/aiter -e TVM_FFI_CACHE_DIR=/work/cache/tvm \
  -e AITER_ROOT_DIR=/work/cache/aiter-root \
  -e NUMBA_CACHE_DIR=/work/cache/numba \
  -e TMPDIR=/work/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_NO_USAGE_STATS=1 \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  --entrypoint python "$image" "$@"
