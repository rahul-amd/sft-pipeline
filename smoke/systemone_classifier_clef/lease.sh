#!/usr/bin/env bash
set -euo pipefail
cd /shared_silo/scratch/rahul.aralikatte@amd.com/sft-pipeline
quest_dir="$PWD/smoke/systemone_classifier_clef"
stop_epoch=$(date -u -d '2026-10-28 00:28:30 UTC' +%s)
remaining=$((stop_epoch-$(date -u +%s)))
((remaining>60)) || exit 2
cleanup() {
  set +e
  mapfile -t owned < <(docker ps -aq --filter "label=clef27.job=$SLURM_JOB_ID")
  if ((${#owned[@]})); then
    timeout 20 docker stop --timeout 8 "${owned[@]}" >> "$quest_dir/logs/lease-cleanup-$SLURM_JOB_ID.log" 2>&1
    docker rm -f "${owned[@]}" >> "$quest_dir/logs/lease-cleanup-$SLURM_JOB_ID.log" 2>&1
  fi
}
(
  sleep "$remaining"
  cleanup
  scancel "$SLURM_JOB_ID"
) &
watchdog=$!
trap 'exit 143' TERM INT
trap 'cleanup; kill "$watchdog" 2>/dev/null || true' EXIT
while [[ ! -f "$quest_dir/runner-ready" ]]; do sleep 5; done
bash "$quest_dir/run.sh"
