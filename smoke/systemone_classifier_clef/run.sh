#!/usr/bin/env bash
set -euo pipefail
quest_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$quest_dir/../.." && pwd)"
cd "$repo_dir"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=/shared_silo/scratch/rahul.aralikatte@amd.com/miniconda3/bin/python
image=vllm/vllm-openai-rocm:nightly-rocm72-5d35ce3934a7804b298559d2d99eec33792e1eb9
job=${SLURM_JOB_ID:?}
case "${SLURM_JOB_PARTITION:-}" in
 hissu) [[ $(date -u +%s) -lt $(date -u -d '2026-10-28 00:28:30 UTC' +%s) ]] || exit 2 ;;
 amd-tw-backlog|amd-tw-verification) ;;
 *) exit 2 ;;
esac
mkdir -p "$quest_dir/logs" "$quest_dir/tmp"
cleanup() {
 set +e
 mapfile -t own < <(docker ps -aq --filter "label=clef27.job=$job")
 for container in "${own[@]}"; do
   name=$(docker inspect "$container" --format '{{.Name}}' 2>/dev/null); name=${name#/}
   docker logs "$container" > "$quest_dir/logs/${name:-$container}.log" 2>&1
 done
 if ((${#own[@]})); then docker stop --timeout 8 "${own[@]}" >/dev/null 2>&1; docker rm -f "${own[@]}" >/dev/null 2>&1; fi
}
trap 'exit 143' TERM INT
trap cleanup EXIT
docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image" > "$quest_dir/logs/image-pull-$job.log" 2>&1
if [[ ! -f "$quest_dir/model-download.json" ]]; then "$python_bin" "$quest_dir/download_model.py" > "$quest_dir/logs/model-download-$job.log" 2>&1; fi
gpu_count=$(tr ',' '\n' <<< "${SLURM_JOB_GPUS:?}" | wc -l)
step=(srun --overlap --ntasks=1 --cpus-per-task=2 --gres="gpu:$gpu_count")
if [[ ! -f "$quest_dir/experiment-manifest.json" ]]; then
 "${step[@]}" env CLEF_CARD_INDEX="${SLURM_JOB_GPUS%%,*}" bash "$quest_dir/container_exec.sh" /work/prepare.py > "$quest_dir/logs/prepare-$job.log" 2>&1
fi
"$python_bin" "$quest_dir/shards.py" assign --count "$gpu_count"
IFS=',' read -r -a cards <<< "$SLURM_JOB_GPUS"
for ((i=0;i<gpu_count;i++)); do
 name="clef27-$job-$i"; port=$((18972+i))
 "${step[@]}" env CLEF_CARD_INDEX="${cards[$i]}" CLEF_PORT="$port" CLEF_CONTAINER_NAME="$name" CLEF_DETACH=1 CLEF_KEEP_CONTAINER=1 \
   bash "$quest_dir/serve.sh" > "$quest_dir/logs/server-launch-$i.log" 2>&1
done
"$python_bin" - "$quest_dir" "$gpu_count" <<'PY'
import sys,time,json,requests,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
r=Path(sys.argv[1]);count=int(sys.argv[2]);s=requests.Session();s.trust_env=False
pending=set(range(count));end=time.monotonic()+1200
while pending and time.monotonic()<end:
 for i in list(pending):
  state=subprocess.check_output(['docker','inspect',f'clef27-{os.environ["SLURM_JOB_ID"]}-{i}','--format','{{.State.Running}}'],text=True).strip()
  if state!='true':raise SystemExit(f'Server {i} exited before health check')
  try:
   if s.get(f'http://127.0.0.1:{18972+i}/health',timeout=3).status_code==200:pending.remove(i)
  except requests.RequestException:pass
 if pending:time.sleep(5)
if pending:raise SystemExit(f'Servers not healthy: {pending}')
manifest={'model':'Cloudflare/clef','revision':'a30162b99d68a481d8bd75d38df069a0cf83df40','job_id':os.environ['SLURM_JOB_ID'],
 'partition':os.environ['SLURM_JOB_PARTITION'],'node':os.uname().nodename,'gpu_count':count,'replicas':count,'tensor_parallel_size':1,
 'dtype':'bfloat16','enforce_eager':True,'max_model_len':98304,'max_num_batched_tokens':98304,'max_num_seqs_per_replica':2,
 'image':'vllm/vllm-openai-rocm:nightly-rocm72-5d35ce3934a7804b298559d2d99eec33792e1eb9','image_digest':'sha256:a1799026344dce3d3a41f38a85ee861fee46b07fe81b3df3c3a0e54b91868390',
 'gpu_mapping':[{'slurm_index':int(i),'render_device':128+8*int(i),'uuid':'GPU-'+Path(f'/sys/class/drm/renderD{128+8*int(i)}/device/unique_id').read_text().strip().removeprefix('0x')} for i in os.environ['SLURM_JOB_GPUS'].split(',')],
 'healthy_utc':datetime.now(timezone.utc).isoformat()}
(r/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
docker exec "clef27-$job-0" python /work/classify.py --url http://127.0.0.1:18972 --pilot --concurrency 2 > "$quest_dir/logs/pilot-$job.log" 2>&1
"$python_bin" - "$quest_dir" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1])/'runtime-manifest.json';x=json.loads(p.read_text());x['classification_started_utc']=datetime.now(timezone.utc).isoformat();p.write_text(json.dumps(x,indent=2)+'\n')
PY
clients=()
for ((i=0;i<gpu_count;i++)); do
 docker exec "clef27-$job-$i" python /work/classify.py --url "http://127.0.0.1:$((18972+i))" --concurrency 2 --shard-count "$gpu_count" --shard-index "$i" > "$quest_dir/logs/classify-shard-$i.log" 2>&1 &
 clients+=("$!")
done
failed=0
for client in "${clients[@]}"; do wait "$client" || failed=1; done
((failed==0)) || exit 1
"$python_bin" "$quest_dir/shards.py" merge > "$quest_dir/logs/merge.log" 2>&1
cleanup
"$python_bin" "$quest_dir/calibrate.py" > "$quest_dir/logs/calibration-report.log" 2>&1
"$python_bin" "$quest_dir/verify.py" > "$quest_dir/logs/verification.log" 2>&1
