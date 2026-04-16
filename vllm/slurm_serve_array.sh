#!/bin/bash
# =============================================================================
# Job array: one vLLM worker per task, 1 node × 2 GPUs each.
# Task 0 additionally acts as the nginx coordinator — no separate job needed.
#
# Submit with:
#   sbatch vllm/slurm_serve_array.sh                  # 16 workers (default)
#   sbatch --array=0-7 vllm/slurm_serve_array.sh      # 8 workers
#
# Task 0 starts vLLM in the background, waits for all workers to register and
# pass /health, then starts nginx on port LB_PORT.  The load balancer URL is
# printed in logs/vllm_worker_<array_id>_0.log once nginx is ready.
#
# Environment overrides:
#   MODEL         HF model id          (default: Qwen/Qwen3-30B-A3B-Thinking-2507)
#   TP            tensor-parallel size (default: 2)
#   MAX_MODEL_LEN max sequence length  (default: unset → model default)
#   PORT_BASE     base HTTP port       (default: 8000); actual port = PORT_BASE + SLURM_ARRAY_TASK_ID
#                 Two tasks on the same node get different ports automatically.
#   LB_PORT       nginx listen port on task-0 node    (default: 9000)
#   GPU_MEM_UTIL  GPU memory fraction  (default: 0.92)
#   SCRATCH       base scratch path    (default: /scratch/project_462000963)
# =============================================================================
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --job-name=vllm_worker
#SBATCH --output=logs/vllm_worker_%A_%a.log

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/project_462000963}"
LB_PORT="${LB_PORT:-9000}"
# Each task gets a unique port: PORT_BASE + task_id.
# If two tasks land on the same node they bind different ports and don't conflict.
PORT=$(( ${PORT_BASE:-8000} + SLURM_ARRAY_TASK_ID ))
export PORT

# ── TP and model defaults ─────────────────────────────────────────────────────
# Must match --gpus-per-node above.  Exported so serve.sh inherits them.
export TP="${TP:-2}"
export MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"

# ── Restrict GPU visibility to the GPUs Slurm allocated ──────────────────────
# When --rocm is used, Singularity bypasses Slurm's cgroup GPU restriction and
# the container sees all GCDs on the node.  Slurm sets SLURM_JOB_GPUS (or
# SLURM_STEP_GPUS) to the allocated GPU indices (e.g. "0,1").  Forwarding this
# as ROCR_VISIBLE_DEVICES ensures vLLM only uses the allocated GPUs.
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export ROCR_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
    export ROCR_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

# ── Rendezvous: advertise this worker's address ───────────────────────────────
# Task 0 reads these files to build the nginx upstream config.
# One file per array task: content is "hostname:port".
RENDEZVOUS_DIR="${SCRATCH}/users/aralikatte/vllm_rendezvous/${SLURM_ARRAY_JOB_ID}"
mkdir -p "${RENDEZVOUS_DIR}"
echo "$(hostname):${PORT}" > "${RENDEZVOUS_DIR}/${SLURM_ARRAY_TASK_ID}.txt"
log "Registered as $(hostname):${PORT} → ${RENDEZVOUS_DIR}/${SLURM_ARRAY_TASK_ID}.txt"

# ── Non-coordinator tasks: just run vLLM ─────────────────────────────────────
if [ "${SLURM_ARRAY_TASK_ID}" -ne 0 ]; then
    exec bash vllm/serve.sh
fi

# ── Task 0: run vLLM in background, then coordinate nginx ────────────────────
log "Task 0: starting vLLM in background and acting as nginx coordinator ..."
bash vllm/serve.sh &
VLLM_PID=$!

N_WORKERS="${SLURM_ARRAY_TASK_COUNT}"
log "Coordinator: expecting ${N_WORKERS} workers (SLURM_ARRAY_TASK_COUNT)"

# Helper: count array tasks that are still PENDING or RUNNING.
array_jobs_active() {
    squeue -j "${SLURM_ARRAY_JOB_ID}" --noheader 2>/dev/null | wc -l
}

# ── Phase 1: wait for all rendezvous files ────────────────────────────────────
log "Waiting for ${N_WORKERS} workers to register ..."
log "(polling squeue — will wait as long as array tasks are active)"
while true; do
    FOUND=$(find "${RENDEZVOUS_DIR}" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)
    if [ "${FOUND}" -ge "${N_WORKERS}" ]; then
        log "All ${N_WORKERS} workers registered."
        break
    fi
    ACTIVE=$(array_jobs_active)
    if [ "${ACTIVE}" -eq 0 ]; then
        err "Array job ${SLURM_ARRAY_JOB_ID} has no active tasks but only ${FOUND} / ${N_WORKERS} registered."
        err "Check logs/vllm_worker_${SLURM_ARRAY_JOB_ID}_*.log for failures."
        exit 1
    fi
    log "  ${FOUND} / ${N_WORKERS} registered, ${ACTIVE} task(s) still active — retrying in 30 s ..."
    sleep 30
done

mapfile -t BACKENDS < <(sort -u "${RENDEZVOUS_DIR}"/*.txt)
log "Backends:"
for B in "${BACKENDS[@]}"; do log "  http://${B}"; done
echo

# ── Phase 2: wait for all vLLM servers to be healthy ─────────────────────────
log "Waiting for all ${N_WORKERS} vLLM servers to be healthy ..."
log "(polling squeue — will wait as long as array tasks are active)"
while true; do
    READY=0
    for B in "${BACKENDS[@]}"; do
        if curl -sf "http://${B}/health" >/dev/null 2>&1; then
            READY=$(( READY + 1 ))
        fi
    done
    if [ "${READY}" -ge "${N_WORKERS}" ]; then
        log "All ${N_WORKERS} backends healthy."
        break
    fi
    ACTIVE=$(array_jobs_active)
    if [ "${ACTIVE}" -eq 0 ]; then
        err "Array job has no active tasks but only ${READY} / ${N_WORKERS} backends are healthy."
        err "Check logs/vllm_worker_${SLURM_ARRAY_JOB_ID}_*.log"
        exit 1
    fi
    log "  ${READY} / ${N_WORKERS} healthy, ${ACTIVE} task(s) still active — retrying in 30 s ..."
    sleep 30
done
echo

# ── Phase 3: generate nginx config and start nginx ───────────────────────────
# nginx is installed in a dedicated conda prefix on scratch (host-accessible,
# no container needed).  One-time setup from a login node:
#   conda create -p ${SCRATCH}/users/aralikatte/sft-nginx -c conda-forge nginx -y
#
# conda-forge nginx uses RPATH ($ORIGIN/../lib) so no LD_LIBRARY_PATH needed.
NGINX_PREFIX="${NGINX_PREFIX:-${SCRATCH}/users/aralikatte/sft-nginx}"
NGINX_BIN="${NGINX_PREFIX}/bin/nginx"

if [ ! -x "${NGINX_BIN}" ]; then
    err "nginx not found at ${NGINX_BIN}."
    err "Install it once from a login node:"
    err "  conda create -p ${NGINX_PREFIX} -c conda-forge nginx -y"
    err ""
    err "The ${N_WORKERS} vLLM backends are still running and can be used directly:"
    for B in "${BACKENDS[@]}"; do err "  http://${B}/v1"; done
    wait "${VLLM_PID}"
    exit 0
fi

NGINX_CONF="${SCRATCH}/users/aralikatte/nginx_${SLURM_ARRAY_JOB_ID}.conf"
NGINX_LOG_DIR="$(pwd)/logs"
NGINX_TMP="${SCRATCH}/users/aralikatte/nginx_tmp_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${NGINX_TMP}"

cat > "${NGINX_CONF}" <<NGINXEOF
worker_processes auto;
pid        /tmp/nginx_${SLURM_ARRAY_JOB_ID}.pid;
error_log ${NGINX_LOG_DIR}/nginx_${SLURM_ARRAY_JOB_ID}_error.log warn;
daemon off;

events {
    worker_connections 4096;
}

http {
    access_log ${NGINX_LOG_DIR}/nginx_${SLURM_ARRAY_JOB_ID}_access.log;

    client_body_temp_path  ${NGINX_TMP}/client_body;
    proxy_temp_path        ${NGINX_TMP}/proxy;
    fastcgi_temp_path      ${NGINX_TMP}/fastcgi;
    uwsgi_temp_path        ${NGINX_TMP}/uwsgi;
    scgi_temp_path         ${NGINX_TMP}/scgi;

    upstream vllm_backends {
        least_conn;
$(for B in "${BACKENDS[@]}"; do echo "        server ${B};"; done)
    }

    server {
        listen ${LB_PORT};

        location / {
            proxy_pass         http://vllm_backends;
            proxy_http_version 1.1;
            proxy_set_header   Connection "";
            proxy_set_header   Host \$host;
            proxy_set_header   X-Real-IP \$remote_addr;

            proxy_connect_timeout 60s;
            proxy_send_timeout    600s;
            proxy_read_timeout    600s;

            proxy_buffering    off;
            proxy_cache        off;
        }
    }
}
NGINXEOF

log "Starting nginx (${NGINX_BIN}) ..."
"${NGINX_BIN}" -c "${NGINX_CONF}" &
NGINX_PID=$!

sleep 2
if ! kill -0 "${NGINX_PID}" 2>/dev/null; then
    err "nginx exited immediately — check ${NGINX_LOG_DIR}/nginx_${SLURM_ARRAY_JOB_ID}_error.log"
    err "The ${N_WORKERS} vLLM backends are still running and can be used directly:"
    for B in "${BACKENDS[@]}"; do err "  http://${B}/v1"; done
    wait "${VLLM_PID}"
    exit 1
fi
log "nginx started (pid ${NGINX_PID}, listening on :${LB_PORT})"
echo

HEAD_NODE="$(hostname)"
log "================================================================"
log "Load balancer endpoint: http://${HEAD_NODE}:${LB_PORT}/v1"
log "Point your pipeline at:"
log "  vllm_base_url: http://${HEAD_NODE}:${LB_PORT}/v1"
log "Direct backends:"
for B in "${BACKENDS[@]}"; do log "  http://${B}/v1"; done
log "Cancel with:  scancel ${SLURM_ARRAY_JOB_ID}"
log "================================================================"
echo

# ── Keep alive: wait for vLLM and nginx to exit (until job is cancelled) ─────
wait "${VLLM_PID}" "${NGINX_PID}"
