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
#SBATCH --time=1-00:00:00
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
    # ROCM_COMPAT left at default (0) — correct for sbatch jobs.
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
if ! command -v nginx >/dev/null 2>&1; then
    err "nginx not found in PATH."
    err "Install it:  conda install -c conda-forge nginx"
    err ""
    err "The ${N_WORKERS} vLLM backends are still running and can be used directly:"
    for B in "${BACKENDS[@]}"; do err "  http://${B}/v1"; done
    wait "${VLLM_PID}"
    exit 0
fi

NGINX_CONF="$(mktemp /tmp/nginx_vllm_XXXXXX.conf)"
NGINX_PID_FILE="/tmp/nginx_vllm_${SLURM_ARRAY_JOB_ID}.pid"
NGINX_LOG_DIR="logs"

{
    echo "worker_processes auto;"
    echo "pid ${NGINX_PID_FILE};"
    echo "error_log ${NGINX_LOG_DIR}/nginx_${SLURM_ARRAY_JOB_ID}_error.log warn;"
    echo ""
    echo "events {"
    echo "    worker_connections 4096;"
    echo "}"
    echo ""
    echo "http {"
    echo "    access_log ${NGINX_LOG_DIR}/nginx_${SLURM_ARRAY_JOB_ID}_access.log;"
    echo ""
    echo "    upstream vllm_backends {"
    echo "        least_conn;   # prefer least-busy backend (better than RR for LLMs)"
    for B in "${BACKENDS[@]}"; do
        echo "        server ${B};"
    done
    echo "    }"
    echo ""
    echo "    server {"
    echo "        listen ${LB_PORT};"
    echo ""
    echo "        location / {"
    echo "            proxy_pass         http://vllm_backends;"
    echo "            proxy_http_version 1.1;"
    echo "            proxy_set_header   Connection \"\";   # upstream keep-alive"
    echo "            proxy_set_header   Host \$host;"
    echo "            proxy_set_header   X-Real-IP \$remote_addr;"
    echo ""
    echo "            # Long timeouts: thinking models can take minutes per request."
    echo "            proxy_connect_timeout 60s;"
    echo "            proxy_send_timeout    600s;"
    echo "            proxy_read_timeout    600s;"
    echo ""
    echo "            # Don't buffer streaming responses (SSE / chunked transfer)."
    echo "            proxy_buffering    off;"
    echo "            proxy_cache        off;"
    echo "        }"
    echo "    }"
    echo "}"
} > "${NGINX_CONF}"

nginx -c "${NGINX_CONF}"
log "nginx started (pid $(cat "${NGINX_PID_FILE}"))"
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

# ── Keep alive: wait for vLLM to exit (i.e. until job is cancelled) ──────────
wait "${VLLM_PID}"
