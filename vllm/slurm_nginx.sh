#!/bin/bash
# =============================================================================
# nginx coordinator for the vLLM job array (slurm_serve_array.sh).
#
# Submit AFTER the array job is running:
#   sbatch --dependency=after:<array_job_id> \
#          --export=ALL,ARRAY_JOB_ID=<array_job_id>,N_WORKERS=<N> \
#          vllm/slurm_nginx.sh
#
# Or use the one-liner in slurm_serve_array.sh's comments, which computes
# N_WORKERS automatically from the array task count.
#
# Required --export values:
#   ARRAY_JOB_ID   Slurm job ID of the array job (sets rendezvous dir path)
#   N_WORKERS      Number of array tasks (how many rendezvous files to expect)
#
# Environment overrides:
#   LB_PORT   nginx listen port  (default: 9000)
#   SCRATCH   base scratch path  (default: /scratch/project_462000963)
# =============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --account=project_462000963
#SBATCH --partition=small
#SBATCH --job-name=vllm_nginx
#SBATCH --output=logs/vllm_nginx_%j.log

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/project_462000963}"
LB_PORT="${LB_PORT:-9000}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

# ── Validate required exports ─────────────────────────────────────────────────
if [ -z "${ARRAY_JOB_ID:-}" ]; then
    err "ARRAY_JOB_ID is not set. Submit with:"
    err "  sbatch --export=ALL,ARRAY_JOB_ID=<id>,N_WORKERS=<n> vllm/slurm_nginx.sh"
    exit 1
fi
if [ -z "${N_WORKERS:-}" ]; then
    err "N_WORKERS is not set. Submit with:"
    err "  sbatch --export=ALL,ARRAY_JOB_ID=<id>,N_WORKERS=<n> vllm/slurm_nginx.sh"
    exit 1
fi

RENDEZVOUS_DIR="${SCRATCH}/users/aralikatte/vllm_rendezvous/${ARRAY_JOB_ID}"

log "nginx coordinator started"
log "Array job   : ${ARRAY_JOB_ID}"
log "Workers     : ${N_WORKERS}"
log "Rendezvous  : ${RENDEZVOUS_DIR}"
log "LB port     : ${LB_PORT}"
echo

mkdir -p logs

# ── Helper: count array tasks that are still PENDING or RUNNING ───────────────
array_jobs_active() {
    squeue -j "${ARRAY_JOB_ID}" --noheader 2>/dev/null | wc -l
}

# ── Phase 1: wait for all rendezvous files ────────────────────────────────────
# Each worker writes hostname:port to ${RENDEZVOUS_DIR}/${SLURM_ARRAY_TASK_ID}.txt
# as its very first action.  We keep waiting as long as tasks are still in the
# queue (PENDING or RUNNING) — no fixed deadline, since the nginx job often
# starts on the small partition before the GPU workers leave the queue.
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
        err "Array job ${ARRAY_JOB_ID} has no active tasks but only ${FOUND} / ${N_WORKERS} registered."
        err "Check logs/vllm_worker_${ARRAY_JOB_ID}_*.log for failures."
        exit 1
    fi
    log "  ${FOUND} / ${N_WORKERS} registered, ${ACTIVE} task(s) still active — retrying in 30 s ..."
    sleep 30
done

# Build backend list from rendezvous files.
mapfile -t BACKENDS < <(sort -u "${RENDEZVOUS_DIR}"/*.txt)
log "Backends registered:"
for B in "${BACKENDS[@]}"; do
    log "  http://${B}"
done
echo

# ── Phase 2: wait for all vLLM servers to be healthy ─────────────────────────
# Model loading can take several minutes.  Again we poll squeue rather than
# using a fixed deadline — stop waiting only if all tasks have exited.
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
        err "Array job ${ARRAY_JOB_ID} has no active tasks but only ${READY} / ${N_WORKERS} backends are healthy."
        err "Check logs/vllm_worker_${ARRAY_JOB_ID}_*.log"
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
    for B in "${BACKENDS[@]}"; do
        err "  http://${B}/v1"
    done
    # Keep this job alive so the array workers stay up.
    wait
    exit 0
fi

NGINX_CONF="$(mktemp /tmp/nginx_vllm_XXXXXX.conf)"
NGINX_PID_FILE="/tmp/nginx_vllm_${SLURM_JOB_ID}.pid"
NGINX_LOG_DIR="logs"

{
    echo "worker_processes auto;"
    echo "pid ${NGINX_PID_FILE};"
    echo "error_log ${NGINX_LOG_DIR}/nginx_${SLURM_JOB_ID}_error.log warn;"
    echo ""
    echo "events {"
    echo "    worker_connections 4096;"
    echo "}"
    echo ""
    echo "http {"
    echo "    access_log ${NGINX_LOG_DIR}/nginx_${SLURM_JOB_ID}_access.log;"
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

log "Generated nginx config: ${NGINX_CONF}"

nginx -c "${NGINX_CONF}"
log "nginx started (pid $(cat "${NGINX_PID_FILE}"))"
echo

HEAD_NODE="$(hostname)"
log "Load balancer endpoint: http://${HEAD_NODE}:${LB_PORT}/v1"
log "Direct backends:"
for B in "${BACKENDS[@]}"; do
    log "  http://${B}/v1"
done
echo
log "Point your pipeline at:"
log "  vllm_base_url: http://${HEAD_NODE}:${LB_PORT}/v1"
echo

# ── Keep job alive until cancelled ───────────────────────────────────────────
log "Serving. Cancel with:  scancel ${ARRAY_JOB_ID} ${SLURM_JOB_ID}"
wait
