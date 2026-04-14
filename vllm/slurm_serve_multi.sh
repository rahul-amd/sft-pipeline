#!/bin/bash
# =============================================================================
# Multi-node vLLM serving with nginx load balancer.
#
# Starts one vLLM server per node (TP=2, one per pair of MI250X GCDs), then
# brings up an nginx upstream on the head node for round-robin load balancing.
#
# Usage:
#   sbatch vllm/slurm_serve_multi.sh               # 4 replicas (default)
#   sbatch --nodes=8 vllm/slurm_serve_multi.sh     # 8 replicas
#
# Environment overrides:
#   MODEL        HF model id         (default: Qwen/Qwen3-30B-A3B-Thinking-2507)
#   TP           tensor-parallel size (default: 2)
#   MAX_MODEL_LEN max sequence length (default: unset → model default)
#   PORT         vLLM port per node  (default: 8000)
#   LB_PORT      nginx port on head  (default: 9000)
#   GPU_MEM_UTIL GPU memory fraction (default: 0.92)
#   SCRATCH      base scratch path   (default: /scratch/project_462000963)
# =============================================================================
#SBATCH --account=project_462000963
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --job-name=vllm_multi
#SBATCH --output=logs/vllm_multi_%j.log

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRATCH="${SCRATCH:-/scratch/project_462000963}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
PORT="${PORT:-8000}"
LB_PORT="${LB_PORT:-9000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

# ── Node inventory ────────────────────────────────────────────────────────────
# scontrol expands e.g. "nid[007006-007009]" into one hostname per line.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"
N_NODES="${#NODES[@]}"

mkdir -p logs

echo
log "Job             : $SLURM_JOB_ID"
log "Nodes ($N_NODES)      : ${NODES[*]}"
log "Head node       : $HEAD_NODE (nginx on :$LB_PORT)"
log "Model           : $MODEL"
log "TP per replica  : $TP  (${TP} GCDs × $N_NODES nodes = $((TP * N_NODES)) GCDs total)"
log "vLLM port       : $PORT  (per node)"
log "Max seq len     : ${MAX_MODEL_LEN:-(model default)}"
echo

# ── Start one vLLM server per node ───────────────────────────────────────────
# Each srun step is pinned to one node and inherits that node's GPU allocation.
for NODE in "${NODES[@]}"; do
    log "Launching vLLM on $NODE ..."
    srun --nodes=1 --ntasks=1 --nodelist="$NODE" --exclusive \
        env \
            MODEL="$MODEL" \
            TP="$TP" \
            PORT="$PORT" \
            GPU_MEM_UTIL="$GPU_MEM_UTIL" \
            MAX_MODEL_LEN="$MAX_MODEL_LEN" \
            SCRATCH="$SCRATCH" \
        bash vllm/serve.sh \
        > "logs/vllm_${SLURM_JOB_ID}_${NODE}.log" 2>&1 &
done

# ── Wait for all backends to pass /health ─────────────────────────────────────
# vLLM can take several minutes to load a large model.
log "Waiting for all $N_NODES vLLM servers to be healthy (up to 30 min) ..."
DEADLINE=$(( $(date +%s) + 1800 ))
while true; do
    READY=0
    for NODE in "${NODES[@]}"; do
        if curl -sf "http://${NODE}:${PORT}/health" >/dev/null 2>&1; then
            READY=$(( READY + 1 ))
        fi
    done
    if [ "$READY" -eq "$N_NODES" ]; then
        log "All $N_NODES backends healthy."
        break
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        err "$READY / $N_NODES backends came up within 30 min. Check logs/vllm_${SLURM_JOB_ID}_*.log"
        exit 1
    fi
    log "  $READY / $N_NODES ready — retrying in 30 s ..."
    sleep 30
done
echo

# ── Generate nginx config ─────────────────────────────────────────────────────
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
    for NODE in "${NODES[@]}"; do
        echo "        server ${NODE}:${PORT};"
    done
    echo "    }"
    echo ""
    echo "    server {"
    echo "        listen ${LB_PORT};"
    echo ""
    echo "        location / {"
    echo "            proxy_pass         http://vllm_backends;"
    echo "            proxy_http_version 1.1;"
    echo "            proxy_set_header   Connection \"\";"   # upstream keep-alive
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
} > "$NGINX_CONF"

log "Generated nginx config: $NGINX_CONF"
log "Upstream backends:"
for NODE in "${NODES[@]}"; do
    log "  http://${NODE}:${PORT}"
done

# ── Launch nginx ──────────────────────────────────────────────────────────────
if ! command -v nginx >/dev/null 2>&1; then
    err "nginx not found in PATH."
    err "Install it:  conda install -c conda-forge nginx"
    err "Then re-run this script, or start nginx manually with:"
    err "  nginx -c $NGINX_CONF"
    err ""
    err "The $N_NODES vLLM backends are still running and can be used directly:"
    for NODE in "${NODES[@]}"; do
        err "  http://${NODE}:${PORT}/v1"
    done
    # Keep the job alive so vLLM servers stay up even without nginx.
    wait
    exit 0
fi

nginx -c "$NGINX_CONF"
log "nginx started (pid $(cat "$NGINX_PID_FILE"))"
echo
log "Load balancer endpoint: http://${HEAD_NODE}:${LB_PORT}/v1"
log "Direct backends:"
for NODE in "${NODES[@]}"; do
    log "  http://${NODE}:${PORT}/v1"
done
echo
log "Point your pipeline at:"
log "  vllm_base_url: http://${HEAD_NODE}:${LB_PORT}/v1"
echo

# ── Keep job alive until cancelled ───────────────────────────────────────────
log "Serving. Cancel with:  scancel $SLURM_JOB_ID"
wait
