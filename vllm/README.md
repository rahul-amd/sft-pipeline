# vLLM ROCm — Singularity SIF

Runs vLLM on the MI250X cluster via a Singularity SIF built from the official
`rocm/vllm` Docker image.  The SIF is self-contained (no overlay needed for
vLLM itself); model weights land in the HF cache on scratch.

---

## Quick start

```bash
# 1. Build the SIF (once, on a login node — takes 10–30 min)
bash vllm/build_sif.sh

# 2a. Slurm batch job (recommended for production)
sbatch vllm/slurm_serve.sh
tail -f logs/vllm_serve_<jobid>.log

# 2b. Interactive — from inside an srun --pty bash shell, set ROCM_COMPAT=1
srun --account=project_462000963 --partition=standard-g \
     --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=128G \
     --time=4:00:00 --pty bash
ROCM_COMPAT=1 bash vllm/serve.sh --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 1 --max-model-len 8192
```

---

## CLI flags

```
bash vllm/serve.sh --model <HF-model-id>  [options]

Options:
  --model                  HuggingFace model id (required)
  --tensor-parallel-size N number of GPUs to use (default: 8)
  --max-model-len N        max sequence length / context window (default: model default)
  --port N                 HTTP port (default: 8000)
  --host ADDR              bind address (default: 0.0.0.0)
```

All flags can also be set as environment variables:

| Env var        | Flag equivalent           | Default                        |
|----------------|---------------------------|--------------------------------|
| `MODEL`        | `--model`                 | (required)                     |
| `TP`           | `--tensor-parallel-size`  | `8`                            |
| `MAX_MODEL_LEN`| `--max-model-len`         | unset (model default)          |
| `PORT`         | `--port`                  | `8000`                         |
| `BIND_HOST`    | `--host`                  | `0.0.0.0`                      |
| `GPU_MEM_UTIL` | —                         | `0.92`                         |
| `HF_HOME`      | —                         | `${SCRATCH}/hf_cache`          |
| `NIC`          | —                         | auto-detected                  |
| `SIF`          | —                         | `${SINCONS_DIR}/vllm_rocm.sif` |
| `ROCM_COMPAT`  | —                         | `0` (see below)                |

---

## Choosing an image tag

`build_sif.sh` defaults to a ROCm 6.3 tag. Browse tags at:
https://hub.docker.com/r/rocm/vllm/tags

Match the ROCm version to your cluster:
```bash
cat /opt/rocm/.info/version   # e.g. 6.3.4
```

Override at build time:
```bash
VLLM_TAG=rocm6.3_mi300_ubuntu22.04_py3.11_vllm_v0.8.5.post1 bash vllm/build_sif.sh
```

**GFX arch matching**: Tags with an explicit `gfxNNN` string (e.g. `gfx950` =
MI350X, `gfx942` = MI300X) only run on that arch.  This cluster is MI250X =
`gfx90a`.  Tags without an explicit arch (e.g. `rocm6.3_mi300_...`) typically
bundle multiple archs including `gfx90a` and work here.  `serve.sh` will catch
a mismatch at startup and print a clear error before any GPU work starts.

---

## Model sizes and tensor parallelism

Each AMD MI250X chip has **2 GCDs** (each GCD ≈ 64 GB HBM2e).
A full node has 8 chips = **16 GCDs**.

| Model size | Recommended TP | `--gpus-per-node` |
|------------|---------------|-------------------|
| 7B         | 1             | 1                 |
| 30B MoE    | 1             | 1                 |
| 72B        | 8             | 8                 |
| 122B MoE   | 16            | 16                |

```bash
# 30B MoE example (fits on 1 GCD with max-model-len cap)
ROCM_COMPAT=1 bash vllm/serve.sh \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 1 \
    --max-model-len 8192

# 72B example
bash vllm/serve.sh --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 8
```

---

## Interactive vs batch mode (`ROCM_COMPAT`)

On LUMI, cgroups v2 restricts `/dev/kfd` access differently depending on how
Singularity is launched:

| How you run serve.sh | `ROCM_COMPAT` | Singularity flags |
|---|---|---|
| `sbatch vllm/slurm_serve.sh` | `0` (default) | `--bind /dev/kfd --bind /dev/dri` |
| `srun singularity exec ...` (direct) | `0` (default) | `--bind /dev/kfd --bind /dev/dri` |
| `srun --pty bash` → `bash vllm/serve.sh` | **`1`** | `--rocm` + LD_LIBRARY_PATH strip |

**Why the difference**: In a `srun --pty bash` interactive shell, Singularity's
child cgroup does not inherit the job's device allowlist, so `/dev/kfd` returns
`EPERM` even though the file shows `crw-rw-rw-`.  `--rocm` fixes cgroup
delegation, but it also injects host ROCm libs into `/.singularity.d/libs/`,
which need glibc 2.38+.  The container is Ubuntu 22.04 (glibc 2.35), so
`import torch` fails with `GLIBC_2.38 not found`.  `ROCM_COMPAT=1` adds a
bash wrapper that strips `/.singularity.d/libs` from `LD_LIBRARY_PATH` before
starting vLLM, resolving both issues.

---

## Using the server (from Stage 2 / Stage 6)

The server exposes an OpenAI-compatible API on `http://<node>:8000/v1`.

```python
import openai
client = openai.OpenAI(base_url="http://<node>:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Thinking-2507",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Or point the pipeline config at it:
```yaml
stage5_inference:
  vllm_base_url: "http://<node>:8000/v1"
```

---

## Scaling throughput: job array + nginx

For maximum scheduler flexibility, use the **job array** approach instead of a
single multi-node allocation.  Each array task is an independent 1-node job
(2 GPUs); the scheduler can backfill them individually rather than waiting for
all N nodes to be free simultaneously.

### How it works

1. `slurm_serve_array.sh` submits an array of worker jobs (default: 16).
   Each task writes `hostname:port` to a shared rendezvous directory on scratch,
   then runs `serve.sh`.
2. `slurm_nginx.sh` runs after the array starts (`--dependency=after:<array_id>`).
   It waits for all rendezvous files to appear, polls `/health` on each backend,
   generates an nginx config, and starts nginx as a load balancer.

### Usage

```bash
# Default: 16 workers (array tasks 0–15), 2 GPUs each
JID=$(sbatch --parsable vllm/slurm_serve_array.sh) && \
echo "Workers: $JID" && \
sbatch --dependency=after:$JID \
       --export=ALL,ARRAY_JOB_ID=$JID,N_WORKERS=16 \
       vllm/slurm_nginx.sh

# 8 workers
JID=$(sbatch --parsable --array=0-7 vllm/slurm_serve_array.sh) && \
sbatch --dependency=after:$JID \
       --export=ALL,ARRAY_JOB_ID=$JID,N_WORKERS=8 \
       vllm/slurm_nginx.sh

# Different model / context length
JID=$(sbatch --parsable \
    vllm/slurm_serve_array.sh) && \
MAX_MODEL_LEN=8192 MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507 \
sbatch --dependency=after:$JID \
       --export=ALL,ARRAY_JOB_ID=$JID,N_WORKERS=16,MAX_MODEL_LEN=8192 \
       vllm/slurm_nginx.sh
```

The nginx job prints the load balancer URL once ready:
```
[14:05:22] Load balancer endpoint: http://nid007006:9000/v1
```

Point your pipeline at it:
```yaml
vllm_base_url: "http://nid007006:9000/v1"
annotation_concurrency: 1024   # N_WORKERS × 64
```

### Environment overrides for array workers

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | HF model id |
| `TP` | `2` | Tensor-parallel size (GCDs per replica) |
| `MAX_MODEL_LEN` | unset | Max sequence length |
| `PORT` | `8000` | vLLM port on each worker node |
| `GPU_MEM_UTIL` | `0.92` | GPU memory fraction |

Pass overrides to the array job via `--export`:
```bash
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct,TP=8 \
       --array=0-3 \
       vllm/slurm_serve_array.sh
```

---

## Files

| File | Purpose |
|------|---------|
| `build_sif.sh` | Pull Docker image → build SIF (run once on login node) |
| `serve.sh`     | Start vLLM server from SIF (single node, interactive or Slurm) |
| `slurm_serve.sh` | Slurm batch wrapper for a single vLLM node |
| `slurm_serve_array.sh` | Job array: one vLLM worker per task |
| `slurm_nginx.sh` | nginx coordinator for job array workers |

---

## Troubleshooting

**`RuntimeError: No HIP GPUs are available` inside the container**

You are running from an interactive `srun --pty bash` shell.  The cgroup device
controller blocks `/dev/kfd` even though `rocm-smi` shows the GPU.  Fix:

```bash
ROCM_COMPAT=1 bash vllm/serve.sh --model <model> --tensor-parallel-size 1
```

**`ImportError: GLIBC_2.38 not found`**

You are using `--rocm` with this SIF.  The SIF has ROCm baked in (Ubuntu 22.04,
glibc 2.35); `--rocm` injects host libs that require glibc 2.38+.  Use
`ROCM_COMPAT=1` (which handles the strip automatically) instead of passing
`--rocm` manually.

**`OSError: [Errno 99] Cannot assign requested address`**

The vLLM HTTP server is trying to bind to a hostname that doesn't resolve to a
bindable IP.  This was caused by the shell variable `HOST` (pre-set to the node
hostname on Cray/LUMI) being used as the bind address.  Fixed in `serve.sh`
(uses `BIND_HOST` instead).  If you still see it, check that `BIND_HOST` is
not accidentally set to a hostname in your environment.

**`torch.AcceleratorError: hipErrorInvalidImage`**

GFX architecture mismatch — the SIF kernels target a different GPU.  `serve.sh`
will catch this and print which arch the tag targets vs what the host has.
Rebuild with a compatible tag:

```bash
VLLM_TAG=rocm6.3_mi300_ubuntu22.04_py3.11_vllm_v0.8.5.post1 bash vllm/build_sif.sh
```

**`HSA_OVERRIDE_GFX_VERSION` warnings**

Do **not** set `HSA_OVERRIDE_GFX_VERSION`.  ROCm 6.3 supports gfx90a (MI250X)
natively; forcing gfx900 loads wrong kernels and causes GPU memory access faults.

**`torch.cuda.get_arch_list()` returns `[]`**

Normal on ROCm builds.  That function is CUDA-specific.  GPU visibility is
confirmed by `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`.

**Out of GPU memory**

Lower `GPU_MEM_UTIL` or cap the KV cache with `--max-model-len`:
```bash
ROCM_COMPAT=1 bash vllm/serve.sh --model <model> \
    --tensor-parallel-size 1 --max-model-len 8192
GPU_MEM_UTIL=0.85 ROCM_COMPAT=1 bash vllm/serve.sh --model <model> \
    --tensor-parallel-size 1
```
