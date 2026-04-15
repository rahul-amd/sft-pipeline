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

# 2b. Interactive — from inside an srun --pty bash shell
srun --account=project_462000963 --partition=standard-g \
     --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=128G \
     --time=4:00:00 --pty bash
bash vllm/serve.sh --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
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
| `ROCM_COMPAT`  | —                         | `1` (see below)                |

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
bash vllm/serve.sh \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 1 \
    --max-model-len 8192

# 72B example
bash vllm/serve.sh --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 8
```

---

## `ROCM_COMPAT` — device delegation on LUMI

On LUMI, cgroups v2 blocks `/dev/kfd` access from inside Singularity containers
in **all** job contexts (interactive and batch).  `--bind /dev/kfd --bind /dev/dri`
makes the device file visible but the cgroup still prevents `open()`.

`ROCM_COMPAT=1` (the default) uses `--rocm` to let Singularity handle AMD device
delegation, then strips the injected host ROCm libs from `LD_LIBRARY_PATH` inside
the container before starting vLLM.  This is necessary because `--rocm` injects
host libs compiled for glibc 2.38+, but the vLLM container is Ubuntu 22.04
(glibc 2.35) — without the strip, `import torch` fails with `GLIBC_2.38 not found`.

In short: `ROCM_COMPAT=1` fixes two problems at once.  There is no job type on
LUMI where `ROCM_COMPAT=0` works.  Only set it to `0` on other clusters where
Slurm propagates device cgroups to Singularity natively.

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

1. `slurm_serve_array.sh` submits an array of independent 1-node jobs.
   Each task writes `hostname:port` to a shared rendezvous directory on scratch,
   then runs `serve.sh`.
2. **Task 0 doubles as the nginx coordinator** — no separate Slurm job needed.
   It starts vLLM in the background, then waits for all other workers to register
   and pass `/health`, generates an nginx config, and starts nginx.
   The load balancer URL is printed in `logs/vllm_worker_<array_id>_0.log`.

This design avoids a second `sbatch` submission, which matters when you are
close to your per-user job limit.

### Usage

```bash
# Default: 16 workers (array tasks 0–15), 2 GPUs each — one command
sbatch vllm/slurm_serve_array.sh

# 8 workers
sbatch --array=0-7 vllm/slurm_serve_array.sh

# Different model or context length
MAX_MODEL_LEN=8192 sbatch vllm/slurm_serve_array.sh
```

Once task 0 is running and all backends are healthy, the URL appears in its log:
```
[14:05:22] Load balancer endpoint: http://nid007006:9000/v1
```

Point your pipeline at it:
```yaml
vllm_base_url: "http://nid007006:9000/v1"
annotation_concurrency: 1024   # N_WORKERS × 64
```

### Environment overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | HF model id |
| `TP` | `2` | Tensor-parallel size (GCDs per replica) |
| `MAX_MODEL_LEN` | unset | Max sequence length |
| `PORT_BASE` | `8000` | Base port; actual port = `PORT_BASE + task_id` |
| `LB_PORT` | `9000` | nginx listen port on task-0 node |
| `GPU_MEM_UTIL` | `0.92` | GPU memory fraction |

Pass overrides via environment or `--export`:
```bash
MAX_MODEL_LEN=8192 sbatch vllm/slurm_serve_array.sh

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
| `slurm_serve_array.sh` | Job array: one vLLM worker per task; task 0 runs nginx |
| `slurm_nginx.sh` | Standalone nginx coordinator (alternative if needed separately) |

---

## Troubleshooting

**`RuntimeError: No HIP GPUs are available` inside the container**

The cgroup device controller is blocking `/dev/kfd`.  This happens in **both**
interactive (`srun --pty bash`) and batch (`sbatch`) contexts on LUMI.
`ROCM_COMPAT=1` is now the default in `serve.sh`; if you're still seeing this,
check that you haven't overridden it to `0`:

```bash
ROCM_COMPAT=1 bash vllm/serve.sh --model <model> --tensor-parallel-size 2
```

**`ImportError: GLIBC_2.38 not found`**

You have `ROCM_COMPAT=0` set.  That path uses `--bind /dev/kfd --bind /dev/dri`
without `--rocm`, which doesn't grant GPU access on LUMI.  Removing `ROCM_COMPAT`
(or setting it to `1`) fixes both issues: `--rocm` for device access, lib strip
to avoid the glibc mismatch.

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
bash vllm/serve.sh --model <model> --tensor-parallel-size 2 --max-model-len 8192
GPU_MEM_UTIL=0.85 bash vllm/serve.sh --model <model> --tensor-parallel-size 2
```
