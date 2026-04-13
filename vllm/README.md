# vLLM ROCm — Singularity SIF

Runs vLLM on the MI250X cluster via a Singularity SIF built from the official
`rocm/vllm` Docker image.  The SIF is self-contained (no overlay needed for
vLLM itself); model weights land in the HF cache on scratch.

---

## Quick start

```bash
# 1. Build the SIF (once, on a login node — takes 10–30 min)
bash vllm/build_sif.sh

# 2a. Interactive — get a GPU node, then serve
srun --account=project_462000963 --partition=standard-g \
     --nodes=1 --ntasks=1 --gpus-per-node=8 --mem=256G \
     --time=4:00:00 --pty bash
bash vllm/serve.sh --model Qwen/Qwen2.5-7B-Instruct

# 2b. Slurm batch job
sbatch vllm/slurm_serve.sh
tail -f logs/vllm_serve_<jobid>.log
```

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

---

## Model sizes and tensor parallelism

Each AMD MI250X chip has **2 GCDs** (each GCD ≈ 64 GB HBM2e).
A full node has 8 chips = **16 GCDs**.

| Model size | Recommended TP | `--gpus-per-node` |
|------------|---------------|-------------------|
| 7B         | 2             | 2                 |
| 72B        | 8             | 8                 |
| 122B MoE   | 16            | 16                |

```bash
# 72B example
MODEL=Qwen/Qwen2.5-72B-Instruct TP=8 bash vllm/serve.sh
```

---

## Using the server (from Stage 2 / Stage 6)

The server exposes an OpenAI-compatible API on `http://<node>:8000/v1`.

```python
import openai
client = openai.OpenAI(base_url="http://<node>:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Or point the pipeline config at it:
```yaml
stage5_inference:
  vllm_base_url: "http://<node>:8000/v1"
```

---

## Files

| File | Purpose |
|------|---------|
| `build_sif.sh` | Pull Docker image → build SIF (run once on login node) |
| `serve.sh`     | Start vLLM server from SIF (interactive or from Slurm) |
| `slurm_serve.sh` | Slurm batch wrapper around `serve.sh` |

---

## Troubleshooting

**`torch.cuda.is_available()` returns False inside the container**

- Check that `--rocm` is passed to `singularity exec` (fixes cgroup device
  delegation when launching from inside `srun --pty bash`)
- Check that `/opt/rocm` is bound: `--bind /opt/rocm`
- Verify on the host first: `/opt/rocm/bin/rocm-smi`

**`PermissionError: [Errno 1] /dev/kfd`**

Use `--rocm` flag instead of `--bind /dev/kfd --bind /dev/dri`.  See the
`CLAUDE.md` "Gotchas" section for the full explanation.

**`HSA_OVERRIDE_GFX_VERSION` warnings**

Do **not** set `HSA_OVERRIDE_GFX_VERSION`.  ROCm 6.3 supports gfx90a (MI250X)
natively; forcing gfx900 loads wrong kernels and causes GPU memory access faults.

**Out of GPU memory**

Lower `GPU_MEM_UTIL` (default 0.92) or add `--max-model-len` to cap KV cache:
```bash
GPU_MEM_UTIL=0.85 MAX_MODEL_LEN=8192 bash vllm/serve.sh --model ...
```
