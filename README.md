# SFT Pipeline

Automated 7-stage pipeline for constructing high-quality post-training datasets for LLM supervised fine-tuning (SFT). Produces **5M prompt–response pairs** with structured reasoning traces from a teacher model.

## Overview

The pipeline chains together:

1. **Prompt collection** — ingest from 45+ public HuggingFace datasets across math, code, science, and general instruction-following
2. **Prompt generation** — synthesize additional prompts from knowledge-rich corpora (arXiv, Wikipedia, textbooks) using a lightweight LLM
3. **Clustering** — embed all prompts, build a FAISS index, cluster by domain and assign difficulty labels (Easy / Medium / Hard)
4. **Sampling** — enforce domain and difficulty quotas, deduplicate by embedding cosine similarity, sort for KV-cache reuse
5. **Inference** — batch inference with Qwen3.5-122B-A10B via vLLM; each response is `<think>reasoning</think><answer>answer</answer>`
6. **Quality filtering** — structural checks, heuristics, SymPy math verification, sandboxed code execution, LLM judge (10% sample)
7. **Multilingual expansion** — *(v2, not yet implemented)*

**Production target**: 32-node AMD MI250X cluster (512 GCDs, ROCm), local disk.
**Dev environment**: Windows/Linux laptop with NVIDIA GPU (CUDA) or CPU-only.

## Installation

```bash
# Clone and install in editable mode
git clone <repo>
cd sft-pipeline
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

**Optional GPU dependencies** (for flash-kmeans clustering):
```bash
pip install flash-kmeans   # Triton GPU kernels for fast K-Means (requires CUDA)
```

**ROCm (cluster)**:
Install the ROCm-specific vLLM wheel separately — the standard `pip install vllm` installs the CUDA build.

## Quickstart

### Run Stage 1 only (dataset collection)

The fastest way to test the pipeline is to collect prompts from existing HuggingFace datasets:

```bash
# Set HF token if any datasets require it (e.g. lmsys/lmsys-chat-1m)
export HF_TOKEN=hf_...

# Run Stage 1 with the research config (45 datasets, 10 examples each)
sft-pipeline run-stage stage1_collect --config config/stage1_research.yaml

# Output: C:/Users/rahul/data/sft-pipeline/runs/research_stage1_001/stage1/
```

The `stage1_research.yaml` config collects from 45 public datasets covering math, code, science, and general instruction-following, capped at `max_examples: 10` per source for fast iteration. Remove or increase `max_examples` for a full collection run.

### Full pipeline (dev)

```bash
sft-pipeline run --config config/dev.yaml
```

### Full pipeline (production cluster)

```bash
# Start Ray head node
ray start --head --num-gpus=16

# Start workers on remaining 31 nodes
ray start --address=<head-node-ip>:6379 --num-gpus=16

# Start vLLM HTTP server for Stage 2 generator + Stage 6 LLM judge
bash scripts/serve_vllm.sh

# Run full pipeline
sft-pipeline run --config config/prod.yaml
```

## CLI Reference

```bash
# Full pipeline run (auto-resumes from DuckDB checkpoint on restart)
sft-pipeline run --config config/prod.yaml

# Run a single stage
sft-pipeline run-stage stage1_collect --config config/stage1_research.yaml
sft-pipeline run-stage stage4_sample  --config config/prod.yaml

# Check stage completion status
sft-pipeline status --config config/prod.yaml

# Dry-run: estimate GPU-hours, wall clock, API cost (no inference)
sft-pipeline estimate --config config/prod.yaml

# Override individual config values at the CLI
sft-pipeline run --config config/prod.yaml --set stage5_inference.n_replicas=32
```

## Running Stages Individually

Each stage can be run independently with `sft-pipeline run-stage <stage> --config <config>`. Dedicated configs exist for the stages that have been run in production; use them when available.

### Stage 1 — Prompt Collection

```bash
# Collect from 45 public HuggingFace research datasets
export HF_TOKEN=hf_...   # required for gated datasets (e.g. lmsys-chat-1m)
sft-pipeline run-stage stage1_collect --config config/stage1_research.yaml

# Output: {base_path}/stage1/part-*.jsonl
```

`stage1_research.yaml` targets `run_id: research_stage1_001d` on the cluster scratch filesystem. It sets `max_examples` per source for fast iteration; remove to collect all available data.

On the cluster, Stage 1 runs distributed across 32 Ray workers via Slurm:
```bash
sbatch scripts/slurm_stage1.sh
```

### Stage 3 — Clustering (two-step workflow)

Stage 3 is split into two separate runs: first cluster the prompts, then annotate them with an LLM. This decoupling lets clustering run on GPU nodes while annotation runs on CPU nodes against an external vLLM server.

**Step 1 — Embed + cluster (requires GPUs + Ray cluster):**
```bash
sbatch scripts/slurm_stage3.sh
# Uses config/stage3_cluster.yaml
# Output: {base_path}/stage3/part-*.jsonl  (cluster labels, heuristic domain/difficulty)
#         {base_path}/stage3/embeddings/   (sharded float16 Parquet)
#         {base_path}/stage3/faiss.index
```

`stage3_cluster.yaml` runs distributed embedding across 32 Ray workers (one per GPU) then torch_kmeans clustering into 100,000 clusters. `annotation_enabled: false` — annotation is handled separately below.

**Step 2 — LLM annotation (offline import workflow):**

If you have the annotation responses as a JSONL file (e.g. generated on another cluster):
```bash
sft-pipeline run-stage stage3_cluster \
    --config config/stage3_cluster.yaml \
    --import-annotations /path/to/responses.jsonl
```

The responses file is a JSONL where each line has `{"prompt_id": "sha256:...", "response": {...}}`. The `response` field is the raw LLM output or a pre-parsed annotation dict with keys `domain`, `difficulty`, `topics`, `language`, `summary`.

To generate the annotation request file (then run inference elsewhere):
```bash
sft-pipeline run-stage stage3_cluster \
    --config config/stage3_cluster.yaml \
    --dump-annotations /path/to/requests.jsonl
# Each line: {"prompt_id": "...", "messages": [{"role": "system", ...}, {"role": "user", ...}]}
```

To annotate online (vLLM server must be running):
```bash
# Start vLLM server first (see vllm/serve.sh or sbatch vllm/slurm_serve_array.sh)
# Then run annotation-only with the CPU config:
sft-pipeline annotate --config config/stage3_annotate.yaml
# Or via Slurm (no GPU allocation):
sbatch scripts/slurm_stage3_annotate.sh
```

`stage3_annotate.yaml` connects to a running vLLM server at `annotation_api_base` with concurrency 4096 (64 replicas × 64 each). It writes `{base_path}/stage3/annotations.parquet` as a resumable checkpoint, then a second `run-stage stage3_cluster --import-annotations` pass merges the results.

### Visualization (after Stage 3)

```bash
# Install viz deps (separate from pipeline)
pip install -r viz/requirements.txt

# Export snapshot from the completed Stage 3 run
python viz/export.py \
    --run-dir /path/to/runs/research_stage3_001 \
    --sample 50000

# Launch Streamlit app
streamlit run viz/app.py
```

The export computes full-data aggregate statistics (domain/difficulty/language/source distributions, cross-tab heatmaps, cluster analysis) over all 15M records via Polars and stores them in `viz/data/meta.json`. The Streamlit app reads these directly — no need to load 15M records at browse time.

Add `--umap` to compute UMAP 2D coords for the sampled prompts (slow; requires `umap-learn` and embedding shards).

---

## Configuration

All pipeline parameters live in a single YAML file validated by Pydantic v2. Config files are layered:

| File | Purpose |
|------|---------|
| `config/default.yaml` | Master config — all defaults documented |
| `config/dev.yaml` | Laptop overrides (CPU/CUDA, small scale) |
| `config/prod.yaml` | Cluster overrides (ROCm, 7M prompts, 64 replicas) |
| `config/stage1_research.yaml` | Stage 1 only; 45 public HF research datasets |
| `config/stage3_cluster.yaml` | Stage 3 clustering; distributed embedding + torch_kmeans |
| `config/stage3_annotate.yaml` | Stage 3 annotation only; async LLM calls, no GPU |

Placeholders `{run_id}` and `{base_path}` are resolved throughout string fields at load time.

### Key config fields

```yaml
global:
  run_id: "my_run"
  base_path: "/data/sft-pipeline/runs/{run_id}"
  device: "cuda"        # "cuda" | "rocm" | "cpu"
  hf_home: "/fast-disk/hf_cache"  # override HuggingFace cache directory

stage1_collect:
  datasets:
    - source: hf_dataset
      hf_repo_id: openai/gsm8k
      hf_split: train
      hf_config: main
      prompt_field: question        # plain string field
      domain_hint: math
      max_examples: 1000            # cap rows from this source (omit for all)
    - source: hf_dataset
      hf_repo_id: nvidia/Nemotron-SFT-SWE-v2
      hf_split: agentless
      prompt_field: messages        # OpenAI/ShareGPT conversation → first user turn
    - source: local_jsonl
      path: /data/my-prompts.jsonl
      prompt_field: instruction

stage3_cluster:
  clustering_algorithm: hdbscan    # "hdbscan" | "kmeans" | "flash_kmeans"
  n_clusters: 50                   # used with kmeans and flash_kmeans

stage4_sample:
  total_prompts: 7000000
  domain_quotas: {math: 0.25, code: 0.20, science: 0.20, general: 0.20, language: 0.15}
  difficulty_quotas: {easy: 0.20, medium: 0.50, hard: 0.30}

stage5_inference:
  model: "Qwen/Qwen3.5-122B-A10B"
  n_replicas: 64
  vllm_engine:
    tensor_parallel_size: 8
  generation:
    n_candidates: 2
    max_tokens: 4096
```

### Prompt field formats

`prompt_field` supports:
- **Plain string**: the field value is used directly
- **Nested field** (dot notation): `responses_create_params.input` extracts `row["responses_create_params"]["input"]`
- **OpenAI conversation** (`[{"role": "user", "content": "..."}]`): first user turn extracted; system prompt prepended if present
- **ShareGPT conversation** (`[{"from": "human", "value": "..."}]`): same extraction logic

## Output Format

Each final record:

```json
{
  "id": "sha256:...",
  "prompt": "Solve: x² + 5x + 6 = 0",
  "reasoning": "...",
  "answer": "x = -2 or x = -3",
  "domain": "math",
  "difficulty": "medium",
  "language": "en",
  "source": "gsm8k",
  "teacher_model": "Qwen/Qwen3.5-122B-A10B"
}
```

Output is written as sharded JSONL (configurable shard size, default 500 MB).

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only (heavier; needs local disk)
python -m pytest tests/integration/ -v

# Flash-kmeans tests (requires CUDA + flash-kmeans installed)
python -m pytest tests/unit/clustering/test_clusterer_flash_kmeans.py -v
```

Tests are structured as:
- `tests/unit/` — per-module unit tests (no GPU, no HF downloads, synthetic fixtures)
- `tests/integration/` — end-to-end smoke test + checkpoint resume test

## Project Structure

```
sft-pipeline/
├── config/                      # YAML config files
├── sft_pipeline/
│   ├── config.py                # Pydantic v2 models + YAML loader
│   ├── checkpoint.py            # DuckDB checkpoint tracker
│   ├── storage.py               # ShardedJSONLWriter + JSONL utilities
│   ├── cli.py                   # Typer CLI
│   ├── stages/                  # One module per pipeline stage
│   ├── filters/                 # Stage 6 filter implementations
│   ├── clustering/              # Embedder, FAISS index, clusterer
│   ├── inference/               # vLLM batch loop, prompt formatter, output parser
│   └── export/                  # Final JSONL export + optional HF Hub push
├── tests/
├── scripts/
├── docker/
└── infra/
```

See `CLAUDE.md` for the full developer guide and design rationale.
