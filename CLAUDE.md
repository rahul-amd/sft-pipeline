# SFT Pipeline — Developer Guide for Claude

This file exists so future Claude sessions can resume work on this project without re-reading all source files. It covers architecture, key design decisions, how to run things, and how to extend the pipeline.

For the original requirements and design rationale, see `docs/requirements-and-plan.md`.

---

## What This Project Does

Builds a 7-stage automated pipeline to produce **5M high-quality prompt–response pairs** for LLM supervised fine-tuning (SFT). A teacher model (Qwen3.5-122B-A10B via vLLM) generates structured reasoning traces (`<think>...</think><answer>...</answer>`) for each prompt. Quality filtering removes ~30%, leaving ~5M examples.

**Production target**: 32-node AMD MI250X cluster (512 GCDs total, ROCm), local disk.
**Dev environment**: Windows 11 laptop, NVIDIA RTX 5060 (CUDA), or CPU-only.
**Stage 7 (translation)**: Deferred to v2 — stub exists but is not implemented.

---

## Project Status

As of 2026-03-25: **Stages 1–6 fully implemented. All 39 tests passing** (flash-kmeans tests skip when GPU/library unavailable).

```
tests/unit/          — 37 tests (config, checkpoint, storage, parser, filters)
tests/unit/clustering/ — flash-kmeans tests (skip without CUDA + flash-kmeans)
tests/integration/   — 2 tests (end-to-end smoke + checkpoint resume)
```

Run tests: `python -m pytest tests/ -v`

---

## Directory Structure

```
sft-pipeline/
├── CLAUDE.md                    ← this file
├── docs/
│   └── requirements-and-plan.md ← original proposal breakdown + design decisions
├── pyproject.toml
├── config/
│   ├── default.yaml             ← master config (all defaults documented here)
│   ├── dev.yaml                 ← laptop overrides (CPU/CUDA, small scale)
│   ├── prod.yaml                ← cluster overrides (ROCm, 7M prompts, 64 replicas)
│   └── stage1_research.yaml    ← Stage 1 only; 45 public HF research datasets
├── sft_pipeline/
│   ├── config.py                ← Pydantic v2 models + YAML loader
│   ├── checkpoint.py            ← DuckDB checkpoint tracker
│   ├── storage.py               ← JSONL read/write + ShardedJSONLWriter
│   ├── cli.py                   ← Typer CLI (run, run-stage, status, estimate)
│   ├── cost_estimator.py        ← dry-run estimation (GPU-hours, wall clock)
│   ├── stages/
│   │   ├── stage1_collect.py    ← HF dataset ingestion + SHA256 exact dedup
│   │   ├── stage2_generate.py   ← corpus chunking + LLM prompt generation
│   │   ├── stage3_cluster.py    ← embed + FAISS index + HDBSCAN + difficulty
│   │   ├── stage4_sample.py     ← quota sampling, near-dedup, similarity sort
│   │   ├── stage5_inference.py  ← vLLM batch inference (single-node + Ray)
│   │   ├── stage6_filter.py     ← quality filter pipeline + filter_report.json
│   │   └── stage7_translate.py  ← stub only (v2)
│   ├── filters/
│   │   ├── structural.py        ← missing fields, length bounds, repetition loops
│   │   ├── heuristic.py         ← type-token ratio, boilerplate, contradiction
│   │   ├── math_verifier.py     ← SymPy LaTeX parse + numeric consistency
│   │   ├── code_verifier.py     ← subprocess / E2B sandbox execution
│   │   └── llm_judge.py         ← 10%-sample LLM quality scoring (Qwen2.5-7B)
│   ├── clustering/
│   │   ├── embedder.py          ← batched sentence-transformer → sharded float16 Parquet
│   │   ├── faiss_index.py       ← IVFFlat build/save/load (CPU only)
│   │   └── clusterer.py         ← HDBSCAN/K-Means/flash-kmeans + difficulty heuristics
│   ├── inference/
│   │   ├── vllm_batch.py        ← offline vLLM batch loop + Ray actor class
│   │   ├── prompt_formatter.py  ← chat template application
│   │   └── output_parser.py     ← <think>/<answer> extraction with fallbacks
│   ├── translation/             ← v2 stubs
│   │   ├── segment_parser.py
│   │   ├── deepl_client.py
│   │   └── google_translate_client.py
│   └── export/
│       ├── jsonl_writer.py      ← final schema normalization + JSONL export
│       └── hf_exporter.py       ← optional HF Hub push (disabled in v1)
├── scripts/
│   ├── run_pipeline.py          ← end-to-end runner script
│   ├── estimate_cost.py         ← dry-run report
│   └── serve_vllm.sh            ← start vLLM HTTP server (Stage 2 + Stage 6 judge)
├── tests/
│   ├── conftest.py              ← shared fixtures + make_prompt_record/make_response_record
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_checkpoint.py
│   │   ├── test_storage.py
│   │   ├── test_output_parser.py
│   │   ├── filters/
│   │   │   ├── test_structural.py
│   │   │   └── test_math_verifier.py
│   │   └── clustering/
│   │       └── test_clusterer_flash_kmeans.py  ← skip without CUDA + flash-kmeans
│   └── integration/
│       └── test_end_to_end.py   ← Stages 4 + mock-5 + 6 smoke test + resume test
├── viz/
│   ├── README.md
│   ├── export.py            ← snapshot export CLI (run after each stage completes)
│   ├── app.py               ← Streamlit entry point
│   ├── pages/
│   │   ├── 1_Stats.py       ← domain/source/difficulty charts
│   │   ├── 2_Prompts.py     ← searchable prompt table
│   │   ├── 3_Clusters.py    ← UMAP scatter plot
│   │   └── 4_Answers.py     ← prompt+reasoning+answer viewer
│   ├── components/
│   │   ├── data_loader.py   ← st.cache_data snapshot loader (mtime-busted)
│   │   └── filters.py       ← shared sidebar filter widgets
│   ├── data/                ← snapshot.parquet + meta.json (gitignored)
│   └── requirements.txt     ← viz-only deps (separate from pipeline)
├── docker/
│   ├── Dockerfile.pipeline
│   └── Dockerfile.vllm
└── infra/
    └── ray_cluster.yaml
```

---

## CLI Usage

```bash
# Install (dev mode)
pip install -e ".[dev]"

# Full run
sft-pipeline run --config config/prod.yaml

# Resume after failure (reads DuckDB; skips completed stages)
sft-pipeline run --config config/prod.yaml

# Run a single stage
sft-pipeline run-stage stage4_sample --config config/prod.yaml

# Check stage status
sft-pipeline status --config config/prod.yaml

# Dry-run (no inference, estimates GPU-hours + wall clock)
sft-pipeline estimate --config config/prod.yaml
```

---

## Config System

Config is loaded via `sft_pipeline.config.load_config(path, overrides=None)`.

- `default.yaml` is the base; `dev.yaml` / `prod.yaml` are override layers (deep-merged)
- `{run_id}` and `{base_path}` placeholders are resolved in string fields after merge
- Pydantic v2 validation runs at load time — type errors surface before any compute starts
- `domain_quotas` and `difficulty_quotas` must each sum to 1.0 (validated)

**Two-pass placeholder resolution**: Pydantic applies model defaults only after `model_validate()`, so fields not present in the YAML (i.e. using defaults) would miss placeholder substitution in a single pass. `load_config()` therefore does:
1. Pass 1: resolve placeholders in the raw YAML dict
2. `model_validate()` to get Pydantic defaults applied
3. Pass 2: `model_dump(by_alias=True)` → resolve again → `model_validate()` again

Both `{base_path}` and `{global.base_path}` are registered as context keys so either syntax works in YAML.

**Notable `GlobalConfig` fields:**
- `hf_home: str | None` — if set, `HF_HOME` env var is set before any HF downloads (useful to avoid filling up the default `~/.cache/huggingface` on small partitions)
- `device: "cuda" | "rocm" | "cpu"` — controls GPU backend; `"rocm"` is mapped to `"cuda"` internally where PyTorch uses it

**Notable `DatasetSource` fields:**
- `max_examples: int | None` — cap rows loaded from this source (`None` = all); applied via `itertools.islice` before normalization/dedup
- `prompt_field` supports **dot notation** for nested fields: `"responses_create_params.input"` → `row["responses_create_params"]["input"]`

**To use prod config on cluster:**
```bash
sft-pipeline run --config config/prod.yaml
```

**To override individual values at CLI:**
```bash
sft-pipeline run --config config/prod.yaml --set stage5_inference.n_replicas=32
```
(The `--set` flag passes dot-path overrides as a dict into `load_config(overrides={...})`.)

---

## Data Flow

```
Stage 1  →  JSONL of prompts with {prompt_id, prompt, domain, source}
Stage 2  →  additional JSONL prompts from corpus (same schema)
Stage 3  →  same JSONL + {cluster_id, difficulty} labels
             + embeddings/ (sharded Parquet dir) + faiss.index (side outputs)
Stage 4  →  sampled subset JSONL (~7M prompts)
Stage 5  →  JSONL of responses {prompt_id, prompt, reasoning, answer, domain, difficulty}
Stage 6  →  filtered JSONL (same schema) + filter_report.json
Export   →  final sharded JSONL with canonical schema (see below)
```

**Final record schema:**
```json
{
  "id": "sha256:...",
  "prompt": "...",
  "reasoning": "...",
  "answer": "...",
  "domain": "math",
  "difficulty": "medium",
  "language": "en",
  "source": "gsm8k",
  "teacher_model": "Qwen/Qwen3.5-122B-A10B"
}
```

---

## Key Design Decisions

### DuckDB checkpointing (`checkpoint.py`)
Every stage calls `cm.is_processed(prompt_id, stage_name)` before processing each item and `cm.mark_processed(prompt_id, stage_name)` after. On crash/restart, the stage rebuilds its "already done" set from DuckDB (`preload_processed()`) and skips those items in the stream. The DB is a single file (`checkpoints.duckdb`) — easy to copy or inspect with any DuckDB client.

### FAISS CPU only (`faiss_index.py`)
FAISS-GPU requires CUDA and won't build on ROCm. `faiss-cpu` works on both laptop and cluster. At 7M × 1024-dim float16 vectors the CPU IVFFlat index is fast enough for ANN lookup.

### HDBSCAN on centroids, not all vectors (`clusterer.py`)
Direct HDBSCAN on 7M vectors is O(n²) — infeasible. Strategy: extract IVF centroids (~1000 points) from the FAISS index, run HDBSCAN on those, then assign each prompt to its nearest centroid's cluster label. Fast and scales to any dataset size.

### flash-kmeans clustering (`clusterer.py`, `stage3_cluster.py`)
Set `clustering_algorithm: "flash_kmeans"` in `Stage3Config` to use GPU-accelerated K-Means via Triton kernels (`pip install flash-kmeans`). Requires CUDA. The flash-kmeans path bypasses the FAISS centroid extraction entirely — it works directly on the full embedding matrix. The FAISS index is still built (needed by Stage 4 for cosine dedup), just not used for clustering in this mode.

API: `batch_kmeans_Euclid(x, n_clusters, tol, verbose)` where `x` is `(1, N, D)` float16 CUDA tensor. Returns `(cluster_ids, centers, _)` with a batch dimension that is squeezed after the call.

Tests in `tests/unit/clustering/test_clusterer_flash_kmeans.py` skip automatically when CUDA or flash-kmeans is unavailable (`@pytest.mark.skipif` markers: `needs_cuda`, `needs_flash`, `needs_cuda_and_flash`).

### Sharded embeddings (`embedder.py`)
Embeddings are saved as sharded float16 Parquet files under `embeddings_dir/`, not a single file. Each shard is `~200MB`. Stage 3 checks for existing shards at startup and skips re-embedding if any shards are found (idempotent on crash). Stage 4 loads shards lazily via Polars `scan_parquet`.

### `get_centroids()` implementation (`faiss_index.py`)
`index.quantizer.xb` (a SWIG-internal attribute) is not available in newer `faiss-cpu` versions. Use `index.quantizer.reconstruct(i)` for each centroid `i` instead — this is the stable public API:
```python
centroids = np.stack([index.quantizer.reconstruct(i) for i in range(nlist)])
```

### ROCm/CUDA abstraction (`embedder.py`, `vllm_batch.py`)
PyTorch uses `"cuda"` as the device string for both CUDA and ROCm. Config uses `device: "rocm"` for clarity; code maps `"rocm"` → `"cuda"` internally when passing to PyTorch. vLLM ROCm build is a separate wheel — install separately on cluster.

### vLLM offline batch mode (`stage5_inference.py`, `vllm_batch.py`)
Uses `vllm.LLM` (offline) instead of the HTTP server. Zero round-trip overhead for millions of requests; direct access to `RequestOutput` for best-of-n candidate selection. Multi-replica scaling via Ray actors (`build_ray_actor_class()`); each actor holds a persistent `LLM` instance.

### Polars for Stage 4 sampling (`stage4_sample.py`)
At 7M prompts, Pandas would materialize tens of GB. Polars lazy evaluation keeps RAM usage low; 5–20x faster for groupby/quota enforcement.

### ShardedJSONLWriter (`storage.py`)
Auto-splits output at configurable `shard_size_mb` boundary. Accepts an `on_shard_complete` callback so each completed shard can be registered in DuckDB's `shard_manifest` table.

### Stage 6 filter ordering
Filters run cheapest-first: structural → heuristic → math → code → llm_judge. Each filter is short-circuit: once a record fails, it is not passed to more expensive filters. The LLM judge only sees a 10% sample (`sample_rate: 0.10`).

### SymPy "uncertain" vs "fail" (`math_verifier.py`)
SymPy can fail to parse valid LaTeX (e.g., custom macros, non-standard notation). Parse failures return `FilterResult(passed=True, reason="uncertain")` — don't reject what the parser can't understand. Only structural inconsistencies (answer numbers absent from reasoning) cause rejection.

---

## Adding a New Dataset Source (Stage 1)

Add a new entry under `stage1_collect.datasets` in the YAML config. No code change needed for standard HF datasets — just specify:
- `source: hf_dataset`, `hf_repo_id`, `hf_split`, and optionally `hf_config`
- `prompt_field`: the field containing the prompt. Use dot notation for nested fields (e.g. `outer_field.inner_field`). If the field holds an OpenAI or ShareGPT conversation list, the first user turn is extracted automatically.
- `domain_hint`: optional manual domain override (otherwise inferred from keywords)
- `max_examples`: optional cap on rows loaded from this source

For a new source type (not `hf_dataset` or `local_jsonl`):
1. Add a new `Literal` value to `DatasetSource.source` in `config.py`
2. Add a new loader function `_load_<type>(src)` in `stage1_collect.py`
3. Add a dispatch branch in `run_stage1()`

## Adding a New Filter (Stage 6)

1. Create `sft_pipeline/filters/my_filter.py` with a `check_my_filter(record, cfg) -> FilterResult` function
2. Add config fields to `Stage6FilterConfig` in `config.py`
3. In `stage6_filter.py`, insert the call in the filter chain (respect cost-ordering)
4. Add unit tests in `tests/unit/filters/test_my_filter.py`

## Adding a New Stage

1. Create `sft_pipeline/stages/stageN_name.py` with `run_stageN(cfg, cm)` signature
2. Add config model to `config.py` and include it in `PipelineConfig`
3. Register the stage in `cli.py`'s stage dispatch dict
4. Add integration test in `tests/integration/`

---

## Gotchas / Non-Obvious Things

- **flash-kmeans CUDA check ordering**: In `_cluster_with_flash_kmeans()`, the `torch.cuda.is_available()` check must come **before** `from flash_kmeans import batch_kmeans_Euclid`. If the import happens first and CUDA is unavailable, the import itself may fail with an unhelpful error rather than the intended RuntimeError.

- **`get_centroids()` and newer faiss-cpu**: `index.quantizer.xb` is not available in recent `faiss-cpu` versions. Always use `index.quantizer.reconstruct(i)` per centroid. The `downcast_index()` approach is also fragile — avoid it.

- **Windows paths in YAML**: Always use `.as_posix()` when embedding `Path` objects into YAML strings in tests. Windows backslashes in double-quoted YAML strings are treated as escape sequences, causing `yaml.scanner.ScannerError`.

- **`make_response_record` reasoning length**: The structural filter requires `min_response_tokens=50` (default). Test fixtures must produce reasoning above this threshold. `conftest.py`'s `make_response_record` is calibrated to ~80 tokens — don't shorten it.

- **ROCm PyTorch + environment setup**: The cluster runs ROCm 6.3.4. The default `pip install torch` gives a CUDA build that fails with "Found no NVIDIA driver" on AMD GPUs. Run `scripts/setup_env.sh` once (with the overlay mounted `:rw`) to install ROCm PyTorch, sentence-transformers, Triton, and flash-kmeans correctly. All Slurm scripts bind `/opt/rocm` so the ROCm runtime `.so` libraries are visible inside the container. Stage 3 also runs a single GPU pre-flight Ray task before dispatching all 32 workers; if CUDA is unavailable you'll see one clear error with the fix rather than 32 identical failures.

- **Singularity GPU binds — three are required, plus `LD_LIBRARY_PATH`**: For ROCm PyTorch to see the GPU inside a container you must bind all three device paths AND export the library search path:
  | Bind | What it provides |
  |------|-----------------|
  | `--bind /opt/rocm` | ROCm runtime libraries (`libamdhip64.so` etc.) |
  | `--bind /dev/kfd` | AMD KFD kernel driver — `torch.cuda.is_available()` needs this |
  | `--bind /dev/dri` | DRI render devices (`renderD128` etc.) |

  Binding `/opt/rocm` makes the files visible inside the container, but Singularity does **not** automatically add `/opt/rocm/bin` to `PATH` or `/opt/rocm/lib` to `LD_LIBRARY_PATH`. Without the latter, `torch` loads but silently fails to open `libamdhip64.so` and `torch.cuda.is_available()` returns `False`. `run_in_env.sh` sets all three env vars automatically; if you run a custom singularity command, set them manually:
  ```bash
  export PATH="/opt/rocm/bin:$PATH"
  export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
  export HSA_OVERRIDE_GFX_VERSION=9.0.0   # MI250X (gfx90a)
  ```
  Missing `/opt/rocm` → "Found no NVIDIA driver". Missing binds or `LD_LIBRARY_PATH` → `is_available()` returns False. All `SING=` lines in Slurm scripts include all three binds; `run_in_env.sh` sets the env vars. For interactive sessions:
  ```bash
  singularity shell \
      --bind /scratch/project_462000963 \
      --bind /users/aralikatte \
      --bind /opt/rocm --bind /dev/kfd --bind /dev/dri \
      --overlay .../python_latest_overlay.img \
      .../python_latest.sif
  # Then inside the container:
  export PATH="/opt/rocm/bin:$PATH"
  export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
  export HSA_OVERRIDE_GFX_VERSION=9.0.0
  ```

- **ROCm vLLM wheel**: Similarly, the standard `pip install vllm` installs the CUDA build. On the cluster, use the ROCm-specific wheel from the vLLM releases page. The `gpu_memory_utilization` and `dtype` settings in `prod.yaml` are tuned for MI250X.

- **Ray cluster connection**: `stage5_inference.py` calls `ray.init(address="auto")` which connects to an existing Ray cluster. Start the cluster first with `ray start --head` on the head node and `ray start --address=<head>` on workers before running Stage 5.

- **DuckDB concurrent access**: DuckDB in WAL mode supports one writer + multiple readers. Don't run two pipeline instances pointing to the same checkpoint DB simultaneously.

- **Stage 3 is idempotent on embeddings**: If `embeddings_dir` already contains any `embeddings_*.parquet` shards, Stage 3 skips re-embedding and goes straight to indexing. Safe to re-run after a crash during FAISS index building.

- **`{global.base_path}` vs `{base_path}`**: Both resolve to the same value. Prefer `{base_path}` in YAML files — it's shorter and guaranteed to work with both pass 1 and pass 2 of placeholder resolution. `{global.base_path}` also works but is only needed if you have a value that needs to match config YAML key paths exactly.

- **`stage1_research.yaml` and `lmsys/lmsys-chat-1m`**: This dataset requires accepting a licence on HuggingFace. Set `HF_TOKEN` env var before running if you haven't accepted it, otherwise the dataset download will fail with an authorization error.

---

## Running on the Cluster

```bash
# 1. Start Ray cluster (on head node)
#    --num-cpus controls how many concurrent Ray tasks can run on this node.
#    Stage 1 tasks use num_cpus=2 each, so 16 CPUs → 8 concurrent tasks/node.
ray start --head --num-cpus=32 --num-gpus=16

# 2. Start workers (on each of the other 31 nodes)
ray start --address=<head-node-ip>:6379 --num-cpus=32 --num-gpus=16

# 3. Start vLLM HTTP server for Stage 2 generator + Stage 6 judge
bash scripts/serve_vllm.sh

# 4. Run pipeline (Stage 1 will distribute sources across all 32 nodes)
sft-pipeline run --config config/prod.yaml

# 5. Monitor
sft-pipeline status --config config/prod.yaml
```

### Stage 1 Deduplication

Both single-node and distributed modes use **SHA256 exact dedup only** — `prompt_id` is the SHA256 of the normalised prompt text and is checked against an in-memory `set[str]`. This is O(1) per record and doesn't degrade with corpus size.

Near-duplicate dedup (semantically similar but not identical prompts) is deliberately deferred to Stage 4, which has embeddings and FAISS cosine similarity available and can do it far more accurately. MinHash LSH was previously used here but caused progressive throughput collapse at scale (datasketch LSH `query()` slows as bucket tables grow).

### Stage 1 Distributed Mode

When `stage1_collect.distributed: true` in config (set in `prod.yaml`), Stage 1 runs in two phases:

**Phase 1 — parallel collection (all nodes)**
Each dataset source becomes a Ray remote task (`num_cpus=2`). Ray distributes tasks across all nodes. Each task streams its source, normalises, SHA256-deduplicates within the source, and writes to `{output_dir}/_phase1/{source_slug}.jsonl` on the shared filesystem. Tasks are idempotent — if the output file exists, the source is skipped on resume.

**Phase 2 — merge + cross-source dedup (head node)**
After all tasks complete, the head node reads all phase1 files, deduplicates by `prompt_id` across sources, writes the final sharded output, and updates DuckDB. This is a simple set-based pass — throughput is limited only by disk I/O.

**Speedup**: `sum(source stream times)` → `max(source stream times)` + fast merge pass.

**Resume**: If Phase 1 crashes, re-run the command. Completed sources (`.jsonl` file exists in `_phase1/`) are skipped automatically. If Phase 2 crashes, re-run — phase1 files are still there.

**Laptop / single-node**: Leave `distributed: false` (default). `dev.yaml` does not set it.

## Running on Laptop (dev)

```bash
# CPU-only (no GPU)
sft-pipeline run --config config/dev.yaml

# With RTX 5060 (small Stage 5 smoke run)
# dev.yaml already sets device: cuda, model: Qwen2.5-7B-Instruct, n_replicas: 1
sft-pipeline run-stage stage5_inference --config config/dev.yaml
```

---

## Visualization App (`viz/`)

See `viz/README.md` for full usage. Quick reference:

```bash
# 1. Install viz deps (separate from pipeline)
pip install -r viz/requirements.txt

# 2. Export snapshot after any stage completes
python viz/export.py --run-dir /path/to/run          # default: 50k sample
python viz/export.py --run-dir /path/to/run --sample 100000

# 3. Launch
streamlit run viz/app.py

# 4. Share
cloudflared tunnel --url http://localhost:8501
```

**How export.py discovers data:**
- Looks for `{run_dir}/stage3/part-*.jsonl` first (enriched prompts with domain/difficulty/cluster_id)
- Falls back to `{run_dir}/stage1/part-*.jsonl` if Stage 3 not done
- Loads `{run_dir}/stage3/embeddings/embeddings_*.parquet` and runs UMAP if present
- Joins Stage 5 responses and Stage 6 filter results if present
- Writes `viz/data/snapshot.parquet` + `viz/data/meta.json`

**Cache invalidation**: `data_loader.py` passes `snapshot.parquet`'s mtime as an argument to `@st.cache_data`, so re-running export automatically invalidates the cache on next app load. No restart needed.

**`viz/data/` is gitignored** — snapshots are not committed.

---

## Test Coverage Gaps (known)

- Stage 1 (dedup): no integration test — needs HF datasets mock
- Stage 2 (corpus chunking): no integration test — needs LLM HTTP mock
- Stage 3 (embedding + FAISS): no integration test — heavy dependencies
- Stage 5 (vLLM): integration test uses mock; no real vLLM test (expected — requires GPU)
- Filters: `heuristic.py`, `code_verifier.py`, `llm_judge.py` have no unit tests yet
