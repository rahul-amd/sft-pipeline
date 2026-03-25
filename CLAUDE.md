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

As of 2026-03-14: **Stages 1–6 fully implemented. All 39 tests passing.**

```
tests/unit/          — 37 tests (config, checkpoint, storage, parser, filters)
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
│   └── prod.yaml                ← cluster overrides (ROCm, 7M prompts, 64 replicas)
├── sft_pipeline/
│   ├── config.py                ← Pydantic v2 models + YAML loader
│   ├── checkpoint.py            ← DuckDB checkpoint tracker
│   ├── storage.py               ← JSONL read/write + ShardedJSONLWriter
│   ├── cli.py                   ← Typer CLI (run, run-stage, status, estimate)
│   ├── cost_estimator.py        ← dry-run estimation (GPU-hours, wall clock)
│   ├── stages/
│   │   ├── stage1_collect.py    ← HF dataset ingestion + MinHash LSH dedup
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
│   │   ├── embedder.py          ← batched sentence-transformer → float16 Parquet
│   │   ├── faiss_index.py       ← IVFFlat build/save/load (CPU only)
│   │   └── clusterer.py         ← HDBSCAN on centroids + difficulty heuristics
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
│   │   └── filters/
│   │       ├── test_structural.py
│   │       └── test_math_verifier.py
│   └── integration/
│       └── test_end_to_end.py   ← Stages 4 + mock-5 + 6 smoke test + resume test
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

1. Open `sft_pipeline/stages/stage1_collect.py`
2. Add a new branch in `_load_dataset_source(ds_cfg)` — check `ds_cfg.source`
3. Yield `dict(prompt=..., source=..., domain=...)` records
4. Add the new source to `config.py`'s `DatasetSourceConfig` if it needs new fields

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

- **Windows paths in YAML**: Always use `.as_posix()` when embedding `Path` objects into YAML strings in tests. Windows backslashes in double-quoted YAML strings are treated as escape sequences, causing `yaml.scanner.ScannerError`.

- **`make_response_record` reasoning length**: The structural filter requires `min_response_tokens=50` (default). Test fixtures must produce reasoning above this threshold. `conftest.py`'s `make_response_record` is calibrated to ~80 tokens — don't shorten it.

- **ROCm vLLM wheel**: The standard `pip install vllm` installs the CUDA build. On the cluster, use the ROCm-specific wheel from the vLLM releases page. The `gpu_memory_utilization` and `dtype` settings in `prod.yaml` are tuned for MI250X.

- **Ray cluster connection**: `stage5_inference.py` calls `ray.init(address="auto")` which connects to an existing Ray cluster. Start the cluster first with `ray start --head` on the head node and `ray start --address=<head>` on workers before running Stage 5.

- **DuckDB concurrent access**: DuckDB in WAL mode supports one writer + multiple readers. Don't run two pipeline instances pointing to the same checkpoint DB simultaneously.

- **Stage 3 is idempotent on embeddings**: If `embeddings_dir` already contains any `embeddings_*.parquet` shards, Stage 3 skips re-embedding and goes straight to indexing. Safe to re-run after a crash during FAISS index building.

---

## Running on the Cluster

```bash
# 1. Start Ray cluster (on head node)
ray start --head --num-gpus=16

# 2. Start workers (on each of the other 31 nodes)
ray start --address=<head-node-ip>:6379 --num-gpus=16

# 3. Start vLLM HTTP server for Stage 2 generator + Stage 6 judge
bash scripts/serve_vllm.sh

# 4. Run pipeline (connects to Ray automatically via address="auto")
sft-pipeline run --config config/prod.yaml

# 5. Monitor
sft-pipeline status --config config/prod.yaml
```

## Running on Laptop (dev)

```bash
# CPU-only (no GPU)
sft-pipeline run --config config/dev.yaml

# With RTX 5060 (small Stage 5 smoke run)
# dev.yaml already sets device: cuda, model: Qwen2.5-7B-Instruct, n_replicas: 1
sft-pipeline run-stage stage5_inference --config config/dev.yaml
```

---

## Test Coverage Gaps (known)

- Stage 1 (MinHash dedup): no integration test — needs HF datasets mock
- Stage 2 (corpus chunking): no integration test — needs LLM HTTP mock
- Stage 3 (embedding + FAISS): no integration test — heavy dependencies
- Stage 5 (vLLM): integration test uses mock; no real vLLM test (expected — requires GPU)
- Filters: `heuristic.py`, `code_verifier.py`, `llm_judge.py` have no unit tests yet
