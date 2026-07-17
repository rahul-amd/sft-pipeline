# SFT Pipeline — Developer Guide for Claude

This file exists so future Claude sessions can resume work on this project without re-reading all source files. It covers architecture, key design decisions, how to run things, and how to extend the pipeline.

For the original requirements and design rationale, see `docs/requirements-and-plan.md`.

---

## What This Project Does

Builds a 7-stage automated pipeline to produce **5M high-quality prompt–response pairs** for LLM supervised fine-tuning (SFT). A teacher model (Qwen3.5-122B-A10B via vLLM) generates structured reasoning traces (`<think>...</think><answer>...</answer>`) for each prompt. Quality filtering removes ~30%, leaving ~5M examples.

**Production target**: 32-node AMD MI250X cluster (512 GCDs total, ROCm), local disk.
**Dev environment**: Windows 11 laptop, NVIDIA RTX 5060 (CUDA), or CPU-only.
**Stage 7 (translation)**: Deferred to v2 — not yet implemented.

---

## Project Status

As of 2026-04-28: **Stages 1–3 fully run in production. Stages 4–6 implemented but not yet run.**

- Stage 1 complete: `runs/research_stage1_001d/` — 14.8M prompts from 45 HF datasets
- Stage 3 complete: `runs/research_stage3_001/` — embeddings, 100k clusters, LLM annotations imported
- Stages 4–6 ready to run; Stage 4 pending domain distribution review via viz app

All 77 tests passing (flash-kmeans tests skip when GPU/library unavailable):
```
tests/unit/          — 64 tests (config, checkpoint, storage, parser, filters, stage1_extract)
tests/unit/clustering/ — 11 flash-kmeans tests (skip without CUDA + flash-kmeans)
tests/integration/   — 2 tests (end-to-end smoke + checkpoint resume)
```

Run tests: `python -m pytest tests/ -v`

---

## Directory Structure

```
sft-pipeline/
├── CLAUDE.md                    ← this file
├── README.md
├── requirements.txt             ← top-level pip requirements
├── pyproject.toml
├── docs/
│   └── requirements-and-plan.md ← original proposal breakdown + design decisions
├── config/
│   ├── default.yaml             ← master config (all defaults documented here)
│   ├── dev.yaml                 ← laptop overrides (CPU/CUDA, small scale)
│   ├── prod.yaml                ← cluster overrides (ROCm, 7M prompts, 64 replicas)
│   ├── stage1_research.yaml     ← Stage 1 only; 45 public HF research datasets
│   ├── stage3_cluster.yaml      ← Stage 3 cluster run config (distributed embedding + clustering, annotation_enabled: false)
│   ├── stage3_annotate.yaml     ← Stage 3 annotation-only run (async LLM calls, CPU, no GPU)
│   └── decontaminate.yaml       ← eval decontamination run config (standalone, CPU)
├── sft_pipeline/
│   ├── config.py                ← Pydantic v2 models + YAML loader
│   ├── checkpoint.py            ← DuckDB checkpoint tracker
│   ├── storage.py               ← JSONL read/write + ShardedJSONLWriter
│   ├── cli.py                   ← Typer CLI (run, run-stage, status, estimate)
│   ├── cost_estimator.py        ← dry-run estimation (GPU-hours, wall clock)
│   ├── stages/
│   │   ├── stage1_collect.py    ← HF dataset ingestion + SHA256 exact dedup
│   │   ├── stage2_generate.py   ← corpus chunking + LLM prompt generation
│   │   ├── decontaminate.py     ← eval decontamination (runs before Stage 3)
│   │   ├── stage3_cluster.py    ← embed + FAISS index + HDBSCAN + difficulty
│   │   ├── stage4_sample.py     ← quota sampling, near-dedup, similarity sort
│   │   ├── stage5_inference.py  ← vLLM batch inference (single-node + Ray)
│   │   └── stage6_filter.py     ← quality filter pipeline + filter_report.json
│   ├── decontam/
│   │   ├── normalize.py         ← aggressive match-time tokenizer (NFKC/lower/strip-punct)
│   │   └── eval_index.py        ← eval loading + 13-word-gram containment index
│   ├── filters/
│   │   ├── structural.py        ← missing fields, length bounds, repetition loops
│   │   ├── heuristic.py         ← type-token ratio, boilerplate, contradiction
│   │   ├── math_verifier.py     ← SymPy LaTeX parse + numeric consistency
│   │   ├── code_verifier.py     ← subprocess / E2B sandbox execution
│   │   └── llm_judge.py         ← 10%-sample LLM quality scoring (Qwen2.5-7B)
│   ├── clustering/
│   │   ├── embedder.py          ← batched sentence-transformer → sharded float16 Parquet
│   │   ├── faiss_index.py       ← IVFFlat build/save/load (CPU only)
│   │   ├── clusterer.py         ← HDBSCAN/K-Means/flash-kmeans + difficulty heuristics
│   │   └── annotator.py         ← async LLM annotation (domain/difficulty/topics/language)
│   ├── inference/
│   │   ├── vllm_batch.py        ← offline vLLM batch loop + Ray actor class
│   │   ├── prompt_formatter.py  ← chat template application
│   │   └── output_parser.py     ← <think>/<answer> extraction with fallbacks
│   └── export/
│       └── jsonl_writer.py      ← final schema normalization + JSONL export
├── scripts/
│   ├── setup_env.sh             ← one-time ROCm PyTorch + deps install (cluster)
│   ├── run_in_env.sh            ← singularity exec entrypoint (activates conda env)
│   ├── install_flash_kmeans.sh  ← flash-kmeans install with ROCm/Triton support
│   ├── slurm_stage1.sh          ← Slurm batch: Stage 1 distributed collection
│   ├── slurm_stage3.sh          ← Slurm batch: Stage 3 distributed embedding + clustering
│   ├── slurm_stage3_annotate.sh ← Slurm batch: Stage 3 annotation-only (CPU node, no GPU)
│   └── test_ray.sh              ← minimal Ray smoke test (single node)
├── vllm/
│   ├── README.md                ← vLLM ROCm SIF usage guide
│   ├── build_sif.sh             ← build Singularity SIF from rocm/vllm Docker image
│   ├── serve.sh                 ← start vLLM server from SIF (interactive or via Slurm)
│   ├── slurm_serve.sh           ← Slurm batch wrapper around serve.sh (single node)
│   ├── slurm_serve_array.sh     ← job array: one vLLM worker per task; task 0 runs nginx
│   └── slurm_nginx.sh           ← standalone nginx coordinator (alternative/fallback)
├── tests/
│   ├── conftest.py              ← shared fixtures + make_prompt_record/make_response_record
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_checkpoint.py
│   │   ├── test_storage.py
│   │   ├── test_output_parser.py
│   │   ├── test_stage1_extract.py  ← _extract_prompt helper (OpenAI/ShareGPT/plain)
│   │   ├── filters/
│   │   │   ├── test_structural.py
│   │   │   └── test_math_verifier.py
│   │   ├── decontam/
│   │   │   ├── test_normalize.py    ← match-time tokenizer
│   │   │   └── test_eval_index.py   ← 13-gram hit, short-item fallback, attribution, fields
│   │   └── clustering/
│   │       └── test_clusterer_flash_kmeans.py  ← skip without CUDA + flash-kmeans
│   └── integration/
│       ├── test_end_to_end.py   ← Stages 4 + mock-5 + 6 smoke test + resume test
│       └── test_decontaminate.py ← plant-contamination, clean pool, report, resume, stage3 wiring
└── viz/
    ├── README.md
    ├── export.py            ← snapshot export CLI (run after each stage completes)
    ├── app.py               ← Streamlit entry point
    ├── .streamlit/
    │   └── config.toml      ← Streamlit server config
    ├── pages/
    │   ├── 1_Stats.py       ← full-data distributions, cross-tab heatmaps, topics-by-domain
    │   ├── 2_Prompts.py     ← searchable prompt table
    │   ├── 3_Clusters.py    ← cluster size histogram, clusters-per-domain, top-clusters table
    │   └── 4_Answers.py     ← prompt+reasoning+answer viewer
    ├── components/
    │   ├── data_loader.py   ← st.cache_data snapshot loader (mtime-busted)
    │   ├── filters.py       ← shared sidebar filter widgets
    │   └── theme.py         ← shared visual theme (call apply_theme() on each page)
    ├── data/                ← snapshot.parquet + meta.json (gitignored)
    └── requirements.txt     ← viz-only deps (separate from pipeline)
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
Decontam →  clean pool (same schema) with eval-overlapping prompts removed
             + removed/ + decontam_report.json (side outputs)
             [runs only if decontaminate.enabled and evals set; Stage 3
              auto-reads this pool when present, else stage1/stage2]
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

**`mark_processed_batch` must use a bulk insert, not `executemany`.** The batch flush does ONE vectorized `INSERT OR REPLACE ... SELECT` from a registered pyarrow table. Do **not** revert it to a row-by-row `executemany` with `ON CONFLICT`: a per-row upsert against the `processed_items` primary-key index costs O(rows × log(table)) and degrades badly as the table grows — measured at **~370s per 100k rows** once the table holds ~1.2M rows (bulk path: **~0.15s**, flat across table size). The row-by-row version froze Stage 6 for ~11 minutes on every 100k checkpoint flush. The batch is deduped in Python first (a single upsert statement can't touch the same PK twice). Semantics vs the old path: `INSERT OR REPLACE` rewrites the whole row (`error_msg` → the batch value, always `None`; `processed_at` → `now()`), whereas the old `DO UPDATE` left `error_msg` intact — harmless because the batch path never carries an error message.

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
PyTorch uses `"cuda"` as the device string for both CUDA and ROCm. Config uses `device: "rocm"` for clarity; code maps `"rocm"` → `"cuda"` internally when passing to PyTorch. On the cluster, vLLM runs from a Singularity SIF (see `vllm/`) rather than a pip wheel.

### vLLM offline batch mode (`stage5_inference.py`, `vllm_batch.py`)
Uses `vllm.LLM` (offline) instead of the HTTP server. Zero round-trip overhead for millions of requests; direct access to `RequestOutput` for best-of-n candidate selection. Multi-replica scaling via Ray actors (`build_ray_actor_class()`); each actor holds a persistent `LLM` instance.

### Polars for Stage 4 sampling (`stage4_sample.py`)
At 7M prompts, Pandas would materialize tens of GB. Polars lazy evaluation keeps RAM usage low; 5–20x faster for groupby/quota enforcement.

### ShardedJSONLWriter (`storage.py`)
Auto-splits output at configurable `shard_size_mb` boundary. Accepts an `on_shard_complete` callback so each completed shard can be registered in DuckDB's `shard_manifest` table.

### Decontamination stage (`stages/decontaminate.py`, `decontam/`)
Runs **between Stage 2 and Stage 3** (stage key `decontaminate`, not renumbered — existing `stageN` run dirs are untouched). Removes any collected/generated prompt that overlaps a downstream eval, so we never train on benchmark questions.

- **Matching**: 13 word-gram containment (`decontam/eval_index.py`). Each eval item's `match_fields` text is tokenized and every contiguous 13-gram is indexed; items **shorter** than 13 tokens index a single gram of their own length (this is the exact whole-item-containment fallback, folded into the same mechanism). A prompt is contaminated if **any** of its grams collides — first hit wins, and the owning eval + matched span are recorded. Aggressive by design: a missed eval leak is worse than dropping a few extra prompts.
- **Normalization** (`decontam/normalize.py`): NFKC → lowercase → drop punctuation/underscore (`[\W_]+`, Unicode-aware so CJK/accents survive) → split. Deliberately more aggressive than the Stage 1 `prompt_id` normalizer (which keeps case/punctuation).
- **Grams are keyed by STRING, not `hash()`** — CPython's `hash()` is per-process randomized, so precomputed int keys would silently miss in `spawn` workers. A string-keyed dict rehashes correctly in every process.
- **Field types** (`extract_field_text`): plain string, dot-notation nested, list→space-joined (e.g. MMLU `choices`), and OpenAI/ShareGPT message list→first-user-turn (reuses Stage 1's `_extract_prompt`).
- **Eval config**: `EvalDatasetSource` supports `hf_configs: "all"` (auto-expand every config via `get_dataset_config_names`) vs a list; `splits` defaults `[test, validation]`; plus `local_jsonl`.
- **Output**: writes a **clean survivor pool** to `decontaminate.output_dir` (one output shard per input shard, `{stage1,stage2}-<name>.jsonl`), an **uncapped** `removed/` dir (`prompt_id, source, matched_eval, matched_ngram`), and `decontam_report.json` (per-eval + per-source removal counts). `_state/shard_stats.jsonl` is the resume ledger — a shard counts as done only after atomic `.tmp`→rename, so re-runs redo at most one partial shard. State/removed live in subdirs so Stage 3's top-level `*.jsonl` glob never reads them.
- **Execution**: single-node `ProcessPoolExecutor`, completion-order bounded window (same pattern as Stage 6). The eval index is shared to workers via **fork copy-on-write** on Linux (module global, no pickling); pickled via the initializer on spawn/forkserver. Matching is deterministic → parallel output == serial (tested).
- **Stage 3 wiring**: `stage3_cluster._resolve_input_dirs(cfg)` returns `[decontam_dir]` if it exists & non-empty, else `[stage1, stage2]`. So Stage 3 auto-uses the clean pool when present and is unchanged otherwise. Enabled-but-no-evals = the stage is a skip/no-op and Stage 3 falls back.
- **Run gating** (`cli.py`): the `run` command runs it only when `enabled and evals` are set.

### Stage 6 filter ordering
Filters run cheapest-first: structural → heuristic → math → code → llm_judge. Each filter is short-circuit: once a record fails, it is not passed to more expensive filters. The LLM judge only sees a 10% sample (`sample_rate: 0.10`).

### Stage 6 parallelism (`stage6_filter.py`)
The filter chain is CPU/subprocess-bound (code sandbox dominates) and each record is independent, so it fans out across processes. Set `stage6_filter.n_workers` (`null` → `os.cpu_count()`, `1` → the original serial loop). `_iter_filtered()` drives it:
- A `ProcessPoolExecutor` with an **initializer** that stashes the config in worker globals once (no per-record re-serialization).
- Work is **chunked** (`worker_chunk_size`, default 32) to amortize IPC. Larger chunks amortize more but hurt load balance when a chunk contains a slow code snippet that hits the sandbox timeout.
- Results are yielded in **completion order** via `concurrent.futures.wait(FIRST_COMPLETED)`, with a bounded in-flight window (`n_workers * 2` chunks) topped up one-for-one as chunks finish. This is deliberate: completion order means one slow chunk ties up only its own worker while the rest keep producing. **Do not** revert to strict input-order yielding (`popleft().result()`) — that head-of-line-blocks the whole pipeline on the slowest chunk and drains the pool (observed as a multi-minute stall). Also do **not** use `executor.map`, which eagerly submits every task and blows memory at 7M records.
- Consequence: Stage 6 output shards are in completion order, not input order. Harmless for a filter stage (only the per-record pass/reject matters), but don't rely on Stage 6 output ordering downstream.
- The main process keeps all state: checkpoint DB, `ShardedJSONLWriter`, counters, report. Workers return the (possibly `_parse_record`-mutated) record so the parent writes the same enriched output the serial path would.
- Determinism caveat: each worker seeds its own `random.Random`, so the `llm_judge` 10% sample subset differs from a serial run (still ~`sample_rate`). Irrelevant when the judge is disabled, which is the common case.
- Debug mode (`debug_rejections`) stays serial — it early-exits after N rejections.

### Code sandbox timeout must kill the whole process group (`code_verifier.py`)
`code.timeout_seconds` is only a real bound if the timeout kills the snippet's **entire process tree**. `subprocess.run(timeout=)` kills only the direct child; a snippet that backgrounds a process (`multiprocessing`, `Popen`, `os.system('… &')`) leaves a grandchild holding the stdout pipe open, and the follow-up `communicate()` then blocks on pipe EOF *indefinitely* — one such record froze a whole Stage 6 run for minutes. `_run_subprocess()` therefore uses `Popen(..., start_new_session=True)` (child leads its own process group) and, on `TimeoutExpired`, `os.killpg(SIGKILL)` the group before a bounded second `communicate(timeout=5)` drains the pipes. `_kill_process_tree()` falls back to `proc.kill()` on Windows / group-lookup failure. Regression test: `test_backgrounded_child_does_not_hang_past_timeout`. `stage6_filter_v2.yaml` sets `code.timeout_seconds: 3`.

### Math filter is non-rejecting (`math_verifier.py`)
`check_math` **never returns `passed=False`** — every path returns `FilterResult(True, ...)`; the numeric-consistency check is informational only (measured against an LLM judge on 995 labeled records, every rejection it would have made was a false positive). It used to also run SymPy's ANTLR `parse_latex` for a "parse sanity" signal, but the caller only reads `.passed` and discards the `reason`, so that parse — tens of ms per expression on every math/science answer — was pure wasted CPU and has been removed. If you ever make this filter actually reject, re-introduce SymPy verification there.

### LLM prompt annotation (`clustering/annotator.py`)
After clustering, each prompt is annotated with `{domain, difficulty, topics, language, summary}` by calling an OpenAI-compatible API (Qwen3-30B-A3B-Thinking-2507 or similar). Key properties:
- **Async with semaphore-bounded concurrency** — avoids overwhelming the vLLM server.
- **Checkpoint to Parquet every N records** — safe to restart mid-run; Stage 3 picks up from the last completed shard.
- **`<think>` block stripping** — thinking models wrap reasoning before the JSON; `annotator.py` strips `<think>…</think>` before parsing.
- **Graceful fallback** — API errors and JSON parse failures return validated defaults (`domain="other"`, `difficulty="medium"`) and never crash the pipeline.
- **Prompt truncation** — keeps only the last 512 whitespace-split tokens of each prompt to stay within the model's context window while preserving the actual question.

**Offline annotation workflow** (run inference on a separate cluster):
```bash
# 1. Dump annotation requests as OpenAI-compatible JSONL
sft-pipeline run-stage stage3_cluster \
    --config config/stage3_cluster.yaml \
    --dump-annotations /path/to/requests.jsonl
# Each line: {"prompt_id": "sha256:...", "messages": [{"role": "system", ...}, {"role": "user", ...}]}

# 2. Run inference elsewhere, produce responses.jsonl:
# Each line: {"prompt_id": "sha256:...", "response": {annotation dict or raw string}}

# 3. Import responses back — merges with cluster labels, rewrites part-*.jsonl
sft-pipeline run-stage stage3_cluster \
    --config config/stage3_cluster.yaml \
    --import-annotations /path/to/responses.jsonl
```
The import path is a fast-path: it loads cluster labels from existing `part-*.jsonl`, merges annotation results, deletes the old shards, and writes fresh ones. No re-embedding or re-clustering.

Empty responses and missing `prompt_id`s fall back to the heuristic cluster labels (`domain`, `difficulty`) already in the cluster output. After import, annotations are also saved to `annotations.parquet` as a checkpoint.

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

  Two separate issues can each cause `is_available()` to return False, and must both be addressed:

  **Issue 1 — `LD_LIBRARY_PATH` not set**: Singularity bind-mounts `/opt/rocm` but does not add it to `PATH` or `LD_LIBRARY_PATH`. Without `LD_LIBRARY_PATH=/opt/rocm/lib`, torch silently fails to open `libamdhip64.so`. Fixed in `run_in_env.sh` which sets these automatically:
  ```bash
  export PATH="/opt/rocm/bin:$PATH"
  export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
  # NOTE: do NOT set HSA_OVERRIDE_GFX_VERSION — ROCm 6.3 supports gfx90a natively.
  # Setting it to 9.0.0 (gfx900/Vega10) loads wrong kernels and causes GPU
  # memory access faults (hipRAND and other compute kernels crash on MI250X).
  ```

  **Issue 2 — cgroups v2 device delegation**: On this cluster, when you run `srun --pty bash` and then launch `singularity` from within that shell, Singularity creates a child cgroup that does not inherit the parent job's device allowlist. Even with `ROCR_VISIBLE_DEVICES=0` set and `/dev/kfd` bound, `open("/dev/kfd")` raises `PermissionError` inside the container. **Fix: use `--rocm` flag**, which tells Singularity to handle ROCm device delegation itself rather than relying on cgroup inheritance:
  ```bash
  singularity exec --rocm \
      --bind /scratch/project_462000963 \
      --bind /users/aralikatte \
      --bind /opt/rocm \          # still needed: --rocm binds device nodes, not the library tree
      --overlay .../python_latest_overlay.img \
      .../python_latest.sif \
      bash .../scripts/setup_env.sh
  ```
  Note: `--bind /dev/kfd` and `--bind /dev/dri` are **not** needed when using `--rocm` — it handles those automatically. The pipeline Slurm scripts (`scripts/slurm_stage1.sh`, `scripts/slurm_stage3.sh`) use `--rocm`. Do **not** use `--bind /dev/kfd --bind /dev/dri` without `--rocm`; the bind makes the device file visible but the cgroup still blocks access.

  **Exception — vLLM SIF (Docker-based, ROCm baked in)**: see the dedicated section below. The vLLM SIF uses `--rocm` + strips `/.singularity.d/libs` from `LD_LIBRARY_PATH` to get device access without the glibc clash. `serve.sh` handles this automatically (`ROCM_COMPAT=1` is the default).

- **ROCm vLLM — use the SIF, not a pip wheel**: The standard `pip install vllm` gives a CUDA build that fails on AMD GPUs. On the cluster, vLLM runs from a pre-built Singularity SIF (`vllm/build_sif.sh` → `vllm/serve.sh`); no pip install needed. The `gpu_memory_utilization` and `dtype` settings in `vllm/serve.sh` are tuned for MI250X.

- **vLLM ROCm Singularity SIF — five hard-won lessons** (`vllm/` directory):

  The vLLM server runs from a pre-built Docker image converted to SIF (`vllm/build_sif.sh` → `vllm/serve.sh`). The rules are different from the plain `python:3.11-slim` SIF used by the pipeline itself.

  **1. Use `--rocm` + strip the injected libs — for ALL job types on LUMI.**
  Naively, `--rocm` looks dangerous for the vLLM SIF: the image has ROCm baked in, and `--rocm` injects host ROCm libs (compiled against glibc 2.38+) into `/.singularity.d/libs/`, prepending that path to `LD_LIBRARY_PATH`. The container is Ubuntu 22.04 (glibc 2.35), so those host libs cause `ImportError: GLIBC_2.38 not found` on `import torch`. However, `--bind /dev/kfd --bind /dev/dri` without `--rocm` does **not** work on LUMI — cgroups v2 blocks device access in both interactive **and** batch contexts. The fix: use `--rocm` to get device delegation, then immediately strip `/.singularity.d/libs` from `LD_LIBRARY_PATH` inside the container so that the container's own ROCm copy is used instead of the injected host libs. `serve.sh` does this automatically with `ROCM_COMPAT=1` (now the default).

  **2. `--bind /dev/kfd` does NOT work on LUMI — cgroups v2 blocks it everywhere.**
  `/dev/kfd` has permissions `crw-rw-rw-` (world-writable) yet `open()` returns `EPERM` in both interactive `srun --pty bash` sessions and `sbatch` jobs. This is the cgroups v2 device controller — Singularity child cgroups don't inherit the job's device allowlist. `--rocm` tells Singularity to handle AMD device delegation itself and is required in all contexts. The `serve.sh` `ROCM_COMPAT=1` mode handles both issues at once:
  ```bash
  # ROCM_COMPAT=1 is now the default; this is equivalent:
  bash vllm/serve.sh --model Qwen/Qwen3-30B-A3B-Thinking-2507 --tensor-parallel-size 2
  ```

  **3. Match the Docker image tag GFX arch to the host GPU.**
  Tag `gfx950` = MI350X. Tag `gfx942` = MI300X. This cluster is MI250X = `gfx90a`. Using the wrong arch causes `hipErrorInvalidImage` (HIP kernel images are not cross-arch compatible). Tags that don't contain an explicit `gfxNNN` string (e.g. `rocm6.3_mi300_...`) typically bundle multiple archs including `gfx90a`. Verify with: `singularity exec "${SIF}" python3 -c "import torch; print(torch.cuda.get_arch_list())"` — or just try it, the `serve.sh` GFX guard will catch explicit mismatches.

  **4. Do NOT name a bash variable `HOST` on LUMI/Cray.**
  The OS pre-sets `HOST=<nodename>`. Any script doing `HOST="${HOST:-0.0.0.0}"` will inherit the node hostname and pass it as `--host` to vLLM's API server, which then fails with `OSError: [Errno 99] Cannot assign requested address` because the hostname resolves to an IP that can't be directly bound. Use `BIND_HOST` or any other name.

  **5. `torch.cuda.get_arch_list()` returns `[]` on ROCm builds — this is normal.**
  That function is CUDA-specific. On a ROCm wheel, `torch.version.hip` is the relevant field. GPU visibility is confirmed by `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`.

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
#    (single node)   sbatch vllm/slurm_serve.sh
#    (job array)     sbatch vllm/slurm_serve_array.sh
#                    → task 0 runs nginx; URL printed in logs/vllm_worker_<id>_0.log
#    (interactive)   ROCM_COMPAT=1 bash vllm/serve.sh --model <model> --tensor-parallel-size 2

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
python viz/export.py --run-dir /path/to/run          # default: 50k sample for browser
python viz/export.py --run-dir /path/to/run --sample 100000
python viz/export.py --run-dir /path/to/run --umap   # also compute UMAP (slow)

# 3. Launch
streamlit run viz/app.py

# 4. Share
cloudflared tunnel --url http://localhost:8501
```

**How export.py works:**
- Prefers `{run_dir}/stage3/part-*.jsonl`; falls back to `{run_dir}/stage1/part-*.jsonl`
- Computes **full-data aggregate stats** (distributions, cross-tabs, cluster analysis) over all records via shard-by-shard Polars — peak memory ≈ one shard at a time. Stored in `meta.json["stats"]`.
- Uses **reservoir sampling** for the browser snapshot (O(sample) memory regardless of dataset size)
- UMAP is opt-in (`--umap`); off by default
- Joins Stage 5 responses and Stage 6 filter results if present
- Writes `viz/data/snapshot.parquet` + `viz/data/meta.json`

**Pages:**
- `1_Stats.py` — domain/difficulty/language/source distributions (full-data counts), Domain×Difficulty and Domain×Language cross-tab heatmaps (normalised + absolute count on hover), topics overall and per-domain
- `2_Prompts.py` — searchable/filterable prompt table (sample only)
- `3_Clusters.py` — cluster size histogram, clusters dominated per domain, top-50 clusters table
- `4_Answers.py` — prompt + reasoning + answer viewer (after Stage 5)

**Cache invalidation**: `data_loader.py` passes `snapshot.parquet`'s mtime to `@st.cache_data`, so re-running export invalidates the cache automatically. No restart needed.

**`viz/data/` is gitignored** — snapshots are not committed.

---

## Test Coverage Gaps (known)

- Stage 1 (dedup): no integration test — needs HF datasets mock
- Stage 2 (corpus chunking): no integration test — needs LLM HTTP mock
- Stage 3 (embedding + FAISS): no integration test — heavy dependencies
- Stage 5 (vLLM): integration test uses mock; no real vLLM test (expected — requires GPU)
- Filters: `heuristic.py`, `code_verifier.py`, `llm_judge.py` have no unit tests yet
