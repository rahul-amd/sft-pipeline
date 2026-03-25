# SFT Dataset Construction Pipeline — Requirements & Implementation Plan

## Context

Based on the proposal "Constructing a High-Quality Post-Training Dataset for SFT", we are building a scalable, automated 7-stage pipeline that produces **5M high-quality prompt–response pairs** with structured reasoning traces for supervised fine-tuning of LLMs. This is a **greenfield Python project** running on an **in-house GPU cluster (512 GPUs, local disk)**, using **Qwen3.5-122B-A10B via vLLM** (a MoE model; configurable) as the teacher model for reasoning generation.

The pipeline addresses the core challenge: manual annotation is too slow/expensive at scale, naive synthetic generation produces low-quality data. The solution combines prompt sourcing, knowledge-driven generation, clustering, teacher-model inference, quality filtering, and multilingual expansion.

---

## Requirements

### Functional Requirements by Stage

#### Stage 1 — Prompt Collection from Existing Datasets
- Ingest from HuggingFace datasets: GSM8K, MMLU, HumanEval, ARC, WizardLM, OpenMathInstruct, etc.
- Extract only prompts (discard answers) — responses will be generated uniformly by teacher model
- Normalize to consistent text format, strip source metadata
- Deduplicate using MinHash LSH (near-duplicate detection, not just exact)
- Tag each prompt with source dataset and inferred domain

#### Stage 2 — Prompt Generation from Knowledge-Rich Corpora
- Input: configurable corpus source — either a **local directory path** (JSONL/text files) or a **HuggingFace dataset ID** (public or private repo); specified in config
- Chunk documents into segments (512–2048 tokens, configurable overlap)
- Use a lightweight LLM (Qwen2.5-7B) to generate 3–5 prompts per chunk
- Prompt types: comprehension questions, explanation tasks, derivation, summarization, multi-step reasoning
- Tag with source document, segment ID, domain

#### Stage 3 — Clustering by Domain and Difficulty
- Embed all prompts with `BAAI/bge-m3` sentence embedding model
- Build a FAISS `IVFFlat` index for efficient ANN lookup
- Cluster domains using HDBSCAN on centroids (not all vectors — infeasible at scale)
- Assign difficulty tiers (Easy/Medium/Hard) via heuristics: token count, multi-step indicators, terminology complexity
- Each prompt exits with `domain` and `difficulty` labels

#### Stage 4 — Prompt Sampling under Compute Constraints
- Target: **~7M prompts for inference** in a single pass (to yield ~5M after ~30% filtering loss)
- Enforce configurable domain quotas (math 25%, code 20%, science 20%, general 20%, language 15%)
- Enforce configurable difficulty quotas (easy 20%, medium 50%, hard 30%)
- Remove near-duplicates via embedding cosine similarity threshold (0.92)
- Sort final sample by embedding similarity to maximize vLLM KV-cache reuse (15–25% throughput gain)
- Deterministic with random seed

#### Stage 5 — Teacher Model Inference (Qwen3.5-122B-A10B via vLLM)
- Use vLLM **offline batch mode** (`vllm.LLM`) — not HTTP server mode (avoids round-trip overhead)
- Each response: `<think>reasoning trace</think><answer>final answer</answer>`
- Generate `n_candidates=2` per prompt; select best by longest valid reasoning trace
- Checkpoint every 5,000 prompts; resume from exact position on failure
- Multi-node scaling: Ray actors, each wrapping one vLLM instance, processing prompt shards

#### Stage 6 — Automated Quality Filtering
Filters run in cost-order (cheapest first):
1. **Structural**: missing fields, repetition loops, length bounds
2. **Heuristic**: low info density, self-contradiction markers
3. **Math**: SymPy validation of LaTeX expressions and internal consistency
4. **Code**: sandboxed subprocess execution (E2B cloud sandboxes in production)
5. **LLM Judge**: 10% sample, Qwen2.5-7B scores 1–10; keep if overall ≥ 6.0
- Produce per-stage, per-domain, per-difficulty filter report

#### Stage 7 — Multilingual Expansion (deferred to v2)
- Translate 15% of final dataset per target language: zh, es, fr, de, ja
- **Segment-aware translation**: preserve `<think>` blocks, code blocks, math LaTeX; translate only natural language segments
- Primary backend: DeepL API; fallback: Google Cloud Translation v3
- Validate via multilingual embedding similarity (source vs translated, cosine > 0.85)
- Tag each record with `language` and link to source English `prompt_id`

### Non-Functional Requirements
- **Scale**: 5M+ final examples (run pipeline in multiple passes with different seeds, combine)
- **Resumability**: DuckDB checkpoint file; every stage skips already-processed items
- **Config**: Single YAML file with Pydantic v2 validation; all parameters configurable
- **Reproducibility**: Identical output given same config + random seed
- **Output format**: Sharded JSONL (500MB shards) + HuggingFace `datasets` export
- **Dry-run mode**: Estimates GPU-hours, API cost, and wall-clock time without processing
- **Observability**: Progress logs per stage; filter report; summary statistics at end of run

### Output Record Schema
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

## Implementation Plan

### Tech Stack

**Cluster spec**: 32 nodes × 8 MI250X (AMD) × 2 GCDs each = 512 logical GPU devices. ROCm stack. Each GCD = 64GB HBM2e. TP=8 per vLLM replica → 64 replicas × 8 GCDs = 512 GCDs.

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Orchestration | Ray + Ray Workflows | Native distributed Python; no JVM; scales across 512-GPU in-house cluster |
| Storage abstraction | pathlib + `datasets` lib | Local filesystem paths; HF `datasets` for loading corpora by HF dataset ID |
| Checkpoint DB | DuckDB | Single-file, no server, SQL, handles 10M rows, stored on local disk |
| Embeddings | `sentence-transformers` (`BAAI/bge-m3`) | Works with ROCm PyTorch |
| ANN Index | **FAISS CPU** (`faiss-cpu`) | FAISS GPU is CUDA-only; CPU IVFFlat handles 7M vectors fine for ANN at this scale |
| Clustering | HDBSCAN on centroids (CPU) | `hdbscan` lib, CPU; GPU variant (cuML) is CUDA-only; on centroids only so CPU is fast |
| DataFrame ops | Polars (lazy) | 5–20x faster than Pandas; Arrow memory; lazy eval = low RAM |
| Teacher inference | vLLM (CUDA on laptop, ROCm on cluster), offline batch | vLLM supports both; `device` config flag switches backend; offline batch = max GPU utilization |
| LLM judge / generator | vLLM HTTP server (Qwen2.5-7B) | Lightweight; shared between Stage 2 and Stage 6; same CUDA/ROCm abstraction |
| Math verification | SymPy | Parse LaTeX, verify internal consistency |
| Code sandbox | E2B (production), subprocess (dev) | E2B: managed, no ops; no CUDA/ROCm dependency |
| Translation | **Deferred to v2** | Stage 7 excluded from v1 scope |
| CLI | Typer | Clean async CLI with automatic --help |
| Config validation | Pydantic v2 + PyYAML | Catches type errors before multi-hour runs begin |
| Testing | pytest + tmp_path fixture | Standard; use `tmp_path` for local FS mocking |

### Project Directory Structure

```
sft-pipeline/
├── config/
│   ├── default.yaml          # Master config
│   ├── dev.yaml              # Small-scale overrides
│   └── prod.yaml             # Production overrides
│
├── sft_pipeline/
│   ├── config.py             # Pydantic config models + YAML loader
│   ├── storage.py            # Local read-write + ShardedJSONLWriter
│   ├── checkpoint.py         # DuckDB checkpoint tracker (critical — build first)
│   ├── cost_estimator.py     # Dry-run GPU/API cost estimation
│   ├── cli.py                # Typer CLI: run, run-stage, status, dry-run
│   │
│   ├── stages/
│   │   ├── stage1_collect.py     # HF dataset ingestion + MinHash dedup
│   │   ├── stage2_generate.py    # Corpus chunking + LLM prompt generation
│   │   ├── stage3_cluster.py     # Embedding, FAISS index, HDBSCAN, difficulty
│   │   ├── stage4_sample.py      # Quota sampling, dedup, ordering optimization
│   │   ├── stage5_inference.py   # vLLM batch inference + checkpointing
│   │   ├── stage6_filter.py      # Quality filtering pipeline
│   │   └── stage7_translate.py   # Segment-aware multilingual translation (v2)
│   │
│   ├── filters/
│   │   ├── structural.py         # Field checks, repetition, length
│   │   ├── heuristic.py          # Info density, contradiction
│   │   ├── math_verifier.py      # SymPy LaTeX verification
│   │   ├── code_verifier.py      # Sandboxed code execution
│   │   └── llm_judge.py          # LLM-based quality scoring
│   │
│   ├── clustering/
│   │   ├── embedder.py           # Batched sentence-transformer inference
│   │   ├── faiss_index.py        # IVFFlat build/stream-add/save/load
│   │   └── clusterer.py          # HDBSCAN + difficulty heuristics
│   │
│   ├── inference/
│   │   ├── vllm_batch.py         # vLLM offline batch loop + best-of-n
│   │   ├── prompt_formatter.py   # Chat template application
│   │   └── output_parser.py      # <think>/<answer> extraction + fallback
│   │
│   ├── translation/
│   │   ├── segment_parser.py     # Preserve code/math blocks; split for translation
│   │   ├── deepl_client.py
│   │   └── google_translate_client.py
│   │
│   └── export/
│       ├── jsonl_writer.py       # Sharded JSONL output
│       └── hf_exporter.py        # datasets.Dataset push_to_hub
│
├── scripts/
│   ├── run_pipeline.py       # End-to-end runner
│   ├── estimate_cost.py      # Dry-run report
│   └── serve_vllm.sh         # Start vLLM HTTP server for judge/generator
│
├── tests/
│   ├── conftest.py           # Shared fixtures + make_prompt_record/make_response_record
│   ├── unit/                 # Per-module unit tests
│   └── integration/          # Per-stage + end-to-end smoke test
│
├── docker/
│   ├── Dockerfile.pipeline   # Pipeline workers
│   └── Dockerfile.vllm       # vLLM inference (pinned commit SHA)
│
├── infra/
│   └── ray_cluster.yaml      # Ray cluster config for in-house GPU nodes
│
└── pyproject.toml
```

### Critical Design Decisions

**1. DuckDB for checkpointing** (not Redis/PostgreSQL): single file, no server, handles 10M rows, stored on local disk. Every stage queries it at startup to build a set of already-processed `prompt_id`s (SHA256 of prompt text), then skips those in the stream.

**2. vLLM offline batch mode for Stage 5** (not HTTP server): zero round-trip overhead for millions of requests; direct `RequestOutput` access for best-of-n selection; maximum GPU utilization.

**3. Polars over Pandas for Stage 4**: at 7M prompts with 1024-dim embeddings, Polars lazy evaluation avoids materializing 30GB in RAM. 5–20x faster for groupby/sort operations.

**4. Single 7M inference pass**: Stage 4 selects ~7M prompts for inference; after Stage 6 filtering (~30% loss expected), yields ~5M high-quality examples. With 512 GPUs and Qwen3.5-122B-A10B (MoE — only ~10B active params, very high throughput), a single pass is tractable.

**5. Segment-aware translation** (v2): parse responses into PRESERVE (code, math, `<think>` delimiters) and TRANSLATE segments before calling the translation API. Prevents systematic corruption of reasoning structure.

**6. HDBSCAN on FAISS centroids, not all 7M vectors**: direct HDBSCAN on 7M points is O(n²) — infeasible. Run HDBSCAN on 500–1000 IVF centroids, assign each point to its nearest centroid's cluster label.

### Stage-by-Stage Implementation Notes

**Stage 1**: Use `DatasetAdapter` protocol — each HF dataset gets an adapter class with `extract_prompt(row) -> str`. Register in a dict. MinHash LSH via `datasketch` for near-duplicate detection at 0.95 similarity threshold.

**Stage 2**: Use `langchain_text_splitters.RecursiveCharacterTextSplitter` with `tiktoken` for token-aware chunking. Generator prompt template instructs Qwen2.5-7B to return JSON array of N questions per passage. Use `json_repair` for malformed outputs + single retry.

**Stage 3**: Stream embeddings in float16 to disk (Parquet) before building FAISS index — halves memory. At 7M × 1024 dims = ~14GB float16. Build IVFFlat index by training on a 500K sample, then streaming all vectors in 100K batches.

**Stage 5**: Batch 256 prompts per `llm.generate()` call. After each batch, flush to shard buffer and update DuckDB checkpoint. Select best-of-2 candidates by longest valid `<think>` trace length. Use Ray actors (one per vLLM instance) for multi-node scaling.

**Stage 6**: Filters run in cost-order. SymPy parse failures are `uncertain` (not `failed`) — don't over-filter on parser fragility. For code execution in production, use E2B API (`from e2b_code_interpreter import Sandbox`) — ~$0.30 for 300K executions, zero ops burden.

**Stage 7** (v2): Validate translated examples with multilingual embedding model (`paraphrase-multilingual-mpnet-base-v2`); cosine similarity between original and translated prompt must exceed 0.85.

### Configuration File (condensed)
```yaml
global:
  seed: 42
  dry_run: false
  run_id: "run_001"
  base_path: "/data/sft-pipeline/runs/{run_id}"   # local disk path

stage1_collect:
  datasets:
    - source: "hf_dataset"             # "hf_dataset" | "local_jsonl"
      hf_repo_id: "openai/gsm8k"
      hf_split: "train"
      prompt_field: "question"
    - source: "local_jsonl"
      path: "/data/datasets/my-prompts.jsonl"
      prompt_field: "instruction"
  dedup_threshold: 0.95

stage2_generate:
  corpora:
    - source: "local"                  # "local" | "hf_dataset"
      path: "/data/corpora/arxiv/"
    - source: "hf_dataset"
      hf_repo_id: "my-org/my-private-corpus"  # private HF repo; needs HF_TOKEN
      hf_split: "train"
      text_field: "text"
  prompts_per_chunk: 4

stage3_cluster:
  embedding_model: "BAAI/bge-m3"
  clustering_algorithm: "hdbscan"

stage4_sample:
  total_prompts: 7000000               # ~7M → ~5M after filtering
  domain_quotas: {math: 0.25, code: 0.20, science: 0.20, general: 0.20, language: 0.15}
  difficulty_quotas: {easy: 0.20, medium: 0.50, hard: 0.30}

stage5_inference:
  model: "Qwen/Qwen3.5-122B-A10B"       # configurable; any vLLM-compatible model
  vllm_engine:
    tensor_parallel_size: 8             # per replica; 512 GPUs / 8 = 64 replicas
    gpu_memory_utilization: 0.90
    pipeline_parallel_size: 1
  n_replicas: 64
  generation: {n_candidates: 2, max_tokens: 4096}
  checkpoint_every: 5000

stage6_filter:
  llm_judge: {sample_rate: 0.10, score_threshold: 6.0}

stage7_translate:
  target_languages: ["zh", "es", "fr", "de", "ja"]
  translation_fraction: 0.15
  provider: "deepl"
```

### Build Order (critical path)

1. `config.py` — Pydantic models; sets interface contract for all stages
2. `checkpoint.py` — DuckDB tracker; every stage depends on this
3. `storage.py` — Local abstraction + `ShardedJSONLWriter`
4. `cli.py` — Typer entrypoint with dry-run and status commands
5. Stage 1 + Stage 2 (can run in parallel)
6. Stage 3 (embedding + FAISS + clustering)
7. Stage 4 (sampling; needs Stage 3 outputs)
8. Stage 5 (teacher inference; longest stage — needs GPU infra ready)
9. Stage 6 (quality filtering; build filters incrementally, start with structural)
10. Stage 7 (translation; v2)
11. `export/hf_exporter.py` (last)

### Verification Plan

**Development environment**: Windows 11 laptop (CPU only, no ROCm). RTX 5060 available for small GPU runs.

- **Unit tests** (laptop, CPU): each filter, checkpoint logic, config validation, output parser — no model loading, synthetic fixtures
- **Integration tests** (laptop, CPU or RTX 5060): per-stage with `tmp_path` pytest fixture; mock vLLM with pre-canned responses
- **End-to-end smoke test** (laptop): 20 prompts, mocked vLLM, Stages 4+5(mock)+6, assert final JSONL schema and checkpoint resume
- **Small GPU smoke run** (RTX 5060): Stage 5 with Qwen2.5-7B-Instruct on 100 prompts to validate inference + parsing before cluster deploy
- **Cluster dry-run**: `sft-pipeline estimate --config config/prod.yaml` — validates config, counts inputs, estimates run time; no inference
- **Cluster validation**: deploy on 2-node subset first; inspect `filter_report.json` and sample outputs before full 32-node run

---

## Confirmed Decisions

- **Infra**: In-house cluster, 32 nodes × 8× AMD MI250X (16 GCDs per node = 512 GCDs total), local disk
- **GPU stack**: ROCm (not CUDA); vLLM ROCm build; FAISS CPU (no FAISS-GPU); CPU HDBSCAN
- **Teacher model**: Qwen3.5-122B-A10B, TP=8 per replica, 64 replicas
- **Corpus input**: Configurable — local path or HF dataset ID (public or private)
- **Dataset input for Stage 1**: Configurable — local JSONL or HF dataset ID
- **Scale strategy**: Single pass, ~7M prompts inferred → ~5M after filtering
- **Stage 7 (translation)**: Deferred to v2
- **Output**: Local disk (sharded JSONL), no HF Hub push in v1
- **Dev environment**: Windows 11 laptop with NVIDIA RTX 5060 (CUDA); GPU-specific code abstracted behind `device` config flag (`cuda` / `rocm` / `cpu`)
