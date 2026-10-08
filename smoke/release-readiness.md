# SFT pipeline release audit

The repository has a validated HTTP path through a 3,805-prompt inference run, but it should not yet be presented as a fully unattended, production-scale dataset builder. Semantic deduplication, multiline formatting, and default code isolation have now been addressed. Remaining limitations include cache invalidation and the difference between quality heuristics and answer verification. A successful smoke run demonstrates stage integration; it does not establish dataset quality or distributed scalability.

This audit covers the Python pipeline, shipped configurations, tests, export, Slurm launch scripts, and calibration tools. The fixes in this working tree focus on the built-in HTTP workflow. Existing production datasets have not been rewritten.

The current follow-up deliberately defers resume invalidation, offline/distributed inference, clean installation, and publication hygiene at the user's request. Those findings below remain a backlog, not work included in this change. See [implemented fixes](implemented-fixes.md) and the [math verification proposal](math-verification-proposal.md).

Validation on 2026-10-08: a fresh, uninterrupted live run completed all eight steps in 284 seconds and exported 18 records, including five generated from custom sources. A no-work resume preserved all JSONL hashes. The test suite passed 203 tests with 12 skips. See the [recorded results and scope](README.md#recorded-validation-2026-10-08) and [machine-readable report](results/validation.json).

Follow-up validation after the requested fixes: 225 tests passed with 12 skips, including live Docker isolation tests. The updated all-stage run exported 17 records in 343 seconds, rejected the added semantic duplicate, and preserved hashes on resume. See the [current results](README.md#previous-validation-deduplication-formatting-and-docker) and [report](results/release-fixes-validation.json).

The subsequent [freeze validation](freeze-validation.md) records fixes for rerun duplication, clustering seeds, judge timeouts, annotation import, and strict benchmark loading, plus the medium run and interruption probes. Earlier results below remain historical.

## Changes made for the unattended HTTP workflow

| Finding | Change | Evidence |
| --- | --- | --- |
| `config/default.yaml` did not validate because it used flat difficulty quotas. | Updated to the required `default` mapping. | Regression test loads every shipped YAML. |
| `run` stopped after filtering and never exported the advertised final schema. | Added automatic export after an enabled Stage 6 and a standalone `run-stage export` command. Empty training fields, duplicate IDs, and empty exports fail explicitly. Re-export replaces shards instead of appending. | CLI integration test checks all eight completion statuses and the final records. |
| Errors containing quotes or newlines could break DuckDB failure recording. | Serialize error metadata with `json.dumps`. CLI single-stage failures now record failed status. | Exception-text regression test. |
| Portable configurations could not reference API credentials from the environment. | Added required `${VARIABLE}` interpolation, resolved before run-path placeholders. Missing variables fail early. | Config tests and the live smoke configuration. |
| Stage 1 swallowed source failures and distributed collection could mark a partial collection complete. | Propagate source failures after draining workers. Missing files and empty collection now fail. Distributed source filenames include the source configuration to distinguish HF configs. | Source review; the live run exercises HF and local ingestion. Failure-path coverage should be expanded further. |
| Stage 2 had no API authentication, treated regular JSON files as JSONL, did not accept individual corpus files, and identified documents using potentially colliding prefixes. | Added bearer authentication, file/directory support, JSON object/array support, and IDs based on paths and full content. | Corpus format and identity regressions; live custom-source generation. |
| Stage 2 accumulated all chunks and futures, checkpointed chunks before writing, and silently completed after generator failures. | Bound pending work, flush output before checkpointing, retry malformed responses, fail the stage on failed chunks, and deduplicate generated prompt IDs. Validate chunk overlap to prevent invalid or non-terminating splitting. | HTTP integration and chunk-config regressions. Crash consistency still has a remaining window described below. |
| Stage 3 could skip incomplete distributed embedding because any Parquet shard existed. `torch_kmeans` was routed through the wrong branch; annotation import discarded centroid similarity. | Consult distributed worker sentinels, require a completion marker for new single-node embeddings, route `torch_kmeans` directly, and retain imported centroid similarity. Disabled decontamination no longer selects an old clean pool. | Source changes; small-run integration exercises CPU k-means, not these GPU branches. |
| Online annotation silently substituted default labels after API or parsing errors. | Fail on missing/invalid domain or difficulty, truncation, or API failure. Completed annotation chunks remain resumable. | Live annotation and HTTP integration. |
| Stage 5 created a task for every prompt, treated API failures and empty responses as successes, and discarded separately returned reasoning. | Bound active tasks, honor configured API retries, persist separate reasoning, retain failed requests for retry, and recover durable output written before a checkpoint. | Real HTTP failure/retry tests and live vLLM requests. |
| Stage 5 checkpoint writes could precede output durability. | Flush and fsync before recording success, including the offline paths. | Source-level ordering and HTTP resume regression. This is not a multi-file transaction. |
| Teacher responses stopped by the token limit were indistinguishable from complete responses. | Store `finish_reason`; Stage 6 explicitly rejects `length` responses. They are quality rejections, while transport failures still fail inference. | Structural regression and live inference. |
| Decontamination and Stage 6 could mark success after Ray shard failures. | Raise after draining failed tasks so the CLI cannot mark the stage complete. Empty benchmark indexes now fail. Stage 6 reads data shards rather than every JSONL file. | Existing local Ray equivalence tests; injected worker-failure tests remain useful. |
| LLM judging had no authentication, a fixed small token budget, and silently passed failed or malformed verdicts. | Add API key and token-budget settings; validate the score; default to raising on judge errors. Legacy fail-open behavior requires explicit `on_error: pass`. | Malformed-score and error-policy regressions; live judging. |

## Blockers before a production release

### Resume must be tied to inputs and configuration

**Priority: critical.** Stage completion is keyed only by stage name in [`checkpoint.py`](../sft_pipeline/checkpoint.py). Embeddings, the FAISS index, annotation Parquet, and shard ledgers have no shared fingerprint of input files, model revision, prompts, thresholds, or configuration. Changing a config and reusing a run can therefore skip required work or combine incompatible artifacts. The CLI's `--no-resume` bypasses stage-level skipping, but does not reset item checkpoints or artifact caches.

Stage 3 and Stage 4 now stage replacement shards before publishing, including annotation import. Direct retries no longer append duplicates; real SIGKILL/retry probes preserved published output and recovered without duplicate IDs. Publication still replaces multiple shard files individually, so a kill between renames requires rerunning the incomplete stage before reading its output. Export also replaces multiple shards individually. Stage 2 still has a write-before-checkpoint window: after a crash it may regenerate a chunk and produce different questions even though earlier questions from that chunk were already persisted.

Define an immutable run manifest with stage-specific fingerprints, atomically publish completed artifact sets, and invalidate downstream stages when inputs change. Test interruption at each write/checkpoint boundary. Until then, use a new run directory after any config/input/model change. The new single-node embedding marker is absent in legacy runs; do not rerun those against this code without an explicit migration or an isolated rebuild.

### Semantic deduplication: addressed; broader quota policy remains

[`stage4_sample.py`](../sft_pipeline/stages/stage4_sample.py) supports opt-in semantic deduplication through `semantic_dedup_enabled`, which defaults to `false`. When enabled, it enforces `dedup_cosine_threshold` with an exact FAISS inner-product index of normalized retained vectors. Checks cross cluster, domain, difficulty, and source boundaries. Each quota cell continues considering candidates after a rejection until its quota is filled or its candidates are exhausted. A small conservative tolerance handles float32 boundary cases. Missing embeddings fail explicitly only when deduplication needs them.

`sampling_report.json` records duplicate rejections, all quota cells (including empty domains/cells), total shortfall, and integer-rounding shortfall. Selection is greedy in configured quota order; rejected candidates do not suppress later candidates. Exact global comparisons can be expensive at millions of prompts and have not been benchmarked at that scale. There is still no separate similarity sort for KV-cache reuse.

Unknown domains are mapped to `general` even when quotas do not contain `general`; those records disappear from the sample. Integer rounding and underfilled quota cells reduce output size without redistribution. A declared fallback domain and a redistribution policy remain possible follow-ups; current reports expose shortfalls rather than redistributing them. The smoke config deliberately permits quota shortfalls.

### Default code isolation: addressed with restricted Docker execution

The default and shipped configs now use [`docker_sandbox.py`](../sft_pipeline/filters/docker_sandbox.py): no host mounts or network, a non-root user, read-only root filesystem, dropped capabilities, no-new-privileges, memory/CPU/PID limits, bounded output, and container cleanup. Startup and cleanup errors fail filtering. E2B no longer falls back to a local subprocess.

Live isolation tests exercise the configured restrictions. Docker's daemon, image, and shared host kernel remain trusted infrastructure; use a dedicated worker/VM for hostile workloads. The legacy `subprocess` backend remains an explicit trusted-input option and has no isolation guarantee. E2B's missing-dependency failure is tested, but its cloud execution was not exercised. The original archived smoke used subprocess; subsequent validation uses Docker.

### Multiline formatting: addressed; conversation scope remains single-turn

[`stage1_collect._normalize`](../sft_pipeline/stages/stage1_collect.py) now only canonicalizes line endings. Both collection and generation preserve multiline indentation, tabs, repeated spaces, blank lines, and Unicode literals. IDs hash the preserved text, so programs with different indentation remain distinct. Plain single-line source fields retain their existing outer-whitespace trimming.

Conversation extraction still keeps the first user turn and prepends a preceding system message into user text; tool context and later turns are discarded. Supporting full conversations is separate work. The formatting regressions cover plain text, serialized conversation formats, and both prompt-producing stages.

### Stage 2 source grounding: demonstrated quality gap

The medium run generated 1,002 prompts successfully, but its [content audit](audit-findings.md) found accepted questions that omit the sport or place needed to identify the intended answer. Source passages are not supplied to the teacher, and the judge can accept unsupported factual answers. Preserve source context, validate self-contained answerability, and calibrate source-aware evaluation before treating generated examples as verified high-quality data.

### Quality checks are not answer verification

**Priority: high for the blog and dataset claims.** The ineffective math filter has been deleted, including its config and calibration hooks. There is currently no dedicated mathematical answer verification. The code filter executes only the last explicitly tagged Python block and does not generally compare against reference answers or task tests. A response can execute successfully and still solve the wrong problem. Model-based judging can also be wrong.

Label these checks accurately. To claim verified answers, preserve source reference answers/tests where available and use task-specific verifiers with measured precision and recall. Re-evaluate calibration on a representative held-out sample with independent labels. The calibration harness takes the first matching records from configured domain offsets; its results should not be presented as an unbiased estimate for the full dataset. The small historical smoke used one model for generation and judging. The medium validation uses Qwen3-4B for answers, Qwen3-8B for annotation/judging, and a separate Qwen3-32B audit. Models sharing a family are not independent ground truth; correctness claims still require external validation.

### Benchmark loading: strict requested-subset coverage implemented

[`eval_index.py`](../sft_pipeline/decontam/eval_index.py) now fails if an explicitly requested HF config/split cannot load, is empty, or has no usable match-field text. Local files fail on missing/malformed input. The decontamination report retains per-subset counts, dataset fingerprints, and failures; explicit row caps record capped and unvisited subsets. Joining multiple-choice options into a match field can create broad or artificial match spans. Space-free languages need language-aware segmentation; short single-token strings fall below the configured minimum and are not indexed.

Immutable HF revision pinning and matching-precision evaluation on realistic benchmark fields remain separate work. Ad-hoc input directories also need collision-free shard keys: two directories with the same basename and shard names currently collide.

### Offline and distributed inference need separate validation

**Priority: high if advertised as supported production paths.** In [`stage5_inference.py`](../sft_pipeline/stages/stage5_inference.py), the single-node offline helper reloads the model for each batch. The Ray path submits and drains one actor's shard before starting the next actor's work; its actors request zero GPUs. This does not establish safe GPU placement or concurrent replica throughput. The API `skip_special_tokens` setting is also not passed to the HTTP server.

Reuse persistent engines, reserve actual GPU resources, submit all replicas concurrently, and test worker failures. Alternatively, document external OpenAI-compatible servers and the dispatcher workflow as the supported scale-out path, and label in-process inference experimental. The present live test validates the HTTP path only.

## Additional release work

| Area | Work remaining |
| --- | --- |
| Memory and throughput | Stage 3, Stage 4, Stage 5, annotation, and visualization retain large record collections. Embedding-loading futures retain completed arrays longer than the comments imply. Add representative scale tests and bounded-memory readers. Stage 2 now bounds work but fsyncs per completed chunk, which can be costly on shared storage. |
| Configuration | The loader uses one YAML plus Pydantic defaults; it does not inherit `default.yaml`. Unknown fields are silently ignored, `--set` is not a CLI option, and several numerical settings lack validation. Some stage-only configs leave unrelated stages enabled by default. Difficulty heuristic thresholds are declared but not passed into the scorer. |
| Reproducibility | Persist source/model revisions, tokenizer versions, RNG seeds, config fingerprints, and dependency versions. Clustering and FAISS initialization now use the configured seed, and torch k-means honors a requested CPU device. This does not promise bitwise reproducibility across hardware/library changes. |
| Source failures | HF schema-retry and permissive JSONL reading can skip malformed records. Report rejected/error counts per source and distinguish expected row rejection from a truncated download. Distributed Stage 1's merge reads all cached phase-one files, including files left by earlier source lists. |
| Packaging | Add a tested dependency/environment lock and clean-install CI. Add an explicit code license chosen by the repository owner, package metadata, and contribution/security instructions. The current environment-based validation is not a clean dependency installation. |
| Publication hygiene | Remove personal cluster paths/hostnames from portable examples, review tracked `.claude` settings and PDF artifacts, and scan the full Git history for credentials or private material before publishing. Stage 2 currently puts absolute local source paths into exported provenance; separate stable public source IDs from private filesystem locations before releasing dataset artifacts. Review upstream dataset terms and retain source provenance for any dataset release. No license or publication policy has been chosen on the owner's behalf. |
| Documentation | Reconcile README, `CLAUDE.md`, stage YAML comments, launch scripts, and the original plan. The plan describes older behavior and is not an implementation contract. Keep multilingual expansion explicitly unimplemented. |
| Visualization | `viz/export.py` expects parsed Stage 5 fields and a `passed_filters` field that the current Stage 6 output does not provide. It also hardcodes `stage6`, so `stage6_v2` results are missed. |
| Tests | Add clean-install CLI CI and broader interruption-boundary coverage. GPU/Docker/Ray tests, direct Stage 4 dedup/quota tests, missing/empty annotation-import fallback, and real Stage 3/4/5 interruption retries have been exercised; the medium run adds a separate model audit. Existing network tests should run in a separate opt-in job. |

## Release scope supported by the smoke run

The reproducible entry point is [`smoke/config.yaml`](config.yaml), driven by [`smoke/run.py`](run.py). It exercises public prompt collection, local source generation, decontamination, real CPU embeddings, CPU k-means, online annotation, quota sampling, live teacher inference, quality filters, online judging, export, and a no-work resume. See [the run instructions](README.md) for the validated setup and results.

A reasonable initial release is an experimental pipeline with this tested example and an explicit limitations section. Semantic deduplication and restricted Docker execution are implemented. Claims of verified mathematical reasoning, resistance to kernel exploits, or unattended multi-million-record crash recovery remain unsupported.
