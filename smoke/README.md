# Running the complete pipeline on a small dataset

Latest validation: [medium-scale freeze results](freeze-validation.md): 259 tests passed; 3,805 real HTTP inferences produced 3,131 exports, with GPU/Docker execution and interruption recovery checks. The [200-example audit](audit-findings.md) identified source-grounding and judge-calibration work before a production-sized run. Reusable config: [config-scale.yaml](config-scale.yaml). Earlier [GPU, Docker and two-node Ray results](gpu-validation.md) remain an archival record. Do not use the `hissu` partition for future runs.

[`smoke/config.yaml`](config.yaml) exercises every implemented stage using six public GSM8K training prompts, eleven distinct local prompt strings, and three short custom source documents. It needs an OpenAI-compatible chat server, Docker for code filtering, internet access for the small public dataset and embedding model on the first run, and a Python pipeline environment. The driver checks stage completion, generated prompts reaching the final dataset, contamination removal, unique IDs, nonempty training fields, semantic deduplication, and unchanged output after resume.

This is a functional integration test. It is not a benchmark of teacher quality, distributed throughput, or recovery from arbitrary machine failure. Stage 7 multilingual expansion is not implemented. Semantic deduplication is disabled by default. Set `stage4_sample.semantic_dedup_enabled: true` in the config to exercise global cosine deduplication and independent pairwise validation. The local fixture separately tests exact deduplication by containing a repeated prompt.

## Folder layout and tests

All smoke-specific files live in this directory: `config.yaml`, `run.py`, `fixtures/`, `tests/`, `results/`, `runs/`, and `release-readiness.md`. Run commands from the repository root. New live outputs go to `smoke/runs/$SFT_RUN_ID/`; run directories are ignored by Git.

Run the deterministic regression and HTTP integration tests without a model server:

```bash
python -m pytest -c smoke/pytest.ini smoke/tests -q
```

Pytest temporary outputs and its cache stay under `smoke/`. The existing repository test suite remains in `tests/`.

The Docker isolation tests are opt-in. On a compute worker with Docker available, run `docker pull python:3.12-slim`, then `SFT_TEST_DOCKER=1 python -m pytest -c smoke/pytest.ini smoke/tests -q`. Tests cover host-file and environment isolation, disabled networking, non-root execution, capability/seccomp settings, resource limits, bounded output, and timeout cleanup. The older partial end-to-end test was also moved into `smoke/tests/` and updated with synthetic embeddings, now required by Stage 4.

The original live run artifacts, including diagnostic attempts and server/test logs, have been moved into `smoke/runs/`. Their historical logs and checkpoint paths are preserved as evidence; start a fresh run with the current config. `results/validated-config.yaml` preserves the exact original configuration corresponding to the hash in `results/validation.json`; it is an archival snapshot, not the runnable configuration. The current config uses the relocated paths and Docker sandbox, disables semantic deduplication, and has no math filter.

## Install and start a model server

From the repository root, install the pipeline into a Python 3.11+ environment:

```bash
python -m pip install -e '.[dev]'
docker pull python:3.12-slim
```

Run a compatible model server in its own GPU environment. The validated model is `Qwen/Qwen3-4B`; the smoke uses its reasoning output. One example for an environment with vLLM installed is:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --served-model-name smoke-teacher \
  --host 127.0.0.1 --port 8000 \
  --api-key local-smoke \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --enforce-eager \
  --reasoning-parser qwen3
```

Use an appropriate CUDA or ROCm vLLM environment for your hardware. The validation used a ROCm container on a Slurm compute node; the pipeline itself ran on CPU on that node. On a cluster, allocate compute resources before starting the server or pipeline. Run both on the same node for the loopback URL above. A remote service requires its actual reachable address and authentication policy.

## Run and validate all stages

In the pipeline environment, from the repository root:

```bash
export SFT_API_BASE=http://127.0.0.1:8000/v1
export SFT_API_KEY=local-smoke
export SFT_MODEL=smoke-teacher
export SFT_RUN_ID=smoke_001
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

python smoke/run.py
```

Use a new `SFT_RUN_ID` whenever inputs, model, or configuration change. `${VARIABLE}` values are required environment substitutions; `{run_id}` and `{base_path}` are pipeline path substitutions. Credentials are not written into the example YAML or validation report.

The driver invokes the ordinary CLI twice. The first invocation runs collection through export; the second must skip all eight completed stages and leave the JSONL artifacts byte-identical. To invoke the pipeline directly:

```bash
python -m sft_pipeline.cli run --config smoke/config.yaml
```

Artifacts are under `smoke/runs/$SFT_RUN_ID/`:

- `pipeline.log` and `resume.log`: execution and resume logs.
- `stage1/` and `stage2/`: collected and generated prompts.
- `stage_decontam/`: clean prompts, removal audit, and report. The artificial library-clock question must be removed.
- `stage3/`: annotated prompts, embeddings, and FAISS index.
- `stage4/`: selected prompts. Its configured 64-prompt ceiling exceeds the small input pool; quota warnings are expected.
- `stage5/`: teacher responses, separately returned reasoning where provided, and completion reasons.
- `stage6/`: survivors and filter counts. Token-limited completions are rejected explicitly.
- `final/part-*.jsonl`: final training schema. `export.final_jsonl_path` currently supplies the directory; output remains sharded rather than a single `dataset.jsonl` file.
- `smoke_report.json`: counts, dependency versions, config hash, output hashes, and validation result. It is written only after all assertions pass.

There is no dedicated math filter; the [math verification proposal](math-verification-proposal.md) is deferred. Code filtering uses a restricted Docker container per snippet. The image must already exist on each worker; no image download or host-execution fallback occurs during filtering. Docker containers share the host kernel; use a dedicated worker or VM for hostile workloads. The legacy `subprocess` option is explicitly for trusted code only. Online judging evaluates every survivor in the smoke, using the same small model; production quality measurement needs independent validation.

If an API request fails or a generator/annotation/judge verdict is invalid, the pipeline marks the stage failed and raises an error. The tested ROCm image can mask nonzero process exit codes after GPU initialization, so use the driver's checkpoint/artifact assertions as well. HTTP teacher completions that reach their output-token limit are retained with their finish reason and rejected by Stage 6; offline completion metadata remains incomplete. Stages 3 and 4 now replace their output on retry; real interruption checks are recorded in the latest validation. A changed config requires a fresh run until artifact invalidation is implemented.

## Historical configuration checks

After making semantic deduplication opt-in and removing math filtering, **220 tests passed and 17 were skipped**. These checks cover disabled/enabled deduplication, sampling without embeddings, configuration loading, and the remaining serial/Ray/filter/API paths. Skips comprise 11 GPU/flash-k-means checks, one network-test module, and five opt-in Docker tests already exercised in the preceding validation. The log is `smoke/runs/optional_dedup/tests.log`. The live results below are historical and retain their original counts and hashes.

## Previous validation: deduplication, formatting, and Docker

The updated configuration completed all eight steps on 2026-10-08 in **343 seconds**, including unchanged-output resume validation: **17 collected + 6 generated → 22 clean → 20 sampled/inferred → 17 exported**. Four generated prompts survived. Stage 4 rejected the differently capitalized semantic duplicate; one additional record fell outside the configured domain quotas. The largest selected pairwise cosine was **0.68267**, below the **0.92** threshold. Stage 6 rejected one response structurally and two through judging.

The suite passed **225 tests**, with 12 skips, including five live Docker isolation tests. GPU/flash-k-means checks and the opt-out network module account for the skips. Docker 29.1.3 ran `python:3.12-slim`; its resolved digest and the complete counts/hashes are in [release-fixes-validation.json](results/release-fixes-validation.json). Logs and raw outputs are retained under `smoke/runs/release_fixes/` and `smoke/runs/release_fixes_final/`. No sandbox containers remained after validation, and the Slurm allocation was released.

See [implementation details and limits](implemented-fixes.md) and the [math verification proposal](math-verification-proposal.md).

## Recorded validation: 2026-10-08

The following is the original run before the additional semantic fixture and Docker default were introduced.

A fresh run completed without manual steps between stages in **284 seconds**, including validation and a second CLI invocation to check resume. It used `Qwen/Qwen3-4B` served by vLLM `0.18.2rc1.dev71+ge92668e83` in a ROCm container on one Slurm GPU allocation; the pipeline ran with CPU embeddings and clustering. Python was 3.12.13. The machine-readable [validation report](results/validation.json) records dependency versions, configuration hash, and artifact hashes.

| Step | Output records |
| --- | ---: |
| Stage 1: public and local collection, exact deduplication | 16 |
| Stage 2: generation from three custom documents | 6 |
| Decontamination: combined pool, one canary removed | 21 |
| Stage 3: real embeddings, clustering, API annotation | 21 |
| Stage 4: quota sampling | 21 |
| Stage 5: authenticated API inference | 21 |
| Stage 6: structural/code/math filters and API judging | 18 |
| Final schema export | 18 |

Five generated prompts reached the final dataset. The final domain counts were 12 math, 3 code, and 3 science. All 18 IDs and prompt texts were unique, and every record had nonempty prompt, reasoning, and answer fields. The teacher returned 20 stopped completions and one token-limited completion; filtering rejected two records structurally (including the token-limited response) and one through judging. The second CLI invocation skipped all eight completed steps and left every JSONL artifact byte-identical.

The test suite passed **203 tests**, with **12 skips**: 11 GPU/flash-k-means checks unavailable in the pipeline environment and one explicitly disabled network-test module. Local Ray tests ran. The new HTTP integration tests cover all-stage orchestration, authentication, failures and retry, separate reasoning, truncation, export, and resume; their HTTP responses and embeddings are deterministic fixtures. The separate live run above used real model responses and embeddings.

This validation used an existing dependency environment plus an editable installation, not a clean dependency install. Counts and hashes are evidence for this particular run, not expected golden outputs: model sampling, serving versions, source revisions, and absolute source paths can change them. Earlier diagnostic attempts exposed integration defects and failed; the recorded run started in a fresh directory after those fixes and completed uninterrupted.

## Large inference runs using a dispatcher

For large runs, an external dispatcher can replace the Stage 5 execution step while keeping the same data contract. Export each Stage 4 record with its `prompt_id`, original metadata, and a `messages` array produced by `build_chat_messages`. Let the dispatcher fan requests out to serving replicas and save responses keyed by `prompt_id`.

Join results back to the Stage 4 records by ID, verify complete coverage and uniqueness, and write Stage 5 `part-*.jsonl` records containing the original fields plus `response`, `teacher_model`, and `finish_reason`. If the server returns a separate reasoning field, preserve it as `reasoning` and the final content as `answer`. If reasoning is embedded in the raw response, configure the teacher's actual delimiters before filtering. Never join by row order: distributed completion order is not input order.

Then run Stage 6 and `run-stage export` against those imported records. This is a suggested integration contract; the repository does not yet have a general dispatcher export/import command for Stage 5. The existing `--dump-annotations` and `--import-annotations` flags apply only to Stage 3. The small validated workflow uses built-in HTTP inference and requires no manual file transfer.
