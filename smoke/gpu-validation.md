# GPU and Docker validation — 2026-10-08

This is the earlier GPU matrix, preserved as historical evidence. The subsequent [freeze validation](freeze-validation.md) fixes the rerun, clustering-seed, judge-timeout and annotation-import defects reported here.

Three fresh, unattended pipeline runs passed all eight stages and the no-work resume check. No artifacts were manually moved between stages. All configurations, fixtures, tests and retained results are under `smoke/`. Production code was not changed during this validation pass.

| Full run | Collected | Generated | Clean pool | Sampled / inferred | Final | Generated in final | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPU + HTTP, dedup off | 17 | 6 | 22 | 21 | 20 | 5 | 275 s |
| Two-node Ray + HTTP, dedup on | 17 | 6 | 22 | 20 | 18 | 4 | 273 s |
| GPU offline teacher, dedup off | 17 | 6 | 22 | 22 | 22 | 6 | 168 s |

Each run removed the planted contamination example, retained generated prompts, exported unique IDs with nonempty prompt/reasoning/answer fields, and left output hashes unchanged on resume. The dedup-enabled run removed the planted semantic duplicate and independently checked every retained embedding pair. Counts differ because model generation and annotation are stochastic. These are functional smoke tests, not a correctness or dataset-quality benchmark.

Configurations: [HTTP GPU](config-gpu.yaml), [distributed GPU](config-distributed.yaml), [offline GPU](config-offline-gpu.yaml). Compact evidence, including hashes and reports: [gpu-validation.json](results/gpu-validation.json).

## Execution coverage

| Branch | Evidence |
|---|---|
| Stage 1 local JSONL + real Hugging Face source, serial and Ray | HTTP and distributed full runs |
| Stage 2 local corpora → chunking → real API generation | All three full runs; JSON/JSONL/Markdown/text loaders also have tests |
| Stage 2 Hugging Face corpus reader + real generation | One streamed TinyStories document, two generated questions; component probe, not a full HF-corpus stage |
| Decontamination serial, process pool and Ray | Full runs and integration tests; real GSM8K/MMLU eval loading passed |
| Embedding on ROCm, local and two-node Ray | Full runs; a GPU kernel also executed on each node in the Ray probe |
| All five clustering algorithms | 512-vector component matrix plus GPU tests, including flash-kmeans benchmarks |
| Flat, IVFFlat and IVFPQ indexes | Built, serialized, reloaded and queried; CPU FAISS fallback with `device: rocm` |
| Annotation online, disabled, dump/import, standalone and resume | Full runs, annotation probe and Flat-index probe |
| Semantic dedup enabled/disabled and quota shortfalls | Full runs; additional tests cover cross-cell dedup and backfilling |
| HTTP teacher, multiple candidates, offline batches and Ray replicas | Full runs and separate two-response, two-candidate probes |
| Filtering serial, process pool, Ray and rejection-debug mode | Full runs and tests; structural, heuristic, code and judge paths |
| Docker execution and isolation | Live tests on compute: host-file/secret/network isolation, privilege/resource limits, bounded output, timeout and descendant cleanup |
| Trusted subprocess backend | Existing tests with controlled fixtures |
| E2B and Hub publication | Mocked protocol/failure tests only; no cloud sandbox or publication performed |
| Export and CLI controls | Full runs, `status`, `estimate`, dry run, disabled stages, no-work resume and `--no-resume` with all stages disabled |

The expanded suite had **237 passed, 1 failed, 0 skipped**. Six additional contract/debug tests passed: **243 passed and 1 failed** across those suites. The failing test is flash-kmeans determinism. A separate Stage 3 rerun probe also failed its uniqueness assertion.

Measured coverage across real runs, component probes and tests: **84.45% of statements** and **71.70% of branches**; combined coverage **81.48%**. This is deliberately not a claim of 100% branch coverage. E2B credentials, FAISS GPU support and NVIDIA hardware were unavailable; real Hub publication was outside this test. Tensor/pipeline parallel degree greater than one was not tested. Some Stage 3 import code is unreachable because the earlier import fast path returns first.

## Findings to address

1. **Stage 3 reruns append duplicate records.** With unchanged inputs/config, running the stage twice reused identical embedding files but produced eight rows for four IDs. This is separate from the previously deferred configuration-change cache invalidation issue. Probe: `cached_flat.json` and `cached-flat-observations.json`.
2. **GPU clustering is not reproducible as currently implemented.** The flash-kmeans determinism test failed twice. Its wrapper does not seed the random initialization; the PyTorch path likewise uses unseeded `randperm`/`randint`. A fixed `global.seed` does not currently make these calls reproducible. Log: `tests-v2.log`.
3. **The judge has a fixed 30-second timeout.** A real run failed with `httpx.ReadTimeout` after both attempts when using a thinking judge. The release smoke configs use a separate non-thinking judge endpoint, which completed filtering in about ten seconds for the whole dataset. The general fix is a configurable request timeout and clearer underlying-error reporting. Log: `judge-diagnostic.log`.
4. **Offline inference still needs the previously deferred polish.** Both offline modes generated responses, but neither records finish reasons, so the downstream truncated-response guard cannot reliably apply. Single-process inference reloads the engine for every batch (observed twice for two prompts). Ray replicas are dispatched/drained sequentially and declare zero GPU resources, so successful execution here does not establish safe GPU scheduling or scale-out. The two-node claim above applies to distributed embedding, not the offline replica probe.
5. **This ROCm image masks nonzero exit codes after GPU initialization.** A minimal `sys.exit(17)` process returns 17 normally, but returns 0 after a GPU tensor allocation or embedding call. This reproduces without coverage instrumentation. It is an environment finding, not an established pipeline-code cause. Use the smoke driver's checkpoint/artifact assertions; raw shell exit status alone is insufficient with this image.
6. **Annotation import fallback differs from the CLI help.** The executed fast path retains heuristic labels for missing responses; it does not perform the advertised online fallback. The later block that implements online fallback is unreachable. Avoid documenting that fallback until the paths are reconciled.

No math filter was reintroduced. Semantic dedup remains optional and false by default. The successful offline output is a functional result; missing completion metadata and model-specific reasoning parsing still need review before treating that route as release quality.

## Environment and artifacts

Two AMD MI325X nodes, two allocated GPUs per node, Slurm job `54411`. GPU software ran inside the existing ROCm Apptainer image with an isolated `smoke/.venv-rocm` overlay; generated-code verification ran in real Docker containers. Versions included Python 3.12.13, PyTorch 2.10.0+git8514f05 / ROCm 7.2, vLLM 0.18.2 development build, Transformers 5.5.0, Ray 2.59.0, sentence-transformers 5.5.1 and faiss-cpu 1.14.2. The model was Qwen/Qwen3-4B and the embedder was all-MiniLM-L6-v2. Docker image digest: `python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f`.

This validates the recorded environment, not a clean installation. The inherited image has pre-existing dependency metadata conflicts; package inventory and preflight results are retained. GPU/ROCm visibility and Ray token propagation needed launcher corrections before successful runs. Failed exploratory attempts are retained separately.

Raw logs, JUnit XML, component results, coverage and Ray worker logs: `smoke/runs/gpu_matrix_20261008/`. Accepted full runs:

- `smoke/runs/gpu_full_v3_20261008/`
- `smoke/runs/gpu_distributed_v3_20261008/`
- `smoke/runs/gpu_offline_full_20261008/`

The smoke driver now snapshots the supplied YAML before launching, so later config edits cannot change its resume invocation. Model credentials remain environment substitutions. Earlier exploratory attempts that used a live config file are not counted as accepted runs.

No sandbox containers remained at the final Docker check. At the user's request, job `54411` was canceled after all accepted runs and evidence collection finished; Slurm then showed no user jobs in `hissu`. **Do not use `hissu` for future runs.**

## Reproduce the accepted flows

Use an allowed compute partition and a working CUDA/ROCm Python environment. Install the pipeline and relevant optional GPU/Ray dependencies. Pre-pull `python:3.12-slim` on every filter worker. Do not launch GPU workloads on the login node.

Start a Qwen3-4B teacher server using the command in [README.md](README.md), including `--reasoning-parser qwen3`. Start a second server for judging with a distinct port/name, `--default-chat-template-kwargs '{"enable_thinking": false}'`, and no reasoning parser. Bind to reachable interfaces when workers are remote. Set GPU visibility according to the runtime; in this ROCm environment, HIP indices refer to the already ROCR-filtered device list.

From the repository root, set endpoints to those running servers:

```bash
export SFT_API_BASE=http://TEACHER_HOST:18237/v1
export SFT_API_KEY=YOUR_LOCAL_SERVER_KEY
export SFT_MODEL=smoke-teacher
export SFT_JUDGE_API_BASE=http://JUDGE_HOST:18238/v1
export SFT_JUDGE_MODEL=smoke-judge
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

SFT_RUN_ID=my_gpu_http python smoke/run.py --config smoke/config-gpu.yaml
SFT_RUN_ID=my_gpu_offline python smoke/run.py --config smoke/config-offline-gpu.yaml

# For a running two-node Ray cluster with shared repository/output paths:
export SFT_RAY_ADDRESS=RAY_HEAD:PORT
SFT_RUN_ID=my_gpu_ray python smoke/run.py --config smoke/config-distributed.yaml
```

The judge uses the same API key as the teacher in these example configs. If the services use different keys, configure the judge key separately. New inputs/configuration require a fresh run ID. On NVIDIA, change `global.device` from `rocm` to `cuda`; that variant was not tested here.

For Ray 2.59, propagate a shared auth-token file to head, workers and driver, reserve GPU resources on both nodes, and choose nonconflicting Ray/worker ports. Do not publish the token. The cluster-specific `gpu_exec.sh` wrapper preserves the tested image, GPU masks and authentication setup, but is not a general-purpose installer or Slurm allocator.

Rerun the suites on compute with Docker and GPUs available:

```bash
SFT_TEST_DOCKER=1 python -m pytest -c smoke/pytest.ini tests smoke/tests \
  --junitxml=smoke/runs/my-test-results.xml
```

The flash-kmeans test should remain failing until its reproducibility defect is fixed. `gpu_branches.py` provides the recorded live component probes; each writes a JSON status even on failure. Its dated case directories should be isolated/copied before rerunning so old artifacts are not mistaken for fresh results.
