# Medium-scale freeze validation

Run date: 2026-10-08. Validation uses the built-in OpenAI-compatible HTTP path on four AMD GPUs with Docker code execution. All validation scripts, configurations, tests and outputs are under `smoke/`. Slurm partition: `amd-tw-verification`; `hissu` is excluded.

The full run is `smoke/runs/freeze_scale_v2/`. Supporting evidence is in `smoke/runs/freeze_validation/`, and the portable summary is [results/freeze-validation.json](results/freeze-validation.json). All eight stages and unchanged-output resume passed without manual steps between stages, exporting **3,131 records in 51 minutes 14 seconds**. The separate 200-example audit found content defects and unreliable reviewer judgments; see [audit findings](audit-findings.md). The HTTP workflow is a functional release candidate, but high-quality-output claims and a production-sized run should wait for source-grounding and judge-calibration improvements.

## End-to-end results

| Stage | Output records |
| --- | ---: |
| Collection | 9,924 |
| Generation from 500 custom-source documents / 501 chunks | 1,002 |
| Decontamination | 10,913 |
| GPU embedding, clustering and online annotation | 10,913 |
| Quota sampling | 3,805 |
| Teacher inference | 3,805 |
| Filtering | 3,131 |
| Final export | 3,131 |

Decontamination removed 13 records: the canary, three GSM8K test overlaps and nine ARC-Challenge test overlaps. Every requested benchmark subset loaded successfully. The final dataset contains 181 generated prompts, unique IDs and nonempty prompt/reasoning/answer fields. Resume skipped all eight completed stages and preserved every JSONL hash.

Inference had zero failed requests: 3,574 stopped completions and 231 token-limited completions. All 231 token-limited records were excluded from export. Filtering rejected 271 records structurally, 24 through code checks and 379 through the judge. Eight workers processed nine input shards. Docker events recorded 772 created and 772 destroyed sandbox containers; no sandbox containers remained at the final check. The Slurm allocation was canceled after validation.

Inference took 2,444 seconds, filtering 375 seconds and embedding/clustering/annotation 225 seconds. The teacher's median sampled generation throughput while at least 32 requests were active was about 3,451 tokens/s. These are observations from a run sharing the allocation with recovery probes, not an isolated throughput benchmark. Peak summed RSS for the driver and descendants was 32.0 GiB during multiprocessing, with shared pages counted more than once; steady inference RSS was about 4.2 GiB. Model servers are excluded from that CPU-memory figure. Per-card GPU reservations and timing details are in the JSON report; vLLM cache reservations are not minimum model VRAM requirements.

The recorded environment used Python 3.12.13, PyTorch 2.10.0 / ROCm, vLLM 0.18.2 development, Transformers 5.5.0, Ray 2.59.0, flash-kmeans 0.3.1 and Docker 29.1.3. Exact package versions and the Docker image digest are in the JSON report. A prior scale attempt failed on an invalid annotation label and is retained under `freeze_scale_v1`; it is not counted as a successful run.

## Fixes covered

- Stages 3 and 4 stage replacement output before publication. Reruns no longer append duplicate IDs, and failed staging preserves previous output. Publication across multiple files still requires a retry if interrupted between renames.
- Torch, flash, sklearn and FAISS clustering/index initialization use the configured seed. The torch path honors an explicitly requested CPU device. Seeds do not guarantee identical results across hardware or library versions.
- The judge timeout is configurable (`stage6_filter.llm_judge.request_timeout`, default 120 seconds), and failures include the underlying error.
- Annotation import performs online annotation for missing/empty responses when enabled, retaining centroid similarity. The unreachable alternate implementation was removed. Annotation checkpoints publish with an atomic file replacement.
- Requested benchmark configs/splits must load and contain usable match-field text. Reports include per-subset counts/fingerprints, explicit row caps and failures.
- Invalid online annotation labels receive up to three attempts with schema feedback. Persistent failures still stop the stage. The first scale attempt exposed an invalid `craft` domain; a fresh run after the fix recovered three such invalid-label responses automatically.
- Stage 5 shard size is configurable, allowing the medium run to exercise filtering across multiple inference shards. Stage 4 reports empty quota cells and rounding shortfalls.

Semantic deduplication remains optional and **false by default**. There is no math filter.

## Tests and fault injection

The final suite passed **259 tests, with zero failures or skips**, including live Docker isolation, GPU flash-kmeans and local Ray tests. Five multiprocessing/fork deprecation warnings remain. Evidence: `full-tests-final-v2.log` and `.xml`.

Real SIGKILL/retry probes passed:

| Stage | Result |
| --- | --- |
| 3 | Prior published output preserved; retry produced 160 unique IDs. |
| 4 | Prior published output preserved; retry produced the same 15 selected IDs. The fixture's quota target was intentionally underfilled. |
| 5 | Killed after 16 durable responses; retry finished all 96, preserving those responses without duplicate IDs. |

These probes target particular write boundaries; they do not establish recovery from every possible filesystem, worker or node failure. Recovery artifacts use separate run directories and do not alter the full run.

A separate live annotation-import probe retained one imported annotation and filled one empty and one missing response through the API, preserving centroid similarity for all three records.

## Sampling comparison

On the same 10,913 annotated prompts, semantic dedup off/on both selected 3,805 records against a 4,000 target. The 195 shortfall came from scarce hard examples. No quota redistribution is performed.

| Metric | Dedup off | Dedup on |
| --- | ---: | ---: |
| Above-threshold selected pairs | 39 | 0 |
| Duplicate candidates rejected | 0 | 27 |
| Largest selected cosine | 1.00000 | 0.91930 |
| Sampling wall time | 0.14 s | 0.43 s |

Threshold: 0.92; independent pairwise checks use float64 vectors and a 1e-6 tolerance. Timings cover this small pool and do not establish million-record performance. The disabled rerun selected exactly the same IDs/order as the full run. Its updated report includes every empty quota cell; the original full-run sampling report is retained unchanged.

## Reproduce the medium run

Use a compute allocation with a working pipeline/GPU environment and a Docker daemon. Pre-pull `python:3.12-slim`. This validates an existing environment, not a clean installation.

Start a Qwen3-4B teacher with a reasoning parser, maximum model length 16,384 and at least 32 concurrent sequences. The recorded server used `--reasoning-parser qwen3`, `--max-num-seqs 48` and eager execution. Start a Qwen3-8B helper with thinking disabled, maximum model length 8,192 and 48 sequences. Both expose authenticated OpenAI-compatible chat APIs. See [the smaller smoke instructions](README.md) for the server command structure.

From the repository root, using reachable server addresses:

```bash
export SFT_API_BASE=http://TEACHER_HOST:8000/v1
export SFT_MODEL=freeze-teacher
export SFT_JUDGE_API_BASE=http://HELPER_HOST:8001/v1
export SFT_JUDGE_MODEL=freeze-helper
export SFT_API_KEY=YOUR_SERVER_KEY
export SFT_RUN_ID=my_medium_run
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python smoke/prepare_scale.py
python smoke/scale_runner.py --config smoke/config-scale.yaml
```

The source preparer creates bounded snapshots of Python instructions, Dolly instructions with context, ARC training questions with answer choices, and 500 custom-source passages. Stage 1 also reads 5,000 GSM8K training prompts directly. References remain separate for auditing and are never supplied to the teacher. The source manifest records observed HF revisions and snapshot hashes; the direct Stage 1 and evaluation HF reads are not revision-pinned by this config. A changed source/config/model requires a fresh run ID.

The teacher uses a 6,144-token output budget; token-limited answers are rejected by Stage 6. The helper generates questions, annotates prompts and judges surviving answers. Every normal stage runs through the CLI without manual data transfer, followed by a no-work resume that must preserve JSONL hashes. Resource sampling records process-tree RSS, server token counters and GPU memory where `rocm-smi` is available.

For a separate audit, start a Qwen3-32B server with thinking disabled and maximum model length 24,576, then run:

```bash
python smoke/audit_scale.py --run-id my_medium_run \
  --api-base http://AUDITOR_HOST:8002/v1 --model freeze-auditor
```

The audit samples 160 accepted and up to 40 rejected records across domain strata, hides pipeline disposition from the reviewer, and supplies source references/passages where available. This is a model review with subsequent spot checks, not independent ground truth or a population-weighted accuracy estimate. The reviewer shares the teacher's model family.

The case-specific `recovery_check.py`, `compare_sampling.py` and `summarize_scale.py` preserve the recorded experiment. Their evidence paths/run labels should be isolated before repeating them. `gpu_exec.sh` captures this cluster's existing Apptainer setup, not a portable installer. Set `RAY_ADDRESS=local` and a private `RAY_TMPDIR` when running local Ray tests on a node with an unrelated Ray cluster.

## Scope and remaining limits

Cache invalidation after config changes, offline/distributed inference polish, clean installation and publication hygiene remain deferred as requested. The small teacher/config is a functional reference, not evidence of production answer quality. Code execution only tests the executed snippet; it does not establish that the answer solves the prompt. No mathematical correctness filter is enabled.

The tested ROCm image can return a zero process exit code after GPU initialization even when Python exits with an error. The smoke driver therefore requires completed checkpoints and validated artifacts before writing its success report. Other environments were not evaluated for this behavior.
