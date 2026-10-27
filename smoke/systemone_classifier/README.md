# Clef versus Sol: four-class SFT review experiment

This isolated experiment evaluates Cloudflare/Clef-Flash on the same 3,400 SFT
and 3,200 Dolci records already reviewed by GPT-6.1 Sol. It changes no pipeline
code. Sol is a reference reviewer, not independently verified truth.

## Results

All 6,600 predictions completed and passed full-input validation. The current
four-class argmax is too permissive to replace Sol: it finds very few of Sol's
major defects. Major-defect probabilities carry a more useful ranking signal,
which warrants a separate calibration and held-out routing experiment.

| Audit | N | Multiclass MCC | Macro-F1 | Major-defect recall | Major-defect AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 3,400 | 0.120 | 0.208 | 7.9% | 0.723 |
| Dolci | 3,200 | 0.102 | 0.167 | 9.6% | 0.769 |

The AUROC supplement excludes Sol-uncertain records; primary metrics retain
all four classes. See [the HTML report](report/index.html) for confidence
intervals, confusion matrices, source results, and limitations.

The recovered bulk run took 74.0 minutes including its infrastructure
interruption, at 15,042 accepted input tokens/s. The uninterrupted resumed
segment processed 2,533 records in 21.7 minutes at 19,685 input tokens/s.
Initial startup and the six-record pilot are excluded from these figures.
All experiment containers and allocations were released.

## Protocol

- One fixed four-class decision: `major_defect`, `minor_defect`, `uncertain`,
  `no_defect_found`. Class definitions and precedence match the earlier audits.
- `questions.json` was frozen before classifier inference. No prompt selection,
  fitting, threshold tuning, or calibration uses these evaluation labels.
- Preserve the complete original content, including raw and parsed answer
  views. Exclude source identifiers, Sol judgments, and Sol explanations from
  requests. Audit IDs are retained only outside the model payload for joining.
- Use the native vLLM `/v1/systemone` pooling endpoint. The decision head returns
  all four probabilities without generating reasoning or output tokens.
- Choose six pilot records by input length: shortest, median, and longest in
  each dataset. This validates the API and full-context handling, not quality.
  Reuse those predictions in the final 6,600-record evaluation.
- Primary correlation is generalized multiclass Matthews correlation (MCC).
  Also report kappa, macro-F1, accuracy, class precision/recall, and confusion
  matrices. `uncertain` is a nominal class, never an ordinal severity score.
- Report equal-record agreement, source-population-weighted agreement within
  each dataset, per-source agreement, and descriptive length subgroups.
- Confidence intervals use 1,000 bootstrap draws within dataset/source strata.
  They condition on the frozen labels and omit systematic reviewer uncertainty.

## Context handling

The checkpoint supports 262,144 positions, but the pinned vLLM adapter and
reference encoder default to a 16,384-token cap. `serve.py` raises only this
adapter cap locally and rejects any request that would be truncated. It does
not change model weights or global vLLM files. The server is configured for
98,304 positions because the largest actual input is 77,065 tokens. All 1,265
inputs above 16,384 tokens must be evaluated in full. Every accepted prediction
must report exactly the independently prepared encoded token count and zero
output tokens. Position capacity alone does not validate decision-head quality
at long lengths; the report explicitly separates this limitation.

## Reproduction

Model revision and input hashes are in `experiment-manifest.json`. The Docker
image is pinned in `container_exec.sh`. Cache directories, model files, logs,
requests, and individual predictions stay under this folder and are ignored by
Git. The original audit sample files and verdicts are prerequisites.

Inside a suitable Slurm GPU allocation, from the repository root:

```bash
bash smoke/systemone_classifier/container_exec.sh /work/download_model.py
bash smoke/systemone_classifier/container_exec.sh /work/prepare.py
CLEF_CONTAINER_NAME=clef-quality-experiment bash smoke/systemone_classifier/serve.sh
```

The server command stays running. From another shell with access to its node:

```bash
python smoke/systemone_classifier/classify.py --url http://GPU_NODE:18971 --pilot
python smoke/systemone_classifier/classify.py --url http://GPU_NODE:18971
python smoke/systemone_classifier/analyze.py
```

The client requires `aiohttp`; analysis requires NumPy, scikit-learn, and
Matplotlib. The GPU container includes model/runtime dependencies. Within a
Slurm allocation, use `srun --ntasks=1` to avoid accidentally starting duplicate
containers. Stop only the experiment's named container and release its
allocation after completion.

The wrapper resolves allocated DRM cards to GPU UUIDs. Numeric ROCr ordinals
and DRM/Slurm card numbering differed on the resumed node; using the UUID
avoids selecting the wrong physical device. Set `CLEF_DETACH=1` for a detached
server, and `CLEF_KEEP_CONTAINER=1` to preserve its logs after exit. In that
case, remove the named container explicitly after saving its logs.

This recorded run was interrupted by a Docker stop after 4,067 successful
predictions. Its initiator was not identified. The remaining requests resumed
on another hissu node with the same model, rubric, encoding, and precision.
`run-segments.json` records the interruption. For this resumed run, execute
`python smoke/systemone_classifier/finalize_timing.py` before analysis; it
combines the segments and preserves the resumed segment's separate completion
record. The final bulk wall time includes the interruption and restart.

## Outputs and interpretation

`predictions.jsonl` preserves all decisions, probabilities, token counts, and
timings; `completion.json` records bulk execution. `report/index.html` presents
the results, with machine-readable metrics and CSV comparisons alongside it.
The report is generated only after exact coverage of all 6,600 records is
verified. Failed requests are never treated as good samples.

Compressed prediction and reference-label archives are included. To regenerate
the report without repeating inference, decompress both `.jsonl.gz` files in
this folder, then run `analyze.py`. `verify_results.py` additionally checks the
original audit files, prepared requests, and agreement statistics against
scikit-learn. `result-verification.json` records the successful checks for this
run. The ignored `docs/` folder contains working copies of upstream references;
the pinned model revision and Docker digest are the runtime provenance.

Execution uses BF16 eager mode on one AMD MI325X GPU at a time. Throughput
includes client overhead, the extra server-side token-validation pass, and
the infrastructure interruption. It excludes initial download, initial server
startup, and the six-record pilot. API-reported input tokens include reused
prefix tokens. This is an agreement experiment, not an optimized inference benchmark.
Sol's remote-service hardware is unknown; these measurements cannot establish
like-for-like compute, dollar, or energy savings. Any subsequently chosen
filtering/routing policy needs a separate held-out evaluation.

References: [vLLM structured decisions](https://docs.vllm.ai/en/latest/models/structured_decisions_models/),
[Clef-Flash model](https://huggingface.co/Cloudflare/clef-flash),
[checkpoint configuration](https://huggingface.co/Cloudflare/clef-flash/blob/f0b7d3cd65296abfe459e6c228b618f5d8b26d2f/config.json).
