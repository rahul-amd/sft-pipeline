# Clef 27B: SFT quality classification and calibration

This isolated experiment repeats the Clef-Flash audit with `Cloudflare/clef` (27B), using the same 6,600 records, rubric, four labels, and frozen Sol reviews. It does not modify either data pipeline. Sol judgments are a comparison reference, not independently established ground truth.

## Results

All 6,600 inputs completed without request failures or truncation. The eight-replica bulk interval took 19.2 minutes. The main hissu allocation ended at **23:06 UTC on 27 October**, after 30 minutes 30 seconds; no experiment containers remained and the unused non-hissu fallback job was canceled. See `resource-release.json`.

Raw MCC on the full audit improved from **0.120 to 0.315** for SFT and **0.102 to 0.323** for Dolci. The fair calibrated comparison uses only the identical held-out half:

| Held-out method | SFT MCC (N=1,700) | Dolci MCC (N=1,600) |
|---|---:|---:|
| Flash, raw | 0.120 | 0.106 |
| Flash, calibrated | 0.324 | 0.297 |
| Clef 27B, raw | 0.287 | 0.334 |
| Clef 27B, calibrated | **0.398** | **0.422** |

Calibration passed the predeclared gate for both models: fitting-only out-of-fold log loss decreased by 23.0% for Flash and 20.7% for Clef. The calibrated Clef-versus-Flash MCC difference has paired 95% intervals of **[0.040, 0.110]** for SFT and **[0.091, 0.160]** for Dolci.

Calibrated Clef's major-defect recall is **63.5% / 79.2%** for SFT/Dolci. Among its no-defect decisions, **5.8% / 8.0%** are Sol-major, and only **70.3% / 55.4%** match Sol's no-defect label. Calibration reduces minor-defect recall and gives **zero recall for Sol-uncertain cases** in both held-out cohorts. This supports further review-routing experiments, not automatic acceptance or replacing Sol.

The first CPU post-processing attempt stopped at a baseline consistency assertion because exact probability ties were resolved differently by a new label-order argmax. Raw metrics now preserve the API's recorded choice, reproducing the original Flash report exactly. This affects four Flash and 39 Clef records; IDs are saved in `report/metrics.json`. No inference, probabilities, calibration coefficients, split, or selection rule changed. Slurm records the original combined job as failed because of that CPU assertion; inference and container cleanup had already completed, and analysis was repaired and verified on CPU after releasing the allocation.

## Design

- SFT: 3,400 records, 100 from each of 34 sources.
- Dolci: 3,200 records, 200 from each of 16 sources.
- Labels: `no_defect_found`, `minor_defect`, `major_defect`, `uncertain`.
- Model revision: `a30162b99d68a481d8bd75d38df069a0cf83df40`.
- Identical record content and questions to `../systemone_classifier`; no source identifiers or Sol decisions are included in classifier inputs.
- Full inputs: 66,876,161 encoded tokens in total, maximum 77,065 per record; 1,265 inputs exceed 16,384 tokens. The local server adapter lifts the encoder default and rejects truncation. The prepared token lengths match the previous Flash experiment exactly.

`protocol.json` and `calibration-split.json` were frozen before new inference. Each source is split equally into calibration and held-out records, giving 3,300 records in each half. No exact duplicate full input states cross the split. This does not establish the absence of near duplicates or shared documents.

For each model, fit one pooled L2-regularized multinomial logistic regression on the four clipped log probabilities (`C=1`, fixed seed, no source features). Adopt it only if five-fold out-of-fold log loss on the calibration half improves by at least 1%. Fit the selected calibrator on the calibration half, then evaluate both models on identical held-out records. No held-out labels are used for fitting or the adoption gate. This is more flexible than temperature scaling: it can change the predicted class.

Report the original raw comparison on all 6,600 records separately from the held-out calibrated comparison. Use MCC for categorical correlation, macro-F1, accuracy, log loss, Brier score, ECE, major-defect recall, and the fraction of Sol-major records among classifier no-defect decisions. Confidence intervals use 1,000 paired bootstrap replicates within dataset/source strata, conditional on the fitted calibrators and reference labels. The held-out set is held out from fitting, not a newly collected benchmark; prior Flash aggregate results were already known.

## Execution and resource limits

Eight independent BF16 replicas run on eight MI325X GPUs, one model per GPU, with two concurrent requests per replica. vLLM's native `/v1/systemone` endpoint returns joint-head probabilities with no generated answer tokens. Its generic implementation class is named `ClefFlashForJointSchema`; the model loaded here is the pinned **27B Clef** checkpoint with its own head dimensions. Server logs record approximately 50.37 GiB of model weights per replica.

Pinned image: `vllm/vllm-openai-rocm:nightly-rocm72-5d35ce3934a7804b298559d2d99eec33792e1eb9`. See `runtime-manifest.json` for the image digest and exact settings. Model caches, logs, requests, predictions, and reports stay under this folder.

The user temporarily authorized hissu for two hours. `resource-authorization.json` overrides only the original protocol's partition restriction. The hard deadline is **2026-10-28 00:30 UTC**; an independent watchdog starts cleanup at 00:28:30 and Slurm job 56641 has an earlier 1h45 wall limit. The scripts reject further hissu use after this window. Do not extend or reuse this authorization.

The benchmark uses parallel replicas unlike the Flash run, so elapsed inference time is not a controlled comparison of model compute efficiency.

## Artifacts and reproduction

- `questions.json`: frozen rubric and class definitions.
- `experiment-manifest.json`: input hashes, token counts, reference encoder parity checks, and model provenance.
- `predictions.jsonl.gz`, `reference_labels.jsonl.gz`: compressed model responses and Sol references.
- `calibration-fit-{flash,clef}.json`: fitting-only diagnostics, adoption gate, and portable calibration coefficients.
- `report/index.html`: self-contained visual report.
- `report/metrics.json`: full results, source-level results, and paired intervals.
- `report/heldout_predictions.csv`: held-out labels and probabilities for both models and policies.
- `inference-verification.json`: completeness, payload identity, and full-context checks.
- `result-verification.json`: independent sklearn metric checks and report link checks.
- `analysis-environment.json`: exact Python and analysis-library versions.

`prepare.py` reconstructs requests from the previous Flash experiment and checks native/reference encoding. `shards.py assign` balances requests by input length without labels. `classify.py` validates probabilities and exact token usage, caches atomic responses by payload/configuration hash, and stops a shard after persistent request failures. `shards.py merge` refuses incomplete evidence.

After complete inference, `calibrate.py` fits both fixed calibration procedures and produces the report; `verify.py` independently checks the results. They require the previous Flash experiment alongside this folder. `verify.py` additionally needs its uncompressed requests and prediction JSONL files; those files are intentionally excluded from Git. No GPU is needed to rerender or verify the completed analysis.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python smoke/systemone_classifier_clef/calibrate.py
PYTHONDONTWRITEBYTECODE=1 \
  python smoke/systemone_classifier_clef/verify.py
```

Review `job.sh`, `run.sh`, and `container_exec.sh` before a new cluster run. They contain site-specific paths, Slurm GPU-to-render-device mapping, and the expired temporary hissu authorization. A new run needs its own authorized allocation; do not copy the old hissu lease.

## Interpretation limits

Agreement with Sol alone cannot validate a production quality filter. Calibration targets this source-balanced audit mixture and may shift under deployment prevalence. Inspect blinded disagreements with human adjudication, then validate any acceptance threshold on fresh records. Do not select a threshold on this held-out set and report its performance as independently validated.
