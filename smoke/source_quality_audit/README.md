# Final SFT source quality audit

Open [`report/index.html`](report/index.html) in a modern browser for the standalone,
offline report, charts, and searchable original records with reviewer evidence.
Download the file first when viewing this repository on GitHub.

The audit covers 3,400 records: 100 uniformly sampled from each of 34 sources in
`runs/research_stage3_001/stage6_v2_decontam/clean` (5,831,367 retained records).
The sampling seed is 20261026. One model, GPT-6.1 Sol with high reasoning effort,
reviewed six quality dimensions. Operational retries and schema/evidence repairs
used the same model; there was no independent adjudication or external execution.
These are reviewer judgments, not verified error rates.

The reviewer flagged 888 sampled records with major defects. Weighting each source
by its retained population gives a 27.2% major-defect estimate. The report explains
sampling uncertainty, source differences, examples, limitations, and recommended
follow-up checks.

## Included evidence

- `samples.jsonl.gz`: lossless compressed original samples, including lineage.
- `verdicts.jsonl`: all 3,400 final validated reviews.
- `sources.json`, `sample-manifest.json`, and `review-manifest.json`: populations,
  sampling metadata, input hashes, and reviewer configuration.
- `completion.json`: final completion status (3,400 records, 629 batches).
- `sample.py`, `review.py`, `repair_reviews.py`, `review_prompt.txt`, and
  `review-schema.json`: sampling and review implementation used for this audit.
- `report/`: analysis, CSV summary, HTML source/assets, figures, and browser
  verification script, results, and preview screenshots.

Intermediate model requests, retries, batch caches, logs, temporary browser
dependencies, and stale progress snapshots remain local and are excluded from Git.
The compressed samples preserve the exact original JSONL bytes and manifest hash.

## Reproduce the analysis

From the repository root, with Python and Matplotlib installed:

```bash
gzip -dk smoke/source_quality_audit/samples.jsonl.gz
PYTHONDONTWRITEBYTECODE=1 python smoke/source_quality_audit/report/build_report.py
```

Skip decompression if `samples.jsonl` already exists. The builder checks the sample
hash, unique IDs, source counts, and frozen verdict totals before rebuilding the
analysis and report. No model calls or access to the full dataset are needed.
The HTML also embeds the sampled records and final reviews for offline browsing.

`report/verify_report.py` checks loading, filters, search, record inspection,
pagination, CSV export, mobile layout, JavaScript errors, and external requests.
It expects Playwright in `report/.preview-deps`, Chromium in `report/.browsers`,
and the local temporary directory `report/.tmp`; optional host library shims live
in `report/.browser-libs`. These machine-specific dependencies are not committed.
Recorded successful checks are in `report/verification.json`.

The sampling and model-review scripts are retained as historical run tooling.
Sampling again requires the original dataset; reviewing again requires an
authenticated Codex CLI and model access. They write audit outputs, so use a
separate working copy when rerunning them. Repair tooling additionally requires
the locally retained operational artifacts. Analysis from the frozen results
does not require those artifacts or any new review calls.
