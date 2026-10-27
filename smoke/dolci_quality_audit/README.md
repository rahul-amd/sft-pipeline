# Dolci-Think-SFT-7B quality audit

Download and open [`report/index.html`](report/index.html) in a modern browser.
It is a standalone, offline report containing charts, source breakdowns, a
carefully qualified comparison with our SFT audit, and an explorer of all 3,200
reviews and complete original conversations. GitHub's file viewer does not run
the interactive HTML.

The frozen sample contains 200 physical records from each of the 16 exact
`dataset_source` labels in `allenai/Dolci-Think-SFT-7B`, revision
`83f618e2fe0aed29ee3440a3658a0326c175d09d`. The source population is 2,268,178
records across 156 Parquet shards. Sampling is uniform without replacement
within each source, using seed 20261027 and independent source RNGs. Duplicate
original IDs are not silently deduplicated.

GPT-6.1 Sol at high reasoning effort assessed six dimensions. Every assistant
turn was evaluated with the context available at that turn. The earliest
technically valid review was retained; only operational, schema, or evidence
failures triggered retries. There was no second reviewer, human adjudication,
external fact-checking, or code execution. All 3,200 records and 1,226 batches
completed, and every issue quote was validated against original text.

The reviewer assigned 1,337 major, 1,044 minor, 157 uncertain, and 662 no-defect
labels. Population weighting gives 55.4% major labels, compared with 41.8% in the
equal-source sample. These are model judgments, not verified error rates. The
report discusses source concentration, prompt ambiguity, conversation length,
and why aggregate differences from our earlier audit do not establish a training
quality ranking.

## Reproduce the report

From the repository root, with Python and Matplotlib installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python smoke/dolci_quality_audit/report/build_report.py
```

The builder accepts the checked-in `samples.jsonl.gz` directly. It checks the
sample/review hashes, complete coverage, and exact evidence excerpts before
building the report. It uses the historical validator and comparison aggregates
from `smoke/source_quality_audit/` in this repository. No model calls are made.

To reproduce the root-level aggregates with `summarize.py`, first decompress the
sample (skip this step if `samples.jsonl` already exists):

```bash
gzip -dk smoke/dolci_quality_audit/samples.jsonl.gz
PYTHONDONTWRITEBYTECODE=1 python smoke/dolci_quality_audit/summarize.py
```

## Evidence and tooling

- `samples.jsonl.gz` preserves the exact original sample bytes and manifest hash.
- `verdicts.jsonl` contains all final reviews and provenance pointers.
- `sources.json`, `sources.csv`, `shards/`, and `inventory-manifest.json` record the
  complete source inventory; `sampling-plan.json` records the chosen row ranks.
- Sampling/review manifests, `protocol.json`, the rubric and schema, and
  `completion.json` document the run and validation policy.
- `analysis.json` and `source_summary.csv` contain final aggregate results;
  `report/analysis.json` adds report-specific analyses and example IDs.
- `report/figures/` contains exportable SVG and PNG charts.
- `report/verify_report.py`, its recorded results, and preview screenshots cover
  offline loading, filters, CSV totals, desktop/mobile layout, and full message
  preservation, including a conversation with 20 assistant turns.

The browser check uses Playwright and the local Chromium installation from the
earlier SFT audit. Browser paths can be overridden with the environment variables
documented in the verification script. Dependencies and temporary profiles are
excluded from Git. All new verification artifacts stay in this audit folder.

The inventory, sampling, reviewer, and driver scripts are retained for provenance.
Rerunning sampling requires Hugging Face access; model reviews require an
authenticated Codex CLI with access to the recorded model. Use a separate working
copy for a new audit: the scripts write audit outputs. Intermediate requests,
retry artifacts, batch caches, logs, stale progress snapshots, and host-specific
runtime files remain local. Review artifact paths in final verdicts identify
those local execution traces; the accepted assessments and original evidence
needed to inspect them are included here.
