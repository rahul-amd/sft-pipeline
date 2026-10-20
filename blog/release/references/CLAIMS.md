# Evidence boundaries for the joint article

This is a compact editorial ledger of previously recorded results, not a new experiment report. The [machine-readable evidence snapshot](validation-evidence.json) identifies source reports and their hashes. Full internal smoke artifacts are not bundled here; public evidence links remain a release task.

| Claim in the article | Supporting record | What it establishes | What it does not establish |
|---|---|---|---|
| SFT latest flow: 288 tests; 20 API completions → 18 exports, four from Stage 2 | `sft-pipeline/smoke/grounding-validation.md` | Exercised current grounded generation, GPU, API, Docker, audit/export paths at small scale | Representative correctness, large-run throughput, clean installation |
| Earlier SFT run: 10,926 merged → 3,805 selected/inferred → 3,131 exports in 51m14s | `sft-pipeline/smoke/freeze-validation.md` | An unattended medium run under its recorded configuration | Dataset release inventory; current grounding/judge performance |
| Semantic dedup: 39 → 0 selected pairs above 0.92 cosine; 3,805 selected in both runs | Same earlier SFT report | Enforcement of the configured embedding constraint on that pool | Semantic equivalence of removed prompts or downstream gains |
| Current judge got 15/16 on a small fresh set with each of two tested teachers | `sft-pipeline/smoke/grounding-validation.md` | A diagnostic result on assistant-curated cases | Human-calibrated or population-level accuracy; zero false accepts |
| LC: 108 tests, 278-example matrix, 36-example shell run | `lc-pipeline/smoke/RESULTS.md` | Task-family, format, tokenizer, API and orchestration checks | 314 independent quality-reviewed records; final release inventory |
| LC synthetic maximum measured length 131,054 tokens | Same LC report | Serialization fit for three synthetic task types at a 128K cap | 128K LLM inference or demonstrated comprehension |
| LC content review found a September→June 2022 answer error | `lc-pipeline/smoke/QUALITY_REVIEW.md` | Concrete evidence that structural success can retain factual errors | A representative error-rate estimate |
| Current pipeline architecture and defaults | [SFT implementation notes](../sections/sft-fact-check.md), [LC implementation notes](../sections/lc-fact-check.md) | Read-only inspection of the frozen working trees | That historical large datasets used every current feature |
| Public dataset comparison choices | [Primary external sources](external-sources.md) | Plausible candidate baselines and evaluation resources | A result favoring either released dataset |

The SFT description includes uncommitted grounding/judge work, so the observed HEAD alone does not identify all described behavior. The LC working tree also contains a pre-existing local script adjustment. [frozen-code.json](frozen-code.json) records the initial tracked-file hashes for this writing task. Final release tags and historical generation revisions must be supplied separately.

Retain these distinctions when editing: optional semantic dedup defaults off; the SFT judge samples 10% by default; code execution is not functional correctness verification; source quotes do not prove entailment; LC model-authored QA has no implemented proof of multi-hop dependence; source/split overlap must be evaluated for the actual data recipe.
