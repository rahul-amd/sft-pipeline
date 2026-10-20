# LC section fact-check notes

Reviewed the frozen checkout under `/shared_silo/scratch/rahul.aralikatte@amd.com/lc-pipeline`. Only blog resources were written. The section deliberately avoids carrying over stronger claims from module docstrings and README prose.

## Code-backed statements

| Claim | Local evidence | Editorial caveat |
|---|---|---|
| PT `text`; SFT `messages`; task-specific auxiliary metadata | `ruler_pp/cli.py:37`, `multihop/multihop.py:306`, `multihop/multihop.py:318`, `summarize/summarize.py:170`, `summarize/summarize.py:181` | Do not say every family has identical JSON schema or a shared `Sample` instance; LLM assembly has custom row shapes. No trainer loss-mask implementation implied. |
| Full training-length budget, HF chat template when present | `ruler_pp/output.py:10`, `ruler_pp/output.py:26`; `ruler_pp/cli.py:59`, `ruler_pp/cli.py:119` | Guarantee is specific to selected tokenizer/template and configured cap; future trainer templates can differ. LLM prepare allows no cap (`None`), so specify configured cap rather than saying all invocations are bounded. |
| Explicit whitespace approximation; tokenizer load errors propagate | `ruler_pp/lengths.py:9`, `ruler_pp/lengths.py:42` | Whitespace tokenizer actually counts regex word/punctuation chunks. It is not exact model-token counting. |
| Content/settings-based identifiers and atomic final JSONL | `ruler_pp/output.py:26`, `ruler_pp/output.py:40` | These IDs are not semantic deduplication. Prepared request/doc files are written directly and covered by a completion manifest; avoid implying every intermediate artifact is atomic. |
| Per-shard schemas, metadata gates, stable provenance | `corpus/loader.py:67`, particularly `:87`, `:112`, `:116`, `:119`, `:123` | Missing score columns skip that filter. Existing duplicate metadata drops all members with cluster size >1, not representative selection. Source token metadata may use another tokenizer. |
| Incomplete/duplicate corpus join fails | `corpus/loader.py:132` | Final task metadata does not uniformly expose individual passage source IDs. Prepared document bundles and cluster membership files carry fuller provenance; do not claim per-passage lineage everywhere. |
| Prefix embedding, default 256 tokens, Qwen3 model | `corpus/cluster.py:67`, `corpus/cluster.py:166` | Earlier README summaries might use a different example prefix length. Character preclip precedes tokenizer truncation; call it a configurable prefix, not an invariant exact token count. |
| Exact neighbor search, greedy single linkage, metadata token budget | `corpus/cluster.py:87`, `corpus/cluster.py:115`, `corpus/cluster.py:168` | “Exact” denotes exhaustive neighbors, not approximate-nearest-neighbor indexing; similarity arithmetic uses fp16. “Mutually near docs” in the module docstring is stronger than the algorithm proves. Search is quadratic. |
| Exactly 22 synthetic task types, ten graph tasks | `ruler_pp/registry.py:23`; `ruler_pp/tasks/graph.py:228` onward | The section table lists 7 word/count/set +4 record/arithmetic +1 tracking +10 graph =22. Task types sample additional operation variants. Do not equate task count with a benchmark suite or claim benchmark replication. |
| Numeric and set variants | `ruler_pp/tasks/numeric_agg.py:15`; `ruler_pp/tasks/set_ops.py:14` | Mean uses two-decimal Python rounding. No math-verification result from the SFT pipeline is implied. |
| Size fitting, seeded per-example generation, workers | `ruler_pp/cli.py:59`, `ruler_pp/cli.py:119`; `ruler_pp/lengths.py:54` | Seeds plus frozen implementation/environment give reproducibility; not an unconditional cross-version bitwise promise. |
| Untie targets and both source modes | `untie/knots.py:52`, `untie/knots.py:123`, `untie/knots.py:209`, `untie/knots.py:217`, `untie/knots.py:251` | Units can combine paragraphs or use fallback splitting. Source rank is only a construction target; ordering recoverability remains unmeasured. |
| MRCR exactly two markers, confounders, prefix copying | `mrcr/mrcr.py:203`, `mrcr/mrcr.py:228`, `mrcr/mrcr.py:270` | It is the repo's passage-marker MRCR-style task; do not imply implementation of all formats in an external MRCR benchmark. |
| IDK answerable, absent, wrong-attribute cases | `idk/idk.py:73`, `idk/idk.py:140` | No “leakage-proof” or natural abstention calibration claim. Binding syntax conflict filtering is narrow. |
| Chunk preparation, default 3–4 chunks, same-document QA | `multihop/multihop.py:47`, `:87`, `:103`, `:157`, `:345` | “Multi-hop” describes authoring intent. No claim-support or per-chunk necessity gate exists. Random sampling can repeat combinations. |
| Full source document in final QA; normalized question dedup | `multihop/multihop.py:275`, `:294`, `:318` | First normalized question wins, irrespective of answer quality. This is neither semantic question dedup nor final global dedup. |
| Full-document summary and optional word target | `summarize/summarize.py:41`, `:54`, `:62`, `:181` | “Faithful” is a prompt requirement, not a measured property. Approximate requested words are not enforced output length. |
| Manifest, bundle integrity, strict IDs/completion checks | `corpus/llm.py:28`, `:35`, `:48`, `:71`, `:117` | Missing `finish_reason` is accepted. Explicit `--allow_partial` allows missing only; duplicates/errors still fail. Legacy bundles require explicit tokenizer and cap, but no historical manifest hashes. |
| API client concurrency, retries, atomic final responses | `corpus/infer.py:20`, `:45` | Materializes request list; no resumable distributed engine. Do not claim the external dispatcher itself is shipped or tested by this repository. |

## Recorded validation evidence

Use `smoke/RESULTS.md:1` onward as the readable evidence source. It records:

- 108 regression passes (`runs/regression.xml` / `runs/regression.log`).
- 278 matrix rows, 123 final files, 41 commands, 379.3 seconds (`runs/e2e-02/RESULT.json`).
- 36 shell-orchestrator rows and 14 API responses (`runs/orchestrator-03/RESULT.json`).
- 34 JSONL files unchanged on a completed-stage resume.
- A post-matrix QA dedup fix checked on saved real responses and a fresh shell run. Do not say the original matrix includes that final fix.
- All 22 synthetic types at 8K/32K in PT/SFT, both Untie flows/source modes, MRCR, all IDK cases, real prepare/infer/assemble for QA/summaries. Three synthetic types exercised 128K budget serialization only.
- Maximum training length 131,054. Real LLM requests sampled roughly 3K–14K-token documents. No 128K model-inference demonstration.
- Qwen3-8B authoring, thinking disabled; Qwen3-Embedding-0.6B GPU embedding. These are smoke settings, not verified settings for historical release data.
- Tests reused an existing ROCm Apptainer environment. Clean installation was not tested.

The reference artifact paths under `smoke/runs/` are local/ignored; a public article should link to deliberately exported evidence or describe them as internal smoke validation. They must not be presented as public downloadable attachments unless copied/sanitized into the release package. No regression, integration, inference, training, or benchmark jobs were run during this writing task.

## Quality and release caveats to preserve

`smoke/QUALITY_REVIEW.md:1` explicitly calls the review a small diagnostic sample, not a calibrated estimate. Its confirmed curriculum-date error is **September 2022 → June 2022**; the same document's summary retained September. The section uses only the factual example, with no personal contact details. The review also distinguishes independent multi-span retrieval from dependent multi-hop reasoning, records an incomplete answer, and proposes human calibration before judge gating.

`smoke/AUDIT.md` remaining-work section supplies the explicit limits: source/cluster splits before generation, final dataset deduplication, source metadata limitations, configuration-insensitive completion markers, clustering/assembly memory bounds, clean-install and licensing work. These are release-review placeholders, not work to implement during the frozen-code blog task.

Suggested controlled comparisons for integration into the parent evaluation section: equal-token training, leave-one-family-out ablations, single versus clustered Untie, random versus entity-linked QA chunk selection (future experiment; entity-linked selection is not implemented), QA evidence validation plus leave-one-chunk-out tests, summary support/coverage by position, and evaluation on contamination-checked held-out tasks. No results for these comparisons currently exist in the cited LC evidence.
