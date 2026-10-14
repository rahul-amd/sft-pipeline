## What would demonstrate that the data helps?

A completed pipeline run establishes that requests, artifacts and training records fit together. It does not establish that its answers are correct or that training on them improves a model. Our release evaluation should make those claims separately: software integrity, content quality and downstream usefulness. **The comparisons below are proposed experiments; their results are placeholders.**

### Compare datasets within the capability they target

For broad instruction tuning, start with [Tülu 3's SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). They provide relevant public instruction mixtures with different construction choices. For a reasoning-focused slice, add [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), whose math, code and science composition makes it a more specific comparator. Our broad mixture should not be declared better or worse from a reasoning-only average.

For long-input instruction tuning, use [LongAlign-10k](https://github.com/THUDM/LongAlign) as a candidate training baseline. Also include a simpler internal baseline: the same underlying documents with ordinary summary/QA supervision, compared with our deterministic task mixture and the full LC mixture. A long-output writing dataset would answer a different question; include one only if long-form generation becomes a stated target.

| Study | Suggested training arms | Primary question |
|---|---|---|
| General SFT | Simple internal baseline; our SFT data; Tülu 3 SFT; SmolTalk | Does the mixture improve held-out instruction following and task performance? |
| Reasoning slice | Matched-domain internal slice; OpenThoughts3 subset | Does the data help math, code and science at comparable budgets? |
| Long-context adaptation | Fixed short-context checkpoint; LongAlign; document-only baseline; our LC mixture | Does added LC supervision improve use of evidence as context grows? |

These are candidate comparisons, not a requirement to run every arm. Pin each data revision and record the exact subsets. Review their terms separately: the Tülu card explicitly notes varying component licenses and non-commercial portions. Do not infer uniform redistribution rights from a collection-level label. The public-dataset comparison measures the entire dataset package—including source and teacher differences—not the isolated causal effect of our pipeline.

### Hold the training contract fixed

Retrain every arm from the same student checkpoint and tokenizer, using one serialization policy, assistant-loss mask, optimizer, schedule and checkpoint-selection rule. Match total training tokens rather than record count; report loss-bearing target tokens as well, because reconstruction and short-answer retrieval allocate supervision differently. For LC experiments, match length buckets and curriculum, packing rules, attention masks and positional-extension settings. If exact token matching requires subsampling, freeze that sample before evaluation and do not silently repeat records to fill a quota.

Equal tokens do not necessarily imply equal compute, especially when sequence-length distributions differ. Report measured GPU-hours and throughput alongside token budgets. Add a compute-matched sensitivity study if efficiency is a claim. Run a feasibility pilot first, then use three paired training seeds for the selected comparison. Report seed variation, paired evaluation uncertainty and regressions, not just the best checkpoint. Keep generation-cost measurements separate from student-training cost.

### Evaluate transfer beyond the generators

For SFT, preregister an instruction-following evaluation and relevant held-out code, math and knowledge tasks. Specify exact dataset versions, scoring scripts, decoding, reasoning mode and output budgets: **[TBD: final evaluation suite]**. Published model leaderboards are useful context but cannot replace retraining matched students; their base models, post-training stages and inference budgets differ.

For LC, use application-oriented tasks from [HELMET](https://github.com/princeton-nlp/HELMET), with [LongBench v2](https://github.com/THUDM/LongBench) as an additional held-out reasoning evaluation. Report task-level scores at actual tokenizer-measured lengths, and sweep evidence position where the task permits it. Measure short-context performance after LC training to expose regressions. [RULER](https://github.com/NVIDIA/RULER) is a useful related-family diagnostic, but our RULER-style training tasks make it weak evidence of independent transfer. Keep that score separate and disclose template overlap.

**The checked-in SFT default recipe includes MMLU and HumanEval test splits as collection sources.** Those tests cannot establish held-out gains for a dataset made with that recipe. Before any comparison, specify a benchmark-clean release recipe and an exact source/split manifest, excluding every evaluation test set from construction; this is a proposed experiment configuration, not a change to frozen code.

Split source documents, near-duplicate families and task templates before constructing development or evaluation data. Neighboring chunks from one document must stay together. Audit public-baseline overlaps as well as our own outputs; exact matching and n-gram checks reduce detectable contamination but cannot establish absence of paraphrases or teacher pretraining exposure. The released training rows, the judge-calibration split and the final benchmark set must have distinct roles.

### Inspect the supervision itself

Start with two independent reviewers on a stratified sample, with disagreements adjudicated and uncertain labels retained. Blind them to dataset identity and pipeline acceptance. For source-generated SFT questions, first assess standalone clarity and subject identification without the source; an unfamiliar named entity alone is not a failure. Then reveal the source to assess answerability, support, and proposed-reference correctness. For answers, label factual correctness, instruction following and completeness separately. For LC QA, distinguish single-span retrieval, independent multi-span questions and genuinely dependent multi-hop reasoning. Remove a claimed supporting chunk and test whether the remaining context still suffices; report this as a diagnostic, not an automatic proof.

For summaries, score supported claims and coverage separately, including information from the beginning, middle and end of each document. A fluent summary can omit the main point; an exact source quote can exist without supporting an answer. Calibrate model judges against held-out human labels before using their scores as quality estimates. Report accepted-set precision, useful-answer rejection and retained yield together, with document-clustered uncertainty and sampling weights where needed. **[TBD: human review sample, adjudicated labels and calibrated error estimates.]**

### Identify what matters, and what it costs

Use matched source pools and cached teacher responses to separate selection from generation changes. The smallest useful SFT ablations compare random versus cluster/quota selection, grounding on/off, semantic deduplication on/off, and filtering policies. Semantic deduplication is optional and defaults off; mathematical correctness is evaluated externally rather than attributed to a nonexistent math filter. For LC, compare family mixtures, reconstruction versus permutation, and random versus related-document grouping. These are experimental recipes to propose after approval, not changes to the frozen code.

Record input documents, candidates, requests, valid outputs, retained examples and training tokens with explicit denominators. Include API retries, truncated responses, authoring/helper/judge tokens, embedding and sandbox work, latency and GPU-hours. Report cost per retained example and, after human review, cost per supported example. **[TBD: frozen-release dataset statistics, cost/yield table, downstream results and confidence intervals.]** Until those results exist, the defensible claim is a tested, inspectable data-production system with documented quality limits—not demonstrated superiority over other datasets.
