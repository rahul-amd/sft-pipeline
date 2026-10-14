# Proposed experiments for the release article

**Status: suggestions only. No experiments in this document have been launched.** Recorded software-validation results are in [the evidence ledger](references/CLAIMS.md); they must not fill these result tables.

The strongest article would show both that the generated supervision is sound and that students benefit from it. A large generation run alone cannot answer either question. Prioritize a small, controlled comparison before choosing the final production scale.

## 1. Establish the dataset being evaluated

Complete the [release manifest](release-manifest.template.yaml) for each immutable dataset revision. Measure records, unique source documents, serialized tokens, loss-bearing tokens, length percentiles/buckets, domain/task mix, languages, duplicates, and removal reasons. Publish denominators: a selected-out prompt is different from an answer rejected by a correctness judgment.

Freeze source splits and benchmark exclusions before constructing evaluation data. The current SFT example recipe collects MMLU and HumanEval test splits; datasets made from that recipe cannot use those tests as held-out evidence. Record whether the actual historical release recipe used those sources. Inspect external baseline overlap as well. For LC, separate source/near-duplicate families and generator templates, and distinguish independent evaluation from RULER-family diagnostics.

**Deliverable:** release inventory table, source/split manifest, length-distribution figure, and contamination report. All values remain **TBD** until measured on the selected release artifacts.

## 2. Review content before paying for full comparisons

A practical starting proposal is **200 SFT records and 200 LC records**, independently labeled by two reviewers with adjudication. This is a pilot allocation, not a statistically justified final sample size. Use its observed variation and error frequency to choose the final sample size and desired interval width. For SFT, stratify by domain, source-generated versus collected prompts, and filter/judge state. For LC, cover task family, length, and single versus clustered sources. Oversample model-authored and previously problematic cases, but retain weights for population estimates.

For source-generated SFT questions, review standalone clarity and subject identification without showing the source; unfamiliar named entities alone should not cause rejection. Then reveal the source to assess answerability, support, and reference correctness. LC questions intentionally depend on their documents: review answerability with the supplied document visible, then assess support and evidence necessity. Score the answer separately for correctness, completeness, and instruction following. Label LC QA as single-span, independent multi-span, or dependent multi-hop. Score summary support and coverage across source positions. Recompute deterministic golds with independent checks; construction-based answers should not be pooled with model-authored factual judgments as one undifferentiated accuracy number.

Review a separate sample of rejected candidates to estimate useful-data loss. Define an eligible rejected population that can be labeled; malformed transport failures and ordinary quality rejections have different denominators. Judge calibration uses separate development and held-out human-label sets. Report uncertainty clustered by source document, and adjudication/disagreement rates.

| Dataset/slice | Reviewed records | Supported and correct | Instruction-complete | Useful candidates rejected | Interval method |
|---|---:|---|---|---|---|
| SFT, collected prompts | TBD | TBD | TBD | TBD | TBD |
| SFT, source-generated prompts | TBD | TBD | TBD | TBD | TBD |
| LC, model-authored QA | TBD | TBD | TBD | TBD | TBD |
| LC, summaries | TBD | TBD | TBD | TBD | TBD |

**Deliverable:** documented rubric, de-identified labels where permitted, supported-answer estimates, rejection/yield analysis, and a small gallery of representative successes and failures. Select examples by a stated rule instead of choosing only attractive outputs.

## 3. Run the smallest useful student comparison

First do one feasibility pilot per selected recipe to validate memory, templates, reasoning serialization, packing, and evaluation parsing. Choose a student checkpoint with the needed context support. Freeze the training and evaluation plans before the scored runs. Suggested main studies:

| Study | Initial trained arms | Additional reference | Optional later arms |
|---|---|---|---|
| General SFT | Our SFT; Tülu 3 SFT subset; simple internal prompt-selection baseline | Same student checkpoint before SFT | SmolTalk; domain-matched OpenThoughts3 reasoning study |
| LC adaptation | Our LC mixture; LongAlign subset; document-only summary/QA baseline | Same short-context checkpoint before LC adaptation | Family ablations; combined SFT+LC recipe |

The simple SFT baseline should use the same candidate pool and teacher settings with random selection at a matched inference budget. The document-only LC baseline should use the same source pool where feasible. Public datasets differ in sources and teachers, so those comparisons measure dataset packages rather than isolating the pipeline algorithm.

For the chosen main comparison, propose three paired training seeds per arm. Keep the student/tokenizer revision, training template, loss mask, optimizer, schedule, checkpoint rule, and decoding fixed. Set budgets in **actual serialized training tokens**, with loss-bearing tokens reported separately. Match LC length distributions/curricula and packing; report measured compute rather than assuming equal tokens imply equal FLOPs. Use a compute-matched secondary study only if making an efficiency claim.

For LC adaptation, start every arm from the same short-context checkpoint. Do not let each arm inherit a different SFT dataset. A later joint study can use a factorial design: reference SFT versus our SFT, crossed with reference LC versus our LC. That joint training recipe is a proposed experiment, not an implemented cross-repository orchestrator.

Select the held-out SFT suite before training: instruction following and relevant code, math, and knowledge tasks, excluding construction/calibration overlap. Use HELMET and explicitly pinned LongBench v2 tasks for LC transfer, plus short-context regression tests. Report task scores and actual input lengths. Evidence-position sweeps and source substitutions can test whether the student uses context. Show all selected tasks and seeds; do not select the suite after seeing results.

| Arm | Data revision | Train / target tokens | Train GPU-hours | Instruction following | Verified task scores | LC score by length | Short-context regression |
|---|---|---|---|---|---|---|---|
| Our SFT | TBD | TBD | TBD | TBD | TBD | Not the primary study | TBD |
| Public SFT baseline | TBD | TBD | TBD | TBD | TBD | Not the primary study | TBD |
| Simple SFT baseline | TBD | TBD | TBD | TBD | TBD | Not the primary study | TBD |
| Our LC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Public LC baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Document-only LC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Deliverable:** the proposed [comparison figure](figures/comparison-placeholder.svg), populated from immutable artifacts with paired-seed variation and evaluation uncertainty. Keep each metric in its native units; do not average unrelated raw scores. Public leaderboard numbers are not substitutes for these matched student runs.

## 4. Explain any observed gain with targeted ablations

Do this after the initial comparison identifies a useful capability or failure. Avoid running every switch combination.

| Question | Controlled contrast | Outcome to inspect |
|---|---|---|
| Does prompt selection help? | Random vs cluster/quota selection from one pool, fixed teacher budget | Coverage, duplication, held-out student scores |
| Does grounding help? | Same generated candidates, grounding on/off | Supported standalone questions, false rejection, generation cost |
| Does semantic dedup help? | On/off at a fixed threshold; same selection target, report underfill | Similar pairs, retained diversity, student scores |
| Do filters help? | Fixed raw teacher outputs, staged filter policies | Human-supported precision, useful-data loss, student scores |
| Which LC tasks help? | Full mix vs selected family removed, matched length/token budget | Relevant task transfer and short-context regressions |
| Is ordering recoverable? | Untie reconstruction vs permutation; single vs clustered documents | Human ordering ambiguity, learning value, target-token cost |
| Is QA actually multi-hop? | Remove a claimed supporting chunk; compare independent vs dependent questions | Necessity of evidence and downstream transfer |

For filtering and selection, report two views if feasible: a fixed generation budget, where yield can differ, and a fixed training-token budget, where replacement data must be declared. The fixed generation-budget result captures the operational tradeoff; the fixed training-budget result helps compare training utility. Never silently fill rejected slots with duplicate rows.

## 5. Report generation efficiency at the right scope

Include authoring, source-blind checks, grounding, annotations, teacher answers, retries, judge calls, embeddings, and sandbox execution. Report API tokens and GPU-hours separately, then document any cost conversion. Use wall time, peak resource requirements, and supported examples per unit cost with human-review uncertainty. Separate generation costs from student-training costs.

The existing historical medium-run chart is useful engineering evidence, but it predates the latest grounding/judge work. It cannot supply the cost or quality of a new release recipe. Do not schedule a new large run solely to improve the article's size headline.

**Decision after review:** select dataset revisions, primary capability claims, student model, training-token budget, evaluation suite, and compute allocation. Those choices remain open; the current task only prepares the article and proposed studies.
