# SFT pipeline blog experiment proposals

This supports the [blog plan](PLAN.md) and is pending approval.

The following are proposed starting sizes and controls, not committed resource estimates. Freeze the code, source snapshots, models, prompts and evaluation protocol before collecting publishable results. Use fresh run IDs after configuration changes.

## 1. Repeat the medium run on the final code

**Question:** Does the current pipeline run unattended beyond a tiny smoke test, and where does it spend resources?

Repeat approximately the earlier 10,000-prompt workload, including hundreds of custom source documents, on the final revision. Capture counts, per-stage latency, API errors/retries, token usage, truncations, peak memory, GPU-hours and sandbox cleanup. Repeat no-work resume and selected interruptions at durable-write boundaries. Record duplicate/lost outputs and requests repeated during recovery.

Produce the retention chart, a stage-cost table and a recovery table. Report startup separately from steady-state inference, and separate serving resource use from the pipeline driver. Translate GPU-hours to money only under a stated pricing assumption. This is the minimum new systems experiment for the release post.

## 2. Measure grounding on independently reviewed sources

**Question:** Does grounding remove underspecified and unsupported questions without discarding too many useful ones?

Start with roughly 100 held-out source chunks across several document types and domains. Compare the original and revised generation prompts on the same chunks. Separately compare no grounding, source-visible validation alone, a batched source-blind check plus source validation, and the current individual source-blind check plus source validation on the **same generated candidates**. This separates generator improvement from checker improvement.

Have two reviewers independently label a manageable held-out sample of roughly 300–500 questions, with disagreement adjudication. For subject identification, reviewers see only the question; for answerability and evidence support, they then see the source. Include good questions, missing subjects, missing numerical information and valid conditional answers. Judge proposed reference correctness separately from question answerability.

Report valid-question yield per source chunk, retained-question precision, false rejections, unsupported references and tokens spent per retained valid question. Group uncertainty estimates by source chunk because questions from the same passage are correlated. Split at document level so neighboring chunks cannot leak between development and evaluation.

## 3. Calibrate the answer judge against human labels

**Question:** Does filtering improve retained answer quality, and what useful answers does it throw away?

Build a roughly 400-example set from a fresh teacher-output pool, sampled across domains and collected/generated origins **before** applying the judge. Divide it into a calibration portion and a locked evaluation portion, keeping related prompts together. Review full questions, reasoning and answers, with sources/references shown separately. Reviewers should not see model identities, pipeline decisions or scores.

Label answer correctness, prompt ambiguity, instruction following and explanation defects separately; retain an uncertain category. Use two reviewers and adjudication. Task-specific code tests can support labels inside the sandbox; do not mistake a successful snippet for a correctness label. Manual mathematical checks here are evaluation work, not a proposal to restore the removed math filter.

Compare the old and current judge, with a stronger model as an optional arm. Select thresholds only on the calibration portion. Report accepted-set precision, fraction of good examples rejected, coverage, uncertainty and domain breakdowns on the locked set. Keep deliberately adversarial challenge cases separate from representative estimates. Any stratified population estimate needs its sampling weights.

## 4. Test whether the resulting data improves a student

**Question:** Does the pipeline produce more useful SFT data than a simpler construction method at the same training budget?

This is the strongest experiment for the claim that the pipeline makes a better training dataset. Use one affordable student checkpoint, one teacher/configuration and four dataset arms:

| Arm | Prompt selection and filtering | Comparison it enables |
| --- | --- | --- |
| A | Random collected prompts; minimum validity checks | Simple baseline |
| B | Cluster/quota sampling of collected prompts; the same minimum checks | Selection effect relative to A |
| C | The same selection policy, collected prompts only; full filtering | Filtering effect relative to B |
| D | The same policy, collected plus grounded generated prompts; full filtering | Custom-source contribution relative to C |

All arms retain baseline sanitation, exact deduplication, configured benchmark decontamination, truncation rejection and safe handling of code. Keep optional semantic deduplication fixed for these arms; study it separately rather than changing another factor at the same time.

Cache identical teacher responses for prompts shared across arms. Match total **training tokens**, student initialization, optimizer, training schedule and evaluation settings; report example counts and generation costs separately. Use an ample candidate pool and a fixed refill rule. If an arm cannot meet the token budget, lower the common budget or collect more candidates rather than silently repeating examples. Audit the training serialization of reasoning and final answers before interpreting learning results.

Select and freeze held-out evaluations appropriate to the actual domains and student, with exact settings disclosed. Include task-based math/code scoring and independent instruction-following evaluation. Apply contamination controls to every arm; n-gram matching reduces observed overlap but cannot prove absence of paraphrases or teacher pretraining exposure.

Start with a feasibility pilot on A and D. If viable, run all four arms with three paired training seeds, plus the unfine-tuned student's evaluation: **12 trained runs**. Report task-level results, seed variation and paired evaluation uncertainty, including regressions. No gain is still a useful result; it should change the post's claims rather than trigger selective reporting.

## 5. Quantify the sampling and deduplication tradeoff

**Question:** What coverage and redundancy do we buy for a fixed number of teacher requests?

Reuse one frozen annotated pool. Compare random selection, cluster/quota sampling, and cluster/quota sampling with semantic deduplication. Match feasible sample sizes, or explicitly report quota shortfalls. Measure source/domain/difficulty distributions, cluster coverage, nearest-neighbor similarity, manually reviewed redundant pairs and selection time/memory.

Sweep a small, predeclared range around the existing cosine threshold. Plot redundancy and quota fill against cost. Validate some annotation labels independently: a visually balanced table of noisy difficulty labels is not proof of balanced task difficulty. Treat embedding coverage as a proxy until the student experiment establishes usefulness.
