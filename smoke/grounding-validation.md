# Stage 2 grounding and judge calibration

This follow-up is on `stage2-grounding-judge-calibration`, based on commit
`854656d` from [PR #12](https://github.com/rahul-amd/sft-pipeline/pull/12).
The live evaluation is complete. Execution checks passed; content probes exposed
remaining model errors. The code is a functional release candidate, not a
correctness certification for generated datasets.

## Implementation

Stage 2 now requests standalone questions with explicit subjects, conditions and
units. It permits fewer questions or an explicit `[]` for unsuitable chunks.
Empty responses and malformed verdicts are operational errors.

`grounding_enabled: true` adds one source-blind context request per generated
question and one source-answerability request per nonempty batch. Each question
is checked individually without its passage or neighboring questions. The source
validator proposes a reference answer and provides an exact source quote. A
question is retained only if both checks pass and the nonempty quote occurs in
the source. The substring check establishes quote provenance, not entailment or
answer correctness; the semantic assessments remain model based.

`grounding_model` defaults to the generator model and shares its endpoint/key.
`grounding_max_tokens` defaults to 2048; increase it for larger batches. Accepted
records retain `source_context`, `reference_answer` and `grounding`. The teacher
receives the standalone question; the judge receives source and reference in
separate fields. Export keeps the existing training schema. Set
`grounding_enabled: false` to bypass these checks; the metadata explicitly records
`validation_disabled`.

Every successfully evaluated chunk has an audit at
`stage2/_state/grounding/<chunk-hash>.json`, including rejected questions. Fully
rejected chunks are checkpointed; a stage producing no prompts still fails
visibly. Malformed/incomplete verdicts cause retry and failure, not silent loss.

The judge receives complete prompts, reasoning and answers, replacing the old
500/1500/500-character cutoffs. The `grounded-v2` rubric separates candidate and
reference, allows valid alternatives, and asks for evidence and candidate checks
**before** verdicts and scores. It rejects ambiguous prompts, incorrect/uncertain
verdicts and scores below the threshold. Numeric-only responses from custom
judges must adopt the new schema. `max_input_chars` defaults to 100000; exceeding
it fails rather than clipping. This character cap is not a server context-token
guarantee.

Stage 6 writes one decision per input to `stage6/_decisions/<shard>.jsonl`,
including rejections and judge scores/reasons. `disabled`, `not_sampled`,
`not_reached`, `error` and `judged` are distinct. The default error policy raises;
explicit `on_error: pass` records the error. Decision/survivor files publish
before the shard completion ledger; interrupted shards are rerun. Surviving
internal records retain judge metadata; training exports keep their existing
schema. Semantic deduplication remains false by default; no math filter was added.

## Execution results

Slurm job `54564` ran on `tus1-p15-g43`, using the user's **one-time** `hissu`
exception. Four GPUs were allocated: Qwen3-4B teacher, Qwen3-8B helper, Qwen3-32B
reviewer, and pipeline/tests. The allocation and servers were shut down afterward.
Future runs must continue to avoid `hissu` unless explicitly authorized again.

- **288 tests passed, zero skipped**, including GPU clustering, real Docker
  isolation and Ray workers; 48.55 seconds. The final review-harness correction
  passed an additional focused run of three tests (two overlap the full suite).
- Final unattended run: `grounding_e2e_final_20261008`, **223 seconds**.
- Collected 17 prompts; Stage 2 generated six candidates and retained five.
  Decontamination removed the canary, leaving 21; sampling selected 20.
- **20 HTTP inferences → 18 exports**, including **four Stage 2 examples**.
  Two responses hit the token limit and were rejected before judging. No teacher
  HTTP failures occurred. The small pool intentionally underfilled target 64.
- GPU embedding/clustering, HTTP generation/annotation/judging, Docker code
  execution and export ran through the ordinary CLI. Resume skipped all eight
  completed stages and preserved data and audit hashes.
- No sandbox containers remained after testing.

Logs and JUnit results: `smoke/runs/grounding_live_20261008/`. Final pipeline
artifacts: `smoke/runs/grounding_e2e_final_20261008/`. Summary:
[`results/grounding-validation.json`](results/grounding-validation.json).
The earlier CPU-only results remain under `smoke/runs/grounding_validation/`.

## Grounding results

The final helper passed **7/7 regression questions**. Of 12 newly generated
questions in that probe, eight passed; the unnamed-parish questions and a
percentage lacking its denominator were rejected.

Fresh holdout checks exposed recall limitations:

| Context/grounding model | Valid questions accepted | Invalid questions rejected |
| --- | ---: | ---: |
| Qwen3-8B | 3/4 | 4/4 |
| Qwen3-32B | 2/4 | 4/4 |

The 8B helper rejected a question explicitly identifying District Q and the Eira
survey. The 32B model rejected valid questions identifying a person/city and a
reactor. Both holdout probes therefore **failed their all-cases-correct criterion**;
their nonzero exits and detailed reports are retained. This is not a passing
quality result, and a larger model did not solve it automatically.

Initial experiments demonstrated why the source-blind check is necessary: a
source-visible validator accepted an unnamed parish. Batching the context check
also caused cross-question errors; separate requests fixed the original cases.
One initial price-question label was corrected because giving both conditional
prices is valid when no single price is demanded. An early proposed reference
miscomputed a net-height difference. All intermediate outputs are retained;
reference answers remain explicitly fallible.

## Judge calibration and separate review

The final v2 case set contains 32 calibration examples and 16 fresh held-out
examples, with related variants in the same split. Labels are assistant-curated,
not human gold or representative production labels. The original case set
informed v2 and was not reused as the final held-out result.

At the unchanged threshold **6.0**:

| Model | Old judge correct decisions | Current judge correct decisions | Current false accepts / false rejects |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 13/16 | 15/16 | 1 / 0 |
| Qwen3-32B | 12/16 | 15/16 | 1 / 0 |

Both missed a prompt with unresolved pronouns when its source was available.
The 8B calibration split selected threshold 6. The 32B split selected 9, which
reduced its held-out result to 13/16. No global threshold was changed from this
small comparison. The age-order case also remained a calibration error, with
different mistakes between the models.

The separate Qwen3-32B reviewer inspected all 20 final-run responses without the
pipeline scores or disposition. It accepted all 18 exports. It agreed with one
rejection and disagreed with the other; both were token-limited with **empty final
answers**, so the structural rejections remain correct. The reviewer shares the
Qwen family and rubric, and its disagreement illustrates that it is not ground
truth. An initial review-harness bug substituted reasoning for an explicitly
empty answer; this was fixed, regression-tested, and the review rerun into
`review-final-e2e-v2/`.

Direct assistant inspection covered every exported final answer and the rejected
responses' metadata. It found explanatory overstatements despite model acceptance;
see [quality findings](grounding-quality-findings.md). Neither this inspection nor
model agreement establishes a production correctness rate. Before making that
claim, collect human-reviewed labels using the intended production teacher and
data distribution.

## Reproduction

Use a fresh run ID after prompt/config changes. On an allocated compute node,
start an OpenAI-compatible thinking teacher and a non-thinking helper with
adequate context (32768 tokens for these helper inputs). Set `SFT_API_BASE`,
`SFT_MODEL`, `SFT_JUDGE_API_BASE`, `SFT_JUDGE_MODEL`, `SFT_API_KEY` and `SFT_RUN_ID`.
Run from the repository root:

```bash
python smoke/check_grounding.py \
  --api-base "$SFT_JUDGE_API_BASE" --model "$SFT_JUDGE_MODEL" \
  --output "smoke/runs/$SFT_RUN_ID/grounding-probe"
python smoke/check_grounding.py \
  --api-base "$SFT_JUDGE_API_BASE" --model "$SFT_JUDGE_MODEL" \
  --cases smoke/fixtures/grounding-holdout.json \
  --output "smoke/runs/$SFT_RUN_ID/grounding-holdout"
python smoke/calibrate_judge.py \
  --api-base "$SFT_JUDGE_API_BASE" --model "$SFT_JUDGE_MODEL" \
  --cases smoke/fixtures/judge-calibration-v2.jsonl \
  --output "smoke/runs/$SFT_RUN_ID/judge-calibration"
python smoke/run.py --config smoke/config-grounding.yaml
```

The holdout probe exits nonzero on any mismatch; inspect its report and do not
silently relabel failures. The calibration harness saves prompts, cases, scores,
threshold sweeps and manifests. Selection uses only calibration labels and never
changes pipeline configuration. Resuming assumes fixed model weights/settings
behind the endpoint. Use `smoke/review_grounding_run.py` with a separate endpoint
and `--run`, `--output`, `--api-base`, `--model` to review all responses in a small
run. Every test and result stays under `smoke/`.
