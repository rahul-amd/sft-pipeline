# Content audit of the medium run

The pipeline completed successfully, but the output should **not** be described as verified high-quality data. A Qwen3-32B reviewer inspected 160 accepted and 40 rejected examples from `freeze_scale_v2`, followed by direct assistant inspection of all 23 accepted examples that the reviewer flagged. Neither the reviewer nor the assistant inspection is human ground truth.

Sampling was deterministic (seed 42) and balanced across domain strata, not weighted to the output distribution. Among the accepted sample, 15 prompts came from Stage 2; 145 had source references. The reviewer shares the teacher's Qwen family. It saw the complete candidate prompt, reasoning and answer, plus a reference or source passage where available; the pipeline's accept/reject decision was hidden.

## Raw reviewer results

| Pipeline decision | Reviewed | Reviewer: correct | Incorrect | Uncertain |
| --- | ---: | ---: | ---: | ---: |
| Accepted | 160 | 137 | 21 | 2 |
| Rejected | 40 | 19 | 18 | 3 |

These are **uncalibrated model verdicts, not accuracy or false-rejection estimates**. Direct inspection found both real defects and incorrect reviewer rationales. Full prompts, references, responses and verdicts remain in `smoke/runs/freeze_validation/audit-freeze_scale_v2/`; the dataset was not edited after the run.

## Confirmed examples that matter before scaling

The IDs below identify retained examples. Findings rely on the supplied prompts and source material, rather than assuming every source reference is correct.

| ID suffix/prefix after `sha256:` | Finding |
| --- | --- |
| `629467a59da8b275afa12c676b4206e5` | Stage 2 asks for the original name of “the parish's governing body” without identifying the parish. The answer assumes a UK context; the source discusses a specific Louisiana parish. The generated question is not self-contained. |
| `fadb9794613b85f4fa6d064aa7529943` | Stage 2 asks for the men's/women's net-height difference without identifying the sport. The teacher answers about tennis; the source describes volleyball and supplies values differing by 0.19 m. Both generation and source grounding need attention. The reviewer's claim that the teacher's numbers are correct for tennis was not accepted as evidence. |
| `92fa9107595c40bf8de58c7aab70087e` | A self-contained generated question asks the distance to NGC 7314. The answer gives 2.5 megalight-years; the source passage gives 54.6. The current teacher/judge combination allowed an answer inconsistent with its source. |
| `36e01d8c98218831944f2c6d3ed9df81` | A generated tardigrade question asks for a survival percentage under combined UV/vacuum exposure. The source's percentage concerns subjects protected from UV, so it does not establish the requested value. The teacher supplies 50% and a mission date inconsistent with the passage. The reviewer also conflates the exposure groups; its proposed correction is unreliable. |
| `6889bf0f3a63391a9038ed8eb4283bd7` | The prompt says the grandmother has an older sister born in 1936, with an eight-year gap. The answer subtracts eight and gives 1928; the stated relationship requires 1944. The reviewer's explanation incorrectly defends the subtraction despite flagging the answer. |
| `1ac39052da5394fc9b5243fb2aa06bc9` | A northward river current and northeast motion require an eastward wind-force component. The prompt asks for force direction; the answer instead selects the direction the wind comes from (west). The reviewer also gives contradictory vector reasoning. |
| `b946efbbb64c0b118f406a14ae764f08` | A passage distinguishes Elizabeth's half-sister Mary from Mary, Queen of Scots. The answer assigns the former's death date to the latter. The supplied reference preserves the distinction; the reviewer's allegation that the reference conflates them is incorrect. |

Other flags include unsupported extra details, conflicts with source references, and questions needing domain-specific review. They were not converted into a numerical error rate. For example, the Abbey Road answer adds precise dates absent from the supplied passage even though its final year-level answer matches the reference. The Azure-storage, audition-practice and practical-instruction examples need external/domain review before assigning gold labels.

## Reviewer and reference failures

Several flagged code answers demonstrate why changing the judge model alone is insufficient:

- `25af4624f592748e5e783d3e9ae61463`: the candidate uses `set(random.sample(...))`, satisfying uniqueness. The reviewer criticizes the separate reference's list-with-replacement implementation and labels the candidate incorrect.
- `9cd47aa5a1bcb34c9ac84e428d58e0d6`: the reviewer invents a single-digit-day requirement absent from the prompt. The candidate supplies a valid single-line `datetime` formatting expression.
- `1c5bf219c3280d62ab28ad8c83c1673c`: the candidate explicitly targets Python 3.7+ insertion-order dictionaries; the reviewer faults it for environments outside that stated scope. The candidate's illustrated formatting and peripheral prose still deserve review, but the claimed primary ordering defect is not established.
- `e6af86fd8ba0a96bd0738dde6090fbc6`: the prompt requests binary sentiment classification; the reviewer faults the candidate for not using the reference's third, neutral category. This does not establish a violation of the requested output contract.

Some references/questions are themselves ambiguous: “10 times more,” unspecified stamp shapes, sand sold in area units, and subjective social questions. A reference disagreement is not automatically a candidate error. The remaining flagged accepted examples were inspected with these limits in mind; no claim is made that unflagged examples are error-free.

Replaying the existing non-judge filters on the 40 rejected examples found that, among the 19 the reviewer called correct, four were token-limited, one triggered repetition detection, and 14 passed that replay. The latter suggests judge disagreement, but original per-record judge scores were not retained and code execution can be nondeterministic. This is diagnostic evidence, not a measured false-rejection rate. Replay output is `non-judge-replay.json` in the audit directory.

## Recommendation before a production-sized run

1. Make Stage 2 source grounding an explicit contract: retain the source chunk, require the question to identify relevant entities/sport/place/time, and validate answerability against that chunk. Decide whether the final training prompt includes the passage or whether a separate source-aware verifier checks a standalone answer.
2. Calibrate the judge on adjudicated examples, clearly separating candidate defects from reference defects and prompt ambiguity. Retain per-record scores/reasons for diagnosis. Raising the score threshold without calibration is not supported by this audit.
3. Repeat a bounded quality evaluation with the intended production teacher after those changes. A million-record run would amplify the observed defects without resolving them.

The tested HTTP pipeline can be tagged as a functional release candidate. Hold the claim of verified high-quality output and a large production run until the source-grounding and evaluation gaps are addressed. No math filter was added, and semantic deduplication remains false by default.
