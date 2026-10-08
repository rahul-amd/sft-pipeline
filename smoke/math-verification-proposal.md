# Proposal: evidence-based math verification

Start with reference-backed final-answer checks. Keep answer correctness, reasoning quality, and unverifiable outputs separate. Do not turn the current number-overlap heuristic into a rejection rule: a number appearing in the reasoning says little about whether the answer is correct.

This proposal is deferred. Math filtering and the previous implementation have been removed; none of the checks below currently run.

## 1. Preserve reference answers through the pipeline

Stage 1 currently extracts only prompt text. Add an optional source adapter with `reference_answer_field`, answer type, units/tolerance, and source revision. For GSM8K, retain the final value after `####` in the source answer. Other datasets need their own adapters; a worked solution is not necessarily a machine-comparable final answer.

Carry an internal `reference` object through Stages 1–6, associated with the exact prompt ID. Never put it into the teacher's messages or embedding text. Exact duplicate prompts with conflicting references should be flagged for review. A semantically similar prompt must not inherit another prompt's reference: a changed sign, number, unit, or constraint can change its answer.

## 2. Compare final answers using an existing verifier

Use [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify) for extraction and numeric/symbolic comparison. It exposes separate gold/prediction parsing and comparison functions and handles expressions, sets, intervals, and other mathematical forms. The argument order matters: gold is the first argument. Its documented unit handling can strip units, so dimensional/unit validation needs a separate explicit check rather than assuming symbolic equality covers it.

Restrict extraction to a declared final answer or boxed result. Do not search the entire reasoning for *any* expression matching the reference. Ambiguous multiple final answers should produce an unknown result. Compare exact integers/rationals exactly; use source-defined tolerances for approximate numeric tasks, with explicit percent and unit conventions. Equation equivalence also needs variable/domain assumptions: equality after simplification is not proof of equality everywhere.

Run parsers and symbolic checks inside the sandbox with CPU/memory/wall-time limits. They process untrusted strings and can take excessive time. Math-Verify's [timeout implementation](https://github.com/huggingface/Math-Verify/blob/main/src/math_verify/parser.py) also constrains threaded use; a separate sandbox process avoids mixing its signal handling into API worker threads. Use a small prebuilt verifier image with pinned versions, not a package installation for each record.

## 3. Record three outcomes, with reasons

Persist `math_verification` alongside the record:

```json
{
  "status": "verified",
  "method": "reference_equivalence",
  "scope": "final_answer",
  "reference_source": "source-id@revision",
  "verifier_version": "pinned-version",
  "reason": "exact_rational_match"
}
```

- **Verified:** supported final answer agrees with a trusted reference or an independently computed exact result.
- **Incorrect:** supported, unambiguous final answer disagrees with that reference.
- **Unknown:** no trusted reference, unsupported answer type, ambiguous extraction, timeout, or incompatible assumptions.

Reject incorrect answers. Keep unknown answers in a separately labeled pool by default; a strict verified-math export excludes them. Infrastructure outages fail the stage, while per-expression timeouts are recorded as unknown. Preserve verification metadata in the export or a linked sidecar, and report both coverage and rejection rates. An unknown result must never silently become verified.

## 4. Handle generated prompts without gold answers honestly

For Stage 2 prompts, a second solution from the same teacher is not independent ground truth. The most reliable first extension is a constrained generator for arithmetic, algebra, probability, and other tasks where trusted code constructs a problem and its answer together. Alternatively, use a trusted executable oracle for a well-defined problem family.

For unrestricted corpus-derived questions, ask an independent solver to solve from the prompt before seeing the teacher response, then use a separate critic to inspect the reasoning. Disagreement triggers review or exclusion. Agreement is useful selection evidence, but label it `model_consensus`, not verified: models can share the same error. No-reference proofs and explanatory science questions remain unknown to an answer-equivalence checker.

## 5. Do not equate a correct final answer with correct reasoning

An answer-equivalence checker can accept a correct answer with an invalid derivation. Keep a separate reasoning-quality assessment, with a critic focused on invalid steps, hidden assumptions, and unsupported claims. Formal proof checking is suitable only when there is a formal statement and a proof accepted by a proof assistant; making that work for arbitrary prose proofs is a separate project.

Before enabling rejection, build a stratified human-reviewed set covering correct answers, subtly wrong answers, equivalent expressions, units, rounding, ambiguous outputs, and correct answers with wrong reasoning. Measure false acceptance, false rejection, and unknown rates by task type. Keep calibration data separate from evaluation benchmarks used for decontamination.

## Recommended first implementation

Add reference retention for GSM8K and one symbolic-answer dataset, sandboxed Math-Verify comparison, the three-outcome record schema, coverage reporting, and strict/unknown export policies. Calibrate these on held-out labels before expanding to generated tasks. Keep independent solving and constrained Stage 2 generation as follow-up work; they have different costs and evidence guarantees.
