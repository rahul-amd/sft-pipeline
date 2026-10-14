# Orchestrator review

Reviewed 14 October 2026. This package is an editorial sketch for approval, not a published announcement or a new experimental result.

## Work allocation and integration

- SFT worker: stage explanations, record walkthrough, defaults, and implementation/evidence anchors.
- LC worker: task families, clustering, inference boundary, serialization contracts, and validation limits.
- Evaluation/visuals worker: primary-source baseline research, proposed evaluation, diagrams, and measured retention plot.
- Orchestrator: opening/release narrative, assembled article, shared evidence section, prioritized experiment plan, release manifest, claim ledger, cross-section review, and final resource checks.

Both implementation workers reviewed the assembled article and experiment plan. All reported precision issues were incorporated. The orchestrator inspected the five figure previews, checked the source-backed retention counts, and reviewed the separation of measured evidence from proposed experiments.

## Corrections retained in the final draft

| Potential misleading impression | Final treatment |
|---|---|
| Every SFT answer receives a judge check | Judge defaults to a 10% sample; latest small integration used 100% |
| Semantic dedup always runs | Optional and off by default; measured cosine constraint is distinct from semantic or learning quality |
| Every final SFT example carries full grounding evidence | Compact export fields are listed; richer evidence stays in intermediates |
| Both repositories emit one common training schema | SFT exports prompt/reasoning/answer; LC emits task-specific PT text or SFT messages |
| Grounding proves correctness | Quote substring checks establish provenance; model verdicts can err |
| Source-blind review requires knowing an answer | Assess clarity and subject identification first; assess support after source reveal; LC QA uses its document |
| IDK absent-item cases guarantee no textual mention | They guarantee no generated binding for the queried item |
| 128K tests demonstrate 128K LLM competence | Coverage was synthetic generation/serialization; real document authoring inputs were roughly 3K–14K tokens |
| Medium SFT counts measure the current recipe or final data release | Historical run is explicitly labeled as preceding current grounding/judge changes |
| Test passes or judge agreement establish dataset superiority | Representative human review and matched student comparisons remain proposed |
| Equal examples/tokens automatically mean equal training compute | Serialized and loss-bearing tokens, LC length curriculum, packing, and measured compute are separate controls |
| MMLU/HumanEval can automatically be held-out tests | Current example sources include their test splits; actual release construction and evaluation overlap must be resolved |

## Open editorial inputs

1. Supply dataset names, immutable hosting revisions, actual release inventories, and selected LC task families/formats. Historical output directories were not treated as finalized release datasets.
2. Identify the exact code/config/model revisions that generated each released dataset. The SFT working tree contains uncommitted grounding/judge improvements; its observed HEAD does not include all described behavior. Do not link the current HEAD as if it identifies those improvements.
3. Fill the release tags, minimal recipe links, environment instructions, source/split manifests, component terms, and redistributable evidence/audit links.
4. Choose the primary learning claims and approve an experiment scope separately. No human-review accuracy estimates, downstream scores, or release generation-cost figures exist in this package.
5. After results exist, replace the empty comparison panels and consider shortening the implementation sections for the publication venue. The current ~5,500-word draft deliberately preserves detail for review.

These are placeholders requested for the sketch, not changes made to frozen code. No publication, commit, push, pipeline test, data-generation run, training job, or Slurm job was performed for this task.

## Verification

All five SVGs were parsed, PNG previews inspected, and relative Markdown links checked. Chart inputs are included as CSV; diagram and plotting sources are included. Derived previews and caches stay within this folder; the generated Matplotlib cache is excluded from version control.

The initial [tracked-file hashes](references/frozen-code.json) were compared again after writing: **168 SFT files and 71 LC files outside `blog/` matched**, and both repository HEADs were unchanged. This check includes the existing tracked working-tree modifications. [Machine-readable verification](references/review-verification.json) records the final check.
