# Joint pipeline release blog

Start with [DRAFT.md](DRAFT.md), the assembled announcement sketch for **sft-pipeline**, **lc-pipeline**, and their generated datasets. It includes the implementation details, figures, recorded engineering evidence, and explicit placeholders for release metadata and unrun quality comparisons.

The draft is for editorial review. Its announcement language is prospective; this work has not published code or data. Dataset names, download links, actual release sizes, source permissions, and downstream results remain to be supplied.

| Resource | Purpose |
|---|---|
| [DRAFT.md](DRAFT.md) | Main article, assembled and reviewed by the orchestrator |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Prioritized proposed studies, controls, and result tables |
| [FIGURES.md](FIGURES.md) | Five figure captions, provenance, and reproduction commands |
| [release-manifest.template.yaml](release-manifest.template.yaml) | Metadata to collect for the actual releases; not a pipeline configuration |
| [REVIEW.md](REVIEW.md) | Editorial review, unresolved details, and freeze verification |
| [references/CLAIMS.md](references/CLAIMS.md) | What the current evidence does and does not support |
| [references/external-sources.md](references/external-sources.md) | Primary sources for proposed public comparisons |
| [references/validation-evidence.json](references/validation-evidence.json) | Compact snapshot of historical validation measurements and source hashes |
| [sections/](sections/) | Worker drafts and implementation fact-check notes |
| [figures/](figures/) | SVG figures, previews, editable sources, and measured chart CSV |

One orchestrator reviewed contributions from three workers: SFT implementation, LC implementation, and evaluation/visuals. The main article is the editorial version; section drafts preserve the contributions and more detailed source notes.

Both codebases were treated as frozen, including pre-existing uncommitted changes. No pipeline execution, training, benchmark evaluation, or Slurm jobs were started for this writing task. All new resources are contained in this directory.
