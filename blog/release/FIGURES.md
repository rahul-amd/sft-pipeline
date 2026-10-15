# Figure inventory and captions

All visuals are original assets for this sketch. SVGs include editable text; diagrams have a standard-library Python source. The plotted retention chart has Matplotlib source and a CSV of measured inputs. No figure asserts downstream superiority.

| Asset | Status | Suggested placement |
|---|---|---|
| [pipeline-overview.svg](figures/pipeline-overview.svg) / [PNG preview](figures/pipeline-overview.png) | Implementation diagram | Opening architecture section |
| [context-budget.svg](figures/context-budget.svg) / [PNG preview](figures/context-budget.png) | Illustrative, not to scale | LC token-budget explanation |
| [sft-medium-retention.svg](figures/sft-medium-retention.svg) / [PNG](figures/sft-medium-retention.png) | Measured historical run | Execution evidence, explicitly separated from release inventory |
| [evaluation-design.svg](figures/evaluation-design.svg) / [PNG preview](figures/evaluation-design.png) | Proposed evidence framework | Evaluation section or appendix |
| [comparison-placeholder.svg](figures/comparison-placeholder.svg) / [PNG preview](figures/comparison-placeholder.png) | Empty planned result panel | Evaluation section; replace after experiments |

**Figure 1. Two independent recipes for training data.** SFT collection and source-grounded question generation converge before decontamination, selection, teacher inference and filtering. LC generators construct deterministic algorithmic/corpus tasks or assemble model-authored QA and summaries. The LC generators support PT text and SFT messages; the SFT repository exports prompt/reasoning/answer records. These outputs do not imply an implemented joint training mixture. Grounding is model based; the SFT judge is sampled according to configuration. Clustering is an input option for Untie, not a prerequisite for every LC task. [Editable Mermaid topology](figures/pipeline-overview.mmd).

**Figure 2. The answer consumes context-window space too.** Reconstruction allocates substantial space to both shuffled input and recovered text; short-answer tasks leave more space for the input. Segment widths are conceptual, not measured proportions. The enforced cap concerns complete serialization. Tests with 128K synthetic outputs do not establish 128K model inference or comprehension.

**Figure 3. Retention in one earlier unattended SFT validation.** The 2026-10-08 medium run combined 9,924 collected and 1,002 generated prompts, selected and inferred 3,805, and exported 3,131 in 51 minutes 14 seconds. These are the counts of one historical validation before the latest grounding/judge changes, not counts of the data being released. The selection target was 4,000; scarce hard examples explain the shortfall. Selection is a budget decision, while filtering accounts for later losses. Counts do not establish content correctness. Source: [the SFT report](../../smoke/freeze-validation.md); [plot data](figures/sft-medium-retention.csv); [plot code](figures/plot_retention.py).

**Figure 4. Separate evidence for separate claims.** Integration tests establish behavior of tested code/configurations. Representative human review estimates supervision quality. Controlled training comparisons measure student utility. The latter two remain proposed studies; passing software checks cannot substitute for them.

**Figure 5. Planned downstream comparison.** Empty panels deliberately contain no points, bars, scores or rankings. Populate only from approved, frozen experiments; include regressions and uncertainty.

## Specification for the planned comparison plot

- Panel A: separate task-level panels for general instruction following and verified task performance. Horizontal axis: selected dataset arms; vertical axis: each task's native score. Plot every paired seed and a declared summary/uncertainty interval. Keep reasoning-only datasets in an appropriate domain-matched panel.
- Panel B: LC task-level scores against measured input-token length; one series per training arm with paired-seed variation. Report out-of-window examples explicitly rather than silently dropping them. Separate related-family RULER diagnostics from independent HELMET/LongBench v2 transfer.
- Accompanying table: base/tokenizer revision, serialized training tokens, loss-bearing tokens, length curriculum, train GPU-hours, inference settings, evaluation revision, seeds and sample counts.
- Required result columns: `experiment_id`, `data_revision`, `arm`, `train_seed`, `eval_task`, `eval_revision`, `context_tokens`, `n_examples`, `score`, `metric`, `train_tokens`, `target_tokens`, `gpu_hours`. Preserve per-example paired scores for uncertainty; one aggregate score alone is insufficient.
- No metric values exist for this planned plot. Do not copy published pretrained/instruct model scores into a matched-training comparison.

## Rebuild

From the repository root, with Python and Matplotlib installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python blog/release/figures/build_figures.py
PYTHONDONTWRITEBYTECODE=1 python blog/release/figures/plot_retention.py
PYTHONDONTWRITEBYTECODE=1 python blog/release/figures/render_previews.py
```

The plotting script keeps its Matplotlib cache under this figure folder. Diagram styling, wording and topology are editable in `build_figures.py`; Mermaid is an alternate editable topology source, not a renderer dependency. SVG titles/descriptions provide text alternatives for diagrams. The measured chart's caption and CSV provide its accessible numeric alternative.

PNG previews of the four diagrams are generated by `render_previews.py` using existing Pillow and Matplotlib font discovery. This small SVG-subset renderer preserves the diagram content; preview dashed frames use square corners. SVGs remain the publication masters.
