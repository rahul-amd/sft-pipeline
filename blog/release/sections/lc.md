<!-- Integration draft. Release links, dataset inventory and evaluation results belong in the parent article. Examples below are illustrative, not released records. -->

## Long-context data: specify what the model must do with the context

A long training sequence is useful only if its task requires the model to use that sequence. `lc-pipeline` builds a mix of tasks with different demands: retrieve a precise passage, aggregate scattered records, follow a graph, recover paragraph order, abstain when a fact is missing, answer questions about a document, and summarize it. These tasks give us explicit controls over the operation, context length, source material, and answer format.

The implementation has two main branches. Deterministic generators construct problems and compute their answers directly. Corpus-based generators transform real documents, either with a known target or with a model-generated answer. Clustering is an optional preparation step for assembling longer sources; it is not a requirement for every task.

**[Figure suggestion: place the LC workflow diagram here. Show synthetic generators alongside corpus filtering, optional clustering, deterministic transformations, and the prepare → inference → assembly branch. All paths converge on token-checked PT/SFT JSONL.]**

## One training interface, with explicit length accounting

Each family supports two trainable representations. PT mode writes a flat `text` field containing the task and answer. SFT mode writes a `messages` array with user and assistant turns. The surrounding metadata varies by task: synthetic examples retain structured golds; document QA retains its question–answer pairs and source identity. This is a shared training interface, rather than an identical metadata schema for every generator. Training loss masking remains a trainer-side choice.

The budget applies to the **complete serialized training example**, including the answer. That distinction matters when the answer can be a long paragraph list or an entire reconstructed document. With a Hugging Face tokenizer that provides a chat template, SFT length includes the template's role markers and other tokens. PT length counts the actual text. When no chat template is available, SFT counting uses newline-joined message content; `length_mode` records that weaker contract. The explicit `whitespace` tokenizer counts word/punctuation chunks and is an approximation, not a model-token guarantee. A requested tokenizer that fails to load causes an error rather than a silent substitution.

For synthetic tasks, the driver probes a task-specific size parameter, searches for a suitable size, then checks each completed example and shrinks it if necessary. Impossible minimum-size examples fail. Outputs record `training_length`, and content/settings-based identifiers distinguish examples across seeds, formats, and budgets. Final JSONL publication uses temporary files followed by replacement, so interrupted generation does not publish a partially written final file.

## Real documents, traceable selection, and optional clustering

The shared corpus reader streams Parquet batches and recognizes common text and identifier column names independently for each shard. Where metadata exists, it can filter short documents, low mean education scores, and records marked as near-duplicates. These are source-selection rules, not fresh quality judgments: missing education fields are not scored, and the near-duplicate rule drops all records whose supplied cluster size exceeds one. It does not select a representative from each duplicate group.

Source tags retain the relative shard path and absolute row offset. Those tags also serve as fallback document IDs when an explicit ID is absent. Cluster joins fail if documents are missing or their IDs are duplicated, which prevents a seemingly successful example from silently losing part of its source.

For sources too short to fill the intended window, clustering provides a way to assemble related documents. The default embedding model is `Qwen/Qwen3-Embedding-0.6B`; it embeds a configurable prefix, **256 tokens by default**. Normalized embeddings feed an exact top-*k* neighbor search implemented with batched GPU matrix multiplication. A greedy single-linkage procedure grows clusters through sufficiently similar neighbors while respecting a token budget. Embeddings and document metadata can be cached for reclustering.

This trades semantic coverage for practical embedding cost. Prefix similarity can miss later topic changes. Single linkage allows a document to join through one member, so membership does not imply that every document pair is similar. Cluster budgets use source token-count metadata; downstream generators enforce their own selected tokenizer's final limit. Exact neighbor search is quadratic, which also bounds how far this implementation should be scaled unchanged.

## Deterministic tasks: diverse operations with inspectable golds

The synthetic registry contains **22 task types**, organized around operations rather than one generic retrieval template:

| Operation group | Task types | What the target requires |
|---|---|---|
| Words, counts, and sets | Common words, frequent words, occurrence count, predicate count, distinct count, top-*k*, set operations | Collect evidence across lists; distinguish multiplicity from membership |
| Records and arithmetic | Category tally, numeric aggregation, group-by, variable arithmetic | Aggregate records or evaluate assignments rather than copy one local span |
| State tracking | Variable tracking | Follow assignment chains to recover the requested value |
| Graph operations | BFS layers, parents, children, descendants, ancestors, shortest path, reachability, cycle detection, component size, maximum degree | Compute a specified property of a rendered directed graph |

Some task types sample among operations: numeric aggregation includes sum, minimum, maximum, mean, and median; set operations include intersection, union size, and difference. Golds are computed from the generated structures, and independent verification can recompute them from the rendered contexts. Optional few-shot examples share the same total budget. Seeded per-example generation permits parallel construction without making worker scheduling part of the data definition.

These tasks provide controllable supervision, not evidence of general long-context competence by themselves. Their contribution to a trained model needs evaluation on held-out tasks and sources, with explicit separation from benchmark examples.

## Three corpus transformations with known targets

**Untie** splits source text into units, samples a contiguous span, shuffles the units, and attaches display labels. Reconstruction asks for the text in source order; permutation asks only for the label sequence. Both can use a single document or a concatenation of cluster members. Reconstruction spends much of the window twice—once in the shuffled input and once in the answer—so fitting must account for both. Permutation offers a shorter target for a similar ordering exercise.

For an illustrative three-unit example, the display might contain `[1] Results`, `[2] Introduction`, `[3] Methods`; the source-order target is `2, 3, 1`. Actual units contain prose rather than these section labels. The limitation is semantic ambiguity: independent paragraphs may admit several reasonable orders, and cluster rank is a construction order rather than an objectively correct discourse order. We therefore treat source-order recovery as a training hypothesis to test, particularly for clustered sources.

**MRCR** builds passage-retrieval problems using exactly two queried markers. One passage carries both markers positively. Distractors carry only one, explicitly negate the other, or contain unrelated markers. The answer must start with a random string and then reproduce the selected passage verbatim. This combines conjunction, negation, precise copying, and a small instruction-following requirement.

An illustrative query asks for the paragraph marked as both *car* and *lemon*. A paragraph marked *car, but not lemon* is a deliberate near miss. The current implementation fixes the conjunction at two markers; it does not support arbitrary-*k* retrieval.

**IDK** adds randomized attribute–item–code bindings to passages and asks a multiple-choice question. It samples three cases: the requested binding exists; the requested item has no generated binding; or it has a binding for a different attribute. The last case plants a plausible but incorrect answer among the options.

For example, “The access code for the lantern is `7F3A-21`” does not answer a question about the lantern's *security* code. The correct choice is “I don't know.” Randomized codes remove the need for external factual knowledge, while explicit bindings make the intended answer checkable. Whether this supervision improves abstention on natural documents remains an evaluation question.

## Model-authored document tasks, with inference as a replaceable step

Multi-hop QA and summarization share a three-phase workflow: **prepare, infer, assemble**. Preparation writes source documents, messages, and stable request IDs to JSONL. A manifest records the tokenizer, optional final length cap, counts, and hashes of the prepared files. This boundary allows the same requests to run through the built-in OpenAI-compatible API client or an external dispatcher.

For QA, preparation splits a document into measured chunks while retaining internal line breaks. By default, each authoring request receives three or four sampled chunks and asks for one question whose answer requires information from all of them. The model can return `cannot generate` when the chunks do not support that request. Assembly attaches accepted pairs to the **full original document**: PT appends question–answer text; SFT presents the document and numbered questions with corresponding assistant answers. Repeated normalized questions within a document are removed, keeping the first answer. This is duplicate suppression, not selection of the best answer.

Summarization supplies the full document to the authoring model, asks for faithful coverage beyond its opening sections, and optionally requests an approximate word count. Final examples retain the document, instruction, and summary in the selected training representation.

Assembly checks the prepared hashes and joins responses by request ID. Unknown or duplicate IDs fail; missing responses require an explicit partial-import option. Explicit transport failures, truncated completions, malformed QA, and empty summaries also fail. External responses that omit completion metadata cannot prove whether generation finished normally. Usable rows must fit the configured final cap, and an empty final dataset is not published as success. Reports expose missing responses, QA abstentions, and duplicate removal.

The built-in client provides bounded concurrency and transient-error retries for small unattended runs. Large inference can use a dispatcher while preserving the request IDs and response metadata needed by assembly. Neither route supplies a factual-quality guarantee.

## What validation establishes—and what remains to demonstrate

The recorded LC validation passed **108 regression tests**, a **278-example integration matrix**, and a separate **36-example unattended shell run**. Coverage included all task families, GPU embeddings, real API calls, and PT/SFT generation at 8K and 32K budgets. Three synthetic tasks also exercised 128K serialization; the maximum measured sequence was 131,054 tokens. Real model requests used sampled documents of roughly 3K–14K tokens. This was not a test of 128K model inference, and these counts describe smoke artifacts, not the released dataset.

Spot review illustrates why software checks and data quality must be reported separately. A generated answer changed a source curriculum date from **September 2022** to **June 2022**. Other supposed multi-hop questions combined independent facts without demonstrating dependent reasoning. The parser accepted their structure; that did not make their content correct.

**[Evaluation placeholder: add human-reviewed grounding and summary-coverage results, leave-one-chunk-out tests for QA necessity, source-split leakage analysis, and controlled training comparisons against public long-context data. Report retained yield and error rates together. Add the actual release inventory and dataset links only after those artifacts are selected and measured.]**
