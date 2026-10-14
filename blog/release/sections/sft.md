# SFT: spend the teacher budget on a deliberate prompt mixture

*Section draft for the joint release post. Describes the frozen working tree; final release tags, dataset identifiers and dataset counts remain publication placeholders. The accompanying `sft-fact-check.md` records implementation and evidence sources.*

The SFT pipeline separates deciding **what to ask** from generating **how to answer**. Public instruction collections supply an initial prompt pool. Custom documents supply additional, source-derived questions. Embeddings, annotations and quotas select a mixture before teacher inference consumes most of the generation budget. Response filters then remove identifiable failures and preserve decisions for inspection.

There are six numbered stages, with benchmark decontamination between Stages 2 and 3 and final export after Stage 6. The intermediate records matter: a final answer alone cannot explain whether a problem came from its source, question, teacher response or filter.

## Two ways to build the prompt pool

Stage 1 reads Hugging Face datasets and local JSONL. A configured field can hold a plain instruction or an OpenAI/ShareGPT conversation. For a conversation, collection takes the first user turn and prepends a preceding system message when present. Existing assistant responses are not used as teacher targets. This is a single-prompt extraction workflow; sources whose questions depend on separate context fields should first be adapted into complete prompts.

Collection normalizes line endings while preserving indentation, line breaks and Unicode literals. It hashes the resulting text for identity and removes repeated IDs. This deliberately conservative normalization matters for code: changing whitespace can change the program. It also makes exact deduplication a limited claim. Two paraphrases, or two differently formatted versions of the same exercise, can remain distinct. Later prompt selection has a separate, optional semantic constraint.

Stage 2 generates questions from local or Hugging Face source documents. Documents are split into overlapping chunks, normally with token-aware splitting; a character-based fallback is available when the splitter dependencies are absent. A generator is asked for diverse, standalone questions whose answers follow from the chunk. It can return fewer questions than requested, including an empty array when the material is unsuitable.

“Standalone” and “supported by the source” are different requirements. A question about “the parish” can be answerable to a validator reading the passage while leaving the eventual student without an identified subject. The pipeline therefore performs two checks when grounding is enabled, which is the default:

1. A source-blind check sees one question at a time and decides whether its subject and necessary context are identified. Other questions cannot accidentally supply the missing information.
2. A source-visible check assesses answerability, proposes a reference answer and returns an exact evidence quote.

A question survives only when both judgments pass, the reference and quote are nonempty, and the quote is a substring of the source. Each evaluated chunk gets an audit containing accepted and rejected candidates. Malformed or incomplete verdicts trigger retry/failure rather than silently becoming rejected examples.

The quote check proves where the quote came from. It does not prove that the quote entails the proposed answer, or that a calculation is correct. Both semantic checks still depend on a model. The accepted record retains `source_context`, `reference_answer` and `grounding` so these decisions can be inspected later. Teacher inference receives the standalone question; filtering can consult the retained evidence separately.

## Remove configured benchmark overlaps before selection

The merged pool next passes through optional benchmark decontamination. Placing this before embedding and teacher inference avoids spending those resources on prompts already identified as overlapping an evaluation set.

The index uses normalized contiguous word spans from explicitly configured benchmark fields. Defaults are 13-word spans, with whole-item matching for shorter items of at least five words. Matching uses a separate normalization that lowercases and removes punctuation; it never rewrites the training prompt. Removed IDs, matched benchmarks and spans are logged. Requested benchmark subsets must load successfully and contain usable text.

This is protection against a specified form of textual overlap. It does not establish absence of paraphrased contamination, teacher-model pretraining exposure or overlap with unconfigured evaluations. No benchmarks are loaded by the default empty evaluation list. The example collection config includes MMLU and HumanEval test sources, making the final release recipe's source/split inventory and benchmark exclusions especially important. Whitespace tokenization also limits partial-overlap detection in scripts without spaces.

## Select across domains, difficulty levels and clusters

Stage 3 embeds prompts using Sentence Transformers, stores normalized vectors in sharded Parquet, builds a FAISS index and assigns clusters. Several clustering backends are configurable. The current embedding implementation keeps the last 512 model tokens to limit memory and retain task-specific endings after boilerplate. That is an explicit coverage tradeoff for long prompts.

Domain and difficulty labels initially come from heuristics. Optional API annotation supplies validated domain, topic, difficulty and language fields, with model labels taking precedence. Clusters describe embedding neighborhoods; annotations describe requested categories. Neither constitutes proof that an exercise is hard or educationally useful. Annotation also uses a bounded tail of long prompts, so it can miss constraints appearing earlier.

Stage 4 turns the annotated pool into a teacher workload. Configuration sets a total prompt target, domain fractions and per-domain difficulty fractions. Within each domain/difficulty cell, it groups examples by cluster, takes each cluster's most central representative first, and then works from the periphery inward while visiting clusters round-robin. This expresses a useful selection hypothesis: cover distinct neighborhoods before repeatedly drawing from a large, repetitive one. Whether it improves student learning remains an experiment.

Semantic deduplication is **off by default**. When enabled, normalized-vector cosine checks reject candidates too similar to anything already selected, including selections from other clusters and quota cells. Replacement candidates come from the same cell. Scarce cells and integer rounding can leave the output below target; the sampler reports shortfalls and does not silently redistribute the budget.

An earlier same-pool comparison illustrates the constraint rather than downstream value: the selected set contained 39 pairs above cosine 0.92 with semantic deduplication off and zero with it on. Both settings selected 3,805 examples against a 4,000 target. This does not show that all rejected pairs were interchangeable or that removing them improves training.

## Generate responses through a replaceable inference boundary

Stage 5 supports an OpenAI-compatible HTTP endpoint and offline vLLM execution. The HTTP path bounds concurrent requests, stores completions as they arrive and checkpoints durable output. It keeps the finish reason and model identifier. If the server separates reasoning from final content, those fields are preserved; otherwise Stage 6 parses configured delimiters and then heuristic fallbacks.

The pipeline sends a user message containing the prompt and lets the teacher use its native response format. A nonempty answer and a reliably separable reasoning field therefore depend on the teacher and serving configuration. Multiple HTTP candidates use a simple length-based choice among eligible completions, not a learned correctness ranking.

For a larger production run, we previously exported prompts to JSONL, used a separate dispatcher to scale inference, and brought responses back for filtering. That is a useful integration pattern to document alongside the release, with stable IDs and a response-field contract. It remains an external handoff; it should not be presented as an automatic dispatcher integration. The unattended validation used the built-in HTTP path.

## Filter identifiable defects and keep the decisions

Stage 6 applies structural checks, heuristics, domain-specific code checks and an optional sampled model judge, in that order.

Structural checks reject token-limited completions, missing fields, responses outside configured approximate length bounds and repetition loops. Heuristics screen low lexical variety and narrow boilerplate nonanswers. These checks identify symptoms, not factual correctness. There is no mathematical correctness filter.

For configured code domains, the checker considers the last explicitly tagged Python block. The default execution backend uses a disposable Docker container with no network, no host data mounts, a read-only root filesystem, an unprivileged user, dropped capabilities and CPU, memory, process and time limits. Docker/image infrastructure is trusted and containers share the host kernel. Passing means the executed snippet did not reveal a rejected defect; a function that is only defined may never be called. Some fragments and environment-related failures pass with uncertainty reasons. This is not task-level testing against a specification.

The judge receives the complete candidate question, reasoning and answer, plus separate source/reference fields when available. Its rubric asks it to establish the expected behavior, compare the actual candidate and assess any reference independently before assigning a verdict. An ambiguous question, an incorrect or uncertain verdict, or a score below threshold causes rejection. Oversized judge inputs fail instead of being silently clipped, and infrastructure errors raise by default.

Judging is sampled: the configuration default is 10%, although the latest integration run used 100%. An exported example is therefore not necessarily judged. The decision audit distinguishes judged, not sampled, disabled, not reached and error states, and records both acceptance and rejection outcomes.

## Follow one illustrative record

Consider this invented source sentence: “At the Halcyon test facility, Pump A moved 240 liters in 6 minutes.” A generated question might be: “What was Pump A's average flow rate at the Halcyon test facility, given that it moved 240 liters in 6 minutes?” This example illustrates the record flow; it is not an observed release example.

| Boundary | What the record carries |
| --- | --- |
| Generation and grounding | Question, content ID, source/chunk identifiers, original sentence, proposed reference of 40 liters/minute, evidence and verdicts |
| Decontamination | Survives only if it does not match the configured benchmark index |
| Clustering and selection | Domain/difficulty labels, cluster assignment and centroid similarity; selection depends on the configured cell quota |
| Teacher inference | The question alone is sent; the response adds teacher identifier, completion status and response fields |
| Filtering | Structural/heuristic results and any judge decision; source and reference remain distinct from the teacher's answer |
| Final export | `id`, `prompt`, `reasoning`, `answer`, `domain`, `difficulty`, `language`, `source`, `teacher_model` |

Final export checks unique IDs and nonempty prompt/reasoning/answer fields. Rich grounding and judge metadata remain in intermediate artifacts rather than the compact training schema. The release should link these audits and source manifests where redistribution permits them.

## What the execution evidence establishes

DuckDB tracks stages/items while sharded files hold outputs; the implementation also uses stage-specific completion ledgers and recovery logic. Output is written before completion is recorded. Resume assumes unchanged inputs, configuration and model endpoints: use a fresh run ID when these change. Publication is not a single transaction across every shard.

An earlier unattended medium run produced 3,131 exports in 51 minutes 14 seconds, before the latest grounding/judge changes. The latest validation passed 288 tests and ran the updated HTTP, GPU clustering, Docker and export flow on a small workload: 20 teacher responses produced 18 exports, four originating in Stage 2. Both excluded responses hit the token limit. A no-work resume preserved the recorded output and audit hashes.

Those are engineering results, not final release dataset sizes or quality estimates. Content probes still found false rejections in grounding and errors missed by the judge. **[EXPERIMENT PLACEHOLDER: independent human review, matched-budget student training, comparisons with public SFT datasets, and selection/filtering ablations.]** These measurements should determine any claim that the released data improves training.
