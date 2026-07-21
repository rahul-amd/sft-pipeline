# Decontamination (`sft_pipeline/decontam/`)

Removes prompts from the collected pool that overlap **downstream eval
benchmarks**, so the teacher never generates training data for questions we plan
to benchmark the student model on. Runs as its own pipeline stage
(`decontaminate`) **between Stage 2 and Stage 3**.

```
Stage 1 (collect) ─┐
                   ├─▶  decontaminate  ─▶  clean pool  ─▶  Stage 3 (cluster) ─▶ …
Stage 2 (generate)─┘        │
                            ├─▶ removed/            (audit: every dropped prompt)
                            └─▶ decontam_report.json (per-eval / per-source counts)
```

- **Stage code:** [`../stages/decontaminate.py`](../stages/decontaminate.py) (`run_decontaminate`)
- **This package:** [`normalize.py`](normalize.py) (tokenizer) + [`eval_index.py`](eval_index.py) (eval loading + the containment index)
- **Config models:** `DecontaminateConfig`, `EvalDatasetSource` in [`../config.py`](../config.py)
- **Example config:** [`../../config/decontaminate.yaml`](../../config/decontaminate.yaml)

---

## How contamination is detected

The method is **word-level n-gram containment** — the standard GPT-3 / LLaMA
decontamination approach — with an exact-match fallback for short eval items.

For each eval item's text:

| eval item length (tokens) | how it is indexed |
|---|---|
| `>= ngram_size` (13) | every contiguous **13-gram** |
| `min_gram_size … ngram_size-1` (5–12) | one gram of its **own length** (exact whole-item containment) |
| `< min_gram_size` (< 5) | **dropped** (counted + logged) |

A collected prompt is **contaminated** if *any* of its n-grams (at any gram
length present in the index) collides with an indexed eval gram. **First
collision wins** — the matched span text and the owning eval are recorded, and
the prompt is dropped.

Why these choices:

- **Any single collision removes the prompt.** Decontamination is asymmetric: a
  missed eval leak silently inflates benchmark scores, while dropping a few
  extra training prompts costs almost nothing. We optimize for recall.
- **Short-item exact fallback.** A 6-word eval question never produces a 13-gram,
  so without a fallback it would be invisible. Indexing it as a single 6-gram
  makes "prompt contains this exact question" detectable through the *same*
  lookup path.
- **`min_gram_size` floor (default 5) — the critical guard.** Without it, a
  1-token eval field like `"True"` or an MMLU `choices` entry like `"0"` becomes
  a 1-gram that removes *every* training prompt containing that token. The floor
  drops such items instead; the dropped count per eval is surfaced in the report
  as `eval_items_dropped_short`. (Real MMLU example: `choices = ['0','4','2','6']`
  — all dropped, so a benign prompt like "Compute 0 plus 1" is not wrongly
  removed. Covered by `test_mmlu_choices_floor_real`.)

### Grams are keyed by string, not `hash()`

The index stores each gram as its joined **string** (`_SEP.join(tokens)`), not a
precomputed integer hash. CPython's `hash()` is per-process randomized
(`PYTHONHASHSEED`), so integer keys computed in the parent would **miss** in a
`spawn` worker. A string-keyed `dict` is rehashed correctly inside every process
(fork *or* spawn), keeping matching decisions process-independent. See the note
at the top of [`eval_index.py`](eval_index.py).

---

## Normalization ([`normalize.py`](normalize.py))

`tokenize(text)` is deliberately **more aggressive** than the Stage 1
`prompt_id` normalizer (which preserves case/punctuation):

```
NFKC → lowercase → replace [\W_]+ (Unicode) with space → split on whitespace
```

So `"What is 2+2?"` → `["what", "is", "2", "2"]`. Punctuation and case can't
block a match, and the regex is Unicode-aware, so accented and CJK **letters/
digits are kept** (not erased).

> **Limitation — space-free scripts.** Tokens are whitespace-delimited, so CJK /
> Thai text tokenizes as one giant token per run and n-gram containment degrades
> to whole-run exact matching (partial overlap won't be caught). Every caller
> goes through this one function, so adding a real segmenter (e.g. `jieba`) here
> would fix it globally.

---

## Eval configuration (`EvalDatasetSource`)

Each eval benchmark is one entry under `decontaminate.evals`:

```yaml
- name: mmlu                  # label used in the per-eval report
  source: hf_dataset         # or: local_jsonl
  hf_repo_id: cais/mmlu
  hf_configs: all            # null → default config; "all" → every config; [a, b] → those configs
  splits: [test, validation] # eval items live in held-out splits, never 'train'
  match_fields: [question, choices]   # >= 1 field; matched INDEPENDENTLY
  max_examples: null         # optional cap on items loaded
```

- **`hf_configs: all`** expands to every config via
  `datasets.get_dataset_config_names` (e.g. MMLU's 57 subjects). To target only a
  config *literally named* `all`, write `hf_configs: ["all"]`.
- **`match_fields`** are matched independently — a prompt is contaminated if it
  matches *any* listed field. Each field value may be
  ([`extract_field_text`](eval_index.py)):
  - a **plain string**
  - **dot-notation nested** (`"a.b"`)
  - a **list** (e.g. MMLU `choices`) → space-joined
  - an **OpenAI / ShareGPT message list** → first user turn (reuses Stage 1's
    `_extract_prompt`)
- **`local_jsonl`** evals point at a `path` instead of a repo/split.
- Splits missing from a given config are skipped quietly (a config often lacks
  `validation`). Gated datasets need `HF_TOKEN` in the environment.

---

## The index (`EvalNGramIndex`)

Built once on the head node by `build_index(evals, ngram_size, min_gram_size)`:

- Internal layout: `dict[gram_length → dict[gram_string → eval_id]]`. Bucketing
  by length means the query only scans gram lengths that actually exist (usually
  just `{13}` plus a handful of short-item lengths).
- **Attribution**: the first eval to produce a gram owns it (`dict.setdefault`),
  so `match()` can name which benchmark caused each removal.
- **Memory**: every gram is a live string dict key on the head node (and per
  worker on spawn platforms). `build_index` logs an approximate key-text
  footprint; for very large eval suites this is the main resource to watch.
- `match(tokens) -> (eval_id, span_text) | None` — first colliding gram wins.

---

## Execution model ([`../stages/decontaminate.py`](../stages/decontaminate.py))

**Parallelism is per-shard, not per-record.** Each worker reads, scans, and
writes one *whole* input shard and returns only a small stats dict — records
never cross the process boundary. Per-record fan-out was measured to *lose* to
serial: the per-record work (tokenize + dict lookups) is microseconds, so
pickling each record to a worker and back dominated.

- `n_workers` (`null` → `os.cpu_count()`, `1` → serial) sets the pool size.
- **Index sharing**: on Linux the index is set as a module global and shared to
  workers via **`fork` copy-on-write** (no pickling). On spawn/forkserver
  (Windows/macOS dev) it's shipped once per worker via the pool initializer.
- Bounded submission window (`n_workers * 2` in-flight) with **no barrier**
  between shards — a slow shard occupies only its own worker.
- Matching is deterministic (no RNG), so **parallel output == serial output**
  (asserted by `test_parallel_matches_serial`).

---

## Output layout & resume

With the default paths, a run under `{base_path}/stage_decontam/` produces:

```
stage_decontam/
├── clean/                     # ← output_dir: LEAF dir, survivor shards ONLY
│   ├── stage1-part-000000.jsonl
│   └── stage2-part-000000.jsonl
├── removed/                   # ← removed_dir: uncapped audit of dropped prompts
│   └── stage1-part-000000.jsonl   # {prompt_id, source, matched_eval, matched_ngram}
├── decontam_report.json       # ← report_path
└── _shard_stats.jsonl         # resume ledger (lives beside the report)
```

- **`clean/` is deliberately a leaf directory** holding only survivor shards.
  The report, `removed/`, and the resume ledger live *outside* it, so Stage 3
  can glob the clean pool (recursively or not) without ever ingesting
  bookkeeping files.
- **One output shard per input shard** (`{stage1,stage2}-<name>.jsonl`), written
  via tmp-file + atomic `rename`.
- **Shard-level resume**: a shard counts as done only after its worker's atomic
  rename *and* the parent recording it in `_shard_stats.jsonl`
  (publish-before-record). A re-run skips shards already listed and reprocesses
  at most the ones that were in flight. Per-shard stats are accumulated from the
  ledger so the final report is correct across resumes.

### `decontam_report.json`

```json
{
  "total_input": 14800000,
  "total_removed": 20431,
  "total_survivors": 14779569,
  "removal_rate": 0.00138,
  "ngram_size": 13,
  "min_gram_size": 5,
  "gram_lengths": [5, 6, 7, 8, 9, 10, 11, 12, 13],
  "total_eval_ngrams": 1830245,
  "eval_items_loaded": {"mmlu": 15908, "gsm8k": 1319, "humaneval": 164},
  "eval_items_dropped_short": {"mmlu": 4021},
  "removed_per_eval": {"mmlu": 12004, "gsm8k": 8100, "humaneval": 327},
  "removed_per_source": {"open-web-math": 9002, "...": 0},
  "elapsed_seconds": 1875.3
}
```

A large `eval_items_dropped_short` value means a `match_field` is mostly noise
(e.g. single-word answers) — expected for MMLU `choices`, a red flag elsewhere.

---

## Stage 3 integration

Stage 3 does **not** hard-code the decontaminated dir. `stage3_cluster.
_resolve_input_dirs(cfg)` returns `[decontaminate.output_dir]` when that dir
exists and is non-empty, otherwise `[stage1, stage2]`. So:

- decontamination **enabled + evals set** → Stage 3 reads the clean pool;
- **disabled**, **no evals**, or **not yet run** → Stage 3 reads the raw pool.

Existing runs are therefore unaffected, and the `run` command only executes the
stage when `enabled and evals` are both set.

---

## Running it

```bash
# Standalone against an existing run (auto-picked up by a later Stage 3):
sft-pipeline run-stage decontaminate --config config/decontaminate.yaml

# As part of a full pipeline run (runs only if enabled and evals are set):
sft-pipeline run --config config/prod.yaml
```

## Tuning

| Knob | Default | Effect |
|---|---|---|
| `ngram_size` | 13 | Shared-span length. Smaller → higher recall, more false positives. |
| `min_gram_size` | 5 | Floor below which eval items are dropped. Lower → more aggressive, risks over-removal from short fields. |
| `n_workers` | 1 (configs: `null`) | Per-shard worker processes. `null` → `os.cpu_count()`. |
| `match_fields` | — | Which eval fields count as contamination signal. |

---

## Tests

- Unit: [`../../tests/unit/decontam/`](../../tests/unit/decontam/) — tokenizer,
  13-gram hit, short-item fallback, `min_gram_size` guard, per-eval attribution,
  the four field shapes.
- Integration (synthetic): [`../../tests/integration/test_decontaminate.py`](../../tests/integration/test_decontaminate.py)
  — planted contamination → clean pool + report + resume + Stage 3 resolution,
  serial == parallel.
- Integration (real, network-gated): [`../../tests/integration/test_decontaminate_real.py`](../../tests/integration/test_decontaminate_real.py)
  — real GSM8K + MMLU; skips when offline or `SFT_SKIP_NETWORK_TESTS` is set.

```bash
python -m pytest tests/unit/decontam tests/integration/test_decontaminate.py -v
SFT_SKIP_NETWORK_TESTS=1 python -m pytest tests/integration/test_decontaminate_real.py  # skips
```

## Extending

- **Add an eval**: append an `EvalDatasetSource` to `decontaminate.evals`. No
  code change for standard HF/JSONL datasets.
- **Support space-free languages**: replace the whitespace split in
  `normalize.tokenize` with a segmenter — all matching flows through it.
- **Swap the index backing** (e.g. sorted numpy arrays + `searchsorted` for a
  smaller footprint at very large eval scale): keep the `EvalNGramIndex.match`
  interface; nothing else needs to change.
