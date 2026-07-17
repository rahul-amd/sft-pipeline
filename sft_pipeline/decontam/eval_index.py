"""
Eval loading + the n-gram containment index used for decontamination.

Contamination model (see DecontaminateConfig):
  - Each eval item's match-field text is tokenized (decontam.normalize.tokenize).
  - Items with >= ngram_size tokens contribute every contiguous ngram_size-gram.
  - Items SHORTER than ngram_size but >= min_gram_size contribute a single gram
    of their own length — the exact whole-item containment fallback, folded into
    the same mechanism (a short item is "contained" iff the prompt has that
    exact gram).
  - Items shorter than min_gram_size are DROPPED (counted, logged). A 1-token
    field like "True" would otherwise remove every prompt containing that word.
  - A prompt is contaminated if ANY of its grams (of any length present in the
    index) collides with an eval gram. First collision wins; the matched span
    and the eval that owns the gram are reported.

Grams are keyed by their joined STRING, not by ``hash()``: CPython's ``hash()``
is per-process randomized (PYTHONHASHSEED), so precomputed integer keys would
miss in ``spawn`` worker processes. A string-keyed dict is rehashed correctly
inside each process (fork or spawn), so matching is process-independent.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Iterator

from sft_pipeline.config import EvalDatasetSource
from sft_pipeline.decontam.normalize import tokenize
from sft_pipeline.stages.stage1_collect import _extract_prompt, _get_field
from sft_pipeline.storage import iter_jsonl

logger = logging.getLogger(__name__)

# Unambiguous gram-token separator (tokens are already whitespace-free).
_SEP = "\x00"


# ---------------------------------------------------------------------------
# Field extraction (the four supported match_field value shapes)
# ---------------------------------------------------------------------------

def extract_field_text(val: Any) -> str | None:
    """Turn an eval field value into a single match string, or None.

    Handles: plain string, JSON-encoded conversation string, list of scalars
    (e.g. MMLU ``choices`` → space-joined), OpenAI/ShareGPT message list (first
    user turn), and single-message dicts. Nested access is done by the caller
    via ``_get_field`` before this is called.
    """
    if val is None:
        return None
    if isinstance(val, str):
        return _extract_prompt(val)
    if isinstance(val, list):
        if val and isinstance(val[0], dict) and ("role" in val[0] or "from" in val[0]):
            return _extract_prompt(val)  # conversation
        parts = [str(x) for x in val if x is not None and str(x).strip()]
        return " ".join(parts) if parts else None
    if isinstance(val, dict):
        return _extract_prompt(val)
    text = str(val).strip()
    return text or None


# ---------------------------------------------------------------------------
# Eval row iteration
# ---------------------------------------------------------------------------

def _iter_eval_rows(src: EvalDatasetSource) -> Iterator[dict]:
    """Yield raw rows from an eval source, honoring configs/splits/max_examples."""
    n_yielded = 0
    limit = src.max_examples

    if src.source == "local_jsonl":
        for row in iter_jsonl(src.path):
            if limit is not None and n_yielded >= limit:
                return
            yield row
            n_yielded += 1
        return

    # hf_dataset
    from datasets import get_dataset_config_names, load_dataset

    if src.hf_configs == "all":
        try:
            configs: list[str | None] = list(get_dataset_config_names(src.hf_repo_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Eval '%s': could not list configs (%s); using default", src.name, exc)
            configs = [None]
    elif isinstance(src.hf_configs, list):
        configs = list(src.hf_configs)
    else:
        configs = [None]

    for config in configs:
        for split in src.splits:
            if limit is not None and n_yielded >= limit:
                return
            try:
                ds = load_dataset(
                    src.hf_repo_id, name=config, split=split,
                    trust_remote_code=False,
                )
            except Exception as exc:  # noqa: BLE001
                # A config often lacks some of the requested splits — skip quietly.
                logger.debug(
                    "Eval '%s': skip %s/%s (%s)", src.name, config, split, exc
                )
                continue
            for row in ds:
                if limit is not None and n_yielded >= limit:
                    return
                yield row
                n_yielded += 1


def iter_eval_texts(src: EvalDatasetSource) -> Iterator[str]:
    """Yield every non-empty match-field text for an eval source."""
    for row in _iter_eval_rows(src):
        if not isinstance(row, dict):
            continue
        for field in src.match_fields:
            text = extract_field_text(_get_field(row, field))
            if text:
                yield text


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

class EvalNGramIndex:
    """Word-gram containment index over one or more eval datasets."""

    def __init__(self, ngram_size: int, min_gram_size: int = 5) -> None:
        if min_gram_size > ngram_size:
            raise ValueError(
                f"min_gram_size ({min_gram_size}) must be <= ngram_size ({ngram_size})"
            )
        self.ngram_size = ngram_size
        self.min_gram_size = min_gram_size
        self.eval_names: list[str] = []
        self._name_to_id: dict[str, int] = {}
        # gram_length -> {gram_string -> eval_id (first eval that produced it)}
        self._by_len: dict[int, dict[str, int]] = {}
        self._gram_lens: tuple[int, ...] = ()
        # Eval items dropped because they tokenize below min_gram_size.
        self.dropped_short: Counter = Counter()

    # -- build -------------------------------------------------------------
    def _eval_id(self, name: str) -> int:
        eid = self._name_to_id.get(name)
        if eid is None:
            eid = self._name_to_id[name] = len(self.eval_names)
            self.eval_names.append(name)
        return eid

    def _index_gram(self, gram_tokens: list[str], eid: int) -> None:
        d = self._by_len.get(len(gram_tokens))
        if d is None:
            d = self._by_len[len(gram_tokens)] = {}
        key = _SEP.join(gram_tokens)
        d.setdefault(key, eid)  # keep first owner for stable attribution

    def add_text(self, text: str, eval_name: str) -> None:
        tokens = tokenize(text)
        n = len(tokens)
        if n < self.min_gram_size:
            # Too short to be a meaningful contamination signal — a tiny gram
            # (e.g. "True", "0 1 2 3") would match legitimate prompts wholesale.
            if n > 0:
                self.dropped_short[eval_name] += 1
            return
        eid = self._eval_id(eval_name)
        if n < self.ngram_size:
            self._index_gram(tokens, eid)  # short-item exact fallback
        else:
            size = self.ngram_size
            for i in range(n - size + 1):
                self._index_gram(tokens[i:i + size], eid)

    def finalize(self) -> None:
        # Scan shorter grams first — the short-item fallback dicts are tiny, so a
        # prompt that hits one returns fast before the big ngram_size pass.
        self._gram_lens = tuple(sorted(self._by_len.keys()))

    @property
    def total_grams(self) -> int:
        return sum(len(d) for d in self._by_len.values())

    @property
    def gram_lens(self) -> tuple[int, ...]:
        return self._gram_lens

    # -- query -------------------------------------------------------------
    def match(self, tokens: list[str]) -> tuple[int, str] | None:
        """Return (eval_id, matched_span_text) for the first colliding gram, else None."""
        n = len(tokens)
        for L in self._gram_lens:
            if n < L:
                continue
            d = self._by_len[L]
            for i in range(n - L + 1):
                gram = tokens[i:i + L]
                eid = d.get(_SEP.join(gram))
                if eid is not None:
                    return eid, " ".join(gram)
        return None


def build_index(
    evals: list[EvalDatasetSource], ngram_size: int, min_gram_size: int = 5,
) -> tuple[EvalNGramIndex, dict[str, int]]:
    """Load every eval and build the containment index.

    Returns the finalized index and a per-eval item-count dict (for the report).
    """
    index = EvalNGramIndex(ngram_size, min_gram_size)
    per_eval_items: dict[str, int] = {}
    for src in evals:
        count = 0
        for text in iter_eval_texts(src):
            index.add_text(text, src.name)
            count += 1
        per_eval_items[src.name] = count
        logger.info(
            "Eval '%s': %d match texts loaded, %d below min_gram_size=%d dropped "
            "(index now %d grams)",
            src.name, count, index.dropped_short.get(src.name, 0),
            min_gram_size, index.total_grams,
        )
    index.finalize()
    # Rough footprint hint: every gram is a string dict key held in RAM on the
    # head node (and per-worker on spawn platforms).
    key_bytes = sum(len(k) for d in index._by_len.values() for k in d)
    logger.info(
        "Eval index finalized: %d grams, ~%.0f MB of key text (dict overhead extra)",
        index.total_grams, key_bytes / 1e6,
    )
    return index, per_eval_items
