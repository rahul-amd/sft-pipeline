"""
vLLM offline batch inference runner.

Runs Qwen3.5-122B-A10B (or any configured model) in offline batch mode
using vLLM's LLM class. Supports multi-node scaling via Ray actors —
each actor wraps one LLM instance and processes a shard of prompts.

For local testing (RTX 5060), set n_replicas=1 and a smaller model.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-node batch inference (used by each Ray actor)
# ---------------------------------------------------------------------------

def run_inference_batch(
    prompts: list[dict],      # list of prompt records from Stage 4
    model_name: str,
    vllm_engine_cfg: Any,     # VllmEngineConfig
    generation_cfg: Any,      # GenerationConfig
    delimiters: Any,          # ReasoningDelimiters
    device: str = "cuda",
) -> Iterator[dict]:
    """
    Run vLLM offline inference on a list of prompt records.
    Yields enriched records with 'reasoning', 'answer', 'teacher_model' fields.

    Args:
        prompts: List of prompt dicts (must have 'prompt_id' and 'prompt').
        model_name: HuggingFace model ID.
        vllm_engine_cfg: VllmEngineConfig with tensor_parallel_size etc.
        generation_cfg: GenerationConfig with temperature, max_tokens, n_candidates.
        delimiters: ReasoningDelimiters for parsing.
        device: 'cuda' or 'rocm' (both map to 'cuda' in PyTorch/vLLM).
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from sft_pipeline.inference.output_parser import select_best_candidate
    from sft_pipeline.inference.prompt_formatter import apply_chat_template

    logger.info(
        "Initializing vLLM: model=%s, TP=%d",
        model_name, vllm_engine_cfg.tensor_parallel_size,
    )

    llm = LLM(
        model=model_name,
        tensor_parallel_size=vllm_engine_cfg.tensor_parallel_size,
        pipeline_parallel_size=vllm_engine_cfg.pipeline_parallel_size,
        gpu_memory_utilization=vllm_engine_cfg.gpu_memory_utilization,
        max_model_len=vllm_engine_cfg.max_model_len,
        dtype=vllm_engine_cfg.dtype,
        enable_chunked_prefill=vllm_engine_cfg.enable_chunked_prefill,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=generation_cfg.temperature,
        top_p=generation_cfg.top_p,
        max_tokens=generation_cfg.max_tokens,
        n=generation_cfg.n_candidates,
    )

    # Format all prompts with chat template
    formatted = [
        apply_chat_template(tokenizer, rec["prompt"], delimiters)
        for rec in prompts
    ]

    logger.info("Running vLLM.generate() on %d prompts", len(formatted))
    outputs = llm.generate(formatted, sampling_params)

    for rec, output in zip(prompts, outputs):
        candidate_texts = [o.text for o in output.outputs]
        best, _ = select_best_candidate(candidate_texts, delimiters)

        yield {
            **rec,
            "reasoning": best.reasoning,
            "answer": best.answer,
            "teacher_model": model_name,
            "used_fallback_parse": best.used_fallback,
        }


# ---------------------------------------------------------------------------
# Ray actor wrapper (used for multi-node / multi-replica)
# ---------------------------------------------------------------------------

def build_ray_actor_class():
    """
    Returns a Ray remote class that wraps a single vLLM inference worker.
    Import is deferred so this module can be imported without Ray installed.
    """
    try:
        import ray
    except ImportError:
        return None

    @ray.remote(num_gpus=0)  # GPU allocation is handled by vLLM internally
    class InferenceActor:
        def __init__(
            self,
            model_name: str,
            vllm_engine_cfg,
            generation_cfg,
            delimiters,
            device: str,
        ) -> None:
            self.model_name = model_name
            self.vllm_engine_cfg = vllm_engine_cfg
            self.generation_cfg = generation_cfg
            self.delimiters = delimiters
            self.device = device
            self._llm = None
            self._tokenizer = None

        def _ensure_loaded(self) -> None:
            if self._llm is not None:
                return
            from vllm import LLM
            from transformers import AutoTokenizer

            self._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.vllm_engine_cfg.tensor_parallel_size,
                pipeline_parallel_size=self.vllm_engine_cfg.pipeline_parallel_size,
                gpu_memory_utilization=self.vllm_engine_cfg.gpu_memory_utilization,
                max_model_len=self.vllm_engine_cfg.max_model_len,
                dtype=self.vllm_engine_cfg.dtype,
                enable_chunked_prefill=self.vllm_engine_cfg.enable_chunked_prefill,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

        def process_batch(self, prompts: list[dict]) -> list[dict]:
            from vllm import SamplingParams
            from sft_pipeline.inference.output_parser import select_best_candidate
            from sft_pipeline.inference.prompt_formatter import apply_chat_template

            self._ensure_loaded()
            sampling_params = SamplingParams(
                temperature=self.generation_cfg.temperature,
                top_p=self.generation_cfg.top_p,
                max_tokens=self.generation_cfg.max_tokens,
                n=self.generation_cfg.n_candidates,
            )
            formatted = [
                apply_chat_template(self._tokenizer, rec["prompt"], self.delimiters)
                for rec in prompts
            ]
            outputs = self._llm.generate(formatted, sampling_params)
            results = []
            for rec, output in zip(prompts, outputs):
                candidate_texts = [o.text for o in output.outputs]
                best, _ = select_best_candidate(candidate_texts, self.delimiters)
                results.append({
                    **rec,
                    "reasoning": best.reasoning,
                    "answer": best.answer,
                    "teacher_model": self.model_name,
                    "used_fallback_parse": best.used_fallback,
                })
            return results

    return InferenceActor
