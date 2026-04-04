"""
Pydantic v2 configuration models for the SFT dataset construction pipeline.
All parameters are loaded from a YAML file; placeholders {run_id} and {base_path}
are resolved at startup before any stage runs.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_placeholders(value: Any, ctx: dict[str, str]) -> Any:
    """Recursively resolve {key} placeholders in strings."""
    if isinstance(value, str):
        for k, v in ctx.items():
            value = value.replace(f"{{{k}}}", v)
        return value
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, ctx) for item in value]
    return value


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class GlobalConfig(BaseModel):
    seed: int = 42
    dry_run: bool = False
    run_id: str = "run_001"
    base_path: str = "/data/sft-pipeline/runs/{run_id}"
    checkpoint_db: str = "{base_path}/checkpoints.duckdb"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # "cuda" | "rocm" | "cpu" — controls GPU backend for embeddings/inference
    device: Literal["cuda", "rocm", "cpu"] = "cpu"
    # Override the HuggingFace cache directory (sets HF_HOME env var at runtime).
    # Useful when the default ~/.cache/huggingface would fill up a small home partition.
    hf_home: str | None = None
    # Ray cluster address for distributed stages ("auto" connects to a running cluster).
    ray_address: str = "auto"


class DatasetSource(BaseModel):
    source: Literal["hf_dataset", "local_jsonl"]
    # hf_dataset fields
    hf_repo_id: str | None = None
    hf_split: str = "train"
    hf_config: str | None = None
    # local_jsonl fields
    path: str | None = None
    # shared
    prompt_field: str = "prompt"
    domain_hint: str | None = None  # optional manual domain override
    max_examples: int | None = None  # cap rows loaded from this source (None = all)

    @model_validator(mode="after")
    def check_source_fields(self) -> DatasetSource:
        if self.source == "hf_dataset" and not self.hf_repo_id:
            raise ValueError("hf_repo_id required when source='hf_dataset'")
        if self.source == "local_jsonl" and not self.path:
            raise ValueError("path required when source='local_jsonl'")
        return self


class Stage1Config(BaseModel):
    enabled: bool = True
    datasets: list[DatasetSource] = Field(default_factory=list)
    batch_size: int = 10_000
    output_path: str = "{base_path}/stage1/prompts.jsonl"
    # Set to true to distribute source collection across a Ray cluster.
    # Each dataset source becomes a Ray task; dedup merge runs on the head node
    # after all tasks complete. Requires ray_address in global config.
    distributed: bool = False


class CorpusSource(BaseModel):
    source: Literal["local", "hf_dataset"]
    # local fields
    path: str | None = None
    # hf_dataset fields
    hf_repo_id: str | None = None
    hf_split: str = "train"
    text_field: str = "text"
    domain_hint: str | None = None

    @model_validator(mode="after")
    def check_source_fields(self) -> CorpusSource:
        if self.source == "local" and not self.path:
            raise ValueError("path required when source='local'")
        if self.source == "hf_dataset" and not self.hf_repo_id:
            raise ValueError("hf_repo_id required when source='hf_dataset'")
        return self


class Stage2Config(BaseModel):
    enabled: bool = True
    corpora: list[CorpusSource] = Field(default_factory=list)
    chunk_size_tokens: int = 1024
    chunk_overlap_tokens: int = 128
    prompts_per_chunk: int = 4
    # Generator model served via vLLM HTTP (lightweight 7B)
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    generator_endpoint: str = "http://localhost:8001/v1"
    generator_temperature: float = 0.8
    generator_max_tokens: int = 512
    max_workers: int = 16
    output_path: str = "{base_path}/stage2/prompts.jsonl"


class Stage3Config(BaseModel):
    enabled: bool = True
    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 512
    # FAISS
    faiss_index_type: Literal["Flat", "IVFFlat", "IVFPQ"] = "IVFFlat"
    faiss_nlist: int = 1000
    faiss_nprobe: int = 50
    faiss_training_sample: int = 500_000
    # Clustering
    clustering_algorithm: Literal["hdbscan", "kmeans", "flash_kmeans"] = "hdbscan"
    hdbscan_min_cluster_size: int = 100
    n_clusters: int = 50  # used with kmeans and flash_kmeans
    # Difficulty heuristics
    difficulty_easy_max_tokens: int = 50
    difficulty_hard_min_tokens: int = 200
    # Output
    embeddings_dir: str = "{base_path}/stage3/embeddings"
    faiss_index_path: str = "{base_path}/stage3/faiss.index"
    output_path: str = "{base_path}/stage3/clustered_prompts.jsonl"


class Stage4Config(BaseModel):
    enabled: bool = True
    total_prompts: int = 7_000_000
    domain_quotas: dict[str, float] = Field(
        default_factory=lambda: {
            "math": 0.25, "code": 0.20, "science": 0.20,
            "general": 0.20, "language": 0.15,
        }
    )
    difficulty_quotas: dict[str, float] = Field(
        default_factory=lambda: {"easy": 0.20, "medium": 0.50, "hard": 0.30}
    )
    dedup_cosine_threshold: float = Field(0.92, ge=0.0, le=1.0)
    output_path: str = "{base_path}/stage4/sampled_prompts.jsonl"

    @model_validator(mode="after")
    def quotas_sum_to_one(self) -> Stage4Config:
        domain_sum = sum(self.domain_quotas.values())
        difficulty_sum = sum(self.difficulty_quotas.values())
        if abs(domain_sum - 1.0) > 0.01:
            raise ValueError(f"domain_quotas must sum to 1.0, got {domain_sum:.3f}")
        if abs(difficulty_sum - 1.0) > 0.01:
            raise ValueError(f"difficulty_quotas must sum to 1.0, got {difficulty_sum:.3f}")
        return self


class VllmEngineConfig(BaseModel):
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = Field(0.90, ge=0.1, le=1.0)
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    enable_chunked_prefill: bool = True


class GenerationConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    n_candidates: int = 2


class ReasoningDelimiters(BaseModel):
    think_start: str = "<think>"
    think_end: str = "</think>"
    answer_start: str = "<answer>"
    answer_end: str = "</answer>"


class Stage5Config(BaseModel):
    enabled: bool = True
    model: str = "Qwen/Qwen3.5-122B-A10B"
    n_replicas: int = 64  # Ray actors; each wraps one vLLM LLM instance
    vllm_engine: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    delimiters: ReasoningDelimiters = Field(default_factory=ReasoningDelimiters)
    batch_size: int = 256
    checkpoint_every: int = 5_000
    max_retries: int = 3
    output_path: str = "{base_path}/stage5/responses.jsonl"


class StructuralFilterConfig(BaseModel):
    min_response_tokens: int = 50
    max_response_tokens: int = 8_000
    max_repetition_ngram: int = 5
    max_repetition_count: int = 3


class HeuristicFilterConfig(BaseModel):
    min_info_density: float = Field(0.3, ge=0.0, le=1.0)
    flag_self_contradiction: bool = True


class MathFilterConfig(BaseModel):
    enabled: bool = True
    domains: list[str] = Field(default_factory=lambda: ["math", "science"])


class CodeFilterConfig(BaseModel):
    enabled: bool = True
    sandbox: Literal["subprocess", "e2b"] = "subprocess"
    timeout_seconds: int = 10
    domains: list[str] = Field(default_factory=lambda: ["code"])


class LLMJudgeConfig(BaseModel):
    enabled: bool = True
    model_endpoint: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    score_threshold: float = Field(6.0, ge=1.0, le=10.0)
    sample_rate: float = Field(0.10, ge=0.0, le=1.0)


class Stage6Config(BaseModel):
    enabled: bool = True
    structural: StructuralFilterConfig = Field(default_factory=StructuralFilterConfig)
    heuristic: HeuristicFilterConfig = Field(default_factory=HeuristicFilterConfig)
    math: MathFilterConfig = Field(default_factory=MathFilterConfig)
    code: CodeFilterConfig = Field(default_factory=CodeFilterConfig)
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)
    output_path: str = "{base_path}/stage6/filtered.jsonl"
    report_path: str = "{base_path}/stage6/filter_report.json"


class ExportConfig(BaseModel):
    final_jsonl_path: str = "{base_path}/final/dataset.jsonl"
    shard_size_mb: int = 500
    hf_repo_id: str | None = None
    push_to_hub: bool = False


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    stage1_collect: Stage1Config = Field(default_factory=Stage1Config)
    stage2_generate: Stage2Config = Field(default_factory=Stage2Config)
    stage3_cluster: Stage3Config = Field(default_factory=Stage3Config)
    stage4_sample: Stage4Config = Field(default_factory=Stage4Config)
    stage5_inference: Stage5Config = Field(default_factory=Stage5Config)
    stage6_filter: Stage6Config = Field(default_factory=Stage6Config)
    export: ExportConfig = Field(default_factory=ExportConfig)

    model_config = {"populate_by_name": True}

    @property
    def base_path(self) -> str:
        return self.global_.base_path

    @property
    def run_id(self) -> str:
        return self.global_.run_id


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """
    Load and validate the pipeline config from a YAML file.
    Resolves {run_id} and {base_path} placeholders throughout.
    Optionally merges ``overrides`` (dotted-key dict) on top of the file values.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raw = {}

    if overrides:
        _deep_merge(raw, overrides)

    # Resolve run_id first, then base_path
    run_id = str(raw.get("global", {}).get("run_id", "run_001"))
    base_path_template = str(
        raw.get("global", {}).get("base_path", "/data/sft-pipeline/runs/{run_id}")
    )
    base_path = base_path_template.replace("{run_id}", run_id)

    # Register both short forms ({base_path}) and qualified forms ({global.base_path})
    # so either syntax works in YAML values.
    ctx = {
        "run_id": run_id,
        "base_path": base_path,
        "global.run_id": run_id,
        "global.base_path": base_path,
    }

    # Pass 1: resolve placeholders in the raw YAML dict (covers explicitly set fields).
    raw = _resolve_placeholders(raw, ctx)
    config = PipelineConfig.model_validate(raw)

    # Pass 2: resolve placeholders that survived in Pydantic model *defaults*
    # (fields not present in the YAML have their defaults applied only after
    # model_validate, so they are missed by Pass 1).
    config_dict = config.model_dump(by_alias=True)
    config_dict = _resolve_placeholders(config_dict, ctx)
    return PipelineConfig.model_validate(config_dict)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
