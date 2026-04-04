"""Environment variables → PipelineConfig construction."""

from __future__ import annotations

import os
import logging
from typing import Any, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    return int(val) if val is not None else default

def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val is not None else default

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")

def _env_int_or_none(key: str, default: int | None) -> int | None:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    return int(val)


def build_pipeline_config(
    source: str,
    template_class: Type[BaseModel] | None,
) -> Any:
    """Build a PipelineConfig from environment variables."""
    from docling_graph import PipelineConfig

    config_kwargs: dict[str, Any] = {
        "source": source,
        "backend": "llm",
        "inference": "local",
        "provider_override": _env_str("DOCLING_GRAPH_LLM_PROVIDER", "ollama"),
        "model_override": _env_str("DOCLING_GRAPH_LLM_MODEL", "granite3-dense:8b"),
        "extraction_contract": _env_str("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        "processing_mode": _env_str("DOCLING_GRAPH_PROCESSING_MODE", "many-to-one"),
        "use_chunking": _env_bool("DOCLING_GRAPH_USE_CHUNKING", True),
        "chunk_max_tokens": _env_int("DOCLING_GRAPH_CHUNK_MAX_TOKENS", 512),
        "llm_batch_token_size": _env_int("DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE", 2048),
        "parallel_workers": _env_int("DOCLING_GRAPH_PARALLEL_WORKERS", 2),
        "staged_pass_retries": _env_int("DOCLING_GRAPH_BATCH_SPLIT_MAX_RETRIES", 1),
        "delta_resolvers_enabled": _env_bool("DOCLING_GRAPH_RESOLVERS_ENABLED", True),
        "delta_resolvers_mode": _env_str("DOCLING_GRAPH_RESOLVERS_MODE", "semantic"),
        "delta_resolver_fuzzy_threshold": _env_float("DOCLING_GRAPH_RESOLVER_FUZZY_THRESHOLD", 0.8),
        "delta_resolver_semantic_threshold": _env_float("DOCLING_GRAPH_RESOLVER_SEMANTIC_THRESHOLD", 0.8),
        "delta_quality_require_root": _env_bool("DOCLING_GRAPH_QUALITY_REQUIRE_ROOT", True),
        "delta_quality_min_instances": _env_int("DOCLING_GRAPH_QUALITY_MIN_INSTANCES", 20),
        "delta_quality_max_parent_lookup_miss": _env_int("DOCLING_GRAPH_QUALITY_MAX_PARENT_MISS", 4),
        "delta_quality_adaptive_parent_lookup": _env_bool("DOCLING_GRAPH_QUALITY_ADAPTIVE_PARENT", True),
        "delta_normalizer_validate_paths": _env_bool("DOCLING_GRAPH_NORMALIZER_VALIDATE_PATHS", True),
        "delta_normalizer_canonicalize_ids": _env_bool("DOCLING_GRAPH_NORMALIZER_CANONICALIZE_IDS", True),
        "delta_normalizer_strip_nested_properties": _env_bool("DOCLING_GRAPH_NORMALIZER_STRIP_NESTED", True),
        "delta_normalizer_attach_provenance": _env_bool("DOCLING_GRAPH_NORMALIZER_ATTACH_PROVENANCE", True),
        "delta_identity_filter_enabled": _env_bool("DOCLING_GRAPH_IDENTITY_FILTER_ENABLED", True),
        "delta_identity_filter_strict": _env_bool("DOCLING_GRAPH_IDENTITY_FILTER_STRICT", False),
        "gleaning_enabled": _env_bool("DOCLING_GRAPH_GLEANING_ENABLED", True),
        "gleaning_max_passes": _env_int("DOCLING_GRAPH_GLEANING_MAX_PASSES", 1),
        "structured_output": _env_bool("DOCLING_GRAPH_STRUCTURED_OUTPUT", True),
        "structured_sparse_check": _env_bool("DOCLING_GRAPH_STRUCTURED_SPARSE_CHECK", True),
        "llm_overrides": {
            "generation": {
                "temperature": _env_float("DOCLING_GRAPH_LLM_TEMPERATURE", 0.1),
                "max_tokens": _env_int_or_none("DOCLING_GRAPH_LLM_MAX_TOKENS", 64000),
            },
            "reliability": {
                "timeout_s": _env_int("DOCLING_GRAPH_LLM_TIMEOUT", 10800),
            },
            "connection": {
                "base_url": _env_str("OLLAMA_LLM_BASE_URL", "http://ollama:11434"),
            },
            "context_limit": _env_int_or_none("DOCLING_GRAPH_LLM_CONTEXT_LIMIT", None),
            "max_output_tokens": _env_int_or_none("DOCLING_GRAPH_LLM_MAX_OUTPUT_TOKENS", None),
        },
        "dump_to_disk": False,
    }

    if template_class is not None:
        config_kwargs["template"] = template_class

    return PipelineConfig(**config_kwargs)
