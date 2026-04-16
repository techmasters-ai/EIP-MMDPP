"""Environment variables → PipelineConfig construction.

Uses pydantic_settings.BaseSettings for type-safe env var parsing.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class DoclingGraphSettings(BaseSettings):
    """Typed settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider and model
    docling_graph_llm_provider: str = "ollama"
    docling_graph_llm_model: str = "granite3-dense:8b"
    docling_graph_extraction_contract: str = "delta"
    docling_graph_processing_mode: str = "many-to-one"

    # Chunking
    docling_graph_use_chunking: bool = True
    docling_graph_chunk_max_tokens: int = 512
    docling_graph_llm_batch_token_size: int = 2048
    docling_graph_parallel_workers: int = 2
    docling_graph_batch_split_max_retries: int = 1

    # Delta resolvers
    docling_graph_resolvers_enabled: bool = True
    docling_graph_resolvers_mode: str = "semantic"
    docling_graph_resolver_fuzzy_threshold: float = 0.8
    docling_graph_resolver_semantic_threshold: float = 0.8

    # Delta quality
    docling_graph_quality_require_root: bool = True
    docling_graph_quality_min_instances: int = 3
    docling_graph_quality_max_parent_miss: int = 4
    docling_graph_quality_adaptive_parent: bool = True

    # Delta normalizer
    docling_graph_normalizer_validate_paths: bool = True
    docling_graph_normalizer_canonicalize_ids: bool = True
    docling_graph_normalizer_strip_nested: bool = True
    docling_graph_normalizer_attach_provenance: bool = True

    # Delta identity filter
    docling_graph_identity_filter_enabled: bool = True
    docling_graph_identity_filter_strict: bool = False

    # Gleaning
    docling_graph_gleaning_enabled: bool = True
    docling_graph_gleaning_max_passes: int = 1

    # Structured output
    docling_graph_structured_output: bool = True
    docling_graph_structured_sparse_check: bool = True

    # LLM overrides
    docling_graph_llm_temperature: float = 0.1
    docling_graph_llm_max_tokens: int | None = 64000
    docling_graph_llm_timeout: int = 10800
    ollama_llm_base_url: str = "http://ollama:11434"
    docling_graph_llm_context_limit: int | None = None
    docling_graph_llm_max_output_tokens: int | None = None

    # Backend: "llm" or "vlm"
    docling_graph_backend: str = "llm"


# Per-pass overrides for the quality gate. ``system_links`` is the
# relationships-only pass (Decision 4) — it legitimately produces zero
# ontology nodes on empty docs, so the default min_instances=3 would
# wrongly fail the gate. Spec §4.6.
_QUALITY_MIN_INSTANCES_PER_PASS: dict[str, int] = {
    "system_links": 1,
}


def build_pipeline_config(
    source: str,
    template_class: Type[BaseModel] | None,
    pass_name: str | None = None,
) -> Any:
    """Build a PipelineConfig from environment variables.

    ``pass_name`` applies per-pass quality-gate overrides (spec §4.6);
    see :data:`_QUALITY_MIN_INSTANCES_PER_PASS`. Unknown pass names
    fall back to the env-var default.
    """
    from docling_graph import PipelineConfig

    settings = DoclingGraphSettings()

    quality_min_instances = settings.docling_graph_quality_min_instances
    if pass_name in _QUALITY_MIN_INSTANCES_PER_PASS:
        quality_min_instances = _QUALITY_MIN_INSTANCES_PER_PASS[pass_name]

    config_kwargs: dict[str, Any] = {
        "source": source,
        "backend": settings.docling_graph_backend,
        "inference": "local",
        "provider_override": settings.docling_graph_llm_provider,
        "model_override": settings.docling_graph_llm_model,
        "extraction_contract": settings.docling_graph_extraction_contract,
        "processing_mode": settings.docling_graph_processing_mode,
        "use_chunking": settings.docling_graph_use_chunking,
        "chunk_max_tokens": settings.docling_graph_chunk_max_tokens,
        "llm_batch_token_size": settings.docling_graph_llm_batch_token_size,
        "parallel_workers": settings.docling_graph_parallel_workers,
        "staged_pass_retries": settings.docling_graph_batch_split_max_retries,
        "delta_resolvers_enabled": settings.docling_graph_resolvers_enabled,
        "delta_resolvers_mode": settings.docling_graph_resolvers_mode,
        "delta_resolver_fuzzy_threshold": settings.docling_graph_resolver_fuzzy_threshold,
        "delta_resolver_semantic_threshold": settings.docling_graph_resolver_semantic_threshold,
        "delta_quality_require_root": settings.docling_graph_quality_require_root,
        "delta_quality_min_instances": quality_min_instances,
        "delta_quality_max_parent_lookup_miss": settings.docling_graph_quality_max_parent_miss,
        "delta_quality_adaptive_parent_lookup": settings.docling_graph_quality_adaptive_parent,
        "delta_normalizer_validate_paths": settings.docling_graph_normalizer_validate_paths,
        "delta_normalizer_canonicalize_ids": settings.docling_graph_normalizer_canonicalize_ids,
        "delta_normalizer_strip_nested_properties": settings.docling_graph_normalizer_strip_nested,
        "delta_normalizer_attach_provenance": settings.docling_graph_normalizer_attach_provenance,
        "delta_identity_filter_enabled": settings.docling_graph_identity_filter_enabled,
        "delta_identity_filter_strict": settings.docling_graph_identity_filter_strict,
        "gleaning_enabled": settings.docling_graph_gleaning_enabled,
        "gleaning_max_passes": settings.docling_graph_gleaning_max_passes,
        "structured_output": settings.docling_graph_structured_output,
        "structured_sparse_check": settings.docling_graph_structured_sparse_check,
        "llm_overrides": {
            "generation": {
                "temperature": settings.docling_graph_llm_temperature,
                "max_tokens": settings.docling_graph_llm_max_tokens,
            },
            "reliability": {
                "timeout_s": settings.docling_graph_llm_timeout,
            },
            "connection": {
                "base_url": settings.ollama_llm_base_url,
            },
            "context_limit": settings.docling_graph_llm_context_limit,
            "max_output_tokens": settings.docling_graph_llm_max_output_tokens,
        },
        "dump_to_disk": False,
    }

    if template_class is not None:
        config_kwargs["template"] = template_class

    return PipelineConfig(**config_kwargs)
