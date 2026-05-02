"""Environment variables → PipelineConfig construction.

Uses pydantic_settings.BaseSettings for type-safe env var parsing.
"""

from __future__ import annotations

import json
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

    # LLM provider and model.
    # llama3.3:70b — larger-context local model, better recall than llama3.1:8b
    # The original default (granite3-dense:8b) isn't referenced in the docs;
    # the env-override value (llama3.3:70b) was stalling Ollama's constrained
    # decoder on our 14KB json_schema — 70B models are more prone to
    # indefinite spins on strict structured output than 8B-class models.
    docling_graph_llm_provider: str = "ollama"
    docling_graph_llm_model: str = "llama3.3:70b"
    docling_graph_extraction_contract: str = "delta"
    docling_graph_processing_mode: str = "many-to-one"

    # Chunking
    docling_graph_use_chunking: bool = True
    docling_graph_chunk_max_tokens: int = 512
    # Library default per docs §Extraction Backends → Provider-Specific Batching:
    # "Ollama/Local: Variable performance → conservative batching." 1024 is the
    # upstream default; our previous 2048 was aggressive for Ollama.
    docling_graph_llm_batch_token_size: int = 1024
    docling_graph_parallel_workers: int = 4
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

    # Gleaning.
    # Upstream treats gleaning_max_passes as a boolean gate in this code path:
    # one follow-up "what did you miss?" pass is what actually runs. Keep the
    # service setting boolean so the config does not imply N-pass behavior.
    docling_graph_gleaning_enabled: bool = True
    docling_graph_gleaning_max_passes: bool = True

    # Structured output
    docling_graph_structured_output: bool = True
    docling_graph_structured_sparse_check: bool = True
    # TODO #81: when True, /extract-pass blanks the text content of web-cruft
    # elements (ad-tracking URLs, navigation link lists, standalone bare URLs)
    # in texts[] before chunking. The element stays in place — only its
    # `text` / `orig` content is replaced with "" — so $refs from body /
    # pictures / tables / groups remain valid and the hierarchy validator
    # passes. The chunker treats empty texts as zero-token contributions
    # so they vanish from the LLM's view of the markdown. Image captions
    # (label='caption') are always preserved. Toggle via the env var
    # DOCLING_GRAPH_SANITIZE_INPUT (false to disable).
    docling_graph_sanitize_input: bool = True

    # LLM overrides.
    # Temperature=0.1 (library default) — tried temperature=0 per Ollama
    # structured-outputs guidance but it caused llama3.3:70b to deterministically
    # emit empty {"nodes":[],"relationships":[]} for our 40-path DeltaGraph
    # schema (Handwritten_Text.pdf extractions at 0.1 succeeded; all extractions
    # at 0.0 returned empty JSON). A small amount of variance is needed for the
    # model to explore valid completions under the structured-output constraint.
    docling_graph_llm_temperature: float = 0.1
    # Bound extraction generations so JSON-mode pathological chunks cannot
    # stream for tens of minutes. Typical per-batch graph outputs observed in
    # the recall harness are comfortably below this (largest seen ~1270
    # tokens). 4096 keeps first-attempt spin-pathology cost low while
    # legitimate truncations are recovered via bounded bumped retries in
    # _stream_with_truncation_retry. Empirically, content=0 truncations are
    # often recoverable, so they are retried; the 600s per-call wall-time cap
    # and orchestrator-level BATCH_HARD_TIMEOUT remain upper-bound safeguards.
    docling_graph_llm_max_tokens: int | None = 4096
    # Upper bound for bounded truncation recovery. The first call still uses
    # docling_graph_llm_max_tokens; default caps at the first 2x retry because
    # 16k retries added wall time without recall gain in Run 8. Raise this
    # only for targeted diagnostics.
    docling_graph_llm_truncation_retry_max_tokens: int | None = 8192
    docling_graph_llm_timeout: int = 1800  # 30 min per LLM call
    # Singular / fallback URLs (back-compat with existing .env files).
    ollama_base_url: str = "http://ollama:11434"
    ollama_llm_base_url: str = ""
    # Plural pool — raw JSON-array string. Parsed in get_ollama_llm_urls().
    ollama_llm_base_urls: str = ""
    # Per-function pool for graph extraction (Chunk 6, NEW). When set,
    # overrides OLLAMA_LLM_BASE_URLS for THIS service only — other
    # LLM-using functions (doc analysis, translation, etc.) consume
    # their own *_LLM_BASE_URLS via the api-side Settings.
    docling_graph_llm_base_urls: str = ""
    # For models missing from LiteLLM's metadata registry (e.g. ollama/llama3.3:70b)
    # the library's resolve_effective_model_config falls back to
    # _DEFAULT_MAX_OUTPUT_TOKENS=4092, then refuses any max_tokens above that.
    # We override explicitly so max_tokens is accepted. 131072 matches
    # llama3.3:70b's full context length (input + output combined).
    docling_graph_llm_context_limit: int | None = 131072
    docling_graph_llm_max_output_tokens: int | None = 4096

    # Backend: "llm" or "vlm"
    docling_graph_backend: str = "llm"

    def get_ollama_llm_urls(self) -> list[str]:
        """Parse priority (4-tier):
          docling_graph_llm_base_urls (function-specific JSON)
          > ollama_llm_base_urls (role-level JSON)
          > ollama_llm_base_url (singular)
          > ollama_base_url (base).
        Always returns a non-empty list.

        Error messages name the actual offending env var so an operator
        hitting a malformed JSON value can tell which variable to fix.
        """
        for env_name, raw in (
            ("DOCLING_GRAPH_LLM_BASE_URLS", self.docling_graph_llm_base_urls),
            ("OLLAMA_LLM_BASE_URLS", self.ollama_llm_base_urls),
        ):
            s = (raw or "").strip()
            if not s:
                continue
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{env_name} is not valid JSON: {exc}; got: {s!r}"
                ) from exc
            if not isinstance(parsed, list) or not all(
                isinstance(x, str) for x in parsed
            ):
                raise ValueError(
                    f"{env_name} must be a JSON array of strings; got: {parsed!r}"
                )
            if not all(x.strip() for x in parsed):
                raise ValueError(
                    f"{env_name} contains blank entries; got: {parsed!r}"
                )
            if parsed:
                return parsed
        if self.ollama_llm_base_url:
            return [self.ollama_llm_base_url]
        return [self.ollama_base_url]


# Per-pass overrides for the quality gate. ``system_links`` is the
# relationships-only pass (Decision 4) — it legitimately produces zero
# ontology nodes on empty docs, so the default min_instances=3 would
# wrongly fail the gate. Spec §4.6.
_QUALITY_MIN_INSTANCES_PER_PASS: dict[str, int] = {
    "system_links": 1,
}

# Per-pass overrides for the library's structured-output sparse-check (TODO #80).
# llm_backend.py:_is_sparse_structured_result computes
# `non_empty_values / schema_leaf_fields < 0.40` and triggers a 25-min
# legacy-mode retry. For identity-only passes (radar_identity, missile_identity)
# most fields are legitimately unknown for any given system — populating
# system_name + nomenclature only is the EXPECTED shape, not under-extraction.
# Disable the heuristic for those passes so a correct sparse output isn't
# wastefully retried.
_STRUCTURED_SPARSE_CHECK_PER_PASS: dict[str, bool] = {
    "radar_identity": False,
    "missile_identity": False,
}


def build_pipeline_config(
    source: str,
    template_class: Type[BaseModel] | None,
    pass_name: str | None = None,
    debug_dir: str | None = None,
    temperature_override: float | None = None,
    llm_batch_token_size_override: int | None = None,
) -> Any:
    """Build a PipelineConfig from environment variables.

    ``pass_name`` applies per-pass quality-gate overrides (spec §4.6);
    see :data:`_QUALITY_MIN_INSTANCES_PER_PASS`. Unknown pass names
    fall back to the env-var default.

    ``debug_dir`` opts into the library's debug trace: when set, the
    library writes ``<debug_dir>/debug/delta_trace.json`` containing the
    batch_errors, quality_gate verdict, identity_filter stats, etc. The
    extract_pass handler reads that file back and surfaces it in the
    response's ``diagnostics`` field.

    The local ``from app.ollama_clients import get_docling_graph_client``
    is wrapped in try/except ImportError so unit tests that exercise this
    function from the host venv still work. In the host venv, ``app``
    resolves to the api-side package, so the import fails — we leave
    ``llm_client`` unset and the library falls back to its own LiteLLMClient.
    Tests of this function only inspect config-dict shape, so the fallback
    is harmless. In-container the import always succeeds.
    """
    from docling_graph import PipelineConfig

    settings = DoclingGraphSettings()

    quality_min_instances = settings.docling_graph_quality_min_instances
    if pass_name in _QUALITY_MIN_INSTANCES_PER_PASS:
        quality_min_instances = _QUALITY_MIN_INSTANCES_PER_PASS[pass_name]

    # TODO #80: identity-only passes don't have relationships and most fields
    # are legitimately unknown — opt them out of the library's sparse-check
    # retry so we don't burn 25min on a "correct" output.
    structured_sparse_check_effective = settings.docling_graph_structured_sparse_check
    if pass_name in _STRUCTURED_SPARSE_CHECK_PER_PASS:
        structured_sparse_check_effective = _STRUCTURED_SPARSE_CHECK_PER_PASS[pass_name]

    # Build via the process-cached factory in app/ollama_clients.py. All
    # generation knobs (top_p / top_k / seed / stop / etc.), the schema
    # transform, force_json_mode, structured_output_threshold_chars, and
    # ClientError + parse_json_fn wiring live inside that factory — this
    # function is just a one-line consumer. provider_override / model_override
    # used to be threaded into config_kwargs too, but pipeline/stages.py:470
    # short-circuits on llm_client and the library's own provider/model
    # selection already lives inside the factory; tests pass without them.
    # See followups.md #23.
    try:
        from app.ollama_clients import get_docling_graph_client
        llm_client: Any | None = get_docling_graph_client()
        if temperature_override is not None and hasattr(
            llm_client, "with_runtime_defaults"
        ):
            llm_client = llm_client.with_runtime_defaults(
                temperature=temperature_override,
            )
    except ImportError:
        llm_client = None

    config_kwargs: dict[str, Any] = {
        "source": source,
        "backend": settings.docling_graph_backend,
        "inference": "local",
        "extraction_contract": settings.docling_graph_extraction_contract,
        "processing_mode": settings.docling_graph_processing_mode,
        "use_chunking": settings.docling_graph_use_chunking,
        "chunk_max_tokens": settings.docling_graph_chunk_max_tokens,
        "llm_batch_token_size": (
            llm_batch_token_size_override
            if llm_batch_token_size_override is not None
            else settings.docling_graph_llm_batch_token_size
        ),
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
        "gleaning_max_passes": 1 if settings.docling_graph_gleaning_max_passes else 0,
        "structured_output": settings.docling_graph_structured_output,
        "structured_sparse_check": structured_sparse_check_effective,
        "llm_overrides": {
            "generation": {
                "temperature": (
                    temperature_override
                    if temperature_override is not None
                    else settings.docling_graph_llm_temperature
                ),
                "max_tokens": settings.docling_graph_llm_max_tokens,
            },
            "reliability": {
                "timeout_s": settings.docling_graph_llm_timeout,
            },
            "context_limit": settings.docling_graph_llm_context_limit,
            "max_output_tokens": settings.docling_graph_llm_max_output_tokens,
        },
        "dump_to_disk": False,
    }

    if llm_client is not None:
        config_kwargs["llm_client"] = llm_client

    if template_class is not None:
        config_kwargs["template"] = template_class

    if debug_dir is not None:
        config_kwargs["debug"] = True
        config_kwargs["output_dir"] = debug_dir

    return PipelineConfig(**config_kwargs)
