"""Request/response schemas for /v1/extraction/chunk-scope (VR Phase C.3).

Worker (C.4) HTTP-calls this endpoint per pass dispatch in narrow_only/shadow
mode. Endpoint returns one of three modes — worker decides what to do per
VECTOR_ROUTER_MODE.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChunkScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_run_id: str = Field(
        description="UUID; scopes which ExtractionChunk vertices to consider"
    )
    bundle_key: str = Field(
        description="e.g. 'air_defense_v3' or 'air_defense_v3_baseline_subset'"
    )
    pass_name: str = Field(description="The field_group pass name")


class ChunkScopeDiagnostics(BaseModel):
    """All diagnostics merged into pipeline_pass_outputs.diagnostics_json.router
    at worker terminal-save time (C.4 work)."""

    mode: Literal["selected_refs", "full", "would_skip"]
    fallback_reason: str | None = None  # "reranker_unavailable", "no_chunks_above_threshold", "would_skip_in_narrow_only_mode", etc.

    # Query construction
    query_text: str

    # Vector retrieval stage
    vector_threshold: float
    vector_score_range: tuple[float, float] | None = None
    candidate_count: int  # semantics vary by exit path: capped pool size in multi-channel error/early paths; post-C5 top_k slice in the multi-channel success path; raw vector-result count in per-element mode

    # Rerank stage
    rerank_score_range: tuple[float, float] | None = None

    # Final selection
    selected_ref_count: int
    selected_token_estimate: int = Field(
        description=(
            "Sum of bge-m3 token counts across selected per-element chunk texts "
            "in per_element mode. In merged-mode runs, equals "
            "selected_chunk_token_estimate; preserved for dashboard "
            "compatibility (Phase 1 Task 6)."
        ),
    )
    full_doc_token_estimate: int

    # Merged-mode diagnostics (Phase 1 Task 6). Default 0 keeps per_element-mode
    # responses byte-identical to the pre-Task-6 shape modulo the new field's
    # presence.
    selected_chunk_count: int = 0
    expanded_ref_count: int = 0
    selected_chunk_token_estimate: int = 0

    # Counterfactual (rev 10 M6) — what mode would have returned if fallback_to_full=false
    would_skip_if_fallback_disabled: bool

    # Per-stage timing (rev 8 M10)
    vector_search_ms: int
    rerank_ms: int

    # Retrieval-stage diagnostics (from ChunkSearchDiagnostics).
    # Field semantics depend on filter_strategy:
    #   "overfetch_post_filter" — HNSW + Python post-filter (legacy path).
    #   "direct_cosine"          — Path B (2026-05-26): per-run SQL pull +
    #                              numpy cosine. ann_top_k_requested is the
    #                              candidate pool size, post_filter_retry_count
    #                              is always 0, short_fetch means "not enough
    #                              matches in this run" (exact, not incomplete).
    # See app/services/extraction_chunk_search.py:ChunkSearchDiagnostics for
    # the full per-field semantics.
    ann_top_k_requested: int
    post_filter_candidate_count: int
    post_filter_retry_count: int
    filter_strategy: str  # "overfetch_post_filter" or "direct_cosine"

    # Short-fetch flag (rev 16 Important #2): propagated from ChunkSearchDiagnostics.
    # HNSW: True if post_filter_candidate_count < desired_top_n even after retry —
    # the result MAY be incomplete.
    # DIRECT: True if the run has fewer matching chunks than desired_top_n —
    # the result IS exact, just not enough matches exist.
    # mode stays selected_refs when short_fetch=True — diagnostic-only for v1.
    # Callers MUST inspect this field; future work can promote it to a fail-open
    # trigger if production data shows incomplete retrieval causes quality regressions.
    short_fetch: bool = False

    # Task C7 — multi-channel diagnostics.
    # All default None; only populated on the multi-channel success path.
    # Per-element paths and all error/early-return paths leave these None.
    channel_counts: dict[str, int] | None = None
    # {"dense": N, "field:<field_name>": M, "lexical": L, "pattern": P}
    # Counts from the MultiChannelDiagnostics per-channel fields.

    field_coverage: dict[str, int] | None = None
    # {field_name: # candidates with evidence for that field}
    # Derived from the merged pool's supported_field_hints.

    score_components: list[dict] | None = None
    # One dict per SELECTED candidate (capped at top_k).
    # Keys: candidate_key, reranker_score (or null), alias_hits, pattern_hits,
    #       negative_hits, section_hits, retrieval_sources (sorted list), final_score.

    fallback_level: str | None = None
    # "none" on the normal multi-channel success path.
    # Phase E will populate "relaxed_dense" | "lexical_table" | "identity_anchor" | "full".


class SelectedChunk(BaseModel):
    """Router-selected merged chunk (Phase 1 Task 6).

    Populated only when ``settings.extraction_index_mode == "merged"``.
    Phase 2 worker reads ``.text`` directly and forwards to docling-graph
    via the chunked-extract path (``selected_chunks=[...]``).

    ``chunk_key`` has no Phase-1 consumer — ``apply_chunk_scope`` does not
    read it and the Phase-1 LLM-batch path goes through the existing
    self_ref-keyed pipeline. It is populated strictly as forward-compat
    payload pre-staging for Phase 2's ``/extract-pass`` chunked-mode
    dispatch (see plan rev 5 Task 6 + Glossary).
    """

    chunk_index: int = Field(
        description="Dense int per pipeline_run_id (see plan Glossary)."
    )
    chunk_key: str = Field(
        description=(
            "``f'chunk_{chunk_index}'``; Phase-2 payload pre-staging only — "
            "no Phase-1 consumer."
        ),
    )
    text: str = Field(
        description="Merged chunk text (``chunker.contextualize`` output)."
    )
    source_refs: list[str] = Field(
        description="Element self_refs covered by this merged chunk."
    )
    token_count: int = Field(
        description="``tokenizer.count_tokens(text)`` for this chunk."
    )


class ChunkScopeResponse(BaseModel):
    mode: Literal["selected_refs", "full", "would_skip"]
    self_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Empty for mode=full or mode=would_skip; "
            "populated only when mode=selected_refs"
        ),
    )
    text_by_ref: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Post-filter ExtractionChunk.chunk_text keyed by self_ref for the "
            "selected refs. The worker uses this to make scoped DoclingDocument "
            "content match the retrieval/rerank evidence text. "
            "UNCHANGED in merged mode — merged chunk text rides on "
            "``selected_chunks[*].text``, not here."
        ),
    )
    selected_chunks: list[SelectedChunk] | None = Field(
        default=None,
        description=(
            "Populated only when ``settings.extraction_index_mode == 'merged'`` "
            "AND ``mode == 'selected_refs'``. ``None`` otherwise. "
            "Each entry corresponds to one merged HybridChunker chunk "
            "selected by the rerank top-K; order is chunk-encounter order "
            "(bge-m3 + reranker rank), NOT lex-sorted by chunk_index. "
            "Phase-2 worker dispatch consumes ``selected_chunks[*].text`` "
            "as pre-built LLM batches."
        ),
    )
    diagnostics: ChunkScopeDiagnostics
