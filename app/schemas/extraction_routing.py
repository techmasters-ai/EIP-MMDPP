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
    candidate_count: int  # post-vector, pre-rerank

    # Rerank stage
    rerank_score_range: tuple[float, float] | None = None

    # Final selection
    selected_ref_count: int
    selected_token_estimate: int
    full_doc_token_estimate: int

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


class ChunkScopeResponse(BaseModel):
    mode: Literal["selected_refs", "full", "would_skip"]
    self_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Empty for mode=full or mode=would_skip; "
            "populated only when mode=selected_refs"
        ),
    )
    diagnostics: ChunkScopeDiagnostics
