"""POST /v1/extraction/chunk-scope — VR Phase C.3 endpoint.

Called by the worker dispatcher in shadow/narrow_only mode. Worker passes
the request body; endpoint returns a ChunkScopeResponse the worker uses to
narrow (or not narrow) the pass's input doc.

Architectural rules:
  * NEVER use vector_search(filters=...) for the vector retrieval — that
    hits ArcadeDB's post-HNSW filter bug. Use search_extraction_chunks()
    which over-fetches and post-filters in Python. See C.1 commit bac8bd8
    + rev 11 in the plan revision history.
  * Reranker error → mode=full (NOT vector-only). Reranker is
    infrastructure degradation, not signal that chunks are irrelevant.
  * Embed query via loop.run_in_executor — embed_texts is sync, endpoint
    is async. Same pattern as /v1/retrieval (app/api/v1/retrieval.py:413-424).
  * chunk_text → content_text mapping at reranker call boundary (rev 9 M2).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from app.schemas.extraction_routing import (
    ChunkScopeDiagnostics,
    ChunkScopeRequest,
    ChunkScopeResponse,
)
from app.db.session import get_graph_store
from app.services.embedding import embed_texts
from app.services.extraction_chunk_search import search_extraction_chunks
from app.services.extraction_query_builder import build_retrieval_query
from app.services.ontology_bundles import load_bundle_manifest
from app.services.ontology_templates import UnknownBundleError
from app.services import reranker as rrk

if TYPE_CHECKING:
    from app.services.graph_store import GraphEntityResult

router = APIRouter(tags=["extraction"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_template_class(pass_def):
    """Import and return the pydantic Pass class for the given pass manifest entry.

    Imports pass_def.module and retrieves pass_def.template_class from it.
    Raises ImportError / AttributeError if the class cannot be found.
    """
    import importlib

    mod = importlib.import_module(pass_def.module)
    return getattr(mod, pass_def.template_class)


def _score_range(
    results: list,
) -> tuple[float, float] | None:
    """Return (min, max) of vector scores across results, or None if empty."""
    if not results:
        return None
    scores = [
        r.score
        for r in results
        if r.score is not None
    ]
    if not scores:
        return None
    return (min(scores), max(scores))


def _rerank_score_range(
    reranked: list[dict],
) -> tuple[float, float] | None:
    """Return (min, max) of reranker_score across reranked candidates, or None."""
    scores = [
        c["reranker_score"]
        for c in reranked
        if "reranker_score" in c
    ]
    if not scores:
        return None
    return (min(scores), max(scores))


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (BPE approximation)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_full_doc_tokens(pipeline_run_id: str, store) -> int:
    """Sum chunk_text lengths across all ExtractionChunk vertices for this run.

    This is a best-effort synchronous query; used only for the ratio warning
    and diagnostics. Returns 0 on any error.
    """
    try:
        # Use the store's synchronous interface to query all chunks for this run
        # via the graph store's fulltext / property lookup. We do a broad vector
        # search using a zero-vector probe to enumerate chunks, then sum their
        # chunk_text lengths.
        # NOTE: This is intentionally approximate — we just need a rough denominator
        # for the ratio check. Use a single large fetch (2000 chunks covers all
        # realistic production runs; see C.1 capacity notes).
        import asyncio as _asyncio

        # We can't run async from sync here without a dedicated loop. Use the
        # store's sync path if available, otherwise return 0 as safe fallback.
        if hasattr(store, "get_all_chunks_for_run_sync"):
            chunks = store.get_all_chunks_for_run_sync(pipeline_run_id)
            total_chars = sum(
                len(c.properties.get("chunk_text", "") or "")
                for c in chunks
            )
            return _estimate_tokens(" " * total_chars)  # chars → tokens

        # Fallback: 0 (ratio warning will not fire, which is the safe side)
        return 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Async full-doc token estimate (used inside the async endpoint)
# ---------------------------------------------------------------------------


async def _async_full_doc_token_estimate(
    pipeline_run_id: str,
    store,
    loop: asyncio.AbstractEventLoop,
) -> int:
    """Async version: over-fetch all ExtractionChunks for the run, sum chunk_text.

    Uses the same over-fetch + post-filter strategy as search_extraction_chunks
    but with a zero-length (dummy) query vector to enumerate all chunks rather
    than doing similarity search.

    Returns estimated token count, or 0 on error.
    """
    try:
        # Build a zero-vector probe of the correct dimensionality.
        # We only need to enumerate vertices; similarity doesn't matter.
        # 1024 is the bge-m3 dimension; zeros produce defined (if uninformative)
        # similarity scores but enumerate all vertices up to top_k.
        _PROBE_DIM = 1024
        probe = [0.0] * _PROBE_DIM

        raw = await store.vector_search(
            vertex_type="ExtractionChunk",
            embedding_property="embedding",
            query_vector=probe,
            top_k=2000,
            score_threshold=None,
            filters=None,
        )
        # Post-filter to this run only (same reason as search_extraction_chunks)
        run_chunks = [
            r for r in raw
            if r.properties.get("pipeline_run_id") == pipeline_run_id
        ]
        total_chars = sum(
            len(r.properties.get("chunk_text", "") or "")
            for r in run_chunks
        )
        return _estimate_tokens(" " * total_chars)
    except Exception as exc:
        logger.debug(
            "chunk-scope: full_doc_token_estimate failed for run=%s: %r",
            pipeline_run_id, exc,
        )
        return 0


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/extraction/chunk-scope", response_model=ChunkScopeResponse)
async def chunk_scope(
    body: ChunkScopeRequest,
) -> ChunkScopeResponse:
    """Compute the per-pass chunk scope via vector retrieval + cross-encoder
    rerank. Returns mode + selected refs (when narrowing) or mode=full /
    mode=would_skip (when no narrowing — worker decides action).

    Endpoint NEVER decides "skip" unilaterally; that is the worker's
    decision per VECTOR_ROUTER_MODE (C.4 wiring).
    """
    # 1. Load the bundle + find pass_def + retrieval profile
    try:
        manifest = load_bundle_manifest(body.bundle_key)
    except UnknownBundleError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    pass_def = next(
        (p for p in manifest.passes if p.name == body.pass_name), None
    )
    if pass_def is None:
        raise HTTPException(
            status_code=404,
            detail=f"pass_name {body.pass_name!r} not in bundle {body.bundle_key!r}",
        )

    # Defensive: identity / required / relationship passes have no retrieval block.
    # Worker C.4 should never route these here, but we handle it gracefully.
    if pass_def.retrieval is None:
        return ChunkScopeResponse(
            mode="full",
            self_refs=[],
            diagnostics=ChunkScopeDiagnostics(
                mode="full",
                fallback_reason="pass_not_routable",
                query_text="",
                vector_threshold=0.0,
                vector_score_range=None,
                candidate_count=0,
                rerank_score_range=None,
                selected_ref_count=0,
                selected_token_estimate=0,
                full_doc_token_estimate=0,
                would_skip_if_fallback_disabled=False,
                vector_search_ms=0,
                rerank_ms=0,
                ann_top_k_requested=0,
                post_filter_candidate_count=0,
                post_filter_retry_count=0,
                filter_strategy="overfetch_post_filter",
            ),
        )

    profile = pass_def.retrieval  # RetrievalProfile

    # 2. Build the query text from the pydantic schema
    template_cls = _resolve_template_class(pass_def)
    query_text = build_retrieval_query(pass_def, template_cls)

    # 3. Embed the query (sync function in executor — rev 10 M3)
    loop = asyncio.get_running_loop()

    def _embed() -> list[float]:
        return embed_texts([query_text], query=True)[0]

    query_vector: list[float] = await loop.run_in_executor(None, _embed)

    # 4. Vector retrieval via the over-fetch helper
    # NEVER use vector_search(filters=...) — ArcadeDB post-HNSW filter bug
    # (see C.1 module docstring + tests/integration/test_extraction_chunk_filter_starvation.py)
    store = get_graph_store()

    vector_t0 = time.monotonic()
    results, search_diag = await search_extraction_chunks(
        store=store,
        query_vector=query_vector,
        pipeline_run_id=body.pipeline_run_id,
        desired_top_n=profile.top_n_candidates,
        score_threshold=profile.min_similarity,
    )
    vector_search_ms = int((time.monotonic() - vector_t0) * 1000)

    # Pre-compute full-doc token estimate (needed by several return paths)
    full_doc_token_estimate = await _async_full_doc_token_estimate(
        body.pipeline_run_id, store, loop
    )

    # 5. Empty retrieval handling
    if not results:
        # Counterfactual (rev 10 M6): what WOULD have happened without fallback
        would_skip_if_fallback_disabled = True
        if profile.fallback_to_full:
            mode = "full"
            fallback_reason = "no_chunks_above_threshold"
        else:
            mode = "would_skip"
            fallback_reason = "no_chunks_above_threshold"

        return ChunkScopeResponse(
            mode=mode,
            self_refs=[],
            diagnostics=ChunkScopeDiagnostics(
                mode=mode,
                fallback_reason=fallback_reason,
                query_text=query_text,
                vector_threshold=profile.min_similarity,
                vector_score_range=None,
                candidate_count=0,
                rerank_score_range=None,
                selected_ref_count=0,
                selected_token_estimate=0,
                full_doc_token_estimate=full_doc_token_estimate,
                would_skip_if_fallback_disabled=would_skip_if_fallback_disabled,
                vector_search_ms=vector_search_ms,
                rerank_ms=0,
                ann_top_k_requested=search_diag.ann_top_k_requested,
                post_filter_candidate_count=search_diag.post_filter_candidate_count,
                post_filter_retry_count=search_diag.post_filter_retry_count,
                filter_strategy=search_diag.filter_strategy,
            ),
        )

    # 6. Rerank — chunk_text → content_text mapping (rev 9 M2)
    candidates_for_rerank = [
        {
            "content_text": r.properties.get("chunk_text", ""),
            "self_ref": r.properties.get("self_ref"),
            "vector_score": r.score if r.score is not None else None,
        }
        for r in results
    ]

    rerank_t0 = time.monotonic()
    try:
        def _rerank() -> list[dict]:
            return rrk.rerank(
                query=query_text,
                candidates=candidates_for_rerank,
                top_k=profile.top_k,
            )

        reranked: list[dict] = await loop.run_in_executor(None, _rerank)
        rerank_ms = int((time.monotonic() - rerank_t0) * 1000)
    except Exception as exc:
        # Rev 10 M6 rule: reranker error → mode=full (NOT vector-only ordering)
        rerank_ms = int((time.monotonic() - rerank_t0) * 1000)
        logger.warning(
            "chunk-scope: reranker error → fail open mode=full: %r", exc
        )
        return ChunkScopeResponse(
            mode="full",
            self_refs=[],
            diagnostics=ChunkScopeDiagnostics(
                mode="full",
                fallback_reason="reranker_unavailable",
                query_text=query_text,
                vector_threshold=profile.min_similarity,
                vector_score_range=_score_range(results),
                candidate_count=len(results),
                rerank_score_range=None,
                selected_ref_count=0,
                selected_token_estimate=0,
                full_doc_token_estimate=full_doc_token_estimate,
                would_skip_if_fallback_disabled=False,
                vector_search_ms=vector_search_ms,
                rerank_ms=rerank_ms,
                ann_top_k_requested=search_diag.ann_top_k_requested,
                post_filter_candidate_count=search_diag.post_filter_candidate_count,
                post_filter_retry_count=search_diag.post_filter_retry_count,
                filter_strategy=search_diag.filter_strategy,
            ),
        )

    # 7. Top-K by reranker_score → selected_refs
    # rerank() already returns top_k items; slice defensively in case caller
    # profile.top_k changed between the rerank call and here.
    top_k_results = reranked[: profile.top_k]
    selected_refs = [c["self_ref"] for c in top_k_results if c.get("self_ref")]
    selected_text_for_estimate = " ".join(
        c.get("content_text", "") for c in top_k_results
    )
    selected_token_estimate = _estimate_tokens(selected_text_for_estimate)

    # Rev 10 M8: narrowing-ineffective WARNING when ratio > 0.80
    if full_doc_token_estimate > 0 and (
        selected_token_estimate / full_doc_token_estimate
    ) > 0.80:
        logger.warning(
            "chunk-scope: narrowing INEFFECTIVE for pass=%s run=%s — "
            "selected_tokens=%d / full_doc_tokens=%d (%.0f%%); "
            "threshold may be too generous",
            body.pass_name,
            body.pipeline_run_id,
            selected_token_estimate,
            full_doc_token_estimate,
            100.0 * selected_token_estimate / full_doc_token_estimate,
        )

    return ChunkScopeResponse(
        mode="selected_refs",
        self_refs=selected_refs,
        diagnostics=ChunkScopeDiagnostics(
            mode="selected_refs",
            fallback_reason=None,
            query_text=query_text,
            vector_threshold=profile.min_similarity,
            vector_score_range=_score_range(results),
            candidate_count=len(results),
            rerank_score_range=_rerank_score_range(reranked),
            selected_ref_count=len(selected_refs),
            selected_token_estimate=selected_token_estimate,
            full_doc_token_estimate=full_doc_token_estimate,
            would_skip_if_fallback_disabled=False,
            vector_search_ms=vector_search_ms,
            rerank_ms=rerank_ms,
            ann_top_k_requested=search_diag.ann_top_k_requested,
            post_filter_candidate_count=search_diag.post_filter_candidate_count,
            post_filter_retry_count=search_diag.post_filter_retry_count,
            filter_strategy=search_diag.filter_strategy,
        ),
    )
