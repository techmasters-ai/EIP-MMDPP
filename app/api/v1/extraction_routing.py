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
    SelectedChunk,
)
from app.db.session import get_graph_store
from app.config import get_settings
from app.services.embedding import embed_texts
from app.services.extraction_chunk_index import (
    read_chunk_index,
    read_chunk_source_refs,
    read_chunk_token_count,
)
from app.services.extraction_chunk_search import search_extraction_chunks
from app.services.extraction_query_builder import build_retrieval_query
from app.services.ontology_bundles import load_bundle_manifest
from app.services.ontology_templates import UnknownBundleError
from app.services import reranker as rrk
from app.services.table_normalization.tokens import count_bge_m3_tokens

if TYPE_CHECKING:
    from app.services.graph_store import GraphEntityResult

router = APIRouter(tags=["extraction"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_template_class(bundle_key: str, pass_def) -> type:
    """Import and return the pydantic Pass class for the given pass manifest entry.

    Manifest module paths are RELATIVE (e.g. 'extraction_schemas.radar_power_rf').
    The worker (pipeline.py:_parse_pass_response, line ~3603) prefixes these with
    'ontology_bundles.{bundle_key}.' before importing. This endpoint must do the same
    — otherwise template resolution fails in production and silently returns mode=full
    via the template_resolution_error fallback (VR never narrows).

    Raises ImportError / AttributeError if the class cannot be found.
    """
    import importlib

    full_module = f"ontology_bundles.{bundle_key}.{pass_def.module}"
    mod = importlib.import_module(full_module)
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


def _estimate_tokens_from_chars(total_chars: int) -> int:
    """Char-count to token estimate without allocating an intermediate string.

    Uses 3 chars/token for technical text — empirically measured on SA-2's
    C.7g run: 39445 chars / 12839 bge-m3 tokens = 3.07 chars/token. The prior
    chars//4 ratio undercounted by ~30%, producing ratios >100% when compared
    against the bge-m3-counted selected_token_estimate.
    """
    return max(1, total_chars // 3) if total_chars > 0 else 0


# ---------------------------------------------------------------------------
# Async full-doc token estimate (used inside the async endpoint)
# ---------------------------------------------------------------------------


async def _async_full_doc_token_estimate(
    pipeline_run_id: str,
    store,
) -> int:
    """Estimate total token count for a pipeline_run's ExtractionChunk rows.

    Rev 17 LOW: switched from zero-vector HNSW probe (which had the same
    filter-starvation shape as the C.1 bug — see bac8bd8 + extraction_routing
    docstring) to a direct non-vector SQL query.  Queries ExtractionChunk by
    pipeline_run_id on the b-tree index (created in C.1 schema migration);
    no HNSW, no top_k cap, no starvation risk.

    Returns estimated token count, or 0 on error (disables the 80%
    narrowing-ineffective warning for that call).
    """
    try:
        # Direct SELECT sum(chunk_text.size()) by the indexed pipeline_run_id
        # property — no vector search involved.  ArcadeDB returns all matching
        # ExtractionChunk vertices for this run without a top_k cap.
        rows = await store._client.query(
            store._database,
            "sql",
            "SELECT sum(chunk_text.size()) AS total_chars "
            "FROM ExtractionChunk WHERE pipeline_run_id = :run_id",
            {"run_id": pipeline_run_id},
        )
        if not rows:
            logger.debug(
                "chunk-scope: full_doc_token_estimate SQL returned no rows for run=%s; "
                "returning 0 (disables narrowing-ineffective warning)",
                pipeline_run_id,
            )
            return 0
        total_chars = rows[0].get("total_chars") or 0
        return _estimate_tokens_from_chars(int(total_chars))
    except Exception as exc:
        logger.debug(
            "chunk-scope: full_doc_token_estimate query failed for run=%s: %r; "
            "returning 0 (disables narrowing-ineffective warning)",
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

    # 2. Build the query text from the pydantic schema.
    # Important #1 (rev 16): wrap in try/except — a bad template_class string in
    # the manifest (ImportError / AttributeError) or a build_retrieval_query
    # failure must return a structured response, not a bare 500.
    try:
        template_cls = _resolve_template_class(body.bundle_key, pass_def)
        query_text = build_retrieval_query(pass_def, template_cls)
    except Exception as exc:
        logger.warning(
            "chunk-scope: template resolution failed for pass=%s: %r — "
            "failing open to mode=full",
            body.pass_name, exc,
        )
        return ChunkScopeResponse(
            mode="full",
            self_refs=[],
            diagnostics=ChunkScopeDiagnostics(
                mode="full",
                fallback_reason="template_resolution_error",
                query_text="",
                vector_threshold=profile.min_similarity,
                candidate_count=0,
                selected_ref_count=0,
                selected_token_estimate=0,
                full_doc_token_estimate=0,  # skip ArcadeDB probe when template failed
                would_skip_if_fallback_disabled=False,
                vector_search_ms=0,
                rerank_ms=0,
                ann_top_k_requested=0,
                post_filter_candidate_count=0,
                post_filter_retry_count=0,
                filter_strategy="overfetch_post_filter",
                short_fetch=False,
            ),
        )

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
        body.pipeline_run_id, store
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
                short_fetch=search_diag.short_fetch,  # Important #2 (rev 16)
            ),
        )

    # 6. Rerank — chunk_text → content_text mapping (rev 9 M2)
    # Phase 1 Task 6: also carry chunk_index / source_refs / token_count
    # so the merged-mode return path can expand source_refs without an
    # extra DB round-trip. The reranker passes unknown keys through
    # (see app/services/reranker.py:rerank — it does dict(candidate)).
    candidates_for_rerank = [
        {
            "content_text": r.properties.get("chunk_text", ""),
            "self_ref": r.properties.get("self_ref"),
            "vector_score": r.score if r.score is not None else None,
            # Merged-mode fields — Task 1 accessors coalesce legacy/missing
            # to safe defaults (-1, [], 0).
            "chunk_index": read_chunk_index(r.properties),
            "source_refs": read_chunk_source_refs(r.properties),
            "token_count": read_chunk_token_count(r.properties),
        }
        for r in results
    ]

    # Minor #6 (rev 16): detect reranker_disabled BEFORE calling rerank so
    # fallback_reason is always explicit.  rrk.rerank() already short-circuits
    # when settings.reranker_enabled=False and returns candidates unchanged, but
    # that leaves fallback_reason=None in the response — diagnostics consumers
    # cannot distinguish a successful cross-encoder run from a disabled one.
    _reranker_enabled = get_settings().reranker_enabled
    _reranker_fallback_reason: str | None = None if _reranker_enabled else "reranker_disabled"

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
                short_fetch=search_diag.short_fetch,  # Important #2 (rev 16)
            ),
        )

    # 7. Top-K by reranker_score → selected_refs
    # rerank() already returns top_k items; slice defensively in case caller
    # profile.top_k changed between the rerank call and here.
    top_k_results = reranked[: profile.top_k]
    selected_refs = [c["self_ref"] for c in top_k_results if c.get("self_ref")]
    text_by_ref: dict[str, str] = {}
    for c in top_k_results:
        self_ref = c.get("self_ref")
        content_text = c.get("content_text")
        if isinstance(self_ref, str) and isinstance(content_text, str) and content_text.strip():
            text_by_ref.setdefault(self_ref, content_text)

    # Rev 17 MED: guard against contract-invalid mode=selected_refs with empty
    # self_refs list.  This can occur when rerank returns candidates that all
    # lack a self_ref key, or when top_k slicing yields zero items.  C.4 would
    # attempt to narrow to an empty scoped doc, which is incorrect.  Fail open
    # to mode=full instead so the pass runs against the full document.
    if not selected_refs:
        logger.warning(
            "chunk-scope: rerank produced top_k_results=%d but 0 valid self_refs for "
            "pass=%s run=%s — failing open to mode=full",
            len(top_k_results), body.pass_name, body.pipeline_run_id,
        )
        return ChunkScopeResponse(
            mode="full",
            self_refs=[],
            diagnostics=ChunkScopeDiagnostics(
                mode="full",
                fallback_reason="no_selected_refs_after_rerank",
                query_text=query_text,
                vector_threshold=profile.min_similarity,
                vector_score_range=_score_range(results),
                candidate_count=len(results),
                rerank_score_range=_rerank_score_range(reranked),
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
                short_fetch=search_diag.short_fetch,
            ),
        )

    # Minor #3 (rev 16): use bge-m3 tokenizer instead of len/4 heuristic.
    # Technical text (frequencies, units, abbreviations) has ~2-3 chars/token
    # for bge-m3; the heuristic undercounts by ~30-50%, causing the 80% ratio
    # warning to fire less often than intended.
    selected_text_for_estimate = " ".join(
        c.get("content_text", "") for c in top_k_results
    )
    selected_token_estimate = count_bge_m3_tokens(selected_text_for_estimate)

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

    # Important #2 (rev 16): short_fetch from search_diag must be surfaced.
    # Mode stays selected_refs — short_fetch is diagnostic-only for v1.
    if search_diag.short_fetch:
        logger.warning(
            "chunk-scope: short-fetch on pass=%s run=%s "
            "(post_filter_candidate_count=%d < desired_top_n=%d); "
            "downstream may see incomplete retrieval",
            body.pass_name, body.pipeline_run_id,
            search_diag.post_filter_candidate_count, profile.top_n_candidates,
        )

    # ---------------------------------------------------------------
    # Phase 1 Task 6: merged-mode expansion path.
    # Gated on ``settings.extraction_index_mode == "merged"`` so per_element
    # mode (the default) stays byte-identical to the pre-Task-6 response.
    # ---------------------------------------------------------------
    selected_chunks_out: list[SelectedChunk] | None = None
    selected_chunk_count = 0
    expanded_ref_count = 0
    selected_chunk_token_estimate = 0
    final_self_refs = selected_refs
    final_selected_ref_count = len(selected_refs)
    final_selected_token_estimate = selected_token_estimate

    if get_settings().extraction_index_mode == "merged":
        # Expand the per-chunk source_refs into a deduplicated list in
        # chunk-encounter order (rerank rank order — do NOT lex-sort;
        # '#/texts/100' would otherwise precede '#/texts/35').
        expanded_refs: list[str] = []
        seen: set[str] = set()
        sc_list: list[SelectedChunk] = []
        for chunk_row in top_k_results:
            refs_for_chunk = chunk_row.get("source_refs") or []
            if not isinstance(refs_for_chunk, list):
                refs_for_chunk = []
            chunk_idx = chunk_row.get("chunk_index", -1)
            if not isinstance(chunk_idx, int):
                try:
                    chunk_idx = int(chunk_idx)
                except (TypeError, ValueError):
                    chunk_idx = -1
            tok_count = chunk_row.get("token_count", 0)
            if not isinstance(tok_count, int):
                try:
                    tok_count = int(tok_count)
                except (TypeError, ValueError):
                    tok_count = 0
            for ref in refs_for_chunk:
                if isinstance(ref, str) and ref not in seen:
                    seen.add(ref)
                    expanded_refs.append(ref)
            # ``content_text`` was sourced from the carve-out column
            # ``chunk_text`` (always non-null on every ExtractionChunk row)
            # so the byte-identity contract is preserved.
            sc_list.append(
                SelectedChunk(
                    chunk_index=chunk_idx,
                    chunk_key=f"chunk_{chunk_idx}",
                    text=chunk_row.get("content_text", ""),
                    source_refs=[str(r) for r in refs_for_chunk if isinstance(r, str)],
                    token_count=tok_count,
                )
            )

        selected_chunks_out = sc_list
        selected_chunk_count = len(sc_list)
        expanded_ref_count = len(expanded_refs)
        selected_chunk_token_estimate = sum(sc.token_count for sc in sc_list)
        # text_by_ref stays UNCHANGED — still self_ref-keyed for
        # apply_chunk_scope. Merged chunk text rides on SelectedChunk.text.
        final_self_refs = expanded_refs
        final_selected_ref_count = expanded_ref_count
        # Legacy field mirrors the chunk-based estimate so existing
        # dashboards keep working (plan rev 5 Task 6 backward-compat).
        final_selected_token_estimate = selected_chunk_token_estimate

    return ChunkScopeResponse(
        mode="selected_refs",
        self_refs=final_self_refs,
        text_by_ref=text_by_ref,
        selected_chunks=selected_chunks_out,
        diagnostics=ChunkScopeDiagnostics(
            mode="selected_refs",
            fallback_reason=_reranker_fallback_reason,  # Minor #6 (rev 16)
            query_text=query_text,
            vector_threshold=profile.min_similarity,
            vector_score_range=_score_range(results),
            candidate_count=len(results),
            rerank_score_range=_rerank_score_range(reranked),
            selected_ref_count=final_selected_ref_count,
            selected_token_estimate=final_selected_token_estimate,
            full_doc_token_estimate=full_doc_token_estimate,
            would_skip_if_fallback_disabled=False,
            vector_search_ms=vector_search_ms,
            rerank_ms=rerank_ms,
            ann_top_k_requested=search_diag.ann_top_k_requested,
            post_filter_candidate_count=search_diag.post_filter_candidate_count,
            post_filter_retry_count=search_diag.post_filter_retry_count,
            filter_strategy=search_diag.filter_strategy,
            short_fetch=search_diag.short_fetch,  # Important #2 (rev 16)
            # Task 6 merged-mode diagnostics (0 in per_element mode).
            selected_chunk_count=selected_chunk_count,
            expanded_ref_count=expanded_ref_count,
            selected_chunk_token_estimate=selected_chunk_token_estimate,
        ),
    )
