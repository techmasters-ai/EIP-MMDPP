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
from app.services.extraction_candidate_scoring import active_fields as _active_fields
from app.db.session import get_graph_store
from app.config import get_settings
from app.services.embedding import embed_texts
from app.services.extraction_chunk_index import (
    read_chunk_index,
    read_chunk_source_refs,
    read_chunk_token_count,
)
from app.services.extraction_chunk_search import (
    build_pool_from_multi_channel_state,
    identity_anchor_queries,
    search_extraction_chunks,
    search_extraction_chunks_multi_channel,  # kept for backward-compat patches in tests
    search_extraction_chunks_multi_channel_full,
)
from app.services.extraction_candidate_scoring import (
    MergedCandidate,
    enough_candidates,
    enough_field_coverage,
    field_coverage,
    score_candidates,
)
from app.services.extraction_query_builder import (
    build_retrieval_profile,
)
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


def _build_lexical_table_candidates(
    pool_keys: "set[str]",
    lexical_hits: "dict[str, dict]",
    pattern_hits: "dict[str, dict]",
    rows_by_key: "dict[str, dict]",
) -> "list[MergedCandidate]":
    """Build MergedCandidate objects for keys that have lexical/pattern hits but
    are ABSENT from the dense merged pool (keyword-only recall net).

    This is the ``lexical_table`` fallback level (E2).  C4's primary
    ``merge_candidates`` deliberately drops keyword-only hits because keyword
    search is precision-not-recall.  This helper ONLY fires when the E2 ladder
    decides coverage is insufficient after relaxed_dense.

    Parameters
    ----------
    pool_keys:
        Set of candidate_keys already present in the dense pool.  Keys in this
        set are skipped — they already have a MergedCandidate and must not be
        duplicated.
    lexical_hits:
        C2 lexical hit counts keyed by candidate_key.
    pattern_hits:
        C3 pattern hit counts keyed by candidate_key.
    rows_by_key:
        Raw ExtractionChunk row dicts keyed by candidate_key (vertex_id or self_ref).
        Used to populate MergedCandidate fields.

    Returns
    -------
    list[MergedCandidate]
        Keyword-only candidates sorted by alias_hits + pattern_hits descending.
        Empty list when no keyword-only keys exist.
    """
    # Collect all candidate_keys from lexical/pattern hits
    kw_keys: set[str] = set()
    for k, v in lexical_hits.items():
        if v.get("alias_hits", 0) > 0 and k not in pool_keys:
            kw_keys.add(k)
    for k, v in pattern_hits.items():
        if v.get("pattern_hits", 0) > 0 and k not in pool_keys:
            kw_keys.add(k)

    result: list[MergedCandidate] = []
    for key in kw_keys:
        row = rows_by_key.get(key)
        if row is None:
            # No row data available — skip (cannot build a valid MergedCandidate)
            continue

        lh = lexical_hits.get(key, {})
        ph = pattern_hits.get(key, {})
        alias_hits = lh.get("alias_hits", 0)
        negative_hits = lh.get("negative_hits", 0)
        pattern_hit_count = ph.get("pattern_hits", 0)
        supported_fields: set[str] = set()
        supported_fields.update(lh.get("supported_fields", set()))
        supported_fields.update(ph.get("supported_fields", set()))

        retrieval_sources: set[str] = set()
        if alias_hits > 0:
            retrieval_sources.add("lexical")
        if pattern_hit_count > 0:
            retrieval_sources.add("pattern")

        result.append(
            MergedCandidate(
                candidate_key=key,
                chunk_index=read_chunk_index(row),
                self_ref=row.get("self_ref", ""),
                chunk_text=row.get("chunk_text", ""),
                source_refs=read_chunk_source_refs(row),
                token_count=read_chunk_token_count(row),
                page_number=row.get("page_number"),
                vector_score=None,
                field_scores={},
                alias_hits=alias_hits,
                pattern_hits=pattern_hit_count,
                negative_hits=negative_hits,
                section_hits=0,
                content_type=None,
                retrieval_sources=retrieval_sources,
                supported_field_hints=supported_fields,
            )
        )

    # Sort by total lexical+pattern signal descending (best keyword hits first)
    result.sort(key=lambda mc: -(mc.alias_hits + mc.pattern_hits))
    return result


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
    #
    # C6: In merged mode we also build the structured PassRetrievalSignals
    # (via build_retrieval_profile) so the multi-channel orchestrator gets
    # per-field queries, aliases, and evidence patterns.  Variable naming:
    #   signals = PassRetrievalSignals  (structured; used by multi-channel path)
    #   profile = RetrievalProfile      (config from pass_def.retrieval; UNCHANGED)
    # NEVER name a variable "retrieval_profile" — see C6 spec naming constraint.
    try:
        template_cls = _resolve_template_class(body.bundle_key, pass_def)
        signals = build_retrieval_profile(pass_def, template_cls)
        query_text = signals.entity_query  # byte-identical to build_retrieval_query output
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

    loop = asyncio.get_running_loop()
    store = get_graph_store()

    # -----------------------------------------------------------------------
    # C6 BRANCH: merged mode with multi-channel retrieval (C1–C4 orchestrator)
    # vs. per-element mode (existing single-vector path).
    #
    # Condition: extraction_index_mode == "merged" AND field_query_top_k > 0.
    # When False: per-element path — byte-identical behavior to pre-C6 code.
    # When True: multi-channel path — ONE per-run SELECT, ZERO HNSW/vector_search.
    # -----------------------------------------------------------------------
    settings = get_settings()
    # Defensive: field_query_top_k may be absent on legacy or test-only profile
    # objects that haven't been updated for C6.  RetrievalProfile always sets
    # this as an int (ge=0 validator); only mock objects may be missing it.
    # isinstance() check ensures we ONLY enter multi-channel for real int values
    # (not MagicMock, which would otherwise compare truthy).
    _fqtk = getattr(profile, "field_query_top_k", 0)
    _use_multi_channel = (
        settings.extraction_index_mode == "merged"
        and isinstance(_fqtk, int)
        and _fqtk > 0
    )

    # C8 identity-anchor count — set in multi-channel path; None in per-element.
    _c8_anchor_count: "int | None" = None

    if _use_multi_channel:
        # ------------------------------------------------------------------
        # MERGED MODE — multi-channel retrieval (C1+C2+C3+C4 orchestrator).
        #
        # Constraints enforced by search_extraction_chunks_multi_channel:
        #   - Exactly ONE per-run SELECT (B-tree-indexed on pipeline_run_id)
        #   - ZERO HNSW / vector_search calls
        #   - Pool capped to profile.top_n_candidates BEFORE this call returns
        #     (Blocker 6: cap is BEFORE rerank)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # C8 — identity-anchor channel (OPPORTUNISTIC, NON-BLOCKING).
        #
        # Derive identity_types from the bundle manifest's identity passes'
        # primary_entity_types.  Never hardcode type names here — the manifest
        # is the single source of truth for which types are identity-scoped.
        # Wrap everything in try/except so ANY error is silently swallowed and
        # the channel simply fires with empty anchors (clean no-op).
        # ------------------------------------------------------------------
        _c8_identity_types: list[str] = []
        try:
            _c8_identity_types = [
                etype
                for p in manifest.passes
                if p.phase == "identity"
                for etype in (p.primary_entity_types or [])
                if etype
            ]
            # Deduplicate while preserving order
            _seen_types: set[str] = set()
            _c8_identity_types_dedup: list[str] = []
            for t in _c8_identity_types:
                if t not in _seen_types:
                    _seen_types.add(t)
                    _c8_identity_types_dedup.append(t)
            _c8_identity_types = _c8_identity_types_dedup
        except Exception as exc:
            logger.debug(
                "chunk-scope C8: could not derive identity_types from manifest for "
                "pass=%s: %r — anchor channel will no-op",
                body.pass_name, exc,
            )

        _c8_anchors: list[str] = await identity_anchor_queries(
            store=store,
            pipeline_run_id=body.pipeline_run_id,
            identity_types=_c8_identity_types,
            worker_anchors=body.identity_anchors,
        )
        _c8_anchor_count: int = len(_c8_anchors)

        vector_t0 = time.monotonic()
        pool, mc_search_diag, _mc_state = await search_extraction_chunks_multi_channel_full(
            signals,
            body.pipeline_run_id,
            profile,
            store=store,
            identity_anchors=_c8_anchors if _c8_anchors else None,
        )
        vector_search_ms = int((time.monotonic() - vector_t0) * 1000)

        # Translate MultiChannelDiagnostics → ChunkSearchDiagnostics-compatible
        # values for the shared diagnostics fields below.
        _mc_ann_top_k = mc_search_diag.raw_row_count
        _mc_post_filter_count = mc_search_diag.pool_size
        _mc_filter_strategy = mc_search_diag.filter_strategy

        # Pre-compute full-doc token estimate
        full_doc_token_estimate = await _async_full_doc_token_estimate(
            body.pipeline_run_id, store
        )

        # Empty pool handling
        if not pool:
            would_skip_if_fallback_disabled = True
            if profile.fallback_to_full:
                _empty_mode = "full"
                _empty_fallback = "no_chunks_above_threshold"
            else:
                _empty_mode = "would_skip"
                _empty_fallback = "no_chunks_above_threshold"
            return ChunkScopeResponse(
                mode=_empty_mode,
                self_refs=[],
                diagnostics=ChunkScopeDiagnostics(
                    mode=_empty_mode,
                    fallback_reason=_empty_fallback,
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
                    ann_top_k_requested=_mc_ann_top_k,
                    post_filter_candidate_count=_mc_post_filter_count,
                    post_filter_retry_count=0,
                    filter_strategy=_mc_filter_strategy,
                    short_fetch=False,
                ),
            )

        # Build rerank-input dicts from MergedCandidates.
        # Each dict carries:
        #   "content_text"          — chunk_text (reranker compat key, rev 9 M2)
        #   "self_ref"              — merged chunk self_ref (e.g. "chunk_0")
        #   "vector_score"          — best dense score from C1
        #   "chunk_index"           — merged chunk index
        #   "source_refs"           — constituent element refs (for expansion)
        #   "token_count"           — for selected_token_estimate
        #   "page_number"           — lineage
        #   C4 signal fields        — alias_hits, pattern_hits, negative_hits,
        #                             section_hits, content_type,
        #                             retrieval_sources, supported_field_hints
        #   "merged_candidate"      — back-ref for C5 score_candidates contract
        pool_dicts: list[dict] = [
            {
                "content_text": mc.chunk_text,
                "self_ref": mc.self_ref,
                "vector_score": mc.vector_score,
                "chunk_index": mc.chunk_index,
                "source_refs": mc.source_refs,
                "token_count": mc.token_count,
                "page_number": mc.page_number,
                "alias_hits": mc.alias_hits,
                "pattern_hits": mc.pattern_hits,
                "negative_hits": mc.negative_hits,
                "section_hits": mc.section_hits,
                "content_type": mc.content_type,
                "retrieval_sources": mc.retrieval_sources,
                "supported_field_hints": mc.supported_field_hints,
                "merged_candidate": mc,
            }
            for mc in pool
        ]

        # Detect reranker disabled state (Minor #6 rev 16 pattern)
        _reranker_enabled = settings.reranker_enabled
        _reranker_fallback_reason: str | None = (
            None if _reranker_enabled else "reranker_disabled"
        )

        rerank_t0 = time.monotonic()
        try:
            # Rerank the ENTIRE capped pool — top_k=len(pool_dicts) so every
            # scoreable candidate gets a reranker_score. The top_k cut happens
            # AFTER C5 scoring (post-rerank boost), not here.
            def _rerank_multi() -> list[dict]:
                return rrk.rerank(
                    query=query_text,
                    candidates=pool_dicts,
                    top_k=len(pool_dicts),
                )

            reranked_pool: list[dict] = await loop.run_in_executor(None, _rerank_multi)
            rerank_ms = int((time.monotonic() - rerank_t0) * 1000)
        except Exception as exc:
            rerank_ms = int((time.monotonic() - rerank_t0) * 1000)
            logger.warning(
                "chunk-scope multi-channel: reranker error → fail open mode=full: %r", exc
            )
            return ChunkScopeResponse(
                mode="full",
                self_refs=[],
                diagnostics=ChunkScopeDiagnostics(
                    mode="full",
                    fallback_reason="reranker_unavailable",
                    query_text=query_text,
                    vector_threshold=profile.min_similarity,
                    vector_score_range=None,
                    candidate_count=len(pool),
                    rerank_score_range=None,
                    selected_ref_count=0,
                    selected_token_estimate=0,
                    full_doc_token_estimate=full_doc_token_estimate,
                    would_skip_if_fallback_disabled=False,
                    vector_search_ms=vector_search_ms,
                    rerank_ms=rerank_ms,
                    ann_top_k_requested=_mc_ann_top_k,
                    post_filter_candidate_count=_mc_post_filter_count,
                    post_filter_retry_count=0,
                    filter_strategy=_mc_filter_strategy,
                    short_fetch=False,
                ),
            )

        # C5 post-rerank precision boost: re-score + re-sort; then take top_k.
        # score_candidates returns list[(MergedCandidate, final_score)] sorted
        # by final desc → reranker_score desc → candidate_key asc.
        c5_scored = score_candidates(reranked_pool, profile)
        # top_k cut is HERE (after C5), not before rerank.
        selected_mcs = c5_scored[: profile.top_k]

        if not selected_mcs:
            logger.warning(
                "chunk-scope multi-channel: C5 scoring produced 0 selected "
                "candidates for pass=%s run=%s — failing open to mode=full",
                body.pass_name, body.pipeline_run_id,
            )
            return ChunkScopeResponse(
                mode="full",
                self_refs=[],
                diagnostics=ChunkScopeDiagnostics(
                    mode="full",
                    fallback_reason="no_selected_refs_after_rerank",
                    query_text=query_text,
                    vector_threshold=profile.min_similarity,
                    vector_score_range=None,
                    candidate_count=len(pool),
                    rerank_score_range=_rerank_score_range(reranked_pool),
                    selected_ref_count=0,
                    selected_token_estimate=0,
                    full_doc_token_estimate=full_doc_token_estimate,
                    would_skip_if_fallback_disabled=False,
                    vector_search_ms=vector_search_ms,
                    rerank_ms=rerank_ms,
                    ann_top_k_requested=_mc_ann_top_k,
                    post_filter_candidate_count=_mc_post_filter_count,
                    post_filter_retry_count=0,
                    filter_strategy=_mc_filter_strategy,
                    short_fetch=False,
                ),
            )

        # -------------------------------------------------------------------
        # E2 FALLBACK LADDER — graduated escalation when coverage is low.
        #
        # Check coverage using E1 helpers AFTER C5 scoring.  If sufficient,
        # skip the ladder (happy path — fallback_level stays "none").
        # If under-covered, escalate IN ORDER stopping when sufficient:
        #   1. relaxed_dense  — re-merge with larger top_n cap (no re-fetch, no re-embed)
        #   2. lexical_table  — admit keyword-only candidates absent from dense pool
        #   3. identity_anchor — include C8 anchors if available (no-op when none)
        #   4. full           — use existing full-document fallback
        #
        # "Coverage" is defined by E1 helpers applied to the MERGED POOL (not
        # just the top_k slice) to reflect the full retrieval signal.
        # -------------------------------------------------------------------
        _e2_fallback_level = "none"

        # E1 coverage check on the full merged pool (before top_k slice).
        # Field coverage is only meaningful when the pass has field queries.
        # Passes with no field queries (entity-only) skip the field-coverage
        # check — their coverage is judged by enough_candidates alone.
        _all_pool_mcs = [d["merged_candidate"] for d in reranked_pool if "merged_candidate" in d]
        _has_field_queries = bool(getattr(signals, "field_queries", ()))
        _enough = (
            enough_candidates(_all_pool_mcs, profile)
            and (not _has_field_queries or enough_field_coverage(_all_pool_mcs, profile))
        )

        # E3 — capture initial (pre-escalation) pool state for diagnostics.
        # Computed once here; never mutated by the ladder levels below.
        _e3_candidate_count_before: int = len(_all_pool_mcs)
        _e3_field_coverage_before: dict[str, int] = field_coverage(_all_pool_mcs)

        if not _enough:
            # ---------------------------------------------------------------
            # Level 1: relaxed_dense — re-run C4 merge with a larger top_n cap
            # (2× top_n_candidates) to admit lower-scoring candidates already
            # in entity_dense/field_dense.  NO re-fetch, NO re-embed: uses
            # _mc_state from the initial pass.
            # ---------------------------------------------------------------
            _relaxed_top_n = profile.top_n_candidates * 2
            _relaxed_pool: list = build_pool_from_multi_channel_state(
                _mc_state,
                profile,
                top_n_override=_relaxed_top_n,
            )

            if _relaxed_pool:
                _e2_fallback_level = "relaxed_dense"
                pool = _relaxed_pool

                _relaxed_pool_dicts = [
                    {
                        "content_text": mc.chunk_text,
                        "self_ref": mc.self_ref,
                        "vector_score": mc.vector_score,
                        "chunk_index": mc.chunk_index,
                        "source_refs": mc.source_refs,
                        "token_count": mc.token_count,
                        "page_number": mc.page_number,
                        "alias_hits": mc.alias_hits,
                        "pattern_hits": mc.pattern_hits,
                        "negative_hits": mc.negative_hits,
                        "section_hits": mc.section_hits,
                        "content_type": mc.content_type,
                        "retrieval_sources": mc.retrieval_sources,
                        "supported_field_hints": mc.supported_field_hints,
                        "merged_candidate": mc,
                    }
                    for mc in _relaxed_pool
                ]

                try:
                    def _rerank_relaxed() -> list[dict]:
                        return rrk.rerank(
                            query=query_text,
                            candidates=_relaxed_pool_dicts,
                            top_k=len(_relaxed_pool_dicts),
                        )
                    reranked_pool = await loop.run_in_executor(None, _rerank_relaxed)
                except Exception:
                    reranked_pool = _relaxed_pool_dicts  # reranker fail-open

                c5_scored = score_candidates(reranked_pool, profile)
                selected_mcs = c5_scored[: profile.top_k]
                pool_dicts = _relaxed_pool_dicts

                # Re-check coverage
                _relaxed_mcs = [d["merged_candidate"] for d in reranked_pool if "merged_candidate" in d]
                _enough = (
                    enough_candidates(_relaxed_mcs, profile)
                    and (not _has_field_queries or enough_field_coverage(_relaxed_mcs, profile))
                )

            if not _enough:
                # ---------------------------------------------------------------
                # Level 2: lexical_table — admit keyword-only candidates (those
                # with lexical/pattern hits absent from the dense pool).
                # Build rows_by_key for lookup; do NOT change C4 primary path.
                # ---------------------------------------------------------------
                _pool_keys = {mc.candidate_key for mc in pool}
                _rows_by_key = {
                    (row.get("vertex_id") or row.get("self_ref", "")): row
                    for row in _mc_state.rows
                }
                _kw_candidates = _build_lexical_table_candidates(
                    pool_keys=_pool_keys,
                    lexical_hits=_mc_state.lex_hits,
                    pattern_hits=_mc_state.pat_hits,
                    rows_by_key=_rows_by_key,
                )

                if _kw_candidates:
                    _e2_fallback_level = "lexical_table"
                    # Extend the pool with keyword-only candidates.
                    # Re-run rerank + C5 over the extended pool.
                    _combined_pool = list(pool) + _kw_candidates
                    _combined_dicts = list(pool_dicts) + [
                        {
                            "content_text": mc.chunk_text,
                            "self_ref": mc.self_ref,
                            "vector_score": mc.vector_score,
                            "chunk_index": mc.chunk_index,
                            "source_refs": mc.source_refs,
                            "token_count": mc.token_count,
                            "page_number": mc.page_number,
                            "alias_hits": mc.alias_hits,
                            "pattern_hits": mc.pattern_hits,
                            "negative_hits": mc.negative_hits,
                            "section_hits": mc.section_hits,
                            "content_type": mc.content_type,
                            "retrieval_sources": mc.retrieval_sources,
                            "supported_field_hints": mc.supported_field_hints,
                            "merged_candidate": mc,
                        }
                        for mc in _kw_candidates
                    ]
                    pool = _combined_pool

                    try:
                        def _rerank_kw() -> list[dict]:
                            return rrk.rerank(
                                query=query_text,
                                candidates=_combined_dicts,
                                top_k=len(_combined_dicts),
                            )
                        reranked_pool = await loop.run_in_executor(None, _rerank_kw)
                    except Exception:
                        reranked_pool = _combined_dicts

                    c5_scored = score_candidates(reranked_pool, profile)
                    selected_mcs = c5_scored[: profile.top_k]
                    pool_dicts = _combined_dicts

                    _kw_mcs = [d["merged_candidate"] for d in reranked_pool if "merged_candidate" in d]
                    _enough = (
                        enough_candidates(_kw_mcs, profile)
                        and (not _has_field_queries or enough_field_coverage(_kw_mcs, profile))
                    )

                if not _enough:
                    # -----------------------------------------------------------
                    # Level 3: identity_anchor — re-merge with C8 anchors injected
                    # (no-op when _c8_anchors is empty; skipped automatically).
                    # Uses build_pool_from_multi_channel_state (no re-fetch/embed).
                    # -----------------------------------------------------------
                    if _c8_anchors:
                        _anchor_pool: list = build_pool_from_multi_channel_state(
                            _mc_state,
                            profile,
                            identity_anchors=_c8_anchors,
                            top_n_override=_relaxed_top_n,
                        )
                        if _anchor_pool:
                            _e2_fallback_level = "identity_anchor"
                            pool = _anchor_pool
                            _anchor_dicts = [
                                {
                                    "content_text": mc.chunk_text,
                                    "self_ref": mc.self_ref,
                                    "vector_score": mc.vector_score,
                                    "chunk_index": mc.chunk_index,
                                    "source_refs": mc.source_refs,
                                    "token_count": mc.token_count,
                                    "page_number": mc.page_number,
                                    "alias_hits": mc.alias_hits,
                                    "pattern_hits": mc.pattern_hits,
                                    "negative_hits": mc.negative_hits,
                                    "section_hits": mc.section_hits,
                                    "content_type": mc.content_type,
                                    "retrieval_sources": mc.retrieval_sources,
                                    "supported_field_hints": mc.supported_field_hints,
                                    "merged_candidate": mc,
                                }
                                for mc in _anchor_pool
                            ]

                            try:
                                def _rerank_anchor() -> list[dict]:
                                    return rrk.rerank(
                                        query=query_text,
                                        candidates=_anchor_dicts,
                                        top_k=len(_anchor_dicts),
                                    )
                                reranked_pool = await loop.run_in_executor(None, _rerank_anchor)
                            except Exception:
                                reranked_pool = _anchor_dicts

                            c5_scored = score_candidates(reranked_pool, profile)
                            selected_mcs = c5_scored[: profile.top_k]
                            pool_dicts = _anchor_dicts

                            _anc_mcs = [d["merged_candidate"] for d in reranked_pool if "merged_candidate" in d]
                            _enough = (
                                enough_candidates(_anc_mcs, profile)
                                and (not _has_field_queries or enough_field_coverage(_anc_mcs, profile))
                            )

                    if not _enough:
                        # -------------------------------------------------------
                        # Level 4: full — only after all cheaper levels fail.
                        # Respects fallback_to_full flag; would_skip when False.
                        # -------------------------------------------------------
                        if not selected_mcs:
                            # Pool is genuinely empty even after all levels
                            _e2_fallback_level = "full" if profile.fallback_to_full else "would_skip"
                            _full_mode = "full" if profile.fallback_to_full else "would_skip"
                            # E3: after-count reflects final pool size (may be 0)
                            _l4_after_mcs = [
                                d["merged_candidate"]
                                for d in reranked_pool
                                if "merged_candidate" in d
                            ]
                            return ChunkScopeResponse(
                                mode=_full_mode,
                                self_refs=[],
                                diagnostics=ChunkScopeDiagnostics(
                                    mode=_full_mode,
                                    fallback_reason="coverage_insufficient_after_all_fallback_levels",
                                    query_text=query_text,
                                    vector_threshold=profile.min_similarity,
                                    vector_score_range=None,
                                    candidate_count=len(pool),
                                    rerank_score_range=_rerank_score_range(reranked_pool),
                                    selected_ref_count=0,
                                    selected_token_estimate=0,
                                    full_doc_token_estimate=full_doc_token_estimate,
                                    would_skip_if_fallback_disabled=not profile.fallback_to_full,
                                    vector_search_ms=vector_search_ms,
                                    rerank_ms=rerank_ms,
                                    ann_top_k_requested=_mc_ann_top_k,
                                    post_filter_candidate_count=_mc_post_filter_count,
                                    post_filter_retry_count=0,
                                    filter_strategy=_mc_filter_strategy,
                                    short_fetch=False,
                                    fallback_level=_e2_fallback_level,
                                    # E3 diagnostics on this early-return path.
                                    field_coverage_before_fallback=_e3_field_coverage_before,
                                    candidate_count_before_fallback=_e3_candidate_count_before,
                                    candidate_count_after_fallback=len(_l4_after_mcs),
                                ),
                            )
                        # If we still have some selected (coverage low but non-zero),
                        # proceed with what we have rather than failing to full.
                        _e2_fallback_level = "full" if profile.fallback_to_full else "would_skip"

        # -------------------------------------------------------------------
        # END E2 FALLBACK LADDER
        # -------------------------------------------------------------------

        # Attach score_components to the pool_dicts for diagnostics (matches
        # the back-ref on each dict via "merged_candidate").
        # Build top_k_results from the C5-selected MergedCandidates.
        # We need the dict form (with reranker_score, source_refs, etc.) for
        # the shared merged-mode expansion logic below. Find each dict in the
        # reranked_pool by matching merged_candidate identity.
        mc_to_dict: dict[int, dict] = {
            id(d["merged_candidate"]): d
            for d in reranked_pool
            if "merged_candidate" in d
        }
        top_k_results = []
        for mc, final_score in selected_mcs:
            d = mc_to_dict.get(id(mc))
            if d is not None:
                top_k_results.append(d)
            else:
                # Defensive fallback: build a minimal dict from mc
                top_k_results.append({
                    "content_text": mc.chunk_text,
                    "self_ref": mc.self_ref,
                    "vector_score": mc.vector_score,
                    "chunk_index": mc.chunk_index,
                    "source_refs": mc.source_refs,
                    "token_count": mc.token_count,
                })

        # The shared merged-mode expansion block below uses top_k_results with
        # the same dict keys as the per-element path. Guard: if no valid self_refs
        # (all merged chunks have "chunk_{n}" self_refs, not "#/..." docling refs,
        # so the text_by_ref guard below will correctly filter them to the
        # SelectedChunk.text path).
        #
        # Diagnostics for merged path (shared diagnostics block reads these)
        _diag_search_diag_ann_top_k = _mc_ann_top_k
        _diag_search_diag_post_filter = _mc_post_filter_count
        _diag_search_diag_retry = 0
        _diag_search_diag_strategy = _mc_filter_strategy
        _diag_search_diag_short_fetch = False
        # vector_score_range: None for multi-channel path (pool is MergedCandidate
        # list; _score_range expects GraphEntityResult with .score attribute).
        _vector_score_range: tuple[float, float] | None = None

    else:
        # ------------------------------------------------------------------
        # PER-ELEMENT MODE — existing single-vector path (BYTE-IDENTICAL).
        # No changes to this branch. All variable names and logic preserved.
        # ------------------------------------------------------------------

        # 3. Embed the query (sync function in executor — rev 10 M3)
        def _embed() -> list[float]:
            return embed_texts([query_text], query=True)[0]

        query_vector: list[float] = await loop.run_in_executor(None, _embed)

        # 4. Vector retrieval via the over-fetch helper
        # NEVER use vector_search(filters=...) — ArcadeDB post-HNSW filter bug
        # (see C.1 module docstring + tests/integration/test_extraction_chunk_filter_starvation.py)
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
        _reranker_enabled = settings.reranker_enabled
        _reranker_fallback_reason: str | None = (
            None if _reranker_enabled else "reranker_disabled"
        )

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

        # Diagnostics aliases for the shared block below.
        _diag_search_diag_ann_top_k = search_diag.ann_top_k_requested
        _diag_search_diag_post_filter = search_diag.post_filter_candidate_count
        _diag_search_diag_retry = search_diag.post_filter_retry_count
        _diag_search_diag_strategy = search_diag.filter_strategy
        _diag_search_diag_short_fetch = search_diag.short_fetch
        # vector_score_range: populated from the per-element GraphEntityResult list.
        _vector_score_range = _score_range(results)
    # -----------------------------------------------------------------------
    # Shared post-retrieval path — used by BOTH merged and per-element branches.
    # top_k_results: list[dict] with keys:
    #   content_text, self_ref, vector_score, chunk_index, source_refs,
    #   token_count (+ merged_candidate in merged mode, + reranker_score).
    # _diag_search_diag_* variables: aliased from branch-specific diag objects.
    # results: the raw candidate pool (GraphEntityResult list in per-element;
    #          MergedCandidate list in merged — used only for _score_range()).
    # -----------------------------------------------------------------------

    selected_refs = [c["self_ref"] for c in top_k_results if c.get("self_ref")]
    text_by_ref: dict[str, str] = {}
    for c in top_k_results:
        self_ref = c.get("self_ref")
        content_text = c.get("content_text")
        # Only docling element refs (``#/...``) belong in ``text_by_ref`` —
        # it is consumed by ``apply_chunk_scope`` to override per-element
        # text on real DoclingDocument elements. Merged-mode rows carry
        # ``self_ref = f"chunk_{idx}"`` (Task 8 cleanup); their text
        # rides on ``SelectedChunk.text`` instead, so they must be
        # filtered out here to avoid polluting the override map.
        if (
            isinstance(self_ref, str)
            and self_ref.startswith("#/")
            and isinstance(content_text, str)
            and content_text.strip()
        ):
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
        # Build the rerank_score_range from top_k_results (works for both branches).
        _shared_rerank_scores = [
            c["reranker_score"] for c in top_k_results if "reranker_score" in c
        ]
        _shared_rerank_range = (
            (min(_shared_rerank_scores), max(_shared_rerank_scores))
            if _shared_rerank_scores else None
        )
        return ChunkScopeResponse(
            mode="full",
            self_refs=[],
            diagnostics=ChunkScopeDiagnostics(
                mode="full",
                fallback_reason="no_selected_refs_after_rerank",
                query_text=query_text,
                vector_threshold=profile.min_similarity,
                vector_score_range=None,
                candidate_count=len(top_k_results),
                rerank_score_range=_shared_rerank_range,
                selected_ref_count=0,
                selected_token_estimate=0,
                full_doc_token_estimate=full_doc_token_estimate,
                would_skip_if_fallback_disabled=False,
                vector_search_ms=vector_search_ms,
                rerank_ms=rerank_ms,
                ann_top_k_requested=_diag_search_diag_ann_top_k,
                post_filter_candidate_count=_diag_search_diag_post_filter,
                post_filter_retry_count=_diag_search_diag_retry,
                filter_strategy=_diag_search_diag_strategy,
                short_fetch=_diag_search_diag_short_fetch,
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

    # Important #2 (rev 16): short_fetch must be surfaced.
    # Mode stays selected_refs — short_fetch is diagnostic-only for v1.
    if _diag_search_diag_short_fetch:
        logger.warning(
            "chunk-scope: short-fetch on pass=%s run=%s "
            "(post_filter_candidate_count=%d < desired_top_n=%d); "
            "downstream may see incomplete retrieval",
            body.pass_name, body.pipeline_run_id,
            _diag_search_diag_post_filter, profile.top_n_candidates,
        )

    # ---------------------------------------------------------------
    # Phase 1 Task 6 + C6: merged-mode expansion path.
    # Gated on ``settings.extraction_index_mode == "merged"`` so per_element
    # mode (the default) stays byte-identical to the pre-Task-6 response.
    # Both the original per-element merged path and the new multi-channel
    # merged path (C6) share this expansion block — top_k_results carries
    # the same dict keys in both cases.
    # ---------------------------------------------------------------
    selected_chunks_out: list[SelectedChunk] | None = None
    selected_chunk_count = 0
    expanded_ref_count = 0
    selected_chunk_token_estimate = 0
    final_self_refs = selected_refs
    final_selected_ref_count = len(selected_refs)
    final_selected_token_estimate = selected_token_estimate

    if _use_multi_channel:
        # Expand the per-chunk source_refs into a deduplicated list in
        # chunk-encounter order (rerank rank order — do NOT lex-sort;
        # '#/texts/100' would otherwise precede '#/texts/35').
        # Gated on _use_multi_channel (not just extraction_index_mode) so
        # the expansion only runs when top_k_results actually carry the
        # multi-channel dict shape (with source_refs from MergedCandidates).
        # Per-element dicts don't carry source_refs and must NOT go through
        # this expansion path.
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

            # Belt-and-suspenders guard: skip rows where chunk_index=-1 AND
            # source_refs is empty AND content_text is blank.  The (removed)
            # C4 keyword-only-bucket pattern that originally created these rows
            # (merge_candidates setting chunk_index=-1) was fixed in commit
            # d0e4e03; this guard is retained as a defensive check against
            # unexpected data shapes (e.g. future schema changes or test stubs).
            # Including such a row would produce SelectedChunk(chunk_index=-1),
            # which breaks the response contract.
            content = chunk_row.get("content_text") or ""
            if chunk_idx == -1 and not refs_for_chunk and not content.strip():
                continue

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
                    text=content,
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

        # Fix 1 (rev 18): fail-open guard — multi-channel expansion produced
        # no source_refs.  This is pathological (a non-empty top_k_results pool
        # where every selected MergedCandidate has an empty source_refs list),
        # but must not silently return mode=selected_refs with self_refs=[],
        # which violates the Rev 17 MED contract.  Mirror the existing shared
        # guard at ~line 715 (same fallback_to_full → full vs would_skip logic).
        if not expanded_refs:
            logger.warning(
                "chunk-scope multi-channel: expansion produced 0 source_refs from "
                "%d top_k candidates for pass=%s run=%s — failing open",
                len(top_k_results), body.pass_name, body.pipeline_run_id,
            )
            _exp_rerank_scores = [
                c["reranker_score"] for c in top_k_results if "reranker_score" in c
            ]
            _exp_rerank_range = (
                (min(_exp_rerank_scores), max(_exp_rerank_scores))
                if _exp_rerank_scores else None
            )
            _exp_mode = "full" if profile.fallback_to_full else "would_skip"
            return ChunkScopeResponse(
                mode=_exp_mode,
                self_refs=[],
                diagnostics=ChunkScopeDiagnostics(
                    mode=_exp_mode,
                    fallback_reason="no_expanded_refs_after_multichannel",
                    query_text=query_text,
                    vector_threshold=profile.min_similarity,
                    vector_score_range=None,
                    candidate_count=len(top_k_results),
                    rerank_score_range=_exp_rerank_range,
                    selected_ref_count=0,
                    selected_token_estimate=0,
                    full_doc_token_estimate=full_doc_token_estimate,
                    would_skip_if_fallback_disabled=not profile.fallback_to_full,
                    vector_search_ms=vector_search_ms,
                    rerank_ms=rerank_ms,
                    ann_top_k_requested=_diag_search_diag_ann_top_k,
                    post_filter_candidate_count=_diag_search_diag_post_filter,
                    post_filter_retry_count=_diag_search_diag_retry,
                    filter_strategy=_diag_search_diag_strategy,
                    short_fetch=_diag_search_diag_short_fetch,
                    selected_chunk_count=selected_chunk_count,
                    expanded_ref_count=0,
                    selected_chunk_token_estimate=selected_chunk_token_estimate,
                ),
            )

    # Build shared rerank_score_range from top_k_results (works for both branches).
    _final_rerank_scores = [
        c["reranker_score"] for c in top_k_results if "reranker_score" in c
    ]
    _final_rerank_range: tuple[float, float] | None = (
        (min(_final_rerank_scores), max(_final_rerank_scores))
        if _final_rerank_scores else None
    )

    # -----------------------------------------------------------------------
    # Task C7 — multi-channel diagnostics (populated on success path only).
    # All four fields default None; set only when _use_multi_channel is True.
    # Per-element path and all error/early-return paths leave them None.
    # -----------------------------------------------------------------------
    _c7_channel_counts: dict[str, int] | None = None
    _c7_field_coverage: dict[str, int] | None = None
    _c7_score_components: list[dict] | None = None
    _c7_fallback_level: str | None = None

    # Task E3 — fallback diagnostics (before/after counts + field_coverage_before).
    # All default None; populated only on the multi-channel success path.
    # The before-values were captured inside the ladder block above.
    _e3_field_coverage_before_diag: dict[str, int] | None = None
    _e3_candidate_count_before_diag: int | None = None
    _e3_candidate_count_after_diag: int | None = None

    # Tasks F2+F4 — subset-schema field subset + diagnostics.
    # None when subset_schema_extraction=False (default) or per-element path.
    _f2_field_subset: list[str] | None = None
    _f4_active_field_count: int | None = None
    _f4_dropped_field_count: int | None = None

    if _use_multi_channel:
        # channel_counts: one key per retrieval channel.
        # "dense" and "field:<name>" come from mc_search_diag;
        # "lexical" and "pattern" are chunk-level hit counts from mc_search_diag.
        _cc: dict[str, int] = {"dense": mc_search_diag.entity_dense_count}
        for field_name, count in mc_search_diag.per_field_dense_counts.items():
            _cc[f"field:{field_name}"] = count
        _cc["lexical"] = mc_search_diag.lexical_hit_count
        _cc["pattern"] = mc_search_diag.pattern_hit_count
        _c7_channel_counts = _cc

        # field_coverage: count of SELECTED candidates with evidence for each field.
        # Derived from the supported_field_hints on the MergedCandidates in top_k_results.
        _fc: dict[str, int] = {}
        for c7_d in top_k_results:
            mc_obj = c7_d.get("merged_candidate")
            if mc_obj is not None:
                for hint in getattr(mc_obj, "supported_field_hints", set()):
                    _fc[hint] = _fc.get(hint, 0) + 1
        _c7_field_coverage = _fc

        # score_components: one dict per selected (top_k) candidate.
        # Sourced from selected_mcs (list[(MergedCandidate, final_score)]).
        # mc_to_dict is already built above — reuse it to get reranker_score.
        _sc_list: list[dict] = []
        for mc_obj, final_score in selected_mcs:
            mc_dict = mc_to_dict.get(id(mc_obj))
            reranker_score = mc_dict.get("reranker_score") if mc_dict else None
            _sc_list.append({
                "candidate_key": mc_obj.candidate_key,
                "reranker_score": reranker_score,
                "alias_hits": mc_obj.alias_hits,
                "pattern_hits": mc_obj.pattern_hits,
                "negative_hits": mc_obj.negative_hits,
                "section_hits": mc_obj.section_hits,
                "retrieval_sources": sorted(mc_obj.retrieval_sources),
                "final_score": final_score,
            })
        _c7_score_components = _sc_list[: profile.top_k]

        # fallback_level: set by E2 ladder ("none" on the happy path).
        _c7_fallback_level = _e2_fallback_level

        # E3 — populate before/after diagnostics from the ladder.
        # _e3_candidate_count_before and _e3_field_coverage_before were captured
        # at the top of the ladder block (initial pool snapshot).
        # After-count: current pool size (== before when no escalation fired).
        _e3_field_coverage_before_diag = _e3_field_coverage_before
        _e3_candidate_count_before_diag = _e3_candidate_count_before
        # Pool may have grown through relaxed_dense / lexical_table / identity_anchor.
        # Reflect the final state from reranked_pool (the last version of the pool
        # that was fed into C5 scoring, which is the authoritative final pool).
        _e3_after_pool_mcs = [d["merged_candidate"] for d in reranked_pool if "merged_candidate" in d]
        _e3_candidate_count_after_diag = len(_e3_after_pool_mcs)

        # Tasks F2+F4 — subset-schema field subset computation.
        # selected_mcs is the C5-scored top_k list: [(MergedCandidate, score), ...].
        # active_fields() is opt-in: when subset_schema_extraction=False (default),
        # it returns ALL fields (no-op). When True, it computes the evidence-based
        # active subset. The subset is derived from the top_k selected_mcs
        # candidates — those are exactly the chunks the LLM will see, so the
        # active fields must reflect that same evidence set. selected_mcs carries
        # the MergedCandidate objects, which hold supported_field_hints.
        if profile.subset_schema_extraction:
            _f2_selected_mcs_only = [mc for mc, _ in selected_mcs]
            _f2_all_fields = list(template_cls.model_fields.keys())
            _f2_active = _active_fields(_f2_selected_mcs_only, template_cls, profile)
            _f2_field_subset = _f2_active
            _f4_active_field_count = len(_f2_active)
            _f4_dropped_field_count = len(_f2_all_fields) - len(_f2_active)

    return ChunkScopeResponse(
        mode="selected_refs",
        self_refs=final_self_refs,
        text_by_ref=text_by_ref,
        selected_chunks=selected_chunks_out,
        field_subset=_f2_field_subset,
        diagnostics=ChunkScopeDiagnostics(
            mode="selected_refs",
            fallback_reason=_reranker_fallback_reason,  # Minor #6 (rev 16)
            query_text=query_text,
            vector_threshold=profile.min_similarity,
            vector_score_range=_vector_score_range,  # per-element: from results; merged: None
            candidate_count=len(top_k_results),
            rerank_score_range=_final_rerank_range,
            selected_ref_count=final_selected_ref_count,
            selected_token_estimate=final_selected_token_estimate,
            full_doc_token_estimate=full_doc_token_estimate,
            would_skip_if_fallback_disabled=False,
            vector_search_ms=vector_search_ms,
            rerank_ms=rerank_ms,
            ann_top_k_requested=_diag_search_diag_ann_top_k,
            post_filter_candidate_count=_diag_search_diag_post_filter,
            post_filter_retry_count=_diag_search_diag_retry,
            filter_strategy=_diag_search_diag_strategy,
            short_fetch=_diag_search_diag_short_fetch,  # Important #2 (rev 16)
            # Task 6 merged-mode diagnostics (0 in per_element mode).
            selected_chunk_count=selected_chunk_count,
            expanded_ref_count=expanded_ref_count,
            selected_chunk_token_estimate=selected_chunk_token_estimate,
            # Task C7 multi-channel diagnostics (None in per_element mode).
            channel_counts=_c7_channel_counts,
            field_coverage=_c7_field_coverage,
            score_components=_c7_score_components,
            fallback_level=_c7_fallback_level,
            # Task C8 identity-anchor channel count (None in per_element mode;
            # 0 when anchor query ran but found no committed entities).
            identity_anchor_count=_c8_anchor_count,
            # Task E3 — fallback diagnostics (None in per_element mode).
            field_coverage_before_fallback=_e3_field_coverage_before_diag,
            candidate_count_before_fallback=_e3_candidate_count_before_diag,
            candidate_count_after_fallback=_e3_candidate_count_after_diag,
            # Tasks F2+F4 — subset-schema diagnostics (None when off/per-element).
            active_field_count=_f4_active_field_count,
            dropped_field_count=_f4_dropped_field_count,
        ),
    )
