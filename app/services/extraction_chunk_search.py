"""Vector search over ExtractionChunk vertices, isolated by pipeline_run_id.

VR Phase C.1 (rev 11) — over-fetch + Python post-filter strategy.

ARCHITECTURE NOTE — ArcadeDB post-HNSW filter behavior
------------------------------------------------------
ArcadeDB's ``vectorNeighbors()`` function traverses the HNSW graph and then
applies WHERE predicates as a post-filter on the top-K results — it does NOT
filter candidates during traversal. This is an acknowledged P1 gap in the
ArcadeDB vector-DB comparison analysis (see
``docker/arcadedb/repo/docs/arcadedb-vs-leading-vector-dbms.md``,
section "Filterable HNSW", which notes: "JVector treats [RIDBitsFilter] as a
per-point skip during traversal" — empirically confirmed to be post-HNSW by
the adversarial test in
``tests/integration/test_extraction_chunk_filter_starvation.py``).

Consequence: calling ``vector_search(filters={"pipeline_run_id": ...})``
causes "filter starvation" when one pipeline_run_id's chunks are cosmetically
closer to the query than another run's chunks — the wrong-run chunks saturate
the top-K and the right-run chunks are never returned.

MUST NOT use ``vector_search(filters={"pipeline_run_id": ...})`` for VR
vector queries. Use ``search_extraction_chunks()`` instead, which over-fetches
unfiltered and post-filters in Python.

``vector_search(filters=...)`` remains valid for non-vector metadata queries
(e.g., looking up vertices by non-vector properties).
"""
# ============================================================================
# CAPACITY ASSUMPTIONS (rev 13 — VR C.1 review nit #1)
# ============================================================================
# The over-fetch + post-filter strategy is calibrated for the EXPECTED production
# workload of <= ~10 concurrent pipeline_runs with ~300 chunks per run (3,000
# total ExtractionChunk rows worst-case under the 24h janitor TTL).
#
# At this scale:
#   - initial_top_k = max(desired_top_n * 10, 500), with desired_top_n typically 50
#     → 500 rows fetched; expected right-run survivors ≈ 500 × (300/3000) = 50.
#   - Retry at _RETRY_TOP_K = 2000 covers the entire current chunk pool.
#
# DEGRADATION POINTS:
#   - At 100 concurrent runs (30K total chunks), 500-top_k yields ~5 right-run
#     survivors; 2000-top_k retry yields ~20 — borderline for desired_top_n=50.
#   - At 200 concurrent runs (60K total), 500 yields ~2.5; 2000 yields ~10.
#     short_fetch=True will fire; callers MUST inspect diagnostics and decide.
#
# If production concurrency grows past ~50 simultaneous runs, revisit:
#   - Increase _RETRY_TOP_K to 5000+
#   - Switch ExtractionChunk to a per-run vertex type (was rev-8 option (c),
#     rejected; reconsider if concurrency growth materializes)
#   - Push ArcadeDB to ship filtered HNSW upstream (P1 gap; may not happen)
# ============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    from app.services.graph_store import GraphEntityResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostics dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChunkSearchDiagnostics:
    """Diagnostic counters from a ``search_extraction_chunks()`` call.

    Ready to merge into the VR router diagnostics block:
      ``pipeline_pass_outputs.diagnostics_json.router.search``

    Field semantics differ slightly between the two retrieval paths
    selected by ``settings.vector_router_retrieval_mode``:

    - ``"hnsw"`` (overfetch + post-filter): the legacy path.
    - ``"direct"`` (Path B, 2026-05-26): per-run SQL pull + numpy cosine.

    Fields
    ------
    ann_top_k_requested:
        HNSW: the top_k value passed to the unfiltered ``vector_search``
        call (initial attempt; if retry fired, the *retry* value — the
        larger cap).
        DIRECT: total number of rows pulled for the pipeline_run_id (the
        candidate pool size; not an ANN cap).
    post_filter_candidate_count:
        HNSW: rows surviving the ``pipeline_run_id`` post-filter.
        DIRECT: rows surviving the optional ``score_threshold`` filter,
        BEFORE the final ``desired_top_n`` slice.
        Both paths: may be < ``desired_top_n`` (signals ``short_fetch``
        for HNSW; signals "not enough matches" for direct).
    post_filter_retry_count:
        HNSW: 0 if initial overfetch had enough survivors; 1 if the
        retry at top_k=2000 was needed.
        DIRECT: always 0 (direct retrieval doesn't retry — it's exact).
    filter_strategy:
        ``"overfetch_post_filter"`` (HNSW) or ``"direct_cosine"``
        (DIRECT). A future ``"filtered_traversal"`` would land if
        ArcadeDB ships filterable HNSW (P1 gap — see module docstring).
    short_fetch:
        HNSW: True if even the retry didn't yield enough survivors —
        the result may be INCOMPLETE (more matching chunks may exist
        in the global index that HNSW didn't surface).
        DIRECT: True if the run has fewer matching chunks than
        ``desired_top_n`` — the result is EXACT (no more matches exist),
        not incomplete. Callers reading this field for fallback logic
        should branch on ``filter_strategy`` if they care about the
        distinction.
    """

    ann_top_k_requested: int
    post_filter_candidate_count: int
    post_filter_retry_count: int
    filter_strategy: str = field(default="overfetch_post_filter")
    short_fetch: bool = field(default=False)


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

# Cap for retry attempt — large enough to cover any realistic single-run
# chunk count without scanning unbounded data.
_RETRY_TOP_K = 2000


async def search_extraction_chunks_hnsw(
    *,
    store: "ArcadeDBGraphStore",
    query_vector: list[float],
    pipeline_run_id: str,
    desired_top_n: int,
    score_threshold: float | None = None,
) -> "tuple[list[GraphEntityResult], ChunkSearchDiagnostics]":
    """Vector search over ExtractionChunk isolated by pipeline_run_id.

    Legacy HNSW path. Subject to post-filter starvation when the global
    HNSW index contains chunks from many concurrent runs — see
    ``search_extraction_chunks_direct`` (Path B, 2026-05-26) for the
    no-starvation alternative selected via
    ``settings.vector_router_retrieval_mode``.

    ArcadeDB HNSW applies WHERE filters POST-traversal — see
    ``tests/integration/test_extraction_chunk_filter_starvation.py`` for
    the adversarial proof. This helper over-fetches unfiltered, then
    post-filters by ``pipeline_run_id`` in Python.

    Strategy
    --------
    1. initial ``top_k = max(desired_top_n * 10, 500)``
    2. Unfiltered ``vector_search`` over ExtractionChunk
       → post-filter by ``pipeline_run_id`` in Python.
    3. If post-filter survivors >= ``desired_top_n`` → return.
    4. Else retry once with ``top_k = _RETRY_TOP_K`` (2000).
    5. Else return whatever survived; ``ChunkSearchDiagnostics.short_fetch``
       is set to True so callers can detect and handle the short-fetch case.

    Parameters
    ----------
    store:
        An ``ArcadeDBGraphStore`` instance.
    query_vector:
        The query embedding (bge-m3, dim=1024).
    pipeline_run_id:
        The pipeline run UUID that scopes this VR query. Only chunks
        belonging to this run are returned.
    desired_top_n:
        How many post-filtered results the caller wants (e.g., profile
        ``top_n_candidates``). Used to size the initial overfetch.
    score_threshold:
        Optional minimum similarity score applied to the unfiltered
        vector_search results before the pipeline_run_id filter.

    Returns
    -------
    tuple[list[GraphEntityResult], ChunkSearchDiagnostics]
        The filtered results and a diagnostics object ready to merge into
        the router diagnostics block.
    """
    initial_top_k = max(desired_top_n * 10, 500)

    # --- Initial overfetch (unfiltered) ------------------------------------
    raw_results = await store.vector_search(
        vertex_type="ExtractionChunk",
        embedding_property="embedding",
        query_vector=query_vector,
        top_k=initial_top_k,
        score_threshold=score_threshold,
        filters=None,  # MUST be None — see module docstring
    )

    filtered = [
        r for r in raw_results
        if r.properties.get("pipeline_run_id") == pipeline_run_id
    ]

    if len(filtered) >= desired_top_n:
        diag = ChunkSearchDiagnostics(
            ann_top_k_requested=initial_top_k,
            post_filter_candidate_count=len(filtered),
            post_filter_retry_count=0,
        )
        return filtered[:desired_top_n], diag

    # --- Retry with larger cap ---------------------------------------------
    logger.debug(
        "search_extraction_chunks: initial overfetch (top_k=%d) yielded only %d "
        "survivors for pipeline_run_id=%r (desired %d). Retrying with top_k=%d.",
        initial_top_k,
        len(filtered),
        pipeline_run_id,
        desired_top_n,
        _RETRY_TOP_K,
    )

    raw_results_retry = await store.vector_search(
        vertex_type="ExtractionChunk",
        embedding_property="embedding",
        query_vector=query_vector,
        top_k=_RETRY_TOP_K,
        score_threshold=score_threshold,
        filters=None,  # MUST be None — see module docstring
    )

    filtered_retry = [
        r for r in raw_results_retry
        if r.properties.get("pipeline_run_id") == pipeline_run_id
    ]

    short_fetch = len(filtered_retry) < desired_top_n

    if short_fetch:
        logger.warning(
            "search_extraction_chunks: short-fetch after retry. "
            "pipeline_run_id=%r desired=%d survivors=%d (retry top_k=%d). "
            "Returning partial results. Check chunk count for this run.",
            pipeline_run_id,
            desired_top_n,
            len(filtered_retry),
            _RETRY_TOP_K,
        )

    diag = ChunkSearchDiagnostics(
        ann_top_k_requested=_RETRY_TOP_K,
        post_filter_candidate_count=len(filtered_retry),
        post_filter_retry_count=1,
        short_fetch=short_fetch,
    )
    return filtered_retry[:desired_top_n], diag


# ---------------------------------------------------------------------------
# Path B — direct cosine retrieval (no HNSW)
# ---------------------------------------------------------------------------


async def search_extraction_chunks_direct(
    *,
    store: "ArcadeDBGraphStore",
    query_vector: list[float],
    pipeline_run_id: str,
    desired_top_n: int,
    score_threshold: float | None = None,
) -> "tuple[list[GraphEntityResult], ChunkSearchDiagnostics]":
    """Per-run vector search via direct cosine in Python (Path B).

    Pulls ALL ExtractionChunk vertices for ``pipeline_run_id`` via SQL
    (B-tree-indexed on pipeline_run_id, so this is O(matching rows) — no
    global scan). Computes cosine similarity client-side against
    ``query_vector``. Returns top-``desired_top_n`` chunks scoring
    >= ``score_threshold``, sorted descending by score with ``self_ref``
    ASC as stable tiebreaker.

    Same return shape as ``search_extraction_chunks_hnsw`` for drop-in
    swap behind the ``vector_router_retrieval_mode`` env var. Eliminates
    the HNSW global-graph post-filter starvation documented at the top
    of this module. Exact (no approximation) and deterministic.

    Retrieval stage is ~50,000× faster than HNSW at our scale (300
    chunks); end-to-end depends on reranker candidate count, which may
    INCREASE if direct returns more candidates than HNSW's starved set.

    Diagnostics semantics for the direct path:

    - ``filter_strategy="direct_cosine"`` (vs ``"overfetch_post_filter"``).
    - ``ann_top_k_requested`` = total rows pulled for the run (proxy for
      "how big was the candidate pool"). NOT HNSW top_k.
    - ``post_filter_candidate_count`` = rows surviving ``score_threshold``,
      BEFORE the final ``desired_top_n`` slice. Use this for plumbing
      tests / threshold tuning.
    - ``post_filter_retry_count=0`` always — direct path doesn't retry.
    - ``short_fetch=True`` iff fewer rows passed the threshold than
      ``desired_top_n``. Means "not enough matches in this run" — not
      "retrieval was incomplete" (direct retrieval is exact).
    """
    import numpy as np
    from app.services.graph_store import GraphEntityResult

    # --- 1. Pull every chunk for this run via SQL (B-tree-indexed). --------
    # ORDER BY self_ref ASC gives a stable iteration order for tiebreaking
    # on identical cosine scores (see step 3).
    # `@rid AS node_id` exposes ArcadeDB's internal vertex id under a
    # numpy/JSON-friendly key — keeps GraphEntityResult.node_id aligned with
    # what the HNSW path returns. `vertex_id` is the synthetic PK kept as
    # secondary fallback for environments where @rid isn't materialized.
    # Merged-mode columns (Phase 1 Task 1 / Task 5): ``chunk_index``,
    # ``source_refs``, ``token_count`` are projected so Task 6's
    # chunk-scope endpoint can read them via the Task 1 accessors
    # (``read_chunk_index`` / ``read_chunk_source_refs`` /
    # ``read_chunk_token_count``). The accessors coalesce missing/None
    # values to legacy defaults (-1 / [] / 0); the projection here lets
    # them see the REAL merged-mode values when present.
    rows = await store._client.query(
        store._database,
        "sql",
        (
            "SELECT @rid AS node_id, vertex_id, self_ref, chunk_text, "
            "embedding, page_number, modality, pipeline_run_id, "
            "chunk_index, source_refs, token_count "
            "FROM ExtractionChunk "
            "WHERE pipeline_run_id = :run_id "
            "ORDER BY self_ref ASC"
        ),
        {"run_id": pipeline_run_id},
    )

    if not rows:
        return [], ChunkSearchDiagnostics(
            ann_top_k_requested=0,
            post_filter_candidate_count=0,
            post_filter_retry_count=0,
            filter_strategy="direct_cosine",
            short_fetch=(desired_top_n > 0),
        )

    # --- 2. Build paired valid-rows / embeddings (same length, same order).
    # Filtering only one side would misalign scores with their source rows.
    valid_rows = [r for r in rows if r.get("embedding")]
    if not valid_rows:
        return [], ChunkSearchDiagnostics(
            ann_top_k_requested=len(rows),
            post_filter_candidate_count=0,
            post_filter_retry_count=0,
            filter_strategy="direct_cosine",
            short_fetch=(desired_top_n > 0),
        )

    embeddings = np.asarray(
        [r["embedding"] for r in valid_rows], dtype=np.float32,
    )
    q = np.asarray(query_vector, dtype=np.float32)

    # Defensive normalize. bge-m3 emits L2-normalized vectors; this
    # protects against writer-side drift and converts cosine to dot product.
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    q /= np.linalg.norm(q) + 1e-12

    scores = embeddings @ q  # shape: (N,)

    # --- 3. Threshold filter (defines candidate_count), then top-N slice. --
    if score_threshold is not None:
        keep_mask = scores >= float(score_threshold)
        kept_rows = [r for r, k in zip(valid_rows, keep_mask.tolist()) if k]
        kept_scores = scores[keep_mask]
    else:
        kept_rows = valid_rows
        kept_scores = scores

    candidate_count = len(kept_rows)

    if candidate_count == 0:
        return [], ChunkSearchDiagnostics(
            ann_top_k_requested=len(rows),
            post_filter_candidate_count=0,
            post_filter_retry_count=0,
            filter_strategy="direct_cosine",
            short_fetch=(desired_top_n > 0),
        )

    # Stable sort: primary key = -score (descending), secondary = self_ref
    # ASC (already enforced by SQL ORDER BY but explicit here is cheap +
    # documents the intent for future maintainers). numpy.lexsort uses the
    # LAST key as the primary sort key, so order is (secondary, primary).
    self_refs = np.asarray([r["self_ref"] for r in kept_rows])
    order = np.lexsort((self_refs, -kept_scores))[:desired_top_n]
    selected_rows = [kept_rows[i] for i in order.tolist()]
    selected_scores = kept_scores[order].tolist()

    # --- 4. Materialize as GraphEntityResult to match the HNSW return shape.
    # GraphEntityResult requires node_id + name + entity_type. The SQL above
    # exposes ArcadeDB's @rid under `node_id` (canonical) and the synthetic
    # vertex_id (PK) as fallback. Cascade order:
    #   @rid (raw, unaliased — defensive for adapter/driver variation)
    #   → node_id (the SELECT alias — the expected path)
    #   → vertex_id (synthetic PK)
    #   → self_ref (string-id last resort)
    results = [
        GraphEntityResult(
            node_id=str(
                row.get("@rid")
                or row.get("node_id")
                or row.get("vertex_id")
                or row["self_ref"]
            ),
            name=row["self_ref"],
            entity_type="ExtractionChunk",
            extraction_confidence=score,
            score=score,
            score_type="vector",
            properties={
                "self_ref": row["self_ref"],
                "chunk_text": row.get("chunk_text", ""),
                "page_number": row.get("page_number"),
                "modality": row.get("modality"),
                "pipeline_run_id": row.get("pipeline_run_id"),
                # Phase 1 Task 5 — merged-mode projection. Pass values
                # through as-is (may be None for rows that bypassed the
                # Task 1 backfill); Task 1 accessors coalesce on read.
                "chunk_index": row.get("chunk_index"),
                "source_refs": row.get("source_refs"),
                "token_count": row.get("token_count"),
            },
        )
        for row, score in zip(selected_rows, selected_scores)
    ]

    short_fetch = candidate_count < desired_top_n
    if short_fetch:
        logger.debug(
            "search_extraction_chunks_direct: short fetch — "
            "pipeline_run_id=%r candidates=%d desired=%d. Not an error: "
            "fewer matching chunks exist than requested (direct retrieval "
            "is exact, not approximate).",
            pipeline_run_id, candidate_count, desired_top_n,
        )

    return results, ChunkSearchDiagnostics(
        ann_top_k_requested=len(rows),
        post_filter_candidate_count=candidate_count,
        post_filter_retry_count=0,
        filter_strategy="direct_cosine",
        short_fetch=short_fetch,
    )


# ---------------------------------------------------------------------------
# Public dispatcher — routes to HNSW or direct based on settings
# ---------------------------------------------------------------------------


async def search_extraction_chunks(
    *,
    store: "ArcadeDBGraphStore",
    query_vector: list[float],
    pipeline_run_id: str,
    desired_top_n: int,
    score_threshold: float | None = None,
) -> "tuple[list[GraphEntityResult], ChunkSearchDiagnostics]":
    """Vector search over ExtractionChunk isolated by pipeline_run_id.

    Dispatches to ``search_extraction_chunks_hnsw`` (legacy, default) or
    ``search_extraction_chunks_direct`` (Path B, no starvation) based on
    ``settings.vector_router_retrieval_mode``.

    Keeps the same signature + return shape as the historical entry point
    so all existing callers (chunk-scope endpoint, integration tests) work
    unchanged. The diagnostics ``filter_strategy`` field indicates which
    path actually ran (``"overfetch_post_filter"`` vs ``"direct_cosine"``).
    """
    from app.config import get_settings

    mode = get_settings().vector_router_retrieval_mode
    if mode == "direct":
        return await search_extraction_chunks_direct(
            store=store,
            query_vector=query_vector,
            pipeline_run_id=pipeline_run_id,
            desired_top_n=desired_top_n,
            score_threshold=score_threshold,
        )
    return await search_extraction_chunks_hnsw(
        store=store,
        query_vector=query_vector,
        pipeline_run_id=pipeline_run_id,
        desired_top_n=desired_top_n,
        score_threshold=score_threshold,
    )
