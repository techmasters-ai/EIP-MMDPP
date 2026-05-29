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
    from app.services.extraction_query_builder import PassRetrievalSignals
    from app.services.ontology_bundles import RetrievalProfile

# Module-level import so tests can patch app.services.extraction_chunk_search.embed_texts
# without needing to intercept a local import inside the function body.
# (patch() replaces the name in the target module's namespace; a local
#  `from ... import embed_texts` inside the function creates a fresh binding
#  each call, bypassing the patch.)
from app.services.embedding import embed_texts  # noqa: E402

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

    # Stable sort: primary key = -score (descending), secondary = vertex_id
    # ASC. vertex_id is the schema's UNIQUE column, so it provides
    # deterministic tie-breaking in BOTH modes — legacy
    # ``{run_id}:{self_ref}`` and merged ``{run_id}:chunk_{idx}``. Using
    # self_ref here was unsafe because merged-mode rows can share a
    # self_ref value (the column was populated from ``source_refs[0]``
    # pre-Task-8 cleanup). numpy.lexsort uses the LAST key as the primary
    # sort key, so order is (secondary, primary).
    vertex_ids = np.asarray([r["vertex_id"] for r in kept_rows])
    order = np.lexsort((vertex_ids, -kept_scores))[:desired_top_n]
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


# ---------------------------------------------------------------------------
# Task C8 — opportunistic, non-blocking identity-anchor channel
# ---------------------------------------------------------------------------


async def identity_anchor_queries(
    *,
    store: "ArcadeDBGraphStore",
    pipeline_run_id: str,
    identity_types: list[str],
    worker_anchors: list[str] | None,
) -> list[str]:
    """Query the graph store for committed identity-type entity names for a run.

    Unions the store results with ``worker_anchors`` (caller-supplied), dedupes,
    and returns the combined anchor name list.

    This function is OPPORTUNISTIC and NON-BLOCKING:
    - On ANY exception (store error, type not found, timeout) it logs a debug
      message and returns ``worker_anchors or []`` without re-raising.
    - When ``identity_types`` is empty, no DB query is made; returns
      ``worker_anchors or []`` immediately (clean no-op).
    - Per-type exceptions are caught individually so one missing type does not
      suppress results from other types that succeeded.

    The anchor names are used by the C8 channel to add dense + lexical signals
    for entity names already committed in the graph store for this run. The
    channel ONLY fires when anchors are non-empty; empty → clean no-op.

    Parameters
    ----------
    store:
        An ``ArcadeDBGraphStore`` instance (provides ``_client.query`` +
        ``_database``).
    pipeline_run_id:
        The pipeline run UUID. Used to scope the entity query so only entities
        committed during this run are returned (no cross-run contamination).
    identity_types:
        List of entity type names (e.g. ``["RADAR_SYSTEM", "MISSILE_SYSTEM"]``)
        derived from the bundle manifest's identity passes'
        ``primary_entity_types``. MUST NOT contain hardcoded names — the caller
        (extraction_routing.py) resolves these from the manifest.
    worker_anchors:
        Optional list of entity names the worker already knows (from
        ``body.identity_anchors``). Unioned with store results; may be None.

    Returns
    -------
    list[str]
        Deduplicated anchor name list. Order: store results first (in query
        order), then worker_anchors not already present. Empty list when no
        anchors found and worker_anchors is None or empty.
    """
    seen: set[str] = set()
    anchors: list[str] = []

    # --- 1. Query the graph store for each identity type (per-type try/except). ---
    for identity_type in identity_types:
        try:
            rows = await store._client.query(
                store._database,
                "sql",
                (
                    "SELECT name, entity_type "
                    f"FROM {identity_type} "
                    "WHERE pipeline_run_id = :run_id"
                ),
                {"run_id": pipeline_run_id},
            )
            for row in (rows or []):
                name = row.get("name")
                if name and isinstance(name, str) and name not in seen:
                    seen.add(name)
                    anchors.append(name)
        except Exception as exc:
            logger.debug(
                "identity_anchor_queries: query for type=%r run=%r raised %r — skipping type",
                identity_type, pipeline_run_id, exc,
            )
            # Continue to the next type; partial results from prior types are kept.

    # --- 2. Union with worker_anchors (dedupe). ---
    for name in (worker_anchors or []):
        if name and isinstance(name, str) and name not in seen:
            seen.add(name)
            anchors.append(name)

    return anchors


# ---------------------------------------------------------------------------
# Task C1 — batched dense entity + per-field multi-query retrieval
# ---------------------------------------------------------------------------
# Design constraints enforced here:
#   - ONE per-run SELECT (B-tree on pipeline_run_id; no global table scan)
#   - ZERO HNSW / vector_search calls (eliminates post-filter starvation)
#   - ONE embed_texts call for all queries (entity + all fields batched)
#   - Pure scoring function (fetch separated for testability)
# ---------------------------------------------------------------------------


async def fetch_extraction_chunks_for_run(
    *,
    store: "ArcadeDBGraphStore",
    pipeline_run_id: str,
) -> "list[dict]":
    """Fetch ALL ExtractionChunk rows for a pipeline_run_id via one SQL query.

    Uses the same B-tree-indexed per-run SQL as
    ``search_extraction_chunks_direct``. Returns the raw row dicts so the
    caller (``search_extraction_chunks_dense_multi_query``) can operate on
    them as a pure in-Python scoring function without any further DB calls.

    Columns projected (mirrors the direct-path SELECT):
      @rid AS node_id, vertex_id, self_ref, chunk_text, embedding,
      page_number, modality, pipeline_run_id,
      chunk_index, source_refs, token_count

    Returns
    -------
    list[dict]
        All rows for the run, ordered by self_ref ASC for stable iteration.
        Empty list when no chunks exist for the run.

    MUST NOT call vector_search or any HNSW path — see module docstring.
    """
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
    return rows or []


# ---------------------------------------------------------------------------
# Task C6 — multi-channel retrieval orchestrator
# ---------------------------------------------------------------------------
# Design constraints enforced here:
#   - Exactly ONE per-run SQL SELECT (via fetch_extraction_chunks_for_run)
#   - ZERO HNSW / vector_search calls
#   - Runs C1 (dense multi-query) + C2 (lexical) + C3 (pattern) on the same rows
#   - C4 merge_candidates produces the unified MergedCandidate pool
#   - Pool ordered by best dense score and CAPPED to cfg.top_n_candidates
#   - Does NOT score (C5) or rerank — that's post-rerank in the endpoint (C6)
# ---------------------------------------------------------------------------


@dataclass
class MultiChannelDiagnostics:
    """Diagnostics from search_extraction_chunks_multi_channel.

    Carries per-channel counts and pool sizes for the router diagnostics block.
    """
    raw_row_count: int             # rows fetched from the per-run SELECT
    entity_dense_count: int        # candidates from C1 entity-dense channel
    field_dense_total_count: int   # sum of all per-field dense candidates (pre-merge)
    lexical_hit_count: int         # chunks with at least one lexical hit (C2)
    pattern_hit_count: int         # chunks with at least one pattern hit (C3)
    pool_size: int                 # merged pool size AFTER cap
    per_field_dense_counts: dict   # {field_name: int} — per-field candidate counts (C7)
    filter_strategy: str = "multi_channel"


async def search_extraction_chunks_multi_channel(
    retrieval_signals: "PassRetrievalSignals",
    pipeline_run_id: str,
    cfg: "RetrievalProfile",
    *,
    store: "ArcadeDBGraphStore",
    identity_anchors: "list[str] | None" = None,
) -> "tuple[list, MultiChannelDiagnostics]":
    """Multi-channel retrieval orchestrator for merged mode (Task C6 + C8).

    Implements the full C1–C4 pipeline over a single per-run SELECT with no
    HNSW / vector_search calls.

    Strategy
    --------
    1. ONE SQL SELECT via ``fetch_extraction_chunks_for_run`` (B-tree-indexed
       on pipeline_run_id; no global scan, no HNSW).
    2. C1 dense multi-query: entity + per-field vectors in ONE ``embed_texts``
       call (already batched inside ``search_extraction_chunks_dense_multi_query``).
    3. C2 lexical alias hits over same rows.
    4. C3 regex/pattern hits over same rows.
    5. C4 ``merge_candidates`` — section_meta={} / table_meta={} (Phase D deferred).
    6. C8 identity-anchor channel (OPPORTUNISTIC, NON-BLOCKING):
       When ``identity_anchors`` is non-empty:
       a. Dense sub-channel: embed anchor names, score against chunk matrix,
          add as ``field_dense["identity_anchor"]`` (source tag
          ``"field:identity_anchor"`` in MergedCandidate.retrieval_sources).
       b. Lexical sub-channel: scan merged pool for chunks whose text contains
          any anchor name (case-insensitive); add ``"identity_anchor"`` to their
          ``retrieval_sources`` set and increment ``alias_hits`` by the hit count.
       When ``identity_anchors`` is empty or None: step 6 is a clean no-op;
       the rest of the pipeline is byte-identical to the no-anchor case.
    7. Order merged pool by best dense score (vector_score, descending; None last).
    8. Cap to ``cfg.top_n_candidates``.

    Parameters
    ----------
    identity_anchors:
        Optional list of entity names (from C8 identity-anchor channel).  When
        non-empty, adds both dense and lexical signals for the named entities.
        When None or empty, this parameter has NO effect (byte-identical to
        pre-C8 behaviour for all existing callers).

    Returns
    -------
    tuple[list[MergedCandidate], MultiChannelDiagnostics]
        - Capped MergedCandidate pool, ready for rerank + C5 scoring in the endpoint.
        - Diagnostics object with per-channel counts.

    MUST NOT call vector_search or any HNSW path — see module docstring.
    """
    import unicodedata

    from app.services.extraction_candidate_scoring import (
        MergedCandidate,
        merge_candidates,
    )
    from app.services.extraction_lexical_search import (
        lexical_hit_counts,
        pattern_hit_counts,
    )

    # ------------------------------------------------------------------
    # 1. ONE per-run SELECT — no HNSW, no vector_search.
    # ------------------------------------------------------------------
    rows = await fetch_extraction_chunks_for_run(
        store=store,
        pipeline_run_id=pipeline_run_id,
    )
    raw_row_count = len(rows)

    if not rows:
        return [], MultiChannelDiagnostics(
            raw_row_count=0,
            entity_dense_count=0,
            field_dense_total_count=0,
            lexical_hit_count=0,
            pattern_hit_count=0,
            pool_size=0,
            per_field_dense_counts={},
        )

    # ------------------------------------------------------------------
    # 2. C1 — batched dense multi-query (entity + per-field vectors).
    #    ONE embed_texts call inside search_extraction_chunks_dense_multi_query.
    # ------------------------------------------------------------------
    entity_dense, field_dense = await search_extraction_chunks_dense_multi_query(
        retrieval_signals, rows, cfg
    )

    entity_dense_count = len(entity_dense)
    field_dense_total_count = sum(len(v) for v in field_dense.values())

    # ------------------------------------------------------------------
    # 3. C2 — lexical alias hits (pure, no DB calls).
    # ------------------------------------------------------------------
    field_queries = retrieval_signals.field_queries
    lex_hits = lexical_hit_counts(rows, field_queries)
    lexical_hit_count = sum(
        1 for v in lex_hits.values() if v.get("alias_hits", 0) > 0
    )

    # ------------------------------------------------------------------
    # 4. C3 — regex/pattern hits (pure, no DB calls).
    # ------------------------------------------------------------------
    pat_hits = pattern_hit_counts(rows, field_queries, cfg.pattern_hit_limit)
    pattern_hit_count = sum(
        1 for v in pat_hits.values() if v.get("pattern_hits", 0) > 0
    )

    # ------------------------------------------------------------------
    # 5. C4 — merge all channels into a unified MergedCandidate pool.
    #    section_meta={} and table_meta={} — Phase D deferred.
    # ------------------------------------------------------------------
    merged_pool: list[MergedCandidate] = merge_candidates(
        entity_dense=entity_dense,
        field_dense=field_dense,
        lexical_hits=lex_hits,
        pattern_hits=pat_hits,
        section_meta={},
        table_meta={},
    )

    # ------------------------------------------------------------------
    # 6. C8 identity-anchor channel (OPPORTUNISTIC, NON-BLOCKING).
    #    Only runs when identity_anchors is non-empty.  Empty → byte-identical
    #    to pre-C8 behaviour.
    # ------------------------------------------------------------------
    if identity_anchors:
        # --- 6a. Dense sub-channel: embed anchor names, score vs chunks. ---
        # We embed the anchor name texts and compute cosine scores against the
        # pre-fetched rows.  Results are added to field_dense under the
        # "_identity_anchor" key so merge_candidates tags them as
        # "field:_identity_anchor" in retrieval_sources.
        # (The post-merge lexical pass below additionally adds the clean
        #  "identity_anchor" tag for chunks whose text contains anchor names.)
        try:
            import numpy as np
            from app.services.graph_store import GraphEntityResult

            anchor_vecs = embed_texts(identity_anchors, query=True)
            anchor_matrix = np.asarray(anchor_vecs, dtype=np.float32)
            anorm = np.linalg.norm(anchor_matrix, axis=1, keepdims=True)
            anorm[anorm == 0] = 1
            anchor_matrix = anchor_matrix / anorm

            valid_rows = [r for r in rows if r.get("embedding")]
            if valid_rows:
                chunk_matrix = np.asarray(
                    [r["embedding"] for r in valid_rows], dtype=np.float32
                )
                cnorm = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
                cnorm[cnorm == 0] = 1
                chunk_matrix = chunk_matrix / cnorm

                # (N_chunks × N_anchors) scores
                anchor_scores = chunk_matrix @ anchor_matrix.T

                # Best anchor score per chunk
                best_anchor_scores = anchor_scores.max(axis=1)  # (N_chunks,)

                vertex_ids = np.asarray(
                    [r.get("vertex_id", "") for r in valid_rows]
                )
                k = min(cfg.field_query_top_k, len(valid_rows))
                order = np.lexsort((vertex_ids, -best_anchor_scores))[:k]

                anchor_dense_results: list[GraphEntityResult] = []
                for idx in order.tolist():
                    row = valid_rows[idx]
                    score = float(best_anchor_scores[idx])
                    anchor_dense_results.append(
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
                                "chunk_index": row.get("chunk_index"),
                                "source_refs": row.get("source_refs"),
                                "token_count": row.get("token_count"),
                            },
                        )
                    )

                if anchor_dense_results:
                    # Re-run merge to add anchor dense channel.  Passing
                    # field_dense with the anchor sub-channel added; lex/pat
                    # hits unchanged. The anchor sub-channel tag in
                    # retrieval_sources will be "field:_identity_anchor".
                    field_dense_with_anchor = dict(field_dense)
                    field_dense_with_anchor["_identity_anchor"] = anchor_dense_results
                    merged_pool = merge_candidates(
                        entity_dense=entity_dense,
                        field_dense=field_dense_with_anchor,
                        lexical_hits=lex_hits,
                        pattern_hits=pat_hits,
                        section_meta={},
                        table_meta={},
                    )
        except Exception as exc:
            logger.debug(
                "search_extraction_chunks_multi_channel: C8 anchor dense step failed "
                "for run=%r: %r — skipping dense anchor sub-channel",
                pipeline_run_id, exc,
            )
            # merged_pool already computed from C4; continue with it unchanged.

        # --- 6b. Lexical sub-channel: tag chunks containing anchor names. ---
        # Scan merged pool; for any MergedCandidate whose chunk_text contains
        # any anchor name (NFC casefold substring), add "identity_anchor" to
        # retrieval_sources and increment alias_hits by the hit count.
        # This is a precision boost (not recall) — same design as C2.
        normalised_anchors = [
            unicodedata.normalize("NFC", a).casefold()
            for a in identity_anchors
            if a
        ]
        for mc in merged_pool:
            haystack = unicodedata.normalize("NFC", mc.chunk_text or "").casefold()
            anchor_hit_count = sum(
                1 for a in normalised_anchors if a in haystack
            )
            if anchor_hit_count > 0:
                mc.retrieval_sources.add("identity_anchor")
                mc.alias_hits += anchor_hit_count

    # ------------------------------------------------------------------
    # 7. Order by best dense score (vector_score desc; None last), then cap.
    # ------------------------------------------------------------------
    merged_pool.sort(
        key=lambda mc: (mc.vector_score is None, -(mc.vector_score or 0.0), mc.candidate_key)
    )
    capped_pool = merged_pool[: cfg.top_n_candidates]

    diag = MultiChannelDiagnostics(
        raw_row_count=raw_row_count,
        entity_dense_count=entity_dense_count,
        field_dense_total_count=field_dense_total_count,
        lexical_hit_count=lexical_hit_count,
        pattern_hit_count=pattern_hit_count,
        pool_size=len(capped_pool),
        per_field_dense_counts={
            field_name: len(results)
            for field_name, results in field_dense.items()
        },
    )
    return capped_pool, diag


async def search_extraction_chunks_dense_multi_query(
    retrieval_signals: "PassRetrievalSignals",
    rows: "list[dict]",
    cfg: "RetrievalProfile",
) -> "tuple[list[GraphEntityResult], dict[str, list[GraphEntityResult]]]":
    """Batched dense entity + per-field cosine scoring over pre-fetched rows.

    Pure scoring function — takes pre-fetched ``rows`` (from
    ``fetch_extraction_chunks_for_run``) so the DB layer is fully separated
    from scoring.  No DB calls are made here; no HNSW path is touched.

    Strategy
    --------
    1. Build the full query-text list:
         [retrieval_signals.entity_query] + [fq.query_text for fq in field_queries]
    2. ONE ``embed_texts(..., query=True)`` call → (1 + N_fields) vectors.
    3. Filter ``rows`` to those with a valid embedding; build a
       (N_chunks × dim) matrix.
    4. ONE ``chunk_matrix @ query_matrix.T`` → (N_chunks × N_queries) scores.
    5. Slice column 0 for entity candidates (top ``cfg.top_n_candidates``).
    6. Slice column i+1 for each field (top ``cfg.field_query_top_k``).
    7. Return (entity_results, {field_name: [GraphEntityResult, ...]}).

    Parameters
    ----------
    retrieval_signals:
        A ``PassRetrievalSignals`` — ``entity_query`` str + ``field_queries``
        tuple of ``FieldRetrievalQuery``. ``field_queries`` must already
        exclude identity / INTERNAL fields (``build_retrieval_profile``
        enforces this).
    rows:
        Pre-fetched ExtractionChunk row dicts from
        ``fetch_extraction_chunks_for_run``.  May be empty.
    cfg:
        A ``RetrievalProfile`` (or any object with ``top_n_candidates: int``
        and ``field_query_top_k: int``).

    Returns
    -------
    tuple[list[GraphEntityResult], dict[str, list[GraphEntityResult]]]
        - entity_results: top ``cfg.top_n_candidates`` chunks by entity-query
          cosine, sorted descending.
        - field_results: mapping field_name → top ``cfg.field_query_top_k``
          chunks by per-field cosine, sorted descending. Keys match
          ``retrieval_signals.field_queries[*].field_name`` exactly.
    """
    import numpy as np
    from app.services.graph_store import GraphEntityResult

    field_queries = retrieval_signals.field_queries

    # ------------------------------------------------------------------
    # 1. Build query text list — entity first, then per-field.
    # ------------------------------------------------------------------
    query_texts: list[str] = [retrieval_signals.entity_query] + [
        fq.query_text for fq in field_queries
    ]

    # ------------------------------------------------------------------
    # 2. ONE embed_texts call (batched; query=True uses BGE query prefix).
    # ------------------------------------------------------------------
    # embed_texts already L2-normalizes its output.
    query_vecs = embed_texts(query_texts, query=True)
    query_matrix = np.asarray(query_vecs, dtype=np.float32)
    # Defensive re-normalize in case of writer-side drift.
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_matrix = query_matrix / norms

    # ------------------------------------------------------------------
    # 3. Filter rows to those with valid embeddings; build chunk matrix.
    # ------------------------------------------------------------------
    valid_rows = [r for r in rows if r.get("embedding")]

    if not valid_rows:
        # No scorable chunks — return empties preserving field key set.
        entity_results: list[GraphEntityResult] = []
        field_results: dict[str, list[GraphEntityResult]] = {
            fq.field_name: [] for fq in field_queries
        }
        return entity_results, field_results

    chunk_matrix = np.asarray(
        [r["embedding"] for r in valid_rows], dtype=np.float32
    )
    # Defensive normalize chunk embeddings (bge-m3 emits normalized vectors;
    # protects against writer-side drift).
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    chunk_norms[chunk_norms == 0] = 1
    chunk_matrix = chunk_matrix / chunk_norms

    # ------------------------------------------------------------------
    # 4. ONE matrix multiply → (N_chunks × N_queries) cosine scores.
    # ------------------------------------------------------------------
    # chunk_matrix: (N_chunks × dim), query_matrix: (N_queries × dim)
    # scores[i, j] = cosine(chunk_i, query_j)
    scores = chunk_matrix @ query_matrix.T  # shape: (N_chunks, N_queries)

    # ------------------------------------------------------------------
    # 5. Helper: top-k from one score column as GraphEntityResult list.
    # ------------------------------------------------------------------
    def _top_k_results(col_idx: int, k: int) -> list[GraphEntityResult]:
        col_scores = scores[:, col_idx]  # (N_chunks,)
        # Stable sort: primary = -score (descending), secondary = vertex_id ASC.
        vertex_ids = np.asarray([r.get("vertex_id", "") for r in valid_rows])
        order = np.lexsort((vertex_ids, -col_scores))[:k]
        results: list[GraphEntityResult] = []
        for idx in order.tolist():
            row = valid_rows[idx]
            score = float(col_scores[idx])
            results.append(
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
                        "chunk_index": row.get("chunk_index"),
                        "source_refs": row.get("source_refs"),
                        "token_count": row.get("token_count"),
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # 6. Entity candidates (column 0).
    # ------------------------------------------------------------------
    entity_results = _top_k_results(0, cfg.top_n_candidates)

    # ------------------------------------------------------------------
    # 7. Per-field candidates (columns 1 … N_fields).
    # ------------------------------------------------------------------
    field_results = {
        fq.field_name: _top_k_results(i + 1, cfg.field_query_top_k)
        for i, fq in enumerate(field_queries)
    }

    return entity_results, field_results
