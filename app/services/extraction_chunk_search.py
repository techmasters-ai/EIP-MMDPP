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

# Router-scoring Part 1 — SECTION signal. None/[]-coalescing accessors for the
# section_path / headings columns projected onto candidate rows below. Imported
# at module scope (extraction_chunk_index does NOT import this module, so no
# cycle); keeps the property-dict construction free of inline imports.
# TABLE signal (is_table wiring): read_chunk_is_table backs the per-call
# table_meta built in _table_meta_from_rows. The Task 7 row-built gated
# candidates now go through extraction_candidate_scoring.merged_candidate_from_row
# (which owns the read_chunk_index/source_refs/token_count reads).
from app.services.extraction_chunk_index import (  # noqa: E402
    read_chunk_headings,
    read_chunk_is_table,
    read_chunk_section_path,
)

# Task 7 — G1 unit-signature recall gate (guarded-ranker spec §3). The gate
# predicate is the SAME matcher the value-grounding label uses (one matcher,
# no drift); nfc() is the label's normaliser. Neither module imports this one
# (extraction_unit_gate → field_value_grounding only) — no cycle.
from app.services.extraction_unit_gate import chunk_passes_unit_gate  # noqa: E402
from app.services.field_value_grounding import has_unit_token, nfc  # noqa: E402
# MergedCandidate + merged_candidate_from_row used by _gate_union at module level.
# extraction_candidate_scoring does NOT import this module — no cycle.
from app.services.extraction_candidate_scoring import (  # noqa: E402
    MergedCandidate,
    merged_candidate_from_row,
)

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
    # Router-scoring Part 1 — SECTION signal: ``section_path`` / ``headings``
    # are projected so candidates carry the section title (read via
    # ``read_chunk_section_path`` / ``read_chunk_headings``). Legacy rows lack
    # the columns; the accessors None/[]-coalesce.
    # TABLE signal: ``is_table`` projected (read via ``read_chunk_is_table``;
    # legacy rows False-coalesce).
    rows = await store._client.query(
        store._database,
        "sql",
        (
            "SELECT @rid AS node_id, vertex_id, self_ref, chunk_text, "
            "embedding, page_number, modality, pipeline_run_id, "
            "chunk_index, source_refs, token_count, "
            "section_path, headings, is_table "
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
                # Router-scoring Part 1 — SECTION signal. Read via the
                # None/[]-coalescing accessors so legacy rows (no columns)
                # carry None / [] rather than crashing.
                "section_path": read_chunk_section_path(row),
                "headings": read_chunk_headings(row),
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
        if not identity_type.isidentifier():
            # Defensive guard: skip types that can't be safe SQL identifiers.
            # identity_type comes from the trusted manifest but ArcadeDB cannot
            # parameterize vertex type names; an invalid identifier would corrupt
            # the query rather than raise a parameterization error.
            continue
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
      chunk_index, source_refs, token_count,
      section_path, headings,  (router-scoring Part 1 — SECTION signal)
      is_table                 (TABLE signal — read via read_chunk_is_table)

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
            "chunk_index, source_refs, token_count, "
            "section_path, headings, is_table "
            "FROM ExtractionChunk "
            "WHERE pipeline_run_id = :run_id "
            "ORDER BY self_ref ASC"
        ),
        {"run_id": pipeline_run_id},
    )
    return rows or []


def _table_meta_from_rows(rows: list[dict]) -> dict[str, str]:
    """Build the C4 ``table_meta`` dict from per-run ExtractionChunk rows.

    TABLE signal (is_table wiring): ``{candidate_key: "table"}`` for every row
    whose persisted ``is_table`` column is true (``read_chunk_is_table``;
    legacy rows False-coalesce → absent → content_type stays None).

    Key convention MUST match ``_candidate_key`` in
    ``extraction_candidate_scoring`` (vertex_id preferred; self_ref fallback)
    or the merge lookup silently misses. Built ONCE per retrieval call and
    passed at ALL merge sites (primary, C8 re-merge, state-rebuild) so the
    pool's content_type is consistent across the fallback ladder.
    """
    out: dict[str, str] = {}
    for r in rows:
        if read_chunk_is_table(r):
            key = r.get("vertex_id") or r.get("self_ref") or ""
            if key:
                out[key] = "table"
    return out


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
    pool_size: int                 # merged pool size AFTER cap (incl. gate-union extras)
    per_field_dense_counts: dict   # {field_name: int} — per-field candidate counts (C7)
    filter_strategy: str = "multi_channel"
    # Task 7 — G1 unit gate. Defaults keep pre-Task-7 constructors working.
    # unit_gate_total: rows carrying flag "unit" (digit + unit token); G1-passing count.
    # unit_gate_added: pool-EXTERNAL G1-flagged chunks admitted beyond the cap.
    #   (In-pool G1-marked chunks are NOT counted here; only net-new additions.)
    unit_gate_total: int = 0
    unit_gate_added: int = 0
    # Task 10 — G2 table gate. Symmetric semantics to the G1 fields above.
    # table_gate_total: rows carrying flag "table" (is_table + unit token, digit-free).
    # table_gate_added: pool-EXTERNAL G2-flagged chunks admitted beyond the cap.
    #   A chunk qualifying for BOTH G1+G2 is counted in BOTH *_total and BOTH
    #   *_added fields (double-count semantics; see _gate_union added_by_flag).
    table_gate_total: int = 0
    table_gate_added: int = 0


@dataclass
class MultiChannelState:
    """Intermediate state from a multi-channel retrieval pass.

    Carried by ``search_extraction_chunks_multi_channel_full`` so the E2
    fallback ladder can re-run C4 merge (relaxed_dense / lexical_table /
    identity_anchor) WITHOUT re-fetching rows or re-embedding queries.

    Constraint: MUST NOT be mutated by the fallback ladder.  All list/dict
    fields are shared references from the original pass — callers that need
    to extend them (e.g. adding identity-anchor dense results) must copy first.
    """
    rows: "list[dict]"                       # pre-fetched ExtractionChunk rows
    entity_dense: "list"                     # C1 entity dense results (GraphEntityResult list)
    field_dense: "dict[str, list]"           # C1 per-field dense results
    lex_hits: "dict[str, dict]"              # C2 lexical hit counts keyed by candidate_key
    pat_hits: "dict[str, dict]"              # C3 pattern hit counts keyed by candidate_key
    raw_row_count: int                       # snapshot of len(rows)
    # SECTION signal (anchor-independent): the pass's expected section names,
    # already NFC+casefolded ONCE — the union of each field's
    # ``json_schema_extra['retrieval']['likely_sections']``. Threaded so the
    # E2 fallback ladder (build_pool_from_multi_channel_state) can reuse it
    # without access to the PassRetrievalSignals. Empty list when the pass has
    # no likely_sections → byte-identical to the anchor-only behaviour.
    likely_sections: "list[str]" = field(default_factory=list)


def build_pool_from_multi_channel_state(
    state: MultiChannelState,
    cfg: "RetrievalProfile",
    *,
    identity_anchors: "list[str] | None" = None,
    top_n_override: "int | None" = None,
) -> "list":
    """Re-run C4 merge using pre-fetched rows and pre-computed dense results.

    Called by the E2 fallback ladder for ``relaxed_dense`` (larger top_n cap)
    and ``identity_anchor`` (with C8 anchors injected).  Skips ALL DB calls
    and ALL embed_texts calls — rows and dense vectors are reused from ``state``.

    Parameters
    ----------
    state:
        ``MultiChannelState`` from the initial ``search_extraction_chunks_multi_channel_full``
        call.  Must not be mutated.
    cfg:
        ``RetrievalProfile`` for the pass.  ``top_n_override`` shadows
        ``cfg.top_n_candidates`` when provided.
    identity_anchors:
        Optional anchor names to inject as a C8 dense + lexical sub-channel.
        Uses the pre-fetched ``state.rows`` — NO new embed call for anchors.
        When None or empty, C8 is a no-op (byte-identical to the initial pass).
    top_n_override:
        When set, overrides the pool cap (``cfg.top_n_candidates``).  Used by
        ``relaxed_dense`` to admit more candidates by raising the cap.

    Returns
    -------
    list[MergedCandidate]
        Sorted (vector_score desc, candidate_key asc) and capped merged pool.

    MUST NOT call fetch_extraction_chunks_for_run or embed_texts.
    """
    import unicodedata
    import numpy as np
    from app.services.extraction_candidate_scoring import merge_candidates
    from app.services.extraction_lexical_search import section_hit_counts
    from app.services.graph_store import GraphEntityResult

    cap = top_n_override if top_n_override is not None else cfg.top_n_candidates

    # SECTION signal (router-scoring): normalise anchors ONCE and compute the
    # anchor-vs-heading hit counts over the pre-fetched rows. The pass's
    # anchor-independent ``likely_sections`` needles ride on ``state`` (already
    # NFC+casefolded once by search_extraction_chunks_multi_channel_full), so
    # the section signal stays live across the fallback ladder even when anchors
    # are empty. Empty anchors AND empty likely_sections → sec_hits keyed-but-
    # zero, byte-identical to the old section_meta={} literal.
    normalised_anchors = [
        unicodedata.normalize("NFC", a).casefold()
        for a in (identity_anchors or [])
        if a
    ]
    sec_hits = section_hit_counts(
        state.rows,
        normalised_anchors,
        likely_sections=getattr(state, "likely_sections", ()) or (),
    )

    # TABLE signal (is_table wiring): rebuild table_meta from the pre-fetched
    # rows ONCE so the fallback-rebuilt pool carries the same content_type as
    # the primary pool. Legacy rows / norm-off runs → {} (content_type=None).
    table_meta = _table_meta_from_rows(state.rows)

    # If no identity anchors, re-run merge as-is (same dense inputs, same lex/pat).
    # This is useful for relaxed_dense (larger cap) without C8.
    field_dense_to_use = state.field_dense

    if identity_anchors:
        # Inject C8 anchor dense sub-channel using the ALREADY-COMPUTED chunk
        # embeddings from state.rows.  No re-embed of query vectors for anchors
        # (we accept this as the C8 path always embeds anchors fresh; the NO-REEMBED
        # invariant specifically covers the entity/field query vectors from C1).
        try:
            anchor_vecs = embed_texts(identity_anchors, query=True)
            anchor_matrix = np.asarray(anchor_vecs, dtype=np.float32)
            anorm = np.linalg.norm(anchor_matrix, axis=1, keepdims=True)
            anorm[anorm == 0] = 1
            anchor_matrix = anchor_matrix / anorm

            valid_rows = [r for r in state.rows if r.get("embedding")]
            if valid_rows:
                chunk_matrix = np.asarray(
                    [r["embedding"] for r in valid_rows], dtype=np.float32
                )
                cnorm = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
                cnorm[cnorm == 0] = 1
                chunk_matrix = chunk_matrix / cnorm

                anchor_scores = chunk_matrix @ anchor_matrix.T
                best_anchor_scores = anchor_scores.max(axis=1)
                vertex_ids = np.asarray([r.get("vertex_id", "") for r in valid_rows])
                k = min(cfg.field_query_top_k, len(valid_rows))
                order = np.lexsort((vertex_ids, -best_anchor_scores))[:k]

                anchor_dense_results: list[GraphEntityResult] = []
                for idx in order.tolist():
                    row = valid_rows[idx]
                    score = float(best_anchor_scores[idx])
                    anchor_dense_results.append(
                        GraphEntityResult(
                            node_id=str(
                                row.get("@rid") or row.get("node_id")
                                or row.get("vertex_id") or row["self_ref"]
                            ),
                            name=row["self_ref"],
                            entity_type="ExtractionChunk",
                            extraction_confidence=score,
                            score=score,
                            score_type="vector",
                            properties={
                                # vertex_id is the merge candidate_key (preferred
                                # over self_ref). Carrying it here aligns the
                                # bucket key with the lexical/pattern/section
                                # hit-count keys (which key by vertex_id), so
                                # those channels actually attach to this chunk.
                                "vertex_id": row.get("vertex_id"),
                                "self_ref": row["self_ref"],
                                "chunk_text": row.get("chunk_text", ""),
                                "page_number": row.get("page_number"),
                                "modality": row.get("modality"),
                                "pipeline_run_id": row.get("pipeline_run_id"),
                                "chunk_index": row.get("chunk_index"),
                                "source_refs": row.get("source_refs"),
                                "token_count": row.get("token_count"),
                                # Router-scoring Part 1 — SECTION signal
                                # (C8 anchor dense sub-channel).
                                "section_path": read_chunk_section_path(row),
                                "headings": read_chunk_headings(row),
                            },
                        )
                    )
                if anchor_dense_results:
                    field_dense_to_use = dict(state.field_dense)
                    field_dense_to_use["_identity_anchor"] = anchor_dense_results
        except Exception as exc:
            logger.debug(
                "build_pool_from_multi_channel_state: C8 anchor dense failed: %r — "
                "continuing without anchor dense sub-channel",
                exc,
            )

    merged_pool = merge_candidates(
        entity_dense=state.entity_dense,
        field_dense=field_dense_to_use,
        lexical_hits=state.lex_hits,
        pattern_hits=state.pat_hits,
        section_meta=sec_hits,
        table_meta=table_meta,
        # row_cosines=None: state reuse path does not re-run the dense scorer,
        # so row_cosines are not available here.  Task 7 (gate-union) will wire
        # them via the saved state; for now max/mean_top3 default to 0.0.
        row_cosines=None,
    )

    # C8 lexical sub-channel on merged pool (if anchors present).
    # ``normalised_anchors`` already computed above for the section signal.
    if identity_anchors:
        for mc in merged_pool:
            haystack = unicodedata.normalize("NFC", mc.chunk_text or "").casefold()
            anchor_hit_count = sum(1 for a in normalised_anchors if a in haystack)
            if anchor_hit_count > 0:
                mc.retrieval_sources.add("identity_anchor")
                # Decomposed-lexical: bump BOTH entity_anchor_text AND alias_hits
                # to preserve alias_hits == field_label_hits + entity_anchor_text.
                mc.entity_anchor_text += anchor_hit_count
                mc.alias_hits += anchor_hit_count

    merged_pool.sort(
        key=lambda mc: (mc.vector_score is None, -(mc.vector_score or 0.0), mc.candidate_key)
    )
    # TODO(Task 18): the G1 gate union (cap-exempt gated chunks; see step 7b in
    # search_extraction_chunks_multi_channel_full) is intentionally NOT applied
    # on this fallback-rebuild path — the E2 ladder already relaxes recall
    # upward (relaxed_dense raises the cap; lexical_table/identity_anchor add
    # candidates). When the Task 18 cut helper lands, route both the primary
    # and fallback pools through it so cap exemption has ONE implementation.
    return merged_pool[:cap]


# ---------------------------------------------------------------------------
# Gate helpers — extracted pre-G2 (anchor-INDEPENDENT; not a C8 sub-step)
# ---------------------------------------------------------------------------


def _gate_scan(
    rows: list[dict],
    retrieval_signals: "PassRetrievalSignals",
    cfg: "RetrievalProfile",
) -> "dict[str, set[str]]":
    """Scan all rows of the run and return a flags map.

    Returns
    -------
    dict[str, set[str]]
        Maps ``candidate_key`` → set of gate flag names (e.g. ``{"unit"}``,
        ``{"table"}``, or ``{"unit", "table"}``).  Returns ``{}`` when no gate
        is enabled or no signature is present.

    Gate rules (each flag is independent; a row may earn multiple flags):

    G1 — flag "unit"
        ``cfg.unit_gate is True`` (strict, not truthiness — MagicMock-safe) AND
        ``unit_signature`` non-empty AND chunk passes
        ``chunk_passes_unit_gate(text_nfc, unit_signature)`` (digit + unit token).

    The ``is True`` guard means MagicMock-shaped ``cfg`` objects in endpoint
    tests can never accidentally enable the gate.  Default off
    (``RetrievalProfile.unit_gate = False``) → returns ``{}``.

    .. note::
        G2 (table gate, flag ``"table"``) is added in Task 10 / Step 1 below.
        The helper signature is already flag-shaped so adding G2 requires only
        a predicate addition here — no call-site changes.
    """
    unit_signature = tuple(getattr(retrieval_signals, "unit_signature", ()) or ())
    if cfg.unit_gate is not True or not unit_signature:
        return {}

    flags_by_key: dict[str, set[str]] = {}
    for row in rows:
        key = row.get("vertex_id") or row.get("self_ref") or ""
        if not key:
            continue
        text_nfc = nfc(row.get("chunk_text") or "")
        flags: set[str] = set()

        # G1 — digit + unit token
        if chunk_passes_unit_gate(text_nfc, unit_signature):
            flags.add("unit")

        if flags:
            flags_by_key[key] = flags

    return flags_by_key


def _gate_union(
    capped_pool: "list[MergedCandidate]",
    merged_pool: "list[MergedCandidate]",
    flags_by_key: "dict[str, set[str]]",
    rows: list[dict],
    row_cosines: "dict | None",
    top_n: int,
    table_meta: "dict | None",
) -> "tuple[list[MergedCandidate], dict[str, int]]":
    """Admit gate-flagged chunks into the candidate pool, cap-exempt.

    Parameters
    ----------
    capped_pool
        The dense-sorted pool already sliced to ``top_n`` (step 7).
    merged_pool
        The FULL merged pool before the cap (sorted in dense order).
    flags_by_key
        Output of :func:`_gate_scan`; ``{}`` → returns ``(capped_pool, {})``.
    rows
        Raw per-run rows (used for group (c) — pool-absent key rebuild).
    row_cosines
        ``vertex_id``-keyed cosine dicts from the Task 6 matmul (may be None).
    top_n
        The ``cfg.top_n_candidates`` value (used to slice beyond-cap members).
    table_meta
        Unused for group (c) row-built candidates (they are pool-absent by
        definition), but accepted for signature symmetry with the primary
        merge call.  Pass ``None`` when not available.

    Returns
    -------
    tuple[list[MergedCandidate], dict[str, int]]
        * Extended pool (capped_pool + gate_extras), same ordering contract as
          the original step 7b: dense-capped first, then beyond-cap gated in
          merged_pool order, then row-built gated sorted by candidate_key.
        * ``added_by_flag`` — per-flag count of chunks ADDED to the pool
          (i.e. NOT already in ``capped_pool``).  In-pool marking is NOT
          counted here.  A chunk qualifying for multiple flags is counted
          once under EACH flag (double-count semantics; documented in
          :class:`MultiChannelDiagnostics`).

    Retrieval-source names are derived from flag names inside this function
    (``f"{flag}_gate"`` → ``"unit_gate"`` / ``"table_gate"``) so stamp sites
    never multiply across call sites.

    Dedup invariant: duplicate ``candidate_keys`` in the final pool raise
    ``RuntimeError`` (converted from the old ``assert`` so it surfaces in
    production logs rather than silently optimising away under -O).
    """
    if not flags_by_key:
        return capped_pool, {}

    gated_keys: set[str] = set(flags_by_key)
    in_pool_keys: set[str] = {mc.candidate_key for mc in capped_pool}
    gate_extras: list[MergedCandidate] = []
    # Tracks how many pool-EXTERNAL candidates were added per flag.
    added_by_flag: dict[str, int] = {}

    def _stamp(mc: "MergedCandidate", flags: "set[str]") -> None:
        for flag in flags:
            mc.gate_flags.add(flag)
            mc.retrieval_sources.add(f"{flag}_gate")

    # (a) Gated members already INSIDE the cap: mark in place — no dup.
    for mc in capped_pool:
        if mc.candidate_key in flags_by_key:
            _stamp(mc, flags_by_key[mc.candidate_key])
            # in-pool: not counted in added_by_flag

    # (b) Gated members of merged_pool BEYOND the cap: re-admit the
    #     EXISTING objects (channel evidence — field_scores, sources,
    #     hit counts — is preserved), in merged_pool (sorted) order.
    merged_keys: set[str] = {mc.candidate_key for mc in merged_pool}
    for mc in merged_pool[top_n:]:
        if mc.candidate_key in gated_keys and mc.candidate_key not in in_pool_keys:
            flags = flags_by_key[mc.candidate_key]
            _stamp(mc, flags)
            gate_extras.append(mc)
            in_pool_keys.add(mc.candidate_key)
            for flag in flags:
                added_by_flag[flag] = added_by_flag.get(flag, 0) + 1

    # (c) Gated keys absent from merged_pool entirely: build minimal
    #     MergedCandidates from the raw rows via the SHARED factory
    #     merged_candidate_from_row (also used by
    #     _build_lexical_table_candidates in extraction_routing) so
    #     row-built candidates carry correct content_type from the
    #     persisted is_table column (table_meta can never reach them —
    #     their keys are pool-absent by definition).
    #     vector_score / cosines come from the Task 6 row_cosines (keyed
    #     identically: vertex_id-or-self_ref); rows without embeddings
    #     have no row_cosines entry → vector_score None, cosines 0.0.
    missing = sorted(gated_keys - merged_keys)
    if missing:
        rows_by_key = {
            (r.get("vertex_id") or r.get("self_ref") or ""): r for r in rows
        }
        for key in missing:
            if key in in_pool_keys:  # dedup guard (unreachable by construction)
                continue
            row = rows_by_key.get(key)
            if row is None:
                continue
            flags = flags_by_key[key]
            rc = (row_cosines or {}).get(key) or {}
            gate_extras.append(
                merged_candidate_from_row(
                    row,
                    candidate_key=key,
                    vector_score=rc.get("entity_cosine"),
                    retrieval_sources={f"{flag}_gate" for flag in flags},
                    gate_flags=set(flags),
                    row_cosines=rc,
                )
            )
            in_pool_keys.add(key)
            for flag in flags:
                added_by_flag[flag] = added_by_flag.get(flag, 0) + 1

    final_pool = capped_pool + gate_extras
    final_keys = [mc.candidate_key for mc in final_pool]
    dupes = {k for k in final_keys if final_keys.count(k) > 1}
    if dupes:
        raise RuntimeError(
            f"gate union produced duplicate candidate_keys: {sorted(dupes)}"
        )

    return final_pool, added_by_flag


async def search_extraction_chunks_multi_channel_full(
    retrieval_signals: "PassRetrievalSignals",
    pipeline_run_id: str,
    cfg: "RetrievalProfile",
    *,
    store: "ArcadeDBGraphStore",
    identity_anchors: "list[str] | None" = None,
) -> "tuple[list, MultiChannelDiagnostics, MultiChannelState]":
    """Like ``search_extraction_chunks_multi_channel`` but also returns the
    intermediate ``MultiChannelState`` for use by the E2 fallback ladder.

    The state carries pre-fetched rows + pre-computed dense/lex/pat results so
    subsequent fallback levels (relaxed_dense, identity_anchor) can re-run C4
    merge WITHOUT re-fetching rows or re-embedding query vectors.

    Returns
    -------
    tuple[list[MergedCandidate], MultiChannelDiagnostics, MultiChannelState]
        Third element is the ``MultiChannelState`` for fallback reuse.

    MUST NOT call vector_search or any HNSW path — see module docstring.
    """
    import unicodedata

    from app.services.extraction_candidate_scoring import (
        MergedCandidate,
        merge_candidates,
        merged_candidate_from_row,
    )
    from app.services.extraction_lexical_search import (
        keyword_hit_counts,
        lexical_hit_counts,
        pattern_hit_counts,
        section_hit_counts,
    )

    # ------------------------------------------------------------------
    # 1. ONE per-run SELECT — no HNSW, no vector_search.
    # ------------------------------------------------------------------
    rows = await fetch_extraction_chunks_for_run(
        store=store,
        pipeline_run_id=pipeline_run_id,
    )
    raw_row_count = len(rows)

    # SECTION signal (anchor-independent): the pass's expected section names —
    # the union of each field's ``likely_sections`` already aggregated onto
    # ``retrieval_signals.likely_sections`` by build_retrieval_profile. NFC +
    # casefold ONCE (consistent with normalised_anchors below) and thread onto
    # the state so the E2 fallback ladder can reuse it. These are schema-typed
    # section-name strings (instance-free) → available even when anchors arrive
    # empty at field-pass routing time, giving the section_norm feature a live
    # source. Empty → byte-identical to the prior anchor-only behaviour.
    normalised_likely_sections = [
        unicodedata.normalize("NFC", s).casefold()
        for s in getattr(retrieval_signals, "likely_sections", ()) or ()
        if s and s.strip()
    ]

    _empty_state = MultiChannelState(
        rows=rows,
        entity_dense=[],
        field_dense={},
        lex_hits={},
        pat_hits={},
        raw_row_count=raw_row_count,
        likely_sections=normalised_likely_sections,
    )

    if not rows:
        return [], MultiChannelDiagnostics(
            raw_row_count=0,
            entity_dense_count=0,
            field_dense_total_count=0,
            lexical_hit_count=0,
            pattern_hit_count=0,
            pool_size=0,
            per_field_dense_counts={},
        ), _empty_state

    # ------------------------------------------------------------------
    # 2. C1 — batched dense multi-query (entity + per-field vectors).
    #    ONE embed_texts call inside search_extraction_chunks_dense_multi_query.
    # ------------------------------------------------------------------
    entity_dense, field_dense, row_cosines = await search_extraction_chunks_dense_multi_query(
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

    # Decomposed-lexical: per-pass keyword channel (NEW feature). Counts the
    # profile's ``lexical_keywords`` SEPARATELY from the field aliases above,
    # then folds the count into the same lex_hits payload under "keyword_hits"
    # so merge_candidates maps it to ``pass_keyword_hits`` (NOT alias_hits).
    # Empty lexical_keywords → every entry gets keyword_hits=0, byte-identical
    # to the pre-decomposition lexical payload.
    kw_hits = keyword_hit_counts(rows, cfg.lexical_keywords)
    for key, lh in lex_hits.items():
        lh["keyword_hits"] = kw_hits.get(key, {}).get("keyword_hits", 0)

    # ------------------------------------------------------------------
    # 4. C3 — regex/pattern hits (pure, no DB calls).
    # ------------------------------------------------------------------
    pat_hits = pattern_hit_counts(rows, field_queries, cfg.pattern_hit_limit)
    pattern_hit_count = sum(
        1 for v in pat_hits.values() if v.get("pattern_hits", 0) > 0
    )

    # ------------------------------------------------------------------
    # 4b. SECTION — anchor-vs-heading hit counts (router-scoring section
    #     signal). Normalise the identity anchors ONCE (NFC + casefold) and
    #     reuse the normalised list both here (section haystack = chunk
    #     headings) and below for the C8 lexical sub-channel (haystack =
    #     chunk body). When no anchors are supplied, sec_hits is {} (the
    #     section_meta read in merge_candidates then yields section_hits=0,
    #     byte-identical to the old section_meta={} literal).
    # ------------------------------------------------------------------
    normalised_anchors = [
        unicodedata.normalize("NFC", a).casefold()
        for a in (identity_anchors or [])
        if a
    ]
    # Anchor matches + likely_section matches both feed section_hits. The
    # likely_sections needles are anchor-independent (see above) so the section
    # signal is non-constant even when anchors are empty.
    sec_hits = section_hit_counts(
        rows, normalised_anchors, likely_sections=normalised_likely_sections
    )

    # Capture state for E2 fallback ladder BEFORE any mutation (C8 may mutate
    # merged_pool but the underlying entity_dense / field_dense are not mutated).
    state = MultiChannelState(
        rows=rows,
        entity_dense=entity_dense,
        field_dense=field_dense,
        lex_hits=lex_hits,
        pat_hits=pat_hits,
        raw_row_count=raw_row_count,
        likely_sections=normalised_likely_sections,
    )

    # ------------------------------------------------------------------
    # 5. C4 — merge all channels into a unified MergedCandidate pool.
    #    section_meta=sec_hits (real data); table_meta from the persisted
    #    is_table column (TABLE signal — built ONCE, reused at the C8
    #    re-merge below). Legacy rows / norm-off runs → {} (None).
    # ------------------------------------------------------------------
    table_meta = _table_meta_from_rows(rows)
    merged_pool: list[MergedCandidate] = merge_candidates(
        entity_dense=entity_dense,
        field_dense=field_dense,
        lexical_hits=lex_hits,
        pattern_hits=pat_hits,
        section_meta=sec_hits,
        table_meta=table_meta,
        row_cosines=row_cosines,
    )

    # ------------------------------------------------------------------
    # 6. C8 identity-anchor channel (OPPORTUNISTIC, NON-BLOCKING).
    #    Only runs when identity_anchors is non-empty.  Empty → byte-identical
    #    to pre-C8 behaviour.
    # ------------------------------------------------------------------
    if identity_anchors:
        # --- 6a. Dense sub-channel: embed anchor names, score vs chunks. ---
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

                anchor_scores = chunk_matrix @ anchor_matrix.T
                best_anchor_scores = anchor_scores.max(axis=1)

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
                                # vertex_id is the merge candidate_key (preferred
                                # over self_ref). Carrying it here aligns the
                                # bucket key with the lexical/pattern/section
                                # hit-count keys (which key by vertex_id), so
                                # those channels actually attach to this chunk.
                                "vertex_id": row.get("vertex_id"),
                                "self_ref": row["self_ref"],
                                "chunk_text": row.get("chunk_text", ""),
                                "page_number": row.get("page_number"),
                                "modality": row.get("modality"),
                                "pipeline_run_id": row.get("pipeline_run_id"),
                                "chunk_index": row.get("chunk_index"),
                                "source_refs": row.get("source_refs"),
                                "token_count": row.get("token_count"),
                                # Router-scoring Part 1 — SECTION signal
                                # (C8 anchor dense sub-channel).
                                "section_path": read_chunk_section_path(row),
                                "headings": read_chunk_headings(row),
                            },
                        )
                    )

                if anchor_dense_results:
                    field_dense_with_anchor = dict(field_dense)
                    field_dense_with_anchor["_identity_anchor"] = anchor_dense_results
                    merged_pool = merge_candidates(
                        entity_dense=entity_dense,
                        field_dense=field_dense_with_anchor,
                        lexical_hits=lex_hits,
                        pattern_hits=pat_hits,
                        section_meta=sec_hits,
                        # TABLE signal: same table_meta as the primary merge
                        # (step 5) — the C8 re-merge must not lose content_type.
                        table_meta=table_meta,
                        row_cosines=row_cosines,
                    )
        except Exception as exc:
            logger.debug(
                "search_extraction_chunks_multi_channel_full: C8 anchor dense step "
                "failed for run=%r: %r — skipping dense anchor sub-channel",
                pipeline_run_id, exc,
            )

        # --- 6b. Lexical sub-channel: tag chunks containing anchor names. ---
        # ``normalised_anchors`` already computed once at step 4b above.
        for mc in merged_pool:
            haystack = unicodedata.normalize("NFC", mc.chunk_text or "").casefold()
            anchor_hit_count = sum(
                1 for a in normalised_anchors if a in haystack
            )
            if anchor_hit_count > 0:
                mc.retrieval_sources.add("identity_anchor")
                # Decomposed-lexical: anchor-in-TEXT is its own feature.
                # Bump BOTH the decomposed entity_anchor_text AND the legacy
                # alias_hits to preserve the invariant
                # alias_hits == field_label_hits + entity_anchor_text.
                mc.entity_anchor_text += anchor_hit_count
                mc.alias_hits += anchor_hit_count

    # ------------------------------------------------------------------
    # Gate scan (anchor-INDEPENDENT — runs over ALL rows of the run, not
    # just the merged pool).  Returns a flags_by_key map; empty when all
    # gates are disabled (default: unit_gate=False).
    # ------------------------------------------------------------------
    flags_by_key = _gate_scan(rows, retrieval_signals, cfg)

    # ------------------------------------------------------------------
    # 7. Order by best dense score (vector_score desc; None last), then cap.
    # ------------------------------------------------------------------
    merged_pool.sort(
        key=lambda mc: (mc.vector_score is None, -(mc.vector_score or 0.0), mc.candidate_key)
    )
    capped_pool = merged_pool[: cfg.top_n_candidates]

    # ------------------------------------------------------------------
    # Gate union — cap-exempt admission (G1 + G2, Task 7 / Task 10).
    #     Final pool order: dense-capped first, then beyond-cap gated in
    #     merged_pool order, then row-built gated sorted by candidate_key.
    # ------------------------------------------------------------------
    capped_pool, added_by_flag = _gate_union(
        capped_pool,
        merged_pool,
        flags_by_key,
        rows,
        row_cosines,
        cfg.top_n_candidates,
        table_meta,
    )

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
        unit_gate_total=sum(1 for f in flags_by_key.values() if "unit" in f),
        unit_gate_added=added_by_flag.get("unit", 0),
        table_gate_total=sum(1 for f in flags_by_key.values() if "table" in f),
        table_gate_added=added_by_flag.get("table", 0),
    )
    return capped_pool, diag, state


async def search_extraction_chunks_multi_channel(
    retrieval_signals: "PassRetrievalSignals",
    pipeline_run_id: str,
    cfg: "RetrievalProfile",
    *,
    store: "ArcadeDBGraphStore",
    identity_anchors: "list[str] | None" = None,
) -> "tuple[list, MultiChannelDiagnostics]":
    """Multi-channel retrieval orchestrator for merged mode (Task C6 + C8).

    Thin 2-tuple wrapper around ``search_extraction_chunks_multi_channel_full``
    for backward compatibility with all existing callers and integration tests.
    The endpoint (E2) calls the ``_full`` variant directly to obtain the
    ``MultiChannelState`` for the fallback ladder.

    Returns
    -------
    tuple[list[MergedCandidate], MultiChannelDiagnostics]

    MUST NOT call vector_search or any HNSW path — see module docstring.
    """
    pool, diag, _state = await search_extraction_chunks_multi_channel_full(
        retrieval_signals,
        pipeline_run_id,
        cfg,
        store=store,
        identity_anchors=identity_anchors,
    )
    return pool, diag


async def search_extraction_chunks_dense_multi_query(
    retrieval_signals: "PassRetrievalSignals",
    rows: "list[dict]",
    cfg: "RetrievalProfile",
) -> "tuple[list[GraphEntityResult], dict[str, list[GraphEntityResult]], dict[str, dict[str, float]]]":
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
    7. Build ``row_cosines`` over ALL valid_rows (not just top-k) with three
       per-row stats: entity_cosine, max_field_cosine, mean_top3_field_cosine.
    8. Return (entity_results, {field_name: [GraphEntityResult, ...]}, row_cosines).

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
    tuple[list[GraphEntityResult], dict[str, list[GraphEntityResult]], dict[str, dict[str, float]]]
        - entity_results: top ``cfg.top_n_candidates`` chunks by entity-query
          cosine, sorted descending.
        - field_results: mapping field_name → top ``cfg.field_query_top_k``
          chunks by per-field cosine, sorted descending. Keys match
          ``retrieval_signals.field_queries[*].field_name`` exactly.
        - row_cosines: per-candidate-key dict of
          ``{"entity_cosine": float, "max_field_cosine": float,
             "mean_top3_field_cosine": float}`` computed over ALL valid_rows
          BEFORE any top-k slice (so chunks outside the top-k still carry
          their true cosine for diagnostics / gate logic).  Empty dict when
          valid_rows is empty.
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
        return entity_results, field_results, {}

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
                        # vertex_id is the merge candidate_key (preferred over
                        # self_ref). Carrying it here aligns the bucket key with
                        # the lexical/pattern/section hit-count keys (which key
                        # by vertex_id), so those channels attach to this chunk.
                        "vertex_id": row.get("vertex_id"),
                        "self_ref": row["self_ref"],
                        "chunk_text": row.get("chunk_text", ""),
                        "page_number": row.get("page_number"),
                        "modality": row.get("modality"),
                        "pipeline_run_id": row.get("pipeline_run_id"),
                        "chunk_index": row.get("chunk_index"),
                        "source_refs": row.get("source_refs"),
                        "token_count": row.get("token_count"),
                        # Router-scoring Part 1 — SECTION signal
                        # (dense multi-query candidates).
                        "section_path": read_chunk_section_path(row),
                        "headings": read_chunk_headings(row),
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

    # ------------------------------------------------------------------
    # 8. Build row_cosines over ALL valid_rows (Task 6 — capture-only).
    #    Computed from the UNSLICED scores matrix (this block merely sits
    #    after the top-k calls; it does not depend on their output) so chunks
    #    ranked outside the top-k still carry their true cosines for
    #    offline diagnostics and future gate logic.
    #
    #    Key convention: vertex_id preferred, self_ref fallback — identical
    #    to _candidate_key() in extraction_candidate_scoring.py.
    # ------------------------------------------------------------------
    entity_col = scores[:, 0]  # (N_chunks,)
    n_fields = len(field_queries)
    if n_fields > 0:
        field_block = scores[:, 1:]  # (N_chunks, N_fields)
        max_field_col = field_block.max(axis=1)  # (N_chunks,)
        # mean of top min(3, n_fields) per row
        k_top = min(3, n_fields)
        sorted_block = np.sort(field_block, axis=1)  # ascending
        mean_top3_col = sorted_block[:, -k_top:].mean(axis=1)  # (N_chunks,)
    else:
        max_field_col = np.zeros(len(valid_rows), dtype=np.float32)
        mean_top3_col = np.zeros(len(valid_rows), dtype=np.float32)

    row_cosines: dict[str, dict[str, float]] = {}
    for i, row in enumerate(valid_rows):
        key = row.get("vertex_id") or row["self_ref"]
        row_cosines[key] = {
            "entity_cosine": float(entity_col[i]),
            "max_field_cosine": float(max_field_col[i]),
            "mean_top3_field_cosine": float(mean_top3_col[i]),
        }

    return entity_results, field_results, row_cosines
