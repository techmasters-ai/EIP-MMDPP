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
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    from app.services.graph_store import GraphEntityResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostics dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChunkSearchDiagnostics:
    """Diagnostic counters from a search_extraction_chunks() call.

    Ready to merge into the VR router diagnostics block:
      ``pipeline_pass_outputs.diagnostics_json.router.search``

    Fields
    ------
    ann_top_k_requested:
        The top_k value passed to the unfiltered ``vector_search`` call
        (initial attempt; if retry fired, the retry value is recorded
        instead as ``ann_top_k_requested`` will equal the larger cap).
        NOTE: if retry fired, the field captures the *larger* retry cap
        so diagnostics always reflect the largest ANN call made.
    post_filter_candidate_count:
        How many results survived the ``pipeline_run_id`` post-filter.
        May be < ``desired_top_n`` if the overfetch cap was too small
        (short-fetch case).
    post_filter_retry_count:
        0 if the initial overfetch yielded >= desired_top_n survivors;
        1 if the retry at top_k=2000 was needed.
    filter_strategy:
        Always ``"overfetch_post_filter"`` for this implementation.
        Future: ``"filtered_traversal"`` if ArcadeDB ships filterable HNSW
        (tracked upstream as P1 gap — see module docstring).
    short_fetch:
        True if post_filter_candidate_count < desired_top_n even after
        retry. Signals to the caller that the result may be incomplete.
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


async def search_extraction_chunks(
    *,
    store: "ArcadeDBGraphStore",
    query_vector: list[float],
    pipeline_run_id: str,
    desired_top_n: int,
    score_threshold: float | None = None,
) -> "tuple[list[GraphEntityResult], ChunkSearchDiagnostics]":
    """Vector search over ExtractionChunk isolated by pipeline_run_id.

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
