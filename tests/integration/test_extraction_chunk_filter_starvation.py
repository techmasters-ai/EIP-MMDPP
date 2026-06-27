"""Integration test: ExtractionChunk filter-starvation adversarial test.

VR Phase C.1 — rev 8 M8 load-bearing adversarial test (filter-starvation gate)
             + rev 11 TestOverFetchPostFilterStrategy (Option b fallback proof).

PURPOSE
-------
Two test classes in this file serve complementary roles:

1. ``test_extraction_chunk_filter_starvation`` (the acceptance gate):
   Validates that bare ``vector_search(filters={"pipeline_run_id": ...})``
   FAILS in the adversarial scenario. This is the bug-catcher — it catches
   anyone accidentally relying on filtered HNSW for VR queries. It is
   EXPECTED to fail (ArcadeDB post-filters, confirmed by rev 11 investigation).

   Originally written to detect IF ArcadeDB applies the WHERE filter before
   HNSW (it does not). The adversarial test PROVES the starvation pattern:
     1. HNSW returns top-20 globally nearest chunks.
     2. The 100 "wrong-run" chunks dominate (cosine ~0.987 to query).
     3. Filter reduces the result to 0 "right-run" chunks (cosine ~0.000).
     4. Assertion that all 5 right-run chunks are present → FAILS.

2. ``TestOverFetchPostFilterStrategy`` (rev 11 Option b proof):
   Uses the SAME adversarial dataset and calls ``search_extraction_chunks()``
   instead of bare ``vector_search(filters=...)``. The over-fetch helper
   fetches top_k=500 (unfiltered), then post-filters in Python. This reliably
   returns all 5 right-run chunks because the unfiltered pool is large enough
   to include them alongside the 100 wrong-run chunks.

REV 11 FINDING
--------------
ArcadeDB's filterable HNSW is a confirmed P1 gap (documented in
``docker/arcadedb/repo/docs/arcadedb-vs-leading-vector-dbms.md``):
"JVector treats [RIDBitsFilter] as a per-point skip during traversal."
This means HNSW top-K is computed globally, then the WHERE filter is applied
as a post-step. Alternative query forms investigated (bounded 1-2h, rev 11):
  - ``SELECT ... FROM ExtractionChunk WHERE pipeline_run_id = :rid ORDER BY
    vectorSimilarity(embedding, :q) LIMIT :k`` — same post-HNSW behavior.
  - MATCH syntax with index hints — no filtered-traversal support in SQL layer.
  - Native ``VECTOR_SIMILARITY(filter=...)`` — does not exist in ArcadeDB.
  - Composite HNSW index on (pipeline_run_id, embedding) — HNSW indexes are
    on single properties; composite vector indexes not supported.
None of these query forms pass this starvation test. Fallback: Option (b),
implemented in ``app/services/extraction_chunk_search.py``.

SKIP CONDITION
--------------
Skipped if ArcadeDB is not reachable (mirrors existing integration test pattern).
"""
from __future__ import annotations

import math
import random
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

# ---------------------------------------------------------------------------
# Embedding geometry helpers
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 1024  # matches bge-m3 + ExtractionChunk.embedding dim=1024


def _unit(v: list[float]) -> list[float]:
    """Return unit-normalized copy of v."""
    mag = math.sqrt(sum(x * x for x in v))
    if mag == 0:
        return v
    return [x / mag for x in v]


def _random_unit(dim: int, rng: random.Random) -> list[float]:
    """Random unit vector of dimension dim."""
    v = [rng.gauss(0, 1) for _ in range(dim)]
    return _unit(v)


def _near_vector(base: list[float], noise_scale: float, rng: random.Random) -> list[float]:
    """Return a unit vector close to `base`.

    In dim=1024, noise_scale=0.005 yields cosine ≈ 0.987 to base.
    Formula: expected_cosine ≈ 1 / sqrt(1 + dim * noise_scale²).
    At noise_scale=0.05 (the intuitive choice) cosine drops to ~0.53 in 1024-D
    due to high-dimensional concentration — always calibrate with this formula.
    """
    noise = [rng.gauss(0, noise_scale) for _ in range(len(base))]
    perturbed = [b + n for b, n in zip(base, noise)]
    return _unit(perturbed)


def _far_vector(dim: int, query: list[float], rng: random.Random) -> list[float]:
    """Return a unit vector orthogonal to query (cosine ≈ 0).

    Strategy: take a random vector and remove its projection onto the query.
    This ensures the far vectors are genuinely distant from the query, making
    the adversarial gap between wrong-run (cosine ~0.987) and right-run
    (cosine ~0.000) as wide as possible.
    """
    rand = _random_unit(dim, rng)
    # Remove projection onto query → strict orthogonal component
    proj = sum(r * q for r, q in zip(rand, query))
    ortho = _unit([r - proj * q for r, q in zip(rand, query)])
    return ortho  # cosine to query ≈ 0


# ---------------------------------------------------------------------------
# ArcadeDB insert helpers (direct SQL, bypass graph_store upsert machinery)
# ---------------------------------------------------------------------------


def _insert_extraction_chunk(
    store: "ArcadeDBGraphStore",
    *,
    vertex_id: str,
    pipeline_run_id: str,
    document_id: str,
    self_ref: str,
    chunk_text: str,
    embedding: list[float],
    page_number: int = 1,
    modality: str = "text",
) -> None:
    """Insert a single ExtractionChunk vertex via synchronous ArcadeDB SQL."""
    sql = (
        "INSERT INTO ExtractionChunk SET "
        "vertex_id = :vertex_id, "
        "pipeline_run_id = :pipeline_run_id, "
        "document_id = :document_id, "
        "self_ref = :self_ref, "
        "chunk_text = :chunk_text, "
        "embedding = :embedding, "
        "page_number = :page_number, "
        "modality = :modality, "
        "created_at = sysdate()"
    )
    store._client.command_sync(
        store._database,
        "sql",
        sql,
        {
            "vertex_id": vertex_id,
            "pipeline_run_id": pipeline_run_id,
            "document_id": document_id,
            "self_ref": self_ref,
            "chunk_text": chunk_text,
            "embedding": embedding,
            "page_number": page_number,
            "modality": modality,
        },
    )


def _delete_chunks_by_run_id(store: "ArcadeDBGraphStore", run_id: str) -> None:
    """Delete all ExtractionChunk rows for a given pipeline_run_id."""
    store._client.command_sync(
        store._database,
        "sql",
        "DELETE FROM ExtractionChunk WHERE pipeline_run_id = :run_id",
        {"run_id": run_id},
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "ArcadeDB applies WHERE filter POST-HNSW top-K (documented P1 gap in "
        "docker/arcadedb/repo/docs/arcadedb-vs-leading-vector-dbms.md — JVector "
        "RIDBitsFilter is per-point post-skip, not pre-filter). VR uses "
        "search_extraction_chunks() over-fetch + Python post-filter as the "
        "production strategy (see TestOverFetchPostFilterStrategy below for the "
        "passing acceptance gate). This bare-vector_search test is retained as an "
        "XFAIL canary: if ArcadeDB ever ships filtered HNSW traversal upstream, "
        "this will surface as XPASS and we can simplify the worker code."
    ),
    strict=False,
)
def test_extraction_chunk_filter_starvation(arcadedb_store: "ArcadeDBGraphStore"):
    """Filtered vector_search must return right-run chunks even when wrong-run chunks
    are globally closer to the query (adversarial filter-starvation scenario).

    Currently EXPECTED TO FAIL — ArcadeDB post-filters HNSW results. See xfail
    decorator above. TestOverFetchPostFilterStrategy is the green acceptance gate
    for the actual VR strategy (search_extraction_chunks).
    """
    import asyncio

    rng = random.Random(42)  # deterministic for reproducibility
    run_suffix = uuid.uuid4().hex[:8]
    wrong_run_id = f"filter-starvation-wrong-{run_suffix}"
    right_run_id = f"filter-starvation-right-{run_suffix}"
    doc_id = f"test-doc-{run_suffix}"

    # Build adversarial query vector
    query_vector = _random_unit(_EMBEDDING_DIM, rng)

    try:
        # Ensure schema has ExtractionChunk (idempotent)
        try:
            asyncio.get_event_loop().run_until_complete(
                arcadedb_store._client.command(
                    arcadedb_store._database, "sql",
                    "CREATE VERTEX TYPE ExtractionChunk IF NOT EXISTS"
                )
            )
        except Exception:
            pass  # type already exists — fine

        # -----------------------------------------------------------------
        # Insert 100 "wrong-run" chunks adversarially near the query.
        # noise_scale=0.005 → cosine ≈ 0.987 in dim=1024.
        # (noise_scale=0.05 only yields cosine ≈ 0.53 due to high-dim
        # concentration — calibrate with 1/sqrt(1 + dim*scale²).)
        # -----------------------------------------------------------------
        for i in range(100):
            emb = _near_vector(query_vector, noise_scale=0.005, rng=rng)
            _insert_extraction_chunk(
                arcadedb_store,
                vertex_id=f"{wrong_run_id}:#/texts/{i}",
                pipeline_run_id=wrong_run_id,
                document_id=doc_id,
                self_ref=f"#/texts/{i}",
                chunk_text=f"wrong chunk {i}",
                embedding=emb,
                page_number=1,
                modality="text",
            )

        # -----------------------------------------------------------------
        # Insert 5 "right-run" chunks orthogonal to the query.
        # cosine to query ≈ 0.00 (strictly orthogonal, not just half-way).
        # The adversarial gap is wrong=0.987 vs right=0.000.
        # -----------------------------------------------------------------
        right_self_refs = [f"#/texts/{200 + j}" for j in range(5)]
        for j, self_ref in enumerate(right_self_refs):
            emb = _far_vector(_EMBEDDING_DIM, query_vector, rng)
            _insert_extraction_chunk(
                arcadedb_store,
                vertex_id=f"{right_run_id}:{self_ref}",
                pipeline_run_id=right_run_id,
                document_id=doc_id,
                self_ref=self_ref,
                chunk_text=f"right chunk {j}",
                embedding=emb,
                page_number=2,
                modality="text",
            )

        # -----------------------------------------------------------------
        # Filtered search: top_k=20, only pipeline_run_id=right_run_id
        # -----------------------------------------------------------------
        results = asyncio.get_event_loop().run_until_complete(
            arcadedb_store.vector_search(
                vertex_type="ExtractionChunk",
                embedding_property="embedding",
                query_vector=query_vector,
                top_k=20,
                score_threshold=0.0,
                filters={"pipeline_run_id": right_run_id},
            )
        )

        returned_run_ids = {
            r.properties.get("pipeline_run_id") for r in results
        }
        returned_self_refs = {
            r.properties.get("self_ref") for r in results
        }

        # CRITICAL ASSERTION: all 5 right-run chunks must be present.
        # If this fails, ArcadeDB is post-filtering after HNSW top-K.
        # STOP: do not silently work around this — report DONE_WITH_CONCERNS.
        assert len(results) == 5, (
            f"FILTER STARVATION DETECTED: expected 5 right-run chunks, got {len(results)}.\n"
            f"Returned run_ids: {returned_run_ids}\n"
            f"Returned self_refs: {returned_self_refs}\n"
            f"This means ArcadeDB applied the pipeline_run_id WHERE filter AFTER HNSW top-K "
            f"(post-filter starvation). The Vector Router architecture requires filter-BEFORE-HNSW.\n"
            f"Per rev 8 M8: STOP and report DONE_WITH_CONCERNS. Do NOT silently work around it.\n"
            f"Options: (a) switch to exact KNN under filter, (b) over-fetch + post-filter at scale."
        )

        assert returned_run_ids == {right_run_id}, (
            f"Expected all results to have pipeline_run_id={right_run_id!r}, "
            f"got run_ids: {returned_run_ids}"
        )

        assert set(right_self_refs) == returned_self_refs, (
            f"Not all right-run self_refs returned.\n"
            f"Expected: {set(right_self_refs)}\n"
            f"Got:      {returned_self_refs}"
        )

    finally:
        # Best-effort cleanup
        try:
            _delete_chunks_by_run_id(arcadedb_store, wrong_run_id)
        except Exception:
            pass
        try:
            _delete_chunks_by_run_id(arcadedb_store, right_run_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rev 11 — Option (b) over-fetch + Python post-filter strategy proof
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOverFetchPostFilterStrategy:
    """Prove that search_extraction_chunks() solves the filter-starvation problem.

    Uses the SAME adversarial dataset as ``test_extraction_chunk_filter_starvation``:
    - 100 wrong-run chunks cosine ~0.987 to query (near-identical embeddings).
    - 5 right-run chunks cosine ~0.000 to query (strictly orthogonal).
    - desired_top_n=5, so initial overfetch top_k=max(5*10, 500)=500.

    With top_k=500 unfiltered, the search retrieves all 105 chunks (100 wrong + 5 right).
    Python post-filter by pipeline_run_id then yields exactly the 5 right-run chunks.

    This test MUST pass — it is the acceptance gate for the rev 11 Option (b) fallback.
    """

    def test_over_fetch_returns_all_right_run_chunks(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        """search_extraction_chunks() returns all 5 right-run chunks against adversarial dataset."""
        import asyncio

        # Path B (2026-05-26): call the HNSW variant directly. The public
        # ``search_extraction_chunks`` is now a dispatcher that routes to either
        # ``_hnsw`` or ``_direct`` based on settings.vector_router_retrieval_mode.
        # This test specifically exercises HNSW overfetch + post-filter behavior,
        # so call ``_hnsw`` directly to remain meaningful regardless of the
        # default mode.
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_hnsw as search_extraction_chunks,
        )

        rng = random.Random(99)  # different seed from starvation test — still deterministic
        run_suffix = uuid.uuid4().hex[:8]
        wrong_run_id = f"overfetch-wrong-{run_suffix}"
        right_run_id = f"overfetch-right-{run_suffix}"
        doc_id = f"test-doc-overfetch-{run_suffix}"

        query_vector = _random_unit(_EMBEDDING_DIM, rng)

        try:
            # Ensure schema has ExtractionChunk (idempotent)
            try:
                asyncio.get_event_loop().run_until_complete(
                    arcadedb_store._client.command(
                        arcadedb_store._database, "sql",
                        "CREATE VERTEX TYPE ExtractionChunk IF NOT EXISTS"
                    )
                )
            except Exception:
                pass  # type already exists

            # Insert 100 wrong-run chunks adversarially near the query.
            for i in range(100):
                emb = _near_vector(query_vector, noise_scale=0.005, rng=rng)
                _insert_extraction_chunk(
                    arcadedb_store,
                    vertex_id=f"{wrong_run_id}:#/texts/{i}",
                    pipeline_run_id=wrong_run_id,
                    document_id=doc_id,
                    self_ref=f"#/texts/{i}",
                    chunk_text=f"wrong chunk {i}",
                    embedding=emb,
                    page_number=1,
                    modality="text",
                )

            # Insert 5 right-run chunks orthogonal to the query.
            right_self_refs = [f"#/texts/{300 + j}" for j in range(5)]
            for j, self_ref in enumerate(right_self_refs):
                emb = _far_vector(_EMBEDDING_DIM, query_vector, rng)
                _insert_extraction_chunk(
                    arcadedb_store,
                    vertex_id=f"{right_run_id}:{self_ref}",
                    pipeline_run_id=right_run_id,
                    document_id=doc_id,
                    self_ref=self_ref,
                    chunk_text=f"right chunk {j}",
                    embedding=emb,
                    page_number=2,
                    modality="text",
                )

            # Call the over-fetch helper — must return all 5 right-run chunks.
            results, diag = asyncio.get_event_loop().run_until_complete(
                search_extraction_chunks(
                    store=arcadedb_store,
                    query_vector=query_vector,
                    pipeline_run_id=right_run_id,
                    desired_top_n=5,
                    score_threshold=None,
                )
            )

            # All 5 right-run chunks must be present.
            assert len(results) == 5, (
                f"Over-fetch helper should return 5 right-run chunks, got {len(results)}.\n"
                f"Diagnostics: {diag}\n"
                f"Returned pipeline_run_ids: "
                f"{[r.properties.get('pipeline_run_id') for r in results]}"
            )

            returned_self_refs = {r.properties.get("self_ref") for r in results}
            assert returned_self_refs == set(right_self_refs), (
                f"Not all right-run self_refs returned by over-fetch helper.\n"
                f"Expected: {set(right_self_refs)}\n"
                f"Got:      {returned_self_refs}"
            )

            returned_run_ids = {r.properties.get("pipeline_run_id") for r in results}
            assert returned_run_ids == {right_run_id}, (
                f"Over-fetch helper returned results from unexpected run_ids: {returned_run_ids}"
            )

            # Diagnostics sanity checks.
            assert diag.filter_strategy == "overfetch_post_filter"
            assert diag.post_filter_candidate_count == 5
            assert diag.short_fetch is False, (
                "short_fetch should be False when desired_top_n=5 survivors are found"
            )
            # Adversarial geometry note (rev 14): in 1024-D, the 100 wrong-run
            # chunks at cosine ~0.987 dominate the HNSW top-K hard enough that
            # some of the 5 orthogonal right-run chunks (cosine ~0.000) fall
            # below position 500. The retry at top_k=2000 reliably captures all
            # 5. The assertion is therefore `<= 1` (initial-success OR
            # one-retry), not `== 0`. The TestOverFetchRetryAndShortFetch class
            # below exercises the explicit short-fetch path separately.
            assert diag.post_filter_retry_count <= 1, (
                f"Expected retry_count<=1 (initial top_k=500 OR retry at 2000 "
                f"must cover the 5 right-run chunks); got "
                f"{diag.post_filter_retry_count}"
            )

        finally:
            # Best-effort cleanup
            try:
                _delete_chunks_by_run_id(arcadedb_store, wrong_run_id)
            except Exception:
                pass
            try:
                _delete_chunks_by_run_id(arcadedb_store, right_run_id)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Rev 14 — C.1 review nit #9: retry-branch + short_fetch=True coverage
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestOverFetchRetryAndShortFetch:
    """C.1 review nit #9: cover the retry branch where initial top_k undershoots
    and the retry-at-2000 also fails to recover desired_top_n survivors. The
    code path is structurally simple but was untested at integration level.

    Strategy:
      - Insert only 3 right-run chunks; nothing else in the DB for this run.
      - Request desired_top_n=10. This forces the initial top_k of
        max(10*10, 500)=500 to return only 3 right-run chunks → survivor count
        (3) < desired_top_n (10) → triggers retry branch.
      - The retry at top_k=2000 still finds only 3 chunks (that's all that
        exist). retry_count=1, short_fetch=True, candidate_count=3.

    NOTE: This test inserts no wrong-run chunks, so there is no adversarial
    interference. The short-fetch fires purely because there are fewer total
    right-run chunks than desired_top_n — the simplest possible trigger.
    """

    def test_retry_branch_with_short_fetch_after_retry(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        """search_extraction_chunks() fires retry and sets short_fetch=True when
        only 3 right-run chunks exist but desired_top_n=10 is requested."""
        import asyncio

        # Path B (2026-05-26): call the HNSW variant directly. The public
        # ``search_extraction_chunks`` is now a dispatcher that routes to either
        # ``_hnsw`` or ``_direct`` based on settings.vector_router_retrieval_mode.
        # This test specifically exercises HNSW overfetch + post-filter behavior,
        # so call ``_hnsw`` directly to remain meaningful regardless of the
        # default mode.
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_hnsw as search_extraction_chunks,
        )

        rng = random.Random(77)  # deterministic seed, distinct from sibling tests
        run_suffix = uuid.uuid4().hex[:8]
        right_run_id = f"retry-shortfetch-{run_suffix}"
        doc_id = f"test-doc-retry-{run_suffix}"

        query_vector = _random_unit(_EMBEDDING_DIM, rng)

        try:
            # Ensure schema has ExtractionChunk (idempotent)
            try:
                asyncio.get_event_loop().run_until_complete(
                    arcadedb_store._client.command(
                        arcadedb_store._database, "sql",
                        "CREATE VERTEX TYPE ExtractionChunk IF NOT EXISTS"
                    )
                )
            except Exception:
                pass  # type already exists — fine

            # Insert only 3 right-run chunks. No wrong-run chunks.
            # Embeddings can be random — no adversarial geometry needed here;
            # short-fetch fires because there are simply too few chunks total.
            right_self_refs = [f"#/texts/{400 + j}" for j in range(3)]
            for j, self_ref in enumerate(right_self_refs):
                emb = _random_unit(_EMBEDDING_DIM, rng)
                _insert_extraction_chunk(
                    arcadedb_store,
                    vertex_id=f"{right_run_id}:{self_ref}",
                    pipeline_run_id=right_run_id,
                    document_id=doc_id,
                    self_ref=self_ref,
                    chunk_text=f"retry chunk {j}",
                    embedding=emb,
                    page_number=1,
                    modality="text",
                )

            # Call with desired_top_n=10 but only 3 chunks exist for this run.
            # Initial top_k = max(10*10, 500) = 500 → 3 survivors < 10 → retry.
            # Retry at top_k=2000 → still only 3 survivors → short_fetch=True.
            results, diag = asyncio.get_event_loop().run_until_complete(
                search_extraction_chunks(
                    store=arcadedb_store,
                    query_vector=query_vector,
                    pipeline_run_id=right_run_id,
                    desired_top_n=10,
                    score_threshold=None,
                )
            )

            # All 3 chunks that exist must be returned (short-fetch, not zero).
            assert len(results) == 3, (
                f"Expected 3 right-run chunks (all that exist), got {len(results)}.\n"
                f"Diagnostics: {diag}"
            )

            returned_self_refs = {r.properties.get("self_ref") for r in results}
            assert returned_self_refs == set(right_self_refs), (
                f"Not all inserted self_refs returned.\n"
                f"Expected: {set(right_self_refs)}\nGot: {returned_self_refs}"
            )

            # Retry branch must have fired.
            assert diag.post_filter_retry_count == 1, (
                f"Expected retry_count=1 (retry branch fired), got {diag.post_filter_retry_count}.\n"
                f"Diagnostics: {diag}"
            )

            # short_fetch must be True — 3 survivors < desired_top_n=10.
            assert diag.short_fetch is True, (
                f"Expected short_fetch=True (3 < 10), got {diag.short_fetch}.\n"
                f"Diagnostics: {diag}"
            )

            # candidate_count reflects how many survived the post-filter.
            assert diag.post_filter_candidate_count == 3, (
                f"Expected post_filter_candidate_count=3, got {diag.post_filter_candidate_count}.\n"
                f"Diagnostics: {diag}"
            )

            # ann_top_k_requested must reflect the retry cap (2000), not the
            # initial top_k (500), because the retry branch fired.
            assert diag.ann_top_k_requested == 2000, (
                f"Expected ann_top_k_requested=2000 (retry value), "
                f"got {diag.ann_top_k_requested}.\nDiagnostics: {diag}"
            )

            # filter_strategy must always be the canonical label.
            assert diag.filter_strategy == "overfetch_post_filter", (
                f"Unexpected filter_strategy: {diag.filter_strategy}"
            )

        finally:
            # Best-effort cleanup
            try:
                _delete_chunks_by_run_id(arcadedb_store, right_run_id)
            except Exception:
                pass
