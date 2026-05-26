"""Unit tests for the Path B direct-cosine retrieval helper.

Covers the 10 direct-function tests + 4 dispatcher tests from the
Path B handoff doc (docs/handoffs/2026-05-26-path-b-direct-cosine-retrieval.md).

Pattern: mock ``store._client.query`` to feed canned rows; verify scoring,
threshold filtering, top-N slicing, deterministic tiebreaking,
null-embedding alignment, and the dispatcher's routing decision.
"""
from __future__ import annotations

import os
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec(*xs: float) -> list[float]:
    """1024-dim vector with the supplied first values, rest zero — bge-m3 dim."""
    v = [0.0] * 1024
    for i, x in enumerate(xs):
        v[i] = float(x)
    return v


def _row(self_ref: str, embedding: list[float] | None, **extra) -> dict:
    return {
        "self_ref": self_ref,
        "chunk_text": extra.pop("chunk_text", f"text for {self_ref}"),
        "embedding": embedding,
        "page_number": extra.pop("page_number", None),
        "modality": extra.pop("modality", "text"),
        "pipeline_run_id": extra.pop("pipeline_run_id", "run-A"),
        "@rid": extra.pop("@rid", f"#100:{self_ref}"),
        **extra,
    }


def _fake_store(rows: list[dict]):
    """Return a SimpleNamespace mimicking ArcadeDBGraphStore.

    Only ``_database`` and ``_client.query`` are referenced by the
    direct path; nothing else is touched.
    """
    client = SimpleNamespace(query=AsyncMock(return_value=rows))
    return SimpleNamespace(_database="eip_knowledge_graph", _client=client)


# ---------------------------------------------------------------------------
# Direct-function tests (10 from the plan)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_returns_empty_when_no_chunks_for_run():
    """Empty result set → empty list + diagnostics fields populated."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    store = _fake_store(rows=[])
    results, diag = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=5,
    )

    assert results == []
    assert diag.filter_strategy == "direct_cosine"
    assert diag.ann_top_k_requested == 0
    assert diag.post_filter_candidate_count == 0
    assert diag.post_filter_retry_count == 0
    assert diag.short_fetch is True  # desired=5 > 0 available


@pytest.mark.asyncio
async def test_direct_returns_top_n_by_score_descending():
    """5 chunks with known embeddings, desired_top_n=3 → 3 highest-scoring."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    # Build 5 chunks. Query = (1, 0, 0, ...). Embeddings have varied x/y
    # components so cosine scores differ after normalization. (Single-axis
    # vectors all normalize to (1, 0, ...) — identical post-norm, can't
    # rank.)  Hand-computed cosines (≈ x / sqrt(x²+y²)):
    #   c1 (0.10, 0.99) → 0.100  c2 (0.90, 0.40) → 0.914
    #   c3 (0.50, 0.87) → 0.497  c4 (0.70, 0.70) → 0.707
    #   c5 (0.30, 0.95) → 0.302
    # Sorted desc: c2, c4, c3, c5, c1.
    rows = [
        _row("c1", _vec(0.10, 0.99)),
        _row("c2", _vec(0.90, 0.40)),
        _row("c3", _vec(0.50, 0.87)),
        _row("c4", _vec(0.70, 0.70)),
        _row("c5", _vec(0.30, 0.95)),
    ]
    store = _fake_store(rows=rows)

    results, diag = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=3,
    )

    assert len(results) == 3
    # Top 3 by score: c2 (0.90), c4 (0.70), c3 (0.50).
    assert [r.name for r in results] == ["c2", "c4", "c3"]
    # Scores are normalized; with these inputs they're all close to 1.0
    # but strictly descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert diag.post_filter_candidate_count == 5


@pytest.mark.asyncio
async def test_direct_filters_by_score_threshold():
    """score_threshold=0.5 → only chunks with cosine >= 0.5 are returned."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    # Build embeddings where 3 score below 0.5 and 2 score above 0.5.
    # Query = (1, 0, 0...). The cosine of an embedding (x, 0, 0...) w.r.t
    # query = x / |emb| = sign(x). To get scores < 0.5 we mix axes.
    rows = [
        _row("low1", _vec(0.3, 0.95)),     # angle dominated by y axis → low x-cosine
        _row("low2", _vec(0.2, 0.97)),
        _row("low3", _vec(0.4, 0.92)),
        _row("high1", _vec(0.95, 0.2)),     # x dominates → cos ≈ 0.978
        _row("high2", _vec(0.85, 0.3)),     # cos ≈ 0.942
    ]
    store = _fake_store(rows=rows)

    results, diag = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=5,
        score_threshold=0.5,
    )

    names = [r.name for r in results]
    # Only high-scoring chunks survive the threshold
    assert set(names) == {"high1", "high2"}
    # All returned scores >= threshold
    assert all(r.score >= 0.5 for r in results)
    # Diagnostics: candidate_count = post-threshold count (BEFORE top-N slice)
    assert diag.post_filter_candidate_count == 2


@pytest.mark.asyncio
async def test_direct_filters_by_pipeline_run_id_via_sql():
    """Verify the SQL string contains the WHERE clause + named param."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    store = _fake_store(rows=[])
    await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-XYZ", desired_top_n=5,
    )

    # SQL accessor was called with the WHERE pipeline_run_id=:run_id
    call_args = store._client.query.call_args
    assert call_args is not None
    pos_args = call_args.args
    # signature: (database, language, sql, params)
    sql = pos_args[2]
    params = pos_args[3]
    assert "WHERE pipeline_run_id = :run_id" in sql
    assert params == {"run_id": "run-XYZ"}


@pytest.mark.asyncio
async def test_direct_return_shape_matches_hnsw_path():
    """Each GraphEntityResult has node_id, name, entity_type, score,
    score_type, properties.pipeline_run_id — same as the HNSW path."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )
    from app.services.graph_store import GraphEntityResult

    rows = [_row("a", _vec(1.0), pipeline_run_id="run-A")]
    store = _fake_store(rows=rows)

    results, _ = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=1,
    )

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, GraphEntityResult)
    assert r.node_id  # non-empty string
    assert r.name == "a"
    assert r.entity_type == "ExtractionChunk"
    assert r.score is not None
    assert r.score_type == "vector"
    assert r.properties["pipeline_run_id"] == "run-A"
    assert r.properties["self_ref"] == "a"


@pytest.mark.asyncio
async def test_direct_handles_null_embedding_alignment():
    """One row has embedding=None — skipped without misaligning the
    score/row pairing for the remaining rows.

    Catches the bug Codex flagged in finding 3: if embeddings are
    filtered without filtering rows, the i-th score points at the wrong
    row.
    """
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    rows = [
        _row("a", _vec(1.0)),         # ✓
        _row("b_null", None),          # ✗ — must NOT shift scores
        _row("c", _vec(0.5, 0.5)),    # ✓ — should score lower than "a"
    ]
    store = _fake_store(rows=rows)

    results, diag = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=5,
    )

    names = [r.name for r in results]
    # "a" should be top because its embedding is most aligned with the query
    assert names[0] == "a"
    # "b_null" must NOT appear
    assert "b_null" not in names
    # "c" should be present (its embedding aligns partially)
    assert "c" in names
    # ann_top_k_requested counts ALL rows from SQL (including the null)
    assert diag.ann_top_k_requested == 3
    # candidate_count counts only valid scored rows (no threshold here)
    assert diag.post_filter_candidate_count == 2


@pytest.mark.asyncio
async def test_direct_returns_empty_when_all_embeddings_null():
    """All embeddings are None — no crash on np.linalg.norm over empty.

    Edge case guard for the bug Codex flagged where np.array of empty
    list + linalg.norm(..., axis=1) would crash.
    """
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    rows = [_row("a", None), _row("b", None)]
    store = _fake_store(rows=rows)

    results, diag = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=5,
    )

    assert results == []
    assert diag.filter_strategy == "direct_cosine"
    assert diag.ann_top_k_requested == 2
    assert diag.post_filter_candidate_count == 0


@pytest.mark.asyncio
async def test_direct_normalizes_query_vector():
    """Unnormalized query → scores still bounded in [-1, 1].

    Cosine similarity is normalize-invariant. Pass a query of length 5
    and verify scores stay in the valid cosine range.
    """
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    unnormalized_q = _vec(5.0)  # length 5, not 1
    rows = [_row("a", _vec(1.0))]
    store = _fake_store(rows=rows)

    results, _ = await search_extraction_chunks_direct(
        store=store, query_vector=unnormalized_q,
        pipeline_run_id="run-A", desired_top_n=1,
    )

    assert len(results) == 1
    assert -1.0 - 1e-6 <= results[0].score <= 1.0 + 1e-6
    # With aligned vectors after normalization, score should be ~1.0
    assert math.isclose(results[0].score, 1.0, abs_tol=1e-5)


@pytest.mark.asyncio
async def test_direct_short_fetch_when_candidates_less_than_top_n():
    """Fewer matching chunks than desired_top_n → short_fetch=True."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    rows = [_row("a", _vec(1.0)), _row("b", _vec(0.5))]
    store = _fake_store(rows=rows)

    results, diag = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=10,
    )

    assert len(results) == 2
    assert diag.short_fetch is True


@pytest.mark.asyncio
async def test_direct_tiebreaker_stable_on_equal_scores():
    """Two chunks with IDENTICAL embeddings → returned in self_ref ASC
    order, deterministic across runs."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_direct,
    )

    # Same embedding for both → identical cosine scores
    same_emb = _vec(1.0)
    # Note: SQL ORDER BY self_ref ASC is mocked here (we control row order),
    # so feed them in REVERSE alphabetical order to confirm the in-Python
    # lexsort tiebreaker still produces ASC.
    rows = [_row("z_chunk", same_emb), _row("a_chunk", same_emb)]
    store = _fake_store(rows=rows)

    results, _ = await search_extraction_chunks_direct(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=2,
    )

    # Tiebreaker: self_ref ASC → a_chunk first, z_chunk second
    assert [r.name for r in results] == ["a_chunk", "z_chunk"]


# ---------------------------------------------------------------------------
# Dispatcher tests (4 from the plan)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_routes_to_direct_when_setting_is_direct(monkeypatch):
    """settings.vector_router_retrieval_mode='direct' → diagnostics show
    filter_strategy='direct_cosine'."""
    from app.services.extraction_chunk_search import search_extraction_chunks
    from app.config import get_settings

    # Override setting + clear the lru_cache so the new value is picked up
    monkeypatch.setenv("VECTOR_ROUTER_RETRIEVAL_MODE", "direct")
    get_settings.cache_clear()

    rows = [_row("a", _vec(1.0))]
    store = _fake_store(rows=rows)
    _, diag = await search_extraction_chunks(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=1,
    )

    assert diag.filter_strategy == "direct_cosine"
    get_settings.cache_clear()  # leave clean state for downstream tests


@pytest.mark.asyncio
async def test_dispatcher_routes_to_hnsw_when_setting_is_hnsw(monkeypatch):
    """settings.vector_router_retrieval_mode='hnsw' → calls the HNSW
    helper (we don't fully exercise HNSW path; we assert direct was NOT
    called)."""
    from app.services.extraction_chunk_search import search_extraction_chunks
    from app.config import get_settings

    monkeypatch.setenv("VECTOR_ROUTER_RETRIEVAL_MODE", "hnsw")
    get_settings.cache_clear()

    direct_called = False
    real_direct = None
    import app.services.extraction_chunk_search as ecs
    real_direct = ecs.search_extraction_chunks_direct

    async def spy_direct(*a, **kw):
        nonlocal direct_called
        direct_called = True
        return await real_direct(*a, **kw)

    monkeypatch.setattr(ecs, "search_extraction_chunks_direct", spy_direct)

    # Stub the HNSW path so we don't need a real ArcadeDB
    async def stub_hnsw(**kw):
        from app.services.extraction_chunk_search import ChunkSearchDiagnostics
        return [], ChunkSearchDiagnostics(
            ann_top_k_requested=500,
            post_filter_candidate_count=0,
            post_filter_retry_count=0,
        )

    monkeypatch.setattr(ecs, "search_extraction_chunks_hnsw", stub_hnsw)

    _, diag = await search_extraction_chunks(
        store=None, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=1,
    )

    assert direct_called is False
    assert diag.filter_strategy == "overfetch_post_filter"
    get_settings.cache_clear()


def test_dispatcher_invalid_mode_fails_settings_validation(monkeypatch):
    """Settings validator rejects unknown values for the mode."""
    from pydantic import ValidationError
    from app.config import Settings, get_settings

    monkeypatch.setenv("VECTOR_ROUTER_RETRIEVAL_MODE", "banana")
    get_settings.cache_clear()

    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.delenv("VECTOR_ROUTER_RETRIEVAL_MODE", raising=False)
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_dispatcher_respects_cleared_settings_cache(monkeypatch):
    """Changing the env var between calls only takes effect if the
    settings lru_cache is cleared. Catches the cache pitfall Codex
    flagged in finding 7."""
    from app.services.extraction_chunk_search import search_extraction_chunks
    from app.config import get_settings

    rows = [_row("a", _vec(1.0))]
    store = _fake_store(rows=rows)

    # First call: mode=direct
    monkeypatch.setenv("VECTOR_ROUTER_RETRIEVAL_MODE", "direct")
    get_settings.cache_clear()
    _, diag1 = await search_extraction_chunks(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=1,
    )
    assert diag1.filter_strategy == "direct_cosine"

    # Change env WITHOUT clearing the cache — call should still route to direct
    monkeypatch.setenv("VECTOR_ROUTER_RETRIEVAL_MODE", "hnsw")
    _, diag2 = await search_extraction_chunks(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=1,
    )
    assert diag2.filter_strategy == "direct_cosine"  # still cached value

    # Now clear cache — second call should pick up the new value
    get_settings.cache_clear()
    # Need to stub HNSW for this assertion since hnsw path needs real store
    import app.services.extraction_chunk_search as ecs

    async def stub_hnsw(**kw):
        from app.services.extraction_chunk_search import ChunkSearchDiagnostics
        return [], ChunkSearchDiagnostics(
            ann_top_k_requested=500, post_filter_candidate_count=0,
            post_filter_retry_count=0,
        )

    monkeypatch.setattr(ecs, "search_extraction_chunks_hnsw", stub_hnsw)
    _, diag3 = await search_extraction_chunks(
        store=store, query_vector=_vec(1.0),
        pipeline_run_id="run-A", desired_top_n=1,
    )
    assert diag3.filter_strategy == "overfetch_post_filter"

    get_settings.cache_clear()
