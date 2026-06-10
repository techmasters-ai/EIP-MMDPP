"""Unit tests for Task C1 — batched dense entity + per-field multi-query retrieval.

Covers:
  - fetch_extraction_chunks_for_run: exactly one SELECT, never vector_search
  - search_extraction_chunks_dense_multi_query: batched embed_texts call,
    entity + per-field result sets, caps respected, identity fields absent

Pattern: synthetic rows with known embeddings + mocked embed_texts; zero DB
calls in the scoring function tests (pure-function separation).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_extraction_chunk_search_direct.py)
# ---------------------------------------------------------------------------

def _vec(*xs: float) -> list[float]:
    """1024-dim vector with the supplied first values, rest zero."""
    v = [0.0] * 1024
    for i, x in enumerate(xs):
        v[i] = float(x)
    return v


def _norm(v: list[float]) -> list[float]:
    """L2-normalise a vector (same as embed_texts does on output)."""
    arr = np.array(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr.tolist()


def _row(self_ref: str, embedding: list[float] | None, **extra) -> dict:
    """Build a fake ArcadeDB ExtractionChunk row.

    Matches the fetch_extraction_chunks_for_run SELECT projection:
      @rid AS node_id, vertex_id, self_ref, chunk_text, embedding,
      page_number, modality, pipeline_run_id, chunk_index, source_refs,
      token_count
    """
    return {
        "node_id": extra.pop("node_id", f"#170:{self_ref}"),
        "vertex_id": extra.pop("vertex_id", f"run-A:{self_ref}"),
        "self_ref": self_ref,
        "chunk_text": extra.pop("chunk_text", f"text for {self_ref}"),
        "embedding": embedding,
        "page_number": extra.pop("page_number", None),
        "modality": extra.pop("modality", "text"),
        "pipeline_run_id": extra.pop("pipeline_run_id", "run-A"),
        "chunk_index": extra.pop("chunk_index", -1),
        "source_refs": extra.pop("source_refs", []),
        "token_count": extra.pop("token_count", 0),
        **extra,
    }


def _fake_store(rows: list[dict]) -> Any:
    """Return a SimpleNamespace mimicking ArcadeDBGraphStore.
    Only _database and _client.query are used by the fetch helper.
    """
    client = SimpleNamespace(query=AsyncMock(return_value=rows))
    return SimpleNamespace(_database="eip_knowledge_graph", _client=client)


def _make_retrieval_signals(
    entity_query: str = "entity query text",
    field_queries: tuple = (),
):
    """Build a minimal PassRetrievalSignals."""
    from app.services.extraction_query_builder import (
        FieldRetrievalQuery,
        PassRetrievalSignals,
    )
    return PassRetrievalSignals(
        pass_name="test_pass",
        entity_doc="entity doc",
        entity_query=entity_query,
        field_queries=field_queries,
        lexical_terms=(),
        negative_terms=(),
        likely_sections=(),
        evidence_patterns=(),
    )


def _make_field_query(field_name: str, query_text: str):
    from app.services.extraction_query_builder import FieldRetrievalQuery
    return FieldRetrievalQuery(
        field_name=field_name,
        query_text=query_text,
        aliases=(),
        negative_terms=(),
        evidence_patterns=(),
        likely_sections=(),
        units=(),
    )


def _make_cfg(top_n_candidates: int = 5, field_query_top_k: int = 3) -> Any:
    """Minimal RetrievalProfile-like cfg object."""
    return SimpleNamespace(
        top_n_candidates=top_n_candidates,
        field_query_top_k=field_query_top_k,
    )


# ---------------------------------------------------------------------------
# 1. fetch_extraction_chunks_for_run — exactly one SELECT, zero vector_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_runs_exactly_one_select():
    """fetch_extraction_chunks_for_run must call store._client.query exactly
    once and never call store.vector_search."""
    from app.services.extraction_chunk_search import (
        fetch_extraction_chunks_for_run,
    )

    rows = [_row("c1", _vec(1.0))]
    store = _fake_store(rows)

    result = await fetch_extraction_chunks_for_run(
        store=store, pipeline_run_id="run-A"
    )

    assert store._client.query.call_count == 1, (
        f"Expected exactly 1 DB query; got {store._client.query.call_count}"
    )
    # Confirm no vector_search attribute was ever touched
    assert not hasattr(store, "vector_search") or not getattr(store, "vector_search", None), (
        "vector_search must not be present/called on the store"
    )


@pytest.mark.asyncio
async def test_fetch_sql_contains_where_pipeline_run_id():
    """SQL must scope by pipeline_run_id — never a global table scan."""
    from app.services.extraction_chunk_search import (
        fetch_extraction_chunks_for_run,
    )

    store = _fake_store([])
    await fetch_extraction_chunks_for_run(store=store, pipeline_run_id="run-XYZ")

    call_args = store._client.query.call_args
    sql = call_args.args[2]
    params = call_args.args[3]
    assert "WHERE pipeline_run_id = :run_id" in sql, (
        f"SQL must contain per-run WHERE clause; got: {sql!r}"
    )
    assert params == {"run_id": "run-XYZ"}, (
        f"Param dict mismatch: {params!r}"
    )


@pytest.mark.asyncio
async def test_fetch_sql_projects_all_required_columns():
    """SELECT must project vertex_id, self_ref, chunk_text, embedding,
    chunk_index, source_refs, token_count, page_number, modality."""
    from app.services.extraction_chunk_search import (
        fetch_extraction_chunks_for_run,
    )

    store = _fake_store([])
    await fetch_extraction_chunks_for_run(store=store, pipeline_run_id="run-A")

    sql = store._client.query.call_args.args[2]
    for col in ("vertex_id", "self_ref", "chunk_text", "embedding",
                "chunk_index", "source_refs", "token_count",
                "page_number", "modality"):
        assert col in sql, f"SQL must project column {col!r}; got: {sql!r}"


@pytest.mark.asyncio
async def test_fetch_returns_all_rows_unfiltered():
    """Helper must return ALL rows for the run, no scoring/truncation."""
    from app.services.extraction_chunk_search import (
        fetch_extraction_chunks_for_run,
    )

    rows = [_row(f"c{i}", _vec(float(i) / 10 + 0.1)) for i in range(20)]
    store = _fake_store(rows)

    result = await fetch_extraction_chunks_for_run(store=store, pipeline_run_id="run-A")

    assert len(result) == 20, "All rows must be returned without truncation"


@pytest.mark.asyncio
async def test_fetch_returns_empty_list_when_no_rows():
    from app.services.extraction_chunk_search import (
        fetch_extraction_chunks_for_run,
    )

    store = _fake_store([])
    result = await fetch_extraction_chunks_for_run(store=store, pipeline_run_id="run-A")
    assert result == []


# ---------------------------------------------------------------------------
# 2. search_extraction_chunks_dense_multi_query — pure scoring, no DB
# ---------------------------------------------------------------------------

# These tests NEVER touch a DB store — they call the pure scoring function
# with pre-fetched synthetic rows and a mocked embed_texts.


def _make_embedding_mock(*query_vectors: list[float]):
    """Return a mock for embed_texts that yields the supplied unit vectors
    (one per query string passed in the same order).

    embed_texts([entity_query, fq1.query_text, fq2.query_text], query=True)
    → [v0, v1, v2]
    """
    return MagicMock(return_value=[_norm(q) for q in query_vectors])


@pytest.mark.asyncio
async def test_dense_multi_embed_texts_called_once_with_all_queries():
    """embed_texts must be called ONCE with entity_query + all field query
    texts in a single list — batched, never per-field individually."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    # Two fields + entity query → 3 strings in one embed_texts call
    signals = _make_retrieval_signals(
        entity_query="entity q",
        field_queries=(
            _make_field_query("field_a", "field a query"),
            _make_field_query("field_b", "field b query"),
        ),
    )
    rows = [_row("c1", _vec(1.0))]
    cfg = _make_cfg(top_n_candidates=5, field_query_top_k=3)

    mock_embed = _make_embedding_mock(_vec(1.0), _vec(1.0), _vec(1.0))

    with patch(
        "app.services.extraction_chunk_search.embed_texts", mock_embed
    ):
        await search_extraction_chunks_dense_multi_query(signals, rows, cfg)

    assert mock_embed.call_count == 1, (
        f"embed_texts must be called exactly ONCE; called {mock_embed.call_count} times"
    )
    call_args = mock_embed.call_args
    texts_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("texts")
    assert texts_arg == ["entity q", "field a query", "field b query"], (
        f"embed_texts must receive [entity_query] + field query_texts; got {texts_arg!r}"
    )
    # Must be called with query=True
    assert call_args.kwargs.get("query") is True, (
        "embed_texts must be called with query=True"
    )


@pytest.mark.asyncio
async def test_dense_multi_never_calls_vector_search():
    """The scoring function must never call the ArcadeDB HNSW path.
    We patch vector_search and assert it is never invoked."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    signals = _make_retrieval_signals(entity_query="q")
    rows = [_row("c1", _vec(1.0))]
    cfg = _make_cfg()

    # Patch both the module-level and arcadedb vector_search
    with patch("app.services.extraction_chunk_search.embed_texts",
               _make_embedding_mock(_vec(1.0))), \
         patch("app.services.arcadedb_graph.ArcadeDBGraphStore.vector_search",
               MagicMock(side_effect=AssertionError("HNSW must not be called")),
               create=True):
        # Should complete without triggering the patched vector_search
        await search_extraction_chunks_dense_multi_query(signals, rows, cfg)


@pytest.mark.asyncio
async def test_dense_multi_returns_entity_candidates_capped_by_top_n():
    """Entity-dense candidate set is capped to cfg.top_n_candidates."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    # 10 rows, cap=3 → only top 3 returned
    rows = [_row(f"c{i}", _vec(float(i + 1) / 11, 0.5)) for i in range(10)]
    signals = _make_retrieval_signals(entity_query="entity q", field_queries=())
    cfg = _make_cfg(top_n_candidates=3, field_query_top_k=5)

    # entity query vector points in dim-0 direction
    mock_embed = _make_embedding_mock(_vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, field_results, _rc = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    assert len(entity_results) <= 3, (
        f"Entity results must be capped at top_n_candidates=3; got {len(entity_results)}"
    )
    assert len(entity_results) > 0


@pytest.mark.asyncio
async def test_dense_multi_returns_per_field_results_capped_by_field_query_top_k():
    """Each field's candidate set is capped to cfg.field_query_top_k."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    rows = [_row(f"c{i}", _vec(float(i + 1) / 11, 0.5)) for i in range(10)]
    signals = _make_retrieval_signals(
        entity_query="entity q",
        field_queries=(
            _make_field_query("field_a", "field a"),
            _make_field_query("field_b", "field b"),
        ),
    )
    cfg = _make_cfg(top_n_candidates=10, field_query_top_k=2)

    # Three queries: entity + field_a + field_b, all pointing dim-0
    mock_embed = _make_embedding_mock(_vec(1.0), _vec(1.0), _vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, field_results, _rc = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    assert "field_a" in field_results, "field_a must be in field_results dict"
    assert "field_b" in field_results, "field_b must be in field_results dict"
    assert len(field_results["field_a"]) <= 2, (
        f"field_a results must be capped at field_query_top_k=2; got {len(field_results['field_a'])}"
    )
    assert len(field_results["field_b"]) <= 2, (
        f"field_b results must be capped at field_query_top_k=2; got {len(field_results['field_b'])}"
    )


@pytest.mark.asyncio
async def test_dense_multi_identity_fields_absent_from_field_results():
    """system_name (and other _SKIP_FIELDS) must never appear in field_results.

    PassRetrievalSignals.field_queries already excludes them (build_retrieval_profile
    enforces this), so we test that the multi-query function passes through
    only the field names from field_queries — never injecting skipped names.
    """
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    # field_queries explicitly does NOT include system_name
    signals = _make_retrieval_signals(
        entity_query="entity q",
        field_queries=(
            _make_field_query("range_km", "range in km"),
            _make_field_query("frequency_ghz", "frequency in GHz"),
        ),
    )
    rows = [_row("c1", _vec(1.0))]
    cfg = _make_cfg(top_n_candidates=5, field_query_top_k=5)
    mock_embed = _make_embedding_mock(_vec(1.0), _vec(1.0), _vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, field_results, _rc = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    assert "system_name" not in field_results, (
        "system_name (identity field) must not appear in field_results"
    )
    assert set(field_results.keys()) == {"range_km", "frequency_ghz"}, (
        f"field_results keys should be exactly the field_query names; got {set(field_results.keys())!r}"
    )


@pytest.mark.asyncio
async def test_dense_multi_result_shape_matches_direct_fetch():
    """Each result in entity_results and field_results must be a
    GraphEntityResult with the same properties schema as search_extraction_chunks_direct:
    self_ref, chunk_text, page_number, modality, pipeline_run_id,
    chunk_index, source_refs, token_count."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )
    from app.services.graph_store import GraphEntityResult

    rows = [
        _row(
            "c1",
            _vec(1.0),
            chunk_text="text c1",
            page_number=3,
            modality="text",
            pipeline_run_id="run-A",
            chunk_index=5,
            source_refs=["#/texts/5"],
            token_count=42,
        )
    ]
    signals = _make_retrieval_signals(
        entity_query="entity q",
        field_queries=(_make_field_query("range_km", "range query"),),
    )
    cfg = _make_cfg(top_n_candidates=5, field_query_top_k=5)
    mock_embed = _make_embedding_mock(_vec(1.0), _vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, field_results, _rc = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    assert len(entity_results) == 1
    r = entity_results[0]
    assert isinstance(r, GraphEntityResult)
    assert r.entity_type == "ExtractionChunk"
    assert r.score is not None
    assert r.score_type == "vector"

    expected_prop_keys = {
        "self_ref", "chunk_text", "page_number", "modality",
        "pipeline_run_id", "chunk_index", "source_refs", "token_count",
    }
    missing = expected_prop_keys - set(r.properties.keys())
    assert not missing, (
        f"GraphEntityResult.properties missing keys {missing}; "
        f"got {sorted(r.properties.keys())}"
    )
    assert r.properties["chunk_text"] == "text c1"
    assert r.properties["chunk_index"] == 5
    assert r.properties["source_refs"] == ["#/texts/5"]
    assert r.properties["token_count"] == 42
    assert r.properties["page_number"] == 3


@pytest.mark.asyncio
async def test_dense_multi_scores_ranked_descending_by_cosine():
    """Results within each candidate set must be sorted score-descending."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    # Rows with varied dim-0 components; query is (1, 0, ...).
    # Post-normalization cosines are monotonically increasing with dim-0 value.
    rows = [
        _row("low",  _vec(0.20, 0.98)),   # low cosine w.r.t. (1,0,...)
        _row("mid",  _vec(0.60, 0.80)),
        _row("high", _vec(0.95, 0.31)),   # highest cosine
    ]
    signals = _make_retrieval_signals(entity_query="q", field_queries=())
    cfg = _make_cfg(top_n_candidates=5, field_query_top_k=5)
    mock_embed = _make_embedding_mock(_vec(1.0))  # entity query → (1,0,...)

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, _, _rc = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    scores = [r.score for r in entity_results]
    assert scores == sorted(scores, reverse=True), (
        f"Entity results must be sorted score-descending; got {scores}"
    )
    assert entity_results[0].name == "high", (
        f"Highest-scoring chunk must be first; got {entity_results[0].name!r}"
    )


@pytest.mark.asyncio
async def test_dense_multi_empty_rows_returns_empty_sets():
    """With zero rows, entity_results is [] and field_results values are []."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    signals = _make_retrieval_signals(
        entity_query="q",
        field_queries=(_make_field_query("f1", "field 1"),),
    )
    cfg = _make_cfg()
    mock_embed = _make_embedding_mock(_vec(1.0), _vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, field_results, row_cosines = await search_extraction_chunks_dense_multi_query(
            signals, [], cfg
        )

    assert entity_results == []
    assert "f1" in field_results
    assert field_results["f1"] == []
    assert row_cosines == {}, "empty rows → row_cosines must be {}"


@pytest.mark.asyncio
async def test_dense_multi_no_field_queries_returns_empty_field_results():
    """With no field_queries, field_results is an empty dict and only
    one embed_texts call is made (for the entity query alone)."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    signals = _make_retrieval_signals(entity_query="just entity", field_queries=())
    rows = [_row("c1", _vec(1.0))]
    cfg = _make_cfg()
    mock_embed = _make_embedding_mock(_vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, field_results, row_cosines = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    assert field_results == {}, (
        f"No field queries → field_results must be empty dict; got {field_results!r}"
    )
    assert len(entity_results) >= 0  # may have entity results
    assert row_cosines  # entity cosines populated even with no field queries


@pytest.mark.asyncio
async def test_dense_multi_skips_null_embedding_rows():
    """Rows with None embeddings must be silently skipped — not crash or
    misalign other scores."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    rows = [
        _row("good",      _vec(1.0)),
        _row("null_emb",  None),          # must be skipped
        _row("also_good", _vec(0.9, 0.4)),
    ]
    signals = _make_retrieval_signals(entity_query="q", field_queries=())
    cfg = _make_cfg(top_n_candidates=5)
    mock_embed = _make_embedding_mock(_vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, _, _rc = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    names = [r.name for r in entity_results]
    assert "null_emb" not in names, "Row with None embedding must be excluded"
    assert "good" in names
    assert "also_good" in names


# ---------------------------------------------------------------------------
# 3. One SELECT + zero HNSW end-to-end — integration of fetch + score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_one_select_zero_hnsw():
    """Full flow (fetch then score): exactly one DB SELECT, never vector_search,
    embed_texts called once with all queries batched."""
    from app.services.extraction_chunk_search import (
        fetch_extraction_chunks_for_run,
        search_extraction_chunks_dense_multi_query,
    )

    rows = [_row(f"c{i}", _vec(float(i + 1) / 6)) for i in range(5)]
    store = _fake_store(rows)

    signals = _make_retrieval_signals(
        entity_query="entity q",
        field_queries=(
            _make_field_query("field_x", "x query"),
            _make_field_query("field_y", "y query"),
        ),
    )
    cfg = _make_cfg(top_n_candidates=3, field_query_top_k=2)
    mock_embed = _make_embedding_mock(_vec(1.0), _vec(1.0), _vec(1.0))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        fetched_rows = await fetch_extraction_chunks_for_run(
            store=store, pipeline_run_id="run-A"
        )
        entity_results, field_results, _rc = await search_extraction_chunks_dense_multi_query(
            signals, fetched_rows, cfg
        )

    # Exactly one DB call total
    assert store._client.query.call_count == 1

    # embed_texts called once for the scoring step
    assert mock_embed.call_count == 1

    # Results are capped
    assert len(entity_results) <= 3
    assert len(field_results.get("field_x", [])) <= 2
    assert len(field_results.get("field_y", [])) <= 2

    # system_name absent
    assert "system_name" not in field_results

    # All entity results are GraphEntityResult
    from app.services.graph_store import GraphEntityResult
    for r in entity_results:
        assert isinstance(r, GraphEntityResult)


# ---------------------------------------------------------------------------
# SECTION wiring — real section_meta flows into MergedCandidate.section_hits
#
# Router-scoring section-signal piece. search_extraction_chunks_multi_channel_full
# now computes section_hit_counts(rows, normalised_anchors) and passes it as the
# REAL section_meta to merge_candidates (replacing the {} literal). A chunk whose
# HEADINGS contain an anchor name must surface with section_hits >= 1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_channel_full_wires_real_section_meta_into_section_hits():
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_multi_channel_full,
    )
    from app.services.ontology_bundles import RetrievalProfile

    # The matching chunk: headings carry the anchor "SNR-75"; aligned embedding
    # so the entity-dense channel surfaces it into the merged pool (buckets).
    match_vec = _norm(_vec(1.0, 0.0))
    other_vec = _norm(_vec(0.0, 1.0))
    rows = [
        _row(
            "#/texts/match",
            match_vec,
            vertex_id="run-A:chunk_match",
            chunk_text="body text without the needle",
            headings=["Chapter 2", "SNR-75 Fire Control Radar"],
            section_path="Chapter 2 > SNR-75 Fire Control Radar",
        ),
        _row(
            "#/texts/other",
            other_vec,
            vertex_id="run-A:chunk_other",
            chunk_text="unrelated body",
            headings=["Appendix A", "General Description"],
        ),
    ]
    store = _fake_store(rows)
    signals = _make_retrieval_signals(entity_query="radar fire control")
    cfg = RetrievalProfile(top_n_candidates=10)

    # embed_texts is called twice: (1) dense multi-query [entity_query] (query=True),
    # (2) anchors [SNR-75] (query=True). Return the match vector each time so the
    # matching chunk lands in the dense + anchor-dense pool.
    def _embed_side_effect(texts, query=False):
        return [match_vec for _ in texts]

    with patch(
        "app.services.extraction_chunk_search.embed_texts",
        MagicMock(side_effect=_embed_side_effect),
    ):
        pool, _diag, _state = await search_extraction_chunks_multi_channel_full(
            signals,
            "run-A",
            cfg,
            store=store,
            identity_anchors=["SNR-75"],
        )

    by_key = {mc.candidate_key: mc for mc in pool}
    assert "run-A:chunk_match" in by_key, (
        f"matching chunk must be in the merged pool; got {list(by_key)}"
    )
    assert by_key["run-A:chunk_match"].section_hits >= 1, (
        "section_hit_counts must have flowed into MergedCandidate.section_hits "
        "for the chunk whose headings contain the anchor"
    )


@pytest.mark.asyncio
async def test_multi_channel_full_wires_pass_keyword_hits_into_candidate():
    """Decomposed-lexical wiring: cfg.lexical_keywords flows through
    search_extraction_chunks_multi_channel_full into MergedCandidate.pass_keyword_hits
    (a SEPARATE feature from field_label_hits / alias_hits).

    The matching chunk's body text contains the keyword needle; the keyword
    count must land in pass_keyword_hits and must NOT leak into alias_hits."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_multi_channel_full,
    )
    from app.services.ontology_bundles import RetrievalProfile

    match_vec = _norm(_vec(1.0, 0.0))
    other_vec = _norm(_vec(0.0, 1.0))
    rows = [
        _row(
            "#/texts/match",
            match_vec,
            vertex_id="run-A:chunk_match",
            chunk_text="the missile reaches a high velocity at burnout",
        ),
        _row(
            "#/texts/other",
            other_vec,
            vertex_id="run-A:chunk_other",
            chunk_text="unrelated body without the needle",
        ),
    ]
    store = _fake_store(rows)
    signals = _make_retrieval_signals(entity_query="missile performance")
    # Per-pass keyword list set on the profile (operator-supplied in production).
    cfg = RetrievalProfile(top_n_candidates=10, lexical_keywords=["velocity"])

    with patch(
        "app.services.extraction_chunk_search.embed_texts",
        MagicMock(side_effect=lambda texts, query=False: [match_vec for _ in texts]),
    ):
        pool, _diag, _state = await search_extraction_chunks_multi_channel_full(
            signals, "run-A", cfg, store=store,
        )

    by_key = {mc.candidate_key: mc for mc in pool}
    assert "run-A:chunk_match" in by_key, (
        f"matching chunk must be in the merged pool; got {list(by_key)}"
    )
    mc = by_key["run-A:chunk_match"]
    assert mc.pass_keyword_hits == 1, (
        "cfg.lexical_keywords must flow into MergedCandidate.pass_keyword_hits"
    )
    # Keyword must NOT leak into alias_hits (legacy invariant): no field aliases
    # and no anchors here, so alias_hits stays 0.
    assert mc.alias_hits == 0
    assert mc.field_label_hits == 0


@pytest.mark.asyncio
async def test_build_pool_from_state_wires_real_section_meta():
    """The pool-builder reuse path (E2 fallback ladder) also supplies real
    section_meta — section_hits is set from the chunk's headings."""
    from app.services.extraction_chunk_search import (
        build_pool_from_multi_channel_state,
        MultiChannelState,
    )
    from app.services.graph_store import GraphEntityResult
    from app.services.ontology_bundles import RetrievalProfile

    match_vec = _norm(_vec(1.0, 0.0))
    row = _row(
        "#/texts/match",
        match_vec,
        vertex_id="run-A:chunk_match",
        chunk_text="body text without the needle",
        headings=["SNR-75 Fire Control Radar"],
    )

    # Pre-seed the dense channel so the chunk is already in buckets.
    entity_dense = [
        GraphEntityResult(
            node_id="run-A:chunk_match",
            name="#/texts/match",
            entity_type="ExtractionChunk",
            extraction_confidence=0.9,
            score=0.9,
            score_type="vector",
            properties={
                "vertex_id": "run-A:chunk_match",
                "self_ref": "#/texts/match",
                "chunk_text": "body text without the needle",
                "headings": ["SNR-75 Fire Control Radar"],
            },
        )
    ]
    state = MultiChannelState(
        rows=[row],
        entity_dense=entity_dense,
        field_dense={},
        lex_hits={},
        pat_hits={},
        raw_row_count=1,
    )
    cfg = RetrievalProfile(top_n_candidates=10)

    # identity_anchors triggers section_hit_counts; embed_texts only called for
    # the anchor dense sub-channel here (state reuse skips the query re-embed).
    with patch(
        "app.services.extraction_chunk_search.embed_texts",
        MagicMock(side_effect=lambda texts, query=False: [match_vec for _ in texts]),
    ):
        pool = build_pool_from_multi_channel_state(
            state, cfg, identity_anchors=["SNR-75"],
        )

    by_key = {mc.candidate_key: mc for mc in pool}
    assert "run-A:chunk_match" in by_key
    assert by_key["run-A:chunk_match"].section_hits >= 1


# ---------------------------------------------------------------------------
# Task 6 — row_cosines: per-row dense cosines retained from the matmul
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_cosines_uncensored_outside_top_k():
    """row_cosines must be computed over ALL valid_rows BEFORE any top-k
    slice.  Row B has field cosines that rank it OUTSIDE field_query_top_k=1
    for EVERY field column — yet row_cosines[key_B]["max_field_cosine"] must
    equal the hand-computed value (not 0.0 / absent).

    Layout (3 rows, 2 field queries; all vectors are unit-norm in 2D):
      - entity query  → (1, 0)
      - field_a query → (0, 1)
      - field_b query → (1/√2, 1/√2)
      - row A  emb  → (1, 0)   entity_cos=1.0, f_a=0.0,    f_b=1/√2
      - row B  emb  → (0, 1)   entity_cos=0.0, f_a=1.0,    f_b=1/√2
      - row C  emb  → (1/√2, 1/√2) entity_cos=1/√2, f_a=1/√2, f_b=1.0

    field_query_top_k=1 → per field column only ONE chunk is returned:
      - field_a col: B (score 1.0) → A and C excluded from field_results
      - field_b col: C (score 1.0) → A and B excluded from field_results
    But row_cosines[key_B] must still carry the true values.
    """
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    # 2D unit vectors (1024-dim with rest zeros; normalisation makes them unit).
    vec_a  = _norm(_vec(1.0, 0.0))            # row A
    vec_b  = _norm(_vec(0.0, 1.0))            # row B
    vec_c  = _norm(_vec(1.0, 1.0))            # row C → norm gives (1/√2, 1/√2, ...)

    q_entity = _norm(_vec(1.0, 0.0))          # entity query
    q_fa     = _norm(_vec(0.0, 1.0))          # field_a query
    q_fb     = _norm(_vec(1.0, 1.0))          # field_b query

    rows = [
        _row("A", vec_a, vertex_id="run-T:A"),
        _row("B", vec_b, vertex_id="run-T:B"),
        _row("C", vec_c, vertex_id="run-T:C"),
    ]
    signals = _make_retrieval_signals(
        entity_query="entity q",
        field_queries=(
            _make_field_query("field_a", "fa"),
            _make_field_query("field_b", "fb"),
        ),
    )
    cfg = _make_cfg(top_n_candidates=10, field_query_top_k=1)  # top-1 per field

    mock_embed = _make_embedding_mock(q_entity, q_fa, q_fb)

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        entity_results, field_results, row_cosines = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    # field_results["field_b"] top-k=1 → only C (score≈1.0 after normalization)
    fb_keys = {r.properties.get("vertex_id") for r in field_results["field_b"]}
    assert "run-T:B" not in fb_keys, (
        "Row B must be outside field_b top-k=1 (sanity check that the uncensored "
        "test is meaningful)"
    )

    # row_cosines must contain ALL three rows
    assert "run-T:A" in row_cosines
    assert "run-T:B" in row_cosines
    assert "run-T:C" in row_cosines

    # For row B:
    #   entity_cosine = dot((0,1,...), (1,0,...)) = 0.0
    #   field_a cosine = dot((0,1,...), (0,1,...)) = 1.0
    #   field_b cosine = dot((0,1,...), (1/√2,1/√2,...)) = 1/√2
    #   max_field_cosine = max(1.0, 1/√2) = 1.0
    #   mean_top3 (k=min(3,2)=2) = mean(1/√2, 1.0) = (1 + 1/√2) / 2
    rc_b = row_cosines["run-T:B"]
    assert isinstance(rc_b["entity_cosine"], float)
    assert isinstance(rc_b["max_field_cosine"], float)
    assert isinstance(rc_b["mean_top3_field_cosine"], float)

    assert rc_b["entity_cosine"] == pytest.approx(0.0, abs=1e-5), (
        f"entity_cosine for B must be 0.0; got {rc_b['entity_cosine']}"
    )
    assert rc_b["max_field_cosine"] == pytest.approx(1.0, abs=1e-5), (
        f"max_field_cosine for B must be 1.0 (field_a cosine); got {rc_b['max_field_cosine']}"
    )
    expected_mean = (1.0 + (1.0 / 2 ** 0.5)) / 2
    assert rc_b["mean_top3_field_cosine"] == pytest.approx(expected_mean, abs=1e-5), (
        f"mean_top3_field_cosine for B must be {expected_mean:.6f}; "
        f"got {rc_b['mean_top3_field_cosine']}"
    )

    # For row A — entity cosine should be highest (≈1.0).
    rc_a = row_cosines["run-T:A"]
    assert rc_a["entity_cosine"] == pytest.approx(1.0, abs=1e-5), (
        f"entity_cosine for A must be 1.0; got {rc_a['entity_cosine']}"
    )


@pytest.mark.asyncio
async def test_row_cosines_zero_field_queries():
    """When there are no field queries, max_field_cosine and
    mean_top3_field_cosine must both be 0.0 for every row."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    rows = [
        _row("r0", _norm(_vec(1.0, 0.0)), vertex_id="run-T:r0"),
        _row("r1", _norm(_vec(0.5, 0.5)), vertex_id="run-T:r1"),
    ]
    signals = _make_retrieval_signals(entity_query="q", field_queries=())
    cfg = _make_cfg(top_n_candidates=10)
    mock_embed = _make_embedding_mock(_norm(_vec(1.0, 0.0)))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        _, _, row_cosines = await search_extraction_chunks_dense_multi_query(
            signals, rows, cfg
        )

    assert len(row_cosines) == 2
    for key, rc in row_cosines.items():
        assert rc["max_field_cosine"] == pytest.approx(0.0), (
            f"max_field_cosine for {key} must be 0.0 with no field queries; got {rc}"
        )
        assert rc["mean_top3_field_cosine"] == pytest.approx(0.0), (
            f"mean_top3_field_cosine for {key} must be 0.0 with no field queries; got {rc}"
        )


@pytest.mark.asyncio
async def test_row_cosines_empty_rows_returns_empty_dict():
    """Empty rows → row_cosines must be {}."""
    from app.services.extraction_chunk_search import (
        search_extraction_chunks_dense_multi_query,
    )

    signals = _make_retrieval_signals(
        entity_query="q",
        field_queries=(_make_field_query("f1", "field 1"),),
    )
    cfg = _make_cfg()
    mock_embed = _make_embedding_mock(_norm(_vec(1.0)), _norm(_vec(0.0, 1.0)))

    with patch("app.services.extraction_chunk_search.embed_texts", mock_embed):
        _, _, row_cosines = await search_extraction_chunks_dense_multi_query(
            signals, [], cfg
        )

    assert row_cosines == {}, f"Empty rows must yield empty row_cosines; got {row_cosines}"
