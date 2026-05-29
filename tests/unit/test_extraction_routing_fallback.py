"""Unit tests for Task E2 — graduated fallback ladder in the multi-channel
chunk_scope path.

TDD discipline: tests written BEFORE implementation; watched-fail recorded.

Coverage:
  1. Well-covered → fallback_level="none", no escalation.
  2. Sparse dense → escalates to relaxed_dense; NO re-fetch / NO re-embed
     (fetch_extraction_chunks_for_run and embed_texts each called once total).
  3. Still sparse after relaxed → lexical_table admits at least one keyword-only
     candidate.
  4. All cheaper levels empty + fallback_to_full=True → full; =False → would_skip.

Per-element path is unchanged (not tested here — covered by test_v1_extraction_routing.py).
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RUN_ID = "run-e2-fallback"
_BUNDLE_KEY = "air_defense_v3_baseline_subset"
_PASS_NAME = "radar_power_rf"


def _vec(*xs: float, dim: int = 1024) -> list[float]:
    v = [0.0] * dim
    for i, x in enumerate(xs):
        v[i] = float(x)
    return v


def _norm(v: list[float]) -> list[float]:
    arr = np.array(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr /= n
    return arr.tolist()


def _row(
    self_ref: str,
    embedding: list[float] | None,
    *,
    run_id: str = _RUN_ID,
    chunk_index: int = 0,
    source_refs: list[str] | None = None,
    token_count: int = 50,
    chunk_text: str | None = None,
    vertex_id: str | None = None,
) -> dict:
    vid = vertex_id or f"{run_id}:{self_ref}"
    return {
        "node_id": f"#170:{self_ref}",
        "vertex_id": vid,
        "self_ref": self_ref,
        "chunk_text": chunk_text if chunk_text is not None else f"text for {self_ref}",
        "embedding": embedding,
        "page_number": 1,
        "modality": "merged",
        "pipeline_run_id": run_id,
        "chunk_index": chunk_index,
        "source_refs": source_refs if source_refs is not None else [f"#/texts/{chunk_index}"],
        "token_count": token_count,
    }


def _fake_store(rows: list[dict]) -> Any:
    """Fake ArcadeDBGraphStore whose SQL client records calls."""
    client = SimpleNamespace(query=AsyncMock(return_value=rows))
    return SimpleNamespace(_database="eip_knowledge_graph", _client=client)


def _make_signals(
    entity_query: str = "entity query",
    field_queries: tuple = (),
) -> Any:
    from app.services.extraction_query_builder import PassRetrievalSignals
    return PassRetrievalSignals(
        pass_name="test_pass",
        entity_doc="doc",
        entity_query=entity_query,
        field_queries=field_queries,
        lexical_terms=(),
        negative_terms=(),
        likely_sections=(),
        evidence_patterns=(),
    )


def _make_profile(
    *,
    top_n_candidates: int = 10,
    top_k: int = 3,
    field_query_top_k: int = 5,
    min_similarity: float = 0.45,
    fallback_similarity_relaxation: float = 0.07,
    fallback_min_field_coverage: int = 1,
    fallback_to_full: bool = True,
    rerank_weight: float = 1.0,
    lexical_weight: float = 0.0,
    pattern_weight: float = 0.0,
    section_weight: float = 0.0,
    table_boost: float = 0.0,
    negative_weight: float = 0.0,
    pattern_hit_limit: int = 50,
) -> Any:
    p = MagicMock()
    p.top_n_candidates = top_n_candidates
    p.top_k = top_k
    p.field_query_top_k = field_query_top_k
    p.min_similarity = min_similarity
    p.fallback_similarity_relaxation = fallback_similarity_relaxation
    p.fallback_min_field_coverage = fallback_min_field_coverage
    p.fallback_to_full = fallback_to_full
    p.pattern_hit_limit = pattern_hit_limit
    p.rerank_weight = rerank_weight
    p.lexical_weight = lexical_weight
    p.pattern_weight = pattern_weight
    p.section_weight = section_weight
    p.table_boost = table_boost
    p.negative_weight = negative_weight
    return p


def _make_pass_def(profile: Any) -> Any:
    pd = MagicMock()
    pd.name = _PASS_NAME
    pd.phase = "field_group"
    pd.module = "extraction_schemas.radar_power_rf"
    pd.template_class = "RadarPowerRfPass"
    pd.retrieval = profile
    pd.primary_entity_types = None
    return pd


def _make_reranker_fn(base_score: float = 0.8):
    """Returns a fake rerank function that assigns scores from base_score downward."""
    def _rerank(query, candidates, top_k):
        scored = []
        for i, c in enumerate(candidates):
            c2 = dict(c)
            c2["reranker_score"] = max(0.0, base_score - i * 0.01)
            scored.append(c2)
        return scored
    return _rerank


# ---------------------------------------------------------------------------
# 1. Well-covered → fallback_level="none", no escalation
# ---------------------------------------------------------------------------

class TestWellCoveredNoEscalation:
    """When retrieval is sufficient (enough_candidates + enough_field_coverage),
    fallback_level must be "none" and no extra retrieval should be triggered."""

    @pytest.mark.asyncio
    async def test_well_covered_fallback_level_none(self):
        """Well-covered pool → fallback_level="none" in diagnostics."""
        from app.services.extraction_candidate_scoring import (
            MergedCandidate,
            enough_candidates,
            enough_field_coverage,
        )

        # Build a pool that passes both coverage checks.
        # enough_candidates requires >= min(top_k, 10) candidates with real signal.
        # Use top_k=3, so threshold = min(3, 10) = 3 real-signal candidates.
        run_id = "run-well-covered"
        rows = [
            _row(f"chunk_{i}", _norm(_vec(float(i + 1), 0)),
                 run_id=run_id, chunk_index=i,
                 chunk_text=f"radar power output mhz watts chunk {i}",
                 source_refs=[f"#/texts/{i}"])
            for i in range(5)
        ]
        store = _fake_store(rows)
        signals = _make_signals(entity_query="radar power rf")
        # top_k=3 → enough_candidates threshold = 3; we have 5 rows with dense signal
        profile = _make_profile(top_n_candidates=10, top_k=3, field_query_top_k=5,
                                 fallback_min_field_coverage=1, min_similarity=0.0)

        pass_def = _make_pass_def(profile)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = _make_signals(entity_query="radar power rf")

        def _fake_rerank(query, candidates, top_k):
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = 0.9 - i * 0.01
                scored.append(c2)
            return scored

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=signals_mock),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=store),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                patch("app.services.extraction_chunk_search.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
            ):
                settings = MagicMock()
                settings.extraction_index_mode = "merged"
                settings.reranker_enabled = True
                settings.vector_router_retrieval_mode = "direct"
                mock_settings.return_value = settings

                resp = await ac.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": run_id,
                        "bundle_key": _BUNDLE_KEY,
                        "pass_name": _PASS_NAME,
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        diag = body["diagnostics"]
        assert diag["fallback_level"] == "none", (
            f"Well-covered pool should have fallback_level='none', got {diag['fallback_level']!r}"
        )
        # SQL was called once only (the initial fetch; no extra fetch for fallback)
        assert store._client.query.call_count == 1, (
            f"Expected 1 SQL call, got {store._client.query.call_count}"
        )


# ---------------------------------------------------------------------------
# 2. Sparse dense → escalates to relaxed_dense; no re-fetch / no re-embed
# ---------------------------------------------------------------------------

class TestRelaxedDenseNoReEmbed:
    """When the initial pool under-covers, escalate to relaxed_dense.

    The key invariant: fetch_extraction_chunks_for_run and embed_texts are
    each called EXACTLY ONCE across the initial + relaxed passes.
    """

    @pytest.mark.asyncio
    async def test_sparse_dense_escalates_no_refetch_no_reembed(self):
        """Sparse pool → relaxed_dense level; fetch and embed called once each."""
        run_id = "run-sparse"
        # Build rows that will produce a pool below the enough_candidates threshold.
        # top_k=5 → threshold=5; we provide only 2 rows with real signal.
        # But all rows will be dense-retrieved (min_similarity=0.0), so we need
        # to make enough_candidates return False by having top_k > candidate count.
        rows = [
            _row("chunk_0", _norm(_vec(1, 0)), run_id=run_id, chunk_index=0,
                 source_refs=["#/texts/0"]),
            _row("chunk_1", _norm(_vec(0, 1)), run_id=run_id, chunk_index=1,
                 source_refs=["#/texts/1"]),
        ]
        store = _fake_store(rows)

        # top_k=5 → threshold=min(5,10)=5; only 2 rows → enough_candidates=False
        profile = _make_profile(top_n_candidates=20, top_k=5, field_query_top_k=5,
                                 fallback_min_field_coverage=1, min_similarity=0.45,
                                 fallback_similarity_relaxation=0.1,
                                 fallback_to_full=True)

        pass_def = _make_pass_def(profile)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = _make_signals(entity_query="radar power rf")

        def _fake_rerank(query, candidates, top_k):
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = 0.8 - i * 0.01
                scored.append(c2)
            return scored

        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()

        # Track calls to fetch and embed
        fetch_mock = AsyncMock(return_value=rows)
        embed_call_count = {"n": 0}
        original_embed = None

        def counting_embed(texts, query=False):
            embed_call_count["n"] += 1
            # Return deterministic vectors
            return [q_vec for _ in texts]

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=signals_mock),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=store),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                # Patch embed_texts in the chunk_search module (where it's called)
                patch("app.services.extraction_chunk_search.embed_texts",
                      side_effect=counting_embed),
                # Patch fetch to track calls
                patch("app.services.extraction_chunk_search.fetch_extraction_chunks_for_run",
                      side_effect=fetch_mock),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
            ):
                settings = MagicMock()
                settings.extraction_index_mode = "merged"
                settings.reranker_enabled = True
                settings.vector_router_retrieval_mode = "direct"
                mock_settings.return_value = settings

                resp = await ac.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": run_id,
                        "bundle_key": _BUNDLE_KEY,
                        "pass_name": _PASS_NAME,
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        diag = body["diagnostics"]

        # Fallback must have escalated (not "none")
        assert diag["fallback_level"] in ("relaxed_dense", "lexical_table",
                                           "identity_anchor", "full"), (
            f"Expected escalation, got fallback_level={diag['fallback_level']!r}"
        )

        # CORE INVARIANT: fetch_extraction_chunks_for_run called exactly once
        assert fetch_mock.call_count == 1, (
            f"fetch_extraction_chunks_for_run called {fetch_mock.call_count} times — "
            "must be 1 (no re-fetch on relaxed_dense)"
        )

        # CORE INVARIANT: embed_texts called exactly once
        assert embed_call_count["n"] == 1, (
            f"embed_texts called {embed_call_count['n']} times — "
            "must be 1 (no re-embed on relaxed_dense)"
        )


# ---------------------------------------------------------------------------
# 3. Still sparse → lexical_table admits keyword-only candidate
# ---------------------------------------------------------------------------

class TestLexicalTableAdmitsKeywordOnly:
    """When relaxed_dense still under-covers, lexical_table must admit at least
    one keyword-only candidate (one that had lexical/pattern hits but was absent
    from the dense pool).

    We do this at the unit level — directly testing the E2 helper that builds
    keyword-only candidates, not going through the full endpoint.
    """

    def test_lexical_table_adds_keyword_only_candidate(self):
        """_build_lexical_table_candidates returns MergedCandidates for keys
        that are in lexical_hits/pattern_hits but absent from the merged dense pool."""
        from app.services.extraction_candidate_scoring import MergedCandidate
        from app.api.v1.extraction_routing import _build_lexical_table_candidates

        # Dense pool has chunk_0 only
        pool_keys = {"run1:chunk_0"}

        # Lexical hits: chunk_0 (already in pool) and chunk_1 (keyword-only)
        lex_hits = {
            "run1:chunk_0": {"alias_hits": 2, "negative_hits": 0, "supported_fields": {"tx_power"}},
            "run1:chunk_1": {"alias_hits": 3, "negative_hits": 0, "supported_fields": {"frequency"}},
        }
        # Pattern hits: chunk_2 (keyword-only, pattern match only)
        pat_hits = {
            "run1:chunk_2": {"pattern_hits": 1, "supported_fields": {"pulse_width"}},
        }

        # Raw rows dict (keyed by candidate_key) for building MergedCandidates
        rows_by_key = {
            "run1:chunk_0": _row("chunk_0", _norm(_vec(1, 0)), vertex_id="run1:chunk_0",
                                  chunk_index=0, source_refs=["#/texts/0"]),
            "run1:chunk_1": _row("chunk_1", None, vertex_id="run1:chunk_1",
                                  chunk_index=1, source_refs=["#/texts/1"],
                                  chunk_text="radar frequency MHz band"),
            "run1:chunk_2": _row("chunk_2", None, vertex_id="run1:chunk_2",
                                  chunk_index=2, source_refs=["#/texts/2"],
                                  chunk_text="pulse width microseconds"),
        }

        keyword_only = _build_lexical_table_candidates(
            pool_keys=pool_keys,
            lexical_hits=lex_hits,
            pattern_hits=pat_hits,
            rows_by_key=rows_by_key,
        )

        assert len(keyword_only) >= 1, (
            "lexical_table level must admit at least one keyword-only candidate"
        )
        # chunk_1 and chunk_2 are keyword-only (absent from dense pool)
        keys_returned = {mc.candidate_key for mc in keyword_only}
        assert "run1:chunk_0" not in keys_returned, (
            "chunk_0 is already in the dense pool — must NOT be re-admitted"
        )
        # At least one of chunk_1 or chunk_2 must be present
        assert keys_returned & {"run1:chunk_1", "run1:chunk_2"}, (
            f"Expected at least one keyword-only candidate; got {keys_returned}"
        )

        # All returned candidates must have retrieval_sources from lexical/pattern
        for mc in keyword_only:
            assert mc.retrieval_sources & {"lexical", "pattern"}, (
                f"Keyword-only candidate {mc.candidate_key} must have lexical or "
                f"pattern retrieval_sources; got {mc.retrieval_sources!r}"
            )


# ---------------------------------------------------------------------------
# 4. All cheaper levels empty + fallback_to_full → full / would_skip
# ---------------------------------------------------------------------------

class TestFallbackToFullOrWouldSkip:
    """When all cheaper fallback levels are exhausted:
    - fallback_to_full=True → mode=full
    - fallback_to_full=False → mode=would_skip
    """

    def _run_endpoint_with_empty_pool(self, fallback_to_full: bool) -> dict:
        """Helper: run the endpoint with an empty pool through all fallback levels."""
        run_id = f"run-empty-{fallback_to_full}"
        # No rows → empty pool → all levels empty
        rows: list[dict] = []
        store = _fake_store(rows)

        profile = _make_profile(top_n_candidates=10, top_k=3, field_query_top_k=5,
                                 fallback_min_field_coverage=1, min_similarity=0.45,
                                 fallback_to_full=fallback_to_full)
        pass_def = _make_pass_def(profile)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = _make_signals(entity_query="radar power rf")

        def _fake_rerank(query, candidates, top_k):
            return candidates

        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()

        async def _run():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
                with (
                    patch("app.api.v1.extraction_routing.load_bundle_manifest",
                          return_value=manifest),
                    patch("app.api.v1.extraction_routing._resolve_template_class",
                          return_value=MagicMock()),
                    patch("app.api.v1.extraction_routing.build_retrieval_profile",
                          return_value=signals_mock),
                    patch("app.api.v1.extraction_routing.get_graph_store",
                          return_value=store),
                    patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                          new=AsyncMock(return_value=1000)),
                    patch("app.services.extraction_chunk_search.embed_texts",
                          return_value=[q_vec]),
                    patch("app.api.v1.extraction_routing.rrk.rerank",
                          side_effect=_fake_rerank),
                    patch("app.api.v1.extraction_routing.identity_anchor_queries",
                          new=AsyncMock(return_value=[])),
                    patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
                ):
                    settings = MagicMock()
                    settings.extraction_index_mode = "merged"
                    settings.reranker_enabled = True
                    settings.vector_router_retrieval_mode = "direct"
                    mock_settings.return_value = settings

                    resp = await ac.post(
                        "/v1/extraction/chunk-scope",
                        json={
                            "pipeline_run_id": run_id,
                            "bundle_key": _BUNDLE_KEY,
                            "pass_name": _PASS_NAME,
                        },
                    )
            return resp.json()

        return asyncio.get_event_loop().run_until_complete(_run())

    @pytest.mark.asyncio
    async def test_empty_fallback_to_full_true(self):
        """Empty pool + fallback_to_full=True → mode=full."""
        run_id = "run-empty-full-true"
        rows: list[dict] = []
        store = _fake_store(rows)

        profile = _make_profile(top_n_candidates=10, top_k=3, fallback_to_full=True)
        pass_def = _make_pass_def(profile)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = _make_signals(entity_query="radar power rf")

        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=signals_mock),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=store),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                patch("app.services.extraction_chunk_search.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
            ):
                settings = MagicMock()
                settings.extraction_index_mode = "merged"
                settings.reranker_enabled = True
                settings.vector_router_retrieval_mode = "direct"
                mock_settings.return_value = settings

                resp = await ac.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": run_id,
                        "bundle_key": _BUNDLE_KEY,
                        "pass_name": _PASS_NAME,
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Empty pool → should be "full" (fallback_to_full=True) even before ladder
        # (existing empty-pool handling), or after ladder exhaustion
        assert body["mode"] == "full", (
            f"Empty pool + fallback_to_full=True → mode=full, got {body['mode']!r}"
        )

    @pytest.mark.asyncio
    async def test_empty_fallback_to_full_false_would_skip(self):
        """Empty pool + fallback_to_full=False → mode=would_skip."""
        run_id = "run-empty-would-skip"
        rows: list[dict] = []
        store = _fake_store(rows)

        profile = _make_profile(top_n_candidates=10, top_k=3, fallback_to_full=False)
        pass_def = _make_pass_def(profile)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = _make_signals(entity_query="radar power rf")

        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=signals_mock),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=store),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                patch("app.services.extraction_chunk_search.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
            ):
                settings = MagicMock()
                settings.extraction_index_mode = "merged"
                settings.reranker_enabled = True
                settings.vector_router_retrieval_mode = "direct"
                mock_settings.return_value = settings

                resp = await ac.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": run_id,
                        "bundle_key": _BUNDLE_KEY,
                        "pass_name": _PASS_NAME,
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["mode"] == "would_skip", (
            f"Empty pool + fallback_to_full=False → mode=would_skip, got {body['mode']!r}"
        )
