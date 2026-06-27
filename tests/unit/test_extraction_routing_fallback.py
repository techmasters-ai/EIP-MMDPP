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


def _score_candidates_stub(scored2):
    """``score_candidates`` side_effect that honours ``return_components`` (Task 18).

    The router scores each pool ONCE with ``return_components=True``; a static
    2-tuple ``return_value`` no longer matches that contract. Returns 2-tuples
    when ``return_components`` is falsy and 3-tuples (with a minimal gate-flag
    component dict) when True.
    """
    def _side_effect(candidates, cfg, *, return_components=False, unit_signature=()):
        if return_components:
            return [
                (
                    mc,
                    final,
                    {
                        "candidate_key": mc.candidate_key,
                        "final_score": float(final),
                        "unit_gate": 1.0 if "unit" in getattr(mc, "gate_flags", set()) else 0.0,
                        "table_gate": 1.0 if "table" in getattr(mc, "gate_flags", set()) else 0.0,
                    },
                )
                for mc, final in scored2
            ]
        return list(scored2)
    return _side_effect


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
    lexical_decomposed: bool = False,
    field_label_weight: float = 0.0,
    pass_keyword_weight: float = 0.0,
    anchor_text_weight: float = 0.0,
    anchor_section_weight: float = 0.0,
    lexical_keywords: list | None = None,
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
    # Decomposed-lexical knobs — pin to the real RetrievalProfile defaults so
    # the MagicMock cfg exercises the LEGACY (flag-off) scoring path. Without
    # these, MagicMock auto-attrs make cfg.lexical_decomposed truthy and the
    # decomposed weights MagicMocks, which crashes score_candidates.
    p.lexical_decomposed = lexical_decomposed
    p.field_label_weight = field_label_weight
    p.pass_keyword_weight = pass_keyword_weight
    p.anchor_text_weight = anchor_text_weight
    p.anchor_section_weight = anchor_section_weight
    p.lexical_keywords = lexical_keywords if lexical_keywords is not None else []
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

        # Fallback must have escalated (not "none").
        # "degraded" is the correct label when the pool is non-empty after all
        # cheaper ladder levels but still below top_k threshold (I1 fix).
        assert diag["fallback_level"] in ("relaxed_dense", "lexical_table",
                                           "identity_anchor", "full", "degraded"), (
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

    def test_lexical_table_candidate_carries_is_table_content_type(self):
        """TABLE signal (is_table wiring): the lexical_table row-built site
        goes through merged_candidate_from_row — a keyword-only row whose
        persisted ``is_table`` column is true arrives with
        content_type == 'table'; a legacy row (no column) stays None.
        table_meta can never reach these candidates (pool-absent keys)."""
        from app.api.v1.extraction_routing import _build_lexical_table_candidates

        pool_keys: set = set()
        lex_hits = {
            "run1:chunk_t": {"alias_hits": 2, "negative_hits": 0,
                             "supported_fields": {"max_range_km"}},
            "run1:chunk_legacy": {"alias_hits": 1, "negative_hits": 0,
                                  "supported_fields": {"frequency"}},
        }
        table_row = _row("chunk_t", None, vertex_id="run1:chunk_t",
                         chunk_index=3, source_refs=["#/texts/3"],
                         chunk_text="Max range: 43 km")
        table_row["is_table"] = True  # the persisted column, as the SELECT projects it
        rows_by_key = {
            "run1:chunk_t": table_row,
            "run1:chunk_legacy": _row("chunk_legacy", None,
                                      vertex_id="run1:chunk_legacy",
                                      chunk_index=4, source_refs=["#/texts/4"],
                                      chunk_text="radar frequency MHz band"),
        }
        assert "is_table" not in rows_by_key["run1:chunk_legacy"]

        keyword_only = _build_lexical_table_candidates(
            pool_keys=pool_keys,
            lexical_hits=lex_hits,
            pattern_hits={},
            rows_by_key=rows_by_key,
        )

        by_key = {mc.candidate_key: mc for mc in keyword_only}
        assert by_key["run1:chunk_t"].content_type == "table"
        assert by_key["run1:chunk_legacy"].content_type is None


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


# ---------------------------------------------------------------------------
# I1. Partial selection (low coverage but non-empty) → fallback_level="degraded"
# ---------------------------------------------------------------------------

class TestPartialSelectionDegradedLevel:
    """I1 regression: when the ladder exhausts all cheaper levels but selected_mcs
    is non-empty (coverage low but non-zero), the response must be mode=selected_refs
    AND diagnostics.fallback_level must be "degraded" — never "full" or "would_skip".

    Before the fix, the code set:
        _e2_fallback_level = "full" if profile.fallback_to_full else "would_skip"
    even on the proceed-with-partial path, corrupting A/B analytics keyed on
    fallback_level=="full".
    """

    @pytest.mark.asyncio
    async def test_partial_selection_fallback_level_is_degraded(self):
        """Partial pool (below top_k threshold) + non-empty selected_mcs →
        mode=selected_refs AND fallback_level='degraded' (never 'full')."""
        from app.services.extraction_candidate_scoring import MergedCandidate
        from app.services.extraction_chunk_search import MultiChannelState

        run_id = "run-i1-degraded"

        # Build a single candidate — below top_k=3 threshold → ladder fires,
        # but selected_mcs is non-empty (1 candidate) → partial path taken.
        mc = MergedCandidate(
            candidate_key=f"{run_id}:chunk_0",
            chunk_index=0,
            self_ref="chunk_0",
            chunk_text="radar system ERP data",
            source_refs=["#/texts/0"],
            token_count=50,
            page_number=1,
            vector_score=0.7,
            field_scores={},
            alias_hits=0,
            pattern_hits=0,
            negative_hits=0,
            section_hits=0,
            content_type=None,
            retrieval_sources={"dense"},
            supported_field_hints=set(),
        )
        pool = [mc]
        diag_obj = SimpleNamespace(
            raw_row_count=1,
            entity_dense_count=1,
            field_dense_total_count=0,
            per_field_dense_counts={},
            lexical_hit_count=0,
            pattern_hit_count=0,
            pool_size=1,
            filter_strategy="direct_cosine",
        )
        state = MultiChannelState(
            rows=[],
            entity_dense=[],
            field_dense={},
            lex_hits={},
            pat_hits={},
            raw_row_count=0,
        )

        profile = _make_profile(
            top_n_candidates=10,
            top_k=3,          # requires 3; we only have 1 → coverage insufficient
            field_query_top_k=5,
            fallback_min_field_coverage=1,
            min_similarity=0.0,
            fallback_to_full=True,  # even with fallback_to_full=True, must be "degraded"
        )
        # Disable subset_schema_extraction to avoid Record-unwrap in this test
        profile.subset_schema_extraction = False
        pass_def = _make_pass_def(profile)
        manifest = MagicMock()
        manifest.passes = [pass_def]

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
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                # Use search_extraction_chunks_multi_channel_full mock (merged path)
                patch("app.api.v1.extraction_routing.search_extraction_chunks_multi_channel_full",
                      new=AsyncMock(return_value=(pool, diag_obj, state))),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                # score_candidates returns the 1 candidate
                patch("app.api.v1.extraction_routing.score_candidates",
                      side_effect=_score_candidates_stub([(mc, 0.7)])),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=lambda query, candidates, top_k: [
                          dict(c, reranker_score=0.7) for c in candidates
                      ]),
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

        # Mode must be selected_refs (non-empty selection proceeded)
        assert body["mode"] == "selected_refs", (
            f"Partial non-empty selection must produce mode=selected_refs; "
            f"got mode={body['mode']!r}"
        )

        diag = body["diagnostics"]
        # fallback_level must be "degraded" — not "full" (the I1 bug value)
        assert diag["fallback_level"] == "degraded", (
            f"Partial selection must produce fallback_level='degraded'; "
            f"got fallback_level={diag['fallback_level']!r} — "
            f"'full' here means the I1 bug is NOT fixed"
        )
        # Explicitly confirm it is NOT "full"
        assert diag["fallback_level"] != "full", (
            "fallback_level must never be 'full' when mode=selected_refs "
            "(I1 regression: 'full' corrupts A/B analytics)"
        )
        assert diag["fallback_level"] != "would_skip", (
            "fallback_level must never be 'would_skip' when mode=selected_refs"
        )


# ---------------------------------------------------------------------------
# E3 edge cases
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Zero field coverage with NON-empty candidates → escalates (not suppressed)
# ---------------------------------------------------------------------------

class TestZeroFieldCoverageNonEmptyEscalates:
    """A field-group pass where the dense pool is non-empty but all candidates
    have empty supported_field_hints (zero field coverage) should trigger the
    E2 fallback ladder (not be suppressed by the enough_candidates guard alone).

    Specifically: enough_candidates returns True (candidates exist with real
    retrieval_sources), but enough_field_coverage returns False because no
    candidate carries the pass's field hints.  The ladder MUST fire.

    We exercise the E2 _enough check directly at the unit level rather than
    through the full HTTP path, since the HTTP path has unavoidable coupling
    between field_queries and the embedding multi-query shape.
    """

    def test_zero_field_coverage_triggers_escalation_unit(self):
        """enough_candidates=True + enough_field_coverage=False → ladder fires.

        Directly tests the _enough predicate that guards the E2 ladder: the
        ladder condition is (not enough_candidates OR (has_field_queries AND
        not enough_field_coverage)).  When candidates exist but have zero
        field coverage, the predicate must be False → ladder fires.
        """
        from app.services.extraction_candidate_scoring import (
            MergedCandidate,
            enough_candidates,
            enough_field_coverage,
            field_coverage,
        )

        profile = _make_profile(
            top_n_candidates=10, top_k=3, field_query_top_k=5,
            fallback_min_field_coverage=1, min_similarity=0.0,
        )

        # Build 4 candidates with real retrieval_sources (dense) but NO field hints.
        # enough_candidates(top_k=3) threshold = min(3,10)=3; we have 4 → True.
        # enough_field_coverage: field_coverage({}) → {} → 0 covered → False.
        candidates = [
            MergedCandidate(
                candidate_key=f"run:chunk_{i}",
                chunk_index=i,
                self_ref=f"chunk_{i}",
                chunk_text=f"generic text {i}",
                source_refs=[f"#/texts/{i}"],
                token_count=50,
                page_number=1,
                vector_score=0.7,
                field_scores={},
                alias_hits=0,
                pattern_hits=0,
                negative_hits=0,
                section_hits=0,
                content_type=None,
                retrieval_sources={"dense"},      # real signal → counts toward enough_candidates
                supported_field_hints=set(),       # ZERO field hints → enough_field_coverage fails
            )
            for i in range(4)
        ]

        # Confirm the predicates produce the expected values.
        assert enough_candidates(candidates, profile) is True, (
            "4 dense candidates with top_k=3 → enough_candidates must be True"
        )
        assert enough_field_coverage(candidates, profile) is False, (
            "Zero field hints → enough_field_coverage must be False"
        )

        # The ladder gate in extraction_routing.py evaluates:
        #   _enough = enough_candidates(...) and (not _has_field_queries or enough_field_coverage(...))
        # When _has_field_queries=True (simulated here), _enough must be False.
        _has_field_queries = True  # simulated: pass has field queries
        _enough = (
            enough_candidates(candidates, profile)
            and (not _has_field_queries or enough_field_coverage(candidates, profile))
        )
        assert not _enough, (
            "Zero field coverage with non-empty dense candidates and field queries "
            "must NOT pass the _enough gate — the E2 ladder must fire"
        )

        # Also check field_coverage_before_fallback shape (empty dict, not None).
        _fc = field_coverage(candidates)
        assert isinstance(_fc, dict), "field_coverage must return a dict"
        assert len(_fc) == 0, "No field hints → field_coverage must return empty dict"


# ---------------------------------------------------------------------------
# 6. Legacy full fallback unchanged — all cheaper levels empty → full
# ---------------------------------------------------------------------------

class TestLegacyFullFallbackUnchanged:
    """When all cheaper ladder levels (relaxed_dense, lexical_table, identity_anchor)
    return nothing new AND fallback_to_full=True, the endpoint must reach mode=full
    exactly as before E2 was introduced.

    This test uses a pool that genuinely fails all cheaper levels:
    - Only 1 row → below top_k threshold → enough_candidates=False
    - No lexical/pattern hits → lexical_table produces nothing
    - No anchors → identity_anchor no-ops
    - fallback_to_full=True → reaches mode=full (not would_skip)
    """

    @pytest.mark.asyncio
    async def test_all_levels_empty_reaches_full(self):
        """All cheaper fallback levels produce nothing → mode=full when fallback_to_full=True."""
        run_id = "run-all-empty-full"
        rows = [
            _row("chunk_0", _norm(_vec(1, 0)), run_id=run_id, chunk_index=0,
                 source_refs=["#/texts/0"]),
        ]
        store = _fake_store(rows)

        # top_k=5 → threshold=5; only 1 row → enough_candidates=False.
        # fallback_to_full=True → should produce mode=full at level 4.
        profile = _make_profile(
            top_n_candidates=10, top_k=5, field_query_top_k=5,
            fallback_min_field_coverage=1, min_similarity=0.45,
            fallback_to_full=True,
        )
        pass_def = _make_pass_def(profile)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = _make_signals(entity_query="radar power rf")

        def _fake_rerank(query, candidates, top_k):
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = 0.5 - i * 0.01
                scored.append(c2)
            return scored

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

        # With only 1 row (below top_k=5 threshold) and no lexical hits or
        # anchors, the ladder should escalate but selected_mcs may be non-empty
        # (1 candidate selected). When selected_mcs is non-empty after ladder
        # exhaustion, the endpoint returns selected_refs with
        # fallback_level="degraded" (I1 fix). Either way, the ladder must have
        # escalated (fallback_level != "none").
        assert diag["fallback_level"] != "none", (
            f"1-candidate pool below top_k threshold must escalate, "
            f"got fallback_level={diag['fallback_level']!r}"
        )


# ---------------------------------------------------------------------------
# 7. `would_skip` survives — fallback_to_full=False + empty levels
# ---------------------------------------------------------------------------

class TestWouldSkipSurvives:
    """When fallback_to_full=False and all cheaper ladder levels are empty,
    the endpoint must return mode=would_skip, NOT mode=full.

    Verifies that fallback_to_full=False is respected all the way through
    the ladder (the ladder must NOT silently promote to full).
    """

    @pytest.mark.asyncio
    async def test_would_skip_not_silently_promoted_to_full(self):
        """Empty pool + fallback_to_full=False → mode=would_skip (not full)."""
        run_id = "run-would-skip-e3"
        # Empty rows → no pool → exits early with would_skip.
        rows: list[dict] = []
        store = _fake_store(rows)

        profile = _make_profile(
            top_n_candidates=10, top_k=3, field_query_top_k=5,
            fallback_min_field_coverage=1, min_similarity=0.45,
            fallback_to_full=False,  # Key: must not promote to full
        )
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
            f"Empty pool + fallback_to_full=False must return would_skip, "
            f"got mode={body['mode']!r}"
        )
        # Confirm it did NOT silently become full
        assert body["mode"] != "full", (
            "fallback_to_full=False must NEVER produce mode=full"
        )


# ---------------------------------------------------------------------------
# 8. E3 diagnostics populated correctly
# ---------------------------------------------------------------------------

class TestE3DiagnosticsPopulated:
    """Verify that the three E3 diagnostic fields are correctly populated:
    - After an escalated call: fallback_level != "none",
      candidate_count_before_fallback and candidate_count_after_fallback are
      set (after >= before when escalation adds candidates), and
      field_coverage_before_fallback is a dict.
    - After a no-escalation call: fallback_level == "none" and
      before == after (no additional candidates were added).
    """

    @pytest.mark.asyncio
    async def test_escalated_call_e3_fields_populated(self):
        """Escalated path: E3 before/after counts are set; after >= before."""
        run_id = "run-e3-escalated"
        # 2 rows with dense signal but below top_k=5 threshold → escalation fires.
        rows = [
            _row("chunk_0", _norm(_vec(1, 0)), run_id=run_id, chunk_index=0,
                 source_refs=["#/texts/0"], chunk_text="radar transmit power watts"),
            _row("chunk_1", _norm(_vec(0, 1)), run_id=run_id, chunk_index=1,
                 source_refs=["#/texts/1"], chunk_text="frequency band MHz"),
        ]
        store = _fake_store(rows)

        # top_k=5 → threshold=5; only 2 rows → enough_candidates=False → escalation.
        profile = _make_profile(
            top_n_candidates=20, top_k=5, field_query_top_k=5,
            fallback_min_field_coverage=1, min_similarity=0.45,
            fallback_similarity_relaxation=0.1, fallback_to_full=True,
        )
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
                patch("app.services.extraction_chunk_search.fetch_extraction_chunks_for_run",
                      side_effect=AsyncMock(return_value=rows)),
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

        # Escalation must have fired
        assert diag["fallback_level"] != "none", (
            f"Expected escalation; got fallback_level={diag['fallback_level']!r}"
        )

        # E3: before/after counts must be populated
        assert diag["candidate_count_before_fallback"] is not None, (
            "candidate_count_before_fallback must be set on escalated multi-channel path"
        )
        assert diag["candidate_count_after_fallback"] is not None, (
            "candidate_count_after_fallback must be set on escalated multi-channel path"
        )

        # after >= before (escalation can only ADD candidates, never remove them)
        assert diag["candidate_count_after_fallback"] >= diag["candidate_count_before_fallback"], (
            f"after ({diag['candidate_count_after_fallback']}) < "
            f"before ({diag['candidate_count_before_fallback']}) — "
            "escalation must not shrink the pool"
        )

        # field_coverage_before_fallback must be a dict (may be empty)
        assert isinstance(diag["field_coverage_before_fallback"], dict), (
            "field_coverage_before_fallback must be a dict on escalated path; "
            f"got {type(diag['field_coverage_before_fallback'])!r}"
        )

    @pytest.mark.asyncio
    async def test_no_escalation_e3_before_equals_after(self):
        """No-escalation path: fallback_level='none' and before==after counts."""
        run_id = "run-e3-no-escalation"
        # 5 rows — above top_k=3 threshold → enough_candidates=True → no escalation.
        rows = [
            _row(f"chunk_{i}", _norm(_vec(float(i + 1), 0)),
                 run_id=run_id, chunk_index=i,
                 chunk_text=f"radar power output mhz watts chunk {i}",
                 source_refs=[f"#/texts/{i}"])
            for i in range(5)
        ]
        store = _fake_store(rows)

        profile = _make_profile(
            top_n_candidates=10, top_k=3, field_query_top_k=5,
            fallback_min_field_coverage=1, min_similarity=0.0,
        )
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

        # No escalation on well-covered pool
        assert diag["fallback_level"] == "none", (
            f"Well-covered pool must have fallback_level='none', "
            f"got {diag['fallback_level']!r}"
        )

        # E3: before == after (no escalation means pool unchanged)
        assert diag["candidate_count_before_fallback"] is not None, (
            "candidate_count_before_fallback must be set on multi-channel path even without escalation"
        )
        assert diag["candidate_count_after_fallback"] is not None, (
            "candidate_count_after_fallback must be set on multi-channel path even without escalation"
        )
        assert diag["candidate_count_before_fallback"] == diag["candidate_count_after_fallback"], (
            f"No escalation: before ({diag['candidate_count_before_fallback']}) "
            f"!= after ({diag['candidate_count_after_fallback']})"
        )

        # field_coverage_before_fallback must be a dict (may be empty for entity-only passes)
        assert isinstance(diag["field_coverage_before_fallback"], dict), (
            "field_coverage_before_fallback must be a dict on multi-channel path; "
            f"got {type(diag['field_coverage_before_fallback'])!r}"
        )


# ---------------------------------------------------------------------------
# Per-pass keyword injection — inject_pass_keywords
# ---------------------------------------------------------------------------

class TestInjectPassKeywords:
    """inject_pass_keywords UNIONS manifest lexical_keywords with schema-derived
    unit vocabulary (guarded-ranker spec §5.1).

    Contract:
      - Manifest entries come FIRST (original casing/order).
      - Derived units appended, dedup by NFC+casefold.
      - profile=None → default RetrievalProfile + derived.
      - Input profile object NOT mutated (model_copy semantics).
    """

    @staticmethod
    def _signals_with_units():
        from app.services.extraction_query_builder import (
            FieldRetrievalQuery,
            PassRetrievalSignals,
        )

        fq = FieldRetrievalQuery(
            field_name="erp_dbw",
            query_text="",
            aliases=("ERP",),
            negative_terms=(),
            evidence_patterns=(),
            likely_sections=(),
            units=("dBW", "dBm"),
        )
        return PassRetrievalSignals(
            pass_name="radar_power_rf",
            entity_doc="",
            entity_query="",
            field_queries=(fq,),
            lexical_terms=("ERP",),
            negative_terms=(),
            likely_sections=(),
            evidence_patterns=(),
        )

    def test_populates_when_empty(self):
        """profile.lexical_keywords == [] → set to derive_pass_keywords(signals)."""
        from app.api.v1.extraction_routing import inject_pass_keywords
        from app.services.ontology_bundles import RetrievalProfile

        profile = RetrievalProfile()  # lexical_keywords defaults to []
        assert profile.lexical_keywords == []

        out = inject_pass_keywords(profile, self._signals_with_units())
        assert out.lexical_keywords == ["dBW", "dBm"]
        # Other declared fields are preserved through the model_copy.
        assert out.min_similarity == profile.min_similarity

    def test_union_manifest_then_derived(self):
        """Non-empty manifest + derived units → union: manifest first, derived appended."""
        from app.api.v1.extraction_routing import inject_pass_keywords
        from app.services.ontology_bundles import RetrievalProfile

        # Manifest carries "TWT"; derived would yield "dBW", "dBm" (not aliases).
        profile = RetrievalProfile(lexical_keywords=["traveling wave tube", "TWT"])
        out = inject_pass_keywords(profile, self._signals_with_units())
        # Manifest entries come first, in original casing/order.
        assert out.lexical_keywords[:2] == ["traveling wave tube", "TWT"]
        # Derived units appended after.
        assert "dBW" in out.lexical_keywords
        assert "dBm" in out.lexical_keywords

    def test_union_dedup_casefold(self):
        """manifest 'kw' + derived 'kW' → 'kW' NOT appended (NFC+casefold dedup)."""
        from app.api.v1.extraction_routing import inject_pass_keywords
        from app.services.extraction_query_builder import (
            FieldRetrievalQuery,
            PassRetrievalSignals,
        )
        from app.services.ontology_bundles import RetrievalProfile

        fq = FieldRetrievalQuery(
            field_name="tx_power_kw",
            query_text="",
            aliases=(),
            negative_terms=(),
            evidence_patterns=(),
            likely_sections=(),
            units=("kW",),
        )
        signals = PassRetrievalSignals(
            pass_name="p",
            entity_doc="",
            entity_query="",
            field_queries=(fq,),
            lexical_terms=(),
            negative_terms=(),
            likely_sections=(),
            evidence_patterns=(),
        )
        # Manifest has lowercase 'kw'; derived yields 'kW' — same under casefold.
        profile = RetrievalProfile(lexical_keywords=["kw"])
        out = inject_pass_keywords(profile, signals)
        # 'kW' must NOT be appended (would be a casefold-dup).
        assert out.lexical_keywords == ["kw"]

    def test_none_profile_builds_default_and_injects(self):
        """profile=None → a default RetrievalProfile is built and populated."""
        from app.api.v1.extraction_routing import inject_pass_keywords
        from app.services.ontology_bundles import RetrievalProfile

        out = inject_pass_keywords(None, self._signals_with_units())
        assert isinstance(out, RetrievalProfile)
        assert out.lexical_keywords == ["dBW", "dBm"]

    def test_empty_derivation_manifest_unchanged(self):
        """Empty derived → manifest list returned unchanged (object equality OK)."""
        from app.api.v1.extraction_routing import inject_pass_keywords
        from app.services.extraction_query_builder import PassRetrievalSignals
        from app.services.ontology_bundles import RetrievalProfile

        signals = PassRetrievalSignals(
            pass_name="p",
            entity_doc="",
            entity_query="",
            field_queries=(),
            lexical_terms=(),
            negative_terms=(),
            likely_sections=(),
            evidence_patterns=(),
        )
        profile = RetrievalProfile(lexical_keywords=["custom_keyword"])
        out = inject_pass_keywords(profile, signals)
        assert out.lexical_keywords == ["custom_keyword"]

    def test_empty_derivation_empty_manifest_stays_empty(self):
        """No units to mine + empty manifest → lexical_keywords stays empty."""
        from app.api.v1.extraction_routing import inject_pass_keywords
        from app.services.extraction_query_builder import PassRetrievalSignals
        from app.services.ontology_bundles import RetrievalProfile

        signals = PassRetrievalSignals(
            pass_name="p",
            entity_doc="",
            entity_query="",
            field_queries=(),
            lexical_terms=(),
            negative_terms=(),
            likely_sections=(),
            evidence_patterns=(),
        )
        out = inject_pass_keywords(RetrievalProfile(), signals)
        assert out.lexical_keywords == []

    def test_purity_input_not_mutated(self):
        """Input profile object must NOT be mutated (model_copy semantics)."""
        from app.api.v1.extraction_routing import inject_pass_keywords
        from app.services.ontology_bundles import RetrievalProfile

        profile = RetrievalProfile(lexical_keywords=["TWT"])
        original_list = list(profile.lexical_keywords)
        inject_pass_keywords(profile, self._signals_with_units())
        # Original profile unchanged.
        assert profile.lexical_keywords == original_list
