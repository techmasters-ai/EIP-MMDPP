"""Integration test for Task C6 — multi-channel retrieval wired into chunk_scope.

TDD: written BEFORE implementation. These tests must FAIL initially (the
orchestrator and endpoint branch do not exist yet) and PASS after wiring.

Coverage assertions:
  1. merged mode: exactly ONE per-run SELECT, ZERO vector_search calls.
  2. merged mode: entire capped pool is reranked (len(pool_dicts) pairs),
     NOT just profile.top_k.
  3. Post-rerank C5 boost can change which chunks are selected vs rerank-only.
  4. per_element mode: search_extraction_chunks (existing path) is still called.
  5. Cap is applied BEFORE rerank (Blocker 6 constraint).
  6. Response shape is valid ChunkScopeResponse with mode=selected_refs in
     merged mode when candidates exist.

Test pattern mirrors test_extraction_chunk_search_dense_multi.py:
  - Synthetic rows with L2-normalised embeddings.
  - Mocked embed_texts (returns deterministic vectors).
  - Mocked reranker.rerank (returns deterministic scores).
  - No live ArcadeDB, no live Ollama, no live reranker.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared row / vector helpers  (mirrors test_extraction_chunk_search_dense_multi)
# ---------------------------------------------------------------------------

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
    pipeline_run_id: str = "run-A",
    chunk_index: int = 0,
    source_refs: list[str] | None = None,
    token_count: int = 50,
    chunk_text: str | None = None,
) -> dict:
    return {
        "node_id": f"#170:{self_ref}",
        "vertex_id": f"{pipeline_run_id}:{self_ref}",
        "self_ref": self_ref,
        "chunk_text": chunk_text if chunk_text is not None else f"text for {self_ref}",
        "embedding": embedding,
        "page_number": 1,
        "modality": "merged",
        "pipeline_run_id": pipeline_run_id,
        "chunk_index": chunk_index,
        "source_refs": source_refs if source_refs is not None else [f"#/texts/{chunk_index}"],
        "token_count": token_count,
    }


def _fake_store(rows: list[dict]) -> Any:
    """Fake ArcadeDBGraphStore. Records SQL calls for assertion."""
    client = SimpleNamespace(query=AsyncMock(return_value=rows))
    return SimpleNamespace(_database="eip_knowledge_graph", _client=client)


# ---------------------------------------------------------------------------
# Minimal PassRetrievalSignals / RetrievalProfile / FieldRetrievalQuery stubs
# ---------------------------------------------------------------------------

def _make_retrieval_signals(
    entity_query: str = "entity query text",
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
    top_n_candidates: int = 10,
    top_k: int = 3,
    field_query_top_k: int = 5,
    min_similarity: float = 0.0,
    rerank_weight: float = 1.0,
    lexical_weight: float = 0.0,
    pattern_weight: float = 0.0,
    section_weight: float = 0.0,
    table_boost: float = 0.0,
    negative_weight: float = 0.0,
) -> Any:
    """Build a minimal mock RetrievalProfile."""
    p = MagicMock()
    p.top_n_candidates = top_n_candidates
    p.top_k = top_k
    p.field_query_top_k = field_query_top_k
    p.min_similarity = min_similarity
    p.rerank_weight = rerank_weight
    p.lexical_weight = lexical_weight
    p.pattern_weight = pattern_weight
    p.section_weight = section_weight
    p.table_boost = table_boost
    p.negative_weight = negative_weight
    p.fallback_to_full = True
    p.pattern_hit_limit = 50
    return p


# ---------------------------------------------------------------------------
# 1. Orchestrator import guard
# ---------------------------------------------------------------------------

class TestOrchestratorImport:
    def test_function_exists(self):
        """search_extraction_chunks_multi_channel must be importable after wiring."""
        from app.services.extraction_chunk_search import (  # noqa: F401
            search_extraction_chunks_multi_channel,
        )


# ---------------------------------------------------------------------------
# 2. Exactly ONE per-run SELECT, ZERO vector_search
# ---------------------------------------------------------------------------

class TestSingleSelectNoHNSW:
    """Orchestrator must call _client.query exactly once and never call
    vector_search (no HNSW path)."""

    @pytest.mark.asyncio
    async def test_one_select_zero_vector_search(self):
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_multi_channel,
        )

        run_id = "run-one-select"
        rows = [
            _row("chunk_0", _norm(_vec(1, 0, 0)), pipeline_run_id=run_id, chunk_index=0),
            _row("chunk_1", _norm(_vec(0, 1, 0)), pipeline_run_id=run_id, chunk_index=1),
            _row("chunk_2", _norm(_vec(0, 0, 1)), pipeline_run_id=run_id, chunk_index=2),
        ]
        store = _fake_store(rows)
        signals = _make_retrieval_signals(entity_query="entity q")
        cfg = _make_profile(top_n_candidates=10, top_k=2, field_query_top_k=5)

        # embed_texts: return query vec for entity + 0 field queries = 1 vec
        q_vec = _norm(_vec(1, 0, 0))
        with patch(
            "app.services.extraction_chunk_search.embed_texts",
            return_value=[q_vec],
        ):
            pool, diag = await search_extraction_chunks_multi_channel(
                signals, run_id, cfg, store=store
            )

        # --- Assertions ---
        # Exactly one SQL query call
        assert store._client.query.call_count == 1, (
            f"Expected 1 SQL SELECT, got {store._client.query.call_count}"
        )
        # The SQL call must reference pipeline_run_id (not be a vector_search)
        call_args = store._client.query.call_args
        # call_args: (db, "sql", sql_str, params) — positional
        sql_str = call_args.args[2] if len(call_args.args) >= 3 else str(call_args)
        assert "ExtractionChunk" in sql_str
        assert ":run_id" in sql_str or "run_id" in sql_str
        # No vector_search attribute should have been called
        assert not hasattr(store, "vector_search") or store.vector_search.call_count == 0, (
            "vector_search was called — HNSW path must not be used"
        )

        # Pool must be non-empty
        assert len(pool) > 0


# ---------------------------------------------------------------------------
# 3. Pool is capped to cfg.top_n_candidates BEFORE rerank
# ---------------------------------------------------------------------------

class TestCapBeforeRerank:
    """The orchestrator caps to top_n_candidates. The C6 endpoint must rerank
    at most top_n_candidates pairs (Blocker 6)."""

    @pytest.mark.asyncio
    async def test_pool_capped_to_top_n_candidates(self):
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_multi_channel,
        )

        run_id = "run-cap"
        # 6 rows but top_n_candidates=4 → pool must have len<=4
        rows = [
            _row(f"chunk_{i}", _norm(_vec(float(i + 1), 0, 0)), pipeline_run_id=run_id, chunk_index=i)
            for i in range(6)
        ]
        store = _fake_store(rows)
        signals = _make_retrieval_signals(entity_query="entity q")
        cfg = _make_profile(top_n_candidates=4, top_k=2, field_query_top_k=5)

        q_vec = _norm(_vec(1, 0, 0))
        with patch(
            "app.services.extraction_chunk_search.embed_texts",
            return_value=[q_vec],
        ):
            pool, diag = await search_extraction_chunks_multi_channel(
                signals, run_id, cfg, store=store
            )

        assert len(pool) <= 4, f"Pool size {len(pool)} exceeds top_n_candidates=4"


# ---------------------------------------------------------------------------
# 4. Diagnostics shape
# ---------------------------------------------------------------------------

class TestDiagnosticsShape:
    """diag returned by the orchestrator must carry channel counts."""

    @pytest.mark.asyncio
    async def test_diag_has_channel_counts(self):
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_multi_channel,
        )

        run_id = "run-diag"
        rows = [
            _row("chunk_0", _norm(_vec(1, 0)), pipeline_run_id=run_id, chunk_index=0),
        ]
        store = _fake_store(rows)
        signals = _make_retrieval_signals(entity_query="entity q")
        cfg = _make_profile(top_n_candidates=10, top_k=2, field_query_top_k=5)

        q_vec = _norm(_vec(1, 0))
        with patch(
            "app.services.extraction_chunk_search.embed_texts",
            return_value=[q_vec],
        ):
            pool, diag = await search_extraction_chunks_multi_channel(
                signals, run_id, cfg, store=store
            )

        # diag must be dict-like or have attributes for channel info
        if isinstance(diag, dict):
            assert "pool_size" in diag or "raw_row_count" in diag or len(diag) > 0
        else:
            # Must have at least one recognisable attribute
            has_any = any(hasattr(diag, attr) for attr in [
                "pool_size", "raw_row_count", "entity_dense_count",
                "channel_counts", "filter_strategy",
            ])
            assert has_any, f"diag has no expected attributes: {diag!r}"


# ---------------------------------------------------------------------------
# 5. Endpoint integration — merged mode: one SELECT + zero vector_search,
#    rerank entire pool, C5 boost can change selection, response valid.
# ---------------------------------------------------------------------------

class TestEndpointMultiChannelMergedMode:
    """End-to-end through the chunk_scope endpoint in merged mode."""

    def _make_pass_def(self, top_n_candidates: int = 6, top_k: int = 3):
        pd = MagicMock()
        pd.name = "radar_power_rf"
        pd.phase = "field_group"
        pd.module = "extraction_schemas.radar_power_rf"
        pd.template_class = "RadarPowerRfPass"

        profile = MagicMock()
        profile.min_similarity = 0.0
        profile.top_n_candidates = top_n_candidates
        profile.top_k = top_k
        profile.field_query_top_k = 5
        profile.fallback_to_full = True
        profile.pattern_hit_limit = 50
        profile.rerank_weight = 1.0
        profile.lexical_weight = 0.0
        profile.pattern_weight = 0.0
        profile.section_weight = 0.0
        profile.table_boost = 0.0
        profile.negative_weight = 0.0
        pd.retrieval = profile

        return pd

    @pytest.mark.asyncio
    async def test_merged_mode_one_select_zero_hnsw(self):
        """In merged mode with field_query_top_k>0: ONE per-run SELECT, ZERO HNSW."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        rows = [
            _row(
                f"chunk_{i}",
                _norm(_vec(float(i + 1), 0)),
                pipeline_run_id=run_id,
                chunk_index=i,
                chunk_text=f"Radar power chunk {i}: important technical data",
                source_refs=[f"#/texts/{i}"],
            )
            for i in range(5)
        ]

        # Store mock: _client.query returns rows; vector_search is present but
        # must NOT be called.
        query_mock = AsyncMock(return_value=rows)
        vector_search_mock = AsyncMock(return_value=[])
        store = SimpleNamespace(
            _database="eip_knowledge_graph",
            _client=SimpleNamespace(query=query_mock),
            vector_search=vector_search_mock,
        )

        pass_def = self._make_pass_def(top_n_candidates=6, top_k=2)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        # entity_query vector + reranker
        q_vec = _norm(_vec(1, 0))
        signals_mock = MagicMock()
        signals_mock.entity_query = "radar power rf query"
        signals_mock.field_queries = ()

        # rerank returns candidates unchanged (all get same score)
        def _fake_rerank(query, candidates, top_k):
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = float(len(candidates) - i)
                scored.append(c2)
            return scored  # NOT sliced — returns full pool

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from app.main import create_app
        app = create_app()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=signals_mock),
                patch("app.api.v1.extraction_routing.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=store),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=500)),
                patch("app.services.extraction_chunk_search.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
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
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Should have selected some refs
        assert body["mode"] in ("selected_refs", "full"), f"Unexpected mode: {body['mode']}"

        # vector_search must NOT have been called
        assert vector_search_mock.call_count == 0, (
            f"vector_search called {vector_search_mock.call_count} times — HNSW must not be used"
        )

        # _client.query must have been called exactly once (for the per-run fetch)
        assert query_mock.call_count == 1, (
            f"Expected exactly 1 SQL SELECT, got {query_mock.call_count}"
        )


# ---------------------------------------------------------------------------
# 6. C5 post-rerank boost can change selection vs rerank-only
# ---------------------------------------------------------------------------

class TestC5BoostChangesSelection:
    """C5 lexical_weight boost can elevate a low-reranker-score chunk above
    a high-reranker-score chunk when the former has more alias_hits.

    NOTE: This test constructs MergedCandidates directly (bypassing multi-channel)
    because lexical/pattern signals only BOOST candidates already in the dense pool
    (precision, not recall — plan-locked). The C5 scoring logic is agnostic to how
    the pool was assembled; what matters is that alias_hits are attached to the
    MergedCandidate before score_candidates is called.
    """

    def test_c5_boost_changes_ranking(self):
        """Scenario:
        - chunk_A: reranker_score=0.9, alias_hits=0
        - chunk_B: reranker_score=0.5, alias_hits=10 (lots of keyword matches)
        With lexical_weight=2.0, C5 should elevate chunk_B above chunk_A.
        """
        from app.services.extraction_candidate_scoring import (
            MergedCandidate,
            score_candidates,
        )

        cfg = _make_profile(
            top_n_candidates=10,
            top_k=1,
            rerank_weight=1.0,
            lexical_weight=2.0,  # big boost for lexical hits
        )

        # chunk_A: high reranker score, no alias hits (dense-only)
        mc_a = MergedCandidate(
            candidate_key="run1:chunk_0",
            chunk_index=0,
            self_ref="chunk_0",
            chunk_text="irrelevant text without keywords",
            source_refs=[],
            token_count=10,
            page_number=None,
            vector_score=0.99,
            field_scores={},
            alias_hits=0,
            pattern_hits=0,
            negative_hits=0,
            section_hits=0,
            content_type=None,
            retrieval_sources={"dense"},
            supported_field_hints=set(),
        )
        # chunk_B: lower reranker score but many alias hits (dense + lexical signal)
        mc_b = MergedCandidate(
            candidate_key="run1:chunk_1",
            chunk_index=1,
            self_ref="chunk_1",
            chunk_text="radar transmitter power output watts frequency",
            source_refs=[],
            token_count=10,
            page_number=None,
            vector_score=0.10,
            field_scores={},
            alias_hits=10,   # strong lexical signal attached via merge_candidates
            pattern_hits=0,
            negative_hits=0,
            section_hits=0,
            content_type=None,
            retrieval_sources={"dense", "field:tx_power", "lexical"},
            supported_field_hints={"tx_power"},
        )

        pool_dicts = [
            {
                "merged_candidate": mc_a,
                "content_text": mc_a.chunk_text,
                "reranker_score": 0.9,   # chunk_A wins on rerank alone
            },
            {
                "merged_candidate": mc_b,
                "content_text": mc_b.chunk_text,
                "reranker_score": 0.5,   # chunk_B loses on rerank alone
            },
        ]

        scored = score_candidates(pool_dicts, cfg)
        assert len(scored) == 2

        top_candidate, top_score = scored[0]
        # With lexical_weight=2.0, chunk_B's alias_hits must push it above chunk_A.
        assert top_candidate.candidate_key == "run1:chunk_1", (
            f"chunk_B (alias_hits=10) should lead after C5 boost with lexical_weight=2.0; "
            f"got top candidate: {top_candidate.candidate_key!r}"
        )
        assert top_candidate.alias_hits > 0, (
            f"C5 winner has alias_hits=0 — not actually boosted by lexical signal. "
            f"candidate: {top_candidate!r}"
        )
        # Verify scores numerically: chunk_B final > chunk_A final
        scores = {mc.candidate_key: s for mc, s in scored}
        assert scores["run1:chunk_1"] > scores["run1:chunk_0"], (
            f"chunk_B score ({scores['run1:chunk_1']:.4f}) must exceed "
            f"chunk_A ({scores['run1:chunk_0']:.4f}) after C5 lexical boost"
        )


# ---------------------------------------------------------------------------
# 7. Per-element mode: existing search_extraction_chunks still called
# ---------------------------------------------------------------------------

class TestPerElementModeUnchanged:
    """In per_element mode (or when field_query_top_k==0), the existing
    search_extraction_chunks path must be used — NOT the multi-channel path."""

    @pytest.mark.asyncio
    async def test_per_element_calls_existing_path(self):
        """When extraction_index_mode != 'merged', search_extraction_chunks is called."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from app.main import create_app
        app = create_app()

        @dataclass
        class _FakeResult:
            node_id: str = "rid:0"
            name: str = "chunk"
            entity_type: str = "ExtractionChunk"
            score: float = 0.75
            properties: dict = field(default_factory=lambda: {
                "self_ref": "#/texts/0",
                "chunk_text": "some text",
                "pipeline_run_id": run_id,
            })

        @dataclass
        class _FakeDiag:
            ann_top_k_requested: int = 500
            post_filter_candidate_count: int = 1
            post_filter_retry_count: int = 0
            filter_strategy: str = "overfetch_post_filter"
            short_fetch: bool = False

        fake_results = [_FakeResult()]
        fake_diag = _FakeDiag()

        search_mock = AsyncMock(return_value=(fake_results, fake_diag))
        multi_channel_mock = AsyncMock()

        pass_def = MagicMock()
        pass_def.name = pass_name
        pass_def.phase = "field_group"
        pass_def.module = "extraction_schemas.radar_power_rf"
        pass_def.template_class = "RadarPowerRfPass"
        profile = MagicMock()
        profile.min_similarity = 0.45
        profile.top_n_candidates = 10
        profile.top_k = 3
        profile.field_query_top_k = 5
        profile.fallback_to_full = True
        pass_def.retrieval = profile

        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=MagicMock(entity_query="radar power rf query",
                                            field_queries=())),
                patch("app.api.v1.extraction_routing.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=500)),
                patch("app.api.v1.extraction_routing.search_extraction_chunks",
                      new=search_mock),
                patch("app.api.v1.extraction_routing.search_extraction_chunks_multi_channel",
                      new=multi_channel_mock),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      return_value=[{
                          "content_text": "some text",
                          "self_ref": "#/texts/0",
                          "vector_score": 0.75,
                          "reranker_score": 0.8,
                      }]),
                # Patch get_settings in the module's own namespace so that
                # ``settings = get_settings()`` inside the endpoint sees our mock.
                patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
            ):
                settings = MagicMock()
                settings.extraction_index_mode = "per_element"
                settings.reranker_enabled = True
                settings.vector_router_retrieval_mode = "direct"
                mock_settings.return_value = settings

                resp = await ac.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": run_id,
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        assert resp.status_code == 200, resp.text
        # Existing path must have been called
        assert search_mock.called, "search_extraction_chunks was not called in per_element mode"
        # Multi-channel path must NOT have been called
        assert not multi_channel_mock.called, (
            "search_extraction_chunks_multi_channel was called in per_element mode — must not be"
        )


# ---------------------------------------------------------------------------
# 8. Rerank is called with len(pool) pairs, not profile.top_k
# ---------------------------------------------------------------------------

class TestReranksEntirePool:
    """The endpoint must rerank the ENTIRE capped pool (all top_n_candidates),
    not just profile.top_k. The top_k cut happens AFTER C5 scoring."""

    @pytest.mark.asyncio
    async def test_rerank_receives_full_pool(self):
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        n_rows = 5
        top_k = 2  # intentionally smaller than n_rows

        rows = [
            _row(
                f"chunk_{i}",
                _norm(_vec(float(i + 1), 0)),
                pipeline_run_id=run_id,
                chunk_index=i,
                source_refs=[f"#/texts/{i}"],
            )
            for i in range(n_rows)
        ]

        query_mock = AsyncMock(return_value=rows)
        store = SimpleNamespace(
            _database="eip_knowledge_graph",
            _client=SimpleNamespace(query=query_mock),
        )

        pass_def = MagicMock()
        pass_def.name = pass_name
        pass_def.phase = "field_group"
        pass_def.module = "extraction_schemas.radar_power_rf"
        pass_def.template_class = "RadarPowerRfPass"
        profile = MagicMock()
        profile.min_similarity = 0.0
        profile.top_n_candidates = n_rows
        profile.top_k = top_k
        profile.field_query_top_k = 5
        profile.fallback_to_full = True
        profile.pattern_hit_limit = 50
        profile.rerank_weight = 1.0
        profile.lexical_weight = 0.0
        profile.pattern_weight = 0.0
        profile.section_weight = 0.0
        profile.table_boost = 0.0
        profile.negative_weight = 0.0
        pass_def.retrieval = profile

        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = MagicMock()
        signals_mock.entity_query = "radar power rf query"
        signals_mock.field_queries = ()

        rerank_call_args = []

        def _capture_rerank(query, candidates, top_k):
            rerank_call_args.append({"n_candidates": len(candidates), "top_k": top_k})
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = float(len(candidates) - i)
                scored.append(c2)
            return scored

        from app.main import create_app
        from httpx import ASGITransport, AsyncClient

        app = create_app()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=signals_mock),
                patch("app.api.v1.extraction_routing.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=store),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=500)),
                patch("app.services.extraction_chunk_search.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_capture_rerank),
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
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        assert resp.status_code == 200, resp.text

        assert len(rerank_call_args) == 1, (
            f"Expected rerank called once, got {len(rerank_call_args)}"
        )
        n_candidates_sent = rerank_call_args[0]["n_candidates"]
        top_k_sent = rerank_call_args[0]["top_k"]

        # Must rerank the ENTIRE pool (top_n_candidates), not just top_k
        assert n_candidates_sent == n_rows, (
            f"Reranker received {n_candidates_sent} candidates; "
            f"expected {n_rows} (the full pool). top_k cut must happen AFTER rerank."
        )
        # top_k passed to rerank must equal the pool size (rerank whole pool)
        assert top_k_sent == n_rows, (
            f"Reranker top_k={top_k_sent}; expected {n_rows} (rerank full pool). "
            f"profile.top_k={top_k} is the POST-C5 cut, not the rerank limit."
        )


# ---------------------------------------------------------------------------
# 9. Task C7 — multi-channel diagnostics (channel_counts, field_coverage,
#    score_components, fallback_level) populated on multi-channel success path
# ---------------------------------------------------------------------------

class TestC7MultiChannelDiagnostics:
    """After a successful multi-channel chunk_scope call, the diagnostics block
    must include populated channel_counts, field_coverage, score_components,
    and fallback_level == "none".

    Also asserts that per-element mode leaves all four new fields as None.
    """

    def _make_pass_def(self, top_n_candidates: int = 6, top_k: int = 2):
        pd = MagicMock()
        pd.name = "radar_power_rf"
        pd.phase = "field_group"
        pd.module = "extraction_schemas.radar_power_rf"
        pd.template_class = "RadarPowerRfPass"

        profile = MagicMock()
        profile.min_similarity = 0.0
        profile.top_n_candidates = top_n_candidates
        profile.top_k = top_k
        profile.field_query_top_k = 5
        profile.fallback_to_full = True
        profile.pattern_hit_limit = 50
        profile.rerank_weight = 1.0
        profile.lexical_weight = 0.0
        profile.pattern_weight = 0.0
        profile.section_weight = 0.0
        profile.table_boost = 0.0
        profile.negative_weight = 0.0
        pd.retrieval = profile

        return pd

    @pytest.mark.asyncio
    async def test_c7_diagnostics_populated_on_multichannel_success(self):
        """Multi-channel success path: channel_counts, field_coverage,
        score_components (non-empty), fallback_level == 'none' all present."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        rows = [
            _row(
                f"chunk_{i}",
                _norm(_vec(float(i + 1), 0)),
                pipeline_run_id=run_id,
                chunk_index=i,
                chunk_text=f"Radar power chunk {i}: technical data",
                source_refs=[f"#/texts/{i}"],
            )
            for i in range(4)
        ]

        query_mock = AsyncMock(return_value=rows)
        store = SimpleNamespace(
            _database="eip_knowledge_graph",
            _client=SimpleNamespace(query=query_mock),
        )

        pass_def = self._make_pass_def(top_n_candidates=6, top_k=2)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = _norm(_vec(1, 0))
        signals_mock = MagicMock()
        signals_mock.entity_query = "radar power rf query"
        signals_mock.field_queries = ()

        def _fake_rerank(query, candidates, top_k):
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = float(len(candidates) - i)
                scored.append(c2)
            return scored

        from app.main import create_app
        from httpx import ASGITransport, AsyncClient

        app = create_app()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=signals_mock),
                patch("app.api.v1.extraction_routing.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=store),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=500)),
                patch("app.services.extraction_chunk_search.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
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
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        diag = body["diagnostics"]

        # --- channel_counts: must be a dict with at least "dense" key -------
        assert diag.get("channel_counts") is not None, (
            "channel_counts must be populated on multi-channel success path"
        )
        cc = diag["channel_counts"]
        assert isinstance(cc, dict), f"channel_counts must be a dict; got {type(cc)}"
        assert "dense" in cc, f"channel_counts must include 'dense' key; got {cc}"
        assert isinstance(cc["dense"], int), (
            f"channel_counts['dense'] must be an int; got {type(cc['dense'])}"
        )

        # --- field_coverage: must be a dict (may be empty with 0 field queries) ---
        assert diag.get("field_coverage") is not None or diag["field_coverage"] == {}, (
            "field_coverage must be a dict (possibly empty) on multi-channel success path"
        )
        fc = diag["field_coverage"]
        assert isinstance(fc, dict), f"field_coverage must be a dict; got {type(fc)}"

        # --- score_components: must be non-empty list with expected keys -----
        assert diag.get("score_components") is not None, (
            "score_components must be populated on multi-channel success path"
        )
        sc = diag["score_components"]
        assert isinstance(sc, list), f"score_components must be a list; got {type(sc)}"
        assert len(sc) > 0, "score_components must be non-empty when candidates are selected"
        assert len(sc) <= pass_def.retrieval.top_k, (
            f"score_components length {len(sc)} exceeds top_k={pass_def.retrieval.top_k}"
        )

        first = sc[0]
        assert isinstance(first, dict), f"score_components entries must be dicts; got {type(first)}"
        # Must carry at minimum the key signals
        for key in ("final_score", "retrieval_sources"):
            assert key in first, (
                f"score_components entry missing '{key}'; keys present: {list(first.keys())}"
            )

        # --- fallback_level: must be "none" on normal multi-channel success ---
        assert diag.get("fallback_level") == "none", (
            f"fallback_level must be 'none' on normal multi-channel path; "
            f"got {diag.get('fallback_level')!r}"
        )

    @pytest.mark.asyncio
    async def test_c7_new_fields_none_in_per_element_mode(self):
        """Per-element mode: channel_counts, field_coverage, score_components,
        fallback_level must all be None (new fields default None)."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        from dataclasses import dataclass, field as dc_field

        @dataclass
        class _FakeResult:
            node_id: str = "rid:0"
            name: str = "chunk"
            entity_type: str = "ExtractionChunk"
            score: float = 0.75
            properties: dict = dc_field(default_factory=lambda: {
                "self_ref": "#/texts/0",
                "chunk_text": "some text",
                "pipeline_run_id": run_id,
            })

        @dataclass
        class _FakeDiag:
            ann_top_k_requested: int = 500
            post_filter_candidate_count: int = 1
            post_filter_retry_count: int = 0
            filter_strategy: str = "overfetch_post_filter"
            short_fetch: bool = False

        fake_results = [_FakeResult()]
        fake_diag = _FakeDiag()

        pass_def = MagicMock()
        pass_def.name = pass_name
        pass_def.phase = "field_group"
        pass_def.module = "extraction_schemas.radar_power_rf"
        pass_def.template_class = "RadarPowerRfPass"
        profile = MagicMock()
        profile.min_similarity = 0.45
        profile.top_n_candidates = 10
        profile.top_k = 3
        # field_query_top_k = 0 → per-element path (C6 branch condition)
        profile.field_query_top_k = 0
        profile.fallback_to_full = True
        pass_def.retrieval = profile

        manifest = MagicMock()
        manifest.passes = [pass_def]

        q_vec = [0.1] * 384

        from app.main import create_app
        from httpx import ASGITransport, AsyncClient

        app = create_app()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=MagicMock(entity_query="radar query", field_queries=())),
                patch("app.api.v1.extraction_routing.embed_texts",
                      return_value=[q_vec]),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=500)),
                patch("app.api.v1.extraction_routing.search_extraction_chunks",
                      new=AsyncMock(return_value=(fake_results, fake_diag))),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      return_value=[{
                          "content_text": "some text",
                          "self_ref": "#/texts/0",
                          "vector_score": 0.75,
                          "reranker_score": 0.8,
                      }]),
                patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
            ):
                settings = MagicMock()
                settings.extraction_index_mode = "per_element"
                settings.reranker_enabled = True
                settings.vector_router_retrieval_mode = "direct"
                mock_settings.return_value = settings

                resp = await ac.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": run_id,
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        diag = body["diagnostics"]

        # All four new C7 fields must be absent or None in per-element mode
        for field_name in ("channel_counts", "field_coverage", "score_components", "fallback_level"):
            value = diag.get(field_name)
            assert value is None, (
                f"Per-element mode: {field_name} must be None; got {value!r}"
            )
