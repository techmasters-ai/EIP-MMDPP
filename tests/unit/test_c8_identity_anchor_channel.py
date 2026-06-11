"""Unit tests for Task C8 — opportunistic non-blocking identity-anchor channel.

TDD: written BEFORE implementation.

Coverage:
  1. identity_anchor_queries helper — run-scoped entity names fetched from
     graph store; union with worker_anchors; dedupe; never raises.
  2. identity_anchor_queries — graph-store error → caught, returns worker_anchors
     or [] without propagating.
  3. Endpoint multi-channel path with non-empty anchors:
     - identity_anchor_count > 0 in diagnostics
     - anchor names added as dense (embed_texts receives anchor texts) AND
       lexical (alias hits recorded) signals
     - "identity_anchor" appears in at least one candidate's retrieval_sources
  4. Endpoint multi-channel path with empty anchors and identity_anchors=None:
     - silently skipped: identity_anchor_count == 0
     - NO error; result byte-identical to no-anchor baseline
  5. ChunkScopeRequest accepts optional identity_anchors field (list[str]|None).
  6. ChunkScopeDiagnostics carries optional identity_anchor_count field (int|None).
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers shared with the other test files (mirrors _vec/_norm/_row/_fake_store)
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
        arr = arr / n
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


def _fake_store(rows: list[dict], entity_rows: list[dict] | None = None) -> Any:
    """Fake ArcadeDBGraphStore. Records SQL calls for assertion.

    entity_rows: if provided, the second query call returns these (identity
    entity rows). If None, the single query mock always returns rows.
    """
    if entity_rows is not None:
        query_mock = AsyncMock(side_effect=[rows, entity_rows])
    else:
        query_mock = AsyncMock(return_value=rows)
    client = SimpleNamespace(query=query_mock)
    return SimpleNamespace(_database="eip_knowledge_graph", _client=client)


# ---------------------------------------------------------------------------
# 1. Schema: ChunkScopeRequest optional identity_anchors field
# ---------------------------------------------------------------------------

class TestChunkScopeRequestSchema:
    def test_identity_anchors_field_optional_and_defaults_none(self):
        """ChunkScopeRequest must accept identity_anchors as an optional list[str]."""
        from app.schemas.extraction_routing import ChunkScopeRequest

        # Without identity_anchors — must not raise
        req = ChunkScopeRequest(
            pipeline_run_id="run-1",
            bundle_key="air_defense_v3",
            pass_name="radar_power_rf",
        )
        assert req.identity_anchors is None

    def test_identity_anchors_field_accepts_list_of_strings(self):
        from app.schemas.extraction_routing import ChunkScopeRequest

        req = ChunkScopeRequest(
            pipeline_run_id="run-1",
            bundle_key="air_defense_v3",
            pass_name="radar_power_rf",
            identity_anchors=["SA-2", "Dvina"],
        )
        assert req.identity_anchors == ["SA-2", "Dvina"]

    def test_identity_anchors_field_accepts_empty_list(self):
        from app.schemas.extraction_routing import ChunkScopeRequest

        req = ChunkScopeRequest(
            pipeline_run_id="run-1",
            bundle_key="air_defense_v3",
            pass_name="radar_power_rf",
            identity_anchors=[],
        )
        assert req.identity_anchors == []


# ---------------------------------------------------------------------------
# 2. Schema: ChunkScopeDiagnostics optional identity_anchor_count field
# ---------------------------------------------------------------------------

class TestChunkScopeDiagnosticsSchema:
    def test_identity_anchor_count_defaults_none(self):
        """ChunkScopeDiagnostics must carry optional identity_anchor_count."""
        from app.schemas.extraction_routing import ChunkScopeDiagnostics

        # Build a minimal valid diagnostics (all required fields)
        diag = ChunkScopeDiagnostics(
            mode="selected_refs",
            query_text="test",
            vector_threshold=0.5,
            candidate_count=1,
            selected_ref_count=1,
            selected_token_estimate=50,
            full_doc_token_estimate=100,
            would_skip_if_fallback_disabled=False,
            vector_search_ms=10,
            rerank_ms=5,
            ann_top_k_requested=50,
            post_filter_candidate_count=1,
            post_filter_retry_count=0,
            filter_strategy="multi_channel",
        )
        # identity_anchor_count must default to None
        assert diag.identity_anchor_count is None

    def test_identity_anchor_count_accepts_int(self):
        from app.schemas.extraction_routing import ChunkScopeDiagnostics

        diag = ChunkScopeDiagnostics(
            mode="selected_refs",
            query_text="test",
            vector_threshold=0.5,
            candidate_count=1,
            selected_ref_count=1,
            selected_token_estimate=50,
            full_doc_token_estimate=100,
            would_skip_if_fallback_disabled=False,
            vector_search_ms=10,
            rerank_ms=5,
            ann_top_k_requested=50,
            post_filter_candidate_count=1,
            post_filter_retry_count=0,
            filter_strategy="multi_channel",
            identity_anchor_count=2,
        )
        assert diag.identity_anchor_count == 2


# ---------------------------------------------------------------------------
# 3. identity_anchor_queries helper — happy path
# ---------------------------------------------------------------------------

class TestIdentityAnchorQueriesHappyPath:
    """identity_anchor_queries should query the graph store for the run's
    committed identity-type entity names and union with worker_anchors."""

    @pytest.mark.asyncio
    async def test_returns_entity_names_from_store(self):
        """When entity rows exist for the run, returns their names."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        entity_rows = [
            {"name": "SA-2", "entity_type": "RADAR_SYSTEM"},
            {"name": "Dvina", "entity_type": "RADAR_SYSTEM"},
        ]
        query_mock = AsyncMock(return_value=entity_rows)
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=["RADAR_SYSTEM"],
            worker_anchors=None,
        )

        assert "SA-2" in result
        assert "Dvina" in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_unions_with_worker_anchors_and_dedupes(self):
        """worker_anchors are unioned with graph-store results; duplicates removed."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        entity_rows = [{"name": "SA-2", "entity_type": "RADAR_SYSTEM"}]
        query_mock = AsyncMock(return_value=entity_rows)
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=["RADAR_SYSTEM"],
            worker_anchors=["SA-2", "FAN SONG"],  # SA-2 is a duplicate
        )

        assert "SA-2" in result
        assert "FAN SONG" in result
        # Deduplicated: SA-2 appears exactly once
        assert result.count("SA-2") == 1

    @pytest.mark.asyncio
    async def test_multiple_identity_types_queried(self):
        """Each identity type is queried separately; results are merged."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        # Two calls: first for RADAR_SYSTEM, second for MISSILE_SYSTEM
        call_results = [
            [{"name": "SA-2", "entity_type": "RADAR_SYSTEM"}],
            [{"name": "S-75", "entity_type": "MISSILE_SYSTEM"}],
        ]
        query_mock = AsyncMock(side_effect=call_results)
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
            worker_anchors=None,
        )

        assert "SA-2" in result
        assert "S-75" in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_result_when_no_committed_entities(self):
        """When the graph store returns no entities, returns empty list."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        query_mock = AsyncMock(return_value=[])
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=["RADAR_SYSTEM"],
            worker_anchors=None,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_identity_types_returns_worker_anchors_only(self):
        """When identity_types is empty, no DB query made; returns worker_anchors."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        query_mock = AsyncMock(return_value=[])
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=[],
            worker_anchors=["SA-2"],
        )

        # No DB calls needed for empty identity_types
        assert "SA-2" in result
        query_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 4. identity_anchor_queries helper — error handling
# ---------------------------------------------------------------------------

class TestIdentityAnchorQueriesErrorHandling:
    """identity_anchor_queries must NEVER raise — errors return worker_anchors or []."""

    @pytest.mark.asyncio
    async def test_graph_store_exception_returns_worker_anchors(self):
        """When the graph store raises, returns worker_anchors (not raising)."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        query_mock = AsyncMock(side_effect=RuntimeError("DB connection failed"))
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        # Must not raise
        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=["RADAR_SYSTEM"],
            worker_anchors=["SA-2"],
        )

        assert result == ["SA-2"]

    @pytest.mark.asyncio
    async def test_graph_store_exception_returns_empty_when_no_worker_anchors(self):
        """When the store raises and worker_anchors is None, returns []."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        query_mock = AsyncMock(side_effect=Exception("timeout"))
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=["RADAR_SYSTEM"],
            worker_anchors=None,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_partial_type_failure_does_not_drop_successes(self):
        """When one type raises, other types' results are still returned."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        # First call succeeds; second raises
        call_results = [
            [{"name": "SA-2", "entity_type": "RADAR_SYSTEM"}],
            RuntimeError("MISSILE_SYSTEM vertex type not found"),
        ]
        query_mock = AsyncMock(side_effect=call_results)
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        # Must not raise; returns what succeeded
        result = await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-X",
            identity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
            worker_anchors=None,
        )

        assert "SA-2" in result


# ---------------------------------------------------------------------------
# 5. Endpoint — identity anchors fire: identity_anchor_count > 0, sources tagged
# ---------------------------------------------------------------------------

class TestEndpointIdentityAnchorFires:
    """When identity anchors are non-empty in the multi-channel path:
    - identity_anchor_count > 0 in diagnostics
    - anchor embed queries included in embed_texts calls
    - "identity_anchor" appears in at least one candidate's retrieval_sources
      (in score_components diagnostics)
    """

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
        # Decomposed-lexical knobs pinned to real RetrievalProfile defaults so
        # the MagicMock profile exercises the legacy (flag-off) path and feeds
        # an EMPTY keyword list into keyword_hit_counts (not an auto-MagicMock).
        profile.lexical_decomposed = False
        profile.field_label_weight = 0.0
        profile.pass_keyword_weight = 0.0
        profile.anchor_text_weight = 0.0
        profile.anchor_section_weight = 0.0
        profile.lexical_keywords = []
        # Gate pin: unit_gate=False (strict is-True guard) so a MagicMock cfg
        # can never accidentally enable G1/G2 and corrupt byte-identical tests.
        profile.unit_gate = False
        pd.retrieval = profile
        return pd

    @pytest.mark.asyncio
    async def test_identity_anchor_count_gt_zero_when_anchors_present(self):
        """With identity anchors committed: diagnostics.identity_anchor_count > 0."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        chunk_rows = [
            _row(
                f"chunk_{i}",
                _norm(_vec(float(i + 1), 0)),
                pipeline_run_id=run_id,
                chunk_index=i,
                chunk_text=f"SA-2 radar system chunk {i}",
                source_refs=[f"#/texts/{i}"],
            )
            for i in range(4)
        ]

        # Mock identity entity rows returned for RADAR_SYSTEM type query
        identity_entity_rows = [
            {"name": "SA-2", "entity_type": "RADAR_SYSTEM"},
        ]

        from app.main import create_app
        from httpx import ASGITransport, AsyncClient

        app = create_app()

        pass_def = self._make_pass_def(top_n_candidates=6, top_k=2)
        manifest = MagicMock()
        manifest.passes = [pass_def]
        manifest.passes = [pass_def]
        # Identity passes to derive identity_types from
        identity_pass = MagicMock()
        identity_pass.phase = "identity"
        identity_pass.primary_entity_types = ["RADAR_SYSTEM"]
        manifest.passes = [pass_def, identity_pass]

        q_vec = _norm(_vec(1, 0))
        signals_mock = MagicMock()
        signals_mock.entity_query = "radar power rf query"
        signals_mock.field_queries = ()

        # The store is queried twice:
        #   1st call: identity_anchor_queries → entity type query (identity_entity_rows)
        #   2nd call: fetch_extraction_chunks_for_run → chunk SELECT (chunk_rows)
        # Order matters: identity_anchor_queries is called BEFORE multi-channel.
        combined_query_mock = AsyncMock(side_effect=[identity_entity_rows, chunk_rows])
        store = SimpleNamespace(
            _database="eip_knowledge_graph",
            _client=SimpleNamespace(query=combined_query_mock),
        )

        def _fake_rerank(query, candidates, top_k):
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = float(len(candidates) - i)
                scored.append(c2)
            return scored

        embed_calls = []

        def _fake_embed(texts, query=False):
            embed_calls.append(list(texts))
            return [_norm(_vec(1, 0))] * len(texts)

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
                      side_effect=_fake_embed),
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
                        "identity_anchors": None,  # worker provides None; endpoint uses store
                    },
                )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        diag = body["diagnostics"]

        # identity_anchor_count must be > 0 when anchors were found
        assert diag.get("identity_anchor_count") is not None, (
            "identity_anchor_count must be present in diagnostics"
        )
        assert diag["identity_anchor_count"] > 0, (
            f"identity_anchor_count expected > 0 when identity entities present; "
            f"got {diag['identity_anchor_count']}"
        )

    @pytest.mark.asyncio
    async def test_identity_anchor_source_in_retrieval_sources(self):
        """Chunks matching anchor names have 'identity_anchor' in retrieval_sources."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        # Chunk with anchor name in text
        anchor_text = "SA-2 radar system operates at high altitude"
        chunk_rows = [
            _row(
                "chunk_0",
                _norm(_vec(1, 0)),
                pipeline_run_id=run_id,
                chunk_index=0,
                chunk_text=anchor_text,
                source_refs=["#/texts/0"],
            ),
        ]
        identity_entity_rows = [{"name": "SA-2", "entity_type": "RADAR_SYSTEM"}]

        from app.main import create_app
        from httpx import ASGITransport, AsyncClient

        app = create_app()

        pass_def = self._make_pass_def(top_n_candidates=6, top_k=2)
        identity_pass = MagicMock()
        identity_pass.phase = "identity"
        identity_pass.primary_entity_types = ["RADAR_SYSTEM"]
        manifest = MagicMock()
        manifest.passes = [pass_def, identity_pass]

        q_vec = _norm(_vec(1, 0))
        signals_mock = MagicMock()
        signals_mock.entity_query = "radar power rf query"
        signals_mock.field_queries = ()

        # 1st call: identity_anchor_queries → identity rows
        # 2nd call: fetch_extraction_chunks_for_run → chunk rows
        combined_query_mock = AsyncMock(side_effect=[identity_entity_rows, chunk_rows])
        store = SimpleNamespace(
            _database="eip_knowledge_graph",
            _client=SimpleNamespace(query=combined_query_mock),
        )

        def _fake_rerank(query, candidates, top_k):
            scored = []
            for i, c in enumerate(candidates):
                c2 = dict(c)
                c2["reranker_score"] = float(len(candidates) - i)
                scored.append(c2)
            return scored

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

        # Check score_components: any candidate with retrieval_sources including "identity_anchor"
        sc = diag.get("score_components") or []
        if sc:
            sources_sets = [set(entry.get("retrieval_sources", [])) for entry in sc]
            has_anchor_source = any("identity_anchor" in s for s in sources_sets)
            # The anchor "SA-2" appears in chunk_0's text, so it should have lexical hit
            # identity_anchor source should appear when chunk text contains anchor name
            assert has_anchor_source, (
                f"Expected 'identity_anchor' in at least one candidate's retrieval_sources; "
                f"got sources: {sources_sets}"
            )


# ---------------------------------------------------------------------------
# 6. Endpoint — empty anchors path: no-op, no regression
# ---------------------------------------------------------------------------

class TestEndpointIdentityAnchorNoOp:
    """When anchors are empty or not resolved: identity_anchor_count == 0,
    no error, multi-channel result identical to the no-anchor baseline."""

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
        # Decomposed-lexical knobs pinned to real RetrievalProfile defaults so
        # the MagicMock profile exercises the legacy (flag-off) path and feeds
        # an EMPTY keyword list into keyword_hit_counts (not an auto-MagicMock).
        profile.lexical_decomposed = False
        profile.field_label_weight = 0.0
        profile.pass_keyword_weight = 0.0
        profile.anchor_text_weight = 0.0
        profile.anchor_section_weight = 0.0
        profile.lexical_keywords = []
        # Gate pin: unit_gate=False (strict is-True guard) so a MagicMock cfg
        # can never accidentally enable G1/G2 and corrupt byte-identical tests.
        profile.unit_gate = False
        pd.retrieval = profile
        return pd

    @pytest.mark.asyncio
    async def test_no_anchors_identity_anchor_count_zero(self):
        """When no identity entities committed and no worker_anchors:
        identity_anchor_count == 0, no error."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        chunk_rows = [
            _row(
                f"chunk_{i}",
                _norm(_vec(float(i + 1), 0)),
                pipeline_run_id=run_id,
                chunk_index=i,
                chunk_text=f"technical data chunk {i}",
                source_refs=[f"#/texts/{i}"],
            )
            for i in range(3)
        ]

        # Graph store returns empty for identity entity query
        combined_query_mock = AsyncMock(side_effect=[chunk_rows, []])
        store = SimpleNamespace(
            _database="eip_knowledge_graph",
            _client=SimpleNamespace(query=combined_query_mock),
        )

        pass_def = self._make_pass_def(top_n_candidates=6, top_k=2)
        # No identity pass → no identity types derived
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

        # identity_anchor_count must be 0 (or None is also acceptable when no anchors)
        anchor_count = diag.get("identity_anchor_count")
        assert anchor_count is None or anchor_count == 0, (
            f"identity_anchor_count expected 0 or None when no anchors; got {anchor_count}"
        )

    @pytest.mark.asyncio
    async def test_graph_store_error_for_identity_not_fail(self):
        """When identity_anchor_queries raises, the multi-channel path continues without error."""
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        chunk_rows = [
            _row(
                f"chunk_{i}",
                _norm(_vec(float(i + 1), 0)),
                pipeline_run_id=run_id,
                chunk_index=i,
                source_refs=[f"#/texts/{i}"],
            )
            for i in range(3)
        ]

        # Call order:
        #   1st call: identity_anchor_queries → raises (simulates DB error)
        #   2nd call: fetch_extraction_chunks_for_run → returns chunk rows (succeeds)
        # identity_anchor_queries must catch the error and return [] — then the
        # multi-channel path proceeds with empty anchors (no-op) and returns 200.
        query_mock = AsyncMock(side_effect=[RuntimeError("ArcadeDB identity type query failed"), chunk_rows])
        store = SimpleNamespace(
            _database="eip_knowledge_graph",
            _client=SimpleNamespace(query=query_mock),
        )

        pass_def = self._make_pass_def(top_n_candidates=6, top_k=2)
        identity_pass = MagicMock()
        identity_pass.phase = "identity"
        identity_pass.primary_entity_types = ["RADAR_SYSTEM"]
        manifest = MagicMock()
        manifest.passes = [pass_def, identity_pass]

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

                # Must return 200 even when identity query fails
                resp = await ac.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": run_id,
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        # Non-blocking: must return 200 (or at least not 500)
        assert resp.status_code == 200, (
            f"Expected 200 when identity query fails (non-blocking); got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_empty_anchors_no_regression_vs_baseline(self):
        """Empty anchors path: multi-channel response must be byte-identical to
        the baseline (no anchors at all).

        We run the endpoint twice:
          - baseline: with identity_anchor_queries mocked to return []
          - empty:    same, with identity_anchors=None in body
        Both responses must have the same mode, selected_ref_count, and
        chunk count.
        """
        run_id = str(uuid.uuid4())
        bundle_key = "air_defense_v3_baseline_subset"
        pass_name = "radar_power_rf"

        chunk_rows = [
            _row(
                f"chunk_{i}",
                _norm(_vec(float(i + 1), 0)),
                pipeline_run_id=run_id,
                chunk_index=i,
                chunk_text=f"technical data chunk {i}",
                source_refs=[f"#/texts/{i}"],
            )
            for i in range(3)
        ]

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

        async def _run_endpoint(extra_body: dict | None = None):
            store = SimpleNamespace(
                _database="eip_knowledge_graph",
                _client=SimpleNamespace(query=AsyncMock(return_value=chunk_rows)),
            )
            body = {
                "pipeline_run_id": run_id,
                "bundle_key": bundle_key,
                "pass_name": pass_name,
            }
            if extra_body:
                body.update(extra_body)

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
                    # Mock identity_anchor_queries to always return [] for this test
                    patch("app.api.v1.extraction_routing.identity_anchor_queries",
                          new=AsyncMock(return_value=[])),
                    patch("app.api.v1.extraction_routing.get_settings") as mock_settings,
                ):
                    settings = MagicMock()
                    settings.extraction_index_mode = "merged"
                    settings.reranker_enabled = True
                    settings.vector_router_retrieval_mode = "direct"
                    mock_settings.return_value = settings

                    resp = await ac.post("/v1/extraction/chunk-scope", json=body)

            return resp

        resp_baseline = await _run_endpoint()
        resp_empty = await _run_endpoint({"identity_anchors": None})

        assert resp_baseline.status_code == 200, resp_baseline.text
        assert resp_empty.status_code == 200, resp_empty.text

        b1 = resp_baseline.json()
        b2 = resp_empty.json()

        # Core fields must match (no regression)
        assert b1["mode"] == b2["mode"], (
            f"mode diverged: baseline={b1['mode']!r} vs empty={b2['mode']!r}"
        )
        d1 = b1["diagnostics"]
        d2 = b2["diagnostics"]
        assert d1["selected_ref_count"] == d2["selected_ref_count"], (
            f"selected_ref_count diverged: baseline={d1['selected_ref_count']} "
            f"vs empty={d2['selected_ref_count']}"
        )
        assert d1.get("selected_chunk_count", 0) == d2.get("selected_chunk_count", 0), (
            "selected_chunk_count diverged between baseline and empty-anchor paths"
        )


# ---------------------------------------------------------------------------
# 7. identity_anchor_queries — query uses pipeline_run_id filter
# ---------------------------------------------------------------------------

class TestIdentityAnchorQueriesScope:
    """The graph query must be scoped to the pipeline_run_id — not global."""

    @pytest.mark.asyncio
    async def test_query_includes_pipeline_run_id(self):
        """The SQL query passed to _client.query must filter by pipeline_run_id."""
        from app.services.extraction_chunk_search import identity_anchor_queries

        query_mock = AsyncMock(return_value=[{"name": "SA-2", "entity_type": "RADAR_SYSTEM"}])
        store = SimpleNamespace(
            _database="eip_test",
            _client=SimpleNamespace(query=query_mock),
        )

        await identity_anchor_queries(
            store=store,
            pipeline_run_id="run-SCOPE-TEST",
            identity_types=["RADAR_SYSTEM"],
            worker_anchors=None,
        )

        # Verify the query was called and the SQL includes the run_id filter
        assert query_mock.call_count >= 1
        call_args = query_mock.call_args_list[0]
        # Arguments: (database, "sql", sql_string, params) positional
        all_args = list(call_args.args) + list(call_args.kwargs.values())
        combined = str(all_args)
        assert "run-SCOPE-TEST" in combined or "run_id" in combined or "pipeline_run_id" in combined, (
            f"pipeline_run_id not found in query args: {all_args}"
        )


# ---------------------------------------------------------------------------
# 8. Search-level separation: alias_hits fires WITH anchors, not WITHOUT.
#
# Locks in the load-bearing behavior the worker-side anchor delivery exists to
# unlock: a chunk whose text NAMES a committed identity entity gets lexical
# credit (alias_hits > 0 + "identity_anchor" source) ONLY when that entity name
# is supplied to the multi-channel search as an anchor.  With empty anchors the
# SAME chunk gets alias_hits == 0 (byte-identical to today's inert C8).
# ---------------------------------------------------------------------------

class TestSearchLevelAliasHitsSeparation:
    """Directly exercise search_extraction_chunks_multi_channel_full with vs
    without identity_anchors and assert the alias_hits separation on one chunk.
    """

    def _make_profile(self, top_n_candidates: int = 6):
        profile = MagicMock()
        profile.min_similarity = 0.0
        profile.top_n_candidates = top_n_candidates
        profile.field_query_top_k = 5
        profile.pattern_hit_limit = 50
        profile.unit_gate = False  # pin: MagicMock attr must not enable the gate
        return profile

    def _signals(self):
        signals = MagicMock()
        signals.entity_query = "radar power rf query"
        signals.field_queries = ()
        return signals

    async def _run_search(self, *, identity_anchors, anchor_text):
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_multi_channel_full,
        )

        run_id = "run-SEP"
        chunk_rows = [
            _row(
                "chunk_0",
                _norm(_vec(1, 0)),
                pipeline_run_id=run_id,
                chunk_index=0,
                chunk_text=anchor_text,
                source_refs=["#/texts/0"],
            ),
        ]
        # Only the chunk SELECT is hit here (identity_anchor_queries is the
        # endpoint's job; this test calls the search function directly).
        store = _fake_store(rows=chunk_rows)

        def _fake_embed(texts, query=False):
            return [_norm(_vec(1, 0))] * len(texts)

        with patch(
            "app.services.extraction_chunk_search.embed_texts",
            side_effect=_fake_embed,
        ):
            pool, _diag, _state = await search_extraction_chunks_multi_channel_full(
                self._signals(),
                run_id,
                self._make_profile(),
                store=store,
                identity_anchors=identity_anchors,
            )
        return pool

    @pytest.mark.asyncio
    async def test_alias_hits_fires_with_anchors(self):
        """A chunk naming the anchor entity gets alias_hits > 0 WITH anchors."""
        pool = await self._run_search(
            identity_anchors=["SA-2"],
            anchor_text="SA-2 radar system operates at high altitude",
        )
        assert pool, "expected a non-empty candidate pool"
        target = next((mc for mc in pool if mc.self_ref == "chunk_0"), None)
        assert target is not None
        assert target.alias_hits > 0, (
            f"chunk naming the anchor should have alias_hits > 0; got {target.alias_hits}"
        )
        assert "identity_anchor" in target.retrieval_sources

    @pytest.mark.asyncio
    async def test_alias_hits_zero_without_anchors(self):
        """The SAME chunk gets alias_hits == 0 when anchors are empty/None."""
        pool = await self._run_search(
            identity_anchors=None,
            anchor_text="SA-2 radar system operates at high altitude",
        )
        assert pool, "expected a non-empty candidate pool"
        target = next((mc for mc in pool if mc.self_ref == "chunk_0"), None)
        assert target is not None
        assert target.alias_hits == 0, (
            f"with no anchors the chunk must have alias_hits == 0; got {target.alias_hits}"
        )
        assert "identity_anchor" not in target.retrieval_sources
