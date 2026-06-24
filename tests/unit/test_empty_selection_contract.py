"""Unit tests for the ``empty_selection`` contract — Task 3.

Verifies that when ``selection_mode == "absolute_union"`` the endpoint:
  - returns ``mode="empty_selection"`` (not ``"full"``/``"would_skip"``) when
    ``select_candidates`` produces no results, and
  - returns ``mode="selected_refs"`` when ``select_candidates`` returns at
    least one candidate, and
  - does NOT enter the E2 fallback ladder (``select_candidates`` called exactly
    once on the primary selection site).

All external deps are mocked.  No live DB, embedder, or reranker required.

Run: pytest tests/unit/test_empty_selection_contract.py -k endpoint -v
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_PIPELINE_RUN_ID = str(uuid.uuid4())
_BUNDLE_KEY = "air_defense_v3"
_PASS_NAME = "missile_kinematics"

_VALID_BODY = {
    "pipeline_run_id": _PIPELINE_RUN_ID,
    "bundle_key": _BUNDLE_KEY,
    "pass_name": _PASS_NAME,
}

# ---------------------------------------------------------------------------
# Helpers — mirrors test_v1_extraction_routing.py patterns
# ---------------------------------------------------------------------------


def _make_mc(idx: int, source_refs: list[str] | None = None):
    """Build a minimal MergedCandidate."""
    from app.services.extraction_candidate_scoring import MergedCandidate
    return MergedCandidate(
        candidate_key=f"run-test:chunk_{idx}",
        chunk_index=idx,
        self_ref=f"chunk_{idx}",
        chunk_text=f"chunk text {idx}",
        source_refs=source_refs if source_refs is not None else [f"#/texts/{idx}"],
        token_count=20,
        page_number=1,
        vector_score=0.8,
        field_scores={},
        alias_hits=0,
        pattern_hits=0,
        negative_hits=0,
        section_hits=0,
        content_type=None,
        retrieval_sources={"dense"},
        supported_field_hints=set(),
    )


def _make_mc_pass_def(selection_mode: str = "absolute_union",
                      fallback_to_full: bool = True):
    """Build a pass def with multi-channel path (field_query_top_k > 0) and
    the given selection_mode on the retrieval profile."""
    rp = MagicMock()
    rp.min_similarity = 0.45
    rp.top_n_candidates = 10
    rp.top_k = 5
    rp.fallback_to_full = fallback_to_full
    rp.field_query_top_k = 3
    rp.rerank_weight = 1.0
    rp.lexical_weight = 0.0
    rp.pattern_weight = 0.0
    rp.section_weight = 0.0
    rp.table_boost = 0.0
    rp.negative_weight = 0.0
    rp.pattern_hit_limit = 50
    rp.lexical_decomposed = False
    rp.field_label_weight = 0.0
    rp.pass_keyword_weight = 0.0
    rp.anchor_text_weight = 0.0
    rp.anchor_section_weight = 0.0
    rp.lexical_keywords = []
    rp.fallback_min_field_coverage = 1
    rp.fallback_similarity_relaxation = 0.07
    # Task 3: selection_mode attribute
    rp.selection_mode = selection_mode
    # signal injection attrs (set by model_copy in the endpoint; MagicMock already
    # passes .model_copy through; we need model_copy to return a similar mock)
    rp.model_copy = lambda update=None, **kw: _rp_copy(rp, update or {})

    pd = MagicMock()
    pd.name = _PASS_NAME
    pd.phase = "field_group"
    pd.module = "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics"
    pd.template_class = "MissileKinematicsPass"
    pd.retrieval = rp
    pd.primary_entity_types = None
    return pd


def _rp_copy(orig: MagicMock, update: dict) -> MagicMock:
    """Lightweight model_copy replacement: return orig with the supplied
    attrs overridden so the absolute_union config-injection path works."""
    copy = MagicMock(wraps=orig)
    copy.selection_mode = orig.selection_mode
    copy.min_similarity = orig.min_similarity
    copy.top_n_candidates = orig.top_n_candidates
    copy.top_k = orig.top_k
    copy.fallback_to_full = orig.fallback_to_full
    copy.field_query_top_k = orig.field_query_top_k
    copy.rerank_weight = orig.rerank_weight
    copy.lexical_weight = orig.lexical_weight
    copy.pattern_weight = orig.pattern_weight
    copy.section_weight = orig.section_weight
    copy.table_boost = orig.table_boost
    copy.negative_weight = orig.negative_weight
    copy.pattern_hit_limit = orig.pattern_hit_limit
    copy.lexical_decomposed = orig.lexical_decomposed
    copy.field_label_weight = orig.field_label_weight
    copy.pass_keyword_weight = orig.pass_keyword_weight
    copy.anchor_text_weight = orig.anchor_text_weight
    copy.anchor_section_weight = orig.anchor_section_weight
    copy.lexical_keywords = orig.lexical_keywords
    copy.fallback_min_field_coverage = orig.fallback_min_field_coverage
    copy.fallback_similarity_relaxation = orig.fallback_similarity_relaxation
    # model_copy on the copy should return the copy itself to avoid recursion
    copy.model_copy = lambda update=None, **kw: copy
    for k, v in update.items():
        setattr(copy, k, v)
    return copy


def _make_mc_state():
    """Minimal MultiChannelState for test mocks."""
    from app.services.extraction_chunk_search import MultiChannelState
    return MultiChannelState(
        rows=[],
        entity_dense=[],
        field_dense={},
        lex_hits={},
        pat_hits={},
        raw_row_count=0,
    )


def _make_mc_diag(pool_size: int = 2):
    return SimpleNamespace(
        raw_row_count=5,
        entity_dense_count=5,
        field_dense_total_count=0,
        per_field_dense_counts={},
        lexical_hit_count=0,
        pattern_hit_count=0,
        pool_size=pool_size,
        filter_strategy="direct_cosine",
    )


def _score_candidates_stub(scored2):
    """Honour return_components=True (Task 18 contract)."""
    def _side_effect(candidates, cfg, *, return_components=False, unit_signature=()):
        if return_components:
            out = []
            for mc, final in scored2:
                comps = {
                    "candidate_key": mc.candidate_key,
                    "final_score": float(final),
                    "unit_gate": 0.0,
                    "table_gate": 0.0,
                }
                out.append((mc, final, comps))
            return out
        return list(scored2)
    return _side_effect


def _make_signals_mock(entity_query: str = "missile kinematics query"):
    signals_mock = MagicMock()
    signals_mock.entity_query = entity_query
    signals_mock.field_queries = ()
    signals_mock.unit_signature = ()
    return signals_mock


def _make_settings_mock():
    settings_mock = MagicMock()
    settings_mock.extraction_index_mode = "merged"
    settings_mock.reranker_enabled = True
    settings_mock.vector_router_retrieval_mode = "direct"
    return settings_mock


def _fake_rerank(query, candidates, top_k):
    return [dict(c, reranker_score=0.85) for c in candidates]


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    from app.main import create_app
    return create_app()


# ---------------------------------------------------------------------------
# Endpoint tests (all names contain "endpoint" so -k endpoint selects them)
# ---------------------------------------------------------------------------

class TestAbsoluteUnionEndpoint:
    """Task 3 — endpoint contract for selection_mode='absolute_union'."""

    @pytest.mark.asyncio
    async def test_endpoint_absolute_union_empty_returns_empty_selection(self, app):
        """absolute_union + select_candidates returns [] → mode=empty_selection,
        self_refs=[], NOT mode=full."""
        pass_def = _make_mc_pass_def(selection_mode="absolute_union")
        manifest = MagicMock()
        manifest.passes = [pass_def]

        mc_a = _make_mc(0)
        pool = [mc_a]
        diag = _make_mc_diag(pool_size=1)
        state = _make_mc_state()

        # select_candidates returns empty — no selections
        scored_empty: list = []

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=_make_signals_mock()),
                patch("app.api.v1.extraction_routing.inject_pass_keywords",
                      side_effect=lambda p, s: p),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                patch("app.api.v1.extraction_routing.search_extraction_chunks_multi_channel_full",
                      new=AsyncMock(return_value=(pool, diag, state))),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.score_candidates",
                      side_effect=_score_candidates_stub(scored_empty)),
                patch("app.api.v1.extraction_routing.select_candidates",
                      return_value=[]),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
                patch("app.api.v1.extraction_routing.get_settings",
                      return_value=_make_settings_mock()),
                patch("app.api.v1.extraction_routing.derive_pass_signal_config",
                      return_value={}),
            ):
                resp = await ac.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "empty_selection", (
            f"absolute_union + empty select_candidates must return mode=empty_selection; "
            f"got mode={data['mode']!r}"
        )
        assert data["self_refs"] == [], data
        assert data["diagnostics"]["mode"] == "empty_selection"
        # fallback_reason is an implementation detail (WHY the pool was empty),
        # not part of the public empty_selection contract — asserted for trace
        # completeness, not as a stability guarantee.
        assert data["diagnostics"]["fallback_reason"] == "no_selected_refs_after_rerank"

    @pytest.mark.asyncio
    async def test_endpoint_absolute_union_nonempty_returns_selected_refs(self, app):
        """absolute_union + select_candidates returns one candidate → mode=selected_refs,
        self_refs non-empty."""
        pass_def = _make_mc_pass_def(selection_mode="absolute_union")
        manifest = MagicMock()
        manifest.passes = [pass_def]

        mc_a = _make_mc(0, source_refs=["#/texts/10"])
        pool = [mc_a]
        diag = _make_mc_diag(pool_size=1)
        state = _make_mc_state()

        scored = [(mc_a, 0.9)]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=_make_signals_mock()),
                patch("app.api.v1.extraction_routing.inject_pass_keywords",
                      side_effect=lambda p, s: p),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                patch("app.api.v1.extraction_routing.search_extraction_chunks_multi_channel_full",
                      new=AsyncMock(return_value=(pool, diag, state))),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.score_candidates",
                      side_effect=_score_candidates_stub(scored)),
                patch("app.api.v1.extraction_routing.select_candidates",
                      return_value=scored),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
                patch("app.api.v1.extraction_routing.get_settings",
                      return_value=_make_settings_mock()),
                patch("app.api.v1.extraction_routing.derive_pass_signal_config",
                      return_value={}),
            ):
                resp = await ac.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "selected_refs", (
            f"absolute_union + non-empty select_candidates must return mode=selected_refs; "
            f"got mode={data['mode']!r}"
        )
        assert len(data["self_refs"]) > 0, (
            f"self_refs must be non-empty for selected_refs mode; got {data['self_refs']!r}"
        )

    @pytest.mark.asyncio
    async def test_endpoint_absolute_union_does_not_enter_ladder(self, app):
        """For absolute_union, the E2 fallback ladder must NOT run.

        Even when coverage is low (enough_candidates returns False), the ladder
        is skipped.  We assert that select_candidates is called exactly ONCE
        (the primary call after the initial C5 scoring — no re-scores inside
        the ladder's Level 1/2/3 sites).
        """
        pass_def = _make_mc_pass_def(selection_mode="absolute_union")
        manifest = MagicMock()
        manifest.passes = [pass_def]

        mc_a = _make_mc(0, source_refs=["#/texts/5"])
        pool = [mc_a]
        diag = _make_mc_diag(pool_size=1)
        state = _make_mc_state()

        scored = [(mc_a, 0.9)]

        # enough_candidates returns False to trigger the ladder guard — but
        # the ladder must be skipped because _is_absolute_union=True.
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=_make_signals_mock()),
                patch("app.api.v1.extraction_routing.inject_pass_keywords",
                      side_effect=lambda p, s: p),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                patch("app.api.v1.extraction_routing.search_extraction_chunks_multi_channel_full",
                      new=AsyncMock(return_value=(pool, diag, state))),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.score_candidates",
                      side_effect=_score_candidates_stub(scored)),
                patch("app.api.v1.extraction_routing.select_candidates",
                      return_value=scored) as mock_select,
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
                patch("app.api.v1.extraction_routing.get_settings",
                      return_value=_make_settings_mock()),
                patch("app.api.v1.extraction_routing.derive_pass_signal_config",
                      return_value={}),
                # enough_candidates returns False to make the coverage check fail
                # (which would normally enter the ladder), but the ladder guard
                # `if not _enough and not _is_absolute_union:` must skip it.
                patch("app.api.v1.extraction_routing.enough_candidates",
                      return_value=False),
                patch("app.api.v1.extraction_routing.enough_field_coverage",
                      return_value=False),
            ):
                resp = await ac.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        # Response must be selected_refs (non-empty pool, ladder skipped)
        assert data["mode"] == "selected_refs", (
            f"absolute_union with non-empty select_candidates must return selected_refs "
            f"even when enough_candidates=False (ladder skipped); got {data['mode']!r}"
        )
        # select_candidates must have been called exactly once (primary site only)
        assert mock_select.call_count == 1, (
            f"select_candidates must be called exactly once for absolute_union "
            f"(no ladder re-scores); called {mock_select.call_count} times"
        )

    @pytest.mark.asyncio
    async def test_endpoint_topk_empty_still_returns_full(self, app):
        """NEGATIVE GATE: a NON-absolute_union (topk) pass with empty
        select_candidates must still return the legacy mode=full — proves the
        empty_selection conversion is correctly gated on _is_absolute_union (an
        inverted condition would surface here)."""
        # selection_mode="topk" → _is_absolute_union must be False.
        pass_def = _make_mc_pass_def(selection_mode="topk", fallback_to_full=True)
        manifest = MagicMock()
        manifest.passes = [pass_def]

        mc_a = _make_mc(0)
        pool = [mc_a]
        diag = _make_mc_diag(pool_size=1)
        state = _make_mc_state()

        scored_empty: list = []

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            with (
                patch("app.api.v1.extraction_routing.load_bundle_manifest",
                      return_value=manifest),
                patch("app.api.v1.extraction_routing._resolve_template_class",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing.build_retrieval_profile",
                      return_value=_make_signals_mock()),
                patch("app.api.v1.extraction_routing.inject_pass_keywords",
                      side_effect=lambda p, s: p),
                patch("app.api.v1.extraction_routing.get_graph_store",
                      return_value=MagicMock()),
                patch("app.api.v1.extraction_routing._async_full_doc_token_estimate",
                      new=AsyncMock(return_value=1000)),
                patch("app.api.v1.extraction_routing.search_extraction_chunks_multi_channel_full",
                      new=AsyncMock(return_value=(pool, diag, state))),
                patch("app.api.v1.extraction_routing.identity_anchor_queries",
                      new=AsyncMock(return_value=[])),
                patch("app.api.v1.extraction_routing.score_candidates",
                      side_effect=_score_candidates_stub(scored_empty)),
                patch("app.api.v1.extraction_routing.select_candidates",
                      return_value=[]),
                patch("app.api.v1.extraction_routing.rrk.rerank",
                      side_effect=_fake_rerank),
                patch("app.api.v1.extraction_routing.get_settings",
                      return_value=_make_settings_mock()),
                patch("app.api.v1.extraction_routing.derive_pass_signal_config",
                      return_value={}),
            ):
                resp = await ac.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "full", (
            f"NON-absolute_union (topk) empty selection must stay mode=full, "
            f"NOT empty_selection; got mode={data['mode']!r}"
        )
        assert data["mode"] != "empty_selection"
        assert data["self_refs"] == []
