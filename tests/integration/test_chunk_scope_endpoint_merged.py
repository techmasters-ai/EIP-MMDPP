"""Integration tests for Phase 1 Task 6 — chunk-scope endpoint merged-mode
expansion.

When ``settings.extraction_index_mode == "merged"`` the endpoint:

1. Expands ``self_refs`` from the **union** of ``source_refs`` across the
   selected merged ExtractionChunk rows (chunk-encounter order, NOT lex-
   sorted).
2. Populates the new ``selected_chunks`` list on the response, where each
   ``SelectedChunk`` carries the merged chunk text byte-for-byte.
3. Leaves ``text_by_ref`` UNCHANGED (still self_ref-keyed; merged chunk
   text rides on ``SelectedChunk.text``, NOT on ``text_by_ref``).
4. Reports ``selected_chunk_count`` and ``expanded_ref_count`` in
   diagnostics.
5. Mirrors ``selected_token_estimate`` to ``selected_chunk_token_estimate``
   for backward-compat dashboards.

Plus a snapshot test that asserts the response shape is backward-compatible
in BOTH modes (no existing field deletions, no semantic changes to existing
fields).

SKIP CONDITIONS
---------------
* ArcadeDB not reachable (inherited from integration conftest).
* Reranker model not available — endpoint short-circuits to mode=full on
  reranker errors, which doesn't exercise the merged-mode population path.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

pytestmark = pytest.mark.integration

_DIM = 1024


def _vec_along(axis: int = 0) -> list[float]:
    """Unit vector with 1.0 at ``axis``, 0 elsewhere — guarantees cosine
    similarity of 1.0 between any two vectors built this way (so all of our
    fixture chunks score above the manifest ``min_similarity`` threshold
    without depending on real bge-m3 semantics).
    """
    v = [0.0] * _DIM
    v[axis] = 1.0
    return v


def _fake_embed(texts: list[str], query: bool = False) -> list[list[float]]:
    """Constant-vector embedder so every chunk + query cosine = 1.0.

    Sidesteps the chicken-and-egg problem of asserting ``mode=selected_refs``
    against a manifest threshold (``min_similarity=0.25``) with synthetic data
    — random unit vectors score near zero and the endpoint short-circuits to
    ``mode=full`` with ``fallback_reason='no_chunks_above_threshold'``.
    """
    return [_vec_along(0) for _ in texts]


def _reranker_available() -> bool:
    """Return True if the reranker model can be loaded."""
    try:
        from app.config import get_settings

        s = get_settings()
        if not s.reranker_enabled:
            return False
        from app.services.reranker import _get_reranker_model

        _get_reranker_model()
        return True
    except Exception:
        return False


# Five merged chunks — token counts and source_refs span values designed to
# (a) make encounter-order vs lex-order distinguishable (`#/texts/100` vs
# `#/texts/35`), (b) verify token-count summation, (c) catch any accidental
# in-place sort.
_MERGED_CHUNKS = [
    {
        "chunk_index": 0,
        "source_refs": ["#/texts/100", "#/texts/35"],
        "token_count": 128,
        "chunk_text": (
            "The SA-2 Dvina radar transmits at 500 kW peak power "
            "on carrier frequency 2900 MHz."
        ),
    },
    {
        "chunk_index": 1,
        "source_refs": ["#/texts/36", "#/texts/37"],
        "token_count": 96,
        "chunk_text": (
            "Effective radiated power 45 dBW, pulse repetition "
            "frequency 1200 Hz."
        ),
    },
    {
        "chunk_index": 2,
        "source_refs": ["#/texts/200"],
        "token_count": 64,
        "chunk_text": "Guidance radar in C-band frequency range.",
    },
    {
        "chunk_index": 3,
        "source_refs": ["#/texts/300", "#/texts/100"],  # repeated ref by design
        "token_count": 80,
        "chunk_text": "Antenna aperture geometry and gain figures.",
    },
    {
        "chunk_index": 4,
        "source_refs": ["#/texts/400"],
        "token_count": 48,
        "chunk_text": "Maximum range performance against low-altitude targets.",
    },
]


def _delete_run(store: "ArcadeDBGraphStore", run_id: str) -> None:
    try:
        store._client.command_sync(
            store._database,
            "sql",
            "DELETE FROM ExtractionChunk WHERE pipeline_run_id = :run_id",
            {"run_id": run_id},
        )
    except Exception:
        pass


def _insert_merged_rows(
    store: "ArcadeDBGraphStore",
    *,
    run_id: str,
    document_id: str,
) -> None:
    """Insert the canonical five-merged-chunk fixture using bge-m3-sized
    deterministic embeddings keyed on chunk_text (so the rerank step sees
    real content_text).
    """
    chunk_texts = [c["chunk_text"] for c in _MERGED_CHUNKS]
    embeddings = _fake_embed(chunk_texts)
    for chunk, emb in zip(_MERGED_CHUNKS, embeddings):
        self_ref = chunk["source_refs"][0]
        store._client.command_sync(
            store._database,
            "sql",
            (
                "INSERT INTO ExtractionChunk SET "
                "vertex_id = :v, pipeline_run_id = :run_id, "
                "document_id = :doc_id, self_ref = :self_ref, "
                "chunk_text = :text, embedding = :emb, page_number = :pn, "
                "modality = 'merged', chunk_index = :ci, "
                "source_refs = :refs, token_count = :tc"
            ),
            {
                "v": f"{run_id}:chunk_{chunk['chunk_index']}",
                "run_id": run_id,
                "doc_id": document_id,
                "self_ref": self_ref,
                "text": chunk["chunk_text"],
                "emb": emb,
                "pn": 1,
                "ci": chunk["chunk_index"],
                "refs": list(chunk["source_refs"]),
                "tc": chunk["token_count"],
            },
        )


async def _post_chunk_scope(
    *,
    arcadedb_store,
    pipeline_run_id: str,
    bundle_key: str = "air_defense_v3_baseline_subset",
    pass_name: str = "radar_power_rf",
):
    """POST to /v1/extraction/chunk-scope with all the standard patches.

    Patches:
      * embed_texts → deterministic fake
      * get_graph_store → use the live fixture store
      * _resolve_template_class + build_retrieval_query → bypass schema-walk
    """
    from pydantic import BaseModel
    from app.main import create_app

    class _FakePassCls(BaseModel):
        pass

    app = create_app()
    with (
        patch(
            "app.api.v1.extraction_routing.embed_texts",
            side_effect=_fake_embed,
        ),
        patch(
            "app.api.v1.extraction_routing.get_graph_store",
            return_value=arcadedb_store,
        ),
        patch(
            "app.api.v1.extraction_routing._resolve_template_class",
            return_value=_FakePassCls,
        ),
        patch(
            "app.api.v1.extraction_routing.build_retrieval_query",
            return_value=(
                "radar power rf ERP frequency transmitter peak power"
            ),
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/v1/extraction/chunk-scope",
                json={
                    "pipeline_run_id": pipeline_run_id,
                    "bundle_key": bundle_key,
                    "pass_name": pass_name,
                },
            )
    return resp


def _force_merged_mode():
    """Override ``settings.extraction_index_mode`` to ``"merged"`` for the
    duration of the test. The Settings object is cached via ``get_settings``
    so we patch the attribute on the singleton.
    """
    from app.config import get_settings

    settings = get_settings()
    return patch.object(settings, "extraction_index_mode", "merged")


# ---------------------------------------------------------------------------
# Step 1a — self_refs populated from source_refs expansion in encounter order
# ---------------------------------------------------------------------------


class TestSelfRefsExpandedInEncounterOrder:
    """``self_refs`` must be the deduplicated union of ``source_refs`` across
    selected merged chunks, preserved in chunk-encounter order (NOT lex).
    """

    @pytest.mark.asyncio
    async def test_self_refs_expanded_in_encounter_order(self, arcadedb_store):
        if not _reranker_available():
            pytest.skip("Reranker not available")

        run_id = f"task6-1a-{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        try:
            _insert_merged_rows(
                arcadedb_store, run_id=run_id, document_id=doc_id,
            )
            with _force_merged_mode():
                resp = await _post_chunk_scope(
                    arcadedb_store=arcadedb_store,
                    pipeline_run_id=run_id,
                )

            assert resp.status_code == 200, resp.text
            data = resp.json()

            if data["mode"] != "selected_refs":
                pytest.skip(
                    f"Endpoint returned mode={data['mode']!r} "
                    f"(reason={data['diagnostics'].get('fallback_reason')!r}) "
                    "— need selected_refs to exercise expansion path"
                )

            # All returned refs must come from the source_refs union of the
            # fixture (lex-or-not, we just need them to be valid).
            all_fixture_refs = {
                ref for c in _MERGED_CHUNKS for ref in c["source_refs"]
            }
            for ref in data["self_refs"]:
                assert ref in all_fixture_refs, (
                    f"Expanded ref {ref!r} not in fixture source_refs"
                )

            # Deduplication: every ref appears at most once.
            assert len(data["self_refs"]) == len(set(data["self_refs"])), (
                "self_refs contains duplicates after expansion — "
                "dedupe-by-seen logic broken"
            )

            # Order assertion: when both `#/texts/100` and `#/texts/35`
            # appear together (chunk 0), 100 must come BEFORE 35 because
            # we preserve insertion order from the source_refs list — NOT
            # lexicographic order.
            if "#/texts/100" in data["self_refs"] and "#/texts/35" in data["self_refs"]:
                assert data["self_refs"].index("#/texts/100") < data["self_refs"].index(
                    "#/texts/35"
                ), (
                    "Expansion appears lex-sorted; '#/texts/100' must precede "
                    "'#/texts/35' under chunk-encounter order"
                )
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1b — selected_chunks[i].text equals row's chunk_text byte-for-byte
# ---------------------------------------------------------------------------


class TestSelectedChunkTextByteIdentity:
    """Each ``selected_chunks[i].text`` must equal the inserted
    ``chunk_text`` byte-for-byte (direct dict access, no sanitization)."""

    @pytest.mark.asyncio
    async def test_selected_chunk_text_byte_identity(self, arcadedb_store):
        if not _reranker_available():
            pytest.skip("Reranker not available")

        run_id = f"task6-1b-{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        try:
            _insert_merged_rows(
                arcadedb_store, run_id=run_id, document_id=doc_id,
            )
            with _force_merged_mode():
                resp = await _post_chunk_scope(
                    arcadedb_store=arcadedb_store,
                    pipeline_run_id=run_id,
                )

            assert resp.status_code == 200, resp.text
            data = resp.json()
            if data["mode"] != "selected_refs":
                pytest.skip(
                    f"Endpoint returned mode={data['mode']!r}, need selected_refs"
                )

            assert data.get("selected_chunks") is not None, (
                "Merged-mode response must populate selected_chunks"
            )
            assert len(data["selected_chunks"]) > 0

            # Index the fixture by chunk_index so we can match regardless of
            # rerank order.
            fixture_by_idx = {c["chunk_index"]: c for c in _MERGED_CHUNKS}
            for sc in data["selected_chunks"]:
                fixture = fixture_by_idx[sc["chunk_index"]]
                assert sc["text"] == fixture["chunk_text"], (
                    f"selected_chunks[{sc['chunk_index']}].text differs from "
                    "the inserted chunk_text byte-for-byte"
                )
                assert sc["chunk_key"] == f"chunk_{sc['chunk_index']}"
                assert sc["source_refs"] == fixture["source_refs"]
                assert sc["token_count"] == fixture["token_count"]
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1c — text_by_ref is UNCHANGED in merged mode (still self_ref-keyed)
# ---------------------------------------------------------------------------


class TestTextByRefUnchangedInMergedMode:
    """``text_by_ref`` MUST stay self_ref-keyed in merged mode. ``chunk_N``
    keys must NOT appear (would silently break ``apply_chunk_scope``).
    """

    @pytest.mark.asyncio
    async def test_text_by_ref_keys_are_self_refs_not_chunk_keys(
        self, arcadedb_store,
    ):
        if not _reranker_available():
            pytest.skip("Reranker not available")

        run_id = f"task6-1c-{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        try:
            _insert_merged_rows(
                arcadedb_store, run_id=run_id, document_id=doc_id,
            )
            with _force_merged_mode():
                resp = await _post_chunk_scope(
                    arcadedb_store=arcadedb_store,
                    pipeline_run_id=run_id,
                )

            assert resp.status_code == 200, resp.text
            data = resp.json()
            if data["mode"] != "selected_refs":
                pytest.skip(
                    f"Endpoint returned mode={data['mode']!r}, need selected_refs"
                )

            text_by_ref = data.get("text_by_ref", {})
            # No keys should start with 'chunk_' — those would be the
            # rev-2 wrong-shape entries that apply_chunk_scope drops.
            for k in text_by_ref.keys():
                assert not k.startswith("chunk_"), (
                    f"text_by_ref must NOT be chunk-keyed in merged mode; "
                    f"got key {k!r}"
                )
            # All keys should look like docling self_refs.
            for k in text_by_ref.keys():
                assert k.startswith("#/"), (
                    f"text_by_ref key {k!r} is not a docling self_ref shape"
                )
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1d — selected_chunk_count + expanded_ref_count reported correctly
# ---------------------------------------------------------------------------


class TestDiagnosticsCountsReportedCorrectly:
    """``diagnostics.selected_chunk_count`` must equal the number of merged
    chunks selected, and ``expanded_ref_count`` must equal the length of
    the expanded ``self_refs`` list.
    """

    @pytest.mark.asyncio
    async def test_diagnostics_counts(self, arcadedb_store):
        if not _reranker_available():
            pytest.skip("Reranker not available")

        run_id = f"task6-1d-{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        try:
            _insert_merged_rows(
                arcadedb_store, run_id=run_id, document_id=doc_id,
            )
            with _force_merged_mode():
                resp = await _post_chunk_scope(
                    arcadedb_store=arcadedb_store,
                    pipeline_run_id=run_id,
                )

            assert resp.status_code == 200, resp.text
            data = resp.json()
            if data["mode"] != "selected_refs":
                pytest.skip(
                    f"Endpoint returned mode={data['mode']!r}, need selected_refs"
                )

            diag = data["diagnostics"]
            assert diag["selected_chunk_count"] == len(data["selected_chunks"]), (
                "selected_chunk_count must equal len(selected_chunks)"
            )
            assert diag["expanded_ref_count"] == len(data["self_refs"]), (
                "expanded_ref_count must equal len(self_refs) in merged mode"
            )
            # selected_ref_count must also equal expanded_ref_count under the
            # plan's contract — it counts the post-expansion refs.
            assert diag["selected_ref_count"] == diag["expanded_ref_count"]
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1e — selected_token_estimate == selected_chunk_token_estimate in merged
# ---------------------------------------------------------------------------


class TestLegacyTokenEstimateMirrorsChunkTokenEstimate:
    """In merged mode, the legacy ``selected_token_estimate`` field MUST
    equal ``selected_chunk_token_estimate`` so existing dashboards continue
    to render meaningful values.
    """

    @pytest.mark.asyncio
    async def test_legacy_token_estimate_mirrors_chunk_token_estimate(
        self, arcadedb_store,
    ):
        if not _reranker_available():
            pytest.skip("Reranker not available")

        run_id = f"task6-1e-{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        try:
            _insert_merged_rows(
                arcadedb_store, run_id=run_id, document_id=doc_id,
            )
            with _force_merged_mode():
                resp = await _post_chunk_scope(
                    arcadedb_store=arcadedb_store,
                    pipeline_run_id=run_id,
                )

            assert resp.status_code == 200, resp.text
            data = resp.json()
            if data["mode"] != "selected_refs":
                pytest.skip(
                    f"Endpoint returned mode={data['mode']!r}, need selected_refs"
                )

            diag = data["diagnostics"]
            assert diag["selected_token_estimate"] == diag["selected_chunk_token_estimate"], (
                "selected_token_estimate must mirror selected_chunk_token_estimate "
                "in merged mode for backward-compat (plan rev 5 Task 6)"
            )
            # And both should equal the sum of token_counts across the
            # selected chunks (the fixture token_counts are deterministic).
            expected_total = sum(
                sc["token_count"] for sc in data["selected_chunks"]
            )
            assert diag["selected_chunk_token_estimate"] == expected_total
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Snapshot test (plan Step 4) — response shape backward-compatible in BOTH modes
# ---------------------------------------------------------------------------


class TestResponseShapeBackwardCompat:
    """Response keys must be a superset of the pre-Task-6 shape in BOTH modes.
    No existing field deletions, no semantic changes to existing fields.

    In per_element mode: ``selected_chunks`` is ``None``; ``self_refs`` is
    the rerank top-K (existing semantics).
    In merged mode: ``selected_chunks`` is a non-null list; ``self_refs``
    is the expanded union.
    """

    _PRE_TASK6_RESPONSE_KEYS = {"mode", "self_refs", "text_by_ref", "diagnostics"}
    _PRE_TASK6_DIAGNOSTICS_KEYS = {
        "mode",
        "fallback_reason",
        "query_text",
        "vector_threshold",
        "vector_score_range",
        "candidate_count",
        "rerank_score_range",
        "selected_ref_count",
        "selected_token_estimate",
        "full_doc_token_estimate",
        "would_skip_if_fallback_disabled",
        "vector_search_ms",
        "rerank_ms",
        "ann_top_k_requested",
        "post_filter_candidate_count",
        "post_filter_retry_count",
        "filter_strategy",
        "short_fetch",
    }

    @pytest.mark.asyncio
    async def test_response_shape_per_element_mode(self, arcadedb_store):
        if not _reranker_available():
            pytest.skip("Reranker not available")

        # In per_element mode the endpoint should behave identically to
        # before Task 6 — selected_chunks=None in the JSON.
        run_id = f"task6-snap-per-{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        try:
            _insert_merged_rows(
                arcadedb_store, run_id=run_id, document_id=doc_id,
            )
            # Do NOT force merged mode — default is per_element.
            resp = await _post_chunk_scope(
                arcadedb_store=arcadedb_store,
                pipeline_run_id=run_id,
            )

            assert resp.status_code == 200, resp.text
            data = resp.json()

            # Pre-Task-6 keys present (additive only).
            missing = self._PRE_TASK6_RESPONSE_KEYS - data.keys()
            assert not missing, f"Missing response keys: {missing}"

            diag = data["diagnostics"]
            missing_d = self._PRE_TASK6_DIAGNOSTICS_KEYS - diag.keys()
            assert not missing_d, f"Missing diagnostics keys: {missing_d}"

            # selected_chunks must be None (not an empty list) in per_element.
            assert data.get("selected_chunks") is None, (
                "selected_chunks must be None in per_element mode to keep "
                "pre-Task-6 byte-identity"
            )

            # New diagnostics fields default to 0 in per_element mode.
            assert diag["selected_chunk_count"] == 0
            assert diag["expanded_ref_count"] == 0
            assert diag["selected_chunk_token_estimate"] == 0
        finally:
            _delete_run(arcadedb_store, run_id)

    @pytest.mark.asyncio
    async def test_response_shape_merged_mode(self, arcadedb_store):
        if not _reranker_available():
            pytest.skip("Reranker not available")

        run_id = f"task6-snap-merged-{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        try:
            _insert_merged_rows(
                arcadedb_store, run_id=run_id, document_id=doc_id,
            )
            with _force_merged_mode():
                resp = await _post_chunk_scope(
                    arcadedb_store=arcadedb_store,
                    pipeline_run_id=run_id,
                )

            assert resp.status_code == 200, resp.text
            data = resp.json()

            missing = self._PRE_TASK6_RESPONSE_KEYS - data.keys()
            assert not missing, f"Missing response keys: {missing}"

            diag = data["diagnostics"]
            missing_d = self._PRE_TASK6_DIAGNOSTICS_KEYS - diag.keys()
            assert not missing_d, f"Missing diagnostics keys: {missing_d}"

            # In merged mode + selected_refs, selected_chunks is populated.
            if data["mode"] == "selected_refs":
                assert isinstance(data.get("selected_chunks"), list)
                assert len(data["selected_chunks"]) > 0
                # Per-item shape.
                for sc in data["selected_chunks"]:
                    assert {
                        "chunk_index",
                        "chunk_key",
                        "text",
                        "source_refs",
                        "token_count",
                    } <= sc.keys()
        finally:
            _delete_run(arcadedb_store, run_id)
