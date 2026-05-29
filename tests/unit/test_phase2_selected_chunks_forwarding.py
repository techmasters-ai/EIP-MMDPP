"""Phase 2 Task 11 — worker forwards SelectedChunk.text to docling-graph.

Verifies the new wire:
  router /v1/extraction/chunk-scope returns selected_chunks
  → _compute_effective_chunk_scope preserves them on the chunk_scope dict
  → _build_extract_pass_request includes them in the POST body
  → byte-identical text passes through

Run standalone:
    python3 -m pytest tests/unit/test_phase2_selected_chunks_forwarding.py -v
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


def _make_pass_def(name: str = "missile_kinematics") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        required=False,
        kind="entities",
        input_mode="document_only",
        module="extraction_schemas.missile_kinematics",
        template_class="MissileKinematicsPass",
        primary_entity_types=["MISSILE_SYSTEM"],
        bridge_entity_types=[],
        extracted_relationship_types=[],
        depends_on=[],
        skip_if_no_upstream_endpoints=False,
        phase="field_group",
        execution=None,
    )


def _sample_router_response_with_selected_chunks() -> dict:
    """Mirror of a real chunk-scope JSON response in narrow_only + merged mode."""
    return {
        "mode": "selected_refs",
        "self_refs": ["#/texts/12", "#/texts/45"],
        "text_by_ref": {
            "#/texts/12": "fragment a",
            "#/texts/45": "fragment b",
        },
        "selected_chunks": [
            {
                "chunk_index": 3,
                "chunk_key": "chunk_3",
                "text": "Merged chunk text — kinematics block 3",
                "source_refs": ["#/texts/12", "#/texts/45"],
                "token_count": 87,
            },
            {
                "chunk_index": 7,
                "chunk_key": "chunk_7",
                "text": "Merged chunk text — kinematics block 7 with envelope rows",
                "source_refs": ["#/texts/100", "#/texts/101"],
                "token_count": 134,
            },
        ],
        "diagnostics": {"selected_chunk_count": 2, "selected_ref_count": 4},
    }


# ---------------------------------------------------------------------------
# _compute_effective_chunk_scope
# ---------------------------------------------------------------------------


def test_compute_effective_preserves_selected_chunks_in_merged_mode(monkeypatch):
    """When extraction_index_mode='merged' + mode='narrow_only', selected_chunks
    from the router response must land on the effective chunk_scope dict so the
    per-pass celery task can forward them to docling-graph."""
    from app.workers import pipeline

    # Pin merged mode + forward-enabled via settings. The settings singleton
    # reads .env at import, and the live .env sets WORKER_FORWARD_SELECTED_CHUNKS
    # =false, so this path must pin the flag True to stay deterministic
    # regardless of the ambient .env value.
    monkeypatch.setattr(pipeline.settings, "extraction_index_mode", "merged",
                        raising=False)
    monkeypatch.setattr(pipeline.settings, "worker_forward_selected_chunks", True,
                        raising=False)

    response = _sample_router_response_with_selected_chunks()
    scope, diag = pipeline._compute_effective_chunk_scope(response, "narrow_only")

    assert scope is not None
    assert scope["mode"] == "selected_refs"
    assert "selected_chunks" in scope
    assert len(scope["selected_chunks"]) == 2
    # byte-identical text
    assert scope["selected_chunks"][0]["text"] == "Merged chunk text — kinematics block 3"
    assert scope["selected_chunks"][1]["text"] == \
        "Merged chunk text — kinematics block 7 with envelope rows"
    assert diag["selected_chunks_forwarded"] is True
    assert diag["selected_chunks_forwarded_count"] == 2


def test_compute_effective_skips_selected_chunks_in_per_element_mode(monkeypatch):
    """When extraction_index_mode='per_element', selected_chunks must NOT
    be retained — preserves byte-identical legacy behavior."""
    from app.workers import pipeline

    monkeypatch.setattr(pipeline.settings, "extraction_index_mode", "per_element",
                        raising=False)

    response = _sample_router_response_with_selected_chunks()
    scope, diag = pipeline._compute_effective_chunk_scope(response, "narrow_only")

    assert scope is not None
    assert "selected_chunks" not in scope
    assert "selected_chunks_forwarded" not in diag


def test_compute_effective_records_no_forward_when_router_omits_chunks(monkeypatch):
    """When merged mode is active but the router did not return selected_chunks
    (e.g. legacy router response, or empty list), diagnostics must record
    forwarded=False rather than silently leaving the field absent."""
    from app.workers import pipeline

    monkeypatch.setattr(pipeline.settings, "extraction_index_mode", "merged",
                        raising=False)
    monkeypatch.setattr(pipeline.settings, "worker_forward_selected_chunks", True,
                        raising=False)

    response = _sample_router_response_with_selected_chunks()
    response["selected_chunks"] = []  # router returned empty list
    scope, diag = pipeline._compute_effective_chunk_scope(response, "narrow_only")

    assert scope is not None
    assert "selected_chunks" not in scope
    assert diag["selected_chunks_forwarded"] is False
    assert diag["selected_chunks_forwarded_count"] == 0


def test_compute_effective_skips_selected_chunks_when_forwarding_disabled(monkeypatch):
    """Kill-switch: with worker_forward_selected_chunks=False, merged mode must
    NOT forward selected_chunks. The worker falls back to the scoped-document
    path so docling-graph runs its pass-aware sanitize + table-normalization +
    re-chunking (recovers recall on table-heavy docs). self_refs still narrow
    the scope; only the verbatim merged-chunk handoff is suppressed. The
    explicit `selected_chunks_forward_disabled` diagnostic makes an A/B run
    verifiable in pipeline_pass_outputs."""
    from app.workers import pipeline

    monkeypatch.setattr(pipeline.settings, "extraction_index_mode", "merged",
                        raising=False)
    monkeypatch.setattr(pipeline.settings, "worker_forward_selected_chunks", False,
                        raising=False)

    response = _sample_router_response_with_selected_chunks()
    scope, diag = pipeline._compute_effective_chunk_scope(response, "narrow_only")

    assert scope is not None
    assert scope["mode"] == "selected_refs"
    # scope still narrows by element refs (scoped-doc path), so recall-bearing
    # self_refs survive — only the verbatim merged-chunk forward is suppressed.
    assert scope["self_refs"] == ["#/texts/12", "#/texts/45"]
    assert "selected_chunks" not in scope
    assert diag["selected_chunks_forwarded"] is False
    assert diag["selected_chunks_forwarded_count"] == 0
    assert diag["selected_chunks_forward_disabled"] is True


# ---------------------------------------------------------------------------
# _build_extract_pass_request
# ---------------------------------------------------------------------------


def test_build_request_includes_selected_chunks_when_provided():
    """_build_extract_pass_request adds selected_chunks to the body when the
    caller passes a non-empty list."""
    from app.workers.pipeline import _build_extract_pass_request

    pass_def = _make_pass_def()
    selected = [
        {"chunk_index": 3, "chunk_key": "chunk_3",
         "text": "merged text 3", "source_refs": ["#/texts/12"], "token_count": 50},
    ]

    body = _build_extract_pass_request(
        bundle_key="air_defense_v3_merged_v1",
        pass_def=pass_def,
        doc_json={"texts": []},
        upstream_refs=None,
        document_id="doc-abc",
        selected_chunks=selected,
    )

    assert body["selected_chunks"] == selected
    assert body["selected_chunks"][0]["text"] == "merged text 3"


def test_build_request_omits_selected_chunks_when_absent_or_empty():
    """Absent or empty selected_chunks must NOT add the field to the request
    body — preserves the legacy /extract-pass shape for per_element mode and
    avoids accidentally triggering docling-graph's chunked-mode branch."""
    from app.workers.pipeline import _build_extract_pass_request

    pass_def = _make_pass_def()
    common = dict(
        bundle_key="air_defense_v3_merged_v1",
        pass_def=pass_def,
        doc_json={"texts": []},
        upstream_refs=None,
        document_id="doc-abc",
    )

    body_default = _build_extract_pass_request(**common)
    assert "selected_chunks" not in body_default

    body_none = _build_extract_pass_request(**common, selected_chunks=None)
    assert "selected_chunks" not in body_none

    body_empty = _build_extract_pass_request(**common, selected_chunks=[])
    assert "selected_chunks" not in body_empty


# ---------------------------------------------------------------------------
# End-to-end wire: _execute_pass_attempt threads chunk_scope through
# ---------------------------------------------------------------------------


def test_execute_pass_attempt_forwards_chunk_scope_selected_chunks():
    """Round-trip: _execute_pass_attempt receives chunk_scope with
    selected_chunks and the resulting HTTP body contains them.

    Mocks _call_extract_pass to capture the request body shape.
    """
    from app.workers import pipeline

    pass_def = _make_pass_def()
    chunk_scope = {
        "mode": "selected_refs",
        "self_refs": ["#/texts/12"],
        "selected_chunks": [
            {"chunk_index": 3, "chunk_key": "chunk_3",
             "text": "merged content", "source_refs": ["#/texts/12"],
             "token_count": 50},
        ],
    }

    captured = {}

    def _fake_call_extract_pass(body, *, timeout):
        captured["body"] = body
        # Minimal valid response shape so _execute_pass_attempt continues happily
        return {
            "pass_output": {"missile_systems": []},
            "metadata": {"node_count": 1, "edge_count": 0,
                         "node_types": {}, "edge_types": {},
                         "extraction_contract": "delta"},
            "diagnostics": {},
        }

    with mock.patch.object(pipeline, "_call_extract_pass",
                           side_effect=_fake_call_extract_pass), \
         mock.patch.object(pipeline, "_should_skip", return_value=False), \
         mock.patch.object(pipeline, "_parse_pass_response",
                           return_value=SimpleNamespace(counts=SimpleNamespace())):
        try:
            pipeline._execute_pass_attempt(
                pipeline_run_id="run-abc",
                pass_def=pass_def,
                manifest=SimpleNamespace(),
                ontology={},
                bundle_key="air_defense_v3_merged_v1",
                doc_json={"texts": []},
                upstream_refs={},
                document_id="doc-abc",
                chunk_scope=chunk_scope,
            )
        except Exception:
            # _execute_pass_attempt's downstream pipeline may raise on the
            # incomplete mocked response; what we care about is the captured
            # body. Re-raise only if no body was captured (= bug in the
            # forwarding path).
            assert captured, "request body was not captured — forwarding broken"

    assert "body" in captured
    body = captured["body"]
    assert "selected_chunks" in body
    assert len(body["selected_chunks"]) == 1
    assert body["selected_chunks"][0]["text"] == "merged content"
