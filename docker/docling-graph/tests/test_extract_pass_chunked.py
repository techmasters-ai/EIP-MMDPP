"""Integration test for Phase 0 Task 0b — chunked-mode /extract-pass.

Asserts that when a request POSTs ``selected_chunks`` to ``/extract-pass``:

* The handler validates and accepts the new optional field.
* ``_sanitize_docling_document`` is NOT invoked (pre-built chunks ARE the
  LLM batches; sanitize would alter docling_document_json byte-for-byte
  identity with the worker's filtered doc).
* ``DocumentChunker.chunk_document`` is NOT invoked.
* The pre-built chunk texts and source_refs round-trip into the
  downstream pipeline (asserted by inspecting the PipelineConfig
  constructed inside ``run_extraction_pass`` via the
  ``build_pipeline_config`` boundary).

The downstream LLM call itself is mocked — these are wire-level tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


_DG_MODULE_NAME = "docling_graph_service_main"


@pytest.fixture
def client(dg_app_module):
    """TestClient that shares the module-level FastAPI app."""
    fastapi_app = dg_app_module.app
    with TestClient(fastapi_app) as c:
        yield c


def _mock_context():
    """Return a stub mimicking run_pipeline's return shape."""
    ctx = MagicMock()
    ctx.knowledge_graph.number_of_nodes.return_value = 0
    ctx.knowledge_graph.number_of_edges.return_value = 0
    ctx.graph_metadata = MagicMock(
        node_count=0, edge_count=0, node_types={}, edge_types={},
    )
    ctx.template_instance.model_dump.return_value = {}
    ctx._upstream_preamble_applied = False
    return ctx


def test_extract_pass_accepts_selected_chunks_field(client):
    """The /extract-pass handler accepts a request with selected_chunks."""
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_context()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_identity",
            "docling_document_json": {"name": "test"},
            "selected_chunks": [
                {
                    "chunk_index": 0,
                    "text": "Pre-built chunk text from worker.",
                    "source_refs": ["#/texts/0", "#/texts/1"],
                    "token_count": 7,
                },
            ],
        })
    assert resp.status_code == 200, resp.text


def test_extract_pass_selected_chunks_silently_drops_extra_fields(client):
    """``SelectedChunkInput`` uses ``extra='ignore'`` — worker-side fields
    like ``chunk_key`` must round-trip without rejection (forward-compat
    with Phase 1's worker-side ``SelectedChunk`` model).
    """
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_context()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_identity",
            "docling_document_json": {"name": "test"},
            "selected_chunks": [
                {
                    "chunk_index": 0,
                    "chunk_key": "chunk_0",  # extra field from worker
                    "text": "Pre-built chunk text.",
                    "source_refs": ["#/texts/0"],
                    "token_count": 5,
                },
            ],
        })
    assert resp.status_code == 200, resp.text


def test_extract_pass_rejects_empty_selected_chunks_list(client):
    """M1 (code-review fix-up): an empty ``selected_chunks`` list is
    ambiguous — ``bool([])`` is False, so the chunked-mode flag inside
    run_extraction_pass would silently fall back to the legacy
    sanitize + DocumentChunker path. The handler must reject ``[]`` at
    the boundary so a chunked-mode caller can never be silently demoted.
    """
    resp = client.post("/extract-pass", json={
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_identity",
        "docling_document_json": {"name": "test"},
        "selected_chunks": [],
    })
    assert resp.status_code == 422, resp.text
    body = resp.json()
    # Surface the explicit reason in the response body — operators
    # debugging a 422 should see why [] is not allowed.
    detail = str(body.get("detail", ""))
    assert "empty list" in detail.lower() or "at least one chunk" in detail.lower(), (
        f"422 detail should explain why [] is rejected; got {detail!r}"
    )


def test_extract_pass_top_level_still_rejects_unknown_fields(client):
    """Parent ``ExtractPassRequest`` retains ``extra='forbid'`` — nested
    ``SelectedChunkInput.extra='ignore'`` must NOT leak into the parent
    contract.
    """
    resp = client.post("/extract-pass", json={
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_identity",
        "docling_document_json": {"name": "test"},
        "fake_unknown_field": "should be rejected",
    })
    assert resp.status_code == 422, resp.text


def test_extract_pass_with_selected_chunks_skips_sanitize(dg_app_module, client):
    """When selected_chunks is present, _sanitize_docling_document MUST NOT
    be invoked. Worker has already filtered the doc; running sanitize on
    top destroys byte-identity with the chunked path.
    """
    sanitize_mock = MagicMock(
        side_effect=lambda doc, stats=None: doc  # passthrough
    )
    # ``run_pipeline`` is imported inside ``run_extraction_pass`` from the
    # ``docling_graph`` library — patch at its origin, not on the main
    # module's namespace.
    with (
        patch.object(dg_app_module, "_sanitize_docling_document", sanitize_mock),
        patch("docling_graph.run_pipeline") as mock_run_pipeline,
        patch.object(dg_app_module, "build_pipeline_config") as mock_build_cfg,
    ):
        mock_run_pipeline.return_value = _mock_context()
        mock_build_cfg.return_value = MagicMock()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_identity",
            "docling_document_json": {
                "name": "test",
                "texts": [{"self_ref": "#/texts/0", "text": "hi"}],
                "body": {"children": [{"$ref": "#/texts/0"}]},
            },
            "selected_chunks": [
                {
                    "chunk_index": 0,
                    "text": "Pre-built chunk.",
                    "source_refs": ["#/texts/0"],
                    "token_count": 4,
                },
            ],
        })

    assert resp.status_code == 200, resp.text
    assert sanitize_mock.call_count == 0, (
        "_sanitize_docling_document must NOT be invoked when "
        "selected_chunks is present in the request."
    )


def test_extract_pass_with_selected_chunks_threads_pre_built_chunks_to_config(
    dg_app_module, client
):
    """When selected_chunks is present, build_pipeline_config MUST receive
    pre_built_chunks as a list of dicts whose ``text`` matches the request
    exactly (byte-equal) and whose ``source_refs`` are preserved.
    """
    chunks_input = [
        {
            "chunk_index": 0,
            "text": "First chunk text — pre-built by worker.",
            "source_refs": ["#/texts/3", "#/texts/4"],
            "token_count": 8,
        },
        {
            "chunk_index": 1,
            "text": "Second chunk text.",
            "source_refs": ["#/texts/9"],
            "token_count": 4,
        },
    ]
    with (
        patch("docling_graph.run_pipeline") as mock_run_pipeline,
        patch.object(dg_app_module, "build_pipeline_config") as mock_build_cfg,
    ):
        mock_run_pipeline.return_value = _mock_context()
        mock_build_cfg.return_value = MagicMock()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_identity",
            "docling_document_json": {
                "name": "test",
                "texts": [{"self_ref": "#/texts/0", "text": "hi"}],
                "body": {"children": [{"$ref": "#/texts/0"}]},
            },
            "selected_chunks": chunks_input,
        })

    assert resp.status_code == 200, resp.text
    assert mock_build_cfg.call_count == 1
    kwargs = mock_build_cfg.call_args.kwargs
    pre_built = kwargs.get("pre_built_chunks")
    assert pre_built is not None, (
        f"build_pipeline_config must receive pre_built_chunks kwarg; "
        f"got kwargs={list(kwargs)}"
    )
    assert len(pre_built) == 2
    # Byte-equal text + identical source_refs round-trip
    assert pre_built[0]["text"] == "First chunk text — pre-built by worker."
    assert pre_built[0]["source_refs"] == ["#/texts/3", "#/texts/4"]
    assert pre_built[0]["chunk_index"] == 0
    assert pre_built[0]["token_count"] == 8
    assert pre_built[1]["text"] == "Second chunk text."
    assert pre_built[1]["source_refs"] == ["#/texts/9"]


def test_extract_pass_without_selected_chunks_still_invokes_sanitize(
    dg_app_module, client
):
    """Backward-compat: absent ``selected_chunks`` keeps existing behavior.
    Sanitize SHOULD run on the regular path."""
    sanitize_mock = MagicMock(
        side_effect=lambda doc, stats=None: doc  # passthrough
    )
    with (
        patch.object(dg_app_module, "_sanitize_docling_document", sanitize_mock),
        patch("docling_graph.run_pipeline") as mock_run_pipeline,
        patch.object(dg_app_module, "build_pipeline_config") as mock_build_cfg,
    ):
        mock_run_pipeline.return_value = _mock_context()
        mock_build_cfg.return_value = MagicMock()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_identity",
            "docling_document_json": {
                "name": "test",
                "texts": [{"self_ref": "#/texts/0", "text": "hi"}],
                "body": {"children": [{"$ref": "#/texts/0"}]},
            },
            # NO selected_chunks
        })

    assert resp.status_code == 200, resp.text
    assert sanitize_mock.call_count == 1, (
        "_sanitize_docling_document MUST run on the existing path "
        "(when selected_chunks is absent)."
    )
    # build_pipeline_config receives no pre_built_chunks on the legacy path
    kwargs = mock_build_cfg.call_args.kwargs
    pre_built = kwargs.get("pre_built_chunks")
    assert pre_built is None or pre_built == [], (
        f"pre_built_chunks must be None/absent on the legacy path; got {pre_built!r}"
    )
