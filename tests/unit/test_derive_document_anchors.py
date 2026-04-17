"""D-4: derive_document_anchors Celery task tests.

The task runs between derive_image_embeddings and derive_ontology_graph.
It loads the persisted docling_document.json via
``_build_docling_document_json``, runs the anchor walker (D-3), upserts
SECTION/FIGURE/TABLE/DOCUMENT vertices via ``upsert_nodes_batch_sync``,
then wires structural edges via ``create_structural_edge_sync``.

Tests mock the MinIO JSON loader and the graph store; they do not hit
the real pipeline run / StageRun tables — those are exercised by
integration tests (D-6 / Chunk G).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.docling_anchors import walk
from app.services.extraction_merge import (
    LogicalIdentity,
    MergedEdgeRecord,
    MergedEntityRecord,
    MergedExtraction,
)
from app.workers.pipeline import derive_document_anchors


def _merged_fixture(doc_uuid: str, run_id: str) -> MergedExtraction:
    """Build a tiny MergedExtraction carrying 1 DOCUMENT + 1 SECTION +
    1 HAS_SECTION edge — enough to exercise upsert + structural-edge
    wiring without dragging in a full Docling fixture.
    """
    doc_identity = LogicalIdentity(
        entity_type="DOCUMENT",
        identity_field_names=("document_number",),
        identity_tuple=("TM 9-1425-386-12",),
        scope="global",
        document_id=None,
    )
    sec_identity = LogicalIdentity(
        entity_type="SECTION",
        identity_field_names=("section_number",),
        identity_tuple=("1",),
        scope="document",
        document_id=doc_uuid,
    )
    doc_rec = MergedEntityRecord(
        identity=doc_identity,
        properties={"document_id": doc_uuid, "title": "Manual"},
        confidence=1.0,
        pass_origins={"document_anchors"},
        display_label="TM 9-1425-386-12",
    )
    sec_rec = MergedEntityRecord(
        identity=sec_identity,
        properties={"document_id": doc_uuid, "heading": "Chapter 1"},
        confidence=1.0,
        pass_origins={"document_anchors"},
        display_label="1",
    )
    return MergedExtraction(
        entities=[doc_rec, sec_rec],
        edges=[
            MergedEdgeRecord(
                from_identity=doc_identity,
                to_identity=sec_identity,
                rel_type="HAS_SECTION",
                confidence=1.0,
                pass_origins={"document_anchors"},
            ),
        ],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id=run_id,
        document_id=doc_uuid,
    )


@pytest.fixture
def stubs(monkeypatch):
    """Common patches: DB accessor, status updates, stage-run writes,
    docling_doc_json loader, ontology loader, graph store, walker."""
    # _get_db → returns a MagicMock with commit/close methods.
    fake_db = MagicMock()
    monkeypatch.setattr("app.workers.pipeline._get_db", lambda: fake_db)
    # _get_pipeline_run_id → stable run id.
    monkeypatch.setattr(
        "app.workers.pipeline._get_pipeline_run_id",
        lambda db, doc_id: "run-id-1",
    )
    # Status + stage-run side effects are fire-and-forget; just swallow them.
    monkeypatch.setattr(
        "app.workers.pipeline._update_document_status",
        lambda *a, **kw: None,
    )
    stage_run_calls: list[tuple] = []
    monkeypatch.setattr(
        "app.workers.pipeline._update_stage_run",
        lambda db, run_id, name, status, **kw: stage_run_calls.append(
            (name, status, kw.get("metrics"))
        ),
    )
    # JSON loader — returns a sentinel dict the walker will never actually
    # parse because we stub walk() next.
    doc_json = {"name": "fake", "fake": True}
    monkeypatch.setattr(
        "app.workers.pipeline._build_docling_document_json",
        lambda document_id: doc_json,
    )
    # Ontology loader — ontology dict is unused by walker-sourced passes
    # (model_config is the SSoT), so returning {} is fine.
    monkeypatch.setattr(
        "app.services.ontology_bundles.load_ontology",
        lambda: {},
        raising=False,
    )
    # Graph store — track upsert + structural-edge calls.
    graph_store = MagicMock()
    graph_store.upsert_nodes_batch_sync = MagicMock(
        return_value=["#1:1", "#1:2"]
    )
    graph_store.create_structural_edge_sync = MagicMock(
        return_value="#2:1"
    )
    monkeypatch.setattr(
        "app.workers.pipeline.get_graph_store",
        lambda: graph_store,
    )
    # Walker — return our fixture; pytest captures the exact call.
    walk_calls: list[tuple] = []

    def fake_walk(docling_doc_json, document_uuid, pipeline_run_id, ontology):
        walk_calls.append((docling_doc_json, document_uuid, pipeline_run_id, ontology))
        return _merged_fixture(document_uuid, pipeline_run_id)

    monkeypatch.setattr("app.services.docling_anchors.walk", fake_walk)

    return {
        "db": fake_db,
        "graph_store": graph_store,
        "stage_run_calls": stage_run_calls,
        "walk_calls": walk_calls,
        "doc_json": doc_json,
    }


def test_derive_document_anchors_wires_upserts_and_structural_edges(stubs):
    result = derive_document_anchors("doc-uuid-1", run_id="run-id-1")

    # Walker was called with the JSON loader output + IDs.
    assert len(stubs["walk_calls"]) == 1
    doc_json, doc_uuid, run_id, _ontology = stubs["walk_calls"][0]
    assert doc_json == stubs["doc_json"]
    assert doc_uuid == "doc-uuid-1"
    assert run_id == "run-id-1"

    # Vertex upsert — single batch call with two node records.
    stubs["graph_store"].upsert_nodes_batch_sync.assert_called_once()
    node_batch_args = stubs["graph_store"].upsert_nodes_batch_sync.call_args
    node_records = node_batch_args.args[0] if node_batch_args.args else node_batch_args.kwargs["records"]
    assert len(node_records) == 2
    entity_types = {r.entity_type for r in node_records}
    assert entity_types == {"DOCUMENT", "SECTION"}

    # Structural edge — one call for the HAS_SECTION edge.
    assert stubs["graph_store"].create_structural_edge_sync.call_count == 1
    edge_call = stubs["graph_store"].create_structural_edge_sync.call_args
    edge_args = edge_call.args
    # (from_rid, to_rid, rel_type) — from_rid is the DOCUMENT (first upsert),
    # to_rid is the SECTION (second upsert), rel_type="HAS_SECTION".
    assert edge_args[0] == "#1:1"
    assert edge_args[1] == "#1:2"
    assert edge_args[2] == "HAS_SECTION"

    # StageRun lifecycle: RUNNING then COMPLETE.
    stage_statuses = [status for _name, status, _m in stubs["stage_run_calls"]]
    assert "RUNNING" in stage_statuses
    assert "COMPLETE" in stage_statuses
    # COMPLETE metrics should name section/figure/table/document counts.
    complete_metrics = [
        m for name, s, m in stubs["stage_run_calls"]
        if s == "COMPLETE" and name == "derive_document_anchors"
    ][0]
    assert complete_metrics["section_count"] == 1
    assert complete_metrics["figure_count"] == 0
    assert complete_metrics["table_count"] == 0
    assert complete_metrics["document_ontology_emitted"] is True

    # Return payload carries the same metrics.
    assert result["stage"] == "derive_document_anchors"
    assert result["status"] == "ok"
    assert result["section_count"] == 1


def test_derive_document_anchors_marks_stage_failed_on_exception(stubs, monkeypatch):
    """When the walker raises, the task marks the StageRun FAILED and
    re-raises. Document status stays PROCESSING (outer orchestrator
    handles PARTIAL_COMPLETE on final retry)."""

    def exploding_walk(*a, **kw):
        raise RuntimeError("walk blew up")

    monkeypatch.setattr("app.services.docling_anchors.walk", exploding_walk)

    with pytest.raises(RuntimeError, match="walk blew up"):
        derive_document_anchors("doc-uuid-1", run_id="run-id-1")

    statuses = [status for _name, status, _m in stubs["stage_run_calls"]]
    assert "FAILED" in statuses
    # upsert_nodes_batch_sync never called because walker raised before it.
    stubs["graph_store"].upsert_nodes_batch_sync.assert_not_called()


def test_derive_document_anchors_skips_edges_when_no_document_emitted(stubs, monkeypatch):
    """When walker returns no DOCUMENT (no designator extractable), the
    task still upserts SECTION/FIGURE/TABLE but the edges list may be
    empty — structural-edge call count matches edges length."""

    def walk_no_doc(docling_doc_json, document_uuid, pipeline_run_id, ontology):
        sec_identity = LogicalIdentity(
            entity_type="SECTION",
            identity_field_names=("section_number",),
            identity_tuple=("0",),
            scope="document",
            document_id=document_uuid,
        )
        return MergedExtraction(
            entities=[
                MergedEntityRecord(
                    identity=sec_identity,
                    properties={"document_id": document_uuid},
                    confidence=1.0,
                    pass_origins={"document_anchors"},
                    display_label="0",
                ),
            ],
            edges=[],
            rejected_edges=[],
            rejections_by_pass={},
            pipeline_run_id=pipeline_run_id,
            document_id=document_uuid,
        )

    monkeypatch.setattr("app.services.docling_anchors.walk", walk_no_doc)
    stubs["graph_store"].upsert_nodes_batch_sync = MagicMock(return_value=["#1:9"])

    result = derive_document_anchors("doc-uuid-1", run_id="run-id-1")

    # Single SECTION upsert.
    stubs["graph_store"].upsert_nodes_batch_sync.assert_called_once()
    # No structural edges to create.
    stubs["graph_store"].create_structural_edge_sync.assert_not_called()
    assert result["section_count"] == 1
    assert result["document_ontology_emitted"] is False
