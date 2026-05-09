"""End-to-end integration test: provenance wire shape through /extract-pass.

Drives the FastAPI handler with a mock run_extraction_pass returning a context
populated with all the provenance side-channels added by Tasks 4, 5, 9, 11
(plus the 5 post-review integration fixes). Asserts the response carries
provenance.evidence_ids, evidence_text, and relationship_provenance with
resolved instance_ids — the wire-shape deliverables of Tasks 9 and 11.

Patching strategy:
  - Patch `run_extraction_pass` on dg_app_module directly (not run_pipeline)
    since the handler calls run_extraction_pass via asyncio.to_thread.
  - The mock context pre-sets all side-channels that run_extraction_pass
    would normally set inside itself:
      * _chunk_to_self_refs         (chunk → [self_ref, ...])
      * _chunk_to_evidence_units    (chunk → [evidence_unit_dict, ...])
      * _delta_merged_graph         (relationship provenance source)
      * _delta_trace                (diagnostics — avoid None guard failure)
      * _upstream_preamble_applied  (False)
  - knowledge_graph carries entity nodes with provenance dicts (chunk_indexes,
    evidence_ids, self_refs, page_numbers) so build_provenance_from_context fires.
  - The input docling_document_json has ONE text element so _is_empty() returns
    False and run_extraction_pass is actually called (not short-circuited before
    our patch fires — the patch replaces the whole function so this is moot,
    but the document shape is kept realistic for the request-validation path
    that runs BEFORE the pipeline call).
"""
import json
import sys
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client_e2e(dg_app_module):
    """TestClient sharing the module-level FastAPI app."""
    fastapi_app = dg_app_module.app
    with TestClient(fastapi_app) as c:
        yield c


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_provenance_test_context():
    """Build a mock context that exercises every new provenance code path.

    Covers:
      - Entity provenance: RADAR_SYSTEM node with chunk_indexes + evidence_ids
      - Evidence-text join: last_chunk_metadata carries evidence_units with text
      - Relationship provenance: _delta_merged_graph with ENGAGES relationship
        linking the two nodes by instance_id (Fix D consistency check)
    """
    g = nx.DiGraph()
    g.add_node("n1", **{
        "label": "RADAR_SYSTEM",
        "instance_id": "i_r1",
        "node_type": "RADAR_SYSTEM",
        "element_uid": "#/texts/12",
        "provenance": {
            "chunk_indexes": [0],
            "evidence_ids": ["#/texts/12"],
            "self_refs": ["#/texts/12"],
            "page_numbers": [3],
        },
    })
    g.add_node("n2", **{
        "label": "MISSILE_SYSTEM",
        "instance_id": "i_m1",
        "node_type": "MISSILE_SYSTEM",
        "element_uid": "#/texts/13",
        "provenance": {
            "chunk_indexes": [0],
            "evidence_ids": ["#/texts/13"],
            "self_refs": ["#/texts/13"],
            "page_numbers": [3],
        },
    })
    g.add_edge("n1", "n2", label="ENGAGES")

    ctx = MagicMock()
    ctx.knowledge_graph = g
    ctx.graph_metadata = MagicMock(
        node_count=2,
        edge_count=1,
        node_types={"RADAR_SYSTEM": 1, "MISSILE_SYSTEM": 1},
        edge_types={"ENGAGES": 1},
    )
    # template_instance.model_dump() → handler builds pass_output from this.
    # Return {} so _summarize_pass_output gets an empty dict (no entities in
    # pass_output means no identity-gate filtering of provenance rows).
    ctx.template_instance.model_dump.return_value = {}
    ctx._upstream_preamble_applied = False

    # Task 5 PRIMARY source: chunk metadata with evidence units.
    # The handler reads:
    #   context.extractor.doc_processor.last_chunk_metadata
    # and builds _chunk_to_self_refs / _chunk_to_evidence_units from it
    # inside run_extraction_pass. Since we patch run_extraction_pass we must
    # set these pre-built maps directly on ctx.
    chunk_metadata = [{
        "chunk_id": 0,
        "chunk_kind": "graph_extraction",
        "self_refs": ["#/texts/12", "#/texts/13"],
        "evidence_ids": ["#/texts/12", "#/texts/13"],
        "evidence_units": [
            {
                "evidence_id": "#/texts/12",
                "self_ref": "#/texts/12",
                "page_numbers": [3],
                "text": "AN/MPQ-65 fire-control radar.",
                "label": "text",
            },
            {
                "evidence_id": "#/texts/13",
                "self_ref": "#/texts/13",
                "page_numbers": [3],
                "text": "PAC-3 missile interceptor.",
                "label": "text",
            },
        ],
        "page_numbers": [3],
        "token_count": 50,
    }]

    # Pre-build the maps that run_extraction_pass would normally build from
    # last_chunk_metadata before setting them on context.
    chunk_to_self_refs: dict = {
        0: ["#/texts/12", "#/texts/13"],
    }
    chunk_to_evidence_units: dict = {
        0: chunk_metadata[0]["evidence_units"],
    }

    ctx._chunk_to_self_refs = chunk_to_self_refs
    ctx._chunk_to_evidence_units = chunk_to_evidence_units

    # Task 11 source: delta_merged_graph with relationship provenance.
    # Nodes carry instance_id "i_r1" / "i_m1" — must match the graph nodes
    # above for Fix D (consistent resolver) to succeed.
    ctx._delta_merged_graph = {
        "nodes": [
            {
                "path": "/radars",
                "node_type": "RADAR_SYSTEM",
                "ids": {"id": "R1"},
                "instance_id": "i_r1",
            },
            {
                "path": "/missiles",
                "node_type": "MISSILE_SYSTEM",
                "ids": {"id": "M1"},
                "instance_id": "i_m1",
            },
        ],
        "relationships": [{
            "edge_label": "ENGAGES",
            "source_path": "/radars",
            "source_ids": {"id": "R1"},
            "target_path": "/missiles",
            "target_ids": {"id": "M1"},
            "provenance": {
                "evidence_ids": ["#/texts/12"],
                "self_refs": ["#/texts/12"],
                "page_numbers": [3],
            },
        }],
    }

    # Provide a minimal _delta_trace so the diagnostics path doesn't choke.
    ctx._delta_trace = {
        "empty_source": False,
        "library_log": "",
        "input_sanitize": {"texts_in": 1, "texts_dropped": 0},
    }

    return ctx


# ---------------------------------------------------------------------------
# Shared request body (minimal document with one real text so _is_empty=False
# at the request-validation layer, and so the batch-evidence collector finds
# some text to work with for the identity gate).
# ---------------------------------------------------------------------------

_MINIMAL_DOC = {
    "name": "test_doc",
    "schema_name": "DoclingDocument",
    "version": "1.0.0",
    "body": {
        "self_ref": "#/body",
        "name": "_root_",
        "label": "unspecified",
        "children": [{"$ref": "#/texts/0"}],
    },
    "texts": [
        {
            "self_ref": "#/texts/0",
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": "text",
            "prov": [],
            "orig": "AN/MPQ-65 fire-control radar.",
            "text": "AN/MPQ-65 fire-control radar.",
        },
    ],
    "pictures": [],
    "tables": [],
    "groups": [],
    "pages": {},
    "furniture": {"children": []},
}

_REQUEST_BODY = {
    "bundle_key": "air_defense_v3",
    "pass_name": "radar_identity",
    "docling_document_json": _MINIMAL_DOC,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_pass_e2e_carries_entity_provenance(client_e2e, dg_app_module):
    """Entity provenance rows must carry non-empty evidence_ids, page_numbers,
    and evidence_text joined from the cited evidence units (Task 9)."""
    mock_ctx = _build_provenance_test_context()

    with patch.object(dg_app_module, "run_extraction_pass", return_value=mock_ctx):
        resp = client_e2e.post("/extract-pass", json=_REQUEST_BODY)

    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:600]}"
    payload = resp.json()

    prov = payload.get("provenance") or []
    assert prov, f"provenance[] should be non-empty; got payload keys: {list(payload)}"

    radar_rows = [p for p in prov if p.get("ontology_name") == "RADAR_SYSTEM"]
    assert radar_rows, f"No RADAR_SYSTEM row in provenance; got: {prov}"
    radar = radar_rows[0]

    assert radar.get("evidence_ids") == ["#/texts/12"], \
        f"evidence_ids mismatch: {radar.get('evidence_ids')}"
    assert radar.get("page_numbers") == [3], \
        f"page_numbers mismatch: {radar.get('page_numbers')}"

    ev_text = radar.get("evidence_text")
    assert ev_text is not None and "AN/MPQ-65" in ev_text, \
        f"evidence_text should contain cited unit text; got: {ev_text!r}"


def test_extract_pass_e2e_carries_relationship_provenance(client_e2e, dg_app_module):
    """relationship_provenance[] must carry resolved source/target instance_ids
    and non-empty evidence_ids built from _delta_merged_graph (Task 11)."""
    mock_ctx = _build_provenance_test_context()

    with patch.object(dg_app_module, "run_extraction_pass", return_value=mock_ctx):
        resp = client_e2e.post("/extract-pass", json=_REQUEST_BODY)

    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:600]}"
    payload = resp.json()

    rel_prov = payload.get("relationship_provenance") or []
    assert rel_prov, (
        "relationship_provenance[] should be non-empty; check that "
        "_delta_merged_graph is being read from context and not overwritten."
    )
    rel = rel_prov[0]

    assert rel.get("relationship_type") == "ENGAGES", \
        f"relationship_type mismatch: {rel.get('relationship_type')}"
    assert rel.get("source_instance_id") == "i_r1", \
        f"source_instance_id mismatch: {rel.get('source_instance_id')}"
    assert rel.get("target_instance_id") == "i_m1", \
        f"target_instance_id mismatch: {rel.get('target_instance_id')}"
    assert rel.get("evidence_ids") == ["#/texts/12"], \
        f"evidence_ids mismatch: {rel.get('evidence_ids')}"


def test_extract_pass_e2e_endpoint_id_consistency(client_e2e, dg_app_module):
    """Fix D regression: every relationship endpoint instance_id must appear
    in the entity provenance rows — the two resolvers must use the same
    instance_id for the same node."""
    mock_ctx = _build_provenance_test_context()

    with patch.object(dg_app_module, "run_extraction_pass", return_value=mock_ctx):
        resp = client_e2e.post("/extract-pass", json=_REQUEST_BODY)

    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:600]}"
    payload = resp.json()

    prov_ids = {p.get("instance_id") for p in (payload.get("provenance") or [])}
    prov_ids.discard(None)

    rel_endpoint_ids: set = set()
    for rel in (payload.get("relationship_provenance") or []):
        rel_endpoint_ids.add(rel.get("source_instance_id"))
        rel_endpoint_ids.add(rel.get("target_instance_id"))
    rel_endpoint_ids.discard(None)

    missing = rel_endpoint_ids - prov_ids
    assert not missing, (
        f"Relationship endpoint instance_ids not found in entity provenance "
        f"(Fix D regression): {missing}. "
        f"Entity provenance ids: {prov_ids}, "
        f"Relationship endpoint ids: {rel_endpoint_ids}"
    )
