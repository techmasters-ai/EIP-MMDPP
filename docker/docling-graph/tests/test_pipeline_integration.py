# tests/test_pipeline_integration.py
"""Integration tests for the complete extraction pipeline flow."""
import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def _make_mock_context():
    """Create a realistic mock PipelineContext with military ontology data."""
    graph = nx.DiGraph()

    graph.add_node("RADAR_SYSTEM_Tombstone", type="RADAR_SYSTEM", name="Tombstone",
                   system_name="Tombstone", nomenclature="64N6", radar_type="SEARCH",
                   _provenance={"batch_id": 0, "chunk_index": 2, "page_numbers": [14, 15]})
    graph.add_node("PLATFORM_SA-20_TEL", type="PLATFORM", name="SA-20 TEL",
                   platform_designation="5P85SE",
                   _provenance={"batch_id": 0, "chunk_index": 1, "page_numbers": [3]})
    graph.add_node("FREQUENCY_BAND_S-band", type="FREQUENCY_BAND", name="S-band",
                   band_name="S-band", designation="S",
                   _provenance={"batch_id": 1, "chunk_index": 0, "page_numbers": [14]})

    graph.add_edge("RADAR_SYSTEM_Tombstone", "PLATFORM_SA-20_TEL",
                   label="INSTALLED_ON", confidence=0.92)
    graph.add_edge("RADAR_SYSTEM_Tombstone", "FREQUENCY_BAND_S-band",
                   label="OPERATES_IN_BAND", confidence=0.88)

    ctx = MagicMock()
    ctx.knowledge_graph = graph
    ctx.graph_metadata = MagicMock(
        node_count=3, edge_count=2,
        node_types={"RADAR_SYSTEM": 1, "PLATFORM": 1, "FREQUENCY_BAND": 1},
        edge_types={"INSTALLED_ON": 1, "OPERATES_IN_BAND": 1})
    return ctx


@pytest.fixture
def integration_client():
    mock_context = _make_mock_context()
    from pydantic import BaseModel
    dummy_template = type("DummyEntity", (BaseModel,), {"__annotations__": {"name": str}})

    with patch("app.main.run_extraction_pipeline", return_value=mock_context):
        from app.main import app
        # Templates are now on app.state (not module-level globals)
        app.state.templates = {"DummyEntity": dummy_template}
        with TestClient(app) as c:
            yield c


class TestFullExtractionFlow:
    def test_full_extraction_returns_valid_graph(self, integration_client):
        resp = integration_client.post("/extract-all", json={
            "document_id": "doc-001",
            "docling_document_json": {"schema_name": "DoclingDocument", "version": "1.0.0",
                                       "body": {"self_ref": "#/body", "children": []}},
        })
        assert resp.status_code == 200
        data = resp.json()
        graph_data = data["graph"]
        assert len(graph_data["nodes"]) == 3
        # main.py calls nx.node_link_data(graph, edges="links") — key is always "links"
        assert len(graph_data["links"]) == 2

    def test_node_types_correct(self, integration_client):
        resp = integration_client.post("/extract-all", json={
            "document_id": "doc-001",
            "docling_document_json": {"schema_name": "DoclingDocument"},
        })
        data = resp.json()
        node_types = {n["type"] for n in data["graph"]["nodes"]}
        assert "RADAR_SYSTEM" in node_types
        assert "PLATFORM" in node_types
        assert "FREQUENCY_BAND" in node_types

    def test_provenance_preserved_on_nodes(self, integration_client):
        resp = integration_client.post("/extract-all", json={
            "document_id": "doc-001",
            "docling_document_json": {"schema_name": "DoclingDocument"},
        })
        data = resp.json()
        nodes = data["graph"]["nodes"]
        radar_node = next(n for n in nodes if n.get("type") == "RADAR_SYSTEM")
        assert "_provenance" in radar_node
        assert "page_numbers" in radar_node["_provenance"]
        assert 14 in radar_node["_provenance"]["page_numbers"]

    def test_extraction_metadata_complete(self, integration_client):
        resp = integration_client.post("/extract-all", json={
            "document_id": "doc-001",
            "docling_document_json": {"schema_name": "DoclingDocument"},
        })
        data = resp.json()
        meta = data["metadata"]
        assert meta["node_count"] == 3
        assert meta["edge_count"] == 2
        assert "RADAR_SYSTEM" in meta["node_types"]

    def test_model_and_provider_in_response(self, integration_client):
        resp = integration_client.post("/extract-all", json={
            "document_id": "doc-001",
            "docling_document_json": {"schema_name": "DoclingDocument"},
        })
        data = resp.json()
        assert "model" in data
        assert "provider" in data
