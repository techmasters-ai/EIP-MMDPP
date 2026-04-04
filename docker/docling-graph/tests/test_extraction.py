"""Tests for /extract-all endpoint with pipeline-based contract."""
import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline_context():
    graph = nx.DiGraph()
    graph.add_node("RADAR_SYSTEM_Tombstone", type="RADAR_SYSTEM", name="Tombstone",
                   _provenance={"batch_id": 0, "chunk_index": 0, "page_numbers": [14]})
    graph.add_node("FREQUENCY_BAND_S-band", type="FREQUENCY_BAND", name="S-band",
                   _provenance={"batch_id": 0, "chunk_index": 0, "page_numbers": [14]})
    graph.add_edge("RADAR_SYSTEM_Tombstone", "FREQUENCY_BAND_S-band", label="OPERATES_IN_BAND")

    ctx = MagicMock()
    ctx.knowledge_graph = graph
    ctx.graph_metadata = MagicMock(node_count=2, edge_count=1,
                                    node_types={"RADAR_SYSTEM": 1, "FREQUENCY_BAND": 1},
                                    edge_types={"OPERATES_IN_BAND": 1})
    return ctx


@pytest.fixture
def client(mock_pipeline_context):
    from pydantic import BaseModel
    dummy_template = type("DummyEntity", (BaseModel,), {"__annotations__": {"name": str}})

    with patch("app.main.run_extraction_pipeline", return_value=mock_pipeline_context), \
         patch("app.main._templates", {"DummyEntity": dummy_template}):
        from app.main import app
        with TestClient(app) as c:
            # Ensure semaphore and templates are set after lifespan startup
            import app.main as m
            m._templates = {"DummyEntity": dummy_template}
            yield c


class TestExtractAll:
    def test_returns_networkx_graph(self, client):
        resp = client.post("/extract-all", json={
            "document_id": "test-001",
            "docling_document_json": {"schema_name": "DoclingDocument", "version": "1.0"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "graph" in data
        assert "nodes" in data["graph"]
        assert "links" in data["graph"]

    def test_returns_metadata(self, client):
        resp = client.post("/extract-all", json={
            "document_id": "test-001",
            "docling_document_json": {"schema_name": "DoclingDocument"},
        })
        data = resp.json()
        assert data["metadata"]["node_count"] == 2
        assert data["metadata"]["edge_count"] == 1

    def test_rejects_missing_document_id(self, client):
        resp = client.post("/extract-all", json={"docling_document_json": {}})
        assert resp.status_code == 422

    def test_rejects_missing_docling_document(self, client):
        resp = client.post("/extract-all", json={"document_id": "test"})
        assert resp.status_code == 422


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
