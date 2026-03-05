"""Integration tests for the graph store endpoints.

POST /v1/graph/ingest/entity — create entity in AGE
POST /v1/graph/ingest/relationship — create relationship in AGE
POST /v1/graph/query — Cypher traversal query
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_graph_entity_ingest_returns_200(async_client):
    """Entity ingest creates a node in the graph."""
    resp = await async_client.post(
        "/v1/graph/ingest/entity",
        json={
            "entity_type": "EQUIPMENT_SYSTEM",
            "name": "Test System Alpha",
            "properties": {"designation": "TSA-001"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"


@pytest.mark.asyncio
async def test_graph_entity_ingest_validation(async_client):
    """Entity ingest requires entity_type and name."""
    resp = await async_client.post(
        "/v1/graph/ingest/entity",
        json={"entity_type": "EQUIPMENT_SYSTEM"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_graph_relationship_ingest_returns_200(async_client):
    """Relationship ingest creates an edge between nodes."""
    resp = await async_client.post(
        "/v1/graph/ingest/relationship",
        json={
            "from_entity": "System A",
            "from_type": "EQUIPMENT_SYSTEM",
            "to_entity": "Part B",
            "to_type": "COMPONENT",
            "relationship_type": "CONTAINS",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"


@pytest.mark.asyncio
async def test_graph_query_returns_200(async_client):
    """Graph query returns 200 even with empty graph."""
    resp = await async_client.post(
        "/v1/graph/query",
        json={"query": "missile system", "top_k": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_graph_query_validates_empty_query(async_client):
    """Empty query should be rejected."""
    resp = await async_client.post(
        "/v1/graph/query",
        json={"query": ""},
    )
    assert resp.status_code == 422
