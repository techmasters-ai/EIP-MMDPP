"""Integration tests for the unified retrieval endpoint.

Tests all five query modes: text_semantic, image_semantic, graph, cross_modal, memory.
The tests mock embeddings so no actual ML model loading is required.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_text_semantic_query_returns_200(async_client, mock_embeddings):
    """Text semantic query returns 200 with sectioned response."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "Patriot PAC-3 guidance computer",
            "modes": ["text_semantic"],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "sections" in data
    assert "text_semantic" in data["sections"]
    assert isinstance(data["sections"]["text_semantic"]["results"], list)
    assert data["sections"]["text_semantic"]["total"] >= 0


@pytest.mark.asyncio
async def test_image_semantic_query_returns_200(async_client, mock_embeddings):
    """Image semantic query via text returns 200."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "schematic of guidance computer",
            "modes": ["image_semantic"],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "image_semantic" in data["sections"]


@pytest.mark.asyncio
async def test_graph_query_returns_200(async_client):
    """Graph query returns 200 even with empty graph."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "Patriot missile",
            "modes": ["graph"],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "graph" in data["sections"]


@pytest.mark.asyncio
async def test_cross_modal_query_returns_200(async_client, mock_embeddings):
    """Cross-modal query returns 200."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "missile guidance system",
            "modes": ["cross_modal"],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "cross_modal" in data["sections"]


@pytest.mark.asyncio
async def test_memory_query_returns_200(async_client, mock_cognee):
    """Memory query returns 200."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "approved knowledge about systems",
            "modes": ["memory"],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "memory" in data["sections"]


@pytest.mark.asyncio
async def test_multi_mode_query(async_client, mock_embeddings, mock_cognee):
    """Multi-mode query returns sections for each requested mode."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "Patriot system specifications",
            "modes": ["text_semantic", "graph", "memory"],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text_semantic" in data["sections"]
    assert "graph" in data["sections"]
    assert "memory" in data["sections"]
    assert len(data["modes"]) == 3


@pytest.mark.asyncio
async def test_invalid_mode_returns_422(async_client):
    """Invalid mode value rejected by Pydantic validation."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "test",
            "modes": ["nonexistent_mode"],
            "top_k": 5,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_query_returns_422(async_client):
    """Request without query_text or query_image returns 422."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"modes": ["text_semantic"], "top_k": 5},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_response_echoes_query(async_client, mock_embeddings):
    """Response includes the original query text."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "test echo query",
            "modes": ["text_semantic"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_text"] == "test echo query"


@pytest.mark.asyncio
async def test_default_mode_is_text_semantic(async_client, mock_embeddings):
    """Request without explicit modes defaults to text_semantic."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query_text": "test defaults"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "text_semantic" in data["sections"]
