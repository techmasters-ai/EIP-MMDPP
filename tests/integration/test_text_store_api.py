"""Integration tests for the text vector store endpoints.

POST /v1/text/ingest — embed and store text chunks
POST /v1/text/query — BGE semantic search on text_chunks
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_text_ingest_returns_created(async_client, mock_embeddings):
    """Text ingest creates chunks and returns IDs."""
    resp = await async_client.post(
        "/v1/text/ingest",
        json={
            "text": "The Patriot PAC-3 guidance computer provides terminal phase guidance.",
            "modality": "text",
            "classification": "UNCLASSIFIED",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_ids" in data
    assert data["chunks_created"] >= 1


@pytest.mark.asyncio
async def test_text_ingest_validates_empty_text(async_client):
    """Empty text should be rejected."""
    resp = await async_client.post(
        "/v1/text/ingest",
        json={"text": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_text_query_returns_200(async_client, mock_embeddings):
    """Text query returns 200 even with no matching chunks."""
    resp = await async_client.post(
        "/v1/text/query",
        json={"query": "guidance computer specifications", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_text_query_validates_empty_query(async_client):
    """Empty query should be rejected."""
    resp = await async_client.post(
        "/v1/text/query",
        json={"query": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_text_query_top_k_bounds(async_client, mock_embeddings):
    """top_k must be between 1 and 100."""
    resp = await async_client.post(
        "/v1/text/query",
        json={"query": "test", "top_k": 0},
    )
    assert resp.status_code == 422

    resp = await async_client.post(
        "/v1/text/query",
        json={"query": "test", "top_k": 101},
    )
    assert resp.status_code == 422
