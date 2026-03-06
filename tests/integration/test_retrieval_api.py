"""Integration tests for the unified retrieval endpoint (legacy compatibility).

These tests are superseded by test_unified_query_api.py which covers the new
query modes. This file is kept for basic schema validation of the endpoint.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_retrieval_query_returns_200(async_client, mock_embeddings):
    """Basic unified query returns 200 with flat results."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query_text": "Patriot PAC-3 guidance computer",
            "mode": "text_basic",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["mode"] == "text_basic"


@pytest.mark.asyncio
async def test_invalid_mode_returns_422(async_client):
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query_text": "test", "mode": "invalid_mode", "top_k": 5},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_query_returns_422(async_client):
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"mode": "text_basic", "top_k": 5},
    )
    assert resp.status_code == 422
