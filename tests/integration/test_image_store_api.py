"""Integration tests for the image vector store endpoints.

POST /v1/images/ingest — embed and store image chunks
POST /v1/images/query — CLIP semantic search on image_chunks
"""

import base64

import pytest

pytestmark = pytest.mark.integration

# Minimal valid 1x1 PNG
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU"
    b"\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
_TINY_PNG_B64 = base64.b64encode(_TINY_PNG).decode()


@pytest.mark.asyncio
async def test_image_ingest_returns_chunk_id(async_client, mock_embeddings):
    """Image ingest embeds with CLIP and returns a chunk ID."""
    resp = await async_client.post(
        "/v1/images/ingest",
        json={
            "image": _TINY_PNG_B64,
            "alt_text": "A schematic diagram",
            "classification": "UNCLASSIFIED",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_id" in data


@pytest.mark.asyncio
async def test_image_query_by_text_returns_200(async_client, mock_embeddings):
    """Text-to-image query returns 200."""
    resp = await async_client.post(
        "/v1/images/query",
        json={"query_text": "schematic diagram of missile", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_image_query_by_image_returns_200(async_client, mock_embeddings):
    """Image-to-image query returns 200."""
    resp = await async_client.post(
        "/v1/images/query",
        json={"query_image": _TINY_PNG_B64, "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_image_query_no_input_returns_empty(async_client):
    """Query without text or image returns empty list."""
    resp = await async_client.post(
        "/v1/images/query",
        json={"top_k": 5},
    )
    assert resp.status_code == 200
    assert resp.json() == []
