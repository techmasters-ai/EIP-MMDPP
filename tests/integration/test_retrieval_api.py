"""Integration tests for the unified retrieval endpoint.

Tests all four query modes: semantic, graph, hybrid, cross_modal.
The tests mock embeddings so no actual ML model loading is required.
Graph queries may return empty results if no data is loaded — that is valid.
"""

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_source(async_client, name: str) -> str:
    resp = await async_client.post("/v1/sources", json={"name": name})
    assert resp.status_code == 201
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Basic endpoint contract tests (no data required)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_query_returns_200(async_client, mock_embeddings):
    """Semantic query must return 200 even when no chunks exist yet."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query": "Patriot PAC-3 guidance computer", "mode": "semantic", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "semantic"
    assert data["query"] == "Patriot PAC-3 guidance computer"
    assert isinstance(data["results"], list)
    assert data["total_results"] == len(data["results"])


@pytest.mark.asyncio
async def test_graph_query_returns_200(async_client):
    """Graph query must return 200 even when AGE graph is empty."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query": "Patriot missile", "mode": "graph", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "graph"
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_hybrid_query_returns_200(async_client, mock_embeddings):
    """Hybrid query must return 200 and combine both retrieval modes."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query": "MK-4 guidance subsystem", "mode": "hybrid", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "hybrid"
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_cross_modal_query_returns_200(async_client):
    """Cross-modal query must return 200 even when no image chunks exist."""
    from unittest.mock import patch

    with patch(
        "app.services.embedding.embed_text_for_image_search",
        return_value=[0.0] * 512,
    ):
        resp = await async_client.post(
            "/v1/retrieval/query",
            json={"query": "schematic of guidance computer", "mode": "cross_modal", "top_k": 5},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "cross_modal"
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_invalid_mode_returns_400(async_client):
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query": "test", "mode": "invalid_mode", "top_k": 5},
    )
    assert resp.status_code == 422  # Pydantic validation rejects unknown enum value


@pytest.mark.asyncio
async def test_missing_query_returns_422(async_client):
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"mode": "semantic", "top_k": 5},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_top_k_default(async_client, mock_embeddings):
    """Request without top_k should use the default."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query": "test query", "mode": "semantic"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_include_context_false_omits_text(async_client, mock_embeddings):
    """When include_context=False, content_text must be null in results."""
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={
            "query": "guidance computer",
            "mode": "semantic",
            "top_k": 5,
            "include_context": False,
        },
    )
    assert resp.status_code == 200
    for item in resp.json()["results"]:
        assert item.get("content_text") is None


# ---------------------------------------------------------------------------
# Result shape validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_result_item_schema(async_client, mock_embeddings):
    """Every result item must have the required fields."""
    import io
    from unittest.mock import patch

    # Upload a document + mock Celery to get some chunks
    src_resp = await async_client.post(
        "/v1/sources", json={"name": "Retrieval Schema Test Source"}
    )
    source_id = src_resp.json()["id"]

    with patch("app.workers.pipeline.start_ingest_pipeline", return_value="mock-task"):
        with patch("app.services.storage.stream_upload_async", return_value=("key", 100, "abc")):
            await async_client.post(
                f"/v1/sources/{source_id}/documents",
                files={"file": ("t.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            )

    # Query — may return 0 results since no actual pipeline ran, but schema must hold
    resp = await async_client.post(
        "/v1/retrieval/query",
        json={"query": "test", "mode": "semantic", "top_k": 5, "include_context": True},
    )
    assert resp.status_code == 200
    for item in resp.json()["results"]:
        assert "score" in item
        assert "modality" in item
        assert "classification" in item
        assert isinstance(item["score"], float)
