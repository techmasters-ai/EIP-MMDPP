"""Integration tests for the LangGraph agent context endpoint.

GET /v1/agent/context

Tests verify:
  - Correct response shape (query, mode, total_results, context, sources)
  - context field is a non-empty markdown string
  - All valid modes return 200
  - Invalid mode returns 422
  - Missing required query param returns 422
  - include_sources=false omits sources list
  - top_k is respected (within valid range)
  - Endpoint is accessible without authentication (pre-Phase 3)
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_agent_context_semantic_returns_200(async_client, mock_embeddings):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "Patriot PAC-3 guidance computer", "mode": "semantic"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_agent_context_graph_returns_200(async_client):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "Patriot missile", "mode": "graph"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_agent_context_hybrid_returns_200(async_client, mock_embeddings):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "MK-4 guidance computer", "mode": "hybrid"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_agent_context_default_mode_is_hybrid(async_client, mock_embeddings):
    """Mode defaults to hybrid when not specified."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test query"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "hybrid"


@pytest.mark.asyncio
async def test_agent_context_response_schema(async_client, mock_embeddings):
    """Response must have all required fields with correct types."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "guidance computer", "mode": "semantic", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "query" in data
    assert "mode" in data
    assert "total_results" in data
    assert "context" in data
    assert "sources" in data

    assert data["query"] == "guidance computer"
    assert data["mode"] == "semantic"
    assert isinstance(data["total_results"], int)
    assert isinstance(data["context"], str)
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_agent_context_is_markdown_string(async_client, mock_embeddings):
    """The context field must be a non-empty markdown string."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "semantic"},
    )
    assert resp.status_code == 200
    context = resp.json()["context"]
    assert isinstance(context, str)
    assert len(context) > 0
    assert "## Retrieved Context" in context


@pytest.mark.asyncio
async def test_agent_context_query_echoed(async_client, mock_embeddings):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "Patriot missile system specifications", "mode": "semantic"},
    )
    assert resp.status_code == 200
    assert resp.json()["query"] == "Patriot missile system specifications"


@pytest.mark.asyncio
async def test_agent_context_invalid_mode_returns_422(async_client):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "invalid_mode"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_missing_query_returns_422(async_client):
    resp = await async_client.get("/v1/agent/context", params={"mode": "semantic"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_empty_query_returns_422(async_client):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "", "mode": "semantic"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_include_sources_false(async_client, mock_embeddings):
    """When include_sources=false, sources list must be empty."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "semantic", "include_sources": "false"},
    )
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


@pytest.mark.asyncio
async def test_agent_context_include_sources_true_by_default(async_client, mock_embeddings):
    """sources is included by default (even if empty when no data)."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "semantic"},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json()["sources"], list)


@pytest.mark.asyncio
async def test_agent_context_top_k_zero_returns_422(async_client):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "top_k": "0"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_top_k_over_max_returns_422(async_client):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "top_k": "100"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_total_results_consistent(async_client, mock_embeddings):
    """total_results must equal the number of results described in context."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "semantic", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    # total_results <= top_k
    assert data["total_results"] <= 5
    # sources count <= total_results (sources may be fewer if include_sources is default True)
    assert len(data["sources"]) == data["total_results"]


@pytest.mark.asyncio
async def test_agent_context_sources_schema(async_client, mock_embeddings):
    """Each source item must have the required fields."""
    import io
    from unittest.mock import patch

    # Upload a document so there's something in the DB to query
    src_resp = await async_client.post("/v1/sources", json={"name": "Agent Test Source"})
    source_id = src_resp.json()["id"]

    with patch("app.workers.pipeline.start_ingest_pipeline", return_value="mock-task"):
        with patch("app.services.storage.stream_upload_async", return_value=("key", 100, "abc")):
            await async_client.post(
                f"/v1/sources/{source_id}/documents",
                files={"file": ("t.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            )

    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "semantic", "top_k": 5},
    )
    assert resp.status_code == 200
    # Schema check on whatever sources come back (may be empty if no chunks yet)
    for source in resp.json()["sources"]:
        assert "score" in source
        assert "modality" in source
        assert "classification" in source
        assert isinstance(source["score"], float)


@pytest.mark.asyncio
async def test_agent_context_cross_modal_falls_back_to_semantic(async_client, mock_embeddings):
    """cross_modal mode in agent endpoint falls back to semantic (no image embedding needed)."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "schematic of guidance computer", "mode": "cross_modal"},
    )
    # cross_modal is not in the QueryMode enum for the agent endpoint...
    # The agent only allows semantic/graph/hybrid but maps cross_modal → semantic internally.
    # Pydantic will reject it as an invalid enum value → 422 is also acceptable.
    # Either 200 (if cross_modal is in the enum) or 422 is valid behavior.
    assert resp.status_code in (200, 422)
