"""Integration tests for the LangGraph agent context endpoint.

GET /v1/agent/context — returns markdown-formatted retrieval context
for LangGraph agents.

Tests verify:
  - Correct response shape (query, mode, total_results, context, sources)
  - context field is a non-empty markdown string
  - All valid modes return 200
  - Invalid mode returns 422
  - Missing required query param returns 422
  - include_sources=false omits sources list
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_agent_context_text_basic_returns_200(async_client, mock_embeddings):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "Patriot PAC-3 guidance computer", "mode": "text_basic"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "text_basic"


@pytest.mark.asyncio
async def test_agent_context_text_only_returns_200(async_client, mock_embeddings):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "Patriot missile", "mode": "text_only"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "text_only"


@pytest.mark.asyncio
async def test_agent_context_images_only_returns_200(async_client, mock_embeddings):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "schematic diagram", "mode": "images_only"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "images_only"


@pytest.mark.asyncio
async def test_agent_context_multi_modal_returns_200(async_client, mock_embeddings):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "guidance computer schematic", "mode": "multi_modal"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "multi_modal"


@pytest.mark.asyncio
async def test_agent_context_memory_returns_200(async_client, mock_cognee):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "approved knowledge", "mode": "memory"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "memory"


@pytest.mark.asyncio
async def test_agent_context_default_mode_is_text_basic(async_client, mock_embeddings):
    """Mode defaults to text_basic when not specified."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test query"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "text_basic"


@pytest.mark.asyncio
async def test_agent_context_response_schema(async_client, mock_embeddings):
    """Response must have all required fields with correct types."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "guidance computer", "mode": "text_basic", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "guidance computer"
    assert data["mode"] == "text_basic"
    assert isinstance(data["total_results"], int)
    assert isinstance(data["context"], str)
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_agent_context_is_markdown_string(async_client, mock_embeddings):
    """The context field must be a non-empty markdown string."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "text_basic"},
    )
    assert resp.status_code == 200
    context = resp.json()["context"]
    assert isinstance(context, str)
    assert len(context) > 0
    assert "## Retrieved Context" in context


@pytest.mark.asyncio
async def test_agent_context_invalid_mode_returns_422(async_client):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "invalid_mode"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_missing_query_returns_422(async_client):
    resp = await async_client.get("/v1/agent/context", params={"mode": "text_basic"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_empty_query_returns_422(async_client):
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "", "mode": "text_basic"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_agent_context_include_sources_false(async_client, mock_embeddings):
    """When include_sources=false, sources list must be empty."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "test", "mode": "text_basic", "include_sources": "false"},
    )
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


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
async def test_agent_context_sources_have_required_fields(async_client, mock_embeddings):
    """Each source must have score, modality, and classification."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "guidance computer", "mode": "text_basic"},
    )
    assert resp.status_code == 200
    for source in resp.json()["sources"]:
        assert "score" in source
        assert "modality" in source
        assert "classification" in source


@pytest.mark.asyncio
async def test_agent_context_query_echoed(async_client, mock_embeddings):
    """Response query field must match the input query."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "radar tracking system", "mode": "text_basic"},
    )
    assert resp.status_code == 200
    assert resp.json()["query"] == "radar tracking system"


@pytest.mark.asyncio
async def test_agent_context_no_results_message(async_client, mock_embeddings):
    """When no data matches, context should contain 'No results found'."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "xyznonexistentquery12345", "mode": "text_basic"},
    )
    assert resp.status_code == 200
    data = resp.json()
    if data["total_results"] == 0:
        assert "No results found" in data["context"]


@pytest.mark.asyncio
async def test_agent_context_total_results_zero_possible(async_client, mock_embeddings):
    """total_results can be 0 when no results are found."""
    resp = await async_client.get(
        "/v1/agent/context",
        params={"query": "zzz_no_match_expected_zzz", "mode": "text_basic"},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json()["total_results"], int)
    assert resp.json()["total_results"] >= 0
