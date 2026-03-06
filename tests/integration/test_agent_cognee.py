"""Integration tests for GET /v1/agent/context?mode=memory.

These tests confirm that the memory mode (Cognee) integrates correctly with the
agent endpoint. They run with LLM_PROVIDER=mock, so no real Cognee or LLM
calls are made. The endpoint should return 200 with a valid AgentContextResponse
schema, even when Cognee returns no data.
"""

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _get_context(client: AsyncClient, **params) -> dict:
    """GET /v1/agent/context with given query parameters."""
    default = {"query": "Patriot PAC-3 guidance computer", "mode": "memory"}
    default.update(params)
    return await client.get("/v1/agent/context", params=default)


def _assert_agent_schema(data: dict) -> None:
    """Verify the AgentContextResponse schema is present and well-formed."""
    assert "query" in data
    assert "mode" in data
    assert "total_results" in data
    assert "context" in data
    assert "sources" in data
    assert isinstance(data["total_results"], int)
    assert isinstance(data["context"], str)
    assert isinstance(data["sources"], list)


class TestMemoryModeSuccess:
    async def test_returns_200(self, async_client: AsyncClient, mock_cognee):
        resp = await _get_context(async_client)
        assert resp.status_code == 200

    async def test_mode_field_is_memory(self, async_client: AsyncClient, mock_cognee):
        resp = await _get_context(async_client)
        assert resp.json()["mode"] == "memory"

    async def test_response_schema_valid(self, async_client: AsyncClient, mock_cognee):
        resp = await _get_context(async_client)
        _assert_agent_schema(resp.json())

    async def test_context_is_markdown_string(self, async_client: AsyncClient, mock_cognee):
        resp = await _get_context(async_client)
        context = resp.json()["context"]
        assert isinstance(context, str)
        assert len(context) > 0

    async def test_no_results_message_with_mock_provider(self, async_client: AsyncClient, mock_cognee):
        """LLM_PROVIDER=mock → Cognee returns [] → context says No results found."""
        resp = await _get_context(async_client, query="missile defense system")
        data = resp.json()
        assert data["total_results"] == 0
        assert "No results found" in data["context"]

    async def test_query_field_echoed(self, async_client: AsyncClient, mock_cognee):
        q = "radar tracking system specifications"
        resp = await _get_context(async_client, query=q)
        assert resp.json()["query"] == q

    async def test_include_sources_false(self, async_client: AsyncClient, mock_cognee):
        resp = await _get_context(async_client, include_sources="false")
        assert resp.json()["sources"] == []


class TestMemoryModeValidation:
    async def test_missing_query_returns_422(self, async_client: AsyncClient):
        resp = await async_client.get(
            "/v1/agent/context", params={"mode": "memory"}
        )
        assert resp.status_code == 422

    async def test_empty_query_returns_422(self, async_client: AsyncClient):
        resp = await async_client.get(
            "/v1/agent/context", params={"query": "", "mode": "memory"}
        )
        assert resp.status_code == 422


class TestModeSchemaParity:
    """All modes return the same AgentContextResponse schema."""

    @pytest.mark.parametrize("mode", ["text_basic", "text_only", "images_only", "multi_modal", "memory"])
    async def test_all_modes_return_same_schema(
        self, async_client: AsyncClient, mock_embeddings, mock_cognee, mode: str
    ):
        resp = await async_client.get(
            "/v1/agent/context",
            params={"query": "guidance computer", "mode": mode},
        )
        assert resp.status_code == 200
        _assert_agent_schema(resp.json())
        assert resp.json()["mode"] == mode
