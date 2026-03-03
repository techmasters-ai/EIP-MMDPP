"""Integration tests for GET /v1/agent/context?mode=cognee_graph.

These tests confirm that the cognee_graph mode integrates correctly with the
agent endpoint.  They run with LLM_PROVIDER=mock (set in .env.test), so no
real Cognee or LLM calls are made.  The endpoint should always return 200
with a valid AgentContextResponse schema, even when Cognee returns no data.
"""

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_context(client: AsyncClient, **params) -> dict:
    """GET /v1/agent/context with given query parameters."""
    default = {"query": "Patriot PAC-3 guidance computer", "mode": "cognee_graph"}
    default.update(params)
    resp = await client.get("/v1/agent/context", params=default)
    return resp


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


# ---------------------------------------------------------------------------
# Successful responses
# ---------------------------------------------------------------------------


class TestCogneeGraphModeSuccess:
    async def test_returns_200(self, async_client: AsyncClient):
        resp = await _get_context(async_client)
        assert resp.status_code == 200

    async def test_mode_field_is_cognee_graph(self, async_client: AsyncClient):
        resp = await _get_context(async_client)
        data = resp.json()
        assert data["mode"] == "cognee_graph"

    async def test_response_schema_valid(self, async_client: AsyncClient):
        resp = await _get_context(async_client)
        _assert_agent_schema(resp.json())

    async def test_context_is_markdown_string(self, async_client: AsyncClient):
        resp = await _get_context(async_client)
        context = resp.json()["context"]
        # With mock provider, Cognee returns [] → "No results found" message
        assert isinstance(context, str)
        assert len(context) > 0

    async def test_no_results_message_with_mock_provider(self, async_client: AsyncClient):
        """LLM_PROVIDER=mock → Cognee returns [] → context says No results found."""
        resp = await _get_context(async_client, query="missile defense system")
        data = resp.json()
        assert data["total_results"] == 0
        assert "No results found" in data["context"]

    async def test_sources_empty_with_no_results(self, async_client: AsyncClient):
        resp = await _get_context(async_client)
        data = resp.json()
        if data["total_results"] == 0:
            assert data["sources"] == []

    async def test_query_field_echoed(self, async_client: AsyncClient):
        q = "radar tracking system specifications"
        resp = await _get_context(async_client, query=q)
        assert resp.json()["query"] == q

    async def test_top_k_respected(self, async_client: AsyncClient):
        resp = await _get_context(async_client, top_k=5)
        data = resp.json()
        assert data["total_results"] <= 5

    async def test_include_sources_false(self, async_client: AsyncClient):
        resp = await _get_context(async_client, include_sources="false")
        data = resp.json()
        assert data["sources"] == []

    async def test_include_sources_true(self, async_client: AsyncClient):
        """include_sources=true (default) — sources field is present (may be [])."""
        resp = await _get_context(async_client, include_sources="true")
        data = resp.json()
        assert isinstance(data["sources"], list)

    async def test_content_type_json(self, async_client: AsyncClient):
        resp = await _get_context(async_client)
        assert "application/json" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestCogneeGraphModeValidation:
    async def test_missing_query_returns_422(self, async_client: AsyncClient):
        resp = await async_client.get(
            "/v1/agent/context", params={"mode": "cognee_graph"}
        )
        assert resp.status_code == 422

    async def test_empty_query_returns_422(self, async_client: AsyncClient):
        resp = await async_client.get(
            "/v1/agent/context", params={"query": "", "mode": "cognee_graph"}
        )
        assert resp.status_code == 422

    async def test_top_k_zero_returns_422(self, async_client: AsyncClient):
        resp = await async_client.get(
            "/v1/agent/context",
            params={"query": "test", "mode": "cognee_graph", "top_k": 0},
        )
        assert resp.status_code == 422

    async def test_top_k_over_limit_returns_422(self, async_client: AsyncClient):
        resp = await async_client.get(
            "/v1/agent/context",
            params={"query": "test", "mode": "cognee_graph", "top_k": 51},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Mode parity — cognee_graph returns same schema as other modes
# ---------------------------------------------------------------------------


class TestCogneeGraphSchemaParity:
    """cognee_graph endpoint returns the same schema as semantic/graph/hybrid."""

    @pytest.mark.parametrize("mode", ["semantic", "graph", "hybrid", "cognee_graph"])
    async def test_all_modes_return_same_schema(
        self, async_client: AsyncClient, mode: str
    ):
        resp = await async_client.get(
            "/v1/agent/context",
            params={"query": "guidance computer", "mode": mode},
        )
        assert resp.status_code == 200
        _assert_agent_schema(resp.json())
        assert resp.json()["mode"] == mode
