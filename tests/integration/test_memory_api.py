"""Integration tests for the Cognee memory layer endpoints.

POST /v1/memory/ingest — propose knowledge
GET  /v1/memory/proposals — list proposals
POST /v1/memory/proposals/{id}/approve — approve proposal
POST /v1/memory/proposals/{id}/reject — reject proposal
POST /v1/memory/query — search Cognee memory
"""

import uuid

import pytest

pytestmark = pytest.mark.integration


async def _create_proposal(async_client) -> dict:
    """Helper to create a memory proposal and return the response data."""
    resp = await async_client.post(
        "/v1/memory/ingest",
        json={
            "content": "The SA-2 Guideline uses Fan Song radar for terminal guidance.",
            "source_context": {"document_id": str(uuid.uuid4())},
            "confidence": 0.85,
        },
    )
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.asyncio
async def test_memory_ingest_creates_proposal(async_client):
    """Memory ingest creates a PROPOSED proposal."""
    data = await _create_proposal(async_client)
    assert data["status"] == "PROPOSED"
    assert "id" in data
    assert data["content"] == "The SA-2 Guideline uses Fan Song radar for terminal guidance."


@pytest.mark.asyncio
async def test_memory_proposals_list(async_client):
    """List proposals returns results."""
    await _create_proposal(async_client)
    resp = await async_client.get("/v1/memory/proposals")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


@pytest.mark.asyncio
async def test_memory_proposals_filter_by_status(async_client):
    """Filter proposals by status."""
    await _create_proposal(async_client)
    resp = await async_client.get("/v1/memory/proposals?status=PROPOSED")
    assert resp.status_code == 200
    for item in resp.json():
        assert item["status"] == "PROPOSED"


@pytest.mark.asyncio
async def test_memory_approve_proposal(async_client, mock_cognee):
    """Approving a proposal sets status to APPROVED."""
    data = await _create_proposal(async_client)
    proposal_id = data["id"]

    resp = await async_client.post(
        f"/v1/memory/proposals/{proposal_id}/approve",
        json={"notes": "Verified by analyst"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "APPROVED"


@pytest.mark.asyncio
async def test_memory_reject_proposal(async_client):
    """Rejecting a proposal sets status to REJECTED."""
    data = await _create_proposal(async_client)
    proposal_id = data["id"]

    resp = await async_client.post(
        f"/v1/memory/proposals/{proposal_id}/reject",
        json={"notes": "Inaccurate information"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_memory_double_approve_returns_409(async_client, mock_cognee):
    """Double approve returns 409 Conflict."""
    data = await _create_proposal(async_client)
    proposal_id = data["id"]

    # First approve succeeds
    resp = await async_client.post(
        f"/v1/memory/proposals/{proposal_id}/approve",
        json={"notes": "First approval"},
    )
    assert resp.status_code == 200

    # Second approve returns 409
    resp = await async_client.post(
        f"/v1/memory/proposals/{proposal_id}/approve",
        json={"notes": "Second attempt"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_memory_approve_nonexistent_returns_404(async_client):
    """Approving nonexistent proposal returns 404."""
    fake_id = str(uuid.uuid4())
    resp = await async_client.post(
        f"/v1/memory/proposals/{fake_id}/approve",
        json={"notes": "test"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_memory_query_returns_200(async_client, mock_cognee):
    """Memory query returns 200."""
    resp = await async_client.post(
        "/v1/memory/query",
        json={"query": "missile guidance", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["results"], list)
