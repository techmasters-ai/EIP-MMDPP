"""Integration tests for the trusted data endpoints.

POST /v1/trusted-data/ingest — propose trusted data
GET  /v1/trusted-data/proposals — list submissions
POST /v1/trusted-data/proposals/{id}/approve — approve submission
POST /v1/trusted-data/proposals/{id}/reject — reject submission
POST /v1/trusted-data/proposals/{id}/reindex — re-enqueue indexing
POST /v1/trusted-data/query — search trusted data
"""

import uuid
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


async def _create_submission(async_client) -> dict:
    """Helper to create a trusted data submission and return the response data."""
    resp = await async_client.post(
        "/v1/trusted-data/ingest",
        json={
            "content": "The SA-2 Guideline uses Fan Song radar for terminal guidance.",
            "source_context": {"document_id": str(uuid.uuid4())},
            "confidence": 0.85,
        },
    )
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.asyncio
async def test_trusted_data_ingest_creates_submission(async_client):
    """Ingest creates a PROPOSED submission."""
    data = await _create_submission(async_client)
    assert data["status"] == "PROPOSED"
    assert "id" in data
    assert data["content"] == "The SA-2 Guideline uses Fan Song radar for terminal guidance."


@pytest.mark.asyncio
async def test_trusted_data_proposals_list(async_client):
    """List proposals returns results."""
    await _create_submission(async_client)
    resp = await async_client.get("/v1/trusted-data/proposals")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


@pytest.mark.asyncio
async def test_trusted_data_proposals_filter_by_status(async_client):
    """Filter proposals by status."""
    await _create_submission(async_client)
    resp = await async_client.get("/v1/trusted-data/proposals?status=PROPOSED")
    assert resp.status_code == 200
    for item in resp.json():
        assert item["status"] == "PROPOSED"


@pytest.mark.asyncio
async def test_trusted_data_approve_submission(async_client):
    """Approving a submission sets status to APPROVED_PENDING_INDEX."""
    data = await _create_submission(async_client)
    submission_id = data["id"]

    with patch("app.workers.trusted_data_tasks.index_trusted_submission.delay"):
        resp = await async_client.post(
            f"/v1/trusted-data/proposals/{submission_id}/approve",
            json={"notes": "Verified by analyst"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "APPROVED_PENDING_INDEX"


@pytest.mark.asyncio
async def test_trusted_data_reject_submission(async_client):
    """Rejecting a submission sets status to REJECTED."""
    data = await _create_submission(async_client)
    submission_id = data["id"]

    resp = await async_client.post(
        f"/v1/trusted-data/proposals/{submission_id}/reject",
        json={"notes": "Inaccurate information"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_trusted_data_double_approve_returns_409(async_client):
    """Double approve returns 409 Conflict."""
    data = await _create_submission(async_client)
    submission_id = data["id"]

    with patch("app.workers.trusted_data_tasks.index_trusted_submission.delay"):
        resp = await async_client.post(
            f"/v1/trusted-data/proposals/{submission_id}/approve",
            json={"notes": "First approval"},
        )
    assert resp.status_code == 200

    resp = await async_client.post(
        f"/v1/trusted-data/proposals/{submission_id}/approve",
        json={"notes": "Second attempt"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_trusted_data_approve_nonexistent_returns_404(async_client):
    """Approving nonexistent submission returns 404."""
    fake_id = str(uuid.uuid4())
    resp = await async_client.post(
        f"/v1/trusted-data/proposals/{fake_id}/approve",
        json={"notes": "test"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trusted_data_response_has_indexing_fields(async_client):
    """Response includes indexing lifecycle fields."""
    data = await _create_submission(async_client)
    assert "index_status" in data
    assert "index_error" in data
    assert "qdrant_point_id" in data
    assert "embedding_model" in data
    assert "embedded_at" in data
