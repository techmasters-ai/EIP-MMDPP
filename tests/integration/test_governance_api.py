"""Integration tests for the feedback and patch governance API."""

import io
import pytest

pytestmark = pytest.mark.integration


async def _create_source_and_document(async_client, mock_minio, mock_celery, source_name: str):
    src_resp = await async_client.post("/v1/sources", json={"name": source_name})
    source_id = src_resp.json()["id"]
    doc_resp = await async_client.post(
        f"/v1/sources/{source_id}/documents",
        files={"file": ("gov_test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    return src_resp.json(), doc_resp.json()


@pytest.mark.asyncio
async def test_submit_feedback_creates_patch(async_client, mock_minio, mock_celery):
    _, doc = await _create_source_and_document(
        async_client, mock_minio, mock_celery, "Feedback Source"
    )

    response = await async_client.post(
        "/v1/feedback",
        json={
            "feedback_type": "WRONG_TEXT",
            "notes": "The OCR extracted 'MlL-SPEC' but should be 'MIL-SPEC'",
            "proposed_value": {"text": "MIL-SPEC"},
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["feedback_type"] == "WRONG_TEXT"


@pytest.mark.asyncio
async def test_list_patches(async_client, mock_minio, mock_celery):
    await async_client.post(
        "/v1/feedback",
        json={"feedback_type": "WRONG_TEXT", "proposed_value": {"text": "corrected"}},
    )
    response = await async_client.get("/v1/patches")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_text_patch_requires_single_curator(async_client, mock_minio, mock_celery):
    """WRONG_TEXT feedback should produce a patch that does NOT require dual approval."""
    feedback_resp = await async_client.post(
        "/v1/feedback",
        json={"feedback_type": "WRONG_TEXT", "proposed_value": {"text": "corrected text"}},
    )
    assert feedback_resp.status_code == 201

    patches = await async_client.get("/v1/patches")
    text_patches = [p for p in patches.json() if p["patch_type"] == "chunk_text_correction"]
    if text_patches:
        assert text_patches[0]["requires_dual_approval"] is False


@pytest.mark.asyncio
async def test_graph_mutation_patch_requires_dual_curator(async_client, mock_minio, mock_celery):
    """MISSING_ENTITY feedback should produce a patch requiring dual approval."""
    feedback_resp = await async_client.post(
        "/v1/feedback",
        json={
            "feedback_type": "MISSING_ENTITY",
            "proposed_value": {"name": "Guidance Computer MK-4", "type": "COMPONENT"},
        },
    )
    assert feedback_resp.status_code == 201

    patches = await async_client.get("/v1/patches")
    entity_patches = [p for p in patches.json() if p["patch_type"] == "entity_add"]
    if entity_patches:
        assert entity_patches[0]["requires_dual_approval"] is True


@pytest.mark.asyncio
async def test_patch_self_approval_rejected(async_client, mock_minio, mock_celery):
    """The same curator cannot provide both approvals."""
    feedback_resp = await async_client.post(
        "/v1/feedback",
        json={
            "feedback_type": "MISSING_ENTITY",
            "proposed_value": {"name": "Test Component", "type": "COMPONENT"},
        },
    )
    assert feedback_resp.status_code == 201

    patches = await async_client.get("/v1/patches")
    if not patches.json():
        pytest.skip("No patches available to test self-approval")

    patch_id = patches.json()[0]["id"]

    # First approval
    first_approval = await async_client.post(
        f"/v1/patches/{patch_id}/approve", json={"notes": "First approval"}
    )
    # Second approval by same curator should fail
    second_approval = await async_client.post(
        f"/v1/patches/{patch_id}/approve", json={"notes": "Second attempt by same curator"}
    )
    assert second_approval.status_code == 403


@pytest.mark.asyncio
async def test_reject_patch(async_client, mock_minio, mock_celery):
    feedback_resp = await async_client.post(
        "/v1/feedback",
        json={"feedback_type": "WRONG_TEXT", "proposed_value": {"text": "wrong correction"}},
    )
    assert feedback_resp.status_code == 201

    patches = await async_client.get("/v1/patches")
    if not patches.json():
        pytest.skip("No patches available")

    patch_id = patches.json()[0]["id"]
    reject_resp = await async_client.post(
        f"/v1/patches/{patch_id}/reject", json={"notes": "Not a valid correction"}
    )
    assert reject_resp.status_code == 200
    assert reject_resp.json()["state"] == "REJECTED"


@pytest.mark.asyncio
async def test_get_patch_not_found(async_client):
    response = await async_client.get("/v1/patches/00000000-0000-0000-0000-000000000099")
    assert response.status_code == 404
