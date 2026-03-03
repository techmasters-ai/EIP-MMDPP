"""Integration tests for the sources and document upload API."""

import io
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_create_source(async_client, mock_minio):
    response = await async_client.post(
        "/v1/sources", json={"name": "Test Program", "description": "Integration test source"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Program"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_source_duplicate_name(async_client, mock_minio):
    await async_client.post("/v1/sources", json={"name": "Duplicate Source"})
    response = await async_client.post("/v1/sources", json={"name": "Duplicate Source"})
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_sources_empty(async_client):
    response = await async_client.get("/v1/sources")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_upload_document(async_client, mock_minio, mock_celery):
    # Create a source first
    src_resp = await async_client.post("/v1/sources", json={"name": "Upload Test Source"})
    source_id = src_resp.json()["id"]

    # Upload a fake PDF
    fake_pdf = b"%PDF-1.4 fake pdf content for testing"
    response = await async_client.post(
        f"/v1/sources/{source_id}/documents",
        files={"file": ("test_doc.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test_doc.pdf"
    assert data["pipeline_status"] == "PENDING"
    assert data["source_id"] == source_id


@pytest.mark.asyncio
async def test_upload_to_nonexistent_source(async_client, mock_minio):
    response = await async_client.post(
        "/v1/sources/00000000-0000-0000-0000-000000000099/documents",
        files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_document_status(async_client, mock_minio, mock_celery):
    src_resp = await async_client.post("/v1/sources", json={"name": "Status Test Source"})
    source_id = src_resp.json()["id"]

    doc_resp = await async_client.post(
        f"/v1/sources/{source_id}/documents",
        files={"file": ("status_test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    doc_id = doc_resp.json()["id"]

    status_resp = await async_client.get(f"/v1/documents/{doc_id}/status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["id"] == doc_id
    assert "pipeline_status" in data


@pytest.mark.asyncio
async def test_create_watch_dir(async_client, mock_minio):
    src_resp = await async_client.post("/v1/sources", json={"name": "Watch Dir Source"})
    source_id = src_resp.json()["id"]

    response = await async_client.post(
        "/v1/watch-dirs",
        json={
            "source_id": source_id,
            "path": "/watch/test_dir",
            "poll_interval_seconds": 30,
            "file_patterns": ["*.pdf"],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["path"] == "/watch/test_dir"
    assert data["source_id"] == source_id


@pytest.mark.asyncio
async def test_delete_watch_dir(async_client, mock_minio):
    src_resp = await async_client.post("/v1/sources", json={"name": "Watch Dir Delete Source"})
    source_id = src_resp.json()["id"]

    create_resp = await async_client.post(
        "/v1/watch-dirs",
        json={"source_id": source_id, "path": "/watch/to_delete"},
    )
    watch_dir_id = create_resp.json()["id"]

    delete_resp = await async_client.delete(f"/v1/watch-dirs/{watch_dir_id}")
    assert delete_resp.status_code == 204
