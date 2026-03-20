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


# ---------------------------------------------------------------------------
# Image descriptions endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_image_descriptions_empty(async_client, mock_minio, mock_celery):
    """Documents with no image elements return an empty list."""
    src_resp = await async_client.post("/v1/sources", json={"name": "ImgDesc Test Source"})
    source_id = src_resp.json()["id"]

    doc_resp = await async_client.post(
        f"/v1/sources/{source_id}/documents",
        files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    doc_id = doc_resp.json()["id"]

    resp = await async_client.get(f"/v1/documents/{doc_id}/image-descriptions")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_image_descriptions_with_image_elements(async_client, async_db_session, mock_minio, mock_celery):
    """Documents with image elements that have content_text return descriptions."""
    import uuid
    from sqlalchemy import text

    src_resp = await async_client.post("/v1/sources", json={"name": "ImgDesc With Images Source"})
    source_id = src_resp.json()["id"]

    doc_resp = await async_client.post(
        f"/v1/sources/{source_id}/documents",
        files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    doc_id = doc_resp.json()["id"]

    # Insert image elements directly into the database
    elem_uid = str(uuid.uuid4())
    await async_db_session.execute(text("""
        INSERT INTO ingest.document_elements (id, document_id, element_uid, element_type, element_order, content_text, page_number)
        VALUES (gen_random_uuid(), cast(:doc_id AS uuid), :elem_uid, 'image', 0, :desc, 1)
    """), {"doc_id": doc_id, "elem_uid": elem_uid, "desc": "A radar installation on a hilltop."})
    await async_db_session.commit()

    resp = await async_client.get(f"/v1/documents/{doc_id}/image-descriptions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["element_uid"] == elem_uid
    assert data[0]["content_text"] == "A radar installation on a hilltop."
    assert data[0]["page_number"] == 1


@pytest.mark.asyncio
async def test_image_descriptions_excludes_null_content(async_client, async_db_session, mock_minio, mock_celery):
    """Image elements without content_text (no description yet) are excluded."""
    import uuid
    from sqlalchemy import text

    src_resp = await async_client.post("/v1/sources", json={"name": "ImgDesc Null Source"})
    source_id = src_resp.json()["id"]

    doc_resp = await async_client.post(
        f"/v1/sources/{source_id}/documents",
        files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    doc_id = doc_resp.json()["id"]

    # Insert image element WITHOUT content_text
    await async_db_session.execute(text("""
        INSERT INTO ingest.document_elements (id, document_id, element_uid, element_type, element_order, content_text, page_number)
        VALUES (gen_random_uuid(), cast(:doc_id AS uuid), :elem_uid, 'image', 0, NULL, 1)
    """), {"doc_id": doc_id, "elem_uid": str(uuid.uuid4())})
    await async_db_session.commit()

    resp = await async_client.get(f"/v1/documents/{doc_id}/image-descriptions")
    assert resp.status_code == 200
    assert resp.json() == []
