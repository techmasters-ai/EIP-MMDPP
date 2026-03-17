"""Sources, document upload, artifacts, and watch directory endpoints."""

import hashlib
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.models.ingest import Artifact, Document, DocumentElement, Source, WatchDir
from app.schemas.common import CursorPage
from app.schemas.sources import (
    ArtifactResponse,
    BatchStatusRequest,
    DocumentResponse,
    DocumentStatusResponse,
    SourceCreate,
    SourceResponse,
    WatchDirCreate,
    WatchDirResponse,
)
from app.services.storage import stream_upload_async, delete_object_async
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(tags=["sources"])

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@router.post("/sources", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def create_source(
    body: SourceCreate,
    db: AsyncSession = Depends(get_async_session),
) -> SourceResponse:
    """Create a named collection (source) for documents."""
    # TODO(Phase 3): replace with actual authenticated user
    system_user = uuid.UUID("00000000-0000-0000-0000-000000000001")

    existing = await db.execute(select(Source).where(Source.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Source with name '{body.name}' already exists.",
        )

    source = Source(
        name=body.name,
        description=body.description,
        created_by=system_user,
    )
    db.add(source)
    await db.flush()
    await db.refresh(source)
    return SourceResponse.model_validate(source)


@router.get("/sources", response_model=list[SourceResponse])
async def list_sources(
    db: AsyncSession = Depends(get_async_session),
) -> list[SourceResponse]:
    """List all sources."""
    result = await db.execute(select(Source).order_by(Source.created_at.desc()))
    sources = result.scalars().all()
    return [SourceResponse.model_validate(s) for s in sources]


# ---------------------------------------------------------------------------
# Documents — upload
# ---------------------------------------------------------------------------


@router.post(
    "/sources/{source_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    source_id: uuid.UUID,
    file: UploadFile,
    db: AsyncSession = Depends(get_async_session),
) -> DocumentResponse:
    """Upload a single document to a source. Streams directly to MinIO."""
    source = await db.get(Source, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")

    # TODO(Phase 3): replace with actual authenticated user
    system_user = uuid.UUID("00000000-0000-0000-0000-000000000001")

    doc_id = uuid.uuid4()
    object_key = f"sources/{source_id}/{doc_id}/{file.filename}"

    # Stream the upload to MinIO (no in-memory buffering for large files)
    key, total_bytes, file_hash = await stream_upload_async(
        file,
        bucket=settings.minio_bucket_raw,
        key=object_key,
        content_type=file.content_type or "application/octet-stream",
    )

    # Check for duplicate within the same source
    existing_result = await db.execute(
        select(Document).where(
            Document.source_id == source_id,
            Document.file_hash == file_hash,
        )
    )
    existing_doc = existing_result.scalar_one_or_none()
    if existing_doc:
        if existing_doc.pipeline_status not in ("FAILED", "ERROR"):
            await delete_object_async(settings.minio_bucket_raw, key)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Duplicate Document Upload",
            )
        # Previous upload failed — remove old record so re-upload can proceed
        await db.delete(existing_doc)
        await db.flush()

    document = Document(
        id=doc_id,
        source_id=source_id,
        filename=file.filename or "unknown",
        mime_type=file.content_type,
        file_size_bytes=total_bytes,
        file_hash=file_hash,
        storage_bucket=settings.minio_bucket_raw,
        storage_key=object_key,
        pipeline_status="PENDING",
        uploaded_by=system_user,
    )
    db.add(document)
    await db.flush()
    await db.refresh(document)

    # Commit the document BEFORE dispatching the Celery task
    # so the worker's sync session can see the row.
    await db.commit()

    # Dispatch ingest pipeline (document is now visible to workers)
    from app.workers.pipeline import start_ingest_pipeline
    task_id = start_ingest_pipeline(str(document.id))

    # Update celery_task_id via ORM (not raw UPDATE) to avoid
    # expiring updated_at and triggering MissingGreenlet.
    document.celery_task_id = task_id

    return DocumentResponse.model_validate(document)


@router.get("/sources/{source_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> list[DocumentResponse]:
    """List all documents in a source."""
    source = await db.get(Source, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")

    result = await db.execute(
        select(Document)
        .where(Document.source_id == source_id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> DocumentStatusResponse:
    """Get the pipeline processing status of a document."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    resp = DocumentStatusResponse.model_validate(doc)

    # Populate stage details from PipelineRun
    from app.models.ingest import PipelineRun, StageRun

    run_result = await db.execute(
        select(PipelineRun)
        .where(PipelineRun.document_id == document_id)
        .order_by(PipelineRun.started_at.desc())
        .limit(1)
    )
    run = run_result.scalar_one_or_none()
    if run:
        resp.pipeline_version = run.pipeline_version
        resp.current_run_id = run.id
        stage_result = await db.execute(
            select(StageRun)
            .where(StageRun.pipeline_run_id == run.id)
            .order_by(StageRun.started_at.nullslast())
        )
        stages = stage_result.scalars().all()
        resp.stage_summary = [
            {
                "stage": s.stage_name,
                "status": s.status,
                "attempt": s.attempt,
                "metrics": s.metrics or {},
            }
            for s in stages
        ]

    return resp


@router.post("/documents/batch-status", response_model=list[DocumentStatusResponse])
async def batch_document_status(
    body: BatchStatusRequest,
    db: AsyncSession = Depends(get_async_session),
) -> list[DocumentStatusResponse]:
    """Get pipeline status for multiple documents in a single query."""
    if not body.document_ids:
        return []

    result = await db.execute(
        select(Document).where(Document.id.in_(body.document_ids))
    )
    docs = result.scalars().all()
    return [DocumentStatusResponse.model_validate(d) for d in docs]


@router.get("/documents/{document_id}/stages")
async def get_document_stages(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Get detailed stage diagnostics for a pipeline run."""
    from app.models.ingest import PipelineRun, StageRun

    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    run_result = await db.execute(
        select(PipelineRun)
        .where(PipelineRun.document_id == document_id)
        .order_by(PipelineRun.started_at.desc())
        .limit(1)
    )
    run = run_result.scalar_one_or_none()
    if not run:
        return {"document_id": str(document_id), "pipeline_version": None, "stages": []}

    stage_result = await db.execute(
        select(StageRun)
        .where(StageRun.pipeline_run_id == run.id)
        .order_by(StageRun.started_at.nullslast())
    )
    stages = stage_result.scalars().all()

    return {
        "document_id": str(document_id),
        "pipeline_version": run.pipeline_version,
        "run_id": str(run.id),
        "run_status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "stages": [
            {
                "stage": s.stage_name,
                "status": s.status,
                "attempt": s.attempt,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "finished_at": s.finished_at.isoformat() if s.finished_at else None,
                "metrics": s.metrics or {},
                "error": s.error_message,
            }
            for s in stages
        ],
    }


@router.post("/documents/{document_id}/reingest")
async def reingest_document(
    document_id: uuid.UUID,
    body: dict = None,
    db: AsyncSession = Depends(get_async_session),
):
    """Re-run the ingest pipeline for an existing document.

    Body (optional):
        {"mode": "full" | "embeddings_only" | "graph_only"}
    """
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if doc.pipeline_status == "PROCESSING":
        raise HTTPException(status_code=409, detail="Pipeline already running for this document.")

    # Reset status so the pipeline picks it up cleanly
    doc.pipeline_status = "PENDING"
    await db.commit()

    mode = (body or {}).get("mode", "full")

    if mode == "full":
        from app.workers.pipeline import start_ingest_pipeline
        task_id = start_ingest_pipeline(str(document_id))
    elif mode == "embeddings_only":
        from app.workers.pipeline import (
            derive_text_chunks_and_embeddings, derive_image_embeddings, finalize_document,
        )
        from celery import chain as celery_chain, chord, group
        result = celery_chain(
            chord(
                group(
                    derive_text_chunks_and_embeddings.si(str(document_id)),
                    derive_image_embeddings.si(str(document_id)),
                ),
                finalize_document.si(str(document_id)),
            ),
        ).apply_async()
        task_id = result.id
    elif mode == "graph_only":
        from app.workers.pipeline import (
            derive_ontology_graph, derive_structure_links, finalize_document,
        )
        from celery import chain as celery_chain
        result = celery_chain(
            derive_ontology_graph.si(str(document_id)),
            derive_structure_links.si(str(document_id)),
            finalize_document.si(str(document_id)),
        ).apply_async()
        task_id = result.id
    else:
        raise HTTPException(status_code=400, detail=f"Unknown reingest mode: {mode}")

    return {"document_id": str(document_id), "mode": mode, "task_id": task_id}


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Hard-delete a document and all its derived data.

    Removes: DB records (document, artifacts, document_elements, text_chunks,
    image_chunks, chunk_links, graph_extractions), Qdrant vectors, Neo4j
    graph nodes/edges, and MinIO objects (raw file + derived artifacts).
    """
    from sqlalchemy import delete as sql_delete
    from app.models.retrieval import TextChunk, ImageChunk, ChunkLink
    from app.models.ingest import DocumentGraphExtraction

    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if doc.pipeline_status == "PROCESSING":
        raise HTTPException(status_code=409, detail="Cannot delete while pipeline is running.")

    doc_id_str = str(document_id)

    # 1. Delete Qdrant vectors
    try:
        from app.services.qdrant_store import delete_by_document_id
        from app.db.session import get_qdrant_client
        client = get_qdrant_client()
        delete_by_document_id(client, doc_id_str)
    except Exception as exc:
        logger.warning("delete_document: Qdrant cleanup failed for %s: %s", doc_id_str, exc)

    # 2. Delete Neo4j graph data
    try:
        from app.db.session import get_neo4j_driver
        driver = get_neo4j_driver()
        with driver.session() as neo_session:
            # Delete ChunkRef nodes and edges linked to this document
            neo_session.run(
                "MATCH (d:Document {document_id: $doc_id})-[r]-() DELETE r, d",
                doc_id=doc_id_str,
            )
            # Delete entity edges tagged with this document's artifact_ids
            artifact_result = await db.execute(
                select(Artifact.id).where(Artifact.document_id == document_id)
            )
            artifact_ids = [str(a) for a in artifact_result.scalars().all()]
            if artifact_ids:
                neo_session.run(
                    "MATCH ()-[r]->() WHERE r.artifact_id IN $aids DELETE r",
                    aids=artifact_ids,
                )
    except Exception as exc:
        logger.warning("delete_document: Neo4j cleanup failed for %s: %s", doc_id_str, exc)

    # 3. Delete MinIO objects (raw file + derived artifacts)
    try:
        # Raw file
        if doc.storage_bucket and doc.storage_key:
            await delete_object_async(doc.storage_bucket, doc.storage_key)
        # Derived artifacts (images, docling docs)
        artifacts = (await db.execute(
            select(Artifact).where(Artifact.document_id == document_id)
        )).scalars().all()
        for art in artifacts:
            if art.storage_bucket and art.storage_key:
                try:
                    await delete_object_async(art.storage_bucket, art.storage_key)
                except Exception:
                    pass
        # DoclingDocument files
        base_key = f"artifacts/{doc_id_str}"
        for suffix in ("docling_document.md", "docling_document.json"):
            try:
                await delete_object_async(settings.minio_bucket_derived, f"{base_key}/{suffix}")
            except Exception:
                pass
    except Exception as exc:
        logger.warning("delete_document: MinIO cleanup failed for %s: %s", doc_id_str, exc)

    # 4. Delete DB records (order matters for FK constraints)
    await db.execute(sql_delete(ChunkLink).where(
        ChunkLink.source_chunk_id.in_(
            select(TextChunk.id).where(TextChunk.document_id == document_id)
        ) | ChunkLink.target_chunk_id.in_(
            select(TextChunk.id).where(TextChunk.document_id == document_id)
        )
    ))
    await db.execute(sql_delete(TextChunk).where(TextChunk.document_id == document_id))
    await db.execute(sql_delete(ImageChunk).where(ImageChunk.document_id == document_id))
    await db.execute(sql_delete(DocumentGraphExtraction).where(DocumentGraphExtraction.document_id == document_id))
    await db.execute(sql_delete(DocumentElement).where(DocumentElement.document_id == document_id))
    await db.execute(sql_delete(Artifact).where(Artifact.document_id == document_id))
    await db.delete(doc)
    await db.commit()

    logger.info("delete_document: deleted document %s and all derived data", doc_id_str)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}/artifacts", response_model=list[ArtifactResponse])
async def list_artifacts(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> list[ArtifactResponse]:
    """List all extracted artifacts for a document."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    result = await db.execute(
        select(Artifact)
        .where(Artifact.document_id == document_id)
        .order_by(Artifact.page_number.asc().nulls_last(), Artifact.created_at.asc())
    )
    artifacts = result.scalars().all()
    return [ArtifactResponse.model_validate(a) for a in artifacts]


@router.get("/artifacts/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> ArtifactResponse:
    """Get a single artifact by ID."""
    artifact = await db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return ArtifactResponse.model_validate(artifact)


# ---------------------------------------------------------------------------
# Watch Directories
# ---------------------------------------------------------------------------


@router.post(
    "/watch-dirs",
    response_model=WatchDirResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_watch_dir(
    body: WatchDirCreate,
    db: AsyncSession = Depends(get_async_session),
) -> WatchDirResponse:
    """Register a directory for automatic document ingestion."""
    system_user = uuid.UUID("00000000-0000-0000-0000-000000000001")

    source = await db.get(Source, body.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")

    existing = await db.execute(select(WatchDir).where(WatchDir.path == body.path))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Watch directory '{body.path}' already registered.",
        )

    watch_dir = WatchDir(
        source_id=body.source_id,
        path=body.path,
        poll_interval_seconds=body.poll_interval_seconds,
        file_patterns=body.file_patterns,
        created_by=system_user,
    )
    db.add(watch_dir)
    await db.flush()
    await db.refresh(watch_dir)
    return WatchDirResponse.model_validate(watch_dir)


@router.get("/watch-dirs", response_model=list[WatchDirResponse])
async def list_watch_dirs(
    db: AsyncSession = Depends(get_async_session),
) -> list[WatchDirResponse]:
    """List all registered watch directories."""
    result = await db.execute(select(WatchDir).order_by(WatchDir.created_at.desc()))
    dirs = result.scalars().all()
    return [WatchDirResponse.model_validate(d) for d in dirs]


@router.delete("/watch-dirs/{watch_dir_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_watch_dir(
    watch_dir_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> None:
    """Remove a watch directory registration."""
    watch_dir = await db.get(WatchDir, watch_dir_id)
    if not watch_dir:
        raise HTTPException(status_code=404, detail="Watch directory not found.")
    await db.delete(watch_dir)


# ---------------------------------------------------------------------------
# DoclingDocument viewer
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}/docling")
async def get_docling_document(
    document_id: uuid.UUID,
    include_json: bool = True,
    db: AsyncSession = Depends(get_async_session),
):
    """Retrieve the persisted DoclingDocument (markdown + JSON) for a processed document."""
    import json as _json

    from app.services.storage import download_bytes_async
    from app.schemas.retrieval import DoclingDocumentResponse, DoclingImageRef

    # Verify document exists
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    base_key = f"artifacts/{str(document_id)}"
    bucket = settings.minio_bucket_derived

    # Fetch markdown (and optionally JSON) from MinIO
    try:
        md_bytes = await download_bytes_async(bucket, f"{base_key}/docling_document.md")
        if include_json:
            json_bytes = await download_bytes_async(bucket, f"{base_key}/docling_document.json")
    except Exception as exc:
        # Fall back to raw file content for text files
        if doc.mime_type and doc.mime_type.startswith("text/"):
            try:
                raw_bytes = await download_bytes_async(
                    doc.storage_bucket, doc.storage_key
                )
                return DoclingDocumentResponse(
                    document_id=str(document_id),
                    filename=doc.filename or "",
                    markdown=raw_bytes.decode("utf-8", errors="replace"),
                    document_json={},
                    images=[],
                )
            except Exception:
                pass  # fall through to 404

        logger.info("get_docling_document: DoclingDocument not found for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=404,
            detail="DoclingDocument not available for this document. Re-ingest to generate.",
        )

    markdown_text = md_bytes.decode("utf-8")
    document_json = {}
    if include_json:
        try:
            document_json = _json.loads(json_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            logger.error("get_docling_document: corrupt JSON for %s: %s", document_id, exc)
            raise HTTPException(status_code=500, detail="DoclingDocument JSON is corrupted. Re-ingest to regenerate.")

    # Build image URL list from artifacts
    from sqlalchemy import select as sa_select

    stmt = (
        sa_select(Artifact)
        .where(
            Artifact.document_id == document_id,
            Artifact.artifact_type.in_(["image", "schematic"]),
            Artifact.storage_key.isnot(None),
        )
        .order_by(Artifact.id)
    )
    result = await db.execute(stmt)
    artifacts = result.scalars().all()

    images: list[DoclingImageRef] = []
    for art in artifacts:
        url = f"/v1/documents/{str(document_id)}/artifacts/{art.id}/image"
        elem_stmt = sa_select(DocumentElement.element_uid).where(
            DocumentElement.artifact_id == art.id
        )
        elem_result = await db.execute(elem_stmt)
        elem_uid = elem_result.scalar_one_or_none()
        if elem_uid:
            images.append(DoclingImageRef(element_uid=elem_uid, url=url))

    # Replace <!-- image --> placeholders with actual image markdown tags.
    # Map Nth placeholder to Nth image artifact (ordered by element_order).
    if images:
        sorted_images = sorted(images, key=lambda img: img.element_uid)
        placeholder = "<!-- image -->"
        for img_ref in sorted_images:
            markdown_text = markdown_text.replace(
                placeholder,
                f"![image]({img_ref.url})",
                1,
            )

    return DoclingDocumentResponse(
        document_id=str(document_id),
        filename=doc.filename or "",
        markdown=markdown_text,
        document_json=document_json,
        images=images,
    )


@router.get("/documents/{document_id}/docling-raw")
async def get_docling_raw_json(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Stream the raw DoclingDocument JSON from MinIO.

    Returns the full DoclingDocument including base64 page images,
    intended for the <docling-img> web component viewer.
    """
    from fastapi.responses import Response
    from app.services.storage import download_bytes_async

    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    base_key = f"artifacts/{str(document_id)}"
    bucket = settings.minio_bucket_derived

    try:
        json_bytes = await download_bytes_async(bucket, f"{base_key}/docling_document.json")
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="DoclingDocument JSON not available. Re-ingest to generate.",
        )

    return Response(content=json_bytes, media_type="application/json")


@router.get("/documents/{document_id}/artifacts/{artifact_id}/image")
async def get_artifact_image(
    document_id: uuid.UUID,
    artifact_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Stream an artifact image from MinIO."""
    from app.services.storage import download_bytes_async
    from fastapi.responses import Response

    art = await db.get(Artifact, artifact_id)
    if not art or art.document_id != document_id or not art.storage_key:
        raise HTTPException(status_code=404, detail="Artifact not found")

    image_bytes = await download_bytes_async(art.storage_bucket, art.storage_key)
    ext = art.storage_key.rsplit(".", 1)[-1].lower() if "." in art.storage_key else "png"
    content_type = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "tiff": "image/tiff", "tif": "image/tiff", "gif": "image/gif",
        "bmp": "image/bmp",
    }.get(ext, "image/png")

    return Response(content=image_bytes, media_type=content_type)
