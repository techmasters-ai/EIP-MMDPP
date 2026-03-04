"""Multi-modal ingest pipeline.

Task graph for a single document:

    validate_and_store
        ↓
    detect_modalities
        ↓
    convert_document  (Docling service — replaces legacy extraction chord)
        ↓
    chunk_and_embed
        ↓
    extract_graph_entities
        ↓
    import_graph
        ↓
    finalize_artifact
        ↓
    ingest_to_cognee
"""

import hashlib
import logging
import uuid
from typing import Optional

from celery import chain, chord, group

from app.workers.celery_app import celery_app
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Pipeline status constants
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETE = "COMPLETE"
STATUS_PARTIAL_COMPLETE = "PARTIAL_COMPLETE"
STATUS_FAILED = "FAILED"
STATUS_PENDING_REVIEW = "PENDING_HUMAN_REVIEW"


def _get_db():
    """Get a synchronous DB session for Celery worker use."""
    from app.db.session import get_sync_session
    return get_sync_session()


def _update_document_status(
    document_id: str,
    status: str,
    stage: Optional[str] = None,
    error: Optional[str] = None,
    failed_stages: Optional[list[str]] = None,
) -> None:
    """Update document pipeline status in the database."""
    from sqlalchemy import update
    from app.models.ingest import Document

    db = _get_db()
    try:
        values = {
            "pipeline_status": status,
            "pipeline_stage": stage,
        }
        if error:
            values["error_message"] = error
        if failed_stages is not None:
            values["failed_stages"] = failed_stages

        db.execute(
            update(Document)
            .where(Document.id == uuid.UUID(document_id))
            .values(**values)
        )
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Entry point — called by the API after upload
# ---------------------------------------------------------------------------

def start_ingest_pipeline(document_id: str) -> str:
    """Enqueue the full ingest pipeline for a document. Returns Celery task ID."""
    pipeline = chain(
        validate_and_store.si(document_id),
        detect_modalities.si(document_id),
        convert_document.si(document_id),
        chunk_and_embed.si(document_id),
        extract_graph_entities.si(document_id),
        import_graph.si(document_id),
        finalize_artifact.si(document_id),
        ingest_to_cognee.si(document_id),
    )
    result = pipeline.apply_async()
    return result.id


# ---------------------------------------------------------------------------
# Pipeline tasks
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def validate_and_store(self, document_id: str) -> None:
    """Validate the uploaded file and confirm it is stored in MinIO."""
    from app.models.ingest import Document
    from app.services.storage import download_bytes_sync

    logger.info("validate_and_store: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="validate_and_store")

    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        # Verify the file exists in MinIO and compute hash if not set
        file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        from sqlalchemy import update
        from app.models.ingest import Document as Doc

        db.execute(
            update(Doc)
            .where(Doc.id == uuid.UUID(document_id))
            .values(
                file_size_bytes=len(file_bytes),
                file_hash=file_hash,
            )
        )
        db.commit()
        logger.info(
            "validate_and_store: document_id=%s hash=%s size=%d",
            document_id,
            file_hash,
            len(file_bytes),
        )
    except Exception as exc:
        logger.error("validate_and_store failed for %s: %s", document_id, exc)
        _update_document_status(
            document_id, STATUS_FAILED, stage="validate_and_store", error=str(exc)
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def detect_modalities(self, document_id: str) -> None:
    """Detect file type/modalities to determine which extraction tasks to run."""
    import magic
    from app.models.ingest import Document
    from app.services.storage import download_bytes_sync

    logger.info("detect_modalities: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="detect_modalities")

    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
        mime_type = magic.from_buffer(file_bytes, mime=True)

        from sqlalchemy import update
        from app.models.ingest import Document as Doc

        db.execute(
            update(Doc)
            .where(Doc.id == uuid.UUID(document_id))
            .values(mime_type=mime_type)
        )
        db.commit()
        logger.info("detect_modalities: document_id=%s mime_type=%s", document_id, mime_type)
    except Exception as exc:
        logger.error("detect_modalities failed for %s: %s", document_id, exc)
        # Non-fatal — continue pipeline
        _update_document_status(document_id, STATUS_PROCESSING, stage="detect_modalities")
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def convert_document(self, document_id: str) -> None:
    """Convert document via the Docling service (granite-docling-258M VLM).

    Replaces the legacy extraction chord (extract_text, extract_images,
    run_ocr, process_schematics) with a single Docling API call.
    Falls back to legacy extraction if the Docling service is unavailable
    and docling_fallback_enabled is True.
    """
    import uuid as uuid_mod
    from app.models.ingest import Document, Artifact
    from app.services.storage import download_bytes_sync, upload_bytes_sync
    from app.services.docling_client import convert_document_sync, check_health_sync

    logger.info("convert_document: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="convert_document")

    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)

        # Try Docling service first
        if check_health_sync():
            result = convert_document_sync(file_bytes, doc.filename or "document")
            logger.info(
                "convert_document: docling returned %d elements, %d pages, %.0fms",
                len(result.elements),
                result.num_pages,
                result.processing_time_ms,
            )
            _persist_extraction_results(db, document_id, result.elements)
        elif settings.docling_fallback_enabled:
            logger.warning(
                "convert_document: Docling service unavailable, falling back to "
                "legacy extraction for document_id=%s",
                document_id,
            )
            _legacy_extract(db, document_id, doc, file_bytes)
        else:
            raise RuntimeError("Docling service unavailable and fallback is disabled")

        db.commit()
    except Exception as exc:
        logger.error("convert_document failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_FAILED, stage="convert_document", error=str(exc)
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


def _persist_extraction_results(db, document_id: str, chunks) -> None:
    """Persist ExtractedChunk list as Artifact rows. Stores images in MinIO."""
    import uuid as uuid_mod
    from app.models.ingest import Artifact
    from app.services.storage import upload_bytes_sync

    for chunk in chunks:
        storage_bucket = None
        storage_key = None

        if chunk.raw_image_bytes:
            ext = chunk.metadata.get("ext", "png")
            img_key = f"artifacts/{document_id}/images/{uuid_mod.uuid4()}.{ext}"
            upload_bytes_sync(
                chunk.raw_image_bytes,
                settings.minio_bucket_derived,
                img_key,
                content_type=f"image/{ext}",
            )
            storage_bucket = settings.minio_bucket_derived
            storage_key = img_key

        artifact = Artifact(
            document_id=uuid.UUID(document_id),
            artifact_type=chunk.modality,
            content_text=chunk.chunk_text,
            content_metadata=chunk.metadata,
            storage_bucket=storage_bucket,
            storage_key=storage_key,
            page_number=chunk.page_number,
            bounding_box=chunk.bounding_box,
            ocr_confidence=chunk.ocr_confidence,
            ocr_engine=chunk.ocr_engine,
            requires_human_review=chunk.requires_human_review,
        )
        db.add(artifact)


def _legacy_extract(db, document_id: str, doc, file_bytes: bytes) -> None:
    """Fallback: run legacy extraction (pdfplumber/pymupdf/tesseract) inline."""
    from app.services.extraction import extract_pdf, extract_docx, extract_image

    mime = doc.mime_type or ""
    chunks = []

    if "pdf" in mime:
        chunks = extract_pdf(file_bytes)
    elif "word" in mime or "docx" in mime or "officedocument" in mime:
        chunks = extract_docx(file_bytes)
    elif "image" in mime:
        chunks = extract_image(file_bytes)

    _persist_extraction_results(db, document_id, chunks)


# ---------------------------------------------------------------------------
# DEPRECATED: Legacy extraction tasks (kept for _legacy_extract fallback path)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def extract_text(self, document_id: str) -> dict:
    """DEPRECATED: Extract text content from the document. Use convert_document instead."""
    from app.models.ingest import Document, Artifact
    from app.services.storage import download_bytes_sync
    from app.services.extraction import extract_pdf, extract_docx

    logger.info("extract_text: document_id=%s", document_id)
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            return {"stage": "extract_text", "status": "skipped", "reason": "document not found"}

        file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
        mime = doc.mime_type or ""

        extracted_chunks = []
        if "pdf" in mime:
            extracted_chunks = extract_pdf(file_bytes)
        elif "word" in mime or "docx" in mime or "officedocument" in mime:
            extracted_chunks = extract_docx(file_bytes)
        else:
            logger.info("extract_text: skipping non-text mime_type=%s", mime)
            return {"stage": "extract_text", "status": "skipped", "reason": f"unsupported mime: {mime}"}

        # Persist text artifacts
        text_artifacts = [c for c in extracted_chunks if c.modality in ("text", "table")]
        artifacts_created = 0
        for chunk in text_artifacts:
            artifact = Artifact(
                document_id=uuid.UUID(document_id),
                artifact_type=chunk.modality,
                content_text=chunk.chunk_text,
                content_metadata=chunk.metadata,
                page_number=chunk.page_number,
                bounding_box=chunk.bounding_box,
                ocr_confidence=chunk.ocr_confidence,
                ocr_engine=chunk.ocr_engine,
                requires_human_review=chunk.requires_human_review,
            )
            db.add(artifact)
            artifacts_created += 1

        db.commit()
        logger.info(
            "extract_text: document_id=%s artifacts_created=%d",
            document_id,
            artifacts_created,
        )
        return {"stage": "extract_text", "status": "ok", "artifacts_created": artifacts_created}

    except Exception as exc:
        logger.error("extract_text failed for %s: %s", document_id, exc)
        db.rollback()
        return {"stage": "extract_text", "status": "error", "error": str(exc)}
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def extract_images(self, document_id: str) -> dict:
    """Extract embedded images from the document and store them in MinIO."""
    import io
    import uuid as uuid_mod
    from app.models.ingest import Document, Artifact
    from app.services.storage import download_bytes_sync, upload_bytes_sync
    from app.services.extraction import extract_pdf

    logger.info("extract_images: document_id=%s", document_id)
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            return {"stage": "extract_images", "status": "skipped"}

        file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
        mime = doc.mime_type or ""

        if "pdf" not in mime:
            return {"stage": "extract_images", "status": "skipped", "reason": "not a pdf"}

        chunks = extract_pdf(file_bytes)
        image_chunks = [c for c in chunks if c.modality == "image" and c.raw_image_bytes]

        artifacts_created = 0
        for chunk in image_chunks:
            img_key = f"artifacts/{document_id}/images/{uuid_mod.uuid4()}.{chunk.metadata.get('ext', 'png')}"
            upload_bytes_sync(
                chunk.raw_image_bytes,
                settings.minio_bucket_derived,
                img_key,
                content_type=f"image/{chunk.metadata.get('ext', 'png')}",
            )
            artifact = Artifact(
                document_id=uuid.UUID(document_id),
                artifact_type="image",
                content_text=chunk.chunk_text or "",
                content_metadata=chunk.metadata,
                storage_bucket=settings.minio_bucket_derived,
                storage_key=img_key,
                page_number=chunk.page_number,
                bounding_box=chunk.bounding_box,
            )
            db.add(artifact)
            artifacts_created += 1

        db.commit()
        return {"stage": "extract_images", "status": "ok", "artifacts_created": artifacts_created}

    except Exception as exc:
        logger.error("extract_images failed for %s: %s", document_id, exc)
        db.rollback()
        return {"stage": "extract_images", "status": "error", "error": str(exc)}
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_ocr(self, document_id: str) -> dict:
    """Run OCR on image-type artifacts that don't yet have text."""
    from app.models.ingest import Document, Artifact
    from app.services.storage import download_bytes_sync
    from app.services.extraction import _ocr_image
    from sqlalchemy import select

    logger.info("run_ocr: document_id=%s", document_id)
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            return {"stage": "run_ocr", "status": "skipped"}

        # Find image artifacts without text content
        result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.artifact_type == "image",
                Artifact.content_text == "",
                Artifact.storage_key.isnot(None),
            )
        )
        image_artifacts = result.scalars().all()

        updated = 0
        for artifact in image_artifacts:
            try:
                image_bytes = download_bytes_sync(
                    artifact.storage_bucket, artifact.storage_key
                )
                ocr_chunks = _ocr_image(
                    image_bytes,
                    page_num=artifact.page_number,
                    tesseract_threshold=settings.ocr_tesseract_confidence_threshold,
                    easyocr_threshold=settings.ocr_easyocr_confidence_threshold,
                )
                if ocr_chunks:
                    chunk = ocr_chunks[0]
                    artifact.content_text = chunk.chunk_text
                    artifact.ocr_confidence = chunk.ocr_confidence
                    artifact.ocr_engine = chunk.ocr_engine
                    artifact.requires_human_review = chunk.requires_human_review
                    updated += 1
            except Exception as e:
                logger.warning("OCR failed for artifact %s: %s", artifact.id, e)

        db.commit()
        return {"stage": "run_ocr", "status": "ok", "updated": updated}

    except Exception as exc:
        logger.error("run_ocr failed for %s: %s", document_id, exc)
        db.rollback()
        return {"stage": "run_ocr", "status": "error", "error": str(exc)}
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=1, default_retry_delay=120)
def process_schematics(self, document_id: str) -> dict:
    """Process schematic/technical drawing artifacts via local Ollama VLM (llava).

    Sends each image artifact to llava to extract structured component/connection
    descriptions, which are then imported into the graph.
    """
    import base64
    import httpx
    from app.models.ingest import Document, Artifact
    from app.services.storage import download_bytes_sync
    from sqlalchemy import select

    logger.info("process_schematics: document_id=%s", document_id)
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            return {"stage": "process_schematics", "status": "skipped"}

        # Find image artifacts (potential schematics)
        result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.artifact_type == "image",
                Artifact.storage_key.isnot(None),
            )
        )
        image_artifacts = result.scalars().all()

        processed = 0
        for artifact in image_artifacts:
            try:
                image_bytes = download_bytes_sync(
                    artifact.storage_bucket, artifact.storage_key
                )
                b64_image = base64.b64encode(image_bytes).decode()

                # Call local Ollama VLM
                response = httpx.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_vlm_model,
                        "prompt": (
                            "This is a technical drawing or schematic from a military document. "
                            "Identify all labeled components, their connections, flow directions, "
                            "and any associated part numbers or specifications. "
                            "Return a JSON object with keys: components (list of {name, part_number, type}), "
                            "connections (list of {from, to, label}), notes (list of strings)."
                        ),
                        "images": [b64_image],
                        "format": "json",
                        "stream": False,
                    },
                    timeout=120.0,
                )

                if response.status_code == 200:
                    vlm_result = response.json().get("response", "{}")
                    import json
                    try:
                        structured = json.loads(vlm_result)
                    except json.JSONDecodeError:
                        structured = {"raw": vlm_result}

                    artifact.artifact_type = "schematic"
                    artifact.content_metadata = {
                        **(artifact.content_metadata or {}),
                        "vlm_extraction": structured,
                    }
                    # Store a textual summary as content_text for search
                    components = structured.get("components", [])
                    artifact.content_text = " ".join(
                        c.get("name", "") for c in components if c.get("name")
                    )
                    processed += 1

            except httpx.ConnectError:
                logger.warning(
                    "Ollama not available for schematic processing (document_id=%s). "
                    "Skipping schematic extraction.",
                    document_id,
                )
                break
            except Exception as e:
                logger.warning(
                    "process_schematics failed for artifact %s: %s", artifact.id, e
                )

        db.commit()
        return {"stage": "process_schematics", "status": "ok", "processed": processed}

    except Exception as exc:
        logger.error("process_schematics failed for %s: %s", document_id, exc)
        db.rollback()
        return {"stage": "process_schematics", "status": "error", "error": str(exc)}
    finally:
        db.close()


@celery_app.task(bind=True)
def collect_metadata(self, extraction_results: list[dict], document_id: str) -> None:
    """Collect and merge results from parallel extraction tasks."""
    logger.info("collect_metadata: document_id=%s results=%s", document_id, extraction_results)
    failed_stages = [
        r["stage"] for r in (extraction_results or []) if r.get("status") == "error"
    ]
    if failed_stages:
        logger.warning(
            "collect_metadata: document_id=%s failed_stages=%s", document_id, failed_stages
        )
    _update_document_status(document_id, STATUS_PROCESSING, stage="collect_metadata")


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def chunk_and_embed(self, document_id: str) -> None:
    """Split text artifacts into chunks and generate embeddings."""
    from app.models.ingest import Document, Artifact
    from app.models.retrieval import Chunk
    from app.services.extraction import chunk_text
    from app.services.embedding import embed_texts, embed_images
    from app.services.storage import download_bytes_sync
    from sqlalchemy import select
    import io

    logger.info("chunk_and_embed: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="chunk_and_embed")
    db = _get_db()
    try:
        # Process text/table/ocr/schematic artifacts → text embeddings
        result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.artifact_type.in_(["text", "table", "ocr", "schematic"]),
                Artifact.content_text.isnot(None),
            )
        )
        text_artifacts = result.scalars().all()

        all_texts = []
        all_artifact_refs = []

        for artifact in text_artifacts:
            if not artifact.content_text:
                continue
            text_chunks = chunk_text(artifact.content_text)
            for idx, chunk_str in enumerate(text_chunks):
                all_texts.append(chunk_str)
                all_artifact_refs.append((artifact, idx))

        if all_texts:
            embeddings = embed_texts(all_texts)
            for (artifact, idx), text, embedding in zip(
                all_artifact_refs, all_texts, embeddings
            ):
                chunk = Chunk(
                    artifact_id=artifact.id,
                    chunk_index=idx,
                    chunk_text=text,
                    embedding=embedding,
                    modality=artifact.artifact_type,
                    page_number=artifact.page_number,
                    bounding_box=artifact.bounding_box,
                    classification=artifact.classification,
                )
                db.add(chunk)

        # Process image artifacts → CLIP embeddings
        image_result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.artifact_type == "image",
                Artifact.storage_key.isnot(None),
            )
        )
        image_artifacts = image_result.scalars().all()

        if image_artifacts:
            from PIL import Image

            pil_images = []
            valid_image_artifacts = []
            for artifact in image_artifacts:
                try:
                    img_bytes = download_bytes_sync(
                        artifact.storage_bucket, artifact.storage_key
                    )
                    pil_images.append(Image.open(io.BytesIO(img_bytes)))
                    valid_image_artifacts.append(artifact)
                except Exception as e:
                    logger.warning("Could not load image for embedding: %s", e)

            if pil_images:
                image_embeddings = embed_images(pil_images)
                for artifact, img_embedding in zip(valid_image_artifacts, image_embeddings):
                    chunk = Chunk(
                        artifact_id=artifact.id,
                        chunk_index=0,
                        chunk_text=artifact.content_text or "",
                        image_embedding=img_embedding,
                        modality="image",
                        page_number=artifact.page_number,
                        bounding_box=artifact.bounding_box,
                        classification=artifact.classification,
                    )
                    db.add(chunk)

        db.commit()
        logger.info("chunk_and_embed: document_id=%s done", document_id)

    except Exception as exc:
        logger.error("chunk_and_embed failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE, stage="chunk_and_embed", error=str(exc)
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def extract_graph_entities(self, document_id: str) -> None:
    """Extract named entities and relationships from text artifacts via NER.

    Results are stored in each artifact's content_metadata under keys
    'extracted_entities' and 'extracted_relationships', which import_graph
    reads in the next pipeline stage.
    """
    from app.models.ingest import Artifact
    from app.services.ner import extract_entities, extract_relationships
    from sqlalchemy import select

    logger.info("extract_graph_entities: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="extract_graph_entities")
    db = _get_db()
    try:
        result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.artifact_type.in_(["text", "table", "ocr", "schematic"]),
                Artifact.content_text.isnot(None),
            )
        )
        text_artifacts = result.scalars().all()

        total_entities = 0
        total_relationships = 0

        for artifact in text_artifacts:
            if not artifact.content_text:
                continue

            entities = extract_entities(artifact.content_text)
            relationships = extract_relationships(artifact.content_text, entities)

            entity_dicts = [
                {
                    "entity_type": e.entity_type,
                    "name": e.name,
                    "confidence": e.confidence,
                    "properties": e.properties,
                }
                for e in entities
            ]
            rel_dicts = [
                {
                    "rel_type": r.rel_type,
                    "from_name": r.from_name,
                    "from_type": r.from_type,
                    "to_name": r.to_name,
                    "to_type": r.to_type,
                    "confidence": r.confidence,
                }
                for r in relationships
            ]

            metadata = dict(artifact.content_metadata or {})
            metadata["extracted_entities"] = entity_dicts
            metadata["extracted_relationships"] = rel_dicts
            artifact.content_metadata = metadata

            total_entities += len(entities)
            total_relationships += len(relationships)

        db.commit()
        logger.info(
            "extract_graph_entities: document_id=%s entities=%d relationships=%d",
            document_id,
            total_entities,
            total_relationships,
        )
    except Exception as exc:
        logger.error("extract_graph_entities failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id,
            STATUS_PARTIAL_COMPLETE,
            stage="extract_graph_entities",
            error=str(exc),
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def import_graph(self, document_id: str) -> None:
    """Import extracted entity/relation candidates into the Apache AGE graph.

    Reads 'extracted_entities' and 'extracted_relationships' stored by
    extract_graph_entities in each artifact's content_metadata.
    """
    from app.models.ingest import Artifact
    from app.services.graph import upsert_node, upsert_relationship
    from sqlalchemy import select

    logger.info("import_graph: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="import_graph")
    db = _get_db()
    try:
        result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.content_metadata.isnot(None),
            )
        )
        artifacts = result.scalars().all()

        nodes_created = 0
        edges_created = 0

        for artifact in artifacts:
            metadata = artifact.content_metadata or {}
            artifact_id_str = str(artifact.id)

            for entity in metadata.get("extracted_entities", []):
                node_id = upsert_node(
                    session=db,
                    entity_type=entity["entity_type"],
                    name=entity["name"],
                    artifact_id=artifact_id_str,
                    confidence=entity["confidence"],
                    properties=entity.get("properties"),
                )
                if node_id:
                    nodes_created += 1

            for rel in metadata.get("extracted_relationships", []):
                ok = upsert_relationship(
                    session=db,
                    from_name=rel["from_name"],
                    from_type=rel["from_type"],
                    to_name=rel["to_name"],
                    to_type=rel["to_type"],
                    rel_type=rel["rel_type"],
                    artifact_id=artifact_id_str,
                    confidence=rel["confidence"],
                )
                if ok:
                    edges_created += 1

        db.commit()
        logger.info(
            "import_graph: document_id=%s nodes=%d edges=%d",
            document_id,
            nodes_created,
            edges_created,
        )
    except Exception as exc:
        logger.error("import_graph failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE, stage="import_graph", error=str(exc)
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True)
def finalize_artifact(self, document_id: str) -> None:
    """Mark the document pipeline as COMPLETE."""
    logger.info("finalize_artifact: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_COMPLETE, stage=None)
    logger.info("finalize_artifact: document_id=%s — pipeline COMPLETE", document_id)


@celery_app.task(bind=True, max_retries=1)
def ingest_to_cognee(self, document_id: str) -> None:
    """Dual-ingest: push document text into Cognee for knowledge graph construction.

    This task is non-fatal — a Cognee failure sets failed_stages but does NOT
    change the pipeline status away from COMPLETE.  The EIP pipeline result is
    always the authoritative source of truth; Cognee is an optional enhancement.
    """
    import asyncio

    logger.info("ingest_to_cognee: document_id=%s", document_id)
    db = _get_db()
    try:
        from sqlalchemy import select
        from app.models.ingest import Artifact, Document

        # Fetch the source name (used as the Cognee dataset name for isolation)
        doc_result = db.execute(
            select(Document).where(Document.id == uuid.UUID(document_id))
        )
        document = doc_result.scalar_one_or_none()
        if document is None:
            logger.warning("ingest_to_cognee: document %s not found", document_id)
            return

        # Collect all artifact texts for this document
        art_result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.content_text.isnot(None),
            )
        )
        artifacts = art_result.scalars().all()
        if not artifacts:
            logger.info("ingest_to_cognee: no text artifacts for %s — skipping", document_id)
            return

        dataset_name = str(document.source_id)

        async def _run():
            from app.services.cognee_service import cognee_add, cognee_cognify
            for artifact in artifacts:
                if artifact.content_text:
                    await cognee_add(artifact.content_text, dataset_name)
            await cognee_cognify(dataset_name)

        asyncio.run(_run())
        logger.info(
            "ingest_to_cognee: completed for document_id=%s dataset=%s",
            document_id,
            dataset_name,
        )

    except Exception as exc:
        logger.warning(
            "ingest_to_cognee failed for %s (non-fatal): %s", document_id, exc
        )
        # Record the failure without changing overall pipeline status
        try:
            _update_document_status(
                document_id,
                STATUS_COMPLETE,
                stage=None,
                error=f"cognee_ingest_failed: {exc}",
            )
        except Exception:
            pass
    finally:
        db.close()
