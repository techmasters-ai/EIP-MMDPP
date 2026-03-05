"""Multi-modal ingest pipeline.

Task graph for a single document:

    validate_and_store
        ↓
    detect_modalities
        ↓
    convert_document  (Docling service — replaces legacy extraction chord)
        ↓
    ┌── embed_text_chunks ──┐
    │                       │  (parallel via chord)
    └── embed_image_chunks ─┘
        ↓
    collect_embeddings  (chord callback)
        ↓
    extract_graph  (docling-graph or NER fallback)
        ↓
    import_graph
        ↓
    connect_document_elements  (DOCUMENT/CHUNK_REF/SAME_PAGE edges)
        ↓
    finalize_artifact
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
        chord(
            group(
                embed_text_chunks.si(document_id),
                embed_image_chunks.si(document_id),
            ),
            collect_embeddings.s(document_id),
        ),
        extract_graph.si(document_id),
        import_graph.si(document_id),
        connect_document_elements.si(document_id),
        finalize_artifact.si(document_id),
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
        docling_healthy = check_health_sync()
        docling_succeeded = False

        if docling_healthy:
            try:
                result = convert_document_sync(file_bytes, doc.filename or "document")
                logger.info(
                    "convert_document: docling returned %d elements, %d pages, %.0fms",
                    len(result.elements),
                    result.num_pages,
                    result.processing_time_ms,
                )
                _persist_extraction_results(db, document_id, result.elements)
                docling_succeeded = True
            except Exception as docling_exc:
                logger.warning(
                    "convert_document: Docling conversion failed for document_id=%s: %s",
                    document_id,
                    docling_exc,
                )
                if not settings.docling_fallback_enabled:
                    raise

        if not docling_succeeded:
            if settings.docling_fallback_enabled:
                logger.warning(
                    "convert_document: falling back to legacy extraction for document_id=%s",
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
    from app.services.extraction import extract_pdf, extract_docx, extract_image, extract_txt

    mime = doc.mime_type or ""
    chunks = []

    if "pdf" in mime:
        chunks = extract_pdf(file_bytes)
    elif "word" in mime or "docx" in mime or "officedocument" in mime:
        chunks = extract_docx(file_bytes)
    elif "image" in mime:
        chunks = extract_image(file_bytes)
    elif "text" in mime:
        chunks = extract_txt(file_bytes)

    _persist_extraction_results(db, document_id, chunks)


# ---------------------------------------------------------------------------
# Embedding tasks (run in parallel via chord)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed")
def embed_text_chunks(self, document_id: str) -> dict:
    """Split text artifacts into chunks, embed with BGE, write to retrieval.text_chunks."""
    from app.models.ingest import Document, Artifact
    from app.models.retrieval import TextChunk
    from app.services.extraction import chunk_text
    from app.services.embedding import embed_texts
    from sqlalchemy import select

    logger.info("embed_text_chunks: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="embed_text_chunks")
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            return {"stage": "embed_text_chunks", "status": "skipped", "reason": "not found"}

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
            text_chunks_list = chunk_text(artifact.content_text)
            for idx, chunk_str in enumerate(text_chunks_list):
                all_texts.append(chunk_str)
                all_artifact_refs.append((artifact, idx))

        chunks_created = 0
        if all_texts:
            embeddings = embed_texts(all_texts)
            for (artifact, idx), text, embedding in zip(
                all_artifact_refs, all_texts, embeddings
            ):
                chunk = TextChunk(
                    artifact_id=artifact.id,
                    document_id=uuid.UUID(document_id),
                    chunk_index=idx,
                    chunk_text=text,
                    embedding=embedding,
                    modality=artifact.artifact_type,
                    page_number=artifact.page_number,
                    bounding_box=artifact.bounding_box,
                    classification=artifact.classification,
                )
                db.add(chunk)
                chunks_created += 1

        db.commit()
        logger.info(
            "embed_text_chunks: document_id=%s chunks=%d", document_id, chunks_created
        )
        return {"stage": "embed_text_chunks", "status": "ok", "chunks": chunks_created}

    except Exception as exc:
        logger.error("embed_text_chunks failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE, stage="embed_text_chunks", error=str(exc)
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed")
def embed_image_chunks(self, document_id: str) -> dict:
    """Embed image artifacts with CLIP, write to retrieval.image_chunks."""
    import io
    from app.models.ingest import Document, Artifact
    from app.models.retrieval import ImageChunk
    from app.services.embedding import embed_images
    from app.services.storage import download_bytes_sync
    from sqlalchemy import select

    logger.info("embed_image_chunks: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="embed_image_chunks")
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            return {"stage": "embed_image_chunks", "status": "skipped", "reason": "not found"}

        image_result = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.artifact_type == "image",
                Artifact.storage_key.isnot(None),
            )
        )
        image_artifacts = image_result.scalars().all()

        chunks_created = 0
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
                    chunk = ImageChunk(
                        artifact_id=artifact.id,
                        document_id=uuid.UUID(document_id),
                        chunk_index=0,
                        chunk_text=artifact.content_text or None,
                        embedding=img_embedding,
                        modality="image",
                        page_number=artifact.page_number,
                        bounding_box=artifact.bounding_box,
                        classification=artifact.classification,
                    )
                    db.add(chunk)
                    chunks_created += 1

        db.commit()
        logger.info(
            "embed_image_chunks: document_id=%s chunks=%d", document_id, chunks_created
        )
        return {"stage": "embed_image_chunks", "status": "ok", "chunks": chunks_created}

    except Exception as exc:
        logger.error("embed_image_chunks failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE, stage="embed_image_chunks", error=str(exc)
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True)
def collect_embeddings(self, embedding_results: list[dict], document_id: str) -> None:
    """Chord callback: collect results from parallel embedding tasks."""
    logger.info(
        "collect_embeddings: document_id=%s results=%s", document_id, embedding_results
    )
    failed = [r["stage"] for r in (embedding_results or []) if r.get("status") == "error"]
    if failed:
        logger.warning(
            "collect_embeddings: document_id=%s failed_stages=%s", document_id, failed
        )
    _update_document_status(document_id, STATUS_PROCESSING, stage="collect_embeddings")


# ---------------------------------------------------------------------------
# Graph extraction & import
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="graph")
def extract_graph(self, document_id: str) -> None:
    """Extract entities and relationships from text artifacts.

    Uses docling-graph + LLM (Phase 5) with fallback to regex NER.
    Results are stored in artifact content_metadata for import_graph.
    """
    from app.models.ingest import Artifact
    from sqlalchemy import select

    logger.info("extract_graph: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="extract_graph")
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

        # Try docling-graph first (Phase 5 implementation)
        docling_graph_succeeded = False
        if settings.llm_provider != "mock":
            try:
                from app.services.docling_graph_service import extract_graph_from_text

                # Concatenate all text for whole-document extraction
                full_text = "\n\n".join(
                    a.content_text for a in text_artifacts if a.content_text
                )
                if full_text.strip():
                    nx_graph = extract_graph_from_text(full_text, document_id)
                    if nx_graph and nx_graph.number_of_nodes() > 0:
                        # Store graph data in first artifact's metadata
                        import json
                        graph_data = {
                            "nodes": [
                                {"id": n, **nx_graph.nodes[n]}
                                for n in nx_graph.nodes
                            ],
                            "edges": [
                                {"from": u, "to": v, **d}
                                for u, v, d in nx_graph.edges(data=True)
                            ],
                        }
                        for artifact in text_artifacts:
                            metadata = dict(artifact.content_metadata or {})
                            metadata["docling_graph_data"] = graph_data
                            artifact.content_metadata = metadata
                        docling_graph_succeeded = True
                        logger.info(
                            "extract_graph: docling-graph extracted %d nodes, %d edges",
                            nx_graph.number_of_nodes(),
                            nx_graph.number_of_edges(),
                        )
            except ImportError:
                logger.debug("docling-graph not available, falling back to NER")
            except Exception as exc:
                logger.warning("docling-graph extraction failed: %s — falling back to NER", exc)

        # Fallback to regex NER
        if not docling_graph_succeeded:
            _extract_graph_ner_fallback(db, document_id, text_artifacts)

        db.commit()
    except Exception as exc:
        logger.error("extract_graph failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE, stage="extract_graph", error=str(exc)
        )
        raise self.retry(exc=exc)
    finally:
        db.close()


def _extract_graph_ner_fallback(db, document_id: str, text_artifacts) -> None:
    """Fallback: use regex NER for entity/relationship extraction."""
    from app.services.ner import extract_entities, extract_relationships

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

    logger.info(
        "extract_graph (NER fallback): document_id=%s entities=%d relationships=%d",
        document_id,
        total_entities,
        total_relationships,
    )


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="graph")
def import_graph(self, document_id: str) -> None:
    """Import extracted entity/relation candidates into the Apache AGE graph.

    Handles both docling-graph format (docling_graph_data) and legacy
    NER format (extracted_entities / extracted_relationships).
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

            # Handle docling-graph format
            graph_data = metadata.get("docling_graph_data")
            if graph_data:
                for node in graph_data.get("nodes", []):
                    node_id = upsert_node(
                        session=db,
                        entity_type=node.get("entity_type", "UNKNOWN"),
                        name=node.get("name", node.get("id", "")),
                        artifact_id=artifact_id_str,
                        confidence=node.get("confidence", 0.8),
                        properties={
                            k: v for k, v in node.items()
                            if k not in ("id", "entity_type", "name", "confidence")
                        },
                    )
                    if node_id:
                        nodes_created += 1

                for edge in graph_data.get("edges", []):
                    ok = upsert_relationship(
                        session=db,
                        from_name=edge.get("from", ""),
                        from_type=edge.get("from_type", "UNKNOWN"),
                        to_name=edge.get("to", ""),
                        to_type=edge.get("to_type", "UNKNOWN"),
                        rel_type=edge.get("rel_type", edge.get("type", "RELATED_TO")),
                        artifact_id=artifact_id_str,
                        confidence=edge.get("confidence", 0.8),
                    )
                    if ok:
                        edges_created += 1
                continue  # Skip legacy format for this artifact

            # Handle legacy NER format
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


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph")
def connect_document_elements(self, document_id: str) -> None:
    """Create structural graph edges connecting text/image chunks from the same document.

    Creates:
    - DOCUMENT node for this document
    - CHUNK_REF nodes for each text_chunk and image_chunk
    - CONTAINS_TEXT edges: DOCUMENT → text CHUNK_REF
    - CONTAINS_IMAGE edges: DOCUMENT → image CHUNK_REF
    - SAME_PAGE edges: text CHUNK_REF ↔ image CHUNK_REF on same page
    """
    from app.models.ingest import Document
    from app.models.retrieval import TextChunk, ImageChunk
    from sqlalchemy import select

    logger.info("connect_document_elements: document_id=%s", document_id)
    _update_document_status(
        document_id, STATUS_PROCESSING, stage="connect_document_elements"
    )
    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            logger.warning("connect_document_elements: document %s not found", document_id)
            return

        # Import graph helpers
        from app.services.graph import (
            upsert_document_node,
            upsert_chunk_ref_node,
            create_structural_edge,
        )

        # 1. Create DOCUMENT node
        upsert_document_node(
            session=db,
            document_id=document_id,
            title=doc.filename,
            properties={"source_id": str(doc.source_id)},
        )

        # 2. Get all text and image chunks for this document
        text_chunks = db.execute(
            select(TextChunk).where(TextChunk.document_id == uuid.UUID(document_id))
        ).scalars().all()

        image_chunks = db.execute(
            select(ImageChunk).where(ImageChunk.document_id == uuid.UUID(document_id))
        ).scalars().all()

        # 3. Create CHUNK_REF nodes and CONTAINS edges
        for tc in text_chunks:
            upsert_chunk_ref_node(db, str(tc.id), "text_chunk")
            create_structural_edge(db, document_id, str(tc.id), "CONTAINS_TEXT")

        for ic in image_chunks:
            upsert_chunk_ref_node(db, str(ic.id), "image_chunk")
            create_structural_edge(db, document_id, str(ic.id), "CONTAINS_IMAGE")

        # 4. Create SAME_PAGE edges between text and image chunks
        page_text_map: dict[int, list[str]] = {}
        for tc in text_chunks:
            if tc.page_number is not None:
                page_text_map.setdefault(tc.page_number, []).append(str(tc.id))

        for ic in image_chunks:
            if ic.page_number is not None and ic.page_number in page_text_map:
                for tc_id in page_text_map[ic.page_number]:
                    create_structural_edge(db, tc_id, str(ic.id), "SAME_PAGE")

        db.commit()
        logger.info(
            "connect_document_elements: document_id=%s text=%d image=%d",
            document_id,
            len(text_chunks),
            len(image_chunks),
        )
    except Exception as exc:
        logger.error("connect_document_elements failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id,
            STATUS_PARTIAL_COMPLETE,
            stage="connect_document_elements",
            error=str(exc),
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


# ---------------------------------------------------------------------------
# DEPRECATED: Legacy tasks (kept for backwards compatibility)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def chunk_and_embed(self, document_id: str) -> None:
    """DEPRECATED: Use embed_text_chunks + embed_image_chunks instead."""
    logger.warning("chunk_and_embed is DEPRECATED — use embed_text_chunks/embed_image_chunks")
    embed_text_chunks(document_id)
    embed_image_chunks(document_id)


@celery_app.task(bind=True, max_retries=1)
def ingest_to_cognee(self, document_id: str) -> None:
    """DEPRECATED: Cognee is now a governed memory layer, not part of the pipeline.

    Use POST /v1/memory/ingest to propose knowledge, then approve via
    POST /v1/memory/proposals/{id}/approve.
    """
    logger.warning(
        "ingest_to_cognee is DEPRECATED — Cognee is now a governed memory layer. "
        "Use /v1/memory/ingest + /v1/memory/proposals/{id}/approve instead."
    )


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def extract_graph_entities(self, document_id: str) -> None:
    """DEPRECATED: Use extract_graph instead."""
    logger.warning("extract_graph_entities is DEPRECATED — use extract_graph")
    extract_graph(document_id)
