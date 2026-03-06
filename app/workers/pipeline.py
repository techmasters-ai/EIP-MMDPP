"""Multi-modal ingest pipeline.

V1 task graph (sequential chain):

    validate_and_store → detect_modalities → convert_document
        → chord(embed_text_chunks, embed_image_chunks) → collect_embeddings
        → extract_graph → import_graph → connect_document_elements → finalize_artifact

V2 task graph (manifest-first, parallel derivations, idempotent):

    prepare_document  (validate + detect + Docling convert + persist document_elements)
        ↓
    ┌── derive_text_chunks_and_embeddings ──┐
    │── derive_image_embeddings             │  (parallel chord)
    │── derive_ontology_graph               │
    └── derive_structure_links ─────────────┘
        ↓
    collect_derivations  (chord callback)
        ↓
    finalize_document_v2

Selected via INGEST_V2_ENABLED env var (default: false).
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
                        # Store graph_data on first artifact only; others reference it
                        if text_artifacts:
                            first_meta = dict(text_artifacts[0].content_metadata or {})
                            first_meta["docling_graph_data"] = graph_data
                            text_artifacts[0].content_metadata = first_meta
                            for artifact in text_artifacts[1:]:
                                ref_meta = dict(artifact.content_metadata or {})
                                ref_meta["docling_graph_source_artifact_id"] = str(text_artifacts[0].id)
                                artifact.content_metadata = ref_meta
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
    - EXTRACTED_FROM edges: ontology entity → CHUNK_REF (provenance)
    """
    from app.models.ingest import Document, Artifact
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
            create_entity_chunk_edge,
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

        # 5. Link ontology entities to their source chunks (EXTRACTED_FROM)
        entity_links = 0
        artifacts_with_entities = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.content_metadata.isnot(None),
            )
        ).scalars().all()

        # Build artifact_id → [chunk_id] map from already-loaded text_chunks
        artifact_chunk_map: dict[str, list[str]] = {}
        for tc in text_chunks:
            artifact_chunk_map.setdefault(str(tc.artifact_id), []).append(str(tc.id))

        for artifact in artifacts_with_entities:
            metadata = artifact.content_metadata or {}
            chunk_ids = artifact_chunk_map.get(str(artifact.id), [])
            if not chunk_ids:
                continue

            # Collect entity (name, type) from either format
            entities: list[tuple[str, str]] = []
            graph_data = metadata.get("docling_graph_data")
            if graph_data:
                for node in graph_data.get("nodes", []):
                    entities.append((
                        node.get("name", node.get("id", "")),
                        node.get("entity_type", "UNKNOWN"),
                    ))
            else:
                for ent in metadata.get("extracted_entities", []):
                    entities.append((ent["name"], ent["entity_type"]))

            for (name, etype) in entities:
                for chunk_id in chunk_ids:
                    if create_entity_chunk_edge(db, name, etype, chunk_id):
                        entity_links += 1

        db.commit()
        logger.info(
            "connect_document_elements: document_id=%s text=%d image=%d entity_links=%d",
            document_id,
            len(text_chunks),
            len(image_chunks),
            entity_links,
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


# ===========================================================================
# V2 Ingest Pipeline — manifest-first, parallel derivations, idempotent
# ===========================================================================

def start_ingest_pipeline_v2(document_id: str) -> str:
    """Enqueue the v2 ingest pipeline. Returns Celery task ID.

    DAG:
        prepare_document
            ↓
        ┌── derive_text_chunks_and_embeddings ──┐
        │── derive_image_embeddings             │  (parallel chord)
        │── derive_ontology_graph               │
        └── derive_structure_links ─────────────┘
            ↓
        collect_derivations  (chord callback)
            ↓
        finalize_document_v2
    """
    pipeline = chain(
        prepare_document.si(document_id),
        chord(
            group(
                derive_text_chunks_and_embeddings.si(document_id),
                derive_image_embeddings.si(document_id),
                derive_ontology_graph.si(document_id),
                derive_structure_links.si(document_id),
            ),
            collect_derivations.s(document_id),
        ),
        finalize_document_v2.si(document_id),
    )
    result = pipeline.apply_async()
    return result.id


def _create_pipeline_run(db, document_id: str) -> str:
    """Create a PipelineRun record and return its id as string."""
    from app.models.ingest import PipelineRun
    import uuid as uuid_mod

    run = PipelineRun(
        document_id=uuid.UUID(document_id),
        pipeline_version="v2",
        status="PROCESSING",
    )
    db.add(run)
    db.flush()
    return str(run.id)


def _update_stage_run(
    db, pipeline_run_id: str, stage_name: str, status: str,
    attempt: int = 1, metrics: dict | None = None, error: str | None = None,
) -> None:
    """Upsert a StageRun record."""
    from app.models.ingest import StageRun
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import datetime

    values = {
        "pipeline_run_id": uuid.UUID(pipeline_run_id),
        "stage_name": stage_name,
        "attempt": attempt,
        "status": status,
    }
    if status == "RUNNING":
        values["started_at"] = datetime.datetime.now(datetime.timezone.utc)
    if status in ("COMPLETE", "FAILED"):
        values["finished_at"] = datetime.datetime.now(datetime.timezone.utc)
    if metrics:
        values["metrics"] = metrics
    if error:
        values["error_message"] = error

    stmt = pg_insert(StageRun).values(**values).on_conflict_do_update(
        constraint="stage_runs_pipeline_run_id_stage_name_attempt_key",
        set_={k: v for k, v in values.items() if k not in ("pipeline_run_id", "stage_name", "attempt")},
    )
    db.execute(stmt)


def _get_pipeline_run_id(db, document_id: str) -> str | None:
    """Get the latest v2 pipeline run id for a document."""
    from app.models.ingest import PipelineRun
    from sqlalchemy import select

    result = db.execute(
        select(PipelineRun.id)
        .where(
            PipelineRun.document_id == uuid.UUID(document_id),
            PipelineRun.pipeline_version == "v2",
        )
        .order_by(PipelineRun.started_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return str(row) if row else None


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def prepare_document(self, document_id: str) -> str:
    """V2 entry point: validate + detect + Docling convert + persist document_elements.

    Creates canonical DocumentElement rows from Docling output, with backward-
    compatible Artifact dual-write.
    """
    import uuid as uuid_mod
    from app.models.ingest import Document, Artifact, DocumentElement
    from app.services.storage import download_bytes_sync, upload_bytes_sync
    from app.services.docling_client import convert_document_sync, check_health_sync
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import magic

    logger.info("prepare_document [v2]: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="prepare_document")

    db = _get_db()
    try:
        # 1. Create PipelineRun
        run_id = _create_pipeline_run(db, document_id)
        _update_stage_run(db, run_id, "prepare_document", "RUNNING")
        db.commit()

        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            raise ValueError(f"Document not found: {document_id}")

        # 2. Download + validate
        file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        mime_type = magic.from_buffer(file_bytes, mime=True)

        from sqlalchemy import update as sql_update
        db.execute(
            sql_update(Document)
            .where(Document.id == uuid.UUID(document_id))
            .values(
                file_size_bytes=len(file_bytes),
                file_hash=file_hash,
                mime_type=mime_type,
            )
        )
        db.commit()

        # 3. Docling conversion
        docling_healthy = check_health_sync()
        if not docling_healthy:
            if settings.docling_fallback_enabled:
                logger.warning("prepare_document: Docling unhealthy, falling back to legacy for %s", document_id)
                _legacy_extract(db, document_id, doc, file_bytes)
                db.commit()
                _update_stage_run(db, run_id, "prepare_document", "COMPLETE", metrics={"fallback": True})
                db.commit()
                return document_id
            raise RuntimeError("Docling service unavailable and fallback is disabled")

        result = convert_document_sync(file_bytes, doc.filename or "document")
        logger.info(
            "prepare_document: docling returned %d elements, %d pages, %.0fms",
            len(result.elements), result.num_pages, result.processing_time_ms,
        )

        # 4. Persist canonical DocumentElement rows (upsert by document_id + element_uid)
        elements_created = 0
        for chunk in result.elements:
            element_uid = (chunk.metadata or {}).get("element_uid")
            if not element_uid:
                # Generate a fallback uid
                content_hash = hashlib.sha256(
                    (chunk.chunk_text or "").encode("utf-8", errors="replace")
                ).hexdigest()[:8]
                element_uid = f"{chunk.page_number or 0}-{elements_created}-{chunk.modality}-{content_hash}"

            storage_bucket = None
            storage_key = None

            # Store image bytes in MinIO
            if chunk.raw_image_bytes:
                ext = (chunk.metadata or {}).get("ext", "png")
                img_key = f"artifacts/{document_id}/images/{uuid_mod.uuid4()}.{ext}"
                upload_bytes_sync(
                    chunk.raw_image_bytes,
                    settings.minio_bucket_derived,
                    img_key,
                    content_type=f"image/{ext}",
                )
                storage_bucket = settings.minio_bucket_derived
                storage_key = img_key

            # Element hash for idempotency
            element_hash = hashlib.sha256(
                f"{document_id}:{element_uid}:{chunk.chunk_text or ''}".encode()
            ).hexdigest()

            element_values = {
                "document_id": uuid.UUID(document_id),
                "element_uid": element_uid,
                "element_type": chunk.modality,
                "element_order": (chunk.metadata or {}).get("element_order", elements_created),
                "page_number": chunk.page_number,
                "bounding_box": chunk.bounding_box,
                "section_path": (chunk.metadata or {}).get("section_path"),
                "heading_level": (chunk.metadata or {}).get("heading_level"),
                "content_text": chunk.chunk_text,
                "storage_bucket": storage_bucket,
                "storage_key": storage_key,
                "metadata": chunk.metadata or {},
                "element_hash": element_hash,
            }

            stmt = pg_insert(DocumentElement).values(**element_values).on_conflict_do_update(
                constraint="document_elements_document_id_element_uid_key",
                set_={
                    "element_type": element_values["element_type"],
                    "element_order": element_values["element_order"],
                    "content_text": element_values["content_text"],
                    "storage_bucket": element_values["storage_bucket"],
                    "storage_key": element_values["storage_key"],
                    "metadata": element_values["metadata"],
                    "element_hash": element_values["element_hash"],
                },
            )
            db.execute(stmt)
            elements_created += 1

        # 5. Dual-write Artifact rows for backward compatibility
        _persist_extraction_results(db, document_id, result.elements)

        db.commit()

        _update_stage_run(
            db, run_id, "prepare_document", "COMPLETE",
            metrics={
                "elements": elements_created,
                "num_pages": result.num_pages,
                "processing_time_ms": result.processing_time_ms,
            },
        )
        db.commit()

        logger.info(
            "prepare_document [v2]: document_id=%s elements=%d pages=%d",
            document_id, elements_created, result.num_pages,
        )
        return document_id

    except Exception as exc:
        logger.error("prepare_document failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_FAILED, stage="prepare_document", error=str(exc)
        )
        run_id_for_err = _get_pipeline_run_id(db, document_id)
        if run_id_for_err:
            _update_stage_run(db, run_id_for_err, "prepare_document", "FAILED", error=str(exc))
            db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed")
def derive_text_chunks_and_embeddings(self, document_id: str) -> dict:
    """V2: Read text/table/heading document_elements → chunk → BGE embed → upsert text_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    from app.models.ingest import DocumentElement
    from app.models.retrieval import TextChunk
    from app.services.extraction import chunk_text
    from app.services.embedding import embed_texts
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    logger.info("derive_text_chunks_and_embeddings [v2]: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_text_embeddings")

    db = _get_db()
    try:
        run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "RUNNING")
            db.commit()

        # Advisory lock to prevent concurrent runs for same document
        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || ':text_embed'))"
            ),
            {"doc_id": document_id},
        )

        elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type.in_(["text", "table", "heading", "equation", "schematic"]),
                DocumentElement.content_text.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        all_texts = []
        all_element_refs = []

        for elem in elements:
            if not elem.content_text:
                continue
            text_chunks_list = chunk_text(elem.content_text)
            for idx, chunk_str in enumerate(text_chunks_list):
                all_texts.append(chunk_str)
                all_element_refs.append((elem, idx))

        chunks_created = 0
        if all_texts:
            embeddings = embed_texts(all_texts)
            model_version = settings.text_embedding_model

            for (elem, idx), text, embedding in zip(all_element_refs, all_texts, embeddings):
                # Deterministic chunk key
                chunk_key = hashlib.sha256(
                    f"{document_id}:{elem.element_uid}:{idx}:{model_version}".encode()
                ).hexdigest()

                chunk_values = {
                    "id": uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest()),
                    "artifact_id": elem.artifact_id,
                    "document_id": uuid.UUID(document_id),
                    "chunk_index": idx,
                    "chunk_text": text,
                    "embedding": embedding,
                    "modality": elem.element_type if elem.element_type != "heading" else "text",
                    "page_number": elem.page_number,
                    "bounding_box": elem.bounding_box,
                }

                stmt = pg_insert(TextChunk).values(**chunk_values).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "chunk_text": chunk_values["chunk_text"],
                        "embedding": chunk_values["embedding"],
                        "modality": chunk_values["modality"],
                    },
                )
                db.execute(stmt)
                chunks_created += 1

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_text_embeddings", "COMPLETE",
                metrics={"chunks": chunks_created, "elements": len(elements)},
            )
            db.commit()

        logger.info(
            "derive_text_chunks_and_embeddings [v2]: document_id=%s chunks=%d",
            document_id, chunks_created,
        )
        return {"stage": "derive_text_embeddings", "status": "ok", "chunks": chunks_created}

    except Exception as exc:
        logger.error("derive_text_chunks_and_embeddings failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE,
            stage="derive_text_embeddings", error=str(exc),
        )
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED", error=str(exc))
            db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed")
def derive_image_embeddings(self, document_id: str) -> dict:
    """V2: Read image document_elements → CLIP embed → upsert image_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    import io
    from app.models.ingest import DocumentElement
    from app.models.retrieval import ImageChunk
    from app.services.embedding import embed_images
    from app.services.storage import download_bytes_sync
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    logger.info("derive_image_embeddings [v2]: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_image_embeddings")

    db = _get_db()
    try:
        run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_image_embeddings", "RUNNING")
            db.commit()

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || ':image_embed'))"
            ),
            {"doc_id": document_id},
        )

        elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type == "image",
                DocumentElement.storage_key.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        chunks_created = 0
        if elements:
            from PIL import Image

            pil_images = []
            valid_elements = []
            for elem in elements:
                try:
                    img_bytes = download_bytes_sync(elem.storage_bucket, elem.storage_key)
                    pil_images.append(Image.open(io.BytesIO(img_bytes)))
                    valid_elements.append(elem)
                except Exception as e:
                    logger.warning("Could not load image element %s: %s", elem.element_uid, e)

            if pil_images:
                image_embeddings = embed_images(pil_images)
                model_version = settings.image_embedding_model

                for elem, img_embedding in zip(valid_elements, image_embeddings):
                    chunk_key = hashlib.sha256(
                        f"{document_id}:{elem.element_uid}:{model_version}".encode()
                    ).hexdigest()

                    chunk_values = {
                        "id": uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest()),
                        "artifact_id": elem.artifact_id,
                        "document_id": uuid.UUID(document_id),
                        "chunk_index": 0,
                        "chunk_text": elem.content_text or None,
                        "embedding": img_embedding,
                        "modality": "image",
                        "page_number": elem.page_number,
                        "bounding_box": elem.bounding_box,
                    }

                    stmt = pg_insert(ImageChunk).values(**chunk_values).on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "chunk_text": chunk_values["chunk_text"],
                            "embedding": chunk_values["embedding"],
                        },
                    )
                    db.execute(stmt)
                    chunks_created += 1

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_image_embeddings", "COMPLETE",
                metrics={"chunks": chunks_created, "elements": len(elements)},
            )
            db.commit()

        logger.info(
            "derive_image_embeddings [v2]: document_id=%s chunks=%d",
            document_id, chunks_created,
        )
        return {"stage": "derive_image_embeddings", "status": "ok", "chunks": chunks_created}

    except Exception as exc:
        logger.error("derive_image_embeddings failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE,
            stage="derive_image_embeddings", error=str(exc),
        )
        if run_id:
            _update_stage_run(db, run_id, "derive_image_embeddings", "FAILED", error=str(exc))
            db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="graph")
def derive_ontology_graph(self, document_id: str) -> dict:
    """V2: Read ordered text elements → LLM/NER extraction → upsert document_graph_extractions → import to AGE.

    Stores graph extraction once per document (not per artifact).
    """
    from app.models.ingest import DocumentElement, DocumentGraphExtraction
    from app.services.graph import upsert_node, upsert_relationship
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    logger.info("derive_ontology_graph [v2]: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_ontology_graph")

    db = _get_db()
    try:
        run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_ontology_graph", "RUNNING")
            db.commit()

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || ':ontology_graph'))"
            ),
            {"doc_id": document_id},
        )

        elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type.in_(["text", "table", "heading", "equation", "schematic"]),
                DocumentElement.content_text.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        full_text = "\n\n".join(e.content_text for e in elements if e.content_text)
        if not full_text.strip():
            logger.info("derive_ontology_graph: no text elements for %s", document_id)
            if run_id:
                _update_stage_run(db, run_id, "derive_ontology_graph", "COMPLETE", metrics={"skipped": True})
                db.commit()
            return {"stage": "derive_ontology_graph", "status": "ok", "nodes": 0, "edges": 0}

        # Try docling-graph + LLM first, fallback to NER
        graph_data = None
        provider = "ner"
        model_name = "regex"

        if settings.llm_provider != "mock":
            try:
                from app.services.docling_graph_service import extract_graph_from_text
                nx_graph = extract_graph_from_text(full_text, document_id)
                if nx_graph and nx_graph.number_of_nodes() > 0:
                    graph_data = {
                        "nodes": [{"id": n, **nx_graph.nodes[n]} for n in nx_graph.nodes],
                        "edges": [{"from": u, "to": v, **d} for u, v, d in nx_graph.edges(data=True)],
                    }
                    provider = "docling-graph"
                    model_name = settings.llm_model_name
            except ImportError:
                logger.debug("docling-graph not available, falling back to NER")
            except Exception as exc:
                logger.warning("docling-graph failed: %s — falling back to NER", exc)

        if graph_data is None:
            from app.services.ner import extract_entities, extract_relationships
            entities = extract_entities(full_text)
            relationships = extract_relationships(full_text, entities)
            graph_data = {
                "nodes": [
                    {"id": e.name, "entity_type": e.entity_type, "name": e.name, "confidence": e.confidence, "properties": e.properties}
                    for e in entities
                ],
                "edges": [
                    {"from": r.from_name, "to": r.to_name, "from_type": r.from_type, "to_type": r.to_type, "rel_type": r.rel_type, "confidence": r.confidence}
                    for r in relationships
                ],
            }

        # Upsert into document_graph_extractions (one row per document)
        extraction_values = {
            "document_id": uuid.UUID(document_id),
            "provider": provider,
            "model_name": model_name,
            "extraction_version": "v2",
            "graph_json": graph_data,
            "status": "COMPLETE",
            "metrics": {"nodes": len(graph_data["nodes"]), "edges": len(graph_data["edges"])},
        }
        stmt = pg_insert(DocumentGraphExtraction).values(**extraction_values).on_conflict_do_update(
            index_elements=["document_id"],
            set_={
                "provider": extraction_values["provider"],
                "model_name": extraction_values["model_name"],
                "graph_json": extraction_values["graph_json"],
                "status": "COMPLETE",
                "metrics": extraction_values["metrics"],
            },
        )
        db.execute(stmt)

        # Import into AGE
        nodes_created = 0
        edges_created = 0

        for node in graph_data.get("nodes", []):
            node_id = upsert_node(
                session=db,
                entity_type=node.get("entity_type", "UNKNOWN"),
                name=node.get("name", node.get("id", "")),
                artifact_id=document_id,  # use document_id as provenance
                confidence=node.get("confidence", 0.8),
                properties={k: v for k, v in node.items() if k not in ("id", "entity_type", "name", "confidence", "properties")},
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
                artifact_id=document_id,
                confidence=edge.get("confidence", 0.8),
            )
            if ok:
                edges_created += 1

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_ontology_graph", "COMPLETE",
                metrics={"nodes": nodes_created, "edges": edges_created, "provider": provider},
            )
            db.commit()

        logger.info(
            "derive_ontology_graph [v2]: document_id=%s nodes=%d edges=%d provider=%s",
            document_id, nodes_created, edges_created, provider,
        )
        return {"stage": "derive_ontology_graph", "status": "ok", "nodes": nodes_created, "edges": edges_created}

    except Exception as exc:
        logger.error("derive_ontology_graph failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE,
            stage="derive_ontology_graph", error=str(exc),
        )
        if run_id:
            _update_stage_run(db, run_id, "derive_ontology_graph", "FAILED", error=str(exc))
            db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph")
def derive_structure_links(self, document_id: str) -> dict:
    """V2: Generate chunk_links and structural AGE edges.

    Creates:
    - NEXT_CHUNK links between consecutive text_chunks
    - SAME_PAGE links between text and image chunks on same page
    - SAME_SECTION links for chunks sharing section_path
    - DOCUMENT node, CHUNK_REF nodes, CONTAINS/SAME_PAGE AGE edges
    - EXTRACTED_FROM edges linking ontology entities to chunk refs
    """
    from app.models.ingest import Document, DocumentElement, Artifact
    from app.models.retrieval import TextChunk, ImageChunk, ChunkLink
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    logger.info("derive_structure_links [v2]: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_structure_links")

    db = _get_db()
    try:
        run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_structure_links", "RUNNING")
            db.commit()

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || ':structure_links'))"
            ),
            {"doc_id": document_id},
        )

        doc = db.get(Document, uuid.UUID(document_id))
        if not doc:
            logger.warning("derive_structure_links: document %s not found", document_id)
            return {"stage": "derive_structure_links", "status": "skipped"}

        # Fetch chunks
        text_chunks = db.execute(
            select(TextChunk)
            .where(TextChunk.document_id == uuid.UUID(document_id))
            .order_by(TextChunk.page_number.nullslast(), TextChunk.chunk_index)
        ).scalars().all()

        image_chunks = db.execute(
            select(ImageChunk)
            .where(ImageChunk.document_id == uuid.UUID(document_id))
        ).scalars().all()

        # Fetch document_elements for section_path metadata
        elements = db.execute(
            select(DocumentElement)
            .where(DocumentElement.document_id == uuid.UUID(document_id))
            .order_by(DocumentElement.element_order)
        ).scalars().all()

        # Build element_uid → section_path map
        element_section_map = {}
        for elem in elements:
            if elem.element_uid and elem.section_path:
                element_section_map[elem.element_uid] = elem.section_path

        # Build artifact_id → element map for section lookups
        artifact_element_map = {}
        for elem in elements:
            if elem.artifact_id:
                artifact_element_map[str(elem.artifact_id)] = elem

        links_created = 0

        def _upsert_link(source_id, target_id, link_type, hop, weight):
            nonlocal links_created
            vals = {
                "source_chunk_id": uuid.UUID(str(source_id)),
                "target_chunk_id": uuid.UUID(str(target_id)),
                "document_id": uuid.UUID(document_id),
                "link_type": link_type,
                "hop": hop,
                "weight": weight,
            }
            stmt = pg_insert(ChunkLink).values(**vals).on_conflict_do_update(
                constraint="chunk_links_pkey",
                set_={"weight": weight},
            )
            db.execute(stmt)
            links_created += 1

        # NEXT_CHUNK links (consecutive text chunks)
        for i in range(len(text_chunks) - 1):
            _upsert_link(
                text_chunks[i].id, text_chunks[i + 1].id,
                "NEXT_CHUNK", 1, settings.retrieval_weight_next_chunk,
            )
            # Bidirectional
            _upsert_link(
                text_chunks[i + 1].id, text_chunks[i].id,
                "NEXT_CHUNK", 1, settings.retrieval_weight_next_chunk,
            )

        # SAME_PAGE links
        page_text_map: dict[int, list] = {}
        for tc in text_chunks:
            if tc.page_number is not None:
                page_text_map.setdefault(tc.page_number, []).append(tc)

        page_image_map: dict[int, list] = {}
        for ic in image_chunks:
            if ic.page_number is not None:
                page_image_map.setdefault(ic.page_number, []).append(ic)

        for page_num, ics in page_image_map.items():
            tcs = page_text_map.get(page_num, [])
            for ic in ics:
                for tc in tcs:
                    _upsert_link(
                        tc.id, ic.id, "SAME_PAGE", 1,
                        settings.retrieval_weight_same_page,
                    )
                    _upsert_link(
                        ic.id, tc.id, "SAME_PAGE", 1,
                        settings.retrieval_weight_same_page,
                    )

        # SAME_SECTION links (chunks whose source elements share section_path)
        section_chunks: dict[str, list] = {}
        for tc in text_chunks:
            if tc.artifact_id and str(tc.artifact_id) in artifact_element_map:
                elem = artifact_element_map[str(tc.artifact_id)]
                if elem.section_path:
                    section_chunks.setdefault(elem.section_path, []).append(tc)

        for section, chunks in section_chunks.items():
            for i, c1 in enumerate(chunks):
                for c2 in chunks[i + 1:]:
                    if c1.id != c2.id:
                        _upsert_link(
                            c1.id, c2.id, "SAME_SECTION", 1,
                            settings.retrieval_weight_same_section,
                        )
                        _upsert_link(
                            c2.id, c1.id, "SAME_SECTION", 1,
                            settings.retrieval_weight_same_section,
                        )

        # SAME_ARTIFACT links (different chunks from same artifact)
        artifact_chunks: dict[str, list] = {}
        for tc in text_chunks:
            if tc.artifact_id:
                artifact_chunks.setdefault(str(tc.artifact_id), []).append(tc)

        for art_id, chunks in artifact_chunks.items():
            if len(chunks) > 1:
                for i, c1 in enumerate(chunks):
                    for c2 in chunks[i + 1:]:
                        _upsert_link(
                            c1.id, c2.id, "SAME_ARTIFACT", 1,
                            settings.retrieval_weight_same_artifact,
                        )
                        _upsert_link(
                            c2.id, c1.id, "SAME_ARTIFACT", 1,
                            settings.retrieval_weight_same_artifact,
                        )

        db.commit()

        # Create AGE structural edges (reuse existing v1 logic)
        from app.services.graph import (
            upsert_document_node,
            upsert_chunk_ref_node,
            create_structural_edge,
            create_entity_chunk_edge,
        )

        upsert_document_node(
            session=db,
            document_id=document_id,
            title=doc.filename,
            properties={"source_id": str(doc.source_id)},
        )

        for tc in text_chunks:
            upsert_chunk_ref_node(db, str(tc.id), "text_chunk")
            create_structural_edge(db, document_id, str(tc.id), "CONTAINS_TEXT")

        for ic in image_chunks:
            upsert_chunk_ref_node(db, str(ic.id), "image_chunk")
            create_structural_edge(db, document_id, str(ic.id), "CONTAINS_IMAGE")

        for page_num, ics in page_image_map.items():
            tcs = page_text_map.get(page_num, [])
            for ic in ics:
                for tc in tcs:
                    create_structural_edge(db, str(tc.id), str(ic.id), "SAME_PAGE")

        # Entity-chunk EXTRACTED_FROM edges
        entity_links = 0
        artifacts_with_entities = db.execute(
            select(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                Artifact.content_metadata.isnot(None),
            )
        ).scalars().all()

        artifact_chunk_map: dict[str, list[str]] = {}
        for tc in text_chunks:
            artifact_chunk_map.setdefault(str(tc.artifact_id), []).append(str(tc.id))

        for artifact in artifacts_with_entities:
            metadata = artifact.content_metadata or {}
            chunk_ids = artifact_chunk_map.get(str(artifact.id), [])
            if not chunk_ids:
                continue

            entities_list: list[tuple[str, str]] = []
            graph_data = metadata.get("docling_graph_data")
            if graph_data:
                for node in graph_data.get("nodes", []):
                    entities_list.append((
                        node.get("name", node.get("id", "")),
                        node.get("entity_type", "UNKNOWN"),
                    ))
            else:
                for ent in metadata.get("extracted_entities", []):
                    entities_list.append((ent["name"], ent["entity_type"]))

            for (name, etype) in entities_list:
                for chunk_id in chunk_ids:
                    if create_entity_chunk_edge(db, name, etype, chunk_id):
                        entity_links += 1

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_structure_links", "COMPLETE",
                metrics={
                    "chunk_links": links_created,
                    "entity_links": entity_links,
                    "text_chunks": len(text_chunks),
                    "image_chunks": len(image_chunks),
                },
            )
            db.commit()

        logger.info(
            "derive_structure_links [v2]: document_id=%s chunk_links=%d entity_links=%d",
            document_id, links_created, entity_links,
        )
        return {"stage": "derive_structure_links", "status": "ok", "links": links_created}

    except Exception as exc:
        logger.error("derive_structure_links failed for %s: %s", document_id, exc)
        db.rollback()
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE,
            stage="derive_structure_links", error=str(exc),
        )
        if run_id:
            _update_stage_run(db, run_id, "derive_structure_links", "FAILED", error=str(exc))
            db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True)
def collect_derivations(self, derivation_results: list[dict], document_id: str) -> None:
    """Chord callback: aggregate derivation stage statuses."""
    logger.info(
        "collect_derivations [v2]: document_id=%s results=%s",
        document_id, derivation_results,
    )
    failed = [r["stage"] for r in (derivation_results or []) if r.get("status") not in ("ok", "skipped")]
    if failed:
        logger.warning(
            "collect_derivations: document_id=%s failed_stages=%s", document_id, failed
        )
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE,
            stage="collect_derivations",
            failed_stages=failed,
        )
    else:
        _update_document_status(document_id, STATUS_PROCESSING, stage="collect_derivations")


@celery_app.task(bind=True)
def finalize_document_v2(self, document_id: str) -> None:
    """V2 finalization: mark pipeline COMPLETE if all required stages succeeded."""
    from app.models.ingest import PipelineRun, StageRun
    from sqlalchemy import select, update as sql_update
    import datetime

    logger.info("finalize_document_v2 [v2]: document_id=%s", document_id)
    db = _get_db()
    try:
        run_id = _get_pipeline_run_id(db, document_id)
        if not run_id:
            _update_document_status(document_id, STATUS_COMPLETE, stage=None)
            return

        # Check for any failed stages
        failed_stages = db.execute(
            select(StageRun.stage_name)
            .where(
                StageRun.pipeline_run_id == uuid.UUID(run_id),
                StageRun.status == "FAILED",
            )
        ).scalars().all()

        if failed_stages:
            final_status = STATUS_PARTIAL_COMPLETE
            logger.warning(
                "finalize_document_v2: document_id=%s has failed stages: %s",
                document_id, failed_stages,
            )
        else:
            final_status = STATUS_COMPLETE

        _update_document_status(document_id, final_status, stage=None)

        # Update PipelineRun
        db.execute(
            sql_update(PipelineRun)
            .where(PipelineRun.id == uuid.UUID(run_id))
            .values(
                status=final_status,
                finished_at=datetime.datetime.now(datetime.timezone.utc),
            )
        )
        db.commit()

        logger.info(
            "finalize_document_v2 [v2]: document_id=%s — pipeline %s",
            document_id, final_status,
        )
    except Exception as exc:
        logger.error("finalize_document_v2 failed for %s: %s", document_id, exc)
        db.rollback()
    finally:
        db.close()


