"""Multi-modal ingest pipeline.

Task graph (manifest-first, parallel derivations, idempotent):

    prepare_document  (validate + detect + Docling convert + persist document_elements)
        ↓
    ┌── derive_text_chunks_and_embeddings ──┐
    │── derive_image_embeddings             │  (parallel chord)
    └── derive_ontology_graph ──────────────┘
        ↓
    collect_derivations  (chord callback)
        ↓
    derive_structure_links  (needs embedding output committed)
        ↓
    derive_canonicalization  (entity alias resolution)
        ↓
    finalize_document
"""

import hashlib
import logging
import uuid
from typing import Optional

import httpx
import redis as redis_lib
from celery import chain, chord, group
from celery.exceptions import Retry as CeleryRetry

from app.workers.celery_app import celery_app
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Redis-based semaphore for Docling concurrency control
_redis_client = redis_lib.Redis.from_url(settings.celery_broker_url)

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
            "error_message": error,  # None clears previous errors
        }
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


def _deterministic_artifact_id(document_id: str, element_uid: str) -> uuid.UUID:
    """Generate a deterministic artifact UUID from document_id + element_uid.

    Uses uuid5 with URL namespace so the same (document_id, element_uid)
    pair always produces the same artifact ID.  This replaces the old
    positional zip-linking approach.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{element_uid}")


def _persist_extraction_results(db, document_id: str, chunks, element_uids: list[str] | None = None) -> list[uuid.UUID]:
    """Persist ExtractedChunk list as Artifact rows. Stores images in MinIO.

    If *element_uids* is provided (one per chunk, same order), each Artifact
    gets a deterministic ID derived from document_id + element_uid.

    Returns the list of artifact IDs (in chunk order).
    """
    import uuid as uuid_mod
    from app.models.ingest import Artifact
    from app.services.storage import upload_bytes_sync

    artifact_ids: list[uuid.UUID] = []
    for idx, chunk in enumerate(chunks):
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

        artifact_id = (
            _deterministic_artifact_id(document_id, element_uids[idx])
            if element_uids
            else uuid_mod.uuid4()
        )

        artifact = Artifact(
            id=artifact_id,
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
        artifact_ids.append(artifact_id)

    return artifact_ids


def _legacy_extract(db, document_id: str, doc, file_bytes: bytes) -> None:
    """Fallback: run legacy extraction (pdfplumber/pymupdf/tesseract) inline.

    Creates both Artifact and DocumentElement rows so downstream derivation
    tasks (embedding, graph, structure links) have input to work with.
    """
    from app.services.extraction import extract_pdf, extract_docx, extract_image, extract_txt
    from app.models.ingest import DocumentElement

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

    # Build element_uids first, then persist Artifacts with deterministic IDs
    element_uids: list[str] = []
    for idx, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(
            (chunk.chunk_text or "").encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        element_uids.append(f"legacy-{idx}-{chunk.modality}-{content_hash}")

    artifact_ids = _persist_extraction_results(db, document_id, chunks, element_uids=element_uids)

    # Create DocumentElement rows with artifact_id linked inline
    for idx, chunk in enumerate(chunks):
        element_uid = element_uids[idx]
        element_hash = hashlib.sha256(
            f"{document_id}:{element_uid}:{chunk.chunk_text or ''}".encode()
        ).hexdigest()

        elem = DocumentElement(
            document_id=uuid.UUID(document_id),
            element_uid=element_uid,
            element_type=chunk.modality,
            element_order=idx,
            page_number=chunk.page_number,
            bounding_box=chunk.bounding_box,
            content_text=chunk.chunk_text,
            element_metadata=chunk.metadata or {},
            element_hash=element_hash,
            artifact_id=artifact_ids[idx],
        )
        db.add(elem)


def start_ingest_pipeline(document_id: str) -> str:
    """Enqueue the ingest pipeline for a document. Returns Celery task ID."""
    # Create PipelineRun before enqueuing so all stages share the same run_id
    db = _get_db()
    try:
        run_id = _create_pipeline_run(db, document_id)
        db.commit()
    finally:
        db.close()

    pipeline = chain(
        prepare_document.si(document_id, run_id),
        purge_document_derivations.si(document_id, run_id),
        chord(
            group(
                derive_text_chunks_and_embeddings.si(document_id, run_id),
                derive_image_embeddings.si(document_id, run_id),
                derive_ontology_graph.si(document_id, run_id),
            ),
            collect_derivations.s(document_id, run_id),
        ),
        derive_structure_links.si(document_id, run_id),
        derive_canonicalization.si(document_id, run_id),
        finalize_document.si(document_id, run_id),
    )
    result = pipeline.apply_async()
    return result.id


def _create_pipeline_run(db, document_id: str) -> str:
    """Create a PipelineRun record and return its id as string."""
    from app.models.ingest import PipelineRun
    import uuid as uuid_mod

    run = PipelineRun(
        document_id=uuid.UUID(document_id),
        pipeline_version="1.0",
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
    """Get the latest pipeline run id for a document."""
    from app.models.ingest import PipelineRun
    from sqlalchemy import select

    result = db.execute(
        select(PipelineRun.id)
        .where(
            PipelineRun.document_id == uuid.UUID(document_id),
            PipelineRun.pipeline_version == "1.0",
        )
        .order_by(PipelineRun.started_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return str(row) if row else None


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30,
                 soft_time_limit=600, time_limit=660)
def prepare_document(self, document_id: str, run_id: str | None = None) -> str:
    """Validate + detect + Docling convert + persist document_elements.

    Creates canonical DocumentElement rows from Docling output, with backward-
    compatible Artifact dual-write.
    """
    import uuid as uuid_mod
    from app.models.ingest import Document, Artifact, DocumentElement
    from app.services.storage import download_bytes_sync, upload_bytes_sync
    from app.services.docling_client import convert_document_sync, check_health_sync
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import magic

    # Apply env-var configurable retry and time-limit settings
    self.max_retries = settings.prepare_max_retries
    self.default_retry_delay = settings.prepare_retry_delay
    self.soft_time_limit = settings.prepare_soft_time_limit
    self.time_limit = settings.prepare_time_limit

    logger.info("prepare_document: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="prepare_document")

    db = _get_db()
    try:
        # Use passed run_id or create one (backward compat)
        if not run_id:
            run_id = _create_pipeline_run(db, document_id)
            db.commit()
        _update_stage_run(db, run_id, "prepare_document", "RUNNING", attempt=self.request.retries + 1)
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

        # 3. Route by format — Docling only handles PDF and images
        _DOCLING_MIMES = {"application/pdf", "image/png", "image/jpeg", "image/tiff", "image/bmp", "image/gif"}
        if mime_type not in _DOCLING_MIMES:
            logger.info("prepare_document: %s not supported by Docling (mime=%s), using legacy extraction", document_id, mime_type)
            _legacy_extract(db, document_id, doc, file_bytes)
            db.commit()
            _update_stage_run(db, run_id, "prepare_document", "COMPLETE", attempt=self.request.retries + 1, metrics={"fallback": True, "reason": "unsupported_format"})
            db.commit()
            return document_id

        # 4. Docling conversion — acquire semaphore permit with busy-wait
        #    Uses in-task backoff loop instead of self.retry() so capacity waits
        #    do NOT consume the task's retry budget (retries reserved for real errors).
        import time as _time

        docling_lock = None
        _wait_start = _time.monotonic()
        _max_wait = settings.docling_lock_timeout  # 600s default
        _wait_attempt = 0
        while docling_lock is None:
            for _permit_i in range(settings.docling_concurrency):
                _candidate = _redis_client.lock(
                    f"docling:permit:{_permit_i}", timeout=settings.docling_lock_timeout, blocking=False,
                )
                if _candidate.acquire(blocking=False):
                    docling_lock = _candidate
                    break
            if docling_lock is not None:
                break
            _elapsed = _time.monotonic() - _wait_start
            if _elapsed >= _max_wait:
                raise RuntimeError(
                    f"Docling at capacity for {_elapsed:.0f}s — all {settings.docling_concurrency} "
                    f"permits held. Document {document_id} cannot proceed."
                )
            _wait_attempt += 1
            _sleep_time = min(30, 5 * _wait_attempt)  # 5s, 10s, 15s, ... capped at 30s
            logger.info(
                "prepare_document: Docling at capacity (%d/%d), waiting %ds for %s (%.0fs elapsed)",
                settings.docling_concurrency, settings.docling_concurrency,
                _sleep_time, document_id, _elapsed,
            )
            _time.sleep(_sleep_time)

        try:
            # Advisory health check — log but don't fail (health endpoint
            # may time out during long CPU conversions on other docs, but
            # convert itself will work once the semaphore permits it).
            docling_healthy = check_health_sync()
            if not docling_healthy:
                logger.warning(
                    "prepare_document: Docling health check failed (advisory) for %s — proceeding with convert",
                    document_id,
                )
            result = convert_document_sync(file_bytes, doc.filename or "document")
        finally:
            try:
                docling_lock.release()
            except redis_lib.exceptions.LockNotOwnedError:
                logger.warning("prepare_document: Docling lock expired before release for %s", document_id)
        logger.info(
            "prepare_document: docling returned %d elements, %d pages, %.0fms",
            len(result.elements), result.num_pages, result.processing_time_ms,
        )

        # 4. Build element_uids, then persist Artifacts with deterministic IDs
        element_uids: list[str] = []
        elements_created = 0
        for chunk in result.elements:
            element_uid = (chunk.metadata or {}).get("element_uid")
            if not element_uid:
                content_hash = hashlib.sha256(
                    (chunk.chunk_text or "").encode("utf-8", errors="replace")
                ).hexdigest()[:8]
                element_uid = f"{chunk.page_number or 0}-{elements_created}-{chunk.modality}-{content_hash}"
            element_uids.append(element_uid)
            elements_created += 1

        # 5. Dual-write Artifact rows with deterministic IDs
        artifact_ids = _persist_extraction_results(db, document_id, result.elements, element_uids=element_uids)
        db.flush()  # Ensure artifact rows visible for FK checks in Core SQL inserts below

        # 6. Persist canonical DocumentElement rows with artifact_id linked inline
        elements_created = 0
        for idx, chunk in enumerate(result.elements):
            element_uid = element_uids[idx]

            storage_bucket = None
            storage_key = None

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
                "element_metadata": chunk.metadata or {},
                "element_hash": element_hash,
                "artifact_id": artifact_ids[idx],
            }

            stmt = pg_insert(DocumentElement).values(**element_values)
            stmt = stmt.on_conflict_do_update(
                constraint="document_elements_document_id_element_uid_key",
                set_={
                    "element_type": stmt.excluded.element_type,
                    "element_order": stmt.excluded.element_order,
                    "content_text": stmt.excluded.content_text,
                    "storage_bucket": stmt.excluded.storage_bucket,
                    "storage_key": stmt.excluded.storage_key,
                    "metadata": stmt.excluded.metadata,
                    "element_hash": stmt.excluded.element_hash,
                    "artifact_id": stmt.excluded.artifact_id,
                },
            )
            db.execute(stmt)
            elements_created += 1

        db.commit()

        # Remove stale DocumentElements/Artifacts not in current extraction
        from sqlalchemy import delete as sql_delete
        stale_elems = db.execute(
            sql_delete(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                ~DocumentElement.element_uid.in_(element_uids),
            )
        )
        stale_elem_count = stale_elems.rowcount

        stale_arts = db.execute(
            sql_delete(Artifact).where(
                Artifact.document_id == uuid.UUID(document_id),
                ~Artifact.id.in_(artifact_ids),
            )
        )
        stale_art_count = stale_arts.rowcount
        if stale_elem_count or stale_art_count:
            db.commit()
            logger.info(
                "prepare_document: cleaned %d stale elements, %d stale artifacts for %s",
                stale_elem_count, stale_art_count, document_id,
            )

        _update_stage_run(
            db, run_id, "prepare_document", "COMPLETE",
            attempt=self.request.retries + 1,
            metrics={
                "elements": elements_created,
                "num_pages": result.num_pages,
                "processing_time_ms": result.processing_time_ms,
                "stale_elements_removed": stale_elem_count,
                "stale_artifacts_removed": stale_art_count,
            },
        )
        db.commit()

        logger.info(
            "prepare_document: document_id=%s elements=%d pages=%d",
            document_id, elements_created, result.num_pages,
        )
        return document_id

    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("prepare_document failed for %s: %s", document_id, exc)
        db.rollback()

        # Docling 5xx: fall back to legacy extraction if enabled
        # doc/file_bytes are assigned before Docling call, so always in scope for 5xx
        if (
            settings.docling_fallback_enabled
            and isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code >= 500
        ):
            logger.warning(
                "prepare_document: Docling %d for %s — falling back to legacy extraction",
                exc.response.status_code, document_id,
            )
            try:
                _legacy_extract(db, document_id, doc, file_bytes)
                db.commit()
                _update_stage_run(
                    db, run_id, "prepare_document", "COMPLETE",
                    attempt=self.request.retries + 1,
                    metrics={"fallback": True, "reason": f"docling_{exc.response.status_code}"},
                )
                db.commit()
                return document_id
            except Exception as fallback_exc:
                logger.error("prepare_document: legacy fallback also failed for %s: %s", document_id, fallback_exc)
                db.rollback()
                # Fall through to normal retry/fail logic

        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_FAILED, stage="prepare_document", error=str(exc)
            )
            run_id_for_err = run_id or _get_pipeline_run_id(db, document_id)
            if run_id_for_err:
                _update_stage_run(db, run_id_for_err, "prepare_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("prepare_document: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, soft_time_limit=120, time_limit=180, queue="ingest")
def purge_document_derivations(self, document_id: str, run_id: str | None = None) -> str:
    """Delete stale derived data for a document before re-deriving.

    Purges: TextChunks, ImageChunks, ChunkLinks (Postgres),
    Qdrant vectors (both collections), Neo4j structural subgraph.
    Idempotent — safe to call on first ingest (no-op if nothing exists).
    """
    from app.models.retrieval import TextChunk, ImageChunk, ChunkLink
    from app.models.ingest import DocumentGraphExtraction
    from app.services.qdrant_store import delete_by_document_id
    from app.db.session import get_neo4j_driver, get_qdrant_client
    from sqlalchemy import delete as sql_delete

    logger.info("purge_document_derivations: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="purge_document_derivations")

    db = _get_db()
    metrics: dict = {}
    try:
        if run_id:
            _update_stage_run(db, run_id, "purge_document_derivations", "RUNNING", attempt=1)
            db.commit()

        doc_uuid = uuid.UUID(document_id)

        # 1. Postgres derived tables
        for model, label in [
            (ChunkLink, "chunk_links"),
            (TextChunk, "text_chunks"),
            (ImageChunk, "image_chunks"),
        ]:
            result = db.execute(
                sql_delete(model).where(model.document_id == doc_uuid)
            )
            metrics[f"{label}_deleted"] = result.rowcount

        result = db.execute(
            sql_delete(DocumentGraphExtraction).where(
                DocumentGraphExtraction.document_id == doc_uuid
            )
        )
        metrics["graph_extractions_deleted"] = result.rowcount
        db.commit()

        # 2. Qdrant vectors
        try:
            qdrant_client = get_qdrant_client()
            delete_by_document_id(qdrant_client, document_id)
            metrics["qdrant_purged"] = True
        except Exception as exc:
            logger.warning("purge: Qdrant cleanup failed for %s: %s", document_id, exc)
            metrics["qdrant_purged"] = False

        # 3. Neo4j — delete document structural subgraph
        try:
            neo4j_driver = get_neo4j_driver()
            with neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (d:Document {document_id: $doc_id})-[]->(c:ChunkRef)
                    DETACH DELETE c
                    RETURN count(c) AS deleted_chunks
                """, doc_id=document_id)
                record = result.single()
                metrics["neo4j_chunks_deleted"] = record["deleted_chunks"] if record else 0

                result = session.run("""
                    MATCH ()-[r]->()
                    WHERE r.artifact_id = $doc_id
                    DELETE r
                    RETURN count(r) AS deleted_edges
                """, doc_id=document_id)
                record = result.single()
                metrics["neo4j_edges_deleted"] = record["deleted_edges"] if record else 0
        except Exception as exc:
            logger.warning("purge: Neo4j cleanup failed for %s: %s", document_id, exc)
            metrics["neo4j_purge_error"] = str(exc)

        if run_id:
            _update_stage_run(db, run_id, "purge_document_derivations", "COMPLETE", attempt=1, metrics=metrics)
            db.commit()

        logger.info("purge_document_derivations: document_id=%s metrics=%s", document_id, metrics)
        return document_id

    except Exception as exc:
        logger.error("purge_document_derivations failed for %s: %s", document_id, exc)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "purge_document_derivations", "FAILED", attempt=1, error=str(exc))
            db.commit()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed",
                 soft_time_limit=300, time_limit=360)
def derive_text_chunks_and_embeddings(self, document_id: str, run_id: str | None = None) -> dict:
    """Read text/table/heading document_elements → chunk → BGE embed → upsert text_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    from app.models.ingest import DocumentElement
    from app.models.retrieval import TextChunk
    from app.services.extraction import chunk_text
    from app.services.embedding import embed_texts
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    self.max_retries = settings.embed_max_retries
    self.default_retry_delay = settings.embed_retry_delay
    self.soft_time_limit = settings.embed_soft_time_limit
    self.time_limit = settings.embed_time_limit

    logger.info("derive_text_chunks_and_embeddings: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_text_embeddings")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Advisory lock to prevent concurrent runs for same document
        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || '_text_embed'))"
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

            # Get Qdrant client for vector upsert
            from app.db.session import get_qdrant_client
            from app.services.qdrant_store import upsert_text_vector
            qdrant = get_qdrant_client()

            for (elem, idx), text, embedding in zip(all_element_refs, all_texts, embeddings):
                # Deterministic chunk key
                chunk_key = hashlib.sha256(
                    f"{document_id}:{elem.element_uid}:{idx}:{model_version}".encode()
                ).hexdigest()

                chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())
                qdrant_point_id = chunk_id  # Use same UUID for Qdrant

                chunk_values = {
                    "id": chunk_id,
                    "artifact_id": elem.artifact_id,
                    "document_id": uuid.UUID(document_id),
                    "chunk_index": idx,
                    "chunk_text": text,
                    "embedding": embedding,
                    "modality": elem.element_type if elem.element_type != "heading" else "text",
                    "page_number": elem.page_number,
                    "bounding_box": elem.bounding_box,
                    "qdrant_point_id": qdrant_point_id,
                }

                stmt = pg_insert(TextChunk).values(**chunk_values).on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "chunk_text": chunk_values["chunk_text"],
                        "embedding": chunk_values["embedding"],
                        "modality": chunk_values["modality"],
                        "qdrant_point_id": qdrant_point_id,
                    },
                )
                db.execute(stmt)

                # Upsert vector to Qdrant
                upsert_text_vector(
                    qdrant,
                    point_id=qdrant_point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": str(chunk_id),
                        "document_id": document_id,
                        "artifact_id": str(elem.artifact_id),
                        "modality": elem.element_type if elem.element_type != "heading" else "text",
                        "page_number": elem.page_number,
                        "classification": "UNCLASSIFIED",
                        "chunk_text": text,
                    },
                )
                chunks_created += 1

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_text_embeddings", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"chunks": chunks_created, "elements": len(elements)},
            )
            db.commit()

        logger.info(
            "derive_text_chunks_and_embeddings: document_id=%s chunks=%d",
            document_id, chunks_created,
        )
        return {"stage": "derive_text_embeddings", "status": "ok", "chunks": chunks_created}

    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("derive_text_chunks_and_embeddings failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_text_embeddings", error=str(exc),
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("derive_text_chunks_and_embeddings: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed",
                 soft_time_limit=300, time_limit=360)
def derive_image_embeddings(self, document_id: str, run_id: str | None = None) -> dict:
    """Read image document_elements → CLIP embed → upsert image_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    import io
    from app.models.ingest import DocumentElement
    from app.models.retrieval import ImageChunk
    from app.services.embedding import embed_images
    from app.services.storage import download_bytes_sync
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    self.max_retries = settings.embed_max_retries
    self.default_retry_delay = settings.embed_retry_delay
    self.soft_time_limit = settings.embed_soft_time_limit
    self.time_limit = settings.embed_time_limit

    logger.info("derive_image_embeddings: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_image_embeddings")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_image_embeddings", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || '_image_embed'))"
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

                # Get Qdrant client for vector upsert
                from app.db.session import get_qdrant_client
                from app.services.qdrant_store import upsert_image_vector
                qdrant = get_qdrant_client()

                for elem, img_embedding in zip(valid_elements, image_embeddings):
                    chunk_key = hashlib.sha256(
                        f"{document_id}:{elem.element_uid}:{model_version}".encode()
                    ).hexdigest()

                    chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())
                    qdrant_point_id = chunk_id

                    chunk_values = {
                        "id": chunk_id,
                        "artifact_id": elem.artifact_id,
                        "document_id": uuid.UUID(document_id),
                        "chunk_index": 0,
                        "chunk_text": elem.content_text or None,
                        "embedding": img_embedding,
                        "modality": "image",
                        "page_number": elem.page_number,
                        "bounding_box": elem.bounding_box,
                        "qdrant_point_id": qdrant_point_id,
                    }

                    stmt = pg_insert(ImageChunk).values(**chunk_values).on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "chunk_text": chunk_values["chunk_text"],
                            "embedding": chunk_values["embedding"],
                            "qdrant_point_id": qdrant_point_id,
                        },
                    )
                    db.execute(stmt)

                    # Upsert vector to Qdrant
                    upsert_image_vector(
                        qdrant,
                        point_id=qdrant_point_id,
                        vector=img_embedding,
                        payload={
                            "chunk_id": str(chunk_id),
                            "document_id": document_id,
                            "artifact_id": str(elem.artifact_id),
                            "modality": "image",
                            "page_number": elem.page_number,
                            "classification": "UNCLASSIFIED",
                            "chunk_text": elem.content_text or "",
                        },
                    )
                    chunks_created += 1

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_image_embeddings", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"chunks": chunks_created, "elements": len(elements)},
            )
            db.commit()

        logger.info(
            "derive_image_embeddings: document_id=%s chunks=%d",
            document_id, chunks_created,
        )
        return {"stage": "derive_image_embeddings", "status": "ok", "chunks": chunks_created}

    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("derive_image_embeddings failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_image_embeddings", error=str(exc),
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_image_embeddings", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("derive_image_embeddings: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


def _build_entity_mentions(
    entities: list[dict],
    elements: list,
    source_provider: str,
) -> list[dict]:
    """Build entity-to-element mentions via normalized text matching.

    Each entity dict must have 'name' and 'entity_type'.
    Short names (≤4 chars) use word-boundary matching to avoid false positives.
    Returns deduplicated list of mention dicts.
    """
    import re

    mentions: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for entity in entities:
        name = entity.get("name", "")
        entity_type = entity.get("entity_type", "UNKNOWN")
        if not name.strip():
            continue

        # Short names use word-boundary regex to avoid false positives
        # (e.g. "RAM" matching "program", "C4" matching "AC400")
        if len(name) <= 4:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            match_type = "word_boundary"
        else:
            pattern = None
            name_lower = name.lower()
            match_type = "substring"

        for elem in elements:
            content = getattr(elem, "content_text", None) or ""
            if not content:
                continue

            if pattern is not None:
                matched = pattern.search(content)
            else:
                matched = name_lower in content.lower()

            if matched:
                key = (name, entity_type, elem.element_uid)
                if key not in seen:
                    seen.add(key)
                    mentions.append({
                        "entity_name": name,
                        "entity_type": entity_type,
                        "element_uid": elem.element_uid,
                        "source_provider": source_provider,
                        "match_type": match_type,
                    })
    return mentions


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="graph",
                 soft_time_limit=600, time_limit=660)
def derive_ontology_graph(self, document_id: str, run_id: str | None = None) -> dict:
    """Read ordered text elements → LLM/NER extraction → upsert document_graph_extractions → import to Neo4j.

    Stores graph extraction once per document (not per artifact).
    """
    from app.models.ingest import DocumentElement, DocumentGraphExtraction
    from app.services.neo4j_graph import upsert_node, upsert_relationship
    from app.db.session import get_neo4j_driver
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    self.max_retries = settings.graph_max_retries
    self.default_retry_delay = settings.graph_retry_delay
    self.soft_time_limit = settings.graph_soft_time_limit
    self.time_limit = settings.graph_time_limit

    logger.info("derive_ontology_graph: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_ontology_graph")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_ontology_graph", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || '_ontology_graph'))"
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
                _update_stage_run(db, run_id, "derive_ontology_graph", "COMPLETE", attempt=self.request.retries + 1, metrics={"skipped": True})
                db.commit()
            return {"stage": "derive_ontology_graph", "status": "ok", "nodes": 0, "edges": 0}

        # Try docling-graph + LLM first, fallback to NER
        graph_data = None
        provider = "ner"
        model_name = "regex"

        if settings.llm_provider != "mock":
            try:
                from app.services.docling_graph_service import extract_graph_from_text_chunked
                nx_graph, actual_provider = extract_graph_from_text_chunked(
                    full_text, document_id,
                    chunk_size=settings.graph_extraction_chunk_size,
                    chunk_overlap=settings.graph_extraction_chunk_overlap,
                )
                if nx_graph and nx_graph.number_of_nodes() > 0:
                    graph_data = {
                        "nodes": [{"id": n, **nx_graph.nodes[n]} for n in nx_graph.nodes],
                        "edges": [
                            {
                                "from": nx_graph.nodes[u].get("name", str(u)),
                                "to": nx_graph.nodes[v].get("name", str(v)),
                                "from_type": nx_graph.nodes[u].get("entity_type", "UNKNOWN"),
                                "to_type": nx_graph.nodes[v].get("entity_type", "UNKNOWN"),
                                "rel_type": d.get("relationship_type", d.get("rel_type", "RELATED_TO")),
                                "confidence": d.get("confidence", 0.5),
                                "properties": d.get("properties", {}),
                            }
                            for u, v, d in nx_graph.edges(data=True)
                        ],
                    }
                    graph_data["mentions"] = _build_entity_mentions(
                        graph_data["nodes"], elements, actual_provider,
                    )
                    provider = actual_provider
                    model_name = settings.docling_graph_model if actual_provider == "docling-graph" else "regex"
            except ImportError:
                if settings.docling_graph_require_llm:
                    raise RuntimeError("docling-graph module not available and DOCLING_GRAPH_REQUIRE_LLM=true")
                logger.debug("docling-graph not available, falling back to NER")
            except Exception as exc:
                if settings.docling_graph_require_llm:
                    raise  # Fail-closed: do not silently fall back to NER
                logger.warning("docling-graph failed: %s — falling back to NER", exc)

        if graph_data is None:
            if settings.docling_graph_require_llm:
                raise RuntimeError(
                    "LLM graph extraction produced no results and DOCLING_GRAPH_REQUIRE_LLM=true — "
                    "refusing to fall back to NER"
                )
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

            graph_data["mentions"] = _build_entity_mentions(
                graph_data["nodes"], elements, "ner",
            )

        # Import into Neo4j with confidence quality gates
        neo4j_driver = get_neo4j_driver()
        nodes_created = 0
        nodes_rejected = 0
        edges_created = 0
        edges_rejected = 0

        node_min_conf = settings.graph_node_min_confidence
        rel_min_conf = settings.graph_rel_min_confidence
        accepted_nodes: set[str] = set()

        for node in graph_data.get("nodes", []):
            conf = node.get("confidence", 0.8)
            if conf < node_min_conf:
                nodes_rejected += 1
                continue
            node_name = node.get("name", node.get("id", ""))
            accepted_nodes.add(node_name)
            node_id = upsert_node(
                driver=neo4j_driver,
                entity_type=node.get("entity_type", "UNKNOWN"),
                name=node_name,
                artifact_id=document_id,
                confidence=conf,
                properties=node.get("properties", {}),
            )
            if node_id:
                nodes_created += 1

        for edge in graph_data.get("edges", []):
            conf = edge.get("confidence", 0.8)
            from_name = edge.get("from", "")
            to_name = edge.get("to", "")
            if conf < rel_min_conf or from_name not in accepted_nodes or to_name not in accepted_nodes:
                edges_rejected += 1
                continue
            ok = upsert_relationship(
                driver=neo4j_driver,
                from_name=from_name,
                from_type=edge.get("from_type", "UNKNOWN"),
                to_name=to_name,
                to_type=edge.get("to_type", "UNKNOWN"),
                rel_type=edge.get("rel_type", edge.get("type", "RELATED_TO")),
                artifact_id=document_id,
                confidence=conf,
                properties=edge.get("properties", {}),
            )
            if ok:
                edges_created += 1

        # Store filter metadata in graph_json for auditability
        graph_data["_ingest_filter"] = {
            "node_min_confidence": node_min_conf,
            "rel_min_confidence": rel_min_conf,
            "nodes_accepted": nodes_created,
            "nodes_rejected": nodes_rejected,
            "edges_accepted": edges_created,
            "edges_rejected": edges_rejected,
        }

        # Upsert into document_graph_extractions (includes filter metadata)
        extraction_values = {
            "document_id": uuid.UUID(document_id),
            "provider": provider,
            "model_name": model_name,
            "extraction_version": "1.0",
            "graph_json": graph_data,
            "status": "COMPLETE",
            "metrics": {
                "nodes": nodes_created, "edges": edges_created,
                "nodes_rejected": nodes_rejected, "edges_rejected": edges_rejected,
            },
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

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_ontology_graph", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={
                    "nodes": nodes_created, "edges": edges_created, "provider": provider,
                    "nodes_rejected": nodes_rejected, "edges_rejected": edges_rejected,
                },
            )
            db.commit()

        logger.info(
            "derive_ontology_graph: document_id=%s nodes=%d(%d rejected) edges=%d(%d rejected) provider=%s",
            document_id, nodes_created, nodes_rejected, edges_created, edges_rejected, provider,
        )
        return {"stage": "derive_ontology_graph", "status": "ok", "nodes": nodes_created, "edges": edges_created}

    except CeleryRetry:
        raise
    except Exception as exc:
        from app.services.docling_graph_service import DeterministicExtractionError

        logger.error("derive_ontology_graph failed for %s: %s", document_id, exc)
        db.rollback()

        # Deterministic errors (empty/non-JSON output) won't resolve on retry
        is_deterministic = isinstance(exc, DeterministicExtractionError)
        if is_deterministic or self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_ontology_graph", error=str(exc),
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_ontology_graph", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            if is_deterministic:
                logger.warning(
                    "derive_ontology_graph: deterministic failure for %s — skipping retries",
                    document_id,
                )
            raise
        logger.info("derive_ontology_graph: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=120, time_limit=180)
def derive_structure_links(self, document_id: str, run_id: str | None = None) -> dict:
    """Generate chunk_links and structural AGE edges.

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

    self.max_retries = settings.finalize_max_retries
    self.default_retry_delay = settings.finalize_retry_delay
    self.soft_time_limit = settings.finalize_soft_time_limit
    self.time_limit = settings.finalize_time_limit

    logger.info("derive_structure_links: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_structure_links")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_structure_links", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        db.execute(
            __import__("sqlalchemy").text(
                "SELECT pg_advisory_xact_lock(hashtext(:doc_id || '_structure_links'))"
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

        # Create Neo4j structural edges
        from app.services.neo4j_graph import (
            upsert_document_node,
            upsert_chunk_ref_node,
            create_structural_edge,
            create_entity_chunk_edge,
        )
        from app.db.session import get_neo4j_driver
        neo4j_driver = get_neo4j_driver()

        upsert_document_node(
            driver=neo4j_driver,
            document_id=document_id,
            title=doc.filename,
            properties={"source_id": str(doc.source_id)},
        )

        for tc in text_chunks:
            upsert_chunk_ref_node(neo4j_driver, str(tc.id), "text_chunk")
            create_structural_edge(neo4j_driver, document_id, str(tc.id), "CONTAINS_TEXT")

        for ic in image_chunks:
            upsert_chunk_ref_node(neo4j_driver, str(ic.id), "image_chunk")
            create_structural_edge(neo4j_driver, document_id, str(ic.id), "CONTAINS_IMAGE")

        for page_num, ics in page_image_map.items():
            tcs = page_text_map.get(page_num, [])
            for ic in ics:
                for tc in tcs:
                    create_structural_edge(neo4j_driver, str(tc.id), str(ic.id), "SAME_PAGE")

        # Entity-chunk EXTRACTED_FROM edges
        entity_links = 0

        # Build element_uid → chunk_ids map (via artifact_id)
        element_uid_chunk_map: dict[str, list[str]] = {}
        artifact_id_to_element_uid: dict[str, str] = {}
        for elem in elements:
            if elem.artifact_id and elem.element_uid:
                artifact_id_to_element_uid[str(elem.artifact_id)] = elem.element_uid
        for tc in text_chunks:
            if tc.artifact_id:
                euid = artifact_id_to_element_uid.get(str(tc.artifact_id))
                if euid:
                    element_uid_chunk_map.setdefault(euid, []).append(str(tc.id))

        # Try graph_json mentions path first (new pipeline)
        from app.models.ingest import DocumentGraphExtraction
        graph_extraction = db.execute(
            select(DocumentGraphExtraction).where(
                DocumentGraphExtraction.document_id == uuid.UUID(document_id),
            )
        ).scalars().first()

        used_mentions_path = False
        if graph_extraction and graph_extraction.graph_json:
            mentions = graph_extraction.graph_json.get("mentions", [])
            if mentions:
                used_mentions_path = True
                for mention in mentions:
                    name = mention.get("entity_name", "")
                    etype = mention.get("entity_type", "UNKNOWN")
                    euid = mention.get("element_uid", "")
                    chunk_ids = element_uid_chunk_map.get(euid, [])
                    for chunk_id in chunk_ids:
                        if create_entity_chunk_edge(neo4j_driver, name, etype, chunk_id):
                            entity_links += 1

        # Fallback: Artifact.content_metadata path (backward compat)
        if not used_mentions_path:
            artifact_chunk_map: dict[str, list[str]] = {}
            for tc in text_chunks:
                artifact_chunk_map.setdefault(str(tc.artifact_id), []).append(str(tc.id))

            artifacts_with_entities = db.execute(
                select(Artifact).where(
                    Artifact.document_id == uuid.UUID(document_id),
                    Artifact.content_metadata.isnot(None),
                )
            ).scalars().all()

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
                        if create_entity_chunk_edge(neo4j_driver, name, etype, chunk_id):
                            entity_links += 1

        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_structure_links", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={
                    "chunk_links": links_created,
                    "entity_links": entity_links,
                    "text_chunks": len(text_chunks),
                    "image_chunks": len(image_chunks),
                },
            )
            db.commit()

        logger.info(
            "derive_structure_links: document_id=%s chunk_links=%d entity_links=%d",
            document_id, links_created, entity_links,
        )
        return {"stage": "derive_structure_links", "status": "ok", "links": links_created}

    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("derive_structure_links failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            _update_document_status(
                document_id, STATUS_PARTIAL_COMPLETE,
                stage="derive_structure_links", error=str(exc),
            )
            if run_id:
                _update_stage_run(db, run_id, "derive_structure_links", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("derive_structure_links: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True)
def collect_derivations(self, derivation_results: list[dict], document_id: str, run_id: str | None = None) -> None:
    """Chord callback: aggregate derivation stage statuses."""
    try:
        logger.info(
            "collect_derivations: document_id=%s results=%s",
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
    except Exception as exc:
        logger.error("collect_derivations failed for %s: %s", document_id, exc)
        _update_document_status(
            document_id, STATUS_PARTIAL_COMPLETE,
            stage="collect_derivations", error=str(exc),
        )


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=120, time_limit=180)
def derive_canonicalization(self, document_id: str, run_id: str | None = None) -> dict:
    """Post-extraction entity canonicalization pass.

    Resolves entity aliases to canonical names via Neo4j fulltext search
    and creates HAS_ALIAS edges for discovered matches.
    """
    from app.db.session import get_neo4j_driver
    from app.services.canonicalization import canonicalize_document_entities

    self.max_retries = settings.finalize_max_retries
    self.default_retry_delay = settings.finalize_retry_delay
    self.soft_time_limit = settings.finalize_soft_time_limit
    self.time_limit = settings.finalize_time_limit

    logger.info("derive_canonicalization: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_canonicalization")

    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if run_id:
            _update_stage_run(db, run_id, "derive_canonicalization", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        neo4j_driver = get_neo4j_driver()
        stats = canonicalize_document_entities(neo4j_driver, document_id)

        if run_id:
            _update_stage_run(
                db, run_id, "derive_canonicalization", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics=stats,
            )
            db.commit()

        logger.info(
            "derive_canonicalization: document_id=%s resolved=%d/%d",
            document_id, stats["resolved"], stats["total"],
        )
        return {"stage": "derive_canonicalization", "status": "ok", **stats}

    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("derive_canonicalization failed for %s: %s", document_id, exc)
        db.rollback()
        if self.request.retries >= self.max_retries:
            if run_id:
                _update_stage_run(db, run_id, "derive_canonicalization", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise
        logger.info("derive_canonicalization: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, soft_time_limit=120, time_limit=180)
def finalize_document(self, document_id: str, run_id: str | None = None) -> None:
    """Mark pipeline COMPLETE if all required stages succeeded."""
    from app.models.ingest import PipelineRun, StageRun
    from sqlalchemy import select, update as sql_update
    import datetime

    self.max_retries = settings.finalize_max_retries
    self.default_retry_delay = settings.finalize_retry_delay
    self.soft_time_limit = settings.finalize_soft_time_limit
    self.time_limit = settings.finalize_time_limit

    logger.info("finalize_document: document_id=%s run_id=%s", document_id, run_id)
    db = _get_db()
    try:
        if not run_id:
            run_id = _get_pipeline_run_id(db, document_id)
        if not run_id:
            _update_document_status(document_id, STATUS_COMPLETE, stage=None)
            return

        # Check for failed, missing, or stuck stages
        REQUIRED_STAGES = {
            "prepare_document",
            "purge_document_derivations",
            "derive_text_embeddings",
            "derive_image_embeddings",
            "derive_ontology_graph",
            "derive_structure_links",
            "derive_canonicalization",
        }

        all_stages = db.execute(
            select(StageRun).where(StageRun.pipeline_run_id == uuid.UUID(run_id))
        ).scalars().all()
        stage_statuses = {s.stage_name: s.status for s in all_stages}

        failed = [n for n, s in stage_statuses.items() if s == "FAILED"]
        missing = REQUIRED_STAGES - set(stage_statuses.keys())
        stuck = [n for n, s in stage_statuses.items() if s in ("RUNNING", "PENDING")]

        if failed or missing or stuck:
            final_status = STATUS_PARTIAL_COMPLETE
            logger.warning(
                "finalize_document: document_id=%s failed=%s missing=%s stuck=%s",
                document_id, failed, list(missing), stuck,
            )
        else:
            # Check if any artifacts need human review
            from app.models.ingest import Artifact as _Artifact
            review_artifacts = db.execute(
                select(_Artifact).where(
                    _Artifact.document_id == uuid.UUID(document_id),
                    _Artifact.requires_human_review == True,  # noqa: E712
                )
            ).scalars().all()
            if review_artifacts:
                final_status = STATUS_PENDING_REVIEW
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
            "finalize_document: document_id=%s — pipeline %s",
            document_id, final_status,
        )
    except CeleryRetry:
        raise
    except Exception as exc:
        logger.error("finalize_document failed for %s: %s", document_id, exc)
        db.rollback()
        # Ensure PipelineRun doesn't get stuck in PROCESSING
        if run_id:
            _update_stage_run(db, run_id, "finalize_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
            db.commit()
        _update_document_status(document_id, STATUS_PARTIAL_COMPLETE, stage="finalize_document", error=str(exc))
    finally:
        db.close()


