"""Multi-modal ingest pipeline.

Task graph (manifest-first, parallel derivations, idempotent):

    prepare_document  (validate + detect + Docling convert + persist document_elements)
        ↓
    derive_document_metadata  (LLM: summary, date, classification, source)
        ↓
    derive_picture_descriptions  (LLM: image descriptions with summary context)
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
from celery.exceptions import Retry as CeleryRetry, SoftTimeLimitExceeded
from celery.signals import worker_ready

from app.workers.celery_app import celery_app
from app.workers._db import get_worker_db as _get_db
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


@worker_ready.connect
def _cleanup_stale_runs(sender, **kwargs):
    """Reset documents stuck in PROCESSING from prior worker crashes."""
    from app.db.session import get_sync_session
    from sqlalchemy import text

    db = get_sync_session()
    try:
        result = db.execute(text("""
            UPDATE ingest.documents
            SET pipeline_status = 'PENDING'
            WHERE pipeline_status = 'PROCESSING'
            RETURNING id
        """))
        stale_ids = [str(r[0]) for r in result.fetchall()]

        db.execute(text("""
            UPDATE ingest.stage_runs
            SET status = 'PENDING'
            WHERE status = 'RUNNING'
        """))

        # Mark stale pipeline_runs as FAILED so queued tasks from those runs
        # will be caught by the supersession guard in prepare_document.
        db.execute(text("""
            UPDATE ingest.pipeline_runs
            SET status = 'FAILED', finished_at = NOW()
            WHERE status = 'PROCESSING'
        """))

        db.commit()
        if stale_ids:
            # Also clear Redis singleflight locks for stale documents
            for stale_id in stale_ids:
                _redis_client.delete(f"prepare:{stale_id}")
            logger.info("Cleaned up %d stale PROCESSING documents (+ Redis locks): %s", len(stale_ids), stale_ids)

        # Clear stale Docling concurrency permits — these are Redis locks that
        # may be held by a previous worker that died mid-conversion.
        docling_permits_cleared = 0
        for i in range(settings.docling_concurrency):
            key = f"docling:permit:{i}"
            if _redis_client.delete(key):
                docling_permits_cleared += 1
        if docling_permits_cleared:
            logger.info("Cleared %d stale Docling concurrency permits", docling_permits_cleared)
    except Exception as e:
        logger.warning("Stale document cleanup failed: %s", e)
        db.rollback()
    finally:
        db.close()


def _dedupe_extracted_elements(chunks: list) -> tuple[list, int]:
    """Remove exact duplicate extracted elements conservatively.

    Dedup key: (modality, page_number, section_path, content_text, bounding_box).
    Preserves first-occurrence order. Keeps duplicates across different pages/sections.
    """
    seen: set[str] = set()
    result = []
    for chunk in chunks:
        section_path = (getattr(chunk, "metadata", None) or {}).get("section_path", "")
        key = f"{chunk.modality}|{chunk.page_number}|{section_path}|{chunk.chunk_text}|{chunk.bounding_box}"
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result, len(chunks) - len(result)



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

    Uses ON CONFLICT DO UPDATE so reingest/retry with the same deterministic
    IDs is idempotent (updates mutable fields, preserves classification).

    Returns the list of artifact IDs (in chunk order).
    """
    import uuid as uuid_mod
    from sqlalchemy import func as sa_func
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.ingest import Artifact
    from app.services.storage import upload_bytes_sync

    artifact_ids: list[uuid.UUID] = []
    for idx, chunk in enumerate(chunks):
        # Compute artifact_id first so image storage keys are deterministic
        artifact_id = (
            _deterministic_artifact_id(document_id, element_uids[idx])
            if element_uids
            else uuid_mod.uuid4()
        )

        storage_bucket = None
        storage_key = None

        if chunk.raw_image_bytes:
            ext = chunk.metadata.get("ext", "png")
            img_key = f"artifacts/{document_id}/images/{artifact_id}.{ext}"
            upload_bytes_sync(
                chunk.raw_image_bytes,
                settings.minio_bucket_derived,
                img_key,
                content_type=f"image/{ext}",
            )
            storage_bucket = settings.minio_bucket_derived
            storage_key = img_key

        values = {
            "id": artifact_id,
            "document_id": uuid.UUID(document_id),
            "artifact_type": chunk.modality,
            "content_text": chunk.chunk_text,
            "content_metadata": chunk.metadata,
            "storage_bucket": storage_bucket,
            "storage_key": storage_key,
            "page_number": chunk.page_number,
            "bounding_box": chunk.bounding_box,
            "ocr_confidence": chunk.ocr_confidence,
            "ocr_engine": chunk.ocr_engine,
            "requires_human_review": chunk.requires_human_review,
        }

        stmt = pg_insert(Artifact).values(**values)
        stmt = stmt.on_conflict_do_update(
            constraint="artifacts_pkey",
            set_={
                "artifact_type": stmt.excluded.artifact_type,
                "content_text": stmt.excluded.content_text,
                "content_metadata": stmt.excluded.content_metadata,
                "storage_bucket": stmt.excluded.storage_bucket,
                "storage_key": stmt.excluded.storage_key,
                "page_number": stmt.excluded.page_number,
                "bounding_box": stmt.excluded.bounding_box,
                "ocr_confidence": stmt.excluded.ocr_confidence,
                "ocr_engine": stmt.excluded.ocr_engine,
                "requires_human_review": stmt.excluded.requires_human_review,
                "updated_at": sa_func.now(),
            },
        )
        db.execute(stmt)
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
    elif "wordprocessingml" in mime or "msword" in mime:
        chunks = extract_docx(file_bytes)
    elif "image" in mime:
        chunks = extract_image(file_bytes)
    elif "text" in mime:
        chunks = extract_txt(file_bytes)
    # Note: PPTX, XLSX, HTML, MD now route to Docling; legacy fallback
    # only handles formats above. Unknown formats produce empty chunks.

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


@celery_app.task(bind=True)
def _chord_error_handler(self, request, exc, traceback, document_id: str, run_id: str | None = None):
    """Errback for chord failures (e.g. hard time limit kills a chord member)."""
    logger.error("Chord failed for document %s: %s", document_id, exc)
    _update_document_status(
        document_id, STATUS_FAILED,
        stage="chord_error", error=str(exc),
    )
    if run_id:
        db = _get_db()
        try:
            from app.models.ingest import PipelineRun
            from sqlalchemy import update as sql_update
            import datetime
            db.execute(
                sql_update(PipelineRun)
                .where(PipelineRun.id == uuid.UUID(run_id))
                .values(status="FAILED", finished_at=datetime.datetime.now(datetime.timezone.utc))
            )
            db.commit()
        except Exception as e:
            logger.warning("_chord_error_handler: failed to update pipeline run %s: %s", run_id, e)
            db.rollback()
        finally:
            db.close()


def start_ingest_pipeline(document_id: str) -> str:
    """Enqueue the ingest pipeline for a document. Returns Celery task ID."""
    from app.models.ingest import PipelineRun
    from sqlalchemy import select

    db = _get_db()
    try:
        # Atomic check: prevent duplicate dispatch if a run is already active
        active = db.execute(
            select(PipelineRun.id)
            .where(
                PipelineRun.document_id == uuid.UUID(document_id),
                PipelineRun.status == "PROCESSING",
            )
            .with_for_update()
            .limit(1)
        ).scalar_one_or_none()

        if active:
            logger.warning(
                "start_ingest_pipeline: skipping document %s — active run %s exists",
                document_id, active,
            )
            db.commit()  # release FOR UPDATE lock
            return str(active)

        run_id = _create_pipeline_run(db, document_id)
        db.commit()
    finally:
        db.close()

    errback = _chord_error_handler.s(document_id, run_id)

    pipeline = chain(
        prepare_document.si(document_id, run_id),
        # Metadata extraction and purge run in parallel — purge only touches
        # derived data (chunks, vectors, graph), not document metadata.
        chord(
            group(
                derive_document_metadata.si(document_id, run_id),
                purge_document_derivations.si(document_id, run_id),
            ),
            # Pictures needs the summary from metadata, so it runs after both complete.
            derive_picture_descriptions.si(document_id, run_id),
        ),
        chord(
            group(
                derive_text_chunks_and_embeddings.si(document_id, run_id),
                derive_image_embeddings.si(document_id, run_id),
                derive_ontology_graph.si(document_id, run_id),
            ),
            collect_derivations.s(document_id, run_id),
        ).on_error(errback),
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
                 soft_time_limit=settings.prepare_soft_time_limit,
                 time_limit=settings.prepare_time_limit)
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

    # Singleflight lock: prevent concurrent prepare_document for same document
    _singleflight_lock = _redis_client.lock(
        f"prepare:{document_id}",
        timeout=settings.prepare_singleflight_timeout,
        blocking=False,
    )
    if not _singleflight_lock.acquire(blocking=False):
        # Lock held — check if there's genuinely an active PROCESSING run.
        # If not, the lock is stale (orphaned by a crashed/timed-out task).
        _check_db = _get_db()
        try:
            from app.models.ingest import PipelineRun as _PR
            from sqlalchemy import select as _sa_select
            _active_run = _check_db.execute(
                _sa_select(_PR.id).where(
                    _PR.document_id == uuid.UUID(document_id),
                    _PR.status == "PROCESSING",
                ).limit(1)
            ).scalar_one_or_none()
        finally:
            _check_db.close()

        if _active_run:
            logger.warning(
                "prepare_document: singleflight lock held for %s (active run %s) — aborting",
                document_id, _active_run,
            )
            return document_id

        # Lock is stale — force-delete and re-acquire
        logger.warning(
            "prepare_document: stale singleflight lock for %s — no active run, force-releasing",
            document_id,
        )
        _redis_client.delete(f"prepare:{document_id}")
        if not _singleflight_lock.acquire(blocking=False):
            logger.error(
                "prepare_document: failed to acquire lock for %s even after force-release",
                document_id,
            )
            return document_id

    _update_document_status(document_id, STATUS_PROCESSING, stage="prepare_document")

    db = _get_db()
    try:
        # Use passed run_id or create one (backward compat)
        if not run_id:
            run_id = _create_pipeline_run(db, document_id)
            db.commit()

        # Supersession guard: bail if a newer pipeline run exists
        from app.models.ingest import PipelineRun
        from sqlalchemy import select as sa_select
        latest_active = db.execute(
            sa_select(PipelineRun.id)
            .where(
                PipelineRun.document_id == uuid.UUID(document_id),
                PipelineRun.status == "PROCESSING",
            )
            .order_by(PipelineRun.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_active and str(latest_active) != run_id:
            logger.warning(
                "prepare_document: run %s superseded by %s for document %s — aborting",
                run_id, latest_active, document_id,
            )
            _update_stage_run(db, run_id, "prepare_document", "FAILED",
                              attempt=self.request.retries + 1, error="superseded")
            db.commit()
            try:
                _singleflight_lock.release()
            except Exception:
                pass
            return document_id

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

        # 3. Route by format — Docling handles PDF, images, office docs, and markup
        _DOCLING_MIMES = {
            # PDF
            "application/pdf",
            # Images
            "image/png", "image/jpeg", "image/tiff", "image/bmp", "image/gif", "image/webp",
            # Office documents
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
            "application/msword",  # legacy DOC
            "application/vnd.ms-powerpoint",  # legacy PPT
            "application/vnd.ms-excel",  # legacy XLS
            # Markup / text
            "text/html",
            "text/markdown",
            "text/csv",
            "text/asciidoc",
        }
        if mime_type not in _DOCLING_MIMES:
            logger.info("prepare_document: %s not supported by Docling (mime=%s), using legacy extraction", document_id, mime_type)
            _legacy_extract(db, document_id, doc, file_bytes)
            db.commit()

            # Persist extracted text as markdown so derive_document_metadata can run
            try:
                from app.models.ingest import DocumentElement
                from sqlalchemy import select as sql_select
                elems = db.execute(
                    sql_select(DocumentElement.content_text)
                    .where(DocumentElement.document_id == uuid.UUID(document_id))
                    .order_by(DocumentElement.element_order)
                ).scalars().all()
                fallback_md = "\n\n".join(t for t in elems if t and t.strip())
                if fallback_md:
                    from app.services.storage import upload_bytes_sync
                    _fb_base = f"artifacts/{document_id}"
                    upload_bytes_sync(
                        fallback_md.encode("utf-8"),
                        settings.minio_bucket_derived,
                        f"{_fb_base}/docling_document.md",
                        content_type="text/markdown; charset=utf-8",
                    )
                    logger.info("prepare_document: persisted legacy markdown for %s (%d chars)", document_id, len(fallback_md))
            except Exception as _fb_err:
                logger.warning("prepare_document: failed to persist legacy markdown for %s: %s", document_id, _fb_err)

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

            # In-task retry loop for 503 (Docling busy with ghost request from
            # a previous timed-out attempt).  Sleeps and retries WITHOUT consuming
            # the Celery retry budget.  The Redis lock stays held during this loop
            # so no other tasks attempt to send to Docling concurrently.
            _max_503_retries = settings.docling_503_max_retries
            for _503_attempt in range(_max_503_retries):
                try:
                    result = convert_document_sync(file_bytes, doc.filename or "document")
                    break  # success
                except httpx.HTTPStatusError as _docling_exc:
                    if _docling_exc.response.status_code != 503 or _503_attempt >= _max_503_retries - 1:
                        raise
                    _wait = min(120, 30 * (_503_attempt + 1))  # 30s, 60s, 90s, 120s cap
                    logger.info(
                        "prepare_document: Docling 503 for %s — in-task wait %ds (%d/%d)",
                        document_id, _wait, _503_attempt + 1, _max_503_retries,
                    )
                    _time.sleep(_wait)
            _docling_convert_ok = True
        except Exception:
            # If Celery retries remain, keep the Docling lock held to prevent
            # another task from grabbing it in the gap. Lock TTL auto-expires.
            _docling_convert_ok = False
            if self.request.retries < self.max_retries:
                logger.info(
                    "prepare_document: keeping Docling lock for %s (retries remain, TTL will expire)",
                    document_id,
                )
            raise
        finally:
            if _docling_convert_ok or self.request.retries >= self.max_retries:
                try:
                    docling_lock.release()
                except redis_lib.exceptions.LockNotOwnedError:
                    logger.warning("prepare_document: Docling lock expired before release for %s", document_id)
        logger.info(
            "prepare_document: docling returned %d elements, %d pages, %.0fms",
            len(result.elements), result.num_pages, result.processing_time_ms,
        )

        # 4. Deduplicate extracted elements (conservative: same modality+page+section+text+bbox)
        result.elements, _dups_dropped = _dedupe_extracted_elements(result.elements)
        if _dups_dropped:
            logger.info(
                "prepare_document: %d elements after dedup (%d duplicates dropped) for %s",
                len(result.elements), _dups_dropped, document_id,
            )

        # 5. Build element_uids, then persist Artifacts with deterministic IDs
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

        # Build a lookup of image storage keys from Artifact uploads to avoid re-uploading
        _image_storage: dict[int, tuple[str, str]] = {}
        for idx, chunk in enumerate(result.elements):
            if chunk.raw_image_bytes:
                # The Artifact was already uploaded in _persist_extraction_results;
                # query its storage_key to reuse for the DocumentElement row.
                art = db.get(Artifact, artifact_ids[idx])
                if art and art.storage_key:
                    _image_storage[idx] = (art.storage_bucket, art.storage_key)

        # 6. Persist canonical DocumentElement rows with artifact_id linked inline
        elements_created = 0
        for idx, chunk in enumerate(result.elements):
            element_uid = element_uids[idx]

            # Reuse image storage from Artifact upload (no duplicate MinIO I/O)
            if idx in _image_storage:
                storage_bucket, storage_key = _image_storage[idx]
            else:
                storage_bucket = None
                storage_key = None

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

        # Persist DoclingDocument markdown and JSON to MinIO for the viewer
        try:
            from app.services.storage import upload_bytes_sync
            _docling_base = f"artifacts/{document_id}"
            if result.markdown:
                upload_bytes_sync(
                    result.markdown.encode("utf-8"),
                    settings.minio_bucket_derived,
                    f"{_docling_base}/docling_document.md",
                    content_type="text/markdown; charset=utf-8",
                )
            if getattr(result, "document_json", None):
                import json as _json
                upload_bytes_sync(
                    _json.dumps(result.document_json, ensure_ascii=False, default=str).encode("utf-8"),
                    settings.minio_bucket_derived,
                    f"{_docling_base}/docling_document.json",
                    content_type="application/json; charset=utf-8",
                )
                logger.info("prepare_document: persisted DoclingDocument md+json for %s", document_id)
        except Exception as _doc_err:
            logger.warning("prepare_document: failed to persist DoclingDocument for %s: %s", document_id, _doc_err)

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
    except SoftTimeLimitExceeded as exc:
        # Task killed mid-wait or mid-conversion — Docling may still be working on
        # this or another document.  Re-queue without consuming the retry budget.
        logger.warning(
            "prepare_document: soft time limit for %s — re-queuing in 180s (attempt %d, not counted)",
            document_id, self.request.retries + 1,
        )
        db.rollback()
        raise self.retry(exc=exc, countdown=180)
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

        # Artifact PK collision — deterministic IDs collided with existing rows.
        # This shouldn't happen after the upsert fix, but if it does, retrying
        # will produce the same collision.  Fail immediately.
        from sqlalchemy.exc import IntegrityError as _IntegrityError
        if isinstance(exc, _IntegrityError) and "artifacts_pkey" in str(exc):
            logger.error(
                "prepare_document: artifact PK collision for %s — failing without retry: %s",
                document_id, exc,
            )
            _update_document_status(
                document_id, STATUS_FAILED, stage="prepare_document", error=str(exc)
            )
            run_id_for_err = run_id or _get_pipeline_run_id(db, document_id)
            if run_id_for_err:
                _update_stage_run(db, run_id_for_err, "prepare_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise

        # Deterministic Docling errors (VlmPipeline failed, unsupported format)
        # won't resolve on retry — fail immediately.
        _deterministic_markers = ("VlmPipeline failed", "unsupported format", "invalid PDF")
        if isinstance(exc, RuntimeError) and any(m in str(exc) for m in _deterministic_markers):
            logger.warning(
                "prepare_document: deterministic Docling failure for %s — skipping retries: %s",
                document_id, exc,
            )
            _update_document_status(
                document_id, STATUS_FAILED, stage="prepare_document", error=str(exc)
            )
            run_id_for_err = run_id or _get_pipeline_run_id(db, document_id)
            if run_id_for_err:
                _update_stage_run(db, run_id_for_err, "prepare_document", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            raise

        # 503 is now handled by the in-task retry loop above; if we still
        # reach here with a 503 it means all in-task retries were exhausted —
        # treat it as a normal error and let the Celery retry budget handle it.
        countdown = settings.prepare_retry_delay

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
        raise self.retry(exc=exc, countdown=countdown)
    finally:
        db.close()
        # Release singleflight lock (safe if already released or expired)
        try:
            _singleflight_lock.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage: derive_document_metadata — LLM metadata extraction
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.pipeline.derive_document_metadata",
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=settings.doc_analysis_timeout + 60,
    time_limit=settings.doc_analysis_timeout + 120,
    queue="ingest",
)
def derive_document_metadata(self, document_id: str, run_id: str | None = None) -> dict:
    """Extract document metadata (summary, date, classification, source) via LLM."""
    import json as json_mod

    logger.info("derive_document_metadata: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_document_metadata")

    if not settings.doc_analysis_enabled:
        logger.info("derive_document_metadata: disabled, skipping for %s", document_id)
        return {"stage": "derive_document_metadata", "status": "skipped"}

    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "derive_document_metadata", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Load markdown from MinIO
        from app.services.storage import download_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived
        try:
            md_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.md")
            markdown = md_bytes.decode("utf-8")
        except Exception:
            logger.info("derive_document_metadata: no markdown available for %s, skipping", document_id)
            return {"stage": "derive_document_metadata", "status": "skipped", "reason": "no_markdown"}

        # Extract metadata via LLM
        from app.services.document_analysis import extract_document_metadata
        metadata = extract_document_metadata(markdown)

        # Store in documents.document_metadata
        from sqlalchemy import text
        db.execute(
            text("UPDATE ingest.documents SET document_metadata = cast(:meta AS jsonb) WHERE id = cast(:doc_id AS uuid)"),
            {"meta": json_mod.dumps(metadata), "doc_id": document_id},
        )
        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_document_metadata", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"summary_length": len(metadata.get("document_summary", ""))},
            )
            db.commit()

        logger.info(
            "derive_document_metadata: document_id=%s classification=%s",
            document_id, metadata.get("classification"),
        )
        return {"stage": "derive_document_metadata", "status": "ok"}

    except Exception as exc:
        logger.error("derive_document_metadata failed for %s: %s", document_id, exc)
        if run_id:
            try:
                _update_stage_run(db, run_id, "derive_document_metadata", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            except Exception:
                pass
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Stage: derive_picture_descriptions — LLM image enrichment
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="app.workers.pipeline.derive_picture_descriptions",
    max_retries=1,
    default_retry_delay=30,
    soft_time_limit=1800,
    time_limit=1860,
    queue="ingest",
)
def derive_picture_descriptions(self, document_id: str, run_id: str | None = None) -> dict:
    """Enrich picture items with LLM-generated descriptions using document summary context."""
    import json as json_mod

    logger.info("derive_picture_descriptions: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_picture_descriptions")

    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "derive_picture_descriptions", "RUNNING", attempt=self.request.retries + 1)
            db.commit()

        # Load document metadata for summary
        from sqlalchemy import text as sa_text
        row = db.execute(
            sa_text("SELECT document_metadata FROM ingest.documents WHERE id = cast(:doc_id AS uuid)"),
            {"doc_id": document_id},
        ).first()
        document_summary = ""
        if row and row[0]:
            meta = row[0] if isinstance(row[0], dict) else json_mod.loads(row[0])
            document_summary = meta.get("document_summary", "")

        # Load Docling JSON from MinIO
        from app.services.storage import download_bytes_sync, upload_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived
        try:
            json_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.json")
            docling_json = json_mod.loads(json_bytes)
        except Exception:
            logger.info("derive_picture_descriptions: no Docling JSON for %s, skipping", document_id)
            return {"stage": "derive_picture_descriptions", "status": "skipped"}

        # For Office formats (DOCX/PPTX), Docling's SimplePipeline doesn't
        # extract actual image bytes. Use python-docx/python-pptx to extract
        # them and inject into the Docling JSON before describing.
        from app.models.ingest import Document as DocModel
        doc = db.get(DocModel, uuid.UUID(document_id))
        if doc:
            mime = doc.mime_type or ""
            if "wordprocessingml" in mime or "msword" in mime:
                from app.services.office_image_extractor import extract_docx_images, inject_images_into_docling_json
                file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
                office_images = extract_docx_images(file_bytes)
                if office_images:
                    inject_images_into_docling_json(docling_json, office_images)
            elif "presentationml" in mime or "ms-powerpoint" in mime:
                from app.services.office_image_extractor import extract_pptx_images, inject_images_into_docling_json
                file_bytes = download_bytes_sync(doc.storage_bucket, doc.storage_key)
                office_images = extract_pptx_images(file_bytes)
                if office_images:
                    inject_images_into_docling_json(docling_json, office_images)

        # Enrich pictures with descriptions
        from app.services.document_analysis import describe_pictures
        updated_json = describe_pictures(docling_json, document_summary)

        # Write updated JSON back to MinIO
        upload_bytes_sync(
            json_mod.dumps(updated_json, ensure_ascii=False, default=str).encode("utf-8"),
            bucket,
            f"{base_key}/docling_document.json",
            content_type="application/json; charset=utf-8",
        )

        # Persist picture descriptions to DocumentElement rows so downstream
        # tasks (text chunking, graph extraction) can see them
        from app.models.ingest import DocumentElement
        from sqlalchemy import select as sa_select
        pictures_updated = 0
        pic_elements = db.execute(
            sa_select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type == "image",
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        described_pics = [
            p for p in updated_json.get("pictures", [])
            if isinstance(p, dict) and p.get("description")
        ]
        # Match by order — first image element gets first described picture
        for elem, pic in zip(pic_elements, described_pics):
            desc = pic["description"]
            if desc and desc != elem.content_text:
                elem.content_text = desc
                pictures_updated += 1
        if pictures_updated:
            db.commit()
            logger.info("derive_picture_descriptions: updated %d DocumentElement rows", pictures_updated)

        # Also update the markdown in MinIO to include picture descriptions
        # so derive_document_metadata and graph extraction see enriched content
        try:
            md_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.md")
            markdown = md_bytes.decode("utf-8")
            appendix_parts = []
            for pic in described_pics:
                desc = pic.get("description", "")
                if desc:
                    appendix_parts.append(f"[Image Description]: {desc}")
            if appendix_parts:
                enriched_md = markdown + "\n\n## Image Descriptions\n\n" + "\n\n".join(appendix_parts)
                upload_bytes_sync(
                    enriched_md.encode("utf-8"),
                    bucket,
                    f"{base_key}/docling_document.md",
                    content_type="text/markdown; charset=utf-8",
                )
        except Exception as md_err:
            logger.debug("derive_picture_descriptions: could not update markdown: %s", md_err)

        if run_id:
            _update_stage_run(
                db, run_id, "derive_picture_descriptions", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"pictures_updated": pictures_updated},
            )
            db.commit()

        logger.info("derive_picture_descriptions: document_id=%s updated=%d", document_id, pictures_updated)
        return {"stage": "derive_picture_descriptions", "status": "ok", "pictures_updated": pictures_updated}

    except Exception as exc:
        logger.error("derive_picture_descriptions failed for %s: %s", document_id, exc)
        if run_id:
            try:
                _update_stage_run(db, run_id, "derive_picture_descriptions", "FAILED", attempt=self.request.retries + 1, error=str(exc))
                db.commit()
            except Exception:
                pass
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit, queue="ingest")
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
                 soft_time_limit=settings.embed_soft_time_limit,
                 time_limit=settings.embed_time_limit)
def derive_text_chunks_and_embeddings(self, document_id: str, run_id: str | None = None) -> dict:
    """Read text/table/heading document_elements → chunk → BGE embed → upsert text_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    from app.models.ingest import Document, DocumentElement
    from app.models.retrieval import TextChunk
    from app.services.chunking import structure_aware_chunk
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

        # Resolve classification from document metadata (fallback: UNCLASSIFIED)
        doc_obj = db.get(Document, uuid.UUID(document_id))
        doc_classification = "UNCLASSIFIED"
        if doc_obj and doc_obj.document_metadata:
            doc_classification = doc_obj.document_metadata.get("classification", "UNCLASSIFIED")

        elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type.in_(["text", "table", "heading", "equation", "schematic"]),
                DocumentElement.content_text.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        # Convert ORM objects to dicts for structure-aware chunker
        element_dicts = [
            {
                "element_type": elem.element_type,
                "content_text": elem.content_text,
                "page_number": elem.page_number,
                "section_path": elem.section_path,
                "element_uid": str(elem.element_uid) if elem.element_uid else "",
                "element_order": elem.element_order,
                "heading_level": elem.heading_level,
            }
            for elem in elements
            if elem.content_text
        ]
        structured_chunks = structure_aware_chunk(element_dicts)

        # Build a lookup from element_uid to the ORM element for artifact_id / bounding_box
        elem_by_uid: dict[str, DocumentElement] = {
            str(e.element_uid): e for e in elements if e.element_uid
        }

        all_texts = []
        all_chunk_refs = []
        _seen_chunk_texts: set[str] = set()

        for sc in structured_chunks:
            if sc.text in _seen_chunk_texts:
                continue
            _seen_chunk_texts.add(sc.text)
            all_texts.append(sc.text)
            all_chunk_refs.append(sc)

        chunks_created = 0
        if all_texts:
            # Batch embedding to limit memory for very large documents
            _embed_batch = settings.embed_text_batch_size
            embeddings: list[list[float]] = []
            for _eb_start in range(0, len(all_texts), _embed_batch):
                embeddings.extend(embed_texts(all_texts[_eb_start:_eb_start + _embed_batch]))
            model_version = settings.text_embedding_model

            # Get Qdrant client for batch vector upsert
            from app.db.session import get_qdrant_client
            from app.services.qdrant_store import upsert_text_vectors_batch
            from qdrant_client.models import PointStruct
            qdrant = get_qdrant_client()

            qdrant_points: list[PointStruct] = []
            for sc, text, embedding in zip(all_chunk_refs, all_texts, embeddings):
                # Resolve artifact_id from the first element_uid in this chunk
                first_uid = sc.element_uids[0] if sc.element_uids else ""
                ref_elem = elem_by_uid.get(first_uid)
                artifact_id = ref_elem.artifact_id if ref_elem else None
                bounding_box = ref_elem.bounding_box if ref_elem else None

                # Deterministic chunk key using element_uids for stability
                uid_key = "|".join(sc.element_uids)
                chunk_key = hashlib.sha256(
                    f"{document_id}:{uid_key}:{sc.chunk_index}:{model_version}".encode()
                ).hexdigest()

                chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())
                qdrant_point_id = chunk_id  # Use same UUID for Qdrant

                chunk_values = {
                    "id": chunk_id,
                    "artifact_id": artifact_id,
                    "document_id": uuid.UUID(document_id),
                    "chunk_index": sc.chunk_index,
                    "chunk_text": text,
                    "embedding": embedding,
                    "modality": sc.modality,
                    "page_number": sc.page_number,
                    "bounding_box": bounding_box,
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

                qdrant_points.append(PointStruct(
                    id=str(qdrant_point_id),
                    vector=embedding,
                    payload={
                        "chunk_id": str(chunk_id),
                        "document_id": document_id,
                        "artifact_id": str(artifact_id) if artifact_id else None,
                        "modality": sc.modality,
                        "page_number": sc.page_number,
                        "classification": doc_classification,
                        "chunk_text": text,
                    },
                ))
                chunks_created += 1

            # Batch upsert text vectors in bounded Qdrant RPCs
            if qdrant_points:
                _upsert_batch = settings.qdrant_upsert_batch_size
                for _qb_start in range(0, len(qdrant_points), _upsert_batch):
                    upsert_text_vectors_batch(qdrant, qdrant_points[_qb_start:_qb_start + _upsert_batch])

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
    except SoftTimeLimitExceeded:
        logger.warning("derive_text_chunks_and_embeddings: soft time limit for %s", document_id)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        return {"stage": "derive_text_embeddings", "status": "failed", "error": "soft time limit exceeded"}
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
            return {"stage": "derive_text_embeddings", "status": "failed", "error": str(exc)}
        if run_id:
            _update_stage_run(db, run_id, "derive_text_embeddings", "FAILED", attempt=self.request.retries + 1, error=str(exc))
            db.commit()
        logger.info("derive_text_chunks_and_embeddings: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed",
                 soft_time_limit=settings.embed_soft_time_limit,
                 time_limit=settings.embed_time_limit)
def derive_image_embeddings(self, document_id: str, run_id: str | None = None) -> dict:
    """Read image document_elements → CLIP embed → upsert image_chunks.

    Uses deterministic chunk keys for idempotent retries.
    """
    import io
    from app.models.ingest import Document, DocumentElement
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

        # Resolve classification from document metadata (fallback: UNCLASSIFIED)
        doc_obj = db.get(Document, uuid.UUID(document_id))
        doc_classification = "UNCLASSIFIED"
        if doc_obj and doc_obj.document_metadata:
            doc_classification = doc_obj.document_metadata.get("classification", "UNCLASSIFIED")

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

                # Get Qdrant client for batch vector upsert
                from app.db.session import get_qdrant_client
                from app.services.qdrant_store import upsert_image_vectors_batch
                from qdrant_client.models import PointStruct
                qdrant = get_qdrant_client()

                qdrant_points: list[PointStruct] = []
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

                    qdrant_points.append(PointStruct(
                        id=str(qdrant_point_id),
                        vector=img_embedding,
                        payload={
                            "chunk_id": str(chunk_id),
                            "document_id": document_id,
                            "artifact_id": str(elem.artifact_id),
                            "modality": "image",
                            "page_number": elem.page_number,
                            "classification": doc_classification,
                            "chunk_text": elem.content_text or "",
                        },
                    ))
                    chunks_created += 1

                # Batch upsert all image vectors in one Qdrant RPC
                if qdrant_points:
                    upsert_image_vectors_batch(qdrant, qdrant_points)

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
    except SoftTimeLimitExceeded:
        logger.warning("derive_image_embeddings: soft time limit for %s", document_id)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "derive_image_embeddings", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        return {"stage": "derive_image_embeddings", "status": "failed", "error": "soft time limit exceeded"}
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
            return {"stage": "derive_image_embeddings", "status": "failed", "error": str(exc)}
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
                 soft_time_limit=settings.graph_soft_time_limit,
                 time_limit=settings.graph_time_limit)
def derive_ontology_graph(self, document_id: str, run_id: str | None = None) -> dict:
    """Read ordered text elements → Docling-Graph service extraction → upsert document_graph_extractions → import to Neo4j.

    Stores graph extraction once per document (not per artifact).
    """
    from app.models.ingest import DocumentElement, DocumentGraphExtraction
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
                DocumentElement.element_type.in_(["text", "table", "heading", "equation", "schematic", "image"]),
                DocumentElement.content_text.isnot(None),
            ).order_by(DocumentElement.element_order)
        ).scalars().all()

        full_text = "\n\n".join(e.content_text for e in elements if e.content_text)

        # Prepend document metadata (summary, classification) for richer graph context
        from sqlalchemy import text as sa_text
        meta_row = db.execute(
            sa_text("SELECT document_metadata FROM ingest.documents WHERE id = cast(:doc_id AS uuid)"),
            {"doc_id": document_id},
        ).first()
        if meta_row and meta_row[0]:
            import json as _json_mod
            meta = meta_row[0] if isinstance(meta_row[0], dict) else _json_mod.loads(meta_row[0])
            doc_summary = meta.get("document_summary", "")
            doc_class = meta.get("classification", "")
            if doc_summary:
                full_text = f"[Document Summary]: {doc_summary}\n[Classification]: {doc_class}\n\n{full_text}"

        if not full_text.strip():
            logger.info("derive_ontology_graph: no text elements for %s", document_id)
            if run_id:
                _update_stage_run(db, run_id, "derive_ontology_graph", "COMPLETE", attempt=self.request.retries + 1, metrics={"skipped": True})
                db.commit()
            return {"stage": "derive_ontology_graph", "status": "ok", "nodes": 0, "edges": 0}

        # ---- Full extraction: all 5 groups in parallel + relationships ----
        from app.services.docling_graph_service import extract_graph_all

        provider = "docling-graph"
        model_name = "unknown"
        group_errors: list[str] = []

        try:
            result = extract_graph_all(full_text, document_id)
            all_entities = result.get("entities", [])
            all_relationships = result.get("relationships", [])
            provider = result.get("provider", provider)
            model_name = result.get("model", model_name)
            logger.info(
                "derive_ontology_graph: extract-all returned entities=%d relationships=%d for %s",
                len(all_entities), len(all_relationships), document_id,
            )
        except Exception as exc:
            logger.warning(
                "derive_ontology_graph: extract-all failed for %s: %s",
                document_id, exc,
            )
            group_errors.append(f"extract-all: {exc}")
            all_entities = []
            all_relationships = []

        graph_data = {
            "nodes": all_entities,
            "edges": all_relationships,
        }
        graph_data["mentions"] = _build_entity_mentions(
            graph_data["nodes"], elements, provider,
        )
        if group_errors:
            graph_data["_group_errors"] = group_errors

        # Import into Neo4j with confidence quality gates (batch)
        neo4j_driver = get_neo4j_driver()
        nodes_rejected = 0
        edges_rejected = 0

        node_min_conf = settings.graph_node_min_confidence
        rel_min_conf = settings.graph_rel_min_confidence
        accepted_nodes: set[str] = set()

        # Collect accepted nodes for batch upsert
        batch_nodes: list[dict] = []
        for node in graph_data.get("nodes", []):
            conf = node.get("confidence", 0.8)
            if conf < node_min_conf:
                nodes_rejected += 1
                continue
            node_name = node.get("name", node.get("id", ""))
            accepted_nodes.add(node_name)
            node_props = dict(node.get("properties", {}))
            node_props.update({
                "id": str(uuid.uuid4()),
                "name": node_name,
                "entity_type": node.get("entity_type", "UNKNOWN"),
                "artifact_id": document_id,
                "confidence": conf,
            })
            batch_nodes.append({
                "entity_type": node.get("entity_type", "UNKNOWN"),
                "name": node_name,
                "artifact_id": document_id,
                "confidence": conf,
                "props": node_props,
            })

        from app.services.neo4j_graph import upsert_nodes_batch, upsert_relationships_batch
        nodes_created = upsert_nodes_batch(neo4j_driver, batch_nodes) if batch_nodes else 0

        # Collect accepted edges for batch upsert
        batch_edges: list[dict] = []
        for edge in graph_data.get("edges", []):
            conf = edge.get("confidence", 0.8)
            from_name = edge.get("from_name", "")
            to_name = edge.get("to_name", "")
            if conf < rel_min_conf or from_name not in accepted_nodes or to_name not in accepted_nodes:
                edges_rejected += 1
                continue
            edge_props = dict(edge.get("properties", {}))
            edge_props["artifact_id"] = document_id
            edge_props["confidence"] = conf
            batch_edges.append({
                "from_name": from_name,
                "from_type": edge.get("from_type", "UNKNOWN"),
                "to_name": to_name,
                "to_type": edge.get("to_type", "UNKNOWN"),
                "rel_type": edge.get("rel_type", edge.get("type", "RELATED_TO")),
                "artifact_id": document_id,
                "confidence": conf,
                "props": edge_props,
            })

        edges_created = upsert_relationships_batch(neo4j_driver, batch_edges) if batch_edges else 0

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
    except SoftTimeLimitExceeded:
        logger.warning("derive_ontology_graph: soft time limit for %s", document_id)
        db.rollback()
        if run_id:
            _update_stage_run(db, run_id, "derive_ontology_graph", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        return {"stage": "derive_ontology_graph", "status": "failed", "error": "soft time limit exceeded"}
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
            return {"stage": "derive_ontology_graph", "status": "failed", "error": str(exc)}
        logger.info("derive_ontology_graph: retrying %s (attempt %d/%d)", document_id, self.request.retries + 1, self.max_retries)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
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

        # SAME_SECTION links — neighbor-only (prev/next by position) to avoid O(n²)
        section_chunks: dict[str, list] = {}
        for tc in text_chunks:
            if tc.artifact_id and str(tc.artifact_id) in artifact_element_map:
                elem = artifact_element_map[str(tc.artifact_id)]
                if elem.section_path:
                    section_chunks.setdefault(elem.section_path, []).append(tc)

        for section, chunks in section_chunks.items():
            for i in range(len(chunks) - 1):
                _upsert_link(
                    chunks[i].id, chunks[i + 1].id, "SAME_SECTION", 1,
                    settings.retrieval_weight_same_section,
                )
                _upsert_link(
                    chunks[i + 1].id, chunks[i].id, "SAME_SECTION", 1,
                    settings.retrieval_weight_same_section,
                )

        # SAME_ARTIFACT links — neighbor-only (prev/next by position) to avoid O(n²)
        artifact_chunks: dict[str, list] = {}
        for tc in text_chunks:
            if tc.artifact_id:
                artifact_chunks.setdefault(str(tc.artifact_id), []).append(tc)

        for art_id, chunks in artifact_chunks.items():
            for i in range(len(chunks) - 1):
                _upsert_link(
                    chunks[i].id, chunks[i + 1].id, "SAME_ARTIFACT", 1,
                    settings.retrieval_weight_same_artifact,
                )
                _upsert_link(
                    chunks[i + 1].id, chunks[i].id, "SAME_ARTIFACT", 1,
                    settings.retrieval_weight_same_artifact,
                )

        db.commit()

        # Create Neo4j structural edges
        from app.services.neo4j_graph import (
            upsert_document_node,
            upsert_chunk_ref_node,
            create_structural_edge,
            batch_create_entity_chunk_edges,
        )
        from app.db.session import get_neo4j_driver
        neo4j_driver = get_neo4j_driver()

        # Include document metadata (summary, classification) as Neo4j properties
        doc_node_props: dict[str, Any] = {"source_id": str(doc.source_id)}
        if doc.document_metadata and isinstance(doc.document_metadata, dict):
            if doc.document_metadata.get("document_summary"):
                doc_node_props["summary"] = doc.document_metadata["document_summary"]
            if doc.document_metadata.get("classification"):
                doc_node_props["classification"] = doc.document_metadata["classification"]
            if doc.document_metadata.get("date_of_information"):
                doc_node_props["date_of_information"] = doc.document_metadata["date_of_information"]
            if doc.document_metadata.get("source_characterization"):
                doc_node_props["source_characterization"] = doc.document_metadata["source_characterization"]

        upsert_document_node(
            driver=neo4j_driver,
            document_id=document_id,
            title=doc.filename,
            properties=doc_node_props,
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

        # Collect all entity-chunk edges, then batch-create in one Cypher call per type
        edge_tuples: list[tuple[str, str, str]] = []  # (name, type, chunk_id)

        used_mentions_path = False
        if graph_extraction and graph_extraction.graph_json:
            mentions = graph_extraction.graph_json.get("mentions", [])
            if mentions:
                used_mentions_path = True
                for mention in mentions:
                    name = mention.get("entity_name", "")
                    etype = mention.get("entity_type", "UNKNOWN")
                    euid = mention.get("element_uid", "")
                    for chunk_id in element_uid_chunk_map.get(euid, []):
                        edge_tuples.append((name, etype, chunk_id))

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
                        edge_tuples.append((name, etype, chunk_id))

        entity_links = batch_create_entity_chunk_edges(neo4j_driver, edge_tuples)

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
        failed = []
        for r in (derivation_results or []):
            if not isinstance(r, dict):
                failed.append(str(r))
            elif r.get("status") not in ("ok", "skipped"):
                failed.append(r.get("stage", "unknown"))
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
                 soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
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


@celery_app.task(bind=True, soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
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
            "derive_document_metadata",
            "derive_picture_descriptions",
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


