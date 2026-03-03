"""Directory watcher — Celery Beat periodic task.

Polls registered watch directories every N seconds (configurable per directory).
Uses SHA-256 hash-based deduplication via ingest.watch_logs.
Files that are still being written are skipped (size-stability check).
"""

import fnmatch
import hashlib
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True)
def scan_watch_directories(self) -> None:
    """Scan all enabled watch directories for new or modified files."""
    from sqlalchemy import select
    from app.db.session import get_sync_session
    from app.models.ingest import WatchDir, WatchLog, Document, Source
    from app.workers.pipeline import start_ingest_pipeline

    db = get_sync_session()
    try:
        result = db.execute(
            select(WatchDir).where(WatchDir.enabled.is_(True))
        )
        watch_dirs = result.scalars().all()

        for watch_dir in watch_dirs:
            _scan_directory(db, watch_dir)

    except Exception as exc:
        logger.error("scan_watch_directories failed: %s", exc)
    finally:
        db.close()


def _scan_directory(db, watch_dir) -> None:
    """Scan a single watch directory for new files."""
    from sqlalchemy import select
    from app.models.ingest import WatchLog, Document, Source
    from app.workers.pipeline import start_ingest_pipeline
    import magic

    dir_path = Path(watch_dir.path)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning("watch_dir not found or not a directory: %s", watch_dir.path)
        return

    for file_path in dir_path.iterdir():
        if not file_path.is_file():
            continue

        # Check against file patterns (fnmatch)
        patterns = watch_dir.file_patterns or ["*.pdf", "*.docx", "*.png", "*.jpg", "*.tiff"]
        if not any(fnmatch.fnmatch(file_path.name, pattern) for pattern in patterns):
            continue

        # File-in-progress guard: check size stability
        if not _is_file_stable(file_path):
            logger.debug("File still being written, skipping: %s", file_path)
            continue

        try:
            file_bytes = file_path.read_bytes()
        except (PermissionError, OSError) as e:
            logger.warning("Cannot read file %s: %s", file_path, e)
            continue

        file_hash = hashlib.sha256(file_bytes).hexdigest()
        file_size = len(file_bytes)

        # Deduplication check
        existing = db.execute(
            select(WatchLog).where(
                WatchLog.watch_dir_id == watch_dir.id,
                WatchLog.file_hash == file_hash,
            )
        ).scalar_one_or_none()

        if existing:
            logger.debug("File already processed (hash match): %s", file_path)
            continue

        # Create a watch log entry and enqueue the ingest pipeline
        logger.info("New file detected: %s (hash=%s)", file_path, file_hash)
        try:
            # Get the source for this watch dir
            source = db.get(Source, watch_dir.source_id)
            if not source:
                logger.error("Source not found for watch_dir %s", watch_dir.id)
                continue

            # Create document record — need a system user UUID
            system_user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
            mime_type = None
            try:
                mime_type = magic.from_buffer(file_bytes[:1024], mime=True)
            except Exception:
                pass

            from app.config import get_settings
            settings = get_settings()

            object_key = f"sources/{watch_dir.source_id}/watch/{uuid.uuid4()}/{file_path.name}"

            from app.services.storage import upload_bytes_sync
            upload_bytes_sync(
                file_bytes,
                settings.minio_bucket_raw,
                object_key,
                content_type=mime_type or "application/octet-stream",
            )

            document = Document(
                source_id=watch_dir.source_id,
                filename=file_path.name,
                mime_type=mime_type,
                file_size_bytes=file_size,
                file_hash=file_hash,
                storage_bucket=settings.minio_bucket_raw,
                storage_key=object_key,
                pipeline_status="PENDING",
                uploaded_by=system_user_id,
            )
            db.add(document)
            db.flush()  # get document.id before commit

            # Create watch log
            watch_log = WatchLog(
                watch_dir_id=watch_dir.id,
                file_path=str(file_path),
                file_hash=file_hash,
                file_size_bytes=file_size,
                document_id=document.id,
                status="enqueued",
            )
            db.add(watch_log)
            db.commit()

            # Enqueue ingest pipeline
            task_id = start_ingest_pipeline(str(document.id))

            # Update document with task ID
            from sqlalchemy import update
            db.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(celery_task_id=task_id)
            )
            db.commit()

            logger.info(
                "Enqueued ingest for watcher file: %s document_id=%s task_id=%s",
                file_path,
                document.id,
                task_id,
            )

        except Exception as e:
            db.rollback()
            logger.error("Failed to enqueue file %s: %s", file_path, e)


def _is_file_stable(file_path: Path, wait_seconds: float = 2.0) -> bool:
    """Return True if the file size did not change over wait_seconds.

    This guards against ingesting a file that is still being written.
    """
    try:
        size_before = file_path.stat().st_size
        time.sleep(wait_seconds)
        size_after = file_path.stat().st_size
        return size_before == size_after and size_after > 0
    except (FileNotFoundError, OSError):
        return False


# Import Document here to avoid circular import issues
from app.models.ingest import Document  # noqa: E402
