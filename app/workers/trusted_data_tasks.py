"""Celery tasks for trusted data embedding and ArcadeDB indexing."""

import logging
import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.workers.celery_app import celery_app
from app.workers._db import get_worker_db as _get_db

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="trusted",
    soft_time_limit=120,
    time_limit=180,
)
def index_trusted_submission(self, submission_id: str):
    """Embed approved text and upsert to ArcadeDB as TrustedTextChunk vertex."""
    from app.models.trusted_data import TrustedDataSubmission

    db = _get_db()
    try:
        submission = db.get(TrustedDataSubmission, uuid.UUID(submission_id))
        if not submission:
            logger.warning("Submission %s not found", submission_id)
            return

        if submission.status not in ("APPROVED_PENDING_INDEX", "INDEX_FAILED"):
            logger.info(
                "Submission %s has status %s, skipping indexing",
                submission_id, submission.status,
            )
            return

        submission.index_status = "INDEXING"
        db.commit()

        # Embed
        from app.services.embedding import embed_texts

        vectors = embed_texts([submission.content])
        vector = vectors[0]

        # Upsert trusted chunk in ArcadeDB — idempotent so retry/reindex
        # can safely re-run after a partial write without failing on uniqueness.
        from app.db.session import get_graph_store

        graph_store = get_graph_store()
        chunk_id = f"trusted:{submission_id}"
        reviewed_at_str = (
            submission.reviewed_at.isoformat() if submission.reviewed_at else None
        )

        # UPSERT: update if chunk_id exists, create if not
        upsert_sql = (
            "UPDATE TextChunk SET text = :text, document_id = :doc_id, "
            "confidence = :confidence, classification = :classification, "
            "modality = :modality, submission_id = :submission_id, "
            "reviewed_at = :reviewed_at, status = :status, "
            "updated_at = sysdate() "
            "UPSERT WHERE chunk_id = :chunk_id"
        )
        result = graph_store.execute_command_sync(
            upsert_sql,
            {
                "chunk_id": chunk_id,
                "text": submission.content,
                "doc_id": chunk_id,
                "confidence": submission.confidence,
                "classification": "UNCLASSIFIED",
                "modality": "trusted_text",
                "submission_id": submission_id,
                "reviewed_at": reviewed_at_str,
                "status": "APPROVED_INDEXED",
            },
        )
        # Get the RID of the upserted vertex
        rid = ""
        if result and isinstance(result[0], dict):
            rid = str(result[0].get("@rid", ""))

        # Attach embedding (uses RID directly, safe for both create and update)
        if rid:
            graph_store.set_vertex_embedding_sync(
                vertex_type="TextChunk",
                vertex_id=rid,
                embedding_property="text_embedding",
                embedding=vector,
            )

        # Update submission
        submission.status = "APPROVED_INDEXED"
        submission.index_status = "COMPLETE"
        submission.embedding_model = settings.text_embedding_model
        submission.embedded_at = datetime.now(timezone.utc)
        submission.index_error = None
        db.commit()

        logger.info("Submission %s indexed successfully", submission_id)

    except Exception as exc:
        db.rollback()
        try:
            submission = db.get(TrustedDataSubmission, uuid.UUID(submission_id))
            if submission:
                submission.index_status = "FAILED"
                submission.index_error = str(exc)[:2000]
                if self.request.retries >= self.max_retries:
                    submission.status = "INDEX_FAILED"
                db.commit()
        except Exception:
            logger.exception("Failed to update submission error state")

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise
    finally:
        db.close()
