"""Celery tasks for trusted data embedding and Qdrant indexing."""

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
    """Embed approved text and upsert to trusted Qdrant collection."""
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

        # Deterministic point ID from submission ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"trusted:{submission_id}"))

        # Upsert to trusted collection
        from app.services.qdrant_store import upsert_trusted_vector
        from app.db.session import get_qdrant_client

        upsert_trusted_vector(
            client=get_qdrant_client(),
            point_id=point_id,
            vector=vector,
            payload={
                "submission_id": submission_id,
                "content_text": submission.content,
                "confidence": submission.confidence,
                "classification": "UNCLASSIFIED",
                "reviewed_at": (
                    submission.reviewed_at.isoformat()
                    if submission.reviewed_at
                    else None
                ),
                "status": submission.status,
            },
        )

        # Update submission
        submission.status = "APPROVED_INDEXED"
        submission.index_status = "COMPLETE"
        submission.qdrant_point_id = uuid.UUID(point_id)
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
