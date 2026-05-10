"""Pipeline stage dispatch — beat-scheduled claim + publish.

See: docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md
"""
import logging

from sqlalchemy import text

from app.services.redis_utils import get_redis
from app.workers._db import get_worker_db as _get_db
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

DISPATCH_BATCH_LIMIT = 50
DISPATCH_LOCK_KEY = "dispatcher:pipeline_stages"
DISPATCH_LOCK_TTL = 30  # seconds


@celery_app.task(
    bind=True,
    name="app.workers.dispatcher.dispatch_pending_pipeline_stages",
)
def dispatch_pending_pipeline_stages(self) -> dict:
    redis = get_redis()
    if not redis.set(DISPATCH_LOCK_KEY, self.request.id, nx=True, ex=DISPATCH_LOCK_TTL):
        return {"skipped": "another dispatcher tick is in flight"}
    try:
        return _run_dispatch_tick()
    finally:
        _release_lock_if_owner(redis, DISPATCH_LOCK_KEY, self.request.id)


def _release_lock_if_owner(redis, key: str, token: str) -> None:
    """Lua-CAS release: only the lock holder can release."""
    redis.eval(
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) else return 0 end",
        1, key, token,
    )


def _run_dispatch_tick() -> dict:
    """Claim up to DISPATCH_BATCH_LIMIT PENDING ledger rows and publish.

    Atomically transitions PENDING -> DISPATCHED via FOR UPDATE SKIP LOCKED,
    then publishes the corresponding Celery task. On publish failure the
    claim is reverted to PENDING by `_undo_claim`.
    """
    db = _get_db()
    try:
        rows = db.execute(text("""
            WITH claimable AS (
                SELECT sr.id
                FROM ingest.stage_runs sr
                JOIN ingest.pipeline_runs pr ON pr.id = sr.pipeline_run_id
                WHERE sr.status       = 'PENDING'
                  AND sr.pass_name    IS NULL
                  AND sr.task_name    IS NOT NULL
                  AND sr.available_at <= NOW()
                  AND pr.status       = 'PROCESSING'
                ORDER BY sr.available_at ASC
                LIMIT :limit
                FOR UPDATE OF sr SKIP LOCKED
            )
            UPDATE ingest.stage_runs sr
            SET status        = 'DISPATCHED',
                dispatched_at = NOW()
            FROM claimable c
            WHERE sr.id = c.id
            RETURNING sr.id, sr.pipeline_run_id, sr.stage_name,
                      sr.task_name, sr.queue_name,
                      (SELECT pr.document_id
                       FROM ingest.pipeline_runs pr
                       WHERE pr.id = sr.pipeline_run_id) AS document_id
        """), {"limit": DISPATCH_BATCH_LIMIT}).fetchall()
        db.commit()
    finally:
        db.close()

    published = 0
    for row in rows:
        try:
            _publish(row.task_name, row.document_id, row.pipeline_run_id, row.id)
            published += 1
        except Exception as exc:
            _undo_claim(row.id, error=str(exc))
            logger.exception("dispatcher: failed to publish stage_run=%s", row.id)

    return {"claimed": len(rows), "published": published}


def _publish(task_name, document_id, run_id, stage_run_id):
    """Publish the Celery task; record celery_task_id on the ledger row.

    Routing is resolved by Celery's `task_routes` configuration — do NOT
    pass `queue=` here (that would override the central routing table).
    """
    task = celery_app.tasks[task_name]
    result = task.apply_async(
        args=[str(document_id), str(run_id)],
        headers={"stage_run_id": str(stage_run_id)},
    )
    db = _get_db()
    try:
        db.execute(text("""
            UPDATE ingest.stage_runs
            SET celery_task_id = :tid
            WHERE id = :id AND status = 'DISPATCHED'
        """), {"tid": result.id, "id": stage_run_id})
        db.commit()
    finally:
        db.close()


def _undo_claim(stage_run_id, *, error: str) -> None:
    """Return a stage_run from DISPATCHED back to PENDING after publish failure.

    Guarded by `WHERE status='DISPATCHED'` so we never accidentally clobber
    a row that has already advanced (e.g. picked up by a parallel worker).
    """
    db = _get_db()
    try:
        db.execute(text("""
            UPDATE ingest.stage_runs
            SET status        = 'PENDING',
                dispatched_at = NULL,
                error_message = COALESCE(error_message, '') || ' publish_failed: ' || :err
            WHERE id = :id AND status = 'DISPATCHED'
        """), {"id": stage_run_id, "err": error})
        db.commit()
    finally:
        db.close()
