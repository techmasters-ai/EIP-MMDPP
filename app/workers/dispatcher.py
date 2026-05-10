"""Pipeline stage dispatch — beat-scheduled claim + publish.

See: docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md
"""
import logging

from app.services.redis_utils import get_redis
from app.workers._db import get_worker_db as _get_db  # noqa: F401  (used in Task 15)
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
    """Stub — implemented in Task 15."""
    return {"claimed": 0, "published": 0}


def _publish(task_name, document_id, run_id, stage_run_id):
    """Stub — implemented in Task 15."""
    raise NotImplementedError("_publish implemented in Task 15")


def _undo_claim(stage_run_id, *, error: str) -> None:
    """Stub — implemented in Task 15."""
    raise NotImplementedError("_undo_claim implemented in Task 15")
