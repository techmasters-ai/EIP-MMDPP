"""Celery application configuration."""

import sys

# The `celery` CLI script sets sys.path[0] to its own directory
# (/usr/local/bin), not the project root, so dynamic imports of sibling
# packages like `ontology_bundles` (imported via importlib during graph
# extraction passes) fail with ModuleNotFoundError.
#
# Insert /app unconditionally. We intentionally do NOT guard with
# `"/app" not in sys.path` — celery's `cwd_in_path` context manager
# (celery/utils/imports.py) briefly adds cwd (=/app) to sys.path for
# the import of this module, then removes it again. A guarded insert
# would see celery's temporary entry and skip, leaving forked workers
# without /app when tasks run. Inserting unconditionally leaves our
# entry behind after celery pops its own copy.
sys.path.insert(0, "/app")

from datetime import timedelta

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "eip_mmdpp",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.pipeline",
        "app.workers.watcher",
        "app.workers.trusted_data_tasks",
        "app.workers.community_tasks",
        "app.workers.dispatcher",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task routing to dedicated queues
    task_routes={
        "app.workers.pipeline.prepare_document": {"queue": "ingest"},
        "app.workers.pipeline.purge_document_derivations": {"queue": "ingest"},
        "app.workers.pipeline.derive_text_chunks_and_embeddings": {"queue": "embed"},
        "app.workers.pipeline.derive_image_embeddings": {"queue": "embed"},
        "app.workers.pipeline.derive_document_anchors": {"queue": "graph"},
        "app.workers.pipeline.derive_ontology_graph": {"queue": "graph_extract"},
        "app.workers.pipeline.derive_structure_links": {"queue": "graph"},
        "app.workers.pipeline.collect_derivations": {"queue": "ingest"},
        "app.workers.pipeline.finalize_document": {"queue": "ingest"},
        # TODO #28: the beat-driven watch-dir poller gets its own queue so its
        # periodic tasks never sit in the `ingest` FIFO ahead of pipeline
        # ingest tasks (prepare_document/collect_derivations/finalize_document).
        # Only the catch-all `worker` consumes `watcher`; `worker-ingest` does
        # not, so heavy ingest can't be starved by scan backlog.
        "app.workers.watcher.scan_watch_directories": {"queue": "watcher"},
        "app.workers.trusted_data_tasks.index_trusted_submission": {"queue": "trusted"},
        "app.workers.pipeline._chord_error_handler": {"queue": "ingest"},
    },
    # Task result expiry
    result_expires=86400,  # 24 hours
    # Worker lifecycle posture (verified 2026-04-23):
    #   task_acks_late=True            -> task ACKed only after return/success
    #   task_reject_on_worker_lost=True -> on worker death, task is requeued
    #   worker_prefetch_multiplier=1   -> at most one queued-but-not-running task per slot
    #   worker_max_tasks_per_child=0   -> no forkpool recycle (Celery default)
    #   worker_max_memory_per_child=0  -> no memory-based recycle (Celery default)
    # Recycling a forkpool child mid-task can silently lose the in-flight task,
    # so the recycle knobs are intentionally left at the default 0. If either
    # is raised in the future, audit every long-running stage for idempotency
    # and rely on periodic_stale_run_sweep to close the gap.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Prevent long tasks from starving short ones
    worker_prefetch_multiplier=1,
    # Redis visibility timeout — prevent redelivery of long-running tasks
    broker_transport_options={"visibility_timeout": settings.celery_visibility_timeout},
    # Beat schedule
    beat_schedule={
        "scan-watch-directories": {
            "task": "app.workers.watcher.scan_watch_directories",
            "schedule": settings.watch_dir_poll_interval_seconds,
        },
        "community-detection": {
            "task": "app.workers.community_tasks.run_community_detection_task",
            "schedule": timedelta(minutes=settings.community_detection_interval_minutes),
            "kwargs": {"mode": "incremental"},
            "options": {"queue": "graph"},
        },
        "periodic-stale-run-sweep": {
            "task": "app.workers.pipeline.periodic_stale_run_sweep",
            "schedule": timedelta(minutes=10),
        },
        "reconcile-ontology-graph-runs": {
            "task": "app.workers.pipeline.reconcile_ontology_graph_runs",
            "schedule": settings.reconciler_period_seconds,
            "options": {"queue": "graph"},
        },
        "poll-extraction-progress": {
            "task": "app.workers.pipeline.poll_extraction_progress",
            "schedule": settings.extraction_progress_poll_seconds,
            "options": {"queue": "graph"},
        },
        "dispatch-pending-pipeline-stages": {
            "task": "app.workers.dispatcher.dispatch_pending_pipeline_stages",
            "schedule": 5.0,
            "options": {"queue": "celery"},
        },
        # VR C.4 (rev 9 H3 + rev 10 H3): hourly cross-store janitor that purges
        # terminated + orphaned ExtractionChunk rows.  Defense-in-depth for runs
        # where _terminalize_doc_and_run's inline cleanup failed.
        #
        # Merged-chunk routing Task 7 — pre-A/B sweep guidance (Task 10):
        # Before kicking off the merged-mode A/B comparison, manually trigger
        # this janitor (or run the equivalent ``cleanup_extraction_index``
        # call) against ExtractionChunk to clear stale per-element rows from
        # prior runs.  The janitor is run-id-scoped so it handles BOTH
        # vertex_id formats (legacy ``f"{run_id}:{self_ref}"`` and merged
        # ``f"{run_id}:chunk_{chunk_index}"``) in one DELETE.  After the
        # sweep, verify ``SELECT count(*) FROM ExtractionChunk WHERE
        # chunk_index = -1`` returns 0 — any non-zero result is a stale
        # per-element row that would pollute the merged-mode index pool.
        "vr-purge-terminated-extraction-chunks": {
            "task": "vr.purge_terminated_extraction_chunks",
            "schedule": timedelta(hours=1),
            "options": {"queue": "graph"},
        },
    },
)


def _post_register_ledger_checks() -> None:
    """Run module-load checks on ledger wiring once all stage decorators are
    registered. Activated in Task 22 (Chunk 6) after Task 20 wired all 9
    stage decorators with lifecycle=True.

    Triggers ``celery_app.finalize()`` first so the ``include=[...]`` modules
    (notably ``app.workers.pipeline``) finish loading and register their
    stage decorators before we inspect the task registry.

    Skips silently if invoked during a re-entrant import — i.e. ``pipeline.py``
    is mid-load and our assertion helpers aren't yet bound. The same checks
    will fire on the next clean entry (worker boot imports celery_app first,
    so re-entry never happens in production).
    """
    pipeline_mod = sys.modules.get("app.workers.pipeline")
    if pipeline_mod is not None and not hasattr(pipeline_mod, "_assert_ledger_wiring"):
        # Re-entrant import: pipeline.py is mid-execution (it imported us at
        # its line 43, before its assertion helpers were defined). Defer.
        return

    celery_app.finalize()
    # Imported here so pipeline.py finishes registering all tasks first.
    from app.workers.pipeline import _assert_ledger_wiring, _assert_threshold_envelope
    _assert_ledger_wiring()
    _assert_threshold_envelope()


_post_register_ledger_checks()  # run at module load
