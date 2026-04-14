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
        "app.workers.pipeline.derive_ontology_graph": {"queue": "graph_extract"},
        "app.workers.pipeline.derive_structure_links": {"queue": "graph"},
        "app.workers.pipeline.collect_derivations": {"queue": "ingest"},
        "app.workers.pipeline.finalize_document": {"queue": "ingest"},
        "app.workers.watcher.scan_watch_directories": {"queue": "ingest"},
        "app.workers.trusted_data_tasks.index_trusted_submission": {"queue": "trusted"},
        "app.workers.pipeline._chord_error_handler": {"queue": "ingest"},
    },
    # Task result expiry
    result_expires=86400,  # 24 hours
    # Retry defaults
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
    },
)


