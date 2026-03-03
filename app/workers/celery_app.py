"""Celery application configuration."""

from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "eip_mmdpp",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.pipeline",
        "app.workers.watcher",
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
        "app.workers.pipeline.validate_and_store": {"queue": "ingest"},
        "app.workers.pipeline.detect_modalities": {"queue": "ingest"},
        "app.workers.pipeline.extract_text": {"queue": "extract"},
        "app.workers.pipeline.extract_images": {"queue": "extract"},
        "app.workers.pipeline.run_ocr": {"queue": "extract"},
        "app.workers.pipeline.process_schematics": {"queue": "vision"},
        "app.workers.pipeline.collect_metadata": {"queue": "ingest"},
        "app.workers.pipeline.chunk_and_embed": {"queue": "embed"},
        "app.workers.pipeline.extract_graph_entities": {"queue": "graph"},
        "app.workers.pipeline.import_graph": {"queue": "graph"},
        "app.workers.pipeline.finalize_artifact": {"queue": "ingest"},
        "app.workers.watcher.scan_watch_directories": {"queue": "ingest"},
    },
    # Task result expiry
    result_expires=86400,  # 24 hours
    # Retry defaults
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Beat schedule
    beat_schedule={
        "scan-watch-directories": {
            "task": "app.workers.watcher.scan_watch_directories",
            "schedule": settings.watch_dir_poll_interval_seconds,
        },
    },
)
