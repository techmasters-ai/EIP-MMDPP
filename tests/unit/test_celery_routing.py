"""Celery task-routing tests (TODO #28: watcher-queue isolation)."""
from __future__ import annotations


def test_scan_watch_directories_routed_to_dedicated_watcher_queue():
    from app.workers.celery_app import celery_app

    routes = celery_app.conf.task_routes
    assert routes["app.workers.watcher.scan_watch_directories"] == {"queue": "watcher"}


def test_pipeline_ingest_tasks_not_sharing_queue_with_watcher():
    from app.workers.celery_app import celery_app

    routes = celery_app.conf.task_routes
    # The pipeline ingest tasks stay on `ingest`...
    assert routes["app.workers.pipeline.prepare_document"]["queue"] == "ingest"
    assert routes["app.workers.pipeline.finalize_document"]["queue"] == "ingest"
    assert routes["app.workers.pipeline.collect_derivations"]["queue"] == "ingest"
    # ...and the watcher poller is NOT on the ingest queue anymore (the #28 fix).
    assert routes["app.workers.watcher.scan_watch_directories"]["queue"] != "ingest"
