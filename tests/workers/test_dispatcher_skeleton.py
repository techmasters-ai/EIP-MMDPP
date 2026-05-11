"""dispatcher module exports the expected task + helpers."""
def test_module_imports():
    from app.workers import dispatcher
    assert hasattr(dispatcher, "dispatch_pending_pipeline_stages")
    assert hasattr(dispatcher, "_run_dispatch_tick")
    assert hasattr(dispatcher, "_publish")
    assert hasattr(dispatcher, "_undo_claim")
    assert hasattr(dispatcher, "_release_lock_if_owner")


def test_task_is_registered_with_celery():
    from app.workers.celery_app import celery_app
    name = "app.workers.dispatcher.dispatch_pending_pipeline_stages"
    assert name in celery_app.tasks


def test_constants_exposed():
    from app.workers.dispatcher import (
        DISPATCH_BATCH_LIMIT, DISPATCH_LOCK_KEY, DISPATCH_LOCK_TTL,
    )
    assert DISPATCH_BATCH_LIMIT == 50
    assert DISPATCH_LOCK_KEY == "dispatcher:pipeline_stages"
    assert DISPATCH_LOCK_TTL == 30
