"""_resolve_queue returns the queue an apply_async() call would route to."""
from app.workers.pipeline import _resolve_queue


def test_task_routes_entry_wins():
    """Tasks with an explicit task_routes entry resolve to that queue."""
    assert _resolve_queue("app.workers.pipeline.prepare_document") == "ingest"
    assert _resolve_queue("app.workers.pipeline.derive_image_embeddings") == "embed"
    assert _resolve_queue("app.workers.pipeline.derive_document_anchors") == "graph"
    assert _resolve_queue("app.workers.pipeline.derive_ontology_graph") == "graph_extract"


def test_decorator_queue_resolves_for_pipeline_stages():
    """Tier 2 (decorator queue=): tasks with no task_routes entry but a decorator queue=.

    detect_and_translate / derive_document_metadata / derive_picture_descriptions
    have no task_routes entry, but their @celery_app.task() decorators set
    queue="ingest". The helper must report that decorator-set queue so the
    ledger's queue_name column matches the runtime destination (verified via
    Task._get_exec_options() which is what apply_async() honors).
    """
    assert _resolve_queue("app.workers.pipeline.detect_and_translate") == "ingest"
    assert _resolve_queue("app.workers.pipeline.derive_document_metadata") == "ingest"
    assert _resolve_queue("app.workers.pipeline.derive_picture_descriptions") == "ingest"


def test_decorator_queue_resolves_tier_2(monkeypatch):
    """Tier 2 (decorator queue=): a task registered with queue= but no task_routes entry."""
    from app.workers.celery_app import celery_app

    class FakeTask:
        queue = "custom-queue"

    monkeypatch.setitem(celery_app.tasks, "fake.module.tier2_task", FakeTask())
    # Ensure no task_routes entry exists for this fake task.
    assert "fake.module.tier2_task" not in (celery_app.conf.task_routes or {})
    assert _resolve_queue("fake.module.tier2_task") == "custom-queue"


def test_unknown_task_falls_to_broker_default():
    """Tier 3 (broker default): task not registered at all."""
    assert _resolve_queue("nonexistent.task.path.that.does.not.exist") == "celery"


def test_every_stage_successor_task_resolves_without_error():
    """Smoke check: every STAGE_SUCCESSORS next_task can be resolved."""
    from app.workers.pipeline import STAGE_SUCCESSORS
    for stage, edge in STAGE_SUCCESSORS.items():
        if edge.next_task is None:
            continue
        queue = _resolve_queue(edge.next_task)
        assert isinstance(queue, str) and queue, f"{stage} → {edge.next_task} resolved to {queue!r}"
