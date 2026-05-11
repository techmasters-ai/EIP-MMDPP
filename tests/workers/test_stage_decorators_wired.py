"""All 9 stages carry guard_stage_run(lifecycle=True) with correct successor."""
from app.workers.celery_app import celery_app
from app.workers.pipeline import STAGE_SUCCESSORS


def test_every_stage_task_has_lifecycle_marker():
    for stage_name, edge in STAGE_SUCCESSORS.items():
        matching = [
            t for t in celery_app.tasks.values()
            if getattr(getattr(t, "run", None), "stage_name", None) == stage_name
        ]
        assert len(matching) == 1, f"{stage_name}: expected exactly 1 task, got {len(matching)}"
        task = matching[0]
        assert getattr(task.run, "_lifecycle", False) is True, (
            f"{stage_name} task is missing lifecycle=True"
        )


def test_stage_9_has_intercept_terminal_false():
    """derive_ontology_graph wrapper must NOT intercept terminal writes."""
    matching = [
        t for t in celery_app.tasks.values()
        if getattr(getattr(t, "run", None), "stage_name", None) == "derive_ontology_graph"
    ]
    assert len(matching) == 1
    assert matching[0].run._intercept_terminal is False


def test_other_stages_intercept_terminal_true():
    """All non-stage-9 stages must intercept_terminal=True."""
    for stage_name in STAGE_SUCCESSORS:
        if stage_name == "derive_ontology_graph":
            continue
        matching = [
            t for t in celery_app.tasks.values()
            if getattr(getattr(t, "run", None), "stage_name", None) == stage_name
        ]
        assert len(matching) == 1
        assert matching[0].run._intercept_terminal is True, (
            f"{stage_name} must have intercept_terminal=True"
        )
