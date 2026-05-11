"""Dispatcher beat entry is registered with the correct schedule."""


def test_dispatch_pending_pipeline_stages_beat_entry():
    from app.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    assert "dispatch-pending-pipeline-stages" in schedule
    entry = schedule["dispatch-pending-pipeline-stages"]
    assert entry["task"] == "app.workers.dispatcher.dispatch_pending_pipeline_stages"
    assert entry["schedule"] == 5.0
    assert entry["options"]["queue"] == "celery"
