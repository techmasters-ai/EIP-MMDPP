"""Stale RUNNING for ledger stages: reset under cap, terminalize over."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _sweep_stale_runs

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_running(db_session, *, stage, dispatch_attempt, started_secs_ago,
                  pass_name=None, run_status="PROCESSING"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',0,:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, :st)
    """), {"r": run, "d": doc, "st": run_status})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             pass_name, task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                :pn,
                'app.workers.pipeline.x',
                :da,
                NOW() - make_interval(secs => :secs))
    """), {"r": run, "s": stage, "pn": pass_name, "da": dispatch_attempt,
           "secs": started_secs_ago})
    db_session.flush()
    return str(run)


def test_ledger_running_under_cap_resets_to_pending(db_session, patched_get_db, monkeypatch):
    from app.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "max_stage_dispatches", 5)
    monkeypatch.setattr(s, "stale_stage_run_threshold_seconds", 60)

    run_id = _seed_running(
        db_session, stage="prepare_document", dispatch_attempt=2,
        started_secs_ago=200,
    )
    _sweep_stale_runs()
    row = db_session.execute(text("""
        SELECT status, dispatch_attempt, started_at IS NULL AS cleared,
               available_at <= NOW() AS due_now
        FROM ingest.stage_runs WHERE pipeline_run_id = :r
    """), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert row.dispatch_attempt == 3
    assert row.cleared
    assert row.due_now


def test_ledger_running_at_cap_terminalizes_with_pipeline_run(db_session, patched_get_db, monkeypatch):
    from app.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "max_stage_dispatches", 3)
    monkeypatch.setattr(s, "stale_stage_run_threshold_seconds", 60)

    run_id = _seed_running(
        db_session, stage="prepare_document", dispatch_attempt=3,
        started_secs_ago=200,
    )
    _sweep_stale_runs()
    sr = db_session.execute(text(
        "SELECT status, dispatch_attempt FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert sr.status == "FAILED"
    assert sr.dispatch_attempt == 4

    pr = db_session.execute(text(
        "SELECT status FROM ingest.pipeline_runs WHERE id = :r"
    ), {"r": run_id}).first()
    assert pr.status == "FAILED"


def test_derive_ontology_graph_running_is_excluded(db_session, patched_get_db, monkeypatch):
    """Stage 9 stays RUNNING — reconcile_ontology_graph_runs owns it."""
    from app.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "stale_stage_run_threshold_seconds", 60)

    run_id = _seed_running(
        db_session, stage="derive_ontology_graph", dispatch_attempt=1,
        started_secs_ago=3600,
    )
    _sweep_stale_runs()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"  # not touched by ledger sweep
