"""_update_stage_run intercepts terminal writes when _CTX is active."""
import uuid
from sqlalchemy import text
from app.workers._stage_lifecycle import _CTX, _LifecycleCtx
from app.workers.pipeline import _update_stage_run

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_running_row(db_session, stage="prepare_document"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,
                'b','k', 0, :u, 'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                'app.workers.pipeline.prepare_document', 1, NOW())
    """), {"r": run, "s": stage})
    db_session.flush()
    return str(run)


def test_intercepts_complete_when_ctx_active_with_matching_run(db_session):
    """Body's _update_stage_run('COMPLETE', metrics=...) stashes metrics in ctx, no DB write."""
    run_id = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage="detect_and_translate", next_task="x",
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "prepare_document", "COMPLETE",
                          attempt=1, metrics={"elements": 124})
    finally:
        _CTX.reset(token)

    # Stashed
    assert ctx.pending_status == "COMPLETE"
    assert ctx.pending_metrics == {"elements": 124}

    # NOT committed to DB
    row = db_session.execute(text(
        "SELECT status, metrics FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"          # unchanged
    # metrics column defaults to {} in the seed row; the key check is that
    # the body's {"elements": 124} payload was NOT persisted (intercepted).
    assert row.metrics in (None, {})        # unchanged from seed


def test_intercepts_failed_when_ctx_active(db_session):
    run_id = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "prepare_document", "FAILED",
                          attempt=1, error="boom")
    finally:
        _CTX.reset(token)
    assert ctx.pending_status == "FAILED"
    assert ctx.pending_error == "boom"


def test_normalizes_uuid_vs_str(db_session):
    """ctx stores str; caller passes UUID; predicate compares str(both)."""
    run_id_str = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id_str, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, uuid.UUID(run_id_str), "prepare_document",
                          "COMPLETE", attempt=1, metrics={"x": 1})
    finally:
        _CTX.reset(token)
    assert ctx.pending_status == "COMPLETE"


def test_does_not_intercept_when_ctx_none(db_session):
    """Legacy path (no _CTX): existing implementation commits the write."""
    run_id = _setup_running_row(db_session)
    _update_stage_run(db_session, run_id, "prepare_document", "COMPLETE",
                      attempt=1, metrics={"elements": 99})
    db_session.flush()
    row = db_session.execute(text(
        "SELECT status, metrics FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "COMPLETE"
    assert row.metrics == {"elements": 99}


def test_does_not_intercept_different_stage(db_session):
    """ctx for stage A; call for stage B → not intercepted."""
    run_id = _setup_running_row(db_session, stage="prepare_document")
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, 'derive_document_metadata', 1, 'RUNNING',
                'app.workers.pipeline.derive_document_metadata', 1, NOW())
    """), {"r": run_id})
    db_session.flush()

    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "derive_document_metadata", "COMPLETE",
                          attempt=1, metrics={"summary_length": 100})
        db_session.flush()
    finally:
        _CTX.reset(token)
    row = db_session.execute(text("""
        SELECT status, metrics FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_document_metadata'
    """), {"r": run_id}).first()
    assert row.status == "COMPLETE"
    assert row.metrics == {"summary_length": 100}


def test_does_not_intercept_when_intercept_terminal_false(db_session):
    """Stage 9 ctx has intercept_terminal=False → writes commit through."""
    run_id = _setup_running_row(db_session, stage="derive_ontology_graph")
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="derive_ontology_graph",
        dispatch_attempt=1, intercept_terminal=False,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "derive_ontology_graph", "COMPLETE",
                          attempt=1, metrics={"node_count": 50})
        db_session.flush()
    finally:
        _CTX.reset(token)
    row = db_session.execute(text("""
        SELECT status, metrics FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_ontology_graph'
    """), {"r": run_id}).first()
    assert row.status == "COMPLETE"
    assert row.metrics == {"node_count": 50}


def test_running_status_intercepted_as_noop(db_session):
    """Body writes RUNNING redundantly; wrapper already wrote it via CLAIM."""
    run_id = _setup_running_row(db_session)
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id, stage_name="prepare_document",
        dispatch_attempt=1, intercept_terminal=True,
        next_stage=None, next_task=None,
    )
    token = _CTX.set(ctx)
    try:
        _update_stage_run(db_session, run_id, "prepare_document", "RUNNING", attempt=1)
    finally:
        _CTX.reset(token)
    assert ctx.pending_status is None
