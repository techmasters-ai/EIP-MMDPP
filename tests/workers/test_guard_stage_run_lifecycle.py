"""guard_stage_run integrates CLAIM, body, _CTX, and finalize."""
import uuid
from unittest.mock import MagicMock, patch
from sqlalchemy import text
from app.workers._stage_lifecycle import _CTX
from app.workers.pipeline import guard_stage_run

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_pending(db_session, stage="prepare_document"):
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
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, :s, 1, 'PENDING',
                'app.workers.pipeline.prepare_document', 1)
    """), {"r": run, "s": stage})
    db_session.flush()
    return str(run)


def _fake_self(retries: int = 0, task_id: str = "task-1", max_retries: int = 2):
    self = MagicMock()
    self.request.retries = retries
    self.request.id = task_id
    self.max_retries = max_retries
    return self


def test_wrapper_marker_attributes_set():
    """guard_stage_run sets stage_name, _lifecycle, _intercept_terminal on wrapper."""
    @guard_stage_run("test_stage", lifecycle=True, next_stage="next",
                     next_task="app.workers.pipeline.detect_and_translate",
                     intercept_terminal=True)
    def body(self, doc_id, run_id=None):
        return {"status": "ok"}
    assert body.stage_name == "test_stage"
    assert body._lifecycle is True
    assert body._intercept_terminal is True


def test_proceed_runs_body_with_ctx_then_finalizes(db_session, patched_get_db):
    run_id = _seed_pending(db_session)
    captured_ctx = {}

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        captured_ctx["ctx"] = _CTX.get()
        return {"status": "ok"}

    with patch("app.workers.pipeline._finalize_after_body") as fin:
        body(_fake_self(), "doc-1", run_id)
        fin.assert_called_once()

    assert captured_ctx["ctx"] is not None
    assert captured_ctx["ctx"].pipeline_run_id == run_id
    assert captured_ctx["ctx"].stage_name == "prepare_document"
    assert _CTX.get() is None


def test_legacy_path_runs_body_without_ctx(db_session, patched_get_db):
    """No ledger row exists → body runs inline (legacy), _CTX stays None."""
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
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.flush()
    run_id = str(run)

    seen = {}

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        seen["ctx"] = _CTX.get()
        return {"status": "ok"}

    with patch("app.workers.pipeline._finalize_after_body") as fin:
        body(_fake_self(), "doc-1", run_id)
        fin.assert_not_called()

    assert seen["ctx"] is None


def test_early_return_already_complete_skips_body(db_session, patched_get_db):
    run_id = _seed_pending(db_session)
    db_session.execute(text(
        "UPDATE ingest.stage_runs SET status='COMPLETE' WHERE pipeline_run_id=:r"
    ), {"r": run_id})
    db_session.flush()

    ran = {"body": False}

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        ran["body"] = True
        return None

    result = body(_fake_self(), "doc-1", run_id)
    assert ran["body"] is False
    assert result == {"stage": "prepare_document",
                      "status": "skipped",
                      "reason": "already_complete"}
    assert _CTX.get() is None


def test_celery_retry_passthrough_no_tx4(db_session, patched_get_db):
    """Body raises CeleryRetry → wrapper passes through, row stays RUNNING."""
    from celery.exceptions import Retry as CeleryRetry
    run_id = _seed_pending(db_session)
    db_session.execute(text(
        "UPDATE ingest.stage_runs SET status='DISPATCHED' WHERE pipeline_run_id=:r"
    ), {"r": run_id})
    db_session.flush()

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        raise CeleryRetry()

    import pytest
    with patch("app.workers.pipeline._finalize_after_body") as fin, \
         patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        with pytest.raises(CeleryRetry):
            body(_fake_self(), "doc-1", run_id)
        fin.assert_not_called()
        tx4.assert_not_called()

    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id=:r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"
    assert _CTX.get() is None


def test_non_celery_exception_triggers_tx4(db_session, patched_get_db):
    run_id = _seed_pending(db_session)

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        raise ValueError("boom")

    import pytest
    with patch("app.workers.pipeline._tx4_finalize_failure") as tx4:
        with pytest.raises(ValueError, match="boom"):
            body(_fake_self(), "doc-1", run_id)
        tx4.assert_called_once()
    assert _CTX.get() is None


def test_ctx_reset_between_invocations(db_session, patched_get_db):
    """Two task invocations in same process: _CTX is None at start of each."""
    run_id = _seed_pending(db_session)

    @guard_stage_run("prepare_document", lifecycle=True,
                     next_stage="detect_and_translate",
                     next_task="app.workers.pipeline.detect_and_translate")
    def body(self, doc_id, run_id=None):
        return {}

    with patch("app.workers.pipeline._finalize_after_body"):
        body(_fake_self(task_id="t1"), "doc-1", run_id)
        assert _CTX.get() is None

        db_session.execute(text(
            "UPDATE ingest.stage_runs SET status='PENDING' WHERE pipeline_run_id=:r"
        ), {"r": run_id})
        db_session.flush()
        body(_fake_self(task_id="t2"), "doc-1", run_id)
        assert _CTX.get() is None
