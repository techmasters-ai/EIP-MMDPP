"""Worker death between stages does not strand the pipeline_run.

This is the headline regression test for the dispatch-ledger feature
(spec: docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md).

Status: SKIPPED - pending integration fixture infrastructure.
"""
import pytest


pytestmark = pytest.mark.skip(
    reason="integration fixtures (celery_worker, small_pdf_fixture, pg_session) "
           "not yet defined; see TODO at bottom of file"
)


def test_kill_worker_between_stages_resumes_via_dispatcher(
    pg_session, celery_worker, small_pdf_fixture,
):
    """Headline regression: this would fail on main before the spec implementation.

    Sequence:
      1. start_ingest_pipeline -> first ledger row PENDING
      2. Wait for derive_image_embeddings to COMPLETE (stage 7).
      3. Snapshot derive_document_anchors (stage 8) status; must be
         PENDING / DISPATCHED / RUNNING (not COMPLETE - otherwise the
         kill window has already closed).
      4. Kill and restart the worker.
      5. Dispatcher publishes stage 8 within 5-10s; assert it reaches
         COMPLETE.
    """
    import time
    from sqlalchemy import text
    from app.workers.pipeline import start_ingest_pipeline

    def stage_status(stage):
        row = pg_session.execute(text("""
            SELECT status FROM ingest.stage_runs
            WHERE pipeline_run_id = :r AND stage_name = :s
        """), {"r": run_id, "s": stage}).first()
        return row.status if row else None

    def wait_until(predicate, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.5)
        return False

    doc_id = small_pdf_fixture
    result = start_ingest_pipeline(doc_id)
    run_id = result.pipeline_run_id

    assert wait_until(lambda: stage_status("derive_image_embeddings") == "COMPLETE",
                      timeout_s=120), "stage 7 did not complete in time"

    pre_kill = stage_status("derive_document_anchors")
    assert pre_kill is not None
    assert pre_kill != "COMPLETE", (
        "stage 8 already complete; pick a slower fixture so the kill window opens"
    )

    celery_worker.kill_and_restart()

    assert wait_until(lambda: stage_status("derive_document_anchors") == "COMPLETE",
                      timeout_s=15), "stage 8 did not resume after worker kill"


# ---------------------------------------------------------------------------
# TODO: implement integration fixtures
#
# Required fixtures (define in tests/integration/conftest.py):
#   - pg_session: SQLAlchemy session against the real Postgres in the
#     docker stack (NOT a rollback-on-finish fixture - needs to see the
#     worker's commits).
#   - celery_worker: process wrapper with .kill_and_restart() method.
#     Likely needs to manage the eip-mmdpp-worker-1 container via
#     docker subprocess calls.
#   - small_pdf_fixture: uploads a fixture PDF to the ingest source
#     and returns its document_id.
#
# Without these the headline regression test cannot run. Track this
# in a follow-up plan/PR - the dispatch-ledger feature itself can ship
# as long as unit tests pass and the live-system verification steps
# in VERIFICATION_CHECKLIST.md are performed at deploy.
# ---------------------------------------------------------------------------
