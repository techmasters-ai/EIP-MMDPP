"""Unit + integration tests for the derive_ontology_graph thin dispatcher (Task 8).

Pattern:
- Real PostgreSQL (port 5438) for DB-state assertions.  Parent rows created
  via ``pipeline_run_factory``; rolled back after each test by ``db_session``.
- ``_get_db`` is patched to return a *non-closing proxy* around ``db_session``
  so the task body shares the same transaction as the test.  The proxy
  suppresses ``.close()`` so ``db_session``'s rollback-on-teardown still works.
- ``derive_ontology_graph_pass.delay`` is patched to capture dispatch calls
  without actually enqueuing.
- ``claim_phase`` and ``mark_phase_dispatched`` are the real implementations
  (they write into dispatched_phases JSONB on pipeline_runs); we verify their
  effects by reading the row directly.

Run with:

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \\
        /home/josh/development/EIP-MMDPP/.venv/bin/python -m pytest \\
        tests/unit/test_derive_ontology_graph_dispatcher.py -v
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Test-local helpers
# ---------------------------------------------------------------------------

_BUNDLE_KEY = "air_defense_v3"


def _fake_pass_def(
    *,
    name: str,
    depends_on: list[str] | None = None,
):
    return SimpleNamespace(
        name=name,
        required=True,
        depends_on=list(depends_on or []),
        input_mode="document_only",
    )


def _fake_manifest(passes: list | None = None):
    """Manifest with two entity passes (no depends_on) + one system_links pass."""
    if passes is None:
        passes = [
            _fake_pass_def(name="radar_identity"),
            _fake_pass_def(name="radar_kinematics"),
            _fake_pass_def(name="system_links", depends_on=["radar_identity"]),
        ]
    return SimpleNamespace(
        bundle_key=_BUNDLE_KEY,
        passes=passes,
    )


def _make_task_self(retries: int = 0, request_id: str = "task-dispatcher-001"):
    """Build a mock Celery task ``self`` for direct invocation of the task body."""
    fake_self = MagicMock()
    fake_self.request.retries = retries
    fake_self.request.id = request_id
    return fake_self


class _NoCloseProxy:
    """Proxy around a SQLAlchemy Session that suppresses ``.close()`` so the
    test's transaction-rollback fixture can still clean up after the task.
    Also wraps commit() as flush() so writes remain visible in the same
    transaction."""

    def __init__(self, session: Session):
        self._session = session

    def close(self):
        pass  # suppress — db_session fixture handles teardown

    def commit(self):
        self._session.flush()

    def rollback(self):
        pass  # tests rollback at teardown

    def __getattr__(self, name: str):
        return getattr(self._session, name)


def _build_proxy(db_session: Session) -> _NoCloseProxy:
    return _NoCloseProxy(db_session)


@contextmanager
def _patched_dispatcher(
    db_session: Session,
    *,
    manifest=None,
):
    """Context manager that patches all heavyweight IO in the dispatcher body.

    Returns a dict of mocks keyed by short names.
    """
    if manifest is None:
        manifest = _fake_manifest()

    proxy = _build_proxy(db_session)
    mock_delay_result = MagicMock()
    mock_delay_result.id = "mock-pass-task-id"

    patches = [
        patch("app.workers.pipeline._get_db", return_value=proxy),
        patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
        patch("app.workers.pipeline.derive_ontology_graph_pass.delay",
              return_value=mock_delay_result),
    ]
    keys = ["_get_db", "load_bundle_manifest", "derive_ontology_graph_pass_delay"]

    mocks: dict[str, Any] = {}
    started = []
    for p, key in zip(patches, keys):
        m = p.start()
        mocks[key] = m
        started.append(p)
    try:
        yield mocks
    finally:
        for p in started:
            p.stop()


def _invoke(self_mock, document_id: str, run_id: str) -> Any:
    """Call the raw task body, bypassing ``guard_stage_run``.

    ``guard_stage_run`` catches unhandled exceptions and calls
    ``_terminalize_doc_and_run`` before re-raising.  Using ``__wrapped__``
    skips the guard decorator entirely.
    """
    from app.workers.pipeline import derive_ontology_graph
    raw_fn = derive_ontology_graph.run.__wrapped__
    return raw_fn(self_mock, document_id, run_id)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDispatcherCreatesSummaryStageRunAsRunning:
    """Test 1: dispatcher creates a RUNNING summary StageRun (pass_name IS NULL)."""

    def test_dispatcher_creates_summary_stagerun_as_running(
        self, db_session, pipeline_run_factory
    ):
        """After dispatching, a StageRun with stage_name='derive_ontology_graph'
        and pass_name IS NULL must exist with status='RUNNING'."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched_dispatcher(db_session) as mocks:
            result = _invoke(_make_task_self(), doc_id, str(run_id))

        # Result indicates dispatched
        assert result["status"] == "dispatched"
        assert result["stage"] == "derive_ontology_graph"

        # DB: RUNNING summary StageRun exists
        row = db_session.execute(
            text(
                "SELECT status, pass_name FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id "
                "  AND stage_name = 'derive_ontology_graph' "
                "  AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert row is not None, "Summary StageRun was not created"
        assert row[0] == "RUNNING"
        assert row[1] is None


class TestDispatcherUsesClaimDispatchFlow:
    """Test 2: dispatcher uses claim_phase → .delay → mark_phase_dispatched for
    each initially dispatched pass (same flow as _try_advance_phase follow-ups)."""

    def test_dispatcher_uses_same_claim_dispatch_flow_as_followups(
        self, db_session, pipeline_run_factory
    ):
        """claim_phase and mark_phase_dispatched are called for each dispatched
        entity pass. Verify via the dispatched_phases JSONB column on pipeline_runs."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        manifest = _fake_manifest(passes=[
            _fake_pass_def(name="radar_identity"),
            _fake_pass_def(name="radar_kinematics"),
            _fake_pass_def(name="system_links", depends_on=["radar_identity"]),
        ])

        with patch("app.workers.pipeline.claim_phase", wraps=__import__(
            "app.services.run_phase_dispatch", fromlist=["claim_phase"]
        ).claim_phase) as mock_claim, \
             patch("app.workers.pipeline.mark_phase_dispatched", wraps=__import__(
            "app.services.run_phase_dispatch", fromlist=["mark_phase_dispatched"]
        ).mark_phase_dispatched) as mock_mark:
            with _patched_dispatcher(db_session, manifest=manifest) as mocks:
                _invoke(_make_task_self(), doc_id, str(run_id))

        # claim_phase called for each dispatched entity pass
        assert mock_claim.call_count >= 1
        # mark_phase_dispatched called for each dispatched entity pass
        assert mock_mark.call_count >= 1

        # Verify dispatched_phases JSONB was written
        row = db_session.execute(
            text(
                "SELECT dispatched_phases FROM ingest.pipeline_runs WHERE id = :run_id"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert row is not None
        dispatched = row[0] or {}
        # At least one entity pass key exists in dispatched_phases
        entity_keys = [k for k in dispatched if k.startswith("entity_pass_")]
        assert len(entity_keys) >= 1


class TestDispatcherDoesNotDispatchSystemLinksDirectly:
    """Test 3: system_links is NOT in the initial dispatch — only entity passes
    (those with empty depends_on) are dispatched by the dispatcher."""

    def test_dispatcher_does_not_dispatch_system_links_directly(
        self, db_session, pipeline_run_factory
    ):
        """derive_ontology_graph_pass.delay must only be called with entity pass
        names (no depends_on), never with 'system_links'."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        manifest = _fake_manifest(passes=[
            _fake_pass_def(name="radar_identity"),
            _fake_pass_def(name="radar_kinematics"),
            _fake_pass_def(name="system_links", depends_on=["radar_identity"]),
        ])

        with _patched_dispatcher(db_session, manifest=manifest) as mocks:
            _invoke(_make_task_self(), doc_id, str(run_id))

        delay_mock = mocks["derive_ontology_graph_pass_delay"]
        # Extract all pass_name arguments passed to .delay(doc_id, run_id, pass_name)
        dispatched_pass_names = [c.args[2] for c in delay_mock.call_args_list]

        assert "system_links" not in dispatched_pass_names, (
            f"system_links was dispatched directly but should not be: {dispatched_pass_names}"
        )
        # Only entity passes (no depends_on) should appear
        for pname in dispatched_pass_names:
            assert pname in {"radar_identity", "radar_kinematics"}, (
                f"Unexpected pass name in initial dispatch: {pname}"
            )


class TestDispatcherHandlesOrphanedRun:
    """Test 4: non-existent run_id → skipped response, no StageRun, no .delay()."""

    def test_dispatcher_handles_orphaned_run(self, db_session, pipeline_run_factory):
        """A run_id that doesn't exist in the DB → dispatcher returns
        {status: 'skipped', reason: 'orphaned_run'} without writing any
        StageRun or dispatching any tasks."""
        missing_run_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())

        with _patched_dispatcher(db_session) as mocks:
            result = _invoke(_make_task_self(), doc_id, missing_run_id)

        assert result["status"] == "skipped"
        assert result["reason"] == "orphaned_run"
        assert result["stage"] == "derive_ontology_graph"

        # No StageRun written
        row = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ingest.stage_runs "
                "WHERE stage_name = 'derive_ontology_graph' "
                "AND pipeline_run_id = :run_id"
            ),
            {"run_id": missing_run_id},
        ).scalar()
        assert row == 0, f"Unexpected StageRun rows: {row}"

        # No task dispatched
        delay_mock = mocks["derive_ontology_graph_pass_delay"]
        delay_mock.assert_not_called()


class TestDispatcherConcurrencyCap:
    """Test 5: dispatcher dispatches only pass_concurrency_per_document initial passes."""

    def test_dispatcher_dispatches_only_concurrency_cap_initial_passes(
        self, db_session, pipeline_run_factory
    ):
        """With 5 entity passes and pass_concurrency_per_document=2 (default),
        exactly 2 calls to derive_ontology_graph_pass.delay are made."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # 5 entity passes, no system_links
        manifest = _fake_manifest(passes=[
            _fake_pass_def(name="radar_identity"),
            _fake_pass_def(name="radar_kinematics"),
            _fake_pass_def(name="radar_power"),
            _fake_pass_def(name="radar_timing"),
            _fake_pass_def(name="radar_antenna"),
        ])

        from app.config import get_settings
        settings = get_settings()

        with _patched_dispatcher(db_session, manifest=manifest) as mocks:
            result = _invoke(_make_task_self(), doc_id, str(run_id))

        delay_mock = mocks["derive_ontology_graph_pass_delay"]
        expected_count = min(5, settings.pass_concurrency_per_document)
        assert delay_mock.call_count == expected_count, (
            f"Expected {expected_count} delay() calls (concurrency cap), "
            f"got {delay_mock.call_count}"
        )
        assert result["entity_passes_dispatched"] == expected_count

        dispatched_names = [c.args[2] for c in delay_mock.call_args_list]
        entity_passes = [
            "radar_identity", "radar_kinematics", "radar_power",
            "radar_timing", "radar_antenna",
        ]
        expected_first_n = entity_passes[: settings.pass_concurrency_per_document]
        assert dispatched_names == expected_first_n, (
            f"Expected first {settings.pass_concurrency_per_document} entity passes, "
            f"got {dispatched_names}"
        )


class TestDispatcherConcurrentRace:
    """Test 6: two concurrent dispatcher invocations for the same (run_id, attempt)
    must not crash with IntegrityError on uq_stage_runs_summary_row.

    Regression: prior to the ON CONFLICT DO NOTHING fix, the second copy would
    raise psycopg2.errors.UniqueViolation on the partial unique index.  Observed
    in Task 12 smoke test on run ea1df7dd."""

    def test_dispatcher_concurrent_dispatch_does_not_raise_integrity_error(
        self, db_session, pipeline_run_factory
    ):
        """Two dispatcher invocations for the same (run_id, attempt) must not
        crash. The second one's INSERT should ON CONFLICT DO NOTHING and the
        function should return cleanly. Regression: previously second crashed
        with IntegrityError on uq_stage_runs_summary_row."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # First dispatcher invocation — creates the RUNNING summary row
        with _patched_dispatcher(db_session) as mocks:
            result_1 = _invoke(_make_task_self(request_id="task-first"), doc_id, str(run_id))

        assert result_1["status"] == "dispatched"

        # Verify the summary row was created exactly once
        rows_after_first = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id "
                "  AND stage_name = 'derive_ontology_graph' "
                "  AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).scalar()
        assert rows_after_first == 1, f"Expected 1 summary row after first dispatch, got {rows_after_first}"

        # Second dispatcher invocation for the same (run, attempt=1) — must NOT raise
        with _patched_dispatcher(db_session) as mocks:
            result_2 = _invoke(_make_task_self(request_id="task-second"), doc_id, str(run_id))

        assert result_2["status"] == "dispatched"

        # Still only one row — ON CONFLICT DO NOTHING suppressed the duplicate
        rows_after_second = db_session.execute(
            text(
                "SELECT COUNT(*) FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id "
                "  AND stage_name = 'derive_ontology_graph' "
                "  AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).scalar()
        assert rows_after_second == 1, (
            f"Second dispatcher wrote a duplicate row (expected 1, got {rows_after_second})"
        )


class TestMergeDispatchQueueRouting:
    """Regression: merge dispatch must route explicitly to the 'graph' queue.

    Without queue='graph', celery_app.send_task falls back to the default
    'celery' queue. Multiple workers in this stack subscribe to 'celery'
    (worker, worker-ingest). If any subscriber holds a stale celery process
    (started before per-pass-fanin commits added derive_ontology_graph_merge),
    it grabs the message and acks-drops with KeyError — silently losing the
    task. The run then hangs until periodic_stale_run_sweep kills it ~hours
    later.

    Production incident 2026-05-08: doc 3ec6b236, runs 53bcff77, 698533c1, and
    0ae7912b all hung 1-10h until sweep. Worker-ingest-1 (Up 6 days, stale
    code) intercepted each merge dispatch:
        ERROR Received unregistered task of type
        'app.workers.pipeline.derive_ontology_graph_merge'
    """

    def test_try_advance_phase_send_task_specifies_graph_queue(self):
        """Static check: the send_task call in _try_advance_phase must
        include queue='graph'. Catches any future code change that drops
        the explicit queue argument."""
        import inspect
        from app.workers import pipeline

        src = inspect.getsource(pipeline._try_advance_phase)
        # The merge dispatch is the only send_task call in this function.
        # If the source ever stops calling send_task, this test should be
        # updated to reflect the new dispatch mechanism.
        assert "send_task" in src, (
            "_try_advance_phase no longer dispatches via send_task — "
            "update this test to match the new mechanism."
        )
        assert (
            'queue="graph"' in src or "queue='graph'" in src
        ), (
            "Merge dispatch via celery_app.send_task MUST include "
            "queue='graph' to prevent fallback to default 'celery' queue. "
            "Stale workers subscribed to 'celery' may ack-drop the task. "
            "See docstring above for production incident details."
        )
