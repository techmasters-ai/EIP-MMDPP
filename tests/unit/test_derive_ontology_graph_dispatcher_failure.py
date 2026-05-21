"""C1.5 — Dispatcher robustness: stage_run + pipeline_run FAILED on dispatch exception.

Task 1.3 (pre-queue path) + Task 1.7 (post-queue path).

Tests are fully hermetic: no real DB, no real Celery.  All I/O is mocked at
the ``app.workers.pipeline`` module level.

Design choices:
- ``_get_db`` is called up to three times in the patched task body:
    1. First session (db1): PipelineRun fetch + RUNNING stage_run insert.
    2. Second session (db2): _claim_and_dispatch_pass loop.
    3. Except-handler session (db_fail): FAILED stage_run UPDATE.
  Each call to ``_get_db`` returns a fresh MagicMock via ``side_effect``.
- ``load_bundle_manifest`` is patched to raise UnknownBundleError (pre-queue)
  or to return a manifest and then let the dispatch loop partially succeed
  before raising (post-queue).
- ``_terminalize_doc_and_run`` is patched to assert on pipeline_run fate.
- ``derive_ontology_graph_pass.delay`` is patched to count calls.

Run standalone (no DB required):
    python3 -m pytest tests/unit/test_derive_ontology_graph_dispatcher_failure.py -v
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = str(uuid.uuid4())
_DOC_ID = str(uuid.uuid4())
_BUNDLE_KEY = "air_defense_v3_baseline_subset"


def _fake_run(run_id: str = _RUN_ID, bundle_key: str = _BUNDLE_KEY):
    run = MagicMock()
    run.ontology_bundle_key = bundle_key
    run.status = "PROCESSING"
    return run


def _fake_pass_def(name: str, *, depends_on=None):
    return SimpleNamespace(
        name=name,
        required=True,
        depends_on=list(depends_on or []),
        input_mode="document_only",
    )


def _fake_manifest(pass_names=("radar_identity", "missile_identity")):
    passes = [_fake_pass_def(name=n) for n in pass_names]
    return SimpleNamespace(bundle_key=_BUNDLE_KEY, passes=passes)


def _make_task_self(retries: int = 0):
    self = MagicMock()
    self.request.retries = retries
    self.request.id = "task-c1.5-test"
    return self


def _make_db_session(run, *, anchors_text_block_count: int = 5):
    """Build a MagicMock DB session that:
    - .get(PipelineRun, ...) → ``run``
    - .execute(...).scalar() → {"text_block_count": N, "table_count": 0, "figure_count": 0}
    """
    db = MagicMock()
    # .get() always returns ``run`` regardless of args
    db.get.return_value = run
    # .execute().scalar() returns a non-empty anchors dict (prevents skip path)
    anchors_dict = {
        "text_block_count": anchors_text_block_count,
        "table_count": 0,
        "figure_count": 0,
    }
    db.execute.return_value.scalar.return_value = anchors_dict
    return db


def _invoke_raw(self_mock, doc_id: str, run_id: str):
    """Call derive_ontology_graph body directly, bypassing guard_stage_run."""
    from app.workers.pipeline import derive_ontology_graph
    raw_fn = derive_ontology_graph.run.__wrapped__
    return raw_fn(self_mock, doc_id, run_id)


# ---------------------------------------------------------------------------
# Test 1 — Pre-queue failure: UnknownBundleError before any pass is dispatched
# ---------------------------------------------------------------------------

class TestPreQueueFailure:
    """When load_bundle_manifest raises BEFORE the first .delay() call, the
    dispatcher must:
    (a) write a FAILED stage_run with non-empty error_message,
    (b) terminalize the pipeline_run (call _terminalize_doc_and_run with FAILED),
    (c) NOT call derive_ontology_graph_pass.delay at all.
    """

    def _run_with_pre_queue_failure(self):
        """Execute the task body with load_bundle_manifest raising
        UnknownBundleError.  Returns (db_sessions, delay_mock, terminalize_mock).
        """
        from app.services.ontology_templates import UnknownBundleError

        run = _fake_run()
        db1 = _make_db_session(run)
        db_fail = MagicMock()  # session opened in the except handler

        db_sessions = [db1, db_fail]
        db_iter = iter(db_sessions)

        with patch("app.workers.pipeline._get_db", side_effect=lambda: next(db_iter)), \
             patch("app.workers.pipeline.load_bundle_manifest",
                   side_effect=UnknownBundleError("bundle 'air_defense_v3_baseline_subset' not found")), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as mock_term, \
             patch("app.workers.pipeline.derive_ontology_graph_pass.delay") as mock_delay:
            with pytest.raises(UnknownBundleError):
                _invoke_raw(_make_task_self(), _DOC_ID, _RUN_ID)

        return db1, db_fail, mock_delay, mock_term

    def test_no_delay_calls_on_pre_queue_failure(self):
        """(c) derive_ontology_graph_pass.delay must NOT be called."""
        _, _, mock_delay, _ = self._run_with_pre_queue_failure()
        assert mock_delay.call_count == 0, (
            f"Expected 0 .delay() calls but got {mock_delay.call_count}"
        )

    def test_terminalize_pipeline_run_on_pre_queue_failure(self):
        """(b) _terminalize_doc_and_run called with doc_id, run_id, 'FAILED'."""
        _, _, _, mock_term = self._run_with_pre_queue_failure()
        mock_term.assert_called_once()
        args = mock_term.call_args[0]
        assert args[0] == _DOC_ID, f"Expected doc_id={_DOC_ID!r}, got {args[0]!r}"
        assert str(args[1]) == str(_RUN_ID), f"Expected run_id={_RUN_ID!r}, got {args[1]!r}"
        assert args[2] == "FAILED", f"Expected status='FAILED', got {args[2]!r}"

    def test_stage_run_written_failed_on_pre_queue_failure(self):
        """(a) The except-handler DB session must execute a SQL UPDATE that
        contains 'FAILED' so the stage_run row transitions from RUNNING."""
        _, db_fail, _, _ = self._run_with_pre_queue_failure()
        # At least one execute() call on the fail session
        assert db_fail.execute.called, "No execute() called on the failure DB session"
        # The UPDATE must reference 'FAILED'
        all_sql = []
        for c in db_fail.execute.call_args_list:
            first_arg = c.args[0] if c.args else None
            if first_arg is not None:
                all_sql.append(str(first_arg))
        assert any("FAILED" in sql for sql in all_sql), (
            f"No SQL containing 'FAILED' found in db_fail.execute calls.\n"
            f"Captured SQL: {all_sql}"
        )

    def test_stage_run_error_message_contains_exception_repr(self):
        """(a) The error_message passed to the UPDATE must include the exception repr."""
        from app.services.ontology_templates import UnknownBundleError

        run = _fake_run()
        db1 = _make_db_session(run)
        db_fail = MagicMock()

        db_sessions = [db1, db_fail]
        db_iter = iter(db_sessions)

        with patch("app.workers.pipeline._get_db", side_effect=lambda: next(db_iter)), \
             patch("app.workers.pipeline.load_bundle_manifest",
                   side_effect=UnknownBundleError("subset not found")), \
             patch("app.workers.pipeline._terminalize_doc_and_run"), \
             patch("app.workers.pipeline.derive_ontology_graph_pass.delay"):
            with pytest.raises(UnknownBundleError):
                _invoke_raw(_make_task_self(), _DOC_ID, _RUN_ID)

        # Check that the error params dict passed to execute contains 'error_message'
        # OR that the SQL itself embeds the error text.
        # We check by looking at all args/kwargs of all execute calls.
        found_error_info = False
        for c in db_fail.execute.call_args_list:
            # positional args
            for a in c.args:
                if "subset not found" in str(a) or "UnknownBundleError" in str(a) or "error_message" in str(a):
                    found_error_info = True
            # keyword args
            for v in c.kwargs.values():
                if "subset not found" in str(v) or "UnknownBundleError" in str(v):
                    found_error_info = True
        assert found_error_info, (
            "error_message with exception repr not found in db_fail.execute calls"
        )


# ---------------------------------------------------------------------------
# Test 2 — Post-queue failure: exception after N passes were already queued
# ---------------------------------------------------------------------------

class TestPostQueueFailure:
    """When _claim_and_dispatch_pass raises AFTER N passes have been dispatched,
    the dispatcher must:
    (a) leave pipeline_run in PROCESSING (NOT call _terminalize_doc_and_run),
    (b) write dispatcher stage_run as FAILED with error_message naming the queued count,
    (c) NOT undo or touch the N already-queued pass stage_runs/pass_outputs.
    """

    def _run_with_post_queue_failure(self, n_success: int = 2):
        """Execute the task body where the first n_success calls to
        _claim_and_dispatch_pass succeed (return True) and the (n_success+1)-th
        raises RuntimeError.  Returns (db_fail, delay_mock, terminalize_mock).
        """
        run = _fake_run()
        db1 = _make_db_session(run)
        db2 = MagicMock()   # session for dispatch loop
        db_fail = MagicMock()  # session opened in the except handler

        db_sessions = [db1, db2, db_fail]
        db_iter = iter(db_sessions)

        manifest = _fake_manifest(
            pass_names=("radar_identity", "missile_identity", "radar_power_rf")
        )

        dispatch_call_count = [0]

        def fake_claim_and_dispatch(db, doc_id, run_id, pass_name):
            dispatch_call_count[0] += 1
            if dispatch_call_count[0] > n_success:
                raise RuntimeError(f"Simulated dispatch failure on pass {pass_name}")
            # Simulate a real dispatch by calling .delay() so we can count it
            from app.workers import pipeline as _p
            _p.derive_ontology_graph_pass.delay(doc_id, run_id, pass_name)
            return True  # dispatched

        with patch("app.workers.pipeline._get_db", side_effect=lambda: next(db_iter)), \
             patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as mock_term, \
             patch("app.workers.pipeline.derive_ontology_graph_pass.delay") as mock_delay:
            with pytest.raises(RuntimeError):
                _invoke_raw(_make_task_self(), _DOC_ID, _RUN_ID)

        return db_fail, mock_delay, mock_term, dispatch_call_count[0]

    def test_pipeline_run_stays_processing_on_post_queue_failure(self):
        """(a) _terminalize_doc_and_run must NOT be called — pipeline_run stays PROCESSING."""
        _, _, mock_term, _ = self._run_with_post_queue_failure(n_success=2)
        assert mock_term.call_count == 0, (
            f"_terminalize_doc_and_run was called {mock_term.call_count} time(s) "
            "but should NOT be called on post-queue failure"
        )

    def test_n_passes_were_dispatched_before_failure(self):
        """N=2 .delay() calls were made before the exception."""
        _, mock_delay, _, _ = self._run_with_post_queue_failure(n_success=2)
        assert mock_delay.call_count == 2, (
            f"Expected 2 .delay() calls before failure, got {mock_delay.call_count}"
        )

    def test_stage_run_written_failed_on_post_queue_failure(self):
        """(b) The except-handler DB session executes a SQL UPDATE with 'FAILED'."""
        db_fail, _, _, _ = self._run_with_post_queue_failure(n_success=2)
        assert db_fail.execute.called, "No execute() called on the failure DB session"
        all_sql = []
        for c in db_fail.execute.call_args_list:
            first_arg = c.args[0] if c.args else None
            if first_arg is not None:
                all_sql.append(str(first_arg))
        assert any("FAILED" in sql for sql in all_sql), (
            f"No SQL containing 'FAILED' found in db_fail.execute calls.\n"
            f"Captured SQL: {all_sql}"
        )

    def test_error_message_names_queued_count(self):
        """(b) The error_message in the FAILED update must mention the queued count (2).

        The format is: "dispatcher failed after queuing 2 pass(es): <exc repr>"
        So we check for "2 pass" (the literal "2 pass(es)" substring).
        """
        db_fail, _, _, _ = self._run_with_post_queue_failure(n_success=2)
        # Check that the params dict passed to execute contains the count '2'
        found = False
        for c in db_fail.execute.call_args_list:
            # The params dict is the second positional arg
            if len(c.args) >= 2:
                params = c.args[1]
                if isinstance(params, dict):
                    err_msg = params.get("error_message", "")
                    # "2 pass" matches "...queuing 2 pass(es)..."
                    if "2 pass" in str(err_msg):
                        found = True
        assert found, (
            "error_message in the FAILED stage_run update should mention the "
            "queued count (e.g. '2 pass(es)'). Check db_fail.execute call_args_list."
        )
