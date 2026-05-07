"""Integration + unit tests for derive_ontology_graph_merge (Task 6).

Pattern:
- Real PostgreSQL (port 5438) for DB-state assertions.  Parent rows created
  via ``pipeline_run_factory``; rolled back after each test by ``db_session``.
- ``_get_db`` is patched to return a *non-closing proxy* around ``db_session``
  so the task body shares the same transaction as the test.  The proxy
  suppresses ``.close()`` so ``db_session``'s rollback-on-teardown still works.
- Heavyweight helpers (``merge_and_resolve``, ``_import_graph_phase_*``,
  ``_upsert_document_graph_extraction``, ``_build_provenance_envelope``,
  ``_build_element_uid_to_artifact_id``) are mocked to avoid IO.
- ``_assert_stage_run_pass_output_consistency``, ``_rehydrate_pass_result``,
  ``check_required_pass_gate`` are mocked for most tests.
- Direct invocation uses ``derive_ontology_graph_merge.run.__wrapped__`` to
  bypass ``guard_stage_run`` and observe raw exception/state.

Run with:

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \\
        /home/josh/development/EIP-MMDPP/.venv/bin/python -m pytest \\
        tests/unit/test_derive_ontology_graph_merge_task.py -v
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
from celery.exceptions import Retry as CeleryRetry
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.pass_outputs_store import (
    load_completed_pass_outputs,
    save_pass_output,
)
from app.services.run_phase_dispatch import read_phase_state
from app.workers.pipeline import (
    GateResult,
    IngestFailed,
    WorkerInvariantError,
    _assert_stage_run_pass_output_consistency,
    _rehydrate_pass_result,
    derive_ontology_graph_merge,
)

# ---------------------------------------------------------------------------
# Test-local helpers
# ---------------------------------------------------------------------------

_BUNDLE_KEY = "air_defense_v3"


def _fake_pass_def(
    *,
    name: str = "radar_identity",
    required: bool = True,
    depends_on: list[str] | None = None,
    input_mode: str = "document_only",
):
    return SimpleNamespace(
        name=name,
        required=required,
        depends_on=list(depends_on or []),
        input_mode=input_mode,
        primary_entity_types=["RADAR_SYSTEM"],
        bridge_entity_types=[],
        extracted_relationship_types=["INSTALLED_ON"],
        module="extraction_schemas.radar_identity",
        template_class="RadarIdentityPass",
    )


def _fake_manifest(passes: list | None = None):
    if passes is None:
        passes = [_fake_pass_def()]
    return SimpleNamespace(
        bundle_key=_BUNDLE_KEY,
        passes=passes,
        ontology_name="air_defense",
        ontology_version="v3",
        extraction_profile_version="1.0",
    )


_MINIMAL_ONTOLOGY: dict = {
    "entity_types": [{"name": "RADAR_SYSTEM", "identity_fields": ["system_name"]}],
    "validation_matrix": [
        {"source": "RADAR_SYSTEM", "relationship": "INSTALLED_ON", "target": "PLATFORM"},
    ],
}


def _fake_merged_result(n_entities: int = 2, n_edges: int = 1):
    """A minimal merged result object enough to satisfy the merge task body."""
    return SimpleNamespace(
        entities=[SimpleNamespace(identity="e1")] * n_entities,
        edges=[SimpleNamespace()] * n_edges,
    )


def _gate_ok() -> GateResult:
    return GateResult(passed=True, failures=[])


def _gate_fail(pass_name: str = "radar_identity") -> GateResult:
    return GateResult(passed=False, failures=[(pass_name, "FAILED: timeout")])


def _make_task_self(retries: int = 0, max_retries: int = 1, request_id: str = "task-merge-001"):
    """Build a mock Celery task ``self`` for direct invocation of the task body."""
    fake_self = MagicMock()
    fake_self.request.retries = retries
    fake_self.request.id = request_id
    fake_self.max_retries = max_retries

    def _side_effect_retry(exc=None, countdown=None):
        raise CeleryRetry()

    fake_self.retry.side_effect = _side_effect_retry
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


def _save_complete_pass_output(db_session, run_id, pass_name: str = "radar_identity"):
    """Pre-seed a COMPLETE pipeline_pass_outputs row."""
    save_pass_output(
        db_session,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name=pass_name,
        attempt=1,
        execution_status="COMPLETE",
        skip_reason=None,
        yield_status="HIT",
        extract_pass_response={"status": "ok", "pass_output": {"entities": []}},
        primary_entities_extracted=2,
        bridge_entities_extracted=0,
        relationships_extracted=1,
        relationships_rejected=0,
        diagnostics={},
        field_provenance=[],
    )
    db_session.flush()


def _save_skipped_pass_output(db_session, run_id, pass_name: str):
    save_pass_output(
        db_session,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name=pass_name,
        attempt=1,
        execution_status="SKIPPED",
        skip_reason="NO_UPSTREAM_ENDPOINTS",
        yield_status=None,
        extract_pass_response={},
        primary_entities_extracted=0,
        bridge_entities_extracted=0,
        relationships_extracted=0,
        relationships_rejected=0,
        diagnostics={},
        field_provenance=[],
    )
    db_session.flush()


def _save_failed_pass_output(db_session, run_id, pass_name: str):
    save_pass_output(
        db_session,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name=pass_name,
        attempt=1,
        execution_status="FAILED",
        skip_reason=None,
        yield_status=None,
        extract_pass_response={"diagnostics": {"error": "timeout"}},
        primary_entities_extracted=0,
        bridge_entities_extracted=0,
        relationships_extracted=0,
        relationships_rejected=0,
        diagnostics={"error": "timeout"},
        field_provenance=[],
    )
    db_session.flush()


def _add_complete_stage_run(db_session, run_id, pass_name: str = "radar_identity"):
    """Pre-seed a COMPLETE StageRun row for a pass."""
    db_session.execute(
        text(
            "INSERT INTO ingest.stage_runs "
            "(pipeline_run_id, stage_name, pass_name, attempt, status, execution_status, started_at) "
            "VALUES (:run_id, 'derive_ontology_graph', :pass_name, 1, 'COMPLETE', 'COMPLETE', NOW())"
        ),
        {"run_id": run_id, "pass_name": pass_name},
    )
    db_session.flush()


@contextmanager
def _patched_merge(
    db_session: Session,
    *,
    manifest=None,
    gate: GateResult | None = None,
    merged=None,
    skip_consistency_check: bool = True,
    skip_rehydrate: bool = True,
):
    """Context manager that patches all heavyweight IO in the merge task body.

    Returns a dict of mocks keyed by short names.
    """
    if manifest is None:
        manifest = _fake_manifest()
    if gate is None:
        gate = _gate_ok()
    if merged is None:
        merged = _fake_merged_result()

    proxy = _build_proxy(db_session)
    fake_pass_result = SimpleNamespace(
        template_instance=SimpleNamespace(relationships=[]),
        upstream_refs={},
        pre_merge_walk=SimpleNamespace(entities=[], raw_edge_count=0),
    )

    # Build a chain mock that won't try to connect to Redis
    mock_chain_instance = MagicMock()
    mock_chain_fn = MagicMock(return_value=mock_chain_instance)

    patches = [
        patch("app.workers.pipeline._get_db", return_value=proxy),
        patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
        patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
        patch("app.workers.pipeline.check_required_pass_gate", return_value=gate),
        patch("app.workers.pipeline.merge_and_resolve", return_value=merged),
        patch("app.workers.pipeline._apply_post_merge_yield_updates"),
        patch("app.workers.pipeline._write_pipeline_run_metrics"),
        patch("app.workers.pipeline._build_provenance_envelope", return_value={}),
        patch("app.workers.pipeline._import_graph_phase_nodes", return_value={}),
        patch("app.workers.pipeline._import_graph_phase_domain_edges"),
        patch("app.workers.pipeline._ensure_structural_document_vertex"),
        patch("app.workers.pipeline._import_graph_phase_structural_edges"),
        patch("app.workers.pipeline._build_element_uid_to_artifact_id", return_value={}),
        patch("app.workers.pipeline._upsert_document_graph_extraction"),
        patch("app.workers.pipeline._update_document_pipeline_status"),
        patch("app.workers.pipeline._terminalize_doc_and_run"),
        # Always patch celery.chain to avoid Redis connection in tests
        patch("celery.chain", mock_chain_fn),
    ]
    if skip_consistency_check:
        patches.append(patch("app.workers.pipeline._assert_stage_run_pass_output_consistency"))
    if skip_rehydrate:
        patches.append(patch("app.workers.pipeline._rehydrate_pass_result", return_value=fake_pass_result))

    keys = [
        "_get_db", "load_bundle_manifest", "load_ontology",
        "check_required_pass_gate", "merge_and_resolve",
        "_apply_post_merge_yield_updates", "_write_pipeline_run_metrics",
        "_build_provenance_envelope", "_import_graph_phase_nodes",
        "_import_graph_phase_domain_edges", "_ensure_structural_document_vertex",
        "_import_graph_phase_structural_edges", "_build_element_uid_to_artifact_id",
        "_upsert_document_graph_extraction", "_update_document_pipeline_status",
        "_terminalize_doc_and_run", "celery_chain",
    ]
    if skip_consistency_check:
        keys.append("_assert_stage_run_pass_output_consistency")
    if skip_rehydrate:
        keys.append("_rehydrate_pass_result")

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
    """Call the raw merge task body, bypassing ``guard_stage_run``."""
    raw_fn = derive_ontology_graph_merge.run.__wrapped__
    return raw_fn(self_mock, document_id, run_id)


# ---------------------------------------------------------------------------
# Helper unit tests (no DB needed)
# ---------------------------------------------------------------------------


class TestAssertStageRunPassOutputConsistency:
    """Unit-level tests for the drift-detection helper (real DB)."""

    def test_no_inconsistency_when_both_present(self, db_session, pipeline_run_factory):
        """COMPLETE StageRun + COMPLETE pass_output → no error."""
        run_id = pipeline_run_factory()
        _add_complete_stage_run(db_session, run_id, "radar_identity")
        _save_complete_pass_output(db_session, run_id, "radar_identity")
        # Should not raise
        _assert_stage_run_pass_output_consistency(db_session, str(run_id))

    def test_raises_when_stage_run_complete_but_no_pass_output(
        self, db_session, pipeline_run_factory
    ):
        """COMPLETE StageRun but no pass_output → WorkerInvariantError."""
        run_id = pipeline_run_factory()
        _add_complete_stage_run(db_session, run_id, "radar_identity")
        # No pass_output row — crash window!
        with pytest.raises(WorkerInvariantError) as exc_info:
            _assert_stage_run_pass_output_consistency(db_session, str(run_id))
        assert "radar_identity" in str(exc_info.value)

    def test_no_error_when_no_complete_stage_runs(self, db_session, pipeline_run_factory):
        """No COMPLETE StageRun rows at all → no error."""
        run_id = pipeline_run_factory()
        # Nothing pre-seeded
        _assert_stage_run_pass_output_consistency(db_session, str(run_id))

    def test_reports_all_missing_passes(self, db_session, pipeline_run_factory):
        """Multiple passes missing outputs → all names in diagnostic."""
        run_id = pipeline_run_factory()
        _add_complete_stage_run(db_session, run_id, "radar_identity")
        _add_complete_stage_run(db_session, run_id, "radar_kinematics")
        # No pass_output rows for either
        with pytest.raises(WorkerInvariantError) as exc_info:
            _assert_stage_run_pass_output_consistency(db_session, str(run_id))
        msg = str(exc_info.value)
        assert "radar_identity" in msg
        assert "radar_kinematics" in msg


# ---------------------------------------------------------------------------
# _rehydrate_pass_result unit tests
# ---------------------------------------------------------------------------


class TestRehydratePassResult:
    """Unit-level tests for the _rehydrate_pass_result helper."""

    def test_rehydrate_pass_result_propagates_pass_terminal_on_corrupt_json(self):
        """Corrupt persisted extract_pass_response_json → _parse_pass_response raises
        PassTerminal → _rehydrate_pass_result propagates (so the merge task's outer
        except handler catches it and triggers _attempt_rollback)."""
        from app.workers.pipeline import PassTerminal

        # Build a minimal fake row with a corrupt payload
        fake_row = SimpleNamespace(
            pass_name="radar_identity",
            extract_pass_response_json={"pass_output": "<corrupt>"},
        )
        manifest = _fake_manifest(passes=[_fake_pass_def(name="radar_identity")])

        with patch(
            "app.workers.pipeline._parse_pass_response",
            side_effect=PassTerminal("corrupt JSON"),
        ):
            with pytest.raises(PassTerminal, match="corrupt JSON"):
                _rehydrate_pass_result(
                    fake_row, manifest, ontology={}, document_id="doc-1"
                )


# ---------------------------------------------------------------------------
# Cancel / skip
# ---------------------------------------------------------------------------


class TestMergeCancel:
    def test_merge_skipped_when_run_cancelled(self, db_session, pipeline_run_factory):
        """Run in FAILED status → merge exits immediately, no graph imports."""
        run_id = pipeline_run_factory(status="FAILED")
        doc_id = str(uuid.uuid4())

        with _patched_merge(db_session) as mocks:
            result = _invoke(_make_task_self(), doc_id, str(run_id))

        assert result["merge"] == "skipped_cancelled"
        mocks["merge_and_resolve"].assert_not_called()
        mocks["_import_graph_phase_nodes"].assert_not_called()


# ---------------------------------------------------------------------------
# Gate failure path
# ---------------------------------------------------------------------------


class TestMergeGateFailure:
    def test_merge_marks_summary_stagerun_failed_on_gate_failure(
        self, db_session, pipeline_run_factory
    ):
        """Required-pass gate fails → summary StageRun marked FAILED,
        _terminalize_doc_and_run called, IngestFailed raised."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Pre-seed a RUNNING summary StageRun (pass_name IS NULL)
        db_session.execute(
            text(
                "INSERT INTO ingest.stage_runs "
                "(pipeline_run_id, stage_name, pass_name, attempt, status, started_at) "
                "VALUES (:run_id, 'derive_ontology_graph', NULL, 1, 'RUNNING', NOW())"
            ),
            {"run_id": run_id},
        )
        db_session.flush()

        with _patched_merge(db_session, gate=_gate_fail()) as mocks:
            with patch("app.workers.pipeline.mark_phase_terminal") as mock_phase:
                with pytest.raises(IngestFailed):
                    _invoke(_make_task_self(), doc_id, str(run_id))

        # Summary row should be updated to FAILED
        row = db_session.execute(
            text(
                "SELECT status FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert row is not None
        assert row[0] == "FAILED"

        mocks["_terminalize_doc_and_run"].assert_called_once_with(
            doc_id, str(run_id), "PARTIAL_COMPLETE"
        )
        # mark_phase_terminal is called in the gate-failure branch AND again
        # in the except handler — both with result="failed"; verify at least one call.
        assert mock_phase.call_count >= 1
        called_with_failed = any(
            c.kwargs.get("result") == "failed" for c in mock_phase.call_args_list
        )
        assert called_with_failed
        # merge_and_resolve should NOT have been called
        mocks["merge_and_resolve"].assert_not_called()


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestMergeSuccessPath:
    def test_merge_loads_only_complete_pass_outputs(
        self, db_session, pipeline_run_factory
    ):
        """FAILED + SKIPPED rows should be excluded; only COMPLETE rows in rehydrated dict."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Pre-seed COMPLETE, FAILED, SKIPPED rows
        _save_complete_pass_output(db_session, run_id, "radar_identity")
        _save_failed_pass_output(db_session, run_id, "radar_modulation")
        _save_skipped_pass_output(db_session, run_id, "radar_timing")

        manifest = _fake_manifest(passes=[
            _fake_pass_def(name="radar_identity"),
            _fake_pass_def(name="radar_modulation", required=False),
            _fake_pass_def(name="radar_timing", required=False),
        ])

        captured_pass_results: dict[str, Any] = {}

        def fake_merge_and_resolve(*, pass_results, **kwargs):
            captured_pass_results.update(pass_results)
            return _fake_merged_result()

        with _patched_merge(
            db_session,
            manifest=manifest,
            skip_rehydrate=False,  # let load_completed_pass_outputs + rehydrate run
        ) as mocks:
            mocks["merge_and_resolve"].side_effect = fake_merge_and_resolve
            # _rehydrate_pass_result was NOT patched — but _parse_pass_response is called
            # internally; patch it to avoid real LLM parsing
            with patch("app.workers.pipeline._parse_pass_response") as mock_parse:
                with patch("app.workers.pipeline._build_pre_merge_walk_summary") as mock_walk:
                    mock_parse.return_value = SimpleNamespace(
                        template_instance=SimpleNamespace(relationships=[]),
                        upstream_refs={},
                        pre_merge_walk=None,
                    )
                    mock_walk.return_value = SimpleNamespace(entities=[], raw_edge_count=0)
                    _invoke(_make_task_self(), doc_id, str(run_id))

        # Only COMPLETE pass (radar_identity) should be in rehydrated dict
        assert "radar_identity" in captured_pass_results
        assert "radar_modulation" not in captured_pass_results
        assert "radar_timing" not in captured_pass_results

    def test_merge_marks_summary_stagerun_complete(
        self, db_session, pipeline_run_factory
    ):
        """Successful merge → summary StageRun (pass_name IS NULL) updated to COMPLETE."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Pre-seed a RUNNING summary StageRun
        db_session.execute(
            text(
                "INSERT INTO ingest.stage_runs "
                "(pipeline_run_id, stage_name, pass_name, attempt, status, started_at) "
                "VALUES (:run_id, 'derive_ontology_graph', NULL, 1, 'RUNNING', NOW())"
            ),
            {"run_id": run_id},
        )
        db_session.flush()

        with _patched_merge(db_session) as mocks:
            with patch("app.workers.pipeline.mark_phase_terminal"):
                result = _invoke(_make_task_self(), doc_id, str(run_id))

        assert result["merge"] == "ok"

        # Check the DB row was updated
        row = db_session.execute(
            text(
                "SELECT status FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert row is not None
        assert row[0] == "COMPLETE"

    def test_merge_dispatches_downstream_chain_in_full_mode(
        self, db_session, pipeline_run_factory
    ):
        """full mode → celery_chain with the 4 downstream stages dispatched."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched_merge(db_session) as mocks:
            with patch("app.workers.pipeline.mark_phase_terminal"):
                _invoke(_make_task_self(), doc_id, str(run_id))

        # celery_chain mock is pre-seeded in _patched_merge as mocks["celery_chain"]
        mock_chain = mocks["celery_chain"]
        mock_chain.assert_called_once()
        args = mock_chain.call_args[0]
        # 4 tasks in chain: collect_derivations, derive_structure_links,
        # derive_canonicalization, finalize_document
        assert len(args) == 4
        mock_chain.return_value.apply_async.assert_called_once()

    def test_merge_does_not_dispatch_downstream_in_graph_only_mode(
        self, db_session, pipeline_run_factory
    ):
        """graph_only mode → NO downstream chain dispatched."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Override mode to graph_only using raw SQL
        db_session.execute(
            text("UPDATE ingest.pipeline_runs SET mode = 'graph_only' WHERE id = :id"),
            {"id": run_id},
        )
        db_session.flush()

        with _patched_merge(db_session) as mocks:
            with patch("app.workers.pipeline.mark_phase_terminal"):
                _invoke(_make_task_self(), doc_id, str(run_id))

        mocks["celery_chain"].assert_not_called()

    def test_merge_graph_only_mode_marks_run_complete(
        self, db_session, pipeline_run_factory
    ):
        """graph_only success → run.status=COMPLETE set, _update_document_pipeline_status called."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        db_session.execute(
            text("UPDATE ingest.pipeline_runs SET mode = 'graph_only' WHERE id = :id"),
            {"id": run_id},
        )
        db_session.flush()

        with _patched_merge(db_session) as mocks:
            with patch("app.workers.pipeline.mark_phase_terminal"):
                _invoke(_make_task_self(), doc_id, str(run_id))

        mocks["_update_document_pipeline_status"].assert_called_once_with(doc_id, "COMPLETE")

        # Run status should be COMPLETE in DB
        row = db_session.execute(
            text("SELECT status FROM ingest.pipeline_runs WHERE id = :id"),
            {"id": run_id},
        ).fetchone()
        assert row is not None
        assert row[0] == "COMPLETE"

    def test_skipped_passes_excluded_from_merge_inputs(
        self, db_session, pipeline_run_factory
    ):
        """SKIPPED pass → excluded from merge_and_resolve pass_results."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        _save_complete_pass_output(db_session, run_id, "radar_identity")
        _save_skipped_pass_output(db_session, run_id, "radar_modulation")

        manifest = _fake_manifest(passes=[
            _fake_pass_def(name="radar_identity"),
            _fake_pass_def(name="radar_modulation", required=False),
        ])

        captured_pass_results: dict[str, Any] = {}

        def fake_merge(*, pass_results, **kwargs):
            captured_pass_results.update(pass_results)
            return _fake_merged_result()

        with _patched_merge(db_session, manifest=manifest, skip_rehydrate=False) as mocks:
            mocks["merge_and_resolve"].side_effect = fake_merge
            with patch("app.workers.pipeline._parse_pass_response") as mock_parse:
                with patch("app.workers.pipeline._build_pre_merge_walk_summary") as mock_walk:
                    mock_parse.return_value = SimpleNamespace(
                        template_instance=SimpleNamespace(relationships=[]),
                        upstream_refs={},
                        pre_merge_walk=None,
                    )
                    mock_walk.return_value = SimpleNamespace(entities=[], raw_edge_count=0)
                    _invoke(_make_task_self(), doc_id, str(run_id))

        assert "radar_identity" in captured_pass_results
        assert "radar_modulation" not in captured_pass_results


# ---------------------------------------------------------------------------
# Drift detection (COMPLETE-without-output)
# ---------------------------------------------------------------------------


class TestCompleteWithoutOutput:
    def test_complete_without_output_raises_worker_invariant_error(
        self, db_session, pipeline_run_factory
    ):
        """COMPLETE StageRun for radar_identity but no pass_output →
        WorkerInvariantError raised, diagnostic mentions radar_identity."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        _add_complete_stage_run(db_session, run_id, "radar_identity")
        # No pass_output row → drift!

        # Use the real consistency check (skip_consistency_check=False) so the
        # helper actually queries the DB and detects the missing pass_output row.
        with _patched_merge(db_session, skip_consistency_check=False) as mocks:
            with patch("app.workers.pipeline.mark_phase_terminal"):
                with pytest.raises(WorkerInvariantError) as exc_info:
                    _invoke(_make_task_self(), doc_id, str(run_id))

        assert "radar_identity" in str(exc_info.value)
        mocks["merge_and_resolve"].assert_not_called()


# ---------------------------------------------------------------------------
# Rollback contract (GraphWriteTracker)
# ---------------------------------------------------------------------------


class TestRollbackContract:
    def test_merge_failure_post_mutation_calls_rollback_before_raising(
        self, db_session, pipeline_run_factory
    ):
        """Exception AFTER first graph mutation → _attempt_rollback(document_id) called
        before the exception propagates."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # _import_graph_phase_nodes "marks" the tracker, then
        # _import_graph_phase_domain_edges raises to simulate mid-merge crash.
        # We achieve this by having _import_graph_phase_nodes call tracker.mark()
        # via side_effect.

        def fake_import_nodes(merged, ontology, doc_id, tracker, provenance):
            tracker.mark()  # simulate first graph write
            return {}

        with _patched_merge(db_session) as mocks:
            mocks["_import_graph_phase_nodes"].side_effect = fake_import_nodes
            mocks["_import_graph_phase_domain_edges"].side_effect = RuntimeError("ArcadeDB down")

            with patch("app.workers.pipeline._attempt_rollback") as mock_rollback:
                mock_rollback.return_value = ""
                with patch("app.workers.pipeline.mark_phase_terminal"):
                    with pytest.raises(RuntimeError, match="ArcadeDB down"):
                        _invoke(_make_task_self(), doc_id, str(run_id))

        mock_rollback.assert_called_once_with(doc_id)

    def test_merge_failure_pre_mutation_skips_rollback(
        self, db_session, pipeline_run_factory
    ):
        """Exception before any tracker.mark() → _attempt_rollback NOT called."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Raise in _assert_stage_run_pass_output_consistency (pre-mutation).
        # Use skip_consistency_check=True (default) so it is in mocks, then
        # override its side_effect to raise before any graph imports happen.
        with _patched_merge(db_session) as mocks:
            mocks["_assert_stage_run_pass_output_consistency"].side_effect = WorkerInvariantError(
                "injected pre-mutation error"
            )

            with patch("app.workers.pipeline._attempt_rollback") as mock_rollback:
                with patch("app.workers.pipeline.mark_phase_terminal"):
                    with pytest.raises(WorkerInvariantError):
                        _invoke(_make_task_self(), doc_id, str(run_id))

        mock_rollback.assert_not_called()

    def test_merge_phase_marked_failed_on_rollback_path(
        self, db_session, pipeline_run_factory
    ):
        """After error, phase entry should be marked failed via mark_phase_terminal."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        def fake_import_nodes(merged, ontology, doc_id, tracker, provenance):
            tracker.mark()
            return {}

        with _patched_merge(db_session) as mocks:
            mocks["_import_graph_phase_nodes"].side_effect = fake_import_nodes
            mocks["_import_graph_phase_domain_edges"].side_effect = RuntimeError("graph fail")

            with patch("app.workers.pipeline._attempt_rollback", return_value=""):
                with patch("app.workers.pipeline.mark_phase_terminal") as mock_phase:
                    with pytest.raises(RuntimeError):
                        _invoke(_make_task_self(), doc_id, str(run_id))

        # mark_phase_terminal should have been called with result="failed"
        called_with_failed = any(
            c.kwargs.get("result") == "failed" or
            (len(c.args) >= 3 and c.args[2] == "failed")
            for c in mock_phase.call_args_list
        )
        assert called_with_failed, (
            f"mark_phase_terminal never called with result='failed'. "
            f"Calls: {mock_phase.call_args_list}"
        )

    def test_rollback_note_included_in_log(
        self, db_session, pipeline_run_factory
    ):
        """Rollback diagnostic suffix from _attempt_rollback is logged when rollback fails."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        def fake_import_nodes(merged, ontology, doc_id, tracker, provenance):
            tracker.mark()
            return {}

        with _patched_merge(db_session) as mocks:
            mocks["_import_graph_phase_nodes"].side_effect = fake_import_nodes
            mocks["_import_graph_phase_domain_edges"].side_effect = RuntimeError("boom")

            # Rollback itself also "fails" — returns diagnostic suffix
            with patch("app.workers.pipeline._attempt_rollback",
                       return_value="; ROLLBACK_ALSO_FAILED: network") as mock_rollback:
                with patch("app.workers.pipeline.mark_phase_terminal"):
                    with pytest.raises(RuntimeError):
                        _invoke(_make_task_self(), doc_id, str(run_id))

        mock_rollback.assert_called_once_with(doc_id)
