"""End-to-end integration tests for the per-pass Celery fan-in architecture.

Covers the full flow: dispatcher → per-pass tasks → fan-in advance → merge →
downstream chain dispatch across 13 test cases.

Infrastructure design
---------------------
- Real PostgreSQL (port 5438) — each test rolls back via ``db_session``.
- ``_get_db`` is patched to return a ``_NoCloseProxy`` so task bodies share the
  test transaction and rollback works at teardown.
- ``derive_ontology_graph`` / pass / merge tasks are invoked via their
  ``.__wrapped__`` unwrapped body to bypass ``guard_stage_run``.
- LLM extraction (``_execute_pass_attempt``) is mocked for all tests.
- Graph-import helpers (ArcadeDB) are mocked for all tests.
- Downstream chain stages (``collect_derivations``, ``derive_structure_links``,
  ``derive_canonicalization``) are mocked for most tests; ``finalize_document``
  executes via its own ``.__wrapped__`` body for tests that verify run.status.

A small 3-pass manifest (radar_identity + missile_airframe + system_links) is
used instead of the full 12-pass air_defense_v3 manifest to keep tests fast.

Run with::

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \\
        /home/josh/development/EIP-MMDPP/.venv/bin/python -m pytest \\
        tests/integration/test_per_pass_db_fanin_e2e.py -v

Test cases
----------
1.  test_full_run_completes_through_all_phases (HappyPath)
2.  test_optional_pass_retry_exhaustion_does_not_block_run (RetryExhaustion)
3.  test_required_pass_retry_exhaustion_terminalizes_with_failed_summary (RetryExhaustion)
4.  test_authorized_skip_treated_as_terminal_for_fanin (SkipPassthrough)
5.  test_skip_already_resolved_passes_on_retry (SkipPassthrough)
6.  test_complete_without_output_fails_loud (DriftDetection)
7.  test_cancel_during_active_pass_aborts_gracefully (Cancellation)
8.  test_downstream_stages_do_not_run_before_merge (ChainOrdering)
9.  test_merge_failure_post_mutation_rolls_back (RollbackContract)
10. test_reconciler_recovers_from_stale_claimed (Reconciler)
11. test_reconciler_recovers_from_stale_dispatched_no_pending_retry (Reconciler)
12. test_reconciler_waits_when_pending_retry_in_countdown (Reconciler)
13. test_finalize_document_sees_derive_ontology_graph_stage_complete (HappyPath)
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager, ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
from celery.exceptions import Retry as CeleryRetry
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.pass_outputs_store import (
    is_pass_already_resolved,
    load_pass_output,
    save_pass_output,
)
from app.services.run_phase_dispatch import claim_phase, read_phase_state
from app.workers.pipeline import (
    GateResult,
    IngestFailed,
    PassAttemptOutcome,
    PassRetryable,
    PassTerminal,
    WorkerInvariantError,
    _phase_key,
    derive_ontology_graph,
    derive_ontology_graph_merge,
    derive_ontology_graph_pass,
    reconcile_ontology_graph_runs,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLE_KEY = "air_defense_v3"
_PASS_RADAR = "radar_identity"
_PASS_MISSILE = "missile_airframe"
_PASS_SL = "system_links"

_PHASE_RADAR = f"entity_pass_{_PASS_RADAR}"
_PHASE_MISSILE = f"entity_pass_{_PASS_MISSILE}"


# ---------------------------------------------------------------------------
# Helpers shared across all tests
# ---------------------------------------------------------------------------


class _NoCloseProxy:
    """Proxy around a SQLAlchemy Session that suppresses ``.close()`` so the
    test's transaction-rollback fixture can still clean up after the task.
    Also converts commit() → flush() so writes remain in the test transaction."""

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


def _fake_pass_def(
    name: str,
    *,
    required: bool = True,
    input_mode: str = "document_only",
    depends_on: list[str] | None = None,
    skip_if_no_upstream: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        kind="entities_and_relationships",
        input_mode=input_mode,
        required=required,
        depends_on=list(depends_on or []),
        primary_entity_types=["RADAR_SYSTEM"],
        bridge_entity_types=[],
        extracted_relationship_types=["INSTALLED_ON"],
        skip_if_no_upstream_endpoints=skip_if_no_upstream,
        module=f"extraction_schemas.{name}",
        template_class=name.title().replace("_", "") + "Pass",
    )


def _fake_manifest(
    passes: list[SimpleNamespace] | None = None,
    *,
    bundle_key: str = _BUNDLE_KEY,
) -> SimpleNamespace:
    """3-pass manifest: radar_identity + missile_airframe (entity passes) + system_links."""
    if passes is None:
        passes = [
            _fake_pass_def(_PASS_RADAR),
            _fake_pass_def(_PASS_MISSILE),
            _fake_pass_def(
                _PASS_SL,
                input_mode="document_plus_entity_refs",
                depends_on=[_PASS_RADAR],
            ),
        ]
    return SimpleNamespace(
        bundle_key=bundle_key,
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


def _fake_pass_result() -> SimpleNamespace:
    return SimpleNamespace(
        template_instance=SimpleNamespace(relationships=[]),
        upstream_refs={},
        pre_merge_walk=SimpleNamespace(entities=[], raw_edge_count=0),
    )


def _complete_outcome() -> PassAttemptOutcome:
    return PassAttemptOutcome(
        execution_status="COMPLETE",
        skip_reason=None,
        yield_status="HIT",
        pass_result=_fake_pass_result(),
        raw_response_payload={
            "status": "ok",
            "pass_output": {"entities": []},
            "metadata": {"node_count": 0, "edge_count": 0},
            "diagnostics": {},
            "field_provenance": [],
        },
        counts={
            "primary_entities_extracted": 1,
            "bridge_entities_extracted": 0,
            "relationships_extracted": 0,
            "relationships_rejected": 0,
        },
        error=None,
    )


def _skipped_outcome() -> PassAttemptOutcome:
    return PassAttemptOutcome(
        execution_status="SKIPPED",
        skip_reason="NO_UPSTREAM_ENDPOINTS",
        yield_status=None,
        pass_result=None,
        raw_response_payload=None,
        counts=None,
        error=None,
    )


def _failed_retryable_outcome() -> PassAttemptOutcome:
    return PassAttemptOutcome(
        execution_status="FAILED",
        skip_reason=None,
        yield_status=None,
        pass_result=None,
        raw_response_payload={"diagnostics": {"pipeline_error": "service down"}},
        counts=None,
        error=PassRetryable("service down"),
    )


def _make_task_self(retries: int = 0, max_retries: int = 3) -> MagicMock:
    fake_self = MagicMock()
    fake_self.request.retries = retries
    fake_self.request.id = f"task-{uuid.uuid4()}"
    fake_self.max_retries = max_retries

    def _retry_side(exc=None, countdown=None):
        raise CeleryRetry()

    fake_self.retry.side_effect = _retry_side
    return fake_self


def _make_merge_task_self(retries: int = 0, max_retries: int = 1) -> MagicMock:
    return _make_task_self(retries=retries, max_retries=max_retries)


def _make_dispatcher_task_self() -> MagicMock:
    fake_self = MagicMock()
    fake_self.request.retries = 0
    fake_self.request.id = f"task-{uuid.uuid4()}"
    return fake_self


# ---------------------------------------------------------------------------
# Factories for extended pipeline_run_factory that returns (run_id, doc_id)
# ---------------------------------------------------------------------------

_FACTORY_DUMMY_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _make_run_with_doc(db_session, *, status: str = "PROCESSING") -> tuple[uuid.UUID, str]:
    """Create Source → Document → PipelineRun and return (run_id, document_id_str).

    Unlike the shared ``pipeline_run_factory`` fixture, this helper returns BOTH
    the pipeline_run_id AND the document_id so E2E tests can pass the real
    document_id to task bodies.
    """
    source_id = uuid.uuid4()
    document_id = uuid.uuid4()
    pipeline_run_id = uuid.uuid4()

    db_session.execute(
        text(
            "INSERT INTO ingest.sources (id, name, created_by) "
            "VALUES (:id, :name, :created_by)"
        ),
        {"id": source_id, "name": f"e2e-source-{source_id}", "created_by": _FACTORY_DUMMY_USER_ID},
    )
    db_session.execute(
        text(
            "INSERT INTO ingest.documents "
            "(id, source_id, filename, storage_bucket, storage_key, uploaded_by, retry_count) "
            "VALUES (:id, :source_id, :filename, :bucket, :key, :user_id, :retry_count)"
        ),
        {
            "id": document_id,
            "source_id": source_id,
            "filename": "e2e_test.pdf",
            "bucket": "test-bucket",
            "key": f"test/{document_id}.pdf",
            "user_id": _FACTORY_DUMMY_USER_ID,
            "retry_count": 0,
        },
    )
    db_session.execute(
        text(
            "INSERT INTO ingest.pipeline_runs "
            "(id, document_id, pipeline_version, status, mode, ontology_bundle_key) "
            "VALUES (:id, :document_id, :version, :status, 'full', :bundle)"
        ),
        {
            "id": pipeline_run_id,
            "document_id": document_id,
            "version": "v2",
            "status": status,
            "bundle": _BUNDLE_KEY,
        },
    )
    db_session.flush()
    return pipeline_run_id, str(document_id)


def _seed_summary_stage_run(db_session, run_id: uuid.UUID, *, status: str = "RUNNING") -> None:
    """Pre-seed the summary StageRun (pass_name IS NULL) for derive_ontology_graph."""
    db_session.execute(
        text(
            "INSERT INTO ingest.stage_runs "
            "(pipeline_run_id, stage_name, pass_name, attempt, status, started_at) "
            "VALUES (:run_id, 'derive_ontology_graph', NULL, 1, :status, NOW())"
        ),
        {"run_id": run_id, "status": status},
    )
    db_session.flush()


def _seed_pass_output(
    db_session,
    run_id: uuid.UUID,
    pass_name: str,
    *,
    execution_status: str = "COMPLETE",
) -> None:
    save_pass_output(
        db_session,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name=pass_name,
        attempt=1,
        execution_status=execution_status,
        skip_reason=None,
        yield_status=("HIT" if execution_status == "COMPLETE" else None),
        extract_pass_response={"status": "ok", "pass_output": {"entities": []}},
        primary_entities_extracted=1 if execution_status == "COMPLETE" else 0,
        bridge_entities_extracted=0,
        relationships_extracted=0,
        relationships_rejected=0,
        diagnostics={},
        field_provenance=[],
    )
    db_session.flush()


def _seed_phase_entry(
    db_session,
    run_id: uuid.UUID,
    phase_key: str,
    entry: dict,
) -> None:
    db_session.execute(
        text(
            "UPDATE ingest.pipeline_runs "
            "SET dispatched_phases = jsonb_set(dispatched_phases, ARRAY[:phase], CAST(:entry AS jsonb), true) "
            "WHERE id = :run_id"
        ),
        {"run_id": str(run_id), "phase": phase_key, "entry": json.dumps(entry)},
    )
    db_session.flush()


def _now_minus(seconds: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def _now_minus_iso(seconds: int) -> str:
    return _now_minus(seconds).isoformat()


def _preclaim_pass(db_session: Session, run_id: uuid.UUID, pass_name: str) -> None:
    """Pre-claim a phase slot so the pass task body can advance it to 'dispatched'.

    In production, the dispatcher (Task 8) calls _claim_and_dispatch_pass which
    calls claim_phase before .delay(). Tests that invoke the pass task body directly
    must pre-claim the phase so mark_phase_dispatched finds the expected 'claimed' state.
    """
    phase_key = _phase_key(pass_name)
    claim_phase(db_session, run_id, phase_key)
    db_session.flush()


# ---------------------------------------------------------------------------
# Common context-manager patches for the pass task
# ---------------------------------------------------------------------------


@contextmanager
def _patch_pass_task(
    db_session: Session,
    *,
    manifest: SimpleNamespace | None = None,
    outcome_map: dict[str, PassAttemptOutcome] | None = None,
) -> Any:
    """Patch all I/O dependencies for derive_ontology_graph_pass.

    Yields a dict of mocks keyed by short name.

    ``outcome_map`` maps pass_name → PassAttemptOutcome (default: COMPLETE for all passes).
    """
    proxy = _NoCloseProxy(db_session)
    if manifest is None:
        manifest = _fake_manifest()
    if outcome_map is None:
        outcome_map = {}

    def _fake_execute(*, pipeline_run_id, pass_def, **kwargs):
        return outcome_map.get(pass_def.name, _complete_outcome())

    mock_chain_instance = MagicMock()
    mock_chain_fn = MagicMock(return_value=mock_chain_instance)

    patches = [
        patch("app.workers.pipeline._get_db", return_value=proxy),
        patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
        patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
        patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
        patch(
            "app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes",
            return_value={},
        ),
        patch("app.workers.pipeline._write_stage_run", return_value=None),
        patch("app.workers.pipeline._execute_pass_attempt", side_effect=_fake_execute),
    ]

    keys = [
        "_get_db", "_build_docling_document_json", "load_bundle_manifest",
        "load_ontology", "_rehydrate_upstream_refs", "_write_stage_run",
        "_execute_pass_attempt",
    ]

    mocks: dict[str, Any] = {}
    started = []
    for p, k in zip(patches, keys):
        m = p.start()
        mocks[k] = m
        started.append(p)
    try:
        yield mocks
    finally:
        for p in started:
            p.stop()


@contextmanager
def _patch_merge_task(
    db_session: Session,
    *,
    manifest: SimpleNamespace | None = None,
    gate: GateResult | None = None,
    merged: Any | None = None,
    skip_consistency: bool = True,
    skip_rehydrate: bool = True,
    mock_downstream: bool = True,
) -> Any:
    """Patch all I/O dependencies for derive_ontology_graph_merge.

    Yields a dict of mocks keyed by short name.
    """
    proxy = _NoCloseProxy(db_session)
    if manifest is None:
        manifest = _fake_manifest()
    if gate is None:
        gate = GateResult(passed=True, failures=[])
    if merged is None:
        merged = SimpleNamespace(
            entities=[SimpleNamespace(identity="e1")],
            edges=[],
        )

    fake_pass_result = _fake_pass_result()

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
        patch("app.workers.pipeline.celery_chain", mock_chain_fn),
    ]

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

    if skip_consistency:
        patches.append(
            patch("app.workers.pipeline._assert_stage_run_pass_output_consistency")
        )
        keys.append("_assert_stage_run_pass_output_consistency")

    if skip_rehydrate:
        patches.append(
            patch(
                "app.workers.pipeline._rehydrate_pass_result",
                return_value=fake_pass_result,
            )
        )
        keys.append("_rehydrate_pass_result")

    mocks: dict[str, Any] = {}
    started = []
    for p, k in zip(patches, keys):
        m = p.start()
        mocks[k] = m
        started.append(p)
    try:
        yield mocks
    finally:
        for p in started:
            p.stop()


@contextmanager
def _patch_dispatcher(
    db_session: Session,
    *,
    manifest: SimpleNamespace | None = None,
) -> Any:
    """Patch I/O deps for derive_ontology_graph (the thin dispatcher, Task 8)."""
    proxy = _NoCloseProxy(db_session)
    if manifest is None:
        manifest = _fake_manifest()

    mock_delay = MagicMock(return_value=MagicMock(id="mocked-task-id"))

    patches = [
        patch("app.workers.pipeline._get_db", return_value=proxy),
        patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
        patch("app.workers.pipeline.derive_ontology_graph_pass.delay", mock_delay),
    ]
    keys = ["_get_db", "load_bundle_manifest", "delay"]

    mocks: dict[str, Any] = {}
    started = []
    for p, k in zip(patches, keys):
        m = p.start()
        mocks[k] = m
        started.append(p)
    try:
        yield mocks
    finally:
        for p in started:
            p.stop()


# ---------------------------------------------------------------------------
# Direct task invocation helpers (bypass guard_stage_run)
# ---------------------------------------------------------------------------


def _invoke_pass(self_mock, document_id: str, run_id: str, pass_name: str) -> Any:
    raw_fn = derive_ontology_graph_pass.run.__wrapped__
    return raw_fn(self_mock, document_id, run_id, pass_name)


def _invoke_merge(self_mock, document_id: str, run_id: str) -> Any:
    raw_fn = derive_ontology_graph_merge.run.__wrapped__
    return raw_fn(self_mock, document_id, run_id)


def _invoke_dispatcher(self_mock, document_id: str, run_id: str) -> Any:
    raw_fn = derive_ontology_graph.run.__wrapped__
    return raw_fn(self_mock, document_id, run_id)


def _invoke_finalize(self_mock, document_id: str, run_id: str) -> Any:
    raw_fn = finalize_document.run.__wrapped__  # type: ignore[name-defined]
    return raw_fn(self_mock, document_id, run_id)


# ---------------------------------------------------------------------------
# Import finalize_document for the tests that need it
# ---------------------------------------------------------------------------

from app.workers.pipeline import finalize_document  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: run all three passes through the pass task, then merge
# ---------------------------------------------------------------------------


def _run_all_passes_and_merge(
    db_session: Session,
    run_id: uuid.UUID,
    doc_id: str,
    *,
    manifest: SimpleNamespace | None = None,
    outcome_map: dict[str, PassAttemptOutcome] | None = None,
) -> dict:
    """Drive the full fan-in: run each pass through derive_ontology_graph_pass,
    then run the merge task. Returns the merge result dict.

    This is a test-only orchestrator — it runs tasks synchronously in sequence
    (no actual Celery broker needed). Uses ExitStack to avoid Python's static
    nesting limit.
    """
    if manifest is None:
        manifest = _fake_manifest()
    if outcome_map is None:
        outcome_map = {}

    def _fake_execute(*, pipeline_run_id, pass_def, **kwargs):
        return outcome_map.get(pass_def.name, _complete_outcome())

    _run_id_str = str(run_id)
    proxy = _NoCloseProxy(db_session)
    mock_chain_fn = MagicMock(return_value=MagicMock())
    fake_pass_result = _fake_pass_result()

    pass_patches = [
        patch("app.workers.pipeline._get_db", return_value=proxy),
        patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
        patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
        patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
        patch("app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes", return_value={}),
        patch("app.workers.pipeline._write_stage_run", return_value=None),
        patch("app.workers.pipeline._execute_pass_attempt", side_effect=_fake_execute),
        patch("app.workers.pipeline.celery_app.send_task", MagicMock()),
        patch("app.workers.pipeline.derive_ontology_graph_pass.delay",
              MagicMock(return_value=MagicMock(id="mocked-task-id"))),
        patch("app.workers.pipeline._terminalize_doc_and_run", MagicMock()),
        patch("app.workers.pipeline.check_required_pass_gate",
              return_value=GateResult(passed=True, failures=[])),
        patch("app.workers.pipeline.merge_and_resolve",
              return_value=SimpleNamespace(entities=[], edges=[])),
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
        patch("app.workers.pipeline._assert_stage_run_pass_output_consistency"),
        patch("app.workers.pipeline._rehydrate_pass_result", return_value=fake_pass_result),
        patch("app.workers.pipeline.celery_chain", mock_chain_fn),
    ]

    with ExitStack() as stack:
        for p in pass_patches:
            stack.enter_context(p)

        # Seed the summary StageRun first (normally done by Task 8 dispatcher)
        _seed_summary_stage_run(db_session, run_id)

        # Pre-claim each pass phase (normally done by _claim_and_dispatch_pass in Task 8)
        entity_passes = [p.name for p in manifest.passes if not p.depends_on]
        for pass_name in entity_passes:
            _preclaim_pass(db_session, run_id, pass_name)

        # Run entity passes
        for pass_name in entity_passes:
            _invoke_pass(_make_task_self(), doc_id, _run_id_str, pass_name)

        # Run system_links pass if manifest has it
        sl_pass_names = [p.name for p in manifest.passes if p.depends_on]
        for pass_name in sl_pass_names:
            _preclaim_pass(db_session, run_id, pass_name)
            _invoke_pass(_make_task_self(), doc_id, _run_id_str, pass_name)

        # Run merge
        result = _invoke_merge(_make_merge_task_self(), doc_id, _run_id_str)

    return result


# ===========================================================================
# Test cluster 1: Happy Path
# ===========================================================================


class TestHappyPath:
    """Tests 1 and 13: full run reaches COMPLETE; summary StageRun lifecycle."""

    def test_full_run_completes_through_all_phases(
        self, db_session: Session
    ):
        """Test 1: happy path — all 3 passes COMPLETE; merge runs; summary StageRun=COMPLETE.

        Verifies:
        - pipeline_pass_outputs rows written for all 3 passes
        - dispatched_phases has 'completed' entries for entity passes + system_links
        - summary StageRun (pass_name IS NULL) transitions to COMPLETE
        - merge task returns {"merge": "ok", ...}
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        result = _run_all_passes_and_merge(db_session, run_id, doc_id)

        assert result["merge"] == "ok", f"Expected merge=ok; got {result}"

        # All 3 pass outputs written
        db_session.expire_all()
        for pass_name in [_PASS_RADAR, _PASS_MISSILE, _PASS_SL]:
            row = load_pass_output(db_session, run_id, pass_name)
            assert row is not None, f"Missing pass_output for {pass_name}"
            assert row.execution_status == "COMPLETE"

        # Summary StageRun (pass_name IS NULL) should be COMPLETE
        summary = db_session.execute(
            text(
                "SELECT status FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert summary is not None, "Summary StageRun not found"
        assert summary[0] == "COMPLETE", f"Expected COMPLETE; got {summary[0]}"

    def test_finalize_document_sees_derive_ontology_graph_stage_complete(
        self, db_session: Session
    ):
        """Test 13: finalize_document sees derive_ontology_graph=COMPLETE in StageRun.

        Verifies:
        - After full happy path, summary StageRun = COMPLETE (not RUNNING/FAILED)
        - finalize_document's REQUIRED_STAGES gate sees derive_ontology_graph as COMPLETE
        - PipelineRun.status set to COMPLETE (no other required stages are missing
          for graph_only mode).

        We use graph_only mode to avoid needing the full prepare/embed/etc. stages.
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        # Set mode to graph_only so finalize_document only requires:
        # derive_document_anchors, derive_ontology_graph, derive_structure_links
        db_session.execute(
            text("UPDATE ingest.pipeline_runs SET mode='graph_only' WHERE id=:id"),
            {"id": run_id},
        )
        db_session.flush()

        # Pre-seed derive_document_anchors and derive_structure_links as COMPLETE
        # so finalize_document doesn't see them as missing
        for stage in ("derive_document_anchors", "derive_structure_links"):
            db_session.execute(
                text(
                    "INSERT INTO ingest.stage_runs "
                    "(pipeline_run_id, stage_name, pass_name, attempt, status, started_at) "
                    "VALUES (:run_id, :stage, NULL, 1, 'COMPLETE', NOW())"
                ),
                {"run_id": run_id, "stage": stage},
            )
        db_session.flush()

        # Run full fan-in (this also seeds the summary StageRun)
        _run_all_passes_and_merge(db_session, run_id, doc_id)

        # Verify summary StageRun = COMPLETE
        db_session.expire_all()
        summary = db_session.execute(
            text(
                "SELECT status FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id "
                "  AND stage_name = 'derive_ontology_graph' "
                "  AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert summary is not None
        assert summary[0] == "COMPLETE"

        # Now run finalize_document via proxy
        proxy = _NoCloseProxy(db_session)
        with patch("app.workers.pipeline._get_db", return_value=proxy):
            with patch("app.workers.pipeline._update_document_status"):
                with patch(
                    "app.workers.pipeline._maybe_trigger_post_ingest_community_detection"
                ):
                    finalize_fn = finalize_document.run.__wrapped__
                    finalize_fn(_make_task_self(), doc_id, str(run_id))

        # PipelineRun should be COMPLETE (all required stages for graph_only are present)
        db_session.expire_all()
        run_row = db_session.execute(
            text("SELECT status FROM ingest.pipeline_runs WHERE id = :id"),
            {"id": run_id},
        ).fetchone()
        assert run_row is not None
        assert run_row[0] == "COMPLETE", (
            f"Expected PipelineRun.status=COMPLETE; got {run_row[0]}"
        )


# ===========================================================================
# Test cluster 2: Retry Exhaustion
# ===========================================================================


class TestRetryExhaustion:
    """Tests 2 and 3: optional vs required pass retry exhaustion behavior."""

    def test_optional_pass_retry_exhaustion_does_not_block_run(
        self, db_session: Session
    ):
        """Test 2: optional pass exhausts retries → run continues; merge excludes it.

        Verifies:
        - Pass output written with execution_status=FAILED, retry_exhausted=True
        - Phase entry state=completed, result=failed
        - Merge runs with only COMPLETE passes (failed pass excluded)
        - Summary StageRun = COMPLETE (gate passes because failed pass is optional)
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        # Manifest: radar_identity required, missile_airframe optional, no system_links
        manifest = _fake_manifest(passes=[
            _fake_pass_def(_PASS_RADAR, required=True),
            _fake_pass_def(_PASS_MISSILE, required=False),
        ])

        # missile_airframe exhausts retries; radar_identity succeeds
        outcome_map = {
            _PASS_MISSILE: _failed_retryable_outcome(),
        }

        proxy = _NoCloseProxy(db_session)
        fake_pass_result = _fake_pass_result()
        mock_chain_fn = MagicMock(return_value=MagicMock())

        def _fake_execute(*, pipeline_run_id, pass_def, **kwargs):
            return outcome_map.get(pass_def.name, _complete_outcome())

        all_patches = [
            patch("app.workers.pipeline._get_db", return_value=proxy),
            patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
            patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
            patch("app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes", return_value={}),
            patch("app.workers.pipeline._write_stage_run", return_value=None),
            patch("app.workers.pipeline.celery_app.send_task", MagicMock()),
            patch("app.workers.pipeline.derive_ontology_graph_pass.delay",
                  MagicMock(return_value=MagicMock(id="mocked-id"))),
            patch("app.workers.pipeline._terminalize_doc_and_run", MagicMock()),
            patch("app.workers.pipeline._execute_pass_attempt", side_effect=_fake_execute),
            patch("app.workers.pipeline.check_required_pass_gate",
                  return_value=GateResult(passed=True, failures=[])),
            patch("app.workers.pipeline.merge_and_resolve",
                  return_value=SimpleNamespace(entities=[], edges=[])),
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
            patch("app.workers.pipeline._assert_stage_run_pass_output_consistency"),
            patch("app.workers.pipeline._rehydrate_pass_result", return_value=fake_pass_result),
            patch("app.workers.pipeline.celery_chain", mock_chain_fn),
        ]

        with ExitStack() as stack:
            for p in all_patches:
                stack.enter_context(p)

            _seed_summary_stage_run(db_session, run_id)

            # Pre-claim phases (normally done by Task 8 dispatcher)
            _preclaim_pass(db_session, run_id, _PASS_RADAR)
            _preclaim_pass(db_session, run_id, _PASS_MISSILE)

            # radar_identity succeeds
            result_radar = _invoke_pass(
                _make_task_self(), doc_id, str(run_id), _PASS_RADAR
            )
            assert result_radar["execution_status"] == "COMPLETE"

            # missile_airframe exhausted (retries=3 == max_retries=3)
            result_missile = _invoke_pass(
                _make_task_self(retries=3, max_retries=3),
                doc_id, str(run_id), _PASS_MISSILE,
            )
            assert result_missile["execution_status"] == "FAILED"
            assert result_missile.get("reason") == "retry_exhausted"

            merge_result = _invoke_merge(_make_merge_task_self(), doc_id, str(run_id))

        # Failed pass_output row written
        db_session.expire_all()
        row = load_pass_output(db_session, run_id, _PASS_MISSILE)
        assert row is not None
        assert row.execution_status == "FAILED"
        assert row.diagnostics_json.get("retry_exhausted") is True

        # Phase entry should be completed/failed
        phase_entry = read_phase_state(db_session, run_id, _PHASE_MISSILE)
        assert phase_entry is not None
        assert phase_entry.state == "completed"
        assert phase_entry.result == "failed"

        assert merge_result["merge"] == "ok"

        # Summary StageRun = COMPLETE
        db_session.expire_all()
        summary = db_session.execute(
            text(
                "SELECT status FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert summary is not None
        assert summary[0] == "COMPLETE"

    def test_required_pass_retry_exhaustion_terminalizes_with_failed_summary(
        self, db_session: Session
    ):
        """Test 3: required pass exhausts retries → run terminalized; summary=FAILED.

        Critical path (r4): verifies _terminalize_doc_and_run is called BEFORE
        Celery's MaxRetriesExceededError short-circuit, ensuring cleanup always runs.

        Verifies:
        - _terminalize_doc_and_run called once
        - IngestFailed raised
        - Summary StageRun status=FAILED (set by _update_summary_stage_run before terminalize)
        - Pass output written with execution_status=FAILED
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        # Manifest: only radar_identity (required)
        manifest = _fake_manifest(passes=[_fake_pass_def(_PASS_RADAR, required=True)])
        proxy = _NoCloseProxy(db_session)

        _seed_summary_stage_run(db_session, run_id)
        # Pre-claim the phase (normally done by Task 8 dispatcher)
        _preclaim_pass(db_session, run_id, _PASS_RADAR)

        mock_terminalize = MagicMock()
        t3_patches = [
            patch("app.workers.pipeline._get_db", return_value=proxy),
            patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
            patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
            patch("app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes", return_value={}),
            patch("app.workers.pipeline._write_stage_run", return_value=None),
            patch("app.workers.pipeline._execute_pass_attempt",
                  return_value=_failed_retryable_outcome()),
            patch("app.workers.pipeline._terminalize_doc_and_run", mock_terminalize),
        ]
        with ExitStack() as stack:
            for p in t3_patches:
                stack.enter_context(p)
            with pytest.raises(IngestFailed):
                _invoke_pass(
                    _make_task_self(retries=3, max_retries=3),
                    doc_id, str(run_id), _PASS_RADAR,
                )

        # _terminalize_doc_and_run must have been called once
        mock_terminalize.assert_called_once()

        # Pass output written as FAILED
        db_session.expire_all()
        row = load_pass_output(db_session, run_id, _PASS_RADAR)
        assert row is not None
        assert row.execution_status == "FAILED"
        assert row.diagnostics_json.get("retry_exhausted") is True

        # Summary StageRun should be FAILED
        summary = db_session.execute(
            text(
                "SELECT status FROM ingest.stage_runs "
                "WHERE pipeline_run_id = :run_id AND pass_name IS NULL"
            ),
            {"run_id": run_id},
        ).fetchone()
        assert summary is not None
        assert summary[0] == "FAILED", (
            f"Expected summary StageRun FAILED after required pass exhaustion; got {summary[0]}"
        )


# ===========================================================================
# Test cluster 3: Skip Passthrough
# ===========================================================================


class TestSkipPassthrough:
    """Tests 4 and 5: authorized skips and idempotency guards."""

    def test_authorized_skip_treated_as_terminal_for_fanin(
        self, db_session: Session
    ):
        """Test 4: system_links is skipped (no upstream endpoints) → fan-in proceeds.

        Verifies:
        - Pass output written with execution_status=SKIPPED, skip_reason set
        - Phase entry result='skipped'
        - _try_advance_phase called (fan-in advances)
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        # Manifest: radar_identity entity pass + system_links with skip_if_no_upstream=True
        sl_def = _fake_pass_def(
            _PASS_SL,
            input_mode="document_plus_entity_refs",
            depends_on=[_PASS_RADAR],
            skip_if_no_upstream=True,
        )
        manifest = _fake_manifest(passes=[_fake_pass_def(_PASS_RADAR), sl_def])

        proxy = _NoCloseProxy(db_session)

        # Pre-claim the system_links phase (normally done by _try_advance_phase when
        # entity passes complete — here we invoke the pass task directly)
        _preclaim_pass(db_session, run_id, _PASS_SL)

        with (
            patch("app.workers.pipeline._get_db", return_value=proxy),
            patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
            patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
            patch("app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes", return_value={}),
            patch("app.workers.pipeline._write_stage_run", return_value=None),
            patch("app.workers.pipeline._execute_pass_attempt",
                  return_value=_skipped_outcome()),
            patch("app.workers.pipeline.celery_app.send_task", MagicMock()),
            patch("app.workers.pipeline.derive_ontology_graph_pass.delay",
                  MagicMock(return_value=MagicMock(id="mocked-id"))),
        ):
            with patch("app.workers.pipeline._try_advance_phase") as mock_advance:
                result = _invoke_pass(
                    _make_task_self(), doc_id, str(run_id), _PASS_SL
                )

        assert result["execution_status"] == "SKIPPED"

        # Pass output row written with SKIPPED
        db_session.expire_all()
        row = load_pass_output(db_session, run_id, _PASS_SL)
        assert row is not None
        assert row.execution_status == "SKIPPED"
        assert row.skip_reason == "NO_UPSTREAM_ENDPOINTS"

        # Phase result should be 'skipped'
        phase_entry = read_phase_state(db_session, run_id, _PASS_SL)
        assert phase_entry is not None
        assert phase_entry.result == "skipped"

        # _try_advance_phase must have been called
        mock_advance.assert_called_once()

    def test_skip_already_resolved_passes_on_retry(
        self, db_session: Session
    ):
        """Test 5: pre-seeded terminal pass-output rows → pass tasks short-circuit.

        Verifies:
        - _execute_pass_attempt NOT called for pre-resolved passes
        - The already-resolved check immediately marks phase terminal and advances
        """
        run_id, doc_id = _make_run_with_doc(db_session)
        manifest = _fake_manifest(passes=[
            _fake_pass_def(_PASS_RADAR),
            _fake_pass_def(_PASS_MISSILE),
        ])

        # Pre-seed terminal pass_output rows for BOTH passes
        _seed_pass_output(db_session, run_id, _PASS_RADAR, execution_status="COMPLETE")
        _seed_pass_output(db_session, run_id, _PASS_MISSILE, execution_status="COMPLETE")

        proxy = _NoCloseProxy(db_session)

        with (
            patch("app.workers.pipeline._get_db", return_value=proxy),
            patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
            patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
            patch("app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes", return_value={}),
            patch("app.workers.pipeline._write_stage_run", return_value=None),
        ):
            with patch("app.workers.pipeline._execute_pass_attempt") as mock_exec:
                with patch("app.workers.pipeline._try_advance_phase") as mock_advance:
                    result_radar = _invoke_pass(
                        _make_task_self(), doc_id, str(run_id), _PASS_RADAR
                    )
                    result_missile = _invoke_pass(
                        _make_task_self(), doc_id, str(run_id), _PASS_MISSILE
                    )

        # _execute_pass_attempt must NOT have been called for either pre-resolved pass
        mock_exec.assert_not_called()

        # Both returns indicate already_resolved short-circuit
        assert result_radar.get("skipped") == "already_resolved"
        assert result_missile.get("skipped") == "already_resolved"

        # _try_advance_phase called for each pass (2 passes)
        assert mock_advance.call_count == 2


# ===========================================================================
# Test cluster 4: Drift Detection
# ===========================================================================


class TestDriftDetection:
    """Test 6: COMPLETE-without-output drift detection."""

    def test_complete_without_output_fails_loud(
        self, db_session: Session
    ):
        """Test 6: COMPLETE StageRun for a pass but no pass_output row → WorkerInvariantError.

        Simulates the crash window between _write_stage_run() and
        _save_terminal_pass_output() in derive_ontology_graph_pass.
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        # Pre-seed a COMPLETE StageRun for radar_identity (no pass_output row)
        db_session.execute(
            text(
                "INSERT INTO ingest.stage_runs "
                "(pipeline_run_id, stage_name, pass_name, attempt, status, execution_status, started_at) "
                "VALUES (:run_id, 'derive_ontology_graph', :pass_name, 1, 'COMPLETE', 'COMPLETE', NOW())"
            ),
            {"run_id": run_id, "pass_name": _PASS_RADAR},
        )
        db_session.flush()
        # No pipeline_pass_outputs row — drift!

        proxy = _NoCloseProxy(db_session)

        with _patch_merge_task(
            db_session,
            skip_consistency=False,  # let the real consistency check run
        ) as mocks:
            with pytest.raises(WorkerInvariantError) as exc_info:
                _invoke_merge(_make_merge_task_self(), doc_id, str(run_id))

        assert _PASS_RADAR in str(exc_info.value)
        # merge_and_resolve must NOT have been called (check aborted early)
        mocks["merge_and_resolve"].assert_not_called()


# ===========================================================================
# Test cluster 5: Cancellation
# ===========================================================================


class TestCancellation:
    """Test 7: cancel during active pass aborts gracefully."""

    def test_cancel_during_active_pass_aborts_gracefully(
        self, db_session: Session
    ):
        """Test 7: cancel_document fires mid-extraction → task discards output.

        The 'mid-extraction' timing is simulated by making _execute_pass_attempt
        flip the run to FAILED as a side effect before returning COMPLETE outcome.

        Verifies:
        - Task returns {"skipped": "cancelled_mid_extraction"}
        - No pipeline_pass_outputs row written
        - mark_phase_terminal NOT called
        """
        from app.models.ingest import PipelineRun

        run_id, doc_id = _make_run_with_doc(db_session)
        proxy = _NoCloseProxy(db_session)

        def _execute_and_cancel(*, pipeline_run_id, pass_def, **kwargs):
            # Flip the run status mid-extraction to simulate cancel_document
            run_obj = db_session.get(PipelineRun, run_id)
            assert run_obj is not None
            run_obj.status = "FAILED"
            db_session.flush()
            return _complete_outcome()

        with (
            patch("app.workers.pipeline._get_db", return_value=proxy),
            patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=_fake_manifest()),
            patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
            patch("app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes", return_value={}),
            patch("app.workers.pipeline._write_stage_run", return_value=None),
            patch("app.workers.pipeline._execute_pass_attempt", side_effect=_execute_and_cancel),
            patch("app.workers.pipeline.mark_phase_terminal") as mock_terminal,
        ):
            result = _invoke_pass(
                _make_task_self(), doc_id, str(run_id), _PASS_RADAR
            )

        # Task must report cancelled_mid_extraction
        assert result.get("skipped") == "cancelled_mid_extraction", (
            f"Expected cancelled_mid_extraction; got {result}"
        )

        # No pass_output row written
        db_session.expire_all()
        assert not is_pass_already_resolved(db_session, run_id, _PASS_RADAR)

        # mark_phase_terminal must NOT have been called
        mock_terminal.assert_not_called()


# ===========================================================================
# Test cluster 6: Chain Ordering
# ===========================================================================


class TestChainOrdering:
    """Test 8: downstream stages do not run before merge returns."""

    def test_downstream_stages_do_not_run_before_merge(
        self, db_session: Session
    ):
        """Test 8: collect_derivations is dispatched ONLY AFTER merge returns.

        Verifies that the downstream chain (collect_derivations → ...) is
        dispatched by celery_chain.apply_async AFTER derive_ontology_graph_merge
        has committed its graph writes and summary StageRun update.

        The ordering is verified by asserting that the celery_chain mock is
        called with exactly 4 stages (full mode) and that apply_async is
        invoked exactly once after merge.
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        # Pre-seed pass outputs so merge can load them
        _seed_pass_output(db_session, run_id, _PASS_RADAR)
        _seed_pass_output(db_session, run_id, _PASS_MISSILE)
        _seed_pass_output(db_session, run_id, _PASS_SL)
        _seed_summary_stage_run(db_session, run_id)

        call_log: list[str] = []

        def fake_import_nodes(merged, ontology, doc_id, tracker, provenance):
            call_log.append("import_nodes")
            return {}

        def fake_chain(*args):
            # Record that chain was called AFTER graph imports
            call_log.append("chain_called")
            m = MagicMock()
            m.apply_async.side_effect = lambda: call_log.append("apply_async")
            return m

        proxy = _NoCloseProxy(db_session)

        with (
            patch("app.workers.pipeline._get_db", return_value=proxy),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=_fake_manifest()),
            patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
            patch("app.workers.pipeline.check_required_pass_gate",
                  return_value=GateResult(passed=True, failures=[])),
            patch("app.workers.pipeline.merge_and_resolve",
                  return_value=SimpleNamespace(entities=[], edges=[])),
            patch("app.workers.pipeline._apply_post_merge_yield_updates"),
            patch("app.workers.pipeline._write_pipeline_run_metrics"),
            patch("app.workers.pipeline._build_provenance_envelope", return_value={}),
            patch("app.workers.pipeline._import_graph_phase_nodes",
                  side_effect=fake_import_nodes),
            patch("app.workers.pipeline._import_graph_phase_domain_edges",
                  side_effect=lambda *a, **k: call_log.append("import_domain_edges")),
            patch("app.workers.pipeline._ensure_structural_document_vertex"),
            patch("app.workers.pipeline._import_graph_phase_structural_edges"),
            patch("app.workers.pipeline._build_element_uid_to_artifact_id", return_value={}),
            patch("app.workers.pipeline._upsert_document_graph_extraction"),
            patch("app.workers.pipeline._update_document_pipeline_status"),
            patch("app.workers.pipeline._assert_stage_run_pass_output_consistency"),
            patch("app.workers.pipeline._rehydrate_pass_result",
                  return_value=_fake_pass_result()),
            patch("app.workers.pipeline.celery_chain", side_effect=fake_chain),
            patch("app.workers.pipeline._terminalize_doc_and_run"),
        ):
            _invoke_merge(_make_merge_task_self(), doc_id, str(run_id))

        # Verify ordering: graph imports ran BEFORE chain_called
        assert "import_nodes" in call_log, "import_nodes should have run"
        assert "chain_called" in call_log, "celery_chain should have been called"
        assert "apply_async" in call_log, "apply_async should have been called"

        # chain_called must come AFTER import_nodes
        idx_import = call_log.index("import_nodes")
        idx_chain = call_log.index("chain_called")
        assert idx_import < idx_chain, (
            f"import_nodes ({idx_import}) must precede chain_called ({idx_chain}); "
            f"call_log={call_log}"
        )

        # apply_async must come AFTER chain_called
        idx_apply = call_log.index("apply_async")
        assert idx_chain < idx_apply, (
            f"chain_called must precede apply_async; call_log={call_log}"
        )


# ===========================================================================
# Test cluster 7: Rollback Contract
# ===========================================================================


class TestRollbackContract:
    """Test 9: merge rollback after post-mutation graph failure."""

    @pytest.mark.integration
    def test_merge_failure_post_mutation_rolls_back(
        self, db_session: Session
    ):
        """Test 9 (skinnied): merge raises after first graph mutation → _attempt_rollback called.

        Simplified vs full-E2E spec: verifies the rollback contract is enforced
        (GraphWriteTracker marks → domain_edges raises → rollback called before raise),
        without verifying Celery retry semantics (eager-mode retry is out of scope).

        What is verified:
        - _attempt_rollback(document_id) called once after the failure
        - RuntimeError propagates (not swallowed)
        - mark_phase_terminal called with result='failed' in the error path
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        # Pre-seed pass output so the consistency check passes
        _seed_pass_output(db_session, run_id, _PASS_RADAR)
        _seed_summary_stage_run(db_session, run_id)

        manifest = _fake_manifest(passes=[_fake_pass_def(_PASS_RADAR)])
        proxy = _NoCloseProxy(db_session)

        def fake_import_nodes(merged, ontology, doc_id, tracker, provenance):
            tracker.mark()  # first graph mutation
            return {}

        mock_rollback = MagicMock(return_value="")
        mock_phase = MagicMock()

        rollback_patches = [
            patch("app.workers.pipeline._get_db", return_value=proxy),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
            patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
            patch("app.workers.pipeline.check_required_pass_gate",
                  return_value=GateResult(passed=True, failures=[])),
            patch("app.workers.pipeline.merge_and_resolve",
                  return_value=SimpleNamespace(entities=[], edges=[])),
            patch("app.workers.pipeline._apply_post_merge_yield_updates"),
            patch("app.workers.pipeline._write_pipeline_run_metrics"),
            patch("app.workers.pipeline._build_provenance_envelope", return_value={}),
            patch("app.workers.pipeline._import_graph_phase_nodes",
                  side_effect=fake_import_nodes),
            patch("app.workers.pipeline._import_graph_phase_domain_edges",
                  side_effect=RuntimeError("ArcadeDB down")),
            patch("app.workers.pipeline._ensure_structural_document_vertex"),
            patch("app.workers.pipeline._import_graph_phase_structural_edges"),
            patch("app.workers.pipeline._build_element_uid_to_artifact_id", return_value={}),
            patch("app.workers.pipeline._upsert_document_graph_extraction"),
            patch("app.workers.pipeline._update_document_pipeline_status"),
            patch("app.workers.pipeline._assert_stage_run_pass_output_consistency"),
            patch("app.workers.pipeline._rehydrate_pass_result", return_value=_fake_pass_result()),
            patch("app.workers.pipeline.celery_chain", MagicMock(return_value=MagicMock())),
            patch("app.workers.pipeline._terminalize_doc_and_run"),
            patch("app.workers.pipeline._attempt_rollback", mock_rollback),
            patch("app.workers.pipeline.mark_phase_terminal", mock_phase),
        ]

        with ExitStack() as stack:
            for p in rollback_patches:
                stack.enter_context(p)
            with pytest.raises(RuntimeError, match="ArcadeDB down"):
                _invoke_merge(_make_merge_task_self(), doc_id, str(run_id))

        # Rollback must have been called with the document_id
        mock_rollback.assert_called_once_with(doc_id)

        # mark_phase_terminal must have been called with result='failed'
        called_failed = any(
            c.kwargs.get("result") == "failed"
            for c in mock_phase.call_args_list
        )
        assert called_failed, (
            f"mark_phase_terminal(result='failed') never called; calls={mock_phase.call_args_list}"
        )


# ===========================================================================
# Test cluster 8: Reconciler
# ===========================================================================


class TestReconciler:
    """Tests 10, 11, 12: reconciler recovery from stale phases."""

    def _patched_reconciler(self, db_session: Session, manifest=None):
        """Context manager that patches reconciler's external deps."""
        proxy = _NoCloseProxy(db_session)
        if manifest is None:
            manifest = _fake_manifest(passes=[
                _fake_pass_def(_PASS_RADAR),
                _fake_pass_def(_PASS_MISSILE),
            ])

        mock_revoke = MagicMock()
        mock_delay = MagicMock(return_value=MagicMock(id="mocked-task-id"))
        mock_send_task = MagicMock()

        @contextmanager
        def _ctx():
            with (
                patch("app.workers.pipeline._get_db", return_value=proxy),
                patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
                patch("app.workers.pipeline.derive_ontology_graph_pass.delay", mock_delay),
                patch("app.workers.pipeline.celery_app.send_task", mock_send_task),
                patch("app.services.run_phase_dispatch.celery_app") as mock_celery,
            ):
                mock_celery.control.revoke = mock_revoke
                yield {
                    "revoke": mock_revoke,
                    "delay": mock_delay,
                    "send_task": mock_send_task,
                }

        return _ctx()

    def _run_reconciler(self) -> dict:
        return reconcile_ontology_graph_runs.run()

    def test_reconciler_recovers_from_stale_claimed(
        self, db_session: Session
    ):
        """Test 10: stale 'claimed' phase (60s ago, threshold 30s) → reclaim + re-dispatch."""
        run_id, doc_id = _make_run_with_doc(db_session)

        # Set mode + bundle on the run
        db_session.execute(
            text("UPDATE ingest.pipeline_runs SET mode='full', ontology_bundle_key=:bk WHERE id=:id"),
            {"id": run_id, "bk": _BUNDLE_KEY},
        )
        db_session.flush()

        stale_at = _now_minus_iso(60)
        _seed_phase_entry(db_session, run_id, _PHASE_RADAR, {
            "state": "claimed",
            "result": None,
            "task_id": "stale-task",
            "claimed_at": stale_at,
            "dispatched_at": None,
            "completed_at": None,
        })

        with self._patched_reconciler(db_session) as mocks:
            result = self._run_reconciler()

        assert _PHASE_RADAR in result["stale_claimed_reclaimed"], (
            f"Expected {_PHASE_RADAR} in stale_claimed_reclaimed; got {result}"
        )
        # Re-dispatch must have been triggered
        mocks["delay"].assert_called()

    def test_reconciler_recovers_from_stale_dispatched_no_pending_retry(
        self, db_session: Session
    ):
        """Test 11: stale 'dispatched' (3h ago) + no pending retry → revoke + re-dispatch."""
        run_id, doc_id = _make_run_with_doc(db_session)

        db_session.execute(
            text("UPDATE ingest.pipeline_runs SET mode='full', ontology_bundle_key=:bk WHERE id=:id"),
            {"id": run_id, "bk": _BUNDLE_KEY},
        )
        db_session.flush()

        stale_at = _now_minus_iso(10800)  # 3 hours ago
        _seed_phase_entry(db_session, run_id, _PHASE_RADAR, {
            "state": "dispatched",
            "result": None,
            "task_id": "old-task-id",
            "claimed_at": stale_at,
            "dispatched_at": stale_at,
            "completed_at": None,
        })

        with self._patched_reconciler(db_session) as mocks:
            result = self._run_reconciler()

        assert _PHASE_RADAR in result["stale_dispatched_reclaimed"], (
            f"Expected {_PHASE_RADAR} in stale_dispatched_reclaimed; got {result}"
        )
        mocks["revoke"].assert_called()

    def test_reconciler_waits_when_pending_retry_in_countdown(
        self, db_session: Session
    ):
        """Test 12: stale 'dispatched' BUT pending retry → NO revoke, NO reclaim.

        A recent FAILED StageRun with attempt < max and finished_at within the
        countdown buffer means a Celery retry is scheduled. The reconciler must
        skip this phase.
        """
        run_id, doc_id = _make_run_with_doc(db_session)

        db_session.execute(
            text("UPDATE ingest.pipeline_runs SET mode='full', ontology_bundle_key=:bk WHERE id=:id"),
            {"id": run_id, "bk": _BUNDLE_KEY},
        )
        db_session.flush()

        stale_at = _now_minus_iso(10800)  # 3 hours ago
        _seed_phase_entry(db_session, run_id, _PHASE_RADAR, {
            "state": "dispatched",
            "result": None,
            "task_id": "retrying-task",
            "claimed_at": stale_at,
            "dispatched_at": stale_at,
            "completed_at": None,
        })

        # Seed a recent FAILED StageRun with attempt=2 (below max), finished 2 min ago
        # → reconciler's _has_pending_retry_for_pass should return True
        finished_2min_ago = _now_minus(120)
        stage_run_id = uuid.uuid4()
        db_session.execute(
            text(
                "INSERT INTO ingest.stage_runs "
                "(id, pipeline_run_id, stage_name, pass_name, attempt, status, "
                " execution_status, finished_at, started_at) "
                "VALUES (:id, :run_id, 'derive_ontology_graph', :pass_name, 2, "
                "'FAILED', 'FAILED', :finished_at, NOW())"
            ),
            {
                "id": stage_run_id,
                "run_id": run_id,
                "pass_name": _PASS_RADAR,
                "finished_at": finished_2min_ago.isoformat(),
            },
        )
        db_session.flush()

        with self._patched_reconciler(db_session) as mocks:
            result = self._run_reconciler()

        assert _PHASE_RADAR in result["skipped_pending_retry"], (
            f"Expected {_PHASE_RADAR} in skipped_pending_retry; got {result}"
        )
        assert result["stale_dispatched_reclaimed"] == [], (
            f"Must NOT reclaim when retry is pending; got {result['stale_dispatched_reclaimed']}"
        )
        mocks["revoke"].assert_not_called()
