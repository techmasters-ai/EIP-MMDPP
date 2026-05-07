"""Integration + unit tests for derive_ontology_graph_pass (Task 5).

Pattern:
- Real PostgreSQL (port 5438) for DB-state assertions.  Parent rows created
  via ``pipeline_run_factory``; rolled back after each test by ``db_session``.
- ``_get_db`` is patched to return a *non-closing proxy* around ``db_session``
  so the task body shares the same transaction as the test.  The proxy
  suppresses ``.close()`` so ``db_session``'s rollback-on-teardown still works.
- ``_execute_pass_attempt`` is mocked to inject ``PassAttemptOutcome`` without
  HTTP or parse overhead.
- ``_write_stage_run`` is mocked (it opens its own session internally).
- ``_try_advance_phase`` is mocked by default (fan-in logic tested separately).

Run with:

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \\
        /home/josh/development/EIP-MMDPP/.venv/bin/python -m pytest \\
        tests/unit/test_derive_ontology_graph_pass_task.py -v
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import Retry as CeleryRetry
from sqlalchemy.orm import Session

from app.services.pass_outputs_store import (
    is_pass_already_resolved,
    load_pass_output,
    save_pass_output,
)
from app.services.run_phase_dispatch import claim_phase, read_phase_state
from app.workers.pipeline import (
    IngestFailed,
    PassAttemptOutcome,
    PassRetryable,
    PassTerminal,
    PassTransportError,
    _phase_key,
    _retry_delay,
    derive_ontology_graph_pass,
)

# ---------------------------------------------------------------------------
# Test-local helpers
# ---------------------------------------------------------------------------

_BUNDLE_KEY = "air_defense_v3"


def _fake_pass_def(
    *,
    name: str = "radar_identity",
    kind: str = "entities_and_relationships",
    input_mode: str = "document_only",
    required: bool = True,
    depends_on: list[str] | None = None,
    primary: tuple[str, ...] = ("RADAR_SYSTEM",),
    bridge: tuple[str, ...] = (),
    rels: tuple[str, ...] = ("INSTALLED_ON",),
    skip_if_no_upstream_endpoints: bool = False,
):
    return SimpleNamespace(
        name=name,
        kind=kind,
        input_mode=input_mode,
        required=required,
        depends_on=list(depends_on or []),
        primary_entity_types=list(primary),
        bridge_entity_types=list(bridge),
        extracted_relationship_types=list(rels),
        skip_if_no_upstream_endpoints=skip_if_no_upstream_endpoints,
        module="extraction_schemas.radar_identity",
        template_class="RadarIdentityPass",
    )


def _fake_manifest(passes: list | None = None, *, required: bool = True):
    """Build a fake manifest.  ``required`` is a shortcut for a single-pass manifest."""
    if passes is None:
        passes = [_fake_pass_def(required=required)]
    return SimpleNamespace(bundle_key=_BUNDLE_KEY, passes=passes)


_MINIMAL_ONTOLOGY: dict = {
    "entity_types": [{"name": "RADAR_SYSTEM", "identity_fields": ["system_name"]}],
    "validation_matrix": [
        {"source": "RADAR_SYSTEM", "relationship": "INSTALLED_ON", "target": "PLATFORM"},
    ],
}


def _fake_pass_result():
    return SimpleNamespace(
        template_instance=SimpleNamespace(relationships=[]),
        upstream_refs={},
        pre_merge_walk=SimpleNamespace(entities=[], raw_edge_count=0),
    )


def _make_complete_outcome(
    *,
    pass_result=None,
    raw_payload: dict | None = None,
    yield_status: str = "HIT",
) -> PassAttemptOutcome:
    if pass_result is None:
        pass_result = _fake_pass_result()
    if raw_payload is None:
        raw_payload = {"status": "ok", "pass_output": {"entities": []}}
    return PassAttemptOutcome(
        execution_status="COMPLETE",
        skip_reason=None,
        yield_status=yield_status,
        pass_result=pass_result,
        raw_response_payload=raw_payload,
        counts={
            "primary_entities_extracted": 5,
            "bridge_entities_extracted": 0,
            "relationships_extracted": 3,
            "relationships_rejected": 0,
        },
        error=None,
    )


def _make_skipped_outcome() -> PassAttemptOutcome:
    return PassAttemptOutcome(
        execution_status="SKIPPED",
        skip_reason="NO_UPSTREAM_ENDPOINTS",
        yield_status=None,
        pass_result=None,
        raw_response_payload=None,
        counts=None,
        error=None,
    )


def _make_failed_outcome(error: Exception) -> PassAttemptOutcome:
    return PassAttemptOutcome(
        execution_status="FAILED",
        skip_reason=None,
        yield_status=None,
        pass_result=None,
        raw_response_payload={"diagnostics": {"pipeline_error": "service down"}},
        counts=None,
        error=error,
    )


def _make_task_self(
    retries: int = 0,
    max_retries: int = 3,
    request_id: str = "task-abc-123",
):
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
    test's transaction-rollback fixture can still clean up after the task."""

    def __init__(self, session: Session):
        self._session = session

    def close(self):
        # Suppress — db_session fixture handles teardown
        pass

    def commit(self):
        # Flush to make writes visible within the same transaction
        self._session.flush()

    def rollback(self):
        pass  # tests rollback at teardown

    def __getattr__(self, name: str):
        return getattr(self._session, name)


def _build_patches(db_session: Session, *, manifest=None, advance_return=None):
    """Return a list of (patch_obj, mock_obj) tuples for the common mocks."""
    proxy = _NoCloseProxy(db_session)
    if manifest is None:
        manifest = _fake_manifest()

    patches = [
        patch("app.workers.pipeline._get_db", return_value=proxy),
        patch("app.workers.pipeline._build_docling_document_json", return_value={"text": "..."}),
        patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
        patch("app.workers.pipeline.load_ontology", return_value=_MINIMAL_ONTOLOGY),
        patch("app.workers.pipeline._rehydrate_upstream_refs_from_persisted_passes", return_value={}),
        patch("app.workers.pipeline._write_stage_run", return_value=None),
        patch("app.workers.pipeline._try_advance_phase", return_value=advance_return),
    ]
    return patches


@contextmanager
def _patched(db_session: Session, *, manifest=None):
    """Context manager that starts/stops all common patches and yields a dict
    of mock objects keyed by a short name for easy access."""
    patches = _build_patches(db_session, manifest=manifest)
    mocks: dict[str, Any] = {}
    started = []
    keys = ["_get_db", "_build_docling_document_json", "load_bundle_manifest",
            "load_ontology", "_rehydrate_upstream_refs", "_write_stage_run",
            "_try_advance_phase"]
    for p, key in zip(patches, keys):
        m = p.start()
        mocks[key] = m
        started.append(p)
    try:
        yield mocks
    finally:
        for p in started:
            p.stop()


def _invoke(self_mock, document_id: str, run_id: str, pass_name: str) -> Any:
    """Call the raw task body, bypassing ``guard_stage_run``.

    ``guard_stage_run`` catches unhandled exceptions and calls
    ``_terminalize_doc_and_run`` before re-raising.  Tests that mock
    ``_terminalize_doc_and_run`` and assert ``called_once`` would see double
    calls if we invoked through the wrapper.  Using ``__wrapped__`` (set by
    ``functools.wraps``) skips the guard decorator entirely.
    """
    raw_fn = derive_ontology_graph_pass.run.__wrapped__
    return raw_fn(self_mock, document_id, run_id, pass_name)


# ---------------------------------------------------------------------------
# Helper-unit tests (no DB needed)
# ---------------------------------------------------------------------------


class TestPhaseKeyHelper:
    def test_entity_pass_prefixed(self):
        assert _phase_key("radar_identity") == "entity_pass_radar_identity"

    def test_system_links_no_prefix(self):
        assert _phase_key("system_links") == "system_links"

    def test_merge_no_prefix(self):
        assert _phase_key("merge") == "merge"


class TestRetryDelayHelper:
    def test_first_retry_30s(self):
        assert _retry_delay(0) == 30

    def test_second_retry_60s(self):
        assert _retry_delay(1) == 60

    def test_third_retry_120s(self):
        assert _retry_delay(2) == 120

    def test_capped_at_300(self):
        assert _retry_delay(10) == 300


# ---------------------------------------------------------------------------
# Cancel / skip / already-resolved tests
# ---------------------------------------------------------------------------


class TestCancelAndSkip:
    def test_skip_when_run_cancelled(self, db_session, pipeline_run_factory):
        """Run in FAILED status → task exits immediately without executing the pass."""
        run_id = pipeline_run_factory(status="FAILED")
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch("app.workers.pipeline._execute_pass_attempt") as mock_exec:
                result = _invoke(
                    _make_task_self(), doc_id, str(run_id), "radar_identity"
                )
        mock_exec.assert_not_called()
        assert result["skipped"] == "cancelled"

    def test_skip_when_run_missing(self, db_session, pipeline_run_factory):
        """Non-existent run_id → is_run_cancelled returns True (hard-deleted)."""
        doc_id = str(uuid.uuid4())
        missing_run_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch("app.workers.pipeline._execute_pass_attempt") as mock_exec:
                result = _invoke(
                    _make_task_self(), doc_id, missing_run_id, "radar_identity"
                )
        mock_exec.assert_not_called()
        assert "skipped" in result

    def test_skip_when_already_resolved_advances_phase(
        self, db_session, pipeline_run_factory
    ):
        """Pre-seeded terminal pass-output → task skips execution, marks phase, advances."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Pre-seed a terminal row within the same transaction
        save_pass_output(
            db_session,
            pipeline_run_id=run_id,
            stage_run_id=None,
            pass_name="radar_identity",
            attempt=1,
            execution_status="COMPLETE",
            skip_reason=None,
            yield_status="HIT",
            extract_pass_response={"status": "ok"},
            primary_entities_extracted=3,
            bridge_entities_extracted=0,
            relationships_extracted=1,
            relationships_rejected=0,
            diagnostics={},
            field_provenance=[],
        )
        db_session.flush()

        with _patched(db_session) as mocks:
            with patch("app.workers.pipeline._execute_pass_attempt") as mock_exec:
                with patch("app.workers.pipeline.mark_phase_terminal") as mock_terminal:
                    result = _invoke(
                        _make_task_self(), doc_id, str(run_id), "radar_identity"
                    )

        mock_exec.assert_not_called()
        mock_terminal.assert_called_once()
        mocks["_try_advance_phase"].assert_called_once()
        assert result["skipped"] == "already_resolved"


# ---------------------------------------------------------------------------
# Successful completion path
# ---------------------------------------------------------------------------


class TestSuccessfulCompletion:
    def test_complete_writes_terminal_pass_output_and_marks_phase_succeeded(
        self, db_session, pipeline_run_factory
    ):
        """COMPLETE outcome → single pass-output row written with execution_status=COMPLETE."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_complete_outcome(),
            ):
                with patch("app.workers.pipeline.mark_phase_terminal") as mock_terminal:
                    result = _invoke(
                        _make_task_self(), doc_id, str(run_id), "radar_identity"
                    )

        assert result["execution_status"] == "COMPLETE"
        db_session.expire_all()
        row = load_pass_output(db_session, run_id, "radar_identity")
        assert row is not None
        assert row.execution_status == "COMPLETE"
        mock_terminal.assert_called_once()
        call_kwargs = mock_terminal.call_args[1]
        assert call_kwargs["result"] == "succeeded"

    def test_persists_full_extract_pass_response(
        self, db_session, pipeline_run_factory
    ):
        """COMPLETE outcome → extract_pass_response_json carries the raw payload."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())
        raw_payload = {"status": "ok", "entities": [{"name": "SA-2"}], "custom_key": 42}

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_complete_outcome(raw_payload=raw_payload),
            ):
                _invoke(_make_task_self(), doc_id, str(run_id), "radar_identity")

        db_session.expire_all()
        row = load_pass_output(db_session, run_id, "radar_identity")
        assert row is not None
        assert row.extract_pass_response_json == raw_payload

    def test_skipped_pass_records_skip_reason_and_phase_result_skipped(
        self, db_session, pipeline_run_factory
    ):
        """SKIPPED outcome → execution_status=SKIPPED, skip_reason set, phase result=skipped."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_skipped_outcome(),
            ):
                with patch("app.workers.pipeline.mark_phase_terminal") as mock_terminal:
                    result = _invoke(
                        _make_task_self(), doc_id, str(run_id), "radar_identity"
                    )

        assert result["execution_status"] == "SKIPPED"
        db_session.expire_all()
        row = load_pass_output(db_session, run_id, "radar_identity")
        assert row is not None
        assert row.execution_status == "SKIPPED"
        assert row.skip_reason == "NO_UPSTREAM_ENDPOINTS"
        call_kwargs = mock_terminal.call_args[1]
        assert call_kwargs["result"] == "skipped"

    def test_skipped_pass_counts_as_phase_terminal_for_fanin(
        self, db_session, pipeline_run_factory
    ):
        """SKIPPED outcome → _try_advance_phase is called so fan-in can proceed."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_skipped_outcome(),
            ):
                _invoke(_make_task_self(), doc_id, str(run_id), "radar_identity")

        mocks["_try_advance_phase"].assert_called_once()


# ---------------------------------------------------------------------------
# Retryable failure path (terminal-only write semantics)
# ---------------------------------------------------------------------------


class TestRetryableFailurePath:
    def test_pass_retryable_with_retries_left_does_not_write_pass_output(
        self, db_session, pipeline_run_factory
    ):
        """PassRetryable + retries remaining → NO pipeline_pass_outputs row written."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassRetryable("service down")),
            ):
                with pytest.raises(CeleryRetry):
                    _invoke(
                        _make_task_self(retries=0, max_retries=3),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )

        db_session.expire_all()
        assert not is_pass_already_resolved(db_session, run_id, "radar_identity")

    def test_pass_transport_error_with_retries_left_does_not_write_pass_output(
        self, db_session, pipeline_run_factory
    ):
        """PassTransportError + retries remaining → NO pipeline_pass_outputs row written."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassTransportError("connection reset")),
            ):
                with pytest.raises(CeleryRetry):
                    _invoke(
                        _make_task_self(retries=0, max_retries=3),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )

        db_session.expire_all()
        assert not is_pass_already_resolved(db_session, run_id, "radar_identity")

    def test_pass_retryable_writes_stagerun_audit_per_attempt(
        self, db_session, pipeline_run_factory
    ):
        """Even when no pass-output row is written, _write_stage_run is called once."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Use _patched but override _write_stage_run so we can spy on calls
        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassRetryable("down")),
            ):
                with pytest.raises(CeleryRetry):
                    _invoke(
                        _make_task_self(retries=0, max_retries=3),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )
        mocks["_write_stage_run"].assert_called_once()


# ---------------------------------------------------------------------------
# Retry exhaustion (r4 — explicit terminal-only write)
# ---------------------------------------------------------------------------


class TestRetryExhaustion:
    def test_retry_exhausted_writes_terminal_pass_output_failed_for_optional(
        self, db_session, pipeline_run_factory
    ):
        """Optional pass + retries exhausted → FAILED pass-output row with retry_exhausted=True."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session, manifest=_fake_manifest(required=False)) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassRetryable("still down")),
            ):
                # retries=3 == max_retries=3 → retries_left=False → terminal
                result = _invoke(
                    _make_task_self(retries=3, max_retries=3),
                    doc_id,
                    str(run_id),
                    "radar_identity",
                )

        assert result["execution_status"] == "FAILED"
        assert result["reason"] == "retry_exhausted"

        db_session.expire_all()
        row = load_pass_output(db_session, run_id, "radar_identity")
        assert row is not None
        assert row.execution_status == "FAILED"
        assert row.diagnostics_json.get("retry_exhausted") is True

    def test_retry_exhausted_marks_phase_result_failed(
        self, db_session, pipeline_run_factory
    ):
        """Retries exhausted → mark_phase_terminal called with result='failed'."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session, manifest=_fake_manifest(required=False)) as mocks:
            with patch("app.workers.pipeline.mark_phase_terminal") as mock_terminal:
                with patch(
                    "app.workers.pipeline._execute_pass_attempt",
                    return_value=_make_failed_outcome(PassRetryable("still down")),
                ):
                    _invoke(
                        _make_task_self(retries=3, max_retries=3),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )

        mock_terminal.assert_called_once()
        call_kwargs = mock_terminal.call_args[1]
        assert call_kwargs["result"] == "failed"

    def test_required_pass_retry_exhausted_terminalizes_run(
        self, db_session, pipeline_run_factory
    ):
        """Required pass + retries exhausted → _terminalize_doc_and_run called, IngestFailed raised.

        r4 critical path: explicit exhaustion handling before MaxRetriesExceededError
        short-circuit ensures cleanup runs even when Celery's error path would bypass it.
        """
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session, manifest=_fake_manifest(required=True)) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassRetryable("down")),
            ):
                with patch(
                    "app.workers.pipeline._terminalize_doc_and_run"
                ) as mock_terminalize:
                    with pytest.raises(IngestFailed):
                        _invoke(
                            _make_task_self(retries=3, max_retries=3),
                            doc_id,
                            str(run_id),
                            "radar_identity",
                        )

        mock_terminalize.assert_called_once()

    def test_optional_pass_retry_exhausted_does_not_terminalize(
        self, db_session, pipeline_run_factory
    ):
        """Optional pass + retries exhausted → run not terminated, _try_advance_phase called."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session, manifest=_fake_manifest(required=False)) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassRetryable("down")),
            ):
                with patch(
                    "app.workers.pipeline._terminalize_doc_and_run"
                ) as mock_terminalize:
                    result = _invoke(
                        _make_task_self(retries=3, max_retries=3),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )

        mock_terminalize.assert_not_called()
        mocks["_try_advance_phase"].assert_called_once()
        assert result["execution_status"] == "FAILED"
        assert result["reason"] == "retry_exhausted"


# ---------------------------------------------------------------------------
# Terminal failure path (non-retryable)
# ---------------------------------------------------------------------------


class TestTerminalFailurePath:
    def test_pass_terminal_writes_terminal_pass_output_immediately(
        self, db_session, pipeline_run_factory
    ):
        """PassTerminal → pass-output written immediately with retry_exhausted=False."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassTerminal("bad schema")),
            ):
                with pytest.raises(IngestFailed):
                    _invoke(
                        _make_task_self(retries=0, max_retries=3),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )

        db_session.expire_all()
        row = load_pass_output(db_session, run_id, "radar_identity")
        assert row is not None
        assert row.execution_status == "FAILED"
        # Non-retryable terminal: retry_exhausted=False
        assert row.diagnostics_json.get("retry_exhausted") is False

    def test_required_pass_terminal_terminalizes_run(
        self, db_session, pipeline_run_factory
    ):
        """Required pass + PassTerminal → _terminalize_doc_and_run called, IngestFailed raised."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassTerminal("bad schema")),
            ):
                with patch(
                    "app.workers.pipeline._terminalize_doc_and_run"
                ) as mock_terminalize:
                    with pytest.raises(IngestFailed):
                        _invoke(
                            _make_task_self(),
                            doc_id,
                            str(run_id),
                            "radar_identity",
                        )

        mock_terminalize.assert_called_once()

    def test_optional_pass_terminal_does_not_terminalize(
        self, db_session, pipeline_run_factory
    ):
        """Optional pass + PassTerminal → run continues, _try_advance_phase called."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        with _patched(db_session, manifest=_fake_manifest(required=False)) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_failed_outcome(PassTerminal("bad schema")),
            ):
                with patch(
                    "app.workers.pipeline._terminalize_doc_and_run"
                ) as mock_terminalize:
                    result = _invoke(
                        _make_task_self(),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )

        mock_terminalize.assert_not_called()
        mocks["_try_advance_phase"].assert_called_once()
        assert result["execution_status"] == "FAILED"
        assert result["reason"] == "terminal"

    def test_pass_terminal_during_parse_carries_raw_payload(
        self, db_session, pipeline_run_factory
    ):
        """Parse-stage failure: raw_response_payload populated even though pass_result=None."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())
        raw_payload = {"status": "ok", "entities": [{"malformed": True}]}

        outcome = PassAttemptOutcome(
            execution_status="FAILED",
            skip_reason=None,
            yield_status=None,
            pass_result=None,
            raw_response_payload=raw_payload,
            counts=None,
            error=PassTerminal("template validation failed"),
        )

        with _patched(db_session) as mocks:
            with patch("app.workers.pipeline._execute_pass_attempt", return_value=outcome):
                with pytest.raises(IngestFailed):
                    _invoke(
                        _make_task_self(),
                        doc_id,
                        str(run_id),
                        "radar_identity",
                    )

        db_session.expire_all()
        row = load_pass_output(db_session, run_id, "radar_identity")
        assert row is not None
        assert row.extract_pass_response_json == raw_payload


# ---------------------------------------------------------------------------
# Attempt counter
# ---------------------------------------------------------------------------


class TestAttemptCounter:
    def test_attempt_field_reflects_celery_retry_counter(
        self, db_session, pipeline_run_factory
    ):
        """Terminal write's ``attempt`` column equals self.request.retries + 1."""
        run_id = pipeline_run_factory()
        doc_id = str(uuid.uuid4())

        # Simulate 3rd attempt (retries=2): attempt_n should be 3.
        with _patched(db_session, manifest=_fake_manifest(required=False)) as mocks:
            with patch(
                "app.workers.pipeline._execute_pass_attempt",
                return_value=_make_complete_outcome(),
            ):
                _invoke(
                    _make_task_self(retries=2, max_retries=3),
                    doc_id,
                    str(run_id),
                    "radar_identity",
                )

        db_session.expire_all()
        row = load_pass_output(db_session, run_id, "radar_identity")
        assert row is not None
        assert row.attempt == 3  # retries=2 → attempt_n = 2 + 1 = 3


# ---------------------------------------------------------------------------
# _rehydrate_upstream_refs_from_persisted_passes unit tests
# ---------------------------------------------------------------------------


class TestRehydrateUpstreamRefs:
    """Direct unit tests for _rehydrate_upstream_refs_from_persisted_passes.

    These tests target the orchestration logic of the helper — the pass-def
    branching, the dependency-skip logic, and the parse-and-walk wiring.
    The heavy helpers (_parse_pass_response, _build_pre_merge_walk_summary,
    _extend_upstream_refs) are mocked so tests stay fast and focused.
    """

    def test_rehydrate_returns_empty_for_document_only_pass(self):
        """input_mode='document_only' → returns {} immediately without any DB query."""
        from app.workers.pipeline import _rehydrate_upstream_refs_from_persisted_passes

        pass_def = _fake_pass_def(input_mode="document_only", depends_on=[])
        manifest = _fake_manifest([pass_def])
        mock_db = MagicMock()

        with patch("app.workers.pipeline.load_pass_output") as mock_load:
            result = _rehydrate_upstream_refs_from_persisted_passes(
                mock_db, str(uuid.uuid4()), pass_def, manifest, _MINIMAL_ONTOLOGY, str(uuid.uuid4())
            )

        assert result == {}
        mock_load.assert_not_called()

    def test_rehydrate_returns_empty_when_no_dependencies(self):
        """input_mode='document_plus_entity_refs' but depends_on=[] → returns {} immediately."""
        from app.workers.pipeline import _rehydrate_upstream_refs_from_persisted_passes

        pass_def = _fake_pass_def(input_mode="document_plus_entity_refs", depends_on=[])
        manifest = _fake_manifest([pass_def])
        mock_db = MagicMock()

        with patch("app.workers.pipeline.load_pass_output") as mock_load:
            result = _rehydrate_upstream_refs_from_persisted_passes(
                mock_db, str(uuid.uuid4()), pass_def, manifest, _MINIMAL_ONTOLOGY, str(uuid.uuid4())
            )

        assert result == {}
        mock_load.assert_not_called()

    def test_rehydrate_skips_non_complete_dependency_outputs(self, db_session, pipeline_run_factory):
        """A FAILED dependency row is loaded but skipped; upstream_refs stays empty."""
        from app.workers.pipeline import _rehydrate_upstream_refs_from_persisted_passes

        run_id = pipeline_run_factory()

        # Pre-seed a FAILED terminal row for the dependency
        save_pass_output(
            db_session,
            pipeline_run_id=run_id,
            stage_run_id=None,
            pass_name="radar_identity",
            attempt=1,
            execution_status="FAILED",
            skip_reason=None,
            yield_status=None,
            extract_pass_response={},
            primary_entities_extracted=0,
            bridge_entities_extracted=0,
            relationships_extracted=0,
            relationships_rejected=0,
            diagnostics={"pipeline_error": "service down"},
            field_provenance=[],
        )
        db_session.flush()

        # system_links depends on radar_identity
        dep_pass_def = _fake_pass_def(name="radar_identity")
        sl_pass_def = _fake_pass_def(
            name="system_links",
            input_mode="document_plus_entity_refs",
            depends_on=["radar_identity"],
        )
        manifest = _fake_manifest([dep_pass_def, sl_pass_def])

        with patch("app.workers.pipeline._parse_pass_response") as mock_parse:
            with patch("app.workers.pipeline._extend_upstream_refs") as mock_extend:
                result = _rehydrate_upstream_refs_from_persisted_passes(
                    db_session, run_id, sl_pass_def, manifest, _MINIMAL_ONTOLOGY, str(uuid.uuid4())
                )

        assert result == {}
        mock_parse.assert_not_called()
        mock_extend.assert_not_called()

    def test_rehydrate_calls_parse_and_walk_for_complete_dependencies(
        self, db_session, pipeline_run_factory
    ):
        """A COMPLETE dependency row is loaded; _parse_pass_response and _extend_upstream_refs
        are both called with the correct arguments."""
        from app.workers.pipeline import _rehydrate_upstream_refs_from_persisted_passes

        run_id = pipeline_run_factory()
        raw_payload = {"status": "ok", "pass_output": {"entities": [{"name": "SA-2"}]}}

        # Pre-seed a COMPLETE terminal row for the dependency
        save_pass_output(
            db_session,
            pipeline_run_id=run_id,
            stage_run_id=None,
            pass_name="radar_identity",
            attempt=1,
            execution_status="COMPLETE",
            skip_reason=None,
            yield_status="HIT",
            extract_pass_response=raw_payload,
            primary_entities_extracted=1,
            bridge_entities_extracted=0,
            relationships_extracted=0,
            relationships_rejected=0,
            diagnostics={},
            field_provenance=[],
        )
        db_session.flush()

        dep_pass_def = _fake_pass_def(name="radar_identity")
        sl_pass_def = _fake_pass_def(
            name="system_links",
            input_mode="document_plus_entity_refs",
            depends_on=["radar_identity"],
        )
        manifest = _fake_manifest([dep_pass_def, sl_pass_def])
        doc_id = str(uuid.uuid4())

        fake_pass_result = _fake_pass_result()
        fake_walk = SimpleNamespace(entities=[], raw_edge_count=0)

        with patch(
            "app.workers.pipeline._parse_pass_response", return_value=fake_pass_result
        ) as mock_parse:
            with patch(
                "app.workers.pipeline._build_pre_merge_walk_summary", return_value=fake_walk
            ) as mock_walk:
                with patch("app.workers.pipeline._extend_upstream_refs") as mock_extend:
                    result = _rehydrate_upstream_refs_from_persisted_passes(
                        db_session, run_id, sl_pass_def, manifest, _MINIMAL_ONTOLOGY, doc_id
                    )

        # _parse_pass_response called with the stored JSON and the dep's pass_def
        mock_parse.assert_called_once()
        parse_call_args = mock_parse.call_args[0]
        assert parse_call_args[0] == raw_payload
        assert parse_call_args[1].name == "radar_identity"

        # _build_pre_merge_walk_summary called with the parse result
        mock_walk.assert_called_once()

        # _extend_upstream_refs called to merge into upstream_refs
        mock_extend.assert_called_once()
        extend_call_args = mock_extend.call_args[0]
        # First arg is the accumulating upstream_refs dict
        assert isinstance(extend_call_args[0], dict)
        # Second arg is the fake pass result with pre_merge_walk attached
        assert extend_call_args[1] is fake_pass_result
        assert fake_pass_result.pre_merge_walk is fake_walk
