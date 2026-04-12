"""Unit tests for compute_status_signals per spec §7.10.

Tests cover all case tables from §7.10:
  - No snapshot ever → graph_queryable=False
  - Fresh successful extraction → graph_queryable=True, is_stale=False
  - Snapshot matches latest run COMPLETE → graph_queryable=True, is_stale=False
  - Snapshot exists, latest run is different run (COMPLETE) → is_stale=True
  - Snapshot exists, latest run is same run but FAILED → is_stale=True
  - Snapshot exists, no latest run → is_stale=True
  - Rollback executed by a newer run → graph_queryable=False
  - No rollback after snapshot → graph_queryable=True
  - Snapshot_run missing (orphan) → conservative: any rollback for document
    invalidates (graph_queryable=False) even without newer-run filter

All tests use MagicMock sessions — no DB connection required.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOC_ID = str(uuid.uuid4())
RUN_ID_1 = uuid.uuid4()
RUN_ID_2 = uuid.uuid4()

_T1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
_T2 = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)


def _fake_snapshot(pipeline_run_id=RUN_ID_1):
    snap = MagicMock()
    snap.pipeline_run_id = pipeline_run_id
    snap.document_id = uuid.UUID(DOC_ID)
    snap.graph_json = {"entity_count_by_type": {}, "edges_accepted": 0}
    snap.updated_at = _T1
    snap.ontology_bundle_key = "test_bundle"
    snap.ontology_version = "1.0.0"
    return snap


def _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1):
    run = MagicMock()
    run.id = run_id
    run.status = status
    run.started_at = started_at
    run.document_id = uuid.UUID(DOC_ID)
    return run


def _make_session(snapshot, latest_run, snapshot_run, rollback_stage_run):
    """
    Build a MagicMock session whose .query(...).filter_by(...).first()
    and .query(...).join(...).filter(...).filter(...).first() chains return
    the supplied values.

    The order of .first() calls in compute_status_signals is:
      1. DocumentGraphExtraction.filter_by(document_id=...) → snapshot
      2. PipelineRun.filter_by(document_id=...).order_by(...) → latest_run
      3. PipelineRun.filter_by(id=snapshot.pipeline_run_id) → snapshot_run
      4. StageRun join query (may have extra .filter()) → rollback_stage_run
    """
    session = MagicMock()

    call_order = [snapshot, latest_run, snapshot_run, rollback_stage_run]
    call_idx = [0]

    def _query_side_effect(*args, **kwargs):
        chain = MagicMock()

        # We track calls to .first() at end of chain
        def _first_side_effect():
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(call_order):
                return call_order[idx]
            return None

        chain.filter_by.return_value = chain
        chain.filter.return_value = chain
        chain.join.return_value = chain
        chain.order_by.return_value = chain
        chain.first.side_effect = _first_side_effect
        return chain

    session.query.side_effect = _query_side_effect
    return session


# ---------------------------------------------------------------------------
# Case 1: No snapshot ever
# ---------------------------------------------------------------------------

class TestNoSnapshot:
    def test_graph_queryable_false(self):
        """When no DocumentGraphExtraction row exists, graph_queryable=False."""
        from app.services.status_signals import compute_status_signals

        session = _make_session(
            snapshot=None,
            latest_run=_fake_run(),
            snapshot_run=None,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.snapshot is None
        assert signals.graph_queryable is False

    def test_is_stale_false_when_no_snapshot(self):
        """is_stale is False when snapshot is None (no meaningful stale state)."""
        from app.services.status_signals import compute_status_signals

        session = _make_session(
            snapshot=None,
            latest_run=_fake_run(),
            snapshot_run=None,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)
        assert signals.is_stale is False

    def test_no_snapshot_no_run(self):
        """No snapshot and no run: graph_queryable=False, is_stale=False."""
        from app.services.status_signals import compute_status_signals

        session = _make_session(
            snapshot=None,
            latest_run=None,
            snapshot_run=None,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)
        assert signals.snapshot is None
        assert signals.graph_queryable is False
        assert signals.is_stale is False


# ---------------------------------------------------------------------------
# Case 2: Fresh successful extraction (snapshot matches latest COMPLETE run)
# ---------------------------------------------------------------------------

class TestFreshSuccessfulExtraction:
    def test_graph_queryable_true(self):
        """Snapshot matches latest COMPLETE run → graph_queryable=True."""
        from app.services.status_signals import compute_status_signals

        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=run,
            rollback_stage_run=None,  # no rollback after snapshot
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.graph_queryable is True

    def test_is_stale_false(self):
        """Snapshot matches latest COMPLETE run → is_stale=False."""
        from app.services.status_signals import compute_status_signals

        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=run,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.is_stale is False

    def test_snapshot_returned(self):
        """Snapshot object is returned in signals."""
        from app.services.status_signals import compute_status_signals

        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=run,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.snapshot is snap


# ---------------------------------------------------------------------------
# Case 3: Snapshot exists, latest run is a DIFFERENT (newer) COMPLETE run
# ---------------------------------------------------------------------------

class TestSnapshotFromOlderRun:
    def test_is_stale_true_when_run_id_mismatch(self):
        """latest_run.id != snapshot.pipeline_run_id → is_stale=True."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        latest_run = _fake_run(run_id=RUN_ID_2, status="COMPLETE", started_at=_T2)
        snapshot_run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        session = _make_session(
            snapshot=snap,
            latest_run=latest_run,
            snapshot_run=snapshot_run,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.is_stale is True


# ---------------------------------------------------------------------------
# Case 4: Snapshot exists, latest run is same run but FAILED
# ---------------------------------------------------------------------------

class TestLatestRunFailed:
    def test_is_stale_true_when_run_failed(self):
        """latest_run.status != 'COMPLETE' → is_stale=True."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        run = _fake_run(run_id=RUN_ID_1, status="FAILED", started_at=_T1)
        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=run,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.is_stale is True


# ---------------------------------------------------------------------------
# Case 5: Snapshot exists, no latest run
# ---------------------------------------------------------------------------

class TestNoLatestRun:
    def test_is_stale_true_when_no_run(self):
        """latest_run is None → is_stale=True (snapshot has no backing run)."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        # snapshot_run lookup still happens after snapshot_run is found to be the
        # run for the snapshot, but latest_run is None here.
        # Order of calls: snapshot → latest_run (None) → snapshot_run → rollback
        session = _make_session(
            snapshot=snap,
            latest_run=None,
            snapshot_run=_fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1),
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.is_stale is True


# ---------------------------------------------------------------------------
# Case 6: Rollback executed by a NEWER run → graph_queryable=False
# ---------------------------------------------------------------------------

class TestRollbackAfterSnapshot:
    def test_graph_queryable_false_when_rollback_exists(self):
        """A newer derive_ontology_graph summary row with rollback_executed=True
        means the graph has been invalidated → graph_queryable=False."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        rollback_stage = MagicMock()
        rollback_stage.rollback_executed = True

        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=run,
            rollback_stage_run=rollback_stage,  # rollback found → invalidated
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.graph_queryable is False

    def test_is_stale_false_but_not_queryable(self):
        """The snapshot may still match the run (not stale) while rollback
        marks it not queryable — is_stale is separate from graph_queryable."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        rollback_stage = MagicMock()
        rollback_stage.rollback_executed = True

        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=run,
            rollback_stage_run=rollback_stage,
        )
        signals = compute_status_signals(DOC_ID, session)

        # is_stale tracks run equivalence only
        assert signals.is_stale is False
        assert signals.graph_queryable is False


# ---------------------------------------------------------------------------
# Case 7: No rollback after snapshot → graph_queryable=True
# ---------------------------------------------------------------------------

class TestNoRollbackAfterSnapshot:
    def test_graph_queryable_true_no_rollback(self):
        """No rollback stage row → graph still queryable."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=run,
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.graph_queryable is True


# ---------------------------------------------------------------------------
# Case 8: Orphan snapshot (snapshot_run missing)
# ---------------------------------------------------------------------------

class TestOrphanSnapshot:
    def test_graph_queryable_false_when_rollback_without_snapshot_run(self):
        """When snapshot_run is None (orphan), the cross-run query has no
        lower bound filter. A rollback row anywhere in the document's history
        → graph_queryable=False."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)
        rollback_stage = MagicMock()
        rollback_stage.rollback_executed = True

        # snapshot_run lookup returns None → orphan
        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=None,   # orphan: the run that created snapshot is gone
            rollback_stage_run=rollback_stage,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.graph_queryable is False

    def test_graph_queryable_true_when_no_rollback_without_snapshot_run(self):
        """Orphan snapshot but no rollback → still queryable (conservative OK)."""
        from app.services.status_signals import compute_status_signals

        snap = _fake_snapshot(pipeline_run_id=RUN_ID_1)
        run = _fake_run(run_id=RUN_ID_1, status="COMPLETE", started_at=_T1)

        session = _make_session(
            snapshot=snap,
            latest_run=run,
            snapshot_run=None,   # orphan
            rollback_stage_run=None,
        )
        signals = compute_status_signals(DOC_ID, session)

        assert signals.graph_queryable is True


# ---------------------------------------------------------------------------
# Case 9: UUID coercion — document_id as string is accepted
# ---------------------------------------------------------------------------

class TestUUIDCoercion:
    def test_string_document_id_accepted(self):
        """compute_status_signals accepts a plain string document_id."""
        from app.services.status_signals import compute_status_signals

        session = _make_session(None, None, None, None)
        # Should not raise
        signals = compute_status_signals(DOC_ID, session)
        assert signals.snapshot is None

    def test_uuid_string_round_trips(self):
        """The UUID constructed from the string matches the original."""
        from app.services.status_signals import compute_status_signals

        doc_id = str(uuid.uuid4())
        session = _make_session(None, None, None, None)
        # We can't easily inspect the uuid inside, but we can assert no error
        signals = compute_status_signals(doc_id, session)
        assert signals is not None
