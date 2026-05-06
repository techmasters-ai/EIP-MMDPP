"""Per-phase dispatch state machine — SSoT for ``pipeline_runs.dispatched_phases``.

This module owns every read and write on the ``dispatched_phases`` JSONB column
of ``ingest.pipeline_runs``.  The column is a map keyed by phase name (e.g.,
``entity_pass_radar_identity``, ``system_links``, ``merge``); each value is a
``PhaseEntry`` JSON object that records the lifecycle state of one dispatch slot.

Lifecycle invariant
-------------------
Every phase follows a strict forward-only state machine::

    (absent) → claimed → dispatched → completed

The only backward transition is *deletion* (``reclaim_stale_phase``), which
resets a stale ``claimed`` or ``dispatched`` entry back to absent so a fresh
claim can be won.  No helper advances state backwards or skips a step.

Compare-and-reset SQL predicates
---------------------------------
Every UPDATE includes the *expected current state* (and timestamp, where
applicable) in its WHERE clause.  This is the "compare-and-reset" pattern:

  * ``claim_phase`` — WHERE … NOT (dispatched_phases ? :phase)
    Guarantees the INSERT-if-absent is atomic: only one caller wins even when
    two workers race to claim the same phase at the same time.

  * ``mark_phase_dispatched`` — WHERE … state = :expected_state
    If a reconciler reset the phase (deleted it and let a fresh claim win)
    between our claim and our post-``.delay()`` mark, the predicate misses and
    we return False rather than clobbering the fresh claim.

  * ``mark_phase_terminal`` — WHERE … state IN ('claimed', 'dispatched')
    Idempotent: if the phase is already completed (e.g., two workers converge),
    the second call returns False without touching the stored row.

  * ``reclaim_stale_phase`` — WHERE … state = :s AND timestamp < :cutoff
    Without the timestamp predicate a reconciler read that predates a fresh
    claim could clobber a valid claimed entry whose wall-clock age happens to
    still look stale.  The timestamp predicate ensures we only delete the *same
    stale entry* we identified during the read.

Caller owns ``db.commit()``
----------------------------
Consistent with ``pass_outputs_store.py`` and the surrounding worker layer,
these helpers do **not** commit.  The caller (typically a Celery task or the
reconciler) owns the transaction boundary so that phase state updates can be
composed with other writes atomically.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reusable type aliases
# ---------------------------------------------------------------------------

PhaseState = Literal["claimed", "dispatched", "completed"]
PhaseResult = Literal["succeeded", "failed", "skipped"]

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

_CANCELLED_STATUSES = frozenset({"FAILED", "COMPLETE", "PARTIAL_COMPLETE", "CANCELLED"})


@dataclass
class PhaseEntry:
    """Parsed representation of one phase slot in ``dispatched_phases``."""

    state: PhaseState
    result: PhaseResult | None
    task_id: str | None
    claimed_at: datetime
    dispatched_at: datetime | None
    completed_at: datetime | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def claim_phase(db: Session, run_id: uuid.UUID | str, phase_name: str) -> bool:
    """Atomic INSERT-if-absent for a phase slot.

    Writes a ``claimed`` entry into ``dispatched_phases[:phase_name]`` only when
    that key is not yet present.  The SQL predicate ``NOT (dispatched_phases ?
    :phase)`` makes this a compare-and-reset claim: if two workers race, exactly
    one UPDATE will match a row (rowcount==1) and the other will find the
    predicate false (rowcount==0).

    Returns True iff this caller won the claim.  Returns False if the phase was
    already present (another worker claimed it first).  Returns False if the run
    row itself is missing — the caller should check separately whether to raise.
    """
    now = _now_iso()
    result = db.execute(
        text(
            """
            UPDATE ingest.pipeline_runs
            SET dispatched_phases = jsonb_set(
                dispatched_phases,
                ARRAY[:phase],
                jsonb_build_object(
                    'state',        'claimed',
                    'result',       NULL,
                    'task_id',      NULL,
                    'claimed_at',   :now_iso,
                    'dispatched_at', NULL,
                    'completed_at', NULL
                ),
                true
            )
            WHERE id = :run_id
              AND NOT (dispatched_phases ? :phase)
            """
        ),
        {"run_id": str(run_id), "phase": phase_name, "now_iso": now},
    )
    return result.rowcount == 1


def mark_phase_dispatched(
    db: Session,
    run_id: uuid.UUID | str,
    phase_name: str,
    task_id: str,
    expected_state: PhaseState = "claimed",
) -> bool:
    """Advance a phase slot from *claimed* → *dispatched*.

    Must be called immediately after ``.delay()`` returns a task ID so the
    Celery task ID is durably associated with the phase slot.

    The compare-and-reset predicate ``state = :expected_state`` guards against
    the following race: a reconciler detects a stale claim, deletes the entry,
    and lets a fresh claim win — all before our post-``.delay()`` mark arrives.
    Without the predicate we would silently overwrite the fresh claim's state
    with our stale task_id.  With it, the UPDATE simply misses and we log a
    warning so the reconciler (Task 9) can handle the situation.

    Returns True if the state advanced.  Returns False (with a warning log) if
    the state has changed under us — the caller should treat this as a
    recoverable soft-failure; do not raise.
    """
    now = _now_iso()
    result = db.execute(
        text(
            """
            UPDATE ingest.pipeline_runs
            SET dispatched_phases = jsonb_set(
                jsonb_set(
                    jsonb_set(
                        dispatched_phases,
                        ARRAY[:phase, 'state'],
                        '"dispatched"'
                    ),
                    ARRAY[:phase, 'task_id'],
                    to_jsonb(CAST(:task_id AS text))
                ),
                ARRAY[:phase, 'dispatched_at'],
                to_jsonb(CAST(:now_iso AS text))
            )
            WHERE id = :run_id
              AND dispatched_phases->:phase->>'state' = :expected_state
            """
        ),
        {
            "run_id": str(run_id),
            "phase": phase_name,
            "task_id": task_id,
            "now_iso": now,
            "expected_state": expected_state,
        },
    )
    if result.rowcount != 1:
        logger.warning(
            "mark_phase_dispatched: state has changed under us — "
            "run_id=%s phase=%s expected_state=%s (reconciler may have re-claimed)",
            run_id,
            phase_name,
            expected_state,
        )
        return False
    return True


def mark_phase_terminal(
    db: Session,
    run_id: uuid.UUID | str,
    phase_name: str,
    *,
    result: PhaseResult,
) -> bool:
    """Advance a phase slot to *completed* with a final result.

    Accepts both ``claimed`` and ``dispatched`` as the predecessor state so that
    a phase that was claimed but never successfully dispatched (e.g., the Celery
    broker was down) can still be terminalized directly.

    The compare-and-reset predicate ``state IN ('claimed', 'dispatched')`` makes
    this idempotent: a second call when the slot is already ``completed`` returns
    False without touching the stored row.  This is safe for at-least-once
    delivery from Celery task callbacks.

    Returns True if the phase was advanced to completed.  Returns False if the
    phase was already completed (no-op) or if the row/phase is missing.
    """
    now = _now_iso()
    db_result = db.execute(
        text(
            """
            UPDATE ingest.pipeline_runs
            SET dispatched_phases = jsonb_set(
                jsonb_set(
                    jsonb_set(
                        dispatched_phases,
                        ARRAY[:phase, 'state'],
                        '"completed"'
                    ),
                    ARRAY[:phase, 'result'],
                    to_jsonb(CAST(:result AS text))
                ),
                ARRAY[:phase, 'completed_at'],
                to_jsonb(CAST(:now_iso AS text))
            )
            WHERE id = :run_id
              AND dispatched_phases->:phase->>'state' IN ('claimed', 'dispatched')
            """
        ),
        {
            "run_id": str(run_id),
            "phase": phase_name,
            "result": result,
            "now_iso": now,
        },
    )
    return db_result.rowcount == 1


def read_phase_state(
    db: Session,
    run_id: uuid.UUID | str,
    phase_name: str,
) -> PhaseEntry | None:
    """Parse the JSONB phase entry into a ``PhaseEntry`` dataclass.

    Returns None if the run row is missing OR if the phase key is absent from
    ``dispatched_phases`` (the phase has never been claimed).
    """
    row = db.execute(
        text(
            """
            SELECT dispatched_phases->:phase AS entry
            FROM ingest.pipeline_runs
            WHERE id = :run_id
            """
        ),
        {"run_id": str(run_id), "phase": phase_name},
    ).fetchone()

    if row is None:
        # Run row does not exist
        return None

    entry = row[0]
    if entry is None:
        # Phase key absent — never claimed
        return None

    return PhaseEntry(
        state=entry["state"],
        result=entry.get("result"),
        task_id=entry.get("task_id"),
        claimed_at=datetime.fromisoformat(entry["claimed_at"]),
        dispatched_at=_parse_dt(entry.get("dispatched_at")),
        completed_at=_parse_dt(entry.get("completed_at")),
    )


def reclaim_stale_phase(
    db: Session,
    run_id: uuid.UUID | str,
    phase_name: str,
    *,
    claim_threshold_s: int,
    dispatch_threshold_s: int,
) -> bool:
    """Compare-and-reset reclaim for stale ``claimed`` or ``dispatched`` entries.

    When a phase has been stuck in ``claimed`` for longer than
    ``claim_threshold_s`` seconds, or in ``dispatched`` for longer than
    ``dispatch_threshold_s`` seconds, the reconciler (Task 9) calls this helper
    to reset the slot back to absent so a fresh ``claim_phase`` call can win.

    The WHERE clause includes BOTH the expected state AND the timestamp cutoff.
    This is the critical safety guard: a reconciler that read the stale entry
    some seconds ago might race with a worker that has since won a fresh claim.
    Without the timestamp predicate the reconciler would delete the fresh claim.
    With it, only an entry whose timestamp predates the cutoff is deleted — a
    fresh claim written moments ago will have a recent timestamp and will not
    match.

    For stale ``dispatched`` entries the known Celery task ID is best-effort
    revoked before the slot is cleared.  Revoke failure (e.g., broker
    unreachable) does NOT block the reclaim — the task may run anyway, but the
    per-task idempotency guard (``is_pass_already_resolved``) will reject the
    result.

    NOTE: this helper does NOT inspect ``pipeline_pass_outputs``.  The decision
    of whether to call reclaim at all (e.g., when a pass-output row already
    exists, or when a Celery retry is in countdown) belongs to the reconciler
    in Task 9.

    Returns True if a reclaim happened (UPDATE rowcount == 1).
    """
    # First read the current entry to determine which branch to take
    row = db.execute(
        text(
            """
            SELECT dispatched_phases->:phase AS entry
            FROM ingest.pipeline_runs
            WHERE id = :run_id
            """
        ),
        {"run_id": str(run_id), "phase": phase_name},
    ).fetchone()

    if row is None or row[0] is None:
        return False

    entry = row[0]
    state = entry.get("state")

    if state == "claimed":
        result = db.execute(
            text(
                """
                UPDATE ingest.pipeline_runs
                SET dispatched_phases = dispatched_phases - :phase
                WHERE id = :run_id
                  AND dispatched_phases->:phase->>'state' = 'claimed'
                  AND (dispatched_phases->:phase->>'claimed_at')::timestamptz
                      < NOW() AT TIME ZONE 'UTC' - make_interval(secs => :threshold_s)
                """
            ),
            {
                "run_id": str(run_id),
                "phase": phase_name,
                "threshold_s": claim_threshold_s,
            },
        )
        return result.rowcount == 1

    elif state == "dispatched":
        dispatched_at_raw = entry.get("dispatched_at")
        if dispatched_at_raw is None:
            return False

        dispatched_at = datetime.fromisoformat(dispatched_at_raw)
        now = datetime.now(timezone.utc)
        age_s = (now - dispatched_at).total_seconds()
        if age_s < dispatch_threshold_s:
            return False

        # Race window: between the SELECT (read_phase_state above) and the SQL UPDATE
        # below, another worker could have advanced this phase to 'completed' (the
        # task itself called mark_phase_terminal). The UPDATE will correctly no-op
        # (WHERE state='dispatched' returns rowcount=0), but revoke() was already
        # issued — it may hit a task that has already succeeded. Operationally
        # benign because the per-task idempotency guard rejects late results, but
        # operators should expect occasional "revoking already-completed task"
        # warnings in worker logs.
        # Best-effort revoke — must not block reclaim on failure
        task_id = entry.get("task_id")
        if task_id:
            try:
                celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
            except Exception:
                logger.warning(
                    "reclaim_stale_phase: revoke failed for task_id=%s "
                    "(run_id=%s phase=%s); proceeding with reclaim anyway",
                    task_id,
                    run_id,
                    phase_name,
                    exc_info=True,
                )

        result = db.execute(
            text(
                """
                UPDATE ingest.pipeline_runs
                SET dispatched_phases = dispatched_phases - :phase
                WHERE id = :run_id
                  AND dispatched_phases->:phase->>'state' = 'dispatched'
                  AND (dispatched_phases->:phase->>'dispatched_at')::timestamptz
                      < NOW() AT TIME ZONE 'UTC' - make_interval(secs => :threshold_s)
                """
            ),
            {
                "run_id": str(run_id),
                "phase": phase_name,
                "threshold_s": dispatch_threshold_s,
            },
        )
        return result.rowcount == 1

    return False


def is_run_cancelled(db: Session, run_id: uuid.UUID | str) -> bool:
    """Return True if the run is in a terminal/cancelled status or does not exist.

    Terminal statuses: FAILED, COMPLETE, PARTIAL_COMPLETE, CANCELLED.  Also
    returns True when the row is missing entirely — ``cancel_document`` can
    hard-delete the pipeline_run row, in which case there is nothing left to
    dispatch.
    """
    row = db.execute(
        text(
            """
            SELECT status
            FROM ingest.pipeline_runs
            WHERE id = :run_id
            """
        ),
        {"run_id": str(run_id)},
    ).fetchone()

    if row is None:
        # Hard-deleted (cancel_document) — treat as cancelled
        return True

    return row[0] in _CANCELLED_STATUSES
