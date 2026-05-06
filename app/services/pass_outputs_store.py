"""Pass outputs persistence service — SSoT for per-pass extraction results.

This module is the single source of truth for ``ingest.pipeline_pass_outputs``
rows.  Every helper enforces the **terminal-only contract**: callers must only
write here when a pass has reached a final outcome (COMPLETE, FAILED with
retries exhausted, or SKIPPED).  Intermediate retry attempts belong in
``stage_runs`` only.

Key distinction maintained throughout:
  - ``count_completed_passes`` / ``load_completed_pass_outputs`` — COMPLETE only.
    Used by the merge step to know which pass outputs are available as inputs.
  - ``is_pass_already_resolved`` / ``count_terminal_passes`` — ALL terminal
    states (COMPLETE + FAILED + SKIPPED).  Used by the per-pass Celery task and
    the fan-in coordinator to detect that a pass slot has been filled and must
    not be retried or re-dispatched.

Never conflate the two: a SKIPPED or FAILED pass is terminal (do not retry)
but produced no extraction output (do not use as merge input).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.ingest import PipelinePassOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = ("COMPLETE", "FAILED", "SKIPPED")
_COMPLETE_STATUS = "COMPLETE"


def _table():
    """Return the underlying Table object for raw-DML operations."""
    return PipelinePassOutput.__table__


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_pass_output(
    db: Session,
    *,
    pipeline_run_id: uuid.UUID | str,
    stage_run_id: uuid.UUID | str | None,
    pass_name: str,
    attempt: int,
    execution_status: str,
    skip_reason: str | None,
    yield_status: str | None,
    extract_pass_response: dict[str, Any],
    primary_entities_extracted: int | None,
    bridge_entities_extracted: int | None,
    relationships_extracted: int | None,
    relationships_rejected: int | None,
    diagnostics: dict[str, Any] | None,
    field_provenance: list[Any] | None,
) -> None:
    """Upsert one terminal row for (pipeline_run_id, pass_name).

    The schema's unique constraint is (pipeline_run_id, pass_name) — only
    one row per (run, pass), regardless of attempt count.  This is
    terminal-only: callers MUST only invoke this when the pass has reached
    a terminal outcome (COMPLETE, FAILED-no-more-retries, or SKIPPED).
    Intermediate retry attempts must NOT call this — they only update
    stage_runs.

    Implementation: PostgreSQL ``INSERT ... ON CONFLICT (pipeline_run_id,
    pass_name) DO UPDATE SET ...`` so re-calling with the same key overwrites
    the prior row (defensive — the same pass should not terminalize twice in
    normal operation, but the upsert is safe if it does).

    The caller is responsible for ``db.commit()`` — this function does NOT
    commit.  Matches the surrounding service style where transaction
    boundaries are owned by the worker/task layer.
    """
    values: dict[str, Any] = {
        "id": uuid.uuid4(),
        "pipeline_run_id": pipeline_run_id,
        "stage_run_id": stage_run_id,
        "pass_name": pass_name,
        "attempt": attempt,
        "execution_status": execution_status,
        "skip_reason": skip_reason,
        "yield_status": yield_status,
        "extract_pass_response_json": extract_pass_response,
        "primary_entities_extracted": primary_entities_extracted,
        "bridge_entities_extracted": bridge_entities_extracted,
        "relationships_extracted": relationships_extracted,
        "relationships_rejected": relationships_rejected,
        "diagnostics_json": diagnostics or {},
        "field_provenance_json": field_provenance or [],
    }

    update_values = {k: v for k, v in values.items() if k != "id"}

    stmt = (
        pg_insert(_table())
        .values(**values)
        .on_conflict_do_update(
            constraint="uq_pipeline_pass_outputs_run_pass",
            set_=update_values,
        )
    )
    db.execute(stmt)


def load_pass_output(
    db: Session,
    pipeline_run_id: uuid.UUID | str,
    pass_name: str,
) -> PipelinePassOutput | None:
    """Return the single terminal row for (run, pass), or None if not yet resolved.

    Covers all execution_status values — COMPLETE, FAILED, and SKIPPED.
    Returns None only if no row has been written yet (the pass has not reached
    any terminal state).
    """
    stmt = select(PipelinePassOutput).where(
        PipelinePassOutput.pipeline_run_id == pipeline_run_id,
        PipelinePassOutput.pass_name == pass_name,
    )
    return db.execute(stmt).scalars().first()


def load_completed_pass_outputs(
    db: Session,
    pipeline_run_id: uuid.UUID | str,
) -> dict[str, PipelinePassOutput]:
    """Return pass_name → row mapping for COMPLETE passes only.

    Used by the merge step to rehydrate extraction inputs.  SKIPPED and FAILED
    rows are intentionally excluded: skipped passes produced no output and
    failed passes did not either.  Only COMPLETE rows carry usable
    extract_pass_response_json.
    """
    stmt = select(PipelinePassOutput).where(
        PipelinePassOutput.pipeline_run_id == pipeline_run_id,
        PipelinePassOutput.execution_status == _COMPLETE_STATUS,
    )
    rows = db.execute(stmt).scalars().all()
    return {row.pass_name: row for row in rows}


def is_pass_already_resolved(
    db: Session,
    pipeline_run_id: uuid.UUID | str,
    pass_name: str,
) -> bool:
    """Return True iff a terminal row exists for (run, pass) in ANY status.

    Covers COMPLETE (don't redo successful work), FAILED (terminal failure is
    final; no retry), and SKIPPED (authorized skip is final).  Used at the top
    of the per-pass Celery task body as a fast idempotency guard.
    """
    stmt = select(func.count()).select_from(PipelinePassOutput).where(
        PipelinePassOutput.pipeline_run_id == pipeline_run_id,
        PipelinePassOutput.pass_name == pass_name,
    )
    count: int = db.execute(stmt).scalar_one()
    return count > 0


def count_completed_passes(
    db: Session,
    pipeline_run_id: uuid.UUID | str,
) -> int:
    """Count distinct passes with execution_status='COMPLETE' for this run.

    Used by the merge step to measure how many pass outputs are available as
    inputs.  SKIPPED and FAILED passes do NOT count.
    """
    stmt = select(func.count()).select_from(PipelinePassOutput).where(
        PipelinePassOutput.pipeline_run_id == pipeline_run_id,
        PipelinePassOutput.execution_status == _COMPLETE_STATUS,
    )
    return db.execute(stmt).scalar_one()


def count_terminal_passes(
    db: Session,
    pipeline_run_id: uuid.UUID | str,
    pass_names: list[str] | None = None,
) -> int:
    """Count distinct passes that have reached ANY terminal state for this run.

    Terminal states: COMPLETE, FAILED, SKIPPED.

    If ``pass_names`` is provided, only rows whose pass_name is in that list
    are counted.  Used by the fan-in coordinator to detect:
      - Whether every entity pass has resolved (so system_links can be dispatched)
      - Whether system_links itself has resolved (so merge can be dispatched)
    """
    stmt = select(func.count()).select_from(PipelinePassOutput).where(
        PipelinePassOutput.pipeline_run_id == pipeline_run_id,
        PipelinePassOutput.execution_status.in_(_TERMINAL_STATUSES),
    )
    if pass_names is not None:
        stmt = stmt.where(PipelinePassOutput.pass_name.in_(pass_names))
    return db.execute(stmt).scalar_one()
