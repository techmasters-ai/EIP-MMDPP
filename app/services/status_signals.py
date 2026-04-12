"""Status signals computation per spec §7.10.

The graph_queryable signal uses a CROSS-RUN query, not a latest-run
query. The composite (started_at, id) ordering ensures a deterministic
total order even when two PipelineRuns for the same document share a
started_at timestamp.

This module is consumed by the /documents/{id}/extraction-status
endpoint in app/api/v1/sources.py. It reads from the database via a
sync Session — the endpoint bridges to async via asyncio.to_thread.
"""
from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models.ingest import (
    DocumentGraphExtraction,
    PipelineRun,
    StageRun,
)
from app.services.ontology_bundles import StatusSignals


def compute_status_signals(document_id: str, session: Session) -> StatusSignals:
    """Spec §7.10 authoritative computation rule."""
    doc_uuid = uuid.UUID(document_id)

    snapshot = (
        session.query(DocumentGraphExtraction)
        .filter_by(document_id=doc_uuid)
        .first()
    )

    latest_run = (
        session.query(PipelineRun)
        .filter_by(document_id=doc_uuid)
        .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
        .first()
    )

    is_stale = False
    if snapshot is not None:
        is_stale = (
            latest_run is None
            or latest_run.id != snapshot.pipeline_run_id
            or latest_run.status != "COMPLETE"
        )

    if snapshot is None:
        graph_queryable = False
    else:
        snapshot_run = (
            session.query(PipelineRun)
            .filter_by(id=snapshot.pipeline_run_id)
            .first()
        )

        q = (
            session.query(StageRun)
            .join(PipelineRun, StageRun.pipeline_run_id == PipelineRun.id)
            .filter(
                PipelineRun.document_id == doc_uuid,
                StageRun.stage_name == "derive_ontology_graph",
                StageRun.pass_name.is_(None),
                StageRun.rollback_executed.is_(True),
            )
        )
        if snapshot_run is not None:
            q = q.filter(
                sa.tuple_(PipelineRun.started_at, PipelineRun.id)
                > sa.tuple_(snapshot_run.started_at, snapshot_run.id)
            )
        graph_invalidated = q.first() is not None
        graph_queryable = not graph_invalidated

    return StatusSignals(
        snapshot=snapshot,
        is_stale=is_stale,
        graph_queryable=graph_queryable,
    )
