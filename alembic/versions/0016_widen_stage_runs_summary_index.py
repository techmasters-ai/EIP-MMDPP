"""widen stage_runs summary unique index to all pass_name IS NULL rows

Revision ID: 0016
Revises: 0015
Create Date: 2026-04-12

Fixes bug where `_update_stage_run` in app/workers/pipeline.py references the
dropped constraint `stage_runs_pipeline_run_id_stage_name_attempt_key` in its
ON CONFLICT clause. Migration 0015 dropped that constraint and added a
narrower `uq_stage_runs_summary_row` index restricted to
`stage_name = 'derive_ontology_graph'`, leaving all other stage-level rows
(pass_name IS NULL for detect_and_translate, derive_document_metadata, etc.)
without any unique index. Every INSERT failed silently in the caller's
try/except, so stage-level monitoring data was never persisted.

This migration widens `uq_stage_runs_summary_row` to cover every
`pass_name IS NULL` row. The per-pass index `uq_stage_runs_run_pass_attempt`
is unchanged.
"""
from alembic import op
import sqlalchemy as sa


revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(
        "uq_stage_runs_summary_row",
        table_name="stage_runs",
        schema="ingest",
    )
    op.create_index(
        "uq_stage_runs_summary_row",
        "stage_runs",
        ["pipeline_run_id", "stage_name", "attempt"],
        unique=True,
        postgresql_where=sa.text("pass_name IS NULL"),
        schema="ingest",
    )


def downgrade() -> None:
    op.drop_index(
        "uq_stage_runs_summary_row",
        table_name="stage_runs",
        schema="ingest",
    )
    op.create_index(
        "uq_stage_runs_summary_row",
        "stage_runs",
        ["pipeline_run_id", "stage_name", "attempt"],
        unique=True,
        postgresql_where=sa.text(
            "pass_name IS NULL AND stage_name = 'derive_ontology_graph'"
        ),
        schema="ingest",
    )
