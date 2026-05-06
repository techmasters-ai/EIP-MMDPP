"""add pipeline_pass_outputs and dispatched_phases

Adds the ``pipeline_pass_outputs`` table (terminal pass-resolution records for
the per-pass Celery fan-in architecture) and the ``dispatched_phases`` JSONB
state-machine column on ``pipeline_runs``.

Design notes:
- The unique constraint on ``pipeline_pass_outputs`` covers only
  ``(pipeline_run_id, pass_name)``, NOT ``attempt``.  The table is
  *terminal-only*: at most one resolved row per (run, pass).  Intermediate
  retry attempts accumulate in ``stage_runs`` instead.
- ``dispatched_phases`` on ``pipeline_runs`` is a per-pass phase-tracking
  dict written by the fan-in coordinator; the server_default of ``'{}'::jsonb``
  back-fills any existing rows without a data migration.

Revision ID: 0019
Revises: 0018
Create Date: 2026-05-06
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade():
    # ------------------------------------------------------------------
    # 1. Add dispatched_phases to pipeline_runs
    # ------------------------------------------------------------------
    op.add_column(
        "pipeline_runs",
        sa.Column(
            "dispatched_phases",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment=(
                "Per-pass phase-tracking state machine. Maps pass_name → phase dict. "
                "Written by the per-pass Celery fan-in coordinator."
            ),
        ),
        schema="ingest",
    )

    # ------------------------------------------------------------------
    # 2. Create pipeline_pass_outputs table
    # ------------------------------------------------------------------
    op.create_table(
        "pipeline_pass_outputs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pipeline_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("pass_name", sa.String(length=64), nullable=False),
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
            comment="Which attempt resolved this pass (audit only; full history in stage_runs).",
        ),
        sa.Column(
            "execution_status",
            sa.String(length=16),
            nullable=False,
            comment="COMPLETE | FAILED | SKIPPED",
        ),
        sa.Column(
            "skip_reason",
            sa.String(length=64),
            nullable=True,
            comment="e.g. NO_UPSTREAM_ENDPOINTS",
        ),
        sa.Column(
            "yield_status",
            sa.String(length=16),
            nullable=True,
            comment="HIT | EMPTY | DEGRADED | BRIDGES_ONLY",
        ),
        sa.Column(
            "extract_pass_response_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="Literal /extract-pass response payload.",
        ),
        sa.Column("primary_entities_extracted", sa.Integer(), nullable=True),
        sa.Column("bridge_entities_extracted", sa.Integer(), nullable=True),
        sa.Column("relationships_extracted", sa.Integer(), nullable=True),
        sa.Column("relationships_rejected", sa.Integer(), nullable=True),
        sa.Column(
            "diagnostics_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "field_provenance_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # PK
        sa.PrimaryKeyConstraint("id"),
        # FKs
        sa.ForeignKeyConstraint(
            ["pipeline_run_id"],
            ["ingest.pipeline_runs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["stage_run_id"],
            ["ingest.stage_runs.id"],
            ondelete="SET NULL",
        ),
        # Unique: terminal-only — one resolved row per (run, pass)
        sa.UniqueConstraint(
            "pipeline_run_id",
            "pass_name",
            name="uq_pipeline_pass_outputs_run_pass",
        ),
        schema="ingest",
    )

    # Index on pipeline_run_id for FK lookups
    op.create_index(
        "ix_pipeline_pass_outputs_pipeline_run_id",
        "pipeline_pass_outputs",
        ["pipeline_run_id"],
        schema="ingest",
    )


def downgrade():
    # Reverse order: drop table first (FK dep on pipeline_runs), then column
    op.drop_index(
        "ix_pipeline_pass_outputs_pipeline_run_id",
        table_name="pipeline_pass_outputs",
        schema="ingest",
    )
    op.drop_table("pipeline_pass_outputs", schema="ingest")
    op.drop_column("pipeline_runs", "dispatched_phases", schema="ingest")
