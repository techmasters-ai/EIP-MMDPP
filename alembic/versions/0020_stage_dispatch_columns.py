"""add dispatch ledger columns to stage_runs

Revision ID: 0020
Revises: 0019
Create Date: 2026-05-10

Adds 6 columns and 1 partial index to ingest.stage_runs, enabling the durable
dispatch-ledger model. Ledger summary rows always have attempt=1; the new
dispatch_attempt column tracks retries. See:
docs/superpowers/specs/2026-05-10-pipeline-stage-dispatch-ledger-design.md
"""
from alembic import op
import sqlalchemy as sa

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "stage_runs",
        sa.Column("queue_name", sa.String(64), nullable=True),
        schema="ingest",
    )
    op.add_column(
        "stage_runs",
        sa.Column("task_name", sa.String(255), nullable=True),
        schema="ingest",
    )
    op.add_column(
        "stage_runs",
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        schema="ingest",
    )
    op.add_column(
        "stage_runs",
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("NOW()"),
        ),
        schema="ingest",
    )
    op.add_column(
        "stage_runs",
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        schema="ingest",
    )
    op.add_column(
        "stage_runs",
        sa.Column(
            "dispatch_attempt",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        schema="ingest",
    )

    op.create_index(
        "ix_stage_runs_dispatcher_claim",
        "stage_runs",
        ["available_at"],
        unique=False,
        postgresql_where=sa.text(
            "status = 'PENDING' AND pass_name IS NULL AND task_name IS NOT NULL"
        ),
        schema="ingest",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_stage_runs_dispatcher_claim",
        table_name="stage_runs",
        schema="ingest",
    )
    op.drop_column("stage_runs", "dispatch_attempt", schema="ingest")
    op.drop_column("stage_runs", "dispatched_at", schema="ingest")
    op.drop_column("stage_runs", "available_at", schema="ingest")
    op.drop_column("stage_runs", "celery_task_id", schema="ingest")
    op.drop_column("stage_runs", "task_name", schema="ingest")
    op.drop_column("stage_runs", "queue_name", schema="ingest")
