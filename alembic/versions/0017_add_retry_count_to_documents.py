"""add retry_count column to ingest.documents

Revision ID: 0017
Revises: 0016
Create Date: 2026-04-24

Sweeper auto-restart feature: when periodic_stale_run_sweep detects a
stage_run past STALE_STAGE_RUN_THRESHOLD_SECONDS, it marks the run FAILED
and re-dispatches start_ingest_pipeline for the document. retry_count
tracks how many times that has happened per doc; when it exceeds
settings.max_doc_retry_count the sweeper stops retrying and marks the
document permanently FAILED.

See docs/superpowers/plans/2026-04-23-reliable-ingest-retry.md Task 3.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "retry_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        schema="ingest",
    )
    # Drop server_default so future inserts rely on the ORM default.
    op.alter_column(
        "documents",
        "retry_count",
        server_default=None,
        schema="ingest",
    )


def downgrade() -> None:
    op.drop_column("documents", "retry_count", schema="ingest")
