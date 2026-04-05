"""Add community_runs table for tracking community detection runs.

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic
revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "community_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("trigger", sa.String(20), nullable=False),
        sa.Column("total_communities", sa.Integer),
        sa.Column("reports_generated", sa.Integer),
        sa.Column("reports_reused", sa.Integer),
        sa.Column("detection_duration_ms", sa.Integer),
        sa.Column("report_duration_ms", sa.Integer),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        schema="retrieval",
    )


def downgrade():
    op.drop_table("community_runs", schema="retrieval")
