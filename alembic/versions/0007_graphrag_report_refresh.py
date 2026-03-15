"""Add generated_at to graphrag_community_reports and unique constraint on community_id.

Revision ID: 0007
Revises: 0006
Create Date: 2026-03-14
"""

from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"


def upgrade():
    # Add unique constraint on community_id so ON CONFLICT (community_id) works
    op.create_unique_constraint(
        "uq_graphrag_reports_community_id",
        "graphrag_community_reports",
        ["community_id"],
        schema="retrieval",
    )

    op.add_column(
        "graphrag_community_reports",
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        schema="retrieval",
    )


def downgrade():
    op.drop_column("graphrag_community_reports", "generated_at", schema="retrieval")
    op.drop_constraint(
        "uq_graphrag_reports_community_id",
        "graphrag_community_reports",
        schema="retrieval",
    )
