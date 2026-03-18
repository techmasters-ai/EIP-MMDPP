"""Drop custom GraphRAG Postgres tables (replaced by Microsoft GraphRAG Parquet/LanceDB).

Revision ID: 0008
Revises: 0007
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("graphrag_community_reports", schema="retrieval")
    op.drop_table("graphrag_communities", schema="retrieval")


def downgrade() -> None:
    op.create_table(
        "graphrag_communities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("community_id", sa.Text(), nullable=False, unique=True),
        sa.Column("level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("entity_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        schema="retrieval",
    )
    op.create_table(
        "graphrag_community_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "community_id",
            sa.Text(),
            sa.ForeignKey(
                "retrieval.graphrag_communities.community_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("report_text", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("rank", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        schema="retrieval",
    )
    op.create_unique_constraint(
        "uq_graphrag_reports_community_id",
        "graphrag_community_reports",
        ["community_id"],
        schema="retrieval",
    )
    op.create_index(
        "ix_graphrag_reports_community_id",
        "graphrag_community_reports",
        ["community_id"],
        schema="retrieval",
    )
