"""Add qdrant_point_id to chunks and GraphRAG tables.

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- Add qdrant_point_id to text_chunks ---
    op.add_column(
        "text_chunks",
        sa.Column("qdrant_point_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema="retrieval",
    )
    op.create_index(
        "ix_text_chunks_qdrant_point_id",
        "text_chunks",
        ["qdrant_point_id"],
        schema="retrieval",
    )

    # --- Add qdrant_point_id to image_chunks ---
    op.add_column(
        "image_chunks",
        sa.Column("qdrant_point_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema="retrieval",
    )
    op.create_index(
        "ix_image_chunks_qdrant_point_id",
        "image_chunks",
        ["qdrant_point_id"],
        schema="retrieval",
    )

    # --- GraphRAG communities ---
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

    # --- GraphRAG community reports ---
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
        schema="retrieval",
    )

    op.create_index(
        "ix_graphrag_reports_community_id",
        "graphrag_community_reports",
        ["community_id"],
        schema="retrieval",
    )


def downgrade() -> None:
    op.drop_table("graphrag_community_reports", schema="retrieval")
    op.drop_table("graphrag_communities", schema="retrieval")

    op.drop_index(
        "ix_image_chunks_qdrant_point_id",
        table_name="image_chunks",
        schema="retrieval",
    )
    op.drop_column("image_chunks", "qdrant_point_id", schema="retrieval")

    op.drop_index(
        "ix_text_chunks_qdrant_point_id",
        table_name="text_chunks",
        schema="retrieval",
    )
    op.drop_column("text_chunks", "qdrant_point_id", schema="retrieval")
