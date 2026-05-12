"""add chunk_metadata JSONB to retrieval.text_chunks for table-aware chunking

Revision ID: 0021
Revises: 0020
Create Date: 2026-05-11

Adds an optional JSONB column carrying chunk-kind / table-ref / cell-refs
provenance for chunks produced by the table normalization layer. Adds a
partial expression index on (chunk_metadata->>'chunk_kind') for fast
filtering by chunk_kind in future retrieval queries.

Spec: docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md §11.1.

WARNING: downgrade drops the chunk_metadata column. Run ./manage.sh
--blow-away before downgrading to avoid silent data loss in retrieval
responses that rely on the table_chunk block.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "text_chunks",
        sa.Column("chunk_metadata", postgresql.JSONB(), nullable=True),
        schema="retrieval",
    )
    op.create_index(
        "ix_text_chunks_chunk_kind",
        "text_chunks",
        [sa.text("(chunk_metadata->>'chunk_kind')")],
        schema="retrieval",
        postgresql_where=sa.text("chunk_metadata IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_text_chunks_chunk_kind",
        table_name="text_chunks",
        schema="retrieval",
    )
    op.drop_column(
        "text_chunks", "chunk_metadata", schema="retrieval",
    )
