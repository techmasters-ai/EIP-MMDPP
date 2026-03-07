"""Rename memory_proposals to trusted_data_submissions and add indexing lifecycle columns.

Revision ID: 0006
Revises: 0005
Create Date: 2026-03-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename table
    op.rename_table(
        "memory_proposals", "trusted_data_submissions", schema="governance"
    )

    # Add indexing lifecycle columns
    op.add_column(
        "trusted_data_submissions",
        sa.Column("index_status", sa.String(50), nullable=True),
        schema="governance",
    )
    op.add_column(
        "trusted_data_submissions",
        sa.Column("index_error", sa.Text, nullable=True),
        schema="governance",
    )
    op.add_column(
        "trusted_data_submissions",
        sa.Column(
            "qdrant_point_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        schema="governance",
    )
    op.add_column(
        "trusted_data_submissions",
        sa.Column("embedding_model", sa.String(100), nullable=True),
        schema="governance",
    )
    op.add_column(
        "trusted_data_submissions",
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        schema="governance",
    )

    # Rename index
    op.execute(
        "ALTER INDEX IF EXISTS governance.ix_memory_proposals_status "
        "RENAME TO ix_trusted_data_submissions_status"
    )

    # Add index on index_status
    op.create_index(
        "ix_trusted_data_submissions_index_status",
        "trusted_data_submissions",
        ["index_status"],
        schema="governance",
    )

    # Backfill: existing APPROVED rows → APPROVED_PENDING_INDEX
    op.execute(
        "UPDATE governance.trusted_data_submissions "
        "SET status = 'APPROVED_PENDING_INDEX' "
        "WHERE status = 'APPROVED'"
    )


def downgrade() -> None:
    # Revert APPROVED_PENDING_INDEX back to APPROVED
    op.execute(
        "UPDATE governance.trusted_data_submissions "
        "SET status = 'APPROVED' "
        "WHERE status IN ('APPROVED_PENDING_INDEX', 'APPROVED_INDEXED', 'INDEX_FAILED')"
    )

    # Drop index_status index
    op.drop_index(
        "ix_trusted_data_submissions_index_status",
        table_name="trusted_data_submissions",
        schema="governance",
    )

    # Rename index back
    op.execute(
        "ALTER INDEX IF EXISTS governance.ix_trusted_data_submissions_status "
        "RENAME TO ix_memory_proposals_status"
    )

    # Drop new columns
    op.drop_column("trusted_data_submissions", "embedded_at", schema="governance")
    op.drop_column("trusted_data_submissions", "embedding_model", schema="governance")
    op.drop_column("trusted_data_submissions", "qdrant_point_id", schema="governance")
    op.drop_column("trusted_data_submissions", "index_error", schema="governance")
    op.drop_column("trusted_data_submissions", "index_status", schema="governance")

    # Rename table back
    op.rename_table(
        "trusted_data_submissions", "memory_proposals", schema="governance"
    )
