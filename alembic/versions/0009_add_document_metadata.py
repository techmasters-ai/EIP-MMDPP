"""Add document_metadata JSONB column to documents table.

Revision ID: 0009
Revises: 0008
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("document_metadata", JSONB, nullable=True),
        schema="ingest",
    )


def downgrade() -> None:
    op.drop_column("documents", "document_metadata", schema="ingest")
