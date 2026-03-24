"""Add translated_text column to document_elements table.

Revision ID: 0010
Revises: 0009
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_elements",
        sa.Column("translated_text", sa.Text, nullable=True),
        schema="ingest",
    )


def downgrade() -> None:
    op.drop_column("document_elements", "translated_text", schema="ingest")
