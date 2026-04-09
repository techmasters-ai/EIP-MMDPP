"""Make text_chunks.artifact_id nullable for HybridChunker chunks.

HybridChunker produces chunks that span multiple elements or derive from
picture descriptions — these have no single parent artifact.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-09
"""
from alembic import op

# revision identifiers, used by Alembic
revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        "text_chunks",
        "artifact_id",
        nullable=True,
        schema="retrieval",
    )


def downgrade():
    op.alter_column(
        "text_chunks",
        "artifact_id",
        nullable=False,
        schema="retrieval",
    )
