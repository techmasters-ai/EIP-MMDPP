"""Add unique partial index on (source_id, file_hash) for duplicate detection.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-05
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove pre-existing duplicates (keep the oldest document per source+hash)
    op.execute("""
        DELETE FROM ingest.documents
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY source_id, file_hash
                           ORDER BY created_at ASC
                       ) AS rn
                FROM ingest.documents
                WHERE file_hash IS NOT NULL
            ) ranked
            WHERE rn > 1
        )
    """)

    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_source_hash
            ON ingest.documents (source_id, file_hash)
            WHERE file_hash IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ingest.uq_documents_source_hash")
