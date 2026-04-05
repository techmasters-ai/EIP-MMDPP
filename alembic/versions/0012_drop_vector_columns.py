"""Drop pgvector embedding and qdrant_point_id columns.

Vectors are now stored in ArcadeDB, not PostgreSQL.

Revision ID: 0012
Revises: 0011_query_profile_registries
Create Date: 2026-04-04
"""
from alembic import op

# revision identifiers, used by Alembic
revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("text_chunks", "embedding", schema="retrieval")
    op.drop_column("text_chunks", "qdrant_point_id", schema="retrieval")
    op.drop_column("image_chunks", "embedding", schema="retrieval")
    op.drop_column("image_chunks", "qdrant_point_id", schema="retrieval")
    op.drop_column("trusted_data_submissions", "qdrant_point_id", schema="governance")
    op.execute("DROP EXTENSION IF EXISTS vector")


def downgrade():
    pass  # No rollback — clean break from pgvector/Qdrant
