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
    # No-op: pgvector columns are no longer created in 0001/0002.
    # Vectors are stored in ArcadeDB from the start.
    pass


def downgrade():
    pass  # No rollback — clean break from pgvector/Qdrant
