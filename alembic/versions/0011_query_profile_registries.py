"""Add query profile registries for ontology-configurable graph search.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-03
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_profile_registries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ontology_name", sa.String(length=255), nullable=True),
        sa.Column("ontology_version", sa.String(length=100), nullable=True),
        sa.Column("ontology_definition", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("profiles", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["ingest.sources.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_query_profile_registries_name"),
        schema="governance",
    )
    op.create_index(
        "ix_query_profile_registries_is_active",
        "query_profile_registries",
        ["is_active"],
        unique=False,
        schema="governance",
    )
    op.create_index(
        "ix_query_profile_registries_source_id",
        "query_profile_registries",
        ["source_id"],
        unique=False,
        schema="governance",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_query_profile_registries_source_id",
        table_name="query_profile_registries",
        schema="governance",
    )
    op.drop_index(
        "ix_query_profile_registries_is_active",
        table_name="query_profile_registries",
        schema="governance",
    )
    op.drop_table("query_profile_registries", schema="governance")
