"""Drop the query-profile registry table + the dead ontology.* tables

Task 5 of the standalone-query-profiles plan. Tasks 0-3 moved every runtime
reader off ``governance.query_profile_registries`` (profiles now live in
``governance.query_profiles``, seeded by 0022) and off ``ontology.*``
(entity/relationship types + validation matrix are now served live from the
``air_defense_v3`` Pydantic SSoT via ``build_ontology_dict()``). Both tables
are write-only/dead at this point (verified via grep — no app/service/api
code reads them). This migration drops them, plus the now-empty ``ontology``
schema itself, and removes the ``scripts/seed_ontology.py`` writer
(deleted alongside this migration; see the same commit).

Downgrade recreates the four tables + the ``ontology`` schema (schema-only,
mirroring the original DDL in 0011/0001) but does NOT restore data — this
matches the existing style in this repo (0022's downgrade drops
``query_profiles`` without restoring the registry it was seeded from).
Anyone needing the actual pre-drop data must restore from the Postgres
backup taken before this migration ran.

Revision ID: 0023
Revises: 0022
Create Date: 2026-07-02
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # governance.query_profile_registries — superseded by governance.query_profiles
    # (Task 0 / migration 0022 already data-migrated the 4 profiles off it).
    op.drop_table("query_profile_registries", schema="governance")

    # ontology.* — write-only since the introspection-backed live ontology
    # (ontology_bundles/air_defense_v3/introspect.py) shipped; drop children
    # before parents (relationship_types FKs -> entity_types; entity_types
    # has a self-referential FK).
    op.drop_table("relationship_types", schema="ontology")
    op.drop_table("entity_types", schema="ontology")
    op.drop_table("versions", schema="ontology")

    # The ontology schema now has nothing left in it.
    op.execute("DROP SCHEMA IF EXISTS ontology CASCADE")


def downgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS ontology")

    op.create_table(
        "versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version_number", sa.String(50), nullable=False, unique=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ontology",
    )

    op.create_table(
        "entity_types",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "parent_type_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ontology.entity_types.id"),
            nullable=True,
        ),
        sa.Column("properties", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("is_abstract", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ontology",
    )

    op.create_table(
        "relationship_types",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "source_type_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ontology.entity_types.id"),
            nullable=True,
        ),
        sa.Column(
            "target_type_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ontology.entity_types.id"),
            nullable=True,
        ),
        sa.Column(
            "cardinality", sa.String(50), nullable=False, server_default="many_to_many"
        ),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="ontology",
    )

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
