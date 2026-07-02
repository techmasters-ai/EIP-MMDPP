"""query_profiles table — data-migrate the 4 profiles off the registry

Task 0 of the standalone-query-profiles plan: removes the registry-layer
indirection by making query profiles first-class rows. Creates
``governance.query_profiles`` and deterministically seeds it from the
active/latest ``governance.query_profile_registries`` row's embedded
``profiles`` JSONB list (falling back to the canonical 4 starter profiles
— copied verbatim from 0018's ``NEW_PROFILES`` — if that row is missing or
doesn't contain exactly 4 profiles).

The ``QueryProfileRegistry`` model/table are left in place; a later task
removes them once callers are migrated onto ``QueryProfile``.

Revision ID: 0022
Revises: 0021
Create Date: 2026-07-01
"""
import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


# Fallback source of truth, copied verbatim from
# alembic/versions/0018_starter_profiles_to_section_properties.py::NEW_PROFILES.
# Used only if the live registry row can't supply exactly 4 profiles.
_CANONICAL_PROFILES = {
    "system_rf_parameters": {
        "id": "system_rf_parameters",
        "label": "System RF Parameters",
        "description": (
            "Frequency, antenna, scan, modulation, and other RF descriptors "
            "of the resolved system."
        ),
        "kind": "section_properties",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": ["rf_parameters"],
        "include_associated_systems": False,
        "section_profile_ids": [],
        "placeholder_query": "e.g. Fan Song",
    },
    "system_components": {
        "id": "system_components",
        "label": "System Components",
        "description": (
            "Antenna, propulsion, seeker, ejector, body, and other physical "
            "components of the resolved system."
        ),
        "kind": "section_properties",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": ["components"],
        "include_associated_systems": True,
        "section_profile_ids": [],
        "placeholder_query": "e.g. SA-2",
    },
    "system_performance": {
        "id": "system_performance",
        "label": "System Performance",
        "description": (
            "Engagement envelope, kinematics, transmit power, and propulsion "
            "timing for the resolved system."
        ),
        "kind": "section_properties",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": ["performance"],
        "include_associated_systems": False,
        "section_profile_ids": [],
        "placeholder_query": "e.g. SA-2",
    },
    "system_dossier": {
        "id": "system_dossier",
        "label": "System Dossier",
        "description": (
            "Composite report of RF parameters, components, and performance "
            "for the resolved system."
        ),
        "kind": "dossier",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": [],
        "include_associated_systems": False,
        "section_profile_ids": [
            "system_rf_parameters",
            "system_components",
            "system_performance",
        ],
        "placeholder_query": "e.g. SA-2",
    },
}

# Keys of a profile dict that get promoted into the flat `definition` JSONB
# column rather than their own `query_profiles` column.
_DEFINITION_KEYS = (
    "target_entity_types",
    "traversals",
    "section_profile_ids",
    "profile_sections",
    "profile_subgroup",
    "include_associated_systems",
)


def upgrade() -> None:
    op.create_table(
        "query_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("profile_key", sa.String(100), nullable=False),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("root_entity_types", JSONB(), nullable=True),
        sa.Column("definition", JSONB(), nullable=True),
        sa.Column(
            "source_id",
            UUID(as_uuid=True),
            sa.ForeignKey("ingest.sources.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_by", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("profile_key", name="uq_query_profiles_profile_key"),
        schema="governance",
    )
    op.create_index(
        "ix_query_profiles_source_id",
        "query_profiles",
        ["source_id"],
        schema="governance",
    )
    op.create_index(
        "ix_query_profiles_enabled",
        "query_profiles",
        ["enabled"],
        schema="governance",
    )

    conn = op.get_bind()

    row = conn.execute(
        sa.text(
            "SELECT profiles FROM governance.query_profile_registries "
            "WHERE is_active IS TRUE ORDER BY updated_at DESC LIMIT 1"
        )
    ).fetchone()
    if row is None:
        row = conn.execute(
            sa.text(
                "SELECT profiles FROM governance.query_profile_registries "
                "ORDER BY updated_at DESC LIMIT 1"
            )
        ).fetchone()

    profiles = row[0] if row else []
    if isinstance(profiles, str):
        profiles = json.loads(profiles)

    if not profiles or len(profiles) != 4:
        profiles = list(_CANONICAL_PROFILES.values())

    for p in profiles:
        definition = {k: p[k] for k in _DEFINITION_KEYS if k in p}
        conn.execute(
            sa.text(
                """
                INSERT INTO governance.query_profiles
                    (id, profile_key, label, description, kind,
                     root_entity_types, definition, source_id, enabled,
                     created_by, created_at, updated_at)
                VALUES
                    (gen_random_uuid(), :profile_key, :label, :description, :kind,
                     CAST(:root_entity_types AS jsonb), CAST(:definition AS jsonb),
                     NULL, TRUE, NULL, now(), now())
                """
            ),
            {
                "profile_key": p["id"],
                "label": p["label"],
                "description": p.get("description"),
                "kind": p["kind"],
                "root_entity_types": json.dumps(p.get("root_entity_types", [])),
                "definition": json.dumps(definition),
            },
        )

    count = conn.execute(
        sa.text("SELECT count(*) FROM governance.query_profiles")
    ).scalar()
    if count != 4:
        raise RuntimeError(
            f"Expected exactly 4 rows in governance.query_profiles after "
            f"data migration, found {count}."
        )


def downgrade() -> None:
    op.drop_table("query_profiles", schema="governance")
