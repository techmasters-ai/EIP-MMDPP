"""starter_profiles_to_section_properties

Phase 2 of the flat-schema profile refactor. Updates the four well-known
starter profile entries in any persisted query_profile_registries row's
JSONB ``profiles`` column to the new ``kind="section_properties"`` /
``kind="dossier"`` shapes.

Structurally reversible: ``downgrade()`` writes back the prior
traversal-based shapes from the OLD_PROFILES table below. NOT
behaviorally compatible by itself — Phase 1 deleted the ontology types
those traversals relied on (CAPABILITY, RF_EMISSION, RADAR_PERFORMANCE,
etc.), so a profile-only rollback returns empty results until Phase 1
is also reverted.

Revision ID: 0018
Revises: 0017
Create Date: 2026-04-26
"""
import json

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


# Source of truth for the new shape — kept here for both up() and the
# rollback's reference list. Mirrors build_default_registry_template().
NEW_PROFILES = {
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


# Old shapes — preserved here so downgrade() restores parseable JSON.
# Behaviorally inert against post-Phase-1 ontology; documented above.
# Mirrors the pre-refactor build_default_registry_template().
OLD_PROFILES = {
    "system_dossier": {
        "id": "system_dossier",
        "label": "System Dossier",
        "description": (
            "Resolve a military system and return the configured exact graph sections "
            "for components, RF parameters, and performance characteristics."
        ),
        "kind": "dossier",
        "exposed": True,
        "root_entity_types": [],
        "target_entity_types": [],
        "traversals": [],
        "section_profile_ids": [
            "system_components",
            "system_rf_parameters",
            "system_performance",
        ],
        "placeholder_query": "e.g. AN/MPQ-65 or PAC-3 MSE",
    },
    "system_components": {
        "id": "system_components",
        "label": "System Components",
        "description": "Traverse subsystem/component structure and part hierarchy.",
        "kind": "section",
        "exposed": True,
        "root_entity_types": [],
        "target_entity_types": [],
        "traversals": [
            {"steps": [{"direction": "out", "rel_types": ["HAS_SUBSYSTEM", "HAS_COMPONENT", "CONTAINS"], "min_hops": 1, "max_hops": 3}]},
            {"steps": [{"direction": "in", "rel_types": ["PART_OF"], "min_hops": 1, "max_hops": 3}]},
        ],
        "placeholder_query": "e.g. AN/MPQ-65",
    },
    "system_rf_parameters": {
        "id": "system_rf_parameters",
        "label": "System RF Parameters",
        "description": "Find band, waveform, emitter, receiver, seeker, and RF specification nodes.",
        "kind": "section",
        "exposed": True,
        "root_entity_types": [],
        "target_entity_types": [],
        "traversals": [
            {"steps": [{"direction": "out", "rel_types": ["EMITS", "USES_WAVEFORM", "OPERATES_IN_BAND"], "min_hops": 1, "max_hops": 2}]},
        ],
        "placeholder_query": "e.g. AN/MPQ-65",
    },
    "system_performance": {
        "id": "system_performance",
        "label": "System Performance",
        "description": "Find performance, capability, engagement, guidance, and specification nodes.",
        "kind": "section",
        "exposed": True,
        "root_entity_types": [],
        "target_entity_types": [],
        "traversals": [
            {"steps": [{"direction": "out", "rel_types": ["HAS_PERFORMANCE", "PROVIDES", "HAS_TIMELINE"], "min_hops": 1, "max_hops": 2}]},
        ],
        "placeholder_query": "e.g. PAC-3 MSE",
    },
}

STARTER_IDS = set(NEW_PROFILES.keys())


def _rewrite_profiles(profiles: list, source_map: dict) -> tuple[list, bool]:
    """Replace any starter-profile entries with the source_map version,
    leaving non-starter profiles untouched. Returns (new_list, changed)."""
    out = []
    changed = False
    for p in profiles:
        pid = p.get("id") if isinstance(p, dict) else None
        if pid in source_map:
            out.append(source_map[pid])
            if p != source_map[pid]:
                changed = True
        else:
            out.append(p)
    return out, changed


def upgrade():
    """For every governance.query_profile_registries row whose profiles
    JSONB column contains one of the four starter IDs, overwrite that
    profile with the NEW_PROFILES shape."""
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT id, profiles FROM governance.query_profile_registries"
    )).fetchall()

    for row_id, profiles_json in rows:
        if profiles_json is None:
            continue
        profiles = (
            profiles_json
            if isinstance(profiles_json, list)
            else json.loads(profiles_json)
        )
        new_profiles, changed = _rewrite_profiles(profiles, NEW_PROFILES)
        if changed:
            conn.execute(
                sa.text(
                    "UPDATE governance.query_profile_registries "
                    "SET profiles = :profiles WHERE id = :id"
                ),
                {"profiles": json.dumps(new_profiles), "id": row_id},
            )


def downgrade():
    """Restore the prior traversal-based shapes. Behaviorally inert
    against the post-Phase-1 ontology; full rollback requires reverting
    Phase 1 ontology changes too."""
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT id, profiles FROM governance.query_profile_registries"
    )).fetchall()

    for row_id, profiles_json in rows:
        if profiles_json is None:
            continue
        profiles = (
            profiles_json
            if isinstance(profiles_json, list)
            else json.loads(profiles_json)
        )
        old_profiles, changed = _rewrite_profiles(profiles, OLD_PROFILES)
        if changed:
            conn.execute(
                sa.text(
                    "UPDATE governance.query_profile_registries "
                    "SET profiles = :profiles WHERE id = :id"
                ),
                {"profiles": json.dumps(old_profiles), "id": row_id},
            )
