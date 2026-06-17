"""VALIDATION_MATRIX and SCORING_WEIGHTS for the air_defense_v3 ontology.

Phase 2 Tasks 12 + 13: Pydantic-native mirrors of the two YAML tables
that constrain and rank relationships. Parity with
``ontology_bundles/air_defense_v3/ontology.yaml`` is pinned by
``tests/unit/test_validation_matrix_parity.py``.

VALIDATION_MATRIX is the authoritative source for "which (source, rel,
target) triples are valid" once YAML is deleted in Phase 6.
SCORING_WEIGHTS keeps the relationship-prior weights used during
candidate-relationship scoring.

YAML bug flagged: ontology.yaml validation_matrix lists
``(RADAR_SYSTEM, INSTALLED_ON, PLATFORM)`` twice. A frozenset naturally
collapses the duplicate to 136 unique triples.
"""
from __future__ import annotations

from .relationships import RelationshipType

# ---------------------------------------------------------------------------
# VALIDATION_MATRIX — every valid (source, rel, target) triple.
# 136 unique triples (YAML has one duplicate; frozenset deduplicates).
# ---------------------------------------------------------------------------

VALIDATION_MATRIX: frozenset[tuple[str, RelationshipType, str]] = frozenset({
    ("AIR_DEFENSE_ARTILLERY_SYSTEM", RelationshipType.ASSOCIATED_WITH, "RADAR_SYSTEM"),
    ("AIR_DEFENSE_ARTILLERY_SYSTEM", RelationshipType.ENGAGES, "PLATFORM"),
    ("AIR_DEFENSE_ARTILLERY_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("AIR_DEFENSE_ARTILLERY_SYSTEM", RelationshipType.IS_A, "WEAPON_SYSTEM"),
    ("COMPONENT", RelationshipType.MANUFACTURED_BY, "ORGANIZATION"),
    ("COMPONENT", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("COMPONENT", RelationshipType.PART_OF, "SUBSYSTEM"),
    ("DOCUMENT", RelationshipType.DERIVED_FROM, "DOCUMENT"),
    ("DOCUMENT", RelationshipType.HAS_FIGURE,  "FIGURE"),
    ("DOCUMENT", RelationshipType.HAS_IMAGE,   "IMAGE"),
    ("DOCUMENT", RelationshipType.HAS_SECTION, "SECTION"),
    ("DOCUMENT", RelationshipType.HAS_TABLE,   "TABLE"),
    ("DOCUMENT", RelationshipType.REVIEWED_BY, "ORGANIZATION"),
    ("DOCUMENT", RelationshipType.SUPERSEDES, "DOCUMENT"),
    ("ELECTRONIC_WARFARE_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.ALIAS_OF, "EQUIPMENT_SYSTEM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.CONTAINS, "COMPONENT"),
    ("EQUIPMENT_SYSTEM", RelationshipType.HAS_COMPONENT, "COMPONENT"),
    ("EQUIPMENT_SYSTEM", RelationshipType.HAS_SUBSYSTEM, "SUBSYSTEM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.INSTANCE_OF, "EQUIPMENT_SYSTEM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.MANUFACTURED_BY, "ORGANIZATION"),
    ("EQUIPMENT_SYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("FIGURE",       RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.CUES, "MISSILE_SYSTEM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.DESIGNATES, "PLATFORM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.GUIDES, "MISSILE_SYSTEM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.TRACKS, "PLATFORM"),
    ("IMAGE",        RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.CONTAINS, "AIR_DEFENSE_ARTILLERY_SYSTEM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.CONTAINS, "MISSILE_SYSTEM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.CONTAINS, "RADAR_SYSTEM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.DEPLOYED_ON, "PLATFORM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.SUPPORTS_ENGAGEMENT_OF, "PLATFORM"),
    ("LAUNCHER_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("LAUNCHER_SYSTEM", RelationshipType.LAUNCHES, "MISSILE_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.ALIAS_OF, "MISSILE_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.ASSOCIATED_WITH, "RADAR_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.DEFENDS, "PLATFORM"),
    ("MISSILE_SYSTEM", RelationshipType.ENGAGES, "PLATFORM"),
    ("MISSILE_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("MISSILE_SYSTEM", RelationshipType.IS_A, "WEAPON_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("PLATFORM", RelationshipType.INSTANCE_OF, "PLATFORM"),
    ("PLATFORM", RelationshipType.MANUFACTURED_BY, "ORGANIZATION"),
    ("PLATFORM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("PLATFORM", RelationshipType.OPERATED_BY, "ORGANIZATION"),
    ("RADAR_SYSTEM", RelationshipType.ALIAS_OF, "RADAR_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.ASSOCIATED_WITH, "AIR_DEFENSE_ARTILLERY_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.ASSOCIATED_WITH, "ELECTRONIC_WARFARE_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.ASSOCIATED_WITH, "MISSILE_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.CUES, "MISSILE_SYSTEM"),
    # Search radar hands off to fire-control / guidance radar. Documented
    # kill-chain pattern, e.g. Spoon Rest CUES Fan Song before Fan Song
    # guides the SA-2. system_links.py prompt examples rely on this
    # triple; without it the merge layer rejects the edge with
    # invalid_triple and the kill-chain loses its first hop.
    ("RADAR_SYSTEM", RelationshipType.CUES, "RADAR_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.DESIGNATES, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.DETECTS, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.IS_A, "EQUIPMENT_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("RADAR_SYSTEM", RelationshipType.SUPPORTS_ENGAGEMENT_OF, "MISSILE_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.TRACKS, "PLATFORM"),
    ("SECTION",  RelationshipType.CHILD_OF,    "SECTION"),
    ("SECTION",  RelationshipType.HAS_FIGURE,  "FIGURE"),
    ("SECTION",  RelationshipType.HAS_IMAGE,   "IMAGE"),
    ("SECTION",  RelationshipType.HAS_TABLE,   "TABLE"),
    ("SUBSYSTEM", RelationshipType.HAS_COMPONENT, "COMPONENT"),
    ("SUBSYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("SUBSYSTEM", RelationshipType.PART_OF, "EQUIPMENT_SYSTEM"),
    ("SUBSYSTEM", RelationshipType.PART_OF, "MISSILE_SYSTEM"),
    ("SUBSYSTEM", RelationshipType.PART_OF, "RADAR_SYSTEM"),
    ("WEAPON_SYSTEM", RelationshipType.CONTAINS, "SUBSYSTEM"),
    ("WEAPON_SYSTEM", RelationshipType.ENGAGES, "PLATFORM"),
    ("WEAPON_SYSTEM", RelationshipType.HAS_COMPONENT, "COMPONENT"),
    ("WEAPON_SYSTEM", RelationshipType.HAS_SUBSYSTEM, "SUBSYSTEM"),
    # VARIANT_OF — a child designation/model/configuration rolls up to a
    # parent family entity of the SAME type (e.g. an SA-2 variant → SA-2
    # family). system_links emits this cross-pass; the merge layer's exact
    # (source, rel, target) triple check requires the same-type self-edge per
    # system family. Mirrors the ALIAS_OF / INSTANCE_OF self-edge pattern.
    ("AIR_DEFENSE_ARTILLERY_SYSTEM", RelationshipType.VARIANT_OF, "AIR_DEFENSE_ARTILLERY_SYSTEM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.VARIANT_OF, "EQUIPMENT_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.VARIANT_OF, "MISSILE_SYSTEM"),
    ("PLATFORM", RelationshipType.VARIANT_OF, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.VARIANT_OF, "RADAR_SYSTEM"),
    ("WEAPON_SYSTEM", RelationshipType.VARIANT_OF, "WEAPON_SYSTEM"),
})


# ---------------------------------------------------------------------------
# SCORING_WEIGHTS — relationship-priors used during candidate scoring.
# Keyed by relationship name (string), NOT RelationshipType, because YAML
# includes a 'default' key that doesn't map to any specific enum member.
# ---------------------------------------------------------------------------

SCORING_WEIGHTS: dict[str, float] = {
    "IS_VARIANT_OF": 0.95,
    "USES_COMPONENT": 0.92,
    "CONTAINS": 0.9,
    "PART_OF": 0.9,
    "INTERFACES_WITH": 0.85,
    "OPERATES_ON": 0.85,
    "RELATED_TO": 0.75,
    "default": 0.7,
}
