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
    ("AIR_DEFENSE_ARTILLERY_SYSTEM", RelationshipType.PROVIDES, "CAPABILITY"),
    ("ANTENNA", RelationshipType.PART_OF, "RADAR_SYSTEM"),
    ("ANTENNA", RelationshipType.RADIATES, "RF_EMISSION"),
    ("ASSEMBLY", RelationshipType.PART_OF, "EQUIPMENT_SYSTEM"),
    # TODO(docs-alignment): ASSERTION triples are dead — the AssertionEntity
    # class was dropped in B-1 (commit 2498a20). They're retained here for
    # parity with relationships.py + the YAML snapshot; once F-4 deletes the
    # parity tests, these rows can be stripped.
    ("ASSERTION", RelationshipType.ABOUT, "COMPONENT"),
    ("ASSERTION", RelationshipType.ABOUT, "EQUIPMENT_SYSTEM"),
    ("ASSERTION", RelationshipType.ABOUT, "MISSILE_SYSTEM"),
    ("ASSERTION", RelationshipType.ABOUT, "PLATFORM"),
    ("ASSERTION", RelationshipType.ABOUT, "RADAR_SYSTEM"),
    ("ASSERTION", RelationshipType.DERIVED_FROM, "DOCUMENT"),
    ("ASSERTION", RelationshipType.REVIEWED_BY, "ORGANIZATION"),
    ("ASSERTION", RelationshipType.SUPPORTED_BY, "DOCUMENT"),
    ("ASSERTION", RelationshipType.SUPPORTED_BY, "FIGURE"),
    ("ASSERTION", RelationshipType.SUPPORTED_BY, "SECTION"),
    ("ASSERTION", RelationshipType.SUPPORTED_BY, "TABLE"),
    ("COMPONENT", RelationshipType.MANUFACTURED_BY, "ORGANIZATION"),
    ("COMPONENT", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("COMPONENT", RelationshipType.PART_OF, "SUBSYSTEM"),
    ("COMPONENT", RelationshipType.SPECIFIED_BY, "STANDARD"),
    ("COMPONENT", RelationshipType.TESTED_IN, "TEST_EVENT"),
    ("DOCUMENT", RelationshipType.DERIVED_FROM, "DOCUMENT"),
    ("DOCUMENT", RelationshipType.HAS_FIGURE,  "FIGURE"),
    ("DOCUMENT", RelationshipType.HAS_IMAGE,   "IMAGE"),
    ("DOCUMENT", RelationshipType.HAS_SECTION, "SECTION"),
    ("DOCUMENT", RelationshipType.HAS_TABLE,   "TABLE"),
    ("DOCUMENT", RelationshipType.REVIEWED_BY, "ORGANIZATION"),
    ("DOCUMENT", RelationshipType.SUPERSEDES, "DOCUMENT"),
    ("ELECTRONIC_WARFARE_SYSTEM", RelationshipType.DETECTS, "RF_EMISSION"),
    ("ELECTRONIC_WARFARE_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("ELECTRONIC_WARFARE_SYSTEM", RelationshipType.OPERATES_IN_BAND, "FREQUENCY_BAND"),
    ("ELECTRONIC_WARFARE_SYSTEM", RelationshipType.PROVIDES, "CAPABILITY"),
    ("EQUIPMENT_SYSTEM", RelationshipType.ALIAS_OF, "EQUIPMENT_SYSTEM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.CONTAINS, "COMPONENT"),
    ("EQUIPMENT_SYSTEM", RelationshipType.HAS_COMPONENT, "COMPONENT"),
    ("EQUIPMENT_SYSTEM", RelationshipType.HAS_SUBSYSTEM, "SUBSYSTEM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.INSTANCE_OF, "EQUIPMENT_SYSTEM"),
    ("EQUIPMENT_SYSTEM", RelationshipType.MANUFACTURED_BY, "ORGANIZATION"),
    ("EQUIPMENT_SYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("EQUIPMENT_SYSTEM", RelationshipType.PROVIDES, "CAPABILITY"),
    ("EQUIPMENT_SYSTEM", RelationshipType.SPECIFIED_BY, "SPECIFICATION"),
    ("EQUIPMENT_SYSTEM", RelationshipType.SPECIFIED_BY, "STANDARD"),
    ("EQUIPMENT_SYSTEM", RelationshipType.TESTED_IN, "TEST_EVENT"),
    ("FAILURE_MODE", RelationshipType.AFFECTS, "COMPONENT"),
    ("FAILURE_MODE", RelationshipType.AFFECTS, "EQUIPMENT_SYSTEM"),
    ("FAILURE_MODE", RelationshipType.AFFECTS, "SUBSYSTEM"),
    ("FIGURE",       RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.CUES, "MISSILE_SYSTEM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.DESIGNATES, "PLATFORM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.GUIDES, "MISSILE_SYSTEM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("FIRE_CONTROL_SYSTEM", RelationshipType.TRACKS, "PLATFORM"),
    ("IF_AMPLIFIER", RelationshipType.PART_OF, "RECEIVER"),
    ("IMAGE",        RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.CONTAINS, "AIR_DEFENSE_ARTILLERY_SYSTEM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.CONTAINS, "MISSILE_SYSTEM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.CONTAINS, "RADAR_SYSTEM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.DEPLOYED_ON, "PLATFORM"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.PROVIDES, "CAPABILITY"),
    ("INTEGRATED_AIR_DEFENSE_SYSTEM", RelationshipType.SUPPORTS_ENGAGEMENT_OF, "PLATFORM"),
    ("LAUNCHER_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("LAUNCHER_SYSTEM", RelationshipType.LAUNCHES, "MISSILE_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.ALIAS_OF, "MISSILE_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.ASSOCIATED_WITH, "RADAR_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.DEFENDS, "PLATFORM"),
    ("MISSILE_SYSTEM", RelationshipType.ENGAGES, "PLATFORM"),
    ("MISSILE_SYSTEM", RelationshipType.HAS_GUIDANCE, "GUIDANCE_METHOD"),
    ("MISSILE_SYSTEM", RelationshipType.HAS_PERFORMANCE, "MISSILE_PERFORMANCE"),
    ("MISSILE_SYSTEM", RelationshipType.HAS_PROPULSION, "PROPULSION_STACK"),
    ("MISSILE_SYSTEM", RelationshipType.HAS_SEEKER, "SEEKER"),
    ("MISSILE_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("MISSILE_SYSTEM", RelationshipType.IS_A, "WEAPON_SYSTEM"),
    ("MISSILE_SYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("MISSILE_SYSTEM", RelationshipType.PROVIDES, "CAPABILITY"),
    ("MISSILE_SYSTEM", RelationshipType.SPECIFIED_BY, "SPECIFICATION"),
    ("MISSILE_SYSTEM", RelationshipType.TESTED_IN, "TEST_EVENT"),
    ("PLATFORM", RelationshipType.INSTANCE_OF, "PLATFORM"),
    ("PLATFORM", RelationshipType.MANUFACTURED_BY, "ORGANIZATION"),
    ("PLATFORM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("PLATFORM", RelationshipType.OPERATED_BY, "ORGANIZATION"),
    ("PROCEDURE", RelationshipType.OPERATED_BY, "ORGANIZATION"),
    ("PROPULSION_STACK", RelationshipType.CONTAINS, "PROPULSION_STAGE"),
    ("PROPULSION_STACK", RelationshipType.HAS_STAGE, "PROPULSION_STAGE"),
    ("RADAR_SYSTEM", RelationshipType.ALIAS_OF, "RADAR_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.ASSOCIATED_WITH, "AIR_DEFENSE_ARTILLERY_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.ASSOCIATED_WITH, "ELECTRONIC_WARFARE_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.ASSOCIATED_WITH, "MISSILE_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.CUES, "MISSILE_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.DESIGNATES, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.DETECTS, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.EMITS, "RF_EMISSION"),
    ("RADAR_SYSTEM", RelationshipType.HAS_ANTENNA, "ANTENNA"),
    ("RADAR_SYSTEM", RelationshipType.HAS_PERFORMANCE, "RADAR_PERFORMANCE"),
    ("RADAR_SYSTEM", RelationshipType.HAS_PROCESSING_CHAIN, "SIGNAL_PROCESSING_CHAIN"),
    ("RADAR_SYSTEM", RelationshipType.HAS_RECEIVER, "RECEIVER"),
    ("RADAR_SYSTEM", RelationshipType.HAS_SCAN, "SCAN_PATTERN"),
    ("RADAR_SYSTEM", RelationshipType.HAS_SIGNATURE, "RF_SIGNATURE"),
    ("RADAR_SYSTEM", RelationshipType.HAS_TIMELINE, "ENGAGEMENT_TIMELINE"),
    ("RADAR_SYSTEM", RelationshipType.HAS_TRANSMITTER, "TRANSMITTER"),
    ("RADAR_SYSTEM", RelationshipType.INSTALLED_ON, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.IS_A, "EQUIPMENT_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("RADAR_SYSTEM", RelationshipType.OPERATES_IN_BAND, "FREQUENCY_BAND"),
    ("RADAR_SYSTEM", RelationshipType.PROVIDES, "CAPABILITY"),
    ("RADAR_SYSTEM", RelationshipType.SPECIFIED_BY, "SPECIFICATION"),
    ("RADAR_SYSTEM", RelationshipType.SUPPORTS_ENGAGEMENT_OF, "MISSILE_SYSTEM"),
    ("RADAR_SYSTEM", RelationshipType.TESTED_IN, "TEST_EVENT"),
    ("RADAR_SYSTEM", RelationshipType.TRACKS, "PLATFORM"),
    ("RADAR_SYSTEM", RelationshipType.USES_WAVEFORM, "WAVEFORM"),
    ("RECEIVER", RelationshipType.PART_OF, "RADAR_SYSTEM"),
    ("RECEIVER", RelationshipType.RECEIVES, "RF_EMISSION"),
    ("RF_EMISSION", RelationshipType.HAS_SIGNATURE, "RF_SIGNATURE"),
    ("SECTION",  RelationshipType.CHILD_OF,    "SECTION"),
    ("SECTION",  RelationshipType.HAS_FIGURE,  "FIGURE"),
    ("SECTION",  RelationshipType.HAS_IMAGE,   "IMAGE"),
    ("SECTION",  RelationshipType.HAS_TABLE,   "TABLE"),
    ("SIGNAL_PROCESSING_CHAIN", RelationshipType.PART_OF, "RADAR_SYSTEM"),
    ("SIGNAL_PROCESSING_CHAIN", RelationshipType.PROCESSES, "RF_EMISSION"),
    ("SPECIFICATION", RelationshipType.SUPERSEDES, "SPECIFICATION"),
    ("STANDARD", RelationshipType.SUPERSEDES, "STANDARD"),
    ("SUBSYSTEM", RelationshipType.HAS_COMPONENT, "COMPONENT"),
    ("SUBSYSTEM", RelationshipType.MENTIONED_IN, "DOCUMENT"),
    ("SUBSYSTEM", RelationshipType.PART_OF, "EQUIPMENT_SYSTEM"),
    ("SUBSYSTEM", RelationshipType.PART_OF, "MISSILE_SYSTEM"),
    ("SUBSYSTEM", RelationshipType.PART_OF, "RADAR_SYSTEM"),
    ("SUBSYSTEM", RelationshipType.PROVIDES, "CAPABILITY"),
    ("TRANSMITTER", RelationshipType.PART_OF, "RADAR_SYSTEM"),
    ("WAVEFORM", RelationshipType.HAS_SIGNATURE, "RF_SIGNATURE"),
    ("WAVEFORM", RelationshipType.USES_MODULATION, "MODULATION"),
    ("WEAPON_SYSTEM", RelationshipType.CONTAINS, "SUBSYSTEM"),
    ("WEAPON_SYSTEM", RelationshipType.ENGAGES, "PLATFORM"),
    ("WEAPON_SYSTEM", RelationshipType.HAS_COMPONENT, "COMPONENT"),
    ("WEAPON_SYSTEM", RelationshipType.HAS_SUBSYSTEM, "SUBSYSTEM"),
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
