"""Relationship-type registry for the air_defense_v3 ontology bundle.

Phase 2 Task 10/11: Pydantic-native relationship metadata mirroring
``ontology.yaml`` 1:1. Consumers that want to enumerate valid relationship
types use :data:`RelationshipType`; consumers that need the full metadata
(label, description, cardinality, directionality) use
:data:`RELATIONSHIP_METADATA`.

Parity with YAML is pinned by dedicated tests in
``tests/unit/test_relationships_parity.py``.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RelationshipType(str, Enum):
    """Every relationship type declared in ``ontology.yaml``.

    ``str`` subclass so Pydantic coerces string inputs via value-match.
    Members are alphabetical for stable diff review. 50 members total.
    """

    ABOUT = "ABOUT"
    AFFECTS = "AFFECTS"
    ALIAS_OF = "ALIAS_OF"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    CONTAINS = "CONTAINS"
    CUES = "CUES"
    DEFENDS = "DEFENDS"
    DEPLOYED_ON = "DEPLOYED_ON"
    DERIVED_FROM = "DERIVED_FROM"
    DESIGNATES = "DESIGNATES"
    DETECTS = "DETECTS"
    EMITS = "EMITS"
    ENGAGES = "ENGAGES"
    GUIDES = "GUIDES"
    HAS_ANTENNA = "HAS_ANTENNA"
    HAS_COMPONENT = "HAS_COMPONENT"
    HAS_GUIDANCE = "HAS_GUIDANCE"
    HAS_PERFORMANCE = "HAS_PERFORMANCE"
    HAS_PROCESSING_CHAIN = "HAS_PROCESSING_CHAIN"
    HAS_PROPULSION = "HAS_PROPULSION"
    HAS_RECEIVER = "HAS_RECEIVER"
    HAS_SCAN = "HAS_SCAN"
    HAS_SEEKER = "HAS_SEEKER"
    HAS_SIGNATURE = "HAS_SIGNATURE"
    HAS_STAGE = "HAS_STAGE"
    HAS_SUBSYSTEM = "HAS_SUBSYSTEM"
    HAS_TIMELINE = "HAS_TIMELINE"
    HAS_TRANSMITTER = "HAS_TRANSMITTER"
    INSTALLED_ON = "INSTALLED_ON"
    INSTANCE_OF = "INSTANCE_OF"
    IS_A = "IS_A"
    LAUNCHES = "LAUNCHES"
    MANUFACTURED_BY = "MANUFACTURED_BY"
    MENTIONED_IN = "MENTIONED_IN"
    OPERATED_BY = "OPERATED_BY"
    OPERATES_IN_BAND = "OPERATES_IN_BAND"
    PART_OF = "PART_OF"
    PROCESSES = "PROCESSES"
    PROVIDES = "PROVIDES"
    RADIATES = "RADIATES"
    RECEIVES = "RECEIVES"
    REVIEWED_BY = "REVIEWED_BY"
    SPECIFIED_BY = "SPECIFIED_BY"
    SUPERSEDES = "SUPERSEDES"
    SUPPORTED_BY = "SUPPORTED_BY"
    SUPPORTS_ENGAGEMENT_OF = "SUPPORTS_ENGAGEMENT_OF"
    TESTED_IN = "TESTED_IN"
    TRACKS = "TRACKS"
    USES_MODULATION = "USES_MODULATION"
    USES_WAVEFORM = "USES_WAVEFORM"


class RelationshipMetadata(BaseModel):
    """Per-relationship metadata mirroring ``ontology.yaml`` relationship_types.

    Populated via :data:`RELATIONSHIP_METADATA`. Every field mirrors a
    YAML key 1:1; deviations are caught by the parity test.
    """

    model_config = ConfigDict(is_entity=False, frozen=True)

    name: RelationshipType = Field(
        ...,
        description="Canonical relationship type. Matches ontology.yaml `name`.",
    )
    label: str = Field(
        ...,
        description="Human-readable label. Matches ontology.yaml `label`.",
    )
    description: str = Field(
        ...,
        description="Sentence description of what this relationship represents.",
    )
    source_type: Optional[str] = Field(
        None,
        description=(
            "Typical source entity type. None when the relationship is valid "
            "across multiple source types (see VALIDATION_MATRIX for the "
            "authoritative per-triple constraints)."
        ),
    )
    target_type: Optional[str] = Field(
        None,
        description="Typical target entity type. None when multi-valued.",
    )
    cardinality: Optional[str] = Field(
        None,
        description=(
            "Cardinality hint: one-to-one | one-to-many | many-to-many. "
            "YAML-derived; not enforced at Pydantic validation time (per-triple "
            "constraints live in VALIDATION_MATRIX and entity-class edge fields)."
        ),
    )


# Static relationship-metadata declarations. Mirrors what ontology.yaml's
# relationship_types list carried pre-Phase-6 deletion. The byte-for-byte
# frozen snapshot lives at tests/fixtures/ontology/air_defense_v3_snapshot.yaml
# and is the oracle for parity tests (tests/unit/test_relationships_parity.py);
# runtime code reads from these static declarations so production no longer
# touches any YAML file.
_STATIC_RELATIONSHIP_METADATA: list[dict] = [
    {"name": "IS_A", "label": "Is A", "description": "Type hierarchy (subclass relationship)", "source_type": None, "target_type": None},
    {"name": "INSTANCE_OF", "label": "Instance Of", "description": "Type-instance relationship", "source_type": None, "target_type": None},
    {"name": "ALIAS_OF", "label": "Alias Of", "description": "Alternative name for the same entity", "source_type": None, "target_type": None},
    {"name": "PART_OF", "label": "Part Of", "description": "Component or subsystem is part of a larger system", "source_type": None, "target_type": None, "cardinality": "many_to_one"},
    {"name": "CONTAINS", "label": "Contains", "description": "System or assembly contains a component or subsystem", "source_type": None, "target_type": None, "cardinality": "one_to_many"},
    {"name": "HAS_SUBSYSTEM", "label": "Has Subsystem", "description": "System has a major functional subsystem", "source_type": None, "target_type": "SUBSYSTEM", "cardinality": "one_to_many"},
    {"name": "HAS_COMPONENT", "label": "Has Component", "description": "System or subsystem has a physical component", "source_type": None, "target_type": "COMPONENT", "cardinality": "one_to_many"},
    {"name": "HAS_STAGE", "label": "Has Stage", "description": "Propulsion stack has a propulsion stage", "source_type": "PROPULSION_STACK", "target_type": "PROPULSION_STAGE", "cardinality": "one_to_many"},
    {"name": "INSTALLED_ON", "label": "Installed On", "description": "System is installed on a platform", "source_type": None, "target_type": "PLATFORM", "cardinality": "many_to_one"},
    {"name": "DEPLOYED_ON", "label": "Deployed On", "description": "System or unit is deployed on a platform or at a site", "source_type": None, "target_type": "PLATFORM", "cardinality": "many_to_one"},
    {"name": "ASSOCIATED_WITH", "label": "Associated With", "description": "System association (radar↔missile, radar↔AAA)", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "OPERATED_BY", "label": "Operated By", "description": "Platform or system is operated by an organization", "source_type": None, "target_type": "ORGANIZATION", "cardinality": "many_to_many"},
    {"name": "MANUFACTURED_BY", "label": "Manufactured By", "description": "System or component is manufactured by an organization", "source_type": None, "target_type": "ORGANIZATION", "cardinality": "many_to_many"},
    {"name": "OPERATES_IN_BAND", "label": "Operates In Band", "description": "System operates in a frequency band", "source_type": None, "target_type": "FREQUENCY_BAND", "cardinality": "many_to_many"},
    {"name": "USES_WAVEFORM", "label": "Uses Waveform", "description": "Radar uses a specific waveform", "source_type": "RADAR_SYSTEM", "target_type": "WAVEFORM", "cardinality": "many_to_many"},
    {"name": "USES_MODULATION", "label": "Uses Modulation", "description": "Waveform uses a modulation scheme", "source_type": "WAVEFORM", "target_type": "MODULATION", "cardinality": "many_to_many"},
    {"name": "EMITS", "label": "Emits", "description": "System generates an RF emission", "source_type": None, "target_type": "RF_EMISSION", "cardinality": "one_to_many"},
    {"name": "RADIATES", "label": "Radiates", "description": "Antenna radiates an RF emission", "source_type": "ANTENNA", "target_type": "RF_EMISSION", "cardinality": "one_to_many"},
    {"name": "RECEIVES", "label": "Receives", "description": "Receiver processes incoming signals", "source_type": "RECEIVER", "target_type": "RF_EMISSION", "cardinality": "many_to_many"},
    {"name": "PROCESSES", "label": "Processes", "description": "Signal processing chain processes signals", "source_type": "SIGNAL_PROCESSING_CHAIN", "target_type": "RF_EMISSION", "cardinality": "many_to_many"},
    {"name": "HAS_SIGNATURE", "label": "Has Signature", "description": "Emission or system has an RF signature", "source_type": None, "target_type": "RF_SIGNATURE", "cardinality": "many_to_many"},
    {"name": "HAS_SCAN", "label": "Has Scan", "description": "Radar has a scan pattern", "source_type": "RADAR_SYSTEM", "target_type": "SCAN_PATTERN", "cardinality": "one_to_many"},
    {"name": "HAS_RECEIVER", "label": "Has Receiver", "description": "Radar has a receiver subsystem", "source_type": "RADAR_SYSTEM", "target_type": "RECEIVER", "cardinality": "one_to_many"},
    {"name": "HAS_TRANSMITTER", "label": "Has Transmitter", "description": "Radar has a transmitter subsystem", "source_type": "RADAR_SYSTEM", "target_type": "TRANSMITTER", "cardinality": "one_to_many"},
    {"name": "HAS_ANTENNA", "label": "Has Antenna", "description": "Radar has an antenna", "source_type": "RADAR_SYSTEM", "target_type": "ANTENNA", "cardinality": "one_to_many"},
    {"name": "HAS_PROCESSING_CHAIN", "label": "Has Processing Chain", "description": "Radar has a signal processing chain", "source_type": "RADAR_SYSTEM", "target_type": "SIGNAL_PROCESSING_CHAIN", "cardinality": "one_to_one"},
    {"name": "HAS_PERFORMANCE", "label": "Has Performance", "description": "System has performance characteristics", "source_type": None, "target_type": None, "cardinality": "one_to_one"},
    {"name": "CUES", "label": "Cues", "description": "Radar cues a missile system for engagement", "source_type": "RADAR_SYSTEM", "target_type": "MISSILE_SYSTEM", "cardinality": "many_to_many"},
    {"name": "GUIDES", "label": "Guides", "description": "System guides a missile to target", "source_type": None, "target_type": "MISSILE_SYSTEM", "cardinality": "many_to_many"},
    {"name": "TRACKS", "label": "Tracks", "description": "System tracks a target", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "ENGAGES", "label": "Engages", "description": "System engages a target", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "DEFENDS", "label": "Defends", "description": "System defends a platform or area", "source_type": None, "target_type": "PLATFORM", "cardinality": "many_to_many"},
    {"name": "DETECTS", "label": "Detects", "description": "System detects targets", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "DESIGNATES", "label": "Designates", "description": "System designates targets for engagement", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "LAUNCHES", "label": "Launches", "description": "Launcher launches a missile", "source_type": "LAUNCHER_SYSTEM", "target_type": "MISSILE_SYSTEM", "cardinality": "many_to_many"},
    {"name": "SUPPORTS_ENGAGEMENT_OF", "label": "Supports Engagement Of", "description": "System supports engagement of a target type", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "HAS_GUIDANCE", "label": "Has Guidance", "description": "Missile uses a guidance method", "source_type": "MISSILE_SYSTEM", "target_type": "GUIDANCE_METHOD", "cardinality": "many_to_one"},
    {"name": "HAS_PROPULSION", "label": "Has Propulsion", "description": "Missile has a propulsion stack", "source_type": "MISSILE_SYSTEM", "target_type": "PROPULSION_STACK", "cardinality": "one_to_one"},
    {"name": "HAS_SEEKER", "label": "Has Seeker", "description": "Missile has a terminal guidance seeker", "source_type": "MISSILE_SYSTEM", "target_type": "SEEKER", "cardinality": "one_to_one"},
    {"name": "PROVIDES", "label": "Provides", "description": "System provides a capability (DoDAF DM2)", "source_type": None, "target_type": "CAPABILITY", "cardinality": "many_to_many"},
    {"name": "HAS_TIMELINE", "label": "Has Timeline", "description": "Weapon system has an engagement timeline", "source_type": None, "target_type": "ENGAGEMENT_TIMELINE", "cardinality": "one_to_one"},
    # TODO(docs-alignment Chunk F): ASSERTION-sourced rows pending removal.
    # The AssertionEntity class was dropped in B-1 (commit 2498a20); these
    # rows remain dead but kept for parity with validation_matrix.py until
    # Chunk F's test cleanup.
    {"name": "SUPPORTED_BY", "label": "Supported By", "description": "Assertion is supported by a document resource", "source_type": "ASSERTION", "target_type": None, "cardinality": "many_to_many"},
    {"name": "MENTIONED_IN", "label": "Mentioned In", "description": "Entity is mentioned in a document", "source_type": None, "target_type": "DOCUMENT", "cardinality": "many_to_many"},
    {"name": "DERIVED_FROM", "label": "Derived From", "description": "Entity or assertion is derived from another source", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "REVIEWED_BY", "label": "Reviewed By", "description": "Assertion or entity was reviewed by an organization or person", "source_type": None, "target_type": "ORGANIZATION", "cardinality": "many_to_many"},
    {"name": "ABOUT", "label": "About", "description": "Assertion is about an entity", "source_type": "ASSERTION", "target_type": None, "cardinality": "many_to_many"},
    {"name": "SPECIFIED_BY", "label": "Specified By", "description": "A system is specified by a performance specification", "source_type": None, "target_type": "SPECIFICATION", "cardinality": "many_to_many"},
    {"name": "AFFECTS", "label": "Affects", "description": "A failure mode affects a component or subsystem", "source_type": "FAILURE_MODE", "target_type": None, "cardinality": "many_to_many"},
    {"name": "SUPERSEDES", "label": "Supersedes", "description": "A newer document or standard supersedes an older one", "source_type": None, "target_type": None, "cardinality": "many_to_one"},
    {"name": "TESTED_IN", "label": "Tested In", "description": "A system was tested in a test event", "source_type": None, "target_type": "TEST_EVENT", "cardinality": "many_to_many"},
]


def _load_relationship_metadata() -> dict[RelationshipType, RelationshipMetadata]:
    """Build RELATIONSHIP_METADATA from the static declarations above.

    Plan Task 51 (Phase 6) deleted ``ontology_bundles/air_defense_v3/ontology.yaml``.
    Runtime metadata is now fully Python-native; the YAML snapshot at
    ``tests/fixtures/ontology/air_defense_v3_snapshot.yaml`` serves as
    the oracle for parity tests only.
    """
    out: dict[RelationshipType, RelationshipMetadata] = {}
    for r in _STATIC_RELATIONSHIP_METADATA:
        rt = RelationshipType(r["name"])
        out[rt] = RelationshipMetadata(
            name=rt,
            label=r.get("label", r["name"]),
            description=r.get("description", ""),
            source_type=r.get("source_type"),
            target_type=r.get("target_type"),
            cardinality=r.get("cardinality"),
        )
    return out


RELATIONSHIP_METADATA: dict[RelationshipType, RelationshipMetadata] = (
    _load_relationship_metadata()
)
