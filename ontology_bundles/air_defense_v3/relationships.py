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
    Members are alphabetical for stable diff review. 31 members total
    (reduced from 56 by the flat-schema profile refactor — spec §3.4
    removed orphan-class predicates).
    """

    ALIAS_OF = "ALIAS_OF"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    CHILD_OF = "CHILD_OF"
    CONTAINS = "CONTAINS"
    CUES = "CUES"
    DEFENDS = "DEFENDS"
    DEPLOYED_ON = "DEPLOYED_ON"
    DERIVED_FROM = "DERIVED_FROM"
    DESIGNATES = "DESIGNATES"
    DETECTS = "DETECTS"
    ENGAGES = "ENGAGES"
    GUIDES = "GUIDES"
    HAS_COMPONENT = "HAS_COMPONENT"
    HAS_FIGURE = "HAS_FIGURE"
    HAS_IMAGE = "HAS_IMAGE"
    HAS_SECTION = "HAS_SECTION"
    HAS_TABLE = "HAS_TABLE"
    HAS_SUBSYSTEM = "HAS_SUBSYSTEM"
    INSTALLED_ON = "INSTALLED_ON"
    INSTANCE_OF = "INSTANCE_OF"
    IS_A = "IS_A"
    LAUNCHES = "LAUNCHES"
    MANUFACTURED_BY = "MANUFACTURED_BY"
    MENTIONED_IN = "MENTIONED_IN"
    NEAR_TEXT = "NEAR_TEXT"
    OPERATED_BY = "OPERATED_BY"
    PART_OF = "PART_OF"
    REVIEWED_BY = "REVIEWED_BY"
    SUPERSEDES = "SUPERSEDES"
    SUPPORTS_ENGAGEMENT_OF = "SUPPORTS_ENGAGEMENT_OF"
    TRACKS = "TRACKS"
    VARIANT_OF = "VARIANT_OF"


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
    {"name": "INSTALLED_ON", "label": "Installed On", "description": "System is installed on a platform", "source_type": None, "target_type": "PLATFORM", "cardinality": "many_to_one"},
    {"name": "DEPLOYED_ON", "label": "Deployed On", "description": "System or unit is deployed on a platform or at a site", "source_type": None, "target_type": "PLATFORM", "cardinality": "many_to_one"},
    {"name": "ASSOCIATED_WITH", "label": "Associated With", "description": "System association (radar↔missile, radar↔AAA)", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "OPERATED_BY", "label": "Operated By", "description": "Platform or system is operated by an organization", "source_type": None, "target_type": "ORGANIZATION", "cardinality": "many_to_many"},
    {"name": "MANUFACTURED_BY", "label": "Manufactured By", "description": "System or component is manufactured by an organization", "source_type": None, "target_type": "ORGANIZATION", "cardinality": "many_to_many"},
    {"name": "CUES", "label": "Cues", "description": "Directional handoff of target data. Valid shapes: (a) search radar cues fire-control / guidance radar (RADAR_SYSTEM → RADAR_SYSTEM), (b) radar cues missile system for engagement (RADAR_SYSTEM → MISSILE_SYSTEM).", "source_type": "RADAR_SYSTEM", "target_type": None, "cardinality": "many_to_many"},
    {"name": "GUIDES", "label": "Guides", "description": "System guides a missile to target", "source_type": None, "target_type": "MISSILE_SYSTEM", "cardinality": "many_to_many"},
    {"name": "TRACKS", "label": "Tracks", "description": "System tracks a target", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "ENGAGES", "label": "Engages", "description": "System engages a target", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "DEFENDS", "label": "Defends", "description": "System defends a platform or area", "source_type": None, "target_type": "PLATFORM", "cardinality": "many_to_many"},
    {"name": "DETECTS", "label": "Detects", "description": "System detects targets", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "DESIGNATES", "label": "Designates", "description": "System designates targets for engagement", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "LAUNCHES", "label": "Launches", "description": "Launcher launches a missile", "source_type": "LAUNCHER_SYSTEM", "target_type": "MISSILE_SYSTEM", "cardinality": "many_to_many"},
    {"name": "SUPPORTS_ENGAGEMENT_OF", "label": "Supports Engagement Of", "description": "System supports engagement of a target type", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "MENTIONED_IN", "label": "Mentioned In", "description": "Entity is mentioned in a document", "source_type": None, "target_type": "DOCUMENT", "cardinality": "many_to_many"},
    {"name": "DERIVED_FROM", "label": "Derived From", "description": "Entity or assertion is derived from another source", "source_type": None, "target_type": None, "cardinality": "many_to_many"},
    {"name": "REVIEWED_BY", "label": "Reviewed By", "description": "Assertion or entity was reviewed by an organization or person", "source_type": None, "target_type": "ORGANIZATION", "cardinality": "many_to_many"},
    {"name": "SUPERSEDES", "label": "Supersedes", "description": "A newer document or standard supersedes an older one", "source_type": None, "target_type": None, "cardinality": "many_to_one"},
    {"name": "HAS_SECTION", "label": "Has Section", "description": "Document or parent contains a section", "source_type": None, "target_type": "SECTION", "cardinality": "one_to_many"},
    {"name": "HAS_FIGURE",  "label": "Has Figure",  "description": "Document or section contains a captioned figure", "source_type": None, "target_type": "FIGURE", "cardinality": "one_to_many"},
    {"name": "HAS_TABLE",   "label": "Has Table",   "description": "Document or section contains a table", "source_type": None, "target_type": "TABLE", "cardinality": "one_to_many"},
    {"name": "CHILD_OF",    "label": "Child Of",    "description": "Hierarchical containment (e.g. sub-section within parent section)", "source_type": None, "target_type": None, "cardinality": "many_to_one"},
    {"name": "HAS_IMAGE",   "label": "Has Image",   "description": "Document or section contains an uncaptioned image or embedded picture", "source_type": None, "target_type": "IMAGE", "cardinality": "one_to_many"},
    {"name": "NEAR_TEXT",   "label": "Near Text",   "description": "Figure or image appears near a text block in reading order", "source_type": None, "target_type": "TEXT_BLOCK", "cardinality": "one_to_many"},
    {"name": "VARIANT_OF",  "label": "Variant Of",  "description": "Variant, model, designation, or configuration belongs to a parent system family. Generic — applies across any product-family taxonomy where a child designation rolls up to a parent family entity of the same type.", "source_type": None, "target_type": None, "cardinality": "many_to_one"},
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
