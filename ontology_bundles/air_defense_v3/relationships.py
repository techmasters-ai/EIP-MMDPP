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


def _load_relationship_metadata() -> dict[RelationshipType, RelationshipMetadata]:
    """Build RELATIONSHIP_METADATA from the YAML ontology.

    Temporary bootstrap during Phase 2–5 while YAML is still the authoritative
    source. Phase 6 deletes YAML; this function is then replaced with static
    declarations (one RelationshipMetadata per member) in a follow-up commit
    to remove the runtime YAML read.
    """
    import yaml
    from pathlib import Path

    yaml_path = Path(__file__).parent / "ontology.yaml"
    with yaml_path.open() as f:
        ont = yaml.safe_load(f)

    out: dict[RelationshipType, RelationshipMetadata] = {}
    for r in ont["relationship_types"]:
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
