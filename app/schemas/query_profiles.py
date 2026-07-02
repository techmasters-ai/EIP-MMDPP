"""Schemas for standalone ontology-backed query profiles and exact graph search.

Profiles are first-class ``governance.query_profiles`` rows (no registry
layer); ``QueryProfileCreate``/``QueryProfileUpdate``/``QueryProfileResponse``
are the flat CRUD payloads. ``QueryProfileDefinition`` remains the canonical
shape validator (``validate_shape`` + the RADAR/MISSILE root frozenset) used to
enforce profile well-formedness on the write path."""

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from app.schemas.common import APIModel
from app.schemas.graph_store import GraphEntityResult


class QueryProfileStep(APIModel):
    direction: Literal["out", "in"] = "out"
    rel_types: list[str] = Field(default_factory=list, min_length=1)
    min_hops: int = Field(default=1, ge=1, le=4)
    max_hops: int = Field(default=1, ge=1, le=4)

    @field_validator("rel_types")
    @classmethod
    def validate_rel_types(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("At least one relationship type is required")
        return cleaned

    @model_validator(mode="after")
    def validate_hops(self):
        if self.max_hops < self.min_hops:
            raise ValueError("max_hops must be greater than or equal to min_hops")
        return self


class QueryProfileTraversal(APIModel):
    steps: list[QueryProfileStep] = Field(default_factory=list, min_length=1, max_length=3)


class QueryProfileFieldEvidence(APIModel):
    """Per-field evidence row (spec §5.8). Combines chunk metadata
    (same surface as GraphEvidenceItem) with field-specific signals
    (supporting_snippet + element_uid)."""
    chunk_id: Optional[uuid.UUID] = None
    chunk_type: Optional[str] = None
    artifact_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    document_name: Optional[str] = None
    modality: Optional[str] = None
    page_number: Optional[int] = None
    classification: str = "UNCLASSIFIED"
    content_text: Optional[str] = None
    source_characterization: Optional[str] = None
    date_of_information: Optional[str] = None
    extraction_confidence: Optional[float] = None
    supporting_snippet: str = ""
    element_uid: Optional[str] = None


class QueryProfileFieldEntry(APIModel):
    """One row in a property table — a single canonical field's value
    with its metadata, plus optional per-field evidence (Phase 3)."""
    name: str
    label: str
    value: Any
    description: Optional[str] = None
    examples: Optional[list[Any]] = None
    enum: Optional[list[str]] = None
    evidence: list[QueryProfileFieldEvidence] = Field(default_factory=list)


class QueryProfileFieldGroup(APIModel):
    """A subgroup of fields rendered as one collapsible card on the
    section UI. `subgroup` is the canonical key from
    json_schema_extra['profile_subgroup']; `subgroup_label` is the
    title-cased display name."""
    subgroup: Optional[str] = None
    subgroup_label: Optional[str] = None
    fields: list[QueryProfileFieldEntry]


# Module-local single source of truth for which canonical entity classes
# section_properties profiles can target. Kept in sync with
# _CANONICAL_BY_ENTITY_TYPE in app.services.query_profiles via a
# contract test (Task 15) — the schema layer must not import from the
# service layer (would create a circular dep).
_CANONICAL_ROOT_ENTITY_TYPES: frozenset[str] = frozenset({
    "RADAR_SYSTEM", "MISSILE_SYSTEM",
})


class QueryProfileDefinition(APIModel):
    id: str = Field(..., min_length=1, max_length=100)
    label: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    kind: Literal["section", "section_properties", "dossier"] = "section"
    exposed: bool = True
    root_entity_types: list[str] = Field(default_factory=list)
    target_entity_types: list[str] = Field(default_factory=list)
    traversals: list[QueryProfileTraversal] = Field(default_factory=list)
    profile_sections: list[str] = Field(default_factory=list)
    include_associated_systems: bool = False
    section_profile_ids: list[str] = Field(default_factory=list)
    placeholder_query: Optional[str] = None

    @field_validator("root_entity_types", "target_entity_types", "section_profile_ids")
    @classmethod
    def strip_string_lists(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]

    @model_validator(mode="after")
    def validate_shape(self):
        if self.kind == "section" and not self.traversals:
            raise ValueError("Section profiles require at least one traversal")
        if self.kind == "dossier" and not self.section_profile_ids:
            raise ValueError("Dossier profiles require at least one section_profile_id")
        if self.kind == "section_properties":
            if not self.profile_sections:
                raise ValueError(
                    "section_properties profiles require non-empty profile_sections"
                )
            unknown = [
                t for t in self.root_entity_types
                if t not in _CANONICAL_ROOT_ENTITY_TYPES
            ]
            if unknown:
                raise ValueError(
                    f"section_properties profiles' root_entity_types must be "
                    f"in {sorted(_CANONICAL_ROOT_ENTITY_TYPES)}; got unknown: {unknown}"
                )
        return self


class QueryProfileCreate(APIModel):
    """Create a first-class query profile row (``governance.query_profiles``).

    ``profile_key`` is the stable EXTERNAL identifier (was the old
    ``QueryProfileDefinition.id``) — dossier ``section_profile_ids`` and search
    ``profile_id`` reference it. The remaining profile body (traversals,
    target_entity_types, section_profile_ids, profile_sections,
    include_associated_systems, placeholder_query, …) rides in the nested
    ``definition`` dict — matching ``services.query_profiles.create_profile``'s
    ``definition=`` parameter."""

    profile_key: str = Field(..., min_length=1, max_length=100)
    label: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    kind: Literal["section", "section_properties", "dossier"] = "section"
    root_entity_types: list[str] = Field(default_factory=list)
    definition: dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[uuid.UUID] = None
    enabled: bool = True

    @field_validator("root_entity_types")
    @classmethod
    def strip_root_entity_types(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class QueryProfileUpdate(APIModel):
    """Partial update — every field optional; only fields present in the
    request body are applied (the route uses ``model_dump(exclude_unset=True)``
    so an omitted ``source_id`` is left untouched while an explicit
    ``source_id: null`` clears the scope back to Global). Field names mirror
    ``services.query_profiles.update_profile``'s keyword parameters."""

    label: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    kind: Optional[Literal["section", "section_properties", "dossier"]] = None
    root_entity_types: Optional[list[str]] = None
    definition: Optional[dict[str, Any]] = None
    source_id: Optional[uuid.UUID] = None
    enabled: Optional[bool] = None


class QueryProfileResponse(APIModel):
    """A query profile row as returned by the flat CRUD + list endpoints."""

    id: Optional[uuid.UUID] = None
    profile_key: str
    label: str
    description: Optional[str] = None
    kind: str
    root_entity_types: list[str] = Field(default_factory=list)
    definition: dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[uuid.UUID] = None
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("root_entity_types", mode="before")
    @classmethod
    def _coerce_roots(cls, value: Any) -> Any:
        return value or []

    @field_validator("definition", mode="before")
    @classmethod
    def _coerce_definition(cls, value: Any) -> Any:
        return value or {}


class QueryProfileSearchRequest(APIModel):
    profile_id: str = Field(..., min_length=1, max_length=100)
    query_text: str = Field(..., min_length=1, max_length=4096)
    include_aliases: bool = True
    include_evidence: bool = True
    evidence_top_k: int = Field(default=3, ge=1, le=10)
    top_k: int = Field(default=25, ge=1, le=100)


class QueryProfileSectionResponse(APIModel):
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult
    field_groups: list[QueryProfileFieldGroup] = Field(default_factory=list)
    related_systems: list[GraphEntityResult] = Field(default_factory=list)
    items: list[GraphEntityResult] = Field(default_factory=list)
    total: int = 0


class QueryProfileDossierSection(APIModel):
    profile_id: str
    profile_label: str
    kind: Literal["section", "section_properties"] = "section"
    field_groups: list[QueryProfileFieldGroup] = Field(default_factory=list)
    related_systems: list[GraphEntityResult] = Field(default_factory=list)
    items: list[GraphEntityResult] = Field(default_factory=list)
    total: int = 0


class QueryProfileDossierResponse(APIModel):
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult
    aliases: list[str] = Field(default_factory=list)
    sections: list[QueryProfileDossierSection] = Field(default_factory=list)


class OntologyEntityType(APIModel):
    name: str
    label: str


class OntologyRelationshipType(APIModel):
    name: str


class OntologySection(APIModel):
    """One query-profile section advertised by ``GET /v1/ontology``:
    the section ``name`` (used as the value in a ``section_properties``
    profile's ``profile_sections`` list) plus a human-readable
    ``description`` sourced from the bundle's ``SECTION_DESCRIPTIONS``."""
    name: str
    description: str = ""


class OntologyResponse(APIModel):
    """Live ontology payload for ``GET /v1/ontology`` — served straight
    from the air_defense_v3 Pydantic SSoT (`app.services.ontology_service`),
    no stored/registry copy."""
    version: str
    entity_types: list[OntologyEntityType] = Field(default_factory=list)
    relationship_types: list[OntologyRelationshipType] = Field(default_factory=list)
    profile_sections: list[OntologySection] = Field(default_factory=list)
