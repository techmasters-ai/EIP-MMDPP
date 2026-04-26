"""Schemas for ontology-backed query profile registries and exact graph search."""

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
    """Phase 3 stub. Populated with snippet + element_uid + chunk
    metadata once the docling-graph extraction emits per-field
    provenance. Empty default for Phase 2."""
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


class QueryProfileDefinition(APIModel):
    id: str = Field(..., min_length=1, max_length=100)
    label: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    kind: Literal["section", "dossier"] = "section"
    exposed: bool = True
    root_entity_types: list[str] = Field(default_factory=list)
    target_entity_types: list[str] = Field(default_factory=list)
    traversals: list[QueryProfileTraversal] = Field(default_factory=list)
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
        return self


class QueryProfileRegistryCreate(APIModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    source_id: Optional[uuid.UUID] = None
    ontology_name: Optional[str] = Field(default=None, max_length=255)
    ontology_version: Optional[str] = Field(default=None, max_length=100)
    ontology_definition: Optional[dict[str, Any]] = None
    profiles: list[QueryProfileDefinition] = Field(default_factory=list)
    is_active: bool = False


class QueryProfileRegistryUpdate(APIModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    source_id: Optional[uuid.UUID] = None
    ontology_name: Optional[str] = Field(default=None, max_length=255)
    ontology_version: Optional[str] = Field(default=None, max_length=100)
    ontology_definition: Optional[dict[str, Any]] = None
    profiles: Optional[list[QueryProfileDefinition]] = None
    is_active: Optional[bool] = None


class QueryProfileRegistryResponse(APIModel):
    id: uuid.UUID
    name: str
    description: Optional[str] = None
    source_id: Optional[uuid.UUID] = None
    ontology_name: Optional[str] = None
    ontology_version: Optional[str] = None
    ontology_definition: Optional[dict[str, Any]] = None
    profiles: list[QueryProfileDefinition] = Field(default_factory=list)
    is_active: bool
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class ActiveQueryProfilesResponse(APIModel):
    registry: Optional[QueryProfileRegistryResponse] = None
    exposed_profiles: list[QueryProfileDefinition] = Field(default_factory=list)


class QueryProfileSearchRequest(APIModel):
    profile_id: str = Field(..., min_length=1, max_length=100)
    query_text: str = Field(..., min_length=1, max_length=4096)
    include_aliases: bool = True
    include_evidence: bool = True
    evidence_top_k: int = Field(default=3, ge=1, le=10)
    top_k: int = Field(default=25, ge=1, le=100)


class QueryProfileSectionResponse(APIModel):
    registry_id: Optional[uuid.UUID] = None
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
    registry_id: Optional[uuid.UUID] = None
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult
    aliases: list[str] = Field(default_factory=list)
    sections: list[QueryProfileDossierSection] = Field(default_factory=list)
