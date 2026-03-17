"""Pydantic request/response models for the Docling-Graph extraction service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request body for the /extract endpoint."""

    document_id: str = Field(..., description="Internal document identifier")
    text: str = Field(..., description="Plain text to extract entities/relationships from")
    ontology_version: str | None = Field(
        None,
        description="Expected ontology version; logged as warning on mismatch",
    )
    template_group: str | None = Field(
        None,
        description="Ontology layer group to extract (reference, equipment, rf_signal, weapon, operational). "
        "If None, uses legacy single-template behavior.",
    )
    mode: str = Field(
        "entities",
        description="Extraction mode: 'entities' for entity extraction, 'relationships' for relationship-only pass.",
    )
    entities_context: list[dict] | None = Field(
        None,
        description="For mode='relationships': list of {name, entity_type} dicts from prior entity passes.",
    )


class ExtractedEntityResponse(BaseModel):
    """A single extracted entity."""

    name: str
    entity_type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict = Field(default_factory=dict)


class ExtractedRelationshipResponse(BaseModel):
    """A single extracted relationship between two entities."""

    from_name: str
    from_type: str
    rel_type: str
    to_name: str
    to_type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractionResponse(BaseModel):
    """Full response from the /extract endpoint."""

    entities: list[ExtractedEntityResponse] = Field(default_factory=list)
    relationships: list[ExtractedRelationshipResponse] = Field(default_factory=list)
    ontology_version: str | None = None
    model: str | None = None
    provider: str | None = None
