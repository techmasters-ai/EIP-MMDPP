"""Pydantic schemas for graph ingest/query endpoints."""

import uuid
from typing import Any, Optional

from pydantic import Field

from app.schemas.common import APIModel


class GraphEntityIngest(APIModel):
    entity_type: str = Field(..., description="Ontology entity type, e.g. EQUIPMENT_SYSTEM")
    name: str = Field(..., min_length=1)
    properties: Optional[dict[str, Any]] = None
    source_chunk_ids: Optional[list[uuid.UUID]] = None


class GraphRelationshipIngest(APIModel):
    from_entity: str = Field(..., min_length=1)
    from_type: str = Field(..., min_length=1)
    to_entity: str = Field(..., min_length=1)
    to_type: str = Field(..., min_length=1)
    relationship_type: str = Field(..., min_length=1)
    properties: Optional[dict[str, Any]] = None


class GraphIngestResponse(APIModel):
    status: str
    node_id: Optional[str] = None
    message: Optional[str] = None


class GraphQueryRequest(APIModel):
    query: str = Field(..., min_length=1, max_length=4096)
    hop_count: int = Field(default=2, ge=1, le=4)
    top_k: int = Field(default=20, ge=1, le=100)


class GraphNeighborhoodRequest(APIModel):
    entity_name: str = Field(..., min_length=1, max_length=4096)
    hop_count: int = Field(default=2, ge=1, le=4)


class GraphNeighborhoodResponse(APIModel):
    center: Optional[dict[str, Any]] = None
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []


class GraphEvidenceItem(APIModel):
    chunk_id: Optional[uuid.UUID] = None
    chunk_type: str
    artifact_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    document_name: Optional[str] = None
    modality: str
    page_number: Optional[int] = None
    classification: str = "UNCLASSIFIED"
    content_text: Optional[str] = None
    # Data lineage — trust, validity, and source characterization
    source_characterization: Optional[str] = None
    date_of_information: Optional[str] = None
    extraction_confidence: Optional[float] = None


class GraphEntityResult(APIModel):
    node_id: Optional[str] = None
    name: str
    entity_type: str
    canonical_name: Optional[str] = None
    score: Optional[float] = None
    hop_count: Optional[int] = None
    relationship_types: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    evidence: list[GraphEvidenceItem] = Field(default_factory=list)


