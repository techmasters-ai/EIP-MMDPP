"""Pydantic schemas for graph (AGE) ingest/query endpoints."""

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
