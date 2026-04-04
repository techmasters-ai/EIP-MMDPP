"""Request and response models for the Docling-Graph extraction service."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request body for /extract-all endpoint."""
    document_id: str = Field(..., description="UUID of the document being processed")
    docling_document_json: dict[str, Any] = Field(..., description="Full DoclingDocument JSON (skips re-conversion)")
    ontology_definition: Optional[dict[str, Any]] = Field(default=None, description="Optional per-request ontology override")
    ontology_version: Optional[str] = Field(default=None, description="Expected ontology version (logged if mismatched)")


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction pipeline run."""
    node_count: int = 0
    edge_count: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)
    extraction_contract: str = "delta"
    gleaning_passes: int = 0
    resolvers_applied: bool = False
    quality_gate_passed: bool = True
    validation_pass_applied: bool = False
    validation_pass_edges_added: int = 0


class ExtractionResponse(BaseModel):
    """Response body for /extract-all endpoint."""
    graph: dict[str, Any] = Field(..., description="Serialized NetworkX graph (node-link JSON)")
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    model: str = "unknown"
    provider: str = "docling-graph"
    ontology_version: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = "ok"
    ontology_version: Optional[str] = None
    template_count: int = 0
    extraction_contract: str = "delta"
    pipeline_version: str = "unknown"
