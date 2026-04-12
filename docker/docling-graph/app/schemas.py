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


class EntityRef(BaseModel):
    """One pre-extracted entity passed to a document_plus_entity_refs pass.
    The 'ref_id' is a compact token (e.g., 'E01') assigned by the worker so
    the LLM can reference these entities in its relationship output.
    Spec §3.5 + §5.9 wire contract."""
    ref_id: str = Field(..., description="Worker-assigned compact token, e.g. 'E01'")
    entity_type: str = Field(..., description="Ontology entity type name, e.g. 'RADAR_SYSTEM'")
    identity_values: dict[str, Any] = Field(
        default_factory=dict,
        description="The identity fields that uniquely pinpoint this entity",
    )
    display_label: Optional[str] = Field(
        default=None,
        description="Human-readable label for prompt preamble rendering",
    )


class ExtractPassRequest(BaseModel):
    """Request body for POST /extract-pass. Spec §5.9 wire contract."""
    bundle_key: str = Field(..., description="Bundle identifier, e.g. 'air_defense_v3'")
    pass_name: str = Field(..., description="Pass name from the bundle manifest, e.g. 'radar_domain'")
    docling_document_json: dict[str, Any] = Field(
        ..., description="Full DoclingDocument JSON"
    )
    upstream_entities: Optional[list[EntityRef]] = Field(
        default=None,
        description="Pre-extracted entity refs for document_plus_entity_refs passes only",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="UUID of the document being processed (for logging correlation)",
    )


class ExtractPassResponse(BaseModel):
    """Response body for POST /extract-pass. Spec §5.9 wire contract.

    The 'pass_output' dict is the Pydantic template instance dumped to
    JSON — the worker re-parses it via the same template class on its side.
    """
    bundle_key: str
    pass_name: str
    pass_output: dict[str, Any] = Field(
        ..., description="Template-class model_dump() output"
    )
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    model: str = "unknown"
    provider: str = "docling-graph"
