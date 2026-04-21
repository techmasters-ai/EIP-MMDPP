"""Request and response models for the Docling-Graph extraction service."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


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
    # Plan 1 observability — populated by the extract_pass handler.
    upstream_ref_count: int = 0
    upstream_preamble_applied: bool = False


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = "ok"
    schema_count: int = 0
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


class ExtractionProvenance(BaseModel):
    """Per-extracted-entity-instance provenance link to a source DoclingDocument element.

    Additive payload attached to ExtractPassResponse — does not change
    ``pass_output``. Consumers that don't read provenance ignore this
    field.

    ``instance_id`` disambiguates distinct extracted instances that
    happen to share the same identity tuple (same ``ontology_name`` +
    ``identity_values``). This matters for:
      * Same-identity duplicates: two separate extractions with the same
        identity don't collapse to one provenance bucket.
      * Empty-identity entities (e.g. PROPULSION_STACK with empty
        ``graph_id_fields``): every instance is separately trackable
        even though logical identity is an empty tuple.

    Downstream merge-preserving dedup (worker-side Task 52a) unions
    provenance by instance_id, not by identity, so information is
    retained even when identities collapse.

    ``element_uid`` is REQUIRED (plan §8 Task 51 strengthened contract).
    The downstream ``derive_structure_links`` mention path at
    ``pipeline.py:4347`` resolves chunks exclusively via ``element_uid``;
    a provenance row with only ``page`` produces zero mention-based
    edges AND cannot be used to compute ``artifact_ids`` for fallback.
    The service drops nodes whose ``element_uid`` cannot be resolved
    (Task 51 Step 3); they never reach this schema.
    """
    instance_id: str = Field(
        ..., description="Unique id per extracted instance in this response (e.g. UUID)."
    )
    ontology_name: str = Field(
        ..., description="Canonical entity_type name (e.g. RADAR_SYSTEM)."
    )
    identity_values: dict[str, Any] = Field(
        ...,
        description="Field-name → value for the entity's graph_id_fields (may be empty).",
    )
    element_uid: str = Field(
        ...,
        description=(
            "DoclingDocument element where the entity was extracted. "
            "REQUIRED for chunk-linking; rows without element_uid cannot "
            "produce mentions and are dropped by the worker before "
            "provenance aggregation (Task 52)."
        ),
    )
    page: Optional[int] = Field(
        None,
        description=(
            "Observational secondary field. NOT sufficient on its own "
            "for chunk linking — element_uid is the authoritative handle."
        ),
    )
    chunk_index: Optional[int] = Field(None)


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
    provenance: list[ExtractionProvenance] = Field(
        default_factory=list,
        description=(
            "Additive payload: per-extracted-instance provenance links "
            "to DoclingDocument elements. Empty by default; populated "
            "by the /extract-pass handler when the service can resolve "
            "element_uid per node in context.knowledge_graph."
        ),
    )
    diagnostics: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Library-level delta extraction trace: batch_errors, "
            "quality_gate verdict, identity_filter stats, path_counts, "
            "merge_stats, property_sparsity, etc. Populated when the "
            "service enables debug=True on PipelineConfig. Exact shape "
            "is the library's trace dict — see docling_graph orchestrator "
            "trace() for schema."
        ),
    )


# Resolve forward references in the module namespace so the classes
# remain instantiable when loaded via importlib under a non-standard
# module name (the docling-graph test conftest loads this file as
# ``docling_graph_service_schemas`` to sidestep the repo-root ``app/``
# package). Without this rebuild, ``from __future__ import annotations``
# leaves ``Any`` / ``Optional`` / ``ExtractionProvenance`` as strings
# that pydantic cannot resolve in the alt-module namespace.
ExtractionProvenance.model_rebuild()
ExtractPassResponse.model_rebuild()
