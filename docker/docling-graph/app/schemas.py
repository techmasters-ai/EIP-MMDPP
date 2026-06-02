"""Request and response models for the Docling-Graph extraction service."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    aliases: Optional[list[str]] = Field(
        default=None,
        description=(
            "Additional names the LLM may match against in document chunks. "
            "Populated from upstream-pass schema fields like ``nomenclature`` "
            "and ``name`` that are aliases of ``identity_values`` (e.g., "
            "Industry Designation 'SA-75', NATO 'SA-2A', common name "
            "'Guideline' for the same missile). The relationship pass uses "
            "these to resolve chunk-cell names back to upstream ref_ids."
        ),
    )
    properties: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional non-identity metadata used by postprocess. Generic — "
            "any key/value pair an upstream pass exposes for downstream "
            "structural use. Item 3 reads ``emitter_function`` here on "
            "RADAR_SYSTEM refs to validate CUES relationship direction."
        ),
    )


class SelectedChunkInput(BaseModel):
    """One pre-built merged chunk supplied by the worker for chunked-mode
    extraction. Receiver side of the merged-chunk routing wire contract
    (Phase 0 Task 0b, plan 2026-05-27-merged-chunk-routing.md).

    When ``ExtractPassRequest.selected_chunks`` is populated, docling-graph
    SKIPS both ``_sanitize_docling_document`` and ``DocumentChunker`` —
    these chunks ARE the LLM batches. Provenance flows back through
    ``source_refs`` (DoclingDocument element self_refs covered by the
    merged chunk).

    ``model_config = ConfigDict(extra="ignore")``: pydantic v2 nested
    config does NOT inherit, so this opt-out is local to the per-chunk
    item. It exists so the worker can serialize its own ``SelectedChunk``
    model (which carries an extra ``chunk_key`` field for forward-compat
    payload pre-staging) directly without a per-callsite ``.exclude=``
    strip. The parent ``ExtractPassRequest.extra='forbid'`` contract
    above is unaffected.
    """
    model_config = ConfigDict(extra="ignore")

    chunk_index: int = Field(
        ..., description="Position of the chunk in HybridChunker output (0-indexed)."
    )
    text: str = Field(
        ...,
        description=(
            "The merged chunk text (output of ``chunker.contextualize()``). "
            "Used verbatim as the LLM batch input — no re-chunking, no "
            "sanitize, no normalization downstream."
        ),
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description=(
            "DoclingDocument element self_refs covered by this merged "
            "chunk (e.g. ``['#/texts/35', '#/texts/36']``). Preserved "
            "for downstream provenance — every extracted entity's "
            "``evidence_units`` resolves back through these refs."
        ),
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "Source page number(s) covered by this merged chunk, supplied "
            "by the worker from ``ExtractionChunk.page_number``. Read into "
            "the pre-built-chunk metadata so synthesized provenance carries "
            "a real page instead of None. Empty when the source row has no "
            "page (e.g. text-only input). Falls back to resolving "
            "``source_refs`` against the DoclingDocument when empty."
        ),
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Token count of ``text`` per the worker-side tokenizer "
            "(bge-m3). Diagnostic only — receiver does not enforce."
        ),
    )


class ExtractPassRequest(BaseModel):
    """Request body for POST /extract-pass. Spec §5.9 wire contract."""

    # C2 review fix: reject unknown keys so direct API / notebook callers can't
    # silently typo an override field name (e.g. ``llm_batch_size_token`` vs
    # the correct ``llm_batch_token_size``) and run with env defaults thinking
    # they overrode it. Worker-produced requests are already protected by the
    # manifest-side ExecutionProfile validator; this closes the same gap at the
    # HTTP boundary for non-worker callers.
    #
    # NOTE: pydantic v2 nested-model ``model_config`` does NOT inherit. The
    # nested ``SelectedChunkInput`` declares ``extra='ignore'`` locally so a
    # worker-side ``SelectedChunk.model_dump()`` (which includes ``chunk_key``
    # for forward-compat payload pre-staging) round-trips without rejection.
    # This top-level forbid is preserved.
    model_config = ConfigDict(extra="forbid")

    bundle_key: str = Field(..., description="Bundle identifier, e.g. 'air_defense_v3'")
    pass_name: str = Field(..., description="Pass name from the bundle manifest, e.g. 'radar_identity'")
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
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "Per-request temperature override for the extraction LLM call. "
            "When None (default), uses the service-wide DOCLING_GRAPH_LLM_TEMPERATURE "
            "(currently 0.1). Used by the extraction_walkthrough notebook to A/B "
            "different temperatures against the same input. Production worker "
            "callers omit this field and inherit the service default."
        ),
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Per-request Ollama model override for the extraction LLM call. "
            "When None (default), uses the service-wide DOCLING_GRAPH_LLM_MODEL. "
            "Used by the extraction_walkthrough notebook to compare different "
            "models (e.g. 'llama3.3:70b' vs 'gpt-oss:120b') across passes. "
            "Production worker callers omit this field and inherit the service "
            "default. The named model must already be pulled in Ollama; an "
            "unknown name fails at chat time."
        ),
    )
    llm_batch_token_size: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Per-request override for chunk_batches_by_token_limit's max_batch_tokens. "
            "When None (default), uses the service-wide DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE "
            "(currently 1024). Larger batches mean fewer LLM calls (faster) but more "
            "text the model must attend to AND emit valid JSON for (potentially worse "
            "JSON-failure rate). Used by the extraction_walkthrough notebook to A/B "
            "batch sizing against the same input. Production worker callers omit this "
            "field and inherit the service default. Note: history shows 2048 was "
            "tried and rolled back to 1024 due to Ollama-local instability — see "
            "config_builder.py:42-44."
        ),
    )
    think: Optional[bool | str] = Field(
        default=None,
        description=(
            "Per-request override for the LLM's thinking-mode toggle. When None "
            "(default), uses the service-wide DOCLING_GRAPH_LLM_THINK. Accepts bool "
            "(True/False) or string ('true'/'false', or 'low'/'medium'/'high' for "
            "gpt-oss-class models). Used by the extraction_walkthrough notebook to "
            "A/B thinking-mode against the same input. Production worker callers "
            "omit this field and inherit the service default."
        ),
    )
    # C2: per-pass execution profile knobs (new in C2 Iter 1).
    chunk_max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Per-request override for HybridChunker's max_tokens per chunk. "
            "When None (default), uses the service-wide "
            "DOCLING_GRAPH_CHUNK_MAX_TOKENS (currently 512). Sourced from "
            "pass_def.execution.chunk_max_tokens when the manifest declares "
            "an execution block; otherwise omitted."
        ),
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Per-request override for the LLM's max_tokens generation cap. "
            "When None (default), uses the service-wide "
            "DOCLING_GRAPH_LLM_MAX_TOKENS (currently 4096). Applied via "
            "OllamaChatClient.with_runtime_defaults(max_tokens=...) so it "
            "reaches the actual outbound HTTP call, not just PipelineConfig. "
            "Sourced from pass_def.execution.max_tokens when the manifest "
            "declares an execution block; otherwise omitted."
        ),
    )
    # Plan 2026-05-27-merged-chunk-routing.md Phase 0 Task 0b: chunked-mode
    # routing. When populated, docling-graph SKIPS both
    # ``_sanitize_docling_document`` and the internal ``DocumentChunker``;
    # the chunks ride straight to the LLM batch loop. Absent / None keeps
    # the existing per-element behavior.
    selected_chunks: Optional[list[SelectedChunkInput]] = Field(
        default=None,
        description=(
            "Pre-built merged chunks supplied by the worker. When set, "
            "docling-graph treats these as the LLM batch inputs directly: "
            "sanitize is skipped, the internal DocumentChunker is skipped, "
            "and ``source_refs`` are preserved for downstream provenance. "
            "Absent / null keeps the legacy per-element chunked path."
        ),
    )
    field_subset: Optional[list[str]] = Field(
        default=None,
        description=(
            "Tasks F2+F4 (§9 subset-schema extraction): the active field names "
            "to include in the LLM extraction prompt, forwarded from the worker's "
            "chunk-scope router response. When None (the default), docling-graph "
            "includes all schema fields — byte-identical to today's behavior. "
            "F3 (applying the subset inside docling-graph) is a separate later task."
        ),
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
    # Additive fields — extraction-time evidence carried through to the
    # response. All optional / default-empty for backward compatibility.
    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "DoclingDocument self_refs (e.g. '#/texts/12') the LLM cited "
            "as evidence for this entity instance. Populated when the "
            "delta normalizer found valid evidence_ids in the LLM output."
        ),
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Pages the cited evidence units span. Sorted, deduped.",
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description=(
            "Concatenated text of the evidence units (best-effort, "
            "truncated to ~500 chars). For human review and downstream "
            "snippet display."
        ),
    )
    # Positional lineage carried from the delta-IR normalizer's per-node
    # stamp (context._delta_merged_graph). The normalizer marks
    # self_refs/chunk_indexes as the AUTHORITATIVE positional lineage of the
    # batch every node spanned; cited_refs holds ONLY the LLM's explicit
    # citations (possibly empty). element_uid == self_refs[0] when present.
    # All additive / default-empty for backward compatibility.
    self_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Authoritative positional DoclingDocument self_refs the entity's "
            "source batch spanned. element_uid == self_refs[0] when present."
        ),
    )
    chunk_indexes: list[int] = Field(
        default_factory=list,
        description="Library chunk indexes (0..N-1) the entity's batch spanned.",
    )
    cited_refs: list[str] = Field(
        default_factory=list,
        description=(
            "DoclingDocument self_refs the LLM EXPLICITLY cited as evidence "
            "for this entity. Subset of self_refs; empty when the LLM cited "
            "nothing (positional self_refs still anchor the entity)."
        ),
    )


class ExtractionFieldProvenance(BaseModel):
    """Wire-shape per-field provenance row (spec §5.3, §5.1.1).

    Built by the service post-process from the pass-template's
    field_provenance list. Joins entity_index → instance_id by indexing
    into the pass's primary entity list (resolved via
    _primary_list_field_name)."""
    instance_id: str = Field(
        ..., description="Resolved entity instance id from the merge layer."
    )
    field_name: str = Field(
        ..., description="Canonical field name on the entity model."
    )
    value: Any = Field(
        None, description="The field value the LLM extracted."
    )
    supporting_snippet: str = Field(
        ..., description="Verbatim quote from the source chunks."
    )
    element_uid: Optional[str] = Field(
        None,
        description=(
            "DoclingDocument element_uid the snippet was matched to. "
            "None when the resolver couldn't locate the snippet — "
            "treated as 'unverified source' by the UI."
        ),
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="Stable evidence_id (== DoclingDocument self_ref) for the source unit.",
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number of the supporting evidence unit.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Document UUID this field's evidence came from.",
    )
    # NEW (spec 2026-05-11 §11.6 — channel A cell-ref provenance):
    chunk_index: Optional[int] = Field(
        default=None,
        description=(
            "Library chunk_id (0..N-1, sequential per pass) the field was "
            "extracted from. Required for the cell_refs lookup. None for "
            "rows where chunk_index cannot be determined."
        ),
    )
    cell_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Cell-level self_refs of the form '#/tables/{N}/data/table_cells/{M}' "
            "when the field was extracted from a chunk synthesized from a "
            "NormalizedTable. Empty for prose chunks. Populated post-construction "
            "by _enrich_field_provenance_with_cell_refs via the two-hop lookup "
            "(chunk_index → chunk_to_self_refs → first #/texts/N → bridge → cell_refs)."
        ),
    )
    # Positional lineage carried from the parent entity node's delta-IR stamp
    # (build_entity_provenance_from_delta_graph). Authoritative positional refs
    # of the batch the entity spanned; additive / default-empty.
    self_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Authoritative positional DoclingDocument self_refs carried from "
            "the parent entity's delta-IR provenance stamp."
        ),
    )
    chunk_indexes: list[int] = Field(
        default_factory=list,
        description="Library chunk indexes the parent entity's batch spanned.",
    )


class ExtractionRelationshipProvenance(BaseModel):
    """Per-extracted-relationship provenance link to source DoclingDocument
    elements. Mirrors ExtractionProvenance but for edges.

    Built from delta-IR relationship.provenance (via context._delta_merged_graph),
    which the Pydantic-to-graph converter does NOT preserve on edges.
    """
    relationship_type: str
    source_instance_id: Optional[str] = None
    target_instance_id: Optional[str] = None
    # Cross-pass DTO ref ids (e.g. "E001"/"E041") carried from a DTO
    # relationship node's properties.from_ref_id / to_ref_id. The worker
    # resolves these through PassResult.upstream_refs to a precise
    # (from_identity, rel_type, to_identity) triple so each edge keys to its
    # OWN source chunks instead of the coarse __rel_type_fallback__ bucket.
    # Distinct from source/target_instance_id (per-instance UUIDs) — additive.
    from_ref_id: Optional[str] = None
    to_ref_id: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)
    self_refs: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    supporting_snippet: Optional[str] = None


class TableFact(BaseModel):
    """Per-cell deterministic fact derived from a variants table row.
    Spec §5.4.
    """
    model_config = ConfigDict(frozen=True)
    canonical_entity: str
    entity_type: str
    schema_field: str
    value: Any
    source_label: str
    section_ctx: Optional[str] = None
    pass_name: str
    raw_text: str


class CrossEntityHint(BaseModel):
    """Row-level cross-entity reference. v1: collected but not applied
    as edges. Spec §5.4."""
    model_config = ConfigDict(frozen=True)
    source_canonical: str
    source_entity_type: str
    target_alias: str
    target_entity_type: str
    relationship_kind: str


class TableOverlay(BaseModel):
    """Doc-level deterministic overlay derived from a variants table.
    Spec §5.4. Mutable defaults via Field(default_factory=...) so each
    instance gets its own dict/list."""
    alias_map_by_entity_type: dict[str, dict[str, str]] = Field(default_factory=dict)
    facts: list[TableFact] = Field(default_factory=list)
    cross_entity_hints: list[CrossEntityHint] = Field(default_factory=list)


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
    field_provenance: list[ExtractionFieldProvenance] = Field(
        default_factory=list,
        description=(
            "Per-field source snippets emitted by the LLM and resolved "
            "to DoclingDocument element_uid by the service post-process. "
            "Phase 3 of the flat-schema profile refactor (spec §5.3)."
        ),
    )
    relationship_provenance: list[ExtractionRelationshipProvenance] = Field(
        default_factory=list,
        description=(
            "Per-relationship provenance, mirrors `provenance` for entities. "
            "Built from delta-IR relationship.provenance when present. "
            "Empty by default — additive on the wire shape."
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
    table_overlay: Optional[TableOverlay] = Field(
        default=None,
        description=(
            "Doc-level deterministic overlay (Mechanism A1, spec §5.4). "
            "None when no qualifying variants table found OR kill switch "
            "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false."
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
ExtractionRelationshipProvenance.model_rebuild()
ExtractPassResponse.model_rebuild()
TableFact.model_rebuild()
CrossEntityHint.model_rebuild()
TableOverlay.model_rebuild()
SelectedChunkInput.model_rebuild()
ExtractPassRequest.model_rebuild()
