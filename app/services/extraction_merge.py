"""Extraction merge and resolve module.

Phase 1 merge: entity de-duplication keyed by LogicalIdentity.
Phase 2 resolve: relationship validation and rejection classification.

Spec §3.6 + §3.7 + §3.8 + §3.9 + §6.2 + §6.7.

ChunkForDerivation and DerivedEdge are defined here (canonical location)
and re-exported; ontology_bundles/air_defense_v3/derive_rules.py imports
them from here instead of redefining them.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel

from app.services.table_overlay import TableOverlay  # type: ignore[import]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class YieldStatus(str, Enum):
    HIT = "HIT"
    EMPTY = "EMPTY"
    BRIDGES_ONLY = "BRIDGES_ONLY"
    DEGRADED = "DEGRADED"


class RelationshipRejectionReason(str, Enum):
    MISSING_REL_TYPE = "missing_rel_type"
    INVALID_IDENTITY_PAYLOAD = "invalid_identity_payload"
    UNKNOWN_REF_ID = "unknown_ref_id"
    FROM_ENDPOINT_NOT_FOUND = "from_endpoint_not_found"
    TO_ENDPOINT_NOT_FOUND = "to_endpoint_not_found"
    INVALID_TRIPLE = "invalid_triple"


# ---------------------------------------------------------------------------
# LogicalIdentity
# ---------------------------------------------------------------------------


def norm(value: Any) -> str:
    """Case/whitespace-insensitive identity key: trim, casefold, collapse whitespace.

    Used to compare LogicalIdentity values so that entities differing only
    by case or incidental whitespace (e.g. ``FAN SONG`` vs ``Fan Song``)
    merge into a single record instead of creating duplicate vertices.
    """
    return " ".join(str(value).strip().casefold().split()) if value is not None else ""


@dataclass(frozen=True, eq=False)
class LogicalIdentity:
    """Hashable identity for an extracted entity.

    Frozen so instances can be used as dict keys in the merge index.
    Spec §3.6.

    Equality/hash are case/whitespace-insensitive (routed through
    ``norm_key``) so that two identities differing only by case or
    incidental whitespace merge into one record. ``identity_tuple`` /
    ``identity_values_dict()`` remain the RAW first-seen values — they
    feed ``build_display_label()`` and must not be normalized in place.
    """

    entity_type: str
    identity_field_names: tuple[str, ...]  # ordered from ontology.yaml
    identity_tuple: tuple[Any, ...]        # parallel values
    scope: Literal["document", "global"]
    document_id: str | None               # populated iff scope == "document"

    @property
    def norm_key(self) -> tuple:
        """Case/whitespace-insensitive key used for equality and hashing.

        Casefold-merge is intentional: two identity values that casefold
        identically WILL merge (accepted per spec — identity fields are
        proper-noun designators). ``document_id`` is kept RAW (it's a UUID,
        not a display string) and only participates for document scope.
        """
        base = tuple(norm(v) for v in self.identity_tuple)
        return (
            self.entity_type,
            base,
            self.document_id if self.scope == "document" else None,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogicalIdentity):
            return NotImplemented
        return self.norm_key == other.norm_key

    def __hash__(self) -> int:
        return hash(self.norm_key)

    def identity_values_dict(self) -> dict[str, Any]:
        """Identity field names zipped with values. Does NOT include document_id."""
        return dict(zip(self.identity_field_names, self.identity_tuple, strict=True))

    def as_upsert_identity_dict(self) -> dict[str, Any]:
        """Shape expected by GraphStore.NodeRecord.identity_fields.

        Adds document_id for document-scoped entities so the composite
        identity distinguishes same-named entities across documents.
        """
        d = dict(zip(self.identity_field_names, self.identity_tuple, strict=True))
        if self.scope == "document":
            assert self.document_id is not None, (
                "document_id required for scope=document"
            )
            d["document_id"] = self.document_id
        return d

    def serialize_as_entity_id(self) -> str:
        """Canonical stable string for this identity (Phase 8 Task 52b).

        Format: ``v1::{entity_type}::{k1}={v1!r}|{k2}={v2!r}|...[|__doc__={document_id}]``

          * Leading ``v1`` is the format-schema version. Future format
            changes bump to ``v2`` and coexist with v1-persisted entity_ids
            during migration. Readers seeing v1 strings under a v2 code
            base must use a pinned v1 parser — no data surgery required.
          * Identity fields appear in declared ``identity_field_names``
            order (matches the canonical tuple order used everywhere else).
          * Values use ``repr()`` so strings containing ``"::"`` or
            ``"|"`` round-trip without colliding with the format's
            delimiters; ints/None/etc. remain unambiguous.
          * Document-scoped identities append ``__doc__={document_id}``
            so the same identity tuple in different documents produces
            distinct entity_ids.

        Used by ``_serialize_for_audit`` (Task 53) for both ``nodes[]``
        and ``mentions[]``, by ``derive_structure_links`` fallback
        suppression (Task 53b), and by persisted ``EntityChunkEdge``
        rows. Single canonical surface — no inline f-strings inventing
        the format at individual call sites.

        **Write-only contract — do not parse.** This is a stable opaque
        string for comparison and edge-property persistence. If a
        consumer needs ``(entity_type, identity_tuple)`` back, it must
        join against the audit blob's ``nodes[]`` entries (which carry
        the fields separately). Versioning the format (the ``v1::``
        prefix) anticipates future format changes; a parser would
        couple consumers to a specific version and invert the point
        of the canonical helper.
        """
        parts = [
            f"{k}={v!r}"
            for k, v in zip(
                self.identity_field_names, self.identity_tuple, strict=True,
            )
        ]
        if self.scope == "document" and self.document_id is not None:
            parts.append(f"__doc__={self.document_id}")
        return f"v1::{self.entity_type}::{'|'.join(parts)}"


# ---------------------------------------------------------------------------
# ExtractionMetadata + PassResult
# ---------------------------------------------------------------------------


@dataclass
class ExtractionMetadata:
    schema_size_chars: int
    structured_output_mode: Literal["strict", "fallback_json"]


@dataclass
class FieldEvidenceRow:
    """Worker-side per-field evidence row (spec §5.6).

    Mirrors ``docker/docling-graph/app/schemas.py::ExtractionFieldProvenance``
    after the worker resolves element_uid → chunk_id. ``chunk_id`` is
    ``None`` when the snippet didn't match any chunk (unverified source);
    the snippet still ships to the UI with an "Unverified source" badge.
    """
    chunk_id: str | None
    snippet: str
    element_uid: str | None
    value: Any = None
    evidence_id: str | None = None
    page: int | None = None
    document_id: str | None = None
    # Task 3: POSITIONAL lineage lists (mirror ExtractionProvenance).
    # ``self_refs`` every source element ref backing this field's
    # evidence (scalar ``element_uid`` is self_refs[0]); ``chunk_indexes``
    # the parallel chunk positions.
    self_refs: list[str] = field(default_factory=list)
    chunk_indexes: list[int] = field(default_factory=list)


@dataclass
class ExtractionProvenance:
    """Worker-side mirror of ``docker/docling-graph/app/schemas.py::ExtractionProvenance``.

    Populated by ``_parse_pass_response`` from the docling-graph service
    response's ``provenance`` list (Phase 8 Task 51 on the service side;
    Task 52 on the worker side). Each row carries per-extracted-instance
    provenance linking the entity back to a source DoclingDocument
    element, used downstream by ``_serialize_for_audit`` to emit
    ``mentions[]`` that ``derive_structure_links`` converts to
    ``EXTRACTED_FROM`` chunk-link edges (Phase 8 Task 53).

    Not frozen — ``identity_values`` is an unhashable dict. Aggregation
    (Task 52a) dedups by ``instance_id`` (string), not by whole-object
    equality.
    """
    instance_id: str
    ontology_name: str
    identity_values: dict[str, Any]
    element_uid: str
    page: int | None = None
    chunk_index: int | None = None
    evidence_ids: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    evidence_text: str | None = None
    # Task 3: POSITIONAL lineage lists carried end-to-end. ``self_refs``
    # holds every source DoclingDocument element ref backing this
    # extracted instance (the scalar ``element_uid`` is self_refs[0]);
    # ``chunk_indexes`` the parallel chunk positions; ``cited_refs`` the
    # subset the LLM explicitly cited. Closes the scalar-collapse
    # boundary where only one element_uid/chunk_index survived.
    self_refs: list[str] = field(default_factory=list)
    chunk_indexes: list[int] = field(default_factory=list)
    cited_refs: list[str] = field(default_factory=list)


@dataclass
class ExtractionRelationshipProvenance:
    """Worker-side mirror of docling-graph ExtractionRelationshipProvenance.

    Wire contract: field set MUST stay byte-for-byte equivalent to
    docker/docling-graph/app/schemas.py::ExtractionRelationshipProvenance.
    """
    relationship_type: str
    source_instance_id: str | None = None
    target_instance_id: str | None = None
    # Cross-pass DTO ref ids (e.g. "E001"/"E041") carried from a DTO
    # relationship node's properties.from_ref_id / to_ref_id. Resolved through
    # PassResult.upstream_refs to a precise (from_identity, rel_type,
    # to_identity) triple in _build_relationship_records (per-edge lineage).
    from_ref_id: str | None = None
    to_ref_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    self_refs: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    supporting_snippet: str | None = None


@dataclass
class PreMergeWalkSummary:
    """Shared pre-merge carrier for entity + edge counts (plan Task 34b).

    The pre-merge path walks each PassResult ONCE via walk_entity_graph
    with both on_entity and on_edge callbacks hooked up. ``entities`` is
    the list of every emitted entity (nested children included);
    ``raw_edge_count`` is the number of edge emissions during the walk
    (no validation yet — VALIDATION_MATRIX triple-check happens at
    merge time). Both classify_yield and _count_pass_output consume this
    pre-built summary; neither re-traverses.

    ``system_links`` DTO special case (Decision 4): entities=[];
    raw_edge_count=len(pass_result.template_instance.relationships).
    The DTO list length is the provisional relationships_extracted so
    classify_yield sees non-zero provisional edges when the LLM emitted
    candidate SystemLinkRelationships. Task 36's post-merge branch
    overwrites yield_status authoritatively from per_pass_edge_metrics.
    """
    entities: list[Any]
    raw_edge_count: int


@dataclass
class PassResult:
    """Handoff type between _run_single_pass (producer) and merge_and_resolve (consumer)."""

    pass_name: str
    template_instance: Any  # instantiated Pydantic template class (or stub in tests)
    metadata: ExtractionMetadata
    pre_merge_rejections: list[tuple[Any, RelationshipRejectionReason]]
    # Optional: pre-populated by system_links pass for cross-pass ref_id lookup.
    upstream_refs: dict[str, "LogicalIdentity"] | None = None
    # Populated by the pass loop (plan Task 34b) — single shared pre-merge
    # traversal; consumed by classify_yield + _count_pass_output without
    # re-walking. None means the PassResult was built outside the pass
    # loop (test fixture, or a code path not yet migrated); consumers fall
    # back to walk_entity_graph in entity-only mode.
    pre_merge_walk: "PreMergeWalkSummary | None" = None
    # Phase 8 Task 52: provenance rows parsed from the docling-graph
    # response's ``provenance`` list. Each row carries
    # instance_id / ontology_name / identity_values / element_uid and
    # optional page / chunk_index. Consumed by Task 52a's aggregation
    # into MergedEntityRecord.provenance and by Task 53's
    # _serialize_for_audit mentions[] emission.
    provenance: list["ExtractionProvenance"] = field(default_factory=list)
    # Phase 3 (flat-schema profile refactor) Task 32: per-field evidence
    # rows, parsed from the docling-graph response's field_provenance
    # list and grouped by (instance_id, field_name) here. Consumed by
    # merge_and_resolve to populate MergedEntityRecord.field_evidence.
    # Outer dict keys by instance_id; inner dict keys by field_name.
    field_evidence: dict[str, dict[str, list["FieldEvidenceRow"]]] = field(
        default_factory=dict,
    )
    # Mechanism A1 (spec §4.4 + §5.5): doc-level overlay parsed from
    # the docling-graph /extract-pass response. None when the parser
    # found no qualifying table OR the parser-side kill switch was off.
    # Consumed by merge_and_resolve._extract_doc_overlay (Task 8).
    table_overlay: TableOverlay | None = None
    # Relationship-level provenance rows parsed from the docling-graph response's
    # ``relationship_provenance`` list. Each row links a relationship triple back
    # to source elements.
    relationship_provenance: list["ExtractionRelationshipProvenance"] = field(
        default_factory=list,
    )
    _walker_entities_cache: list[Any] | None = field(default=None, init=False, repr=False)

    def _cached_entities(self) -> list[Any]:
        """Return every entity reachable from ``template_instance``, memoized.

        Preference order (plan Task 35a):
        1. ``pre_merge_walk.entities`` if the pass loop already built the
           shared pre-merge summary (Task 34b) — guarantees this filter and
           the pre-merge counters see the same traversal, avoiding
           count-vs-upstream-ref drift.
        2. Fresh walk via ``walk_entity_graph`` in entity-only mode
           (``on_edge=None``) when the template is a Pydantic BaseModel —
           ontology / document_id are optional in this mode.
        3. Heuristic fallback for non-Pydantic template stubs
           (SimpleNamespace in tests): return ``None`` to signal the caller
           to fall back to the legacy attribute-name lookup.
        """
        if self._walker_entities_cache is not None:
            return self._walker_entities_cache
        if self.pre_merge_walk is not None:
            self._walker_entities_cache = list(self.pre_merge_walk.entities)
            return self._walker_entities_cache
        if isinstance(self.template_instance, BaseModel):
            out: list[Any] = []
            walk_entity_graph(
                self.template_instance,
                on_entity=out.append,
                ontology=None,
                document_id=None,
                on_edge=None,
                visited_objects=set(),
                at_pass_root=True,
            )
            self._walker_entities_cache = out
            return self._walker_entities_cache
        # Non-BaseModel template (SimpleNamespace stub) — caller falls back.
        return None  # type: ignore[return-value]

    def iter_entities_of_type(self, entity_type: str) -> Iterable[Any]:
        """Yield entity instances whose ``model_config['ontology_name']``
        matches ``entity_type`` — recursive via ``walk_entity_graph``.

        For Pydantic BaseModel templates this walks the typed-edge entity
        graph (nested children behind ``edge_label`` fields are reached).
        For non-Pydantic test stubs (SimpleNamespace), falls back to the
        legacy attribute-name heuristic:
          1. '<lower>_list'  — test-fixture convention
          2. '<lower>s'      — plural snake_case
          3. '<lower>'       — bare singular
        """
        entities = self._cached_entities()
        if entities is not None:
            for e in entities:
                cfg = getattr(e, "model_config", {}) or {}
                if cfg.get("ontology_name") == entity_type:
                    yield e
            return

        # Fallback path for SimpleNamespace templates.
        lower = entity_type.lower()
        for attr in (f"{lower}_list", f"{lower}s", lower):
            val = getattr(self.template_instance, attr, None)
            if val is not None and isinstance(val, list):
                for item in val:
                    yield item
                return

    @property
    def relationships(self) -> list[Any]:
        """Return the relationships field (empty list for entities-only passes)."""
        return getattr(self.template_instance, "relationships", []) or []


# ---------------------------------------------------------------------------
# Merged record types
# ---------------------------------------------------------------------------


@dataclass
class MergedEntityRecord:
    identity: LogicalIdentity
    properties: dict[str, Any]   # merged from all source passes
    confidence: float             # highest confidence across merges
    pass_origins: set[str]        # which passes contributed
    display_label: str            # derived via build_display_label
    # Phase 8 Task 52a: aggregated provenance rows — one per distinct
    # (instance_id, element_uid) pair across all passes that produced
    # this record. Populated by merge_and_resolve after the entity-merge
    # loop; consumed by _serialize_for_audit (Task 53) to emit mentions[].
    # Default empty so existing fixtures that construct the record
    # without this kwarg continue to work.
    provenance: list["ExtractionProvenance"] = field(default_factory=list)
    # Phase 3 (flat-schema profile refactor) Task 32: per-field evidence
    # rows aggregated from PassResult.field_evidence across passes that
    # produced this record. Outer dict keys by canonical field_name;
    # inner list is one FieldEvidenceRow per source quotation.
    field_evidence: dict[str, list["FieldEvidenceRow"]] = field(
        default_factory=dict,
    )


@dataclass
class MergedEdgeRecord:
    from_identity: LogicalIdentity
    to_identity: LogicalIdentity
    rel_type: str
    confidence: float
    # Plan Task 36: one edge may be emitted by multiple passes (e.g. a typed
    # edge and a system_links DTO that resolve to the same triple). The
    # cross-pass reducer unions pass_origins across contributing passes.
    pass_origins: set[str]


@dataclass
class PerPassEdgeMetrics:
    """Per-pass post-merge edge accounting (plan Task 36).

    Populated by ``merge_and_resolve`` for EVERY pass — typed-edge and
    system_links alike — so ``_apply_post_merge_yield_updates`` can read
    a uniform-shape carrier without branching on pass kind at the
    accounting level. Field semantics:
      * ``attempted`` — raw pre-validation count (walker edge emissions
        for typed-edge passes; ``len(DTO list)`` for ``system_links``).
      * ``accepted`` — after VALIDATION_MATRIX triple check + Pydantic
        parse. Maps to ``StageRun.relationships_extracted`` authoritatively.
      * ``rejected`` — ``attempted - accepted``. Includes INVALID_TRIPLE
        and any pass-specific rejection reasons. Maps to
        ``StageRun.relationships_rejected`` authoritatively.
      * ``rejection_sample`` — up to N sampled rejected tuples in
        ``_rel_to_dict`` shape for observability.
      * ``rejections_by_reason`` — per-reason counts; preserves parity
        with the current ``_build_rejections_by_reason`` path.
    """
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_sample: list[dict] = field(default_factory=list)
    rejections_by_reason: dict[str, int] = field(default_factory=dict)


@dataclass
class MergedExtraction:
    entities: list[MergedEntityRecord]
    edges: list[MergedEdgeRecord]
    rejected_edges: list[tuple[str, Any, RelationshipRejectionReason]]  # (source_pass, raw_rel, reason)
    rejections_by_pass: dict[str, int]
    pipeline_run_id: str
    document_id: str
    # Plan Task 36 post-merge accounting: populated per pass by
    # merge_and_resolve. _apply_post_merge_yield_updates reads this
    # uniformly for typed-edge and system_links passes.
    per_pass_edge_metrics: dict[str, PerPassEdgeMetrics] = field(default_factory=dict)
    # old_identity -> final_identity map produced by cross-pass identity
    # canonicalization (Mechanism A1). The SAME map _resolve_relationship used
    # to canonicalize MergedEdgeRecord.from_identity/to_identity. Exposed so the
    # worker can canonicalize the RAW (pre-alias) upstream_refs identities when
    # building per-edge relationship lineage, keeping the precise triple key in
    # sync with the already-canonicalized edge keys. Empty in the common case.
    identity_aliases: dict[LogicalIdentity, LogicalIdentity] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DTOs for derive_rules (spec §3.8) — canonical location
# ---------------------------------------------------------------------------


@dataclass
class ChunkForDerivation:
    """DTO used by derive_structural_edges. Distinct from the SQLAlchemy
    TextChunk ORM model — carries only the fields derivation needs.
    Constructed by the worker from TextChunk rows before calling
    derive_rules.derive_structural_edges."""
    rid: str              # ArcadeDB vertex RID of this chunk
    text_normalized: str  # lowercased, whitespace-collapsed text


@dataclass
class DerivedEdge:
    """Output of derive_structural_edges. Uses RID-based endpoints because
    both source (extracted entity) and target (Document/TextChunk) RIDs are
    already known at derivation time."""
    from_id: str           # extracted entity RID (from identity_to_rid)
    to_id: str             # Document or TextChunk RID
    rel_type: str
    confidence: float | None


# ---------------------------------------------------------------------------
# build_display_label (spec §3.9)
# ---------------------------------------------------------------------------

_NAME_LIKE_KEYS = ("system_name", "name", "title", "section_number", "document_id")


def build_display_label(
    entity_type: str,
    identity_values: dict[str, Any],
    properties: dict[str, Any],
) -> str:
    """Spec §3.9 resolution order:
    1. First 'name-like' key in identity_values with a truthy value.
    2. Concatenation of non-empty identity_values joined by ' / '.
    3. First name-like key in properties with a truthy value.
    4. Deterministic fallback: '{entity_type}_{short-hash-of-identity-tuple}'.
    """
    for key in _NAME_LIKE_KEYS:
        v = identity_values.get(key)
        if v:
            return str(v)

    non_empty = [str(v) for v in identity_values.values() if v]
    if non_empty:
        return " / ".join(non_empty)

    for key in _NAME_LIKE_KEYS:
        v = properties.get(key)
        if v:
            return str(v)

    identity_hash = hashlib.sha1(
        json.dumps(identity_values, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{entity_type}_{identity_hash}"


# ---------------------------------------------------------------------------
# Yield classification (spec §6.2)
# ---------------------------------------------------------------------------


def classify_yield_from_counts(
    *,
    primary: int,
    bridge: int,
    extracted_rels: int,
    rejected_rels: int,
) -> YieldStatus:
    """Precedence (top wins):
    - DEGRADED when total_rels >= 4 AND rejected/total >= 0.75
    - EMPTY when primary == 0 and bridge == 0 and extracted_rels == 0
    - BRIDGES_ONLY when primary == 0 and bridge > 0
    - HIT otherwise
    """
    total_rels = extracted_rels + rejected_rels
    if total_rels >= 4 and rejected_rels / total_rels >= 0.75:
        return YieldStatus.DEGRADED
    if primary == 0 and bridge == 0 and extracted_rels == 0:
        return YieldStatus.EMPTY
    if primary == 0 and bridge > 0:
        return YieldStatus.BRIDGES_ONLY
    return YieldStatus.HIT


def classify_yield(
    result: PassResult,
    pass_def: Any,
    ontology: dict,
) -> YieldStatus:
    """Pre-merge yield classifier (plan Task 35c).

    Counts come from the shared ``PreMergeWalkSummary`` that the pass
    loop built (Task 34b): entities via ``iter_entities_of_type`` —
    which consumes ``pre_merge_walk.entities`` through
    ``_cached_entities`` — and ``extracted_rels`` from
    ``raw_edge_count`` directly. The walker never runs again here.

    Post-merge ``_apply_post_merge_yield_updates`` is the authoritative
    source for ``relationships_rejected``; at pre-merge we FORCE the
    rejected count to 0 so yield classification mirrors raw walker
    emissions without counting merge-time validation failures that
    haven't happened yet. Fallback for test-built ``PassResult``s
    without ``pre_merge_walk``: ``extracted_rels`` falls back to
    ``len(result.relationships)`` — the legacy DTO-list count.
    """
    primary_types = getattr(pass_def, "primary_entity_types", []) or []
    bridge_types = getattr(pass_def, "bridge_entity_types", []) or []

    primary = sum(
        len(list(result.iter_entities_of_type(t))) for t in primary_types
    )
    bridge = sum(
        len(list(result.iter_entities_of_type(t))) for t in bridge_types
    )
    if result.pre_merge_walk is not None:
        extracted_rels = result.pre_merge_walk.raw_edge_count
    else:
        extracted_rels = len(result.relationships)
    rejected_rels = 0  # Forced: post-merge path is authoritative.

    return classify_yield_from_counts(
        primary=primary,
        bridge=bridge,
        extracted_rels=extracted_rels,
        rejected_rels=rejected_rels,
    )


# ---------------------------------------------------------------------------
# Private helpers for merge_and_resolve
# ---------------------------------------------------------------------------


def _build_logical_identity(
    entity_type: str,
    entity_instance: Any,
    ontology: dict,
    document_id: str,
) -> LogicalIdentity | None:
    """Construct a LogicalIdentity from an instance + ontology entity def.
    Returns None if the entity_type isn't in the ontology."""
    # Plan A0-1: content-based identity for is_entity=False components.
    # Per docs:17235 "All fields are used for deduplication": the identity
    # tuple is every declared field in declaration order, including None
    # values, with lists canonicalized to tuples so the frozen dataclass
    # remains hashable. This branch runs before the ontology lookup because
    # components are not (and should not be) present in
    # ontology["entity_types"].
    cfg = getattr(entity_instance, "model_config", {}) or {}
    if cfg.get("is_entity") is False:
        model_fields = getattr(type(entity_instance), "model_fields", {}) or {}
        field_names = tuple(model_fields.keys())
        values: list[Any] = []
        for fname in field_names:
            raw = getattr(entity_instance, fname, None)
            if isinstance(raw, list):
                values.append(tuple(raw))
            else:
                values.append(raw)
        scope = cfg.get("identity_scope", "document")
        return LogicalIdentity(
            entity_type=entity_type,
            identity_field_names=field_names,
            identity_tuple=tuple(values),
            scope=scope,
            document_id=document_id if scope == "document" else None,
        )

    entity_def = next(
        (e for e in ontology.get("entity_types", []) if e["name"] == entity_type),
        None,
    )
    if entity_def is None:
        return None
    identity_fields = tuple(entity_def.get("identity_fields") or ())
    scope = entity_def.get("identity_scope", "document")
    identity_values = tuple(
        getattr(entity_instance, name, None) for name in identity_fields
    )
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=identity_fields,
        identity_tuple=identity_values,
        scope=scope,
        document_id=document_id if scope == "document" else None,
    )


def logical_identity_from_dict(
    entity_type: str,
    identity_dict: dict,
    ontology: dict,
    document_id: str,
) -> LogicalIdentity | None:
    """Build a LogicalIdentity from a raw identity dict.

    This is the canonical way the worker converts an upstream entity ref's
    ``identity_values`` into a ``LogicalIdentity`` suitable for
    ``PassResult.upstream_refs``. The merge resolver compares these
    objects by value (``@dataclass(frozen=True)``) against the merged
    entity index, so the identity tuple must come straight from the
    ontology's ``identity_fields`` list in declared order.

    Returns None if the entity_type is unknown or the payload is missing
    a required identity key — in that case the caller should drop the ref.
    """
    entity_def = next(
        (e for e in ontology.get("entity_types", []) if e["name"] == entity_type),
        None,
    )
    if entity_def is None:
        return None
    identity_fields = tuple(entity_def.get("identity_fields") or ())
    scope = entity_def.get("identity_scope", "document")
    if not all(f in identity_dict for f in identity_fields):
        return None  # payload missing required keys
    identity_values = tuple(identity_dict[f] for f in identity_fields)
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=identity_fields,
        identity_tuple=identity_values,
        scope=scope,
        document_id=document_id if scope == "document" else None,
    )


def _snapshot_entity_identities(
    pass_results: dict[str, "PassResult"],
    ontology: dict,
    document_id: str,
) -> dict[int, LogicalIdentity]:
    """Capture each live entity instance's current LogicalIdentity.

    ``system_links`` stores ref-id endpoints before merge-time identity
    canonicalization runs. Capturing object-id keyed identities before and
    after canonicalization lets the resolver translate those stable ref IDs
    onto the final entity_index identities.
    """
    snapshot: dict[int, LogicalIdentity] = {}
    for entity_def in ontology.get("entity_types", []):
        entity_type = entity_def["name"]
        for pass_name, pass_result in pass_results.items():
            try:
                instances = list(pass_result.iter_entities_of_type(entity_type))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "identity snapshot failed for pass=%s entity_type=%s: %s",
                    pass_name, entity_type, exc,
                )
                continue
            for instance in instances:
                identity = _build_logical_identity(
                    entity_type, instance, ontology, document_id,
                )
                if identity is not None:
                    snapshot[id(instance)] = identity
    return snapshot


def _build_identity_aliases(
    before: dict[int, LogicalIdentity],
    after: dict[int, LogicalIdentity],
) -> dict[LogicalIdentity, LogicalIdentity]:
    """Return ``old_identity -> final_identity`` for rewritten instances."""
    aliases: dict[LogicalIdentity, LogicalIdentity] = {}
    for object_id, old_identity in before.items():
        new_identity = after.get(object_id)
        if new_identity is not None and new_identity != old_identity:
            aliases[old_identity] = new_identity
    return aliases


def _resolve_identity_alias(
    identity: LogicalIdentity,
    identity_aliases: dict[LogicalIdentity, LogicalIdentity] | None,
) -> LogicalIdentity:
    """Map a pre-canonicalization identity onto its final merge identity."""
    if not identity_aliases:
        return identity
    current = identity
    seen: set[LogicalIdentity] = set()
    while current in identity_aliases and current not in seen:
        seen.add(current)
        current = identity_aliases[current]
    return current


def _to_merged_entity_record(
    model: BaseModel,
    ontology: dict,
    document_id: str,
    pass_origin: str = "document_anchors",
) -> MergedEntityRecord:
    """Convert a Pydantic entity model → MergedEntityRecord (spec §3.4 + §8.2).

    Used by walker-sourced passes (e.g. the Docling anchor walker, Chunk D)
    that emit Pydantic models directly instead of going through the
    PassResult / ontology-lookup path. Identity is derived from the
    model's ``model_config['graph_id_fields']`` (or, for is_entity=False
    components, delegated to ``_build_logical_identity`` so the A0-1
    content-based branch runs). The caller's ``document_id`` is stamped
    onto ``properties`` so downstream TextChunk joins work regardless of
    whether the model declares its own ``document_id`` field.

    ``ontology`` is accepted for signature parity with
    ``_build_logical_identity`` and is only consulted by the component
    branch; passing ``{}`` is normal for walker-sourced entities.
    """
    cfg = getattr(model, "model_config", {}) or {}
    entity_type = cfg["ontology_name"]

    if cfg.get("is_entity") is False:
        identity = _build_logical_identity(entity_type, model, ontology, document_id)
    else:
        id_fields = tuple(cfg.get("graph_id_fields") or ())
        scope = cfg.get("identity_scope", "document")
        identity_tuple = tuple(getattr(model, f, None) for f in id_fields)
        identity = LogicalIdentity(
            entity_type=entity_type,
            identity_field_names=id_fields,
            identity_tuple=identity_tuple,
            scope=scope,
            document_id=document_id if scope == "document" else None,
        )

    id_field_set = set(cfg.get("graph_id_fields") or ())
    dumped = model.model_dump(mode="json")
    properties: dict[str, Any] = {}
    for fname, value in dumped.items():
        if fname in id_field_set:
            continue
        finfo = type(model).model_fields.get(fname)
        extra = finfo.json_schema_extra if finfo else None
        if isinstance(extra, dict) and extra.get("edge_label"):
            continue
        properties[fname] = value
    # Stamp document_id unconditionally: SectionEntity / DocumentEntity etc.
    # declare document_id as Optional[str]=None, so model_dump yields None
    # when the caller did not populate it. Walker-sourced passes never
    # populate document_id on the model itself; the caller's argument is
    # authoritative.
    if properties.get("document_id") is None:
        properties["document_id"] = document_id

    identity_values_dict = dict(
        zip(identity.identity_field_names, identity.identity_tuple)
    )
    display_label = build_display_label(entity_type, identity_values_dict, properties)

    return MergedEntityRecord(
        identity=identity,
        properties=properties,
        confidence=1.0,
        pass_origins={pass_origin},
        display_label=display_label,
    )


# ---------------------------------------------------------------------------
# Unified entity-graph walker (plan Task 35a/35b)
# ---------------------------------------------------------------------------


def walk_entity_graph(
    node: Any,
    on_entity: Callable[[Any], None],
    *,
    ontology: dict | None = None,
    document_id: str | None = None,
    on_edge: Callable[[LogicalIdentity, str, Any], None] | None = None,
    visited_objects: set[int] | None = None,
    at_pass_root: bool = True,
) -> None:
    """Walk the typed-edge entity graph rooted at ``node``.

    Single unified walker with two modes gated by the ``on_edge`` callback.
    When ``on_edge is None`` the walker runs in entity-only mode: it emits
    every reachable entity via ``on_entity`` but skips edge-identity
    construction, so ``ontology`` / ``document_id`` may be ``None``
    (``PassResult.iter_entities_of_type`` fallback). When ``on_edge`` is
    provided both ``ontology`` and ``document_id`` are required — the walker
    builds ``LogicalIdentity`` for the parent of each emitted edge.

    Traversal rules (graph-only, per docs):
    - ``at_pass_root=True`` (only the initial pass-root container at top-level):
      walk plain ``list`` / scalar ``BaseModel`` fields to reach top-level
      entities. The pass-root container is NOT emitted as an entity.
      Children are entered with ``at_pass_root=False``.
    - Entity nodes (``is_entity=True``): emit via ``on_entity``; then follow
      ONLY fields marked with ``json_schema_extra.edge_label``. Components
      (``is_entity=False``) reached via ``edge_label`` are emitted as
      first-class graph nodes via ``on_entity`` with an edge from the parent
      (spec §4.8 step 2 + docs:17500-17509 — e.g. a shared Address node);
      the walker does NOT recurse into such components (A0-5 contract test
      forbids components from carrying edge_label fields).
    - Component nodes (``is_entity=False``) encountered inside the graph
      (not at pass-root): treat as embedded data. Do NOT recurse, do NOT
      emit. Value objects live in their parent entity's properties, not as
      graph endpoints.
    - Plain nested ``BaseModel`` entity fields without ``edge_label``:
      embedded data, not graph-relevant. Do NOT recurse.
    """
    full_mode = on_edge is not None
    if full_mode:
        if ontology is None or document_id is None:
            raise ValueError(
                "walk_entity_graph: on_edge requires ontology and document_id "
                "(full mode builds LogicalIdentity for edge parents)."
            )
    if visited_objects is None:
        visited_objects = set()

    if id(node) in visited_objects:
        return
    visited_objects.add(id(node))

    cfg = getattr(node, "model_config", {}) or {}

    node_cls = type(node)
    model_fields = getattr(node_cls, "model_fields", {}) if isinstance(node, BaseModel) else {}

    if at_pass_root:
        # Pass-root container: walk plain fields to reach top-level entities.
        # Do NOT emit as entity. Children are entered with at_pass_root=False.
        for fname in model_fields:
            value = getattr(node, fname, None)
            if value is None:
                continue
            items = value if isinstance(value, list) else [value]
            for child in items:
                if isinstance(child, BaseModel):
                    walk_entity_graph(
                        child,
                        on_entity,
                        ontology=ontology,
                        document_id=document_id,
                        on_edge=on_edge,
                        visited_objects=visited_objects,
                        at_pass_root=False,
                    )
        return

    if cfg.get("is_entity") is False:
        # Component encountered inside the entity graph → embedded data.
        # Do NOT emit, do NOT recurse.
        return

    # Entity node — emit, then follow edge_label fields only.
    on_entity(node)

    parent_identity: LogicalIdentity | None = None
    if full_mode:
        entity_type = cfg.get("ontology_name")
        if entity_type is not None:
            parent_identity = _build_logical_identity(
                entity_type, node, ontology or {}, document_id or "",
            )

    for fname, finfo in model_fields.items():
        extra = finfo.json_schema_extra or {}
        edge_label = extra.get("edge_label") if isinstance(extra, dict) else None
        if not edge_label:
            continue  # Non-edge field; embedded data or scalar, not graph-relevant.
        value = getattr(node, fname, None)
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for child in items:
            if not isinstance(child, BaseModel):
                continue
            child_cfg = getattr(child, "model_config", {}) or {}
            if child_cfg.get("is_entity") is not True:
                # Per spec §4.8 + docs:17500-17509: components reached via
                # edge(label=...) become first-class graph nodes with
                # content-based identity (A0-1). Do NOT recurse — components
                # cannot carry edge_label fields (enforced by A0-5 contract
                # test at schema-validation time).
                on_entity(child)
                if full_mode and on_edge is not None and parent_identity is not None:
                    on_edge(parent_identity, edge_label, child)
                continue
            if full_mode and on_edge is not None and parent_identity is not None:
                on_edge(parent_identity, edge_label, child)
            walk_entity_graph(
                child,
                on_entity,
                ontology=ontology,
                document_id=document_id,
                on_edge=on_edge,
                visited_objects=visited_objects,
                at_pass_root=False,
            )


def _is_valid_triple(
    ontology: dict,
    from_type: str,
    rel_type: str,
    to_type: str,
) -> bool:
    for row in ontology.get("validation_matrix", []):
        if (
            row.get("source") == from_type
            and row.get("relationship") == rel_type
            and row.get("target") == to_type
        ):
            return True
    return False


def _resolve_relationship(
    rel: Any,
    pass_name: str,
    pass_result: PassResult,
    entity_index: dict[LogicalIdentity, MergedEntityRecord],
    ontology: dict,
    document_id: str,
    identity_aliases: dict[LogicalIdentity, LogicalIdentity] | None = None,
) -> MergedEdgeRecord | RelationshipRejectionReason:
    """Return either a MergedEdgeRecord or a rejection reason.

    Rejection ordering (spec §6.7):
    MISSING_REL_TYPE → INVALID_IDENTITY_PAYLOAD → UNKNOWN_REF_ID
    → FROM_ENDPOINT_NOT_FOUND → TO_ENDPOINT_NOT_FOUND → INVALID_TRIPLE
    """
    rel_type = getattr(rel, "rel_type", None)
    if not rel_type:
        return RelationshipRejectionReason.MISSING_REL_TYPE

    # system_links-style: ref_id-based cross-pass resolution
    from_ref_id = getattr(rel, "from_ref_id", None)
    to_ref_id = getattr(rel, "to_ref_id", None)
    if from_ref_id is not None or to_ref_id is not None:
        upstream_refs = getattr(pass_result, "upstream_refs", None) or {}
        if from_ref_id not in upstream_refs or to_ref_id not in upstream_refs:
            return RelationshipRejectionReason.UNKNOWN_REF_ID
        from_identity = upstream_refs[from_ref_id]
        to_identity = upstream_refs[to_ref_id]
    else:
        # Same-pass identity-dict lookup
        from_type = getattr(rel, "from_type", None)
        to_type = getattr(rel, "to_type", None)
        from_identity_dict = getattr(rel, "from_identity", None)
        to_identity_dict = getattr(rel, "to_identity", None)

        if not isinstance(from_identity_dict, dict) or not isinstance(to_identity_dict, dict):
            return RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD

        from_identity = logical_identity_from_dict(
            from_type, from_identity_dict, ontology, document_id
        )
        to_identity = logical_identity_from_dict(
            to_type, to_identity_dict, ontology, document_id
        )

        if from_identity is None or to_identity is None:
            return RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD

    from_identity = _resolve_identity_alias(from_identity, identity_aliases)
    to_identity = _resolve_identity_alias(to_identity, identity_aliases)

    if from_identity not in entity_index:
        return RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND
    if to_identity not in entity_index:
        return RelationshipRejectionReason.TO_ENDPOINT_NOT_FOUND

    if not _is_valid_triple(
        ontology,
        from_identity.entity_type,
        rel_type,
        to_identity.entity_type,
    ):
        return RelationshipRejectionReason.INVALID_TRIPLE

    raw_conf = getattr(rel, "confidence", None)
    # IMPORTANT: use explicit None check — not `raw_conf or 0.8`
    # so that explicit 0.0 is preserved (regression guard).
    confidence = 0.8 if raw_conf is None else raw_conf

    return MergedEdgeRecord(
        from_identity=from_identity,
        to_identity=to_identity,
        rel_type=rel_type,
        confidence=confidence,
        pass_origins={pass_name},
    )


# ---------------------------------------------------------------------------
# Cross-pass identity canonicalization (Path B fix for cross-pass system_name
# drift).
#
# Field-group passes are run independently. For the same physical entity,
# two passes can emit different `system_name` values (e.g. one pass picks
# "MIM-104F", another picks "PAC-3" from the same prose). The merge phase
# below keys on `LogicalIdentity` (built from `graph_id_fields`), so distinct
# `system_name` values produce distinct vertices even when they refer to the
# same physical missile.
#
# This pass runs BEFORE merge_and_resolve's entity loop, detects entities
# whose system_name and other identity-side fields strongly imply they refer
# to the same physical entity, and mutates the instances to share a single
# canonical `system_name`. After mutation, the existing identity-keyed merge
# collapses them onto one vertex automatically.
# ---------------------------------------------------------------------------


_CANONICALIZE_MIN_TOKEN_LEN = 2  # tokens shorter than this don't count for matching


def _identity_token_bag(model: Any, max_fields: int = 3) -> set[str]:
    """Tokenize system_name + nomenclature + name (and similar identity-side
    fields) into a UPPERCASE token set, dropping single-char tokens.
    Used for deciding whether two entity instances refer to the same physical
    entity."""
    import re

    parts: list[str] = []
    for fname in ("system_name", "nomenclature", "name"):
        v = getattr(model, fname, None)
        if isinstance(v, str) and v:
            parts.append(v)
    if not parts:
        return set()
    text = " ".join(parts)
    return {
        t.upper()
        for t in re.split(r"[^A-Za-z0-9]+", text)
        if len(t) >= _CANONICALIZE_MIN_TOKEN_LEN
    }


def _likely_same_entity(a: Any, b: Any) -> bool:
    """Return True iff `a` and `b` plausibly refer to the same physical entity.

    Decision rule:
      - Same `system_name` → trivially same.
      - Otherwise: tokenize a's `system_name` (the bare canonical token), and
        check whether ALL of those tokens appear in b's combined identity
        token bag (system_name + nomenclature + name). Symmetric.

    This catches the cross-pass drift case where:
      • missile_identity emits system_name='MIM-104F' AND nomenclature='MIM-104F
        Patriot Advanced Capability-3 (PAC-3)' AND name='Patriot'
      • missile_airframe emits system_name='PAC-3' (no nomenclature/name)
    The 'PAC-3' system_name's tokens {PAC} are a subset of MIM-104F's combined
    token bag {MIM, 104F, PATRIOT, ADVANCED, CAPABILITY, PAC} → match.

    It also correctly REJECTS unrelated systems: 'PAC-3' tokens {PAC} are not
    a subset of 'SM-6' / 'RIM-174 Standard Missile 6 (SM-6) Block IA'.
    """
    a_name = getattr(a, "system_name", None)
    b_name = getattr(b, "system_name", None)
    if not isinstance(a_name, str) or not isinstance(b_name, str):
        return False
    if a_name == b_name:
        return True

    import re
    a_only_tokens = {
        t.upper()
        for t in re.split(r"[^A-Za-z0-9]+", a_name)
        if len(t) >= _CANONICALIZE_MIN_TOKEN_LEN
    }
    b_only_tokens = {
        t.upper()
        for t in re.split(r"[^A-Za-z0-9]+", b_name)
        if len(t) >= _CANONICALIZE_MIN_TOKEN_LEN
    }
    if not a_only_tokens or not b_only_tokens:
        return False

    a_full = _identity_token_bag(a)
    b_full = _identity_token_bag(b)

    return a_only_tokens.issubset(b_full) or b_only_tokens.issubset(a_full)


def _pick_canonical_name(names: Iterable[str]) -> str:
    """Pick the canonical system_name from a set of variants.
    Rule: shortest wins; ties broken by lex order. Empty set → empty string."""
    candidates = sorted(set(n for n in names if isinstance(n, str) and n),
                        key=lambda n: (len(n), n))
    return candidates[0] if candidates else ""


def canonicalize_cross_pass_identities(
    pass_results: dict[str, "PassResult"],
    ontology: dict,
    *,
    table_alias_map_by_entity_type: dict[str, dict[str, str]] | None = None,
) -> int:
    """Mutate entity instances in-place so cross-pass duplicates share a
    single canonical `system_name`.

    Returns the number of entity instances whose system_name was changed
    (useful for logging / metrics).

    Only acts on entity types whose graph_id_fields is exactly ['system_name'].
    Other identity schemes (multi-field, content-hashed, etc.) are left alone.

    When ``table_alias_map_by_entity_type`` is non-None, runs Mechanism A1
    Phase 0 (table-overlay alias rewrite) BEFORE the existing token-overlap
    pass. Spec §5.3.
    """
    rewrites = 0

    # Mechanism A1 alias-map rewrite (Phase 0). Runs BEFORE the existing
    # token-overlap pass so its rewrites land first.
    if table_alias_map_by_entity_type:
        try:
            from app.services.table_overlay import apply_identity_rewrite
            stats = apply_identity_rewrite(
                pass_results, table_alias_map_by_entity_type, ontology,
            )
            rewrites += stats.rewrites
            logger.info(
                "IDENTITY_REWRITE rewrites=%d unique_canonicals=%d "
                "passes_touched=%d",
                stats.rewrites, stats.unique_canonicals, stats.passes_touched,
            )
        except Exception as exc:
            logger.warning(
                "apply_identity_rewrite failed: %s — falling through to "
                "existing token-overlap canonicalization", exc,
            )

    for entity_def in ontology.get("entity_types", []):
        entity_type = entity_def["name"]

        # Gather all instances of this type, with provenance back to (pass, model).
        all_instances: list[Any] = []
        for pass_name, pass_result in pass_results.items():
            for inst in pass_result.iter_entities_of_type(entity_type):
                cfg = getattr(inst, "model_config", {}) or {}
                id_fields = tuple(cfg.get("graph_id_fields") or ())
                if id_fields != ("system_name",):
                    continue
                all_instances.append(inst)

        if len(all_instances) <= 1:
            continue

        # Union-Find on similarity links.
        n = len(all_instances)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                if _likely_same_entity(all_instances[i], all_instances[j]):
                    union(i, j)

        # Group by component.
        from collections import defaultdict
        components: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            components[find(i)].append(i)

        # For each multi-member component, pick canonical and rewrite.
        for member_idxs in components.values():
            if len(member_idxs) <= 1:
                continue
            names = [getattr(all_instances[i], "system_name", "") for i in member_idxs]
            canonical = _pick_canonical_name(names)
            if not canonical:
                continue
            for i in member_idxs:
                inst = all_instances[i]
                if getattr(inst, "system_name", None) != canonical:
                    try:
                        inst.system_name = canonical
                        rewrites += 1
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "canonicalize: failed to mutate system_name on "
                            "%s instance: %s", entity_type, exc,
                        )

    if rewrites:
        logger.info(
            "canonicalize_cross_pass_identities: rewrote %d system_name(s) "
            "to collapse cross-pass duplicates",
            rewrites,
        )
    return rewrites


# ---------------------------------------------------------------------------
# merge_and_resolve (spec §3.7)
# ---------------------------------------------------------------------------


# NOTE: _extract_doc_overlay used to live here; promoted to
# app.services.table_overlay.extract_doc_overlay (and called via the
# top-level apply_table_overlay_phases orchestrator below) so the
# Mechanism A1 worker-side surface is one self-contained module. This
# stub is kept as a doc-comment so future-you grepping for the old name
# lands at the new home.


def merge_and_resolve(
    pass_results: dict[str, PassResult],
    manifest: Any,  # BundleManifest or compatible stub
    ontology: dict,
    document_id: str,
    pipeline_run_id: str,
) -> MergedExtraction:
    """Phase 1: merge entities; Phase 2: resolve edges against logical identity.

    Key properties (spec §3.7):
    1. Entities keyed by LogicalIdentity. Bridge entities with identical
       identity across passes collapse into one MergedEntityRecord with
       both pass names in pass_origins.
    2. Relationships resolved post-merge by LogicalIdentity lookup.
    3. Same-pass edges use identity-dict lookup; cross-pass edges use
       ref_id lookup against an upstream ref set (system_links).
    4. Rejections counted per pass and per reason.
    5. confidence = 0.8 if rel.confidence is None. Explicit 0.0 preserved.
    """
    # ----- Mechanism A1 (Phase 0 + Phase 0.5): table overlay -----
    # Field-group passes may emit different `system_name` values for the
    # same physical entity (e.g. missile_identity emits 'MIM-104F' while
    # missile_airframe emits 'PAC-3' from the same prose). Mutate
    # instances so cross-pass duplicates share a canonical system_name
    # before LogicalIdentity is built. See
    # canonicalize_cross_pass_identities for the matching heuristic.
    #
    # When a doc-level table overlay was extracted by docling-graph
    # (spec §5.4-5.5), it carries (a) an alias_map_by_entity_type used
    # by Phase 0 to deterministically collapse table aliases BEFORE the
    # token-overlap heuristic, and (b) a list of TableFacts applied in
    # Phase 0.5 to overwrite per-cell field values where the table
    # disagrees with the LLM extraction. The worker-side kill switch
    # (DOCLING_GRAPH_TABLE_OVERLAY_ENABLED) is authoritative even over
    # cached overlay payloads loaded from prior runs (spec §4.3).
    #
    # The orchestrator encapsulates: env-flag read, kill-switch
    # logging, extract_doc_overlay, Phase 0 (delegated back to
    # canonicalize_cross_pass_identities), and Phase 0.5
    # (apply_field_overlay). When future refactors split this merge
    # function into a different dispatcher (e.g., the
    # derive_ontology_graph_merge of feat/per-pass-celery-fanin), the
    # orchestrator call below should be threaded into that dispatcher
    # at the same point — between identity canonicalization and entity
    # merge. Spec §5.5.
    pre_canonical_identity_snapshot = _snapshot_entity_identities(
        pass_results, ontology, document_id,
    )
    from app.services.table_overlay import apply_table_overlay_phases
    apply_table_overlay_phases(
        pass_results,
        ontology=ontology,
        document_id=document_id,
        canonicalize_fn=canonicalize_cross_pass_identities,
    )
    post_canonical_identity_snapshot = _snapshot_entity_identities(
        pass_results, ontology, document_id,
    )
    identity_aliases = _build_identity_aliases(
        pre_canonical_identity_snapshot, post_canonical_identity_snapshot,
    )
    if identity_aliases:
        logger.info(
            "merge_and_resolve: identity_aliases=%d from canonicalization",
            len(identity_aliases),
        )
    # ---------- end Mechanism A1 ----------

    # --- Pass 1: merge entities ---
    entity_index: dict[LogicalIdentity, MergedEntityRecord] = {}

    for pass_name, pass_result in pass_results.items():
        for entity_def in ontology.get("entity_types", []):
            entity_type = entity_def["name"]
            for instance in pass_result.iter_entities_of_type(entity_type):
                identity = _build_logical_identity(
                    entity_type, instance, ontology, document_id,
                )
                if identity is None:
                    continue

                # Extract properties from the instance.
                # entity_def["properties"] is a JSON-Schema-style dict
                # ({"type":"object","properties":{<field>:{<schema>}, ...}})
                # — NOT a flat list of property names. Pull the nested
                # property keys; if the ontology ever ships a flat
                # list[str] (older shape), fall through to that.
                _props_node = entity_def.get("properties", [])
                if isinstance(_props_node, dict):
                    _prop_names = list(_props_node.get("properties", {}).keys())
                else:
                    _prop_names = list(_props_node)
                props = {
                    p: getattr(instance, p, None)
                    for p in _prop_names
                }
                props = {k: v for k, v in props.items() if v is not None}

                raw_conf = getattr(instance, "confidence", None)
                confidence = 1.0 if raw_conf is None else raw_conf

                existing = entity_index.get(identity)
                if existing is None:
                    entity_index[identity] = MergedEntityRecord(
                        identity=identity,
                        properties=props,
                        confidence=confidence,
                        pass_origins={pass_name},
                        display_label=build_display_label(
                            entity_type,
                            identity.identity_values_dict(),
                            props,
                        ),
                    )
                else:
                    # Merge: update non-None props, max confidence, add pass
                    for k, v in props.items():
                        if v is not None:
                            existing.properties[k] = v
                    existing.confidence = max(existing.confidence, confidence)
                    existing.pass_origins.add(pass_name)

    # --- Phase 1.5: provenance aggregation (plan Task 52a) ---
    #
    # For each pass's ExtractionProvenance rows, resolve the (entity_type,
    # identity_values) dict to a LogicalIdentity via
    # logical_identity_from_dict — the same helper upstream-refs uses, so
    # document_id / scope / key order all stay in lockstep with the
    # entity-merge path. Rows that can't normalize (missing identity key,
    # unknown entity type) are dropped with a WARNING distinguishing the
    # "malformed" reason from the "post-merge-absent" reason for
    # observability.
    #
    # Dedup key is (instance_id, element_uid): a single logical instance
    # can legitimately reference multiple source elements (different
    # element_uid rows sharing the same instance_id are retained); pure
    # echoes with identical pairs collapse.
    for pass_name, pass_result in pass_results.items():
        seen_keys: set[tuple[str, str]] = set()
        for prov in pass_result.provenance:
            identity = logical_identity_from_dict(
                prov.ontology_name,
                prov.identity_values,
                ontology,
                document_id,
            )
            if identity is None:
                logger.warning(
                    "drop malformed provenance instance_id=%s type=%s pass=%s",
                    prov.instance_id, prov.ontology_name, pass_name,
                )
                continue
            identity = _resolve_identity_alias(identity, identity_aliases)
            record = entity_index.get(identity)
            if record is None:
                logger.warning(
                    "drop post-merge-absent provenance instance_id=%s type=%s pass=%s",
                    prov.instance_id, prov.ontology_name, pass_name,
                )
                continue
            key = (prov.instance_id, prov.element_uid)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            record.provenance.append(prov)

    # --- Phase 1.6: per-field evidence aggregation (Phase 3 task 32) ---
    #
    # PassResult.field_evidence is keyed by instance_id (outer) and
    # field_name (inner). Walk each instance's provenance rows to find
    # the matching record (same instance_id appears in record.provenance),
    # and append the FieldEvidenceRow under the canonical field_name.
    # Cross-pass dedup by (chunk_id, snippet, element_uid).
    for pass_name, pass_result in pass_results.items():
        if not pass_result.field_evidence:
            continue
        # Build instance_id → record index from the provenance rows we
        # just aggregated; same logical record can have many instance_ids
        # across passes, so this is a multi-map.
        instance_to_record: dict[str, MergedEntityRecord] = {}
        for record in entity_index.values():
            for prov in record.provenance:
                instance_to_record[prov.instance_id] = record
        for instance_id, fields_dict in pass_result.field_evidence.items():
            record = instance_to_record.get(instance_id)
            if record is None:
                logger.warning(
                    "drop post-merge-absent field_evidence instance_id=%s pass=%s",
                    instance_id, pass_name,
                )
                continue
            for field_name, rows in fields_dict.items():
                bucket = record.field_evidence.setdefault(field_name, [])
                seen_pairs = {
                    (r.chunk_id, r.snippet, r.element_uid) for r in bucket
                }
                for row in rows:
                    key = (row.chunk_id, row.snippet, row.element_uid)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    bucket.append(row)

    # --- Pass 2: resolve relationships (plan Task 36) ---
    #
    # Two edge-producing paths, run ADDITIVELY per pass, feeding a uniform
    # per-pass accounting:
    #   * DTO path: always consumes ``pass_result.relationships`` via
    #     ``_resolve_relationship`` — handles ``system_links`` in
    #     production AND any test fixture that puts DTO rels on a
    #     SimpleNamespace template. Same VALIDATION_MATRIX and
    #     rejection-reason semantics as before.
    #   * Walker path: fires when ``template_instance`` is a Pydantic
    #     ``BaseModel`` (typed-edge templates, post-Phase-5). Each
    #     ``on_edge`` emission is a raw attempted edge; triple-check +
    #     endpoint lookup decide accepted vs rejected. Fresh
    #     ``visited_objects`` set per pass (reviewer finding #1).
    #
    # In production the two paths don't overlap because typed-edge
    # passes carry no DTOs post-Task 64 and system_links has no
    # walker-reachable edges — so both branches contribute to the same
    # per-pass counters without double-counting.
    edges: list[MergedEdgeRecord] = []
    rejected_edges: list[tuple[str, Any, RelationshipRejectionReason]] = []
    rejections_by_pass: dict[str, int] = {}
    per_pass_edge_metrics: dict[str, PerPassEdgeMetrics] = {}

    MAX_REJECTION_SAMPLE = 20

    for pass_name, pass_result in pass_results.items():
        attempted = 0
        accepted_count = 0
        rejection_sample: list[dict] = []
        rejections_by_reason: dict[str, int] = {}

        def _record_rejection(raw_rel: Any, reason: RelationshipRejectionReason):
            rejected_edges.append((pass_name, raw_rel, reason))
            rejections_by_pass[pass_name] = rejections_by_pass.get(pass_name, 0) + 1
            key = reason.value if hasattr(reason, "value") else str(reason)
            rejections_by_reason[key] = rejections_by_reason.get(key, 0) + 1
            if len(rejection_sample) < MAX_REJECTION_SAMPLE:
                rejection_sample.append(_edge_to_rejection_dict(raw_rel, reason))

        # --- DTO path (system_links + legacy test fixtures) ---
        for rel in pass_result.relationships:
            attempted += 1
            result = _resolve_relationship(
                rel, pass_name, pass_result,
                entity_index, ontology, document_id, identity_aliases,
            )
            if isinstance(result, MergedEdgeRecord):
                edges.append(result)
                accepted_count += 1
            else:
                _record_rejection(rel, result)

        # --- Walker path (typed-edge BaseModel templates) ---
        if isinstance(pass_result.template_instance, BaseModel):
            raw_edges: list[tuple[LogicalIdentity, str, Any]] = []

            def _collect_edge(parent_identity, label, child):
                raw_edges.append((parent_identity, label, child))

            walk_entity_graph(
                pass_result.template_instance,
                on_entity=lambda _e: None,
                ontology=ontology,
                document_id=document_id,
                on_edge=_collect_edge,
                visited_objects=set(),
                at_pass_root=True,
            )

            for parent_identity, label, child in raw_edges:
                attempted += 1
                child_cfg = getattr(child, "model_config", {}) or {}
                child_type = child_cfg.get("ontology_name")
                if child_type is None:
                    _record_rejection(
                        {"from_type": parent_identity.entity_type,
                         "rel_type": label, "to_type": None},
                        RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD,
                    )
                    continue
                child_identity = _build_logical_identity(
                    child_type, child, ontology, document_id,
                )
                if child_identity is None:
                    _record_rejection(
                        {"from_type": parent_identity.entity_type,
                         "rel_type": label, "to_type": child_type},
                        RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD,
                    )
                    continue
                if not _is_valid_triple(
                    ontology,
                    parent_identity.entity_type, label, child_type,
                ):
                    _record_rejection(
                        {"from_type": parent_identity.entity_type,
                         "rel_type": label, "to_type": child_type},
                        RelationshipRejectionReason.INVALID_TRIPLE,
                    )
                    continue
                if parent_identity not in entity_index:
                    _record_rejection(
                        {"from_type": parent_identity.entity_type,
                         "rel_type": label, "to_type": child_type},
                        RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND,
                    )
                    continue
                if child_identity not in entity_index:
                    _record_rejection(
                        {"from_type": parent_identity.entity_type,
                         "rel_type": label, "to_type": child_type},
                        RelationshipRejectionReason.TO_ENDPOINT_NOT_FOUND,
                    )
                    continue

                raw_conf = getattr(child, "confidence", None)
                confidence = 0.8 if raw_conf is None else raw_conf
                edge_record = MergedEdgeRecord(
                    from_identity=parent_identity,
                    to_identity=child_identity,
                    rel_type=label,
                    confidence=confidence,
                    pass_origins={pass_name},
                )
                edges.append(edge_record)
                accepted_count += 1

        per_pass_edge_metrics[pass_name] = PerPassEdgeMetrics(
            attempted=attempted,
            accepted=accepted_count,
            rejected=attempted - accepted_count,
            rejection_sample=rejection_sample,
            rejections_by_reason=rejections_by_reason,
        )

    # --- Cross-pass edge reducer (plan Step 2a) ---
    # Group edges by (from_logical_identity, rel_type, to_logical_identity).
    # Keep max confidence; union pass_origins across contributing passes.
    edge_dedup: dict[tuple[LogicalIdentity, str, LogicalIdentity], MergedEdgeRecord] = {}
    for e in edges:
        key = (e.from_identity, e.rel_type, e.to_identity)
        existing = edge_dedup.get(key)
        if existing is None:
            edge_dedup[key] = MergedEdgeRecord(
                from_identity=e.from_identity,
                to_identity=e.to_identity,
                rel_type=e.rel_type,
                confidence=e.confidence,
                pass_origins=set(e.pass_origins),
            )
        else:
            existing.confidence = max(existing.confidence, e.confidence)
            existing.pass_origins |= e.pass_origins

    return MergedExtraction(
        entities=list(entity_index.values()),
        edges=list(edge_dedup.values()),
        rejected_edges=rejected_edges,
        rejections_by_pass=rejections_by_pass,
        pipeline_run_id=pipeline_run_id,
        document_id=document_id,
        per_pass_edge_metrics=per_pass_edge_metrics,
        identity_aliases=identity_aliases,
    )


def _pass_kind(manifest: Any, pass_name: str) -> str | None:
    """Helper: return manifest.find_pass(pass_name).kind if available."""
    if manifest is None:
        return None
    try:
        pass_def = manifest.find_pass(pass_name)
        return getattr(pass_def, "kind", None)
    except (KeyError, AttributeError):
        return None


def _edge_to_rejection_dict(raw_rel: Any, reason: RelationshipRejectionReason) -> dict:
    """Serialize a rejected edge (DTO or walker-derived dict) to the
    metrics rejection_sample shape."""
    reason_val = reason.value if hasattr(reason, "value") else str(reason)
    if isinstance(raw_rel, dict):
        return {**raw_rel, "reason": reason_val}
    # DTO-ish object: pull common attributes.
    return {
        "rel_type": getattr(raw_rel, "rel_type", None),
        "from_ref_id": getattr(raw_rel, "from_ref_id", None),
        "to_ref_id": getattr(raw_rel, "to_ref_id", None),
        "reason": reason_val,
    }
