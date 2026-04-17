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


@dataclass(frozen=True)
class LogicalIdentity:
    """Hashable identity for an extracted entity.

    Frozen so instances can be used as dict keys in the merge index.
    Spec §3.6.
    """

    entity_type: str
    identity_field_names: tuple[str, ...]  # ordered from ontology.yaml
    identity_tuple: tuple[Any, ...]        # parallel values
    scope: Literal["document", "global"]
    document_id: str | None               # populated iff scope == "document"

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


# ---------------------------------------------------------------------------
# ExtractionMetadata + PassResult
# ---------------------------------------------------------------------------


@dataclass
class ExtractionMetadata:
    schema_size_chars: int
    structured_output_mode: Literal["strict", "fallback_json"]


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

_NAME_LIKE_KEYS = ("system_name", "name", "title", "heading", "document_id")


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
# merge_and_resolve (spec §3.7)
# ---------------------------------------------------------------------------


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

                # Extract properties from the instance
                props = {
                    p: getattr(instance, p, None)
                    for p in entity_def.get("properties", [])
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
                entity_index, ontology, document_id,
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
