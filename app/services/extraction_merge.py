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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel


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
class PassResult:
    """Handoff type between _run_single_pass (producer) and merge_and_resolve (consumer)."""

    pass_name: str
    template_instance: Any  # instantiated Pydantic template class (or stub in tests)
    metadata: ExtractionMetadata
    pre_merge_rejections: list[tuple[Any, RelationshipRejectionReason]]
    # Optional: pre-populated by system_links pass for cross-pass ref_id lookup.
    upstream_refs: dict[str, "LogicalIdentity"] | None = None

    def iter_entities_of_type(self, entity_type: str) -> Iterable[Any]:
        """Return entity model instances matching the given type.

        Tries multiple attribute name conventions:
        1. '<lower>_list'  — test-fixture convention (e.g. radar_system_list)
        2. '<lower>s'      — plural snake_case (e.g. radar_systems)
        3. '<lower>'       — bare singular fallback

        Returns an empty list if nothing matches.
        """
        lower = entity_type.lower()
        candidates = (
            f"{lower}_list",
            f"{lower}s",
            lower,
        )
        for attr in candidates:
            val = getattr(self.template_instance, attr, None)
            if val is not None and isinstance(val, list):
                return val
        return []

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
    source_pass: str


@dataclass
class MergedExtraction:
    entities: list[MergedEntityRecord]
    edges: list[MergedEdgeRecord]
    rejected_edges: list[tuple[str, Any, RelationshipRejectionReason]]  # (source_pass, raw_rel, reason)
    rejections_by_pass: dict[str, int]
    pipeline_run_id: str
    document_id: str


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
    """Convenience wrapper — extracts counts from PassResult and delegates."""
    primary_types = getattr(pass_def, "primary_entity_types", []) or []
    bridge_types = getattr(pass_def, "bridge_entity_types", []) or []

    primary = sum(
        len(list(result.iter_entities_of_type(t))) for t in primary_types
    )
    bridge = sum(
        len(list(result.iter_entities_of_type(t))) for t in bridge_types
    )
    extracted_rels = len(result.relationships)
    rejected_rels = len(result.pre_merge_rejections)

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


def _identity_from_dict(
    entity_type: str,
    identity_dict: dict,
    ontology: dict,
    document_id: str,
) -> LogicalIdentity | None:
    """Build a LogicalIdentity from a raw identity dict (as emitted on
    a relationship's from_identity / to_identity field).
    Returns None if the entity_type isn't in the ontology or the payload
    is missing required identity keys."""
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

        from_identity = _identity_from_dict(
            from_type, from_identity_dict, ontology, document_id
        )
        to_identity = _identity_from_dict(
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
        source_pass=pass_name,
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

    # --- Pass 2: resolve relationships ---
    edges: list[MergedEdgeRecord] = []
    rejected_edges: list[tuple[str, Any, RelationshipRejectionReason]] = []
    rejections_by_pass: dict[str, int] = {}

    for pass_name, pass_result in pass_results.items():
        for rel in pass_result.relationships:
            result = _resolve_relationship(
                rel, pass_name, pass_result,
                entity_index, ontology, document_id,
            )
            if isinstance(result, MergedEdgeRecord):
                edges.append(result)
            else:
                rejected_edges.append((pass_name, rel, result))
                rejections_by_pass[pass_name] = (
                    rejections_by_pass.get(pass_name, 0) + 1
                )

    return MergedExtraction(
        entities=list(entity_index.values()),
        edges=edges,
        rejected_edges=rejected_edges,
        rejections_by_pass=rejections_by_pass,
        pipeline_run_id=pipeline_run_id,
        document_id=document_id,
    )
