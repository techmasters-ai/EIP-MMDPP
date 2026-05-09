"""Provenance extraction from a pipeline context (Phase 8 Task 51).

Walks ``context.knowledge_graph`` and emits an ``ExtractionProvenance``
row per extracted entity instance. Nodes whose ``element_uid`` cannot
be resolved are dropped with a WARNING — the downstream worker path
(``derive_structure_links`` → mention edges) resolves chunks
exclusively via ``element_uid``, so rows without one cannot produce
mention-based edges and must not enter the response.

Node-attribute conventions checked (first match wins):

  1. Direct ``element_uid`` attribute on the node data dict.
  2. Nested ``provenance.element_uid`` — some extractor contracts
     attach provenance as a sub-dict.
  3. ``chunk_indexes[0]`` on the node's provenance dict, mapped to
     the first ``doc_item.self_ref`` of that chunk via the
     ``chunk_to_self_refs`` mapping built by ``main.py`` before
     calling this function. This is the fallback for the delta-IR
     normalizer path, which stores chunk indexes instead of element
     uids (see
     ``docling_graph/core/extractors/contracts/delta/ir_normalizer.py``).
     The previous implementation indexed into ``docling_document.texts``
     with a chunk index, which is a type mismatch (chunks group
     multiple doc_items; the two index spaces are unrelated) and was
     the reason production passes returned empty provenance lists.

Entity-typed nodes are identified via ``node.get("label")`` or
``node.get("node_type")``; nodes without one of the entity sentinel
values are skipped (they're likely property/edge helper nodes rather
than entities). The sentinel set is permissive — anything that looks
like an uppercase ontology name is treated as an entity.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from app._numeric_evidence import (
    normalize_text as _normalize_text,
    value_match_candidates as _value_match_candidates,
)

logger = logging.getLogger(__name__)


def _resolve_instance_id(node_data: dict, fallback: str | None = None) -> str | None:
    """Unified resolver for entity instance_id.

    Used by both entity-row construction and relationship endpoint lookup so
    both sides resolve the same node to the same ID. Fallback chain:
      1. node_data["instance_id"]
      2. node_data["node_uid"]
      3. node_data["__delta_node_uid"]
      4. fallback (e.g. networkx node id) if provided
    Returns None when nothing matches (caller decides whether to UUID or skip).
    """
    for key in ("instance_id", "node_uid", "__delta_node_uid"):
        v = node_data.get(key) if isinstance(node_data, dict) else None
        if v:
            return str(v)
    if fallback:
        return str(fallback)
    return None


def _find_model_class(annotation: Any) -> type[BaseModel] | None:
    """Return the first BaseModel subclass reached through the annotation
    (unwraps Optional / List / etc.)."""
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        return None
    for arg in get_args(annotation):
        nested = _find_model_class(arg)
        if nested is not None:
            return nested
    return None


def synthesize_provenance_from_pass_output(
    pass_output: dict[str, Any],
    template_cls: type[BaseModel],
    chunk_to_self_refs: dict[int, list[str]] | None,
    provenance_cls: type,
) -> list[Any]:
    """Emit one ExtractionProvenance row per entity in ``pass_output``.

    Used as a fallback when ``build_provenance_from_context`` returns [].
    The library's salvage path (missing_root_instance → empty_output →
    legacy prompt-schema retry → direct mode) populates the knowledge
    graph with nodes but drops the element-tracking attributes
    (chunk_indexes / self_ref / element_uid), so the pass response ends
    up carrying entities with no provenance rows — downstream mentions
    are empty, EXTRACTED_FROM never fires, entities aren't reachable
    from the Document vertex.

    This synthesizer walks ``template_cls.model_fields`` to find
    is_entity=True list fields, and emits a minimal
    ``ExtractionProvenance`` per entry:

    * ``instance_id`` — a fresh UUID (entity identity dedup is
      downstream's job)
    * ``ontology_name`` — item_cls ``model_config.ontology_name`` if
      set, else the class name
    * ``identity_values`` — copied from identity_fields of the item
    * ``element_uid`` — first self_ref from chunk 0 (via
      chunk_to_self_refs) if available, else empty
    * ``page`` / ``chunk_index`` — None / 0

    Coarse but deterministic. Enough to seed MENTIONED_IN /
    EXTRACTED_FROM edges so the entity reaches chunk provenance.
    """
    if not isinstance(pass_output, dict):
        return []

    if chunk_to_self_refs:
        first_refs = chunk_to_self_refs.get(0) or next(
            iter(chunk_to_self_refs.values()), None
        )
        first_element_uid = first_refs[0] if first_refs else ""
    else:
        first_element_uid = ""

    out: list[Any] = []
    for field_name, field_info in template_cls.model_fields.items():
        items = pass_output.get(field_name)
        if not isinstance(items, list) or not items:
            continue
        item_cls = _find_model_class(getattr(field_info, "annotation", None))
        if item_cls is None:
            continue
        model_config = item_cls.model_config or {}
        if not model_config.get("is_entity"):
            continue

        ontology_name = str(model_config.get("ontology_name") or item_cls.__name__)
        id_fields = tuple(model_config.get("graph_id_fields", []) or [])

        for item in items:
            if not isinstance(item, dict):
                continue
            identity_values = {
                f: item.get(f) for f in id_fields if item.get(f) is not None
            }
            out.append(
                provenance_cls(
                    instance_id=str(uuid.uuid4()),
                    ontology_name=ontology_name,
                    identity_values=identity_values,
                    element_uid=first_element_uid,
                    page=None,
                    chunk_index=0 if chunk_to_self_refs else None,
                )
            )
    return out


def _is_entity_label(label: Any) -> bool:
    """Treat any non-empty string that looks like an ontology name
    (uppercase / underscore / digits) as an entity label.

    Non-entity helper nodes in the library often carry lowercase
    labels (``path``, ``property``, etc.) so the uppercase heuristic
    is a reasonable filter.
    """
    if not isinstance(label, str) or not label:
        return False
    return label == label.upper() and any(c.isalpha() for c in label)


def _resolve_element_uid(
    node_data: dict[str, Any],
    chunk_to_self_refs: dict[int, list[str]] | None,
) -> str | None:
    """Return element_uid for a knowledge-graph node, or None if none.

    Resolution order:
      1. Direct `element_uid` attribute on the node dict.
      2. Nested `provenance.element_uid`.
      3. Nested `provenance.self_refs[0]` (extraction-time, authoritative).
      4. Nested `provenance.evidence_ids[0]` IF it is a Docling self_ref
         (starts with '#/').
      5. `provenance.chunk_indexes[0]` → first self_ref via
         chunk_to_self_refs (extraction-time map from main.py).
    """
    direct = node_data.get("element_uid")
    if isinstance(direct, str) and direct:
        return direct

    prov = node_data.get("provenance")
    if not isinstance(prov, dict):
        return None

    nested = prov.get("element_uid")
    if isinstance(nested, str) and nested:
        return nested

    self_refs = prov.get("self_refs")
    if isinstance(self_refs, list) and self_refs:
        first = self_refs[0]
        if isinstance(first, str) and first:
            return first

    evidence_ids = prov.get("evidence_ids")
    if isinstance(evidence_ids, list):
        for eid in evidence_ids:
            if isinstance(eid, str) and eid.startswith("#/"):
                return eid

    if chunk_to_self_refs:
        chunk_indexes = prov.get("chunk_indexes")
        if isinstance(chunk_indexes, list) and chunk_indexes:
            first = chunk_indexes[0]
            if isinstance(first, int):
                refs = chunk_to_self_refs.get(first)
                if refs:
                    return refs[0]

    return None


_EVIDENCE_TEXT_CAP = 500


def _join_evidence_text_for_node(
    prov_dict: dict,
    chunk_to_evidence_units: dict[int, list[dict]] | None,
) -> str | None:
    """Join cited evidence units' text into a single capped snippet.

    Selects evidence units by:
      1. Walking prov_dict["chunk_indexes"] to look up the chunk's units.
      2. Filtering to units whose evidence_id appears in
         prov_dict["evidence_ids"] (per-node citations) — falls back to
         all units when the per-node list is empty.
    """
    if not chunk_to_evidence_units:
        return None
    chunk_indexes = prov_dict.get("chunk_indexes") or []
    cited_ids = set(prov_dict.get("evidence_ids") or [])

    parts: list[str] = []
    for ci in chunk_indexes:
        units = chunk_to_evidence_units.get(ci) or []
        for u in units:
            if not isinstance(u, dict):
                continue
            text = u.get("text") or ""
            if not text.strip():
                continue
            if cited_ids and u.get("evidence_id") not in cited_ids:
                continue
            parts.append(text.strip())
    if not parts:
        return None
    joined = " ".join(parts)
    if len(joined) <= _EVIDENCE_TEXT_CAP:
        return joined
    return joined[: _EVIDENCE_TEXT_CAP - 1] + "…"


def _resolve_identity_values(
    node_data: dict[str, Any],
    ontology_name: str,
) -> dict[str, Any]:
    """Return the node's identity-values dict. Falls back to an empty
    dict when the library doesn't surface identity fields — that is
    the correct shape for is_entity=False components, and the worker's
    downstream aggregation (Task 52a) uses ``instance_id`` anyway."""
    iv = node_data.get("identity_values")
    if isinstance(iv, dict):
        return iv
    # The delta-IR normalizer sometimes stores identity scalars inline
    # on the node (``node.properties``); we don't try to reconstruct
    # the identity tuple here — the worker side re-reads the dumped
    # template_instance via logical_identity_from_dict.
    return {}


def _resolve_page(node_data: dict[str, Any]) -> int | None:
    direct = node_data.get("page")
    if isinstance(direct, int):
        return direct
    prov = node_data.get("provenance")
    if isinstance(prov, dict):
        pages = prov.get("page_numbers")
        if isinstance(pages, list) and pages:
            first = pages[0]
            if isinstance(first, int):
                return first
    return None


def _resolve_chunk_index(node_data: dict[str, Any]) -> int | None:
    direct = node_data.get("chunk_index")
    if isinstance(direct, int):
        return direct
    prov = node_data.get("provenance")
    if isinstance(prov, dict):
        idxs = prov.get("chunk_indexes")
        if isinstance(idxs, list) and idxs:
            first = idxs[0]
            if isinstance(first, int):
                return first
    return None


def build_provenance_from_context(
    context: Any,
    provenance_cls: type,
    chunk_to_self_refs: dict[int, list[str]] | None = None,
    chunk_to_evidence_units: dict[int, list[dict]] | None = None,
) -> list[Any]:
    """Walk ``context.knowledge_graph`` and return a list of
    ``ExtractionProvenance`` instances, one per entity-typed node whose
    element_uid could be resolved.

    ``provenance_cls`` is injected rather than imported so this module
    plays nicely with the docling-graph test conftest's sys.path swap
    (the repo-root ``app.schemas`` package shadows the service's
    ``app.schemas`` when the test suite runs from repo root).

    ``chunk_to_self_refs`` maps ``chunk_index → [doc_item.self_ref, ...]``
    and is built by the caller (``main.py``) by re-running the chunker
    on the DoclingDocument after extraction succeeds. ``_resolve_element_uid``
    uses the first self_ref of the chunk referenced by
    ``provenance.chunk_indexes[0]``. Pass ``None`` to disable step 3 and
    fall through to drop-with-warning.

    Dropped node classes:
      * Nodes without an entity-looking ``label`` / ``node_type``.
      * Nodes where ``element_uid`` cannot be resolved via any strategy
        (logged WARNING with the node id).
    """
    graph = getattr(context, "knowledge_graph", None)
    if graph is None:
        return []

    out: list[Any] = []
    for node_id, data in graph.nodes(data=True):
        label = data.get("label") or data.get("node_type")
        if not _is_entity_label(label):
            continue

        element_uid = _resolve_element_uid(data, chunk_to_self_refs)
        if element_uid is None:
            # Fallback: when the library's "direct" / salvage paths fire
            # (missing_root_instance → empty_output → legacy prompt-schema
            # retry), graph nodes lose their element_uid / chunk_indexes /
            # self_ref attributes. Drop-with-warning in that case produces
            # an empty provenance list, which cascades: no mentions[] →
            # no EXTRACTED_FROM edges → get_document_entities_sync returns
            # 0 via traversal. Anchor entities to the FIRST chunk's first
            # self_ref as a coarse but deterministic fallback. Good enough
            # to seed an EXTRACTED_FROM edge; downstream consumers treat
            # mentions as weak evidence already.
            if chunk_to_self_refs:
                first_chunk_refs = chunk_to_self_refs.get(0) or next(
                    iter(chunk_to_self_refs.values()), None
                )
                if first_chunk_refs:
                    element_uid = first_chunk_refs[0]
            if element_uid is None:
                logger.warning(
                    "build_provenance: dropping node %s (label=%s) — no resolvable element_uid",
                    node_id, label,
                )
                continue

        identity_values = _resolve_identity_values(data, ontology_name=label)
        # Per-instance id: explicit one if the library provides it, else
        # a fresh UUID so aggregation keeps instances distinct even when
        # identity collapses (empty-identity components, duplicate
        # extractions).
        instance_id = _resolve_instance_id(data, fallback=str(node_id)) or str(uuid.uuid4())

        prov_dict = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
        out.append(
            provenance_cls(
                instance_id=str(instance_id),
                ontology_name=str(label),
                identity_values=identity_values,
                element_uid=element_uid,
                page=_resolve_page(data),
                chunk_index=_resolve_chunk_index(data),
                evidence_ids=[
                    eid for eid in (prov_dict.get("evidence_ids") or [])
                    if isinstance(eid, str)
                ],
                page_numbers=sorted({
                    p for p in (prov_dict.get("page_numbers") or [])
                    if isinstance(p, int)
                }),
                evidence_text=_join_evidence_text_for_node(prov_dict, chunk_to_evidence_units),
            )
        )
    return out


def _normalize_field_evidence_inputs(input_chunks) -> list[dict]:
    """Accept either (element_uid, text) tuples (legacy) or evidence-unit
    dicts (new). Return a normalized list of dicts ready for matching."""
    out: list[dict] = []
    for entry in input_chunks or []:
        if isinstance(entry, dict):
            out.append({
                "element_uid": entry.get("self_ref") or entry.get("evidence_id"),
                "text": entry.get("text") or "",
                "evidence_id": entry.get("evidence_id"),
                "page": (entry.get("page_numbers") or [None])[0],
                "document_id": entry.get("document_id"),
            })
        elif isinstance(entry, tuple) and len(entry) >= 2:
            out.append({
                "element_uid": entry[0],
                "text": entry[1],
                "evidence_id": entry[0] if isinstance(entry[0], str) and entry[0].startswith("#/") else None,
                "page": None,
                "document_id": None,
            })
    return out


def _excerpt_around(text: str, needle_norm: str, max_chars: int = 240) -> str:
    """Return a short window of *text* centred on the first match of
    *needle_norm* (already normalized). Falls back to text[:max_chars]."""
    norm = _normalize_text(text)
    pos = norm.find(needle_norm)
    if pos < 0:
        return text[:max_chars].strip()
    half = max_chars // 2
    raw_start = max(0, pos - half)
    raw_end = min(len(text), raw_start + max_chars)
    return text[raw_start:raw_end].strip()


def build_auto_field_evidence(
    primary_entities: list[Any],
    instance_ids: list[str],
    input_chunks: list[tuple[str, str]],
    skip_fields: set[str],
    provenance_cls: type,
) -> list[Any]:
    """Deterministic per-field provenance — no LLM involvement.

    For each non-None populated field on each primary entity, produce
    one ExtractionFieldProvenance row per input chunk that contains
    any string form of the field value (numeric+unit-aware variants).
    When no chunk matches, falls back to attaching every input chunk
    as a candidate so the field still has *some* provenance row.

    Args:
        primary_entities: the pass's primary entity list (e.g. the
            populated radar_systems / missile_systems).
        instance_ids: list aligned with primary_entities — i-th entry
            is the resolved instance_id for primary_entities[i].
        input_chunks: list of (element_uid, text) tuples — one per
            DoclingDocument chunk fed to the LLM.
        skip_fields: set of canonical field names to never treat as
            extracted properties (system_field, edge_label, identity
            tagged on json_schema_extra). The caller pre-computes this.
        provenance_cls: ExtractionFieldProvenance class to instantiate.

    Returns:
        list of provenance rows, ordered by (entity_index, field_name)
        for stable diffs.
    """
    normalized_chunks = _normalize_field_evidence_inputs(input_chunks)
    out: list[Any] = []
    for idx, entity in enumerate(primary_entities):
        instance_id = instance_ids[idx] if idx < len(instance_ids) else ""
        if not hasattr(entity, "model_fields"):
            continue
        for fname in entity.__class__.model_fields:
            if fname in skip_fields:
                continue
            value = getattr(entity, fname, None)
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (list, tuple, set, dict)) and not value:
                continue  # empty default-factory collections aren't extracted data
            candidates = _value_match_candidates(value, fname)
            if not candidates:
                continue
            normalized = [_normalize_text(c) for c in candidates if c]
            normalized = [n for n in normalized if n]
            matches: list[dict] = []  # each entry is a normalized chunk dict + snippet
            for row in normalized_chunks:
                ctext = row["text"]
                if not ctext:
                    continue
                ctext_norm = _normalize_text(ctext)
                hit = next((n for n in normalized if n in ctext_norm), None)
                if hit is None:
                    continue
                snippet = _excerpt_around(ctext, hit)
                matches.append({**row, "snippet": snippet})
            if not matches and normalized_chunks:
                # Fall back to batch-level: every chunk a candidate.
                for row in normalized_chunks[:3]:
                    ctext = row["text"]
                    if not ctext:
                        continue
                    snippet = ctext[:240].strip()
                    matches.append({**row, "snippet": snippet})
            for row in matches:
                out.append(provenance_cls(
                    instance_id=instance_id,
                    field_name=fname,
                    value=value,
                    supporting_snippet=row["snippet"],
                    element_uid=row.get("element_uid"),
                    evidence_id=row.get("evidence_id"),
                    page=row.get("page"),
                    document_id=row.get("document_id"),
                ))
    return out


def build_relationship_provenance_from_delta_trace(
    context: Any,
    provenance_cls: type,
) -> list[Any]:
    """Read normalized relationships from context._delta_merged_graph and emit
    one ExtractionRelationshipProvenance per relationship.

    Source: context._delta_merged_graph is the post-pipeline merged graph dict
    loaded by main.py from <debug_dir>/debug/delta_merged_graph.json. It carries
    the normalized delta IR's relationships WITH provenance — which the
    Pydantic-to-graph converter does NOT preserve on context.knowledge_graph
    edges (graph_converter.py creates Edge with properties={}).

    Note: context._delta_trace (from delta_trace.json) has stats/diagnostics only,
    NOT relationship data. delta_merged_graph.json is the correct source for
    relationship provenance.
    """
    merged_graph = getattr(context, "_delta_merged_graph", None)
    if not isinstance(merged_graph, dict):
        return []

    relationships = merged_graph.get("relationships") or []
    nodes = merged_graph.get("nodes") or []

    # Build lookup: (path, ids-frozenset) → instance_id. Keying by path
    # ALONE collides when multiple nodes share a path but differ by ids.
    def _node_key(node_path: str, ids_dict: dict | None) -> tuple:
        if isinstance(ids_dict, dict):
            id_items = tuple(sorted((str(k), str(v)) for k, v in ids_dict.items()))
        else:
            id_items = ()
        return (str(node_path), id_items)

    node_lookup: dict[tuple, str] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        path = n.get("path")
        instance_id = _resolve_instance_id(n)  # no fallback — rel lookup needs a real source
        if isinstance(path, str) and instance_id:
            node_lookup[_node_key(path, n.get("ids"))] = instance_id

    out: list[Any] = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        label = rel.get("edge_label") or rel.get("label")
        if not label:
            continue
        prov = rel.get("provenance") if isinstance(rel.get("provenance"), dict) else {}
        source_key = _node_key(rel.get("source_path") or "", rel.get("source_ids"))
        target_key = _node_key(rel.get("target_path") or "", rel.get("target_ids"))
        out.append(
            provenance_cls(
                relationship_type=str(label),
                source_instance_id=node_lookup.get(source_key),
                target_instance_id=node_lookup.get(target_key),
                evidence_ids=[
                    eid for eid in (prov.get("evidence_ids") or [])
                    if isinstance(eid, str)
                ],
                self_refs=[
                    r for r in (prov.get("self_refs") or [])
                    if isinstance(r, str)
                ],
                page_numbers=sorted({
                    p for p in (prov.get("page_numbers") or [])
                    if isinstance(p, int)
                }),
                supporting_snippet=None,
            )
        )
    return out
