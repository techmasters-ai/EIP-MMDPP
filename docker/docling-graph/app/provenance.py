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
from typing import Any

logger = logging.getLogger(__name__)


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
    """Return the element_uid for a knowledge-graph node, or None if
    none can be resolved.

    Resolution order:
      1. Direct ``element_uid`` attribute on the node dict.
      2. Nested ``provenance.element_uid``.
      3. ``provenance.chunk_indexes[0]`` → first ``doc_item.self_ref``
         of that chunk, via ``chunk_to_self_refs`` (built by main.py
         after run_pipeline via the service's own HybridChunker pass).

    ``chunk_to_self_refs`` may be ``None`` when the caller couldn't
    build the mapping (e.g. chunker init failed). In that case step 3
    is skipped and the node falls through to the drop-with-warning
    branch.
    """
    # 1. Direct attribute
    direct = node_data.get("element_uid")
    if isinstance(direct, str) and direct:
        return direct

    # 2. Nested provenance dict
    prov = node_data.get("provenance")
    if isinstance(prov, dict):
        nested = prov.get("element_uid")
        if isinstance(nested, str) and nested:
            return nested

        # 3. chunk_indexes → chunk_to_self_refs[idx][0]
        if chunk_to_self_refs:
            chunk_indexes = prov.get("chunk_indexes")
            if isinstance(chunk_indexes, list) and chunk_indexes:
                first = chunk_indexes[0]
                if isinstance(first, int):
                    refs = chunk_to_self_refs.get(first)
                    if refs:
                        return refs[0]

    return None


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
        instance_id = (
            data.get("instance_id")
            or data.get("node_uid")
            or str(node_id)
            or str(uuid.uuid4())
        )

        out.append(
            provenance_cls(
                instance_id=str(instance_id),
                ontology_name=str(label),
                identity_values=identity_values,
                element_uid=element_uid,
                page=_resolve_page(data),
                chunk_index=_resolve_chunk_index(data),
            )
        )
    return out
