"""Layered graph extraction: run per-layer passes and merge results.

Splits the ontology by layer so each extraction pass has a small, focused
schema. Merges entities across passes with deduplication and canonical IDs.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from app.services.ontology_layers import build_all_layer_subsets, load_layer_map

logger = logging.getLogger(__name__)


def _canonical_key(entity: dict[str, Any]) -> str:
    """Build canonical dedup key from entity_type + normalized name."""
    etype = (entity.get("entity_type") or "UNKNOWN").upper()
    name = (entity.get("name") or "").strip().lower()
    return f"{etype}::{name}"


def _entity_fingerprint(entity: dict[str, Any]) -> str:
    """Content-based fingerprint for entity dedup across passes."""
    key = _canonical_key(entity)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _edge_key(rel: dict[str, Any]) -> str:
    """Dedup key for relationships."""
    from_name = (rel.get("from_name") or rel.get("source_name") or "").strip().lower()
    to_name = (rel.get("to_name") or rel.get("target_name") or "").strip().lower()
    rel_type = (rel.get("relationship_type") or rel.get("rel_type") or "RELATED_TO").upper()
    return f"{from_name}::{rel_type}::{to_name}"


def merge_extraction_results(
    pass_results: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Merge multiple extraction pass results into one.

    Deduplicates entities by (entity_type, name) and relationships by
    (from_name, rel_type, to_name). Preserves pass-level provenance.

    Args:
        pass_results: List of (pass_name, extraction_result) tuples.

    Returns:
        Merged extraction result dict with entities, relationships, metadata.
    """
    seen_entities: dict[str, dict[str, Any]] = {}  # fingerprint → entity
    seen_edges: dict[str, dict[str, Any]] = {}  # edge_key → relationship
    total_nodes = 0
    total_edges = 0
    pass_metadata: list[dict[str, Any]] = []

    for pass_name, result in pass_results:
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        total_nodes += len(entities)
        total_edges += len(relationships)

        for entity in entities:
            entity["_extraction_pass"] = pass_name
            fp = _entity_fingerprint(entity)
            if fp not in seen_entities:
                seen_entities[fp] = entity
            else:
                # Merge: prefer longer name, higher confidence, merge properties
                existing = seen_entities[fp]
                if len(entity.get("name", "")) > len(existing.get("name", "")):
                    existing["name"] = entity["name"]
                if entity.get("confidence", 0) > existing.get("confidence", 0):
                    existing["confidence"] = entity["confidence"]
                # Merge properties
                existing_props = existing.get("properties", {})
                new_props = entity.get("properties", {})
                for k, v in new_props.items():
                    if v and not existing_props.get(k):
                        existing_props[k] = v
                existing["properties"] = existing_props

        for rel in relationships:
            rel["_extraction_pass"] = pass_name
            ek = _edge_key(rel)
            if ek not in seen_edges:
                seen_edges[ek] = rel
            else:
                # Keep higher confidence
                existing = seen_edges[ek]
                if rel.get("confidence", 0) > existing.get("confidence", 0):
                    seen_edges[ek] = rel

        pass_metadata.append({
            "pass_name": pass_name,
            "entities_extracted": len(entities),
            "relationships_extracted": len(relationships),
            "provider": result.get("provider", "docling-graph"),
            "model": result.get("model", "unknown"),
        })

    merged_entities = list(seen_entities.values())
    merged_relationships = list(seen_edges.values())

    logger.info(
        "Merged %d passes: %d raw entities → %d deduped, %d raw edges → %d deduped",
        len(pass_results), total_nodes, len(merged_entities),
        total_edges, len(merged_relationships),
    )

    return {
        "entities": merged_entities,
        "relationships": merged_relationships,
        "provider": "docling-graph",
        "model": pass_metadata[0].get("model", "unknown") if pass_metadata else "unknown",
        "metadata": {
            "extraction_mode": "layered",
            "passes": pass_metadata,
            "raw_entity_count": total_nodes,
            "raw_edge_count": total_edges,
            "deduped_entity_count": len(merged_entities),
            "deduped_edge_count": len(merged_relationships),
        },
    }


def run_layered_extraction(
    docling_document_json: dict[str, Any],
    document_id: str,
    ontology: dict[str, Any],
    max_passes: int = 10,
    cross_layer_enabled: bool = True,
) -> dict[str, Any]:
    """Run layer-by-layer extraction and merge results.

    Args:
        docling_document_json: The enriched DoclingDocument JSON.
        document_id: Document ID for logging.
        ontology: Full ontology definition.
        max_passes: Maximum number of extraction passes.
        cross_layer_enabled: Whether to run cross-layer relationship pass.

    Returns:
        Merged extraction result.
    """
    from app.services.docling_graph_service import extract_graph_all

    layer_map = load_layer_map()
    subsets = build_all_layer_subsets(ontology, layer_map)

    if not cross_layer_enabled:
        subsets = [(name, sub) for name, sub in subsets if name != "cross_layer"]

    subsets = subsets[:max_passes]

    pass_results: list[tuple[str, dict[str, Any]]] = []
    for pass_name, subset in subsets:
        entity_count = len(subset.get("entity_types", []))
        rel_count = len(subset.get("relationship_types", []))
        logger.info(
            "Layered extraction pass '%s' for %s: %d entities, %d rels",
            pass_name, document_id, entity_count, rel_count,
        )

        if entity_count == 0 and rel_count == 0:
            logger.info("Skipping empty pass '%s'", pass_name)
            continue

        try:
            result = extract_graph_all(
                docling_document_json,
                document_id,
                ontology_definition=subset,
            )
            pass_results.append((pass_name, result))
        except Exception as exc:
            logger.warning(
                "Layered extraction pass '%s' failed for %s: %s",
                pass_name, document_id, exc,
            )
            pass_results.append((pass_name, {"entities": [], "relationships": []}))

    if not pass_results:
        return {"entities": [], "relationships": [], "provider": "docling-graph", "model": "unknown"}

    return merge_extraction_results(pass_results)
