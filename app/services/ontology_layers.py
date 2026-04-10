"""Ontology layer slicing for layered graph extraction.

Builds per-layer subsets from the full ontology so each extraction pass
only processes entity types from one layer, keeping prompts small.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_LAYER_MAP_PATH = Path(__file__).resolve().parents[2] / "ontology" / "layer_map.yaml"


def load_layer_map(path: Path | None = None) -> dict[int, list[str]]:
    """Load layer_map.yaml → {layer_number: [entity_type_names]}."""
    p = path or _LAYER_MAP_PATH
    with open(p) as f:
        raw = yaml.safe_load(f)
    layers = raw.get("layers", {})
    return {int(k): v.get("entity_types", []) for k, v in layers.items()}


def _entity_names_set(ontology: dict[str, Any]) -> set[str]:
    return {e["name"] for e in ontology.get("entity_types", [])}


def _relationship_names_set(ontology: dict[str, Any]) -> set[str]:
    return {r["name"] for r in ontology.get("relationship_types", [])}


def build_layer_subset(
    ontology: dict[str, Any],
    layer: int,
    layer_map: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    """Build an ontology subset containing only entity types from one layer.

    Includes relationship types where BOTH source and target are in the layer
    (within-layer relationships).

    Returns a dict with the same shape as the full ontology (entity_types,
    relationship_types, validation_matrix) for direct use with /extract-all.
    """
    if layer_map is None:
        layer_map = load_layer_map()

    layer_entities = set(layer_map.get(layer, []))
    all_ontology_entities = _entity_names_set(ontology)
    # Only include entities that actually exist in the ontology
    layer_entities &= all_ontology_entities

    entity_defs = [
        e for e in ontology.get("entity_types", [])
        if e["name"] in layer_entities
    ]

    # Within-layer relationships: both endpoints in the same layer
    rel_defs = []
    for r in ontology.get("relationship_types", []):
        vm = r.get("validation_matrix", [])
        # Keep if any validation row has both source and target in layer
        layer_rows = [
            row for row in vm
            if row.get("source_type") in layer_entities
            and row.get("target_type") in layer_entities
        ]
        if layer_rows:
            rel_copy = dict(r)
            rel_copy["validation_matrix"] = layer_rows
            rel_defs.append(rel_copy)

    subset = dict(ontology)
    subset["entity_types"] = entity_defs
    subset["relationship_types"] = rel_defs

    logger.info(
        "Layer %d subset: %d entities, %d relationships",
        layer, len(entity_defs), len(rel_defs),
    )
    return subset


def build_cross_layer_subset(
    ontology: dict[str, Any],
    layer_map: dict[int, list[str]] | None = None,
) -> dict[str, Any]:
    """Build an ontology subset for cross-layer relationships only.

    Includes ALL entity types (needed as endpoints) but only relationship
    types where source and target are in DIFFERENT layers.
    """
    if layer_map is None:
        layer_map = load_layer_map()

    # Build entity → layer lookup
    entity_to_layer: dict[str, int] = {}
    for layer_num, entities in layer_map.items():
        for ename in entities:
            entity_to_layer[ename] = layer_num

    # Keep all entity types (cross-layer edges need both endpoints)
    entity_defs = list(ontology.get("entity_types", []))

    # Cross-layer relationships: endpoints in different layers
    rel_defs = []
    for r in ontology.get("relationship_types", []):
        vm = r.get("validation_matrix", [])
        cross_rows = [
            row for row in vm
            if entity_to_layer.get(row.get("source_type"), 0)
            != entity_to_layer.get(row.get("target_type"), 0)
            and row.get("source_type") in entity_to_layer
            and row.get("target_type") in entity_to_layer
        ]
        if cross_rows:
            rel_copy = dict(r)
            rel_copy["validation_matrix"] = cross_rows
            rel_defs.append(rel_copy)

    subset = dict(ontology)
    subset["entity_types"] = entity_defs
    subset["relationship_types"] = rel_defs

    logger.info(
        "Cross-layer subset: %d entities, %d relationships",
        len(entity_defs), len(rel_defs),
    )
    return subset


def build_all_layer_subsets(
    ontology: dict[str, Any],
    layer_map: dict[int, list[str]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Build all extraction subsets: per-layer + cross-layer.

    Returns list of (pass_name, ontology_subset) tuples.
    """
    if layer_map is None:
        layer_map = load_layer_map()

    subsets = []
    for layer_num in sorted(layer_map.keys()):
        subset = build_layer_subset(ontology, layer_num, layer_map)
        if subset["entity_types"]:
            subsets.append((f"layer_{layer_num}", subset))

    cross = build_cross_layer_subset(ontology, layer_map)
    if cross["relationship_types"]:
        subsets.append(("cross_layer", cross))

    return subsets
