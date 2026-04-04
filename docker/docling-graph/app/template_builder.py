"""Ontology YAML → Pydantic templates with edge() fields and graph_id_fields."""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

_COMPOSITE_ID_TYPES: dict[str, list[str]] = {
    "SPECIFICATION": ["parameter", "value"],
}


def derive_graph_id_fields(entity_type: str, properties: dict[str, Any]) -> list[str]:
    """Derive graph_id_fields for an entity type from its ontology properties.

    Priority:
    1. Composite identity (if entity_type in _COMPOSITE_ID_TYPES)
    2. 'name' or 'system_name' property
    3. Property with '_id' suffix
    4. 'title' or 'heading' property
    5. Fallback: first property
    """
    if entity_type in _COMPOSITE_ID_TYPES:
        return _COMPOSITE_ID_TYPES[entity_type]

    prop_names = list(properties.keys())
    if not prop_names:
        return ["name"]

    for field_name in ("name", "system_name"):
        if field_name in prop_names:
            return [field_name]

    for field_name in prop_names:
        if field_name.endswith("_id") and field_name != "entity_type":
            return [field_name]

    for field_name in ("title", "heading"):
        if field_name in prop_names:
            return [field_name]

    return [prop_names[0]]
