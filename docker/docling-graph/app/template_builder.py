"""Ontology YAML → Pydantic templates with edge() fields and graph_id_fields."""

from __future__ import annotations
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)

RESERVED_WORD_MAP: dict[str, str] = {"TABLE": "TABLE_REF"}

_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

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


def _safe_type_name(ontology_name: str) -> str:
    """Map an ontology entity type name through RESERVED_WORD_MAP."""
    return RESERVED_WORD_MAP.get(ontology_name, ontology_name)


def _build_single_template(entity_type_def: dict[str, Any]) -> tuple[str, type[BaseModel]]:
    """Build one Pydantic model class from an ontology entity type definition.

    Returns (safe_class_name, model_class).
    """
    ontology_name: str = entity_type_def["name"]
    class_name = _safe_type_name(ontology_name)

    # The ontology uses JSON Schema structure: properties.properties
    raw_props: dict[str, Any] = entity_type_def.get("properties", {}).get("properties", {})

    id_fields = derive_graph_id_fields(ontology_name, raw_props)

    field_definitions: dict[str, Any] = {}
    for prop_name, prop_schema in raw_props.items():
        py_type = _TYPE_MAP.get(prop_schema.get("type", "string"), str)
        description = prop_schema.get("description", "")
        example = prop_schema.get("example")

        field_kwargs: dict[str, Any] = {"description": description}
        if example is not None:
            field_kwargs["examples"] = [example]

        if prop_name in id_fields:
            # Required field: annotation only, no default
            field_definitions[prop_name] = (py_type, Field(**field_kwargs))
        else:
            # Optional field with default None
            field_definitions[prop_name] = (Optional[py_type], Field(default=None, **field_kwargs))

    model_cls = create_model(
        class_name,
        __config__=ConfigDict(graph_id_fields=id_fields),
        **field_definitions,
    )
    return class_name, model_cls


def build_templates(ontology: dict[str, Any]) -> dict[str, type[BaseModel]]:
    """Build Pydantic model classes for all entity types in the ontology.

    Only entity types with defined properties are included.
    Returns a dict mapping safe class name → model class.
    """
    templates: dict[str, type[BaseModel]] = {}
    for entity_type_def in ontology.get("entity_types", []):
        raw_props = entity_type_def.get("properties", {}).get("properties")
        if not raw_props:
            continue
        name, model_cls = _build_single_template(entity_type_def)
        templates[name] = model_cls
        logger.debug("Built template %s (id_fields=%s)", name, model_cls.model_config.get("graph_id_fields"))
    return templates
