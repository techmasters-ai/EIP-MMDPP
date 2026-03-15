"""YAML-to-Pydantic template generator for docling-graph.

Reads the unified ontology YAML and builds dynamic Pydantic model classes that
docling-graph uses to type entity extraction results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

# Mapping from ontology YAML type strings to Python types
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def load_ontology(path: str | Path) -> dict[str, Any]:
    """Load and return the parsed ontology YAML."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_ontology_version(ontology: dict[str, Any]) -> str:
    """Return the version string from the ontology dict."""
    return ontology.get("version", "unknown")


def build_templates(ontology: dict[str, Any]) -> dict[str, type[BaseModel]]:
    """Generate Pydantic model classes from ontology entity types.

    Each entity type becomes a Pydantic model whose fields are derived from
    the entity's ``properties.properties`` mapping in the YAML.  The model's
    ``model_config`` carries ``is_entity=True`` and ``graph_id_fields`` so
    that docling-graph can identify them.

    Returns a dict mapping entity type name -> Pydantic model class.
    """
    templates: dict[str, type[BaseModel]] = {}

    entity_types = ontology.get("entity_types", [])
    for et in entity_types:
        name: str = et["name"]
        props_schema = et.get("properties", {})
        prop_fields = props_schema.get("properties", {})

        # Build field definitions: (type, default)
        field_defs: dict[str, Any] = {}
        id_fields: list[str] = []

        for field_name, field_spec in prop_fields.items():
            py_type = _TYPE_MAP.get(field_spec.get("type", "string"), str)
            # All fields are optional with None default
            field_defs[field_name] = (py_type | None, None)

        # First string field is the graph ID by convention; fall back to "name"
        if prop_fields:
            id_fields = [next(iter(prop_fields))]

        model = create_model(
            name,
            __config__=None,
            **field_defs,
        )

        # Attach graph metadata via model_config (Pydantic v2 style)
        model.model_config["is_entity"] = True
        model.model_config["graph_id_fields"] = id_fields

        templates[name] = model

    logger.info(
        "Built %d entity templates from ontology v%s",
        len(templates),
        get_ontology_version(ontology),
    )

    # Attempt to register edge declarations with docling-graph if available
    _register_edges(ontology)

    return templates


def _register_edges(ontology: dict[str, Any]) -> None:
    """Try to register relationship types with docling_graph.utils.edge.

    If the module is not available (older docling-graph versions), log a
    warning and skip.
    """
    relationship_types = ontology.get("relationship_types", [])
    if not relationship_types:
        return

    try:
        from docling_graph.utils.edge import register_edge_type  # type: ignore[import-untyped]
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "docling_graph.utils.edge not available — skipping %d edge declarations",
            len(relationship_types),
        )
        return

    registered = 0
    for rt in relationship_types:
        try:
            register_edge_type(
                name=rt["name"],
                from_types=rt.get("from_types", []),
                to_types=rt.get("to_types", []),
            )
            registered += 1
        except Exception:
            logger.warning("Failed to register edge type %s", rt.get("name"), exc_info=True)

    logger.info("Registered %d / %d edge types with docling-graph", registered, len(relationship_types))
