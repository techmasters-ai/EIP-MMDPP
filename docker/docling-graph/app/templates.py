"""YAML-to-Pydantic template generator for docling-graph.

Reads the unified ontology YAML and builds dynamic Pydantic model classes that
docling-graph uses to type entity extraction results.

Entity types are grouped into five ontology layers.  ``build_templates()``
returns one combined Pydantic model per group (not per entity type) so that a
single LLM call can extract all entity types in a layer at once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, create_model

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

# Five ontology layers — each group becomes a single combined Pydantic model
# so that one LLM call extracts all entity types in that layer.
GROUP_MAP: dict[str, list[str]] = {
    "reference": [
        "DOCUMENT", "SECTION", "FIGURE", "TABLE", "SPREADSHEET", "ASSERTION",
    ],
    "equipment": [
        "PLATFORM", "RADAR_SYSTEM", "MISSILE_SYSTEM",
        "AIR_DEFENSE_ARTILLERY_SYSTEM", "ELECTRONIC_WARFARE_SYSTEM",
        "FIRE_CONTROL_SYSTEM", "INTEGRATED_AIR_DEFENSE_SYSTEM",
        "LAUNCHER_SYSTEM", "WEAPON_SYSTEM", "ORGANIZATION",
    ],
    "rf_signal": [
        "FREQUENCY_BAND", "RF_EMISSION", "WAVEFORM", "MODULATION",
        "RF_SIGNATURE", "SCAN_PATTERN", "ANTENNA", "TRANSMITTER",
        "RECEIVER", "IF_AMPLIFIER", "SIGNAL_PROCESSING_CHAIN", "SEEKER",
    ],
    "weapon": [
        "GUIDANCE_METHOD", "MISSILE_PERFORMANCE",
        "MISSILE_PHYSICAL_CHARACTERISTICS", "PROPULSION_STACK",
        "PROPULSION_STAGE", "SUBSYSTEM", "COMPONENT",
    ],
    "operational": [
        "CAPABILITY", "RADAR_PERFORMANCE", "ENGAGEMENT_TIMELINE",
        "FORCE_STRUCTURE", "EQUIPMENT_SYSTEM", "ASSEMBLY",
        "SPECIFICATION", "STANDARD", "PROCEDURE", "FAILURE_MODE",
        "TEST_EVENT",
    ],
}


def load_ontology(path: str | Path) -> dict[str, Any]:
    """Load and return the parsed ontology YAML."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def get_ontology_version(ontology: dict[str, Any]) -> str:
    """Return the version string from the ontology dict."""
    return ontology.get("version", "unknown")


def _build_entity_model(
    name: str, props_schema: dict[str, Any]
) -> type[BaseModel]:
    """Build a single Pydantic model for one ontology entity type.

    Returns the model class with ``is_entity`` and ``graph_id_fields``
    metadata attached to ``model_config``.
    """
    prop_fields = props_schema.get("properties", {})

    field_defs: dict[str, Any] = {}
    id_fields: list[str] = []

    for field_name, field_spec in prop_fields.items():
        py_type = _TYPE_MAP.get(field_spec.get("type", "string"), str)
        field_defs[field_name] = (py_type | None, None)

    if prop_fields:
        id_fields = [next(iter(prop_fields))]

    model = create_model(name, __config__=None, **field_defs)
    model.model_config["is_entity"] = True
    model.model_config["graph_id_fields"] = id_fields
    return model


def build_templates(ontology: dict[str, Any]) -> dict[str, dict[str, type[BaseModel]]]:
    """Generate Pydantic entity models grouped by ontology layer.

    Each ontology entity type becomes its own Pydantic model via
    ``_build_entity_model``.  Models are organized into groups matching
    ``GROUP_MAP`` so the service can iterate through entity types per group.

    Returns a dict mapping **group name** -> dict of **entity name** -> model class.
    """
    # Step 1: build individual entity models keyed by entity type name
    entity_models: dict[str, type[BaseModel]] = {}
    entity_types = ontology.get("entity_types", [])
    for et in entity_types:
        name: str = et["name"]
        props_schema = et.get("properties", {})
        entity_models[name] = _build_entity_model(name, props_schema)

    # Step 2: organize entity models into groups.
    # docling-graph expects a single Pydantic model class per run_pipeline call.
    # We store individual entity models grouped by layer so the service can
    # iterate through each entity type within a group.
    from app.prompts import GROUP_PROMPTS

    # Add ontology descriptions to entity model docstrings
    for et in entity_types:
        desc = et.get("description", "")
        name = et["name"]
        if desc and name in entity_models:
            entity_models[name].__doc__ = desc

    templates: dict[str, dict[str, type[BaseModel]]] = {}
    for group_name, member_names in GROUP_MAP.items():
        group_models: dict[str, type[BaseModel]] = {}
        for entity_name in member_names:
            if entity_name not in entity_models:
                logger.warning(
                    "Entity type %s (group %s) not found in ontology — skipping",
                    entity_name,
                    group_name,
                )
                continue
            group_models[entity_name] = entity_models[entity_name]
        templates[group_name] = group_models

    logger.info(
        "Built %d group templates (%d entity types) from ontology v%s",
        len(templates),
        len(entity_models),
        get_ontology_version(ontology),
    )

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
