"""Ontology YAML → Pydantic templates with edge() fields and graph_id_fields."""

from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)

# Loaded from the shared JSON file so both the main app and docling-graph
# service use the same mapping (see ontology/arcadedb_reserved_words.json).
_RESERVED_WORDS_PATH = Path(__file__).resolve().parents[1] / "ontology" / "arcadedb_reserved_words.json"
try:
    RESERVED_WORD_MAP: dict[str, str] = json.loads(_RESERVED_WORDS_PATH.read_text())
except Exception:
    RESERVED_WORD_MAP = {"TABLE": "TABLE_REF"}

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
    id_fields, _ = _derive_graph_id_fields_with_source(entity_type, properties)
    return id_fields


def _derive_graph_id_fields_with_source(
    entity_type: str, properties: dict[str, Any]
) -> tuple[list[str], bool]:
    """Internal: returns (id_fields, has_natural_id).

    has_natural_id=False means the id field is the fallback (first property);
    callers should make such fields Optional in Pydantic to avoid validation
    failures when the LLM emits differently-named or incomplete identifiers.
    """
    if entity_type in _COMPOSITE_ID_TYPES:
        return _COMPOSITE_ID_TYPES[entity_type], True

    prop_names = list(properties.keys())
    if not prop_names:
        return ["name"], False

    for field_name in ("name", "system_name"):
        if field_name in prop_names:
            return [field_name], True

    for field_name in prop_names:
        if field_name.endswith("_id") and field_name != "entity_type":
            return [field_name], True

    for field_name in ("title", "heading"):
        if field_name in prop_names:
            return [field_name], True

    return [prop_names[0]], False


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

    id_fields, has_natural_id = _derive_graph_id_fields_with_source(ontology_name, raw_props)

    field_definitions: dict[str, Any] = {}
    for prop_name, prop_schema in raw_props.items():
        py_type = _TYPE_MAP.get(prop_schema.get("type", "string"), str)
        description = prop_schema.get("description", "")
        example = prop_schema.get("example")

        field_kwargs: dict[str, Any] = {"description": description}
        if example is not None:
            field_kwargs["examples"] = [example]

        # Required only when the id field is a natural identifier
        # (name, system_name, *_id, title, heading, or declared composite).
        # For fallback id fields (first-property heuristic), keep Optional
        # so LLM extractions aren't rejected when the exact field is missing.
        if prop_name in id_fields and has_natural_id:
            field_definitions[prop_name] = (py_type, Field(**field_kwargs))
        else:
            field_definitions[prop_name] = (Optional[py_type], Field(default=None, **field_kwargs))

    # Canonical docling-graph pattern: is_entity=True for entity types
    # (things with identity/graph_id_fields), is_entity=False for components
    # (subordinate parts without their own identity). The library's catalog
    # reads these from model_config to determine entity vs component kind.
    is_entity = len(id_fields) > 0
    model_cls = create_model(
        class_name,
        __config__=ConfigDict(
            graph_id_fields=id_fields,
            is_entity=is_entity,
        ),
        **field_definitions,
    )
    return class_name, model_cls


def build_unified_template(ontology: dict[str, Any]) -> type[BaseModel] | None:
    """Build a single Pydantic template containing all entity types.

    The docling-graph library expects ``PipelineConfig.template`` to be a
    single ``BaseModel`` subclass. This function builds one model with a
    list field per entity type so the LLM extracts all types in one pass.

    Returns ``None`` if the ontology has no entity types with properties.
    """
    per_type = build_templates(ontology)
    if not per_type:
        return None

    fields: dict[str, Any] = {}
    for safe_name, model_cls in per_type.items():
        fields[safe_name.lower()] = (
            list[model_cls],  # type: ignore[valid-type]
            Field(default_factory=list, description=f"Extracted {safe_name} entities"),
        )

    unified = create_model(
        "UnifiedExtractionTemplate",
        __config__=ConfigDict(graph_id_fields=[], is_entity=True),
        **fields,
    )
    logger.info("Built unified template with %d entity types", len(fields))
    return unified


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


def _build_edge_map(ontology: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Build a mapping from source entity type → list of edge definitions.

    Each edge definition is a dict with keys: rel_type, target_type, cardinality.
    Entries are deduplicated on (source, rel_type, target_type). When the same
    source+rel_type appears with multiple target types, one entry per unique target
    is kept; the caller groups by field name (rel_type.lower()).
    """
    cardinality_map: dict[str, str] = {
        rel_def["name"]: rel_def.get("cardinality", "many_to_many")
        for rel_def in ontology.get("relationship_types", [])
        if rel_def.get("name")
    }

    seen: set[tuple[str, str, str]] = set()
    edge_map: dict[str, list[dict[str, str]]] = defaultdict(list)

    for entry in ontology.get("validation_matrix", []):
        source = entry.get("source", "")
        rel_type = entry.get("relationship", "")
        target = entry.get("target", "")
        if not (source and rel_type and target):
            continue

        key = (source, rel_type, target)
        if key in seen:
            continue
        seen.add(key)

        edge_map[source].append(
            {"rel_type": rel_type, "target_type": target, "cardinality": cardinality_map.get(rel_type, "many_to_many")}
        )

    return dict(edge_map)


def build_templates_with_edges(ontology: dict[str, Any]) -> dict[str, type[BaseModel]]:
    """Extend base templates with edge fields derived from the validation_matrix.

    For each entity type that appears as a source in the validation_matrix, edge
    fields are added to the Pydantic model.  Field names are the lowercase
    relationship type (e.g. ``installed_on``, ``has_antenna``).

    Cardinality rules:
    - one_to_one / many_to_one  → ``Optional[TargetModel] = Field(default=None, ...)``
    - one_to_many / many_to_many → ``list[TargetModel] = Field(default_factory=list, ...)``

    When the same relationship type points to multiple target types for the same
    source (e.g. ASSOCIATED_WITH → MISSILE_SYSTEM and AIR_DEFENSE_ARTILLERY_SYSTEM),
    the edge field uses BaseModel as the annotation.

    Returns a new dict mapping safe class name → extended model class.
    """
    base_templates = build_templates(ontology)
    edge_map = _build_edge_map(ontology)

    # Build a reverse lookup once: safe_name → ontology_name (only matters for reserved words)
    reverse_reserved = {v: k for k, v in RESERVED_WORD_MAP.items()}

    result: dict[str, type[BaseModel]] = {}

    for safe_name, base_model in base_templates.items():
        ontology_name = reverse_reserved.get(safe_name, safe_name)
        edges = edge_map.get(ontology_name, [])

        if not edges:
            result[safe_name] = base_model
            continue

        existing_fields: dict[str, Any] = {
            name: (info.annotation, info)
            for name, info in base_model.model_fields.items()
        }

        # Group by field name to detect same rel_type pointing at multiple targets
        edges_by_field: dict[str, list[dict[str, str]]] = defaultdict(list)
        for edge_def in edges:
            edges_by_field[edge_def["rel_type"].lower()].append(edge_def)

        edge_fields: dict[str, Any] = {}
        for field_name, field_edges in edges_by_field.items():
            rel_type = field_edges[0]["rel_type"]
            cardinality = field_edges[0]["cardinality"]

            unique_targets = {e["target_type"] for e in field_edges}
            if len(unique_targets) == 1:
                target_safe = RESERVED_WORD_MAP.get(next(iter(unique_targets)), next(iter(unique_targets)))
                target_model: type[BaseModel] = base_templates.get(target_safe, BaseModel)
            else:
                target_model = BaseModel

            if cardinality in ("one_to_one", "many_to_one"):
                edge_fields[field_name] = (
                    Optional[target_model],
                    Field(default=None, json_schema_extra={"edge_label": rel_type}),
                )
            else:
                edge_fields[field_name] = (
                    list[target_model],
                    Field(default_factory=list, json_schema_extra={"edge_label": rel_type}),
                )

        new_model = create_model(
            safe_name,
            __config__=base_model.model_config,
            **{**existing_fields, **edge_fields},
        )
        result[safe_name] = new_model
        logger.debug("Built edge-extended template %s with %d edge fields", safe_name, len(edge_fields))

    return result
