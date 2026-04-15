"""Pydantic → YAML-parity introspection for the air_defense_v3 bundle.

Plan v32 Phase 3. Builds the ``entity_types`` list (and, in later tasks,
relationship_types / validation_matrix / scoring_weights / full ontology
dict) by reading canonical Pydantic classes in ``entities.py``. Output
is deep-equal to the legacy ``ontology.yaml`` load so consumers can
migrate off YAML under the ``ONTOLOGY_SOURCE`` feature flag.
"""
from __future__ import annotations

from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES
from ontology_bundles.air_defense_v3.relationships import RELATIONSHIP_METADATA
from ontology_bundles.air_defense_v3.validation_matrix import (
    SCORING_WEIGHTS,
    VALIDATION_MATRIX,
)


_PY_TO_YAML_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _python_type_to_yaml(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_yaml(non_none[0])
    if origin in (list,):
        return "array"
    return _PY_TO_YAML_TYPE.get(annotation, "string")


def _parse_label_description(cls: type[BaseModel]) -> tuple[str, str]:
    """Extract YAML ``label`` and ``description`` from the class docstring.

    Generator format: first paragraph is ``<label> — <description>``;
    subsequent paragraphs are anti-pattern notes and are ignored.
    """
    doc = (cls.__doc__ or "").strip()
    first_paragraph = doc.split("\n\n", 1)[0].strip()
    if " — " in first_paragraph:
        label, _, description = first_paragraph.partition(" — ")
        return label.strip(), description.strip()
    return first_paragraph, ""


def _extract_pattern(finfo: FieldInfo) -> str | None:
    for item in finfo.metadata:
        pat = getattr(item, "pattern", None)
        if pat is not None:
            return pat
    return None


def _build_property_schema(finfo: FieldInfo) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": _python_type_to_yaml(finfo.annotation)}
    if finfo.description is not None:
        prop["description"] = finfo.description
    if finfo.examples:
        prop["example"] = finfo.examples[0]
    extra = finfo.json_schema_extra or {}
    enum_values = extra.get("enum") if isinstance(extra, dict) else None
    if enum_values is not None:
        prop["enum"] = list(enum_values)
    pattern = _extract_pattern(finfo)
    if pattern is not None:
        prop["pattern"] = pattern
    return prop


def _is_skippable_field(finfo: FieldInfo) -> bool:
    extra = finfo.json_schema_extra
    if not isinstance(extra, dict):
        return False
    if extra.get("edge_label"):
        return True
    if extra.get("system_field"):
        return True
    return False


def _build_entity_entry(name: str, cls: type[BaseModel]) -> dict[str, Any]:
    cfg = cls.model_config
    label, description = _parse_label_description(cls)

    properties: dict[str, Any] = {}
    for fname, finfo in cls.model_fields.items():
        if _is_skippable_field(finfo):
            continue
        properties[fname] = _build_property_schema(finfo)

    entry: dict[str, Any] = {"name": name, "label": label}

    graph_id_fields = list(cfg.get("graph_id_fields") or [])
    identity_scope = cfg.get("identity_scope", "global")
    if graph_id_fields or identity_scope != "global":
        entry["identity_fields"] = graph_id_fields
        entry["identity_scope"] = identity_scope

    if description:
        entry["description"] = description

    parent = cfg.get("dodaf_parent")
    if parent:
        entry["parent"] = parent

    entry["properties"] = {"type": "object", "properties": properties}
    return entry


def build_entity_types_list() -> list[dict[str, Any]]:
    """Introspect ``ALL_ENTITIES`` and produce a YAML-parity ``entity_types`` list.

    Each entry matches the legacy ``ontology.yaml`` shape: ``name``,
    ``label``, optional ``identity_fields`` + ``identity_scope``,
    ``description``, ``parent``, and ``properties`` (nested
    ``type: object`` with per-field scalar schemas). Typed-edge fields
    (``json_schema_extra.edge_label``) and system-only fields
    (``json_schema_extra.system_field``) are excluded from ``properties``.
    """
    return [_build_entity_entry(name, cls) for name, cls in ALL_ENTITIES.items()]


def build_validation_matrix_list() -> list[dict[str, str]]:
    """Introspect ``VALIDATION_MATRIX`` and produce a YAML-parity list.

    Each entry is ``{source, relationship, target}`` (enum values
    serialized as strings). The frozenset is ordered deterministically by
    ``(source, relationship.value, target)`` so list equality does not
    depend on hash insertion order. Note: ``ontology.yaml`` has one
    duplicate triple (``RADAR_SYSTEM``, ``INSTALLED_ON``, ``PLATFORM``);
    the frozenset collapses it, so this list is 127 unique entries.
    Parity tests must dedupe the YAML side before comparing.
    """
    sorted_triples = sorted(
        VALIDATION_MATRIX, key=lambda t: (t[0], t[1].value, t[2])
    )
    return [
        {"source": src, "relationship": rel.value, "target": tgt}
        for (src, rel, tgt) in sorted_triples
    ]


def build_scoring_weights() -> dict[str, float]:
    """Return ``SCORING_WEIGHTS`` as-is — already YAML-shaped (dict of
    ``{relationship_name_or_default: float}``)."""
    return dict(SCORING_WEIGHTS)


def build_relationship_types_list() -> list[dict[str, Any]]:
    """Introspect ``RELATIONSHIP_METADATA`` and produce a YAML-parity
    ``relationship_types`` list.

    Each entry matches the legacy ``ontology.yaml`` shape: ``name``,
    ``label``, ``description``, ``source_type``, ``target_type``
    (always emitted; YAML carries ``null`` for unscoped relationships),
    and optional ``cardinality`` (omitted when unset). The enum values
    serialize as plain strings — same shape as the YAML.
    """
    entries: list[dict[str, Any]] = []
    for rt, meta in RELATIONSHIP_METADATA.items():
        entry: dict[str, Any] = {
            "name": rt.value,
            "label": meta.label,
            "description": meta.description,
            "source_type": meta.source_type,
            "target_type": meta.target_type,
        }
        if meta.cardinality is not None:
            entry["cardinality"] = meta.cardinality
        entries.append(entry)
    return entries
