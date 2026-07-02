"""Live ontology service — serves the ontology straight from the
``air_defense_v3`` bundle's Pydantic SSoT (no stored/registry copy).

Plan: docs/superpowers/plans/2026-07-01-remove-registry-standalone-profiles.md,
Task 1.
"""
from __future__ import annotations

from typing import Any

from app.services.query_profiles import _CANONICAL_BY_ENTITY_TYPE
from ontology_bundles.air_defense_v3.introspect import build_ontology_dict


def _collect_profile_sections() -> list[str]:
    """Walk every canonical Pydantic class registered in
    ``_CANONICAL_BY_ENTITY_TYPE`` and collect every
    ``json_schema_extra["profile_sections"]`` value found across their
    ``model_fields``. Deduped and sorted — NOT hardcoded, so new
    profile_sections tags on the canonical classes surface automatically.
    """
    sections: set[str] = set()
    for cls in _CANONICAL_BY_ENTITY_TYPE.values():
        for finfo in cls.model_fields.values():
            extra = finfo.json_schema_extra or {}
            if not isinstance(extra, dict):
                continue
            raw = extra.get("profile_sections")
            # Guard the value type: a str would silently iterate its
            # characters. Only list/tuple of section names is valid.
            if not isinstance(raw, (list, tuple)):
                continue
            for section in raw:
                if isinstance(section, str):
                    sections.add(section)
    return sorted(sections)


def get_live_ontology() -> dict[str, Any]:
    """Return the live ontology payload for ``GET /v1/ontology``.

    Shape: ``{version, entity_types: [{name, label}],
    relationship_types: [{name}], profile_sections: [str, ...]}``.
    ``version`` and the entity/relationship types come straight from
    ``build_ontology_dict()`` (the air_defense_v3 introspection SSoT);
    ``profile_sections`` is derived from the canonical Pydantic classes'
    field tags, not hardcoded.
    """
    ontology = build_ontology_dict()

    entity_types = [
        {"name": entry["name"], "label": entry.get("label") or entry["name"]}
        for entry in ontology["entity_types"]
    ]
    relationship_types = [
        {"name": entry["name"]} for entry in ontology["relationship_types"]
    ]

    return {
        "version": ontology["version"],
        "entity_types": entity_types,
        "relationship_types": relationship_types,
        "profile_sections": _collect_profile_sections(),
    }
