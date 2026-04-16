"""Tests for walk_entity_graph component emission via edge_label (Task A0-2).

Per docs:17500-17509 — "Same address node is shared across multiple
people/organizations": components (``is_entity=False``) reached through
``edge(label=...)`` fields must become first-class graph nodes (with
content-based identity per A0-1).

A0-5 contract test forbids components from carrying ``edge_label`` fields,
so the walker does NOT recurse into emitted components.
"""
from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field

from app.services.extraction_merge import walk_entity_graph


def _edge(label: str, **field_kwargs: Any):
    """Mirror ontology_bundles/air_defense_v3/entities.py edge() helper."""
    extra = field_kwargs.pop("json_schema_extra", None) or {}
    extra["edge_label"] = label
    return Field(json_schema_extra=extra, **field_kwargs)


class TinyComp(BaseModel):
    model_config = ConfigDict(is_entity=False, ontology_name="TINY_COMP")
    value: str


class TinyEntity(BaseModel):
    model_config = ConfigDict(
        is_entity=True,
        ontology_name="TINY",
        graph_id_fields=["name"],
        identity_scope="document",
    )
    name: str
    comps: List[TinyComp] = _edge(label="HAS_COMP", default_factory=list)


def test_walker_emits_component_via_edge_label():
    """Per docs:17500-17509 — components reachable via edge() must become graph nodes."""
    entity = TinyEntity(name="e1", comps=[TinyComp(value="x"), TinyComp(value="y")])
    entities: list = []
    edges: list = []
    walk_entity_graph(
        entity,
        on_entity=lambda n: entities.append(n),
        on_edge=lambda parent_id, label, child: edges.append((label, type(child).__name__)),
        ontology={
            "entity_types": [
                {"name": "TINY", "identity_fields": ["name"], "identity_scope": "document"},
                {"name": "TINY_COMP", "identity_fields": ["value"], "identity_scope": "document"},
            ],
        },
        document_id="doc-1",
        at_pass_root=False,
    )
    names = [type(e).__name__ for e in entities]
    assert "TinyEntity" in names
    assert names.count("TinyComp") == 2  # both components emitted
    assert ("HAS_COMP", "TinyComp") in edges
