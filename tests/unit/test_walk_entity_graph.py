"""Tests for walk_entity_graph (plan Task 35b).

Unified walker emits entities AND edges in one traversal.
- on_edge=None: entity-only mode; ontology/document_id tolerated as None.
- on_edge!=None: full mode; ontology and document_id required (for identity).

Traversal rules (per plan):
- at_pass_root=True: walk plain list/scalar BaseModel fields to reach top-level
  entities. Do NOT emit container as entity. Children entered with at_pass_root=False.
- Entity nodes (is_entity=True): emit via on_entity; follow ONLY fields with edge_label.
- Component nodes (is_entity=False) inside the graph: do NOT emit, do NOT recurse.
- Plain nested BaseModel entity fields without edge_label: do NOT recurse.
"""
from __future__ import annotations

from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.services.extraction_merge import walk_entity_graph


# --- Fixture entity classes ------------------------------------------------

def _edge(label: str, **field_kwargs):
    """Mirror ontology_bundles/air_defense_v3/entities.py edge() helper."""
    extra = field_kwargs.pop("json_schema_extra", None) or {}
    extra["edge_label"] = label
    return Field(json_schema_extra=extra, **field_kwargs)


class AntennaEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="ANTENNA",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )
    name: str = Field(...)
    gain_dbi: Optional[float] = None


class FrequencyBandEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="FREQUENCY_BAND",
        graph_id_fields=["band_name"],
        identity_scope="document",
        is_entity=True,
    )
    band_name: str = Field(...)


class RadarSystemEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="document",
        is_entity=True,
    )
    system_name: str = Field(...)
    antennas: List[AntennaEntity] = _edge(label="HAS_ANTENNA", default_factory=list)
    # Plain nested BaseModel entity field (no edge_label) — must NOT be recursed.
    related_radar: Optional["RadarSystemEntity"] = None


RadarSystemEntity.model_rebuild()


class SelfRefEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="SELF_REF",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )
    name: str = Field(...)
    child: Optional["SelfRefEntity"] = _edge(label="CHILDREN", default=None)


SelfRefEntity.model_rebuild()


class MutualA(BaseModel):
    model_config = ConfigDict(
        ontology_name="MUTUAL_A",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )
    name: str = Field(...)
    partner: Optional["MutualB"] = _edge(label="PARTNERED_WITH", default=None)


class MutualB(BaseModel):
    model_config = ConfigDict(
        ontology_name="MUTUAL_B",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )
    name: str = Field(...)
    partner: Optional[MutualA] = _edge(label="PARTNERED_WITH", default=None)


MutualA.model_rebuild()
MutualB.model_rebuild()


class RangeComponent(BaseModel):
    """is_entity=False — component/value-object nested inside an entity."""
    model_config = ConfigDict(is_entity=False)
    min_km: float = Field(...)
    max_km: float = Field(...)


class RadarWithComponent(BaseModel):
    model_config = ConfigDict(
        ontology_name="RADAR_WITH_COMP",
        graph_id_fields=["system_name"],
        identity_scope="document",
        is_entity=True,
    )
    system_name: str = Field(...)
    # Plain nested component — embedded, NOT a graph edge.
    operating_range: Optional[RangeComponent] = None


class RadarWithComponentViaEdge(BaseModel):
    """Schema bug: edge_label pointing at a component. Contract test 9e forbids
    this; walker's runtime guard logs and skips."""
    model_config = ConfigDict(
        ontology_name="RADAR_WITH_COMP_VIA_EDGE",
        graph_id_fields=["system_name"],
        identity_scope="document",
        is_entity=True,
    )
    system_name: str = Field(...)
    operating_range: Optional[RangeComponent] = _edge(
        label="HAS_RANGE", default=None,
    )


class RadarDomainPass(BaseModel):
    """Pass-root container — is_entity not set (acts as non-entity root)."""
    model_config = ConfigDict()
    radar_systems: List[RadarSystemEntity] = Field(default_factory=list)


# --- Ontology fixture (minimal, matches the classes above) ----------------

ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "document"},
        {"name": "ANTENNA", "identity_fields": ["name"], "identity_scope": "document"},
        {"name": "FREQUENCY_BAND", "identity_fields": ["band_name"], "identity_scope": "document"},
        {"name": "SELF_REF", "identity_fields": ["name"], "identity_scope": "document"},
        {"name": "MUTUAL_A", "identity_fields": ["name"], "identity_scope": "document"},
        {"name": "MUTUAL_B", "identity_fields": ["name"], "identity_scope": "document"},
        {"name": "RADAR_WITH_COMP", "identity_fields": ["system_name"], "identity_scope": "document"},
        {"name": "RADAR_WITH_COMP_VIA_EDGE", "identity_fields": ["system_name"], "identity_scope": "document"},
    ],
}
DOCUMENT_ID = "doc-fixture-1"


# --- Test helpers ----------------------------------------------------------

def _collect(node, *, on_edge_hook=None, ontology=None, document_id=None, at_pass_root=True):
    entities: list = []
    edges: list = []
    def on_edge(parent_identity, label, child):
        edges.append((parent_identity, label, child))
        if on_edge_hook is not None:
            on_edge_hook(parent_identity, label, child)
    walk_entity_graph(
        node,
        on_entity=entities.append,
        ontology=ontology,
        document_id=document_id,
        on_edge=on_edge,
        visited_objects=set(),
        at_pass_root=at_pass_root,
    )
    return entities, edges


# --- Step 1: walker in isolation ------------------------------------------

def test_a_radar_with_two_antennas():
    """(a) RadarDomainPass with RadarSystem containing two nested Antennas →
    on_entity called 3× (1 radar + 2 antennas); on_edge called 2×."""
    pass_root = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="AN/TPY-2",
                antennas=[
                    AntennaEntity(name="A1"),
                    AntennaEntity(name="A2"),
                ],
            ),
        ],
    )
    entities, edges = _collect(pass_root, ontology=ONTOLOGY, document_id=DOCUMENT_ID)
    assert len(entities) == 3
    types = sorted(e.model_config["ontology_name"] for e in entities)
    assert types == ["ANTENNA", "ANTENNA", "RADAR_SYSTEM"]
    assert len(edges) == 2
    for parent_identity, label, child in edges:
        assert label == "HAS_ANTENNA"
        assert parent_identity.entity_type == "RADAR_SYSTEM"
        assert parent_identity.identity_tuple == ("AN/TPY-2",)
        assert isinstance(child, AntennaEntity)


def test_b_self_reference_cycle_terminates():
    """(b) Self-reference: entity points to itself → walker terminates; on_entity
    called once; on_edge called once (the self-edge is emitted before the
    visited-guard aborts the recursion)."""
    node = SelfRefEntity(name="Root")
    node.child = node  # self-reference
    entities, edges = _collect(node, ontology=ONTOLOGY, document_id=DOCUMENT_ID, at_pass_root=False)
    assert len(entities) == 1
    assert entities[0] is node
    assert len(edges) == 1
    assert edges[0][1] == "CHILDREN"


def test_b_mutual_reference_cycle_terminates():
    """(b variant) Mutual reference A↔B → each visited once, edge in each
    direction, no infinite recursion."""
    a = MutualA(name="A")
    b = MutualB(name="B")
    a.partner = b
    b.partner = a
    entities, edges = _collect(a, ontology=ONTOLOGY, document_id=DOCUMENT_ID, at_pass_root=False)
    assert len(entities) == 2
    assert {e.model_config["ontology_name"] for e in entities} == {"MUTUAL_A", "MUTUAL_B"}
    assert len(edges) == 2
    directions = {(parent.entity_type, label) for parent, label, _ in edges}
    assert directions == {("MUTUAL_A", "PARTNERED_WITH"), ("MUTUAL_B", "PARTNERED_WITH")}


def test_c_same_logical_entity_two_python_objects():
    """(c) Two distinct Python instances with the same logical identity →
    both visited; dedup is the merger's responsibility, not the walker's."""
    ant_a = AntennaEntity(name="A1", gain_dbi=38.0)
    ant_b = AntennaEntity(name="A1", gain_dbi=None)  # same identity, different object
    pass_root = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(system_name="R1", antennas=[ant_a]),
            RadarSystemEntity(system_name="R2", antennas=[ant_b]),
        ],
    )
    entities, edges = _collect(pass_root, ontology=ONTOLOGY, document_id=DOCUMENT_ID)
    antennas = [e for e in entities if isinstance(e, AntennaEntity)]
    assert len(antennas) == 2
    assert {id(e) for e in antennas} == {id(ant_a), id(ant_b)}


def test_d_component_nested_via_plain_field_not_recursed():
    """(d) Component with is_entity=False, nested via plain (no edge_label)
    field on an entity → NOT recursed into, NOT emitted."""
    node = RadarWithComponent(
        system_name="R1",
        operating_range=RangeComponent(min_km=0.0, max_km=100.0),
    )
    entities, edges = _collect(node, ontology=ONTOLOGY, document_id=DOCUMENT_ID, at_pass_root=False)
    assert [e.model_config.get("ontology_name") for e in entities] == ["RADAR_WITH_COMP"]
    assert edges == []


def test_e_component_reached_via_edge_label_warns_and_skips(caplog):
    """(e) Component reached via edge_label field — contract test 9e catches
    this at schema-validation time. Walker's runtime defense logs and skips
    without emitting the component as an entity or an edge."""
    import logging
    caplog.set_level(logging.WARNING, logger="app.services.extraction_merge")
    node = RadarWithComponentViaEdge(
        system_name="R1",
        operating_range=RangeComponent(min_km=0.0, max_km=100.0),
    )
    entities, edges = _collect(node, ontology=ONTOLOGY, document_id=DOCUMENT_ID, at_pass_root=False)
    assert [e.model_config.get("ontology_name") for e in entities] == ["RADAR_WITH_COMP_VIA_EDGE"]
    assert edges == []  # no edge emitted — target is not is_entity=True
    assert any("contract violation" in rec.getMessage() for rec in caplog.records)


def test_f_plain_basemodel_entity_field_no_edge_label_not_recursed():
    """(f) Plain nested BaseModel entity field (no edge_label) on an entity →
    NOT recursed into, NOT emitted. RadarSystemEntity.related_radar is the
    fixture field under test."""
    child = RadarSystemEntity(system_name="Child")
    parent = RadarSystemEntity(system_name="Parent", related_radar=child)
    entities, edges = _collect(parent, ontology=ONTOLOGY, document_id=DOCUMENT_ID, at_pass_root=False)
    assert [e.system_name for e in entities] == ["Parent"]
    assert edges == []


def test_g_at_pass_root_transitions_to_false_for_children():
    """(g) at_pass_root=True root iteration enters children with
    at_pass_root=False → child entities are emitted normally, not re-walked
    as a container. Verified by: a RadarDomainPass (pass root) containing one
    RadarSystemEntity (entity) that has both a plain nested BaseModel
    (related_radar — must NOT be entered) and an edge_label'd list (antennas —
    MUST be entered). If at_pass_root leaked to children, related_radar would
    be walked as a pass-root child."""
    leaked_child = RadarSystemEntity(system_name="LeakedChild")
    pass_root = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="Parent",
                related_radar=leaked_child,
                antennas=[AntennaEntity(name="A1")],
            ),
        ],
    )
    entities, _ = _collect(pass_root, ontology=ONTOLOGY, document_id=DOCUMENT_ID)
    names = [getattr(e, "system_name", None) or getattr(e, "name", None) for e in entities]
    assert "LeakedChild" not in names  # NOT walked — plain field on entity is embedded
    assert "Parent" in names
    assert "A1" in names


# --- Step 2: duplicate-preserving collection ------------------------------

def test_step2_duplicate_preserving_collection():
    """Step 2: same logical entity appears twice with complementary non-null
    fields. Walker emits both instances. Dedup/field-union is merger's job."""
    ant_1 = AntennaEntity(name="A1", gain_dbi=38.0)   # pass 1: gain filled
    ant_2 = AntennaEntity(name="A1", gain_dbi=None)   # pass-like: gain missing
    pass_root = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(system_name="R1", antennas=[ant_1]),
            RadarSystemEntity(system_name="R2", antennas=[ant_2]),
        ],
    )
    entities, _ = _collect(pass_root, ontology=ONTOLOGY, document_id=DOCUMENT_ID)
    antennas = [e for e in entities if isinstance(e, AntennaEntity)]
    assert len(antennas) == 2
    gain_values = sorted([a.gain_dbi for a in antennas], key=lambda v: (v is None, v))
    assert gain_values == [38.0, None]


# --- Entity-only mode (on_edge=None) ---------------------------------------

def test_entity_only_mode_tolerates_no_ontology():
    """When on_edge=None (entity-only mode), ontology/document_id may be None
    and the walker does not touch identity construction. This is the
    PassResult.iter_entities_of_type fallback path (plan Task 35a)."""
    pass_root = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="R1",
                antennas=[AntennaEntity(name="A1"), AntennaEntity(name="A2")],
            ),
        ],
    )
    collected: list = []
    walk_entity_graph(
        pass_root,
        on_entity=collected.append,
        ontology=None,
        document_id=None,
        on_edge=None,
        visited_objects=set(),
        at_pass_root=True,
    )
    # Still reaches nested antennas via edge_label even without ontology —
    # entity-only mode skips identity building for edges but keeps walking.
    assert len(collected) == 3
    types = sorted(e.model_config["ontology_name"] for e in collected)
    assert types == ["ANTENNA", "ANTENNA", "RADAR_SYSTEM"]


def test_full_mode_requires_ontology_and_document_id():
    """When on_edge is provided, ontology AND document_id must be non-None.
    Clear error at call time rather than a KeyError deep in identity building."""
    node = RadarSystemEntity(system_name="R1", antennas=[AntennaEntity(name="A1")])
    with pytest.raises((AssertionError, ValueError)):
        walk_entity_graph(
            node,
            on_entity=lambda _: None,
            ontology=None,
            document_id="doc-1",
            on_edge=lambda *_: None,
            visited_objects=set(),
            at_pass_root=False,
        )
    with pytest.raises((AssertionError, ValueError)):
        walk_entity_graph(
            node,
            on_entity=lambda _: None,
            ontology=ONTOLOGY,
            document_id=None,
            on_edge=lambda *_: None,
            visited_objects=set(),
            at_pass_root=False,
        )


def test_visited_objects_defaulted_when_none():
    """Convenience: callers may omit visited_objects and the walker creates
    its own set. Required by the _cached_entities fallback."""
    node = RadarSystemEntity(system_name="R1", antennas=[AntennaEntity(name="A1")])
    collected: list = []
    walk_entity_graph(
        node,
        on_entity=collected.append,
        ontology=None,
        document_id=None,
        on_edge=None,
        visited_objects=None,
        at_pass_root=False,
    )
    assert [e.model_config["ontology_name"] for e in collected] == ["RADAR_SYSTEM", "ANTENNA"]
