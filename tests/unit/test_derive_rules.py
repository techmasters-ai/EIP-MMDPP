"""Tests for derive_rules.derive_structural_edges.
HAS_PROVENANCE is NOT created here — see spec §3.8."""
from dataclasses import dataclass, field

from ontology_bundles.air_defense_v3 import derive_rules
from ontology_bundles.air_defense_v3.derive_rules import (
    ChunkForDerivation, DerivedEdge,
)


@dataclass(frozen=True)
class FakeIdentity:
    entity_type: str
    identity_field_names: tuple = ("name",)
    identity_tuple: tuple = ("Test",)
    scope: str = "global"
    document_id: str | None = None


@dataclass
class FakeMergedEntity:
    identity: FakeIdentity
    properties: dict = field(default_factory=dict)
    confidence: float = 0.9
    pass_origins: set = field(default_factory=set)
    display_label: str = "Test"


@dataclass
class FakeMerged:
    entities: list
    edges: list = field(default_factory=list)
    rejected_edges: list = field(default_factory=list)


def test_derive_structural_edges_no_has_provenance():
    """HAS_PROVENANCE must be handled by upsert_nodes_batch_sync,
    NOT by derive_rules. See spec §3.8."""
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="Fan Song",
    )
    merged = FakeMerged(entities=[entity])
    identity_to_rid = {entity.identity: "#10:1"}
    chunks = []

    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid="#11:1",
    )

    rel_types = [e.rel_type for e in edges]
    assert "HAS_PROVENANCE" not in rel_types, (
        "HAS_PROVENANCE must come from upsert_nodes_batch_sync, not derive_rules"
    )


def test_derive_structural_edges_mentioned_in():
    """MENTIONED_IN edges are created from entity display labels to
    chunks that contain them."""
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="Fan Song",
    )
    merged = FakeMerged(entities=[entity])
    identity_to_rid = {entity.identity: "#10:1"}
    chunks = [
        ChunkForDerivation(rid="#5:1", text_normalized="the fan song radar system"),
        ChunkForDerivation(rid="#5:2", text_normalized="unrelated text"),
    ]

    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid="#11:1",
    )

    mentioned = [e for e in edges if e.rel_type == "MENTIONED_IN"]
    assert len(mentioned) == 1
    assert mentioned[0].from_id == "#10:1"
    assert mentioned[0].to_id == "#5:1"


def test_derive_structural_edges_skips_entities_without_rid():
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="Fan Song",
    )
    merged = FakeMerged(entities=[entity])
    # Empty identity_to_rid — simulates a merged entity that wasn't upserted
    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid={},
        chunks=[ChunkForDerivation(rid="#5:1", text_normalized="fan song")],
        document_rid="#11:1",
    )
    assert edges == []


def test_derive_structural_edges_skips_empty_display_label():
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="",
    )
    merged = FakeMerged(entities=[entity])
    identity_to_rid = {entity.identity: "#10:1"}
    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=[ChunkForDerivation(rid="#5:1", text_normalized="some text")],
        document_rid="#11:1",
    )
    # No edges because canonical label is empty
    assert edges == []
