"""Phase 8 Task 53 — _serialize_for_audit writes real mentions + nodes.

The audit blob now carries ``nodes[]``, ``mentions[]`` and
``element_to_artifact`` so ``derive_structure_links`` can build
``EXTRACTED_FROM`` chunk-link edges without re-querying Postgres.

Task 52b's ``LogicalIdentity.serialize_as_entity_id`` backs the
entity_id field on both ``nodes[]`` and ``mentions[]`` — the same
string across both, so joining by entity_id is trivial.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.extraction_merge import (
    ExtractionProvenance,
    LogicalIdentity,
    MergedEdgeRecord,
    MergedEntityRecord,
    MergedExtraction,
)
from app.workers.pipeline import _serialize_for_audit


def _identity(
    entity_type="RADAR_SYSTEM",
    field_names=("system_name",),
    values=("Tombstone",),
    scope="global",
    document_id=None,
):
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=field_names,
        identity_tuple=values,
        scope=scope,
        document_id=document_id,
    )


def _manifest_with_radar_bundle():
    pass_def = SimpleNamespace(
        name="radar_domain",
        bridge_entity_types=["PLATFORM"],
    )
    return SimpleNamespace(bundle_key="air_defense_v3", passes=[pass_def])


def _merged_with_one_record(provenance=None) -> MergedExtraction:
    identity = _identity()
    record = MergedEntityRecord(
        identity=identity,
        properties={"system_name": "Tombstone"},
        confidence=1.0,
        pass_origins={"radar_domain"},
        display_label="Tombstone",
        provenance=provenance or [],
    )
    return MergedExtraction(
        entities=[record],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
    )


def _prov(**kwargs):
    base = dict(
        instance_id="uuid-1",
        ontology_name="RADAR_SYSTEM",
        identity_values={"system_name": "Tombstone"},
        element_uid="#/texts/0",
    )
    base.update(kwargs)
    return ExtractionProvenance(**base)


# ---------------------------------------------------------------------------
# nodes[] emission
# ---------------------------------------------------------------------------


def test_nodes_entry_per_merged_record_with_entity_id_and_rid_and_artifact_ids():
    identity = _identity()
    prov = _prov(element_uid="#/texts/5")
    merged = _merged_with_one_record(provenance=[prov])

    identity_to_rid = {identity: "#42:17"}
    element_uid_to_artifact_id = {"#/texts/5": "artifact-uuid-a"}

    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid, element_uid_to_artifact_id,
    )

    assert "nodes" in blob
    assert len(blob["nodes"]) == 1
    node = blob["nodes"][0]
    assert node["name"] == "Tombstone"
    assert node["entity_type"] == "RADAR_SYSTEM"
    assert node["entity_id"] == identity.serialize_as_entity_id()
    assert node["rid"] == "#42:17"
    assert node["artifact_ids"] == ["artifact-uuid-a"]


def test_nodes_artifact_ids_dedup_and_sort_for_determinism():
    identity = _identity()
    prov1 = _prov(instance_id="i1", element_uid="#/texts/1")
    prov2 = _prov(instance_id="i2", element_uid="#/texts/2")
    prov3 = _prov(instance_id="i3", element_uid="#/texts/3")
    merged = _merged_with_one_record(provenance=[prov1, prov2, prov3])

    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={identity: "#42:17"},
        element_uid_to_artifact_id={
            "#/texts/1": "B",
            "#/texts/2": "A",
            "#/texts/3": "B",  # duplicate artifact
        },
    )
    node = blob["nodes"][0]
    # Deduped ({A,B}) and sorted
    assert node["artifact_ids"] == ["A", "B"]


def test_nodes_artifact_ids_empty_when_no_provenance():
    identity = _identity()
    merged = _merged_with_one_record(provenance=[])

    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={identity: "#1:1"},
        element_uid_to_artifact_id={},
    )
    assert blob["nodes"][0]["artifact_ids"] == []


# ---------------------------------------------------------------------------
# mentions[] emission
# ---------------------------------------------------------------------------


def test_mentions_one_entry_per_provenance_row_with_entity_id_rid_element_uid():
    identity = _identity()
    prov1 = _prov(instance_id="i1", element_uid="#/texts/1", page=42, chunk_index=7)
    prov2 = _prov(instance_id="i2", element_uid="#/texts/2")
    merged = _merged_with_one_record(provenance=[prov1, prov2])

    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={identity: "#1:1"},
        element_uid_to_artifact_id={},
    )

    assert "mentions" in blob
    assert len(blob["mentions"]) == 2
    m1 = blob["mentions"][0]
    assert m1["entity_name"] == "Tombstone"
    assert m1["entity_type"] == "RADAR_SYSTEM"
    assert m1["entity_id"] == identity.serialize_as_entity_id()
    assert m1["rid"] == "#1:1"
    assert m1["element_uid"] == "#/texts/1"
    assert m1["page"] == 42
    assert m1["chunk_index"] == 7


def test_mention_rid_matches_corresponding_node_rid_consistency_invariant():
    """For any mention m and the matching node n (same entity_id),
    m['rid'] == n['rid']. Both derive from the same identity_to_rid
    map, so divergence would be a serializer bug."""
    identity = _identity()
    prov = _prov()
    merged = _merged_with_one_record(provenance=[prov])

    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={identity: "#33:5"},
        element_uid_to_artifact_id={},
    )
    node = blob["nodes"][0]
    mention = blob["mentions"][0]
    assert node["entity_id"] == mention["entity_id"]
    assert node["rid"] == mention["rid"] == "#33:5"


# ---------------------------------------------------------------------------
# element_to_artifact persistence
# ---------------------------------------------------------------------------


def test_element_to_artifact_persisted_in_audit_blob():
    """The element_uid_to_artifact_id map passed into _serialize_for_audit
    is written into graph_json under the 'element_to_artifact' key so
    derive_structure_links can read it from the persisted blob instead
    of re-querying Postgres (plan §53 snapshot-consistency contract)."""
    identity = _identity()
    merged = _merged_with_one_record()

    element_map = {"#/texts/0": "artifact-A", "#/texts/1": "artifact-B"}
    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={identity: "#1:1"},
        element_uid_to_artifact_id=element_map,
    )
    assert blob["element_to_artifact"] == element_map


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_extractions_emits_empty_nodes_and_mentions_no_exception():
    """A document that produced zero entities writes empty lists — the
    serializer must not raise, and downstream readers must see [] not
    KeyError."""
    merged = MergedExtraction(
        entities=[],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
    )
    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={},
        element_uid_to_artifact_id={},
    )
    assert blob["nodes"] == []
    assert blob["mentions"] == []
    assert blob["element_to_artifact"] == {}


def test_preserves_existing_audit_keys():
    """The new keys are additive — existing keys (primary_entities_total,
    bridge_entities_total, entity_count_by_type, edges_accepted,
    edges_rejected, rejection_reasons) must still be present."""
    merged = _merged_with_one_record()

    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={_identity(): "#1:1"},
        element_uid_to_artifact_id={},
    )
    for existing_key in (
        "primary_entities_total",
        "bridge_entities_total",
        "entity_count_by_type",
        "edges_accepted",
        "edges_rejected",
        "rejection_reasons",
    ):
        assert existing_key in blob, f"audit blob missing legacy key {existing_key}"


def test_same_name_same_type_distinct_identities_produce_distinct_entity_ids():
    """Two SECTION entities with identical heading but different
    section_number values have distinct entity_ids (plan Task 53's key
    bug fix for derive_structure_links fallback suppression)."""
    id1 = _identity(
        entity_type="SECTION",
        field_names=("section_number",),
        values=("3.2",),
        scope="document",
        document_id="doc-1",
    )
    id2 = _identity(
        entity_type="SECTION",
        field_names=("section_number",),
        values=("4.7",),
        scope="document",
        document_id="doc-1",
    )
    r1 = MergedEntityRecord(
        identity=id1, properties={"section_number": "3.2", "heading": "Overview"},
        confidence=1.0, pass_origins={"radar_domain"}, display_label="Overview",
    )
    r2 = MergedEntityRecord(
        identity=id2, properties={"section_number": "4.7", "heading": "Overview"},
        confidence=1.0, pass_origins={"radar_domain"}, display_label="Overview",
    )
    merged = MergedExtraction(
        entities=[r1, r2], edges=[], rejected_edges=[],
        rejections_by_pass={}, pipeline_run_id="run-1", document_id="doc-1",
    )
    blob = _serialize_for_audit(
        merged, _manifest_with_radar_bundle(),
        identity_to_rid={id1: "#42:1", id2: "#42:2"},
        element_uid_to_artifact_id={},
    )
    entity_ids = {n["entity_id"] for n in blob["nodes"]}
    assert len(entity_ids) == 2  # same display label, distinct entity_ids
