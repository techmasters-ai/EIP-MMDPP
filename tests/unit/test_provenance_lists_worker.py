"""Task 3 — worker carries POSITIONAL lineage LISTS end-to-end.

Closes the "scalar-collapse boundary": docling-graph (Task 2) now emits
``ExtractionProvenance`` with ``self_refs`` / ``chunk_indexes`` /
``cited_refs`` LISTS in the /extract-pass response, but the worker parse
previously dropped them and only carried a single scalar
element_uid / chunk_index / page.

This module verifies, against the REAL worker parse + serialize logic:

  (a) ``_parse_pass_response`` populates self_refs / chunk_indexes /
      cited_refs on each worker ``ExtractionProvenance`` from the
      response row's lists, and the same for field-evidence rows.
  (b) Back-compat: a scalar-only response row (no self_refs key)
      defaults self_refs to [element_uid] and chunk_indexes to
      [chunk_index].
  (c) ``_serialize_for_audit`` mentions[] forward the
      self_refs / chunk_indexes / page_numbers LISTS alongside the
      preserved scalars.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.extraction_merge import (
    ExtractionProvenance,
    LogicalIdentity,
    MergedEntityRecord,
    MergedExtraction,
)
from app.workers.pipeline import _parse_pass_response, _serialize_for_audit


# ---------------------------------------------------------------------------
# Fixtures (mirror the established test patterns in
# test_pass_result_carries_provenance.py / test_serialize_for_audit_phase8.py)
# ---------------------------------------------------------------------------


def _pass_def_for_radar():
    return SimpleNamespace(
        name="radar_domain",
        module="extraction_schemas.radar_domain",
        template_class="RadarDomainPass",
    )


def _manifest_with_radar_bundle():
    return SimpleNamespace(bundle_key="air_defense_v3")


def _response_with_provenance(provenance_rows, field_provenance_rows=None) -> dict:
    return {
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_domain",
        "pass_output": {"radar_systems": [], "specifications": []},
        "metadata": {"schema_size_chars": 0, "structured_output_mode": "strict"},
        "provenance": provenance_rows,
        "field_provenance": field_provenance_rows or [],
    }


def _identity(
    entity_type="RADAR_SYSTEM",
    field_names=("system_name",),
    values=("Tombstone",),
):
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=field_names,
        identity_tuple=values,
        scope="global",
        document_id=None,
    )


def _manifest_for_audit():
    pass_def = SimpleNamespace(name="radar_domain", bridge_entity_types=[])
    return SimpleNamespace(bundle_key="air_defense_v3", passes=[pass_def])


# ---------------------------------------------------------------------------
# (a) parse populates the lists from the response row's lists
# ---------------------------------------------------------------------------


def test_parse_pass_response_carries_self_refs_chunk_indexes_cited_refs_lists():
    raw = _response_with_provenance([
        {
            "instance_id": "uuid-1",
            "ontology_name": "RADAR_SYSTEM",
            "identity_values": {"system_name": "Tombstone"},
            "element_uid": "#/texts/3",
            "self_refs": ["#/texts/3", "#/texts/4"],
            "chunk_indexes": [0, 1],
            "cited_refs": ["#/texts/3"],
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    assert len(result.provenance) == 1
    prov = result.provenance[0]
    assert prov.self_refs == ["#/texts/3", "#/texts/4"]
    assert prov.chunk_indexes == [0, 1]
    assert prov.cited_refs == ["#/texts/3"]
    # Scalars preserved.
    assert prov.element_uid == "#/texts/3"


def test_parse_pass_response_carries_field_evidence_lists():
    raw = _response_with_provenance(
        provenance_rows=[
            {
                "instance_id": "uuid-1",
                "ontology_name": "RADAR_SYSTEM",
                "identity_values": {"system_name": "Tombstone"},
                "element_uid": "#/texts/3",
            },
        ],
        field_provenance_rows=[
            {
                "instance_id": "uuid-1",
                "field_name": "frequency_band",
                "supporting_snippet": "operates in S-band",
                "element_uid": "#/texts/7",
                "self_refs": ["#/texts/7", "#/texts/8"],
                "chunk_indexes": [2, 3],
            },
        ],
    )

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    rows = result.field_evidence["uuid-1"]["frequency_band"]
    assert len(rows) == 1
    fe = rows[0]
    assert fe.self_refs == ["#/texts/7", "#/texts/8"]
    assert fe.chunk_indexes == [2, 3]
    # Scalar element_uid preserved.
    assert fe.element_uid == "#/texts/7"


# ---------------------------------------------------------------------------
# (b) back-compat: scalar-only response row
# ---------------------------------------------------------------------------


def test_parse_pass_response_backcompat_scalar_only_provenance_row():
    """Older service shape: no self_refs/chunk_indexes keys → derive the
    lists from the scalar element_uid / chunk_index."""
    raw = _response_with_provenance([
        {
            "instance_id": "uuid-5",
            "ontology_name": "RADAR_SYSTEM",
            "identity_values": {},
            "element_uid": "#/texts/5",
            "chunk_index": 9,
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    prov = result.provenance[0]
    assert prov.self_refs == ["#/texts/5"]
    assert prov.chunk_indexes == [9]
    assert prov.cited_refs == []


def test_parse_pass_response_backcompat_scalar_only_no_chunk_index():
    """element_uid present but chunk_index absent → chunk_indexes == []."""
    raw = _response_with_provenance([
        {
            "instance_id": "uuid-6",
            "ontology_name": "RADAR_SYSTEM",
            "identity_values": {},
            "element_uid": "#/texts/6",
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    prov = result.provenance[0]
    assert prov.self_refs == ["#/texts/6"]
    assert prov.chunk_indexes == []


def test_parse_pass_response_backcompat_scalar_only_field_evidence_row():
    raw = _response_with_provenance(
        provenance_rows=[
            {
                "instance_id": "uuid-1",
                "ontology_name": "RADAR_SYSTEM",
                "identity_values": {},
                "element_uid": "#/texts/3",
            },
        ],
        field_provenance_rows=[
            {
                "instance_id": "uuid-1",
                "field_name": "frequency_band",
                "supporting_snippet": "S-band",
                "element_uid": "#/texts/9",
            },
        ],
    )

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    fe = result.field_evidence["uuid-1"]["frequency_band"][0]
    assert fe.self_refs == ["#/texts/9"]
    assert fe.chunk_indexes == []


# ---------------------------------------------------------------------------
# (c) mentions[] forward the lists alongside preserved scalars
# ---------------------------------------------------------------------------


def test_mentions_forward_self_refs_chunk_indexes_page_numbers_lists():
    identity = _identity()
    prov = ExtractionProvenance(
        instance_id="uuid-1",
        ontology_name="RADAR_SYSTEM",
        identity_values={"system_name": "Tombstone"},
        element_uid="#/texts/3",
        page=42,
        chunk_index=0,
        self_refs=["#/texts/3", "#/texts/4"],
        chunk_indexes=[0, 1],
        page_numbers=[42, 43],
        cited_refs=["#/texts/3"],
    )
    record = MergedEntityRecord(
        identity=identity,
        properties={"system_name": "Tombstone"},
        confidence=1.0,
        pass_origins={"radar_domain"},
        display_label="Tombstone",
        provenance=[prov],
    )
    merged = MergedExtraction(
        entities=[record],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
    )

    blob = _serialize_for_audit(
        merged, _manifest_for_audit(),
        identity_to_rid={identity: "#1:1"},
        element_uid_to_artifact_id={},
    )

    assert len(blob["mentions"]) == 1
    m = blob["mentions"][0]
    # New lists forwarded.
    assert m["self_refs"] == ["#/texts/3", "#/texts/4"]
    assert m["chunk_indexes"] == [0, 1]
    assert m["page_numbers"] == [42, 43]
    # Scalars preserved, scalar element_uid == self_refs[0].
    assert m["element_uid"] == "#/texts/3"
    assert m["element_uid"] == m["self_refs"][0]
    assert m["page"] == 42
    assert m["chunk_index"] == 0
