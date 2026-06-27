"""Field lineage is CHUNK lineage (element_uid → chunk), not prose.

Regression for the verified bug: ``_parse_pass_response`` dropped EVERY
field_provenance row whose ``supporting_snippet`` was empty, even when the
row carried a resolvable chunk anchor (``element_uid`` / ``self_refs``).
The docling-graph service emits such rows in real runs, so the drop
collapsed ``field_evidence`` to empty → ``merge_and_resolve`` skipped field
aggregation → 0 field-lineage rows committed.

A field_provenance row that has a chunk anchor is USEFUL even with an empty
snippet. A row with NEITHER snippet NOR any chunk anchor is truly useless
and is still dropped.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.workers.pipeline import _parse_pass_response


def _pass_def_for_radar():
    return SimpleNamespace(
        name="radar_domain",
        module="extraction_schemas.radar_domain",
        template_class="RadarDomainPass",
    )


def _manifest_with_radar_bundle():
    return SimpleNamespace(bundle_key="air_defense_v3")


def _response_with_field_provenance(field_provenance_rows) -> dict:
    return {
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_domain",
        "pass_output": {"radar_systems": [], "specifications": []},
        "metadata": {"schema_size_chars": 0, "structured_output_mode": "strict"},
        "provenance": [
            {
                "instance_id": "uuid-1",
                "ontology_name": "RADAR_SYSTEM",
                "identity_values": {"system_name": "Tombstone"},
                "element_uid": "#/texts/3",
            },
        ],
        "field_provenance": field_provenance_rows,
    }


def test_empty_snippet_with_element_uid_is_kept():
    """A field_provenance row with instance_id+field_name+element_uid but
    an empty supporting_snippet carries a resolvable chunk anchor and MUST
    be kept (field lineage is chunk lineage, not prose)."""
    raw = _response_with_field_provenance([
        {
            "instance_id": "uuid-1",
            "field_name": "frequency_band",
            "supporting_snippet": "",
            "element_uid": "#/texts/12",
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    rows = result.field_evidence["uuid-1"]["frequency_band"]
    assert len(rows) == 1
    fe = rows[0]
    assert fe.element_uid == "#/texts/12"
    assert fe.snippet == ""
    # element_uid back-fills self_refs.
    assert fe.self_refs == ["#/texts/12"]


def test_empty_snippet_with_self_refs_is_kept():
    """No element_uid, but self_refs carries a chunk anchor → kept."""
    raw = _response_with_field_provenance([
        {
            "instance_id": "uuid-1",
            "field_name": "frequency_band",
            "supporting_snippet": "",
            "self_refs": ["#/texts/20", "#/texts/21"],
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    rows = result.field_evidence["uuid-1"]["frequency_band"]
    assert len(rows) == 1
    assert rows[0].self_refs == ["#/texts/20", "#/texts/21"]
    assert rows[0].snippet == ""


def test_empty_snippet_no_anchor_is_dropped():
    """Truly useless row: empty snippet AND no element_uid AND no
    self_refs → still dropped."""
    raw = _response_with_field_provenance([
        {
            "instance_id": "uuid-1",
            "field_name": "frequency_band",
            "supporting_snippet": "",
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    assert result.field_evidence == {}


def test_present_snippet_still_kept():
    """A row with a present snippet is unchanged by the relaxed condition."""
    raw = _response_with_field_provenance([
        {
            "instance_id": "uuid-1",
            "field_name": "frequency_band",
            "supporting_snippet": "operates in S-band",
            "element_uid": "#/texts/7",
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    rows = result.field_evidence["uuid-1"]["frequency_band"]
    assert len(rows) == 1
    assert rows[0].snippet == "operates in S-band"
    assert rows[0].element_uid == "#/texts/7"


def test_missing_instance_id_or_field_name_still_dropped():
    """Even with an element_uid, a row missing instance_id or field_name is
    still dropped (those are genuinely required)."""
    raw = _response_with_field_provenance([
        {
            # no instance_id
            "field_name": "frequency_band",
            "supporting_snippet": "",
            "element_uid": "#/texts/12",
        },
        {
            "instance_id": "uuid-1",
            # no field_name
            "supporting_snippet": "",
            "element_uid": "#/texts/13",
        },
    ])

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )

    assert result.field_evidence == {}
