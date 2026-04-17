"""Phase 8 Task 52 — worker-side propagation of provenance.

_parse_pass_response reads the ``provenance`` list from the docling-
graph /extract-pass response JSON and attaches it to ``PassResult.provenance``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _minimal_radar_response_json() -> dict:
    """Minimum viable response the worker-side parser accepts."""
    return {
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_domain",
        "pass_output": {
            "radar_systems": [],
            "specifications": [],
        },
        "metadata": {"schema_size_chars": 0, "structured_output_mode": "strict"},
        "provenance": [
            {
                "instance_id": "uuid-1",
                "ontology_name": "RADAR_SYSTEM",
                "identity_values": {"system_name": "Tombstone"},
                "element_uid": "#/texts/5",
                "page": 42,
                "chunk_index": 7,
            },
            {
                "instance_id": "uuid-2",
                "ontology_name": "PROPULSION_STACK",
                "identity_values": {},
                "element_uid": "#/texts/12",
            },
        ],
    }


def _pass_def_for_radar():
    return SimpleNamespace(
        name="radar_domain",
        module="extraction_schemas.radar_domain",
        template_class="RadarDomainPass",
    )


def _manifest_with_radar_bundle():
    return SimpleNamespace(bundle_key="air_defense_v3")


def test_extraction_provenance_dataclass_importable():
    from app.services.extraction_merge import ExtractionProvenance  # noqa: F401


def test_pass_result_has_provenance_field_defaulting_empty():
    """PassResult must accept an optional provenance argument defaulting
    to an empty list — existing test fixtures that don't pass one must
    not break."""
    from app.services.extraction_merge import (
        ExtractionMetadata,
        PassResult,
    )

    result = PassResult(
        pass_name="radar_domain",
        template_instance=SimpleNamespace(radar_systems=[]),
        metadata=ExtractionMetadata(
            schema_size_chars=0, structured_output_mode="strict",
        ),
        pre_merge_rejections=[],
    )
    assert result.provenance == []


def test_parse_pass_response_carries_provenance_through_to_pass_result():
    from app.workers.pipeline import _parse_pass_response

    result = _parse_pass_response(
        _minimal_radar_response_json(),
        _pass_def_for_radar(),
        _manifest_with_radar_bundle(),
    )
    assert len(result.provenance) == 2
    first = result.provenance[0]
    assert first.instance_id == "uuid-1"
    assert first.ontology_name == "RADAR_SYSTEM"
    assert first.identity_values == {"system_name": "Tombstone"}
    assert first.element_uid == "#/texts/5"
    assert first.page == 42
    assert first.chunk_index == 7
    # Second row is a component with empty identity_values.
    assert result.provenance[1].identity_values == {}
    assert result.provenance[1].element_uid == "#/texts/12"


def test_parse_pass_response_empty_provenance_yields_empty_list():
    """Response without a ``provenance`` key (e.g. from an older service
    build) must parse cleanly with provenance=[]."""
    from app.workers.pipeline import _parse_pass_response

    raw = _minimal_radar_response_json()
    del raw["provenance"]

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )
    assert result.provenance == []


def test_parse_pass_response_skips_malformed_provenance_rows():
    """Rows missing required fields (instance_id / ontology_name /
    element_uid) are dropped — the worker must survive service-side
    drift without failing the whole pass."""
    from app.workers.pipeline import _parse_pass_response

    raw = _minimal_radar_response_json()
    raw["provenance"] = [
        {"instance_id": "uuid-ok", "ontology_name": "RADAR_SYSTEM",
         "identity_values": {"system_name": "X"}, "element_uid": "#/texts/0"},
        {"ontology_name": "RADAR_SYSTEM", "identity_values": {},
         "element_uid": "#/texts/1"},  # missing instance_id
        {"instance_id": "uuid-no-uid", "ontology_name": "RADAR_SYSTEM",
         "identity_values": {}},  # missing element_uid
    ]

    result = _parse_pass_response(
        raw, _pass_def_for_radar(), _manifest_with_radar_bundle(),
    )
    assert len(result.provenance) == 1
    assert result.provenance[0].instance_id == "uuid-ok"
