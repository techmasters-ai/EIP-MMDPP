"""Phase 3 (post-LLM-quote refactor) — deterministic auto-evidence.

The build_auto_field_evidence helper substring-matches each populated
field's value against the DoclingDocument chunks fed to the LLM and
emits ExtractionFieldProvenance rows for every match. Replaces the
LLM-driven verbatim-quote path; works regardless of LLM compliance.
"""
import importlib.util
import pathlib
import sys

_SERVICE_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "docker" / "docling-graph" / "app"
)


def _load(modname: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_provenance = _load("_dgp_provenance", _SERVICE_ROOT / "provenance.py")
_schemas = _load("_dgp_schemas", _SERVICE_ROOT / "schemas.py")

build_auto_field_evidence = _provenance.build_auto_field_evidence
ExtractionFieldProvenance = _schemas.ExtractionFieldProvenance


def _radar_entity(**props):
    """Minimal RadarSystemEntity-like instance for tests.
    Uses the canonical class so json_schema_extra is real."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    return RadarSystemEntity(**props)


def test_auto_evidence_matches_string_value_in_chunk():
    entity = _radar_entity(system_name="Fan Song", scan_type="phased-array")
    chunks = [
        ("#/texts/0", "The Fan Song uses a phased-array antenna."),
        ("#/texts/1", "Unrelated text about other systems."),
    ]
    rows = build_auto_field_evidence(
        primary_entities=[entity],
        instance_ids=["i1"],
        input_chunks=chunks,
        skip_fields={"system_name", "confidence"},
        provenance_cls=ExtractionFieldProvenance,
    )
    scan = [r for r in rows if r.field_name == "scan_type"]
    assert len(scan) == 1
    assert scan[0].element_uid == "#/texts/0"
    assert "phased-array" in scan[0].supporting_snippet


def test_auto_evidence_numeric_with_unit_suffix():
    entity = _radar_entity(system_name="Fan Song", gain_dbi=35.0)
    chunks = [
        ("#/texts/0", "The antenna gain is 35 dBi nominal."),
    ]
    rows = build_auto_field_evidence(
        primary_entities=[entity],
        instance_ids=["i1"],
        input_chunks=chunks,
        skip_fields={"system_name", "confidence"},
        provenance_cls=ExtractionFieldProvenance,
    )
    gain = [r for r in rows if r.field_name == "gain_dbi"]
    assert len(gain) == 1
    assert gain[0].element_uid == "#/texts/0"


def test_auto_evidence_multiple_chunks_match():
    entity = _radar_entity(system_name="Fan Song", nominal_rf_mhz=3000.0)
    chunks = [
        ("#/texts/0", "Operating at 3000 MHz."),
        ("#/texts/1", "Center freq 3000 MHz with bandwidth..."),
    ]
    rows = build_auto_field_evidence(
        primary_entities=[entity],
        instance_ids=["i1"],
        input_chunks=chunks,
        skip_fields={"system_name", "confidence"},
        provenance_cls=ExtractionFieldProvenance,
    )
    rf = [r for r in rows if r.field_name == "nominal_rf_mhz"]
    # Both chunks contain "3000 MHz" — both attached as evidence.
    assert len(rf) == 2
    assert {r.element_uid for r in rf} == {"#/texts/0", "#/texts/1"}


def test_auto_evidence_falls_back_to_batch_when_no_match():
    entity = _radar_entity(system_name="Fan Song", scan_type="phased-array")
    chunks = [
        ("#/texts/0", "No relevant text here."),
        ("#/texts/1", "Some other unrelated content."),
    ]
    rows = build_auto_field_evidence(
        primary_entities=[entity],
        instance_ids=["i1"],
        input_chunks=chunks,
        skip_fields={"system_name", "confidence"},
        provenance_cls=ExtractionFieldProvenance,
    )
    scan = [r for r in rows if r.field_name == "scan_type"]
    # Fallback: batch-level — still gets evidence rows attached.
    assert len(scan) >= 1


def test_auto_evidence_skips_none_values():
    entity = _radar_entity(system_name="Fan Song")  # no other fields
    chunks = [("#/texts/0", "Some text.")]
    rows = build_auto_field_evidence(
        primary_entities=[entity],
        instance_ids=["i1"],
        input_chunks=chunks,
        skip_fields={"system_name", "confidence"},
        provenance_cls=ExtractionFieldProvenance,
    )
    # Only system_name was populated; it's in skip_fields → no rows.
    assert rows == []


def test_auto_evidence_skips_skipfields():
    entity = _radar_entity(system_name="Fan Song", scan_type="phased-array")
    chunks = [("#/texts/0", "Fan Song uses phased-array scan.")]
    rows = build_auto_field_evidence(
        primary_entities=[entity],
        instance_ids=["i1"],
        input_chunks=chunks,
        # Explicitly skip scan_type to ensure it's honored.
        skip_fields={"system_name", "confidence", "scan_type"},
        provenance_cls=ExtractionFieldProvenance,
    )
    assert all(r.field_name not in {"system_name", "scan_type"} for r in rows)
