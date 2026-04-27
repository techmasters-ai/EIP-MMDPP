"""Cross-unit smoke for the radar field-group split (Option A landed).

Mirrors tests/integration/test_radar_field_groups_smoke.py but exercises
the cross-unit conversion path that was added to _numeric_evidence.py
per docs/superpowers/plans/2026-04-27-cross-unit-conversion-followup-todo.md.

Cases probe values stated in non-canonical units that the LLM must
extract in the field's canonical unit; the §4.8 evidence gate then
verifies via cross-unit candidates ("10 GHz" for nominal_rf_mhz=10000).

The fixture text is the "Tall King + Tombstone" two-paragraph harder-test
from the cross-unit TODO investigation. It also probes:
- FORBIDDEN-name decoy (mentions SA-3, SA-5)
- Multi-system separation (two distinct radars in one input)
- Numeric collisions (multiple "1.0" values across fields)
- Single-token identity (Tombstone)

This test locks in the regression: before Option A, the Tombstone power_rf
fields nulled because the LLM emitted canonical-unit values (10000 MHz,
1400 kW) that the same-unit-suffix matcher couldn't find in source text
that used "10 GHz" / "1.4 megawatts". After Option A, the matcher
generates cross-unit candidates and preserves them.

Marked @pytest.mark.integration; skipped when service is offline.
"""
import os
import pytest
import requests

DOCLING_GRAPH_URL = os.environ.get(
    "DOCLING_GRAPH_URL", "http://localhost:8002/extract-pass"
)


HARDER_TEST_TEXT = (
    "Tall King, GRAU index 1RL117, is a long-range Soviet 2D early-warning "
    "radar typically deployed alongside SA-3 and SA-5 surface-to-air missile "
    "batteries to provide initial target acquisition for downstream "
    "fire-control systems. Assigned ELNOT BAR LOCK, Tall King operates as a "
    "SEARCH emitter and remains OPERATIONAL across former Warsaw Pact "
    "operators. The system is reviewed annually under MDE oversight by NASIC "
    "and uses a CIRCULAR mechanical scan, completing one full sweep every "
    "10 seconds. The 35-meter horizontal aperture array provides 30 dBi peak "
    "gain with a 4 degree azimuth beamwidth.\n"
    "\n"
    "In contrast, Tombstone (30N6E2) is a multi-function X-band fire-control "
    "radar operating at a nominal carrier frequency of 10 GHz with reported "
    "peak transmitter output of 1.4 megawatts. Its phased-array antenna "
    "delivers 40 dBi gain with a 1 degree beamwidth in both azimuth and "
    "elevation, dramatically narrower than Tall King's beam. Elevation "
    "coverage extends to 60 degrees. Pulse parameters: pulse duration 1 "
    "microsecond, pulse repetition interval 1000 microseconds, 100 pulses "
    "integrated per beam-position dwell. Intra-pulse modulation is "
    "LFM_CHIRP across a 1 MHz frequency excursion, with inter-pulse "
    "modulation CONSTANT_PRI."
)


def _build_doc(text: str) -> dict:
    """Minimal valid DoclingDocument with one paragraph (label='text')."""
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "test-radar-cross-unit",
        "origin": {
            "mimetype": "text/plain",
            "binary_hash": 1,
            "filename": "harder.txt",
        },
        "furniture": {
            "name": "_root_", "self_ref": "#/furniture", "children": [],
        },
        "body": {
            "name": "_root_", "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}],
        },
        "groups": [], "pictures": [], "tables": [],
        "key_value_items": [], "form_items": [], "pages": {},
        "texts": [{
            "self_ref": "#/texts/0",
            "parent": {"$ref": "#/body"},
            "label": "text",
            "prov": [],
            "orig": text,
            "text": text,
        }],
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    "system_name,field,expected",
    [
        # The two cases that regressed BEFORE Option A:
        ("Tombstone", "nominal_rf_mhz", 10000.0),   # source: "10 GHz"
        ("Tombstone", "tx_peak_power_kw", 1400.0),  # source: "1.4 megawatts"
    ],
    ids=["10GHz-to-MHz", "1.4MW-to-kW"],
)
def test_radar_power_rf_cross_unit_extraction(system_name, field, expected):
    """Regression: cross-unit values must populate after Option A."""
    body = {
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_power_rf",
        "document_id": f"cross-unit-radar_power_rf-{field}",
        "docling_document_json": _build_doc(HARDER_TEST_TEXT),
    }
    try:
        resp = requests.post(DOCLING_GRAPH_URL, json=body, timeout=180)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"docling-graph not available at {DOCLING_GRAPH_URL}")

    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    payload = resp.json()
    pass_output = payload.get("pass_output", {})
    radar_systems = pass_output.get("radar_systems", []) or []
    entity = next(
        (e for e in radar_systems if (e.get("system_name") or "") == system_name),
        None,
    )
    assert entity is not None, (
        f"{system_name} not found in radar_power_rf pass output; "
        f"got {[e.get('system_name') for e in radar_systems]!r}"
    )

    value = entity.get(field)
    assert value is not None, (
        f"{field}={expected} expected (cross-unit conversion from source); "
        f"got None — Option A regressed?"
    )
    assert value == expected, (
        f"{field}={value} but expected exactly {expected} "
        f"(source text states value in non-canonical unit; LLM must "
        f"convert + evidence-gate must preserve via cross-unit candidate)"
    )


@pytest.mark.integration
def test_radar_identity_no_forbidden_name_leak():
    """Sanitizer drops SA-3/SA-5 even though they appear in operational context."""
    body = {
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_identity",
        "document_id": "cross-unit-radar_identity-forbidden-decoy",
        "docling_document_json": _build_doc(HARDER_TEST_TEXT),
    }
    try:
        resp = requests.post(DOCLING_GRAPH_URL, json=body, timeout=180)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"docling-graph not available at {DOCLING_GRAPH_URL}")

    assert resp.status_code == 200
    payload = resp.json()
    radar_systems = payload.get("pass_output", {}).get("radar_systems") or []
    names = {(e.get("system_name") or "") for e in radar_systems}
    # Must NOT contain SA-* (those are SAM systems, not radars).
    forbidden_hits = {n for n in names if n.startswith("SA-")}
    assert forbidden_hits == set(), (
        f"sanitizer should drop SA-* identities; found {forbidden_hits!r} "
        f"in {names!r}"
    )
    # Should contain the legitimate radars.
    assert "Tall King" in names
    assert "Tombstone" in names


@pytest.mark.integration
def test_radar_multi_system_separation():
    """Tall King and Tombstone are distinct entities — never merged."""
    body = {
        "bundle_key": "air_defense_v3",
        "pass_name": "radar_antenna",
        "document_id": "cross-unit-radar_antenna-multi-system",
        "docling_document_json": _build_doc(HARDER_TEST_TEXT),
    }
    try:
        resp = requests.post(DOCLING_GRAPH_URL, json=body, timeout=180)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"docling-graph not available at {DOCLING_GRAPH_URL}")

    payload = resp.json()
    radar_systems = payload.get("pass_output", {}).get("radar_systems") or []
    names = {(e.get("system_name") or "") for e in radar_systems}
    assert "Tall King" in names
    assert "Tombstone" in names
    assert len(names) >= 2
