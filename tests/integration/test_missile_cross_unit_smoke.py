"""Cross-unit smoke for the missile field-group split (Option A landed).

Mirrors tests/integration/test_radar_cross_unit_smoke.py for the missile
bundle. Probes cross-unit conversion paths added to _numeric_evidence.py
per docs/superpowers/plans/2026-04-27-cross-unit-conversion-followup-todo.md.

Cases probe values stated in non-canonical units that the LLM must
extract in the field's canonical unit; the §4.8 evidence gate then
verifies via cross-unit candidates ("1.5 tonnes" for total_mass_kg=1500).

Marked @pytest.mark.integration; skipped when service is offline.
"""
import os
import pytest
import requests

DOCLING_GRAPH_URL = os.environ.get(
    "DOCLING_GRAPH_URL", "http://localhost:8002/extract-pass"
)


def _build_doc(text: str) -> dict:
    """Minimal valid DoclingDocument with one paragraph (label='text')."""
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "test-missile-cross-unit",
        "origin": {
            "mimetype": "text/plain",
            "binary_hash": 1,
            "filename": "smoke.txt",
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
    "pass_name,text,system_name,field,expected",
    [
        # Cross-unit: tonnes → kg (the named test case from the cross-unit TODO)
        ("missile_airframe",
         "The 5V55K missile total launch weight is 1.5 tonnes.",
         "5V55K", "total_mass_kg", 1500.0),
        # Cross-unit: meters → km (range stated in m, field in km)
        ("missile_kinematics",
         "The 9M82 missile maximum intercept range is 43000 m.",
         "9M82", "max_intercept_km", 43.0),
    ],
    ids=["1.5tonnes-to-kg", "43000m-to-km"],
)
def test_missile_cross_unit_extraction(
    pass_name, text, system_name, field, expected
):
    """Regression: missile cross-unit values must populate after Option A."""
    body = {
        "bundle_key": "air_defense_v3",
        "pass_name": pass_name,
        "document_id": f"cross-unit-{pass_name}-{field}",
        "docling_document_json": _build_doc(text),
    }
    try:
        resp = requests.post(DOCLING_GRAPH_URL, json=body, timeout=180)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"docling-graph not available at {DOCLING_GRAPH_URL}")

    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    payload = resp.json()
    pass_output = payload.get("pass_output", {})
    missile_systems = pass_output.get("missile_systems", []) or []
    entity = next(
        (e for e in missile_systems if (e.get("system_name") or "") == system_name),
        None,
    )
    assert entity is not None, (
        f"{system_name} not found in {pass_name} pass output; "
        f"got {[e.get('system_name') for e in missile_systems]!r}"
    )

    value = entity.get(field)
    assert value is not None, (
        f"{field}={expected} expected (cross-unit conversion); got None"
    )
    assert value == expected, (
        f"{field}={value} but expected exactly {expected} "
        f"(source text states value in non-canonical unit)"
    )
