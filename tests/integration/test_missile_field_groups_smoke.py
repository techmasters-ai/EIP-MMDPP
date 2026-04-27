"""Smoke harness for missile field-group extraction (mirror of
tests/integration/test_radar_field_groups_smoke.py).

Three known cases that exercise the numeric extraction the field-group
split is intended to improve. Each test parametrizes:
- pass_name to invoke
- source text containing the proper-noun missile name + numeric value
- target field
- exact-match system_name
- acceptable [lower, upper] range bracketing the source-text value

Marked @pytest.mark.integration; default `pytest tests/unit tests/pipeline`
does not pick this up — run explicitly with the marker.

Range calibration policy: ranges bracket the source-text value with
tolerance for unit-conversion rounding, NOT the model's observed output.
If a future model emits 5000 km for a doc that says 43 km, this test
SHOULD fail. Recalibrating to model output would mask regressions.
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
        "name": "test-missile-smoke",
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
    "pass_name,text,system_name,field,lower,upper",
    [
        ("missile_kinematics",
         "The 5V55K missile has a maximum intercept range of 43 km.",
         "5V55K", "max_intercept_km", 30.0, 60.0),
        ("missile_speed_timing",
         "The 5V28 missile achieves a maximum speed of 1200 m/s.",
         "5V28", "max_speed_mps", 800.0, 1500.0),
        ("missile_airframe",
         "The 9M82 missile body length is 7.5 m.",
         "9M82", "body_length_m", 4.0, 9.0),
    ],
    ids=["5V55K-range-43km", "5V28-speed-1200mps", "9M82-length-7.5m"],
)
def test_missile_field_group_numeric_smoke(
    pass_name, text, system_name, field, lower, upper
):
    body = {
        "bundle_key": "air_defense_v3",
        "pass_name": pass_name,
        "document_id": f"smoke-{pass_name}-{field}",
        "docling_document_json": _build_doc(text),
        # NOTE: omit upstream_entities entirely for document_only passes;
        # the endpoint rejects document_only requests when the key is
        # present (even with an empty list).
    }
    try:
        resp = requests.post(DOCLING_GRAPH_URL, json=body, timeout=180)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"docling-graph not available at {DOCLING_GRAPH_URL}")

    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    payload = resp.json()
    pass_output = payload.get("pass_output", {})
    missile_systems = pass_output.get("missile_systems", []) or []
    assert len(missile_systems) >= 1, (
        f"expected >=1 missile_system; pass_output={pass_output!r}"
    )

    # Exact-match on system_name (not substring) — missile names like
    # 5V55K / 5V28 / 9M82 are unambiguous identifiers, unlike radar's
    # multi-token "Fan Song". Substring matching could collide with
    # near-similar designations.
    entity = next(
        (e for e in missile_systems
         if (e.get("system_name") or "") == system_name),
        None,
    )
    assert entity is not None, (
        f"{system_name} not found; got system_names="
        f"{[e.get('system_name') for e in missile_systems]!r}"
    )

    value = entity.get(field)
    if value is None:
        print(f"\n--- FAILURE DEBUG: pass_output ---\n{pass_output}\n---")
        pytest.fail(
            f"{pass_name}.{field} was None for {system_name}; "
            f"expected value in [{lower}, {upper}]"
        )
    assert isinstance(value, (int, float)), (
        f"{field} is {type(value).__name__}, want number; got {value!r}"
    )
    assert lower <= float(value) <= upper, (
        f"{field}={value} not in [{lower}, {upper}]"
    )
