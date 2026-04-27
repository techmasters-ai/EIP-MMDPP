"""Phase B Session 1 — smoke harness for the field-group split (spec §5.2).

Hits live docling-graph at http://localhost:8002/extract-pass with 3
minimal DoclingDocuments. Each test parametrizes:
- pass_name to invoke
- source text
- target field
- acceptable [lower, upper] range bracketing the source-text value

Skipped when docling-graph is unreachable. Marked @pytest.mark.integration
so it stays out of the default pytest tests/unit invocation.

Range calibration policy (spec §5.2): ranges bracket the source-text
value with tolerance for unit-conversion rounding, NOT the model's
observed output. If a future model emits 3500 MHz for a doc that says
3000 MHz, this test SHOULD fail. Recalibrating to model output would
mask regressions.
"""
import os
import pytest
import requests

DOCLING_GRAPH_URL = os.environ.get(
    "DOCLING_GRAPH_URL", "http://localhost:8002/extract-pass"
)


def _build_doc(text: str) -> dict:
    """Minimal valid DoclingDocument with one paragraph.

    label: "text" — existing fixtures and service-injected text use
    this. "paragraph" may be accepted by some Docling versions but
    leads to debugging document-shape issues instead of measuring
    extraction.
    """
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "test-fansong-smoke",
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
    "pass_name,text,field,lower,upper",
    [
        ("radar_power_rf",
         "Fan Song transmitter peak power is 600 kW.",
         "tx_peak_power_kw", 400.0, 800.0),
        ("radar_power_rf",
         "Fan Song operates at 3000 MHz.",
         "nominal_rf_mhz", 2900.0, 3100.0),
        ("radar_antenna",
         "Fan Song antenna gain is 35 dBi.",
         "gain_dbi", 33.0, 37.0),
    ],
    ids=["power-600kW", "freq-3000MHz", "gain-35dBi"],
)
def test_radar_field_group_numeric_smoke(pass_name, text, field, lower, upper):
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
    radar_systems = pass_output.get("radar_systems", []) or []
    assert len(radar_systems) >= 1, (
        f"expected ≥1 radar_system; pass_output={pass_output!r}"
    )

    entity = next(
        (e for e in radar_systems
         if "Fan Song" in (e.get("system_name") or "")),
        None,
    )
    assert entity is not None, f"Fan Song not found; got {radar_systems!r}"

    value = entity.get(field)
    if value is None:
        print(f"\n--- FAILURE DEBUG: pass_output ---\n{pass_output}\n---")
        pytest.fail(
            f"{pass_name}.{field} was None; expected value in [{lower}, {upper}]"
        )
    assert isinstance(value, (int, float)), (
        f"{field} is {type(value).__name__}, want number; got {value!r}"
    )
    assert lower <= float(value) <= upper, (
        f"{field}={value} not in [{lower}, {upper}]"
    )
