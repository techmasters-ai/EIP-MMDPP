"""Test Step 2: properties propagation from radar/missile identity passes
into the upstream_entities sent to system_links.

Production path: `_collect_upstream_properties` harvests emitter_function
from scratch; `_extend_upstream_refs` puts it on the SimpleNamespace ref;
`_build_extract_pass_request` serializes it into the POST body so the
docling-graph service's `EntityRef.properties` field carries it.
"""
import sys
import types
from pathlib import Path

# Stub the workers.pipeline imports — we only need the two helpers.
# Import strategy: load the module under a partial-import guard.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.workers.pipeline import (
    _collect_upstream_properties,
    _UPSTREAM_PROPERTY_FIELDS,
)


def test_collect_emitter_function():
    scratch = {"emitter_function": "FIRE_CONTROL", "nomenclature": "SNR-75"}
    props = _collect_upstream_properties(scratch)
    assert props == {"emitter_function": "FIRE_CONTROL"}


def test_returns_none_when_no_properties():
    scratch = {"nomenclature": "SNR-75"}  # no emitter_function
    assert _collect_upstream_properties(scratch) is None


def test_returns_none_when_empty_emitter():
    scratch = {"emitter_function": ""}
    assert _collect_upstream_properties(scratch) is None


def test_returns_none_when_emitter_is_whitespace():
    scratch = {"emitter_function": "   "}
    assert _collect_upstream_properties(scratch) is None


def test_emitter_function_is_in_tracked_fields():
    """If this fails, someone changed _UPSTREAM_PROPERTY_FIELDS to remove
    emitter_function — that breaks role-aware CUES retype."""
    assert "emitter_function" in _UPSTREAM_PROPERTY_FIELDS


def test_request_body_includes_properties():
    """End-to-end: _build_extract_pass_request must serialize properties."""
    from app.workers.pipeline import _build_extract_pass_request

    pass_def = types.SimpleNamespace(name="system_links")
    ref = types.SimpleNamespace(
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
        aliases=["SNR-75"],
        properties={"emitter_function": "FIRE_CONTROL"},
    )
    upstream_refs = {"E001": ref}
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3",
        pass_def=pass_def,
        doc_json={},
        upstream_refs=upstream_refs,
        document_id="test-doc",
    )
    entities = body["upstream_entities"]
    assert len(entities) == 1
    assert entities[0]["properties"] == {"emitter_function": "FIRE_CONTROL"}


def test_request_body_omits_properties_when_none():
    """Backward-compat: if ref has no properties, don't include the field."""
    from app.workers.pipeline import _build_extract_pass_request

    pass_def = types.SimpleNamespace(name="system_links")
    ref = types.SimpleNamespace(
        entity_type="MISSILE_SYSTEM",
        identity_values={"system_name": "1D"},
        display_label="1D",
        aliases=None,
        properties=None,
    )
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3",
        pass_def=pass_def,
        doc_json={},
        upstream_refs={"E001": ref},
        document_id="test-doc",
    )
    assert "properties" not in body["upstream_entities"][0]
