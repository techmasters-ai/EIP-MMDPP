"""Parity tests for ``pipeline.py`` ontology consumers under both
``ONTOLOGY_SOURCE`` modes.

Plan v32 Task 27 (Phase 4). Lines 1831, 1868, 1928, 1965 read ontology
dict keys. Since build_ontology_dict() is canonical-JSON-equivalent to
the YAML load, these helpers should behave identically across modes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.ontology_templates import invalidate_ontology_cache, load_ontology
from app.workers.pipeline import (
    _endpoint_types_for_rel_types,
    _is_valid_upstream_ref,
)


@pytest.fixture(autouse=True)
def _flush_cache():
    invalidate_ontology_cache()
    yield
    invalidate_ontology_cache()


@pytest.fixture(params=["yaml", "pydantic"])
def ontology(request, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_SOURCE", request.param)
    return load_ontology()


def test_is_valid_upstream_ref_valid_section(ontology):
    ref = SimpleNamespace(
        entity_type="SECTION",
        identity_values={"heading": "Maintenance", "page_start": 5},
    )
    assert _is_valid_upstream_ref(ref, ontology) is True


def test_is_valid_upstream_ref_missing_identity_field(ontology):
    ref = SimpleNamespace(
        entity_type="SECTION",
        identity_values={"heading": "Maintenance"},  # missing page_start
    )
    assert _is_valid_upstream_ref(ref, ontology) is False


def test_is_valid_upstream_ref_empty_identity_fields(ontology):
    """DOCUMENT has graph_id_fields=[] — no anchors → not a valid upstream ref."""
    ref = SimpleNamespace(
        entity_type="DOCUMENT",
        identity_values={"title": "x"},
    )
    assert _is_valid_upstream_ref(ref, ontology) is False


def test_is_valid_upstream_ref_unknown_type(ontology):
    ref = SimpleNamespace(entity_type="UNKNOWN_X", identity_values={"k": "v"})
    assert _is_valid_upstream_ref(ref, ontology) is False


def test_endpoint_types_for_rel_types_installed_on(ontology):
    """INSTALLED_ON relationship has many sources; target is always PLATFORM."""
    endpoints = _endpoint_types_for_rel_types(ontology, ["INSTALLED_ON"])
    assert "PLATFORM" in endpoints
    assert "RADAR_SYSTEM" in endpoints
    assert "MISSILE_SYSTEM" in endpoints


def test_endpoint_types_for_rel_types_empty_rel_list(ontology):
    assert _endpoint_types_for_rel_types(ontology, []) == set()


def test_endpoint_types_parity_across_modes(monkeypatch):
    """For every relationship type, the set of endpoint entity types must
    match between yaml and pydantic modes."""
    monkeypatch.setenv("ONTOLOGY_SOURCE", "yaml")
    yaml_ont = load_ontology()
    monkeypatch.setenv("ONTOLOGY_SOURCE", "pydantic")
    pyd_ont = load_ontology()

    rel_types = {r["name"] for r in yaml_ont["relationship_types"]}
    for rel in sorted(rel_types):
        yaml_endpoints = _endpoint_types_for_rel_types(yaml_ont, [rel])
        pyd_endpoints = _endpoint_types_for_rel_types(pyd_ont, [rel])
        assert yaml_endpoints == pyd_endpoints, (
            f"endpoint-type divergence for {rel}: "
            f"yaml={yaml_endpoints} pyd={pyd_endpoints}"
        )
