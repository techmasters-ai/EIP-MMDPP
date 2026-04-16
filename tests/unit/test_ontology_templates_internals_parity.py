"""Parity tests for ontology_templates helper functions under both
``ONTOLOGY_SOURCE`` modes.

Plan v32 Task id 40 (Phase 4). The loader module hosts several
derived helpers — load_validation_matrix, build_entity_type_names,
build_relationship_type_names, get_ontology_cache_signature — that
all flow through ``load_ontology()``. Feature-flag dispatch must
propagate to them without code changes.

Note: ``get_ontology_cache_signature`` always reads the YAML bundle
signature (mtime) because it's a CACHE signature, not a content hash.
It stays stable across ONTOLOGY_SOURCE modes as long as the YAML file
doesn't change — which is the correct semantic for Phase 3-5 (YAML
still present).
"""
from __future__ import annotations

import pytest

from app.services.ontology_templates import (
    build_entity_type_names,
    build_relationship_type_names,
    invalidate_ontology_cache,
    load_validation_matrix,
)


@pytest.fixture(autouse=True)
def _flush_cache():
    invalidate_ontology_cache()
    yield
    invalidate_ontology_cache()


def _run_under(monkeypatch, source: str, fn):
    monkeypatch.setenv("ONTOLOGY_SOURCE", source)
    invalidate_ontology_cache()
    return fn()


def test_load_validation_matrix_parity(monkeypatch):
    yaml_matrix = _run_under(monkeypatch, "yaml", load_validation_matrix)
    pyd_matrix = _run_under(monkeypatch, "pydantic", load_validation_matrix)
    assert yaml_matrix == pyd_matrix


def test_build_entity_type_names_parity(monkeypatch):
    yaml_names = set(
        _run_under(monkeypatch, "yaml", build_entity_type_names)
    )
    pyd_names = set(
        _run_under(monkeypatch, "pydantic", build_entity_type_names)
    )
    assert yaml_names == pyd_names
    assert len(yaml_names) == 45


def test_build_relationship_type_names_parity(monkeypatch):
    yaml_names = set(
        _run_under(monkeypatch, "yaml", build_relationship_type_names)
    )
    pyd_names = set(
        _run_under(monkeypatch, "pydantic", build_relationship_type_names)
    )
    assert yaml_names == pyd_names
    assert len(yaml_names) == 50


def test_helpers_accept_caller_supplied_ontology(monkeypatch):
    """When a caller passes an ontology dict, the helpers must not
    re-enter load_ontology() — they operate on the provided dict."""
    custom = {
        "entity_types": [{"name": "CUSTOM_A"}, {"name": "CUSTOM_B"}],
        "relationship_types": [{"name": "REL_X"}],
    }
    for source in ("yaml", "pydantic"):
        monkeypatch.setenv("ONTOLOGY_SOURCE", source)
        invalidate_ontology_cache()
        assert set(build_entity_type_names(custom)) == {"CUSTOM_A", "CUSTOM_B"}
        assert build_relationship_type_names(custom) == ["REL_X"]
