"""Parity tests for ``query_profiles.py`` under both ``ONTOLOGY_SOURCE`` modes.

Plan v32 Task 29 (Phase 4). ``_ontology_subset`` at line 143 reads
``version`` / ``entity_types`` / ``relationship_types`` /
``validation_matrix`` via ``load_ontology()``. The downstream
``build_default_registry_template`` filters known type names against
this subset — any name-set divergence would produce different default
registry templates.
"""
from __future__ import annotations

import pytest

from app.services.ontology_templates import invalidate_ontology_cache


@pytest.fixture(autouse=True)
def _flush_cache():
    invalidate_ontology_cache()
    yield
    invalidate_ontology_cache()


def _ontology_subset_for(monkeypatch, source: str) -> dict:
    monkeypatch.setenv("ONTOLOGY_SOURCE", source)
    invalidate_ontology_cache()
    from app.services.query_profiles import _ontology_subset

    return _ontology_subset(repository_only=False)


def test_ontology_subset_name_sets_match(monkeypatch):
    """Entity and relationship type name SETS must match under both modes.
    Order is not part of the contract — _filter_known uses set semantics."""
    yaml_subset = _ontology_subset_for(monkeypatch, "yaml")
    pyd_subset = _ontology_subset_for(monkeypatch, "pydantic")

    yaml_entities = {e["name"] for e in yaml_subset["entity_types"]}
    pyd_entities = {e["name"] for e in pyd_subset["entity_types"]}
    assert yaml_entities == pyd_entities

    yaml_rels = {r["name"] for r in yaml_subset["relationship_types"]}
    pyd_rels = {r["name"] for r in pyd_subset["relationship_types"]}
    assert yaml_rels == pyd_rels


def test_ontology_subset_version_matches(monkeypatch):
    yaml_subset = _ontology_subset_for(monkeypatch, "yaml")
    pyd_subset = _ontology_subset_for(monkeypatch, "pydantic")
    assert yaml_subset["version"] == pyd_subset["version"]


def test_ontology_subset_validation_matrix_set_matches(monkeypatch):
    """validation_matrix triple-set (deduped) must match across modes."""
    yaml_subset = _ontology_subset_for(monkeypatch, "yaml")
    pyd_subset = _ontology_subset_for(monkeypatch, "pydantic")

    yaml_triples = {
        (e["source"], e["relationship"], e["target"])
        for e in yaml_subset["validation_matrix"]
    }
    pyd_triples = {
        (e["source"], e["relationship"], e["target"])
        for e in pyd_subset["validation_matrix"]
    }
    assert yaml_triples == pyd_triples


def test_build_default_registry_template_runs_under_both_sources(monkeypatch):
    """build_default_registry_template reads repository ontology only
    (YAML), so it's source-invariant at the template step. Still
    exercise under both env-var values to catch import regressions."""
    for source in ("yaml", "pydantic"):
        monkeypatch.setenv("ONTOLOGY_SOURCE", source)
        invalidate_ontology_cache()
        from app.services.query_profiles import build_default_registry_template

        template = build_default_registry_template()
        assert template is not None
