"""Parity test for ``canonicalization.py`` under both ``ONTOLOGY_SOURCE`` modes.

Plan v32 Task 28 (Phase 4). The canonicalization module does NOT
consume the ontology dict — its public helpers take ``entity_type``
as a string argument from the pipeline. This test pins that contract:
the module import + helper signatures must remain ontology-source
agnostic.
"""
from __future__ import annotations

import pytest

from app.services.ontology_templates import invalidate_ontology_cache


@pytest.fixture(autouse=True)
def _flush_cache():
    invalidate_ontology_cache()
    yield
    invalidate_ontology_cache()


@pytest.fixture(params=["yaml", "pydantic"])
def ontology_source(request, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_SOURCE", request.param)
    return request.param


def test_module_imports_under_both_sources(ontology_source):
    from app.services import canonicalization

    assert hasattr(canonicalization, "canonicalize_entity")
    assert hasattr(canonicalization, "_exact_match")
    assert hasattr(canonicalization, "_fuzzy_match")


def test_canonicalization_does_not_read_ontology_dict():
    """The module must not import load_ontology — entity_type strings
    flow in as function arguments from the caller."""
    import inspect

    from app.services import canonicalization

    source = inspect.getsource(canonicalization)
    assert "load_ontology" not in source, (
        "canonicalization.py must stay ontology-source agnostic; "
        "entity_type values flow in as function arguments, not via "
        "load_ontology()"
    )
