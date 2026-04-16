"""Parity test for ``dossier_service.py`` under both ``ONTOLOGY_SOURCE`` modes.

Plan v32 Task id 38 (Phase 4). ``dossier_service`` does NOT call
``load_ontology()``; ``ROOT_ENTITY_TYPES`` is a static module-level
list and ``entity_type`` values flow in via function arguments or
attribute access on passed-in objects. Pin the contract.
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
    from app.services import dossier_service

    assert hasattr(dossier_service, "ROOT_ENTITY_TYPES")
    assert isinstance(dossier_service.ROOT_ENTITY_TYPES, list)
    assert dossier_service.ROOT_ENTITY_TYPES


def test_dossier_service_does_not_read_ontology_dict():
    """Source inspection: no ``load_ontology()`` call anywhere in
    the module — ``entity_type`` values must flow through callers."""
    import inspect

    from app.services import dossier_service

    source = inspect.getsource(dossier_service)
    assert "load_ontology" not in source, (
        "dossier_service.py must stay ontology-source agnostic"
    )
