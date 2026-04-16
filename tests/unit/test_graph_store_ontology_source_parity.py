"""Parity test for ``graph_store.py`` under both ``ONTOLOGY_SOURCE`` modes.

Plan v32 Task id 39 (Phase 4). ``graph_store`` is a Protocol-style
abstraction (``GraphStore.sync_schema`` receives ``ontology: dict``
from the caller). It does not call ``load_ontology()`` — callers
such as ``arcadedb_graph.AsyncArcadeDBGraphStore`` pass in whatever
dict they want. Parity at THIS module is therefore about the
contract, not about runtime behavior.
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
    from app.services import graph_store

    assert hasattr(graph_store, "SchemaSyncReport")


def test_graph_store_does_not_call_load_ontology():
    """graph_store.py must not embed ``load_ontology()`` — the caller
    (e.g. arcadedb_graph) owns the ontology dict and passes it in."""
    import inspect

    from app.services import graph_store

    source = inspect.getsource(graph_store)
    assert "load_ontology(" not in source, (
        "graph_store.py must not call load_ontology() — ontology "
        "dict flows in via sync_schema(ontology=...)"
    )
