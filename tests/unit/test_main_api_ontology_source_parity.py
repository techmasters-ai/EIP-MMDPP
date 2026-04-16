"""Parity tests for main/API consumers of ``load_ontology`` under both
``ONTOLOGY_SOURCE`` modes.

Plan v32 Task id 41 (Phase 4). Two consumers:
- ``app.main`` lifespan loads the ontology once and hands it to
  ``graph_store.sync_schema(ontology)``. Since sync_schema receives
  the dict, the important parity contract is that ``load_ontology()``
  returns canonical-JSON-equivalent data (already pinned by Task 23 +
  Task 24). Module-level contract here: no regressions in the import
  path.
- ``app.api.v1._retrieval_helpers`` loads ``scoring_weights`` via
  ``load_ontology()``. Parity test: the returned weight dict is
  identical across sources.
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


def test_main_lifespan_import_under_both_sources(ontology_source):
    import app.main  # noqa: F401


def test_retrieval_helpers_import_under_both_sources(ontology_source):
    from app.api.v1 import _retrieval_helpers

    assert hasattr(_retrieval_helpers, "get_ontology_relation_weights")


def test_ontology_relation_weights_parity(monkeypatch):
    """_retrieval_helpers.get_ontology_relation_weights caches per
    ontology cache signature. After invalidating, the weights returned
    under yaml and pydantic must match value-for-value."""
    from app.api.v1 import _retrieval_helpers

    monkeypatch.setenv("ONTOLOGY_SOURCE", "yaml")
    invalidate_ontology_cache()
    _retrieval_helpers._load_scoring_weights.cache_clear()
    yaml_weights = _retrieval_helpers.get_ontology_relation_weights()

    monkeypatch.setenv("ONTOLOGY_SOURCE", "pydantic")
    invalidate_ontology_cache()
    _retrieval_helpers._load_scoring_weights.cache_clear()
    pyd_weights = _retrieval_helpers.get_ontology_relation_weights()

    assert yaml_weights == pyd_weights
    assert "default" in yaml_weights


def test_retrieval_helpers_clears_cache_on_invalidation():
    """register_invalidation_hook wires the weight-cache clear to the
    global invalidation call."""
    from app.api.v1 import _retrieval_helpers

    _retrieval_helpers._load_scoring_weights.cache_clear()
    _retrieval_helpers.get_ontology_relation_weights()
    assert _retrieval_helpers._load_scoring_weights.cache_info().currsize == 1
    invalidate_ontology_cache()
    assert _retrieval_helpers._load_scoring_weights.cache_info().currsize == 0
