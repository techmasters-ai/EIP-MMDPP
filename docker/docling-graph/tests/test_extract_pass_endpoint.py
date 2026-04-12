"""Protocol tests for POST /extract-pass. Spec §8.6.

These are contract tests — the extraction pipeline itself is mocked.
We assert the handler's request-validation and 400/404/200 behavior,
not extraction correctness."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


_DG_MODULE_NAME = "docling_graph_service_main"
_DG_SERVICE_ROOT = Path(__file__).resolve().parent.parent


def _ensure_dg_app_package() -> None:
    """Ensure the docling-graph `app.*` sub-modules are importable as `app.*`.

    When the combined test suite runs from repo root, the repo-root `app/`
    package is already in sys.modules['app']. We temporarily swap it out
    for the docling-graph `app/` package so that `from app.config_builder
    import ...` in main.py resolves to the docling-graph package.

    This is called once; subsequent calls are no-ops.
    """
    import importlib
    import importlib.util
    import sys

    if _DG_MODULE_NAME in sys.modules:
        return  # already done

    service_root = _DG_SERVICE_ROOT
    dg_app_path = service_root / "app"

    # Save state
    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    saved_path = list(sys.path)

    # Swap: point sys.path[0] at service root so `import app` finds DG's app/
    sys.path.insert(0, str(service_root))
    # Remove any cached repo-root `app.*` modules so fresh imports find DG's
    for key in list(saved.keys()):
        del sys.modules[key]

    try:
        spec = importlib.util.spec_from_file_location(
            _DG_MODULE_NAME,
            service_root / "app" / "main.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_DG_MODULE_NAME] = mod
        spec.loader.exec_module(mod)
    finally:
        # Restore: remove DG's app.* from cache, re-insert repo-root's
        for key in list(sys.modules.keys()):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]
        sys.modules.update(saved)
        # Restore sys.path
        sys.path[:] = saved_path


@pytest.fixture(scope="module")
def dg_app_module():
    """Load the docling-graph app module once per test module."""
    import sys
    _ensure_dg_app_package()
    return sys.modules[_DG_MODULE_NAME]


@pytest.fixture
def client(dg_app_module):
    """TestClient that shares the module-level FastAPI app.

    The app's lifespan hook calls preload_all_templates() which walks
    ontology_bundles/ and imports every extraction_schemas module.
    TestClient triggers the lifespan on first request.
    """
    fastapi_app = dg_app_module.app
    with TestClient(fastapi_app) as c:
        yield c


# Helper to patch out the real pipeline invocation
def _mock_run_pipeline_return():
    """Return a stub mimicking docling_graph.run_pipeline's return shape
    so the handler can build a response without hitting the LLM."""
    ctx = MagicMock()
    ctx.knowledge_graph.number_of_nodes.return_value = 0
    ctx.knowledge_graph.number_of_edges.return_value = 0
    ctx.graph_metadata = MagicMock(
        node_count=0, edge_count=0, node_types={}, edge_types={},
    )
    # The handler walks pass_result.template_instance.model_dump() — make
    # the stub return something model-dump-able (a real dict).
    ctx.template_instance.model_dump.return_value = {}
    return ctx


def test_extract_pass_unknown_bundle_key_returns_404(client):
    resp = client.post("/extract-pass", json={
        "bundle_key": "does_not_exist",
        "pass_name": "reference",
        "docling_document_json": {"name": "test"},
    })
    assert resp.status_code == 404
    assert "bundle_key" in resp.json().get("detail", "").lower() or \
           "does_not_exist" in resp.json().get("detail", "")


def test_extract_pass_unknown_pass_name_returns_404(client):
    resp = client.post("/extract-pass", json={
        "bundle_key": "air_defense_v3",
        "pass_name": "nonexistent_pass",
        "docling_document_json": {"name": "test"},
    })
    assert resp.status_code == 404
    assert "pass_name" in resp.json().get("detail", "").lower() or \
           "nonexistent_pass" in resp.json().get("detail", "")


def test_extract_pass_document_only_with_upstream_entities_returns_400(client):
    """document_only pass with unexpected upstream_entities → 400"""
    resp = client.post("/extract-pass", json={
        "bundle_key": "air_defense_v3",
        "pass_name": "reference",  # input_mode == document_only
        "docling_document_json": {"name": "test"},
        "upstream_entities": [
            {"ref_id": "E01", "entity_type": "RADAR_SYSTEM", "identity_values": {}},
        ],
    })
    assert resp.status_code == 400
    assert "document_only" in resp.json().get("detail", "") or \
           "upstream_entities" in resp.json().get("detail", "")


def test_extract_pass_document_plus_entity_refs_missing_upstream_returns_400(client):
    """document_plus_entity_refs pass with missing upstream_entities → 400"""
    resp = client.post("/extract-pass", json={
        "bundle_key": "air_defense_v3",
        "pass_name": "system_links",  # input_mode == document_plus_entity_refs
        "docling_document_json": {"name": "test"},
        # no upstream_entities
    })
    assert resp.status_code == 400
    assert "document_plus_entity_refs" in resp.json().get("detail", "") or \
           "upstream_entities" in resp.json().get("detail", "")


def test_extract_pass_document_plus_entity_refs_empty_upstream_returns_400(client):
    """document_plus_entity_refs pass with empty upstream_entities list → 400"""
    resp = client.post("/extract-pass", json={
        "bundle_key": "air_defense_v3",
        "pass_name": "system_links",
        "docling_document_json": {"name": "test"},
        "upstream_entities": [],
    })
    assert resp.status_code == 400


def test_extract_pass_valid_document_only_returns_200(client):
    """Valid document_only request → 200."""
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "reference",
            "docling_document_json": {"name": "test"},
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["bundle_key"] == "air_defense_v3"
    assert body["pass_name"] == "reference"
    assert "pass_output" in body


def test_extract_pass_valid_document_plus_entity_refs_returns_200(client):
    """Valid document_plus_entity_refs request → 200."""
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "system_links",
            "docling_document_json": {"name": "test"},
            "upstream_entities": [
                {"ref_id": "E01", "entity_type": "RADAR_SYSTEM", "identity_values": {"system_name": "Foo"}},
            ],
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pass_name"] == "system_links"
