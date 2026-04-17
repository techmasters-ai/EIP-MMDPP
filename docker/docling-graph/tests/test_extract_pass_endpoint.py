"""Protocol tests for POST /extract-pass. Spec §8.6.

These are contract tests — the extraction pipeline itself is mocked.
We assert the handler's request-validation and 400/404/200 behavior,
not extraction correctness."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


_DG_MODULE_NAME = "docling_graph_service_main"
_DG_SERVICE_ROOT = Path(__file__).resolve().parent.parent


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
def _mock_run_pipeline_return(preamble_applied: bool = False):
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
    ctx._upstream_preamble_applied = preamble_applied
    return ctx


def test_extract_pass_unknown_bundle_key_returns_404(client):
    resp = client.post("/extract-pass", json={
        "bundle_key": "does_not_exist",
        "pass_name": "radar_domain",
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
        "pass_name": "radar_domain",  # input_mode == document_only
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
            "pass_name": "radar_domain",
            "docling_document_json": {"name": "test"},
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["bundle_key"] == "air_defense_v3"
    assert body["pass_name"] == "radar_domain"
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


def test_metadata_reports_upstream_ref_count_for_document_plus_entity_refs(client):
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "system_links",
            "docling_document_json": {"name": "test"},
            "upstream_entities": [
                {"ref_id": "E001", "entity_type": "RADAR_SYSTEM",
                 "identity_values": {"system_name": "Fan Song"},
                 "display_label": "Fan Song"},
                {"ref_id": "E002", "entity_type": "MISSILE_SYSTEM",
                 "identity_values": {"system_name": "SA-2"},
                 "display_label": "SA-2"},
            ],
        })
    assert resp.status_code == 200, resp.text
    meta = resp.json()["metadata"]
    assert meta["upstream_ref_count"] == 2
    assert meta["upstream_preamble_applied"] is False  # flipped to True in Task 5b


def test_metadata_reports_zero_refs_for_document_only(client):
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_domain",
            "docling_document_json": {"name": "test"},
        })
    assert resp.status_code == 200, resp.text
    meta = resp.json()["metadata"]
    assert meta["upstream_ref_count"] == 0
    assert meta["upstream_preamble_applied"] is False


def test_extract_pass_logs_end_even_on_extraction_failure(client, caplog):
    """Regression: the END log must fire whether extraction succeeded or
    raised. Previously it was placed after metadata construction and was
    skipped when run_extraction_pass raised before that point."""
    import logging
    caplog.set_level(logging.INFO)
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.side_effect = RuntimeError("simulated extraction failure")
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_domain",
            "docling_document_json": {"name": "test"},
        })
    assert resp.status_code == 500
    start_logs = [r for r in caplog.records if "extract-pass: START" in r.message]
    end_logs = [r for r in caplog.records if "extract-pass: END" in r.message]
    assert len(start_logs) == 1
    assert len(end_logs) == 1, "END log must fire even when extraction raises"


# ---------------------------------------------------------------------------
# Step 4 — direct run_extraction_pass unit tests (Path B injection)
# These stub sys.modules["docling_graph"] so they work on hosts that don't
# have the real docling_graph library installed.
# ---------------------------------------------------------------------------

def _make_docling_graph_stub():
    """Create a minimal sys.modules stub for docling_graph.

    Must carry both run_pipeline (used inside run_extraction_pass) and
    PipelineConfig (imported by app.config_builder.build_pipeline_config).
    """
    stub_mod = types.ModuleType("docling_graph")

    def fake_run_pipeline(config):
        ctx = MagicMock()
        ctx.knowledge_graph.number_of_nodes.return_value = 0
        ctx.knowledge_graph.number_of_edges.return_value = 0
        ctx.graph_metadata = MagicMock(
            node_count=0, edge_count=0, node_types={}, edge_types={},
        )
        ctx.template_instance.model_dump.return_value = {}
        return ctx

    class FakePipelineConfig:
        """Minimal PipelineConfig stand-in that accepts any kwargs."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    stub_mod.run_pipeline = fake_run_pipeline
    stub_mod.PipelineConfig = FakePipelineConfig
    return stub_mod


def test_run_extraction_pass_exercises_chosen_injection_hook(dg_app_module, monkeypatch):
    """Path B: preamble is appended to texts and prepended to body.children in the
    tmp-file body, and context._upstream_preamble_applied is True."""
    stub_mod = _make_docling_graph_stub()
    monkeypatch.setitem(sys.modules, "docling_graph", stub_mod)
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "true")

    EntityRef = dg_app_module.EntityRef
    upstream_entities = [
        EntityRef(
            ref_id="E001",
            entity_type="RADAR_SYSTEM",
            identity_values={"system_name": "Fan Song"},
            display_label="Fan Song",
        ),
    ]

    # Spy on json.dump to capture what's written to the tmp file
    captured = {}
    original_json_dump = json.dump

    def spy_json_dump(obj, fp, **kwargs):
        if hasattr(fp, "name") and str(getattr(fp, "name", "")).endswith(".json"):
            captured["tmp_body_dict"] = obj
        return original_json_dump(obj, fp, **kwargs)

    monkeypatch.setattr(dg_app_module.json, "dump", spy_json_dump)

    # Input document has no existing texts
    input_doc = {
        "schema_name": "DoclingDocument",
        "version": "2.0.0",
        "name": "test_doc",
        "body": {
            "self_ref": "#/body",
            "name": "_root_",
            "label": "unspecified",
            "children": [],
        },
        "texts": [],
    }

    context = dg_app_module.run_extraction_pass(
        docling_document_json=input_doc,
        template_cls=MagicMock(),
        upstream_entities=upstream_entities,
    )

    # Preamble flag must be set on the returned context
    assert context._upstream_preamble_applied is True

    # Path-B concrete assertion: the tmp file texts[0] contains the preamble
    assert "tmp_body_dict" in captured, "json.dump spy did not capture the tmp file write"
    texts = captured["tmp_body_dict"]["texts"]
    assert len(texts) > 0
    assert texts[0]["text"].startswith("Upstream entities:")

    # Body's first child should reference the preamble item
    body_children = captured["tmp_body_dict"]["body"]["children"]
    assert len(body_children) > 0
    assert body_children[0]["$ref"] == "#/texts/0"


def test_run_extraction_pass_env_flag_off_does_not_invoke_hook(dg_app_module, monkeypatch):
    """When DOCLING_GRAPH_UPSTREAM_PREAMBLE=false, document is not mutated and
    context._upstream_preamble_applied is False."""
    stub_mod = _make_docling_graph_stub()
    monkeypatch.setitem(sys.modules, "docling_graph", stub_mod)
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "false")

    EntityRef = dg_app_module.EntityRef
    upstream_entities = [
        EntityRef(
            ref_id="E001",
            entity_type="RADAR_SYSTEM",
            identity_values={"system_name": "Fan Song"},
            display_label="Fan Song",
        ),
    ]

    # Spy on json.dump to capture what's written to the tmp file
    captured = {}
    original_json_dump = json.dump

    def spy_json_dump(obj, fp, **kwargs):
        if hasattr(fp, "name") and str(getattr(fp, "name", "")).endswith(".json"):
            captured["tmp_body_dict"] = obj
        return original_json_dump(obj, fp, **kwargs)

    monkeypatch.setattr(dg_app_module.json, "dump", spy_json_dump)

    original_input_texts = [
        {
            "self_ref": "#/texts/0",
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": "text",
            "prov": [],
            "orig": "Original document content.",
            "text": "Original document content.",
        },
    ]
    input_doc = {
        "schema_name": "DoclingDocument",
        "version": "2.0.0",
        "name": "test_doc",
        "body": {
            "self_ref": "#/body",
            "name": "_root_",
            "label": "unspecified",
            "children": [{"$ref": "#/texts/0"}],
        },
        "texts": list(original_input_texts),
    }

    context = dg_app_module.run_extraction_pass(
        docling_document_json=input_doc,
        template_cls=MagicMock(),
        upstream_entities=upstream_entities,
    )

    # Preamble must NOT be applied
    assert context._upstream_preamble_applied is False

    # Path-B negative assertion: document should have the same N texts, unchanged
    assert "tmp_body_dict" in captured, "json.dump spy did not capture the tmp file write"
    texts = captured["tmp_body_dict"]["texts"]
    assert len(texts) == len(original_input_texts)
    assert texts[0]["text"] == original_input_texts[0]["text"]


# ---------------------------------------------------------------------------
# Step 5 — endpoint flag-flip tests
# These patch run_extraction_pass at the module level.
# ---------------------------------------------------------------------------

def test_preamble_applied_flag_flips_true_for_document_plus_entity_refs(client):
    """When the mock returns preamble_applied=True, the response metadata reflects it."""
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return(preamble_applied=True)
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "system_links",
            "docling_document_json": {"name": "test"},
            "upstream_entities": [
                {"ref_id": "E001", "entity_type": "RADAR_SYSTEM",
                 "identity_values": {"system_name": "Fan Song"},
                 "display_label": "Fan Song"},
            ],
        })
    assert resp.status_code == 200, resp.text
    meta = resp.json()["metadata"]
    assert meta["upstream_preamble_applied"] is True


def test_preamble_applied_flag_false_for_document_only(client):
    """document_only pass: preamble_applied must be False."""
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return(preamble_applied=False)
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_domain",
            "docling_document_json": {"name": "test"},
        })
    assert resp.status_code == 200, resp.text
    meta = resp.json()["metadata"]
    assert meta["upstream_preamble_applied"] is False


def test_preamble_applied_flag_false_when_env_disables(client, monkeypatch):
    """DOCLING_GRAPH_UPSTREAM_PREAMBLE=false → mock returns False → metadata is False."""
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "false")
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return(preamble_applied=False)
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "system_links",
            "docling_document_json": {"name": "test"},
            "upstream_entities": [
                {"ref_id": "E001", "entity_type": "RADAR_SYSTEM",
                 "identity_values": {"system_name": "Fan Song"},
                 "display_label": "Fan Song"},
            ],
        })
    assert resp.status_code == 200, resp.text
    meta = resp.json()["metadata"]
    assert meta["upstream_preamble_applied"] is False
