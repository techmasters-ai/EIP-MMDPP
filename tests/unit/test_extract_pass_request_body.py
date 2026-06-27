import pytest
pytestmark = pytest.mark.unit
from types import SimpleNamespace
from app.workers.pipeline import _build_extract_pass_request

def _pass_def():
    return SimpleNamespace(name="radar_identity", execution=None)

def test_run_id_included_when_provided():
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3", pass_def=_pass_def(), doc_json={"x": 1},
        upstream_refs=None, document_id="doc-1", pipeline_run_id="run-1",
    )
    assert body["pipeline_run_id"] == "run-1"

def test_run_id_omitted_when_none():
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3", pass_def=_pass_def(), doc_json={"x": 1},
        upstream_refs=None, document_id="doc-1", pipeline_run_id=None,
    )
    assert "pipeline_run_id" not in body
