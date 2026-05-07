"""Integration: main.py /extract-pass populates response.table_overlay
when a qualifying variants table exists, and respects the kill switch.

Uses the existing conftest fixtures `dg_app_module` (loads
docker/docling-graph/app/main.py under stable module name
`docling_graph_service_main`) and patches `run_extraction_pass` —
same pattern as test_extract_pass_endpoint.py."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_DG_MODULE_NAME = "docling_graph_service_main"


@pytest.fixture
def client(dg_app_module):
    """Wraps the docling-graph FastAPI app in a TestClient.
    Module-scope fixture `dg_app_module` from conftest.py loads main.py
    under the swap-safe `_DG_MODULE_NAME` name."""
    fastapi_app = dg_app_module.app
    with TestClient(fastapi_app) as c:
        yield c


@pytest.fixture
def sa2_like_doc_with_table_fixture():
    """Minimal SA-2-shaped DoclingDocument carrying ONE qualifying
    column-major variants table. Borrowed cell shape from the
    test_table_overlay_qualification.py _make_qualifying_missile_table
    helper (Task 4)."""
    cells = []
    label_rows = (
        ("Missile Type", True),
        ("NATO Designation", True),
        ("Length mm", True),
        ("Diameter mm", True),
    )
    for r, (label, is_header) in enumerate(label_rows):
        cells.append({
            "start_row_offset_idx": r, "start_col_offset_idx": 0,
            "end_col_offset_idx": 1, "row_header": is_header,
            "text": label,
        })
    for col_idx in range(1, 6):  # 5 entity columns
        for r, val in enumerate(
            (f"M{col_idx}", f"NATO{col_idx}", "10726", "654")
        ):
            cells.append({
                "start_row_offset_idx": r,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1,
                "row_header": False, "text": val,
            })
    return {
        "tables": [
            {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 6}}
        ],
        "texts": [], "body": {"children": []},
        "name": "sa2_test_doc",
    }


def _stub_run_extraction_pass_return():
    """Mimic run_extraction_pass's return shape. Same shape as the
    existing _mock_run_pipeline_return helper in
    test_extract_pass_endpoint.py."""
    ctx = MagicMock()
    ctx.knowledge_graph.number_of_nodes.return_value = 0
    ctx.knowledge_graph.number_of_edges.return_value = 0
    ctx.graph_metadata = MagicMock(
        node_count=0, edge_count=0, node_types={}, edge_types={},
    )
    ctx.template_instance.model_dump.return_value = {}
    ctx._upstream_preamble_applied = False
    return ctx


def _make_minimal_request_payload(doc_with_table: dict):
    return {
        "bundle_key": "air_defense_v3",
        "pass_name": "missile_propulsion",
        "docling_document_json": doc_with_table,
    }


def test_extract_pass_includes_table_overlay_when_table_qualifies(
    client, sa2_like_doc_with_table_fixture, monkeypatch,
):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _stub_run_extraction_pass_return()
        r = client.post(
            "/extract-pass",
            json=_make_minimal_request_payload(sa2_like_doc_with_table_fixture),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("table_overlay") is not None
    overlay = body["table_overlay"]
    assert "MISSILE_SYSTEM" in overlay["alias_map_by_entity_type"]
    diag = body.get("diagnostics") or {}
    svc = diag.get("service_table_overlay") or {}
    assert svc.get("kill_switch_active_parser") is False
    assert svc.get("tables_processed") == 1


def test_extract_pass_kill_switch_returns_no_overlay(
    client, sa2_like_doc_with_table_fixture, monkeypatch,
):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "false")
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _stub_run_extraction_pass_return()
        r = client.post(
            "/extract-pass",
            json=_make_minimal_request_payload(sa2_like_doc_with_table_fixture),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("table_overlay") is None
    diag = body.get("diagnostics") or {}
    svc = diag.get("service_table_overlay") or {}
    assert svc.get("kill_switch_active_parser") is True
