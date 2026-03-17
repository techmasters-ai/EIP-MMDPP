"""Tests for the /extract endpoint with template_group and mode routing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the app package is importable
_DOCLING_GRAPH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DOCLING_GRAPH_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ONTOLOGY_PATH = _REPO_ROOT / "ontology" / "ontology.yaml"


@pytest.fixture()
def mock_templates():
    from app.templates import build_templates, load_ontology

    for p in [_ONTOLOGY_PATH, Path(os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml"))]:
        if p.exists():
            ontology = load_ontology(p)
            return build_templates(ontology)
    pytest.skip("Ontology file not found")


@pytest.fixture()
def client(mock_templates):
    from app import main

    main._templates = mock_templates
    main._ontology_version = "3.0.0"

    from fastapi.testclient import TestClient

    return TestClient(main.app)


class TestExtractEndpointGrouped:
    def test_mock_mode_with_group(self, client):
        import app.main as m

        orig = m.LLM_PROVIDER
        m.LLM_PROVIDER = "mock"
        try:
            resp = client.post(
                "/extract",
                json={
                    "document_id": "test-123",
                    "text": "The AN/MPQ-53 radar operates in C-band.",
                    "template_group": "equipment",
                    "mode": "entities",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "entities" in data
            assert data["provider"] == "mock"
        finally:
            m.LLM_PROVIDER = orig

    def test_mock_mode_relationships(self, client):
        import app.main as m

        orig = m.LLM_PROVIDER
        m.LLM_PROVIDER = "mock"
        try:
            resp = client.post(
                "/extract",
                json={
                    "document_id": "test-123",
                    "text": "The AN/MPQ-53 radar is installed on the Patriot system.",
                    "mode": "relationships",
                    "entities_context": [
                        {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM"},
                        {"name": "Patriot", "entity_type": "MISSILE_SYSTEM"},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "relationships" in data
            # In relationship mode, entities should be empty
            assert data["entities"] == []
        finally:
            m.LLM_PROVIDER = orig

    def test_mock_mode_entities_no_relationships(self, client):
        """Entity mode should return entities but no relationships."""
        import app.main as m

        orig = m.LLM_PROVIDER
        m.LLM_PROVIDER = "mock"
        try:
            resp = client.post(
                "/extract",
                json={
                    "document_id": "test-123",
                    "text": "The AN/MPQ-53 radar operates in C-band.",
                    "template_group": "equipment",
                    "mode": "entities",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["entities"]) > 0
            assert data["relationships"] == []
        finally:
            m.LLM_PROVIDER = orig

    def test_legacy_no_group(self, client):
        """Request with no template_group should still work (backward compat)."""
        import app.main as m

        orig = m.LLM_PROVIDER
        m.LLM_PROVIDER = "mock"
        try:
            resp = client.post(
                "/extract",
                json={
                    "document_id": "test-123",
                    "text": "Some text.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "entities" in data
            assert "relationships" in data
        finally:
            m.LLM_PROVIDER = orig

    def test_invalid_group_returns_422(self, client):
        resp = client.post(
            "/extract",
            json={
                "document_id": "test-123",
                "text": "Some text.",
                "template_group": "nonexistent",
                "mode": "entities",
            },
        )
        assert resp.status_code == 422

    def test_invalid_mode_returns_422(self, client):
        resp = client.post(
            "/extract",
            json={
                "document_id": "test-123",
                "text": "Some text.",
                "mode": "invalid_mode",
            },
        )
        assert resp.status_code == 422

    def test_health_includes_groups(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "groups" in data
        assert set(data["groups"]) == {"reference", "equipment", "rf_signal", "weapon", "operational"}
