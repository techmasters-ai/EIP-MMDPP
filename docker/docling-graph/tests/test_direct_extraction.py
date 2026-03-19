"""Tests for direct LLM entity extraction and the /extract-all endpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
def setup_app(mock_templates):
    """Set up the FastAPI app in mock mode."""
    from app import main

    main._templates = mock_templates
    main._ontology_version = "3.0.0"
    # Build schema prompts from ontology
    from app.templates import load_ontology
    for p in [_ONTOLOGY_PATH, Path(os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml"))]:
        if p.exists():
            ontology = load_ontology(p)
            main._group_schema_prompts = main._build_group_schema_prompts(ontology)
            break
    return main


@pytest.fixture()
def client(setup_app):
    from fastapi.testclient import TestClient
    return TestClient(setup_app.app)


# ---------------------------------------------------------------------------
# _parse_json_from_llm
# ---------------------------------------------------------------------------

class TestParseJsonFromLlm:
    def test_plain_json_object(self, setup_app):
        result = setup_app._parse_json_from_llm('{"entities": []}')
        assert result == {"entities": []}

    def test_plain_json_array(self, setup_app):
        result = setup_app._parse_json_from_llm('[{"name": "test"}]')
        assert result == [{"name": "test"}]

    def test_markdown_wrapped(self, setup_app):
        raw = '```json\n{"entities": [{"name": "test"}]}\n```'
        result = setup_app._parse_json_from_llm(raw)
        assert result == {"entities": [{"name": "test"}]}

    def test_text_before_json(self, setup_app):
        raw = 'Here are the entities: {"entities": [{"name": "AN/MPQ-53"}]}'
        result = setup_app._parse_json_from_llm(raw)
        assert result["entities"][0]["name"] == "AN/MPQ-53"

    def test_none_input(self, setup_app):
        assert setup_app._parse_json_from_llm(None) is None

    def test_empty_string(self, setup_app):
        assert setup_app._parse_json_from_llm("") is None

    def test_no_json(self, setup_app):
        assert setup_app._parse_json_from_llm("no json here") is None


# ---------------------------------------------------------------------------
# _build_group_schema_prompts
# ---------------------------------------------------------------------------

class TestBuildGroupSchemaPrompts:
    def test_generates_all_groups(self, setup_app):
        prompts = setup_app._group_schema_prompts
        assert "reference" in prompts
        assert "equipment" in prompts
        assert "rf_signal" in prompts
        assert "weapon" in prompts
        assert "operational" in prompts

    def test_equipment_group_includes_radar_system(self, setup_app):
        prompt = setup_app._group_schema_prompts["equipment"]
        assert "RADAR_SYSTEM" in prompt
        assert "PLATFORM" in prompt

    def test_prompts_include_property_descriptions(self, setup_app):
        prompt = setup_app._group_schema_prompts["equipment"]
        assert "system_name" in prompt or "nomenclature" in prompt


# ---------------------------------------------------------------------------
# /extract-all (mock mode)
# ---------------------------------------------------------------------------

class TestExtractAllEndpoint:
    def test_mock_mode_returns_entities(self, client, setup_app):
        orig = setup_app.LLM_PROVIDER
        setup_app.LLM_PROVIDER = "mock"
        try:
            resp = client.post(
                "/extract-all",
                json={
                    "document_id": "test-all-1",
                    "text": "The AN/MPQ-53 radar operates in C-band on the Patriot system.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "entities" in data
            assert "relationships" in data
            assert data["provider"] == "mock"
            assert len(data["entities"]) > 0
        finally:
            setup_app.LLM_PROVIDER = orig

    def test_service_not_ready(self, setup_app):
        """If templates aren't loaded, should return 503."""
        from fastapi.testclient import TestClient

        saved = setup_app._templates
        setup_app._templates = None
        try:
            test_client = TestClient(setup_app.app)
            resp = test_client.post(
                "/extract-all",
                json={"document_id": "test", "text": "text"},
            )
            assert resp.status_code == 503
        finally:
            setup_app._templates = saved


# ---------------------------------------------------------------------------
# _extract_entities_for_group (with mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractEntitiesForGroup:
    @patch("app.main.litellm")
    def test_parses_llm_response(self, mock_litellm, setup_app):
        """Should parse JSON entity array from LLM response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "entities": [
                {"name": "S-300", "entity_type": "MISSILE_SYSTEM", "confidence": 0.9, "properties": {}},
                {"name": "Tombstone", "entity_type": "RADAR_SYSTEM", "confidence": 0.85, "properties": {}},
            ]
        })
        mock_litellm.completion.return_value = mock_response

        result = setup_app._extract_entities_for_group(
            "The S-300 uses the Tombstone radar.", "equipment"
        )
        assert len(result) == 2
        assert result[0]["name"] == "S-300"
        assert result[1]["entity_type"] == "RADAR_SYSTEM"

    @patch("app.main.litellm")
    def test_llm_failure_returns_empty(self, mock_litellm, setup_app):
        """LLM exception should return empty list, not crash."""
        mock_litellm.completion.side_effect = Exception("LLM timeout")

        result = setup_app._extract_entities_for_group("text", "equipment")
        assert result == []

    @patch("app.main.litellm")
    def test_invalid_json_returns_empty(self, mock_litellm, setup_app):
        """Non-JSON LLM output should return empty list."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I cannot extract entities from this text."
        mock_litellm.completion.return_value = mock_response

        result = setup_app._extract_entities_for_group("text", "equipment")
        assert result == []


# ---------------------------------------------------------------------------
# _extract_relationships (with mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractRelationships:
    @patch("app.main.litellm")
    def test_parses_relationship_response(self, mock_litellm, setup_app):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relationships": [
                {
                    "from_name": "S-300", "from_type": "MISSILE_SYSTEM",
                    "rel_type": "HAS_COMPONENT",
                    "to_name": "Tombstone", "to_type": "RADAR_SYSTEM",
                    "confidence": 0.9,
                },
            ]
        })
        mock_litellm.completion.return_value = mock_response

        entities = [
            {"name": "S-300", "entity_type": "MISSILE_SYSTEM"},
            {"name": "Tombstone", "entity_type": "RADAR_SYSTEM"},
        ]
        result = setup_app._extract_relationships("text about S-300", entities)
        assert len(result) == 1
        assert result[0]["rel_type"] == "HAS_COMPONENT"

    @patch("app.main.litellm")
    def test_llm_failure_returns_empty(self, mock_litellm, setup_app):
        mock_litellm.completion.side_effect = Exception("LLM down")
        result = setup_app._extract_relationships("text", [])
        assert result == []


# ---------------------------------------------------------------------------
# _run_full_extraction (with mocked LLM)
# ---------------------------------------------------------------------------

class TestRunFullExtraction:
    @patch("app.main._extract_relationships")
    @patch("app.main._extract_entities_for_group")
    def test_runs_all_groups_and_relationships(self, mock_entities, mock_rels, setup_app):
        """Should call entity extraction for all 5 groups + 1 relationship pass."""
        mock_entities.return_value = [
            {"name": "Test", "entity_type": "PLATFORM", "confidence": 0.9, "properties": {}},
        ]
        mock_rels.return_value = []

        entities, rels = setup_app._run_full_extraction("some document text")

        # 5 groups should produce 5 entities (1 per group call)
        assert len(entities) == 5
        assert mock_entities.call_count == 5
        assert mock_rels.call_count == 1

    @patch("app.main._extract_relationships")
    @patch("app.main._extract_entities_for_group")
    def test_passes_entities_to_relationship_extraction(self, mock_entities, mock_rels, setup_app):
        """Relationship extraction should receive all extracted entities as context."""
        mock_entities.return_value = [
            {"name": "Radar A", "entity_type": "RADAR_SYSTEM", "confidence": 0.9, "properties": {}},
        ]
        mock_rels.return_value = []

        setup_app._run_full_extraction("text")

        # Check entities_context passed to relationship extraction
        rel_call_args = mock_rels.call_args
        entities_context = rel_call_args[0][1]
        assert len(entities_context) == 5  # 5 groups × 1 entity each
        assert all(e.get("name") for e in entities_context)
