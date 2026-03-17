"""Tests for ontology-aware GraphRAG report generation."""
from unittest.mock import patch, MagicMock

import pytest

from app.services.graphrag_service import _generate_community_reports

pytestmark = pytest.mark.unit


def test_report_prompt_includes_ontology_context():
    """The LLM prompt should include entity type and relationship descriptions."""
    communities = [{
        "community_id": "0",
        "entity_names": ["Tombstone", "X-band"],
        "title": "Community 0",
        "level": 0,
    }]
    entities = [
        {"name": "Tombstone", "entity_type": "RADAR_SYSTEM"},
        {"name": "X-band", "entity_type": "FREQUENCY_BAND"},
    ]
    relationships = [
        {"source": "Tombstone", "target": "X-band", "relationship": "OPERATES_IN_BAND"},
    ]

    captured_prompts = []

    def mock_ollama_call(*args, **kwargs):
        captured_prompts.append(kwargs.get("json", {}).get("messages", []))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "Test report about radar systems."}
        }
        return mock_resp

    settings = MagicMock()
    settings.llm_provider = "ollama"
    settings.graphrag_model = "llama3.2"
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_num_ctx = 16384
    settings.ollama_think = ""

    with patch("httpx.post", side_effect=mock_ollama_call):
        _generate_community_reports(communities, entities, relationships, settings)

    assert len(captured_prompts) == 1
    system_msg = captured_prompts[0][0]["content"]
    assert "RADAR_SYSTEM" in system_msg
    assert "FREQUENCY_BAND" in system_msg
    assert "OPERATES_IN_BAND" in system_msg
