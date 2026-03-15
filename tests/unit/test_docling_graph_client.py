"""Tests for Docling-Graph HTTP client."""
from unittest.mock import patch, MagicMock
import httpx
import pytest
from app.services.docling_graph_service import extract_graph, DeterministicExtractionError


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_extraction_response():
    return {
        "entities": [
            {"name": "Tombstone", "entity_type": "RADAR_SYSTEM", "confidence": 0.9, "properties": {}},
            {"name": "X-band", "entity_type": "FREQUENCY_BAND", "confidence": 0.85, "properties": {}},
        ],
        "relationships": [
            {
                "from_name": "Tombstone", "from_type": "RADAR_SYSTEM",
                "rel_type": "OPERATES_IN_BAND",
                "to_name": "X-band", "to_type": "FREQUENCY_BAND",
                "confidence": 0.8,
            },
        ],
        "ontology_version": "3.0.0",
        "model": "llama3.2",
        "provider": "ollama",
    }


def test_extract_graph_success(mock_extraction_response):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_extraction_response
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = extract_graph("Some radar text about Tombstone", "doc-123")

    assert len(result["entities"]) == 2
    assert result["entities"][0]["name"] == "Tombstone"
    assert len(result["relationships"]) == 1
    assert result["ontology_version"] == "3.0.0"


def test_extract_graph_service_error():
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Service Unavailable", request=MagicMock(), response=mock_response,
    )

    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            extract_graph("Some text", "doc-456")


def test_extract_graph_empty_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "entities": [], "relationships": [],
        "ontology_version": "3.0.0", "model": "llama3.2", "provider": "ollama",
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        result = extract_graph("No entities here", "doc-789")

    assert result["entities"] == []
    assert result["relationships"] == []
