"""Tests for Docling-Graph HTTP client."""
from unittest.mock import patch, MagicMock
import httpx
import pytest
from app.services.docling_graph_service import (
    extract_graph,
    DoclingGraphCapacityError,
    DeterministicExtractionError,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def mock_redis():
    """Mock the Redis client used for concurrency gating."""
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = True
    mock_lock.release.return_value = None

    mock_redis_client = MagicMock()
    mock_redis_client.lock.return_value = mock_lock

    with patch("app.services.docling_graph_service._get_redis", return_value=mock_redis_client):
        yield mock_redis_client


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


def test_extract_graph_capacity_error():
    """When all Redis permits are taken, DoclingGraphCapacityError is raised."""
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = False  # All permits busy

    mock_redis_client = MagicMock()
    mock_redis_client.lock.return_value = mock_lock

    with patch("app.services.docling_graph_service._get_redis", return_value=mock_redis_client):
        with pytest.raises(DoclingGraphCapacityError):
            extract_graph("Some text", "doc-cap")


# ---------------------------------------------------------------------------
# extract_graph_all
# ---------------------------------------------------------------------------

class TestExtractGraphAll:
    def test_success(self, mock_extraction_response):
        from app.services.docling_graph_service import extract_graph_all

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_extraction_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            result = extract_graph_all("Some radar text about Tombstone", "doc-all-1")

        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1
        assert result["model"] == "llama3.2"

    def test_calls_extract_all_endpoint(self, mock_extraction_response):
        from app.services.docling_graph_service import extract_graph_all

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_extraction_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            extract_graph_all("text", "doc-123")

        call_url = mock_post.call_args[0][0]
        assert call_url.endswith("/extract-all")

    def test_sends_document_id_and_text(self, mock_extraction_response):
        from app.services.docling_graph_service import extract_graph_all

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_extraction_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            extract_graph_all("the document text", "doc-456")

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["document_id"] == "doc-456"
        assert payload["text"] == "the document text"

    def test_capacity_error(self):
        from app.services.docling_graph_service import extract_graph_all

        mock_lock = MagicMock()
        mock_lock.acquire.return_value = False

        mock_redis_client = MagicMock()
        mock_redis_client.lock.return_value = mock_lock

        with patch("app.services.docling_graph_service._get_redis", return_value=mock_redis_client):
            with pytest.raises(DoclingGraphCapacityError):
                extract_graph_all("text", "doc-cap")

    def test_http_error(self):
        from app.services.docling_graph_service import extract_graph_all

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=mock_response,
        )

        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                extract_graph_all("text", "doc-err")
