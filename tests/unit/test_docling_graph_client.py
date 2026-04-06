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
    """Mimics the Docling-Graph service's NetworkX node-link response shape."""
    return {
        "graph": {
            "nodes": [
                {"id": "tombstone_1", "name": "Tombstone", "type": "RADAR_SYSTEM", "confidence": 0.9},
                {"id": "xband_1", "name": "X-band", "type": "FREQUENCY_BAND", "confidence": 0.85},
            ],
            "links": [
                {
                    "source": "tombstone_1", "target": "xband_1",
                    "label": "OPERATES_IN_BAND", "confidence": 0.8,
                },
            ],
        },
        "metadata": {"node_count": 2, "edge_count": 1},
        "ontology_version": "3.0.0",
        "model": "llama3.2",
        "provider": "docling-graph",
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

        doc_json = {"schema_name": "DoclingDocument", "body": {}}
        with patch("httpx.post", return_value=mock_response):
            result = extract_graph_all(doc_json, "doc-all-1")

        # Response adapter normalizes graph→entities/relationships
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1
        assert result["model"] == "llama3.2"
        # Verify entity field mapping
        assert result["entities"][0]["name"] == "Tombstone"
        assert result["entities"][0]["entity_type"] == "RADAR_SYSTEM"
        # Verify relationship field mapping
        assert result["relationships"][0]["from_name"] == "Tombstone"
        assert result["relationships"][0]["to_name"] == "X-band"
        assert result["relationships"][0]["rel_type"] == "OPERATES_IN_BAND"

    def test_calls_extract_all_endpoint(self, mock_extraction_response):
        from app.services.docling_graph_service import extract_graph_all

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_extraction_response
        mock_response.raise_for_status = MagicMock()

        doc_json = {"schema_name": "DoclingDocument"}
        with patch("httpx.post", return_value=mock_response) as mock_post:
            extract_graph_all(doc_json, "doc-123")

        call_url = mock_post.call_args[0][0]
        assert call_url.endswith("/extract-all")

    def test_sends_docling_document_json(self, mock_extraction_response):
        from app.services.docling_graph_service import extract_graph_all

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_extraction_response
        mock_response.raise_for_status = MagicMock()

        doc_json = {"schema_name": "DoclingDocument", "body": {"text": "the document text"}}
        with patch("httpx.post", return_value=mock_response) as mock_post:
            extract_graph_all(doc_json, "doc-456")

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["document_id"] == "doc-456"
        assert payload["docling_document_json"] == doc_json
        assert "text" not in payload  # old field should NOT be sent

    def test_capacity_error(self):
        from app.services.docling_graph_service import extract_graph_all

        mock_lock = MagicMock()
        mock_lock.acquire.return_value = False

        mock_redis_client = MagicMock()
        mock_redis_client.lock.return_value = mock_lock

        with patch("app.services.docling_graph_service._get_redis", return_value=mock_redis_client):
            with pytest.raises(DoclingGraphCapacityError):
                extract_graph_all({"schema_name": "DoclingDocument"}, "doc-cap")

    def test_http_error(self):
        from app.services.docling_graph_service import extract_graph_all

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=mock_response,
        )

        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                extract_graph_all({"schema_name": "DoclingDocument"}, "doc-err")
