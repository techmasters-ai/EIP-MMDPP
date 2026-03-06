"""Unit tests for Docling document conversion client.

Tests element-to-chunk mapping, document conversion, and health check
with mocked HTTP responses.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _map_elements_to_chunks
# ---------------------------------------------------------------------------

class TestMapElementsToChunks:
    def test_text_element(self):
        from app.services.docling_client import _map_elements_to_chunks
        elems = [{"element_type": "text", "content_text": "Hello world."}]
        chunks = _map_elements_to_chunks(elems)
        assert len(chunks) == 1
        assert chunks[0].modality == "text"
        assert chunks[0].chunk_text == "Hello world."

    def test_table_element(self):
        from app.services.docling_client import _map_elements_to_chunks
        elems = [{"element_type": "table", "content_text": "| A | B |"}]
        chunks = _map_elements_to_chunks(elems)
        assert chunks[0].modality == "table"

    def test_heading_mapped_to_text(self):
        from app.services.docling_client import _map_elements_to_chunks
        elems = [{"element_type": "heading", "content_text": "Title"}]
        chunks = _map_elements_to_chunks(elems)
        assert chunks[0].modality == "text"

    def test_image_element(self):
        from app.services.docling_client import _map_elements_to_chunks
        elems = [{"element_type": "image", "content_text": "Figure 1"}]
        chunks = _map_elements_to_chunks(elems)
        assert chunks[0].modality == "image"

    def test_image_base64_decoded(self):
        from app.services.docling_client import _map_elements_to_chunks
        raw = b"PNG_DATA"
        b64 = base64.b64encode(raw).decode()
        elems = [{"element_type": "image", "content_text": "fig", "image_base64": b64}]
        chunks = _map_elements_to_chunks(elems)
        assert chunks[0].raw_image_bytes == raw

    def test_unknown_type_defaults_to_text(self):
        from app.services.docling_client import _map_elements_to_chunks
        elems = [{"element_type": "footnote", "content_text": "Note 1"}]
        chunks = _map_elements_to_chunks(elems)
        assert chunks[0].modality == "text"

    def test_structural_metadata_preserved(self):
        from app.services.docling_client import _map_elements_to_chunks
        elems = [{
            "element_type": "text", "content_text": "text",
            "element_uid": "uid-1", "element_order": 5,
            "heading_level": 2, "section_path": "Sec > Sub",
            "metadata": {"extra": "val"},
        }]
        chunks = _map_elements_to_chunks(elems)
        meta = chunks[0].metadata
        assert meta["element_uid"] == "uid-1"
        assert meta["element_order"] == 5
        assert meta["heading_level"] == 2
        assert meta["section_path"] == "Sec > Sub"
        assert meta["extra"] == "val"

    def test_empty_elements(self):
        from app.services.docling_client import _map_elements_to_chunks
        assert _map_elements_to_chunks([]) == []

    def test_no_image_base64_means_none(self):
        from app.services.docling_client import _map_elements_to_chunks
        elems = [{"element_type": "image", "content_text": "fig"}]
        chunks = _map_elements_to_chunks(elems)
        assert chunks[0].raw_image_bytes is None


# ---------------------------------------------------------------------------
# convert_document_sync
# ---------------------------------------------------------------------------

class TestConvertDocumentSync:
    @patch("app.services.docling_client.get_settings")
    @patch("app.services.docling_client.httpx.post")
    def test_successful_conversion(self, mock_post, mock_gs):
        from app.services.docling_client import convert_document_sync
        mock_gs.return_value.docling_service_url = "http://docling:8000"
        mock_gs.return_value.docling_timeout_seconds = 120
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ok",
            "elements": [{"element_type": "text", "content_text": "Hello"}],
            "markdown": "# Hello",
            "num_pages": 1,
            "processing_time_ms": 500,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        result = convert_document_sync(b"pdf data", "test.pdf")
        assert len(result.elements) == 1
        assert result.num_pages == 1

    @patch("app.services.docling_client.get_settings")
    @patch("app.services.docling_client.httpx.post")
    def test_error_status_raises(self, mock_post, mock_gs):
        from app.services.docling_client import convert_document_sync
        mock_gs.return_value.docling_service_url = "http://docling:8000"
        mock_gs.return_value.docling_timeout_seconds = 120
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "error", "error": "bad format"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        with pytest.raises(RuntimeError, match="bad format"):
            convert_document_sync(b"data", "bad.pdf")

    @patch("app.services.docling_client.get_settings")
    @patch("app.services.docling_client.httpx.post")
    def test_result_structure(self, mock_post, mock_gs):
        from app.services.docling_client import convert_document_sync, DoclingConversionResult
        mock_gs.return_value.docling_service_url = "http://docling:8000"
        mock_gs.return_value.docling_timeout_seconds = 60
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "elements": [], "markdown": "md", "num_pages": 3, "processing_time_ms": 200,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        result = convert_document_sync(b"data", "test.pdf")
        assert isinstance(result, DoclingConversionResult)
        assert result.markdown == "md"


# ---------------------------------------------------------------------------
# check_health_sync
# ---------------------------------------------------------------------------

class TestCheckHealthSync:
    @patch("app.services.docling_client.get_settings")
    @patch("app.services.docling_client.httpx.get")
    def test_healthy(self, mock_get, mock_gs):
        from app.services.docling_client import check_health_sync
        mock_gs.return_value.docling_service_url = "http://docling:8000"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"model_loaded": True}
        mock_get.return_value = mock_resp
        assert check_health_sync() is True

    @patch("app.services.docling_client.get_settings")
    @patch("app.services.docling_client.httpx.get")
    def test_bad_status_code(self, mock_get, mock_gs):
        from app.services.docling_client import check_health_sync
        mock_gs.return_value.docling_service_url = "http://docling:8000"
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp
        assert check_health_sync() is False

    @patch("app.services.docling_client.get_settings")
    @patch("app.services.docling_client.httpx.get")
    def test_model_not_loaded(self, mock_get, mock_gs):
        from app.services.docling_client import check_health_sync
        mock_gs.return_value.docling_service_url = "http://docling:8000"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"model_loaded": False}
        mock_get.return_value = mock_resp
        assert check_health_sync() is False

    @patch("app.services.docling_client.get_settings")
    @patch("app.services.docling_client.httpx.get", side_effect=Exception("connection refused"))
    def test_connection_error(self, mock_get, mock_gs):
        from app.services.docling_client import check_health_sync
        mock_gs.return_value.docling_service_url = "http://docling:8000"
        assert check_health_sync() is False
