"""Tests for LLM-based document metadata extraction and picture descriptions."""

from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# extract_document_metadata
# ---------------------------------------------------------------------------

class TestExtractDocumentMetadata:
    @patch("app.services.document_analysis.httpx.Client")
    @patch("app.services.document_analysis.get_settings")
    def test_returns_all_keys(self, mock_settings, MockClient):
        """Should return summary, date, classification, source, generated_at."""
        s = MagicMock()
        s.doc_analysis_llm_model = "test-model"
        s.doc_analysis_timeout = 30
        s.ollama_base_url = "http://ollama:11434"
        s.ollama_num_ctx = 4096
        s.doc_analysis_summary_prompt = "Summarize"
        s.doc_analysis_date_prompt = "Date"
        s.doc_analysis_source_prompt = "Source"
        s.doc_analysis_classification_prompt = "Classification"
        mock_settings.return_value = s

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Test result"}}]
        }
        mock_resp.raise_for_status = MagicMock()

        client_inst = MagicMock()
        client_inst.post.return_value = mock_resp
        MockClient.return_value = client_inst

        from app.services.document_analysis import extract_document_metadata
        result = extract_document_metadata("Some document text")

        assert "document_summary" in result
        assert "date_of_information" in result
        assert "classification" in result
        assert "source_characterization" in result
        assert "generated_at" in result

    @patch("app.services.document_analysis.httpx.Client")
    @patch("app.services.document_analysis.get_settings")
    def test_classification_normalization(self, mock_settings, MockClient):
        """Invalid classification strings should default to UNCLASSIFIED."""
        s = MagicMock()
        s.doc_analysis_llm_model = "m"
        s.doc_analysis_timeout = 10
        s.ollama_base_url = "http://x"
        s.ollama_num_ctx = 1024
        s.doc_analysis_summary_prompt = "s"
        s.doc_analysis_date_prompt = "d"
        s.doc_analysis_source_prompt = "c"
        s.doc_analysis_classification_prompt = "cl"
        mock_settings.return_value = s

        # Return "BOGUS" for classification, which should normalise to UNCLASSIFIED
        def side_effect(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": "BOGUS"}}]
            }
            resp.raise_for_status = MagicMock()
            return resp

        client_inst = MagicMock()
        client_inst.post.side_effect = side_effect
        MockClient.return_value = client_inst

        from app.services.document_analysis import extract_document_metadata
        result = extract_document_metadata("text")
        assert result["classification"] == "UNCLASSIFIED"

    @patch("app.services.document_analysis.httpx.Client")
    @patch("app.services.document_analysis.get_settings")
    def test_valid_classification_preserved(self, mock_settings, MockClient):
        """Valid classification (e.g. SECRET) should be preserved."""
        s = MagicMock()
        s.doc_analysis_llm_model = "m"
        s.doc_analysis_timeout = 10
        s.ollama_base_url = "http://x"
        s.ollama_num_ctx = 1024
        s.doc_analysis_summary_prompt = "s"
        s.doc_analysis_date_prompt = "d"
        s.doc_analysis_source_prompt = "c"
        s.doc_analysis_classification_prompt = "cl"
        mock_settings.return_value = s

        def side_effect(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": "SECRET"}}]
            }
            resp.raise_for_status = MagicMock()
            return resp

        client_inst = MagicMock()
        client_inst.post.side_effect = side_effect
        MockClient.return_value = client_inst

        from app.services.document_analysis import extract_document_metadata
        result = extract_document_metadata("classified text")
        assert result["classification"] == "SECRET"

    @patch("app.services.document_analysis.httpx.Client")
    @patch("app.services.document_analysis.get_settings")
    def test_truncates_long_text(self, mock_settings, MockClient):
        """Text longer than ollama_num_ctx * 3 should be truncated."""
        s = MagicMock()
        s.doc_analysis_llm_model = "m"
        s.doc_analysis_timeout = 10
        s.ollama_base_url = "http://x"
        s.ollama_num_ctx = 10  # max_chars = 30
        s.doc_analysis_summary_prompt = "s"
        s.doc_analysis_date_prompt = "d"
        s.doc_analysis_source_prompt = "c"
        s.doc_analysis_classification_prompt = "cl"
        mock_settings.return_value = s

        captured_texts = []

        def side_effect(*args, **kwargs):
            body = kwargs.get("json", args[1] if len(args) > 1 else {})
            user_msg = body.get("messages", [{}])[-1].get("content", "")
            captured_texts.append(user_msg)
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}]
            }
            resp.raise_for_status = MagicMock()
            return resp

        client_inst = MagicMock()
        client_inst.post.side_effect = side_effect
        MockClient.return_value = client_inst

        from app.services.document_analysis import extract_document_metadata
        long_text = "A" * 100
        extract_document_metadata(long_text)

        # All calls should receive truncated text (max 30 chars)
        for text in captured_texts:
            assert len(text) <= 30

    @patch("app.services.document_analysis.httpx.Client")
    @patch("app.services.document_analysis.get_settings")
    def test_llm_failure_returns_defaults(self, mock_settings, MockClient):
        """If LLM calls fail, fallback values should be used."""
        s = MagicMock()
        s.doc_analysis_llm_model = "m"
        s.doc_analysis_timeout = 10
        s.ollama_base_url = "http://x"
        s.ollama_num_ctx = 4096
        s.doc_analysis_summary_prompt = "s"
        s.doc_analysis_date_prompt = "d"
        s.doc_analysis_source_prompt = "c"
        s.doc_analysis_classification_prompt = "cl"
        mock_settings.return_value = s

        client_inst = MagicMock()
        client_inst.post.side_effect = Exception("LLM down")
        MockClient.return_value = client_inst

        from app.services.document_analysis import extract_document_metadata
        result = extract_document_metadata("text")

        # Should not crash; return fallback values
        assert result["classification"] == "UNCLASSIFIED"
        assert "generated_at" in result


# ---------------------------------------------------------------------------
# describe_pictures
# ---------------------------------------------------------------------------

class TestDescribePictures:
    @patch("app.services.document_analysis.get_settings")
    def test_no_pictures_returns_unchanged(self, mock_settings):
        s = MagicMock()
        mock_settings.return_value = s

        from app.services.document_analysis import describe_pictures
        docling_json = {"pictures": []}
        result = describe_pictures(docling_json, "summary")
        assert result is docling_json

    @patch("app.services.document_analysis.get_settings")
    def test_no_describable_images(self, mock_settings):
        """Pictures without data URIs should be skipped."""
        s = MagicMock()
        mock_settings.return_value = s

        from app.services.document_analysis import describe_pictures
        docling_json = {
            "pictures": [
                {"image": {"uri": ""}},
                {"image": {"uri": "https://example.com/img.png"}},
            ]
        }
        result = describe_pictures(docling_json, "summary")
        assert result is docling_json

    @patch("app.services.document_analysis._describe_single_image")
    @patch("app.services.document_analysis.get_settings")
    def test_describable_images_get_descriptions(self, mock_settings, mock_describe):
        s = MagicMock()
        s.picture_description_model = "gemma3:27b"
        s.picture_description_timeout = 60
        s.picture_description_prompt = "Describe this image. {document_summary}"
        mock_settings.return_value = s
        mock_describe.return_value = "A radar installation on a vehicle."

        from app.services.document_analysis import describe_pictures
        import base64
        b64 = base64.b64encode(b"\x89PNG fake").decode()
        docling_json = {
            "pictures": [
                {"image": {"uri": f"data:image/png;base64,{b64}"}},
            ]
        }
        result = describe_pictures(docling_json, "Doc about radar systems.")
        pic = result["pictures"][0]
        assert pic["description"] == "A radar installation on a vehicle."
        assert len(pic["annotations"]) == 1
        ann = pic["annotations"][0]
        assert ann["kind"] == "description"
        assert ann["text"] == "A radar installation on a vehicle."
        assert ann["source"] == "llm"
        assert ann["model"] == "gemma3:27b"


# ---------------------------------------------------------------------------
# _describe_single_image
# ---------------------------------------------------------------------------

class TestDescribeSingleImage:
    @patch("app.services.document_analysis.httpx.Client")
    def test_returns_content_on_success(self, MockClient):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "This is a missile diagram."}}]
        }
        mock_resp.raise_for_status = MagicMock()

        client_inst = MagicMock()
        client_inst.__enter__ = MagicMock(return_value=client_inst)
        client_inst.__exit__ = MagicMock(return_value=False)
        client_inst.post.return_value = mock_resp
        MockClient.return_value = client_inst

        s = MagicMock()
        s.ollama_base_url = "http://ollama:11434"

        from app.services.document_analysis import _describe_single_image
        result = _describe_single_image("base64data", "prompt", "gemma3:27b", 60, s)
        assert result == "This is a missile diagram."

    @patch("app.services.document_analysis.httpx.Client")
    def test_returns_none_on_failure(self, MockClient):
        client_inst = MagicMock()
        client_inst.__enter__ = MagicMock(return_value=client_inst)
        client_inst.__exit__ = MagicMock(return_value=False)
        client_inst.post.side_effect = Exception("timeout")
        MockClient.return_value = client_inst

        s = MagicMock()
        s.ollama_base_url = "http://ollama:11434"

        from app.services.document_analysis import _describe_single_image
        result = _describe_single_image("base64data", "prompt", "gemma3:27b", 60, s)
        assert result is None
