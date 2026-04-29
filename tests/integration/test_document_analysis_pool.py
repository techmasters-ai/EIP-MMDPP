import os
import pytest


@pytest.mark.skipif(os.environ.get("OLLAMA_LIVE") != "1",
                    reason="requires live Ollama")
def test_extract_document_metadata_via_pool():
    from app.services.document_analysis import extract_document_metadata
    md = "# Hello\n\nThis is a test document about radar systems."
    result = extract_document_metadata(md)
    assert "document_summary" in result
    assert isinstance(result["document_summary"], str)
