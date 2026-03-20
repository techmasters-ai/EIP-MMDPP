"""Tests for the cross-encoder reranker service."""

import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


def test_rerank_returns_sorted_by_score():
    """Reranker should return candidates sorted by cross-encoder score."""
    from app.services.reranker import rerank

    candidates = [
        {"chunk_id": "a", "content_text": "Patriot missile system overview"},
        {"chunk_id": "b", "content_text": "Weather forecast for tomorrow"},
        {"chunk_id": "c", "content_text": "PAC-3 guidance section specifications"},
    ]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.8, 0.1, 0.95]

    mock_settings = MagicMock()
    mock_settings.reranker_enabled = True
    with patch("app.services.reranker.get_settings", return_value=mock_settings), \
         patch("app.services.reranker._get_reranker_model", return_value=mock_model):
        result = rerank("Patriot PAC-3 guidance", candidates, top_k=2)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "c"
    assert result[1]["chunk_id"] == "a"


def test_rerank_disabled_returns_unchanged():
    """When reranker is disabled, return input unchanged."""
    from app.services.reranker import rerank

    candidates = [{"chunk_id": "a", "content_text": "text", "score": 0.5}]

    with patch("app.services.reranker.get_settings") as mock_settings_fn:
        mock_s = MagicMock()
        mock_s.reranker_enabled = False
        mock_settings_fn.return_value = mock_s
        result = rerank("query", candidates, top_k=10)

    assert len(result) == 1
    assert result[0]["chunk_id"] == "a"


def test_rerank_empty_candidates():
    """Empty candidates should return empty list."""
    from app.services.reranker import rerank

    with patch("app.services.reranker.get_settings") as mock_settings_fn:
        mock_s = MagicMock()
        mock_s.reranker_enabled = True
        mock_settings_fn.return_value = mock_s
        result = rerank("query", [], top_k=10)

    assert result == []


def test_rerank_unscorable_appended():
    """Candidates without content_text should be appended at the end."""
    from app.services.reranker import rerank

    candidates = [
        {"chunk_id": "a", "content_text": "some text"},
        {"chunk_id": "b"},  # no content_text
    ]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9]

    with patch("app.services.reranker._get_reranker_model", return_value=mock_model):
        result = rerank("query", candidates, top_k=10)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "a"
    assert result[1]["chunk_id"] == "b"
