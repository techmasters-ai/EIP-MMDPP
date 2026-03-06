"""Unit tests for embedding services.

Tests text embedding (BGE prefix logic, batching) and image embedding
(CLIP) with mocked model loading.
"""

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    def test_empty_list_returns_empty(self):
        from app.services.embedding import embed_texts
        assert embed_texts([]) == []

    @patch("app.services.embedding._get_text_model")
    @patch("app.services.embedding.settings")
    def test_bge_prefix_added(self, mock_settings, mock_get_model):
        from app.services.embedding import embed_texts
        mock_settings.text_embedding_model = "BAAI/bge-large-en-v1.5"
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2]])
        mock_get_model.return_value = model
        embed_texts(["hello"])
        call_args = model.encode.call_args
        assert call_args[0][0][0].startswith("Represent this sentence:")

    @patch("app.services.embedding._get_text_model")
    @patch("app.services.embedding.settings")
    def test_no_prefix_for_non_bge(self, mock_settings, mock_get_model):
        from app.services.embedding import embed_texts
        mock_settings.text_embedding_model = "paraphrase-MiniLM-L6-v2"
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2]])
        mock_get_model.return_value = model
        embed_texts(["hello"])
        call_args = model.encode.call_args
        assert call_args[0][0] == ["hello"]

    @patch("app.services.embedding._get_text_model")
    @patch("app.services.embedding.settings")
    def test_batch_size_passed(self, mock_settings, mock_get_model):
        from app.services.embedding import embed_texts
        mock_settings.text_embedding_model = "paraphrase-MiniLM-L6-v2"
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2]])
        mock_get_model.return_value = model
        embed_texts(["hello"], batch_size=32)
        call_args = model.encode.call_args
        assert call_args[1]["batch_size"] == 32

    @patch("app.services.embedding._get_text_model")
    @patch("app.services.embedding.settings")
    def test_returns_list_of_lists(self, mock_settings, mock_get_model):
        from app.services.embedding import embed_texts
        mock_settings.text_embedding_model = "miniLM"
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = model
        result = embed_texts(["a", "b"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    @patch("app.services.embedding._get_text_model")
    @patch("app.services.embedding.settings")
    def test_returns_single_vector(self, mock_settings, mock_get_model):
        from app.services.embedding import embed_query
        mock_settings.text_embedding_model = "miniLM"
        model = MagicMock()
        model.encode.return_value = np.array([[0.5, 0.6]])
        mock_get_model.return_value = model
        result = embed_query("test query")
        assert result == [0.5, 0.6]


# ---------------------------------------------------------------------------
# embed_images
# ---------------------------------------------------------------------------

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False


class TestEmbedImages:
    def test_empty_list_returns_empty(self):
        from app.services.embedding import embed_images
        assert embed_images([]) == []

    @pytest.mark.skipif(not _has_torch, reason="torch not installed")
    @patch("app.services.embedding._get_clip_model")
    def test_returns_normalized_vectors(self, mock_get_clip):
        from app.services.embedding import embed_images

        mock_model = MagicMock()
        mock_preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
        mock_tokenizer = MagicMock()

        # encode_image returns a tensor
        features = torch.tensor([[3.0, 4.0]])  # norm = 5
        mock_model.encode_image.return_value = features

        mock_get_clip.return_value = (mock_model, mock_preprocess, mock_tokenizer)

        mock_image = MagicMock()
        result = embed_images([mock_image])
        # After normalization: [3/5, 4/5] = [0.6, 0.8]
        assert len(result) == 1
        assert abs(result[0][0] - 0.6) < 0.01
        assert abs(result[0][1] - 0.8) < 0.01


# ---------------------------------------------------------------------------
# embed_text_for_clip
# ---------------------------------------------------------------------------

class TestEmbedTextForClip:
    @pytest.mark.skipif(not _has_torch, reason="torch not installed")
    @patch("app.services.embedding._get_clip_model")
    def test_returns_clip_space_vector(self, mock_get_clip):
        from app.services.embedding import embed_text_for_clip

        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_tokenizer = MagicMock(return_value=torch.zeros(1, 77, dtype=torch.long))

        features = torch.tensor([[1.0, 0.0]])
        mock_model.encode_text.return_value = features

        mock_get_clip.return_value = (mock_model, mock_preprocess, mock_tokenizer)

        result = embed_text_for_clip("radar system")
        assert result == [1.0, 0.0]
