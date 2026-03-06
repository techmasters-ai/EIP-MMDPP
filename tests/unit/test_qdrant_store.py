"""Unit tests for Qdrant vector store operations.

Tests collection management, filter building, upsert, search (sync + async),
and delete operations with a mocked Qdrant client.

Requires: qdrant_client package (skipped if not installed).
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant_client not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_settings(**overrides):
    """Return a mock Settings with Qdrant defaults."""
    s = MagicMock()
    s.qdrant_text_collection = "eip_text_chunks"
    s.qdrant_image_collection = "eip_image_chunks"
    s.text_embedding_dim = 1024
    s.image_embedding_dim = 512
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _mock_point(pid="abc", score=0.9, payload=None):
    """Build a mock Qdrant ScoredPoint."""
    p = MagicMock()
    p.id = pid
    p.score = score
    p.payload = payload or {"chunk_id": "c1", "modality": "text"}
    return p


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------

class TestBuildFilter:
    def test_empty_dict_returns_empty_filter(self):
        from app.services.qdrant_store import _build_filter
        f = _build_filter({})
        assert f.must == [] or f.must is None or len(f.must) == 0

    def test_single_field_creates_condition(self):
        from app.services.qdrant_store import _build_filter
        f = _build_filter({"document_id": "abc"})
        assert len(f.must) == 1
        assert f.must[0].key == "document_id"

    def test_none_values_excluded(self):
        from app.services.qdrant_store import _build_filter
        f = _build_filter({"a": "val", "b": None})
        assert len(f.must) == 1

    def test_multiple_fields(self):
        from app.services.qdrant_store import _build_filter
        f = _build_filter({"classification": "SECRET", "modality": "text"})
        assert len(f.must) == 2

    def test_values_stringified(self):
        from app.services.qdrant_store import _build_filter
        f = _build_filter({"page_number": 5})
        assert f.must[0].match.value == "5"

    def test_list_value_uses_match_any(self):
        from app.services.qdrant_store import _build_filter
        from qdrant_client.models import MatchAny
        f = _build_filter({"document_id": ["a", "b", "c"]})
        assert len(f.must) == 1
        assert f.must[0].key == "document_id"
        assert isinstance(f.must[0].match, MatchAny)
        assert f.must[0].match.any == ["a", "b", "c"]

    def test_mixed_scalar_and_list(self):
        from app.services.qdrant_store import _build_filter
        from qdrant_client.models import MatchAny, MatchValue
        f = _build_filter({"classification": "SECRET", "document_id": ["a", "b"]})
        assert len(f.must) == 2
        by_key = {c.key: c for c in f.must}
        assert isinstance(by_key["classification"].match, MatchValue)
        assert isinstance(by_key["document_id"].match, MatchAny)


# ---------------------------------------------------------------------------
# ensure_collections
# ---------------------------------------------------------------------------

class TestEnsureCollections:
    @patch("app.services.qdrant_store.get_settings")
    def test_creates_both_when_none_exist(self, mock_gs):
        from app.services.qdrant_store import ensure_collections
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.get_collections.return_value.collections = []
        ensure_collections(client)
        assert client.create_collection.call_count == 2

    @patch("app.services.qdrant_store.get_settings")
    def test_skips_existing_text_collection(self, mock_gs):
        from app.services.qdrant_store import ensure_collections
        mock_gs.return_value = _mock_settings()
        existing = MagicMock()
        existing.name = "eip_text_chunks"
        client = MagicMock()
        client.get_collections.return_value.collections = [existing]
        ensure_collections(client)
        assert client.create_collection.call_count == 1

    @patch("app.services.qdrant_store.get_settings")
    def test_skips_both_if_both_exist(self, mock_gs):
        from app.services.qdrant_store import ensure_collections
        mock_gs.return_value = _mock_settings()
        t = MagicMock(); t.name = "eip_text_chunks"
        i = MagicMock(); i.name = "eip_image_chunks"
        client = MagicMock()
        client.get_collections.return_value.collections = [t, i]
        ensure_collections(client)
        assert client.create_collection.call_count == 0

    @patch("app.services.qdrant_store.get_settings")
    def test_creates_payload_indexes(self, mock_gs):
        from app.services.qdrant_store import ensure_collections
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.get_collections.return_value.collections = []
        ensure_collections(client)
        # 5 fields × 2 collections = 10 index calls
        assert client.create_payload_index.call_count == 10


# ---------------------------------------------------------------------------
# upsert operations
# ---------------------------------------------------------------------------

class TestUpsertTextVector:
    @patch("app.services.qdrant_store.get_settings")
    def test_calls_upsert_with_correct_collection(self, mock_gs):
        from app.services.qdrant_store import upsert_text_vector
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        pid = uuid.uuid4()
        upsert_text_vector(client, pid, [0.1, 0.2], {"chunk_id": "c1"})
        client.upsert.assert_called_once()
        args = client.upsert.call_args
        assert args.kwargs["collection_name"] == "eip_text_chunks"

    @patch("app.services.qdrant_store.get_settings")
    def test_uuid_converted_to_string(self, mock_gs):
        from app.services.qdrant_store import upsert_text_vector
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        pid = uuid.uuid4()
        upsert_text_vector(client, pid, [0.1], {"k": "v"})
        points = client.upsert.call_args.kwargs["points"]
        assert points[0].id == str(pid)


class TestUpsertImageVector:
    @patch("app.services.qdrant_store.get_settings")
    def test_calls_upsert_on_image_collection(self, mock_gs):
        from app.services.qdrant_store import upsert_image_vector
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        upsert_image_vector(client, uuid.uuid4(), [0.1], {"k": "v"})
        assert client.upsert.call_args.kwargs["collection_name"] == "eip_image_chunks"


class TestUpsertBatch:
    @patch("app.services.qdrant_store.get_settings")
    def test_text_batch(self, mock_gs):
        from app.services.qdrant_store import upsert_text_vectors_batch
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        upsert_text_vectors_batch(client, [MagicMock(), MagicMock()])
        assert client.upsert.call_args.kwargs["collection_name"] == "eip_text_chunks"

    @patch("app.services.qdrant_store.get_settings")
    def test_image_batch(self, mock_gs):
        from app.services.qdrant_store import upsert_image_vectors_batch
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        upsert_image_vectors_batch(client, [MagicMock()])
        assert client.upsert.call_args.kwargs["collection_name"] == "eip_image_chunks"


# ---------------------------------------------------------------------------
# sync search
# ---------------------------------------------------------------------------

class TestSearchTextVectors:
    @patch("app.services.qdrant_store.get_settings")
    def test_returns_formatted_results(self, mock_gs):
        from app.services.qdrant_store import search_text_vectors
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.query_points.return_value.points = [_mock_point()]
        results = search_text_vectors(client, [0.1, 0.2])
        assert len(results) == 1
        assert results[0]["score"] == 0.9
        assert "payload" in results[0]

    @patch("app.services.qdrant_store.get_settings")
    def test_filter_applied(self, mock_gs):
        from app.services.qdrant_store import search_text_vectors
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.query_points.return_value.points = []
        search_text_vectors(client, [0.1], filters={"classification": "SECRET"})
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    @patch("app.services.qdrant_store.get_settings")
    def test_no_filter_when_none(self, mock_gs):
        from app.services.qdrant_store import search_text_vectors
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.query_points.return_value.points = []
        search_text_vectors(client, [0.1], filters=None)
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    @patch("app.services.qdrant_store.get_settings")
    def test_empty_results(self, mock_gs):
        from app.services.qdrant_store import search_text_vectors
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.query_points.return_value.points = []
        assert search_text_vectors(client, [0.1]) == []


class TestSearchImageVectors:
    @patch("app.services.qdrant_store.get_settings")
    def test_results_from_image_collection(self, mock_gs):
        from app.services.qdrant_store import search_image_vectors
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.query_points.return_value.points = [_mock_point()]
        results = search_image_vectors(client, [0.1])
        assert client.query_points.call_args.kwargs["collection_name"] == "eip_image_chunks"
        assert len(results) == 1

    @patch("app.services.qdrant_store.get_settings")
    def test_filter_applied(self, mock_gs):
        from app.services.qdrant_store import search_image_vectors
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.query_points.return_value.points = []
        search_image_vectors(client, [0.1], filters={"modality": "image"})
        assert client.query_points.call_args.kwargs["query_filter"] is not None


# ---------------------------------------------------------------------------
# async search
# ---------------------------------------------------------------------------

class TestSearchTextVectorsAsync:
    @pytest.mark.asyncio
    @patch("app.services.qdrant_store.get_settings")
    async def test_returns_formatted_results(self, mock_gs):
        from app.services.qdrant_store import search_text_vectors_async
        mock_gs.return_value = _mock_settings()
        client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.points = [_mock_point()]
        client.query_points.return_value = mock_resp
        results = await search_text_vectors_async(client, [0.1])
        assert len(results) == 1

    @pytest.mark.asyncio
    @patch("app.services.qdrant_store.get_settings")
    async def test_filter_applied(self, mock_gs):
        from app.services.qdrant_store import search_text_vectors_async
        mock_gs.return_value = _mock_settings()
        client = AsyncMock()
        client.query_points.return_value.points = []
        await search_text_vectors_async(client, [0.1], filters={"a": "b"})
        assert client.query_points.call_args.kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    @patch("app.services.qdrant_store.get_settings")
    async def test_empty_results(self, mock_gs):
        from app.services.qdrant_store import search_text_vectors_async
        mock_gs.return_value = _mock_settings()
        client = AsyncMock()
        client.query_points.return_value.points = []
        assert await search_text_vectors_async(client, [0.1]) == []


class TestSearchImageVectorsAsync:
    @pytest.mark.asyncio
    @patch("app.services.qdrant_store.get_settings")
    async def test_image_collection(self, mock_gs):
        from app.services.qdrant_store import search_image_vectors_async
        mock_gs.return_value = _mock_settings()
        client = AsyncMock()
        client.query_points.return_value.points = [_mock_point()]
        await search_image_vectors_async(client, [0.1])
        assert client.query_points.call_args.kwargs["collection_name"] == "eip_image_chunks"

    @pytest.mark.asyncio
    @patch("app.services.qdrant_store.get_settings")
    async def test_filter_applied(self, mock_gs):
        from app.services.qdrant_store import search_image_vectors_async
        mock_gs.return_value = _mock_settings()
        client = AsyncMock()
        client.query_points.return_value.points = []
        await search_image_vectors_async(client, [0.1], filters={"x": "y"})
        assert client.query_points.call_args.kwargs["query_filter"] is not None


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDeleteByDocumentId:
    @patch("app.services.qdrant_store.get_settings")
    def test_deletes_from_both_collections(self, mock_gs):
        from app.services.qdrant_store import delete_by_document_id
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        delete_by_document_id(client, "doc-123")
        assert client.delete.call_count == 2

    @patch("app.services.qdrant_store.get_settings")
    def test_swallows_exception_per_collection(self, mock_gs):
        from app.services.qdrant_store import delete_by_document_id
        mock_gs.return_value = _mock_settings()
        client = MagicMock()
        client.delete.side_effect = [Exception("fail"), None]
        # Should not raise
        delete_by_document_id(client, "doc-123")
        assert client.delete.call_count == 2
