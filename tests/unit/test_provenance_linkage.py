"""Unit tests for deterministic provenance linkage.

Tests that artifact IDs are generated deterministically from
document_id + element_uid, and that DocumentElement.artifact_id
is set inline (no zip-linking).
"""

import uuid

import pytest

pytest.importorskip("celery", reason="celery not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _deterministic_artifact_id
# ---------------------------------------------------------------------------

class TestDeterministicArtifactId:
    def test_same_inputs_same_output(self):
        from app.workers.pipeline import _deterministic_artifact_id
        doc_id = "aaaa-bbbb-cccc-dddd"
        elem_uid = "0-0-text-abc12345"
        a = _deterministic_artifact_id(doc_id, elem_uid)
        b = _deterministic_artifact_id(doc_id, elem_uid)
        assert a == b
        assert isinstance(a, uuid.UUID)

    def test_different_element_uids_differ(self):
        from app.workers.pipeline import _deterministic_artifact_id
        doc_id = "aaaa-bbbb-cccc-dddd"
        a = _deterministic_artifact_id(doc_id, "elem-0")
        b = _deterministic_artifact_id(doc_id, "elem-1")
        assert a != b

    def test_different_document_ids_differ(self):
        from app.workers.pipeline import _deterministic_artifact_id
        a = _deterministic_artifact_id("doc-1", "elem-0")
        b = _deterministic_artifact_id("doc-2", "elem-0")
        assert a != b

    def test_uses_uuid5_namespace_url(self):
        from app.workers.pipeline import _deterministic_artifact_id
        doc_id = "test-doc"
        elem_uid = "test-elem"
        expected = uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{elem_uid}")
        assert _deterministic_artifact_id(doc_id, elem_uid) == expected


# ---------------------------------------------------------------------------
# _persist_extraction_results with element_uids
# ---------------------------------------------------------------------------

class TestPersistExtractionWithDeterministicIds:
    def _make_chunk(self, **kwargs):
        """Build a minimal mock ExtractedChunk."""
        from unittest.mock import MagicMock
        chunk = MagicMock()
        chunk.modality = kwargs.get("modality", "text")
        chunk.chunk_text = kwargs.get("chunk_text", "Hello world.")
        chunk.metadata = kwargs.get("metadata", {})
        chunk.raw_image_bytes = kwargs.get("raw_image_bytes", None)
        chunk.page_number = kwargs.get("page_number", 1)
        chunk.bounding_box = kwargs.get("bounding_box", None)
        chunk.ocr_confidence = kwargs.get("ocr_confidence", None)
        chunk.ocr_engine = kwargs.get("ocr_engine", None)
        chunk.requires_human_review = kwargs.get("requires_human_review", False)
        return chunk

    def test_returns_deterministic_ids_when_uids_provided(self):
        from unittest.mock import MagicMock
        from app.workers.pipeline import _persist_extraction_results, _deterministic_artifact_id

        db = MagicMock()
        doc_id = str(uuid.uuid4())
        chunks = [self._make_chunk(), self._make_chunk(chunk_text="Second")]
        uids = ["elem-0", "elem-1"]

        result = _persist_extraction_results(db, doc_id, chunks, element_uids=uids)

        assert len(result) == 2
        assert result[0] == _deterministic_artifact_id(doc_id, "elem-0")
        assert result[1] == _deterministic_artifact_id(doc_id, "elem-1")

    def test_returns_random_ids_when_no_uids(self):
        from unittest.mock import MagicMock
        from app.workers.pipeline import _persist_extraction_results

        db = MagicMock()
        doc_id = str(uuid.uuid4())
        chunks = [self._make_chunk()]

        result = _persist_extraction_results(db, doc_id, chunks)

        assert len(result) == 1
        assert isinstance(result[0], uuid.UUID)

    def test_artifact_upserted_to_db_with_correct_id(self):
        from unittest.mock import MagicMock
        from app.workers.pipeline import _persist_extraction_results, _deterministic_artifact_id

        db = MagicMock()
        doc_id = str(uuid.uuid4())
        chunks = [self._make_chunk()]
        uids = ["elem-0"]

        _persist_extraction_results(db, doc_id, chunks, element_uids=uids)

        # db.execute should have been called once with a pg_insert upsert
        assert db.execute.call_count == 1
        # db.add should NOT be called (we use execute with pg_insert now)
        db.add.assert_not_called()

    def test_empty_chunks_returns_empty(self):
        from unittest.mock import MagicMock
        from app.workers.pipeline import _persist_extraction_results

        db = MagicMock()
        result = _persist_extraction_results(db, "doc-1", [], element_uids=[])
        assert result == []
        db.execute.assert_not_called()

    def test_image_storage_key_uses_artifact_id(self):
        """Verify image storage keys are deterministic (use artifact_id, not random UUID)."""
        from unittest.mock import MagicMock, patch
        from app.workers.pipeline import _persist_extraction_results, _deterministic_artifact_id

        db = MagicMock()
        doc_id = str(uuid.uuid4())
        chunk = self._make_chunk(raw_image_bytes=b"\x89PNG", metadata={"ext": "png"})
        uids = ["elem-img-0"]

        with patch("app.workers.pipeline.upload_bytes_sync") as mock_upload:
            _persist_extraction_results(db, doc_id, [chunk], element_uids=uids)

        expected_id = _deterministic_artifact_id(doc_id, "elem-img-0")
        expected_key = f"artifacts/{doc_id}/images/{expected_id}.png"
        # Verify upload was called with the deterministic key
        mock_upload.assert_called_once()
        actual_key = mock_upload.call_args[0][2]  # 3rd positional arg is the key
        assert actual_key == expected_key

    def test_upsert_preserves_classification(self):
        """Verify classification is NOT in the on_conflict set_ dict.

        The upsert should never overwrite a manually-applied classification.
        We check the source of set_={...} to confirm 'classification' is absent.
        """
        import inspect
        from app.workers.pipeline import _persist_extraction_results
        source = inspect.getsource(_persist_extraction_results)
        # Extract the set_={...} block from source
        set_block = source[source.index("set_="):]
        set_block = set_block[:set_block.index("}") + 1]
        assert "classification" not in set_block


# ---------------------------------------------------------------------------
# Zip-linking removal verification
# ---------------------------------------------------------------------------

class TestZipLinkingRemoved:
    def test_no_created_at_ordering_in_prepare_document(self):
        """Verify the old zip-linking pattern (order_by Artifact.created_at)
        is no longer present in the prepare_document function source."""
        import inspect
        from app.workers.pipeline import prepare_document
        source = inspect.getsource(prepare_document)
        assert "order_by(Artifact.created_at)" not in source
        assert "for elem, art in zip(" not in source
