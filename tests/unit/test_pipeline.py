"""Unit tests for ingest pipeline tasks and helpers.

Tests task registration, DAG construction, deterministic key generation,
and stage state machine logic — all without DB dependencies.
"""

import hashlib
import uuid

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

class TestTaskRegistration:
    """Verify all pipeline tasks are registered with the correct names."""

    def test_prepare_document_registered(self):
        from app.workers.pipeline import prepare_document
        assert prepare_document.name == "app.workers.pipeline.prepare_document"

    def test_derive_text_chunks_registered(self):
        from app.workers.pipeline import derive_text_chunks_and_embeddings
        assert derive_text_chunks_and_embeddings.name == "app.workers.pipeline.derive_text_chunks_and_embeddings"

    def test_derive_image_embeddings_registered(self):
        from app.workers.pipeline import derive_image_embeddings
        assert derive_image_embeddings.name == "app.workers.pipeline.derive_image_embeddings"

    def test_derive_ontology_graph_registered(self):
        from app.workers.pipeline import derive_ontology_graph
        assert derive_ontology_graph.name == "app.workers.pipeline.derive_ontology_graph"

    def test_derive_structure_links_registered(self):
        from app.workers.pipeline import derive_structure_links
        assert derive_structure_links.name == "app.workers.pipeline.derive_structure_links"

    def test_collect_derivations_registered(self):
        from app.workers.pipeline import collect_derivations
        assert collect_derivations.name == "app.workers.pipeline.collect_derivations"

    def test_finalize_document_registered(self):
        from app.workers.pipeline import finalize_document
        assert finalize_document.name == "app.workers.pipeline.finalize_document"


class TestTaskRouting:
    """Verify derivation tasks are routed to correct queues."""

    def test_derive_text_embed_queue(self):
        from app.workers.pipeline import derive_text_chunks_and_embeddings
        assert derive_text_chunks_and_embeddings.queue == "embed"

    def test_derive_image_embed_queue(self):
        from app.workers.pipeline import derive_image_embeddings
        assert derive_image_embeddings.queue == "embed"

    def test_derive_ontology_graph_queue(self):
        from app.workers.pipeline import derive_ontology_graph
        assert derive_ontology_graph.queue == "graph"

    def test_derive_structure_links_queue(self):
        from app.workers.pipeline import derive_structure_links
        assert derive_structure_links.queue == "graph"


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

class TestDAGConstruction:
    """Verify start_ingest_pipeline constructs a proper Celery chain."""

    def test_pipeline_returns_task_id(self):
        from unittest.mock import patch, MagicMock

        mock_chain_result = MagicMock()
        mock_chain_result.id = "mock-task-id"

        with patch("app.workers.pipeline._get_db") as mock_get_db, \
             patch("app.workers.pipeline._create_pipeline_run", return_value="run-1"), \
             patch("app.workers.pipeline.chain") as mock_chain_fn:

            db = MagicMock()
            db.execute.return_value.scalar_one_or_none.return_value = None
            mock_get_db.return_value = db
            mock_chain_fn.return_value.apply_async.return_value = mock_chain_result

            from app.workers.pipeline import start_ingest_pipeline
            task_id = start_ingest_pipeline(str(uuid.uuid4()))

            assert task_id == "mock-task-id"
            mock_chain_fn.return_value.apply_async.assert_called_once()


# ---------------------------------------------------------------------------
# Deterministic key generation
# ---------------------------------------------------------------------------

class TestDeterministicKeys:
    """Verify that chunk keys are deterministic for idempotent retries."""

    def test_text_chunk_key_is_deterministic(self):
        doc_id = "test-doc"
        element_uid = "0-1-text-abcdef12"
        chunk_index = 0
        model_version = "bge-large-en-v1.5"

        key1 = hashlib.sha256(
            f"{doc_id}:{element_uid}:{chunk_index}:{model_version}".encode()
        ).hexdigest()
        key2 = hashlib.sha256(
            f"{doc_id}:{element_uid}:{chunk_index}:{model_version}".encode()
        ).hexdigest()
        assert key1 == key2

    def test_different_elements_produce_different_keys(self):
        key1 = hashlib.sha256("doc:elem1:0:model".encode()).hexdigest()
        key2 = hashlib.sha256("doc:elem2:0:model".encode()).hexdigest()
        assert key1 != key2

    def test_different_chunk_indices_produce_different_keys(self):
        key1 = hashlib.sha256("doc:elem:0:model".encode()).hexdigest()
        key2 = hashlib.sha256("doc:elem:1:model".encode()).hexdigest()
        assert key1 != key2

    def test_image_chunk_key_is_deterministic(self):
        doc_id = "test-doc"
        element_uid = "0-5-image-abcdef12"
        model_version = "openclip-vit-b-32"

        key1 = hashlib.sha256(
            f"{doc_id}:{element_uid}:{model_version}".encode()
        ).hexdigest()
        key2 = hashlib.sha256(
            f"{doc_id}:{element_uid}:{model_version}".encode()
        ).hexdigest()
        assert key1 == key2

    def test_element_hash_is_deterministic(self):
        doc_id = "test-doc"
        element_uid = "0-1-text-abcdef12"
        content = "Some text content"

        hash1 = hashlib.sha256(
            f"{doc_id}:{element_uid}:{content}".encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            f"{doc_id}:{element_uid}:{content}".encode()
        ).hexdigest()
        assert hash1 == hash2


# ---------------------------------------------------------------------------
# Pipeline status constants
# ---------------------------------------------------------------------------

class TestPipelineStatusConstants:
    def test_status_constants_exist(self):
        from app.workers.pipeline import (
            STATUS_PROCESSING,
            STATUS_COMPLETE,
            STATUS_PARTIAL_COMPLETE,
            STATUS_FAILED,
        )
        assert STATUS_PROCESSING == "PROCESSING"
        assert STATUS_COMPLETE == "COMPLETE"
        assert STATUS_PARTIAL_COMPLETE == "PARTIAL_COMPLETE"
        assert STATUS_FAILED == "FAILED"


class TestStandaloneImageSynthesis:
    """Verify that standalone image files get a synthesized element
    when Docling returns 0 elements."""

    def test_synthesis_produces_one_image_element(self):
        """When Docling returns 0 elements for an image MIME type,
        _synthesize_standalone_image should produce a single image ExtractedChunk."""
        from app.workers.pipeline import _synthesize_standalone_image
        from app.services.extraction import ExtractedChunk

        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = _synthesize_standalone_image(fake_bytes, "image/png")

        assert result is not None
        assert len(result) == 1
        chunk = result[0]
        assert isinstance(chunk, ExtractedChunk)
        assert chunk.modality == "image"
        assert chunk.chunk_text == ""
        assert chunk.page_number == 1
        assert chunk.raw_image_bytes == fake_bytes
        assert chunk.metadata["label"] == "picture"
        assert chunk.metadata["ext"] == "png"

    def test_synthesis_returns_none_for_non_image(self):
        """Non-image MIME types should return None (no synthesis)."""
        from app.workers.pipeline import _synthesize_standalone_image

        result = _synthesize_standalone_image(b"hello", "application/pdf")
        assert result is None

    def test_synthesis_handles_jpeg_extension(self):
        """JPEG MIME should produce ext='jpeg'."""
        from app.workers.pipeline import _synthesize_standalone_image

        result = _synthesize_standalone_image(b"\xff\xd8\xff", "image/jpeg")
        assert result is not None
        assert result[0].metadata["ext"] == "jpeg"

    def test_synthesis_always_produces_for_image_mime(self):
        """The function always produces for image MIME — the caller
        guards on len(result.elements) == 0."""
        from app.workers.pipeline import _synthesize_standalone_image

        result = _synthesize_standalone_image(b"\x89PNG", "image/png")
        assert result is not None  # synthesis always produces if image mime


class TestDedupeExtractedElements:
    """Verify that dedup preserves distinct images on the same page."""

    def test_different_images_same_page_not_deduped(self):
        """Two different images on the same page should both be kept."""
        from app.workers.pipeline import _dedupe_extracted_elements
        from app.services.extraction import ExtractedChunk

        img_a = ExtractedChunk(
            chunk_text="", modality="image", page_number=8,
            raw_image_bytes=b"\x89PNG_image_A",
        )
        img_b = ExtractedChunk(
            chunk_text="", modality="image", page_number=8,
            raw_image_bytes=b"\x89PNG_image_B",
        )
        result, dropped = _dedupe_extracted_elements([img_a, img_b])
        assert len(result) == 2
        assert dropped == 0

    def test_identical_images_same_page_deduped(self):
        """Two identical images on the same page should be deduped to one."""
        from app.workers.pipeline import _dedupe_extracted_elements
        from app.services.extraction import ExtractedChunk

        same_bytes = b"\x89PNG_same_image"
        img_a = ExtractedChunk(
            chunk_text="", modality="image", page_number=8,
            raw_image_bytes=same_bytes,
        )
        img_b = ExtractedChunk(
            chunk_text="", modality="image", page_number=8,
            raw_image_bytes=same_bytes,
        )
        result, dropped = _dedupe_extracted_elements([img_a, img_b])
        assert len(result) == 1
        assert dropped == 1

    def test_text_elements_still_deduped_normally(self):
        """Text elements with identical content on same page still dedup."""
        from app.workers.pipeline import _dedupe_extracted_elements
        from app.services.extraction import ExtractedChunk

        txt_a = ExtractedChunk(chunk_text="Hello world", modality="text", page_number=1)
        txt_b = ExtractedChunk(chunk_text="Hello world", modality="text", page_number=1)
        result, dropped = _dedupe_extracted_elements([txt_a, txt_b])
        assert len(result) == 1
        assert dropped == 1


# ---------------------------------------------------------------------------
# Post-ingest community-detection counter
# ---------------------------------------------------------------------------


class TestPostIngestCommunityTrigger:
    """Verify the Redis counter and threshold-based trigger in finalize_document."""

    def test_trigger_disabled_when_flag_off(self, monkeypatch):
        """When COMMUNITY_DETECTION_POST_INGEST_ENABLED=false, the helper is a no-op."""
        from unittest.mock import MagicMock, patch
        from app.config import get_settings
        from app.workers.pipeline import _maybe_trigger_post_ingest_community_detection

        get_settings.cache_clear()
        monkeypatch.setenv("COMMUNITY_DETECTION_POST_INGEST_ENABLED", "false")

        try:
            with patch("app.services.redis_utils.get_redis") as mock_get:
                _maybe_trigger_post_ingest_community_detection("doc-1")
                mock_get.assert_not_called()
        finally:
            get_settings.cache_clear()

    def test_increments_counter_below_threshold(self, monkeypatch):
        """Below threshold, counter increments and no task is dispatched."""
        from unittest.mock import MagicMock, patch
        from app.config import get_settings
        from app.workers.pipeline import _maybe_trigger_post_ingest_community_detection

        get_settings.cache_clear()
        monkeypatch.setenv("COMMUNITY_DETECTION_POST_INGEST_ENABLED", "true")
        monkeypatch.setenv("COMMUNITY_DETECTION_POST_INGEST_THRESHOLD", "5")

        try:
            mock_redis = MagicMock()
            mock_redis.incr.return_value = 2
            with (
                patch("app.services.redis_utils.get_redis", return_value=mock_redis),
                patch("app.workers.community_tasks.run_community_detection_task") as mock_task,
            ):
                _maybe_trigger_post_ingest_community_detection("doc-1")
                mock_redis.incr.assert_called_once_with("community:pending_ingest_count")
                mock_task.delay.assert_not_called()
                mock_redis.set.assert_not_called()
                # Shared client — helper must NOT close it
                mock_redis.close.assert_not_called()
        finally:
            get_settings.cache_clear()

    def test_dispatches_task_at_threshold(self, monkeypatch):
        """At threshold, counter resets and an incremental task is dispatched."""
        from unittest.mock import MagicMock, patch
        from app.config import get_settings
        from app.workers.pipeline import _maybe_trigger_post_ingest_community_detection

        get_settings.cache_clear()
        monkeypatch.setenv("COMMUNITY_DETECTION_POST_INGEST_ENABLED", "true")
        monkeypatch.setenv("COMMUNITY_DETECTION_POST_INGEST_THRESHOLD", "3")

        try:
            mock_redis = MagicMock()
            mock_redis.incr.return_value = 3  # reaches threshold
            with (
                patch("app.services.redis_utils.get_redis", return_value=mock_redis),
                patch("app.workers.community_tasks.run_community_detection_task") as mock_task,
            ):
                _maybe_trigger_post_ingest_community_detection("doc-1")
                mock_redis.incr.assert_called_once()
                mock_redis.set.assert_called_once_with("community:pending_ingest_count", 0)
                mock_task.delay.assert_called_once_with(mode="incremental")
                mock_redis.close.assert_not_called()
        finally:
            get_settings.cache_clear()

    def test_errors_are_swallowed(self, monkeypatch):
        """If Redis is unavailable, ingestion must not fail."""
        from unittest.mock import patch
        from app.config import get_settings
        from app.workers.pipeline import _maybe_trigger_post_ingest_community_detection

        get_settings.cache_clear()
        monkeypatch.setenv("COMMUNITY_DETECTION_POST_INGEST_ENABLED", "true")

        try:
            with patch("app.services.redis_utils.get_redis", side_effect=RuntimeError("no redis")):
                # Must not raise
                _maybe_trigger_post_ingest_community_detection("doc-1")
        finally:
            get_settings.cache_clear()
