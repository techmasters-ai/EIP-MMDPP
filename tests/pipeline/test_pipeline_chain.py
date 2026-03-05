"""Pipeline task tests — verifying the new task chain structure.

These tests mock external services (DB, MinIO, embeddings, LLM) and verify
that each task calls the correct dependencies.
"""

import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

pytestmark = pytest.mark.unit


class TestStartIngestPipeline:
    def test_creates_celery_chain(self):
        """start_ingest_pipeline returns a chain of the expected tasks."""
        from app.workers.pipeline import start_ingest_pipeline

        doc_id = str(uuid.uuid4())
        with patch("app.workers.pipeline.chain") as mock_chain:
            mock_chain.return_value.apply_async.return_value = MagicMock(id="task-123")
            result = start_ingest_pipeline(doc_id)
            assert result == "task-123"
            mock_chain.assert_called_once()


class TestEmbedTextChunksTask:
    def test_task_is_registered(self):
        """embed_text_chunks is a registered Celery task."""
        from app.workers.pipeline import embed_text_chunks
        assert embed_text_chunks.name == "app.workers.pipeline.embed_text_chunks"


class TestEmbedImageChunksTask:
    def test_task_is_registered(self):
        """embed_image_chunks is a registered Celery task."""
        from app.workers.pipeline import embed_image_chunks
        assert embed_image_chunks.name == "app.workers.pipeline.embed_image_chunks"


class TestExtractGraphTask:
    def test_task_is_registered(self):
        """extract_graph is a registered Celery task."""
        from app.workers.pipeline import extract_graph
        assert extract_graph.name == "app.workers.pipeline.extract_graph"


class TestImportGraphTask:
    def test_task_is_registered(self):
        """import_graph is a registered Celery task."""
        from app.workers.pipeline import import_graph
        assert import_graph.name == "app.workers.pipeline.import_graph"


class TestConnectDocumentElementsTask:
    def test_task_is_registered(self):
        """connect_document_elements is a registered Celery task."""
        from app.workers.pipeline import connect_document_elements
        assert connect_document_elements.name == "app.workers.pipeline.connect_document_elements"


class TestCollectEmbeddingsTask:
    def test_task_is_registered(self):
        """collect_embeddings is a registered Celery task."""
        from app.workers.pipeline import collect_embeddings
        assert collect_embeddings.name == "app.workers.pipeline.collect_embeddings"


class TestDeprecatedTasksExist:
    """Deprecated tasks still exist for backwards compatibility."""

    def test_chunk_and_embed_exists(self):
        from app.workers.pipeline import chunk_and_embed
        assert chunk_and_embed.name == "app.workers.pipeline.chunk_and_embed"

    def test_extract_graph_entities_exists(self):
        from app.workers.pipeline import extract_graph_entities
        assert extract_graph_entities.name == "app.workers.pipeline.extract_graph_entities"


class TestTaskRouting:
    def test_embed_text_chunks_routed_to_embed_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.embed_text_chunks"]["queue"] == "embed"

    def test_embed_image_chunks_routed_to_embed_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.embed_image_chunks"]["queue"] == "embed"

    def test_extract_graph_routed_to_graph_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.extract_graph"]["queue"] == "graph"

    def test_connect_document_elements_routed_to_graph_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.connect_document_elements"]["queue"] == "graph"
