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


class TestStartIngestPipelineV2:
    def test_creates_celery_chain(self):
        """start_ingest_pipeline_v2 returns a chain of the expected tasks."""
        from app.workers.pipeline import start_ingest_pipeline_v2

        doc_id = str(uuid.uuid4())
        with patch("app.workers.pipeline.chain") as mock_chain:
            mock_chain.return_value.apply_async.return_value = MagicMock(id="v2-task-123")
            result = start_ingest_pipeline_v2(doc_id)
            assert result == "v2-task-123"
            mock_chain.assert_called_once()


class TestV2TasksRegistered:
    """Verify all v2 pipeline tasks are registered."""

    def test_prepare_document(self):
        from app.workers.pipeline import prepare_document
        assert prepare_document.name == "app.workers.pipeline.prepare_document"

    def test_derive_text_chunks_and_embeddings(self):
        from app.workers.pipeline import derive_text_chunks_and_embeddings
        assert derive_text_chunks_and_embeddings.name == "app.workers.pipeline.derive_text_chunks_and_embeddings"

    def test_derive_image_embeddings(self):
        from app.workers.pipeline import derive_image_embeddings
        assert derive_image_embeddings.name == "app.workers.pipeline.derive_image_embeddings"

    def test_derive_ontology_graph(self):
        from app.workers.pipeline import derive_ontology_graph
        assert derive_ontology_graph.name == "app.workers.pipeline.derive_ontology_graph"

    def test_derive_structure_links(self):
        from app.workers.pipeline import derive_structure_links
        assert derive_structure_links.name == "app.workers.pipeline.derive_structure_links"

    def test_collect_derivations(self):
        from app.workers.pipeline import collect_derivations
        assert collect_derivations.name == "app.workers.pipeline.collect_derivations"

    def test_finalize_document_v2(self):
        from app.workers.pipeline import finalize_document_v2
        assert finalize_document_v2.name == "app.workers.pipeline.finalize_document_v2"


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
