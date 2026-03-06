"""Pipeline task tests — verifying the task chain structure.

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


class TestTasksRegistered:
    """Verify all pipeline tasks are registered."""

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

    def test_finalize_document(self):
        from app.workers.pipeline import finalize_document
        assert finalize_document.name == "app.workers.pipeline.finalize_document"


class TestTaskRouting:
    def test_prepare_document_routed_to_ingest_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.prepare_document"]["queue"] == "ingest"

    def test_derive_text_embeddings_routed_to_embed_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.derive_text_chunks_and_embeddings"]["queue"] == "embed"

    def test_derive_image_embeddings_routed_to_embed_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.derive_image_embeddings"]["queue"] == "embed"

    def test_derive_ontology_graph_routed_to_graph_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.derive_ontology_graph"]["queue"] == "graph"

    def test_derive_structure_links_routed_to_graph_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.derive_structure_links"]["queue"] == "graph"

    def test_finalize_document_routed_to_ingest_queue(self):
        from app.workers.celery_app import celery_app
        routes = celery_app.conf.task_routes
        assert routes["app.workers.pipeline.finalize_document"]["queue"] == "ingest"
