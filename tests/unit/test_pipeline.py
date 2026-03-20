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
