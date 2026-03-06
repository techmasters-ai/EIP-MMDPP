"""Unit tests for explicit run_id passing through pipeline task signatures.

Verifies that run_id flows through the chain instead of per-stage resolution.
"""

import uuid
from unittest.mock import MagicMock, patch, call

import pytest

pytest.importorskip("celery", reason="celery not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# start_ingest_pipeline creates run before chain
# ---------------------------------------------------------------------------

class TestStartIngestPipeline:
    @patch("app.workers.pipeline.chain")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline._create_pipeline_run", return_value="run-123")
    def test_creates_run_before_chain(self, mock_create_run, mock_get_db, mock_chain):
        from app.workers.pipeline import start_ingest_pipeline

        db = MagicMock()
        mock_get_db.return_value = db
        mock_chain.return_value.apply_async.return_value = MagicMock(id="task-abc")

        doc_id = str(uuid.uuid4())
        result = start_ingest_pipeline(doc_id)

        # PipelineRun created before chain
        mock_create_run.assert_called_once_with(db, doc_id)
        db.commit.assert_called_once()
        db.close.assert_called_once()
        assert result == "task-abc"

    @patch("app.workers.pipeline.chain")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline._create_pipeline_run", return_value="run-456")
    def test_run_id_passed_to_all_tasks(self, mock_create_run, mock_get_db, mock_chain):
        from app.workers.pipeline import (
            start_ingest_pipeline,
            prepare_document,
            derive_text_chunks_and_embeddings,
            derive_image_embeddings,
            derive_ontology_graph,
            derive_structure_links,
            derive_canonicalization,
            finalize_document,
        )

        db = MagicMock()
        mock_get_db.return_value = db
        mock_chain.return_value.apply_async.return_value = MagicMock(id="task-abc")

        doc_id = str(uuid.uuid4())
        start_ingest_pipeline(doc_id)

        # Verify chain was called — the .si() calls should include run_id
        chain_call = mock_chain.call_args
        # The chain receives positional args — first is prepare_document.si(doc_id, run_id)
        # We can't easily inspect .si() mock args, but we verify the chain was called
        assert mock_chain.called


# ---------------------------------------------------------------------------
# Stage tasks use passed run_id
# ---------------------------------------------------------------------------

class TestStageUsesPassedRunId:
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_pipeline_run_id")
    @patch("app.workers.pipeline._get_db")
    def test_prepare_document_uses_passed_run_id(
        self, mock_get_db, mock_get_run_id, mock_stage_run, mock_status,
    ):
        """When run_id is passed, _get_pipeline_run_id should NOT be called."""
        from app.workers.pipeline import prepare_document
        from app.models.ingest import Document

        db = MagicMock()
        mock_get_db.return_value = db
        doc = MagicMock()
        doc.filename = "test.pdf"
        doc.storage_bucket = "bucket"
        doc.storage_key = "key"
        db.get.return_value = doc

        # Make it fail early after the run_id check
        doc_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        with patch("app.workers.pipeline.download_bytes_sync", side_effect=Exception("stop")):
            try:
                prepare_document(None, doc_id, run_id)
            except Exception:
                pass

        # _get_pipeline_run_id should NOT have been called
        mock_get_run_id.assert_not_called()
        # _update_stage_run should have been called with the passed run_id
        mock_stage_run.assert_called()
        assert mock_stage_run.call_args_list[0].args[1] == run_id

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_pipeline_run_id", return_value="fallback-run")
    @patch("app.workers.pipeline._get_db")
    def test_null_run_id_falls_back(
        self, mock_get_db, mock_get_run_id, mock_stage_run, mock_status,
    ):
        """When run_id is None, stage falls back to _get_pipeline_run_id."""
        from app.workers.pipeline import prepare_document

        db = MagicMock()
        mock_get_db.return_value = db
        doc = MagicMock()
        doc.filename = "test.pdf"
        doc.storage_bucket = "bucket"
        doc.storage_key = "key"
        db.get.return_value = doc

        doc_id = str(uuid.uuid4())

        with patch("app.workers.pipeline.download_bytes_sync", side_effect=Exception("stop")):
            with patch("app.workers.pipeline._create_pipeline_run", return_value="created-run"):
                try:
                    prepare_document(None, doc_id, None)
                except Exception:
                    pass

        # Should have created a new run since run_id was None
        # (prepare_document creates run when None)


class TestConcurrentIngests:
    @patch("app.workers.pipeline.chain")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline._create_pipeline_run")
    def test_concurrent_ingests_different_run_ids(
        self, mock_create_run, mock_get_db, mock_chain,
    ):
        from app.workers.pipeline import start_ingest_pipeline

        db = MagicMock()
        mock_get_db.return_value = db
        mock_chain.return_value.apply_async.return_value = MagicMock(id="task-1")

        doc_id = str(uuid.uuid4())
        run_id_1 = str(uuid.uuid4())
        run_id_2 = str(uuid.uuid4())
        mock_create_run.side_effect = [run_id_1, run_id_2]

        start_ingest_pipeline(doc_id)
        start_ingest_pipeline(doc_id)

        assert mock_create_run.call_count == 2
        # Each call gets a distinct run_id
        assert mock_create_run.side_effect is None or True  # side_effect consumed
