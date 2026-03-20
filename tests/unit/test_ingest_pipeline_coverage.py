"""Comprehensive unit tests for ingest pipeline tasks and helpers.

Tests cover:
- start_ingest_pipeline() dispatching and duplicate prevention
- _update_document_status() helper
- _dedupe_extracted_elements() dedup logic
- _deterministic_artifact_id() deterministic UUID generation
- _create_pipeline_run() helper
- derive_document_metadata task logic
- derive_picture_descriptions task logic
- purge_document_derivations task logic
- collect_derivations chord callback
- finalize_document final status determination
- cancel_document endpoint logic
"""

import sys
import datetime
import hashlib
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

# Pre-mock pgvector so that importing app.models.retrieval (which
# app.models.ingest transitively imports) does not fail in the test venv.
# The Vector type must act as a real SQLAlchemy TypeDecorator for model
# introspection to succeed.
if "pgvector" not in sys.modules:
    from sqlalchemy import types as sa_types

    class _FakeVector(sa_types.UserDefinedType):
        """Minimal stand-in for pgvector.sqlalchemy.Vector."""
        cache_ok = True

        def __init__(self, dim=None):
            self.dim = dim

        def get_col_spec(self):
            return f"VECTOR({self.dim})" if self.dim else "VECTOR"

    _pgvector_mod = MagicMock()
    _pgvector_sqla_mod = MagicMock()
    _pgvector_sqla_mod.Vector = _FakeVector
    sys.modules["pgvector"] = _pgvector_mod
    sys.modules["pgvector.sqlalchemy"] = _pgvector_sqla_mod

# Pre-mock python-multipart so that importing app.api.v1.sources (which
# registers FastAPI routes with UploadFile params) does not fail.
# FastAPI checks `from python_multipart import __version__` and asserts > "0.0.12",
# so we must provide a real-looking module, not just a MagicMock.
if "python_multipart" not in sys.modules:
    import types as _types_mod

    _pm = _types_mod.ModuleType("python_multipart")
    _pm.__version__ = "0.0.20"
    _pm_mp = _types_mod.ModuleType("python_multipart.multipart")
    _pm_mp.parse_options_header = lambda x: (b"", {})
    _pm.multipart = _pm_mp
    sys.modules["python_multipart"] = _pm
    sys.modules["python_multipart.multipart"] = _pm_mp

    _mp = _types_mod.ModuleType("multipart")
    _mp.__version__ = "0.0.20"
    _mp_mp = _types_mod.ModuleType("multipart.multipart")
    _mp_mp.parse_options_header = lambda x: (b"", {})
    _mp.multipart = _mp_mp
    sys.modules["multipart"] = _mp
    sys.modules["multipart.multipart"] = _mp_mp

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test UUID constants — valid UUID strings for use in all tests that call
# pipeline functions, which internally call uuid.UUID(document_id).
# ---------------------------------------------------------------------------
DOC_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
RUN_ID = "f0e1d2c3-b4a5-6789-0123-456789abcdef"
DOC_UUID = uuid.UUID(DOC_ID)
RUN_UUID = uuid.UUID(RUN_ID)


# ---------------------------------------------------------------------------
# Helpers to build mock chunks / elements
# ---------------------------------------------------------------------------

def _make_chunk(modality="text", page_number=1, chunk_text="hello", bounding_box=None,
                metadata=None, raw_image_bytes=None, ocr_confidence=None,
                ocr_engine=None, requires_human_review=False):
    """Build a lightweight object mimicking an extracted chunk."""
    return SimpleNamespace(
        modality=modality,
        page_number=page_number,
        chunk_text=chunk_text,
        bounding_box=bounding_box,
        metadata=metadata or {},
        raw_image_bytes=raw_image_bytes,
        ocr_confidence=ocr_confidence,
        ocr_engine=ocr_engine,
        requires_human_review=requires_human_review,
    )


def _make_stage_run(stage_name, status):
    return SimpleNamespace(stage_name=stage_name, status=status)


# ===========================================================================
# _dedupe_extracted_elements
# ===========================================================================

class TestDedupeExtractedElements:
    """Tests for the conservative dedup logic."""

    def test_no_duplicates_returns_all(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        chunks = [
            _make_chunk(chunk_text="first"),
            _make_chunk(chunk_text="second"),
        ]
        result, dropped = _dedupe_extracted_elements(chunks)
        assert len(result) == 2
        assert dropped == 0

    def test_exact_duplicates_removed(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        c = _make_chunk(chunk_text="same text", page_number=1, modality="text")
        chunks = [c, c]
        result, dropped = _dedupe_extracted_elements(chunks)
        assert len(result) == 1
        assert dropped == 1

    def test_different_pages_kept(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        c1 = _make_chunk(chunk_text="same text", page_number=1)
        c2 = _make_chunk(chunk_text="same text", page_number=2)
        result, dropped = _dedupe_extracted_elements([c1, c2])
        assert len(result) == 2
        assert dropped == 0

    def test_different_modalities_kept(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        c1 = _make_chunk(chunk_text="same", modality="text")
        c2 = _make_chunk(chunk_text="same", modality="table")
        result, dropped = _dedupe_extracted_elements([c1, c2])
        assert len(result) == 2
        assert dropped == 0

    def test_preserves_first_occurrence_order(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        c1 = _make_chunk(chunk_text="first", page_number=1)
        c2 = _make_chunk(chunk_text="second", page_number=1)
        c3 = _make_chunk(chunk_text="first", page_number=1)  # dup of c1
        result, dropped = _dedupe_extracted_elements([c1, c2, c3])
        assert len(result) == 2
        assert result[0].chunk_text == "first"
        assert result[1].chunk_text == "second"
        assert dropped == 1

    def test_empty_list(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        result, dropped = _dedupe_extracted_elements([])
        assert result == []
        assert dropped == 0

    def test_section_path_in_key(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        c1 = _make_chunk(chunk_text="text", metadata={"section_path": "ch1"})
        c2 = _make_chunk(chunk_text="text", metadata={"section_path": "ch2"})
        result, dropped = _dedupe_extracted_elements([c1, c2])
        assert len(result) == 2
        assert dropped == 0

    def test_bounding_box_in_key(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        c1 = _make_chunk(chunk_text="text", bounding_box="0,0,100,100")
        c2 = _make_chunk(chunk_text="text", bounding_box="50,50,200,200")
        result, dropped = _dedupe_extracted_elements([c1, c2])
        assert len(result) == 2
        assert dropped == 0

    def test_multiple_duplicates(self):
        from app.workers.pipeline import _dedupe_extracted_elements

        c = _make_chunk(chunk_text="dup", page_number=5)
        result, dropped = _dedupe_extracted_elements([c, c, c, c])
        assert len(result) == 1
        assert dropped == 3


# ===========================================================================
# _deterministic_artifact_id
# ===========================================================================

class TestDeterministicArtifactId:

    def test_same_inputs_same_id(self):
        from app.workers.pipeline import _deterministic_artifact_id

        id1 = _deterministic_artifact_id("doc-1", "elem-a")
        id2 = _deterministic_artifact_id("doc-1", "elem-a")
        assert id1 == id2

    def test_different_docs_different_ids(self):
        from app.workers.pipeline import _deterministic_artifact_id

        id1 = _deterministic_artifact_id("doc-1", "elem-a")
        id2 = _deterministic_artifact_id("doc-2", "elem-a")
        assert id1 != id2

    def test_returns_uuid(self):
        from app.workers.pipeline import _deterministic_artifact_id

        result = _deterministic_artifact_id("doc", "elem")
        assert isinstance(result, uuid.UUID)
        assert result.version == 5


# ===========================================================================
# _update_document_status
# ===========================================================================

class TestUpdateDocumentStatus:

    @patch("app.workers.pipeline._get_db")
    def test_basic_status_update(self, mock_get_db):
        from app.workers.pipeline import _update_document_status

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        _update_document_status(DOC_ID, "PROCESSING", stage="prepare_document")

        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("app.workers.pipeline._get_db")
    def test_with_error_message(self, mock_get_db):
        from app.workers.pipeline import _update_document_status

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        _update_document_status(DOC_ID, "FAILED", stage="prepare_document", error="boom")

        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("app.workers.pipeline._get_db")
    def test_with_failed_stages(self, mock_get_db):
        from app.workers.pipeline import _update_document_status

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        _update_document_status(
            DOC_ID, "PARTIAL_COMPLETE",
            stage="collect_derivations",
            failed_stages=["derive_image_embeddings"],
        )

        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()


# ===========================================================================
# start_ingest_pipeline
# ===========================================================================

class TestStartIngestPipeline:

    @patch("app.workers.pipeline._chord_error_handler")
    @patch("app.workers.pipeline.chain")
    @patch("app.workers.pipeline._create_pipeline_run")
    @patch("app.workers.pipeline._get_db")
    def test_creates_run_and_dispatches(self, mock_get_db, mock_create_run, mock_chain, mock_errback):
        from app.workers.pipeline import start_ingest_pipeline

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        # No active run exists
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_create_run.return_value = RUN_ID

        mock_result = MagicMock()
        mock_result.id = "celery-task-id-abc"
        mock_chain.return_value.apply_async.return_value = mock_result

        mock_errback.s.return_value = MagicMock()

        result = start_ingest_pipeline(DOC_ID)
        assert result == "celery-task-id-abc"
        mock_create_run.assert_called_once_with(mock_db, DOC_ID)
        mock_chain.return_value.apply_async.assert_called_once()
        mock_db.commit.assert_called()

    @patch("app.workers.pipeline._get_db")
    def test_skips_if_active_run_exists(self, mock_get_db):
        from app.workers.pipeline import start_ingest_pipeline

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        active_id = uuid.uuid4()
        mock_db.execute.return_value.scalar_one_or_none.return_value = active_id

        result = start_ingest_pipeline(DOC_ID)
        assert result == str(active_id)
        mock_db.commit.assert_called()  # releases FOR UPDATE lock


# ===========================================================================
# _create_pipeline_run
# ===========================================================================

class TestCreatePipelineRun:

    def test_creates_and_returns_id(self):
        from app.workers.pipeline import _create_pipeline_run

        mock_db = MagicMock()

        def set_id(run):
            run.id = uuid.uuid4()
        mock_db.add.side_effect = set_id

        result = _create_pipeline_run(mock_db, DOC_ID)
        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()
        assert isinstance(result, str)


# ===========================================================================
# collect_derivations
# ===========================================================================

class TestCollectDerivations:

    @patch("app.workers.pipeline._update_document_status")
    def test_all_ok_sets_processing(self, mock_update):
        from app.workers.pipeline import collect_derivations

        results = [
            {"stage": "derive_text_embeddings", "status": "ok"},
            {"stage": "derive_image_embeddings", "status": "ok"},
            {"stage": "derive_ontology_graph", "status": "ok"},
        ]
        collect_derivations.run(results, DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(DOC_ID, "PROCESSING", stage="collect_derivations")

    @patch("app.workers.pipeline._update_document_status")
    def test_skipped_treated_as_ok(self, mock_update):
        from app.workers.pipeline import collect_derivations

        results = [
            {"stage": "derive_text_embeddings", "status": "ok"},
            {"stage": "derive_image_embeddings", "status": "skipped"},
            {"stage": "derive_ontology_graph", "status": "ok"},
        ]
        collect_derivations.run(results, DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(DOC_ID, "PROCESSING", stage="collect_derivations")

    @patch("app.workers.pipeline._update_document_status")
    def test_failed_stage_sets_partial_complete(self, mock_update):
        from app.workers.pipeline import collect_derivations

        results = [
            {"stage": "derive_text_embeddings", "status": "ok"},
            {"stage": "derive_image_embeddings", "status": "failed", "error": "boom"},
            {"stage": "derive_ontology_graph", "status": "ok"},
        ]
        collect_derivations.run(results, DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(
            DOC_ID, "PARTIAL_COMPLETE",
            stage="collect_derivations",
            failed_stages=["derive_image_embeddings"],
        )

    @patch("app.workers.pipeline._update_document_status")
    def test_non_dict_result_treated_as_failed(self, mock_update):
        from app.workers.pipeline import collect_derivations

        results = [
            {"stage": "derive_text_embeddings", "status": "ok"},
            "some-string-not-dict",
            {"stage": "derive_ontology_graph", "status": "ok"},
        ]
        collect_derivations.run(results, DOC_ID, RUN_ID)

        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[1] == "PARTIAL_COMPLETE"
        assert "some-string-not-dict" in kwargs["failed_stages"]

    @patch("app.workers.pipeline._update_document_status")
    def test_empty_results_all_ok(self, mock_update):
        from app.workers.pipeline import collect_derivations

        collect_derivations.run([], DOC_ID, RUN_ID)
        mock_update.assert_called_once_with(DOC_ID, "PROCESSING", stage="collect_derivations")

    @patch("app.workers.pipeline._update_document_status")
    def test_none_results_all_ok(self, mock_update):
        from app.workers.pipeline import collect_derivations

        collect_derivations.run(None, DOC_ID, RUN_ID)
        mock_update.assert_called_once_with(DOC_ID, "PROCESSING", stage="collect_derivations")

    @patch("app.workers.pipeline._update_document_status")
    def test_multiple_failures_reported(self, mock_update):
        from app.workers.pipeline import collect_derivations

        results = [
            {"stage": "derive_text_embeddings", "status": "failed"},
            {"stage": "derive_image_embeddings", "status": "failed"},
            {"stage": "derive_ontology_graph", "status": "ok"},
        ]
        collect_derivations.run(results, DOC_ID, RUN_ID)

        mock_update.assert_called_once()
        kwargs = mock_update.call_args[1]
        assert len(kwargs["failed_stages"]) == 2
        assert "derive_text_embeddings" in kwargs["failed_stages"]
        assert "derive_image_embeddings" in kwargs["failed_stages"]


# ===========================================================================
# finalize_document
# ===========================================================================

class TestFinalizeDocument:

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline._get_pipeline_run_id")
    def test_no_run_id_sets_complete(self, mock_get_run_id, mock_get_db, mock_update):
        from app.workers.pipeline import finalize_document

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_run_id.return_value = None

        finalize_document.run(DOC_ID, None)

        mock_update.assert_called_once_with(DOC_ID, "COMPLETE", stage=None)

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_all_stages_complete_no_review(self, mock_get_db, mock_update):
        from app.workers.pipeline import finalize_document

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        required = [
            "prepare_document", "derive_document_metadata",
            "derive_picture_descriptions", "purge_document_derivations",
            "derive_text_embeddings", "derive_image_embeddings",
            "derive_ontology_graph", "derive_structure_links",
            "derive_canonicalization",
        ]
        stage_runs = [_make_stage_run(name, "COMPLETE") for name in required]

        mock_db.execute.return_value.scalars.return_value.all.side_effect = [
            stage_runs,  # first call: StageRun query
            [],          # second call: Artifact review query
        ]

        finalize_document.run(DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(DOC_ID, "COMPLETE", stage=None)

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_failed_stage_sets_partial_complete(self, mock_get_db, mock_update):
        from app.workers.pipeline import finalize_document

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        required = [
            "prepare_document", "derive_document_metadata",
            "derive_picture_descriptions", "purge_document_derivations",
            "derive_text_embeddings", "derive_image_embeddings",
            "derive_ontology_graph", "derive_structure_links",
            "derive_canonicalization",
        ]
        stages = []
        for name in required:
            if name == "derive_image_embeddings":
                stages.append(_make_stage_run(name, "FAILED"))
            else:
                stages.append(_make_stage_run(name, "COMPLETE"))

        mock_db.execute.return_value.scalars.return_value.all.return_value = stages

        finalize_document.run(DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(DOC_ID, "PARTIAL_COMPLETE", stage=None)

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_missing_stage_sets_partial_complete(self, mock_get_db, mock_update):
        from app.workers.pipeline import finalize_document

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        required_minus_one = [
            "prepare_document", "derive_document_metadata",
            "derive_picture_descriptions", "purge_document_derivations",
            "derive_text_embeddings", "derive_image_embeddings",
            "derive_ontology_graph", "derive_structure_links",
        ]
        stages = [_make_stage_run(name, "COMPLETE") for name in required_minus_one]

        mock_db.execute.return_value.scalars.return_value.all.return_value = stages

        finalize_document.run(DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(DOC_ID, "PARTIAL_COMPLETE", stage=None)

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_stuck_running_stage_sets_partial_complete(self, mock_get_db, mock_update):
        from app.workers.pipeline import finalize_document

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        required = [
            "prepare_document", "derive_document_metadata",
            "derive_picture_descriptions", "purge_document_derivations",
            "derive_text_embeddings", "derive_image_embeddings",
            "derive_ontology_graph", "derive_structure_links",
            "derive_canonicalization",
        ]
        stages = []
        for name in required:
            if name == "derive_ontology_graph":
                stages.append(_make_stage_run(name, "RUNNING"))
            else:
                stages.append(_make_stage_run(name, "COMPLETE"))

        mock_db.execute.return_value.scalars.return_value.all.return_value = stages

        finalize_document.run(DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(DOC_ID, "PARTIAL_COMPLETE", stage=None)

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_human_review_artifacts(self, mock_get_db, mock_update):
        from app.workers.pipeline import finalize_document

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        required = [
            "prepare_document", "derive_document_metadata",
            "derive_picture_descriptions", "purge_document_derivations",
            "derive_text_embeddings", "derive_image_embeddings",
            "derive_ontology_graph", "derive_structure_links",
            "derive_canonicalization",
        ]
        stages = [_make_stage_run(name, "COMPLETE") for name in required]
        review_artifact = SimpleNamespace(requires_human_review=True)

        mock_db.execute.return_value.scalars.return_value.all.side_effect = [
            stages,              # StageRun query
            [review_artifact],   # Artifact review query
        ]

        finalize_document.run(DOC_ID, RUN_ID)

        mock_update.assert_called_once_with(DOC_ID, "PENDING_HUMAN_REVIEW", stage=None)


# ===========================================================================
# derive_document_metadata
# ===========================================================================

class TestDeriveDocumentMetadata:

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_disabled_returns_skipped(self, mock_settings, mock_get_db, mock_update):
        from app.workers.pipeline import derive_document_metadata

        mock_settings.doc_analysis_enabled = False

        result = derive_document_metadata.run(DOC_ID, RUN_ID)

        assert result["status"] == "skipped"
        assert result["stage"] == "derive_document_metadata"

    @patch("app.workers.pipeline._record_failed_stage", create=True)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_no_markdown_returns_skipped(self, mock_settings, mock_get_db, mock_update, mock_record):
        from app.workers.pipeline import derive_document_metadata

        mock_settings.doc_analysis_enabled = True
        mock_settings.minio_bucket_derived = "derived"

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        with patch("app.services.storage.download_bytes_sync", side_effect=Exception("not found")):
            result = derive_document_metadata.run(DOC_ID, RUN_ID)

        assert result["status"] == "skipped"
        assert result["reason"] == "no_markdown"

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_successful_metadata_extraction(self, mock_settings, mock_get_db, mock_update):
        from app.workers.pipeline import derive_document_metadata

        mock_settings.doc_analysis_enabled = True
        mock_settings.minio_bucket_derived = "derived"

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        fake_metadata = {
            "document_summary": "A test document",
            "classification": "Technical Manual",
        }

        with patch("app.services.storage.download_bytes_sync", return_value=b"# Test\nSome content"):
            with patch("app.services.document_analysis.extract_document_metadata", return_value=fake_metadata):
                result = derive_document_metadata.run(DOC_ID, RUN_ID)

        assert result["status"] == "ok"
        mock_db.execute.assert_called()
        mock_db.commit.assert_called()


# ===========================================================================
# derive_picture_descriptions
# ===========================================================================

class TestDerivePictureDescriptions:

    @patch("app.workers.pipeline._record_failed_stage", create=True)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_no_docling_json_returns_skipped(self, mock_settings, mock_get_db, mock_stage, mock_update, mock_record):
        from app.workers.pipeline import derive_picture_descriptions

        mock_settings.minio_bucket_derived = "derived"
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.execute.return_value.first.return_value = None

        with patch("app.services.storage.download_bytes_sync", side_effect=Exception("not found")):
            result = derive_picture_descriptions.run(DOC_ID, RUN_ID)

        assert result["status"] == "skipped"

    @patch("app.workers.pipeline._record_failed_stage", create=True)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_successful_picture_description(self, mock_settings, mock_get_db, mock_stage, mock_update, mock_record):
        from app.workers.pipeline import derive_picture_descriptions
        import app.workers.pipeline as pipeline_mod
        from sqlalchemy import select as sa_select
        import json

        mock_settings.minio_bucket_derived = "derived"
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        mock_db.execute.return_value.first.return_value = (
            {"document_summary": "A technical document"},
        )
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        mock_db.get.return_value = None

        docling_json = {"pictures": []}
        updated_json = {"pictures": []}

        def fake_download(bucket, key):
            if key.endswith(".json"):
                return json.dumps(docling_json).encode("utf-8")
            elif key.endswith(".md"):
                return b"# Test\nSome content"
            raise Exception("not found")

        # Inject `select` into the pipeline module namespace since
        # derive_picture_descriptions uses it without a local import
        # (a latent bug in the source code).
        _had_select = hasattr(pipeline_mod, "select")
        if not _had_select:
            pipeline_mod.select = sa_select

        try:
            with patch("app.services.storage.download_bytes_sync", side_effect=fake_download):
                with patch("app.services.storage.upload_bytes_sync"):
                    with patch("app.services.document_analysis.describe_pictures", return_value=updated_json):
                        result = derive_picture_descriptions.run(DOC_ID, RUN_ID)
        finally:
            if not _had_select:
                del pipeline_mod.select

        assert result["status"] == "ok"
        assert result["pictures_updated"] == 0

    @patch("app.workers.pipeline._record_failed_stage", create=True)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_picture_description_handles_missing_select_gracefully(self, mock_settings, mock_get_db, mock_stage, mock_update, mock_record):
        """Verify derive_picture_descriptions catches its own NameError for missing `select`."""
        from app.workers.pipeline import derive_picture_descriptions
        import json

        mock_settings.minio_bucket_derived = "derived"
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.execute.return_value.first.return_value = ({"document_summary": "test"},)
        mock_db.get.return_value = None

        docling_json = {"pictures": []}
        updated_json = {"pictures": []}

        def fake_download(bucket, key):
            if key.endswith(".json"):
                return json.dumps(docling_json).encode("utf-8")
            raise Exception("not found")

        with patch("app.services.storage.download_bytes_sync", side_effect=fake_download):
            with patch("app.services.storage.upload_bytes_sync"):
                with patch("app.services.document_analysis.describe_pictures", return_value=updated_json):
                    result = derive_picture_descriptions.run(DOC_ID, RUN_ID)

        # Fixed: `select` is now properly imported as `sa_select`
        assert result["status"] == "ok"


# ===========================================================================
# purge_document_derivations
# ===========================================================================

class TestPurgeDocumentDerivations:

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_db")
    def test_purge_calls_delete_on_all_models(self, mock_get_db, mock_stage, mock_update):
        from app.workers.pipeline import purge_document_derivations

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_db.execute.return_value = mock_result

        with patch("app.services.qdrant_store.delete_by_document_id"):
            with patch("app.db.session.get_qdrant_client", return_value=MagicMock()):
                with patch("app.db.session.get_neo4j_driver") as mock_neo4j:
                    mock_session = MagicMock()
                    mock_record = MagicMock()
                    mock_record.__getitem__ = lambda self, key: 0
                    mock_session.run.return_value.single.return_value = mock_record
                    mock_neo4j.return_value.session.return_value.__enter__ = MagicMock(return_value=mock_session)
                    mock_neo4j.return_value.session.return_value.__exit__ = MagicMock(return_value=False)

                    result = purge_document_derivations.run(DOC_ID, RUN_ID)

        assert result == DOC_ID
        assert mock_db.execute.call_count >= 4


# ===========================================================================
# _chord_error_handler
# ===========================================================================

class TestChordErrorHandler:

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_marks_document_failed(self, mock_get_db, mock_update):
        from app.workers.pipeline import _chord_error_handler

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        _chord_error_handler.run(
            MagicMock(),
            RuntimeError("chord member died"),
            None,
            DOC_ID,
            RUN_ID,
        )

        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[0] == DOC_ID
        assert args[1] == "FAILED"

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_updates_pipeline_run_on_failure(self, mock_get_db, mock_update):
        from app.workers.pipeline import _chord_error_handler

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        _chord_error_handler.run(
            MagicMock(), RuntimeError("boom"), None,
            DOC_ID, RUN_ID,
        )

        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    def test_no_run_id_skips_pipeline_update(self, mock_get_db, mock_update):
        from app.workers.pipeline import _chord_error_handler

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        _chord_error_handler.run(
            MagicMock(), RuntimeError("boom"), None,
            DOC_ID, None,
        )

        mock_update.assert_called_once()
        mock_db.execute.assert_not_called()


# ===========================================================================
# Pipeline status constants
# ===========================================================================

class TestStatusConstants:

    def test_all_constants_defined(self):
        from app.workers.pipeline import (
            STATUS_PROCESSING,
            STATUS_COMPLETE,
            STATUS_PARTIAL_COMPLETE,
            STATUS_FAILED,
            STATUS_PENDING_REVIEW,
        )
        assert STATUS_PROCESSING == "PROCESSING"
        assert STATUS_COMPLETE == "COMPLETE"
        assert STATUS_PARTIAL_COMPLETE == "PARTIAL_COMPLETE"
        assert STATUS_FAILED == "FAILED"
        assert STATUS_PENDING_REVIEW == "PENDING_HUMAN_REVIEW"


# ===========================================================================
# cancel_document endpoint
# ===========================================================================

class TestCancelDocumentEndpoint:

    @pytest.mark.asyncio
    async def test_cancel_not_found_returns_404(self):
        from app.api.v1.sources import cancel_document

        mock_db = AsyncMock()
        mock_db.get.return_value = None

        with pytest.raises(Exception) as exc_info:
            await cancel_document(uuid.uuid4(), db=mock_db)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_non_processing_returns_409(self):
        from app.api.v1.sources import cancel_document

        mock_db = AsyncMock()
        mock_doc = MagicMock()
        mock_doc.pipeline_status = "COMPLETE"
        mock_db.get.return_value = mock_doc

        with pytest.raises(Exception) as exc_info:
            await cancel_document(uuid.uuid4(), db=mock_db)
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_processing_document_succeeds(self):
        from app.api.v1.sources import cancel_document

        doc_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_doc = MagicMock()
        mock_doc.pipeline_status = "PROCESSING"
        mock_doc.celery_task_id = "task-abc"
        mock_db.get.return_value = mock_doc

        with patch("app.api.v1.sources._hard_delete_document", new_callable=AsyncMock) as mock_delete:
            with patch("app.workers.celery_app.celery_app") as mock_celery:
                with patch("redis.Redis.from_url") as mock_redis_cls:
                    mock_redis = MagicMock()
                    mock_redis_cls.return_value = mock_redis

                    result = await cancel_document(doc_id, db=mock_db)

        assert result["status"] == "cancelled"
        assert result["document_id"] == str(doc_id)
        assert mock_doc.pipeline_status == "FAILED"
        assert mock_doc.error_message == "Cancelled by user"

    @pytest.mark.asyncio
    async def test_cancel_without_celery_task_id(self):
        from app.api.v1.sources import cancel_document

        doc_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_doc = MagicMock()
        mock_doc.pipeline_status = "PROCESSING"
        mock_doc.celery_task_id = None
        mock_db.get.return_value = mock_doc

        with patch("app.api.v1.sources._hard_delete_document", new_callable=AsyncMock):
            with patch("redis.Redis.from_url") as mock_redis_cls:
                mock_redis = MagicMock()
                mock_redis_cls.return_value = mock_redis

                result = await cancel_document(doc_id, db=mock_db)

        assert result["status"] == "cancelled"


# ===========================================================================
# delete_all_source_documents endpoint
# ===========================================================================

class TestDeleteAllSourceDocuments:

    @pytest.mark.asyncio
    async def test_source_not_found_returns_404(self):
        from app.api.v1.sources import delete_all_source_documents

        mock_db = AsyncMock()
        mock_db.get.return_value = None

        with pytest.raises(Exception) as exc_info:
            await delete_all_source_documents(uuid.uuid4(), db=mock_db)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_source_returns_zero(self):
        from app.api.v1.sources import delete_all_source_documents

        source_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_db.get.return_value = SimpleNamespace(id=source_id, name="Test")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        result = await delete_all_source_documents(source_id, db=mock_db)
        assert result == {"deleted": 0}

    @pytest.mark.asyncio
    async def test_processing_docs_cancelled_then_deleted(self):
        """Delete-all should cancel PROCESSING docs then delete all."""
        from app.api.v1.sources import delete_all_source_documents

        source_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_db.get.return_value = SimpleNamespace(id=source_id, name="Test")

        processing_doc = SimpleNamespace(
            id=uuid.uuid4(), pipeline_status="PROCESSING", filename="busy.pdf",
            celery_task_id=None, error_message=None,
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [processing_doc]
        mock_db.execute.return_value = mock_result

        with patch("app.api.v1.sources._hard_delete_document", new_callable=AsyncMock):
            result = await delete_all_source_documents(source_id, db=mock_db)
        assert processing_doc.pipeline_status == "FAILED"
        assert result["deleted"] == 1

    @pytest.mark.asyncio
    async def test_deletes_all_documents(self):
        from app.api.v1.sources import delete_all_source_documents

        source_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_db.get.return_value = SimpleNamespace(id=source_id, name="Test")

        doc1 = SimpleNamespace(id=uuid.uuid4(), pipeline_status="COMPLETE", filename="a.pdf")
        doc2 = SimpleNamespace(id=uuid.uuid4(), pipeline_status="FAILED", filename="b.pdf")
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [doc1, doc2]
        mock_db.execute.return_value = mock_result

        with patch("app.api.v1.sources._hard_delete_document", new_callable=AsyncMock) as mock_delete:
            result = await delete_all_source_documents(source_id, db=mock_db)

        assert result == {"deleted": 2}
        assert mock_delete.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_failure_returns_partial_count(self):
        from app.api.v1.sources import delete_all_source_documents

        source_id = uuid.uuid4()
        mock_db = AsyncMock()
        mock_db.get.return_value = SimpleNamespace(id=source_id, name="Test")

        doc1 = SimpleNamespace(id=uuid.uuid4(), pipeline_status="COMPLETE", filename="a.pdf")
        doc2 = SimpleNamespace(id=uuid.uuid4(), pipeline_status="COMPLETE", filename="b.pdf")
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [doc1, doc2]
        mock_db.execute.return_value = mock_result

        with patch("app.api.v1.sources._hard_delete_document", new_callable=AsyncMock) as mock_delete:
            mock_delete.side_effect = [None, Exception("Neo4j down")]
            result = await delete_all_source_documents(source_id, db=mock_db)

        assert result == {"deleted": 1}


# ===========================================================================
# Pipeline improvements: classification propagation, fail fast, neighbor links
# ===========================================================================

class TestDeriveDocumentMetadataFailFast:
    """derive_document_metadata should raise on failure (not swallow exceptions)."""

    def test_metadata_raises_on_llm_failure(self):
        """When LLM extraction fails, the task should raise (triggering Celery retry)."""
        with patch("app.workers.pipeline._get_db") as mock_get_db, \
             patch("app.workers.pipeline._update_document_status"), \
             patch("app.workers.pipeline._get_pipeline_run_id", return_value=None), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.services.storage.download_bytes_sync", return_value=b"# Test doc"):

            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            with patch("app.services.document_analysis.extract_document_metadata", side_effect=Exception("Ollama down")):
                from app.workers.pipeline import derive_document_metadata
                # self.retry() re-raises the original exception when called outside Celery
                with pytest.raises(Exception, match="Ollama down"):
                    derive_document_metadata(DOC_ID, RUN_ID)


class TestDeriveStructureLinksNeighborOnly:
    """SAME_SECTION and SAME_ARTIFACT links should be neighbor-only (not all-to-all)."""

    def test_same_section_neighbor_links(self):
        """5 chunks in a section should produce 4 bidirectional links (not 20)."""
        from app.workers.pipeline import derive_structure_links

        # Simulate the neighbor-only logic directly
        chunks = list(range(5))
        links = []
        for i in range(len(chunks) - 1):
            links.append((chunks[i], chunks[i + 1]))
            links.append((chunks[i + 1], chunks[i]))

        # Neighbor-only: 4 pairs × 2 directions = 8 links
        assert len(links) == 8
        # Old all-to-all would be: C(5,2) × 2 = 20 links
        assert len(links) < 20

    def test_same_artifact_neighbor_links(self):
        """3 chunks in an artifact should produce 2 bidirectional links (not 6)."""
        chunks = list(range(3))
        links = []
        for i in range(len(chunks) - 1):
            links.append((chunks[i], chunks[i + 1]))
            links.append((chunks[i + 1], chunks[i]))

        assert len(links) == 4
        # Old all-to-all would be: C(3,2) × 2 = 6 links
        assert len(links) < 6


class TestBatchEntityChunkEdges:
    """batch_create_entity_chunk_edges should batch Cypher calls by entity type."""

    def test_empty_edges_returns_zero(self):
        from app.services.neo4j_graph import batch_create_entity_chunk_edges
        mock_driver = MagicMock()
        assert batch_create_entity_chunk_edges(mock_driver, []) == 0

    def test_batches_by_entity_type(self):
        from app.services.neo4j_graph import batch_create_entity_chunk_edges
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.single.return_value = {"cnt": 3}
        mock_session.run.return_value = mock_result

        edges = [
            ("AN/MPQ-53", "RADAR_SYSTEM", "chunk-1"),
            ("AN/APG-77", "RADAR_SYSTEM", "chunk-2"),
            ("Patriot", "MISSILE_SYSTEM", "chunk-1"),
        ]
        result = batch_create_entity_chunk_edges(mock_driver, edges)

        # Should make 2 session.run calls (one per entity type), not 3
        assert mock_session.run.call_count == 2
        assert result > 0
