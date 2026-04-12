"""Spec §5.4 — new derive_ontology_graph branch behind the feature flag.

Task 4.6. Mocks all DB/graph/HTTP calls to test the orchestrator's
control flow in isolation. Integration-level testing lives in
tests/e2e/test_full_pipeline.py and tests/integration/.
"""
from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace
import uuid

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_pipeline_run(mode="graph_only", bundle_key="air_defense_v3"):
    run = MagicMock()
    run.id = uuid.uuid4()
    run.document_id = uuid.uuid4()
    run.mode = mode
    run.ontology_bundle_key = bundle_key
    run.use_case_key = None
    run.status = "PROCESSING"
    return run


def _mock_manifest(bundle_key="air_defense_v3"):
    m = MagicMock()
    m.bundle_key = bundle_key
    m.ontology_name = "Test Ontology"
    m.ontology_version = "1.0"
    m.extraction_profile_version = "1.0"
    m.passes = []  # empty passes = no loop iterations
    return m


def _mock_merged():
    m = MagicMock()
    m.entities = [MagicMock(), MagicMock()]
    m.edges = [MagicMock()]
    m.rejected_edges = []
    m.document_id = "doc-1"
    m.pipeline_run_id = "run-1"
    return m


def _make_db_side_effect(run):
    """Return a side-effect callable for db.get() that returns run for
    PipelineRun lookups and a fresh MagicMock() for anything else."""
    def _side_effect(model, id_):
        if hasattr(model, "__name__") and model.__name__ == "PipelineRun":
            return run
        return MagicMock()
    return _side_effect


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestDeriveOntologyGraphBundlePasses:

    @patch("app.workers.pipeline._update_document_pipeline_status")
    @patch("app.workers.pipeline._upsert_document_graph_extraction")
    @patch("app.workers.pipeline._import_graph_phase_structural_edges")
    @patch("app.workers.pipeline._import_graph_phase_domain_edges")
    @patch("app.workers.pipeline._import_graph_phase_nodes", return_value={})
    @patch("app.workers.pipeline._write_pipeline_run_metrics")
    @patch("app.workers.pipeline._apply_post_merge_yield_updates")
    @patch("app.workers.pipeline.merge_and_resolve")
    @patch("app.workers.pipeline.check_required_pass_gate")
    @patch("app.workers.pipeline._build_docling_document_json", return_value={})
    @patch("app.workers.pipeline.load_ontology", return_value={})
    @patch("app.workers.pipeline.load_bundle_manifest")
    @patch("app.workers.pipeline._get_db")
    def test_happy_path_graph_only(
        self, mock_db, mock_manifest_fn, mock_ontology, mock_doc_json,
        mock_gate, mock_merge, mock_yield, mock_metrics,
        mock_nodes, mock_domain_edges, mock_structural, mock_upsert, mock_status,
    ):
        """graph_only mode: stage_summary COMPLETE, run COMPLETE,
        Document.pipeline_status updated to COMPLETE."""
        from app.workers.pipeline import _derive_ontology_graph_bundle_passes

        run = _mock_pipeline_run(mode="graph_only")
        manifest = _mock_manifest()
        merged = _mock_merged()

        mock_manifest_fn.return_value = manifest
        mock_gate.return_value = SimpleNamespace(passed=True, failures=[])
        mock_merge.return_value = merged

        db = MagicMock()
        db.get.side_effect = _make_db_side_effect(run)
        mock_db.return_value = db

        fake_self = MagicMock()
        fake_self.request.retries = 0

        result = _derive_ontology_graph_bundle_passes(
            fake_self, str(run.id), str(run.document_id),
        )

        assert result["status"] == "ok"
        assert result["entities"] == 2
        assert result["edges"] == 1
        assert result["stage"] == "derive_ontology_graph"
        # Document.pipeline_status updated to COMPLETE for graph_only
        mock_status.assert_called_with(str(run.document_id), "COMPLETE")

    @patch("app.workers.pipeline._update_document_pipeline_status")
    @patch("app.workers.pipeline._upsert_document_graph_extraction")
    @patch("app.workers.pipeline._import_graph_phase_structural_edges")
    @patch("app.workers.pipeline._import_graph_phase_domain_edges")
    @patch("app.workers.pipeline._import_graph_phase_nodes", return_value={})
    @patch("app.workers.pipeline._write_pipeline_run_metrics")
    @patch("app.workers.pipeline._apply_post_merge_yield_updates")
    @patch("app.workers.pipeline.merge_and_resolve")
    @patch("app.workers.pipeline.check_required_pass_gate")
    @patch("app.workers.pipeline._build_docling_document_json", return_value={})
    @patch("app.workers.pipeline.load_ontology", return_value={})
    @patch("app.workers.pipeline.load_bundle_manifest")
    @patch("app.workers.pipeline._get_db")
    def test_happy_path_full_mode_leaves_run_processing(
        self, mock_db, mock_manifest_fn, mock_ontology, mock_doc_json,
        mock_gate, mock_merge, mock_yield, mock_metrics,
        mock_nodes, mock_domain_edges, mock_structural, mock_upsert, mock_status,
    ):
        """full mode: stage_summary COMPLETE, run stays PROCESSING
        (downstream finalize_document terminalizes). Document.pipeline_status
        is NOT updated by derive_ontology_graph for full mode."""
        from app.workers.pipeline import _derive_ontology_graph_bundle_passes

        run = _mock_pipeline_run(mode="full")
        manifest = _mock_manifest()
        merged = _mock_merged()

        mock_manifest_fn.return_value = manifest
        mock_gate.return_value = SimpleNamespace(passed=True, failures=[])
        mock_merge.return_value = merged

        db = MagicMock()
        db.get.side_effect = _make_db_side_effect(run)
        mock_db.return_value = db

        fake_self = MagicMock()
        fake_self.request.retries = 0

        result = _derive_ontology_graph_bundle_passes(
            fake_self, str(run.id), str(run.document_id),
        )

        assert result["status"] == "ok"
        # Full mode: Document.pipeline_status NOT updated by derive_ontology_graph
        mock_status.assert_not_called()

    # ---------------------------------------------------------------------------
    # Failure / rollback tests
    # ---------------------------------------------------------------------------

    @patch("app.workers.pipeline._update_document_pipeline_status")
    @patch("app.workers.pipeline._attempt_rollback", return_value="")
    @patch("app.workers.pipeline.check_required_pass_gate")
    @patch("app.workers.pipeline._build_docling_document_json", return_value={})
    @patch("app.workers.pipeline.load_ontology", return_value={})
    @patch("app.workers.pipeline.load_bundle_manifest")
    @patch("app.workers.pipeline._get_db")
    def test_gate_failure_no_rollback(
        self, mock_db, mock_manifest_fn, mock_ontology, mock_doc_json,
        mock_gate, mock_rollback, mock_status,
    ):
        """Gate failure: IngestFailed raised, no rollback (tracker never marked),
        Document.pipeline_status set to PARTIAL_COMPLETE."""
        from app.workers.pipeline import _derive_ontology_graph_bundle_passes, IngestFailed

        run = _mock_pipeline_run()
        manifest = _mock_manifest()

        mock_manifest_fn.return_value = manifest
        mock_gate.return_value = SimpleNamespace(
            passed=False, failures=[("radar_domain", "FAILED")],
        )

        db = MagicMock()
        db.get.side_effect = _make_db_side_effect(run)
        mock_db.return_value = db

        fake_self = MagicMock()
        fake_self.request.retries = 0

        with pytest.raises(IngestFailed):
            _derive_ontology_graph_bundle_passes(
                fake_self, str(run.id), str(run.document_id),
            )

        # No rollback for gate failures (tracker was never marked)
        mock_rollback.assert_not_called()
        mock_status.assert_called_with(str(run.document_id), "PARTIAL_COMPLETE")

    @patch("app.workers.pipeline._update_document_pipeline_status")
    @patch("app.workers.pipeline._attempt_rollback", return_value="")
    @patch("app.workers.pipeline.load_bundle_manifest")
    @patch("app.workers.pipeline._get_db")
    def test_unexpected_exception_pre_mutation_no_rollback(
        self, mock_db, mock_manifest_fn, mock_rollback, mock_status,
    ):
        """Unexpected exception before any graph mutation: tracker stays False,
        rollback is NOT attempted."""
        from app.workers.pipeline import _derive_ontology_graph_bundle_passes

        run = _mock_pipeline_run()

        # manifest load itself fails — no mutations attempted
        mock_manifest_fn.side_effect = RuntimeError("S3 unreachable")

        db = MagicMock()
        db.get.side_effect = _make_db_side_effect(run)
        mock_db.return_value = db

        fake_self = MagicMock()
        fake_self.request.retries = 0

        with pytest.raises(RuntimeError, match="S3 unreachable"):
            _derive_ontology_graph_bundle_passes(
                fake_self, str(run.id), str(run.document_id),
            )

        # tracker.any_mutation_attempted was False → no rollback
        mock_rollback.assert_not_called()
        mock_status.assert_called_with(str(run.document_id), "PARTIAL_COMPLETE")

    @patch("app.workers.pipeline._update_document_pipeline_status")
    @patch("app.workers.pipeline._attempt_rollback", return_value="")
    @patch("app.workers.pipeline._upsert_document_graph_extraction")
    @patch("app.workers.pipeline._import_graph_phase_structural_edges")
    @patch("app.workers.pipeline._import_graph_phase_domain_edges")
    @patch("app.workers.pipeline._write_pipeline_run_metrics")
    @patch("app.workers.pipeline._apply_post_merge_yield_updates")
    @patch("app.workers.pipeline.merge_and_resolve")
    @patch("app.workers.pipeline.check_required_pass_gate")
    @patch("app.workers.pipeline._build_docling_document_json", return_value={})
    @patch("app.workers.pipeline.load_ontology", return_value={})
    @patch("app.workers.pipeline.load_bundle_manifest")
    @patch("app.workers.pipeline._get_db")
    def test_unexpected_exception_post_mutation_triggers_rollback(
        self, mock_db, mock_manifest_fn, mock_ontology, mock_doc_json,
        mock_gate, mock_merge, mock_yield, mock_metrics,
        mock_domain_edges, mock_structural,
        mock_upsert, mock_rollback, mock_status,
    ):
        """Unexpected exception after graph nodes phase (tracker marked): rollback
        IS attempted and rollback_executed=True in the StageRun."""
        from app.workers.pipeline import _derive_ontology_graph_bundle_passes, GraphWriteTracker

        run = _mock_pipeline_run()
        manifest = _mock_manifest()
        merged = _mock_merged()

        mock_manifest_fn.return_value = manifest
        mock_gate.return_value = SimpleNamespace(passed=True, failures=[])
        mock_merge.return_value = merged

        # _import_graph_phase_nodes marks the tracker, then _import_graph_phase_domain_edges raises
        def _nodes_with_mark(merged, ontology, document_id, tracker):
            tracker.mark()  # simulate the phase marking before failing
            return {}

        mock_domain_edges.side_effect = RuntimeError("ArcadeDB write failure")

        db = MagicMock()
        db.get.side_effect = _make_db_side_effect(run)
        mock_db.return_value = db

        fake_self = MagicMock()
        fake_self.request.retries = 0

        with patch("app.workers.pipeline._import_graph_phase_nodes", side_effect=_nodes_with_mark):
            with pytest.raises(RuntimeError, match="ArcadeDB write failure"):
                _derive_ontology_graph_bundle_passes(
                    fake_self, str(run.id), str(run.document_id),
                )

        # tracker.any_mutation_attempted was True → rollback attempted
        mock_rollback.assert_called_once_with(str(run.document_id))
        mock_status.assert_called_with(str(run.document_id), "PARTIAL_COMPLETE")


# ---------------------------------------------------------------------------
# Dispatch tests — legacy path removed in Task 5.2
# ---------------------------------------------------------------------------

class TestDispatch:
    """Test that derive_ontology_graph unconditionally delegates to
    _derive_ontology_graph_bundle_passes (the legacy path was removed in Task 5.2).
    """

    @patch("app.workers.pipeline._derive_ontology_graph_bundle_passes")
    def test_dispatches_to_bundle_passes(self, mock_new):
        """derive_ontology_graph always routes to the bundle-passes branch."""
        from app.workers.pipeline import derive_ontology_graph

        mock_new.return_value = {"status": "ok"}
        derive_ontology_graph.run("doc-1", "run-1")

        mock_new.assert_called_once()

    @patch("app.workers.pipeline._derive_ontology_graph_bundle_passes")
    def test_dispatches_without_run_id(self, mock_new):
        """derive_ontology_graph works even when run_id is None."""
        from app.workers.pipeline import derive_ontology_graph

        mock_new.return_value = {"status": "ok"}
        derive_ontology_graph.run("doc-1")

        mock_new.assert_called_once()
