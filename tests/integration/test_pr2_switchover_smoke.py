"""PR 2 switchover smoke test — updated for Task 5.2 (PR 3 cleanup).

Task 5.2 deleted the legacy path and feature flag. derive_ontology_graph now
unconditionally delegates to _derive_ontology_graph_bundle_passes. The
feature-flag dispatch tests have been replaced with a simpler wiring check.

The actual end-to-end extraction (ingest a document, produce a graph)
requires the full compose stack — tests/e2e/test_full_pipeline.py
covers that. This smoke test focuses on the code-level wiring: can the
orchestrator import chain resolve, and are the new symbols importable.
"""
from __future__ import annotations

from unittest.mock import patch


class TestBundlePassesCodePath:
    """Verify the bundle_passes branch is the sole dispatch target."""

    def test_feature_flag_field_removed(self):
        """graph_extraction_engine was removed in Task 5.2; extra=ignore
        means the field simply does not exist on Settings."""
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert not hasattr(s, "graph_extraction_engine")

    def test_derive_ontology_graph_always_dispatches_to_bundle_passes(self):
        """derive_ontology_graph unconditionally routes to bundle_passes branch."""
        from app.workers.pipeline import derive_ontology_graph

        with patch("app.workers.pipeline._derive_ontology_graph_bundle_passes") as mock_new:
            mock_new.return_value = {"status": "ok"}
            # Use .run() to bypass Celery middleware; bind=True passes self implicitly
            derive_ontology_graph.run("doc-1", "run-1")

        mock_new.assert_called_once()


class TestNewSymbolsImportable:
    """Every symbol added in Chunks 3–4 must be importable."""

    def test_extraction_merge_full_surface(self):
        from app.services.extraction_merge import (  # noqa: F401
            LogicalIdentity, PassResult, MergedEntityRecord,
            MergedEdgeRecord, MergedExtraction, ChunkForDerivation,
            DerivedEdge, RelationshipRejectionReason, YieldStatus,
            merge_and_resolve, build_display_label,
            classify_yield, classify_yield_from_counts,
        )

    def test_status_signals_importable(self):
        from app.services.status_signals import compute_status_signals  # noqa: F401
        from app.services.ontology_bundles import StatusSignals  # noqa: F401

    def test_pipeline_orchestrator_helpers_importable(self):
        from app.workers.pipeline import (  # noqa: F401
            GraphWriteTracker,
            PassRetryable, PassTerminal, IngestFailed,
            WorkerInvariantError, GateResult,
            _run_single_pass, _should_skip, check_required_pass_gate,
            _import_graph_phase_nodes,
            _import_graph_phase_domain_edges,
            _import_graph_phase_structural_edges,
            _write_pipeline_run_metrics,
            _apply_post_merge_yield_updates,
            _upsert_document_graph_extraction,
            _attempt_rollback,
            _delete_extraction_layer_graph,
            _update_document_pipeline_status,
        )

    def test_dispatch_types_importable(self):
        from app.workers.dispatch_types import IngestDispatchResult  # noqa: F401

    def test_baseline_harness_importable(self):
        from tools.extraction_baseline_harness import compare  # noqa: F401

    def test_extraction_status_endpoint_exists(self):
        """The new /extraction-status endpoint is registered on the router."""
        from app.api.v1.sources import router
        routes = [r.path for r in router.routes]
        assert "/documents/{document_id}/extraction-status" in routes
