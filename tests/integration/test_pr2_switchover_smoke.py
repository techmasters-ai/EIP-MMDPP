"""PR 2 switchover smoke test. Task 4.9.

Verifies the new bundle_passes extraction path can be activated
per-test via monkeypatch without breaking the legacy default, and
that both code paths coexist cleanly behind the feature flag.

The actual end-to-end extraction (ingest a document, produce a graph)
requires the full compose stack — tests/e2e/test_full_pipeline.py
covers that. This smoke test focuses on the code-level wiring: can the
orchestrator import chain resolve, does the feature-flag dispatch work,
and are the new symbols importable.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestBundlePassesCodePath:
    """Verify the new bundle_passes branch is callable and wired."""

    def test_feature_flag_defaults_to_legacy(self):
        """After PR 2 merges, the flag STILL defaults to legacy.
        The flip happens during soak, not at merge time."""
        from app.config import Settings
        s = Settings(_env_file=None, postgres_password="test")
        assert s.graph_extraction_engine == "legacy"

    def test_derive_ontology_graph_dispatches_to_new_branch_when_flag_set(self):
        """Setting graph_extraction_engine=bundle_passes routes to the new path."""
        from app.workers.pipeline import derive_ontology_graph

        with patch("app.workers.pipeline._derive_ontology_graph_bundle_passes") as mock_new, \
             patch("app.workers.pipeline._derive_ontology_graph_legacy") as mock_legacy, \
             patch("app.workers.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.graph_extraction_engine = "bundle_passes"
            mock_new.return_value = {"status": "ok"}

            # Use .run() to bypass Celery middleware; bind=True passes self implicitly
            derive_ontology_graph.run("doc-1", "run-1")

        mock_new.assert_called_once()
        mock_legacy.assert_not_called()

    def test_derive_ontology_graph_dispatches_to_legacy_when_flag_default(self):
        """Default flag (legacy) still routes to the unchanged legacy path."""
        from app.workers.pipeline import derive_ontology_graph

        with patch("app.workers.pipeline._derive_ontology_graph_bundle_passes") as mock_new, \
             patch("app.workers.pipeline._derive_ontology_graph_legacy") as mock_legacy, \
             patch("app.workers.pipeline.get_settings") as mock_settings:
            mock_settings.return_value.graph_extraction_engine = "legacy"
            mock_legacy.return_value = {"status": "ok"}

            derive_ontology_graph.run("doc-1", "run-1")

        mock_legacy.assert_called_once()
        mock_new.assert_not_called()


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
