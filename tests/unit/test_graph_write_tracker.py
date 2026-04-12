"""Spec §5.4 ``GraphWriteTracker`` — worker-local rollback gate.

Task 4.1 of the extraction-refactor plan. Asserts the tracker's default
state, its one-way transition semantics (``.mark()`` is idempotent), and
that the helper stubs raise NotImplementedError with a task-ID back-
pointer so partial runs during Chunk 4 incremental delivery fail loudly.
"""
import pytest

from app.workers.pipeline import GraphWriteTracker


class TestGraphWriteTracker:
    def test_default_is_false(self):
        """A fresh tracker starts unmarked — no mutation attempted."""
        t = GraphWriteTracker()
        assert t.any_mutation_attempted is False

    def test_mark_flips_true(self):
        """One call to .mark() flips the gate open."""
        t = GraphWriteTracker()
        t.mark()
        assert t.any_mutation_attempted is True

    def test_mark_is_idempotent(self):
        """Multiple .mark() calls stay True — the tracker never flips back."""
        t = GraphWriteTracker()
        t.mark()
        t.mark()
        t.mark()
        assert t.any_mutation_attempted is True

    def test_independent_instances(self):
        """Two trackers don't share state (guard against accidental class-level default)."""
        a = GraphWriteTracker()
        b = GraphWriteTracker()
        a.mark()
        assert a.any_mutation_attempted is True
        assert b.any_mutation_attempted is False


class TestOrchestratorHelperStubs:
    """Every helper stub added in Task 4.1 raises NotImplementedError with
    a task-ID back-pointer so partial deliveries fail loudly instead of
    silently no-op'ing. The back-pointer text is asserted so nobody
    accidentally swallows the marker while filling in the body later."""

    def test_attempt_rollback_stub(self):
        from app.workers.pipeline import _attempt_rollback
        with pytest.raises(NotImplementedError, match="Task 4.6"):
            _attempt_rollback("doc-1")

    def test_delete_extraction_layer_graph_stub(self):
        from app.workers.pipeline import _delete_extraction_layer_graph
        with pytest.raises(NotImplementedError, match="Task 4.6"):
            _delete_extraction_layer_graph("doc-1")

    def test_write_pipeline_run_metrics_stub(self):
        from app.workers.pipeline import _write_pipeline_run_metrics
        with pytest.raises(NotImplementedError, match="Task 4.5"):
            _write_pipeline_run_metrics("run-1", None, None)

    def test_run_single_pass_stub(self):
        from app.workers.pipeline import _run_single_pass
        with pytest.raises(NotImplementedError, match="Task 4.3"):
            _run_single_pass()

    def test_should_skip_stub(self):
        from app.workers.pipeline import _should_skip
        with pytest.raises(NotImplementedError, match="Task 4.3"):
            _should_skip(None, {}, {})

    def test_apply_post_merge_yield_updates_stub(self):
        from app.workers.pipeline import _apply_post_merge_yield_updates
        with pytest.raises(NotImplementedError, match="Task 4.5"):
            _apply_post_merge_yield_updates("run-1", None)

    def test_import_graph_phase_nodes_stub(self):
        from app.workers.pipeline import _import_graph_phase_nodes
        with pytest.raises(NotImplementedError, match="Task 4.4"):
            _import_graph_phase_nodes(None, {}, "doc-1", GraphWriteTracker())

    def test_import_graph_phase_domain_edges_stub(self):
        from app.workers.pipeline import _import_graph_phase_domain_edges
        with pytest.raises(NotImplementedError, match="Task 4.4"):
            _import_graph_phase_domain_edges(None, {}, GraphWriteTracker())

    def test_import_graph_phase_structural_edges_stub(self):
        from app.workers.pipeline import _import_graph_phase_structural_edges
        with pytest.raises(NotImplementedError, match="Task 4.4"):
            _import_graph_phase_structural_edges(
                None, {}, "doc-1", "run-1", GraphWriteTracker(),
            )

    def test_update_document_pipeline_status_stub(self):
        from app.workers.pipeline import _update_document_pipeline_status
        with pytest.raises(NotImplementedError, match="Task 4.7"):
            _update_document_pipeline_status("doc-1", "COMPLETE")

    def test_check_required_pass_gate_stub(self):
        from app.workers.pipeline import check_required_pass_gate
        with pytest.raises(NotImplementedError, match="Task 4.3"):
            check_required_pass_gate("run-1")

    def test_upsert_document_graph_extraction_stub(self):
        from app.workers.pipeline import _upsert_document_graph_extraction
        with pytest.raises(NotImplementedError, match="Task 4.5"):
            _upsert_document_graph_extraction()
