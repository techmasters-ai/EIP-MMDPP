"""Spec §5.4 ``GraphWriteTracker`` — worker-local rollback gate.

Task 4.1 of the extraction-refactor plan. Asserts the tracker's default
state and its one-way transition semantics (``.mark()`` is idempotent).
The helper stubs from Task 4.1 were replaced with real implementations
in Task 4.6.
# Reference: integration tests for the rollback path now live in
# `test_derive_ontology_graph_merge_task.py` (Task 6's GraphWriteTracker
# rollback contract).
"""
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
