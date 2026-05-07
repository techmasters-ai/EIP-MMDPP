"""Verify trimmed chain shapes after Task 7 (per-pass-celery-fanin).

After the outer-chain trim:
- start_ingest_pipeline: 13 → 9 stages (ends at derive_ontology_graph)
- reingest_graph_only: 4 → 2 stages (ends at derive_ontology_graph)

Downstream stages are dispatched by derive_ontology_graph_merge post fan-in.

Run with:

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \\
        /home/josh/development/EIP-MMDPP/.venv/bin/python -m pytest \\
        tests/unit/test_chain_shape_changes.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_DOC_ID = "00000000-0000-0000-0000-000000000099"


def _fake_manifest():
    m = MagicMock()
    m.bundle_key = "air_defense_v3"
    m.ontology_name = "EIP Military Equipment Ontology"
    m.ontology_version = "3.0.0"
    m.extraction_profile_version = "1.0.0"
    return m


def _fake_db_full():
    """Minimal DB mock that satisfies start_ingest_pipeline's guard checks."""
    db = MagicMock()
    db.execute.return_value.scalar_one_or_none.return_value = None  # no active run
    db.get.return_value = None  # no Document row needed
    return db


def _fake_db_graph():
    """Minimal DB mock for reingest_graph_only."""
    db = MagicMock()
    db.get.return_value = None
    db.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = None
    return db


def _fake_request():
    """Minimal request stub for reingest_graph_only."""
    r = MagicMock()
    r.ontology_bundle_key = None
    r.use_case_key = None
    return r


# ---------------------------------------------------------------------------
# Test 1: start_ingest_pipeline chain ends at derive_ontology_graph
# ---------------------------------------------------------------------------


class TestStartIngestPipelineChainShape:
    """Verify start_ingest_pipeline outer chain is exactly 9 stages after Task 7 trim."""

    def test_start_ingest_pipeline_chain_ends_at_derive_ontology_graph(self):
        """Chain must be exactly 9 stages ending at derive_ontology_graph.

        NO collect_derivations, derive_structure_links, derive_canonicalization,
        or finalize_document in the outer chain — those are dispatched by
        derive_ontology_graph_merge after the per-pass fan-in.
        """
        mock_chain = MagicMock()
        mock_chain.return_value.apply_async.return_value = MagicMock(id="task-xyz")

        with patch("app.workers.pipeline._get_db", return_value=_fake_db_full()), \
             patch("app.workers.pipeline.chain", mock_chain), \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=_fake_manifest()):

            from app.workers.pipeline import start_ingest_pipeline
            start_ingest_pipeline(_DOC_ID)

        mock_chain.assert_called_once()
        chain_args = mock_chain.call_args[0]  # positional args = the task signatures

        # Must be exactly 9 stages
        assert len(chain_args) == 9, (
            f"Expected 9 stages in start_ingest_pipeline chain, got {len(chain_args)}. "
            f"Task names: {[getattr(a, 'task', repr(a)) for a in chain_args]}"
        )

        # Verify task name presence via .task attribute on each Signature
        task_names = [getattr(sig, "task", "") for sig in chain_args]

        # Assert the full ordered sequence — a stage-reorder regression will fail here
        expected_sequence = [
            "prepare_document",
            "detect_and_translate",
            "derive_document_metadata",
            "purge_document_derivations",
            "derive_picture_descriptions",
            "derive_text_chunks_and_embeddings",
            "derive_image_embeddings",
            "derive_document_anchors",
            "derive_ontology_graph",
        ]
        actual_sequence = [name.rsplit(".", 1)[-1] for name in task_names]
        assert actual_sequence == expected_sequence, (
            f"Chain order mismatch: expected {expected_sequence}, got {actual_sequence}"
        )

        # Excluded stages must be absent
        excluded = [
            "collect_derivations",
            "derive_structure_links",
            "derive_canonicalization",
            "finalize_document",
        ]
        for exc_name in excluded:
            assert not any(exc_name in n for n in task_names), (
                f"Stage '{exc_name}' must NOT appear in outer chain. "
                f"Captured task names: {task_names}"
            )


# ---------------------------------------------------------------------------
# Test 2: reingest_graph_only chain ends at derive_ontology_graph
# ---------------------------------------------------------------------------


class TestReingestGraphOnlyChainShape:
    """Verify reingest_graph_only outer chain is exactly 2 stages after Task 7 trim."""

    def test_reingest_graph_only_chain_ends_at_derive_ontology_graph(self):
        """Chain must be exactly 2 stages: derive_document_anchors, derive_ontology_graph.

        NO derive_structure_links or finalize_document in the outer chain —
        those are dispatched by derive_ontology_graph_merge in graph_only mode.
        """
        mock_celery_chain = MagicMock()
        mock_celery_chain.return_value.apply_async.return_value = MagicMock(id="task-graph")

        with patch("app.workers.pipeline._get_db", return_value=_fake_db_graph()), \
             patch("app.workers.pipeline.celery_chain", mock_celery_chain), \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=_fake_manifest()):

            from app.workers.pipeline import reingest_graph_only
            result = reingest_graph_only(_DOC_ID, _fake_request())

        mock_celery_chain.assert_called_once()
        chain_args = mock_celery_chain.call_args[0]

        # Must be exactly 2 stages
        assert len(chain_args) == 2, (
            f"Expected 2 stages in reingest_graph_only chain, got {len(chain_args)}. "
            f"Task names: {[getattr(a, 'task', repr(a)) for a in chain_args]}"
        )

        task_names = [getattr(sig, "task", "") for sig in chain_args]

        # Assert the full ordered sequence — a stage-reorder regression will fail here
        expected_sequence = ["derive_document_anchors", "derive_ontology_graph"]
        actual_sequence = [name.rsplit(".", 1)[-1] for name in task_names]
        assert actual_sequence == expected_sequence, (
            f"Chain order mismatch: expected {expected_sequence}, got {actual_sequence}"
        )

        # Excluded stages must be absent
        excluded = ["derive_structure_links", "finalize_document"]
        for exc_name in excluded:
            assert not any(exc_name in n for n in task_names), (
                f"Stage '{exc_name}' must NOT appear in outer chain. "
                f"Captured task names: {task_names}"
            )

        assert result["pipeline_run_id"]  # non-empty


# ---------------------------------------------------------------------------
# Test 3: all chain stages use (doc_id, run_id) signature — 2 positional args
# ---------------------------------------------------------------------------


class TestChainStageSignatures:
    """Verify every .si() call in both chains passes exactly 2 positional args."""

    def test_all_chain_stages_use_doc_id_run_id_signature(self):
        """Every stage in both chains must be called with .si(doc_id, run_id).

        Patches each pipeline task's .si() to record (task_name, args), then
        runs both start_ingest_pipeline and reingest_graph_only, asserting
        every captured .si() call has exactly 2 positional arguments.
        """
        si_calls: list[tuple[str, tuple]] = []  # [(task_name, args), ...]

        def make_si_tracker(task_name: str):
            def tracked_si(*args, **kwargs):
                si_calls.append((task_name, args))
                sig = MagicMock()
                sig.task = task_name
                return sig
            return tracked_si

        # Tasks appearing in the full chain
        full_chain_tasks = [
            "prepare_document",
            "detect_and_translate",
            "derive_document_metadata",
            "purge_document_derivations",
            "derive_picture_descriptions",
            "derive_text_chunks_and_embeddings",
            "derive_image_embeddings",
            "derive_document_anchors",
            "derive_ontology_graph",
        ]
        # Tasks appearing in the graph_only chain (subset)
        graph_only_tasks = ["derive_document_anchors", "derive_ontology_graph"]

        # Build patch list — deduplicate (derive_document_anchors + derive_ontology_graph
        # appear in both chains but each patch.start() replaces the same attribute once)
        all_task_names = list(dict.fromkeys(full_chain_tasks + graph_only_tasks))

        mock_chain_full = MagicMock()
        mock_chain_full.return_value.apply_async.return_value = MagicMock(id="t1")

        mock_chain_graph = MagicMock()
        mock_chain_graph.return_value.apply_async.return_value = MagicMock(id="t2")

        task_patches = [
            patch(f"app.workers.pipeline.{name}.si", side_effect=make_si_tracker(name))
            for name in all_task_names
        ]
        infra_patches = [
            patch("app.workers.pipeline._get_db", side_effect=[_fake_db_full(), _fake_db_graph()]),
            patch("app.workers.pipeline.chain", mock_chain_full),
            patch("app.workers.pipeline.celery_chain", mock_chain_graph),
            patch(
                "app.services.ontology_bundles.load_bundle_manifest",
                return_value=_fake_manifest(),
            ),
        ]

        all_patches = infra_patches + task_patches
        for p in all_patches:
            p.start()
        try:
            from app.workers.pipeline import start_ingest_pipeline, reingest_graph_only
            start_ingest_pipeline(_DOC_ID)
            reingest_graph_only(_DOC_ID, _fake_request())
        finally:
            for p in all_patches:
                p.stop()

        assert len(si_calls) > 0, (
            "No .si() calls captured — task patching likely failed"
        )

        bad_calls = [
            (name, args)
            for name, args in si_calls
            if len(args) != 2
        ]
        assert not bad_calls, (
            f"These .si() calls do NOT pass exactly 2 positional args (doc_id, run_id): "
            f"{bad_calls}"
        )
