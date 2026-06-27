"""Verify the ingest entry-point dispatch contract.

Originally this asserted the trimmed outer Celery *chain* shapes (Task 7
per-pass-celery-fanin: start_ingest_pipeline = 9 stages, reingest_graph_only
= 2 stages). The 2026-05-10 ledger-seed refactor removed the outer chain
entirely: both entry points now seed a single PENDING `stage_runs` ledger row
via `_seed_first_stage` and return `celery_task_id=""`. An external
dispatcher-poller publishes the seeded stage's Celery task within ~5s, and
each stage's lifecycle wrapper commits the next stage's PENDING row in the
same transaction as its own COMPLETE — so there is no chain to assert.

These tests now pin the surviving contract: which first stage each entry point
seeds, that no Celery chain is constructed, and the empty task id.

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
    """start_ingest_pipeline seeds prepare_document and builds NO Celery chain."""

    def test_start_ingest_pipeline_chain_ends_at_derive_ontology_graph(self):
        """Ledger-seed contract: start_ingest_pipeline seeds the first stage
        (prepare_document) for the dispatcher-poller and constructs no chain.

        The downstream stages (detect_and_translate ... derive_ontology_graph
        and beyond) are advanced by each stage's lifecycle wrapper / the merge
        fan-in — none of them are wired here.
        """
        mock_chain = MagicMock()

        with patch("app.workers.pipeline._get_db", return_value=_fake_db_full()), \
             patch("app.workers.pipeline.chain", mock_chain), \
             patch("app.workers.pipeline._seed_first_stage") as mock_seed, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=_fake_manifest()):

            from app.workers.pipeline import start_ingest_pipeline
            result = start_ingest_pipeline(_DOC_ID)

        # No outer Celery chain is constructed any more.
        mock_chain.assert_not_called()

        # Exactly one ledger row seeded, and it's prepare_document.
        mock_seed.assert_called_once()
        assert mock_seed.call_args.kwargs.get("stage_name") == "prepare_document"
        assert (
            mock_seed.call_args.kwargs.get("task_name")
            == "app.workers.pipeline.prepare_document"
        )
        # Empty task id signals "ledger-seeded; poller will publish".
        assert result.celery_task_id == ""


# ---------------------------------------------------------------------------
# Test 2: reingest_graph_only chain ends at derive_ontology_graph
# ---------------------------------------------------------------------------


class TestReingestGraphOnlyChainShape:
    """reingest_graph_only seeds derive_document_anchors and builds NO chain."""

    def test_reingest_graph_only_chain_ends_at_derive_ontology_graph(self):
        """Ledger-seed contract (graph_only): reingest_graph_only seeds the
        first stage (derive_document_anchors) for the dispatcher-poller and
        constructs no celery_chain. derive_ontology_graph is advanced by the
        anchors stage's lifecycle wrapper; downstream stages by the merge.
        """
        mock_celery_chain = MagicMock()

        with patch("app.workers.pipeline._get_db", return_value=_fake_db_graph()), \
             patch("app.workers.pipeline.celery_chain", mock_celery_chain), \
             patch("app.workers.pipeline._seed_first_stage") as mock_seed, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=_fake_manifest()):

            from app.workers.pipeline import reingest_graph_only
            result = reingest_graph_only(_DOC_ID, _fake_request())

        # No outer Celery chain is constructed any more.
        mock_celery_chain.assert_not_called()

        # Exactly one ledger row seeded, and it's derive_document_anchors.
        mock_seed.assert_called_once()
        assert mock_seed.call_args.kwargs.get("stage_name") == "derive_document_anchors"
        assert (
            mock_seed.call_args.kwargs.get("task_name")
            == "app.workers.pipeline.derive_document_anchors"
        )
        assert result["pipeline_run_id"]  # non-empty
        assert result["celery_task_id"] == ""


# ---------------------------------------------------------------------------
# Test 3: all chain stages use (doc_id, run_id) signature — 2 positional args
# ---------------------------------------------------------------------------


class TestChainStageSignatures:
    """Both entry points seed exactly one ledger row (no chain, no .si() fan-out)."""

    def test_all_chain_stages_use_doc_id_run_id_signature(self):
        """Originally verified every chain stage was called as .si(doc_id, run_id).

        After the ledger-seed refactor there are no .si() chain signatures at the
        entry points: start_ingest_pipeline and reingest_graph_only each seed a
        single PENDING ledger row. This test now asserts each entry point makes
        exactly one _seed_first_stage call (for its respective first stage) and
        constructs no Celery chain.
        """
        seed_calls: list[tuple] = []

        def record_seed(db, **kwargs):
            seed_calls.append(kwargs)

        mock_chain_full = MagicMock()
        mock_chain_graph = MagicMock()

        infra_patches = [
            patch("app.workers.pipeline._get_db", side_effect=[_fake_db_full(), _fake_db_graph()]),
            patch("app.workers.pipeline.chain", mock_chain_full),
            patch("app.workers.pipeline.celery_chain", mock_chain_graph),
            patch("app.workers.pipeline._seed_first_stage", side_effect=record_seed),
            patch(
                "app.services.ontology_bundles.load_bundle_manifest",
                return_value=_fake_manifest(),
            ),
        ]

        for p in infra_patches:
            p.start()
        try:
            from app.workers.pipeline import start_ingest_pipeline, reingest_graph_only
            start_ingest_pipeline(_DOC_ID)
            reingest_graph_only(_DOC_ID, _fake_request())
        finally:
            for p in infra_patches:
                p.stop()

        # No Celery chain constructed at either entry point.
        mock_chain_full.assert_not_called()
        mock_chain_graph.assert_not_called()

        # One seed per entry point, naming the correct first stage.
        seeded_stages = [kw.get("stage_name") for kw in seed_calls]
        assert seeded_stages == ["prepare_document", "derive_document_anchors"], (
            f"Expected one seed per entry point; got {seeded_stages}"
        )
