"""Spec §5.2 / §5.3 — ingest entry points and bundle resolution.

Mocks the DB session and celery chain to isolate the refactor's
contract (return type, bundle snapshotting, precedence) from
infrastructure. Integration-level verification comes from
tests/e2e/test_full_pipeline.py which still runs the legacy path.
"""
from unittest.mock import MagicMock, patch
import uuid

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_db_session():
    """Yields a MagicMock-backed fake session that tracks .add(run) and
    returns it via .get(Document, id) lookups."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


@pytest.fixture
def fake_manifest():
    """A BundleManifest-ish stub with the three attributes the refactor reads."""
    m = MagicMock()
    m.bundle_key = "air_defense_v3"
    m.ontology_name = "EIP Military Equipment Ontology"
    m.ontology_version = "3.0.0"
    m.extraction_profile_version = "1.0.0"
    return m


class TestStartIngestPipeline:

    def test_returns_ingest_dispatch_result(self, fake_db_session, fake_manifest):
        """The refactored entry point returns IngestDispatchResult.

        Post 2026-05-10 ledger-seed refactor: start_ingest_pipeline no longer
        builds/dispatches a Celery chain. It seeds the first PENDING ledger row
        via _seed_first_stage (the dispatcher-poller publishes prepare_document
        within 5s) and returns celery_task_id="" by contract. The non-empty
        task id from the old chain().apply_async().id path no longer exists.
        """
        from app.workers.dispatch_types import IngestDispatchResult

        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline._seed_first_stage") as mock_seed, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=fake_manifest):
            # No active run; fake_db_session.execute returns None for the active check
            fake_db_session.execute.return_value.scalar_one_or_none.return_value = None
            fake_db_session.get = MagicMock(return_value=None)  # no Document row is ok for this test

            from app.workers.pipeline import start_ingest_pipeline
            result = start_ingest_pipeline("00000000-0000-0000-0000-000000000001")

        assert isinstance(result, IngestDispatchResult)
        # Ledger-seed contract: empty task id, first stage seeded for the poller.
        assert result.celery_task_id == ""
        assert result.pipeline_run_id  # non-empty
        mock_seed.assert_called_once()
        # The seeded first stage is prepare_document.
        assert mock_seed.call_args.kwargs.get("stage_name") == "prepare_document"

    def test_snapshots_bundle_on_pipeline_run(self, fake_db_session, fake_manifest):
        """PipelineRun row gets ontology_bundle_key, ontology_name, ontology_version,
        extraction_profile_version, mode='full'."""
        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline.chain") as mock_chain, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=fake_manifest):
            fake_db_session.execute.return_value.scalar_one_or_none.return_value = None
            fake_db_session.get = MagicMock(return_value=None)
            mock_chain.return_value.apply_async.return_value = MagicMock(id="task-xyz")

            from app.workers.pipeline import start_ingest_pipeline
            start_ingest_pipeline("00000000-0000-0000-0000-000000000001")

        # Inspect the PipelineRun row that was passed to session.add()
        added_rows = [call.args[0] for call in fake_db_session.add.call_args_list]
        pipeline_run = next(
            (r for r in added_rows if r.__class__.__name__ == "PipelineRun"), None,
        )
        assert pipeline_run is not None
        assert pipeline_run.ontology_bundle_key == "air_defense_v3"
        assert pipeline_run.ontology_name == "EIP Military Equipment Ontology"
        assert pipeline_run.ontology_version == "3.0.0"
        assert pipeline_run.extraction_profile_version == "1.0.0"
        assert pipeline_run.mode == "full"
        assert pipeline_run.status == "PROCESSING"

    def test_explicit_override_wins(self, fake_db_session, fake_manifest):
        """Caller-provided ontology_bundle_key beats source default."""
        # Mock a Document with a source that has a different default
        mock_source = MagicMock()
        mock_source.default_ontology_bundle_key = "source_override_bundle"
        mock_source.default_use_case_key = None
        mock_document = MagicMock()
        mock_document.source = mock_source

        fake_manifest.bundle_key = "explicit_bundle"

        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline.chain") as mock_chain, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=fake_manifest) as mock_load:
            fake_db_session.execute.return_value.scalar_one_or_none.return_value = None
            fake_db_session.get = MagicMock(return_value=mock_document)
            mock_chain.return_value.apply_async.return_value = MagicMock(id="task-1")

            from app.workers.pipeline import start_ingest_pipeline
            start_ingest_pipeline(
                "00000000-0000-0000-0000-000000000001",
                ontology_bundle_key="explicit_bundle",
            )

        # Manifest was loaded with the explicit bundle, not the source default
        mock_load.assert_called_with("explicit_bundle")

    def test_source_default_beats_system_default(self, fake_db_session, fake_manifest):
        """When caller passes no override, the Source's default_ontology_bundle_key wins."""
        mock_source = MagicMock()
        mock_source.default_ontology_bundle_key = "source_bundle"
        mock_source.default_use_case_key = None
        mock_document = MagicMock()
        mock_document.source = mock_source

        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline.chain") as mock_chain, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=fake_manifest) as mock_load:
            fake_db_session.execute.return_value.scalar_one_or_none.return_value = None
            fake_db_session.get = MagicMock(return_value=mock_document)
            mock_chain.return_value.apply_async.return_value = MagicMock(id="task-1")

            from app.workers.pipeline import start_ingest_pipeline
            start_ingest_pipeline("00000000-0000-0000-0000-000000000001")

        mock_load.assert_called_with("source_bundle")

    def test_system_default_final_fallback(self, fake_db_session, fake_manifest):
        """No explicit, no source default → system default."""
        mock_source = MagicMock()
        mock_source.default_ontology_bundle_key = None
        mock_source.default_use_case_key = None
        mock_document = MagicMock()
        mock_document.source = mock_source

        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline.chain") as mock_chain, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=fake_manifest) as mock_load:
            fake_db_session.execute.return_value.scalar_one_or_none.return_value = None
            fake_db_session.get = MagicMock(return_value=mock_document)
            mock_chain.return_value.apply_async.return_value = MagicMock(id="task-1")

            from app.workers.pipeline import start_ingest_pipeline
            start_ingest_pipeline("00000000-0000-0000-0000-000000000001")

        # System default from settings
        from app.config import get_settings
        assert mock_load.call_args[0][0] == get_settings().default_ontology_bundle_key

    def test_active_run_returns_existing(self, fake_db_session):
        """Duplicate-dispatch guard: return the existing run id without creating a new one."""
        existing_run_id = uuid.uuid4()
        fake_db_session.execute.return_value.scalar_one_or_none.return_value = existing_run_id

        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline.chain") as mock_chain:
            from app.workers.pipeline import start_ingest_pipeline
            result = start_ingest_pipeline("00000000-0000-0000-0000-000000000001")

        from app.workers.dispatch_types import IngestDispatchResult
        assert isinstance(result, IngestDispatchResult)
        assert result.pipeline_run_id == str(existing_run_id)
        # No new chain was dispatched
        mock_chain.assert_not_called()


class TestReingestGraphOnly:

    def test_inherits_bundle_from_latest_run(self, fake_db_session, fake_manifest):
        """Spec §5.3: latest run's ontology_bundle_key wins when caller passes no override."""
        latest_run = MagicMock()
        latest_run.ontology_bundle_key = "inherited_bundle"
        latest_run.use_case_key = None

        mock_source = MagicMock()
        mock_source.default_ontology_bundle_key = "source_bundle"
        mock_source.default_use_case_key = None
        mock_document = MagicMock()
        mock_document.source = mock_source

        fake_db_session.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = latest_run

        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline.celery_chain") as mock_chain, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=fake_manifest) as mock_load:
            fake_db_session.get = MagicMock(return_value=mock_document)
            mock_chain.return_value.apply_async.return_value = MagicMock(id="task-graph")

            from app.workers.pipeline import reingest_graph_only
            request = MagicMock(ontology_bundle_key=None, use_case_key=None)
            result = reingest_graph_only(
                "00000000-0000-0000-0000-000000000001",
                request,
            )

        mock_load.assert_called_with("inherited_bundle")
        assert "pipeline_run_id" in result

    def test_falls_back_to_source_when_latest_is_legacy(self, fake_db_session, fake_manifest):
        """Legacy latest run has ontology_bundle_key=None → fall through to source default."""
        latest_run = MagicMock()
        latest_run.ontology_bundle_key = None
        latest_run.use_case_key = None

        mock_source = MagicMock()
        mock_source.default_ontology_bundle_key = "source_bundle"
        mock_source.default_use_case_key = None
        mock_document = MagicMock()
        mock_document.source = mock_source

        fake_db_session.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = latest_run

        with patch("app.workers.pipeline._get_db", return_value=fake_db_session), \
             patch("app.workers.pipeline.celery_chain") as mock_chain, \
             patch("app.services.ontology_bundles.load_bundle_manifest", return_value=fake_manifest) as mock_load:
            fake_db_session.get = MagicMock(return_value=mock_document)
            mock_chain.return_value.apply_async.return_value = MagicMock(id="task-graph")

            from app.workers.pipeline import reingest_graph_only
            request = MagicMock(ontology_bundle_key=None, use_case_key=None)
            reingest_graph_only(
                "00000000-0000-0000-0000-000000000001",
                request,
            )

        mock_load.assert_called_with("source_bundle")
