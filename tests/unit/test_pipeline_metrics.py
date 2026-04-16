"""Spec §5.4 + §5.7 + §6.6 — merge post-processing, metrics, snapshot write.

Task 4.5 of the extraction-refactor plan. Mocks the DB session so
tests stay hermetic. Integration-level verification happens via the
e2e pipeline test.
"""
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest


# --- Shared fixtures --------------------------------------------------------

def _fake_identity(entity_type="RADAR_SYSTEM", name="S-400"):
    from app.services.extraction_merge import LogicalIdentity
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=("system_name",),
        identity_tuple=(name,),
        scope="global",
        document_id=None,
    )


def _fake_merged(
    entities=None, edges=None, rejected_edges=None,
    pipeline_run_id="run-1", document_id="doc-1",
):
    from app.services.extraction_merge import MergedExtraction, MergedEntityRecord, MergedEdgeRecord
    if entities is None:
        entities = [
            MergedEntityRecord(
                identity=_fake_identity(),
                properties={"system_name": "S-400"},
                confidence=0.95,
                pass_origins={"radar_domain"},
                display_label="S-400",
            ),
        ]
    return MergedExtraction(
        entities=entities,
        edges=edges or [],
        rejected_edges=rejected_edges or [],
        rejections_by_pass={},
        pipeline_run_id=pipeline_run_id,
        document_id=document_id,
    )


def _fake_manifest(bundle_key="air_defense_v3"):
    """BundleManifest-ish stub."""
    return SimpleNamespace(
        bundle_key=bundle_key,
        ontology_name="EIP Military Equipment Ontology",
        ontology_version="3.0.0",
        extraction_profile_version="1.0.0",
        passes=[
            SimpleNamespace(
                name="radar_domain",
                primary_entity_types=["RADAR_SYSTEM"],
                bridge_entity_types=["PLATFORM", "SPECIFICATION"],
                extracted_relationship_types=["INSTALLED_ON"],
            ),
            SimpleNamespace(
                name="missile_domain",
                primary_entity_types=["MISSILE_SYSTEM"],
                bridge_entity_types=["PLATFORM", "SPECIFICATION"],
                extracted_relationship_types=[],
            ),
            SimpleNamespace(
                name="other_systems",
                primary_entity_types=["AIR_DEFENSE_ARTILLERY_SYSTEM"],
                bridge_entity_types=["PLATFORM", "SPECIFICATION"],
                extracted_relationship_types=[],
            ),
        ],
    )


def _fake_pipeline_run(bundle_key="air_defense_v3", ontology_name="EIP Military Equipment Ontology"):
    run = MagicMock()
    run.metrics = None
    run.ontology_bundle_key = bundle_key
    run.ontology_name = ontology_name
    run.ontology_version = "3.0.0"
    run.use_case_key = None
    run.extraction_profile_version = "1.0.0"
    return run


# --- _write_pipeline_run_metrics --------------------------------------------

class TestWritePipelineRunMetrics:

    def test_populates_all_required_keys(self):
        from app.workers.pipeline import _write_pipeline_run_metrics

        merged = _fake_merged()
        manifest = _fake_manifest()
        fake_run = _fake_pipeline_run()

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.get = MagicMock(return_value=fake_run)
        # _build_pass_outcomes_rollup queries via session.execute
        session.execute.return_value.all.return_value = []

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _write_pipeline_run_metrics("run-1", merged, manifest)

        assert fake_run.metrics is not None
        keys = set(fake_run.metrics.keys())
        expected = {
            "pass_outcomes",
            "document_extraction_anomaly",
            "pass_degraded_count",
            "overall_relationship_rejection_ratio",
            "rejected_relationships_sample",
            "bundle_legacy",
            "bundle_key_display",
        }
        assert expected.issubset(keys)
        assert fake_run.metrics["bundle_legacy"] is False
        assert fake_run.metrics["bundle_key_display"] == "air_defense_v3"

    def test_document_extraction_anomaly_true_when_all_core_passes_empty(self):
        from app.workers.pipeline import _write_pipeline_run_metrics
        from types import SimpleNamespace

        merged = _fake_merged()
        manifest = _fake_manifest()
        fake_run = _fake_pipeline_run()

        # Simulate v_latest_pass_attempts: all 3 core passes end EMPTY
        fake_rows = [
            SimpleNamespace(
                pass_name="radar_domain", execution_status="COMPLETE",
                yield_status="EMPTY", attempt=1,
                primary_entities_extracted=0, bridge_entities_extracted=0,
                relationships_extracted=0, relationships_rejected=0,
            ),
            SimpleNamespace(
                pass_name="missile_domain", execution_status="COMPLETE",
                yield_status="EMPTY", attempt=1,
                primary_entities_extracted=0, bridge_entities_extracted=0,
                relationships_extracted=0, relationships_rejected=0,
            ),
            SimpleNamespace(
                pass_name="other_systems", execution_status="COMPLETE",
                yield_status="BRIDGES_ONLY", attempt=1,
                primary_entities_extracted=0, bridge_entities_extracted=2,
                relationships_extracted=0, relationships_rejected=0,
            ),
        ]

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.get = MagicMock(return_value=fake_run)
        session.execute.return_value.all.return_value = fake_rows

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _write_pipeline_run_metrics("run-1", merged, manifest)

        assert fake_run.metrics["document_extraction_anomaly"] is True

    def test_document_extraction_anomaly_false_when_any_core_pass_hits(self):
        from app.workers.pipeline import _write_pipeline_run_metrics
        from types import SimpleNamespace

        merged = _fake_merged()
        manifest = _fake_manifest()
        fake_run = _fake_pipeline_run()

        fake_rows = [
            SimpleNamespace(
                pass_name="radar_domain", execution_status="COMPLETE",
                yield_status="HIT", attempt=1,
                primary_entities_extracted=5, bridge_entities_extracted=0,
                relationships_extracted=3, relationships_rejected=0,
            ),
            SimpleNamespace(
                pass_name="missile_domain", execution_status="COMPLETE",
                yield_status="EMPTY", attempt=1,
                primary_entities_extracted=0, bridge_entities_extracted=0,
                relationships_extracted=0, relationships_rejected=0,
            ),
            SimpleNamespace(
                pass_name="other_systems", execution_status="COMPLETE",
                yield_status="EMPTY", attempt=1,
                primary_entities_extracted=0, bridge_entities_extracted=0,
                relationships_extracted=0, relationships_rejected=0,
            ),
        ]

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.get = MagicMock(return_value=fake_run)
        session.execute.return_value.all.return_value = fake_rows

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _write_pipeline_run_metrics("run-1", merged, manifest)

        assert fake_run.metrics["document_extraction_anomaly"] is False

    def test_bundle_legacy_false_with_real_bundle(self):
        """bundle_legacy is always False for runs going through the new path
        (they always have a real bundle_key). The legacy flag is for the
        cross-run aggregator, not individual rows."""
        from app.workers.pipeline import _write_pipeline_run_metrics

        merged = _fake_merged()
        manifest = _fake_manifest()
        fake_run = _fake_pipeline_run()

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.get = MagicMock(return_value=fake_run)
        session.execute.return_value.all.return_value = []

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _write_pipeline_run_metrics("run-1", merged, manifest)

        assert fake_run.metrics["bundle_legacy"] is False


# --- _apply_post_merge_yield_updates ----------------------------------------

class TestApplyPostMergeYieldUpdates:

    def test_hit_to_degraded_transition(self):
        """HIT pass becomes DEGRADED when post-merge rejections push ratio >= 75%."""
        from app.workers.pipeline import _apply_post_merge_yield_updates

        merged = _fake_merged(
            edges=[SimpleNamespace(pass_origins={"radar_domain"}, rel_type="INSTALLED_ON")],
            rejected_edges=[
                ("radar_domain", SimpleNamespace(), SimpleNamespace(value="invalid_triple"))
                for _ in range(9)
            ],
        )

        # StageRun row that was classified HIT pre-merge
        stage_row = MagicMock()
        stage_row.pass_name = "radar_domain"
        stage_row.yield_status = "HIT"
        stage_row.primary_entities_extracted = 3
        stage_row.bridge_entities_extracted = 0

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.all.return_value = [stage_row]

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _apply_post_merge_yield_updates("run-1", merged)

        # HIT → DEGRADED because 9/10 = 90% rejection ratio, above 75% threshold
        assert stage_row.yield_status == "DEGRADED"
        assert stage_row.relationships_extracted == 1
        assert stage_row.relationships_rejected == 9

    def test_hit_stays_hit_when_rejection_ratio_low(self):
        """HIT pass stays HIT when most edges survive merge."""
        from app.workers.pipeline import _apply_post_merge_yield_updates

        merged = _fake_merged(
            edges=[SimpleNamespace(pass_origins={"radar_domain"}, rel_type="INSTALLED_ON") for _ in range(5)],
            rejected_edges=[],
        )

        stage_row = MagicMock()
        stage_row.pass_name = "radar_domain"
        stage_row.yield_status = "HIT"
        stage_row.primary_entities_extracted = 3
        stage_row.bridge_entities_extracted = 0

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.all.return_value = [stage_row]

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _apply_post_merge_yield_updates("run-1", merged)

        assert stage_row.yield_status == "HIT"

    def test_empty_stays_empty(self):
        """EMPTY pass is NOT promoted to HIT even if post-merge edges arrive.
        (Post-merge can't create edges out of nothing — this is a sanity check.)"""
        from app.workers.pipeline import _apply_post_merge_yield_updates

        merged = _fake_merged(edges=[], rejected_edges=[])

        stage_row = MagicMock()
        stage_row.pass_name = "radar_domain"
        stage_row.yield_status = "EMPTY"
        stage_row.primary_entities_extracted = 0
        stage_row.bridge_entities_extracted = 0

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.all.return_value = [stage_row]

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _apply_post_merge_yield_updates("run-1", merged)

        # Stays EMPTY — the only allowed transition is HIT→DEGRADED
        assert stage_row.yield_status == "EMPTY"


# --- _upsert_document_graph_extraction --------------------------------------

class TestUpsertDocumentGraphExtraction:

    def test_insert_on_first_write(self):
        from app.workers.pipeline import _upsert_document_graph_extraction

        merged = _fake_merged()
        run = _fake_pipeline_run()
        manifest = _fake_manifest()

        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        # No existing row
        session.query.return_value.filter_by.return_value.first.return_value = None

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _upsert_document_graph_extraction(
                document_id="doc-1",
                pipeline_run_id="run-1",
                run=run,
                merged=merged,
                manifest=manifest,
            )

        # One .add() call for the new row
        assert session.add.call_count == 1

    def test_update_on_subsequent_write(self):
        from app.workers.pipeline import _upsert_document_graph_extraction

        merged = _fake_merged()
        run = _fake_pipeline_run()
        manifest = _fake_manifest()

        existing = MagicMock()
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter_by.return_value.first.return_value = existing

        with patch("app.workers.pipeline.get_sync_session", return_value=session):
            _upsert_document_graph_extraction(
                document_id="doc-1",
                pipeline_run_id="run-1",
                run=run,
                merged=merged,
                manifest=manifest,
            )

        # No new row inserted — existing was updated in place
        assert session.add.call_count == 0
        # Key columns got set via setattr
        assert existing.pipeline_run_id == "run-1"
        assert existing.ontology_bundle_key == "air_defense_v3"


# --- _serialize_for_audit ---------------------------------------------------

class TestSerializeForAudit:

    def test_distinguishes_primary_and_bridge_by_manifest(self):
        from app.workers.pipeline import _serialize_for_audit
        from app.services.extraction_merge import MergedEntityRecord

        manifest = _fake_manifest()
        # 2 RADAR_SYSTEM (primary in radar_domain) + 1 PLATFORM (bridge)
        merged = _fake_merged(entities=[
            MergedEntityRecord(
                identity=_fake_identity(entity_type="RADAR_SYSTEM", name="S-400"),
                properties={}, confidence=0.9,
                pass_origins={"radar_domain"}, display_label="S-400",
            ),
            MergedEntityRecord(
                identity=_fake_identity(entity_type="RADAR_SYSTEM", name="S-300"),
                properties={}, confidence=0.9,
                pass_origins={"radar_domain"}, display_label="S-300",
            ),
            MergedEntityRecord(
                identity=_fake_identity(entity_type="PLATFORM", name="Truck"),
                properties={}, confidence=0.9,
                pass_origins={"radar_domain"}, display_label="Truck",
            ),
        ])

        blob = _serialize_for_audit(merged, manifest)

        assert blob["primary_entities_total"] == 2
        assert blob["bridge_entities_total"] == 1
        assert blob["entity_count_by_type"] == {"RADAR_SYSTEM": 2, "PLATFORM": 1}
        assert blob["edges_accepted"] == 0
        assert blob["edges_rejected"] == 0
