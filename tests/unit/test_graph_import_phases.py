"""Spec §5.6 three-phase import helpers.

Unit-level tests focused on tracker semantics (tracker.mark() is called
before the first graph_store mutation in each phase) and correct record
assembly. Integration-level verification of the rollback contract lives
in tests/integration/test_arcadedb_extraction_rollback.py.

Task 4.4 of the extraction-refactor plan.
"""
from unittest.mock import MagicMock, patch

import pytest


# --- Shared fixtures --------------------------------------------------------

def _fake_identity(entity_type="RADAR_SYSTEM", values=("S-400",), document_id=None):
    """LogicalIdentity stub. Mirrors Task 3.4's frozen dataclass shape."""
    from app.services.extraction_merge import LogicalIdentity
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=("system_name",),
        identity_tuple=values,
        scope="global" if document_id is None else "document",
        document_id=document_id,
    )


def _fake_merged_entity(entity_type="RADAR_SYSTEM", name="S-400", confidence=0.95):
    from app.services.extraction_merge import MergedEntityRecord
    return MergedEntityRecord(
        identity=_fake_identity(entity_type=entity_type, values=(name,)),
        properties={"system_name": name, "nomenclature": "X"},
        confidence=confidence,
        pass_origins={"radar_domain"},
        display_label=name,
    )


def _fake_merged_edge(
    from_entity="S-400", to_entity="Truck-1", rel_type="INSTALLED_ON", confidence=0.9,
):
    from app.services.extraction_merge import MergedEdgeRecord
    return MergedEdgeRecord(
        from_identity=_fake_identity(entity_type="RADAR_SYSTEM", values=(from_entity,)),
        to_identity=_fake_identity(entity_type="PLATFORM", values=(to_entity,)),
        rel_type=rel_type,
        confidence=confidence,
        pass_origins={"radar_domain"},
    )


def _fake_merged(entities=None, edges=None):
    from app.services.extraction_merge import MergedExtraction
    return MergedExtraction(
        entities=entities or [_fake_merged_entity()],
        edges=edges or [],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
    )


# --- Phase 2: node upsert ---------------------------------------------------

class TestPhaseNodes:
    def test_marks_tracker_before_mutation(self):
        """Tracker flips to True when upsert_nodes_batch_sync is called."""
        from app.workers.pipeline import _import_graph_phase_nodes, GraphWriteTracker

        merged = _fake_merged()
        tracker = GraphWriteTracker()
        ontology = {
            "entity_types": [
                {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
            ],
        }

        mock_store = MagicMock()
        mock_store.upsert_nodes_batch_sync.return_value = ["#10:1"]

        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            identity_to_rid = _import_graph_phase_nodes(
                merged, ontology, "doc-1", tracker,
            )

        # Tracker was flipped
        assert tracker.any_mutation_attempted is True
        # upsert_nodes_batch_sync was called exactly once with 1 NodeRecord
        assert mock_store.upsert_nodes_batch_sync.call_count == 1
        node_records = mock_store.upsert_nodes_batch_sync.call_args.args[0]
        assert len(node_records) == 1
        # Returned identity_to_rid has the entity
        assert len(identity_to_rid) == 1

    def test_leaves_tracker_false_on_pre_mutation_failure(self):
        """If building NodeRecord list raises, tracker stays False.

        Simulates build_display_label raising — the tracker should NOT
        be marked because no mutation was attempted.
        """
        from app.workers.pipeline import _import_graph_phase_nodes, GraphWriteTracker

        merged = _fake_merged()
        tracker = GraphWriteTracker()
        ontology = {"entity_types": []}

        mock_store = MagicMock()

        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store), \
             patch("app.workers.pipeline.build_display_label", side_effect=ValueError("bad label")):
            with pytest.raises(ValueError):
                _import_graph_phase_nodes(merged, ontology, "doc-1", tracker)

        assert tracker.any_mutation_attempted is False
        assert mock_store.upsert_nodes_batch_sync.call_count == 0

    def test_passes_pipeline_run_id_in_provenance(self):
        """ProvenanceMetadata includes the new pipeline_run_id field from Task 3.3."""
        from app.workers.pipeline import _import_graph_phase_nodes, GraphWriteTracker

        merged = _fake_merged()
        tracker = GraphWriteTracker()
        ontology = {"entity_types": []}

        mock_store = MagicMock()
        mock_store.upsert_nodes_batch_sync.return_value = ["#10:1"]

        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            _import_graph_phase_nodes(merged, ontology, "doc-1", tracker)

        provenance = mock_store.upsert_nodes_batch_sync.call_args.args[1]
        assert provenance.document_id == "doc-1"
        assert provenance.pipeline_run_id == "run-1"


# --- Phase 3: domain edges --------------------------------------------------

class TestPhaseDomainEdges:
    def test_uses_existing_tracker_state(self):
        """Phase 3 does not reset the tracker — phase 2 may have already marked it."""
        from app.workers.pipeline import _import_graph_phase_domain_edges, GraphWriteTracker

        merged = _fake_merged(edges=[_fake_merged_edge()])
        tracker = GraphWriteTracker()
        # Pretend phase 2 already ran
        tracker.mark()

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            _import_graph_phase_domain_edges(merged, {}, tracker)

        # Tracker still True (idempotent), one call to upsert_relationships_batch_sync
        assert tracker.any_mutation_attempted is True
        assert mock_store.upsert_relationships_batch_sync.call_count == 1

    def test_marks_tracker_on_first_edge(self):
        """Phase 3 also marks the tracker (idempotent) before its mutation,
        defensively in case phase 2 was skipped (e.g., no entities)."""
        from app.workers.pipeline import _import_graph_phase_domain_edges, GraphWriteTracker

        merged = _fake_merged(edges=[_fake_merged_edge()])
        tracker = GraphWriteTracker()
        # Start with tracker False
        assert tracker.any_mutation_attempted is False

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            _import_graph_phase_domain_edges(merged, {}, tracker)

        assert tracker.any_mutation_attempted is True

    def test_empty_edges_still_calls_upsert(self):
        """Even with an empty edges list, phase 3 calls upsert_relationships_batch_sync
        with an empty list — matches the existing graph_store semantics."""
        from app.workers.pipeline import _import_graph_phase_domain_edges, GraphWriteTracker

        merged = _fake_merged(edges=[])
        tracker = GraphWriteTracker()

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            _import_graph_phase_domain_edges(merged, {}, tracker)

        # Called with empty list
        assert mock_store.upsert_relationships_batch_sync.call_count == 1
        rel_records = mock_store.upsert_relationships_batch_sync.call_args.args[0]
        assert rel_records == []


# --- Phase 4: derived structural edges --------------------------------------

class TestPhaseStructuralEdges:
    def test_noop_for_empty_derived_list(self):
        """Empty derived list → no graph_store calls, tracker state unchanged."""
        from app.workers.pipeline import (
            _import_graph_phase_structural_edges,
            GraphWriteTracker,
        )

        merged = _fake_merged()
        tracker = GraphWriteTracker()
        tracker.mark()  # pretend phase 2 already marked

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store), \
             patch("app.workers.pipeline._load_chunks_for_derivation", return_value=[]), \
             patch("app.workers.pipeline._get_structural_document_rid", return_value="#1:0"), \
             patch("ontology_bundles.air_defense_v3.derive_rules.derive_structural_edges", return_value=[]):
            _import_graph_phase_structural_edges(
                merged, {}, "doc-1", "run-1", tracker,
            )

        # No create_structural_edge_sync calls
        assert mock_store.create_structural_edge_sync.call_count == 0
        # Tracker unchanged (still True from before)
        assert tracker.any_mutation_attempted is True

    def test_writes_derived_edges_and_marks_tracker(self):
        """Non-empty derived list → each edge creates a call + marks tracker."""
        from app.workers.pipeline import (
            _import_graph_phase_structural_edges,
            GraphWriteTracker,
        )
        from app.services.extraction_merge import DerivedEdge

        merged = _fake_merged()
        tracker = GraphWriteTracker()  # not yet marked

        derived = [
            DerivedEdge(from_id="#10:1", to_id="#5:1", rel_type="MENTIONED_IN", confidence=0.9),
            DerivedEdge(from_id="#10:1", to_id="#5:2", rel_type="MENTIONED_IN", confidence=0.8),
        ]

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store), \
             patch("app.workers.pipeline._load_chunks_for_derivation", return_value=[]), \
             patch("app.workers.pipeline._get_structural_document_rid", return_value="#1:0"), \
             patch("ontology_bundles.air_defense_v3.derive_rules.derive_structural_edges", return_value=derived):
            _import_graph_phase_structural_edges(
                merged, {}, "doc-1", "run-1", tracker,
            )

        assert tracker.any_mutation_attempted is True
        assert mock_store.create_structural_edge_sync.call_count == 2
        # Each call carries the document_id + pipeline_run_id + source='derive_rules' in properties
        for call in mock_store.create_structural_edge_sync.call_args_list:
            props = call.kwargs["properties"]
            assert props["document_id"] == "doc-1"
            assert props["pipeline_run_id"] == "run-1"
            assert props["source"] == "derive_rules"

    def test_tracker_stays_false_for_empty_derived_without_prior_mark(self):
        """Empty derived list with a fresh tracker leaves tracker False (true no-op)."""
        from app.workers.pipeline import (
            _import_graph_phase_structural_edges,
            GraphWriteTracker,
        )

        merged = _fake_merged()
        tracker = GraphWriteTracker()  # not yet marked
        assert tracker.any_mutation_attempted is False

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store), \
             patch("app.workers.pipeline._load_chunks_for_derivation", return_value=[]), \
             patch("app.workers.pipeline._get_structural_document_rid", return_value="#1:0"), \
             patch("ontology_bundles.air_defense_v3.derive_rules.derive_structural_edges", return_value=[]):
            _import_graph_phase_structural_edges(
                merged, {}, "doc-1", "run-1", tracker,
            )

        # True no-op: tracker stays False
        assert tracker.any_mutation_attempted is False
        assert mock_store.create_structural_edge_sync.call_count == 0
