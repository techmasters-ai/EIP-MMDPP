"""Fix N — per-run retraction of the document's DOMAIN relationship edges.

``derive_ontology_graph_merge`` must AUTOMATICALLY retract this document's prior
domain-relationship edges every run, BEFORE re-committing the run's domain edges,
so the doc's domain-edge population after any run == exactly what THIS run
extracted (every edge lineage-bearing), with no manual purge. This is required so
narrowed re-runs (emitting different edge subsets) don't re-accumulate stale global
edges that break the doc-scoped fail-closed lineage gate.

These tests cover:
  * ``_domain_relationship_edge_types(ontology)`` — gate-aligned enumeration of the
    24 air_defense_v3 domain edge types (structural rels excluded).
  * ANTI-DRIFT: the retract's structural exclusion set == the gate's
    ``STRUCTURAL_EDGE_TYPES`` (imported from scripts/verify_lineage_e2e.py).
  * ``ArcadeDBGraphStore.retract_document_domain_edges_sync`` — the two-statement
    remove-UPDATE + size()=0-DELETE idiom per domain edge type; structural /
    HAS_PROVENANCE / EXTRACTED_FROM untouched; per-statement try/except isolation.
  * ``_retract_document_domain_edges`` pipeline wrapper — tracker semantics,
    empty-set no-op, type-list-build-raises-before-mark.

Mirrors tests/unit/test_graph_import_phases.py (MagicMock store + patched
get_graph_store).
"""
from unittest.mock import MagicMock, patch

import pytest

import scripts.verify_lineage_e2e as gate
from app.services.ontology_templates import load_ontology


# The 24 air_defense_v3 DOMAIN relationship edge types (every ontology rel type
# MINUS the 8 structural rels CHILD_OF, CONTAINS, HAS_FIGURE, HAS_IMAGE,
# HAS_SECTION, HAS_TABLE, MENTIONED_IN, NEAR_TEXT). Empirically confirmed.
_EXPECTED_DOMAIN_24 = {
    "ALIAS_OF", "ASSOCIATED_WITH", "CUES", "DEFENDS", "DEPLOYED_ON",
    "DERIVED_FROM", "DESIGNATES", "DETECTS", "ENGAGES", "GUIDES",
    "HAS_COMPONENT", "HAS_SUBSYSTEM", "INSTALLED_ON", "INSTANCE_OF", "IS_A",
    "LAUNCHES", "MANUFACTURED_BY", "OPERATED_BY", "PART_OF", "REVIEWED_BY",
    "SUPERSEDES", "SUPPORTS_ENGAGEMENT_OF", "TRACKS", "VARIANT_OF",
}


# --- _domain_relationship_edge_types ----------------------------------------

class TestDomainRelationshipEdgeTypes:
    def test_air_defense_v3_yields_exactly_the_24_domain_types(self):
        from app.workers.pipeline import _domain_relationship_edge_types

        ontology = load_ontology(bundle_key="air_defense_v3")
        types = _domain_relationship_edge_types(ontology)

        assert len(types) == 24
        assert set(types) == _EXPECTED_DOMAIN_24

    def test_includes_keystone_domain_rels(self):
        """VARIANT_OF / ASSOCIATED_WITH / CUES must be present (gate-aligned)."""
        from app.workers.pipeline import _domain_relationship_edge_types

        types = set(_domain_relationship_edge_types(load_ontology(bundle_key="air_defense_v3")))
        assert {"VARIANT_OF", "ASSOCIATED_WITH", "CUES"} <= types

    def test_excludes_all_structural_and_anchor_edge_types(self):
        """No structural rel survives: the schema _STRUCTURAL_EDGE_TYPES, the 4
        anchor/derive extras, and the doc-anchor subset CHILD_OF / HAS_SECTION /
        HAS_FIGURE / HAS_TABLE are all absent from the domain set."""
        from app.workers.pipeline import _domain_relationship_edge_types
        from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES

        types = set(_domain_relationship_edge_types(load_ontology(bundle_key="air_defense_v3")))

        for s in _STRUCTURAL_EDGE_TYPES:
            assert s not in types, f"structural edge {s} leaked into domain set"
        for anchor in ("HAS_IMAGE", "NEAR_TEXT", "CONTAINS", "MENTIONED_IN"):
            assert anchor not in types
        for doc_anchor in ("CHILD_OF", "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE"):
            assert doc_anchor not in types

    def test_empty_ontology_yields_empty_list(self):
        from app.workers.pipeline import _domain_relationship_edge_types

        assert _domain_relationship_edge_types({}) == []
        assert _domain_relationship_edge_types({"relationship_types": []}) == []


# --- ANTI-DRIFT: retract exclusion set == gate STRUCTURAL_EDGE_TYPES ---------

class TestAntiDrift:
    def test_retract_structural_exclusion_matches_gate(self):
        """The structural set the retract subtracts MUST equal the gate's
        STRUCTURAL_EDGE_TYPES — they are built from the same shared schema
        constants, so 'what is a domain edge' can never diverge between the
        per-run retract and the doc-scoped fail-closed gate."""
        from app.services.arcadedb_schema import (
            _STRUCTURAL_EDGE_TYPES,
            _ANCHOR_DERIVE_STRUCTURAL_EDGE_TYPES,
        )

        retract_structural = set(_STRUCTURAL_EDGE_TYPES) | set(
            _ANCHOR_DERIVE_STRUCTURAL_EDGE_TYPES
        )
        assert retract_structural == set(gate.STRUCTURAL_EDGE_TYPES)

    def test_gate_consumes_shared_anchor_constant(self):
        """The gate's _ANCHOR_STRUCTURAL must BE the shared schema constant
        (consumed, not a divergent local literal)."""
        from app.services.arcadedb_schema import _ANCHOR_DERIVE_STRUCTURAL_EDGE_TYPES

        assert set(gate._ANCHOR_STRUCTURAL) == set(_ANCHOR_DERIVE_STRUCTURAL_EDGE_TYPES)

    def test_domain_set_disjoint_from_gate_structural(self):
        """No domain edge type the retract touches is in the gate's structural
        exclusion set (the two partitions don't overlap)."""
        from app.workers.pipeline import _domain_relationship_edge_types

        domain = set(_domain_relationship_edge_types(load_ontology(bundle_key="air_defense_v3")))
        assert domain.isdisjoint(set(gate.STRUCTURAL_EDGE_TYPES))


# --- ArcadeDBGraphStore.retract_document_domain_edges_sync -------------------

def _make_store(command_sync_result=None):
    """Build an ArcadeDBGraphStore over a MagicMock client (mirrors
    tests/unit/test_arcadedb_graph.py::_make_client)."""
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = MagicMock()
    client.command_sync = MagicMock(return_value=command_sync_result or [{"count": 0}])
    return ArcadeDBGraphStore(client=client, database="testdb"), client


class TestRetractStoreMethod:
    def test_issues_two_statements_per_domain_type(self):
        """Each domain edge type gets the remove-UPDATE + size()=0-DELETE pair,
        both parameterized with {"doc_id": document_id}."""
        store, client = _make_store()
        types = ["ASSOCIATED_WITH", "VARIANT_OF", "CUES"]

        count = store.retract_document_domain_edges_sync("doc-9", types)

        # 2 statements per type, all executed.
        assert client.command_sync.call_count == 2 * len(types)
        assert count == 2 * len(types)

        calls = client.command_sync.call_args_list
        sqls = [c.args[2] for c in calls]

        for etype in types:
            update_stmt = next(
                s for s in sqls
                if s.startswith(f"UPDATE {etype} ") and "document_ids.remove(:doc_id)" in s
            )
            assert "WHERE document_ids CONTAINS :doc_id" in update_stmt
            delete_stmt = next(
                s for s in sqls
                if s.startswith(f"DELETE FROM {etype} ")
                and "document_ids.size() = 0" in s
            )
            assert "document_ids IS NOT NULL" in delete_stmt

        # every statement carries the {"doc_id": document_id} param
        assert all(c.args[3] == {"doc_id": "doc-9"} for c in calls)

    def test_never_touches_structural_provenance_or_extracted_from(self):
        """No statement references EXTRACTED_FROM / HAS_PROVENANCE /
        CONTAINS_TEXT / MENTIONED_IN / HAS_IMAGE / NEAR_TEXT — only the domain
        edge classes the caller passed."""
        store, client = _make_store()
        types = ["ASSOCIATED_WITH", "CUES"]

        store.retract_document_domain_edges_sync("doc-1", types)

        joined = " ".join(c.args[2] for c in client.command_sync.call_args_list)
        for forbidden in (
            "EXTRACTED_FROM", "HAS_PROVENANCE", "CONTAINS_TEXT",
            "MENTIONED_IN", "HAS_IMAGE", "NEAR_TEXT", "CONTAINS_IMAGE",
        ):
            assert forbidden not in joined, f"retract touched {forbidden}"

    def test_empty_type_list_issues_no_commands(self):
        store, client = _make_store()
        count = store.retract_document_domain_edges_sync("doc-1", [])
        assert count == 0
        assert client.command_sync.call_count == 0

    def test_missing_edge_class_is_noop_not_crash(self):
        """A command_sync raising for one type (e.g. class absent on a fresh DB)
        must not abort the others — they are still processed."""
        store, client = _make_store()

        def _side_effect(database, language, sql, params=None):
            if "VARIANT_OF" in sql:
                raise RuntimeError("class VARIANT_OF not found")
            return [{"count": 0}]

        client.command_sync.side_effect = _side_effect

        # Does not raise; the two VARIANT_OF statements fail but ASSOCIATED_WITH
        # and CUES (2 statements each) still execute.
        count = store.retract_document_domain_edges_sync(
            "doc-1", ["ASSOCIATED_WITH", "VARIANT_OF", "CUES"]
        )
        # 6 attempted, 4 succeeded (VARIANT_OF's 2 raised).
        assert client.command_sync.call_count == 6
        assert count == 4


# --- _retract_document_domain_edges (pipeline wrapper) ----------------------

def _ontology_with_rels(*rel_names):
    return {"relationship_types": [{"name": n} for n in rel_names]}


class TestRetractPipelineWrapper:
    def test_marks_tracker_then_calls_store(self):
        """Non-empty domain set → tracker.mark() then
        retract_document_domain_edges_sync with the computed type list."""
        from app.workers.pipeline import (
            _retract_document_domain_edges,
            GraphWriteTracker,
        )

        merged = MagicMock()
        tracker = GraphWriteTracker()
        ontology = load_ontology(bundle_key="air_defense_v3")

        mock_store = MagicMock()
        mock_store.retract_document_domain_edges_sync.return_value = 48
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            n = _retract_document_domain_edges(merged, ontology, "doc-1", tracker)

        assert tracker.any_mutation_attempted is True
        assert mock_store.retract_document_domain_edges_sync.call_count == 1
        call = mock_store.retract_document_domain_edges_sync.call_args
        assert call.args[0] == "doc-1"
        passed_types = call.args[1]
        assert set(passed_types) == _EXPECTED_DOMAIN_24
        assert n == 48

    def test_retracts_types_not_emitted_this_run(self):
        """Ontology-driven, NOT emit-driven: even when merged carries only
        ASSOCIATED_WITH edges, the retract still issues VARIANT_OF (and the rest
        of the 24)."""
        from app.workers.pipeline import (
            _retract_document_domain_edges,
            GraphWriteTracker,
        )

        # merged has only ASSOCIATED_WITH edges; retract must NOT narrow to that.
        merged = MagicMock()
        merged.edges = [MagicMock(rel_type="ASSOCIATED_WITH")]
        tracker = GraphWriteTracker()
        ontology = load_ontology(bundle_key="air_defense_v3")

        mock_store = MagicMock()
        mock_store.retract_document_domain_edges_sync.return_value = 0
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            _retract_document_domain_edges(merged, ontology, "doc-1", tracker)

        passed_types = set(
            mock_store.retract_document_domain_edges_sync.call_args.args[1]
        )
        assert "VARIANT_OF" in passed_types
        assert passed_types == _EXPECTED_DOMAIN_24

    def test_empty_domain_set_is_noop_and_does_not_mark_tracker(self):
        """No domain edge types → zero store calls, tracker untouched, return 0."""
        from app.workers.pipeline import (
            _retract_document_domain_edges,
            GraphWriteTracker,
        )

        merged = MagicMock()
        tracker = GraphWriteTracker()
        assert tracker.any_mutation_attempted is False

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store):
            n = _retract_document_domain_edges(merged, {}, "doc-1", tracker)

        assert n == 0
        assert tracker.any_mutation_attempted is False
        assert mock_store.retract_document_domain_edges_sync.call_count == 0

    def test_tracker_not_marked_if_type_build_raises_before_first_command(self):
        """If the type-list build raises BEFORE the first command, tracker stays
        False (no mutation attempted, rollback-free)."""
        from app.workers.pipeline import (
            _retract_document_domain_edges,
            GraphWriteTracker,
        )

        merged = MagicMock()
        tracker = GraphWriteTracker()
        ontology = load_ontology(bundle_key="air_defense_v3")

        mock_store = MagicMock()
        with patch("app.workers.pipeline.get_graph_store", return_value=mock_store), \
             patch(
                 "app.workers.pipeline._domain_relationship_edge_types",
                 side_effect=ValueError("boom"),
             ):
            with pytest.raises(ValueError):
                _retract_document_domain_edges(merged, ontology, "doc-1", tracker)

        assert tracker.any_mutation_attempted is False
        assert mock_store.retract_document_domain_edges_sync.call_count == 0
