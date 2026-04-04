"""Tests for the post-extraction validation pass."""
import os
import networkx as nx
import pytest
from unittest.mock import patch


class TestValidationPass:
    def test_validation_pass_enabled_by_default(self):
        from app.main import _should_run_validation_pass
        # Clear any env override
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DOCLING_GRAPH_VALIDATION_PASS_ENABLED", None)
            assert _should_run_validation_pass() is True

    def test_validation_pass_disabled_by_env(self):
        from app.main import _should_run_validation_pass
        with patch.dict(os.environ, {"DOCLING_GRAPH_VALIDATION_PASS_ENABLED": "false"}):
            assert _should_run_validation_pass() is False

    def test_apply_validation_edges_adds_edges(self):
        from app.main import _apply_validation_edges
        graph = nx.DiGraph()
        graph.add_node("RADAR_SYSTEM_Test", type="RADAR_SYSTEM", name="Test")
        graph.add_node("SPECIFICATION_Range_300", type="SPECIFICATION", name="Range")

        new_edges = [{
            "source": "RADAR_SYSTEM_Test",
            "target": "SPECIFICATION_Range_300",
            "label": "SPECIFIED_BY",
            "confidence": 0.85,
        }]

        added = _apply_validation_edges(graph, new_edges)
        assert added == 1
        assert graph.has_edge("RADAR_SYSTEM_Test", "SPECIFICATION_Range_300")

    def test_apply_validation_edges_skips_existing(self):
        from app.main import _apply_validation_edges
        graph = nx.DiGraph()
        graph.add_node("A", type="X")
        graph.add_node("B", type="Y")
        graph.add_edge("A", "B", label="EXISTS")

        new_edges = [{"source": "A", "target": "B", "label": "DUPLICATE"}]
        added = _apply_validation_edges(graph, new_edges)
        assert added == 0

    def test_apply_validation_edges_skips_missing_nodes(self):
        from app.main import _apply_validation_edges
        graph = nx.DiGraph()
        graph.add_node("A", type="X")

        new_edges = [{"source": "A", "target": "NONEXISTENT", "label": "REL"}]
        added = _apply_validation_edges(graph, new_edges)
        assert added == 0
