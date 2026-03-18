"""Unit tests for Microsoft GraphRAG service (indexer + searcher)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _mock_settings(**overrides):
    s = MagicMock()
    s.graphrag_indexing_enabled = True
    s.graphrag_llm_provider = "ollama"
    s.graphrag_llm_model = "llama3.2"
    s.graphrag_llm_api_base = "http://ollama:11434/v1"
    s.graphrag_api_key = ""
    s.graphrag_embedding_model = "nomic-embed-text"
    s.graphrag_data_dir = "/tmp/test_graphrag"
    s.graphrag_community_level = 2
    s.graphrag_max_cluster_size = 10
    s.graphrag_tune_interval_minutes = 1440
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# run_graphrag_indexing
# ---------------------------------------------------------------------------


class TestRunGraphragIndexing:
    @patch("app.services.graphrag_service.get_settings")
    def test_disabled_returns_zeros(self, mock_gs):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings(graphrag_indexing_enabled=False)
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result == {"communities_created": 0, "reports_generated": 0}

    @patch("app.services.graphrag_service.export_all", return_value={"entities": 0})
    @patch("app.services.graphrag_service.get_settings")
    def test_empty_graph_returns_zeros(self, mock_gs, mock_export):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings()
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 0

    @patch("app.services.graphrag_service._run_graphrag_pipeline")
    @patch("app.services.graphrag_service.export_all")
    @patch("app.services.graphrag_service.get_settings")
    def test_full_pipeline_returns_stats(self, mock_gs, mock_export, mock_pipeline):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings()
        mock_export.return_value = {"entities": 5, "relationships": 3}
        mock_pipeline.return_value = {"communities_created": 2, "reports_generated": 2}

        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 2
        assert result["reports_generated"] == 2

    @patch("app.services.graphrag_service.export_all", side_effect=Exception("fail"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_zeros(self, mock_gs, mock_export):
        from app.services.graphrag_service import run_graphrag_indexing

        mock_gs.return_value = _mock_settings()
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 0


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------


def _make_mock_search_data():
    """Create minimal mock search data dict."""
    return {
        "entities": pd.DataFrame(columns=["id", "title", "type", "description"]),
        "communities": pd.DataFrame(columns=["id", "title", "level"]),
        "community_reports": pd.DataFrame(
            columns=["id", "community_id", "full_content"]
        ),
        "text_units": pd.DataFrame(columns=["id", "text"]),
        "relationships": pd.DataFrame(columns=["id", "source", "target"]),
    }


class TestLocalSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import local_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_local_search") as mock_run:
            mock_run.return_value = ("Answer text", {"entities": []})
            result = local_search("S-400 capabilities")

        assert result["response"] == "Answer text"

    @patch(
        "app.services.graphrag_service._load_search_data",
        side_effect=Exception("no data"),
    )
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_empty(self, mock_gs, mock_load):
        from app.services.graphrag_service import local_search

        mock_gs.return_value = _mock_settings()
        result = local_search("query")
        assert result["response"] == ""
        assert result["context"] == {}


class TestGlobalSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import global_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_global_search") as mock_run:
            mock_run.return_value = ("Global answer", {"reports": []})
            result = global_search("broad question")

        assert result["response"] == "Global answer"


class TestDriftSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import drift_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_drift_search") as mock_run:
            mock_run.return_value = ("DRIFT answer", {"entities": []})
            result = drift_search("guidance methods")

        assert result["response"] == "DRIFT answer"


class TestBasicSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.get_settings")
    def test_returns_results(self, mock_gs, mock_load):
        from app.services.graphrag_service import basic_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = _make_mock_search_data()

        with patch("app.services.graphrag_service._run_basic_search") as mock_run:
            mock_run.return_value = ("Basic answer", {"chunks": []})
            result = basic_search("simple query")

        assert result["response"] == "Basic answer"
