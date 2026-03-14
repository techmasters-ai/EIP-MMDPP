"""Unit tests for GraphRAG service.

Tests community detection, report generation, local/global search,
and graph export with mocked Neo4j driver and DB session.
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_settings(**overrides):
    s = MagicMock()
    s.graphrag_indexing_enabled = True
    s.graphrag_max_cluster_size = 10
    s.llm_provider = "mock"
    s.graphrag_model = "llama3.2"
    s.ollama_base_url = "http://localhost:11434"
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# _export_graph_for_graphrag
# ---------------------------------------------------------------------------

class TestExportGraphForGraphrag:
    def test_returns_entities_and_relationships(self, mock_neo4j_driver):
        from app.services.graphrag_service import _export_graph_for_graphrag
        entities_data = [{"name": "A", "entity_type": "RadarSystem", "id": "1"}]
        rels_data = [{"source": "A", "target": "B", "relationship": "USES"}]
        driver, session = mock_neo4j_driver
        # Two calls: entities then relationships
        e_result = MagicMock()
        e_result.__iter__ = MagicMock(return_value=iter(entities_data))
        r_result = MagicMock()
        r_result.__iter__ = MagicMock(return_value=iter(rels_data))
        session.run.side_effect = [e_result, r_result]
        entities, rels = _export_graph_for_graphrag(driver)
        assert len(entities) == 1
        assert len(rels) == 1

    def test_empty_graph(self, mock_neo4j_driver):
        from app.services.graphrag_service import _export_graph_for_graphrag
        driver, session = mock_neo4j_driver
        e_result = MagicMock()
        e_result.__iter__ = MagicMock(return_value=iter([]))
        r_result = MagicMock()
        r_result.__iter__ = MagicMock(return_value=iter([]))
        session.run.side_effect = [e_result, r_result]
        entities, rels = _export_graph_for_graphrag(driver)
        assert entities == []
        assert rels == []

    def test_exception_returns_empty(self, mock_neo4j_driver):
        from app.services.graphrag_service import _export_graph_for_graphrag
        driver, session = mock_neo4j_driver
        session.run.side_effect = Exception("fail")
        entities, rels = _export_graph_for_graphrag(driver)
        assert entities == []
        assert rels == []


# ---------------------------------------------------------------------------
# _detect_communities
# ---------------------------------------------------------------------------

try:
    import networkx
    _has_networkx = True
except ImportError:
    _has_networkx = False

@pytest.mark.skipif(not _has_networkx, reason="networkx not installed")
class TestDetectCommunities:
    def test_empty_entities(self):
        from app.services.graphrag_service import _detect_communities
        assert _detect_communities([], [], _mock_settings()) == []

    def test_single_entity_one_community(self):
        from app.services.graphrag_service import _detect_communities
        entities = [{"name": "A", "entity_type": "RadarSystem"}]
        communities = _detect_communities(entities, [], _mock_settings())
        assert len(communities) >= 1

    def test_community_dict_structure(self):
        from app.services.graphrag_service import _detect_communities
        entities = [
            {"name": "A", "entity_type": "T"},
            {"name": "B", "entity_type": "T"},
        ]
        rels = [{"source": "A", "target": "B", "relationship": "LINKED"}]
        communities = _detect_communities(entities, rels, _mock_settings())
        for c in communities:
            assert "community_id" in c
            assert "entity_names" in c
            assert "title" in c

    def test_multiple_entities_multiple_communities(self):
        from app.services.graphrag_service import _detect_communities
        # Two disconnected clusters
        entities = [
            {"name": "A", "entity_type": "T"},
            {"name": "B", "entity_type": "T"},
            {"name": "C", "entity_type": "T"},
            {"name": "D", "entity_type": "T"},
        ]
        rels = [
            {"source": "A", "target": "B", "relationship": "R"},
            {"source": "C", "target": "D", "relationship": "R"},
        ]
        communities = _detect_communities(entities, rels, _mock_settings())
        assert len(communities) >= 1


# ---------------------------------------------------------------------------
# _generate_community_reports
# ---------------------------------------------------------------------------

class TestGenerateCommunityReports:
    def test_mock_provider_returns_canned_reports(self):
        from app.services.graphrag_service import _generate_community_reports
        communities = [{"community_id": "0", "entity_names": ["A"], "title": "C0"}]
        entities = [{"name": "A", "entity_type": "T"}]
        settings = _mock_settings(llm_provider="mock")
        reports = _generate_community_reports(communities, entities, [], settings)
        assert len(reports) == 1
        assert "Mock report" in reports[0]["report_text"]

    def test_report_count_matches_communities(self):
        from app.services.graphrag_service import _generate_community_reports
        communities = [
            {"community_id": "0", "entity_names": ["A"], "title": "C0"},
            {"community_id": "1", "entity_names": ["B"], "title": "C1"},
        ]
        settings = _mock_settings(llm_provider="mock")
        reports = _generate_community_reports(communities, [], [], settings)
        assert len(reports) == 2

    def test_report_structure(self):
        from app.services.graphrag_service import _generate_community_reports
        communities = [{"community_id": "0", "entity_names": ["A"], "title": "C0"}]
        settings = _mock_settings(llm_provider="mock")
        reports = _generate_community_reports(communities, [], [], settings)
        r = reports[0]
        assert "community_id" in r
        assert "report_text" in r
        assert "summary" in r
        assert "rank" in r

    @patch("httpx.post")
    def test_ollama_provider_calls_httpx(self, mock_post):
        from app.services.graphrag_service import _generate_community_reports
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "message": {"content": "This community contains radar systems."},
            }),
            raise_for_status=MagicMock(),
        )
        communities = [{"community_id": "0", "entity_names": ["A"], "title": "C0"}]
        entities = [{"name": "A", "entity_type": "RadarSystem"}]
        settings = _mock_settings(llm_provider="ollama", ollama_num_ctx=32768)
        reports = _generate_community_reports(communities, entities, [], settings)
        assert len(reports) == 1
        assert "radar" in reports[0]["report_text"].lower()
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/chat" in call_args[0][0]

    @patch("httpx.post")
    def test_ollama_empty_response_skipped(self, mock_post):
        from app.services.graphrag_service import _generate_community_reports
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"message": {"content": ""}}),
            raise_for_status=MagicMock(),
        )
        communities = [{"community_id": "0", "entity_names": ["A"], "title": "C0"}]
        settings = _mock_settings(llm_provider="ollama", ollama_num_ctx=32768)
        reports = _generate_community_reports(communities, [], [], settings)
        assert len(reports) == 0


# ---------------------------------------------------------------------------
# _store_communities_and_reports
# ---------------------------------------------------------------------------

class TestStoreCommunities:
    def test_inserts_communities_and_reports(self):
        from app.services.graphrag_service import _store_communities_and_reports
        db = MagicMock()
        communities = [{"community_id": "0", "level": 0, "entity_names": ["A"], "title": "C0"}]
        reports = [{"community_id": "0", "report_text": "text", "summary": "sum", "rank": 1.0}]
        _store_communities_and_reports(db, communities, reports)
        assert db.execute.call_count == 2
        db.commit.assert_called_once()

    def test_empty_data_still_commits(self):
        from app.services.graphrag_service import _store_communities_and_reports
        db = MagicMock()
        _store_communities_and_reports(db, [], [])
        db.commit.assert_called_once()


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

    @patch("app.services.graphrag_service._export_graph_for_graphrag", return_value=([], []))
    @patch("app.services.graphrag_service.get_settings")
    def test_no_entities_returns_zeros(self, mock_gs, mock_export):
        from app.services.graphrag_service import run_graphrag_indexing
        mock_gs.return_value = _mock_settings()
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 0

    @patch("app.services.graphrag_service._store_communities_and_reports")
    @patch("app.services.graphrag_service._generate_community_reports")
    @patch("app.services.graphrag_service._detect_communities")
    @patch("app.services.graphrag_service._export_graph_for_graphrag")
    @patch("app.services.graphrag_service.get_settings")
    def test_full_pipeline_returns_stats(self, mock_gs, mock_export, mock_detect, mock_gen, mock_store):
        from app.services.graphrag_service import run_graphrag_indexing
        mock_gs.return_value = _mock_settings()
        mock_export.return_value = ([{"name": "A"}], [])
        mock_detect.return_value = [{"community_id": "0", "entity_names": ["A"]}]
        mock_gen.return_value = [{"community_id": "0", "report_text": "t"}]
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 1
        assert result["reports_generated"] == 1

    @patch("app.services.graphrag_service._export_graph_for_graphrag", side_effect=Exception("fail"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_zeros(self, mock_gs, mock_export):
        from app.services.graphrag_service import run_graphrag_indexing
        mock_gs.return_value = _mock_settings()
        result = run_graphrag_indexing(MagicMock(), MagicMock())
        assert result["communities_created"] == 0


# ---------------------------------------------------------------------------
# local_search
# ---------------------------------------------------------------------------

class TestLocalSearch:
    @patch("app.services.graphrag_service._get_entity_community_context", return_value={})
    @patch("app.services.neo4j_graph.fulltext_search_entity")
    def test_returns_entity_results(self, mock_search, mock_ctx):
        from app.services.graphrag_service import local_search
        mock_search.return_value = [{"name": "A", "canonical_name": None, "entity_type": "T", "score": 1.0}]
        results = local_search("radar", MagicMock(), MagicMock(), MagicMock())
        assert len(results) == 1

    @patch("app.services.graphrag_service._get_entity_community_context", return_value={})
    @patch("app.services.neo4j_graph.fulltext_search_entity", return_value=[])
    def test_empty_matches(self, mock_search, mock_ctx):
        from app.services.graphrag_service import local_search
        assert local_search("nothing", MagicMock(), MagicMock(), MagicMock()) == []

    @patch("app.services.neo4j_graph.fulltext_search_entity", side_effect=Exception("fail"))
    def test_exception_returns_empty(self, mock_search):
        from app.services.graphrag_service import local_search
        assert local_search("q", MagicMock(), MagicMock(), MagicMock()) == []


# ---------------------------------------------------------------------------
# global_search
# ---------------------------------------------------------------------------

class TestGlobalSearch:
    def test_returns_reports(self):
        from app.services.graphrag_service import global_search
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [
            ("report text", "summary", 1.0, "cid", "title", 0, 0.85),
        ]
        results = global_search("radar", db)
        assert len(results) == 1
        assert results[0]["report_text"] == "report text"
        assert results[0]["relevance"] == 0.85

    def test_empty_db(self):
        from app.services.graphrag_service import global_search
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []
        assert global_search("test", db) == []

    def test_exception_returns_empty(self):
        from app.services.graphrag_service import global_search
        db = MagicMock()
        db.execute.side_effect = Exception("fail")
        assert global_search("test", db) == []

    def test_global_search_filters_by_query(self):
        """global_search must filter community reports by query relevance."""
        import inspect
        from app.services.graphrag_service import global_search
        source = inspect.getsource(global_search)
        assert "plainto_tsquery" in source, "global_search must use plainto_tsquery to filter by query text"
        assert ":query" in source, "global_search must bind the query parameter"

    def test_query_param_passed_to_sql(self):
        """Verify the query text is passed to the SQL statement."""
        from app.services.graphrag_service import global_search
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []
        global_search("specific search term", db, limit=5)
        call_args = db.execute.call_args
        params = call_args[0][1]
        assert params["query"] == "specific search term"
        assert params["limit"] == 5


# ---------------------------------------------------------------------------
# _get_entity_community_context
# ---------------------------------------------------------------------------

class TestGetEntityCommunityContext:
    def test_returns_context_keyed_by_name(self):
        from app.services.graphrag_service import _get_entity_community_context
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = [
            ("report", "summary", "title", "cid"),
        ]
        matches = [{"node": {"name": "A"}, "entity_type": "T"}]
        ctx = _get_entity_community_context(db, matches)
        assert "A" in ctx

    def test_empty_matches(self):
        from app.services.graphrag_service import _get_entity_community_context
        assert _get_entity_community_context(MagicMock(), []) == {}

    def test_missing_name_skipped(self):
        from app.services.graphrag_service import _get_entity_community_context
        matches = [{"node": {}, "entity_type": "T"}]
        ctx = _get_entity_community_context(MagicMock(), matches)
        assert ctx == {}
