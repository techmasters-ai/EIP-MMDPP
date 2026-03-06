"""Unit tests for entity canonicalization service.

Tests exact match, alias match, fuzzy match, and document-level
canonicalization with mocked Neo4j driver.
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_driver(run_return=None):
    """Return (driver, session) with session.run returning a mock result."""
    session = MagicMock()
    result = MagicMock()
    result.single.return_value = run_return
    result.__iter__ = MagicMock(return_value=iter(run_return if isinstance(run_return, list) else []))
    session.run.return_value = result

    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


# ---------------------------------------------------------------------------
# _exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_match_found(self):
        from app.services.canonicalization import _exact_match
        driver, _ = _mock_driver({"canonical_name": "AN/SPY-1"})
        assert _exact_match(driver, "SPY-1", "RadarSystem") == "AN/SPY-1"

    def test_no_match(self):
        from app.services.canonicalization import _exact_match
        driver, _ = _mock_driver(None)
        assert _exact_match(driver, "unknown", "RadarSystem") is None

    def test_exception_returns_none(self):
        from app.services.canonicalization import _exact_match
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert _exact_match(driver, "test", "T") is None


# ---------------------------------------------------------------------------
# _alias_match
# ---------------------------------------------------------------------------

class TestAliasMatch:
    def test_alias_found(self):
        from app.services.canonicalization import _alias_match
        driver, _ = _mock_driver({"canonical_name": "SA-11 Gadfly"})
        assert _alias_match(driver, "Buk-M1") == "SA-11 Gadfly"

    def test_no_alias(self):
        from app.services.canonicalization import _alias_match
        driver, _ = _mock_driver(None)
        assert _alias_match(driver, "unknown") is None

    def test_exception_returns_none(self):
        from app.services.canonicalization import _alias_match
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert _alias_match(driver, "test") is None


# ---------------------------------------------------------------------------
# _fuzzy_match
# ---------------------------------------------------------------------------

class TestFuzzyMatch:
    @patch("app.services.neo4j_graph.fulltext_search_entity")
    def test_high_score_match(self, mock_search):
        from app.services.canonicalization import _fuzzy_match
        mock_search.return_value = [
            {"name": "AN/SPY-1", "canonical_name": None, "entity_type": "RadarSystem", "score": 0.95}
        ]
        result = _fuzzy_match(MagicMock(), "SPY-1", "RadarSystem")
        assert result == "AN/SPY-1"

    @patch("app.services.neo4j_graph.fulltext_search_entity")
    def test_low_score_returns_none(self, mock_search):
        from app.services.canonicalization import _fuzzy_match
        mock_search.return_value = [
            {"name": "AN/SPY-1", "canonical_name": None, "entity_type": "RadarSystem", "score": 0.3}
        ]
        assert _fuzzy_match(MagicMock(), "SPY-1", "RadarSystem") is None

    @patch("app.services.neo4j_graph.fulltext_search_entity")
    def test_wrong_entity_type_returns_none(self, mock_search):
        from app.services.canonicalization import _fuzzy_match
        mock_search.return_value = [
            {"name": "AN/SPY-1", "canonical_name": None, "entity_type": "MissileSystem", "score": 0.95}
        ]
        assert _fuzzy_match(MagicMock(), "SPY-1", "RadarSystem") is None

    @patch("app.services.neo4j_graph.fulltext_search_entity")
    def test_same_name_skipped(self, mock_search):
        from app.services.canonicalization import _fuzzy_match
        mock_search.return_value = [
            {"name": "AN/SPY-1", "canonical_name": None, "entity_type": "RadarSystem", "score": 0.99}
        ]
        # When the candidate equals the query name, it should be skipped
        assert _fuzzy_match(MagicMock(), "AN/SPY-1", "RadarSystem") is None

    @patch("app.services.neo4j_graph.fulltext_search_entity")
    def test_uses_canonical_name_field(self, mock_search):
        from app.services.canonicalization import _fuzzy_match
        mock_search.return_value = [
            {"name": "spy1", "canonical_name": "AN/SPY-1", "entity_type": "RadarSystem", "score": 0.9}
        ]
        result = _fuzzy_match(MagicMock(), "SPY-1-variant", "RadarSystem")
        assert result == "AN/SPY-1"


# ---------------------------------------------------------------------------
# canonicalize_entity
# ---------------------------------------------------------------------------

class TestCanonicalizeEntity:
    @patch("app.services.canonicalization._fuzzy_match", return_value=None)
    @patch("app.services.canonicalization._alias_match", return_value=None)
    @patch("app.services.canonicalization._exact_match", return_value="CANONICAL")
    def test_exact_match_found(self, mock_exact, mock_alias, mock_fuzzy):
        from app.services.canonicalization import canonicalize_entity
        result = canonicalize_entity(MagicMock(), "test", "T")
        assert result == "CANONICAL"
        mock_alias.assert_not_called()
        mock_fuzzy.assert_not_called()

    @patch("app.services.canonicalization._fuzzy_match", return_value=None)
    @patch("app.services.canonicalization._alias_match", return_value="ALIAS_CANONICAL")
    @patch("app.services.canonicalization._exact_match", return_value=None)
    def test_alias_match_found(self, mock_exact, mock_alias, mock_fuzzy):
        from app.services.canonicalization import canonicalize_entity
        result = canonicalize_entity(MagicMock(), "test", "T")
        assert result == "ALIAS_CANONICAL"
        mock_fuzzy.assert_not_called()

    @patch("app.services.canonicalization._fuzzy_match", return_value="FUZZY_CANONICAL")
    @patch("app.services.canonicalization._alias_match", return_value=None)
    @patch("app.services.canonicalization._exact_match", return_value=None)
    def test_fuzzy_match_found(self, mock_exact, mock_alias, mock_fuzzy):
        from app.services.canonicalization import canonicalize_entity
        result = canonicalize_entity(MagicMock(), "test", "T")
        assert result == "FUZZY_CANONICAL"

    @patch("app.services.canonicalization._fuzzy_match", return_value=None)
    @patch("app.services.canonicalization._alias_match", return_value=None)
    @patch("app.services.canonicalization._exact_match", return_value=None)
    def test_no_match_returns_none(self, mock_exact, mock_alias, mock_fuzzy):
        from app.services.canonicalization import canonicalize_entity
        assert canonicalize_entity(MagicMock(), "test", "T") is None

    @patch("app.services.canonicalization._fuzzy_match", return_value="FUZZY")
    @patch("app.services.canonicalization._alias_match", return_value="ALIAS")
    @patch("app.services.canonicalization._exact_match", return_value="EXACT")
    def test_priority_exact_first(self, mock_exact, mock_alias, mock_fuzzy):
        from app.services.canonicalization import canonicalize_entity
        result = canonicalize_entity(MagicMock(), "test", "T")
        assert result == "EXACT"


# ---------------------------------------------------------------------------
# canonicalize_document_entities
# ---------------------------------------------------------------------------

class TestCanonicalizeDocumentEntities:
    @patch("app.services.canonicalization.canonicalize_entity")
    @patch("app.services.neo4j_graph.create_alias_edge")
    def test_returns_stats_dict(self, mock_alias, mock_canon):
        from app.services.canonicalization import canonicalize_document_entities
        entities = [{"name": "A", "entity_type": "T"}]
        driver, _ = _mock_driver(run_return=entities)
        mock_canon.return_value = "CANONICAL_A"
        mock_alias.return_value = True
        stats = canonicalize_document_entities(driver, "doc-1")
        assert "resolved" in stats
        assert "new_aliases" in stats
        assert "total" in stats
        assert stats["total"] == 1

    def test_no_entities_returns_zeros(self):
        from app.services.canonicalization import canonicalize_document_entities
        driver, _ = _mock_driver(run_return=[])
        stats = canonicalize_document_entities(driver, "doc-1")
        assert stats["total"] == 0
        assert stats["resolved"] == 0

    @patch("app.services.canonicalization.canonicalize_entity")
    @patch("app.services.neo4j_graph.create_alias_edge")
    def test_resolved_counted(self, mock_alias, mock_canon):
        from app.services.canonicalization import canonicalize_document_entities
        entities = [
            {"name": "A", "entity_type": "T"},
            {"name": "B", "entity_type": "T"},
        ]
        driver, _ = _mock_driver(run_return=entities)
        mock_canon.side_effect = ["CANONICAL_A", None]  # A resolved, B not
        mock_alias.return_value = True
        stats = canonicalize_document_entities(driver, "doc-1")
        assert stats["resolved"] == 1
        assert stats["total"] == 2

    @patch("app.services.canonicalization.canonicalize_entity")
    @patch("app.services.neo4j_graph.create_alias_edge")
    def test_alias_edges_created_for_resolved(self, mock_alias, mock_canon):
        from app.services.canonicalization import canonicalize_document_entities
        entities = [{"name": "A", "entity_type": "T"}]
        driver, _ = _mock_driver(run_return=entities)
        mock_canon.return_value = "CANONICAL_A"
        mock_alias.return_value = True
        canonicalize_document_entities(driver, "doc-1")
        mock_alias.assert_called_once()

    def test_exception_returns_empty_stats(self):
        from app.services.canonicalization import canonicalize_document_entities
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        stats = canonicalize_document_entities(driver, "doc-1")
        assert stats["total"] == 0


# ---------------------------------------------------------------------------
# _set_canonical_name
# ---------------------------------------------------------------------------

class TestSetCanonicalName:
    def test_runs_update_query(self):
        from app.services.canonicalization import _set_canonical_name
        driver, session = _mock_driver()
        _set_canonical_name(driver, "A", "T", "CANONICAL")
        session.run.assert_called_once()

    def test_exception_does_not_propagate(self):
        from app.services.canonicalization import _set_canonical_name
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        # Should not raise
        _set_canonical_name(driver, "A", "T", "C")
