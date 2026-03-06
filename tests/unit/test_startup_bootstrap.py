"""Unit tests for FastAPI lifespan bootstrap and Neo4j ensure_indexes.

Tests that startup calls ensure_collections and ensure_indexes,
and that failures propagate (no silent swallow).
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ensure_indexes
# ---------------------------------------------------------------------------

class TestEnsureIndexes:
    def test_runs_create_queries(self):
        from app.services.neo4j_graph import ensure_indexes

        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        ensure_indexes(driver)

        # Should have run 3 statements (fulltext index, constraint, index)
        assert session.run.call_count == 3
        calls = [c.args[0] for c in session.run.call_args_list]
        assert any("entity_name_fulltext" in c for c in calls)
        assert any("document_id_unique" in c for c in calls)
        assert any("chunk_ref_chunk_id" in c for c in calls)

    def test_exception_propagates(self):
        from app.services.neo4j_graph import ensure_indexes

        driver = MagicMock()
        session = MagicMock()
        session.run.side_effect = Exception("Neo4j unavailable")
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(Exception, match="Neo4j unavailable"):
            ensure_indexes(driver)

    def test_idempotent(self):
        """Calling ensure_indexes twice doesn't error (IF NOT EXISTS)."""
        from app.services.neo4j_graph import ensure_indexes

        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        ensure_indexes(driver)
        ensure_indexes(driver)

        # 3 statements × 2 calls = 6
        assert session.run.call_count == 6


# ---------------------------------------------------------------------------
# Lifespan bootstrap
# ---------------------------------------------------------------------------

try:
    import qdrant_client  # noqa: F401
    _has_qdrant = True
except ImportError:
    _has_qdrant = False


@pytest.mark.skipif(not _has_qdrant, reason="qdrant_client not installed")
class TestLifespanBootstrap:
    @pytest.mark.asyncio
    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.ensure_indexes")
    @patch("app.db.session.get_qdrant_client")
    @patch("app.services.qdrant_store.ensure_collections")
    async def test_lifespan_calls_ensure_collections(
        self, mock_ensure_qdrant, mock_get_qdrant, mock_ensure_neo4j, mock_get_neo4j,
    ):
        from app.main import lifespan

        mock_app = MagicMock()
        async with lifespan(mock_app):
            pass

        mock_get_qdrant.assert_called_once()
        mock_ensure_qdrant.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.ensure_indexes")
    @patch("app.db.session.get_qdrant_client")
    @patch("app.services.qdrant_store.ensure_collections")
    async def test_lifespan_calls_ensure_indexes(
        self, mock_ensure_qdrant, mock_get_qdrant, mock_ensure_neo4j, mock_get_neo4j,
    ):
        from app.main import lifespan

        mock_app = MagicMock()
        async with lifespan(mock_app):
            pass

        mock_get_neo4j.assert_called_once()
        mock_ensure_neo4j.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.ensure_indexes")
    @patch("app.db.session.get_qdrant_client")
    @patch("app.services.qdrant_store.ensure_collections", side_effect=Exception("Qdrant down"))
    async def test_bootstrap_failure_raises(
        self, mock_ensure_qdrant, mock_get_qdrant, mock_ensure_neo4j, mock_get_neo4j,
    ):
        from app.main import lifespan

        mock_app = MagicMock()
        with pytest.raises(Exception, match="Qdrant down"):
            async with lifespan(mock_app):
                pass
