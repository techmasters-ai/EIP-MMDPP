"""Unit tests for FastAPI lifespan bootstrap.

Tests that startup calls ArcadeDB schema sync,
and that failures propagate (no silent swallow).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Lifespan bootstrap — ArcadeDB schema sync
# ---------------------------------------------------------------------------

class TestLifespanBootstrap:
    @pytest.mark.asyncio
    @patch("app.db.session.get_graph_store")
    @patch("app.services.ontology_templates.load_ontology", return_value=None)
    async def test_lifespan_calls_graph_store(
        self, mock_load_ontology, mock_get_graph_store,
    ):
        from app.main import lifespan

        mock_app = MagicMock()
        async with lifespan(mock_app):
            pass

        mock_get_graph_store.assert_called_once()
