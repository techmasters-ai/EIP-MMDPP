"""Unit tests for FastAPI lifespan bootstrap.

Tests that startup calls ensure_collections,
and that failures propagate (no silent swallow).
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


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
    @patch("app.db.session.get_qdrant_client")
    @patch("app.services.qdrant_store.ensure_collections")
    async def test_lifespan_calls_ensure_collections(
        self, mock_ensure_qdrant, mock_get_qdrant,
    ):
        from app.main import lifespan

        mock_app = MagicMock()
        async with lifespan(mock_app):
            pass

        mock_get_qdrant.assert_called_once()
        mock_ensure_qdrant.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.db.session.get_qdrant_client")
    @patch("app.services.qdrant_store.ensure_collections", side_effect=Exception("Qdrant down"))
    async def test_bootstrap_failure_raises(
        self, mock_ensure_qdrant, mock_get_qdrant,
    ):
        from app.main import lifespan

        mock_app = MagicMock()
        with pytest.raises(Exception, match="Qdrant down"):
            async with lifespan(mock_app):
                pass
