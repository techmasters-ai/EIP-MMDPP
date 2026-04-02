"""Unit tests for get_neighborhood_graph_async co-occurrence fallback."""
from unittest.mock import AsyncMock, MagicMock
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_async_driver():
    """Mock async Neo4j driver returning configurable query results."""
    session = AsyncMock()
    ctx_mgr = MagicMock()
    ctx_mgr.__aenter__ = AsyncMock(return_value=session)
    ctx_mgr.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = ctx_mgr
    return driver, session


class TestGetNeighborhoodGraphAsync:
    @pytest.mark.asyncio
    async def test_orphan_entity_triggers_cooccurrence(self, mock_async_driver):
        """When main query returns 0 rows, co-occurrence query runs."""
        from app.services.neo4j_graph import get_neighborhood_graph_async

        driver, session = mock_async_driver

        call_count = 0
        async def mock_run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            result = AsyncMock()
            if call_count == 1:
                # Main query: no rows (orphan entity)
                result.data = AsyncMock(return_value=[])
            elif call_count == 2:
                # Fallback center node query
                result.data = AsyncMock(return_value=[{
                    "props": {"id": "center-uuid", "name": "Max Range",
                              "entity_type": "SPECIFICATION"},
                    "entity_type": "SPECIFICATION",
                }])
            elif call_count == 3:
                # Co-occurrence query
                result.data = AsyncMock(return_value=[
                    {"other_props": {"id": "sys-uuid", "name": "SA-2 Guideline",
                                     "entity_type": "MISSILE_SYSTEM"},
                     "other_type": "MISSILE_SYSTEM"},
                ])
            return result

        session.run = mock_run

        result = await get_neighborhood_graph_async(driver, "Max Range")
        assert len(result["nodes"]) == 2  # center + co-occurring
        assert len(result["edges"]) == 1
        assert result["edges"][0]["rel_type"] == "CO_OCCURS_WITH"

    @pytest.mark.asyncio
    async def test_connected_entity_skips_cooccurrence(self, mock_async_driver):
        """When main query returns edges, co-occurrence is NOT run."""
        from app.services.neo4j_graph import get_neighborhood_graph_async

        driver, session = mock_async_driver

        call_count = 0
        async def mock_run(query, **kwargs):
            nonlocal call_count
            call_count += 1
            result = AsyncMock()
            if call_count == 1:
                # Main query: has neighbors
                result.data = AsyncMock(return_value=[{
                    "center_props": {"id": "c-id", "name": "SNR-75"},
                    "center_type": "RADAR_SYSTEM",
                    "source": "SNR-75", "source_type": "RADAR_SYSTEM",
                    "source_props": {"id": "c-id", "name": "SNR-75"},
                    "rel_type": "SPECIFIED_BY",
                    "rel_props": {},
                    "target": "Search PRF", "target_type": "SPECIFICATION",
                    "target_props": {"id": "t-id", "name": "Search PRF"},
                }])
            return result

        session.run = mock_run

        result = await get_neighborhood_graph_async(driver, "SNR-75")
        assert len(result["edges"]) == 1
        assert result["edges"][0]["rel_type"] == "SPECIFIED_BY"
        # Only 1 session.run call — no fallback, no co-occurrence
        assert call_count == 1
