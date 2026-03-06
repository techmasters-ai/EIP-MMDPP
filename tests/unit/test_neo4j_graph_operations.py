"""Unit tests for Neo4j graph operations beyond _sanitize_label.

Tests upsert, structural edges, search, neighborhood, and async operations
using a mocked Neo4j driver.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_driver(run_return=None):
    """Return (driver, mock_session) with session.run returning run_return."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.single.return_value = run_return
    mock_result.__iter__ = MagicMock(return_value=iter(run_return if isinstance(run_return, list) else []))
    mock_session.run.return_value = mock_result

    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, mock_session


def _mock_async_driver(data_return=None, single_return=None):
    """Return (driver, mock_session, mock_result) for async tests."""
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=data_return or [])
    mock_result.single = AsyncMock(return_value=single_return)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)

    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)
    return driver, mock_session, mock_result


# ---------------------------------------------------------------------------
# upsert_node
# ---------------------------------------------------------------------------

class TestUpsertNode:
    def test_returns_node_id_on_success(self):
        from app.services.neo4j_graph import upsert_node
        driver, session = _mock_driver({"node_id": "id-123"})
        result = upsert_node(driver, "RadarSystem", "AN/SPY-1", "art-1", 0.9)
        assert result is not None

    def test_properties_passed(self):
        from app.services.neo4j_graph import upsert_node
        driver, session = _mock_driver({"node_id": "id-1"})
        upsert_node(driver, "RadarSystem", "AN/SPY-1", "art-1", 0.9, {"freq": "S-band"})
        call_kwargs = session.run.call_args
        assert "props" in call_kwargs.kwargs
        assert call_kwargs.kwargs["props"]["freq"] == "S-band"

    def test_exception_returns_none(self):
        from app.services.neo4j_graph import upsert_node
        driver, session = _mock_driver()
        session.run.side_effect = Exception("connection lost")
        result = upsert_node(driver, "RadarSystem", "TEST", "art-1", 0.5)
        assert result is None

    def test_sanitized_label_in_query(self):
        from app.services.neo4j_graph import upsert_node
        driver, session = _mock_driver({"node_id": "id-1"})
        upsert_node(driver, "Air Defense System", "SA-11", "art-1", 0.9)
        query = session.run.call_args.args[0]
        assert "Air_Defense_System" in query


# ---------------------------------------------------------------------------
# upsert_relationship
# ---------------------------------------------------------------------------

class TestUpsertRelationship:
    def test_returns_true_on_success(self):
        from app.services.neo4j_graph import upsert_relationship
        driver, _ = _mock_driver()
        result = upsert_relationship(driver, "A", "TypeA", "B", "TypeB", "REL", "art-1", 0.8)
        assert result is True

    def test_both_labels_sanitized(self):
        from app.services.neo4j_graph import upsert_relationship
        driver, session = _mock_driver()
        upsert_relationship(driver, "A", "Type A", "B", "Type B", "rel type", "a1", 0.5)
        query = session.run.call_args.args[0]
        assert "Type_A" in query
        assert "Type_B" in query
        assert "rel_type" in query

    def test_exception_returns_false(self):
        from app.services.neo4j_graph import upsert_relationship
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert upsert_relationship(driver, "A", "T", "B", "T", "R", "a1", 0.5) is False


# ---------------------------------------------------------------------------
# upsert_document_node
# ---------------------------------------------------------------------------

class TestUpsertDocumentNode:
    def test_returns_document_id(self):
        from app.services.neo4j_graph import upsert_document_node
        driver, _ = _mock_driver()
        result = upsert_document_node(driver, "doc-1", "Test Doc")
        assert result == "doc-1"

    def test_exception_returns_none(self):
        from app.services.neo4j_graph import upsert_document_node
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert upsert_document_node(driver, "doc-1", "Test") is None


# ---------------------------------------------------------------------------
# upsert_chunk_ref_node
# ---------------------------------------------------------------------------

class TestUpsertChunkRefNode:
    def test_returns_chunk_id(self):
        from app.services.neo4j_graph import upsert_chunk_ref_node
        driver, _ = _mock_driver()
        assert upsert_chunk_ref_node(driver, "chunk-1", "text") == "chunk-1"

    def test_exception_returns_none(self):
        from app.services.neo4j_graph import upsert_chunk_ref_node
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert upsert_chunk_ref_node(driver, "c1", "text") is None


# ---------------------------------------------------------------------------
# create_structural_edge
# ---------------------------------------------------------------------------

class TestCreateStructuralEdge:
    def test_contains_text(self):
        from app.services.neo4j_graph import create_structural_edge
        driver, session = _mock_driver()
        assert create_structural_edge(driver, "doc-1", "chunk-1", "CONTAINS_TEXT") is True
        query = session.run.call_args.args[0]
        assert "Document" in query
        assert "ChunkRef" in query

    def test_contains_image(self):
        from app.services.neo4j_graph import create_structural_edge
        driver, _ = _mock_driver()
        assert create_structural_edge(driver, "doc-1", "chunk-1", "CONTAINS_IMAGE") is True

    def test_same_page(self):
        from app.services.neo4j_graph import create_structural_edge
        driver, session = _mock_driver()
        assert create_structural_edge(driver, "c1", "c2", "SAME_PAGE") is True
        query = session.run.call_args.args[0]
        assert "SAME_PAGE" in query

    def test_exception_returns_false(self):
        from app.services.neo4j_graph import create_structural_edge
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert create_structural_edge(driver, "a", "b", "CONTAINS_TEXT") is False


# ---------------------------------------------------------------------------
# create_entity_chunk_edge
# ---------------------------------------------------------------------------

class TestCreateEntityChunkEdge:
    def test_creates_extracted_from_edge(self):
        from app.services.neo4j_graph import create_entity_chunk_edge
        driver, session = _mock_driver()
        assert create_entity_chunk_edge(driver, "AN/SPY-1", "RadarSystem", "chunk-1") is True
        query = session.run.call_args.args[0]
        assert "EXTRACTED_FROM" in query

    def test_exception_returns_false(self):
        from app.services.neo4j_graph import create_entity_chunk_edge
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert create_entity_chunk_edge(driver, "A", "T", "c1") is False


# ---------------------------------------------------------------------------
# search_nodes_by_name
# ---------------------------------------------------------------------------

class TestSearchNodesByName:
    def test_returns_list_of_dicts(self):
        from app.services.neo4j_graph import search_nodes_by_name
        record = {"n": {"name": "AN/SPY-1"}, "entity_type": "RadarSystem"}
        driver, _ = _mock_driver(run_return=[record])
        results = search_nodes_by_name(driver, "SPY")
        assert len(results) == 1

    def test_empty_results(self):
        from app.services.neo4j_graph import search_nodes_by_name
        driver, _ = _mock_driver(run_return=[])
        assert search_nodes_by_name(driver, "nothing") == []

    def test_exception_returns_empty(self):
        from app.services.neo4j_graph import search_nodes_by_name
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert search_nodes_by_name(driver, "test") == []


# ---------------------------------------------------------------------------
# get_node_neighborhood
# ---------------------------------------------------------------------------

class TestGetNodeNeighborhood:
    def test_returns_neighborhood(self):
        from app.services.neo4j_graph import get_node_neighborhood
        record = {"start": {"name": "A"}, "rel_type": "HAS", "neighbor": {"name": "B"}}
        driver, _ = _mock_driver(run_return=[record])
        results = get_node_neighborhood(driver, "A")
        assert len(results) == 1

    def test_hop_count_clamped(self):
        from app.services.neo4j_graph import get_node_neighborhood
        driver, session = _mock_driver(run_return=[])
        get_node_neighborhood(driver, "A", hop_count=0)
        query0 = session.run.call_args.args[0]
        assert "*1..1" in query0

        get_node_neighborhood(driver, "A", hop_count=10)
        query10 = session.run.call_args.args[0]
        assert "*1..4" in query10

    def test_exception_returns_empty(self):
        from app.services.neo4j_graph import get_node_neighborhood
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert get_node_neighborhood(driver, "A") == []


# ---------------------------------------------------------------------------
# get_graph_stats
# ---------------------------------------------------------------------------

class TestGetGraphStats:
    def test_returns_counts(self):
        from app.services.neo4j_graph import get_graph_stats
        driver, session = _mock_driver()
        # First call for nodes, second for edges
        node_result = MagicMock()
        node_result.single.return_value = {"cnt": 42}
        edge_result = MagicMock()
        edge_result.single.return_value = {"cnt": 10}
        session.run.side_effect = [node_result, edge_result]
        stats = get_graph_stats(driver)
        assert stats["nodes"] == 42
        assert stats["edges"] == 10

    def test_exception_returns_zeros(self):
        from app.services.neo4j_graph import get_graph_stats
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        stats = get_graph_stats(driver)
        assert stats == {"nodes": 0, "edges": 0}


# ---------------------------------------------------------------------------
# Async operations
# ---------------------------------------------------------------------------

class TestSearchNodesAsync:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        from app.services.neo4j_graph import search_nodes_async
        data = [{"n": {"name": "X"}, "entity_type": "RadarSystem"}]
        driver, _, _ = _mock_async_driver(data_return=data)
        results = await search_nodes_async(driver, "X")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        from app.services.neo4j_graph import search_nodes_async
        driver, session, _ = _mock_async_driver()
        session.run.side_effect = Exception("fail")
        assert await search_nodes_async(driver, "test") == []


class TestGetNeighborhoodAsync:
    @pytest.mark.asyncio
    async def test_returns_neighborhood(self):
        from app.services.neo4j_graph import get_neighborhood_async
        data = [{"start": {"name": "A"}, "rel_type": "R", "neighbor": {"name": "B"}}]
        driver, _, _ = _mock_async_driver(data_return=data)
        results = await get_neighborhood_async(driver, "A")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_hop_count_clamped(self):
        from app.services.neo4j_graph import get_neighborhood_async
        driver, session, _ = _mock_async_driver()
        await get_neighborhood_async(driver, "A", hop_count=0)
        query = session.run.call_args.args[0]
        assert "*1..1" in query


class TestGetOntologyLinkedChunksAsync:
    @pytest.mark.asyncio
    async def test_returns_linked_chunks(self):
        from app.services.neo4j_graph import get_ontology_linked_chunks_async
        data = [{
            "target_chunk_id": "c2", "target_chunk_type": "text",
            "rel_type": "USES", "entity_name": "A", "related_name": "B",
        }]
        driver, _, _ = _mock_async_driver(data_return=data)
        results = await get_ontology_linked_chunks_async(driver, "c1")
        assert len(results) == 1
        assert results[0]["target_chunk_id"] == "c2"

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        from app.services.neo4j_graph import get_ontology_linked_chunks_async
        driver, session, _ = _mock_async_driver()
        session.run.side_effect = Exception("fail")
        assert await get_ontology_linked_chunks_async(driver, "c1") == []


class TestGetGraphStatsAsync:
    @pytest.mark.asyncio
    async def test_returns_counts(self):
        from app.services.neo4j_graph import get_graph_stats_async
        driver, session, _ = _mock_async_driver()
        # Two calls — nodes then edges
        node_result = AsyncMock()
        node_result.single = AsyncMock(return_value={"cnt": 5})
        edge_result = AsyncMock()
        edge_result.single = AsyncMock(return_value={"cnt": 3})
        session.run = AsyncMock(side_effect=[node_result, edge_result])
        stats = await get_graph_stats_async(driver)
        assert stats["nodes"] == 5
        assert stats["edges"] == 3


# ---------------------------------------------------------------------------
# fulltext_search_entity
# ---------------------------------------------------------------------------

class TestFulltextSearchEntity:
    def test_returns_results_with_score(self):
        from app.services.neo4j_graph import fulltext_search_entity
        records = [
            {"name": "AN/SPY-1", "canonical_name": None, "entity_type": "RadarSystem", "score": 2.5}
        ]
        driver, _ = _mock_driver(run_return=records)
        results = fulltext_search_entity(driver, "SPY")
        assert len(results) == 1
        assert results[0]["score"] == 2.5

    def test_exception_returns_empty(self):
        from app.services.neo4j_graph import fulltext_search_entity
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert fulltext_search_entity(driver, "test") == []


# ---------------------------------------------------------------------------
# create_alias_edge
# ---------------------------------------------------------------------------

class TestCreateAliasEdge:
    def test_returns_true_on_success(self):
        from app.services.neo4j_graph import create_alias_edge
        driver, _ = _mock_driver()
        assert create_alias_edge(driver, "AN/SPY-1", "SPY-1", "RadarSystem") is True

    def test_exception_returns_false(self):
        from app.services.neo4j_graph import create_alias_edge
        driver, session = _mock_driver()
        session.run.side_effect = Exception("fail")
        assert create_alias_edge(driver, "A", "B", "T") is False


# ---------------------------------------------------------------------------
# ensure_indexes
# ---------------------------------------------------------------------------

class TestEnsureIndexes:
    def test_runs_create_queries(self):
        from app.services.neo4j_graph import ensure_indexes
        driver, session = _mock_driver()
        ensure_indexes(driver)
        assert session.run.call_count == 3
        queries = [c.args[0] for c in session.run.call_args_list]
        assert any("entity_name_fulltext" in q for q in queries)
        assert any("document_id_unique" in q for q in queries)

    def test_exception_propagates(self):
        from app.services.neo4j_graph import ensure_indexes
        driver, session = _mock_driver()
        session.run.side_effect = Exception("down")
        with pytest.raises(Exception, match="down"):
            ensure_indexes(driver)
