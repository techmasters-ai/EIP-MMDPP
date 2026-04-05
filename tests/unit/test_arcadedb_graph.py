"""Unit tests for ArcadeDB GraphStore implementation.

Tests verify that ArcadeDBGraphStore generates correct ArcadeDB SQL and
delegates to the appropriate client methods. All tests mock the client —
no real ArcadeDB server required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(
    command_result=None,
    query_result=None,
    command_sync_result=None,
    query_sync_result=None,
):
    """Build a mock ArcadeDBClient with configurable return values."""
    client = MagicMock()
    client.command = AsyncMock(return_value=command_result or [{"@rid": "#1:0"}])
    client.query = AsyncMock(return_value=query_result or [])
    client.command_sync = MagicMock(return_value=command_sync_result or [{"@rid": "#1:0"}])
    client.query_sync = MagicMock(return_value=query_sync_result or [])
    client.close_sync = MagicMock()
    return client


def _graph(client=None, database="testdb"):
    """Build an ArcadeDBGraphStore with a mock client.

    By default the validation matrix is pre-initialized to the empty set so
    unit tests that focus on other behaviors don't get their relationships
    rejected by lazy-loaded ontology validation.
    """
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    store = ArcadeDBGraphStore(client=client or _make_client(), database=database)
    store._validation_matrix = set()
    return store


def _node_record(entity_type="RADAR_SYSTEM", identity_fields=None, name="APG-77", **kwargs):
    from app.services.graph_store import NodeRecord

    return NodeRecord(
        entity_type=entity_type,
        identity_fields=identity_fields or {"system_name": "APG-77"},
        name=name,
        **kwargs,
    )


def _relationship_record(**kwargs):
    from app.services.graph_store import RelationshipRecord

    defaults = dict(
        from_type="RADAR_SYSTEM",
        from_identity={"system_name": "APG-77"},
        to_type="PLATFORM",
        to_identity={"platform_name": "F-22"},
        rel_type="INSTALLED_ON",
    )
    defaults.update(kwargs)
    return RelationshipRecord(**defaults)


def _provenance(document_id="doc-001"):
    from app.services.graph_store import ProvenanceMetadata

    return ProvenanceMetadata(document_id=document_id, page_numbers=[1, 2])


# ---------------------------------------------------------------------------
# Test: upsert_node generates correct SQL
# ---------------------------------------------------------------------------

class TestUpsertNodeGeneratesCorrectSql:
    async def test_upsert_node_generates_correct_sql(self):
        """SQL should contain UPDATE...UPSERT with identity fields in WHERE."""
        client = _make_client()
        store = _graph(client)
        record = _node_record()

        await store.upsert_node(record)

        client.command.assert_called_once()
        call_args = client.command.call_args
        sql = call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs.get("command", "")
        db = call_args.args[0] if call_args.args else call_args.kwargs.get("database", "")

        assert db == "testdb"
        assert "UPDATE" in sql.upper()
        assert "UPSERT" in sql.upper()
        assert "RADAR_SYSTEM" in sql
        assert "system_name" in sql

    async def test_upsert_node_sets_name_and_confidence(self):
        """SQL should SET name and extraction_confidence."""
        client = _make_client()
        store = _graph(client)
        record = _node_record(extraction_confidence=0.92)

        await store.upsert_node(record)

        sql = client.command.call_args.args[2]
        assert "name" in sql.lower()
        assert "extraction_confidence" in sql.lower()

    async def test_upsert_node_with_provenance_creates_extracted_from(self):
        """When provenance is provided, a second command should create the edge."""
        client = _make_client()
        store = _graph(client)
        record = _node_record()
        prov = _provenance()

        await store.upsert_node(record, provenance=prov)

        # Should have at least 2 calls: upsert node + extracted_from edge
        assert client.command.call_count >= 2


# ---------------------------------------------------------------------------
# Test: upsert_node_sync calls command_sync
# ---------------------------------------------------------------------------

class TestUpsertNodeSyncCallsCommandSync:
    def test_upsert_node_sync_calls_command_sync(self):
        """Sync variant should call client.command_sync, not async command."""
        client = _make_client()
        store = _graph(client)
        record = _node_record()

        store.upsert_node_sync(record)

        client.command_sync.assert_called()
        client.command.assert_not_called()

        sql = client.command_sync.call_args.args[2]
        assert "UPDATE" in sql.upper()
        assert "UPSERT" in sql.upper()


# ---------------------------------------------------------------------------
# Test: upsert_relationship includes document_ids
# ---------------------------------------------------------------------------

class TestUpsertRelationshipIncludesDocumentIds:
    async def test_upsert_relationship_includes_document_ids(self):
        """Relationship SQL should reference document_ids list."""
        client = _make_client()
        store = _graph(client)
        rel = _relationship_record()
        prov = _provenance(document_id="doc-99")

        await store.upsert_relationship(rel, provenance=prov)

        client.command.assert_called()
        sql = client.command.call_args.args[2]
        assert "INSTALLED_ON" in sql
        assert "document_ids" in sql.lower()

    async def test_upsert_relationship_sets_confidence(self):
        """SQL should carry extraction_confidence."""
        client = _make_client()
        store = _graph(client)
        rel = _relationship_record(extraction_confidence=0.85)

        await store.upsert_relationship(rel)

        sql = client.command.call_args.args[2]
        assert "extraction_confidence" in sql.lower()


# ---------------------------------------------------------------------------
# Test: search_nodes uses LUCENE
# ---------------------------------------------------------------------------

class TestSearchNodesUsesLucene:
    async def test_fulltext_search_uses_lucene(self):
        """Fulltext search SQL should use LUCENE keyword."""
        client = _make_client(query_result=[
            {"@rid": "#1:0", "name": "APG-77", "entity_type": "RADAR_SYSTEM"},
        ])
        store = _graph(client)

        await store.fulltext_search("radar", entity_types=["RADAR_SYSTEM"])

        client.query.assert_called_once()
        sql = client.query.call_args.args[2]
        assert "LUCENE" in sql.upper()

    async def test_fulltext_search_filters_entity_types(self):
        """When entity_types given, SQL should filter by them."""
        client = _make_client(query_result=[])
        store = _graph(client)

        await store.fulltext_search("radar", entity_types=["RADAR_SYSTEM", "PLATFORM"])

        sql = client.query.call_args.args[2]
        assert "entity_type" in sql.lower()


# ---------------------------------------------------------------------------
# Test: vector_search uses vectorNeighbors
# ---------------------------------------------------------------------------

class TestVectorSearchUsesVectorNeighbors:
    async def test_vector_search_uses_vector_neighbors(self):
        """SQL should contain vectorNeighbors function call."""
        client = _make_client(query_result=[
            {"@rid": "#2:0", "name": "chunk-1", "entity_type": "TextChunk"},
        ])
        store = _graph(client)

        embedding = [0.1] * 384
        await store.vector_search("TextChunk", "text_embedding", embedding, top_k=5)

        client.query.assert_called_once()
        sql = client.query.call_args.args[2]
        assert "vectorNeighbors" in sql
        assert "TextChunk[text_embedding]" in sql

    async def test_vector_search_passes_limit(self):
        """Limit should appear in the SQL."""
        client = _make_client(query_result=[])
        store = _graph(client)

        await store.vector_search("TextChunk", "text_embedding", [0.1] * 384, top_k=15)

        sql = client.query.call_args.args[2]
        params = client.query.call_args.args[3] if len(client.query.call_args.args) > 3 else client.query.call_args.kwargs.get("params", {})
        assert params.get("top_k") == 15 or "15" in sql

    async def test_vector_search_image_chunk(self):
        """Should build correct index for ImageChunk."""
        client = _make_client(query_result=[
            {"@rid": "#3:0", "name": "img-1", "entity_type": "ImageChunk"},
        ])
        store = _graph(client)

        await store.vector_search("ImageChunk", "image_embedding", [0.1] * 512, top_k=5)

        sql = client.query.call_args.args[2]
        assert "ImageChunk[image_embedding]" in sql


# ---------------------------------------------------------------------------
# Test: cross_model_search
# ---------------------------------------------------------------------------

class TestCrossModelSearch:
    async def test_cross_model_search(self):
        """SQL should combine vectorNeighbors with graph traversal."""
        client = _make_client(query_result=[
            {"@rid": "#3:0", "name": "entity-1", "entity_type": "RADAR_SYSTEM"},
        ])
        store = _graph(client)

        text_emb = [0.1] * 384
        image_emb = [0.2] * 128
        await store.cross_model_search(text_emb, image_emb, limit=10)

        # Should call query at least once
        assert client.query.call_count >= 1
        # Check that vectorNeighbors appears in at least one query
        all_sqls = [call.args[2] for call in client.query.call_args_list]
        combined = " ".join(all_sqls)
        assert "vectorNeighbors" in combined


# ---------------------------------------------------------------------------
# Test: delete_document_graph
# ---------------------------------------------------------------------------

class TestDeleteDocumentGraph:
    async def test_delete_document_graph(self):
        """Should issue DELETE VERTEX commands for chunks and Document."""
        client = _make_client()
        store = _graph(client)

        await store.delete_document_graph("doc-42")

        assert client.command.call_count >= 3  # TextChunk, ImageChunk, Document + orphan cleanup
        all_sqls = [call.args[2] for call in client.command.call_args_list]
        combined = " ".join(all_sqls).upper()

        assert "DELETE VERTEX" in combined or "DELETE" in combined
        assert "TEXTCHUNK" in combined or "TextChunk" in " ".join(all_sqls)
        assert "DOCUMENT" in combined

    async def test_delete_document_graph_returns_count(self):
        """Should return number of elements deleted."""
        client = _make_client(command_result=[{"count": 5}])
        store = _graph(client)

        result = await store.delete_document_graph("doc-42")

        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Test: create_text_chunk_vertex
# ---------------------------------------------------------------------------

class TestCreateTextChunkVertex:
    async def test_create_text_chunk_vertex(self):
        """Should create a TextChunk vertex with chunk_id, text, document_id."""
        client = _make_client()
        store = _graph(client)

        await store.create_text_chunk_vertex(
            chunk_id="chunk-1",
            text="sample text content",
            document_id="doc-1",
            properties={"page_number": 3, "modality": "text"},
        )

        client.command.assert_called()
        sql = client.command.call_args.args[2]
        assert "TextChunk" in sql
        assert "chunk_id" in sql.lower()
        assert "document_id" in sql.lower()

    async def test_create_text_chunk_vertex_returns_id(self):
        """Should return a backend ID string."""
        client = _make_client(command_result=[{"@rid": "#5:0"}])
        store = _graph(client)

        result = await store.create_text_chunk_vertex("c1", "text", "d1")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test: create_alias
# ---------------------------------------------------------------------------

class TestCreateAlias:
    async def test_create_alias(self):
        """Should create Alias vertex + HAS_ALIAS edge."""
        client = _make_client()
        store = _graph(client)

        await store.create_alias("#1:0", "APG-77 Radar")

        assert client.command.call_count >= 1
        all_sqls = [call.args[2] for call in client.command.call_args_list]
        combined = " ".join(all_sqls)
        assert "Alias" in combined or "alias" in combined.lower()
        assert "HAS_ALIAS" in combined or "has_alias" in combined.lower()


# ---------------------------------------------------------------------------
# Test: ensure_ready retries
# ---------------------------------------------------------------------------

class TestEnsureReadyRetries:
    async def test_ensure_ready_retries(self):
        """Should retry on failure before succeeding."""
        client = _make_client()
        # First two calls fail, third succeeds
        client.query = AsyncMock(side_effect=[
            Exception("connection refused"),
            Exception("connection refused"),
            [{"name": "testdb"}],
        ])
        store = _graph(client)

        # Should not raise — retries should handle the initial failures
        await store.ensure_ready()

        assert client.query.call_count == 3

    async def test_ensure_ready_gives_up_after_max_retries(self):
        """If all retries fail, should raise."""
        client = _make_client()
        client.query = AsyncMock(side_effect=Exception("always fails"))
        store = _graph(client)

        with pytest.raises(Exception):
            await store.ensure_ready()


# ---------------------------------------------------------------------------
# Test: sync variants delegate correctly
# ---------------------------------------------------------------------------

class TestSyncVariants:
    def test_upsert_nodes_batch_sync(self):
        """Batch sync should issue a single sqlscript call containing all rows."""
        client = _make_client(
            command_sync_result=[{"@rid": "#10:0"}, {"@rid": "#10:1"}],
        )
        store = _graph(client)
        records = [_node_record(name="A"), _node_record(name="B")]

        result = store.upsert_nodes_batch_sync(records)

        # One sqlscript call, not one-per-record
        assert client.command_sync.call_count == 1
        language = client.command_sync.call_args.args[1]
        script = client.command_sync.call_args.args[2]
        assert language == "sqlscript"
        # Both rows should appear as separate statements
        assert script.count("UPSERT WHERE") == 2
        assert "name_0" in script and "name_1" in script
        assert result == ["#10:0", "#10:1"]

    def test_upsert_relationships_batch_sync_single_call(self):
        """Relationship batch should issue one sqlscript call with per-row params."""
        from app.services.graph_store import RelationshipRecord
        client = _make_client(
            command_sync_result=[{"@rid": "#20:0"}, {"@rid": "#20:1"}],
        )
        store = _graph(client)
        records = [
            RelationshipRecord(
                from_type="RADAR_SYSTEM",
                from_identity={"name": "APG-77", "entity_type": "RADAR_SYSTEM"},
                to_type="MISSILE",
                to_identity={"name": "AIM-120", "entity_type": "MISSILE"},
                rel_type="USES",
            ),
            RelationshipRecord(
                from_type="RADAR_SYSTEM",
                from_identity={"name": "APG-63", "entity_type": "RADAR_SYSTEM"},
                to_type="MISSILE",
                to_identity={"name": "AIM-9", "entity_type": "MISSILE"},
                rel_type="USES",
            ),
        ]

        result = store.upsert_relationships_batch_sync(records)

        assert client.command_sync.call_count == 1
        language = client.command_sync.call_args.args[1]
        script = client.command_sync.call_args.args[2]
        params = client.command_sync.call_args.args[3]
        assert language == "sqlscript"
        assert script.count("CREATE EDGE USES") == 2
        # Prefixed, row-suffixed params to avoid collisions across statements
        assert params["f_name_0"] == "APG-77"
        assert params["f_name_1"] == "APG-63"
        assert params["t_name_0"] == "AIM-120"
        assert params["t_name_1"] == "AIM-9"
        assert result == ["#20:0", "#20:1"]

    def test_create_text_chunks_batch_sync_folds_embedding(self):
        """Batch chunk create should fold the embedding into CREATE VERTEX."""
        from app.services.graph_store import TextChunkRecord
        client = _make_client(
            command_sync_result=[{"@rid": "#30:0"}, {"@rid": "#30:1"}],
        )
        store = _graph(client)
        records = [
            TextChunkRecord(
                chunk_id="c1", text="hello", document_id="d1",
                embedding=[0.1, 0.2, 0.3],
            ),
            TextChunkRecord(
                chunk_id="c2", text="world", document_id="d1",
                embedding=[0.4, 0.5, 0.6],
            ),
        ]

        result = store.create_text_chunks_batch_sync(records)

        assert client.command_sync.call_count == 1
        language = client.command_sync.call_args.args[1]
        script = client.command_sync.call_args.args[2]
        params = client.command_sync.call_args.args[3]
        assert language == "sqlscript"
        assert script.count("CREATE VERTEX TextChunk") == 2
        assert params["text_embedding_0"] == [0.1, 0.2, 0.3]
        assert params["text_embedding_1"] == [0.4, 0.5, 0.6]
        assert result == ["#30:0", "#30:1"]

    def test_batch_create_entity_chunk_edges_sync(self):
        """Entity-chunk edges should be created in one sqlscript call."""
        from app.services.graph_store import EntityChunkEdge
        client = _make_client()
        store = _graph(client)
        edges = [
            EntityChunkEdge(entity_name="APG-77", entity_type="RADAR_SYSTEM", chunk_rid="#40:0"),
            EntityChunkEdge(entity_name="AIM-120", entity_type="MISSILE", chunk_rid="#40:1"),
        ]

        n = store.batch_create_entity_chunk_edges_sync(edges)

        assert n == 2
        assert client.command_sync.call_count == 1
        language = client.command_sync.call_args.args[1]
        script = client.command_sync.call_args.args[2]
        params = client.command_sync.call_args.args[3]
        assert language == "sqlscript"
        assert script.count("CREATE EDGE EXTRACTED_FROM") == 2
        # Each statement uses type-scoped resolution
        assert "FROM RADAR_SYSTEM" in script
        assert "FROM MISSILE" in script
        assert params["name_0"] == "APG-77"
        assert params["rid_1"] == "#40:1"

    def test_batch_create_entity_chunk_edges_sync_empty(self):
        """Empty input should short-circuit without issuing any call."""
        client = _make_client()
        store = _graph(client)

        n = store.batch_create_entity_chunk_edges_sync([])

        assert n == 0
        client.command_sync.assert_not_called()


# ---------------------------------------------------------------------------
# Validation matrix enforcement
# ---------------------------------------------------------------------------


class TestValidationMatrix:
    def _store_with_matrix(self, client, matrix):
        store = _graph(client)
        store._validation_matrix = matrix  # bypass lazy load
        return store

    def test_valid_triple_passes(self):
        """Triples in the matrix should be written normally."""
        from app.services.graph_store import RelationshipRecord
        client = _make_client(
            command_sync_result=[{"@rid": "#50:0"}],
        )
        store = self._store_with_matrix(
            client,
            {("RADAR_SYSTEM", "INSTALLED_ON", "PLATFORM")},
        )
        rec = RelationshipRecord(
            from_type="RADAR_SYSTEM",
            from_identity={"name": "APG-77", "entity_type": "RADAR_SYSTEM"},
            to_type="PLATFORM",
            to_identity={"name": "F-22", "entity_type": "PLATFORM"},
            rel_type="INSTALLED_ON",
        )

        result = store.upsert_relationships_batch_sync([rec])

        assert client.command_sync.call_count == 1
        assert result == ["#50:0"]

    def test_invalid_triple_skipped_when_warn_mode(self):
        """With reject_invalid=false, invalid triples are silently dropped."""
        from app.services.graph_store import RelationshipRecord
        client = _make_client()
        store = self._store_with_matrix(
            client,
            {("RADAR_SYSTEM", "INSTALLED_ON", "PLATFORM")},
        )
        rec = RelationshipRecord(
            from_type="FREQUENCY_BAND",
            from_identity={"name": "X-band", "entity_type": "FREQUENCY_BAND"},
            to_type="PLATFORM",
            to_identity={"name": "F-22", "entity_type": "PLATFORM"},
            rel_type="INSTALLED_ON",
        )

        result = store.upsert_relationships_batch_sync([rec])

        # No HTTP call should have been issued
        client.command_sync.assert_not_called()
        assert result == []

    def test_invalid_triple_raises_when_reject_mode(self, monkeypatch):
        """With reject_invalid=true, invalid triples raise ValueError."""
        from app.services.graph_store import RelationshipRecord
        from app.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("GRAPH_REJECT_INVALID_RELATIONSHIPS", "true")

        try:
            client = _make_client()
            store = self._store_with_matrix(
                client,
                {("RADAR_SYSTEM", "INSTALLED_ON", "PLATFORM")},
            )
            rec = RelationshipRecord(
                from_type="FREQUENCY_BAND",
                from_identity={"name": "X-band", "entity_type": "FREQUENCY_BAND"},
                to_type="PLATFORM",
                to_identity={"name": "F-22", "entity_type": "PLATFORM"},
                rel_type="INSTALLED_ON",
            )

            with pytest.raises(ValueError, match="validation_matrix"):
                store.upsert_relationships_batch_sync([rec])
        finally:
            get_settings.cache_clear()

    def test_empty_matrix_is_permissive(self):
        """When the ontology defines no matrix, all triples are allowed."""
        from app.services.graph_store import RelationshipRecord
        client = _make_client(
            command_sync_result=[{"@rid": "#50:0"}],
        )
        store = self._store_with_matrix(client, set())
        rec = RelationshipRecord(
            from_type="RADAR_SYSTEM",
            from_identity={"name": "APG-77", "entity_type": "RADAR_SYSTEM"},
            to_type="PLATFORM",
            to_identity={"name": "F-22", "entity_type": "PLATFORM"},
            rel_type="INSTALLED_ON",
        )

        result = store.upsert_relationships_batch_sync([rec])

        assert client.command_sync.call_count == 1
        assert result == ["#50:0"]

    def test_sync_schema_invalidates_matrix_cache(self):
        """After sync_schema runs, the cache is cleared for reload."""
        client = _make_client()
        store = _graph(client)
        store._validation_matrix = {("A", "REL", "B")}

        async def _run():
            await store.sync_schema()

        import asyncio
        asyncio.get_event_loop().run_until_complete(_run())

        assert store._validation_matrix is None

    def test_create_text_chunk_vertex_sync(self):
        """Sync variant should use command_sync."""
        client = _make_client()
        store = _graph(client)

        result = store.create_text_chunk_vertex_sync("c1", "text", "d1")

        client.command_sync.assert_called()
        sql = client.command_sync.call_args.args[2]
        assert "TextChunk" in sql

    def test_delete_document_graph_sync(self):
        """Sync variant should use command_sync for all deletes."""
        client = _make_client()
        store = _graph(client)

        result = store.delete_document_graph_sync("doc-1")

        assert client.command_sync.call_count >= 3
        assert isinstance(result, int)

    def test_ensure_ready_sync(self):
        """Sync ensure_ready should use query_sync."""
        client = _make_client(query_sync_result=[{"name": "testdb"}])
        store = _graph(client)

        store.ensure_ready_sync()

        client.query_sync.assert_called()

    def test_close_sync(self):
        """close_sync should delegate to client."""
        client = _make_client()
        store = _graph(client)

        store.close_sync()

        client.close_sync.assert_called_once()


# ---------------------------------------------------------------------------
# Test: additional operations
# ---------------------------------------------------------------------------

class TestAdditionalOperations:
    async def test_upsert_document_node(self):
        """Should create/update a Document vertex."""
        client = _make_client()
        store = _graph(client)

        result = await store.upsert_document_node("doc-1", {"title": "Manual"})

        client.command.assert_called_once()
        sql = client.command.call_args.args[2]
        assert "Document" in sql
        assert "document_id" in sql.lower()

    async def test_create_structural_edge(self):
        """Should create an edge between two RIDs."""
        client = _make_client()
        store = _graph(client)

        result = await store.create_structural_edge("#1:0", "#2:0", "EXTRACTED_FROM")

        client.command.assert_called_once()
        sql = client.command.call_args.args[2]
        assert "EXTRACTED_FROM" in sql
        assert "#1:0" in sql or "from_id" in sql.lower()

    async def test_set_vertex_embedding(self):
        """Should update a vertex with an embedding vector."""
        client = _make_client()
        store = _graph(client)
        embedding = [0.1, 0.2, 0.3]

        await store.set_vertex_embedding("TextChunk", "#1:0", "text_embedding", embedding)

        client.command.assert_called_once()
        sql = client.command.call_args.args[2]
        assert "UPDATE" in sql.upper()

    async def test_get_neighborhood(self):
        """Should traverse graph from a node."""
        client = _make_client(query_result=[
            {"@rid": "#2:0", "name": "F-22", "entity_type": "PLATFORM"},
        ])
        store = _graph(client)

        results = await store.get_neighborhood("#1:0", depth=2)

        client.query.assert_called_once()
        sql = client.query.call_args.args[2]
        # Should contain variable-depth traversal syntax
        assert "1," in sql or "depth" in sql.lower() or "BOTH" in sql.upper() or "out(" in sql.lower() or "both(" in sql.lower()

    async def test_search_nodes(self):
        """Should query with optional filters."""
        client = _make_client(query_result=[
            {"@rid": "#1:0", "name": "APG-77", "entity_type": "RADAR_SYSTEM"},
        ])
        store = _graph(client)

        results = await store.search_nodes("RADAR_SYSTEM", filters={"band": "X"})

        client.query.assert_called_once()
        sql = client.query.call_args.args[2]
        assert "RADAR_SYSTEM" in sql

    async def test_set_canonical_name(self):
        """Should update canonical_name on a node."""
        client = _make_client()
        store = _graph(client)

        await store.set_canonical_name("#1:0", "APG-77 Radar System")

        client.command.assert_called_once()
        sql = client.command.call_args.args[2]
        assert "canonical_name" in sql.lower()

    async def test_search_community_reports_by_vector(self):
        """Should build vectorNeighbors SQL against CommunityReport."""
        client = _make_client(query_result=[
            {"community_id": 1, "title": "Radar cluster", "summary": "summary",
             "member_count": 3, "report_rid": "#99:0"},
        ])
        store = _graph(client)

        results = await store.search_community_reports_by_vector(
            query_vector=[0.1, 0.2, 0.3], top_k=5,
        )

        client.query.assert_called_once()
        sql = client.query.call_args.args[2]
        assert "CommunityReport[report_embedding]" in sql
        assert "vectorNeighbors" in sql
        params = client.query.call_args.kwargs["params"]
        assert params["query_vector"] == [0.1, 0.2, 0.3]
        assert params["top_k"] == 5
        assert len(results) == 1
        assert results[0]["community_id"] == 1

    async def test_get_relationships_between_entities_excludes_structural(self):
        """Should filter out EXTRACTED_FROM and other structural edges."""
        client = _make_client(query_result=[
            {"rel_type": "USES", "from_name": "APG-77", "from_type": "RADAR_SYSTEM",
             "to_name": "AIM-120", "to_type": "MISSILE"},
            {"rel_type": "EXTRACTED_FROM", "from_name": "APG-77", "from_type": "RADAR_SYSTEM",
             "to_name": "chunk-1", "to_type": "TextChunk"},
        ])
        store = _graph(client)

        results = await store.get_relationships_between_entities(["APG-77", "AIM-120"])

        assert len(results) == 1
        assert results[0]["rel_type"] == "USES"
        # Verify structural edge was filtered out
        assert all(r["rel_type"] != "EXTRACTED_FROM" for r in results)

    async def test_get_relationships_between_entities_empty_names(self):
        """Empty input should short-circuit to an empty result with no DB call."""
        client = _make_client()
        store = _graph(client)

        results = await store.get_relationships_between_entities([])

        assert results == []
        client.query.assert_not_called()
