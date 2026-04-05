"""ArcadeDB implementation of the GraphStore Protocol.

Maps every Protocol method to ArcadeDB SQL executed via ArcadeDBClient.
Uses ``command()`` for writes and ``query()`` for reads, with sync variants
for Celery workers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.services.arcadedb_client import ArcadeDBClient
from app.services.graph_store import (
    GraphEntityResult,
    NodeRecord,
    ProvenanceMetadata,
    RelationshipRecord,
    SchemaSyncReport,
)

logger = logging.getLogger(__name__)

# Maximum retries for ensure_ready
_READY_MAX_RETRIES = 5
_READY_BACKOFF_BASE = 0.5  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rid(result: list[dict[str, Any]]) -> str:
    """Extract the @rid string from a single-row command result."""
    if result and isinstance(result[0], dict):
        return str(result[0].get("@rid", ""))
    return ""


def _count(result: list[dict[str, Any]]) -> int:
    """Extract a count from a command result."""
    if result and isinstance(result[0], dict):
        return int(result[0].get("count", 0))
    return 0


def _to_entity(row: dict[str, Any]) -> GraphEntityResult:
    """Map an ArcadeDB row dict to a GraphEntityResult."""
    props = {k: v for k, v in row.items() if k not in (
        "@rid", "name", "entity_type", "canonical_name", "extraction_confidence",
        "@type", "@cat",
    )}
    return GraphEntityResult(
        node_id=str(row.get("@rid", row.get("node_id", ""))),
        name=row.get("name", ""),
        entity_type=row.get("entity_type", ""),
        canonical_name=row.get("canonical_name"),
        extraction_confidence=row.get("extraction_confidence"),
        properties=props,
    )


def _build_where(identity_fields: dict[str, Any]) -> str:
    """Build a WHERE clause from identity fields using parameter placeholders."""
    parts = [f"{k} = :{k}" for k in identity_fields]
    return " AND ".join(parts)


def _build_set(fields: dict[str, Any]) -> str:
    """Build a SET clause from a dict using parameter placeholders."""
    parts = [f"{k} = :{k}" for k in fields]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# ArcadeDBGraphStore
# ---------------------------------------------------------------------------


class ArcadeDBGraphStore:
    """GraphStore implementation backed by ArcadeDB.

    Parameters
    ----------
    client:
        An :class:`ArcadeDBClient` instance.
    database:
        The ArcadeDB database name to operate on.
    """

    def __init__(self, client: ArcadeDBClient, database: str) -> None:
        self._client = client
        self._database = database

    # ==================================================================
    # Node operations
    # ==================================================================

    async def upsert_node(
        self,
        record: NodeRecord,
        provenance: ProvenanceMetadata | None = None,
    ) -> str:
        """Upsert a single node and return its backend ID."""
        rid = await self._upsert_node_impl(record)
        if provenance:
            await self._create_provenance_edge(rid, provenance)
        return rid

    async def _upsert_node_impl(self, record: NodeRecord) -> str:
        set_fields: dict[str, Any] = {
            "name": record.name,
            "entity_type": record.entity_type,
            "extraction_confidence": record.extraction_confidence,
        }
        set_fields.update(record.properties)

        where_clause = _build_where(record.identity_fields)
        set_clause = _build_set(set_fields)

        sql = (
            f"UPDATE {record.entity_type} SET {set_clause}, "
            f"updated_at = sysdate() "
            f"UPSERT WHERE {where_clause} AND entity_type = :entity_type"
        )
        params = {**set_fields, **record.identity_fields}

        result = await self._client.command(
            self._database, "sql", sql, params,
        )
        return _rid(result)

    async def _create_provenance_edge(
        self, node_rid: str, provenance: ProvenanceMetadata,
    ) -> None:
        sql = (
            "CREATE EDGE EXTRACTED_FROM FROM :node_rid "
            "TO (SELECT FROM Document WHERE document_id = :document_id) "
            "SET page_numbers = :page_numbers, created_at = sysdate()"
        )
        params = {
            "node_rid": node_rid,
            "document_id": provenance.document_id,
            "page_numbers": provenance.page_numbers,
        }
        await self._client.command(self._database, "sql", sql, params)

    async def upsert_nodes_batch(
        self,
        records: list[NodeRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Upsert multiple nodes and return their backend IDs."""
        return [await self.upsert_node(r, provenance) for r in records]

    async def upsert_document_node(
        self,
        document_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Upsert the root Document vertex."""
        props = dict(properties or {})
        extra_set = ""
        if props:
            extra_set = ", " + _build_set(props)

        sql = (
            f"UPDATE Document SET document_id = :document_id{extra_set}, "
            f"updated_at = sysdate() "
            f"UPSERT WHERE document_id = :document_id"
        )
        params = {"document_id": document_id, **props}
        result = await self._client.command(self._database, "sql", sql, params)
        return _rid(result)

    async def create_text_chunk_vertex(
        self,
        chunk_id: str,
        text: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a TextChunk vertex."""
        props = dict(properties or {})
        extra_set = ""
        if props:
            extra_set = ", " + _build_set(props)

        sql = (
            f"CREATE VERTEX TextChunk SET chunk_id = :chunk_id, "
            f"text = :text, document_id = :document_id, "
            f"created_at = sysdate(){extra_set}"
        )
        params = {"chunk_id": chunk_id, "text": text, "document_id": document_id, **props}
        result = await self._client.command(self._database, "sql", sql, params)
        return _rid(result)

    async def create_image_chunk_vertex(
        self,
        chunk_id: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create an ImageChunk vertex."""
        props = dict(properties or {})
        extra_set = ""
        if props:
            extra_set = ", " + _build_set(props)

        sql = (
            f"CREATE VERTEX ImageChunk SET chunk_id = :chunk_id, "
            f"document_id = :document_id, "
            f"created_at = sysdate(){extra_set}"
        )
        params = {"chunk_id": chunk_id, "document_id": document_id, **props}
        result = await self._client.command(self._database, "sql", sql, params)
        return _rid(result)

    # ==================================================================
    # Edge / relationship operations
    # ==================================================================

    async def upsert_relationship(
        self,
        record: RelationshipRecord,
        provenance: ProvenanceMetadata | None = None,
    ) -> str:
        """Upsert a relationship edge between two entity vertices."""
        from_where = _build_where(record.from_identity)
        to_where = _build_where(record.to_identity)

        doc_ids_expr = "[]"
        doc_id_param: dict[str, Any] = {}
        if provenance:
            doc_ids_expr = "[:doc_id]"
            doc_id_param = {"doc_id": provenance.document_id}

        extra_props = ""
        if record.properties:
            extra_props = ", " + _build_set(record.properties)

        sql = (
            f"CREATE EDGE {record.rel_type} "
            f"FROM (SELECT FROM {record.from_type} WHERE {from_where}) "
            f"TO (SELECT FROM {record.to_type} WHERE {to_where}) "
            f"SET extraction_confidence = :extraction_confidence, "
            f"document_ids = {doc_ids_expr}, "
            f"created_at = sysdate(), updated_at = sysdate(){extra_props}"
        )
        params = {
            **record.from_identity,
            **record.to_identity,
            "extraction_confidence": record.extraction_confidence,
            **doc_id_param,
            **record.properties,
        }
        result = await self._client.command(self._database, "sql", sql, params)
        return _rid(result)

    async def upsert_relationships_batch(
        self,
        records: list[RelationshipRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Upsert multiple relationships."""
        return [await self.upsert_relationship(r, provenance) for r in records]

    async def create_structural_edge(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a structural edge between two RIDs."""
        props = dict(properties or {})
        extra_set = ""
        if props:
            extra_set = ", " + _build_set(props)

        sql = (
            f"CREATE EDGE {rel_type} FROM {from_id} TO {to_id} "
            f"SET created_at = sysdate(){extra_set}"
        )
        result = await self._client.command(self._database, "sql", sql, props or None)
        return _rid(result)

    # ==================================================================
    # Query operations
    # ==================================================================

    async def search_nodes(
        self,
        entity_type: str,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> list[GraphEntityResult]:
        """Search for nodes matching optional filters."""
        where_parts = [f"entity_type = '{entity_type}'"]
        params: dict[str, Any] = {"limit": limit}

        if filters:
            for k, v in filters.items():
                where_parts.append(f"{k} = :{k}")
                params[k] = v

        where = " AND ".join(where_parts)
        sql = (
            f"SELECT *, @rid AS node_id FROM {entity_type} "
            f"WHERE {where} LIMIT :limit"
        )
        rows = await self._client.query(self._database, "sql", sql, params)
        return [_to_entity(r) for r in rows]

    async def resolve_root_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> GraphEntityResult | None:
        """Resolve a name to its canonical root entity."""
        type_filter = ""
        params: dict[str, Any] = {"name": name}
        if entity_type:
            type_filter = " AND entity_type = :entity_type"
            params["entity_type"] = entity_type

        sql = (
            f"SELECT *, @rid AS node_id FROM V "
            f"WHERE name = :name{type_filter} LIMIT 1"
        )
        rows = await self._client.query(self._database, "sql", sql, params)
        return _to_entity(rows[0]) if rows else None

    async def fulltext_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[GraphEntityResult]:
        """Full-text search using LUCENE index."""
        type_filter = ""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if entity_types:
            type_list = ", ".join(f"'{t}'" for t in entity_types)
            type_filter = f" AND entity_type IN [{type_list}]"

        sql = (
            f"SELECT *, @rid AS node_id FROM V "
            f"WHERE name LUCENE :query{type_filter} "
            f"ORDER BY $score DESC LIMIT :limit"
        )
        rows = await self._client.query(self._database, "sql", sql, params)
        return [_to_entity(r) for r in rows]

    async def get_neighborhood(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> list[GraphEntityResult]:
        """Return nodes in the k-hop neighbourhood of *node_id*."""
        edge_filter = ""
        if rel_types:
            edge_filter = ", ".join(f"'{t}'" for t in rel_types)
            edge_filter = f"both({edge_filter})"
        else:
            edge_filter = "both()"

        sql = (
            f"SELECT expand({edge_filter}"
            f"{{1,{depth}}}) FROM {node_id}"
        )
        rows = await self._client.query(self._database, "sql", sql)
        return [_to_entity(r) for r in rows]

    async def get_neighborhood_graph(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return neighbourhood as a {nodes, edges} dict."""
        entities = await self.get_neighborhood(node_id, depth, rel_types)

        edge_filter = ""
        if rel_types:
            edge_filter = ", ".join(f"'{t}'" for t in rel_types)
            edge_filter = f"bothE({edge_filter})"
        else:
            edge_filter = "bothE()"

        sql = (
            f"SELECT expand({edge_filter}"
            f"{{1,{depth}}}) FROM {node_id}"
        )
        edge_rows = await self._client.query(self._database, "sql", sql)

        return {
            "nodes": [
                {
                    "id": e.node_id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    **e.properties,
                }
                for e in entities
            ],
            "edges": edge_rows,
        }

    async def get_ontology_linked_chunks(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Return text/image chunks linked to *node_id* via EXTRACTED_FROM."""
        sql = (
            "SELECT expand(in('EXTRACTED_FROM')) FROM :node_id"
        )
        rows = await self._client.query(
            self._database, "sql", sql, {"node_id": node_id},
        )
        return rows

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return backend-level graph statistics."""
        v_sql = "SELECT count(*) AS cnt FROM V"
        e_sql = "SELECT count(*) AS cnt FROM E"
        v_rows = await self._client.query(self._database, "sql", v_sql)
        e_rows = await self._client.query(self._database, "sql", e_sql)
        return {
            "vertex_count": _count(v_rows),
            "edge_count": _count(e_rows),
        }

    async def get_relationship_count(
        self,
        rel_type: str | None = None,
    ) -> int:
        """Return the number of relationships."""
        source = rel_type or "E"
        sql = f"SELECT count(*) AS count FROM {source}"
        rows = await self._client.query(self._database, "sql", sql)
        return _count(rows)

    async def get_co_extracted_entities(
        self,
        node_id: str,
        limit: int = 10,
    ) -> list[GraphEntityResult]:
        """Return entities that co-occur with *node_id* in the same source chunk."""
        sql = (
            "SELECT expand(out('EXTRACTED_FROM').in('EXTRACTED_FROM')) "
            "FROM :node_id LIMIT :limit"
        )
        rows = await self._client.query(
            self._database, "sql", sql, {"node_id": node_id, "limit": limit},
        )
        return [_to_entity(r) for r in rows]

    # ==================================================================
    # Alias operations
    # ==================================================================

    async def create_alias(
        self,
        node_id: str,
        alias: str,
    ) -> None:
        """Create an Alias vertex and HAS_ALIAS edge to the target node."""
        sql = (
            "CREATE VERTEX Alias SET alias = :alias, created_at = sysdate()"
        )
        alias_result = await self._client.command(
            self._database, "sql", sql, {"alias": alias},
        )
        alias_rid = _rid(alias_result)

        edge_sql = (
            f"CREATE EDGE HAS_ALIAS FROM {node_id} TO {alias_rid}"
        )
        await self._client.command(self._database, "sql", edge_sql)

    async def search_by_alias(
        self,
        alias: str,
        entity_type: str | None = None,
    ) -> list[GraphEntityResult]:
        """Find nodes by alias."""
        type_filter = ""
        params: dict[str, Any] = {"alias": alias}
        if entity_type:
            type_filter = " AND entity_type = :entity_type"
            params["entity_type"] = entity_type

        sql = (
            "SELECT expand(in('HAS_ALIAS')) FROM Alias "
            f"WHERE alias = :alias{type_filter}"
        )
        rows = await self._client.query(self._database, "sql", sql, params)
        return [_to_entity(r) for r in rows]

    async def set_canonical_name(
        self,
        node_id: str,
        canonical_name: str,
    ) -> None:
        """Set the canonical_name on a node."""
        sql = f"UPDATE {node_id} SET canonical_name = :canonical_name"
        await self._client.command(
            self._database, "sql", sql, {"canonical_name": canonical_name},
        )

    # ==================================================================
    # Vector operations
    # ==================================================================

    async def vector_search(
        self,
        embedding: list[float],
        entity_types: list[str] | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[GraphEntityResult]:
        """ANN search over node embeddings via vectorNeighbors."""
        sql = (
            "SELECT *, @rid AS node_id "
            "FROM (SELECT expand(vectorNeighbors('TextChunk[text_embedding]', "
            ":query_vector, :top_k)))"
        )
        params: dict[str, Any] = {
            "query_vector": embedding,
            "top_k": limit,
        }
        rows = await self._client.query(self._database, "sql", sql, params)
        results = [_to_entity(r) for r in rows]

        if score_threshold is not None:
            results = [
                r for r in results
                if (r.extraction_confidence or 0) >= score_threshold
            ]
        return results

    async def set_vertex_embedding(
        self,
        node_id: str,
        embedding: list[float],
        model_name: str | None = None,
    ) -> None:
        """Attach a vector embedding to a node."""
        model_set = ""
        params: dict[str, Any] = {"embedding": embedding}
        if model_name:
            model_set = ", embedding_model = :model_name"
            params["model_name"] = model_name

        sql = (
            f"UPDATE {node_id} SET text_embedding = :embedding{model_set}, "
            f"updated_at = sysdate()"
        )
        await self._client.command(self._database, "sql", sql, params)

    async def cross_model_search(
        self,
        text_embedding: list[float],
        image_embedding: list[float],
        limit: int = 10,
    ) -> list[GraphEntityResult]:
        """Search using both text and image embeddings + graph traversal."""
        sql = (
            "SELECT chunk.*, entity.name AS entity_name, "
            "entity.entity_type AS entity_entity_type "
            "FROM ("
            "  SELECT expand(vectorNeighbors('TextChunk[text_embedding]', "
            "  :text_vector, :top_k))"
            ") AS chunk "
            "LET entity = chunk.in('EXTRACTED_FROM')"
        )
        params: dict[str, Any] = {
            "text_vector": text_embedding,
            "top_k": limit,
        }
        text_rows = await self._client.query(self._database, "sql", sql, params)

        img_sql = (
            "SELECT *, @rid AS node_id "
            "FROM (SELECT expand(vectorNeighbors('ImageChunk[image_embedding]', "
            ":image_vector, :top_k)))"
        )
        img_params: dict[str, Any] = {
            "image_vector": image_embedding,
            "top_k": limit,
        }
        img_rows = await self._client.query(self._database, "sql", img_sql, img_params)

        combined = text_rows + img_rows
        return [_to_entity(r) for r in combined[:limit]]

    # ==================================================================
    # Lifecycle operations
    # ==================================================================

    async def delete_document_graph(
        self,
        document_id: str,
    ) -> int:
        """Delete all graph elements associated with *document_id*."""
        total = 0
        params = {"doc_id": document_id}

        # 1. Delete text chunks
        sql = "DELETE VERTEX FROM TextChunk WHERE document_id = :doc_id"
        result = await self._client.command(self._database, "sql", sql, params)
        total += _count(result)

        # 2. Delete image chunks
        sql = "DELETE VERTEX FROM ImageChunk WHERE document_id = :doc_id"
        result = await self._client.command(self._database, "sql", sql, params)
        total += _count(result)

        # 3. Delete Document vertex
        sql = "DELETE VERTEX FROM Document WHERE document_id = :doc_id"
        result = await self._client.command(self._database, "sql", sql, params)
        total += _count(result)

        # 4. Orphan cleanup: remove entities with no remaining EXTRACTED_FROM edges
        orphan_sql = (
            "DELETE VERTEX FROM V WHERE @cat NOT IN "
            "['Document', 'TextChunk', 'ImageChunk', 'Alias'] "
            "AND out('EXTRACTED_FROM').size() = 0 "
            "AND in('EXTRACTED_FROM').size() = 0"
        )
        result = await self._client.command(self._database, "sql", orphan_sql)
        total += _count(result)

        return total

    async def sync_schema(self) -> SchemaSyncReport:
        """Ensure the backend schema matches the current ontology."""
        report = SchemaSyncReport()

        # Core vertex types
        vertex_types = [
            "Document", "TextChunk", "ImageChunk", "Alias", "BaseEntity",
        ]
        for vtype in vertex_types:
            try:
                sql = f"CREATE VERTEX TYPE {vtype} IF NOT EXISTS"
                await self._client.command(self._database, "sql", sql)
                report.types_created += 1
            except Exception as e:
                report.errors.append(f"Failed to create {vtype}: {e}")

        # Core edge types
        edge_types = [
            "EXTRACTED_FROM", "HAS_ALIAS", "HAS_CHUNK",
        ]
        for etype in edge_types:
            try:
                sql = f"CREATE EDGE TYPE {etype} IF NOT EXISTS"
                await self._client.command(self._database, "sql", sql)
                report.types_created += 1
            except Exception as e:
                report.errors.append(f"Failed to create {etype}: {e}")

        return report

    async def ensure_indexes(self) -> None:
        """Create any missing indexes required for efficient queries."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS ON Document (document_id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON TextChunk (chunk_id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON TextChunk (document_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON ImageChunk (chunk_id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON ImageChunk (document_id) NOTUNIQUE",
        ]
        for sql in indexes:
            try:
                await self._client.command(self._database, "sql", sql)
            except Exception as e:
                logger.warning("Index creation failed: %s", e)

    async def ensure_ready(self) -> None:
        """Block until the backend is ready, with retry/backoff."""
        last_exc: Exception | None = None
        for attempt in range(_READY_MAX_RETRIES):
            try:
                result = await self._client.query(
                    self._database, "sql", "SELECT name FROM schema:database",
                )
                logger.info("ArcadeDB ready (attempt %d)", attempt + 1)
                return
            except Exception as exc:
                last_exc = exc
                wait = _READY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "ArcadeDB not ready (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _READY_MAX_RETRIES, exc, wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"ArcadeDB not ready after {_READY_MAX_RETRIES} retries"
        ) from last_exc

    # ==================================================================
    # Sync variants (for Celery workers)
    # ==================================================================

    def upsert_node_sync(
        self,
        record: NodeRecord,
        provenance: ProvenanceMetadata | None = None,
    ) -> str:
        """Synchronous upsert_node."""
        rid = self._upsert_node_impl_sync(record)
        if provenance:
            self._create_provenance_edge_sync(rid, provenance)
        return rid

    def _upsert_node_impl_sync(self, record: NodeRecord) -> str:
        set_fields: dict[str, Any] = {
            "name": record.name,
            "entity_type": record.entity_type,
            "extraction_confidence": record.extraction_confidence,
        }
        set_fields.update(record.properties)

        where_clause = _build_where(record.identity_fields)
        set_clause = _build_set(set_fields)

        sql = (
            f"UPDATE {record.entity_type} SET {set_clause}, "
            f"updated_at = sysdate() "
            f"UPSERT WHERE {where_clause} AND entity_type = :entity_type"
        )
        params = {**set_fields, **record.identity_fields}
        result = self._client.command_sync(self._database, "sql", sql, params)
        return _rid(result)

    def _create_provenance_edge_sync(
        self, node_rid: str, provenance: ProvenanceMetadata,
    ) -> None:
        sql = (
            "CREATE EDGE EXTRACTED_FROM FROM :node_rid "
            "TO (SELECT FROM Document WHERE document_id = :document_id) "
            "SET page_numbers = :page_numbers, created_at = sysdate()"
        )
        params = {
            "node_rid": node_rid,
            "document_id": provenance.document_id,
            "page_numbers": provenance.page_numbers,
        }
        self._client.command_sync(self._database, "sql", sql, params)

    def upsert_nodes_batch_sync(
        self,
        records: list[NodeRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Synchronous batch upsert."""
        return [self.upsert_node_sync(r, provenance) for r in records]

    def upsert_relationships_batch_sync(
        self,
        records: list[RelationshipRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Synchronous batch relationship upsert."""
        results = []
        for r in records:
            from_where = _build_where(r.from_identity)
            to_where = _build_where(r.to_identity)

            doc_ids_expr = "[]"
            doc_id_param: dict[str, Any] = {}
            if provenance:
                doc_ids_expr = "[:doc_id]"
                doc_id_param = {"doc_id": provenance.document_id}

            sql = (
                f"CREATE EDGE {r.rel_type} "
                f"FROM (SELECT FROM {r.from_type} WHERE {from_where}) "
                f"TO (SELECT FROM {r.to_type} WHERE {to_where}) "
                f"SET extraction_confidence = :extraction_confidence, "
                f"document_ids = {doc_ids_expr}, "
                f"created_at = sysdate(), updated_at = sysdate()"
            )
            params = {
                **r.from_identity,
                **r.to_identity,
                "extraction_confidence": r.extraction_confidence,
                **doc_id_param,
            }
            result = self._client.command_sync(self._database, "sql", sql, params)
            results.append(_rid(result))
        return results

    def create_structural_edge_sync(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous structural edge creation."""
        props = dict(properties or {})
        extra_set = ""
        if props:
            extra_set = ", " + _build_set(props)

        sql = (
            f"CREATE EDGE {rel_type} FROM {from_id} TO {to_id} "
            f"SET created_at = sysdate(){extra_set}"
        )
        result = self._client.command_sync(self._database, "sql", sql, props or None)
        return _rid(result)

    def set_vertex_embedding_sync(
        self,
        node_id: str,
        embedding: list[float],
        model_name: str | None = None,
    ) -> None:
        """Synchronous embedding set."""
        model_set = ""
        params: dict[str, Any] = {"embedding": embedding}
        if model_name:
            model_set = ", embedding_model = :model_name"
            params["model_name"] = model_name

        sql = (
            f"UPDATE {node_id} SET text_embedding = :embedding{model_set}, "
            f"updated_at = sysdate()"
        )
        self._client.command_sync(self._database, "sql", sql, params)

    def create_text_chunk_vertex_sync(
        self,
        chunk_id: str,
        text: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous TextChunk creation."""
        props = dict(properties or {})
        extra_set = ""
        if props:
            extra_set = ", " + _build_set(props)

        sql = (
            f"CREATE VERTEX TextChunk SET chunk_id = :chunk_id, "
            f"text = :text, document_id = :document_id, "
            f"created_at = sysdate(){extra_set}"
        )
        params = {"chunk_id": chunk_id, "text": text, "document_id": document_id, **props}
        result = self._client.command_sync(self._database, "sql", sql, params)
        return _rid(result)

    def create_image_chunk_vertex_sync(
        self,
        chunk_id: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous ImageChunk creation."""
        props = dict(properties or {})
        extra_set = ""
        if props:
            extra_set = ", " + _build_set(props)

        sql = (
            f"CREATE VERTEX ImageChunk SET chunk_id = :chunk_id, "
            f"document_id = :document_id, "
            f"created_at = sysdate(){extra_set}"
        )
        params = {"chunk_id": chunk_id, "document_id": document_id, **props}
        result = self._client.command_sync(self._database, "sql", sql, params)
        return _rid(result)

    def delete_document_graph_sync(
        self,
        document_id: str,
    ) -> int:
        """Synchronous document graph deletion."""
        total = 0
        params = {"doc_id": document_id}

        for sql in [
            "DELETE VERTEX FROM TextChunk WHERE document_id = :doc_id",
            "DELETE VERTEX FROM ImageChunk WHERE document_id = :doc_id",
            "DELETE VERTEX FROM Document WHERE document_id = :doc_id",
        ]:
            result = self._client.command_sync(self._database, "sql", sql, params)
            total += _count(result)

        orphan_sql = (
            "DELETE VERTEX FROM V WHERE @cat NOT IN "
            "['Document', 'TextChunk', 'ImageChunk', 'Alias'] "
            "AND out('EXTRACTED_FROM').size() = 0 "
            "AND in('EXTRACTED_FROM').size() = 0"
        )
        result = self._client.command_sync(self._database, "sql", orphan_sql)
        total += _count(result)

        return total

    def ensure_ready_sync(self) -> None:
        """Synchronous ensure_ready with retry/backoff."""
        last_exc: Exception | None = None
        for attempt in range(_READY_MAX_RETRIES):
            try:
                self._client.query_sync(
                    self._database, "sql", "SELECT name FROM schema:database",
                )
                logger.info("ArcadeDB ready (attempt %d)", attempt + 1)
                return
            except Exception as exc:
                last_exc = exc
                wait = _READY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "ArcadeDB not ready (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _READY_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"ArcadeDB not ready after {_READY_MAX_RETRIES} retries"
        ) from last_exc

    def fulltext_search_sync(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[GraphEntityResult]:
        """Synchronous full-text search using LUCENE index."""
        type_filter = ""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if entity_types:
            type_list = ", ".join(f"'{t}'" for t in entity_types)
            type_filter = f" AND entity_type IN [{type_list}]"

        sql = (
            f"SELECT *, @rid AS node_id FROM V "
            f"WHERE name LUCENE :query{type_filter} "
            f"ORDER BY $score DESC LIMIT :limit"
        )
        rows = self._client.query_sync(self._database, "sql", sql, params)
        return [_to_entity(r) for r in rows]

    def search_by_alias_sync(
        self,
        alias: str,
        entity_type: str | None = None,
    ) -> list[GraphEntityResult]:
        """Synchronous alias search."""
        type_filter = ""
        params: dict[str, Any] = {"alias": alias}
        if entity_type:
            type_filter = " AND entity_type = :entity_type"
            params["entity_type"] = entity_type

        sql = (
            "SELECT expand(in('HAS_ALIAS')) FROM Alias "
            f"WHERE alias = :alias{type_filter}"
        )
        rows = self._client.query_sync(self._database, "sql", sql, params)
        return [_to_entity(r) for r in rows]

    def create_alias_sync(
        self,
        node_id: str,
        alias: str,
    ) -> None:
        """Synchronous alias creation."""
        sql = (
            "CREATE VERTEX Alias SET alias = :alias, created_at = sysdate()"
        )
        alias_result = self._client.command_sync(
            self._database, "sql", sql, {"alias": alias},
        )
        alias_rid = _rid(alias_result)

        edge_sql = (
            f"CREATE EDGE HAS_ALIAS FROM {node_id} TO {alias_rid}"
        )
        self._client.command_sync(self._database, "sql", edge_sql)

    def set_canonical_name_sync(
        self,
        node_id: str,
        canonical_name: str,
    ) -> None:
        """Synchronous canonical name setter."""
        sql = f"UPDATE {node_id} SET canonical_name = :canonical_name"
        self._client.command_sync(
            self._database, "sql", sql, {"canonical_name": canonical_name},
        )

    def resolve_root_entity_sync(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> GraphEntityResult | None:
        """Synchronous root entity resolution."""
        type_filter = ""
        params: dict[str, Any] = {"name": name}
        if entity_type:
            type_filter = " AND entity_type = :entity_type"
            params["entity_type"] = entity_type

        sql = (
            f"SELECT *, @rid AS node_id FROM V "
            f"WHERE name = :name{type_filter} LIMIT 1"
        )
        rows = self._client.query_sync(self._database, "sql", sql, params)
        return _to_entity(rows[0]) if rows else None

    def close_sync(self) -> None:
        """Release any held resources."""
        self._client.close_sync()
