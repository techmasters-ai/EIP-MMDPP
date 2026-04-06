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
    EntityChunkEdge,
    GraphEntityResult,
    ImageChunkRecord,
    NodeRecord,
    ProvenanceMetadata,
    RelationshipRecord,
    SchemaSyncReport,
    TextChunkRecord,
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
    """Map an ArcadeDB row dict to a GraphEntityResult.

    When the row originates from a ``vectorNeighbors`` query, ArcadeDB
    includes a ``$distance`` (or ``distance``) field.  For COSINE distance
    the value is in the range [0, 2]; we convert it to a similarity score
    ``1 - distance`` and store it in ``extraction_confidence`` so callers
    can filter on a threshold without knowing about the distance semantics.
    """
    # Determine extraction_confidence, preferring vector distance when present.
    distance = row.get("$distance", row.get("distance"))
    if distance is not None:
        extraction_confidence = 1.0 - float(distance)
    else:
        extraction_confidence = row.get("extraction_confidence")

    props = {k: v for k, v in row.items() if k not in (
        "@rid", "name", "entity_type", "canonical_name", "extraction_confidence",
        "@type", "@cat", "@class", "$distance", "distance",
        # Exclude embedding vectors — they're large and not needed in query results
        "text_embedding", "image_embedding", "report_embedding",
    )}
    return GraphEntityResult(
        node_id=str(row.get("@rid", row.get("node_id", ""))),
        name=row.get("name", ""),
        entity_type=row.get("entity_type", ""),
        canonical_name=row.get("canonical_name"),
        extraction_confidence=extraction_confidence,
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


def _build_upsert_node_script(
    records: list[NodeRecord],
) -> tuple[str, dict[str, Any]]:
    """Build a multi-statement sqlscript that upserts a batch of nodes.

    Each row gets parameter keys suffixed with ``_{i}`` so they don't collide
    across statements. Returns (script, params).
    """
    statements: list[str] = []
    params: dict[str, Any] = {}

    for i, record in enumerate(records):
        set_fields: dict[str, Any] = {
            "name": record.name,
            "entity_type": record.entity_type,
            "extraction_confidence": record.extraction_confidence,
        }
        set_fields.update(record.properties)

        set_parts: list[str] = []
        for k, v in set_fields.items():
            pk = f"{k}_{i}"
            params[pk] = v
            set_parts.append(f"{k} = :{pk}")

        where_parts: list[str] = []
        for k, v in record.identity_fields.items():
            pk = f"{k}_{i}"
            # Identity fields may overlap with set fields — same key, same value,
            # so reusing the param key is safe.
            params[pk] = v
            where_parts.append(f"{k} = :{pk}")

        set_clause = ", ".join(set_parts)
        where_clause = " AND ".join(where_parts)

        sql = (
            f"UPDATE {record.entity_type} SET {set_clause}, updated_at = sysdate() "
            f"UPSERT WHERE {where_clause} AND entity_type = :entity_type_{i}"
        )
        statements.append(sql)

    return ";\n".join(statements), params


def _build_upsert_relationship_script(
    records: list[RelationshipRecord],
    provenance: ProvenanceMetadata | None,
) -> tuple[str, dict[str, Any]]:
    """Build a multi-statement sqlscript that creates a batch of relationships.

    Uses prefixed param keys (``f_`` for from-identity, ``t_`` for to-identity,
    ``p_`` for edge properties) and a row suffix to avoid collisions.
    """
    statements: list[str] = []
    params: dict[str, Any] = {}

    doc_id_param_key = None
    if provenance:
        doc_id_param_key = "doc_id"
        params[doc_id_param_key] = provenance.document_id

    for i, record in enumerate(records):
        from_where_parts: list[str] = []
        for k, v in record.from_identity.items():
            pk = f"f_{k}_{i}"
            params[pk] = v
            from_where_parts.append(f"{k} = :{pk}")

        to_where_parts: list[str] = []
        for k, v in record.to_identity.items():
            pk = f"t_{k}_{i}"
            params[pk] = v
            to_where_parts.append(f"{k} = :{pk}")

        conf_key = f"extraction_confidence_{i}"
        params[conf_key] = record.extraction_confidence

        extra_parts: list[str] = []
        for k, v in record.properties.items():
            pk = f"p_{k}_{i}"
            params[pk] = v
            extra_parts.append(f"{k} = :{pk}")
        extra = (", " + ", ".join(extra_parts)) if extra_parts else ""

        doc_ids_expr = f"[:{doc_id_param_key}]" if doc_id_param_key else "[]"

        sql = (
            f"CREATE EDGE {record.rel_type} "
            f"FROM (SELECT FROM {record.from_type} WHERE {' AND '.join(from_where_parts)}) "
            f"TO (SELECT FROM {record.to_type} WHERE {' AND '.join(to_where_parts)}) "
            f"SET extraction_confidence = :{conf_key}, "
            f"document_ids = {doc_ids_expr}, "
            f"created_at = sysdate(), updated_at = sysdate(){extra}"
        )
        statements.append(sql)

    return ";\n".join(statements), params


def _build_text_chunk_sql(
    record: TextChunkRecord,
    row_idx: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a single CREATE VERTEX TextChunk statement, optionally folding in embedding.

    When ``row_idx`` is provided, param keys are suffixed with ``_{row_idx}``
    so the statement can be combined with others in a sqlscript batch.
    """
    suffix = f"_{row_idx}" if row_idx is not None else ""
    props = dict(record.properties or {})
    if record.embedding is not None:
        props["text_embedding"] = record.embedding

    set_fields: dict[str, Any] = {
        "chunk_id": record.chunk_id,
        "text": record.text,
        "document_id": record.document_id,
        **props,
    }

    set_parts: list[str] = []
    params: dict[str, Any] = {}
    for k, v in set_fields.items():
        pk = f"{k}{suffix}"
        params[pk] = v
        set_parts.append(f"{k} = :{pk}")

    sql = (
        f"CREATE VERTEX TextChunk SET {', '.join(set_parts)}, "
        f"created_at = sysdate()"
    )
    return sql, params


def _build_image_chunk_sql(
    record: ImageChunkRecord,
    row_idx: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a single CREATE VERTEX ImageChunk statement, optionally folding in embedding."""
    suffix = f"_{row_idx}" if row_idx is not None else ""
    props = dict(record.properties or {})
    if record.embedding is not None:
        props["image_embedding"] = record.embedding

    set_fields: dict[str, Any] = {
        "chunk_id": record.chunk_id,
        "document_id": record.document_id,
        **props,
    }

    set_parts: list[str] = []
    params: dict[str, Any] = {}
    for k, v in set_fields.items():
        pk = f"{k}{suffix}"
        params[pk] = v
        set_parts.append(f"{k} = :{pk}")

    sql = (
        f"CREATE VERTEX ImageChunk SET {', '.join(set_parts)}, "
        f"created_at = sysdate()"
    )
    return sql, params


def _extract_rids(result: list, expected: int) -> list[str]:
    """Extract a list of RIDs from a sqlscript result, padding to *expected* length.

    ArcadeDB's sqlscript endpoint returns a flat list of result rows, one per
    statement. This helper maps them back to the input order, filling missing
    entries with empty strings so callers always get a same-length list.
    """
    rids: list[str] = [""] * expected
    if not isinstance(result, list):
        return rids
    for i, row in enumerate(result[:expected]):
        if isinstance(row, dict):
            rids[i] = str(row.get("@rid", ""))
    return rids


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
        # Cache of valid (source_type, rel_type, target_type) triples. Loaded
        # lazily on first use and refreshed whenever sync_schema is called.
        self._validation_matrix: set[tuple[str, str, str]] | None = None
        # Time-based ensure_ready cache — avoid repeated schema:database probes.
        self._last_ready_check: float = 0

    # ------------------------------------------------------------------
    # Validation matrix
    # ------------------------------------------------------------------

    def _get_validation_matrix(self) -> set[tuple[str, str, str]]:
        """Return the cached validation matrix, loading it on first access."""
        if self._validation_matrix is None:
            try:
                from app.services.ontology_templates import load_validation_matrix
                self._validation_matrix = load_validation_matrix()
            except Exception as exc:
                logger.warning("Failed to load validation matrix: %s", exc)
                self._validation_matrix = set()
        return self._validation_matrix

    def _check_relationship_triple(
        self,
        from_type: str,
        rel_type: str,
        to_type: str,
    ) -> bool:
        """Return True if the triple is allowed by the validation matrix.

        An empty matrix (ontology doesn't define one) is treated as permissive
        so that the check never blocks ingestion when no matrix is configured.
        """
        matrix = self._get_validation_matrix()
        if not matrix:
            return True
        return (from_type, rel_type, to_type) in matrix

    def _enforce_relationship_triple(
        self,
        from_type: str,
        rel_type: str,
        to_type: str,
    ) -> bool:
        """Validate a triple, raising or logging per settings.

        Returns True when the caller should proceed with the write, False when
        the caller should skip the record.
        """
        if self._check_relationship_triple(from_type, rel_type, to_type):
            return True
        from app.config import get_settings
        if get_settings().graph_reject_invalid_relationships:
            raise ValueError(
                f"Invalid relationship triple: ({from_type})-[{rel_type}]->({to_type}) "
                f"is not in the active ontology validation_matrix"
            )
        logger.warning(
            "Skipping invalid relationship triple: (%s)-[%s]->(%s)",
            from_type, rel_type, to_type,
        )
        return False

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
            f"CREATE EDGE EXTRACTED_FROM FROM {node_rid} "
            "TO (SELECT FROM Document WHERE document_id = :document_id) "
            "SET page_numbers = :page_numbers, created_at = sysdate()"
        )
        params = {
            "document_id": provenance.document_id,
            "page_numbers": provenance.page_numbers,
        }
        await self._client.command(self._database, "sql", sql, params)

    async def upsert_nodes_batch(
        self,
        records: list[NodeRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Upsert multiple nodes in one sqlscript call. Returns their RIDs."""
        if not records:
            return []
        script, params = _build_upsert_node_script(records)
        result = await self._client.command(
            self._database, "sqlscript", script, params,
        )
        rids = _extract_rids(result, len(records))
        if provenance:
            await self._create_provenance_edges_batch(rids, provenance)
        return rids

    async def _create_provenance_edges_batch(
        self,
        node_rids: list[str],
        provenance: ProvenanceMetadata,
    ) -> None:
        """Create EXTRACTED_FROM edges from multiple nodes to a Document in one call."""
        targets = [rid for rid in node_rids if rid]
        if not targets:
            return
        statements: list[str] = []
        params: dict[str, Any] = {
            "document_id": provenance.document_id,
            "page_numbers": provenance.page_numbers,
        }
        for rid in targets:
            statements.append(
                f"CREATE EDGE EXTRACTED_FROM FROM {rid} "
                f"TO (SELECT FROM Document WHERE document_id = :document_id) "
                f"SET page_numbers = :page_numbers, created_at = sysdate()"
            )
        await self._client.command(
            self._database, "sqlscript", ";\n".join(statements), params,
        )

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
        embedding: list[float] | None = None,
    ) -> str:
        """Create a TextChunk vertex, optionally folding in the embedding."""
        sql, params = _build_text_chunk_sql(
            TextChunkRecord(
                chunk_id=chunk_id,
                text=text,
                document_id=document_id,
                properties=properties or {},
                embedding=embedding,
            )
        )
        result = await self._client.command(self._database, "sql", sql, params)
        return _rid(result)

    async def create_image_chunk_vertex(
        self,
        chunk_id: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Create an ImageChunk vertex, optionally folding in the embedding."""
        sql, params = _build_image_chunk_sql(
            ImageChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                properties=properties or {},
                embedding=embedding,
            )
        )
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
        if not self._enforce_relationship_triple(
            record.from_type, record.rel_type, record.to_type,
        ):
            return ""

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
        """Create multiple relationships in one sqlscript call. Returns their RIDs."""
        if not records:
            return []
        valid_records = [
            r for r in records
            if self._enforce_relationship_triple(r.from_type, r.rel_type, r.to_type)
        ]
        if not valid_records:
            return []
        script, params = _build_upsert_relationship_script(valid_records, provenance)
        result = await self._client.command(
            self._database, "sqlscript", script, params,
        )
        return _extract_rids(result, len(valid_records))

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
        """Resolve a name to its canonical root entity.

        When *entity_type* is provided, queries that specific type to leverage
        type-specific indexes.  Falls back to ``V`` when type is unknown.
        """
        params: dict[str, Any] = {"name": name}
        source = entity_type if entity_type else "V"
        type_filter = ""
        if entity_type:
            type_filter = " AND entity_type = :entity_type"
            params["entity_type"] = entity_type

        sql = (
            f"SELECT *, @rid AS node_id FROM {source} "
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
        """Return ontology-linked chunks for a given chunk.

        The traversal is: chunk ←(EXTRACTED_FROM)— entities —(EXTRACTED_FROM)→ other chunks.
        ``in('EXTRACTED_FROM')`` gives entities that point to this chunk, then
        ``out('EXTRACTED_FROM')`` from those entities gives the other chunks they mention.

        *node_id* may be an ArcadeDB RID (``#10:5``) or a chunk UUID string.
        If it's a UUID, we resolve it to a RID first.
        """
        rid = node_id
        # If node_id looks like a UUID rather than a RID, resolve it
        if not node_id.startswith("#"):
            lookup = await self._client.query(
                self._database, "sql",
                "SELECT @rid FROM TextChunk WHERE chunk_id = :cid LIMIT 1",
                {"cid": node_id},
            )
            if not lookup:
                # Try ImageChunk
                lookup = await self._client.query(
                    self._database, "sql",
                    "SELECT @rid FROM ImageChunk WHERE chunk_id = :cid LIMIT 1",
                    {"cid": node_id},
                )
            if lookup and isinstance(lookup[0], dict):
                rid = str(lookup[0].get("@rid", node_id))
            else:
                return []

        # Traverse: chunk ← entities → other chunks
        sql = (
            f"SELECT expand(out('EXTRACTED_FROM')) "
            f"FROM (SELECT expand(in('EXTRACTED_FROM')) FROM {rid})"
        )
        rows = await self._client.query(self._database, "sql", sql)
        # Exclude the original chunk from results
        return [r for r in rows if str(r.get("@rid", "")) != rid]

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return backend-level graph statistics."""
        v_sql = "SELECT count(*) AS cnt FROM V"
        e_sql = "SELECT count(*) AS cnt FROM E"
        v_rows, e_rows = await asyncio.gather(
            self._client.query(self._database, "sql", v_sql),
            self._client.query(self._database, "sql", e_sql),
        )
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
        """Create an Alias vertex (upsert) and HAS_ALIAS edge to the target node."""
        sql = (
            "UPDATE Alias SET alias_name = :alias, created_at = sysdate() "
            "UPSERT WHERE alias_name = :alias"
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

        # Filter entity_type on the linked entity (via traversal), not the
        # Alias vertex itself. Alias vertices don't have entity_type.
        sql = (
            "SELECT expand(in('HAS_ALIAS')) FROM Alias "
            f"WHERE alias_name = :alias"
        )
        rows = await self._client.query(self._database, "sql", sql, params)
        results = [_to_entity(r) for r in rows]
        if entity_type:
            results = [r for r in results if r.entity_type == entity_type]
        return results

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
        vertex_type: str,
        embedding_property: str,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[GraphEntityResult]:
        """ANN search over node embeddings via vectorNeighbors.

        Parameters
        ----------
        vertex_type:
            The ArcadeDB vertex type to search (e.g. "TextChunk", "ImageChunk").
        embedding_property:
            The embedding property on that vertex type (e.g. "text_embedding").
        query_vector:
            The query embedding vector.
        top_k:
            Maximum number of results.
        score_threshold:
            Optional minimum similarity score filter.
        """
        index_name = f"{vertex_type}[{embedding_property}]"
        # Use SELECT * so chunk metadata (chunk_id, document_id, artifact_id,
        # modality, text) flows through to retrieval callers. The embedding
        # vector itself is excluded by _to_entity()'s property filter.
        sql = (
            "SELECT *, $distance, @rid AS node_id "
            f"FROM (SELECT expand(vectorNeighbors('{index_name}', "
            ":query_vector, :top_k)))"
        )
        params: dict[str, Any] = {
            "query_vector": query_vector,
            "top_k": top_k,
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
        vertex_type: str,
        vertex_id: str,
        embedding_property: str,
        embedding: list[float],
    ) -> None:
        """Attach a vector embedding to a node."""
        params: dict[str, Any] = {"embedding": embedding}

        sql = (
            f"UPDATE {vertex_id} SET {embedding_property} = :embedding, "
            f"updated_at = sysdate()"
        )
        await self._client.command(self._database, "sql", sql, params)

    async def set_vertex_embedding_by_chunk_id(
        self,
        chunk_id: str,
        embedding: list[float],
        model_name: str | None = None,
    ) -> None:
        """Update embedding on a TextChunk vertex looked up by chunk_id."""
        model_set = ""
        params: dict[str, Any] = {"embedding": embedding, "chunk_id": chunk_id}
        if model_name:
            model_set = ", embedding_model = :model_name"
            params["model_name"] = model_name

        sql = (
            f"UPDATE TextChunk SET text_embedding = :embedding{model_set}, "
            f"updated_at = sysdate() WHERE chunk_id = :chunk_id"
        )
        await self._client.command(self._database, "sql", sql, params)

    async def cross_model_search(
        self,
        text_embedding: list[float],
        image_embedding: list[float],
        limit: int = 10,
    ) -> list[GraphEntityResult]:
        """Search using both text and image embeddings + graph traversal."""
        text_index = "TextChunk[text_embedding]"
        image_index = "ImageChunk[image_embedding]"
        sql = (
            "SELECT chunk.*, entity.name AS entity_name, "
            "entity.entity_type AS entity_entity_type "
            "FROM ("
            f"  SELECT expand(vectorNeighbors('{text_index}', "
            "  :text_vector, :top_k))"
            ") AS chunk "
            "LET entity = chunk.in('EXTRACTED_FROM')"
        )
        img_sql = (
            "SELECT name, entity_type, canonical_name, extraction_confidence, "
            "$distance, @rid AS node_id "
            f"FROM (SELECT expand(vectorNeighbors('{image_index}', "
            ":image_vector, :top_k)))"
        )
        text_rows, img_rows = await asyncio.gather(
            self._client.query(self._database, "sql", sql, {
                "text_vector": text_embedding,
                "top_k": limit,
            }),
            self._client.query(self._database, "sql", img_sql, {
                "image_vector": image_embedding,
                "top_k": limit,
            }),
        )

        combined = text_rows + img_rows
        return [_to_entity(r) for r in combined[:limit]]

    # ==================================================================
    # Community operations
    # ==================================================================

    async def run_community_algorithm(
        self,
        algorithm: str,
        params: dict,
    ) -> list[dict]:
        """Run a community detection algorithm and return per-node results."""
        algo_params = ", ".join(f"{k}: {v}" for k, v in params.items())
        cypher = (
            f"CALL algo.{algorithm}({{{algo_params}}}) YIELD node, communityId "
            f"RETURN node.name AS name, node.entity_type AS entity_type, "
            f"communityId AS community_id"
        )
        return await self._client.query(self._database, "cypher", cypher)

    async def get_community_reports(self) -> list[dict]:
        """Return all existing CommunityReport rows."""
        return await self._client.query(
            self._database, "sql",
            "SELECT community_id, membership_hash FROM CommunityReport",
        )

    async def upsert_community_report(
        self,
        community_id: int,
        title: str,
        summary: str,
        member_count: int,
        membership_hash: str,
        model_name: str,
    ) -> str:
        """Upsert a CommunityReport vertex and return its RID."""
        sql = (
            "UPDATE CommunityReport SET title = :title, summary = :summary, "
            "member_count = :count, membership_hash = :hash, "
            "generated_at = sysdate(), model_name = :model "
            "UPSERT WHERE community_id = :cid"
        )
        result = await self._client.command(
            self._database, "sql", sql,
            params={
                "cid": community_id,
                "title": title,
                "summary": summary,
                "count": member_count,
                "hash": membership_hash,
                "model": model_name,
            },
        )
        return _rid(result)

    async def list_community_reports(
        self,
        limit: int = 100,
    ) -> list[dict]:
        """Return community reports (summary view) up to *limit*."""
        return await self._client.query(
            self._database, "sql",
            "SELECT community_id, title, summary, member_count, generated_at "
            "FROM CommunityReport ORDER BY community_id LIMIT :limit",
            params={"limit": limit},
        )

    async def get_community_report(
        self,
        community_id: int,
    ) -> dict | None:
        """Return a single community report by community_id, or None."""
        rows = await self._client.query(
            self._database, "sql",
            "SELECT * FROM CommunityReport WHERE community_id = :cid",
            params={"cid": community_id},
        )
        return rows[0] if rows else None

    async def search_community_reports_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """ANN search over CommunityReport.report_embedding."""
        sql = (
            "SELECT community_id, title, summary, member_count, "
            "generated_at, model_name, $distance, @rid AS report_rid "
            "FROM (SELECT expand(vectorNeighbors('CommunityReport[report_embedding]', "
            ":query_vector, :top_k)))"
        )
        rows = await self._client.query(
            self._database, "sql", sql,
            params={"query_vector": query_vector, "top_k": top_k},
        )
        # Map $distance → score (similarity = 1 - cosine_distance)
        for r in rows:
            dist = r.pop("$distance", r.pop("distance", None))
            r["score"] = (1.0 - float(dist)) if dist is not None else 0.0
        return rows

    async def get_relationships_between_entities(
        self,
        entity_names: list[str],
    ) -> list[dict]:
        """Return ontology edges where both endpoints are in *entity_names*.

        Excludes structural edges (EXTRACTED_FROM, CONTAINS_TEXT, etc.) —
        only returns domain relationships between community members.
        """
        if not entity_names:
            return []
        from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES
        structural = set(_STRUCTURAL_EDGE_TYPES)

        sql = (
            "SELECT @class AS rel_type, "
            "out.name AS from_name, out.@class AS from_type, "
            "in.name AS to_name, in.@class AS to_type "
            "FROM (SELECT expand(outE()) FROM V WHERE name IN :names) "
            "WHERE in.name IN :names"
        )
        rows = await self._client.query(
            self._database, "sql", sql,
            params={"names": entity_names},
        )
        return [r for r in rows if r.get("rel_type") not in structural]

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

        sql = "DELETE VERTEX FROM TextChunk WHERE document_id = :doc_id"
        result = await self._client.command(self._database, "sql", sql, params)
        total += _count(result)

        sql = "DELETE VERTEX FROM ImageChunk WHERE document_id = :doc_id"
        result = await self._client.command(self._database, "sql", sql, params)
        total += _count(result)

        sql = "DELETE VERTEX FROM Document WHERE document_id = :doc_id"
        result = await self._client.command(self._database, "sql", sql, params)
        total += _count(result)

        await self._client.command(self._database, "sql",
            "UPDATE E SET document_ids = document_ids.remove(:doc_id) "
            "WHERE document_ids CONTAINS :doc_id",
            params=params,
        )
        await self._client.command(self._database, "sql",
            "DELETE EDGE WHERE document_ids IS NOT NULL AND document_ids.size() = 0",
        )

        # Orphan cleanup: remove entities with no remaining EXTRACTED_FROM edges
        orphan_sql = (
            "DELETE VERTEX FROM V WHERE @class NOT IN "
            "['Document', 'TextChunk', 'ImageChunk', 'Alias'] "
            "AND out('EXTRACTED_FROM').size() = 0 "
            "AND in('EXTRACTED_FROM').size() = 0"
        )
        result = await self._client.command(self._database, "sql", orphan_sql)
        total += _count(result)

        return total

    async def sync_schema(
        self,
        ontology: dict | None = None,
    ) -> SchemaSyncReport:
        """Ensure the backend schema matches the current ontology.

        If *ontology* is provided, delegates to the full ontology-driven
        ``sync_schema_from_ontology``.  Otherwise falls back to creating
        the core structural types only.  Invalidates the cached validation
        matrix so the next relationship write picks up any ontology changes.
        """
        self._validation_matrix = None
        if ontology is not None:
            from app.services.arcadedb_schema import sync_schema_from_ontology
            return await sync_schema_from_ontology(
                self._client, self._database, ontology,
            )

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
            "CREATE INDEX IF NOT EXISTS ON Document (document_id) UNIQUE_HASH_INDEX",
            "CREATE INDEX IF NOT EXISTS ON TextChunk (chunk_id) UNIQUE_HASH_INDEX",
            "CREATE INDEX IF NOT EXISTS ON TextChunk (document_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON ImageChunk (chunk_id) UNIQUE_HASH_INDEX",
            "CREATE INDEX IF NOT EXISTS ON ImageChunk (document_id) NOTUNIQUE",
        ]
        for sql in indexes:
            try:
                await self._client.command(self._database, "sql", sql)
            except Exception as e:
                logger.warning("Index creation failed: %s", e)

    async def ensure_ready(self) -> None:
        """Block until the backend is ready, with retry/backoff.

        Uses a time-based cache so repeated calls within the same pipeline
        (e.g., 5 graph-writing stages per document) skip the probe.
        """
        from app.config import get_settings as _gs
        ttl = _gs().arcadedb_ready_cache_seconds
        if time.monotonic() - self._last_ready_check < ttl:
            logger.debug("ArcadeDB ready (cached)")
            return

        last_exc: Exception | None = None
        for attempt in range(_READY_MAX_RETRIES):
            try:
                await self._client.query(
                    self._database, "sql", "SELECT name FROM schema:database",
                )
                self._last_ready_check = time.monotonic()
                logger.info("ArcadeDB ready (attempt %d)", attempt + 1)
                return
            except Exception as exc:
                self._last_ready_check = 0
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
            f"CREATE EDGE EXTRACTED_FROM FROM {node_rid} "
            "TO (SELECT FROM Document WHERE document_id = :document_id) "
            "SET page_numbers = :page_numbers, created_at = sysdate()"
        )
        params = {
            "document_id": provenance.document_id,
            "page_numbers": provenance.page_numbers,
        }
        self._client.command_sync(self._database, "sql", sql, params)

    def upsert_nodes_batch_sync(
        self,
        records: list[NodeRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Upsert a batch of nodes in one sqlscript call. Returns their RIDs."""
        if not records:
            return []
        script, params = _build_upsert_node_script(records)
        result = self._client.command_sync(
            self._database, "sqlscript", script, params,
        )
        rids = _extract_rids(result, len(records))
        if provenance:
            self._create_provenance_edges_batch_sync(rids, provenance)
        return rids

    def _create_provenance_edges_batch_sync(
        self,
        node_rids: list[str],
        provenance: ProvenanceMetadata,
    ) -> None:
        """Create EXTRACTED_FROM edges for a batch of nodes in one sqlscript call."""
        targets = [rid for rid in node_rids if rid]
        if not targets:
            return
        statements: list[str] = []
        params: dict[str, Any] = {
            "document_id": provenance.document_id,
            "page_numbers": provenance.page_numbers,
        }
        for rid in targets:
            statements.append(
                f"CREATE EDGE EXTRACTED_FROM FROM {rid} "
                f"TO (SELECT FROM Document WHERE document_id = :document_id) "
                f"SET page_numbers = :page_numbers, created_at = sysdate()"
            )
        self._client.command_sync(
            self._database, "sqlscript", ";\n".join(statements), params,
        )

    def upsert_relationships_batch_sync(
        self,
        records: list[RelationshipRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Create a batch of relationships in one sqlscript call. Returns their RIDs."""
        if not records:
            return []
        valid_records = [
            r for r in records
            if self._enforce_relationship_triple(r.from_type, r.rel_type, r.to_type)
        ]
        if not valid_records:
            return []
        script, params = _build_upsert_relationship_script(valid_records, provenance)
        result = self._client.command_sync(
            self._database, "sqlscript", script, params,
        )
        return _extract_rids(result, len(valid_records))

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
        vertex_type: str,
        vertex_id: str,
        embedding_property: str,
        embedding: list[float],
    ) -> None:
        """Synchronous embedding set."""
        params: dict[str, Any] = {"embedding": embedding}

        sql = (
            f"UPDATE {vertex_id} SET {embedding_property} = :embedding, "
            f"updated_at = sysdate()"
        )
        self._client.command_sync(self._database, "sql", sql, params)

    def create_text_chunk_vertex_sync(
        self,
        chunk_id: str,
        text: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Create a TextChunk vertex, optionally folding in the embedding."""
        sql, params = _build_text_chunk_sql(
            TextChunkRecord(
                chunk_id=chunk_id,
                text=text,
                document_id=document_id,
                properties=properties or {},
                embedding=embedding,
            )
        )
        result = self._client.command_sync(self._database, "sql", sql, params)
        return _rid(result)

    def create_image_chunk_vertex_sync(
        self,
        chunk_id: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Create an ImageChunk vertex, optionally folding in the embedding."""
        sql, params = _build_image_chunk_sql(
            ImageChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                properties=properties or {},
                embedding=embedding,
            )
        )
        result = self._client.command_sync(self._database, "sql", sql, params)
        return _rid(result)

    def create_text_chunks_batch_sync(
        self,
        records: list[TextChunkRecord],
    ) -> list[str]:
        """Create a batch of TextChunk vertices (with embeddings) in one sqlscript call."""
        if not records:
            return []
        statements: list[str] = []
        all_params: dict[str, Any] = {}
        for i, record in enumerate(records):
            sql, params = _build_text_chunk_sql(record, row_idx=i)
            statements.append(sql)
            all_params.update(params)
        result = self._client.command_sync(
            self._database, "sqlscript", ";\n".join(statements), all_params,
        )
        return _extract_rids(result, len(records))

    def create_image_chunks_batch_sync(
        self,
        records: list[ImageChunkRecord],
    ) -> list[str]:
        """Create a batch of ImageChunk vertices (with embeddings) in one sqlscript call."""
        if not records:
            return []
        statements: list[str] = []
        all_params: dict[str, Any] = {}
        for i, record in enumerate(records):
            sql, params = _build_image_chunk_sql(record, row_idx=i)
            statements.append(sql)
            all_params.update(params)
        result = self._client.command_sync(
            self._database, "sqlscript", ";\n".join(statements), all_params,
        )
        return _extract_rids(result, len(records))

    def batch_create_entity_chunk_edges_sync(
        self,
        edges: list[EntityChunkEdge],
    ) -> int:
        """Create EXTRACTED_FROM edges from entities (by name+type) to chunk RIDs in one call.

        Each statement uses an inline subquery to resolve the entity, so rows
        where the entity is missing become no-ops without failing the batch.
        """
        if not edges:
            return 0
        statements: list[str] = []
        params: dict[str, Any] = {}
        for i, edge in enumerate(edges):
            params[f"name_{i}"] = edge.entity_name
            params[f"etype_{i}"] = edge.entity_type
            params[f"rid_{i}"] = edge.chunk_rid
            statements.append(
                f"CREATE EDGE EXTRACTED_FROM "
                f"FROM (SELECT FROM {edge.entity_type} "
                f"WHERE name = :name_{i} AND entity_type = :etype_{i} LIMIT 1) "
                f"TO :rid_{i} SET created_at = sysdate()"
            )
        self._client.command_sync(
            self._database, "sqlscript", ";\n".join(statements), params,
        )
        return len(edges)

    def get_chunk_rid_sync(
        self,
        chunk_id: str,
        chunk_type: str = "TextChunk",
    ) -> str | None:
        """Return the ArcadeDB RID for an existing chunk vertex, or None if not found."""
        sql = f"SELECT @rid FROM {chunk_type} WHERE chunk_id = :chunk_id LIMIT 1"
        result = self._client.query_sync(self._database, "sql", sql, {"chunk_id": chunk_id})
        if result and isinstance(result, list) and len(result) > 0:
            row = result[0]
            if isinstance(row, dict):
                return row.get("@rid") or row.get("rid")
        return None

    def create_entity_chunk_edge_sync(
        self,
        entity_name: str,
        entity_type: str,
        chunk_rid: str,
    ) -> bool:
        """Create an EXTRACTED_FROM edge from an entity to a chunk vertex.

        Looks up the entity by name+type, then creates the edge to the chunk
        identified by its ArcadeDB RID.  Returns True if the edge was created,
        False if the entity vertex could not be found.
        """
        entity = self.resolve_root_entity_sync(entity_name, entity_type)
        if not entity:
            return False
        sql = (
            f"CREATE EDGE EXTRACTED_FROM FROM {entity.node_id} TO {chunk_rid} "
            f"SET created_at = sysdate()"
        )
        self._client.command_sync(self._database, "sql", sql)
        return True

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

        # Remove document_id from relationship edge document_ids lists
        # Delete edges where document_ids becomes empty
        self._client.command_sync(self._database, "sql",
            "UPDATE E SET document_ids = document_ids.remove(:doc_id) "
            "WHERE document_ids CONTAINS :doc_id",
            params,
        )
        self._client.command_sync(self._database, "sql",
            "DELETE EDGE WHERE document_ids IS NOT NULL AND document_ids.size() = 0",
        )

        orphan_sql = (
            "DELETE VERTEX FROM V WHERE @class NOT IN "
            "['Document', 'TextChunk', 'ImageChunk', 'Alias'] "
            "AND out('EXTRACTED_FROM').size() = 0 "
            "AND in('EXTRACTED_FROM').size() = 0"
        )
        result = self._client.command_sync(self._database, "sql", orphan_sql)
        total += _count(result)

        return total

    def ensure_ready_sync(self) -> None:
        """Synchronous ensure_ready with time-based caching."""
        from app.config import get_settings as _gs
        ttl = _gs().arcadedb_ready_cache_seconds
        if time.monotonic() - self._last_ready_check < ttl:
            logger.debug("ArcadeDB ready (cached)")
            return

        last_exc: Exception | None = None
        for attempt in range(_READY_MAX_RETRIES):
            try:
                self._client.query_sync(
                    self._database, "sql", "SELECT name FROM schema:database",
                )
                self._last_ready_check = time.monotonic()
                logger.info("ArcadeDB ready (attempt %d)", attempt + 1)
                return
            except Exception as exc:
                self._last_ready_check = 0
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
        params: dict[str, Any] = {"alias": alias}

        sql = (
            "SELECT expand(in('HAS_ALIAS')) FROM Alias "
            "WHERE alias_name = :alias"
        )
        rows = self._client.query_sync(self._database, "sql", sql, params)
        results = [_to_entity(r) for r in rows]
        if entity_type:
            results = [r for r in results if r.entity_type == entity_type]
        return results

    def create_alias_sync(
        self,
        node_id: str,
        alias: str,
    ) -> None:
        """Synchronous alias creation (upsert)."""
        sql = (
            "UPDATE Alias SET alias_name = :alias, created_at = sysdate() "
            "UPSERT WHERE alias_name = :alias"
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
        """Synchronous root entity resolution.

        When *entity_type* is provided, queries that specific type to leverage
        type-specific indexes.  Falls back to ``V`` when type is unknown.
        """
        params: dict[str, Any] = {"name": name}
        source = entity_type if entity_type else "V"
        type_filter = ""
        if entity_type:
            type_filter = " AND entity_type = :entity_type"
            params["entity_type"] = entity_type

        sql = (
            f"SELECT *, @rid AS node_id FROM {source} "
            f"WHERE name = :name{type_filter} LIMIT 1"
        )
        rows = self._client.query_sync(self._database, "sql", sql, params)
        return _to_entity(rows[0]) if rows else None

    def get_document_entities_sync(
        self,
        document_id: str,
    ) -> list[dict]:
        """Return domain entities linked to a document via graph traversal.

        Traversal: Document →(CONTAINS_TEXT)→ TextChunk ←(EXTRACTED_FROM)← Entity
        """
        sql = (
            "SELECT DISTINCT name, entity_type, @rid AS node_id "
            "FROM ("
            "  SELECT expand(in('EXTRACTED_FROM')) "
            "  FROM ("
            "    SELECT expand(out('CONTAINS_TEXT')) "
            "    FROM Document WHERE document_id = :doc_id"
            "  )"
            ")"
        )
        return self._client.query_sync(
            self._database, "sql", sql, {"doc_id": document_id},
        )

    def close_sync(self) -> None:
        """Release any held resources."""
        self._client.close_sync()
