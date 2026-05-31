"""ArcadeDB implementation of the GraphStore Protocol.

Maps every Protocol method to ArcadeDB SQL executed via ArcadeDBClient.
Uses ``command()`` for writes and ``query()`` for reads, with sync variants
for Celery workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.services.arcadedb_client import ArcadeDBClient
from app.services.arcadedb_schema import _safe_type_name
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

# Durability re-query retry (Task 3). The node-durability assertion reads back
# just-upserted RIDs over a SEPARATE HTTP session; ArcadeDB's
# asyncOperationsQueueSize=2048 (docker-compose.yml) lets that read race ahead
# of the async flush, so a short count can be transient. Mirror the
# batch_create_entity_chunk_edges_sync read-after-write retry: a few attempts
# with linear backoff before treating a short count as a real durability fail.
_DURABILITY_MAX_ATTEMPTS = 3
_DURABILITY_BACKOFF_BASE = 0.2  # seconds

# Local mirror — kept consistent with app.services.query_profiles via
# the Task 15 sync contract test. section_properties profiles dispatch
# off this set when traversing related systems.
_CANONICAL_ROOT_ENTITY_TYPES_RUNTIME: frozenset[str] = frozenset({
    "RADAR_SYSTEM", "MISSILE_SYSTEM",
})


class UpsertMissingRIDError(RuntimeError):
    """Raised when ``upsert_nodes_batch_sync`` completes the sqlscript but
    one or more records produce no ``@rid`` in the result.

    This is a genuine server-side failure (e.g. the UPSERT's INSERT branch
    hit a schema constraint we didn't anticipate, or the response was
    truncated). It's distinct from the caller-side "malformed identity"
    case, which is caught earlier by entry-validation and raises
    ``ValueError`` instead. Surfacing the @rid gap as an exception at the
    upsert boundary — rather than silently padding with ``""`` — prevents
    downstream edge creation from f-stringing the blank into a
    syntactically invalid CREATE EDGE statement (see the historical
    fix at pipeline.py:4352-4374 that guarded the edge loop after this
    class of bug kept reaching production).
    """


class UpsertNotDurableError(RuntimeError):
    """Raised when ``upsert_nodes_batch_sync`` returns ``@rid`` values for
    every record yet a post-upsert re-query shows fewer rows are actually
    committed/queryable than were written.

    This converts the SILENT zero-commit documented in the Task 0 lineage
    diagnostic (reports/collection/lineage_diagnostic_findings.md) into a
    loud failure. In that incident the SA-2 merge upserted 22 entities,
    ``upsert_nodes_batch_sync`` returned 22 RIDs (so the RID-presence sanity
    check — ``UpsertMissingRIDError`` — passed), the merge phase was marked
    ``result='succeeded'``, and yet 0 of the 22 entities were queryable in
    ArcadeDB afterward. RID presence is therefore NECESSARY BUT NOT
    SUFFICIENT: a returned RID does not guarantee a durably committed,
    re-queryable row. The durability re-query that raises this error is the
    safety net that guarantees a zero/partial-commit can never again report
    success — it raises here so the caller's existing rollback/FAILED path
    (``derive_ontology_graph_merge``) fires instead of marking the merge
    succeeded.

    Distinct from ``UpsertMissingRIDError`` (blank @rid in the response) and
    the entry-validation ``ValueError`` (malformed identity). Those fire
    BEFORE this check; this one fires only when RIDs look valid but the rows
    don't survive a read-back.
    """


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


def _truncate_rid_list(rids: list[str], sample: int = 10) -> str:
    """Render a RID list for error messages, capped to a small sample.

    A large upsert batch otherwise dumps multi-KB of RIDs into the exception
    text. Show the first ``sample`` and a ``+N more`` tail.
    """
    if len(rids) <= sample:
        return str(rids)
    head = rids[:sample]
    return f"{head} ...+{len(rids) - sample} more"


def _to_entity(row: dict[str, Any]) -> GraphEntityResult:
    """Map an ArcadeDB row dict to a GraphEntityResult.

    Score priority (native ArcadeDB semantics preserved):
    1. ``$distance`` from vectorNeighbors → converted to similarity (1 - distance)
    2. ``$score`` from Lucene FULL_TEXT search → native BM25 relevance score
    3. ``extraction_confidence`` → ingestion-time confidence from the extraction pipeline
    """
    distance = row.get("$distance", row.get("distance"))
    lucene_score = row.get("$score")

    # Map native ArcadeDB scores with type tracking
    score: float | None = None
    score_type: str | None = None
    if distance is not None:
        score = 1.0 - float(distance)
        score_type = "vector"
    elif lucene_score is not None:
        score = float(lucene_score)
        score_type = "fulltext"

    # extraction_confidence: prefer native score, fall back to ingestion confidence
    extraction_confidence = score if score is not None else row.get("extraction_confidence")

    props = {k: v for k, v in row.items() if k not in (
        "@rid", "name", "entity_type", "canonical_name", "extraction_confidence",
        "@type", "@cat", "@class", "$distance", "distance", "$score",
        "text_embedding", "image_embedding", "report_embedding",
        "embedding",  # VR C.1: ExtractionChunk.embedding (1024-d float array) — strip from properties dict per the existing pattern; avoid 4KB-per-result bandwidth on chunk-scope retrievals.
    )}
    return GraphEntityResult(
        node_id=str(row.get("@rid", row.get("node_id", ""))),
        name=row.get("name", ""),
        entity_type=row.get("entity_type", ""),
        canonical_name=row.get("canonical_name"),
        extraction_confidence=extraction_confidence,
        score=score,
        score_type=score_type,
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

    Vertex class names are routed through ``_safe_type_name`` so ArcadeDB
    reserved words (``TABLE`` → ``TABLE_REF``) resolve to the class
    actually created in the schema. Without this, an UPSERT against
    ``TABLE`` would either fail parsing or hit the wrong class.
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

        # ArcadeDB sqlscript returns only the LAST statement's result from
        # a plain ";"-joined batch (manual, "SQL Script" section, p. 275).
        # To collect per-statement @rid values we must bind each UPSERT to
        # a LET variable and append a RETURN [$v0,$v1,...] at the end —
        # otherwise 17/18 records silently come back with no @rid and the
        # caller has no way to wire edges to them. See the historical
        # incident that introduced UpsertMissingRIDError for context.
        sql = (
            f"LET v{i} = UPDATE {_safe_type_name(record.entity_type)} "
            f"SET {set_clause}, updated_at = sysdate() "
            f"UPSERT RETURN AFTER @rid WHERE {where_clause} AND entity_type = :entity_type_{i}"
        )
        statements.append(sql)

    # Collect every LET-bound value in a single RETURN so the HTTP response
    # carries one result row per input record (each wrapped in "value": [{@rid}]).
    return_clause = "RETURN [" + ", ".join(f"$v{i}" for i in range(len(records))) + "]"
    script = ";\n".join(statements) + ";\n" + return_clause
    return script, params


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
        if record.provenance is not None:
            prov_key = f"provenance_{i}"
            params[prov_key] = json.dumps(record.provenance)
            extra_parts.append(f"provenance = :{prov_key}")
        extra = (", " + ", ".join(extra_parts)) if extra_parts else ""

        doc_ids_expr = f"[:{doc_id_param_key}]" if doc_id_param_key else "[]"

        # Find-or-create pattern. Plain CREATE EDGE would produce
        # duplicate parallel edges on re-ingest (Doc B extracts the same
        # Fan Song → SA-2 pairing that Doc A already created, and
        # CREATE EDGE has no built-in uniqueness constraint against
        # (@out, rel, @in)). We resolve endpoint RIDs first, then check
        # for an existing edge between them; if found we UPDATE — adding
        # this document's id to document_ids when it's not already present
        # — otherwise we CREATE with the initial document_ids list.
        #
        # ArcadeDB sqlscript IF/ELSE is scope-flat, so each branch must
        # bind v{i} to a result vector of the same shape ([{@rid: ...}])
        # for the trailing RETURN to aggregate correctly. See manual
        # p. 275 (multi-statement sqlscript) + the _build_upsert_node_script
        # companion fix.
        rtype = _safe_type_name(record.rel_type)
        safe_from = _safe_type_name(record.from_type)
        safe_to = _safe_type_name(record.to_type)
        from_clause = " AND ".join(from_where_parts)
        to_clause = " AND ".join(to_where_parts)

        if doc_id_param_key:
            # Union doc_id only when the edge exists AND doc_id is not
            # already in the list, preventing duplicate entries on
            # repeat ingest of the same doc.
            update_sql = (
                f"  UPDATE {rtype} "
                f"SET document_ids = document_ids || [:{doc_id_param_key}], "
                f"updated_at = sysdate(){extra} "
                f"WHERE @out = $src_{i}[0].@rid AND @in = $dst_{i}[0].@rid "
                f"AND NOT (document_ids CONTAINS :{doc_id_param_key})"
            )
        else:
            update_sql = (
                f"  UPDATE {rtype} SET updated_at = sysdate(){extra} "
                f"WHERE @out = $src_{i}[0].@rid AND @in = $dst_{i}[0].@rid"
            )

        sql = (
            f"LET src_{i} = SELECT FROM {safe_from} WHERE {from_clause};\n"
            f"LET dst_{i} = SELECT FROM {safe_to} WHERE {to_clause};\n"
            f"LET existing_{i} = SELECT @rid FROM {rtype} "
            f"WHERE @out = $src_{i}[0].@rid AND @in = $dst_{i}[0].@rid;\n"
            f"IF ($existing_{i}.size() = 0) {{\n"
            f"  LET v{i} = CREATE EDGE {rtype} "
            f"FROM $src_{i}[0] TO $dst_{i}[0] "
            f"SET extraction_confidence = :{conf_key}, "
            f"document_ids = {doc_ids_expr}, "
            f"created_at = sysdate(), updated_at = sysdate(){extra} "
            f"RETURN @rid;\n"
            f"}} ELSE {{\n"
            f"{update_sql};\n"
            f"  LET v{i} = $existing_{i};\n"
            f"}}"
        )
        statements.append(sql)

    return_clause = "RETURN [" + ", ".join(f"$v{i}" for i in range(len(records))) + "]"
    script = ";\n".join(statements) + ";\n" + return_clause
    return script, params


def _build_text_chunk_sql(
    record: TextChunkRecord,
    row_idx: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build an idempotent UPDATE…UPSERT for a TextChunk, folding in embedding.

    Uses UPSERT keyed on the UNIQUE ``chunk_id`` index so repeat calls (e.g.
    Celery retries of ``derive_text_chunks_and_embeddings`` after a transient
    ArcadeDB error) replace the existing vertex instead of throwing a
    duplicate-key 503. When ``row_idx`` is provided, param keys are suffixed
    with ``_{row_idx}`` so the statement can be combined in a sqlscript batch.
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
        f"UPDATE TextChunk SET {', '.join(set_parts)}, updated_at = sysdate() "
        f"UPSERT RETURN AFTER @rid WHERE chunk_id = :chunk_id{suffix}"
    )
    return sql, params


def _build_image_chunk_sql(
    record: ImageChunkRecord,
    row_idx: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build an idempotent UPDATE…UPSERT for an ImageChunk, folding in embedding.

    Uses UPSERT keyed on the UNIQUE ``chunk_id`` index so repeat calls (e.g.
    Celery retries of ``derive_image_embeddings`` after a transient ArcadeDB
    error) replace the existing vertex instead of throwing a duplicate-key
    503.
    """
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
        f"UPDATE ImageChunk SET {', '.join(set_parts)}, updated_at = sysdate() "
        f"UPSERT RETURN AFTER @rid WHERE chunk_id = :chunk_id{suffix}"
    )
    return sql, params


def _extract_rids(result: list, expected: int) -> list[str]:
    """Extract a list of RIDs from a sqlscript result, padding to *expected* length.

    Handles two response shapes:

      1. ``LET vN = ... ; RETURN [$v0, $v1, ...]`` (the multi-statement
         pattern used by ``_build_upsert_node_script`` +
         ``_build_upsert_relationship_script``) → each row in ``result``
         looks like ``{"value": [{"@rid": "#4:13", ...}]}``. We unwrap the
         ``value`` list and read the first element's ``@rid``.

      2. Direct ``[{"@rid": "#4:13"}]`` rows — single-statement callers
         that still go through ``_extract_rids`` without the LET/RETURN
         wrapping.

    ArcadeDB's sqlscript endpoint returns a flat list of result rows, one per
    statement. This helper maps them back to the input order, filling missing
    entries with empty strings so callers always get a same-length list.
    """
    rids: list[str] = [""] * expected
    if not isinstance(result, list):
        return rids
    for i, row in enumerate(result[:expected]):
        if not isinstance(row, dict):
            continue
        # Shape 1 — LET/RETURN pattern: {"value": [{"@rid": "..."}]}.
        value = row.get("value")
        if isinstance(value, list) and value and isinstance(value[0], dict):
            rid = value[0].get("@rid")
            if rid:
                rids[i] = str(rid)
                continue
        # Shape 2 — direct @rid on the row (legacy single-statement path).
        rid = row.get("@rid")
        if rid:
            rids[i] = str(rid)
    return rids


def _build_structural_edge_sql(
    from_id: str,
    to_id: str,
    rel_type: str,
    properties: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Build a CREATE EDGE statement for a structural edge between two RIDs."""
    props = dict(properties or {})
    extra_set = ""
    if props:
        extra_set = ", " + _build_set(props)

    sql = (
        f"CREATE EDGE {rel_type} FROM {from_id} TO {to_id} "
        f"SET created_at = sysdate(){extra_set}"
    )
    return sql, props or None


def _build_resolve_root_entity_sql(
    name: str,
    entity_type: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a SELECT query that resolves a name to its canonical root entity."""
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
    return sql, params


def _build_fulltext_search_sql(
    query: str,
    entity_types: list[str] | None = None,
    limit: int = 20,
) -> tuple[str, dict[str, Any]] | tuple[list[str], dict[str, Any]]:
    """Build full-text search query/queries.

    ArcadeDB doesn't have an abstract 'V' base type — queries must target
    concrete vertex types. For a single type, returns one SQL string.
    For multiple types, returns a list of SQL strings (one per type) to
    be executed and merged by the caller.
    """
    params: dict[str, Any] = {"query": query, "limit": limit}

    types = entity_types or []
    if len(types) == 1:
        sql = (
            f"SELECT *, @rid AS node_id FROM {types[0]} "
            f"WHERE name CONTAINSTEXT :query "
            f"LIMIT :limit"
        )
        return sql, params
    elif len(types) > 1:
        # Multiple specified types — query each and merge
        sqls = [
            f"SELECT *, @rid AS node_id FROM {t} "
            f"WHERE name CONTAINSTEXT :query "
            f"LIMIT :limit"
            for t in types
        ]
        return sqls, params
    else:
        # No type filter — must query all known entity types.
        # Return a sentinel that the caller handles by loading ontology types.
        return [], params


def _build_search_by_alias_sql(
    alias: str,
) -> tuple[str, dict[str, Any]]:
    """Build a query that finds nodes by alias traversal."""
    sql = (
        "SELECT expand(in('HAS_ALIAS')) FROM Alias "
        "WHERE alias_name = :alias"
    )
    return sql, {"alias": alias}


def _build_create_alias_sql(
    alias: str,
) -> tuple[str, dict[str, Any]]:
    """Build the UPSERT statement for an Alias vertex."""
    sql = (
        "UPDATE Alias SET alias_name = :alias, created_at = sysdate() "
        "UPSERT RETURN AFTER @rid WHERE alias_name = :alias"
    )
    return sql, {"alias": alias}


def _build_create_alias_edge_sql(
    node_id: str,
    alias_rid: str,
) -> str:
    """Build the CREATE EDGE HAS_ALIAS statement."""
    return f"CREATE EDGE HAS_ALIAS FROM {node_id} TO {alias_rid}"


def _build_set_canonical_name_sql(
    node_id: str,
    canonical_name: str,
) -> tuple[str, dict[str, Any]]:
    """Build the UPDATE statement for setting canonical_name on a node."""
    sql = f"UPDATE {node_id} SET canonical_name = :canonical_name"
    return sql, {"canonical_name": canonical_name}


def _build_delete_document_graph_sql(
    document_id: str,
) -> dict[str, Any]:
    """Build the SQL statements for deleting a document's graph elements.

    ArcadeDB has no abstract V/E base types, so all queries must target
    concrete types.  We load the ontology to discover relationship and
    entity type names at runtime.

    Returns a dict with keys: delete_vertex_sqls, edge_cleanup_sqls,
    edge_prune_sqls, orphan_sqls, params.
    """
    from app.services.arcadedb_schema import _safe_type_name, _STRUCTURAL_EDGE_TYPES

    params = {"doc_id": document_id}

    # --- structural vertex deletions (always the same) ---
    delete_vertex_sqls = [
        "DELETE VERTEX FROM TextChunk WHERE document_id = :doc_id",
        "DELETE VERTEX FROM ImageChunk WHERE document_id = :doc_id",
        "DELETE VERTEX FROM Document WHERE document_id = :doc_id",
    ]

    # --- structural edge deletions (carry document_id as STRING) ---
    for etype in _STRUCTURAL_EDGE_TYPES:
        delete_vertex_sqls.append(
            f"DELETE FROM {etype} WHERE document_id = :doc_id"
        )

    # --- ontology relationship edges (carry document_ids as LIST) ---
    # Remove doc_id from the list, then prune edges with empty lists.
    try:
        from app.services.ontology_templates import load_ontology
        ontology = load_ontology()
    except Exception:
        ontology = {}

    rel_types = [r["name"] for r in ontology.get("relationship_types", [])]
    # Orphan reap targets only global-scope entities (RADAR_SYSTEM /
    # MISSILE_SYSTEM / PLATFORM / ...). Document-scoped entities (SECTION
    # / FIGURE / TABLE / IMAGE / TEXT_BLOCK) have no EXTRACTED_FROM edges
    # by design — their graph-edges are HAS_IMAGE / CHILD_OF / NEAR_TEXT /
    # HAS_SECTION — so an unconditional orphan query would delete every
    # anchor-walker vertex across EVERY document whenever any single
    # document is purged. The explicit delete_vertex_sqls at the top of
    # this builder already wipe document-scoped anchors for THIS doc via
    # document_id filters; global entities are the only class that needs
    # the post-purge "is this still referenced by any remaining doc?"
    # check.
    global_entity_types = [
        _safe_type_name(e["name"]) for e in ontology.get("entity_types", [])
        if e.get("identity_scope", "global") != "document"
    ]

    edge_cleanup_sqls = [
        f"UPDATE {rtype} SET document_ids = document_ids.remove(:doc_id) "
        f"WHERE document_ids CONTAINS :doc_id"
        for rtype in rel_types
    ]

    edge_prune_sqls = [
        f"DELETE FROM {rtype} WHERE document_ids IS NOT NULL "
        f"AND document_ids.size() = 0"
        for rtype in rel_types
    ]

    # --- orphan entity cleanup (global entities with no remaining refs) ---
    orphan_sqls = [
        f"DELETE VERTEX FROM {etype} "
        f"WHERE out('EXTRACTED_FROM').size() = 0 "
        f"AND in('EXTRACTED_FROM').size() = 0"
        for etype in global_entity_types
    ]

    return {
        "delete_vertex_sqls": delete_vertex_sqls,
        "edge_cleanup_sqls": edge_cleanup_sqls,
        "edge_prune_sqls": edge_prune_sqls,
        "orphan_sqls": orphan_sqls,
        "params": params,
    }


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

    async def _resolve_rid(self, node_id: str) -> str | None:
        """Resolve a node identifier to an ArcadeDB RID.

        If *node_id* already looks like a RID (starts with '#'), return it
        unchanged. Otherwise treat it as a UUID and look it up across
        TextChunk, ImageChunk, and entity vertex types.
        """
        if node_id.startswith("#"):
            return node_id
        for vtype in ("TextChunk", "ImageChunk", "V"):
            key = "chunk_id" if vtype != "V" else "name"
            rows = await self._client.query(
                self._database, "sql",
                f"SELECT @rid FROM {vtype} WHERE {key} = :id LIMIT 1",
                {"id": node_id},
            )
            if rows and isinstance(rows[0], dict):
                rid = rows[0].get("@rid")
                if rid:
                    return str(rid)
        return None

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
            f"UPDATE {_safe_type_name(record.entity_type)} SET {set_clause}, "
            f"updated_at = sysdate() "
            f"UPSERT RETURN AFTER @rid WHERE {where_clause} AND entity_type = :entity_type"
        )
        params = {**set_fields, **record.identity_fields}

        result = await self._client.command(
            self._database, "sql", sql, params,
        )
        return _rid(result)

    async def _create_provenance_edge(
        self, node_rid: str, provenance: ProvenanceMetadata,
    ) -> None:
        # HAS_PROVENANCE links entities to their source Document.
        # Distinct from EXTRACTED_FROM which links entities to specific chunks.
        sql = (
            f"CREATE EDGE HAS_PROVENANCE FROM {node_rid} "
            "TO (SELECT FROM Document WHERE document_id = :document_id) "
            "SET document_id = :document_id, pipeline_run_id = :pipeline_run_id, "
            "page_numbers = :page_numbers, "
            "upload_datetime = :upload_datetime, document_datetime = :document_datetime, "
            "created_at = sysdate()"
        )
        params = {
            "document_id": provenance.document_id,
            "pipeline_run_id": provenance.pipeline_run_id,
            "page_numbers": provenance.page_numbers,
            "upload_datetime": provenance.upload_datetime,
            "document_datetime": provenance.document_datetime,
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
        """Create HAS_PROVENANCE edges from multiple nodes to a Document in one call."""
        targets = [rid for rid in node_rids if rid]
        if not targets:
            return
        statements: list[str] = []
        params: dict[str, Any] = {
            "document_id": provenance.document_id,
            "pipeline_run_id": provenance.pipeline_run_id,
            "page_numbers": provenance.page_numbers,
            "upload_datetime": provenance.upload_datetime,
            "document_datetime": provenance.document_datetime,
        }
        for rid in targets:
            statements.append(
                f"CREATE EDGE HAS_PROVENANCE FROM {rid} "
                f"TO (SELECT FROM Document WHERE document_id = :document_id) "
                f"SET document_id = :document_id, "
                f"pipeline_run_id = :pipeline_run_id, "
                f"page_numbers = :page_numbers, "
                f"upload_datetime = :upload_datetime, "
                f"document_datetime = :document_datetime, "
                f"created_at = sysdate()"
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
            f"UPSERT RETURN AFTER @rid WHERE document_id = :document_id"
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

        # Use prefixed param keys to avoid collision when from_identity and
        # to_identity have the same field names (e.g., both have "name").
        from_parts = [f"{k} = :f_{k}" for k in record.from_identity]
        to_parts = [f"{k} = :t_{k}" for k in record.to_identity]
        from_where = " AND ".join(from_parts)
        to_where = " AND ".join(to_parts)

        doc_ids_expr = "[]"
        doc_id_param: dict[str, Any] = {}
        if provenance:
            doc_ids_expr = "[:doc_id]"
            doc_id_param = {"doc_id": provenance.document_id}

        extra_props = ""
        extra_params: dict[str, Any] = {}
        if record.properties:
            prop_parts = [f"{k} = :p_{k}" for k in record.properties]
            extra_props = ", " + ", ".join(prop_parts)
            extra_params = {f"p_{k}": v for k, v in record.properties.items()}

        sql = (
            f"CREATE EDGE {record.rel_type} "
            f"FROM (SELECT FROM {record.from_type} WHERE {from_where}) "
            f"TO (SELECT FROM {record.to_type} WHERE {to_where}) "
            f"SET extraction_confidence = :extraction_confidence, "
            f"document_ids = {doc_ids_expr}, "
            f"created_at = sysdate(), updated_at = sysdate(){extra_props}"
        )
        params = {
            **{f"f_{k}": v for k, v in record.from_identity.items()},
            **{f"t_{k}": v for k, v in record.to_identity.items()},
            "extraction_confidence": record.extraction_confidence,
            **doc_id_param,
            **extra_params,
        }
        result = await self._client.command(self._database, "sql", sql, params)
        return _rid(result)

    async def upsert_relationships_batch(
        self,
        records: list[RelationshipRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Create multiple relationships, one sqlscript per edge.

        Mirrors upsert_relationships_batch_sync's per-edge loop — see
        that docstring for the ArcadeDB sqlscript batching regression
        that forced this structure.
        """
        if not records:
            return []
        valid_records = [
            r for r in records
            if self._enforce_relationship_triple(r.from_type, r.rel_type, r.to_type)
        ]
        if not valid_records:
            return []
        rids: list[str] = []
        for rec in valid_records:
            script, params = _build_upsert_relationship_script([rec], provenance)
            result = await self._client.command(
                self._database, "sqlscript", script, params,
            )
            rids.extend(_extract_rids(result, 1))
        return rids

    async def create_structural_edge(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a structural edge between two RIDs."""
        sql, params = _build_structural_edge_sql(from_id, to_id, rel_type, properties)
        result = await self._client.command(self._database, "sql", sql, params)
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
        type-specific indexes. When unknown, iterate ontology entity types
        (ArcadeDB has no root vertex type — `FROM V` and `FROM Vertex` both
        raise SchemaException). Returns the first match.
        """
        if entity_type:
            sql, params = _build_resolve_root_entity_sql(name, entity_type)
            rows = await self._client.query(self._database, "sql", sql, params)
            return _to_entity(rows[0]) if rows else None

        try:
            from app.services.ontology_templates import load_ontology
            from app.services.arcadedb_schema import STRUCTURAL_TYPES
            ont = load_ontology()
            ontology_types = [
                _safe_type_name(e["name"]) for e in ont.get("entity_types", [])
            ]
            # Search domain entities first; structural types (SECTION,
            # FIGURE, etc.) last. The SECTION class in particular has an
            # ArcadeDB index quirk where `WHERE name = 'SA-2'` returns
            # SECTIONs whose name is "2", so picking up a domain hit
            # first avoids returning bogus structural matches.
            ontology_types.sort(key=lambda t: (t in STRUCTURAL_TYPES, t))
        except Exception:
            ontology_types = []
        if not ontology_types:
            return None

        for etype in ontology_types:
            sql, params = _build_resolve_root_entity_sql(name, etype)
            try:
                rows = await self._client.query(self._database, "sql", sql, params)
            except Exception:
                # Type not yet in ArcadeDB schema (no docs ingested with
                # that type yet) — skip and keep looking.
                continue
            if rows:
                # Defensive: SECTION's broken index can return rows that
                # don't actually match the requested name (e.g. WHERE
                # name='SA-2' returning rows whose name is '2'). Verify
                # the returned row's name matches case-insensitively —
                # ArcadeDB's WHERE is case-insensitive too, so the DB
                # may legitimately return 'FAN SONG' for a 'Fan Song'
                # query and we want to accept that.
                row_name = rows[0].get("name") or ""
                if row_name.casefold() == name.casefold():
                    return _to_entity(rows[0])
        return None

    async def fulltext_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[GraphEntityResult]:
        """Full-text search across entity types using CONTAINSTEXT.

        ArcadeDB doesn't have an abstract V base type, so cross-type
        searches query each concrete entity type and merge results.
        """
        result = _build_fulltext_search_sql(query, entity_types, limit)

        if isinstance(result[0], str):
            # Single SQL string
            sql, params = result
            rows = await self._client.query(self._database, "sql", sql, params)
            return [_to_entity(r) for r in rows]

        sqls, params = result
        if not sqls:
            # No type filter — load all ontology entity types
            try:
                from app.services.ontology_templates import load_ontology
                from app.services.arcadedb_schema import _safe_type_name
                ont = load_ontology()
                sqls = [
                    f"SELECT *, @rid AS node_id FROM {_safe_type_name(e['name'])} "
                    f"WHERE name CONTAINSTEXT :query LIMIT :limit"
                    for e in ont.get("entity_types", [])
                ]
            except Exception:
                return []

        # Query each type and merge (use asyncio.gather for parallelism)
        all_rows: list[dict] = []
        for sql in sqls:
            try:
                rows = await self._client.query(self._database, "sql", sql, params)
                all_rows.extend(rows)
            except Exception:
                continue  # Skip types that error (empty types, etc.)

        # Sort by extraction_confidence descending, cap at limit
        entities = [_to_entity(r) for r in all_rows]
        entities.sort(key=lambda e: e.extraction_confidence or 0, reverse=True)
        return entities[:limit]

    async def get_neighborhood(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> list[GraphEntityResult]:
        """Return nodes in the k-hop neighbourhood of *node_id*.

        *node_id* may be an ArcadeDB RID (#10:5) or a UUID string; UUIDs
        are resolved to RIDs first. Uses MATCH syntax per ArcadeDB manual.
        """
        rid = await self._resolve_rid(node_id)
        if not rid:
            return []

        # ArcadeDB doesn't support MATCH without a concrete type or
        # both(){depth} syntax. Build nested expand for multi-hop.
        if rel_types:
            edge_list = ", ".join(f"'{t}'" for t in rel_types)
            traversal = f"both({edge_list})"
        else:
            traversal = "both()"

        # Build nested query for depth: depth=1 → one expand, depth=2 → nested expand
        inner = f"SELECT expand({traversal}) FROM {rid}"
        for _ in range(1, depth):
            inner = f"SELECT expand({traversal}) FROM ({inner})"
        sql = f"SELECT *, @rid AS node_id FROM ({inner})"
        rows = await self._client.query(self._database, "sql", sql)
        return [_to_entity(r) for r in rows]

    async def get_neighborhood_graph(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return neighbourhood as a {nodes, edges} dict.

        *node_id* may be an ArcadeDB RID or UUID string.
        """
        rid = await self._resolve_rid(node_id)
        if not rid:
            return {"nodes": [], "edges": []}

        # Get neighbor nodes
        entities = await self.get_neighborhood(node_id, depth, rel_types)

        # Get edges — query outgoing + incoming edges from the root vertex
        if rel_types:
            edge_list = ", ".join(f"'{t}'" for t in rel_types)
            edge_traversal = f"bothE({edge_list})"
        else:
            edge_traversal = "bothE()"

        edge_sql = f"SELECT *, @rid AS edge_rid FROM (SELECT expand({edge_traversal}) FROM {rid})"
        edge_rows = await self._client.query(self._database, "sql", edge_sql)

        nodes: list[dict] = [
            {"id": e.node_id, "name": e.name, "entity_type": e.entity_type, **e.properties}
            for e in entities
        ]
        edges: list[dict] = []
        for r in edge_rows:
            raw_prov = r.get("provenance")
            if isinstance(raw_prov, str):
                try:
                    provenance = json.loads(raw_prov)
                except (ValueError, TypeError):
                    provenance = None
            else:
                provenance = raw_prov  # already a dict or None
            edges.append({
                "id": str(r.get("edge_rid", "")),
                "rel_type": str(r.get("@type", r.get("@class", ""))),
                "from": str(r.get("@out", r.get("out", ""))),
                "to": str(r.get("@in", r.get("in", ""))),
                "provenance": provenance,
            })

        return {"nodes": nodes, "edges": edges}

    async def get_directed_traversal(
        self,
        node_id: str,
        steps: list[dict],
        target_entity_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[GraphEntityResult]:
        """Execute a multi-step directed MATCH traversal.

        Generates native ArcadeDB MATCH from query-profile traversal steps
        instead of collapsing them into an undirected .both() walk.
        """
        rid = await self._resolve_rid(node_id)
        if not rid:
            return []

        # ArcadeDB MATCH quirks worked around here (mirrors the
        # get_ontology_linked_chunks fix):
        #   * The first node MUST declare a concrete `type:` — omitting
        #     it throws java.lang.UnsupportedOperationException. Resolve
        #     @type via a quick lookup before building the MATCH.
        #   * Edge type filters in `.out()/.in()/.both()` are positional
        #     string args (`.out('A','B')`), NOT a `{class: ...}` map.
        #     The `class:` keyword is rejected by the parser here.
        #   * The `while:` predicate for variable-depth traversal goes
        #     inside the *target node* alias spec, not inside the edge
        #     traversal call.
        type_rows = await self._client.query(
            self._database, "sql",
            f"SELECT @type AS node_type FROM {rid}",
        )
        seed_type = None
        if type_rows and isinstance(type_rows[0], dict):
            seed_type = type_rows[0].get("node_type")
        if not seed_type:
            return []

        # Build MATCH pattern: root → step1 → step2 → ... → target
        match_parts = [
            f"{{type: {seed_type}, as: root, where: (@rid = {rid})}}"
        ]
        for i, step in enumerate(steps):
            direction = step.get("direction", "out")
            rel_types = step.get("rel_types", [])
            min_hops = step.get("min_hops", 1)
            max_hops = step.get("max_hops", 1)

            # Direction: .out() / .in() / .both()
            dir_fn = "out" if direction == "out" else "in" if direction == "in" else "both"
            # Rel type filter as positional string args
            rel_args = ",".join(f"'{t}'" for t in rel_types) if rel_types else ""
            # Variable-depth `while:` belongs in the alias spec
            while_clause = f", while: ($depth >= {min_hops} AND $depth <= {max_hops})"
            match_parts.append(
                f".{dir_fn}({rel_args}) "
                f"{{as: step{i}{while_clause}}}"
            )

        # The last step alias is the target
        last_alias = f"step{len(steps) - 1}"

        # Type filter on the final target — inject into the last alias
        # spec, which already carries `while:`. Match `{as: stepN, while: ...}`
        # and slot in `where: (...)` after the alias name.
        if target_entity_types and match_parts:
            type_list = ", ".join(f"'{t}'" for t in target_entity_types)
            type_filter = f"where: (entity_type IN [{type_list}]), "
            last_part = match_parts[-1]
            match_parts[-1] = last_part.replace(
                f"{{as: {last_alias}, ",
                f"{{as: {last_alias}, {type_filter}",
            )

        sql = (
            f"MATCH {''.join(match_parts)} "
            f"RETURN {last_alias}.name AS name, "
            f"{last_alias}.entity_type AS entity_type, "
            f"{last_alias}.canonical_name AS canonical_name, "
            f"{last_alias}.extraction_confidence AS extraction_confidence, "
            f"{last_alias}.@rid AS node_id "
            f"LIMIT {limit}"
        )
        rows = await self._client.query(self._database, "sql", sql)
        return [_to_entity(r) for r in rows]

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
        rid = await self._resolve_rid(node_id)
        if not rid:
            return []

        # ArcadeDB MATCH requires a concrete `type:` on the first node:
        # omitting it (or using `type: V`) throws UnsupportedOperationException
        # on transaction commit. We don't know whether the seed is a TextChunk
        # or an ImageChunk at the call site, so resolve @type with one
        # lightweight lookup before building the typed MATCH. Also:
        # `{class: V, ...}` is rejected by the parser; only `type:` works.
        # `.@class` in a projection / alias is a parser error; `.@type`
        # resolves cleanly inside the MATCH alias scope.
        type_rows = await self._client.query(
            self._database, "sql",
            f"SELECT @type AS node_type FROM {rid}",
        )
        seed_type = None
        if type_rows and isinstance(type_rows[0], dict):
            seed_type = type_rows[0].get("node_type")
        if seed_type not in ("TextChunk", "ImageChunk"):
            return []

        sql = (
            f"MATCH "
            f"{{type: {seed_type}, as: seed, where: (@rid = {rid})}}"
            f".in('EXTRACTED_FROM') {{as: entity}}"
            f".out('EXTRACTED_FROM') {{as: chunk, where: (@rid <> {rid})}} "
            f"RETURN chunk.@rid AS chunk_rid, chunk.@type AS chunk_type, "
            f"chunk.chunk_id AS chunk_id, chunk.document_id AS document_id, "
            f"chunk.text AS text, chunk.chunk_text AS chunk_text, "
            f"chunk.modality AS modality, chunk.page_number AS page_number, "
            f"entity.name AS entity_name, entity.entity_type AS entity_type"
        )
        rows = await self._client.query(self._database, "sql", sql)
        # Deduplicate by chunk_rid (same chunk may be reached via multiple entities)
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for r in rows:
            crid = str(r.get("chunk_rid", ""))
            if crid and crid not in seen:
                seen.add(crid)
                result.append(r)
        return result

    async def get_entity_by_rid(self, rid: str) -> dict[str, Any]:
        """Return all properties of the vertex at the given RID, including
        nullable ones. Used by section_properties profiles to feed
        _project_field_groups (spec §4.6).

        ArcadeDB note: SELECT * already includes @rid and @type in the
        projection — explicit projection syntax (`SELECT @rid, @type, *`)
        is rejected as a SQL parse error.
        """
        if not rid.startswith("#"):
            raise ValueError(f"get_entity_by_rid expects an ArcadeDB RID (got {rid!r})")
        rows = await self._client.query(
            self._database, "sql",
            f"SELECT * FROM {rid}",
        )
        return rows[0] if rows else {}

    async def get_associated_systems(self, node_id: str) -> list[GraphEntityResult]:
        """Return systems linked by ASSOCIATED_WITH or CUES in either direction.

        Spec §4.6. Used by the System Components profile's `related_systems`
        block. Resolves @type for typed MATCH (ArcadeDB 26.5.x throws
        UnsupportedOperationException without it), traverses bothE() across
        the two relevant edge labels, deduplicates by RID, returns up to 25.
        """
        rid = await self._resolve_rid(node_id)
        if not rid:
            return []
        type_rows = await self._client.query(
            self._database, "sql",
            f"SELECT @type AS node_type FROM {rid}",
        )
        if not type_rows:
            return []
        seed_type = type_rows[0].get("node_type")
        if seed_type not in _CANONICAL_ROOT_ENTITY_TYPES_RUNTIME:
            return []

        sql = (
            f"MATCH {{type: {seed_type}, as: src, where: (@rid = {rid})}}"
            f".bothE('ASSOCIATED_WITH', 'CUES') {{as: e}}"
            f".bothV() {{as: tgt, where: (@rid <> {rid})}} "
            f"RETURN tgt.@rid AS node_id, tgt.name AS name, "
            f"tgt.@type AS entity_type, tgt.canonical_name AS canonical_name, "
            f"e.@type AS rel_type "
            f"LIMIT 25"
        )
        try:
            rows = await self._client.query(self._database, "sql", sql)
        except Exception:
            return []

        seen: set[str] = set()
        out: list[GraphEntityResult] = []
        for r in rows:
            nid = str(r.get("node_id", ""))
            if not nid or nid in seen:
                continue
            seen.add(nid)
            out.append(GraphEntityResult(
                node_id=nid,
                name=r.get("name", ""),
                entity_type=r.get("entity_type", ""),
                canonical_name=r.get("canonical_name"),
                relationship_types=[r.get("rel_type", "")] if r.get("rel_type") else [],
            ))
        return out

    async def get_structural_neighbors(
        self,
        chunk_id: str,
        max_hops: int = 2,
        limit: int = 5,
    ) -> list[dict]:
        """Return structurally linked chunks via ArcadeDB graph edges.

        Traverses NEXT_CHUNK, SAME_PAGE, SAME_SECTION, SAME_ARTIFACT edges
        from the given chunk, returning neighbor chunks with link metadata
        for scoring.
        """
        rid = await self._resolve_rid(chunk_id)
        if not rid:
            return []

        # ArcadeDB SQL quirk: `target.@class` / `edge.@class` is a parser error
        # ("no viable alternative at input '.@'"). MATCH + `as: X` + `X.@type`
        # works because @type is resolved inside the MATCH scope rather than as
        # a dotted attribute lookup on a LET variable. Return weight via
        # explicit projection and handle null weight in ORDER BY.
        sql = (
            f"MATCH "
            f"{{type: TextChunk, as: src, where: (@rid = {rid})}}"
            f".bothE('NEXT_CHUNK','SAME_PAGE','SAME_SECTION','SAME_ARTIFACT'){{as: e}}"
            f".bothV(){{as: tgt, where: (@rid <> {rid})}} "
            f"RETURN "
            f"tgt.chunk_id AS chunk_id, "
            f"tgt.@type AS chunk_type, "
            f"tgt.document_id AS document_id, "
            f"tgt.modality AS modality, "
            f"e.@type AS link_type, "
            f"e.weight AS weight "
            f"ORDER BY weight DESC "
            f"LIMIT :limit"
        )
        rows = await self._client.query(
            self._database, "sql", sql, {"limit": limit},
        )
        return rows

    async def get_entity_evidence_chunks(
        self,
        entity_rid: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return chunks that an entity was extracted from.

        Traverses entity →(EXTRACTED_FROM)→ chunks. Returns chunk rows with
        @class (TextChunk/ImageChunk), chunk_id, document_id, and text.
        """
        rid = await self._resolve_rid(entity_rid)
        if not rid:
            return []
        # ArcadeDB SQL: `SELECT *, <alias>, <alias>` is a parser error
        # ("mismatched input ','") — the parser rejects additional
        # projections when `*` is present. `@class` is likewise not a valid
        # projection expression; `@type` is. Enumerate fields explicitly.
        sql = (
            f"SELECT @rid AS chunk_rid, @type AS chunk_type, "
            f"chunk_id, document_id, text, chunk_text, modality, page_number "
            f"FROM (SELECT expand(out('EXTRACTED_FROM')) FROM {rid}) "
            f"LIMIT :limit"
        )
        rows = await self._client.query(
            self._database, "sql", sql, {"limit": limit},
        )
        # Strip embedding vectors from results
        for r in rows:
            r.pop("text_embedding", None)
            r.pop("image_embedding", None)
        return rows

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
        sql, params = _build_create_alias_sql(alias)
        alias_result = await self._client.command(
            self._database, "sql", sql, params,
        )
        alias_rid = _rid(alias_result)

        edge_sql = _build_create_alias_edge_sql(node_id, alias_rid)
        await self._client.command(self._database, "sql", edge_sql)

    async def search_by_alias(
        self,
        alias: str,
        entity_type: str | None = None,
    ) -> list[GraphEntityResult]:
        """Find nodes by alias."""
        sql, params = _build_search_by_alias_sql(alias)
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
        sql, params = _build_set_canonical_name_sql(node_id, canonical_name)
        await self._client.command(
            self._database, "sql", sql, params,
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
        document_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
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
        document_ids:
            Optional list of document UUIDs to restrict results to. Pushed into
            the ArcadeDB WHERE clause so the vector index budget isn't wasted on
            documents that will be filtered out.
        filters:
            Optional dict of additional vertex property filters (VR C.1). Each
            key must be a property declared in the schema for this vertex type;
            unknown keys raise ValueError to catch typos early. Compiled to
            parameterized WHERE clauses (``{k} = :{k}`` pattern) and ANDed with
            the document_ids clause when both are provided.
            ``None`` or ``{}`` → unfiltered (no extra WHERE clause emitted).
        """
        # --- Validate filters keys against the declared schema -----------------
        # Import lazily to avoid circular dependency at module load time.
        from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES
        if filters:
            declared_props: set[str] = set()
            if vertex_type in _STRUCTURAL_VERTEX_TYPES:
                declared_props = {name for name, _ in _STRUCTURAL_VERTEX_TYPES[vertex_type]}
            # For ontology entity types (not in _STRUCTURAL_VERTEX_TYPES), we
            # skip validation because their property set is dynamic. Unknown
            # structural-type keys are caught; dynamic types pass through.
            if declared_props:
                unknown = set(filters.keys()) - declared_props
                if unknown:
                    raise ValueError(
                        f"vector_search filters contains unknown properties for vertex type "
                        f"{vertex_type!r}: {sorted(unknown)}. "
                        f"Known properties: {sorted(declared_props)}"
                    )

        # --- Build WHERE clause -------------------------------------------------
        index_name = f"{vertex_type}[{embedding_property}]"
        from app.config import get_settings as _gs
        ef = _gs().arcadedb_vector_ef_search
        ef_arg = f", {ef}" if ef > 0 else ""

        params: dict[str, Any] = {
            "query_vector": query_vector,
            "top_k": top_k,
        }

        where_parts: list[str] = []

        # Existing document_ids clause (string-interpolated, preserving the
        # existing pattern — values are UUIDs, safe against SQL injection).
        if document_ids:
            doc_list = ", ".join(f"'{d}'" for d in document_ids)
            where_parts.append(f"document_id IN [{doc_list}]")

        # New filters clause — parameterized to avoid SQL injection.
        if filters:
            for k, v in filters.items():
                param_key = f"filter_{k}"
                where_parts.append(f"{k} = :{param_key}")
                params[param_key] = v

        where_clause = ""
        if where_parts:
            where_clause = " WHERE " + " AND ".join(where_parts)

        # ArcadeDB quirk: when vectorNeighbors is expanded inside a subquery,
        # `$distance` is NOT projected in the outer SELECT (only available while
        # vectorNeighbors is being evaluated). Referencing it in ORDER BY forces
        # the engine to surface it as a `distance` column, which _to_entity then
        # converts to a similarity score. Without this, every hit arrives with
        # extraction_confidence=None and the retrieval min-score filter drops
        # 100% of results (empty search).
        sql = (
            f"SELECT *, @rid AS node_id "
            f"FROM (SELECT expand(vectorNeighbors('{index_name}', "
            f":query_vector, :top_k{ef_arg})))"
            f"{where_clause} "
            f"ORDER BY $distance ASC"
        )
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
        """Search using both text and image embeddings.

        Runs text and image vector searches in parallel, then merges
        results with score-based interleaving (highest similarity first).
        """
        text_rows, img_rows = await asyncio.gather(
            self._client.query(self._database, "sql",
                "SELECT *, $distance, @rid AS node_id "
                "FROM (SELECT expand(vectorNeighbors('TextChunk[text_embedding]', "
                ":query_vector, :top_k)))",
                {"query_vector": text_embedding, "top_k": limit},
            ),
            self._client.query(self._database, "sql",
                "SELECT *, $distance, @rid AS node_id "
                "FROM (SELECT expand(vectorNeighbors('ImageChunk[image_embedding]', "
                ":query_vector, :top_k)))",
                {"query_vector": image_embedding, "top_k": limit},
            ),
        )

        combined = [_to_entity(r) for r in text_rows + img_rows]
        # Sort by similarity score (descending) and deduplicate by node_id
        combined.sort(key=lambda e: e.extraction_confidence or 0, reverse=True)
        seen: set[str] = set()
        result: list[GraphEntityResult] = []
        for entity in combined:
            if entity.node_id not in seen:
                seen.add(entity.node_id)
                result.append(entity)
            if len(result) >= limit:
                break
        return result

    async def graph_vector_search(
        self,
        root_id: str,
        query_vector: list[float],
        embedding_property: str = "text_embedding",
        traversal: str = "out('EXTRACTED_FROM')",
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Cross-model query: graph traversal filtered by vector similarity.

        Implements the pattern from ArcadeDB Manual §4.13.6: use MATCH to
        traverse graph edges, then filter results by ``vectorCosineSimilarity``
        on their embeddings. This finds graph-connected records that are also
        semantically similar to a query vector — in a single ArcadeDB query.

        Example: from an entity, traverse EXTRACTED_FROM to chunks, return
        only chunks whose embedding is similar to the query.
        """
        sql = (
            f"SELECT *, vectorCosineSimilarity({embedding_property}, :query_vector) AS vscore "
            f"FROM (SELECT expand({traversal}) FROM {root_id}) "
            f"WHERE vectorCosineSimilarity({embedding_property}, :query_vector) > :threshold "
            f"ORDER BY vscore DESC "
            f"LIMIT :limit"
        )
        return await self._client.query(
            self._database, "sql", sql,
            {
                "query_vector": query_vector,
                "threshold": similarity_threshold,
                "limit": limit,
            },
        )

    # ==================================================================
    # Community operations
    # ==================================================================

    async def run_community_algorithm(
        self,
        algorithm: str,
        params: dict,
        exclude_types: set[str] | None = None,
    ) -> list[dict]:
        """Run a community detection algorithm and return per-node results.

        If *exclude_types* is provided, nodes whose label is in that set are
        filtered out server-side via Cypher label negation, so structural
        vertices (Document, TextChunk, etc.) do not influence community
        assignment.

        ArcadeDB quirks worked around here:

        * ``CALL algo.{leiden,louvain}() YIELD node`` returns a
          ``DatabaseRID`` value, not a vertex object — so ``node.name`` and
          ``labels(node)`` are unbound (``node.name`` returned NULL for
          every row, killing community detection silently). The fix is to
          carry the RID forward via ``toString(node)`` and ``MATCH (n)
          WHERE toString(id(n)) = ...`` to bind a real vertex.
        * Cypher cannot reference ``@class`` or ``@rid``; filter via
          ``NOT n:Label`` predicates against the matched vertex.
        * In ArcadeDB 26.5.x ``algo.leiden`` populates ``node`` but always
          returns ``communityId = NULL`` — the algorithm doesn't actually
          assign communities. ``algo.louvain`` works. Callers should
          configure ``community_detection_algorithm=louvain`` until
          upstream fixes leiden.
        """
        algo_params = ", ".join(f"{k}: {v}" for k, v in params.items())

        where_clause = "WHERE toString(id(n)) = rid"
        if exclude_types:
            where_clause += " AND " + " AND ".join(
                f"NOT n:{t}" for t in sorted(exclude_types)
            )

        cypher = (
            f"CALL algo.{algorithm}({{{algo_params}}}) YIELD node, communityId "
            f"WITH toString(node) AS rid, communityId AS cid "
            f"MATCH (n) "
            f"{where_clause} "
            f"RETURN n.name AS name, labels(n)[0] AS entity_type, "
            f"id(n) AS node_rid, cid AS community_id"
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
        source_documents: list[dict] | None = None,
    ) -> str:
        """Upsert a CommunityReport vertex and return its RID."""
        sql = (
            "UPDATE CommunityReport SET title = :title, summary = :summary, "
            "member_count = :count, membership_hash = :hash, "
            "source_documents = :sources, "
            "generated_at = sysdate(), model_name = :model "
            "UPSERT RETURN AFTER @rid WHERE community_id = :cid"
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
                "sources": source_documents or [],
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
        from app.config import get_settings as _gs
        ef = _gs().arcadedb_vector_ef_search
        ef_arg = f", {ef}" if ef > 0 else ""
        sql = (
            "SELECT community_id, title, summary, member_count, "
            "generated_at, model_name, source_documents, $distance, @rid AS report_rid "
            "FROM (SELECT expand(vectorNeighbors('CommunityReport[report_embedding]', "
            f":query_vector, :top_k{ef_arg})))"
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
        from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES, _safe_type_name
        from app.services.ontology_templates import load_ontology
        structural = set(_STRUCTURAL_EDGE_TYPES)

        # ArcadeDB SQL quirks (same class as the get_ontology_linked_chunks
        # fix in the same file):
        #  - MATCH first node without `type:` throws
        #    java.lang.UnsupportedOperationException; `type: V` fails
        #    because there's no root type registered.
        #  - Top-level `@class` projection is a parser error; use `@type`.
        #  - `out.@class` / `in.@class` fail with "no viable alternative at input '.@'".
        #    MATCH aliases (`e.@type`, `src.@type`, etc.) resolve correctly.
        #
        # The seed type is intrinsically unknown here (names may span
        # MISSILE_SYSTEM, RADAR_SYSTEM, etc.), so iterate the ontology's
        # entity types and run one MATCH per type with `type: T` on src.
        # dst carries no type constraint, so cross-type edges (e.g.
        # RADAR→MISSILE) are still captured in the iteration where the
        # src-side type matches either endpoint. Types that aren't yet
        # registered in ArcadeDB's schema raise SchemaException; skip
        # those and continue.
        try:
            ontology = load_ontology()
            entity_types = [
                _safe_type_name(e["name"]) for e in ontology.get("entity_types", [])
            ]
        except Exception:
            entity_types = []
        if not entity_types:
            return []

        all_rows: list[dict] = []
        for etype in entity_types:
            sql = (
                f"MATCH {{type: {etype}, as: src, where: (name IN :names)}}"
                f".outE() {{as: e}}"
                f".inV() {{as: dst, where: (name IN :names)}} "
                f"RETURN e.@type AS rel_type, "
                f"src.name AS from_name, src.@type AS from_type, "
                f"dst.name AS to_name, dst.@type AS to_type"
            )
            try:
                rows = await self._client.query(
                    self._database, "sql", sql,
                    params={"names": entity_names},
                )
                all_rows.extend(rows)
            except Exception:
                # Type not yet in ArcadeDB schema (nothing ingested with
                # that type) or a per-type query failed — continue.
                continue
        return [r for r in all_rows if r.get("rel_type") not in structural]

    # ==================================================================
    # Lifecycle operations
    # ==================================================================

    async def delete_document_graph(
        self,
        document_id: str,
    ) -> int:
        """Delete all graph elements associated with *document_id*."""
        parts = _build_delete_document_graph_sql(document_id)
        params = parts["params"]

        total = 0
        for sql in parts["delete_vertex_sqls"]:
            try:
                result = await self._client.command(self._database, "sql", sql, params)
                total += _count(result)
            except Exception:
                pass  # type may not exist yet on fresh DB

        for sql in parts["edge_cleanup_sqls"]:
            try:
                await self._client.command(self._database, "sql", sql, params)
            except Exception:
                pass

        for sql in parts["edge_prune_sqls"]:
            try:
                await self._client.command(self._database, "sql", sql)
            except Exception:
                pass

        for sql in parts["orphan_sqls"]:
            try:
                result = await self._client.command(self._database, "sql", sql)
                total += _count(result)
            except Exception:
                pass

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
            f"UPDATE {_safe_type_name(record.entity_type)} SET {set_clause}, "
            f"updated_at = sysdate() "
            f"UPSERT RETURN AFTER @rid WHERE {where_clause} AND entity_type = :entity_type"
        )
        params = {**set_fields, **record.identity_fields}
        result = self._client.command_sync(self._database, "sql", sql, params)
        return _rid(result)

    def _create_provenance_edge_sync(
        self, node_rid: str, provenance: ProvenanceMetadata,
    ) -> None:
        sql = (
            f"CREATE EDGE HAS_PROVENANCE FROM {node_rid} "
            "TO (SELECT FROM Document WHERE document_id = :document_id) "
            "SET document_id = :document_id, pipeline_run_id = :pipeline_run_id, "
            "page_numbers = :page_numbers, "
            "upload_datetime = :upload_datetime, document_datetime = :document_datetime, "
            "created_at = sysdate()"
        )
        params = {
            "document_id": provenance.document_id,
            "pipeline_run_id": provenance.pipeline_run_id,
            "page_numbers": provenance.page_numbers,
            "upload_datetime": provenance.upload_datetime,
            "document_datetime": provenance.document_datetime,
        }
        self._client.command_sync(self._database, "sql", sql, params)

    def upsert_nodes_batch_sync(
        self,
        records: list[NodeRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Upsert a batch of nodes in one sqlscript call. Returns their RIDs.

        Raises:
            ValueError: if any record's ``identity_fields`` contains a
                ``None`` or empty-string value. Catches malformed identity
                at the ingestion boundary (where the fix is obvious) rather
                than letting it propagate into ArcadeDB which returns a
                blank ``@rid`` that then corrupts downstream CREATE EDGE
                statements.
            UpsertMissingRIDError: if the sqlscript completes but one or
                more records have no ``@rid`` in the result. Distinct from
                the caller-side "malformed identity" case caught above —
                this fires for genuine server-side failures (schema drift,
                truncated response, etc.) that used to be silently masked
                by ``_extract_rids`` padding with ``""``.
        """
        if not records:
            return []

        # --- Entry validation: reject records with malformed identity ---
        # None or empty-string values in identity_fields break the UPSERT's
        # WHERE clause — the MATCH misses every row so the INSERT branch
        # fires but may return no @rid, or inserts with a useless
        # zero-identity that will never be looked up again. Refuse at the
        # boundary with a message that points at the offending record.
        for i, rec in enumerate(records):
            bad = [
                k for k, v in rec.identity_fields.items()
                if v is None or (isinstance(v, str) and not v)
            ]
            if bad:
                raise ValueError(
                    f"upsert_nodes_batch_sync: record index {i} "
                    f"(entity_type={rec.entity_type!r}, name={rec.name!r}) "
                    f"has empty / None identity field(s) {bad}. Full "
                    f"identity_fields={rec.identity_fields!r}. Reject at "
                    f"upsert boundary instead of corrupting the graph."
                )

        script, params = _build_upsert_node_script(records)
        result = self._client.command_sync(
            self._database, "sqlscript", script, params,
        )
        rids = _extract_rids(result, len(records))

        # --- Server-side sanity: every record must come back with a @rid ---
        # After entry validation, any still-blank RID is a genuine server
        # issue. Raise loud instead of letting "" propagate — downstream
        # edge emission would turn the blank into invalid SQL.
        missing = [
            i for i, rid in enumerate(rids) if not rid
        ]
        if missing:
            missing_detail = [
                {
                    "index": i,
                    "entity_type": records[i].entity_type,
                    "name": records[i].name,
                    "identity_fields": records[i].identity_fields,
                }
                for i in missing
            ]
            raise UpsertMissingRIDError(
                f"upsert_nodes_batch_sync: {len(missing)}/{len(records)} "
                f"records returned no @rid despite valid identity input. "
                f"Missing: {missing_detail}"
            )

        # --- Durability assertion: every returned RID must be queryable ---
        # Task 3 (lineage_diagnostic_findings.md). RID presence above is
        # necessary but NOT sufficient: the SA-2 silent zero-commit returned
        # valid RIDs while persisting 0 rows. Re-read the just-written RIDs
        # with a per-vertex-type existence query and assert the committed
        # count matches what we wrote. If the read-back is short, the write
        # did not durably commit — raise so the caller's rollback/FAILED path
        # fires instead of marking the merge succeeded. ``records`` is passed
        # so each RID can be grouped by its vertex type (rids[i] ↔ records[i],
        # _extract_rids pads positionally).
        self._assert_nodes_durable_sync(rids, records)

        if provenance:
            self._create_provenance_edges_batch_sync(rids, provenance)
        return rids

    def _assert_nodes_durable_sync(
        self,
        rids: list[str],
        records: list[NodeRecord],
    ) -> None:
        """Re-query just-upserted RIDs and assert all are committed/queryable.

        ``rids`` are the (non-blank, already RID-checked) RIDs returned by the
        upsert sqlscript; ``records`` is the parallel input list — ``rids[i]``
        corresponds to ``records[i]`` because ``_extract_rids`` pads
        positionally. We re-read each RID with a **type-scoped existence
        query** and raise :class:`UpsertNotDurableError` if fewer rows are
        actually present than we wrote.

        Why type-scoped and not ``SELECT count(*) FROM [<rid>, ...]``:
        proven against live ArcadeDB 26.5.1, ``count(*) FROM [#37:0,
        #37:999999]`` returns **2** — it counts LIST ENTRIES, not existing
        rows, so a non-durable RID never produces a short count and the
        assertion was a no-op against the exact silent zero-commit it was
        built to catch. ``SELECT count(*) FROM <Type> WHERE @rid IN [...]``
        correctly returns only the rows that exist (verified: real+fake → 1).

        Because a batch can span multiple vertex types (RADAR_SYSTEM,
        MISSILE_SYSTEM, ...), a single ``FROM <Type>`` would under-count the
        rest → false-fail. We group RIDs by their vertex type (routed through
        ``_safe_type_name`` so reserved words like ``TABLE`` → ``TABLE_REF``
        resolve to the class actually upserted) and issue one query per type,
        summing the existing rows. Bounded by #types (~9), not per-entity.

        Each per-type read is wrapped in the same read-after-write retry the
        ``batch_create_entity_chunk_edges_sync`` edge loop uses: ArcadeDB's
        ``asyncOperationsQueueSize=2048`` (docker-compose.yml) lets a separate
        HTTP read race ahead of the async flush of a just-written row, so a
        short count can be transient. We retry a few times with backoff before
        treating a short count as a real durability failure — without this the
        check could spuriously roll back a *good* merge under load.
        """
        if not rids:
            return
        # RIDs come straight from ArcadeDB's own response (e.g. "#10:0"); they
        # are not user input, so embedding them in the @rid IN list is safe.
        # Validate the shape defensively before interpolation. Keep the parallel
        # record so we can route each RID to its vertex type.
        type_to_rids: dict[str, list[str]] = {}
        safe_rids: list[str] = []
        for rid, rec in zip(rids, records):
            if not (isinstance(rid, str) and rid.startswith("#")):
                continue
            safe_rids.append(rid)
            vtype = _safe_type_name(rec.entity_type)
            type_to_rids.setdefault(vtype, []).append(rid)
        if not safe_rids:
            return

        expected = set(safe_rids)
        committed_rids: set[str] = set()
        for vtype, rids_for_type in type_to_rids.items():
            committed_rids.update(
                self._query_committed_rids_with_retry(vtype, rids_for_type)
            )

        missing = [r for r in safe_rids if r not in committed_rids]
        if missing:
            raise UpsertNotDurableError(
                f"upsert_nodes_batch_sync: durability re-query found "
                f"{len(committed_rids)}/{len(safe_rids)} upserted node(s) "
                f"actually committed/queryable in ArcadeDB. "
                f"Missing (not durable): {_truncate_rid_list(missing)}. "
                f"Returned RIDs were {_truncate_rid_list(safe_rids)}. A "
                f"returned @rid did NOT result in a durable row — refusing to "
                f"report success on a silent zero/partial commit (see "
                f"lineage_diagnostic_findings.md, Task 3)."
            )

    def _query_committed_rids_with_retry(
        self,
        vtype: str,
        rids_for_type: list[str],
    ) -> set[str]:
        """Return the subset of ``rids_for_type`` that exist as ``vtype`` rows.

        Issues ``SELECT @rid FROM <vtype> WHERE @rid IN [...]`` and returns the
        @rids that came back. Retries on a SHORT result (fewer rows than asked
        for) to ride out the async-flush read-after-write lag documented for
        ``batch_create_entity_chunk_edges_sync`` — the same separate-HTTP
        read-of-just-written-RIDs pattern. Returns whatever was found on the
        last attempt; the caller decides whether the total is short.
        """
        rid_list = ", ".join(rids_for_type)
        sql = f"SELECT @rid AS rid FROM {vtype} WHERE @rid IN [{rid_list}]"
        found: set[str] = set()
        for attempt in range(_DURABILITY_MAX_ATTEMPTS):
            rows = self._client.query_sync(self._database, "sql", sql)
            found = {
                str(row.get("rid") or row.get("@rid"))
                for row in rows
                if isinstance(row, dict) and (row.get("rid") or row.get("@rid"))
            }
            if len(found) >= len(rids_for_type):
                break
            if attempt + 1 < _DURABILITY_MAX_ATTEMPTS:
                logger.warning(
                    "_assert_nodes_durable_sync: re-query found %d/%d %s "
                    "rows on attempt %d — retrying (async-flush lag?)",
                    len(found), len(rids_for_type), vtype, attempt + 1,
                )
                time.sleep(_DURABILITY_BACKOFF_BASE * (attempt + 1))
        return found

    def _create_provenance_edges_batch_sync(
        self,
        node_rids: list[str],
        provenance: ProvenanceMetadata,
    ) -> None:
        """Create HAS_PROVENANCE edges for a batch of nodes in one sqlscript call."""
        targets = [rid for rid in node_rids if rid]
        if not targets:
            return
        statements: list[str] = []
        params: dict[str, Any] = {
            "document_id": provenance.document_id,
            "pipeline_run_id": provenance.pipeline_run_id,
            "page_numbers": provenance.page_numbers,
            "upload_datetime": provenance.upload_datetime,
            "document_datetime": provenance.document_datetime,
        }
        for rid in targets:
            statements.append(
                f"CREATE EDGE HAS_PROVENANCE FROM {rid} "
                f"TO (SELECT FROM Document WHERE document_id = :document_id) "
                f"SET document_id = :document_id, "
                f"pipeline_run_id = :pipeline_run_id, "
                f"page_numbers = :page_numbers, "
                f"upload_datetime = :upload_datetime, "
                f"document_datetime = :document_datetime, "
                f"created_at = sysdate()"
            )
        self._client.command_sync(
            self._database, "sqlscript", ";\n".join(statements), params,
        )

    def upsert_relationships_batch_sync(
        self,
        records: list[RelationshipRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Create a batch of relationships. Returns their RIDs.

        Issues one sqlscript per edge rather than batching them into a
        single multi-block script.

        Why: ArcadeDB Manual p.275 ("Conditional execution") documents
        ONLY the bare IF form::

            IF(<condition>){
              <statement>; [<statement>;]*
            }

        ``ELSE`` is not in the documented grammar. Our
        ``_build_upsert_relationship_script`` emits IF/ELSE for the
        find-or-create pattern, which worked for single-edge scripts
        but failed silently when two IF/ELSE blocks were stacked in
        one sqlscript: only the first IF-branch fires, subsequent
        IF-branch CREATE EDGE statements no-op, and the trailing
        ``RETURN [$v0, $v1, ...]`` collapses to ``[{value: None}]``.
        The ELSE branch (UPDATE document_ids on an already-existing
        edge) doesn't trip the regression, which is why re-ingests of
        the same document saw the edges while first-ingest of a
        second relationship silently dropped it.

        Per-edge sqlscripts sidestep the batch regression while
        keeping the find-or-create semantics (needed for cross-doc
        document_ids union). Each sqlscript is still atomic per the
        manual's "all-or-nothing" guarantee. Cost: N HTTP round-trips
        instead of 1. Typical per-doc edge counts are small (1-10),
        so latency impact is negligible compared to the correctness
        cost of silently losing edges.
        """
        if not records:
            return []
        valid_records = [
            r for r in records
            if self._enforce_relationship_triple(r.from_type, r.rel_type, r.to_type)
        ]
        if not valid_records:
            return []
        rids: list[str] = []
        for rec in valid_records:
            script, params = _build_upsert_relationship_script([rec], provenance)
            result = self._client.command_sync(
                self._database, "sqlscript", script, params,
            )
            rids.extend(_extract_rids(result, 1))
        return rids

    def create_structural_edge_sync(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous structural edge creation."""
        sql, params = _build_structural_edge_sql(from_id, to_id, rel_type, properties)
        result = self._client.command_sync(self._database, "sql", sql, params)
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
        document_id: str | None = None,
        pipeline_run_id: str | None = None,
    ) -> int:
        """Create EXTRACTED_FROM edges in a single sqlscript call.

        Post-Phase-8 Task 53b: when ``edge.source_rid`` is populated,
        uses a direct RID-to-RID CREATE EDGE that attaches to exactly
        that vertex and persists ``entity_id`` on the edge. When
        ``source_rid`` is None (legacy callers), falls back to the
        original name+type LIMIT-1 subquery path so nothing silently
        breaks during the migration window — a WARNING is logged so
        the fallback is visible.

        When ``document_id`` / ``pipeline_run_id`` are supplied they're
        persisted on every emitted edge so ``get_document_edges_sync``
        can return these rows. Both default to ``None`` only for
        backward compatibility; ``derive_structure_links`` always
        passes them.

        Rows where the subquery resolves to zero vertices become
        no-ops; the batch stays intact.
        """
        if not edges:
            return 0
        statements: list[str] = []
        params: dict[str, Any] = {}
        doc_set = ""
        if document_id is not None:
            params["_doc_id"] = document_id
            doc_set += ", document_id = :_doc_id"
        if pipeline_run_id is not None:
            params["_pipeline_run_id"] = pipeline_run_id
            doc_set += ", pipeline_run_id = :_pipeline_run_id"
        fallback_count = 0
        for i, edge in enumerate(edges):
            params[f"rid_{i}"] = edge.chunk_rid
            if edge.source_rid:
                # New path — direct RID-to-RID. entity_id persisted on the edge.
                params[f"src_{i}"] = edge.source_rid
                params[f"eid_{i}"] = edge.entity_id or ""
                statements.append(
                    f"CREATE EDGE EXTRACTED_FROM "
                    f"FROM :src_{i} TO :rid_{i} "
                    f"SET entity_id = :eid_{i}, created_at = sysdate(){doc_set}"
                )
            else:
                # Legacy path — name+type subquery. LIMIT 1 means siblings
                # with the same name+type but different identity tuples may
                # attach to the wrong vertex; rely on callers to populate
                # source_rid for correctness.
                fallback_count += 1
                params[f"name_{i}"] = edge.entity_name
                params[f"etype_{i}"] = edge.entity_type
                statements.append(
                    f"CREATE EDGE EXTRACTED_FROM "
                    f"FROM (SELECT FROM {edge.entity_type} "
                    f"WHERE name = :name_{i} AND entity_type = :etype_{i} LIMIT 1) "
                    f"TO :rid_{i} SET created_at = sysdate(){doc_set}"
                )
        if fallback_count:
            logger.warning(
                "batch_create_entity_chunk_edges_sync: %d/%d edges used the "
                "legacy name+type subquery path (EntityChunkEdge.source_rid "
                "was None). Same-name-same-type entities may attach to the "
                "wrong vertex. Callers should populate source_rid from the "
                "post-merge identity_to_rid map.",
                fallback_count, len(edges),
            )

        # Retry on ArcadeDB's "Record #N:M not found" race (read-after-write
        # visibility lag between vertex upsert and edge create in separate
        # HTTP sessions). See docs/superpowers/plans/2026-04-24-ingest-hardening-v2.md Task 4.
        import httpx
        import re
        import time as _time
        _RECORD_NOT_FOUND = re.compile(r"Record #\d+:\d+\s+not found", re.IGNORECASE)

        script = ";\n".join(statements)
        max_attempts = 3
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self._client.command_sync(
                    self._database, "sqlscript", script, params,
                )
                break
            except httpx.HTTPStatusError as exc:
                body = getattr(getattr(exc, "response", None), "text", "") or ""
                if _RECORD_NOT_FOUND.search(body) and attempt + 1 < max_attempts:
                    logger.warning(
                        "batch_create_entity_chunk_edges_sync: RecordNotFound on attempt %d — retrying",
                        attempt + 1,
                    )
                    _time.sleep(0.2 * (attempt + 1))
                    last_exc = exc
                    continue
                raise
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

    def delete_document_graph_sync(
        self,
        document_id: str,
    ) -> int:
        """Synchronous document graph deletion."""
        parts = _build_delete_document_graph_sql(document_id)
        params = parts["params"]

        total = 0
        for sql in parts["delete_vertex_sqls"]:
            try:
                result = self._client.command_sync(self._database, "sql", sql, params)
                total += _count(result)
            except Exception:
                pass

        for sql in parts["edge_cleanup_sqls"]:
            try:
                self._client.command_sync(self._database, "sql", sql, params)
            except Exception:
                pass

        for sql in parts["edge_prune_sqls"]:
            try:
                self._client.command_sync(self._database, "sql", sql)
            except Exception:
                pass

        for sql in parts["orphan_sqls"]:
            try:
                result = self._client.command_sync(self._database, "sql", sql)
                total += _count(result)
            except Exception:
                pass

        return total

    def delete_extraction_layer_graph_sync(self, document_id: str) -> int:
        """Narrower rollback primitive. Spec §6.8 + residual check #1.

        Deleted:
          - Document-scoped extracted entity vertices (SECTION, FIGURE, TABLE,
            WAVEFORM, ANTENNA, TRANSMITTER, RECEIVER, SIGNAL_PROCESSING_CHAIN,
            SEEKER, SPECIFICATION, PROCEDURE, FAILURE_MODE, TEST_EVENT,
            FORCE_STRUCTURE, SUBSYSTEM, COMPONENT, ASSEMBLY, CAPABILITY, and
            the is_entity=False components such as MODULATION, RF_SIGNATURE,
            RF_EMISSION, SCAN_PATTERN, IF_AMPLIFIER, PROPULSION_STACK,
            MISSILE_PERFORMANCE, MISSILE_PHYSICAL_CHARACTERISTICS,
            PROPULSION_STAGE, RADAR_PERFORMANCE, ENGAGEMENT_TIMELINE — per
            model_config identity_scope='document'; ASSERTION was dropped
            in B-1). The concrete list is sourced at runtime from the
            active ontology so new/retired entities are picked up without
            edits here.
          - Domain edges via ``document_ids CONTAINS :doc_id`` — removes this
            doc_id from the list, then prunes edges whose list is now empty.
            (Mirrors _build_delete_document_graph_sql; covers MENTIONED_IN
            edges written through upsert_relationships_batch_sync.)
          - HAS_PROVENANCE edges whose *in* vertex is the structural Document
            vertex with this document_id (target-scoped; edges from global
            sources are also removed without touching the source vertices).
          - MENTIONED_IN structural edges written by derive_rules via
            create_structural_edge_sync — these have no document_id field, so
            they are found by traversal: where the *in* (target) vertex is a
            TextChunk with document_id = :doc_id.

        Preserved:
          - TextChunk, ImageChunk vertices (owned by upstream stages)
          - The structural Document vertex itself
          - Global entity vertices (RADAR_SYSTEM, PLATFORM, MISSILE_SYSTEM, …)
          - HAS_PROVENANCE edges from global entities to OTHER documents
          - Domain edges that reference other documents in their document_ids
            lists (the list shrinks but is not emptied)

        Returns the total number of SQL statements that executed successfully
        (approximates the count of modified classes, NOT the count of
        individual vertices/edges). This matches delete_document_graph_sync's
        return-value semantics for logging parity.
        """
        try:
            from app.services.ontology_templates import load_ontology
            ontology = load_ontology()
        except Exception:
            ontology = {}

        # Route through _safe_type_name so reserved words (e.g. TABLE → TABLE_REF)
        # match the actual class names created in arcadedb_schema.py.
        document_scoped_entity_classes = [
            _safe_type_name(e["name"]) for e in ontology.get("entity_types", [])
            if e.get("identity_scope") == "document"
        ]
        domain_edge_classes = [
            _safe_type_name(r["name"]) for r in ontology.get("relationship_types", [])
        ]

        executed = 0
        params = {"doc_id": document_id}

        # 1. Look up the structural Document vertex RID for HAS_PROVENANCE
        #    target-scoped deletion.
        doc_rid: str | None = None
        try:
            rows = self._client.query_sync(
                self._database, "sql",
                "SELECT @rid AS rid FROM Document WHERE document_id = :doc_id",
                params,
            )
            if rows:
                doc_rid = str(rows[0].get("rid", ""))
        except Exception as exc:
            logger.warning(
                "delete_extraction_layer_graph_sync: failed to look up "
                "Document RID for %s: %s",
                document_id, exc,
            )

        # 2. Delete HAS_PROVENANCE edges whose *in* (target) is this Document
        #    vertex.  Global source vertices are untouched.
        #    ArcadeDB uses @in (not plain 'in') to reference the target vertex
        #    RID in WHERE clauses on edge types.
        if doc_rid:
            try:
                self._client.command_sync(
                    self._database, "sql",
                    "DELETE FROM HAS_PROVENANCE WHERE @in = :doc_rid",
                    {"doc_rid": doc_rid},
                )
                executed += 1
            except Exception as exc:
                logger.warning(
                    "delete_extraction_layer_graph_sync: HAS_PROVENANCE "
                    "delete failed for %s: %s",
                    document_id, exc,
                )

        # 3. Delete document-scoped extracted entity vertices.
        for vertex_class in document_scoped_entity_classes:
            try:
                self._client.command_sync(
                    self._database, "sql",
                    f"DELETE VERTEX FROM {vertex_class} WHERE document_id = :doc_id",
                    params,
                )
                executed += 1
            except Exception as exc:
                logger.debug(
                    "delete_extraction_layer_graph_sync: %s vertex delete "
                    "skipped: %s",
                    vertex_class, exc,
                )

        # 4. Domain edges: remove doc_id from document_ids list, then prune
        #    edges whose list is now empty.  Mirrors _build_delete_document_graph_sql.
        for edge_class in domain_edge_classes:
            try:
                self._client.command_sync(
                    self._database, "sql",
                    f"UPDATE {edge_class} "
                    f"SET document_ids = document_ids.remove(:doc_id) "
                    f"WHERE document_ids CONTAINS :doc_id",
                    params,
                )
                executed += 1
            except Exception as exc:
                logger.debug(
                    "delete_extraction_layer_graph_sync: %s cleanup skipped: %s",
                    edge_class, exc,
                )

        for edge_class in domain_edge_classes:
            try:
                self._client.command_sync(
                    self._database, "sql",
                    f"DELETE FROM {edge_class} "
                    f"WHERE document_ids IS NOT NULL AND document_ids.size() = 0",
                )
                executed += 1
            except Exception as exc:
                logger.debug(
                    "delete_extraction_layer_graph_sync: %s prune skipped: %s",
                    edge_class, exc,
                )

        # 5. MENTIONED_IN structural edges written by derive_rules via
        #    create_structural_edge_sync — no document_id stored on the edge
        #    itself, so filter by in-vertex (TextChunk) document_id.
        #    ArcadeDB uses @in.<prop> (not plain in.<prop>) for property
        #    traversal on edge target vertices in WHERE clauses.
        try:
            self._client.command_sync(
                self._database, "sql",
                "DELETE FROM MENTIONED_IN WHERE @in.document_id = :doc_id",
                params,
            )
            executed += 1
        except Exception as exc:
            logger.debug(
                "delete_extraction_layer_graph_sync: MENTIONED_IN "
                "(derive_rules) delete skipped: %s",
                exc,
            )

        logger.info(
            "delete_extraction_layer_graph_sync: %d SQL statements executed "
            "for document %s (preserved chunks, Document vertex, global "
            "entities, cross-doc HAS_PROVENANCE)",
            executed, document_id,
        )
        return executed

    def count_ontology_nodes_sync(
        self,
        entity_type: str,
        document_id: str | None = None,
    ) -> int:
        """Count ontology vertices of ``entity_type``, optionally scoped
        to a single ``document_id``. Missing vertex classes return 0 so
        callers need not pre-check existence.
        """
        if document_id:
            sql = (
                f"SELECT count(*) AS count FROM {entity_type} "
                "WHERE document_id = :doc_id"
            )
            params = {"doc_id": document_id}
        else:
            sql = f"SELECT count(*) AS count FROM {entity_type}"
            params = {}
        try:
            rows = self._client.query_sync(self._database, "sql", sql, params)
        except Exception as exc:
            logger.debug(
                "count_ontology_nodes_sync: %s returned error (class may "
                "not exist yet): %s",
                entity_type, exc,
            )
            return 0
        return _count(rows)

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
        """Synchronous full-text search across entity types."""
        result = _build_fulltext_search_sql(query, entity_types, limit)

        if isinstance(result[0], str):
            sql, params = result
            rows = self._client.query_sync(self._database, "sql", sql, params)
            return [_to_entity(r) for r in rows]

        sqls, params = result
        if not sqls:
            try:
                from app.services.ontology_templates import load_ontology
                from app.services.arcadedb_schema import _safe_type_name
                ont = load_ontology()
                sqls = [
                    f"SELECT *, @rid AS node_id FROM {_safe_type_name(e['name'])} "
                    f"WHERE name CONTAINSTEXT :query LIMIT :limit"
                    for e in ont.get("entity_types", [])
                ]
            except Exception:
                return []

        all_rows: list[dict] = []
        for sql in sqls:
            try:
                rows = self._client.query_sync(self._database, "sql", sql, params)
                all_rows.extend(rows)
            except Exception:
                continue

        entities = [_to_entity(r) for r in all_rows]
        entities.sort(key=lambda e: e.extraction_confidence or 0, reverse=True)
        return entities[:limit]

    def search_by_alias_sync(
        self,
        alias: str,
        entity_type: str | None = None,
    ) -> list[GraphEntityResult]:
        """Synchronous alias search."""
        sql, params = _build_search_by_alias_sql(alias)
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
        sql, params = _build_create_alias_sql(alias)
        alias_result = self._client.command_sync(
            self._database, "sql", sql, params,
        )
        alias_rid = _rid(alias_result)

        edge_sql = _build_create_alias_edge_sql(node_id, alias_rid)
        self._client.command_sync(self._database, "sql", edge_sql)

    def set_canonical_name_sync(
        self,
        node_id: str,
        canonical_name: str,
    ) -> None:
        """Synchronous canonical name setter."""
        sql, params = _build_set_canonical_name_sql(node_id, canonical_name)
        self._client.command_sync(
            self._database, "sql", sql, params,
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
        sql, params = _build_resolve_root_entity_sql(name, entity_type)
        rows = self._client.query_sync(self._database, "sql", sql, params)
        return _to_entity(rows[0]) if rows else None

    def fuzzy_match_sync(
        self,
        name: str,
        entity_type: str,
        threshold: float = 0.8,
    ) -> list[dict]:
        """Find entities with similar names using ArcadeDB's levenshteinDistance.

        Computes normalized similarity = 1 - (distance / max(len(name), len(candidate))).
        Returns candidates above *threshold*, sorted by distance.
        """
        max_dist = int(len(name) * (1.0 - threshold)) + 1
        sql = (
            f"SELECT name, canonical_name, entity_type, @rid AS node_id, "
            f"text.levenshteinDistance(name, :name) AS dist "
            f"FROM {entity_type} "
            f"WHERE text.levenshteinDistance(name, :name) <= :max_dist "
            f"AND name <> :name "
            f"ORDER BY dist LIMIT 5"
        )
        return self._client.query_sync(
            self._database, "sql", sql,
            {"name": name, "max_dist": max_dist},
        )

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

    def get_document_vertices_sync(
        self,
        document_id: str,
    ) -> list[dict]:
        """Return every vertex owned by this document.

        Two passes, unioned + deduplicated by ``@rid``:

        1. Enumerate vertex classes via ``schema:types WHERE
           type='vertex'`` and SELECT rows where ``document_id = :doc_id``.
           Covers Document, TextChunk, ImageChunk, anchor entities, and
           any document-scoped domain entities.
        2. Expand endpoints of every document-scoped edge (same filter
           as ``get_document_edges_sync``). Picks up global-scope
           entities (RADAR_SYSTEM / MISSILE_SYSTEM / etc.) that don't
           store ``document_id`` on the vertex itself but are edge
           endpoints for this doc.

        Display label is coalesced across ``name`` / ``title`` /
        ``chunk_id`` / ``document_id``.
        """
        all_rows: list[dict] = []
        seen_rids: set[str] = set()

        def _add(rows: list[dict]) -> None:
            for row in rows:
                rid = row.get("@rid")
                if not rid or rid in seen_rids:
                    continue
                seen_rids.add(rid)
                all_rows.append(row)

        # Pass 1 — per-class SELECT filtered by document_id.
        vertex_types_rows = self._client.query_sync(
            self._database, "sql",
            "SELECT name FROM schema:types WHERE type = 'vertex'",
        )
        vertex_types = [r.get("name") for r in vertex_types_rows if r.get("name")]
        for vtype in vertex_types:
            sql = (
                f"SELECT "
                f"  @type AS `@type`, @rid AS `@rid`, "
                f"  COALESCE(name, title, chunk_id, document_id) AS _label, "
                f"  entity_type, name, title, chunk_id, document_id, page "
                f"FROM {vtype} "
                f"WHERE document_id = :doc_id"
            )
            try:
                rows = self._client.query_sync(
                    self._database, "sql", sql, {"doc_id": document_id},
                )
            except Exception as exc:
                logger.debug(
                    "get_document_vertices_sync: skipped %s (%s)", vtype, exc,
                )
                continue
            _add(rows)

        # Pass 2 — collect distinct endpoint RIDs from document-scoped
        # edges, then fetch those vertices. This is how global-scope
        # entities (identity_scope='global' ontology entities) show up
        # here: they don't store document_id as a property, but their
        # edges do, so we reach them via the edge filter.
        edge_types_rows = self._client.query_sync(
            self._database, "sql",
            "SELECT name FROM schema:types WHERE type = 'edge'",
        )
        edge_types = [r.get("name") for r in edge_types_rows if r.get("name")]
        endpoint_rids: set[str] = set()
        for etype in edge_types:
            sql = (
                f"SELECT @out AS out_rid, @in AS in_rid "
                f"FROM {etype} "
                f"WHERE document_id = :doc_id "
                f"   OR document_ids CONTAINS :doc_id"
            )
            try:
                rows = self._client.query_sync(
                    self._database, "sql", sql, {"doc_id": document_id},
                )
            except Exception:
                continue
            for row in rows:
                for rid_key in ("out_rid", "in_rid"):
                    rid = row.get(rid_key)
                    if rid and rid not in seen_rids:
                        endpoint_rids.add(rid)

        for rid in endpoint_rids:
            try:
                rows = self._client.query_sync(
                    self._database, "sql",
                    f"SELECT @type AS `@type`, @rid AS `@rid`, "
                    f"COALESCE(name, title, chunk_id, document_id) AS _label, "
                    f"entity_type, name, title, chunk_id, document_id, page "
                    f"FROM {rid}",
                )
            except Exception:
                continue
            _add(rows)

        return all_rows

    def get_document_edges_sync(
        self,
        document_id: str,
    ) -> list[dict]:
        """Return all edges whose provenance includes this document.

        ArcadeDB keeps each edge type as its own top-level class (not as
        a subclass of a shared `E`), so `SELECT FROM E` raises
        SchemaException. Enumerate edge classes from `schema:types`,
        then pull rows from each one that reference this document.

        Structural edges carry `document_id` (scalar). Domain edges
        carry `document_ids` (LIST unioned across documents — see
        `_build_upsert_relationship_script`). Per-type queries cover
        both shapes — an unknown-property predicate on one branch is
        simply false, not an error.
        """
        edge_types_rows = self._client.query_sync(
            self._database, "sql",
            "SELECT name FROM schema:types WHERE type = 'edge'",
        )
        edge_types = [r.get("name") for r in edge_types_rows if r.get("name")]

        all_rows: list[dict] = []
        for etype in edge_types:
            sql = (
                f"SELECT @type AS `@type`, @rid AS `@rid`, "
                # Endpoints: coalesce naming fields across vertex classes so
                # TextChunk/ImageChunk (chunk_id), SECTION (name = section
                # uid), and entities (name) all render with something.
                f"  COALESCE(@out.name, @out.title, @out.chunk_id, @out.document_id) AS _from_label, "
                f"  COALESCE(@in.name, @in.title, @in.chunk_id, @in.document_id) AS _to_label, "
                f"  rel_type, document_id, document_ids, "
                f"  extraction_confidence "
                f"FROM {etype} "
                f"WHERE document_id = :doc_id "
                f"   OR document_ids CONTAINS :doc_id"
            )
            try:
                rows = self._client.query_sync(
                    self._database, "sql", sql, {"doc_id": document_id},
                )
            except Exception as exc:
                logger.debug(
                    "get_document_edges_sync: skipped %s (%s)", etype, exc,
                )
                continue
            all_rows.extend(rows)
        return all_rows

    def close_sync(self) -> None:
        """Release any held resources."""
        self._client.close_sync()
