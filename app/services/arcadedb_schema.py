"""Ontology-driven ArcadeDB schema sync."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.services.graph_store import SchemaSyncReport

logger = logging.getLogger(__name__)

# Loaded from the shared JSON file so both the main app and docling-graph
# service use the same mapping (see ontology/arcadedb_reserved_words.json).
_RESERVED_WORDS_PATH = Path(__file__).resolve().parents[2] / "ontology" / "arcadedb_reserved_words.json"
try:
    RESERVED_WORD_MAP: dict[str, str] = json.loads(_RESERVED_WORDS_PATH.read_text())
except Exception:
    RESERVED_WORD_MAP = {"TABLE": "TABLE_REF"}

_YAML_TO_ARCADE: dict[str, str] = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "DOUBLE",
    "boolean": "BOOLEAN",
}

# Structural vertex types (not from ontology)
_STRUCTURAL_VERTEX_TYPES = {
    # VR Phase C.1 (rev 10): ephemeral per-pipeline-run chunk index used by the
    # Vector Router to narrow extraction scope.  Indexed on pipeline_run_id
    # (filter dimension) and created_at (janitor age-sweep — rev 8 M5).
    # HNSW LSM_VECTOR index on `embedding` (dim=1024, cosine) matches bge-m3
    # output and the existing TextChunk.text_embedding dim.
    # NOTE: `chunk_text` is the stored field name; callers remap to
    # "content_text" at the reranker call site ONLY — not here.
    "ExtractionChunk": [
        ("vertex_id", "STRING"),         # synthetic PK: f"{pipeline_run_id}:{self_ref}" (legacy)
                                         #            or f"{pipeline_run_id}:chunk_{chunk_index}" (merged)
        ("pipeline_run_id", "STRING"),   # filter dimension (B-tree indexed)
        ("document_id", "STRING"),
        ("self_ref", "STRING"),          # e.g. "#/texts/12" (legacy per-element rows only)
        ("chunk_text", "STRING"),
        ("embedding", "ARRAY_OF_FLOATS"),  # dim=1024, cosine HNSW
        ("page_number", "INTEGER"),
        ("modality", "STRING"),          # text | table | picture_caption
        ("created_at", "DATETIME"),      # set via sysdate(); janitor sweep key (rev 8 M5)
        # --- Phase 1 Task 1 (merged-chunk routing) -------------------
        # Defaults applied APPLICATION-SIDE via read_chunk_* accessors
        # in extraction_chunk_index.py; NO ArcadeDB DEFAULT clause.
        # Legacy per-element rows carry: chunk_index=-1, source_refs=[],
        # token_count=0. New merged-mode rows carry real values.
        ("chunk_index", "INTEGER"),      # position in HybridChunker output (-1 = legacy)
        ("source_refs", "LIST"),         # element self_refs covered by this merged chunk
        ("token_count", "INTEGER"),      # tokenizer.count_tokens(chunk_text); diagnostics
    ],
    "TextChunk": [
        ("chunk_id", "STRING"),
        ("document_id", "STRING"),
        ("page_number", "INTEGER"),
        ("modality", "STRING"),
        ("classification", "STRING"),
        ("text_embedding", "ARRAY_OF_FLOATS"),
        # Spec 2026-05-11-table-aware-chunking §11.4 — populated from
        # chunk_metadata.chunk_kind on the Postgres TextChunk row. NULL
        # for prose / heading / non-table chunks.
        ("chunk_kind", "STRING"),
    ],
    "ImageChunk": [
        ("chunk_id", "STRING"),
        ("document_id", "STRING"),
        ("artifact_id", "STRING"),
        ("page_number", "INTEGER"),
        ("image_embedding", "ARRAY_OF_FLOATS"),
    ],
    "TrustedTextChunk": [
        ("chunk_id", "STRING"),
        ("document_id", "STRING"),
        ("content_text", "STRING"),
        ("source", "STRING"),
        ("classification", "STRING"),
        ("text_embedding", "ARRAY_OF_FLOATS"),
    ],
    "Document": [
        ("document_id", "STRING"),
        ("title", "STRING"),
        ("upload_datetime", "DATETIME"),
        ("document_datetime", "DATETIME"),
        # Free-text mirror of the owning collection's name (the Collection
        # vertex carries the same value; threaded here so a Document row is
        # self-describing without a BELONGS_TO traversal).
        ("source_name", "STRING"),
    ],
    # First-class collection ("source") vertex. One per ingest.sources row;
    # keyed on source_id (UNIQUE — see Phase 7). Documents in the same
    # collection share ONE Collection vertex via Document -BELONGS_TO-> Collection.
    # NOT an ontology entity type — structural infrastructure only.
    "Collection": [
        ("source_id", "STRING"),
        ("name", "STRING"),
    ],
    "Alias": [
        ("alias_name", "STRING"),
    ],
    "CommunityReport": [
        ("community_id", "INTEGER"),
        ("membership_hash", "STRING"),
        ("title", "STRING"),
        ("summary", "STRING"),
        ("member_count", "INTEGER"),
        ("key_entities", "LIST"),
        ("key_relationships", "LIST"),
        ("report_embedding", "ARRAY_OF_FLOATS"),
        ("source_documents", "LIST"),
        ("model_name", "STRING"),
        ("generated_at", "DATETIME"),
    ],
}

# Structural edge types
_STRUCTURAL_EDGE_TYPES = [
    "CONTAINS_TEXT",
    "CONTAINS_IMAGE",
    "SAME_PAGE",
    "SAME_SECTION",
    "SAME_ARTIFACT",
    "NEXT_CHUNK",
    "HAS_PROVENANCE",
    "EXTRACTED_FROM",
    "HAS_ALIAS",
    # Document-anchor edges (per spec §3.5a).
    # Intentionally OUTSIDE ontology validation (RelationshipType/VALIDATION_MATRIX).
    "HAS_SECTION",
    "HAS_FIGURE",
    "HAS_TABLE",
    "CHILD_OF",
    # Document -> Collection membership. Structural (no source_chunk_ids
    # lineage, not an ontology relationship type) so the per-run domain-edge
    # retract (Fix N) and the doc-scoped lineage gate both exclude it.
    "BELONGS_TO",
]

# Document-anchor + derive-rules STRUCTURAL edge types that ARE declared in the
# ontology relationship_types registry (so they would otherwise look like domain
# rels) but are built by the docling document-anchor walker / derive_rules
# structural path (via create_structural_edge_sync), NOT the ontology
# relationship path — they legitimately carry no source_chunk_ids lineage and are
# NOT per-document-retractable domain edges.
#
# Single source of truth for the four extras that the schema's
# _STRUCTURAL_EDGE_TYPES list omits (it lists the HAS_SECTION/HAS_FIGURE/HAS_TABLE/
# CHILD_OF doc-anchor subset but not HAS_IMAGE / NEAR_TEXT / the chunk-containment
# CONTAINS / the derive-rules MENTIONED_IN). Consumed by BOTH:
#   * scripts/verify_lineage_e2e.py — STRUCTURAL_EDGE_TYPES (the doc-scoped
#     fail-closed lineage gate's domain/structural partition).
#   * app/workers/pipeline.py::_domain_relationship_edge_types — the per-run
#     domain-edge retract's structural exclusion set.
# Promoting it here guarantees the retract and the gate cannot diverge on what
# counts as a domain edge (anti-drift).
_ANCHOR_DERIVE_STRUCTURAL_EDGE_TYPES = ("HAS_IMAGE", "NEAR_TEXT", "CONTAINS", "MENTIONED_IN")

# Common properties on all ontology entity vertex types
_COMMON_ENTITY_PROPS = [
    ("id", "STRING"),
    ("name", "STRING"),
    ("entity_type", "STRING"),
    ("canonical_name", "STRING"),
    ("extraction_confidence", "DOUBLE"),
    ("created_at", "DATETIME"),
    ("updated_at", "DATETIME"),
]

# Common properties on ontology relationship edge types
_COMMON_EDGE_PROPS = [
    ("document_ids", "LIST"),
    ("pipeline_run_id", "STRING"),
    ("extraction_confidence", "DOUBLE"),
    ("created_at", "DATETIME"),
    ("updated_at", "DATETIME"),
    # Task 5: per-edge source-chunk lineage resolved in the merge phase from
    # the relationship's positional self_refs (no fan-out, no RID round-trip).
    ("source_chunk_ids", "LIST"),
    ("source_pages", "LIST"),
    ("source_self_refs", "LIST"),
]

# Common properties on structural edge types
_STRUCTURAL_EDGE_PROPS = [
    ("document_id", "STRING"),
    ("pipeline_run_id", "STRING"),
    ("extraction_confidence", "DOUBLE"),
    ("created_at", "DATETIME"),
    ("page_numbers", "LIST"),
    ("upload_datetime", "DATETIME"),
    ("document_datetime", "DATETIME"),
]


# Exported for use by arcadedb_community.py and other modules that need to
# distinguish structural (infrastructure) types from domain entity types.
STRUCTURAL_TYPES: frozenset[str] = frozenset(
    list(_STRUCTURAL_VERTEX_TYPES.keys()) + ["TABLE_REF"]
)


def _safe_type_name(ontology_name: str) -> str:
    """Map reserved ArcadeDB SQL words to safe alternatives."""
    return RESERVED_WORD_MAP.get(ontology_name, ontology_name)


async def _run_ddl_batch(
    client: Any,
    database: str,
    statements: list[str],
    *,
    phase: str,
    report: SchemaSyncReport,
) -> None:
    """Execute a batch of idempotent DDL statements as a single sqlscript.

    Falls back to per-statement execution on batch failure so one bad row
    doesn't silently drop the rest of the phase. All statements are expected
    to use ``IF NOT EXISTS`` so re-runs are no-ops.
    """
    if not statements:
        return
    script = ";\n".join(statements)
    try:
        await client.command(database, "sqlscript", script)
        return
    except Exception as exc:
        logger.warning(
            "Schema %s batch failed (%s); falling back to per-statement execution",
            phase, exc,
        )

    for sql in statements:
        try:
            await client.command(database, "sql", sql)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" not in msg:
                report.errors.append(f"{phase}: {exc}")


async def _backfill_merged_chunk_columns(client: Any, database: str) -> None:
    """Coalesce legacy NULLs on ExtractionChunk's merged-chunk columns to defaults.

    Three columns were added in Phase 1 Task 1 of the merged-chunk routing
    plan without ArcadeDB DEFAULT clauses (application-side coalescing via
    the ``read_chunk_*`` accessors). Pre-existing rows ingested before that
    change carry NULL in these columns; this helper applies the documented
    legacy defaults:

      * ``chunk_index = -1`` (legacy marker; merged rows have a real position)
      * ``token_count = 0``  (legacy rows: no tokenizer counted them)
      * ``source_refs = []`` (legacy rows: no constituent ref list known)

    Idempotency: each statement is filtered by ``WHERE col IS NULL`` so a
    second invocation against an already-defaulted DB updates zero rows.
    Safe to run on every startup. Errors are logged but do NOT propagate —
    the schema-sync caller continues so a transient backfill failure does
    not block the rest of the schema sync.
    """
    backfills: list[tuple[str, str]] = [
        (
            "chunk_index",
            "UPDATE ExtractionChunk SET chunk_index = -1 WHERE chunk_index IS NULL",
        ),
        (
            "token_count",
            "UPDATE ExtractionChunk SET token_count = 0 WHERE token_count IS NULL",
        ),
        (
            "source_refs",
            "UPDATE ExtractionChunk SET source_refs = [] WHERE source_refs IS NULL",
        ),
    ]

    for col, sql in backfills:
        try:
            result = await client.command(database, "sql", sql)
            # ArcadeDB UPDATE returns [{"count": N}] (per-statement) on success.
            count = 0
            if isinstance(result, list) and result and isinstance(result[0], dict):
                count = int(result[0].get("count", 0))
            if count > 0:
                logger.info(
                    "ExtractionChunk backfill: column=%s rows_updated=%d "
                    "(coalesced NULL → legacy default)",
                    col, count,
                )
            else:
                logger.debug(
                    "ExtractionChunk backfill: column=%s no NULL rows to update",
                    col,
                )
        except Exception as exc:
            # Non-fatal: the application-side accessors still coalesce on read.
            # Most common cause early in a fresh DB lifecycle is that the
            # ExtractionChunk vertex type was created moments earlier and the
            # schema is still propagating — the next schema-sync will retry.
            logger.warning(
                "ExtractionChunk backfill failed for column=%s — %s: %s "
                "(read-side accessors still coalesce NULLs to the default)",
                col, type(exc).__name__, exc,
            )


async def sync_schema_from_ontology(
    client: Any,  # ArcadeDBClient
    database: str,
    ontology: dict[str, Any],
) -> SchemaSyncReport:
    """Sync ArcadeDB schema to match ontology definition.

    Additive only — creates new types/properties, never drops.
    Idempotent — uses IF NOT EXISTS on all CREATE statements.
    Batched — each phase is submitted as a single sqlscript call instead of
    issuing ~200 sequential HTTP requests.
    """
    report = SchemaSyncReport()

    # --- Phase 1: ontology entity vertex types + their properties ---
    entity_ddl: list[str] = []
    entity_types: list[str] = []
    for entity_def in ontology.get("entity_types", []):
        etype = _safe_type_name(entity_def["name"])
        entity_types.append(etype)
        entity_ddl.append(f"CREATE VERTEX TYPE {etype} IF NOT EXISTS")
        for prop_name, prop_type in _COMMON_ENTITY_PROPS:
            entity_ddl.append(
                f"CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {prop_type}"
            )
        props_schema = entity_def.get("properties", {}).get("properties", {})
        declared_props: set[str] = set()
        for prop_name, prop_def in props_schema.items():
            yaml_type = prop_def.get("type", "string")
            arcade_type = _YAML_TO_ARCADE.get(yaml_type, "STRING")
            entity_ddl.append(
                f"CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {arcade_type}"
            )
            declared_props.add(prop_name)
        # Phase 8 follow-up (Phase 6b upsert-index fix): document-scoped
        # entities get indexed on a composite that includes document_id.
        # Some Pydantic entity classes don't declare document_id as a
        # field (walker stamps it at upsert-time via
        # _to_merged_entity_record's ``properties.setdefault``), so the
        # introspected properties_schema lacks it — the subsequent
        # CREATE INDEX would fail "property does not exist". Declare it
        # explicitly here for document-scoped types.
        if entity_def.get("identity_scope", "global") == "document" \
                and "document_id" not in declared_props:
            entity_ddl.append(
                f"CREATE PROPERTY {etype}.document_id IF NOT EXISTS STRING"
            )
    await _run_ddl_batch(client, database, entity_ddl, phase="entity_types", report=report)
    report.types_created += len(entity_types)
    # Property count is an upper bound — we don't parse sqlscript result per-statement
    report.properties_added += (
        len(entity_types) * len(_COMMON_ENTITY_PROPS)
        + sum(
            len(e.get("properties", {}).get("properties", {}))
            for e in ontology.get("entity_types", [])
        )
    )

    # --- Phase 2: ontology relationship edge types + their properties ---
    rel_ddl: list[str] = []
    rel_types: list[str] = []
    for rel_def in ontology.get("relationship_types", []):
        rtype = rel_def["name"]
        rel_types.append(rtype)
        rel_ddl.append(f"CREATE EDGE TYPE {rtype} IF NOT EXISTS")
        for prop_name, prop_type in _COMMON_EDGE_PROPS:
            rel_ddl.append(
                f"CREATE PROPERTY {rtype}.{prop_name} IF NOT EXISTS {prop_type}"
            )
    await _run_ddl_batch(client, database, rel_ddl, phase="relationship_types", report=report)
    report.types_created += len(rel_types)

    # --- Phase 3: structural vertex types + properties ---
    struct_vertex_ddl: list[str] = []
    for stype, props in _STRUCTURAL_VERTEX_TYPES.items():
        struct_vertex_ddl.append(f"CREATE VERTEX TYPE {stype} IF NOT EXISTS")
        for prop_name, prop_type in props:
            struct_vertex_ddl.append(
                f"CREATE PROPERTY {stype}.{prop_name} IF NOT EXISTS {prop_type}"
            )
    await _run_ddl_batch(
        client, database, struct_vertex_ddl,
        phase="structural_vertex_types", report=report,
    )
    report.types_created += len(_STRUCTURAL_VERTEX_TYPES)

    # --- Phase 3b: idempotent backfill of merged-chunk legacy defaults ---
    # M1 (Phase 1 Task 1 code-review fix): the three columns added for
    # merged-chunk routing — chunk_index, token_count, source_refs — have
    # NO ArcadeDB DEFAULT clause. Application-side coalescing via
    # read_chunk_* accessors handles NULLs on read, but a fresh DB / a
    # restore-from-snapshot would leave legacy rows with NULL until the
    # operator re-ran the migration UPDATEs by hand. Wiring the same
    # UPDATEs into the schema-sync helper makes the backfill automatic.
    #
    # Idempotency: each WHERE col IS NULL filter makes the UPDATE a no-op
    # on a row that already carries the default — safe to re-run on every
    # startup.
    await _backfill_merged_chunk_columns(client, database)

    # --- Phase 4: structural edge types + properties ---
    struct_edge_ddl: list[str] = []
    for etype in _STRUCTURAL_EDGE_TYPES:
        struct_edge_ddl.append(f"CREATE EDGE TYPE {etype} IF NOT EXISTS")
        for prop_name, prop_type in _STRUCTURAL_EDGE_PROPS:
            struct_edge_ddl.append(
                f"CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {prop_type}"
            )
    await _run_ddl_batch(
        client, database, struct_edge_ddl,
        phase="structural_edge_types", report=report,
    )
    report.types_created += len(_STRUCTURAL_EDGE_TYPES)

    # --- Phase 5: vector indexes ---
    from app.config import get_settings as _get_settings
    _image_dim = _get_settings().image_embedding_dim
    vector_indexes = [
        ("TextChunk", "text_embedding", 1024, "COSINE", "INT8", True),
        ("ImageChunk", "image_embedding", _image_dim, "COSINE", "INT8", False),
        ("CommunityReport", "report_embedding", 1024, "COSINE", "INT8", True),
        ("TrustedTextChunk", "text_embedding", 1024, "COSINE", "INT8", True),
        # VR C.1: ExtractionChunk HNSW index — dim=1024 matches bge-m3 + TextChunk
        ("ExtractionChunk", "embedding", 1024, "COSINE", "INT8", True),
    ]
    vector_ddl: list[str] = []
    for vtype, vprop, dims, sim, quant, hier in vector_indexes:
        meta = f"dimensions: {dims}, similarity: '{sim}', quantization: '{quant}'"
        if hier:
            meta += ", addHierarchy: true"
        vector_ddl.append(
            f"CREATE INDEX IF NOT EXISTS ON {vtype} ({vprop}) LSM_VECTOR METADATA {{{meta}}}"
        )
    await _run_ddl_batch(client, database, vector_ddl, phase="vector_indexes", report=report)
    report.indexes_created += len(vector_ddl)

    # --- Phase 6: fulltext indexes on ontology entity names ---
    fulltext_ddl = [
        f"CREATE INDEX IF NOT EXISTS ON {_safe_type_name(e['name'])} (name) FULL_TEXT"
        for e in ontology.get("entity_types", [])
    ]
    await _run_ddl_batch(client, database, fulltext_ddl, phase="fulltext_indexes", report=report)
    report.indexes_created += len(fulltext_ddl)

    # --- Phase 6b: composite UNIQUE indexes for UPSERT on entity types ---
    # ArcadeDB UPSERT requires a UNIQUE index on the WHERE fields.
    # Without this, UPDATE...UPSERT WHERE {identity_fields} AND entity_type=:entity_type
    # fails with "Upsert must involve an index to retrieve the records."
    #
    # Post-docs-alignment (B-3..B-6): identity fields are ontology-driven
    # (section_number for SECTION, figure_ref for FIGURE, table_ref for TABLE,
    # document_number for DOCUMENT, system_name for RADAR_SYSTEM, etc.) NOT
    # a universal ``(name, entity_type)`` anymore. The index must mirror the
    # upsert WHERE clause exactly:
    #   * document-scoped entities include ``document_id`` in the composite
    #     (LogicalIdentity.as_upsert_identity_dict appends it).
    #   * ``entity_type`` closes the composite since the SQL filters on it.
    #   * Components (identity_fields=[]) have nothing indexable for UPSERT
    #     and are skipped — they don't go through upsert_nodes_batch_sync,
    #     they're embedded via the A0-1 walker.
    #
    # Case-insensitive identity (Task 2): the write layer (Task 1,
    # ``app/services/arcadedb_graph.py::_key_fields``) now matches/persists a
    # normalized ``<field>_key`` (= ``norm(value)``) alongside every raw
    # identity field so case/whitespace variants (``FAN SONG`` vs
    # ``Fan Song``) converge on one vertex. This block declares that
    # ``<field>_key`` property (STRING, mirrors the Phase 1 CREATE PROPERTY
    # idiom exactly) and a UNIQUE index over the normalized keys, so
    # ArcadeDB UPSERT can dedup on ``<field>_key`` the same way it already
    # dedups on the raw field below. The raw-field UNIQUE index is
    # intentionally KEPT (other code may still read it; it's also a
    # rollback fallback) — this is additive, not a replacement.
    # ``document_id`` is never ``_key``-normalized (opaque UUID, not a
    # case/whitespace-variant display string) — mirrors ``_key_fields``.
    upsert_ddl: list[str] = []
    for e in ontology.get("entity_types", []):
        id_fields: list[str] = list(e.get("identity_fields") or [])
        if not id_fields:
            continue  # component-class — no upsert path, no index needed
        etype = _safe_type_name(e["name"])
        scope = e.get("identity_scope", "document")
        doc_scoped = scope == "document" and "document_id" not in id_fields

        # (a) CREATE PROPERTY for every normalized <field>_key, BEFORE any
        # index DDL for this type — the key index below references these
        # columns and ArcadeDB requires the property to exist first.
        # ``document_id`` is skipped: mirrors ``_key_fields`` (Task 1), it's
        # an opaque UUID, not a case/whitespace-variant display string.
        # Mutual-consistency invariant: this exclusion must match
        # ``_key_fields`` in arcadedb_graph.py exactly. If a doc-scoped type
        # ever listed ``document_id`` inside its own identity_fields, BOTH the
        # write layer and this index would drop document scoping from the
        # ``_key`` — no live type does this today (air_defense_v3 checked).
        key_fields = [f for f in id_fields if f != "document_id"]
        for field in key_fields:
            upsert_ddl.append(
                f"CREATE PROPERTY {etype}.{field}_key IF NOT EXISTS STRING"
            )

        # Raw-field UNIQUE index (unchanged, kept for rollback fallback and
        # other readers).
        fields = list(id_fields)
        if doc_scoped:
            fields.append("document_id")
        fields.append("entity_type")
        upsert_ddl.append(
            f"CREATE INDEX IF NOT EXISTS ON {etype} "
            f"({', '.join(fields)}) UNIQUE"
        )

        # (b) Normalized <field>_key UNIQUE index — what UPSERT dedups on
        # under case-insensitive identity. This index is created BOTH here
        # (always-on schema sync, main.py lifespan) AND idempotently
        # (IF NOT EXISTS) by the Task 3 migration.
        # VERIFIED on live ArcadeDB: a UNIQUE (LSM_TREE) index permits
        # MULTIPLE NULLs, so creating it against existing rows whose _key is
        # still NULL (pre-backfill) succeeds and is inert until Task 3
        # backfills real values.
        # Ordering dependency: Task 3 MUST merge the case-collision pairs
        # BEFORE backfilling _key — otherwise two rows would backfill to the
        # SAME _key and violate this UNIQUE index. See
        # docs/superpowers/plans/2026-07-01-case-insensitive-identity.md
        # Task 3 ("merge → backfill → index").
        key_index_fields = [f"{f}_key" for f in key_fields]
        if doc_scoped:
            key_index_fields.append("document_id")
        key_index_fields.append("entity_type")
        upsert_ddl.append(
            f"CREATE INDEX IF NOT EXISTS ON {etype} "
            f"({', '.join(key_index_fields)}) UNIQUE"
        )
    await _run_ddl_batch(client, database, upsert_ddl, phase="upsert_indexes", report=report)
    report.indexes_created += len(upsert_ddl)

    # --- Phase 7: unique indexes ---
    unique_indexes = [
        ("TextChunk", "chunk_id"),
        ("ImageChunk", "chunk_id"),
        ("Document", "document_id"),
        ("Alias", "alias_name"),
        ("TrustedTextChunk", "chunk_id"),
        ("CommunityReport", "community_id"),
        # VR C.1: ExtractionChunk synthetic PK — unique per pipeline_run_id:self_ref pair
        ("ExtractionChunk", "vertex_id"),
        # First-class collection: one Collection vertex per source_id. The
        # UNIQUE index is also what lets the population path UPSERT on source_id
        # (ArcadeDB UPSERT requires a UNIQUE index on the WHERE field).
        ("Collection", "source_id"),
    ]
    unique_ddl = [
        f"CREATE INDEX IF NOT EXISTS ON {utype} ({uprop}) UNIQUE"
        for utype, uprop in unique_indexes
    ]
    await _run_ddl_batch(client, database, unique_ddl, phase="unique_indexes", report=report)
    report.indexes_created += len(unique_ddl)

    # --- Phase 7b: non-unique secondary indexes for filtering ---
    # Spec 2026-05-11-table-aware-chunking §11.4 — TextChunk.chunk_kind
    # enables fast filtering by chunk_kind in retrieval queries.
    # VR C.1: ExtractionChunk.pipeline_run_id — vector_search filter dimension.
    # VR C.1: ExtractionChunk.created_at — janitor age-sweep (rev 8 M5).
    # (vertex_id UNIQUE index handled separately in Phase 7 above — do NOT add
    # it here; adding it to secondary_indexes would create a NOTUNIQUE shadow
    # index on a column that already has a UNIQUE index.)
    secondary_indexes = [
        ("TextChunk", "chunk_kind"),
        ("ExtractionChunk", "pipeline_run_id"),
        ("ExtractionChunk", "created_at"),
    ]
    secondary_ddl = [
        f"CREATE INDEX IF NOT EXISTS ON {stype} ({sprop}) NOTUNIQUE"
        for stype, sprop in secondary_indexes
    ]
    await _run_ddl_batch(client, database, secondary_ddl, phase="secondary_indexes", report=report)
    report.indexes_created += len(secondary_ddl)

    # --- Phase 8: BucketSelectionStrategy 'thread' for write-heavy types ---
    # Eliminates contention and ConcurrentModificationException on parallel
    # pipeline ingestion (ArcadeDB Manual §5.5.24).
    # VR C.1: ExtractionChunk is bulk-written by build_extraction_index (C.2)
    # at every pipeline_run start — hundreds of vertices per run, potentially
    # concurrent across runs. Apply the thread bucket strategy before C.2 lands.
    write_heavy_types = ["TextChunk", "ImageChunk", "TrustedTextChunk", "ExtractionChunk"]
    for e in ontology.get("entity_types", []):
        write_heavy_types.append(_safe_type_name(e["name"]))
    bucket_ddl = [
        f"ALTER TYPE {t} BucketSelectionStrategy `thread`"
        for t in write_heavy_types
    ]
    await _run_ddl_batch(client, database, bucket_ddl, phase="bucket_strategy", report=report)

    logger.info(
        "Schema sync: %d types, %d properties, %d indexes, %d errors",
        report.types_created,
        report.properties_added,
        report.indexes_created,
        len(report.errors),
    )

    return report


def sync_schema_from_ontology_sync(
    client: Any,
    database: str,
    ontology: dict[str, Any],
) -> SchemaSyncReport:
    """Synchronous wrapper for use during startup (e.g., before event loop is running)."""
    import asyncio

    return asyncio.run(sync_schema_from_ontology(client, database, ontology))


async def ensure_schema(client: Any, database: str) -> SchemaSyncReport:
    """Drive ArcadeDB DDL purely from Pydantic introspection (no YAML read).

    Additive-only (CREATE ... IF NOT EXISTS throughout). Any destructive
    DROP must be issued by the caller *before* ensure_schema runs — the
    Chunk G migration script owns that boundary (spec §5.2 step 4).

    Composes :func:`build_ontology_dict` with
    :func:`sync_schema_from_ontology`; consumers no longer load
    ``ontology.yaml`` to prime the schema.
    """
    from ontology_bundles.air_defense_v3.introspect import build_ontology_dict

    return await sync_schema_from_ontology(client, database, build_ontology_dict())


def ensure_schema_sync(client: Any, database: str) -> SchemaSyncReport:
    """Synchronous adapter for :func:`ensure_schema`.

    For the migration script and other callers that run outside an
    event loop. Mirrors :func:`sync_schema_from_ontology_sync`.
    """
    import asyncio

    return asyncio.run(ensure_schema(client, database))
