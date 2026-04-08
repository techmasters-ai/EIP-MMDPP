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
    "TextChunk": [
        ("chunk_id", "STRING"),
        ("document_id", "STRING"),
        ("page_number", "INTEGER"),
        ("modality", "STRING"),
        ("classification", "STRING"),
        ("text_embedding", "ARRAY_OF_FLOATS"),
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
    "HAS_PROVENANCE",
    "EXTRACTED_FROM",
    "HAS_ALIAS",
]

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
    ("extraction_confidence", "DOUBLE"),
    ("created_at", "DATETIME"),
    ("updated_at", "DATETIME"),
]

# Common properties on structural edge types
_STRUCTURAL_EDGE_PROPS = [
    ("document_id", "STRING"),
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
        for prop_name, prop_def in props_schema.items():
            yaml_type = prop_def.get("type", "string")
            arcade_type = _YAML_TO_ARCADE.get(yaml_type, "STRING")
            entity_ddl.append(
                f"CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {arcade_type}"
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
    vector_indexes = [
        ("TextChunk", "text_embedding", 1024, "COSINE", "INT8", True),
        ("ImageChunk", "image_embedding", 512, "COSINE", "INT8", False),
        ("CommunityReport", "report_embedding", 1024, "COSINE", "INT8", True),
        ("TrustedTextChunk", "text_embedding", 1024, "COSINE", "INT8", True),
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

    # --- Phase 7: unique indexes ---
    unique_indexes = [
        ("TextChunk", "chunk_id"),
        ("ImageChunk", "chunk_id"),
        ("Document", "document_id"),
        ("Alias", "alias_name"),
        ("TrustedTextChunk", "chunk_id"),
    ]
    unique_ddl = [
        f"CREATE INDEX IF NOT EXISTS ON {utype} ({uprop}) UNIQUE_HASH_INDEX"
        for utype, uprop in unique_indexes
    ]
    await _run_ddl_batch(client, database, unique_ddl, phase="unique_indexes", report=report)
    report.indexes_created += len(unique_ddl)

    # --- Phase 8: BucketSelectionStrategy 'thread' for write-heavy types ---
    # Eliminates contention and ConcurrentModificationException on parallel
    # pipeline ingestion (ArcadeDB Manual §5.5.24).
    write_heavy_types = ["TextChunk", "ImageChunk", "TrustedTextChunk"]
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
