"""Ontology-driven ArcadeDB schema sync."""

from __future__ import annotations

import logging
from typing import Any

from app.services.graph_store import SchemaSyncReport

logger = logging.getLogger(__name__)

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
        ("text_embedding", "LIST"),
    ],
    "ImageChunk": [
        ("chunk_id", "STRING"),
        ("document_id", "STRING"),
        ("artifact_id", "STRING"),
        ("page_number", "INTEGER"),
        ("image_embedding", "LIST"),
    ],
    "TrustedTextChunk": [
        ("chunk_id", "STRING"),
        ("document_id", "STRING"),
        ("content_text", "STRING"),
        ("source", "STRING"),
        ("classification", "STRING"),
        ("text_embedding", "LIST"),
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
        ("report_embedding", "LIST"),
        ("model_name", "STRING"),
        ("generated_at", "DATETIME"),
    ],
}

# Structural edge types
_STRUCTURAL_EDGE_TYPES = [
    "CONTAINS_TEXT",
    "CONTAINS_IMAGE",
    "SAME_PAGE",
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


def _safe_type_name(ontology_name: str) -> str:
    """Map reserved ArcadeDB SQL words to safe alternatives."""
    return RESERVED_WORD_MAP.get(ontology_name, ontology_name)


async def sync_schema_from_ontology(
    client: Any,  # ArcadeDBClient
    database: str,
    ontology: dict[str, Any],
) -> SchemaSyncReport:
    """Sync ArcadeDB schema to match ontology definition.

    Additive only — creates new types/properties, never drops.
    Idempotent — uses IF NOT EXISTS on all CREATE statements.
    """
    report = SchemaSyncReport()

    # 1. Create ontology entity vertex types
    for entity_def in ontology.get("entity_types", []):
        etype = _safe_type_name(entity_def["name"])
        try:
            await client.command(database, "sql", f"CREATE VERTEX TYPE {etype} IF NOT EXISTS")
            report.types_created += 1
        except Exception as e:
            if "already exists" not in str(e).lower():
                report.errors.append(f"Failed to create type {etype}: {e}")
                continue

        # Add common entity properties
        for prop_name, prop_type in _COMMON_ENTITY_PROPS:
            try:
                await client.command(
                    database, "sql",
                    f"CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {prop_type}",
                )
                report.properties_added += 1
            except Exception:
                pass

        # Add ontology-specific properties
        props_schema = entity_def.get("properties", {}).get("properties", {})
        for prop_name, prop_def in props_schema.items():
            yaml_type = prop_def.get("type", "string")
            arcade_type = _YAML_TO_ARCADE.get(yaml_type, "STRING")
            try:
                await client.command(
                    database, "sql",
                    f"CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {arcade_type}",
                )
                report.properties_added += 1
            except Exception:
                pass

    # 2. Create ontology relationship edge types
    for rel_def in ontology.get("relationship_types", []):
        rtype = rel_def["name"]
        try:
            await client.command(database, "sql", f"CREATE EDGE TYPE {rtype} IF NOT EXISTS")
            report.types_created += 1
        except Exception as e:
            if "already exists" not in str(e).lower():
                report.errors.append(f"Failed to create edge type {rtype}: {e}")
                continue

        for prop_name, prop_type in _COMMON_EDGE_PROPS:
            try:
                await client.command(
                    database, "sql",
                    f"CREATE PROPERTY {rtype}.{prop_name} IF NOT EXISTS {prop_type}",
                )
            except Exception:
                pass

    # 3. Create structural vertex types
    for stype, props in _STRUCTURAL_VERTEX_TYPES.items():
        try:
            await client.command(database, "sql", f"CREATE VERTEX TYPE {stype} IF NOT EXISTS")
            report.types_created += 1
        except Exception:
            pass
        for prop_name, prop_type in props:
            try:
                await client.command(
                    database, "sql",
                    f"CREATE PROPERTY {stype}.{prop_name} IF NOT EXISTS {prop_type}",
                )
            except Exception:
                pass

    # 4. Create structural edge types
    for etype in _STRUCTURAL_EDGE_TYPES:
        try:
            await client.command(database, "sql", f"CREATE EDGE TYPE {etype} IF NOT EXISTS")
            report.types_created += 1
        except Exception:
            pass
        for prop_name, prop_type in _STRUCTURAL_EDGE_PROPS:
            try:
                await client.command(
                    database, "sql",
                    f"CREATE PROPERTY {etype}.{prop_name} IF NOT EXISTS {prop_type}",
                )
            except Exception:
                pass

    # 5. Create vector indexes
    vector_indexes = [
        ("TextChunk", "text_embedding", 1024, "COSINE", "INT8", True),
        ("ImageChunk", "image_embedding", 512, "COSINE", "INT8", False),
        ("CommunityReport", "report_embedding", 1024, "COSINE", "INT8", True),
        ("TrustedTextChunk", "text_embedding", 1024, "COSINE", "INT8", True),
    ]
    for vtype, vprop, dims, sim, quant, hier in vector_indexes:
        meta = f"dimensions: {dims}, similarity: '{sim}', quantization: '{quant}'"
        if hier:
            meta += ", addHierarchy: true"
        try:
            await client.command(
                database, "sql",
                f"CREATE INDEX IF NOT EXISTS ON {vtype} ({vprop}) LSM_VECTOR METADATA {{{meta}}}",
            )
            report.indexes_created += 1
        except Exception as e:
            if "already exists" not in str(e).lower():
                report.errors.append(f"Vector index on {vtype}.{vprop}: {e}")

    # 6. Create fulltext indexes on entity names (for each ontology entity type)
    for entity_def in ontology.get("entity_types", []):
        etype = _safe_type_name(entity_def["name"])
        try:
            await client.command(
                database, "sql",
                f"CREATE INDEX IF NOT EXISTS ON {etype} (name) FULL_TEXT",
            )
            report.indexes_created += 1
        except Exception:
            pass

    # 7. Create unique indexes on structural types
    unique_indexes = [
        ("TextChunk", "chunk_id"),
        ("ImageChunk", "chunk_id"),
        ("Document", "document_id"),
        ("Alias", "alias_name"),
    ]
    for utype, uprop in unique_indexes:
        try:
            await client.command(
                database, "sql",
                f"CREATE INDEX IF NOT EXISTS ON {utype} ({uprop}) UNIQUE",
            )
            report.indexes_created += 1
        except Exception:
            pass

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
