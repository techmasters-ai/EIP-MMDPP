"""Unit tests for ExtractionChunk vertex type registration in arcadedb_schema.py.

VR Phase C.1 — rev 10 locked schema decisions.

These tests are unit (mocked) — no ArcadeDB server required.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# Expected fields IN ORDER. Originally rev 10 locked the first 9; Phase 1
# Task 1 of the merged-chunk routing plan appended three additional columns
# (chunk_index / source_refs / token_count) with application-side defaults
# applied via read_chunk_* accessors in extraction_chunk_index.py.
_EXPECTED_FIELDS = [
    ("vertex_id", "STRING"),
    ("pipeline_run_id", "STRING"),
    ("document_id", "STRING"),
    ("self_ref", "STRING"),
    ("chunk_text", "STRING"),
    ("embedding", "ARRAY_OF_FLOATS"),
    ("page_number", "INTEGER"),
    ("modality", "STRING"),
    ("created_at", "DATETIME"),
    ("chunk_index", "INTEGER"),
    ("source_refs", "LIST"),
    ("token_count", "INTEGER"),
]


def test_extraction_chunk_vertex_type_registered():
    """ExtractionChunk must be present in _STRUCTURAL_VERTEX_TYPES with all expected fields in order."""
    from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES

    assert "ExtractionChunk" in _STRUCTURAL_VERTEX_TYPES, (
        "ExtractionChunk not found in _STRUCTURAL_VERTEX_TYPES — "
        "add it per VR C.1 rev 10 schema spec"
    )

    fields = _STRUCTURAL_VERTEX_TYPES["ExtractionChunk"]
    assert fields == _EXPECTED_FIELDS, (
        f"ExtractionChunk field list mismatch.\n"
        f"Expected: {_EXPECTED_FIELDS}\n"
        f"Got:      {fields}"
    )


def test_extraction_chunk_has_merged_routing_columns():
    """Phase 1 Task 1: chunk_index / source_refs / token_count must be declared.

    These columns are required by the merged-chunk routing plan
    (`docs/superpowers/plans/2026-05-27-merged-chunk-routing.md`). Legacy
    per-element rows carry safe defaults applied application-side via the
    ``read_chunk_*`` accessors in extraction_chunk_index.py — NO ArcadeDB
    DEFAULT clause is used.
    """
    from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES

    fields = dict(_STRUCTURAL_VERTEX_TYPES.get("ExtractionChunk", []))
    assert fields.get("chunk_index") == "INTEGER", (
        f"ExtractionChunk.chunk_index must be INTEGER, got {fields.get('chunk_index')!r}"
    )
    assert fields.get("source_refs") == "LIST", (
        f"ExtractionChunk.source_refs must be LIST (matching CommunityReport.key_entities), "
        f"got {fields.get('source_refs')!r}"
    )
    assert fields.get("token_count") == "INTEGER", (
        f"ExtractionChunk.token_count must be INTEGER, got {fields.get('token_count')!r}"
    )


def test_extraction_chunk_embedding_dim_matches_textchunk():
    """ExtractionChunk.embedding must use ARRAY_OF_FLOATS (dim=1024 verified via index declaration).

    bge-m3 produces 1024-dim embeddings, matching TextChunk.text_embedding in Phase 5 vector
    indexes. The type ARRAY_OF_FLOATS is the required ArcadeDB type for HNSW LSM_VECTOR index.
    """
    from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES

    fields = dict(_STRUCTURAL_VERTEX_TYPES.get("ExtractionChunk", []))
    assert fields.get("embedding") == "ARRAY_OF_FLOATS", (
        "ExtractionChunk.embedding must be ARRAY_OF_FLOATS to support HNSW LSM_VECTOR index "
        "at dim=1024 (matching bge-m3 output and TextChunk.text_embedding dim)"
    )


def test_extraction_chunk_has_created_at():
    """ExtractionChunk must have created_at DATETIME — load-bearing for janitor orphan cleanup (rev 8 M5)."""
    from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES

    fields = dict(_STRUCTURAL_VERTEX_TYPES.get("ExtractionChunk", []))
    assert "created_at" in fields, (
        "ExtractionChunk is missing created_at field. "
        "This field is load-bearing for janitor age-sweep (rev 8 M5): "
        "janitor queries ArcadeDB for chunks WHERE created_at < NOW() - INTERVAL '24 hours'."
    )
    # Must be DATETIME, NOT TIMESTAMP: ArcadeDB's com.arcadedb.schema.Type enum has
    # no TIMESTAMP member, so a TIMESTAMP declaration fails the CREATE PROPERTY DDL
    # at runtime ("No enum constant com.arcadedb.schema.Type.TIMESTAMP") and the
    # dependent created_at index never gets built. sysdate() returns a DATETIME, and
    # every other timestamp column in this schema (entity/edge created_at) uses DATETIME.
    assert fields["created_at"] == "DATETIME", (
        f"ExtractionChunk.created_at must be DATETIME (sysdate() return type; matches "
        f"_COMMON_ENTITY_PROPS/_COMMON_EDGE_PROPS), got {fields['created_at']!r}"
    )


def test_all_structural_vertex_types_use_valid_arcadedb_types():
    """Every declared property type must be a real ArcadeDB ``Type`` enum member.

    Regression guard for the TIMESTAMP bug: a unit test that only checks the Python
    dict can lock in a type string that ArcadeDB rejects at runtime. ArcadeDB passes
    the type string straight into ``CREATE PROPERTY ... <TYPE>``, so any value outside
    the enum fails the live schema sync (and silently — sync swallows the error and
    continues). This test validates the dict against the actual enum set so an invalid
    type fails CI instead of only at container boot.
    """
    from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES

    # com.arcadedb.schema.Type enum members (scalars + the array specializations used here).
    valid_arcadedb_types = {
        "BOOLEAN", "INTEGER", "SHORT", "LONG", "FLOAT", "DOUBLE", "DECIMAL",
        "STRING", "DATE", "DATETIME", "BINARY", "BYTE", "LIST", "MAP",
        "EMBEDDED", "LINK",
        "ARRAY_OF_SHORTS", "ARRAY_OF_INTEGERS", "ARRAY_OF_LONGS",
        "ARRAY_OF_FLOATS", "ARRAY_OF_DOUBLES",
    }

    offenders: list[str] = []
    for vtype, props in _STRUCTURAL_VERTEX_TYPES.items():
        for prop_name, prop_type in props:
            if prop_type not in valid_arcadedb_types:
                offenders.append(f"{vtype}.{prop_name} = {prop_type!r}")

    assert not offenders, (
        "Invalid ArcadeDB property type(s) declared (would fail CREATE PROPERTY at "
        "schema-sync time):\n  " + "\n  ".join(offenders)
    )
