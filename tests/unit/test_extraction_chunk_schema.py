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
    ("created_at", "TIMESTAMP"),
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
    """ExtractionChunk must have created_at TIMESTAMP — load-bearing for janitor orphan cleanup (rev 8 M5)."""
    from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES

    fields = dict(_STRUCTURAL_VERTEX_TYPES.get("ExtractionChunk", []))
    assert "created_at" in fields, (
        "ExtractionChunk is missing created_at field. "
        "This field is load-bearing for janitor age-sweep (rev 8 M5): "
        "janitor queries ArcadeDB for chunks WHERE created_at < NOW() - INTERVAL '24 hours'."
    )
    assert fields["created_at"] == "TIMESTAMP", (
        f"ExtractionChunk.created_at must be TIMESTAMP (for sysdate() default + interval arithmetic), "
        f"got {fields['created_at']!r}"
    )
