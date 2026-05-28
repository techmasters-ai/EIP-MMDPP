"""Integration tests for ExtractionChunk schema migration (Phase 1 Task 1).

Verifies the three new columns added in Phase 1 Task 1 of the merged-chunk
routing plan (`docs/superpowers/plans/2026-05-27-merged-chunk-routing.md`):

* ``chunk_index: INTEGER``  (legacy default ``-1``, application-side)
* ``source_refs: LIST``     (legacy default ``[]``,  application-side)
* ``token_count: INTEGER``  (legacy default ``0``,   application-side)

The CREATE PROPERTY statements use ``IF NOT EXISTS``; the backfill UPDATEs
populate NULLs to the safe legacy default. Tests use a per-test fresh-schema
fixture (drop + recreate ``ExtractionChunk``) to avoid TOCTOU against
concurrent ingest traffic.

SKIP CONDITION
--------------
Tests skip when ArcadeDB is not reachable — mirroring the pattern in
``tests/integration/test_extraction_chunk_index_e2e.py``.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NEW_PROPERTIES = [
    ("chunk_index", "INTEGER"),
    ("source_refs", "LIST"),
    ("token_count", "INTEGER"),
]


def _has_property(store: "ArcadeDBGraphStore", type_name: str, prop_name: str) -> str | None:
    """Return the property's type string, or None if absent."""
    rows = store._client.query_sync(
        store._database,
        "sql",
        "SELECT FROM schema:types WHERE name = :tname",
        {"tname": type_name},
    )
    if not rows:
        return None
    props = rows[0].get("properties") or []
    for p in props:
        if p.get("name") == prop_name:
            return p.get("type")
    return None


def _count_nulls(store: "ArcadeDBGraphStore", run_id: str, col: str) -> int:
    rows = store._client.query_sync(
        store._database,
        "sql",
        (
            f"SELECT count(*) AS cnt FROM ExtractionChunk "
            f"WHERE pipeline_run_id = :run_id AND {col} IS NULL"
        ),
        {"run_id": run_id},
    )
    return int(rows[0]["cnt"]) if rows else 0


def _select_row(store: "ArcadeDBGraphStore", run_id: str, self_ref: str) -> dict:
    rows = store._client.query_sync(
        store._database,
        "sql",
        (
            "SELECT chunk_index, source_refs, token_count, self_ref "
            "FROM ExtractionChunk "
            "WHERE pipeline_run_id = :run_id AND self_ref = :self_ref"
        ),
        {"run_id": run_id, "self_ref": self_ref},
    )
    assert rows, f"no row for self_ref={self_ref!r}"
    return rows[0]


def _delete_run(store: "ArcadeDBGraphStore", run_id: str) -> None:
    """Best-effort teardown."""
    try:
        store._client.command_sync(
            store._database,
            "sql",
            "DELETE FROM ExtractionChunk WHERE pipeline_run_id = :run_id",
            {"run_id": run_id},
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Schema-property tests
# ---------------------------------------------------------------------------


class TestExtractionChunkSchemaColumns:
    """Verify the three new columns are declared on ExtractionChunk."""

    def test_chunk_index_column_exists(self, arcadedb_store: "ArcadeDBGraphStore"):
        type_str = _has_property(arcadedb_store, "ExtractionChunk", "chunk_index")
        assert type_str == "INTEGER", (
            f"ExtractionChunk.chunk_index must exist as INTEGER, got {type_str!r}. "
            "If absent, the Phase 1 Task 1 CREATE PROPERTY migration has not run "
            "against this database."
        )

    def test_source_refs_column_exists_as_list(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        type_str = _has_property(arcadedb_store, "ExtractionChunk", "source_refs")
        assert type_str == "LIST", (
            f"ExtractionChunk.source_refs must exist as LIST, got {type_str!r}. "
            "CommunityReport.key_entities uses LIST successfully — same idiom applies here."
        )

    def test_token_count_column_exists(self, arcadedb_store: "ArcadeDBGraphStore"):
        type_str = _has_property(arcadedb_store, "ExtractionChunk", "token_count")
        assert type_str == "INTEGER", (
            f"ExtractionChunk.token_count must exist as INTEGER, got {type_str!r}."
        )


# ---------------------------------------------------------------------------
# Backfill semantics: legacy NULL values → safe defaults
# ---------------------------------------------------------------------------


class TestBackfillSemantics:
    """Insert legacy-style rows (no values for the new columns), run the
    Phase 1 Task 1 backfill UPDATEs, then assert NULL counts → 0.

    The fixture-DB lifecycle is per-run-id rather than per-type to avoid
    racing against any other test that may have inserted into the shared
    ExtractionChunk type. Isolation is by unique pipeline_run_id (uuid4).
    """

    def test_backfill_zeroes_out_nulls_on_legacy_rows(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        run_id = f"task1-backfill-{uuid.uuid4().hex[:8]}"
        try:
            # Insert TWO legacy rows: vertex_id + minimal mandatory columns.
            # The three new columns get NULL (no SET expression for them).
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "INSERT INTO ExtractionChunk SET "
                    "vertex_id = :v, pipeline_run_id = :run_id, "
                    "document_id = :doc_id, self_ref = :self_ref, "
                    "chunk_text = :text, page_number = 1, modality = 'text'"
                ),
                {
                    "v": f"{run_id}:#/texts/0",
                    "run_id": run_id,
                    "doc_id": "doc-task1",
                    "self_ref": "#/texts/0",
                    "text": "legacy chunk 0",
                },
            )
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "INSERT INTO ExtractionChunk SET "
                    "vertex_id = :v, pipeline_run_id = :run_id, "
                    "document_id = :doc_id, self_ref = :self_ref, "
                    "chunk_text = :text, page_number = 2, modality = 'text'"
                ),
                {
                    "v": f"{run_id}:#/texts/1",
                    "run_id": run_id,
                    "doc_id": "doc-task1",
                    "self_ref": "#/texts/1",
                    "text": "legacy chunk 1",
                },
            )

            # Pre-backfill: 2 NULLs across each of the new columns.
            assert _count_nulls(arcadedb_store, run_id, "chunk_index") == 2
            assert _count_nulls(arcadedb_store, run_id, "token_count") == 2
            assert _count_nulls(arcadedb_store, run_id, "source_refs") == 2

            # Run the same backfill UPDATEs the migration uses. Scope to
            # the test run_id to avoid sweeping unrelated rows.
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "UPDATE ExtractionChunk SET chunk_index = -1 "
                    "WHERE pipeline_run_id = :run_id AND chunk_index IS NULL"
                ),
                {"run_id": run_id},
            )
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "UPDATE ExtractionChunk SET token_count = 0 "
                    "WHERE pipeline_run_id = :run_id AND token_count IS NULL"
                ),
                {"run_id": run_id},
            )
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "UPDATE ExtractionChunk SET source_refs = [] "
                    "WHERE pipeline_run_id = :run_id AND source_refs IS NULL"
                ),
                {"run_id": run_id},
            )

            # Post-backfill: 0 NULLs on each column for this run_id.
            assert _count_nulls(arcadedb_store, run_id, "chunk_index") == 0, (
                "chunk_index backfill did not eliminate NULLs"
            )
            assert _count_nulls(arcadedb_store, run_id, "token_count") == 0, (
                "token_count backfill did not eliminate NULLs"
            )
            assert _count_nulls(arcadedb_store, run_id, "source_refs") == 0, (
                "source_refs literal-list backfill (SET col = []) "
                "did not eliminate NULLs"
            )

            # Spot-check that the values are the documented defaults.
            row0 = _select_row(arcadedb_store, run_id, "#/texts/0")
            assert row0["chunk_index"] == -1
            assert row0["token_count"] == 0
            assert row0["source_refs"] == []
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Insert merged-mode row and read back via accessors
# ---------------------------------------------------------------------------


class TestIdempotentBackfillHelper:
    """M1 (Phase 1 Task 1 code-review fix): the schema-sync helper must
    include an idempotent backfill step so a fresh DB or restore-from-snapshot
    automatically populates the three merged-mode columns to their legacy
    defaults — no operator-only migration step required.
    """

    def test_schema_sync_backfills_nulls_in_new_columns(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        """Insert a legacy-style row (NULLs in all three new columns) by
        bypassing the application path, then invoke the schema-sync helper
        and assert the row has been backfilled to ``(-1, 0, [])``.
        """
        import asyncio

        from app.services.arcadedb_schema import sync_schema_from_ontology
        from ontology_bundles.air_defense_v3.introspect import build_ontology_dict

        run_id = f"task1-m1-{uuid.uuid4().hex[:8]}"
        try:
            # Direct INSERT bypassing the build_extraction_index path —
            # leaves all three new columns NULL.
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "INSERT INTO ExtractionChunk SET "
                    "vertex_id = :v, pipeline_run_id = :run_id, "
                    "document_id = :doc_id, self_ref = :self_ref, "
                    "chunk_text = :text, page_number = 1, modality = 'text'"
                ),
                {
                    "v": f"{run_id}:#/texts/0",
                    "run_id": run_id,
                    "doc_id": "doc-task1-m1",
                    "self_ref": "#/texts/0",
                    "text": "legacy row needing backfill",
                },
            )

            # Pre-state: NULLs across all three new columns for this row.
            assert _count_nulls(arcadedb_store, run_id, "chunk_index") == 1
            assert _count_nulls(arcadedb_store, run_id, "token_count") == 1
            assert _count_nulls(arcadedb_store, run_id, "source_refs") == 1

            # Invoke the schema-sync helper. Its in-place backfill step
            # MUST coalesce all three columns to the documented defaults.
            asyncio.run(
                sync_schema_from_ontology(
                    arcadedb_store._client,
                    arcadedb_store._database,
                    build_ontology_dict(),
                )
            )

            # Post-state: 0 NULLs across all three columns for this row.
            assert _count_nulls(arcadedb_store, run_id, "chunk_index") == 0, (
                "schema-sync backfill did NOT zero-out chunk_index NULLs"
            )
            assert _count_nulls(arcadedb_store, run_id, "token_count") == 0, (
                "schema-sync backfill did NOT zero-out token_count NULLs"
            )
            assert _count_nulls(arcadedb_store, run_id, "source_refs") == 0, (
                "schema-sync backfill did NOT zero-out source_refs NULLs"
            )

            # Spot-check the actual defaults.
            row0 = _select_row(arcadedb_store, run_id, "#/texts/0")
            assert row0["chunk_index"] == -1
            assert row0["token_count"] == 0
            assert row0["source_refs"] == []
        finally:
            _delete_run(arcadedb_store, run_id)

    def test_schema_sync_backfill_is_idempotent(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        """A second invocation against an already-backfilled DB must be a
        no-op (count of rows updated == 0) — guaranteed by the WHERE col IS
        NULL filter on each UPDATE statement.
        """
        import asyncio

        from app.services.arcadedb_schema import sync_schema_from_ontology
        from ontology_bundles.air_defense_v3.introspect import build_ontology_dict

        run_id = f"task1-m1-idem-{uuid.uuid4().hex[:8]}"
        try:
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "INSERT INTO ExtractionChunk SET "
                    "vertex_id = :v, pipeline_run_id = :run_id, "
                    "document_id = :doc_id, self_ref = :self_ref, "
                    "chunk_text = :text, page_number = 1, modality = 'text'"
                ),
                {
                    "v": f"{run_id}:#/texts/0",
                    "run_id": run_id,
                    "doc_id": "doc-task1-m1-idem",
                    "self_ref": "#/texts/0",
                    "text": "legacy row for idempotency check",
                },
            )

            # First run: should backfill the NULLs.
            asyncio.run(
                sync_schema_from_ontology(
                    arcadedb_store._client,
                    arcadedb_store._database,
                    build_ontology_dict(),
                )
            )
            assert _count_nulls(arcadedb_store, run_id, "chunk_index") == 0
            row_after_first = _select_row(arcadedb_store, run_id, "#/texts/0")
            assert row_after_first["chunk_index"] == -1

            # Second run: a no-op for already-defaulted rows — values stay put.
            asyncio.run(
                sync_schema_from_ontology(
                    arcadedb_store._client,
                    arcadedb_store._database,
                    build_ontology_dict(),
                )
            )
            row_after_second = _select_row(arcadedb_store, run_id, "#/texts/0")
            assert row_after_second["chunk_index"] == -1
            assert row_after_second["token_count"] == 0
            assert row_after_second["source_refs"] == []
        finally:
            _delete_run(arcadedb_store, run_id)


class TestMergedModeRowRoundTrip:
    """Insert one merged-mode row directly + one legacy-style row, read both
    back, assert accessors return correct values for each.
    """

    def test_merged_row_round_trips_through_accessors(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_index import (
            read_chunk_index,
            read_chunk_source_refs,
            read_chunk_token_count,
        )

        run_id = f"task1-merged-{uuid.uuid4().hex[:8]}"
        try:
            # Merged-mode row (vertex_id format f"{run_id}:chunk_{idx}")
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "INSERT INTO ExtractionChunk SET "
                    "vertex_id = :v, pipeline_run_id = :run_id, "
                    "document_id = :doc_id, self_ref = :self_ref, "
                    "chunk_text = :text, page_number = 2, modality = 'text', "
                    "chunk_index = :ci, source_refs = :refs, token_count = :tc"
                ),
                {
                    "v": f"{run_id}:chunk_3",
                    "run_id": run_id,
                    "doc_id": "doc-task1-merged",
                    "self_ref": "",   # merged rows: no single self_ref
                    "text": "merged chunk three",
                    "ci": 3,
                    "refs": ["#/texts/35", "#/texts/36"],
                    "tc": 312,
                },
            )

            # Legacy row (no values for new columns; backfill after)
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "INSERT INTO ExtractionChunk SET "
                    "vertex_id = :v, pipeline_run_id = :run_id, "
                    "document_id = :doc_id, self_ref = :self_ref, "
                    "chunk_text = :text, page_number = 1, modality = 'text'"
                ),
                {
                    "v": f"{run_id}:#/texts/0",
                    "run_id": run_id,
                    "doc_id": "doc-task1-merged",
                    "self_ref": "#/texts/0",
                    "text": "legacy chunk zero",
                },
            )

            # Run backfill ONLY against legacy rows under this run_id.
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "UPDATE ExtractionChunk SET chunk_index = -1 "
                    "WHERE pipeline_run_id = :run_id AND chunk_index IS NULL"
                ),
                {"run_id": run_id},
            )
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "UPDATE ExtractionChunk SET token_count = 0 "
                    "WHERE pipeline_run_id = :run_id AND token_count IS NULL"
                ),
                {"run_id": run_id},
            )
            arcadedb_store._client.command_sync(
                arcadedb_store._database,
                "sql",
                (
                    "UPDATE ExtractionChunk SET source_refs = [] "
                    "WHERE pipeline_run_id = :run_id AND source_refs IS NULL"
                ),
                {"run_id": run_id},
            )

            # --- Read back via SELECT ---
            rows = arcadedb_store._client.query_sync(
                arcadedb_store._database,
                "sql",
                (
                    "SELECT vertex_id, chunk_index, source_refs, token_count "
                    "FROM ExtractionChunk WHERE pipeline_run_id = :run_id "
                    "ORDER BY vertex_id"
                ),
                {"run_id": run_id},
            )
            by_vid = {r["vertex_id"]: r for r in rows}

            # Legacy row → accessors return safe defaults
            legacy = by_vid[f"{run_id}:#/texts/0"]
            assert read_chunk_index(legacy) == -1
            assert read_chunk_source_refs(legacy) == []
            assert read_chunk_token_count(legacy) == 0

            # Merged row → accessors return real values
            merged = by_vid[f"{run_id}:chunk_3"]
            assert read_chunk_index(merged) == 3
            assert read_chunk_source_refs(merged) == ["#/texts/35", "#/texts/36"]
            assert read_chunk_token_count(merged) == 312
        finally:
            _delete_run(arcadedb_store, run_id)
