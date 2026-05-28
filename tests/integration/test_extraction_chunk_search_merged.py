"""Integration tests for Phase 1 Task 5 — direct-cosine SELECT projection
extension for merged-mode columns.

Verifies the SELECT in ``search_extraction_chunks_direct`` projects the three
new ``ExtractionChunk`` columns (``chunk_index``, ``source_refs``,
``token_count``) added in Phase 1 Task 1, and that the values flow through
into ``GraphEntityResult.properties`` so Task 6's chunk-scope endpoint can
read them via the Task 1 accessors.

Tests cover:

* Step 1a — a MERGED row (chunk_index=3, source_refs=[…], token_count=42)
  round-trips through the search function with real values.
* Step 1b — a LEGACY per-element row with backfilled defaults (-1 / [] / 0)
  also surfaces those defaults rather than ``None``.
* Step 1c — a mixed result set keeps the pre-Task-5 contract intact AND adds
  the three new keys.
* Step 1d — the Task 1 accessors (``read_chunk_index`` /
  ``read_chunk_source_refs`` / ``read_chunk_token_count``) read the new
  keys transparently from the projected ``properties`` dict.
* Step 1e — a row whose ``chunk_index`` slipped through the backfill (still
  ``None``) still SELECTs cleanly; the accessor coalesces to ``-1``.

Mirrors ``tests/integration/test_extraction_chunk_schema.py`` fixture style
(unique pipeline_run_id per test for isolation; teardown via DELETE).

SKIP CONDITION
--------------
Tests skip when ArcadeDB is not reachable — inherited from the integration
conftest fixture.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

pytestmark = pytest.mark.integration


# bge-m3 dimensionality. Match the writer-side normalization assumption in
# search_extraction_chunks_direct so cosine scores are well-defined.
_EMB_DIM = 1024


def _vec_along(axis: int, dim: int = _EMB_DIM) -> list[float]:
    """Return a unit vector with 1.0 at ``axis``, zeros elsewhere."""
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def _delete_run(store: "ArcadeDBGraphStore", run_id: str) -> None:
    """Best-effort teardown — mirrors test_extraction_chunk_schema.py."""
    try:
        store._client.command_sync(
            store._database,
            "sql",
            "DELETE FROM ExtractionChunk WHERE pipeline_run_id = :run_id",
            {"run_id": run_id},
        )
    except Exception:
        pass


def _insert_merged_row(
    store: "ArcadeDBGraphStore",
    *,
    run_id: str,
    chunk_index: int,
    source_refs: list[str],
    token_count: int,
    chunk_text: str,
    embedding: list[float],
    page_number: int = 1,
    document_id: str = "doc-task5",
) -> None:
    """Insert one merged-mode row by direct SQL (mirrors Task 3's path).

    Uses the merged vertex_id format ``f"{run_id}:chunk_{chunk_index}"`` so
    we exercise the same INSERT shape ``build_extraction_index_hybrid``
    would write in production.
    """
    self_ref = source_refs[0] if source_refs else ""
    store._client.command_sync(
        store._database,
        "sql",
        (
            "INSERT INTO ExtractionChunk SET "
            "vertex_id = :v, pipeline_run_id = :run_id, "
            "document_id = :doc_id, self_ref = :self_ref, "
            "chunk_text = :text, embedding = :emb, page_number = :pn, "
            "modality = 'merged', chunk_index = :ci, "
            "source_refs = :refs, token_count = :tc"
        ),
        {
            "v": f"{run_id}:chunk_{chunk_index}",
            "run_id": run_id,
            "doc_id": document_id,
            "self_ref": self_ref,
            "text": chunk_text,
            "emb": embedding,
            "pn": page_number,
            "ci": chunk_index,
            "refs": list(source_refs),
            "tc": token_count,
        },
    )


def _insert_legacy_row(
    store: "ArcadeDBGraphStore",
    *,
    run_id: str,
    self_ref: str,
    chunk_text: str,
    embedding: list[float],
    page_number: int = 1,
    document_id: str = "doc-task5",
) -> None:
    """Insert one legacy per-element row (no values for the new columns).

    Caller is expected to run the backfill UPDATEs afterwards to coalesce
    NULL → (-1, [], 0) — mirroring Phase 1 Task 1's migration semantics.
    """
    store._client.command_sync(
        store._database,
        "sql",
        (
            "INSERT INTO ExtractionChunk SET "
            "vertex_id = :v, pipeline_run_id = :run_id, "
            "document_id = :doc_id, self_ref = :self_ref, "
            "chunk_text = :text, embedding = :emb, page_number = :pn, "
            "modality = 'text'"
        ),
        {
            "v": f"{run_id}:{self_ref}",
            "run_id": run_id,
            "doc_id": document_id,
            "self_ref": self_ref,
            "text": chunk_text,
            "emb": embedding,
            "pn": page_number,
        },
    )


def _backfill_run(store: "ArcadeDBGraphStore", run_id: str) -> None:
    """Run the same backfill UPDATEs the Task 1 migration uses, scoped to
    one run_id so we don't sweep unrelated test rows."""
    store._client.command_sync(
        store._database,
        "sql",
        (
            "UPDATE ExtractionChunk SET chunk_index = -1 "
            "WHERE pipeline_run_id = :run_id AND chunk_index IS NULL"
        ),
        {"run_id": run_id},
    )
    store._client.command_sync(
        store._database,
        "sql",
        (
            "UPDATE ExtractionChunk SET token_count = 0 "
            "WHERE pipeline_run_id = :run_id AND token_count IS NULL"
        ),
        {"run_id": run_id},
    )
    store._client.command_sync(
        store._database,
        "sql",
        (
            "UPDATE ExtractionChunk SET source_refs = [] "
            "WHERE pipeline_run_id = :run_id AND source_refs IS NULL"
        ),
        {"run_id": run_id},
    )


# ---------------------------------------------------------------------------
# Step 1a — Merged-mode row surfaces real chunk_index / source_refs / token_count
# ---------------------------------------------------------------------------


class TestMergedRowProjection:
    """A row inserted with non-default merged-mode values must surface those
    values in the GraphEntityResult.properties dict returned by
    ``search_extraction_chunks_direct``.
    """

    @pytest.mark.asyncio
    async def test_merged_row_projects_chunk_index_source_refs_token_count(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_direct,
        )

        run_id = f"task5-merged-{uuid.uuid4().hex[:8]}"
        try:
            emb = _vec_along(0)
            _insert_merged_row(
                arcadedb_store,
                run_id=run_id,
                chunk_index=3,
                source_refs=["#/texts/35", "#/texts/36"],
                token_count=42,
                chunk_text="merged chunk three text",
                embedding=emb,
            )

            results, diag = await search_extraction_chunks_direct(
                store=arcadedb_store,
                query_vector=emb,
                pipeline_run_id=run_id,
                desired_top_n=5,
            )

            assert diag.filter_strategy == "direct_cosine"
            assert len(results) == 1
            props = results[0].properties
            assert props["chunk_index"] == 3
            assert props["source_refs"] == ["#/texts/35", "#/texts/36"]
            assert props["token_count"] == 42
            # Pre-Task-5 contract preserved.
            assert props["chunk_text"] == "merged chunk three text"
            assert props["modality"] == "merged"
            assert props["pipeline_run_id"] == run_id
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1b — Legacy row surfaces backfilled defaults (-1, [], 0)
# ---------------------------------------------------------------------------


class TestLegacyRowProjection:
    """A pre-Task-1 per-element row (whose NULLs the backfill coalesced to
    (-1, [], 0)) must surface those defaults rather than ``None`` —
    Task 6's accessors will then return the same defaults.
    """

    @pytest.mark.asyncio
    async def test_legacy_row_projects_backfilled_defaults(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_direct,
        )

        run_id = f"task5-legacy-{uuid.uuid4().hex[:8]}"
        try:
            emb = _vec_along(0)
            _insert_legacy_row(
                arcadedb_store,
                run_id=run_id,
                self_ref="#/texts/0",
                chunk_text="legacy chunk zero",
                embedding=emb,
            )
            _backfill_run(arcadedb_store, run_id)

            results, _ = await search_extraction_chunks_direct(
                store=arcadedb_store,
                query_vector=emb,
                pipeline_run_id=run_id,
                desired_top_n=5,
            )

            assert len(results) == 1
            props = results[0].properties
            assert props["chunk_index"] == -1, (
                "Legacy row should surface the backfilled -1 default, not None"
            )
            assert props["source_refs"] == [], (
                "Legacy row should surface the backfilled [] default, not None"
            )
            assert props["token_count"] == 0, (
                "Legacy row should surface the backfilled 0 default, not None"
            )
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1c — Mixed result set: snapshot of pre-Task-5 contract + new fields
# ---------------------------------------------------------------------------


class TestMixedResultSet:
    """A mixed result set (merged + per-element rows for one run) must keep
    the pre-Task-5 contract intact (self_ref / chunk_text / page_number /
    modality / pipeline_run_id) AND add the three new fields.
    """

    @pytest.mark.asyncio
    async def test_mixed_run_projection_shape(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_direct,
        )

        run_id = f"task5-mixed-{uuid.uuid4().hex[:8]}"
        try:
            emb = _vec_along(0)
            _insert_merged_row(
                arcadedb_store,
                run_id=run_id,
                chunk_index=0,
                source_refs=["#/texts/0", "#/texts/1"],
                token_count=128,
                chunk_text="merged chunk zero",
                embedding=emb,
            )
            _insert_legacy_row(
                arcadedb_store,
                run_id=run_id,
                self_ref="#/texts/2",
                chunk_text="legacy chunk two",
                embedding=emb,
            )
            _backfill_run(arcadedb_store, run_id)

            results, _ = await search_extraction_chunks_direct(
                store=arcadedb_store,
                query_vector=emb,
                pipeline_run_id=run_id,
                desired_top_n=5,
            )

            assert len(results) == 2
            # Snapshot the union of keys that MUST be present on every row.
            required_keys = {
                "self_ref",
                "chunk_text",
                "page_number",
                "modality",
                "pipeline_run_id",
                # New under Task 5:
                "chunk_index",
                "source_refs",
                "token_count",
            }
            for r in results:
                missing = required_keys - r.properties.keys()
                assert not missing, (
                    f"GraphEntityResult.properties missing keys {missing} "
                    f"after Task 5 projection extension; full keys: "
                    f"{sorted(r.properties.keys())}"
                )

            # Per-row spot checks — group by self_ref to be index-order-agnostic.
            by_ref = {r.properties["self_ref"]: r.properties for r in results}
            merged = by_ref["#/texts/0"]
            legacy = by_ref["#/texts/2"]
            assert merged["chunk_index"] == 0
            assert merged["source_refs"] == ["#/texts/0", "#/texts/1"]
            assert merged["token_count"] == 128
            assert legacy["chunk_index"] == -1
            assert legacy["source_refs"] == []
            assert legacy["token_count"] == 0
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1d — Task 1 accessors read the new keys transparently
# ---------------------------------------------------------------------------


class TestAccessorsReadProjectedFields:
    """The Task 1 accessors (``read_chunk_index`` /
    ``read_chunk_source_refs`` / ``read_chunk_token_count``) MUST work
    against the projected ``properties`` dict for both merged and legacy
    rows. This is the contract Task 6's chunk-scope endpoint relies on.
    """

    @pytest.mark.asyncio
    async def test_accessors_work_on_projected_properties(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_index import (
            read_chunk_index,
            read_chunk_source_refs,
            read_chunk_token_count,
        )
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_direct,
        )

        run_id = f"task5-acc-{uuid.uuid4().hex[:8]}"
        try:
            emb = _vec_along(0)
            _insert_merged_row(
                arcadedb_store,
                run_id=run_id,
                chunk_index=7,
                source_refs=["#/texts/100"],
                token_count=64,
                chunk_text="merged chunk seven",
                embedding=emb,
            )
            _insert_legacy_row(
                arcadedb_store,
                run_id=run_id,
                self_ref="#/texts/200",
                chunk_text="legacy chunk two-hundred",
                embedding=emb,
            )
            _backfill_run(arcadedb_store, run_id)

            results, _ = await search_extraction_chunks_direct(
                store=arcadedb_store,
                query_vector=emb,
                pipeline_run_id=run_id,
                desired_top_n=5,
            )

            assert len(results) == 2
            by_ref = {r.properties["self_ref"]: r.properties for r in results}

            merged_props = by_ref["#/texts/100"]
            assert read_chunk_index(merged_props) == 7
            assert read_chunk_source_refs(merged_props) == ["#/texts/100"]
            assert read_chunk_token_count(merged_props) == 64

            legacy_props = by_ref["#/texts/200"]
            assert read_chunk_index(legacy_props) == -1
            assert read_chunk_source_refs(legacy_props) == []
            assert read_chunk_token_count(legacy_props) == 0
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Step 1e — NULL chunk_index (bypass scenario) still SELECTs + accessor coalesces
# ---------------------------------------------------------------------------


class TestNullChunkIndexCoalescing:
    """If a future row's ``chunk_index`` is NULL despite Task 1's backfill
    (e.g., a bug bypasses the migration), the SELECT must still succeed
    and ``read_chunk_index`` must coalesce to ``-1`` rather than raising.
    """

    @pytest.mark.asyncio
    async def test_null_chunk_index_row_does_not_break_search(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_index import read_chunk_index
        from app.services.extraction_chunk_search import (
            search_extraction_chunks_direct,
        )

        run_id = f"task5-null-{uuid.uuid4().hex[:8]}"
        try:
            emb = _vec_along(0)
            # Insert legacy row but DO NOT run the backfill — chunk_index
            # stays NULL on the row.
            _insert_legacy_row(
                arcadedb_store,
                run_id=run_id,
                self_ref="#/texts/0",
                chunk_text="row that bypassed backfill",
                embedding=emb,
            )

            results, _ = await search_extraction_chunks_direct(
                store=arcadedb_store,
                query_vector=emb,
                pipeline_run_id=run_id,
                desired_top_n=5,
            )

            assert len(results) == 1
            props = results[0].properties
            # The accessor must coalesce NULL → -1 without raising.
            assert read_chunk_index(props) == -1
        finally:
            _delete_run(arcadedb_store, run_id)
