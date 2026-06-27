"""Integration tests for cleanup_extraction_index against MERGED-mode rows
(Phase 1 Task 7 of the merged-chunk routing plan).

Verifies the claim documented at Task 7 of
``docs/superpowers/plans/2026-05-27-merged-chunk-routing.md``:

    The existing janitor ``cleanup_extraction_index`` deletes
    ``WHERE pipeline_run_id = :run_id`` — run-id scoped, NOT vertex-id
    scoped.  So the new merged-mode vertex_id format
    (``f"{pipeline_run_id}:chunk_{chunk_index}"``) is automatically
    handled WITHOUT any code changes to the janitor.

The tests insert one or more rows in each format under a single
pipeline_run_id, plus a row under a DIFFERENT pipeline_run_id, then call
``cleanup_extraction_index`` against the first run_id and assert:

* every row matching that run_id is deleted (both formats)
* rows for the other run_id are untouched (scope correctness)
* per-element-mode behavior is byte-identical to the pre-Task-1 contract

SKIP CONDITION
--------------
Tests skip when ArcadeDB is not reachable — mirroring
``tests/integration/test_extraction_chunk_schema.py``.
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


def _count_rows(store: "ArcadeDBGraphStore", run_id: str) -> int:
    rows = store._client.query_sync(
        store._database,
        "sql",
        (
            "SELECT count(*) AS cnt FROM ExtractionChunk "
            "WHERE pipeline_run_id = :run_id"
        ),
        {"run_id": run_id},
    )
    return int(rows[0]["cnt"]) if rows else 0


def _insert_merged_row(
    store: "ArcadeDBGraphStore",
    *,
    run_id: str,
    document_id: str,
    chunk_index: int,
    source_refs: list[str],
    text: str,
    page_no: int = 1,
    token_count: int = 0,
) -> None:
    """Insert one MERGED-mode row mimicking the path taken by
    ``build_extraction_index_hybrid`` — vertex_id format
    ``f"{run_id}:chunk_{chunk_index}"``, modality='merged', populated
    chunk_index/source_refs/token_count.
    """
    store._client.command_sync(
        store._database,
        "sql",
        (
            "INSERT INTO ExtractionChunk SET "
            "vertex_id = :v, pipeline_run_id = :run_id, "
            "document_id = :doc_id, self_ref = :self_ref, "
            "chunk_text = :text, page_number = :page_no, modality = 'merged', "
            "chunk_index = :ci, source_refs = :refs, token_count = :tc"
        ),
        {
            "v": f"{run_id}:chunk_{chunk_index}",
            "run_id": run_id,
            "doc_id": document_id,
            "self_ref": "",
            "text": text,
            "page_no": page_no,
            "ci": chunk_index,
            "refs": source_refs,
            "tc": token_count,
        },
    )


def _insert_legacy_row(
    store: "ArcadeDBGraphStore",
    *,
    run_id: str,
    document_id: str,
    self_ref: str,
    text: str,
    page_no: int = 1,
) -> None:
    """Insert one LEGACY per-element row mimicking the path taken by
    ``build_extraction_index`` — vertex_id format
    ``f"{run_id}:{self_ref}"``, no chunk_index/source_refs/token_count.
    """
    store._client.command_sync(
        store._database,
        "sql",
        (
            "INSERT INTO ExtractionChunk SET "
            "vertex_id = :v, pipeline_run_id = :run_id, "
            "document_id = :doc_id, self_ref = :self_ref, "
            "chunk_text = :text, page_number = :page_no, modality = 'text'"
        ),
        {
            "v": f"{run_id}:{self_ref}",
            "run_id": run_id,
            "doc_id": document_id,
            "self_ref": self_ref,
            "text": text,
            "page_no": page_no,
        },
    )


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
# Cleanup against merged-mode rows
# ---------------------------------------------------------------------------


class TestCleanupRemovesMergedRows:
    """1a + 1c + 1d: merged-mode rows are deleted by run-id-scoped janitor."""

    def test_cleanup_deletes_merged_mode_rows(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_index import cleanup_extraction_index

        run_id = f"task7-merged-{uuid.uuid4().hex[:8]}"
        try:
            # Three merged-mode rows under this run_id.
            for idx, refs in enumerate(
                [
                    ["#/texts/10", "#/texts/11"],
                    ["#/texts/12"],
                    ["#/texts/13", "#/texts/14", "#/texts/15"],
                ]
            ):
                _insert_merged_row(
                    arcadedb_store,
                    run_id=run_id,
                    document_id="doc-task7-merged",
                    chunk_index=idx,
                    source_refs=refs,
                    text=f"merged chunk {idx}",
                    page_no=idx + 1,
                    token_count=100 + idx,
                )

            assert _count_rows(arcadedb_store, run_id) == 3

            deleted = cleanup_extraction_index(run_id, store=arcadedb_store)

            # All three merged-mode rows removed by run-id-scope.
            assert _count_rows(arcadedb_store, run_id) == 0, (
                "cleanup_extraction_index left merged-mode rows behind — "
                "run-id-scoped delete did not match the new vertex_id format. "
                "Task 7 BLOCKED: janitor needs vertex-id-format aware logic."
            )
            # ArcadeDB DELETE may or may not return the count reliably;
            # the authoritative check is the post-DELETE count above.
            assert deleted >= 0
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Cleanup against legacy per-element rows (regression / per-element parity)
# ---------------------------------------------------------------------------


class TestCleanupRemovesLegacyRows:
    """1b + 1c + 1d: legacy per-element rows are deleted exactly as before."""

    def test_cleanup_deletes_legacy_per_element_rows(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_index import cleanup_extraction_index

        run_id = f"task7-legacy-{uuid.uuid4().hex[:8]}"
        try:
            for self_ref in ("#/texts/0", "#/texts/1", "#/tables/0"):
                _insert_legacy_row(
                    arcadedb_store,
                    run_id=run_id,
                    document_id="doc-task7-legacy",
                    self_ref=self_ref,
                    text=f"legacy chunk for {self_ref}",
                )

            assert _count_rows(arcadedb_store, run_id) == 3

            cleanup_extraction_index(run_id, store=arcadedb_store)

            assert _count_rows(arcadedb_store, run_id) == 0, (
                "cleanup_extraction_index regression: per-element-mode behavior "
                "broke — legacy rows survived a run-id-scoped DELETE."
            )
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Mixed-format cleanup under a single run_id
# ---------------------------------------------------------------------------


class TestCleanupRemovesMixedFormats:
    """1a + 1b + 1c + 1d: both vertex_id formats coexist under one run_id
    (e.g. mid-rollout, or backfill artifact); both must be deleted in one
    run-id-scoped DELETE.
    """

    def test_cleanup_deletes_both_formats_under_one_run_id(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_index import cleanup_extraction_index

        run_id = f"task7-mixed-{uuid.uuid4().hex[:8]}"
        try:
            # 2 legacy rows
            for self_ref in ("#/texts/0", "#/texts/1"):
                _insert_legacy_row(
                    arcadedb_store,
                    run_id=run_id,
                    document_id="doc-task7-mixed",
                    self_ref=self_ref,
                    text=f"legacy {self_ref}",
                )

            # 2 merged-mode rows under the SAME run_id
            for idx, refs in enumerate(
                [["#/texts/100", "#/texts/101"], ["#/texts/102"]]
            ):
                _insert_merged_row(
                    arcadedb_store,
                    run_id=run_id,
                    document_id="doc-task7-mixed",
                    chunk_index=idx,
                    source_refs=refs,
                    text=f"merged {idx}",
                    page_no=idx + 1,
                    token_count=50 + idx,
                )

            assert _count_rows(arcadedb_store, run_id) == 4

            cleanup_extraction_index(run_id, store=arcadedb_store)

            assert _count_rows(arcadedb_store, run_id) == 0, (
                "cleanup_extraction_index did not delete all rows when both "
                "vertex_id formats coexisted under one pipeline_run_id."
            )
        finally:
            _delete_run(arcadedb_store, run_id)


# ---------------------------------------------------------------------------
# Scope correctness: other run_ids untouched
# ---------------------------------------------------------------------------


class TestCleanupScopedToRunId:
    """1e: cleanup must NOT touch rows under a different pipeline_run_id."""

    def test_cleanup_leaves_other_run_id_intact(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        from app.services.extraction_chunk_index import cleanup_extraction_index

        run_id_target = f"task7-scope-target-{uuid.uuid4().hex[:8]}"
        run_id_other = f"task7-scope-other-{uuid.uuid4().hex[:8]}"
        try:
            # Target run_id: one merged + one legacy
            _insert_merged_row(
                arcadedb_store,
                run_id=run_id_target,
                document_id="doc-task7-scope",
                chunk_index=0,
                source_refs=["#/texts/0"],
                text="target merged 0",
                token_count=10,
            )
            _insert_legacy_row(
                arcadedb_store,
                run_id=run_id_target,
                document_id="doc-task7-scope",
                self_ref="#/texts/0",
                text="target legacy",
            )

            # Other run_id: one merged + one legacy. These MUST survive.
            _insert_merged_row(
                arcadedb_store,
                run_id=run_id_other,
                document_id="doc-task7-scope",
                chunk_index=0,
                source_refs=["#/texts/0"],
                text="other merged 0",
                token_count=10,
            )
            _insert_legacy_row(
                arcadedb_store,
                run_id=run_id_other,
                document_id="doc-task7-scope",
                self_ref="#/texts/0",
                text="other legacy",
            )

            assert _count_rows(arcadedb_store, run_id_target) == 2
            assert _count_rows(arcadedb_store, run_id_other) == 2

            cleanup_extraction_index(run_id_target, store=arcadedb_store)

            assert _count_rows(arcadedb_store, run_id_target) == 0, (
                "Target run_id rows not cleaned up."
            )
            assert _count_rows(arcadedb_store, run_id_other) == 2, (
                "Cleanup leaked across pipeline_run_id boundary — "
                "other run_id rows were touched."
            )
        finally:
            _delete_run(arcadedb_store, run_id_target)
            _delete_run(arcadedb_store, run_id_other)
