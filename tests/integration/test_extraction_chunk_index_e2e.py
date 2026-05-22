"""Integration tests: build_extraction_index + cleanup_extraction_index end-to-end.

VR Phase C.2 — full cycle against a live ArcadeDB instance.

SKIP CONDITION
--------------
All tests skip when ArcadeDB is not reachable (mirrors the pattern from
``tests/integration/test_extraction_chunk_filter_starvation.py``).

Tests use the shared ``arcadedb_store`` fixture from
``tests/integration/conftest.py``.

TDD discipline: tests were written BEFORE the implementation module.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Embedding stub — returns deterministic 1024-d vectors without Ollama
# ---------------------------------------------------------------------------

_DIM = 1024


def _fake_embed(texts: list[str], query: bool = False) -> list[list[float]]:
    """Deterministic unit-norm embeddings for testing (no Ollama required)."""
    import hashlib
    import math

    result = []
    for text in texts:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Build a pseudo-random unit vector deterministically from the seed
        vec = []
        for i in range(_DIM):
            # Simple LCG hash for reproducibility
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vec.append((seed / 0xFFFFFFFF) * 2 - 1)
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude == 0:
            magnitude = 1.0
        result.append([v / magnitude for v in vec])
    return result


# ---------------------------------------------------------------------------
# Synthetic docling doc_json fixture (3 elements, no Ollama embedding needed)
# ---------------------------------------------------------------------------

def _make_test_doc_json(run_suffix: str) -> dict:
    """3-element synthetic doc: 2 text + 1 table-with-caption."""
    return {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": f"Radar tracking parameters for run {run_suffix}.",
                "label": "text",
                "prov": [{"page_no": 1}],
            },
            {
                "self_ref": "#/texts/1",
                "text": f"Maximum engagement altitude 30 km for run {run_suffix}.",
                "label": "text",
                "prov": [{"page_no": 2}],
            },
            {
                "self_ref": "#/texts/99",
                "text": f"Table 1 — SA-2 Guidance for {run_suffix}",
                "label": "title",   # excluded from text indexing; used as caption
                "prov": [],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "captions": [{"$ref": "#/texts/99"}],
                "prov": [{"page_no": 3}],
                "data": {
                    "table_cells": [
                        {"text": "Parameter", "start_row_offset_idx": 0,
                         "end_row_offset_idx": 0, "start_col_offset_idx": 0,
                         "end_col_offset_idx": 0},
                        {"text": "Value", "start_row_offset_idx": 0,
                         "end_row_offset_idx": 0, "start_col_offset_idx": 1,
                         "end_col_offset_idx": 1},
                        {"text": "PRF", "start_row_offset_idx": 1,
                         "end_row_offset_idx": 1, "start_col_offset_idx": 0,
                         "end_col_offset_idx": 0},
                        {"text": "200 Hz", "start_row_offset_idx": 1,
                         "end_row_offset_idx": 1, "start_col_offset_idx": 1,
                         "end_col_offset_idx": 1},
                    ]
                },
            }
        ],
        "pictures": [],
    }


# ---------------------------------------------------------------------------
# SQL count helper
# ---------------------------------------------------------------------------

def _count_chunks_for_run(store: "ArcadeDBGraphStore", run_id: str) -> int:
    """Query ArcadeDB for the number of ExtractionChunk rows for a run_id."""
    rows = store._client.query_sync(
        store._database,
        "sql",
        "SELECT count(*) AS cnt FROM ExtractionChunk WHERE pipeline_run_id = :run_id",
        {"run_id": run_id},
    )
    if rows:
        return int(rows[0].get("cnt", 0))
    return 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildThenCleanupFullCycle:
    """test_build_then_cleanup_full_cycle"""

    def test_build_inserts_chunks_cleanup_deletes_them(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        """Build inserts 3 chunks; cleanup deletes them; SQL count verifies both states."""
        from unittest.mock import patch

        from app.services.extraction_chunk_index import (
            build_extraction_index,
            cleanup_extraction_index,
        )

        run_suffix = uuid.uuid4().hex[:8]
        run_id = f"e2e-cycle-{run_suffix}"
        doc_id = f"doc-e2e-{run_suffix}"
        doc_json = _make_test_doc_json(run_suffix)

        try:
            with patch(
                "app.services.extraction_chunk_index.embed_texts",
                side_effect=_fake_embed,
            ):
                diag = build_extraction_index(
                    doc_json, run_id, doc_id, store=arcadedb_store
                )

            # 2 text + 1 table = 3 chunks (0 skipped)
            assert diag.chunks_inserted == 3, (
                f"Expected 3 inserts, got {diag.chunks_inserted}. "
                f"Diagnostics: {diag}"
            )
            assert diag.chunks_skipped == 0

            # Verify via SQL count
            count_after_build = _count_chunks_for_run(arcadedb_store, run_id)
            assert count_after_build == 3, (
                f"ArcadeDB should have 3 ExtractionChunk rows for run_id={run_id!r}, "
                f"found {count_after_build}"
            )

            # --- Cleanup ---
            deleted = cleanup_extraction_index(run_id, store=arcadedb_store)
            assert deleted == 3, (
                f"cleanup_extraction_index should return 3, got {deleted}"
            )

            # Verify via SQL count
            count_after_cleanup = _count_chunks_for_run(arcadedb_store, run_id)
            assert count_after_cleanup == 0, (
                f"ArcadeDB should have 0 ExtractionChunk rows after cleanup, "
                f"found {count_after_cleanup}"
            )

        finally:
            # Best-effort cleanup in case of test failure
            try:
                cleanup_extraction_index(run_id, store=arcadedb_store)
            except Exception:
                pass


class TestConcurrentBuildIdempotency:
    """test_concurrent_build_idempotency"""

    def test_second_build_same_run_id_does_not_double_rows(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        """After two sequential builds for the same run_id, row count == element count (not doubled)."""
        from unittest.mock import patch

        from app.services.extraction_chunk_index import (
            build_extraction_index,
            cleanup_extraction_index,
        )

        run_suffix = uuid.uuid4().hex[:8]
        run_id = f"e2e-idem-{run_suffix}"
        doc_id = f"doc-idem-{run_suffix}"
        doc_json = _make_test_doc_json(run_suffix)

        try:
            with patch(
                "app.services.extraction_chunk_index.embed_texts",
                side_effect=_fake_embed,
            ):
                diag1 = build_extraction_index(
                    doc_json, run_id, doc_id, store=arcadedb_store
                )
                # Second build: should DELETE then re-INSERT, not append
                diag2 = build_extraction_index(
                    doc_json, run_id, doc_id, store=arcadedb_store
                )

            assert diag1.chunks_inserted == 3
            assert diag2.chunks_inserted == 3

            # Row count must equal element count, not 6
            count = _count_chunks_for_run(arcadedb_store, run_id)
            assert count == 3, (
                f"After two builds for the same run_id, expected 3 rows (not doubled). "
                f"Got {count}. The idempotent DELETE-before-INSERT must be working."
            )

        finally:
            try:
                cleanup_extraction_index(run_id, store=arcadedb_store)
            except Exception:
                pass


class TestCleanupOnlyDeletesMatchingRunId:
    """test_cleanup_only_deletes_matching_run_id"""

    def test_cleanup_one_run_leaves_other_intact(
        self, arcadedb_store: "ArcadeDBGraphStore"
    ):
        """Insert chunks for two run_ids; cleanup one; the other survives."""
        from unittest.mock import patch

        from app.services.extraction_chunk_index import (
            build_extraction_index,
            cleanup_extraction_index,
        )

        run_suffix = uuid.uuid4().hex[:8]
        run_a = f"e2e-scope-a-{run_suffix}"
        run_b = f"e2e-scope-b-{run_suffix}"
        doc_id = f"doc-scope-{run_suffix}"
        doc_json = _make_test_doc_json(run_suffix)

        try:
            with patch(
                "app.services.extraction_chunk_index.embed_texts",
                side_effect=_fake_embed,
            ):
                build_extraction_index(
                    doc_json, run_a, doc_id, store=arcadedb_store
                )
                build_extraction_index(
                    doc_json, run_b, doc_id, store=arcadedb_store
                )

            count_a_before = _count_chunks_for_run(arcadedb_store, run_a)
            count_b_before = _count_chunks_for_run(arcadedb_store, run_b)
            assert count_a_before == 3
            assert count_b_before == 3

            # Cleanup run_a only
            deleted = cleanup_extraction_index(run_a, store=arcadedb_store)
            assert deleted == 3

            # run_a is gone; run_b survives
            count_a_after = _count_chunks_for_run(arcadedb_store, run_a)
            count_b_after = _count_chunks_for_run(arcadedb_store, run_b)

            assert count_a_after == 0, (
                f"run_a should have 0 chunks after cleanup, got {count_a_after}"
            )
            assert count_b_after == 3, (
                f"run_b should still have 3 chunks (unaffected by run_a cleanup), "
                f"got {count_b_after}"
            )

        finally:
            # Best-effort cleanup of both
            try:
                cleanup_extraction_index(run_a, store=arcadedb_store)
            except Exception:
                pass
            try:
                cleanup_extraction_index(run_b, store=arcadedb_store)
            except Exception:
                pass
