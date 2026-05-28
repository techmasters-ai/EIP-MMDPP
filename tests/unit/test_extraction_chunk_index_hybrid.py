"""Unit tests for ``build_extraction_index_hybrid`` (merged-chunk routing).

Tests per plan ``docs/superpowers/plans/2026-05-27-merged-chunk-routing.md``
Task 3 Steps 1a-1g:

* 1a chunk count matches the shared ``build_hybrid_chunks_for_extraction``
  helper output
* 1b every inserted row's ``source_refs`` is non-empty and references real
  ``texts[]/tables[]/pictures[]`` indices
* 1c each chunk text contains the parent heading prefix (HybridChunker's
  ``contextualize`` prepends title + heading)
* 1d deterministic ``chunk_index`` across two runs on the same fixture
* 1e empty doc → ``BuildIndexDiagnostics(chunks_inserted=0,
  chunks_inserted_zero_reason="empty_chunker_output")``
* 1f malformed doc_json → ``ValueError`` propagates (NOT silently swallowed)
* 1g ``embed_texts`` partial response → ``RuntimeError``

The fixtures piggy-back on ``tests/unit/test_hybrid_chunking.py`` — the
Dvina-shaped synthetic doc + the empty-doc fixture — so the two test
suites share a single source of truth for the synthetic input shape.

TDD discipline: failing tests landed BEFORE the implementation.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixture imports — share with test_hybrid_chunking
# ---------------------------------------------------------------------------

from tests.unit.test_hybrid_chunking import (  # noqa: E402
    _build_dvina_like_doc_json,
    _build_empty_doc_json,
    _heading_terms_in,
)


# ---------------------------------------------------------------------------
# Mock store factory — captures INSERT SQL + params without talking to ArcadeDB
# ---------------------------------------------------------------------------


def _make_mock_store() -> MagicMock:
    """Return a mock ``ArcadeDBGraphStore``-like object.

    ``store._client.command_sync`` is captured so tests can introspect every
    DELETE / INSERT call's SQL and params. Return value mimics ArcadeDB's
    DELETE-with-count response shape.
    """
    store = MagicMock()
    store._database = "test_db"
    store._client.command_sync.return_value = [{"count": 0}]
    return store


def _captured_insert_params(store: MagicMock) -> list[dict]:
    """Extract the params dict from every INSERT call recorded on the store.

    Filters out the DELETE call by SQL substring match. Returns the params
    in call order, so tests can assert on per-row column values.
    """
    rows: list[dict] = []
    for c in store._client.command_sync.call_args_list:
        args = c[0]
        if len(args) < 3:
            continue
        sql = args[2]
        if not isinstance(sql, str) or "INSERT" not in sql.upper():
            continue
        # params is either positional [3] or kwarg
        params = args[3] if len(args) > 3 else c[1].get("params", {})
        if isinstance(params, dict):
            rows.append(params)
    return rows


# ---------------------------------------------------------------------------
# Step 1a — chunk count matches shared helper output
# ---------------------------------------------------------------------------


def test_step1a_chunk_count_matches_helper_output() -> None:
    """``chunks_inserted`` equals ``len(build_hybrid_chunks_for_extraction(doc))``.

    The indexer must NOT drop chunks emitted by the shared helper — any
    discrepancy means the indexer is silently filtering, which would
    invalidate the byte-identity invariant with docling-graph's per-pass
    chunker (Phase 2 dependency).
    """
    from app.services.extraction_chunk_index import build_extraction_index_hybrid
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    doc_json = _build_dvina_like_doc_json()
    expected_chunks = build_hybrid_chunks_for_extraction(doc_json)
    assert expected_chunks, "fixture must produce >=1 chunk"

    store = _make_mock_store()
    fake_embeddings = [[0.1] * 1024] * len(expected_chunks)

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=fake_embeddings,
    ):
        diag = build_extraction_index_hybrid(
            doc_json,
            pipeline_run_id="run-1a",
            document_id="doc-1a",
            store=store,
        )

    assert diag.chunks_inserted == len(expected_chunks)
    # And the diagnostics' zero-reason is unset (not the empty path)
    assert diag.chunks_inserted_zero_reason is None
    # mean_token_count is a positive int (every chunk had positive token_count)
    assert isinstance(diag.mean_token_count, int)
    assert diag.mean_token_count > 0


# ---------------------------------------------------------------------------
# Step 1b — source_refs populated + reference valid indices
# ---------------------------------------------------------------------------


_REF_RE = re.compile(r"^#/(texts|tables|pictures)/(\d+)$")


def test_step1b_source_refs_non_empty_and_reference_real_indices() -> None:
    """Every INSERTed row has a non-empty ``source_refs`` list, and every
    ref resolves to a real ``texts[i]/tables[i]/pictures[i]`` index in the
    source doc_json.
    """
    from app.services.extraction_chunk_index import (
        build_extraction_index_hybrid,
        read_chunk_source_refs,
    )

    doc_json = _build_dvina_like_doc_json()
    store = _make_mock_store()

    # Size fake embeddings to match the chunker's actual output.
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction
    n = len(build_hybrid_chunks_for_extraction(doc_json))
    fake_embeddings = [[0.1] * 1024] * n

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=fake_embeddings,
    ):
        build_extraction_index_hybrid(
            doc_json,
            pipeline_run_id="run-1b",
            document_id="doc-1b",
            store=store,
        )

    n_texts = len(doc_json.get("texts", []))
    n_tables = len(doc_json.get("tables", []))
    n_pictures = len(doc_json.get("pictures", []))

    rows = _captured_insert_params(store)
    assert rows, "expected >=1 INSERT row captured"

    for row in rows:
        # Use the Task 1 accessor — belt-and-suspenders for legacy/None coalescing.
        refs = read_chunk_source_refs(row)
        assert refs, f"row vertex_id={row.get('vertex_id')!r} has empty source_refs"
        for ref in refs:
            m = _REF_RE.match(ref)
            assert m, f"source_ref {ref!r} not in #/(texts|tables|pictures)/N form"
            kind, idx_str = m.group(1), m.group(2)
            idx = int(idx_str)
            cap = {"texts": n_texts, "tables": n_tables, "pictures": n_pictures}[kind]
            assert 0 <= idx < cap, (
                f"source_ref {ref!r} index {idx} out of range "
                f"(doc has {cap} {kind})"
            )


# ---------------------------------------------------------------------------
# Step 1c — chunk text includes heading prefix
# ---------------------------------------------------------------------------


def test_step1c_chunk_text_includes_heading_prefix() -> None:
    """Each merged chunk's stored ``chunk_text`` contains the nearest
    parent heading (HybridChunker's ``contextualize`` prepends title +
    heading). Uses the heading-terms match for tolerance vs docling
    formatting drift.
    """
    from app.services.extraction_chunk_index import build_extraction_index_hybrid
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    doc_json = _build_dvina_like_doc_json()
    expected_chunks = build_hybrid_chunks_for_extraction(doc_json)
    assert expected_chunks

    store = _make_mock_store()
    fake_embeddings = [[0.1] * 1024] * len(expected_chunks)

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=fake_embeddings,
    ):
        build_extraction_index_hybrid(
            doc_json,
            pipeline_run_id="run-1c",
            document_id="doc-1c",
            store=store,
        )

    rows = _captured_insert_params(store)
    assert rows

    # Same fragment→heading mapping as test_hybrid_chunking::test_chunk_text_contains_parent_heading.
    expected_heading_per_text = {
        "Soviet-era surface-to-air": "Chapter 1: Overview",
        "Maximum engagement range": "Chapter 2: Kinematics",
        "Radar tracking accuracy": "Section 2.1: Tracking",
        "comprises the Fan Song": "Chapter 3: Components",
    }

    matched: dict[str, str] = {}
    for row in rows:
        text = row.get("chunk_text") or ""
        for body_fragment, heading in expected_heading_per_text.items():
            if body_fragment in text:
                matched[body_fragment] = text
                assert _heading_terms_in(text, heading), (
                    f"row chunk_text containing {body_fragment!r} is missing "
                    f"heading terms from {heading!r}; text was:\n{text!r}"
                )
                assert _heading_terms_in(text, "S-75 Dvina System Manual"), (
                    f"row chunk_text containing {body_fragment!r} is missing "
                    f"title terms; text was:\n{text!r}"
                )

    assert matched, (
        "No INSERT row's chunk_text matched any expected body fragment — "
        "HybridChunker output diverged from synthetic input."
    )


# ---------------------------------------------------------------------------
# Step 1d — deterministic chunk_index across two runs
# ---------------------------------------------------------------------------


def test_step1d_deterministic_chunk_index_across_runs() -> None:
    """Calling ``build_extraction_index_hybrid`` twice on the same doc_json
    produces identical ``chunk_index`` values in identical insertion order.

    HybridChunker is RNG-free; the indexer must not introduce any
    non-determinism (e.g. unordered set iteration).
    """
    from app.services.extraction_chunk_index import (
        build_extraction_index_hybrid,
        read_chunk_index,
    )
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    doc_json = _build_dvina_like_doc_json()
    n = len(build_hybrid_chunks_for_extraction(doc_json))
    fake_embeddings = [[0.1] * 1024] * n

    def _run(run_id: str) -> list[int]:
        store = _make_mock_store()
        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=fake_embeddings,
        ):
            build_extraction_index_hybrid(
                doc_json,
                pipeline_run_id=run_id,
                document_id="doc-1d",
                store=store,
            )
        rows = _captured_insert_params(store)
        return [read_chunk_index(r) for r in rows]

    first = _run("run-1d-a")
    second = _run("run-1d-b")

    assert first == second, (
        f"chunk_index order drifted between runs: {first!r} vs {second!r}"
    )
    # And the chunk_index sequence is dense from 0 — matches MergedChunk
    # density invariant from Task 2.
    assert first == list(range(len(first)))


# ---------------------------------------------------------------------------
# Step 1e — empty doc returns zero-reason diagnostics
# ---------------------------------------------------------------------------


def test_step1e_empty_doc_returns_zero_reason_diagnostics() -> None:
    """An empty doc (HybridChunker emits nothing) returns
    ``BuildIndexDiagnostics(chunks_inserted=0,
    chunks_inserted_zero_reason="empty_chunker_output")`` and never calls
    ``embed_texts``.
    """
    from app.services.extraction_chunk_index import (
        BuildIndexDiagnostics,
        build_extraction_index_hybrid,
    )

    doc_json = _build_empty_doc_json()
    store = _make_mock_store()

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=[],
    ) as mock_embed:
        diag = build_extraction_index_hybrid(
            doc_json,
            pipeline_run_id="run-1e",
            document_id="doc-1e",
            store=store,
        )

    assert isinstance(diag, BuildIndexDiagnostics)
    assert diag.chunks_inserted == 0
    assert diag.chunks_inserted_zero_reason == "empty_chunker_output"
    # Don't waste an embed call on zero chunks.
    assert mock_embed.call_count == 0

    # DELETE was still called (idempotent pre-clean) — exactly once.
    delete_calls = [
        c for c in store._client.command_sync.call_args_list
        if "DELETE" in str(c[0][2]).upper()
    ]
    assert len(delete_calls) == 1


# ---------------------------------------------------------------------------
# Step 1f — malformed doc_json raises ValueError
# ---------------------------------------------------------------------------


def test_step1f_malformed_doc_json_raises_value_error() -> None:
    """A malformed dict (not a valid ``DoclingDocument``) must propagate
    ``ValueError`` (pydantic ``ValidationError`` is a subclass). NEVER
    silently swallowed — the C.4 dispatcher's try/except converts this
    to RUN_FULL fallback.
    """
    from app.services.extraction_chunk_index import build_extraction_index_hybrid

    bad = {"this": "is not a docling document"}
    store = _make_mock_store()

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=[],
    ):
        with pytest.raises(ValueError):
            build_extraction_index_hybrid(
                bad,
                pipeline_run_id="run-1f",
                document_id="doc-1f",
                store=store,
            )


# ---------------------------------------------------------------------------
# Step 1g — embed_texts partial response → RuntimeError
# ---------------------------------------------------------------------------


def test_step1g_embed_partial_response_raises_runtime_error() -> None:
    """If ``embed_texts`` returns fewer (or more) vectors than chunks,
    raise ``RuntimeError``. Mirrors the length-guard contract of the
    legacy ``build_extraction_index``.
    """
    from app.services.extraction_chunk_index import build_extraction_index_hybrid
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    doc_json = _build_dvina_like_doc_json()
    n = len(build_hybrid_chunks_for_extraction(doc_json))
    assert n >= 2, "fixture must yield >=2 chunks for a meaningful partial-response test"

    # Return ONE FEWER embedding than chunks.
    truncated = [[0.1] * 1024] * (n - 1)
    store = _make_mock_store()

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=truncated,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            build_extraction_index_hybrid(
                doc_json,
                pipeline_run_id="run-1g",
                document_id="doc-1g",
                store=store,
            )

    # Message should mention the count mismatch, for log-grep ergonomics.
    msg = str(exc_info.value).lower()
    assert "embed" in msg
    assert str(n - 1) in msg or str(n) in msg


# ---------------------------------------------------------------------------
# Sanity: vertex_id format is the new merged-mode format
# ---------------------------------------------------------------------------


def test_merged_vertex_id_format_is_pipeline_run_id_colon_chunk_index() -> None:
    """Each INSERT's ``vertex_id`` follows the new merged format:
    ``f"{pipeline_run_id}:chunk_{chunk_index}"``.

    This locks the format at the indexer boundary — Task 6's dispatcher
    relies on this shape to back-map selected chunks to their indices.
    """
    from app.services.extraction_chunk_index import (
        build_extraction_index_hybrid,
        read_chunk_index,
    )
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    doc_json = _build_dvina_like_doc_json()
    n = len(build_hybrid_chunks_for_extraction(doc_json))
    fake_embeddings = [[0.1] * 1024] * n
    store = _make_mock_store()

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=fake_embeddings,
    ):
        build_extraction_index_hybrid(
            doc_json,
            pipeline_run_id="run-merged-vid",
            document_id="doc-merged-vid",
            store=store,
        )

    rows = _captured_insert_params(store)
    assert rows
    for row in rows:
        idx = read_chunk_index(row)
        assert row.get("vertex_id") == f"run-merged-vid:chunk_{idx}", (
            f"vertex_id format drifted: got {row.get('vertex_id')!r}, "
            f"expected f'run-merged-vid:chunk_{idx}'"
        )


# ---------------------------------------------------------------------------
# Self-ref uniqueness — guards the post-Task-8 cleanup fix
# ---------------------------------------------------------------------------


def test_merged_self_ref_is_chunk_index_and_unique_per_row() -> None:
    """Merged rows write ``self_ref = f"chunk_{chunk_index}"`` so the column
    is unique-per-row across a single pipeline_run_id.

    Background: pre-fix, ``self_ref`` was set to ``source_refs[0]`` for
    legacy-consumer compatibility. Because two merged chunks can share the
    same first source_ref (e.g. both contain ``#/texts/0`` as their first
    element after the chunker's contextualization step), this made the
    column NON-UNIQUE — silently breaking the lexsort tiebreaker in
    ``search_extraction_chunks_direct`` (Task 8 calibration finding).

    The unique-key index in the schema is on ``vertex_id`` so INSERTs
    don't fail; this test guards against regression by asserting both
    the value shape AND uniqueness.
    """
    from app.services.extraction_chunk_index import (
        build_extraction_index_hybrid,
        read_chunk_index,
    )
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    doc_json = _build_dvina_like_doc_json()
    n = len(build_hybrid_chunks_for_extraction(doc_json))
    assert n >= 2, "fixture must yield >=2 chunks to exercise uniqueness"
    fake_embeddings = [[0.1] * 1024] * n
    store = _make_mock_store()

    with patch(
        "app.services.extraction_chunk_index.embed_texts",
        return_value=fake_embeddings,
    ):
        build_extraction_index_hybrid(
            doc_json,
            pipeline_run_id="run-self-ref-uniq",
            document_id="doc-self-ref-uniq",
            store=store,
        )

    rows = _captured_insert_params(store)
    assert rows

    self_refs = [row.get("self_ref") for row in rows]

    # Value shape: each self_ref == f"chunk_{chunk_index}".
    for row in rows:
        idx = read_chunk_index(row)
        assert row.get("self_ref") == f"chunk_{idx}", (
            f"self_ref drift on merged row: got {row.get('self_ref')!r}, "
            f"expected f'chunk_{idx}'"
        )

    # Uniqueness: no two merged rows share a self_ref within a run.
    assert len(set(self_refs)) == len(self_refs), (
        f"self_ref values are non-unique across merged rows: {self_refs!r}"
    )
