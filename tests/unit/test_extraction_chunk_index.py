"""Unit tests for extraction_chunk_index.py — VR Phase C.2.

All tests are mocked (no ArcadeDB, no Ollama). Tests exercise the
rendering helpers, diagnostics, and the build/cleanup public API.

TDD discipline: all tests were written BEFORE the implementation module.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic docling doc-json fixtures
# ---------------------------------------------------------------------------

def _make_text_elem(idx: int, text: str, page_no: int = 1, label: str = "text",
                    parent_section_ref: str | None = None) -> dict:
    """Build a minimal docling text element dict."""
    elem: dict[str, Any] = {
        "self_ref": f"#/texts/{idx}",
        "text": text,
        "label": label,
        "prov": [{"page_no": page_no}],
    }
    if parent_section_ref:
        elem["parent"] = {"$ref": parent_section_ref}
    return elem


def _make_table_elem(idx: int, caption_text: str | None, cells: list[list[str]],
                     page_no: int = 1) -> dict:
    """Build a minimal docling table element dict.

    ``cells`` is a 2D list of strings — rows × cols.
    The function stores caption as a ref to a synthetic text element
    at '#/texts/900{idx}'.

    NOTE: Uses real docling EXCLUSIVE end-index convention (Python slice):
      end_row_offset_idx = row_idx + 1  (i.e. range(start, end) = [start])
    Includes num_rows / num_cols as authoritative grid size (real docling shape).
    """
    table: dict[str, Any] = {
        "self_ref": f"#/tables/{idx}",
        "prov": [{"page_no": page_no}],
        "data": {},
    }
    if caption_text is not None:
        # Caption stored as $ref to a texts element (standard docling shape)
        table["captions"] = [{"$ref": f"#/texts/900{idx}"}]
    # Encode cells as flat table_cells list with start_row/col offsets.
    # Use EXCLUSIVE end indices (Python-slice convention, as in real docling):
    #   end_row_offset_idx = row_idx + 1  (occupies ONLY row_idx)
    num_rows = len(cells)
    num_cols = max((len(r) for r in cells), default=0)
    flat_cells: list[dict] = []
    for row_idx, row in enumerate(cells):
        for col_idx, text in enumerate(row):
            flat_cells.append({
                "text": text,
                "start_row_offset_idx": row_idx,
                "end_row_offset_idx": row_idx + 1,   # EXCLUSIVE
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1,   # EXCLUSIVE
            })
    table["data"]["table_cells"] = flat_cells
    table["data"]["num_rows"] = num_rows
    table["data"]["num_cols"] = num_cols
    return table


def _make_picture_elem(idx: int, caption_text: str | None, page_no: int = 1) -> dict:
    """Build a minimal docling picture element dict."""
    pic: dict[str, Any] = {
        "self_ref": f"#/pictures/{idx}",
        "prov": [{"page_no": page_no}],
    }
    if caption_text is not None:
        pic["captions"] = [{"$ref": f"#/texts/800{idx}"}]
    return pic


def _make_section_heading_elem(idx: int, text: str, page_no: int = 1) -> dict:
    """Build a minimal docling section-header text element."""
    return {
        "self_ref": f"#/texts/{idx}",
        "text": text,
        "label": "section_header",
        "prov": [{"page_no": page_no}],
    }


def _make_doc_json(
    texts: list[dict],
    tables: list[dict],
    pictures: list[dict],
    *,
    extra_texts: dict[str, str] | None = None,
) -> dict:
    """Assemble a synthetic docling doc_json dict.

    ``extra_texts`` maps self_ref like '#/texts/9000' → caption text.
    These are appended to the texts array as caption-label items (label
    "caption_ref") so the caption resolver can find them, but they are NOT
    independently indexed as extraction chunks (the walker skips them because
    "caption_ref" is not in the walkable-text labels set — see NOTE below).

    NOTE: In real docling documents, caption text items have label "caption"
    and CAN be indexed. The test helper uses a synthetic excluded label to
    avoid double-counting captions as standalone text chunks AND as embedded
    table/picture content. Production documents will behave differently per
    their actual label distributions.
    """
    all_texts = list(texts)
    if extra_texts:
        for ref, text in extra_texts.items():
            all_texts.append({
                "self_ref": ref,
                "text": text,
                # Use "caption_ref" — a synthetic label that the walker's
                # _HEADING_LABELS exclusion doesn't catch but the
                # _TEXT_CHUNK_LABELS allow-list also doesn't include, so the
                # walker's label check (or lack thereof) gates it. We rely on
                # the walker NOT having an allow-list for text labels (it
                # only excludes heading labels). Therefore, to truly keep
                # these out of the index, we use a heading label variant.
                # "title" is in _HEADING_LABELS → excluded from indexing.
                "label": "title",
                "prov": [],
            })
    # Build body.children from texts in order — mirrors real docling structure
    # where all texts parent to {cref: "#/body"} and body.children lists them
    # in document order. This is required for _resolve_parent_section_heading
    # which now uses body.children traversal (body.cref key, not parent.$ref).
    body_children = [
        {"cref": t["self_ref"]}
        for t in all_texts
        if isinstance(t, dict) and t.get("self_ref")
    ]
    return {
        "texts": all_texts,
        "tables": tables,
        "pictures": pictures,
        "body": {"children": body_children},
    }


# ---------------------------------------------------------------------------
# Helper: small doc with mixed modalities
# ---------------------------------------------------------------------------

def _make_mixed_doc() -> dict:
    """3 text elements + 1 table + 2 pictures (1 with caption, 1 without).

    Expected result:
      - texts: 3 inserted
      - table: 1 inserted
      - pictures: 1 inserted (captioned), 1 skipped (no caption)
      Total: 5 inserts, 1 skip
    """
    texts = [
        _make_text_elem(0, "Radar tracking parameters and specifications."),
        _make_text_elem(1, "Maximum engagement altitude is 30 km."),
        _make_text_elem(2, "Minimum slant range 500 m."),
    ]
    tables = [
        _make_table_elem(
            0,
            caption_text="Table 1 — SA-2 Guidance Parameters",
            cells=[["Parameter", "Value"], ["PRF", "200 Hz"], ["Freq", "E-band"]],
        )
    ]
    pictures = [
        _make_picture_elem(0, caption_text="Fan Song E radar front view"),
        _make_picture_elem(1, caption_text=None),  # no caption → skip
    ]
    return _make_doc_json(
        texts, tables, pictures,
        extra_texts={
            "#/texts/9000": "Table 1 — SA-2 Guidance Parameters",
            "#/texts/8000": "Fan Song E radar front view",
        },
    )


# ---------------------------------------------------------------------------
# Mock store factory
# ---------------------------------------------------------------------------

def _make_mock_store() -> MagicMock:
    """Return a mock ArcadeDBGraphStore with command_sync stubbed."""
    store = MagicMock()
    store._database = "test_db"
    # command_sync returns a list with a count dict (like ArcadeDB DELETE result)
    store._client.command_sync.return_value = [{"count": 0}]
    store._client.query_sync.return_value = []
    return store


# ---------------------------------------------------------------------------
# Tests: BuildIndexDiagnostics and overall counts
# ---------------------------------------------------------------------------


class TestBuildReturnsDiagnostics:
    """test_build_returns_diagnostics_with_chunk_counts"""

    def test_build_returns_correct_counts(self):
        """5 inserts (1 picture-no-caption skipped), modality_counts correct."""
        from app.services.extraction_chunk_index import build_extraction_index

        doc_json = _make_mixed_doc()
        store = _make_mock_store()

        fake_embeddings = [[0.1] * 1024] * 5  # 5 non-empty elements

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=fake_embeddings,
        ) as mock_embed:
            diag = build_extraction_index(
                doc_json,
                "run-abc",
                "doc-xyz",
                store=store,
            )

        assert diag.chunks_inserted == 5
        assert diag.chunks_skipped == 1  # picture with no caption
        assert diag.modality_counts == {"text": 3, "table": 1, "picture_caption": 1}
        # embed_texts called once with all 5 non-empty texts as a batch
        assert mock_embed.call_count == 1

    def test_build_diagnostics_embed_ms_and_insert_ms_are_ints(self):
        """embed_ms and insert_ms must be non-negative ints."""
        from app.services.extraction_chunk_index import build_extraction_index

        doc_json = _make_mixed_doc()
        store = _make_mock_store()
        fake_embeddings = [[0.1] * 1024] * 5

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=fake_embeddings,
        ):
            diag = build_extraction_index(doc_json, "run-t", "doc-t", store=store)

        assert isinstance(diag.embed_ms, int)
        assert isinstance(diag.insert_ms, int)
        assert diag.embed_ms >= 0
        assert diag.insert_ms >= 0


class TestBuildSkipsEmptyTextElements:
    """test_build_skips_empty_text_elements"""

    def test_empty_text_element_is_skipped(self):
        """sanitized-blank text element (text='') → skipped, not embedded."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [
            _make_text_elem(0, "Real content here."),
            _make_text_elem(1, ""),       # blank
            _make_text_elem(2, "   "),    # whitespace only
        ]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        fake_embeddings = [[0.1] * 1024] * 1  # only 1 non-empty

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=fake_embeddings,
        ) as mock_embed:
            diag = build_extraction_index(doc_json, "run-empty", "doc-e", store=store)

        assert diag.chunks_inserted == 1
        assert diag.chunks_skipped == 2
        # embed_texts should only receive 1 text (the non-empty one)
        call_args = mock_embed.call_args[0][0]
        assert len(call_args) == 1

    def test_empty_rendered_text_not_sent_to_embed(self):
        """embed_texts is never called with an empty string in its input list."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "")]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[],
        ) as mock_embed:
            diag = build_extraction_index(doc_json, "run-e2", "doc-e2", store=store)

        assert diag.chunks_inserted == 0
        assert diag.chunks_skipped == 1
        # embed_texts either not called or called with empty list
        for c in mock_embed.call_args_list:
            texts_arg = c[0][0]
            assert all(t.strip() for t in texts_arg), (
                "embed_texts received an empty/blank string"
            )


class TestBuildIdempotentDeleteThenInsert:
    """test_build_idempotent_delete_then_insert"""

    def test_delete_called_before_insert(self):
        """DELETE for pipeline_run_id is called before any INSERT."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "Some content.")]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024],
        ):
            build_extraction_index(doc_json, "run-idem", "doc-i", store=store)

        # command_sync must have been called at least twice:
        # once for DELETE, at least once for INSERT
        calls = store._client.command_sync.call_args_list
        assert len(calls) >= 2

        # First call must be the DELETE
        first_cmd = calls[0][0][2]  # positional: (db, lang, sql, params)
        assert "DELETE" in first_cmd.upper()
        assert "ExtractionChunk" in first_cmd

    def test_second_build_same_run_id_produces_same_insert_count(self):
        """Calling build twice with same run_id → same chunk count each time."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [
            _make_text_elem(0, "Radar specs."),
            _make_text_elem(1, "Altitude range."),
        ]
        doc_json = _make_doc_json(texts, [], [])
        fake_embeddings = [[0.1] * 1024, [0.2] * 1024]

        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=fake_embeddings,
        ):
            diag1 = build_extraction_index(doc_json, "run-idem2", "doc-i2", store=store)
            diag2 = build_extraction_index(doc_json, "run-idem2", "doc-i2", store=store)

        assert diag1.chunks_inserted == diag2.chunks_inserted == 2

    def test_delete_uses_correct_run_id_param(self):
        """DELETE WHERE pipeline_run_id = :run_id uses the correct value."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "Content.")]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024],
        ):
            build_extraction_index(doc_json, "my-specific-run-id", "doc-x", store=store)

        # Find the DELETE call and check params
        for c in store._client.command_sync.call_args_list:
            args = c[0]
            if len(args) >= 3 and "DELETE" in args[2].upper():
                params = args[3] if len(args) > 3 else c[1].get("params", {})
                run_id_val = (params or {}).get("run_id")
                assert run_id_val == "my-specific-run-id", (
                    f"DELETE used wrong run_id: {run_id_val!r}"
                )
                break
        else:
            pytest.fail("No DELETE call found in command_sync calls")


# ---------------------------------------------------------------------------
# Tests: table rendering
# ---------------------------------------------------------------------------


class TestTableRendering:
    """test_build_renders_table_as_markdown_with_caption"""

    def test_table_with_caption_and_cells_renders_correctly(self):
        """Table elem with caption + cells → rendered text has caption + markdown."""
        from app.services.extraction_chunk_index import _render_table_chunk

        table = _make_table_elem(
            0,
            caption_text="Table 1 — SA-2 Guidance Parameters",
            cells=[["Parameter", "Value"], ["PRF", "200 Hz"]],
        )
        doc_json = _make_doc_json(
            [], [table], [],
            extra_texts={"#/texts/9000": "Table 1 — SA-2 Guidance Parameters"},
        )

        rendered = _render_table_chunk(table, doc_json, include_caption=True)

        assert "Table 1 — SA-2 Guidance Parameters" in rendered
        assert "Parameter" in rendered
        assert "200 Hz" in rendered
        # Markdown table separator present
        assert "---" in rendered or "| ---" in rendered

    def test_table_without_caption_flag_omits_caption(self):
        """include_caption=False → caption not in rendered text."""
        from app.services.extraction_chunk_index import _render_table_chunk

        table = _make_table_elem(
            0,
            caption_text="Table 1 — SA-2 Guidance Parameters",
            cells=[["Parameter", "Value"], ["PRF", "200 Hz"]],
        )
        doc_json = _make_doc_json(
            [], [table], [],
            extra_texts={"#/texts/9000": "Table 1 — SA-2 Guidance Parameters"},
        )

        rendered = _render_table_chunk(table, doc_json, include_caption=False)

        assert "Table 1 — SA-2 Guidance Parameters" not in rendered
        # Cells still present
        assert "Parameter" in rendered or "PRF" in rendered

    def test_table_no_cells_and_no_caption_is_empty(self):
        """Table with no caption + no cells → empty string."""
        from app.services.extraction_chunk_index import _render_table_chunk

        table = {
            "self_ref": "#/tables/0",
            "prov": [{"page_no": 1}],
            "data": {"table_cells": []},
        }
        doc_json = _make_doc_json([], [table], [])

        rendered = _render_table_chunk(table, doc_json, include_caption=True)

        assert rendered.strip() == ""

    def test_table_with_caption_only_no_cells_still_has_caption(self):
        """Table with caption but no cells → caption text returned."""
        from app.services.extraction_chunk_index import _render_table_chunk

        table = {
            "self_ref": "#/tables/0",
            "captions": [{"$ref": "#/texts/9000"}],
            "prov": [{"page_no": 1}],
            "data": {"table_cells": []},
        }
        doc_json = _make_doc_json(
            [], [table], [],
            extra_texts={"#/texts/9000": "Summary table"},
        )

        rendered = _render_table_chunk(table, doc_json, include_caption=True)

        assert "Summary table" in rendered

    def test_build_uses_include_table_caption_flag(self):
        """build_extraction_index include_table_caption=False → no caption in chunks."""
        from app.services.extraction_chunk_index import build_extraction_index

        table = _make_table_elem(
            0,
            caption_text="Table 2 — Engagement Envelope",
            cells=[["Range", "Value"], ["Max", "100 km"]],
        )
        doc_json = _make_doc_json(
            [], [table], [],
            extra_texts={"#/texts/9000": "Table 2 — Engagement Envelope"},
        )
        store = _make_mock_store()

        captured_chunks: list[str] = []

        original_command_sync = store._client.command_sync.side_effect

        def capture_command(db, lang, sql, params=None):
            if params and "chunk_text" in params:
                captured_chunks.append(params["chunk_text"])
            return [{"count": 0}]

        store._client.command_sync.side_effect = capture_command

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024],
        ):
            build_extraction_index(
                doc_json, "run-tc", "doc-tc", store=store,
                include_table_caption=False,
            )

        assert any("Table 2" not in chunk for chunk in captured_chunks if chunk), (
            "Caption should not appear when include_table_caption=False"
        )
        # More precise: caption specifically absent from the table chunk
        for chunk in captured_chunks:
            if "Range" in chunk or "Max" in chunk:  # table chunk
                assert "Table 2 — Engagement Envelope" not in chunk


# ---------------------------------------------------------------------------
# Tests: text rendering with parent section heading
# ---------------------------------------------------------------------------


class TestTextRenderingWithHeading:
    """test_build_renders_text_with_parent_section_heading_by_default"""

    def _make_doc_with_heading(self) -> tuple[dict, dict, dict]:
        """Returns (section_elem, text_elem, doc_json) with parent linkage."""
        section = _make_section_heading_elem(0, "Radar Specifications")
        # text elem references the section as parent
        text_elem = _make_text_elem(
            1,
            "PRF is 200 Hz.",
            parent_section_ref="#/texts/0",
        )
        doc_json = _make_doc_json([section, text_elem], [], [])
        return section, text_elem, doc_json

    def test_text_with_parent_heading_prepended_by_default(self):
        """Text elem with parent section → rendered text starts with heading."""
        from app.services.extraction_chunk_index import _render_text_chunk

        section, text_elem, doc_json = self._make_doc_with_heading()

        rendered = _render_text_chunk(
            text_elem, doc_json, include_parent_section_heading=True
        )

        assert "Radar Specifications" in rendered
        assert "PRF is 200 Hz." in rendered

    def test_text_without_heading_flag_renders_just_text(self):
        """include_parent_section_heading=False → just raw text."""
        from app.services.extraction_chunk_index import _render_text_chunk

        section, text_elem, doc_json = self._make_doc_with_heading()

        rendered = _render_text_chunk(
            text_elem, doc_json, include_parent_section_heading=False
        )

        assert "Radar Specifications" not in rendered
        assert "PRF is 200 Hz." in rendered

    def test_text_no_parent_still_renders(self):
        """Text elem with no parent → renders raw text, no heading prefix."""
        from app.services.extraction_chunk_index import _render_text_chunk

        text_elem = _make_text_elem(0, "Standalone text.")
        doc_json = _make_doc_json([text_elem], [], [])

        rendered = _render_text_chunk(
            text_elem, doc_json, include_parent_section_heading=True
        )

        assert "Standalone text." in rendered

    def test_build_passes_heading_flag_to_text_render(self):
        """build_extraction_index include_parent_section_heading=False → no heading."""
        from app.services.extraction_chunk_index import build_extraction_index

        section = _make_section_heading_elem(0, "Radar Specifications")
        text_elem = _make_text_elem(
            1, "PRF is 200 Hz.", parent_section_ref="#/texts/0"
        )
        doc_json = _make_doc_json([section, text_elem], [], [])
        store = _make_mock_store()

        captured_chunks: list[str] = []

        def capture_command(db, lang, sql, params=None):
            if params and "chunk_text" in params:
                captured_chunks.append(params["chunk_text"])
            return [{"count": 0}]

        store._client.command_sync.side_effect = capture_command

        # section_header elements are skipped (label-based filter), so only 1 insert
        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024],
        ):
            build_extraction_index(
                doc_json, "run-head", "doc-head", store=store,
                include_parent_section_heading=False,
            )

        for chunk in captured_chunks:
            assert "Radar Specifications" not in chunk, (
                f"Heading should not appear when flag=False, got: {chunk!r}"
            )


# ---------------------------------------------------------------------------
# Tests: picture rendering
# ---------------------------------------------------------------------------


class TestPictureRendering:
    """test_build_renders_picture_caption_only"""

    def test_picture_caption_is_returned(self):
        """Picture with caption → exactly the caption text."""
        from app.services.extraction_chunk_index import _render_picture_chunk

        pic = _make_picture_elem(0, caption_text="Fan Song E radar")
        doc_json = _make_doc_json(
            [], [], [pic],
            extra_texts={"#/texts/8000": "Fan Song E radar"},
        )

        rendered = _render_picture_chunk(pic, doc_json)

        assert rendered == "Fan Song E radar"

    def test_picture_no_caption_returns_empty(self):
        """Picture with no caption → empty string."""
        from app.services.extraction_chunk_index import _render_picture_chunk

        pic = _make_picture_elem(0, caption_text=None)
        doc_json = _make_doc_json([], [], [pic])

        rendered = _render_picture_chunk(pic, doc_json)

        assert rendered.strip() == ""

    def test_picture_no_inferred_description(self):
        """No description / OCR fallback — only caption text returned."""
        from app.services.extraction_chunk_index import _render_picture_chunk

        pic = {
            "self_ref": "#/pictures/0",
            "prov": [{"page_no": 1}],
            "captions": [{"$ref": "#/texts/8000"}],
            "inferred_description": "A large radar dish",  # must NOT be used
        }
        doc_json = _make_doc_json(
            [], [], [pic],
            extra_texts={"#/texts/8000": "Fan Song E"},
        )

        rendered = _render_picture_chunk(pic, doc_json)

        assert "Fan Song E" in rendered
        assert "A large radar dish" not in rendered


# ---------------------------------------------------------------------------
# Tests: self_ref format
# ---------------------------------------------------------------------------


class TestSelfRefFormat:
    """test_build_self_ref_format"""

    def test_self_refs_match_docling_convention(self):
        """Inserted self_refs match '#/texts/N', '#/tables/N', '#/pictures/N'."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [
            _make_text_elem(3, "Content A."),
            _make_text_elem(7, "Content B."),
        ]
        tables = [_make_table_elem(2, "Cap", cells=[["X", "Y"]])]
        pictures = [_make_picture_elem(5, "A picture caption")]
        doc_json = _make_doc_json(
            texts, tables, pictures,
            extra_texts={
                "#/texts/9002": "Cap",
                "#/texts/8005": "A picture caption",
            },
        )
        store = _make_mock_store()

        inserted_self_refs: list[str] = []

        def capture_command(db, lang, sql, params=None):
            if params and "self_ref" in params:
                inserted_self_refs.append(params["self_ref"])
            return [{"count": 0}]

        store._client.command_sync.side_effect = capture_command

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024] * 4,
        ):
            build_extraction_index(doc_json, "run-ref", "doc-ref", store=store)

        assert "#/texts/3" in inserted_self_refs
        assert "#/texts/7" in inserted_self_refs
        assert "#/tables/2" in inserted_self_refs
        assert "#/pictures/5" in inserted_self_refs

    def test_vertex_id_format_is_run_id_colon_self_ref(self):
        """vertex_id must be f'{pipeline_run_id}:{self_ref}'."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "A text element.")]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        inserted_vertex_ids: list[str] = []

        def capture_command(db, lang, sql, params=None):
            if params and "vertex_id" in params:
                inserted_vertex_ids.append(params["vertex_id"])
            return [{"count": 0}]

        store._client.command_sync.side_effect = capture_command

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024],
        ):
            build_extraction_index(doc_json, "run-vid", "doc-vid", store=store)

        assert "run-vid:#/texts/0" in inserted_vertex_ids


# ---------------------------------------------------------------------------
# Tests: embedding batch count
# ---------------------------------------------------------------------------


class TestEmbeddingBatchCount:
    """test_build_embedding_batch_count"""

    def test_embed_texts_called_once_with_full_batch(self):
        """embed_texts called ONCE with all non-empty texts, not once per chunk."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [
            _make_text_elem(0, f"Text element {i}.") for i in range(8)
        ]
        tables = [_make_table_elem(0, "Cap", cells=[["A", "B"]])]
        doc_json = _make_doc_json(
            texts, tables, [],
            extra_texts={"#/texts/9000": "Cap"},
        )
        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024] * 9,  # 8 texts + 1 table
        ) as mock_embed:
            build_extraction_index(doc_json, "run-batch", "doc-batch", store=store)

        assert mock_embed.call_count == 1, (
            f"embed_texts should be called once with the full batch, "
            f"got {mock_embed.call_count} calls"
        )

    def test_embed_texts_batch_size_matches_non_empty_count(self):
        """The single embed_texts call receives exactly the non-empty rendered texts."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [
            _make_text_elem(0, "Alpha."),
            _make_text_elem(1, ""),       # skip
            _make_text_elem(2, "Gamma."),
        ]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024, [0.2] * 1024],
        ) as mock_embed:
            build_extraction_index(doc_json, "run-bs", "doc-bs", store=store)

        call_args = mock_embed.call_args[0][0]
        assert len(call_args) == 2
        assert all(t.strip() for t in call_args)

    def test_embed_texts_not_called_when_all_elements_empty(self):
        """When every element renders to empty, embed_texts is not called at all."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [
            _make_text_elem(0, ""),
            _make_text_elem(1, "   "),
        ]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[],
        ) as mock_embed:
            diag = build_extraction_index(doc_json, "run-zero", "doc-zero", store=store)

        assert diag.chunks_inserted == 0
        for c in mock_embed.call_args_list:
            texts_arg = c[0][0]
            assert texts_arg == [], (
                "embed_texts should not be called with non-empty content"
            )

    def test_embed_calls_counter_reflects_batch_count(self):
        """diag.embed_calls == 1 when a single batch is embedded."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(i, f"Element {i}.") for i in range(5)]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 1024] * 5,
        ):
            diag = build_extraction_index(doc_json, "run-ec", "doc-ec", store=store)

        assert diag.embed_calls == 1


# ---------------------------------------------------------------------------
# Tests: _walk_docling_elements modality/self_ref coverage
# ---------------------------------------------------------------------------


class TestWalkDoclingElements:
    """Tests for the _walk_docling_elements private helper."""

    def test_yields_texts_tables_pictures(self):
        """All three modalities are yielded."""
        from app.services.extraction_chunk_index import _walk_docling_elements

        texts = [_make_text_elem(0, "T0"), _make_text_elem(1, "T1")]
        tables = [_make_table_elem(0, None, cells=[["A", "B"]])]
        pictures = [_make_picture_elem(0, "A picture")]
        doc_json = _make_doc_json(texts, tables, pictures)

        results = list(_walk_docling_elements(doc_json))
        self_refs = {r[0] for r in results}
        modalities = {r[1] for r in results}

        assert "#/texts/0" in self_refs
        assert "#/texts/1" in self_refs
        assert "#/tables/0" in self_refs
        assert "#/pictures/0" in self_refs
        assert "text" in modalities
        assert "table" in modalities
        assert "picture_caption" in modalities

    def test_yields_correct_self_ref_format(self):
        """self_refs follow '#/section/N' convention."""
        from app.services.extraction_chunk_index import _walk_docling_elements

        texts = [_make_text_elem(5, "Five")]
        tables = [_make_table_elem(3, None, cells=[["X"]])]
        pictures = [_make_picture_elem(7, "P")]
        doc_json = _make_doc_json(texts, tables, pictures)

        results = list(_walk_docling_elements(doc_json))
        self_refs = [r[0] for r in results]

        assert "#/texts/5" in self_refs
        assert "#/tables/3" in self_refs
        assert "#/pictures/7" in self_refs

    def test_empty_doc_yields_nothing(self):
        """Empty doc → no elements yielded."""
        from app.services.extraction_chunk_index import _walk_docling_elements

        doc_json = _make_doc_json([], [], [])
        results = list(_walk_docling_elements(doc_json))
        assert results == []

    def test_heading_only_doc_excludes_headings_from_walk(self):
        """Rev 14 code-quality review Minor #4: explicit walker-level test
        for the heading exclusion contract. Elements labeled section_header,
        section-header, or title are NEVER yielded as indexable chunks even
        if they have non-empty text. They serve only as parent-context
        providers for adjacent text elements (see _resolve_parent_section_heading).
        """
        from app.services.extraction_chunk_index import _walk_docling_elements

        # Doc with ONLY heading-labeled text elements + no tables + no pictures.
        texts = [
            {"label": "section_header", "text": "Radar Specifications",
             "self_ref": "#/texts/0"},
            {"label": "section-header", "text": "Hyphenated heading",
             "self_ref": "#/texts/1"},
            {"label": "title", "text": "Document Title",
             "self_ref": "#/texts/2"},
        ]
        doc_json = _make_doc_json(texts, [], [])
        results = list(_walk_docling_elements(doc_json))
        assert results == [], (
            f"Heading-only doc should yield zero indexable elements; "
            f"got {[r[0] for r in results]}"
        )


# ---------------------------------------------------------------------------
# Fixture path helper
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "docling_anchors"


# ---------------------------------------------------------------------------
# HIGH fix: fixture-backed heading-context tests (body.children traversal)
# ---------------------------------------------------------------------------


class TestRealDoclingFixtureHeadingContext:
    """Fixture-backed tests for _resolve_parent_section_heading.

    Real docling JSON has texts parenting to {cref: '#/body'} — not to
    #/texts/N.  Heading resolution MUST use body.children traversal.
    Uses tests/fixtures/docling_anchors/interleaved_pictures.json which
    has section_header at texts[0] followed by non-heading texts at
    texts[1,3,4,6].
    """

    @pytest.fixture(scope="class")
    def interleaved_doc(self):
        path = _FIXTURE_DIR / "interleaved_pictures.json"
        with open(path) as f:
            return json.load(f)

    def test_real_docling_fixture_resolves_heading_context(self, interleaved_doc):
        """Non-heading texts after a section_header in body.children must
        resolve to that heading via body.children traversal.
        Uses interleaved_pictures.json: texts[0]='Chapter 1' (section_header),
        texts[1]='Opening paragraph...' (text label).
        """
        from app.services.extraction_chunk_index import _resolve_parent_section_heading

        doc = interleaved_doc
        # Find the first non-heading text
        non_heading = None
        for t in doc["texts"]:
            if t.get("label") not in ("title", "section_header", "section-header"):
                non_heading = t
                break
        assert non_heading is not None, (
            "Fixture has no non-heading texts; cannot test heading resolution"
        )

        heading = _resolve_parent_section_heading(non_heading, doc)
        assert heading, (
            f"Expected a heading for {non_heading['self_ref']!r}; got empty string. "
            f"Real docling uses cref='#/body' — body.children traversal required."
        )

        # The resolved heading must be an actual heading text in the fixture
        heading_texts = {
            t["text"]
            for t in doc["texts"]
            if t.get("label") in ("title", "section_header", "section-header")
        }
        assert heading in heading_texts, (
            f"Resolved heading {heading!r} not in fixture headings {heading_texts!r}"
        )

    def test_heading_is_most_recent_before_target(self, interleaved_doc):
        """The resolved heading is the NEAREST preceding heading in body order.
        texts[0] = 'Chapter 1' (section_header)
        texts[1] = non-heading → should resolve to 'Chapter 1'
        """
        from app.services.extraction_chunk_index import _resolve_parent_section_heading

        doc = interleaved_doc
        # texts[1] is "Opening paragraph before any picture." (label=text)
        target = doc["texts"][1]
        assert target.get("label") not in ("title", "section_header", "section-header"), (
            f"Fixture changed: texts[1] is now a heading ({target.get('label')!r})"
        )
        heading = _resolve_parent_section_heading(target, doc)
        assert heading == "Chapter 1", (
            f"Expected 'Chapter 1' (texts[0]); got {heading!r}"
        )

    def test_element_before_any_heading_returns_empty(self, interleaved_doc):
        """If the target self_ref appears BEFORE any heading in body.children,
        return ''. This fixture's texts[0] IS a heading, so use a synthetic
        element injected before the first body child."""
        from app.services.extraction_chunk_index import _resolve_parent_section_heading

        # Inject a synthetic non-heading element at the front of body.children
        # without modifying the shared fixture
        doc_copy = dict(interleaved_doc)
        doc_copy = {k: v for k, v in interleaved_doc.items()}
        # Deep-copy just what we need to mutate
        import copy
        doc_copy = copy.deepcopy(interleaved_doc)
        doc_copy["texts"].insert(0, {
            "self_ref": "#/texts/99",
            "label": "text",
            "text": "Before any heading",
            "parent": {"cref": "#/body"},
        })
        doc_copy["body"]["children"].insert(0, {"cref": "#/texts/99"})

        heading = _resolve_parent_section_heading("#/texts/99", doc_copy)
        assert heading == "", (
            f"Element before any heading should return ''; got {heading!r}"
        )

    def test_build_prepends_heading_to_text_chunks_real_fixture(self, interleaved_doc):
        """build_extraction_index on interleaved_pictures.json with
        include_parent_section_heading=True: the chunk for texts[1]
        ('Opening paragraph...') must be prefixed with 'Chapter 1'.
        """
        from app.services.extraction_chunk_index import build_extraction_index

        store = _make_mock_store()
        captured: dict[str, str] = {}  # self_ref → chunk_text

        def capture_cmd(db, lang, sql, params=None):
            if params and "chunk_text" in params:
                captured[params.get("self_ref", "")] = params["chunk_text"]
            return [{"count": 0}]

        store._client.command_sync.side_effect = capture_cmd

        # Count non-heading texts to know how many embeddings to produce
        non_heading_count = sum(
            1 for t in interleaved_doc["texts"]
            if t.get("label") not in ("title", "section_header", "section-header")
        )
        # Add picture captions and table cells if any
        pic_count = sum(
            1 for p in interleaved_doc.get("pictures", [])
            if any(True for _ in (p.get("captions") or []))
        )
        tbl_count = len(interleaved_doc.get("tables", []))
        total = non_heading_count + pic_count + tbl_count

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 128] * total,
        ):
            diag = build_extraction_index(
                interleaved_doc,
                "run-heading-fixture",
                "doc-heading-fixture",
                store=store,
                include_parent_section_heading=True,
            )

        # texts[1] is the first non-heading text; its chunk must have heading
        chunk = captured.get("#/texts/1", "")
        assert "Chapter 1" in chunk, (
            f"Expected 'Chapter 1' prefix in chunk for #/texts/1; got: {chunk!r}. "
            f"All captured chunks: {captured!r}"
        )
        assert "Opening paragraph" in chunk, (
            f"Expected body text in chunk for #/texts/1; got: {chunk!r}"
        )


# ---------------------------------------------------------------------------
# MED fix: fixture-backed table shape test (exclusive end indices)
# ---------------------------------------------------------------------------


class TestRealDoclingTableShape:
    """Fixture-backed tests for _render_table_chunk.
    Uses tests/fixtures/docling_anchors/with_figures_tables.json which has a
    1×1 table where end_row_offset_idx=1 / end_col_offset_idx=1 (exclusive).
    """

    @pytest.fixture(scope="class")
    def figures_tables_doc(self):
        path = _FIXTURE_DIR / "with_figures_tables.json"
        with open(path) as f:
            return json.load(f)

    def test_real_docling_table_renders_1x1_not_2x2(self, figures_tables_doc):
        """The 1×1 table with end indices=1 (exclusive) must produce a
        1-row markdown table, NOT a 2×2 table with a blank row/column."""
        from app.services.extraction_chunk_index import _render_table_chunk

        doc = figures_tables_doc
        tbl = doc["tables"][0]

        # Confirm fixture shape: num_rows=1, num_cols=1, end_*=1 (exclusive)
        assert tbl["data"]["num_rows"] == 1
        assert tbl["data"]["num_cols"] == 1
        cell = tbl["data"]["table_cells"][0]
        assert cell["end_row_offset_idx"] == 1
        assert cell["end_col_offset_idx"] == 1
        assert cell["text"] == "a"

        rendered = _render_table_chunk(tbl, doc, include_caption=False)

        assert "a" in rendered, f"Cell text 'a' missing from rendering: {rendered!r}"

        lines = [ln for ln in rendered.splitlines() if ln.strip()]
        # A 1×1 table is ONLY a header row (+ optional separator) = 1 or 2 lines.
        # 3 lines would mean a spurious extra data row was emitted.
        # The old buggy code produced 3 lines (header + sep + blank data row).
        assert len(lines) <= 2, (
            f"1×1 table should be ≤2 lines (header row + separator); "
            f"got {len(lines)} lines: {lines!r}\n"
            f"Bug: end_*=1 is EXCLUSIVE; old code treated it as inclusive + added blank row."
        )

        # Header line must have exactly 1 pipe-delimited column (not 2)
        header_line = lines[0]
        cols_in_header = [c.strip() for c in header_line.strip("|").split("|")]
        assert len(cols_in_header) == 1, (
            f"1×1 table header must have 1 column; got {cols_in_header!r}\n"
            f"Bug: end_col=1 is EXCLUSIVE; old code added spurious blank column."
        )
        assert cols_in_header[0] == "a", (
            f"Header cell must be 'a'; got {cols_in_header[0]!r}"
        )

    def test_1x1_table_no_extra_columns(self, figures_tables_doc):
        """A 1×1 table must produce a markdown table with exactly 1 column.
        The old code produced 2 columns because end_col=1 was treated inclusive.
        Verifies by counting pipe separators in the header line.
        """
        from app.services.extraction_chunk_index import _render_table_chunk

        doc = figures_tables_doc
        tbl = doc["tables"][0]
        rendered = _render_table_chunk(tbl, doc, include_caption=False)

        lines = [ln for ln in rendered.splitlines() if ln.strip()]
        assert lines, "Rendered table must not be empty"

        # Count total lines — must be ≤ 2 (header + separator only for 1 row)
        assert len(lines) <= 2, (
            f"1×1 fixture table should have ≤2 lines; got {lines!r}"
        )

        # The header line must contain exactly 1 pipe-delimited cell (value "a")
        header = lines[0]
        # Remove leading/trailing pipes and split; filter empty strings from split
        parts = [p.strip() for p in header.strip("|").split("|")]
        assert len(parts) == 1, (
            f"Header must have 1 column; got {parts!r} in {header!r}\n"
            f"Bug: end_col=1 is EXCLUSIVE; old code treated inclusive => 2 columns."
        )
        assert parts[0] == "a", f"Header cell must be 'a'; got {parts[0]!r}"

    def test_table_uses_num_rows_num_cols_as_grid_size(self):
        """Grid is sized from data.num_rows/num_cols, NOT from max(end_*)+1.
        Synthetic test: 1×1 table with end_*=1 (exclusive) must yield 1×1 grid
        with exactly 1 row and 1 column — NOT the 2×2 the old code produced.
        """
        from app.services.extraction_chunk_index import _render_table_chunk

        tbl = {
            "self_ref": "#/tables/0",
            "prov": [],
            "data": {
                "num_rows": 1,
                "num_cols": 1,
                "table_cells": [
                    {
                        "text": "hello",
                        "start_row_offset_idx": 0,
                        "end_row_offset_idx": 1,    # exclusive — occupies ONLY row 0
                        "start_col_offset_idx": 0,
                        "end_col_offset_idx": 1,    # exclusive — occupies ONLY col 0
                    }
                ],
            },
        }
        doc = {"texts": [], "tables": [tbl], "pictures": []}
        rendered = _render_table_chunk(tbl, doc, include_caption=False)

        lines = [ln for ln in rendered.splitlines() if ln.strip()]
        # 1×1 table = 1 header row + 1 separator = 2 lines max.
        # 3 lines means old inclusive bug added a blank extra row.
        assert len(lines) <= 2, (
            f"1×1 synthetic table: expected ≤2 lines (header + sep); got {lines!r}\n"
            f"Bug: end_*=1 is EXCLUSIVE; old code treated it inclusive => blank extra row."
        )
        assert "hello" in rendered, f"Cell text missing: {rendered!r}"

        # Exactly 1 pipe-delimited non-empty column in the header
        header = lines[0]
        header_cells = [c.strip() for c in header.strip("|").split("|") if c.strip()]
        assert len(header_cells) == 1, (
            f"Expected 1 column; got {header_cells!r} in {header!r}"
        )


# ---------------------------------------------------------------------------
# MED fix: embed length mismatch raises RuntimeError
# ---------------------------------------------------------------------------


class TestEmbedLengthMismatch:
    """embed_texts returning fewer vectors than pending must raise, not truncate."""

    def test_build_raises_on_embedding_count_mismatch(self):
        """If embed_texts returns fewer vectors than pending, build must raise
        RuntimeError (not silently truncate via zip). C.4 catches + falls back."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [
            _make_text_elem(0, "Alpha content."),
            _make_text_elem(1, "Beta content."),
            _make_text_elem(2, "Gamma content."),
        ]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        # Return only 2 embeddings for 3 pending elements
        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 128, [0.2] * 128],  # short by 1
        ):
            with pytest.raises(RuntimeError) as exc_info:
                build_extraction_index(doc_json, "run-mismatch", "doc-mm", store=store)

        msg = str(exc_info.value)
        assert "2" in msg and "3" in msg, (
            f"Error message must cite both counts; got: {msg!r}"
        )
        assert "embed_texts" in msg.lower() or "embedding" in msg.lower(), (
            f"Error message must mention embeddings; got: {msg!r}"
        )

    def test_build_raises_on_extra_embeddings(self):
        """embed_texts returning MORE vectors than pending also raises."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "One item.")]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        # Return 3 embeddings for 1 pending element
        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 128, [0.2] * 128, [0.3] * 128],
        ):
            with pytest.raises(RuntimeError):
                build_extraction_index(doc_json, "run-extra", "doc-extra", store=store)

    def test_matching_length_does_not_raise(self):
        """Correct 1:1 count between pending and embeddings must NOT raise."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "Exactly one.")]
        doc_json = _make_doc_json(texts, [], [])
        store = _make_mock_store()

        # Return exactly 1 embedding for 1 pending element
        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            return_value=[[0.1] * 128],
        ):
            diag = build_extraction_index(doc_json, "run-ok", "doc-ok", store=store)

        assert diag.chunks_inserted == 1


# ---------------------------------------------------------------------------
# MED/LOW fix: strict=True delete propagates failures from build
# ---------------------------------------------------------------------------


class TestStrictDeleteBehavior:
    """_delete_by_run_id strict=True (used by build) propagates failures.
    strict=False (used by cleanup) keeps best-effort swallowing behavior.
    """

    def test_build_propagates_delete_failure(self):
        """When DELETE raises and strict=True (build path), the exception
        propagates out of build_extraction_index. C.4 catches and falls back."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "Some content.")]
        doc_json = _make_doc_json(texts, [], [])

        store = MagicMock()
        store._database = "test_db"
        store._client.command_sync.side_effect = RuntimeError("ArcadeDB offline")

        with pytest.raises(RuntimeError, match="ArcadeDB offline"):
            build_extraction_index(doc_json, "run-del-fail", "doc-df", store=store)

    def test_cleanup_swallows_delete_failure_unchanged(self):
        """cleanup_extraction_index must keep best-effort (strict=False)
        semantics after the strict refactor — no behavior change."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = MagicMock()
        store._database = "test_db"
        store._client.command_sync.side_effect = RuntimeError("ArcadeDB offline")

        # Must NOT raise
        result = cleanup_extraction_index("run-cleanup-fail", store=store)
        assert result == 0, f"Expected 0 on failure; got {result!r}"

    def test_build_delete_strict_raises_before_embed_called(self):
        """When strict DELETE fails, embed_texts should NOT be called
        (build must abort before embedding on DELETE failure)."""
        from app.services.extraction_chunk_index import build_extraction_index

        texts = [_make_text_elem(0, "Content.")]
        doc_json = _make_doc_json(texts, [], [])

        store = MagicMock()
        store._database = "test_db"
        store._client.command_sync.side_effect = RuntimeError("DB down")

        with patch(
            "app.services.extraction_chunk_index.embed_texts",
        ) as mock_embed:
            with pytest.raises(RuntimeError):
                build_extraction_index(doc_json, "run-abort", "doc-abort", store=store)

        mock_embed.assert_not_called()


# ---------------------------------------------------------------------------
# HIGH fix: recursive group traversal in _resolve_parent_section_heading
# Rev 15 review finding: SA-2 has 40/308 texts inside #/groups/N refs.
# ---------------------------------------------------------------------------


class TestRealDoclingGroupParentedTexts:
    """Rev 15 review HIGH finding: SA-2 has 40 of 308 texts inside groups.
    _resolve_parent_section_heading must recursively walk #/groups/N
    children to find heading context for these texts.
    """

    @pytest.fixture
    def fixture_doc(self):
        path = _FIXTURE_DIR / "with_groups.json"
        with open(path) as f:
            return json.load(f)

    def test_group_parented_text_resolves_to_preceding_section_header(self, fixture_doc):
        """Bullets under 'Equipment Specifications' (inside #/groups/1) must
        resolve to that heading, NOT empty string."""
        from app.services.extraction_chunk_index import _resolve_parent_section_heading

        # texts[4] = "RSNA-75 Fan Song radar" is inside groups/1, which is
        # preceded by texts[3] = "Equipment Specifications" (section_header).
        heading = _resolve_parent_section_heading("#/texts/4", fixture_doc)
        assert heading == "Equipment Specifications", (
            f"Expected 'Equipment Specifications'; got {heading!r}. The "
            f"group-traversal fix is incomplete."
        )

    def test_first_group_text_resolves_to_section_header_before_group(self, fixture_doc):
        """texts[1] is inside groups[0] which is preceded by texts[0] (Introduction).
        Heading should be 'Introduction'."""
        from app.services.extraction_chunk_index import _resolve_parent_section_heading

        heading = _resolve_parent_section_heading("#/texts/1", fixture_doc)
        assert heading == "Introduction"

    def test_section_header_state_persists_across_groups(self, fixture_doc):
        """After walking through groups[0], the section_header 'Introduction'
        is the running heading until the next section_header is encountered
        (texts[3] 'Equipment Specifications'). This is implicit: texts[1] and
        texts[2] (both in groups[0]) both resolve to 'Introduction'."""
        from app.services.extraction_chunk_index import _resolve_parent_section_heading

        assert _resolve_parent_section_heading("#/texts/2", fixture_doc) == "Introduction"
        assert _resolve_parent_section_heading("#/texts/5", fixture_doc) == "Equipment Specifications"

    def test_unknown_self_ref_returns_empty(self, fixture_doc):
        """Self_ref that doesn't exist in the doc returns empty string (no exception)."""
        from app.services.extraction_chunk_index import _resolve_parent_section_heading

        heading = _resolve_parent_section_heading("#/texts/99999", fixture_doc)
        assert heading == ""

    def test_recursion_depth_guard_handles_deep_nesting(self, fixture_doc):
        """Pathological case: group nested inside group inside group. The walker
        must not stack-overflow. Add max_depth guard."""
        # Build a synthetic doc with nested groups; verify no exception.
        nested_doc = {
            "body": {"children": [{"$ref": "#/groups/0"}]},
            "texts": [
                {"self_ref": "#/texts/0", "label": "section_header", "text": "Outer Heading"},
                {"self_ref": "#/texts/1", "label": "list_item", "text": "Deeply nested item"},
            ],
            "groups": [
                {"self_ref": "#/groups/0", "children": [{"$ref": "#/texts/0"}, {"$ref": "#/groups/1"}]},
                {"self_ref": "#/groups/1", "children": [{"$ref": "#/groups/2"}]},
                {"self_ref": "#/groups/2", "children": [{"$ref": "#/texts/1"}]},
            ],
        }
        from app.services.extraction_chunk_index import _resolve_parent_section_heading
        # Should resolve to 'Outer Heading' (set in groups[0] before nested groups).
        heading = _resolve_parent_section_heading("#/texts/1", nested_doc)
        assert heading == "Outer Heading"
