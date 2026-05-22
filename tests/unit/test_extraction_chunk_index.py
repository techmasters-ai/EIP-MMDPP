"""Unit tests for extraction_chunk_index.py — VR Phase C.2.

All tests are mocked (no ArcadeDB, no Ollama). Tests exercise the
rendering helpers, diagnostics, and the build/cleanup public API.

TDD discipline: all tests were written BEFORE the implementation module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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
    """
    table: dict[str, Any] = {
        "self_ref": f"#/tables/{idx}",
        "prov": [{"page_no": page_no}],
        "data": {},
    }
    if caption_text is not None:
        # Caption stored as $ref to a texts element (standard docling shape)
        table["captions"] = [{"$ref": f"#/texts/900{idx}"}]
    # Encode cells as flat table_cells list with start_row/col offsets
    flat_cells: list[dict] = []
    for row_idx, row in enumerate(cells):
        for col_idx, text in enumerate(row):
            flat_cells.append({
                "text": text,
                "start_row_offset_idx": row_idx,
                "end_row_offset_idx": row_idx,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx,
            })
    table["data"]["table_cells"] = flat_cells
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
    return {
        "texts": all_texts,
        "tables": tables,
        "pictures": pictures,
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
