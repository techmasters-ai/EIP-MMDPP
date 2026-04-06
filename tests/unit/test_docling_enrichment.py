"""Tests for DoclingDocument enrichment helpers."""

import pytest

pytestmark = pytest.mark.unit


class TestParseSelfRef:
    def test_texts(self):
        from app.services.docling_enrichment import _parse_self_ref
        assert _parse_self_ref("#/texts/0") == ("texts", 0)
        assert _parse_self_ref("#/texts/42") == ("texts", 42)

    def test_pictures(self):
        from app.services.docling_enrichment import _parse_self_ref
        assert _parse_self_ref("#/pictures/1") == ("pictures", 1)

    def test_tables(self):
        from app.services.docling_enrichment import _parse_self_ref
        assert _parse_self_ref("#/tables/3") == ("tables", 3)

    def test_invalid(self):
        from app.services.docling_enrichment import _parse_self_ref
        assert _parse_self_ref("") == ("", -1)
        assert _parse_self_ref("invalid") == ("", -1)
        assert _parse_self_ref("#/") == ("", -1)
        assert _parse_self_ref(None) == ("", -1)


class TestBuildEnrichedCopyForChunking:
    def test_applies_text_translation(self):
        from app.services.docling_enrichment import _build_enriched_copy_for_chunking
        doc = {
            "texts": [{"text": "\u041f\u0440\u0438\u0432\u0435\u0442", "self_ref": "#/texts/0"}],
            "_enrichments": {
                "translations": {
                    "#/texts/0": {"original_text": "\u041f\u0440\u0438\u0432\u0435\u0442", "translated_text": "Hello"}
                }
            },
        }
        result = _build_enriched_copy_for_chunking(doc)
        assert result["texts"][0]["text"] == "Hello"
        # Original dict unchanged
        assert doc["texts"][0]["text"] == "\u041f\u0440\u0438\u0432\u0435\u0442"

    def test_replaces_table_data(self):
        from app.services.docling_enrichment import _build_enriched_copy_for_chunking
        doc = {
            "tables": [{"data": {"table_cells": [{"text": "\u0420\u0430\u043a\u0435\u0442\u0430"}], "num_rows": 1, "num_cols": 1}, "self_ref": "#/tables/0"}],
            "_enrichments": {
                "translations": {
                    "#/tables/0": {"original_text": "\u0420\u0430\u043a\u0435\u0442\u0430", "translated_text": "Missile"}
                }
            },
        }
        result = _build_enriched_copy_for_chunking(doc)
        cells = result["tables"][0]["data"]["table_cells"]
        assert len(cells) == 1
        assert cells[0]["text"] == "Missile"

    def test_no_translations_passthrough(self):
        from app.services.docling_enrichment import _build_enriched_copy_for_chunking
        doc = {"texts": [{"text": "Hello"}], "_enrichments": {"translations": {}}}
        result = _build_enriched_copy_for_chunking(doc)
        assert result["texts"][0]["text"] == "Hello"

    def test_no_enrichments_passthrough(self):
        from app.services.docling_enrichment import _build_enriched_copy_for_chunking
        doc = {"texts": [{"text": "Hello"}]}
        result = _build_enriched_copy_for_chunking(doc)
        assert result["texts"][0]["text"] == "Hello"

    def test_does_not_include_context_node(self):
        from app.services.docling_enrichment import _build_enriched_copy_for_chunking
        doc = {
            "texts": [{"text": "Original"}],
            "_enrichments": {
                "translations": {},
                "context": {"summary": "A document about radars"},
            },
        }
        result = _build_enriched_copy_for_chunking(doc)
        # Should NOT have a context node — only chunking copy
        assert len(result.get("texts", [])) == 1


class TestRegenerateTranslatedMarkdown:
    def test_includes_picture_descriptions(self):
        from app.services.docling_enrichment import _regenerate_translated_markdown
        doc = {
            "texts": [],
            "pictures": [
                {"meta": {"description": {"text": "A radar system image"}}}
            ],
            "_enrichments": {"translations": {}},
        }
        md = _regenerate_translated_markdown(doc)
        assert "A radar system image" in md
        assert "Image Descriptions" in md

    def test_no_pictures_no_appendix(self):
        from app.services.docling_enrichment import _regenerate_translated_markdown
        doc = {"texts": [], "_enrichments": {"translations": {}}}
        md = _regenerate_translated_markdown(doc)
        assert "Image Descriptions" not in md
