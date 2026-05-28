"""Tests for the worker-side port of docling-graph's _sanitize_docling_document.

Verifies the ported sanitizer behaves byte-identically to docling-graph's
original. Critical because the merged-chunk routing path (Phase 2) skips
docling-graph's sanitize step — the worker's index builder must clean
chunks upstream.
"""
from __future__ import annotations

import pytest

from app.services.web_cruft_sanitizer import (
    _contains_encoded_blob,
    _looks_like_nav_or_tracking,
    sanitize_docling_document,
)


class TestLooksLikeNavOrTracking:
    """Rule predicates — mirror of docling-graph's behavior."""

    def test_empty_or_whitespace_returns_false(self):
        assert _looks_like_nav_or_tracking("") is False
        assert _looks_like_nav_or_tracking("   \n  ") is False
        assert _looks_like_nav_or_tracking(None) is False  # type: ignore

    def test_real_prose_returns_false(self):
        assert _looks_like_nav_or_tracking(
            "The S-75 Dvina was deployed in 1957 by the Soviet Union."
        ) is False
        assert _looks_like_nav_or_tracking(
            "Maximum effective range: 18 miles. Maximum altitude: 82,000 feet."
        ) is False

    def test_rule1_ad_tracking_domain_drops(self):
        assert _looks_like_nav_or_tracking(
            "Visit https://googletagmanager.com/gtm.js?id=GTM-XYZ for tracking"
        ) is True
        assert _looks_like_nav_or_tracking(
            "[Click here](https://doubleclick.net/foo)"
        ) is True

    def test_rule2_pure_link_lines_drop(self):
        # Sidebar nav
        assert _looks_like_nav_or_tracking(
            "[Home](https://example.com/home)\n"
            "[About](https://example.com/about)\n"
            "[Contact](https://example.com/contact)"
        ) is True
        # Bare URL list
        assert _looks_like_nav_or_tracking(
            "https://example.com/page1\nhttps://example.com/page2"
        ) is True
        # Mixed with prose → kept
        assert _looks_like_nav_or_tracking(
            "[Click here](https://example.com) for more information"
        ) is False

    def test_rule3_base64_blob_drops(self):
        # Long base64 with padding
        blob = "A" + "abcdef" * 12 + "+" + "Z" * 30 + "==" + "9" * 5
        assert _looks_like_nav_or_tracking(blob) is True

    def test_caption_like_prose_kept(self):
        """Image-description prose should never match the rules."""
        assert _looks_like_nav_or_tracking(
            "A black and white photograph of a missile launcher in a clearing"
        ) is False


class TestContainsEncodedBlob:
    def test_short_tokens_excluded(self):
        # UUIDs, hex hashes, semvers stay below 64-char floor
        assert _contains_encoded_blob("abc-123 v1.2.3 abcdef0123456789") is False
        assert _contains_encoded_blob(
            "uuid: 550e8400-e29b-41d4-a716-446655440000"
        ) is False

    def test_long_base64_with_padding_detected(self):
        token = "X" * 30 + "abcDEF12+/9" * 10 + "=="
        assert _contains_encoded_blob(f"prefix {token} suffix") is True

    def test_long_percent_encoded_url_detected(self):
        # Long token with 6+ %XX triplets
        token = "https%3A%2F%2Fex.com%2Fpath%2Fwith%2Flots%2Fparams%3D%26x%3D%26y%3D" + "z" * 20
        assert len(token) >= 64
        assert _contains_encoded_blob(token) is True


class TestSanitizeDoclingDocument:
    def test_blanks_ad_tracking_element_in_place(self):
        doc = {
            "texts": [
                {"label": "text", "text": "Real prose content here.", "orig": "Real prose content here."},
                {"label": "text", "text": "Visit doubleclick.net/foo", "orig": "Visit doubleclick.net/foo"},
                {"label": "text", "text": "More real content.", "orig": "More real content."},
            ],
            "body": {"children": [
                {"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/texts/2"},
            ]},
        }
        stats: dict[str, int] = {}
        result = sanitize_docling_document(doc, stats)

        # Real content preserved
        assert result["texts"][0]["text"] == "Real prose content here."
        assert result["texts"][2]["text"] == "More real content."
        # Tracker element blanked but kept in place (so $refs stay valid)
        assert result["texts"][1]["text"] == ""
        assert result["texts"][1]["orig"] == ""
        # Element count unchanged — only content blanked
        assert len(result["texts"]) == 3
        # Refs in body.children still resolve
        assert result["body"]["children"][1]["$ref"] == "#/texts/1"
        # Stats
        assert stats["texts_in"] == 3
        assert stats["texts_dropped"] == 1

    def test_caption_protected(self):
        """label=='caption' must never be blanked, even if it matches the rules."""
        doc = {
            "texts": [
                {"label": "caption",
                 "text": "Image: doubleclick.net/foo (this would normally match Rule 1)",
                 "orig": "..."},
            ],
        }
        stats: dict[str, int] = {}
        result = sanitize_docling_document(doc, stats)
        assert "doubleclick.net" in result["texts"][0]["text"]
        assert stats["texts_dropped"] == 0

    def test_tables_untouched(self):
        """The sanitizer must NEVER walk doc['tables'][]."""
        doc = {
            "texts": [
                {"label": "text", "text": "doubleclick.net/foo", "orig": "doubleclick.net/foo"},
            ],
            "tables": [
                {
                    "table_id": "t1",
                    "data": {"grid": [[{"text": "cell value"}]]},
                },
            ],
        }
        result = sanitize_docling_document(doc, {})
        # Table untouched
        assert result["tables"][0]["data"]["grid"][0][0]["text"] == "cell value"
        # Text element blanked
        assert result["texts"][0]["text"] == ""

    def test_hyperlink_rewrite_protection(self):
        """When element has separate hyperlink annotation, sanitizer renders
        [text](hyperlink) for the rule check AND clears the hyperlink when
        blanking — so the chunker won't re-render the URL."""
        doc = {
            "texts": [
                # Visible text is innocent, but hyperlink is to a tracker
                {"label": "text", "text": "Read more",
                 "orig": "Read more",
                 "hyperlink": "https://doubleclick.net/click?id=123"},
            ],
        }
        stats: dict[str, int] = {}
        result = sanitize_docling_document(doc, stats)
        assert stats["texts_dropped"] == 1
        assert result["texts"][0]["text"] == ""
        assert result["texts"][0]["hyperlink"] is None

    def test_no_changes_returns_original(self):
        """When nothing matches, return the doc unchanged (no copy)."""
        doc = {
            "texts": [
                {"label": "text", "text": "Real content."},
                {"label": "text", "text": "More real content."},
            ],
        }
        result = sanitize_docling_document(doc, {})
        # Identity check — when no blanks, return same dict to avoid copy
        assert result is doc

    def test_empty_texts_array_safe(self):
        doc = {"texts": []}
        result = sanitize_docling_document(doc, {})
        assert result["texts"] == []

    def test_missing_texts_field_safe(self):
        doc: dict = {}
        result = sanitize_docling_document(doc, {})
        # No crash, no field added
        assert "texts" not in result or result.get("texts") == []
