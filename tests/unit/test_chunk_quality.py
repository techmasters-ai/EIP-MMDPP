"""Unit tests for app/services/chunk_quality.py — shared v2 filter primitives.

These tests exercise the primitives in isolation, with no DoclingDocument
or ArcadeDB context. They are the contract for what every extraction-pass
consumer of the v2 filter relies on.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestConstants:
    def test_min_chunk_text_chars(self):
        from app.services.chunk_quality import MIN_CHUNK_TEXT_CHARS
        assert MIN_CHUNK_TEXT_CHARS == 20

    def test_min_residual_chars(self):
        from app.services.chunk_quality import MIN_RESIDUAL_CHARS
        assert MIN_RESIDUAL_CHARS == 40

    def test_web_chrome_phrases_is_frozenset_of_lowercase_strings(self):
        from app.services.chunk_quality import WEB_CHROME_PHRASES
        assert isinstance(WEB_CHROME_PHRASES, frozenset)
        for p in WEB_CHROME_PHRASES:
            assert isinstance(p, str)
            assert p == p.lower(), f"phrase {p!r} must be lowercase"


class TestNormalizeForDedup:
    def test_lowercases(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("HELLO World") == "hello world"

    def test_collapses_whitespace(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("a   b\n\nc\td") == "a b c d"

    def test_strips_edges(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("  hi  ") == "hi"

    def test_empty_returns_empty(self):
        from app.services.chunk_quality import normalize_for_dedup
        assert normalize_for_dedup("") == ""
        assert normalize_for_dedup("   \n\n  ") == ""


class TestIsChromeLine:
    def test_chrome_phrase_exact_match(self):
        from app.services.chunk_quality import is_chrome_line
        assert is_chrome_line("Audio Coming Soon") is True
        assert is_chrome_line("SUBSCRIBE NOW") is True
        assert is_chrome_line("audio coming soon") is True

    def test_empty_or_whitespace_is_chrome(self):
        from app.services.chunk_quality import is_chrome_line
        # Empty / whitespace lines are treated as chrome so they get
        # pulled along with adjacent real chrome lines during strip.
        assert is_chrome_line("") is True
        assert is_chrome_line("   ") is True
        assert is_chrome_line("\n") is True

    def test_substring_match_does_not_count(self):
        from app.services.chunk_quality import is_chrome_line
        # "subscribe now" appears inside a real paragraph — NOT chrome.
        assert is_chrome_line("Please subscribe now to receive updates") is False

    def test_real_text_is_not_chrome(self):
        from app.services.chunk_quality import is_chrome_line
        assert is_chrome_line("S-75 Dvina is a Soviet SAM system") is False


class TestStripChromeLines:
    def test_no_chrome_returns_unchanged(self):
        from app.services.chunk_quality import strip_chrome_lines
        stripped, l, t = strip_chrome_lines("Real article body paragraph.")
        assert stripped == "Real article body paragraph."
        assert l == 0
        assert t == 0

    def test_strips_leading_chrome_lines(self):
        from app.services.chunk_quality import strip_chrome_lines
        text = "Audio Coming Soon\n\nReal article body here."
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == "Real article body here."
        assert l == 2  # chrome line + blank line
        assert t == 0

    def test_strips_trailing_chrome_lines(self):
        from app.services.chunk_quality import strip_chrome_lines
        text = "Real article body here.\n\nSubscribe Now"
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == "Real article body here."
        assert l == 0
        assert t == 2

    def test_preserves_middle_chrome(self):
        from app.services.chunk_quality import strip_chrome_lines
        # Middle chrome lines stay — they sit between real content
        text = "Real one.\nSubscribe Now\nReal two."
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == text
        assert l == 0
        assert t == 0

    def test_all_chrome_returns_empty(self):
        from app.services.chunk_quality import strip_chrome_lines
        text = "Audio Coming Soon\nSponsored\nSubscribe Now"
        stripped, l, t = strip_chrome_lines(text)
        assert stripped == ""
        assert l == 3
        assert t == 0


class TestClassifyChunk:
    """The unified decision predicate used by every consumer."""

    def test_too_short_is_dropped(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        decision = classify_chunk("a b c d", seen)
        assert decision.keep is False
        assert decision.reason == "short"

    def test_real_text_is_kept_unchanged(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        decision = classify_chunk(
            "The S-75 Dvina is a Soviet surface-to-air missile system used widely.", seen,
        )
        assert decision.keep is True
        assert decision.reason == "kept"
        assert decision.stripped_text is None

    def test_chrome_prefix_is_stripped_and_kept(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        decision = classify_chunk(
            "Audio Coming Soon\n\nThe S-75 Dvina is a real radar described here in detail.", seen,
        )
        assert decision.keep is True
        assert decision.reason == "stripped"
        assert decision.stripped_text is not None
        assert "Audio Coming Soon" not in decision.stripped_text
        assert "S-75 Dvina" in decision.stripped_text

    def test_residue_too_short_after_strip_is_dropped(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        # 20-39 chars after strip → drop
        decision = classify_chunk("Audio Coming Soon\n\nshort residue body.", seen)
        assert decision.keep is False
        assert decision.reason == "after_strip"

    def test_exact_duplicate_after_strip_is_dropped(self):
        from app.services.chunk_quality import classify_chunk
        seen: set[str] = set()
        d1 = classify_chunk(
            "Audio Coming Soon\n\nIdentical body paragraph for dedup testing here.", seen,
        )
        assert d1.keep is True
        d2 = classify_chunk(
            "SUBSCRIBE NOW\n\nIdentical body paragraph for dedup testing here.", seen,
        )
        assert d2.keep is False
        assert d2.reason == "dedup"

    def test_classify_chunk_mutates_seen(self):
        """When a chunk is kept, its normalized text is added to `seen`."""
        from app.services.chunk_quality import classify_chunk, normalize_for_dedup
        seen: set[str] = set()
        text = "A real article paragraph with substantive content for indexing."
        decision = classify_chunk(text, seen)
        assert decision.keep
        # The key in `seen` is the stripped+normalized form
        assert normalize_for_dedup(text) in seen
