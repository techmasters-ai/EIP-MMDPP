"""Unit tests for structure-aware chunking service.

Tests StructuredChunk dataclass, _approx_tokens, _split_text, and
structure_aware_chunk with various element combinations.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# StructuredChunk dataclass
# ---------------------------------------------------------------------------

class TestStructuredChunkDataclass:
    def test_defaults(self):
        from app.services.chunking import StructuredChunk
        sc = StructuredChunk(text="hello", chunk_index=0, modality="text")
        assert sc.text == "hello"
        assert sc.chunk_index == 0
        assert sc.modality == "text"
        assert sc.page_number is None
        assert sc.section_path is None
        assert sc.element_uids == []
        assert sc.heading_text is None

    def test_all_fields_set(self):
        from app.services.chunking import StructuredChunk
        sc = StructuredChunk(
            text="content",
            chunk_index=3,
            modality="table",
            page_number=5,
            section_path="Chapter 1 > Section 2",
            element_uids=["uid-1", "uid-2"],
            heading_text="My Heading",
        )
        assert sc.page_number == 5
        assert sc.section_path == "Chapter 1 > Section 2"
        assert len(sc.element_uids) == 2
        assert sc.heading_text == "My Heading"


# ---------------------------------------------------------------------------
# _approx_tokens
# ---------------------------------------------------------------------------

class TestApproxTokens:
    def test_empty_string_returns_zero(self):
        from app.services.chunking import _approx_tokens
        assert _approx_tokens("") == 0

    def test_known_word_count(self):
        from app.services.chunking import _approx_tokens
        # "hello world" = 2 words → int(2 * 1.3) = 2
        assert _approx_tokens("hello world") == 2

    def test_single_word(self):
        from app.services.chunking import _approx_tokens
        # 1 word → int(1 * 1.3) = 1
        assert _approx_tokens("hello") == 1

    def test_ten_words(self):
        from app.services.chunking import _approx_tokens
        text = " ".join(["word"] * 10)
        assert _approx_tokens(text) == 13  # int(10 * 1.3) = 13


# ---------------------------------------------------------------------------
# _split_text
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_short_text_returns_single_chunk(self):
        from app.services.chunking import _split_text
        result = _split_text("Short text.", max_tokens=100, overlap_tokens=10)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_paragraph_splitting(self):
        from app.services.chunking import _split_text
        # Build text with multiple paragraphs that exceed max_tokens
        paragraphs = ["Paragraph " + str(i) + " " + "word " * 30 for i in range(5)]
        text = "\n\n".join(paragraphs)
        result = _split_text(text, max_tokens=50, overlap_tokens=10)
        assert len(result) > 1

    def test_word_based_fallback(self):
        from app.services.chunking import _split_text
        # Single very long paragraph (no \n\n) → word-based split
        text = "word " * 200
        result = _split_text(text, max_tokens=50, overlap_tokens=10)
        assert len(result) > 1
        assert len(result) < 20
        for chunk in result:
            assert len(chunk) > 0

    def test_overlap_preserved_between_chunks(self):
        from app.services.chunking import _split_text
        # Multi-paragraph text that gets split
        paragraphs = ["Paragraph " + str(i) + " content here." for i in range(10)]
        text = "\n\n".join(paragraphs)
        result = _split_text(text, max_tokens=20, overlap_tokens=10)
        # With overlap, chunks should share some content
        if len(result) > 1:
            # The last paragraph of chunk 0 should appear in chunk 1
            last_para_chunk0 = result[0].split("\n\n")[-1]
            assert last_para_chunk0 in result[1]

    def test_zero_overlap_no_duplication(self):
        from app.services.chunking import _split_text
        text = "word " * 200
        result = _split_text(text, max_tokens=50, overlap_tokens=0)
        # With zero overlap, total words should equal original
        all_words = []
        for chunk in result:
            all_words.extend(chunk.split())
        # May not be exact due to rounding but should be close
        assert len(all_words) == 200


# ---------------------------------------------------------------------------
# structure_aware_chunk
# ---------------------------------------------------------------------------

class TestStructureAwareChunk:
    def _make_elem(self, **kwargs):
        """Build a minimal element dict."""
        defaults = {
            "element_type": "text",
            "content_text": "Some text content.",
            "page_number": 1,
            "section_path": None,
            "element_uid": "uid-0",
            "element_order": 0,
            "heading_level": None,
        }
        defaults.update(kwargs)
        return defaults

    def test_empty_elements_returns_empty(self):
        from app.services.chunking import structure_aware_chunk
        assert structure_aware_chunk([]) == []

    def test_single_text_element_returns_one_chunk(self):
        from app.services.chunking import structure_aware_chunk
        elems = [self._make_elem()]
        result = structure_aware_chunk(elems)
        assert len(result) == 1
        assert result[0].modality == "text"
        assert result[0].chunk_index == 0

    def test_heading_flushes_buffer_and_starts_new_section(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_type="text", content_text="Before heading.", element_order=0),
            self._make_elem(element_type="heading", content_text="Section Title", element_order=1),
            self._make_elem(element_type="text", content_text="After heading.", element_order=2),
        ]
        result = structure_aware_chunk(elems, max_chunk_tokens=9999)
        # "Before heading." should be its own chunk, heading+after merged
        assert len(result) == 2
        assert "Before heading." in result[0].text
        assert "Section Title" in result[1].text
        assert "After heading." in result[1].text

    def test_table_always_own_chunk(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_type="text", content_text="Intro text.", element_order=0),
            self._make_elem(element_type="table", content_text="| A | B |\n|---|---|\n| 1 | 2 |", element_order=1),
            self._make_elem(element_type="text", content_text="After table.", element_order=2),
        ]
        result = structure_aware_chunk(elems, max_chunk_tokens=9999)
        modalities = [c.modality for c in result]
        assert "table" in modalities
        # Table should be standalone
        table_chunk = [c for c in result if c.modality == "table"][0]
        assert "| A | B |" in table_chunk.text

    def test_equation_always_own_chunk(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_type="equation", content_text="E = mc^2", element_order=0),
        ]
        result = structure_aware_chunk(elems)
        assert len(result) == 1
        assert result[0].modality == "equation"

    def test_image_with_content_gets_own_chunk(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_type="image", content_text="Figure 1: Schematic.", element_order=0),
        ]
        result = structure_aware_chunk(elems)
        assert len(result) == 1
        assert result[0].modality == "image"

    def test_image_without_content_skipped(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_type="image", content_text="", element_order=0),
        ]
        result = structure_aware_chunk(elems)
        assert len(result) == 0

    def test_text_accumulation_respects_max_tokens(self):
        from app.services.chunking import structure_aware_chunk
        # Create many small text elements that together exceed max_tokens
        elems = [
            self._make_elem(content_text=f"Sentence number {i}.", element_order=i)
            for i in range(20)
        ]
        result = structure_aware_chunk(elems, max_chunk_tokens=20)
        # Should produce multiple chunks
        assert len(result) > 1

    def test_element_order_sorting(self):
        from app.services.chunking import structure_aware_chunk
        # Pass elements out of order
        elems = [
            self._make_elem(content_text="Second.", element_order=2, element_uid="uid-2"),
            self._make_elem(content_text="First.", element_order=1, element_uid="uid-1"),
            self._make_elem(content_text="Third.", element_order=3, element_uid="uid-3"),
        ]
        result = structure_aware_chunk(elems, max_chunk_tokens=9999)
        assert len(result) == 1
        assert "First." in result[0].text
        # First should appear before Second in the combined text
        text = result[0].text
        assert text.index("First.") < text.index("Second.")

    def test_section_path_preserved(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(section_path="Chapter 1 > Intro"),
        ]
        result = structure_aware_chunk(elems)
        assert result[0].section_path == "Chapter 1 > Intro"

    def test_page_number_preserved(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(page_number=42),
        ]
        result = structure_aware_chunk(elems)
        assert result[0].page_number == 42

    def test_element_uids_tracked(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_uid="uid-a", element_order=0),
            self._make_elem(element_uid="uid-b", element_order=1),
        ]
        result = structure_aware_chunk(elems, max_chunk_tokens=9999)
        assert "uid-a" in result[0].element_uids
        assert "uid-b" in result[0].element_uids

    def test_heading_text_propagates_to_table_chunk(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_type="heading", content_text="My Section", element_order=0),
            self._make_elem(element_type="table", content_text="| data |", element_order=1),
        ]
        result = structure_aware_chunk(elems, max_chunk_tokens=9999)
        table_chunks = [c for c in result if c.modality == "table"]
        assert len(table_chunks) == 1
        assert table_chunks[0].heading_text == "My Section"

    def test_mixed_elements_correct_ordering(self):
        from app.services.chunking import structure_aware_chunk
        elems = [
            self._make_elem(element_type="heading", content_text="Title", element_order=0),
            self._make_elem(element_type="text", content_text="Intro paragraph.", element_order=1),
            self._make_elem(element_type="table", content_text="Table data", element_order=2),
            self._make_elem(element_type="text", content_text="After table.", element_order=3),
            self._make_elem(element_type="equation", content_text="x=1", element_order=4),
        ]
        result = structure_aware_chunk(elems, max_chunk_tokens=9999)
        modalities = [c.modality for c in result]
        assert modalities == ["text", "table", "text", "equation"]

    def test_oversized_text_element_split(self):
        from app.services.chunking import structure_aware_chunk
        # A single element with very long text
        long_text = ("word " * 500).strip()
        elems = [self._make_elem(content_text=long_text)]
        result = structure_aware_chunk(elems, max_chunk_tokens=50, overlap_tokens=10)
        assert len(result) > 1


# ---------------------------------------------------------------------------
# split_description_sections
# ---------------------------------------------------------------------------

from app.services.chunking import split_description_sections


class TestSplitDescriptionSections:
    def test_markdown_headers(self):
        desc = "# Executive Summary\nThis is a missile.\n\n## Technical Details\nLength: 10m.\n\n## Markings\nNone visible."
        sections = split_description_sections(desc)
        assert len(sections) == 3
        assert sections[0].startswith("# Executive Summary")
        assert "missile" in sections[0]
        assert sections[1].startswith("## Technical Details")

    def test_numbered_headers_parenthesis(self):
        desc = "1) Classification: Category=photo\n\n2) Why This Category: visible cues\n\n3) General Description: missile image"
        sections = split_description_sections(desc)
        assert len(sections) == 3
        assert sections[0].startswith("1)")

    def test_numbered_headers_dot(self):
        desc = "1. Executive Summary\nMissile system.\n\n2. Source Context\nPDF summary.\n\n3. Full Scene\nOutdoor."
        sections = split_description_sections(desc)
        assert len(sections) == 3

    def test_bold_headers(self):
        desc = "**Executive Summary:** This is a radar.\n\n**Technical Details:** Frequency band X."
        sections = split_description_sections(desc)
        assert len(sections) == 2
        assert "radar" in sections[0]

    def test_fallback_paragraph_split(self):
        desc = "First paragraph about the missile.\n\nSecond paragraph about the radar.\n\nThird paragraph."
        sections = split_description_sections(desc)
        assert len(sections) == 3

    def test_skip_short_sections(self):
        desc = "# Summary\nGood content here about the system.\n\n## Empty\n\n\n## Details\nMore content here."
        sections = split_description_sections(desc)
        assert all(len(s) >= 20 for s in sections)

    def test_preamble_before_first_header(self):
        desc = "This is introductory text before any header.\n\n# First Section\nContent here."
        sections = split_description_sections(desc)
        assert len(sections) == 2
        assert "introductory" in sections[0]

    def test_single_section_no_headers(self):
        desc = "This is a single block of text describing a missile system with sufficient length to pass minimum."
        sections = split_description_sections(desc)
        assert len(sections) == 1

    def test_empty_description(self):
        sections = split_description_sections("")
        assert sections == []

    def test_whitespace_only(self):
        sections = split_description_sections("   \n\n   ")
        assert sections == []
