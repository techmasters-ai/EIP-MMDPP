"""Unit tests for filter_docling_document — the worker-side v2 quality filter
applied to DoclingDocument JSON so ALL extraction passes (identity, field_group,
system_links) see a noise-filtered doc."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _make_text(idx: int, text: str, label: str = "text") -> dict:
    return {
        "self_ref": f"#/texts/{idx}",
        "text": text,
        "orig": text,
        "label": label,
        "prov": [{"page_no": 1}],
    }


def _make_doc(texts: list[dict]) -> dict:
    return {
        "texts": texts,
        "tables": [],
        "pictures": [],
        "body": {"children": [{"cref": t["self_ref"]} for t in texts]},
    }


class TestBasicShape:
    def test_returns_doc_and_diagnostics(self):
        from app.services.scoped_docling_document import filter_docling_document
        doc = _make_doc([_make_text(0, "Real radar article paragraph with substantive content here.")])
        filtered, diag = filter_docling_document(doc)
        assert isinstance(filtered, dict)
        assert hasattr(diag, "blanked_short")
        assert hasattr(diag, "blanked_dedup")
        assert hasattr(diag, "blanked_after_strip")
        assert hasattr(diag, "stripped_in_place")

    def test_preserves_texts_array_length(self):
        """Filter must blank-in-place, NOT remove entries. Array indices and
        $refs from body.children / pictures.children / tables.children depend
        on positional stability."""
        from app.services.scoped_docling_document import filter_docling_document
        texts = [
            _make_text(0, "Real radar article paragraph one with substantive content."),
            _make_text(1, "™"),  # short - will be blanked
            _make_text(2, "Real radar article paragraph two with substantive content."),
        ]
        doc = _make_doc(texts)
        filtered, _ = filter_docling_document(doc)
        assert len(filtered["texts"]) == 3, "array length must be preserved"

    def test_preserves_self_refs_on_all_entries_including_blanked(self):
        from app.services.scoped_docling_document import filter_docling_document
        texts = [_make_text(0, "Real content here for indexing purposes."), _make_text(1, "™")]
        doc = _make_doc(texts)
        filtered, _ = filter_docling_document(doc)
        for i, t in enumerate(filtered["texts"]):
            assert t["self_ref"] == f"#/texts/{i}"


class TestBlanking:
    def test_short_chunk_text_orig_become_empty(self):
        from app.services.scoped_docling_document import filter_docling_document
        doc = _make_doc([_make_text(0, "™")])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == ""
        assert filtered["texts"][0]["orig"] == ""
        assert diag.blanked_short == 1

    def test_short_chunk_clears_hyperlink_when_present(self):
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "Log in")
        elem["hyperlink"] = "https://tracker.example.com/click?x=1"
        doc = _make_doc([elem])
        filtered, _ = filter_docling_document(doc)
        assert filtered["texts"][0]["hyperlink"] is None

    def test_duplicate_after_strip_is_blanked_keeping_first(self):
        from app.services.scoped_docling_document import filter_docling_document
        body = "Identical body paragraph for dedup testing here with extra words."
        texts = [
            _make_text(0, f"Audio Coming Soon\n\n{body}"),
            _make_text(1, f"SUBSCRIBE NOW\n\n{body}"),  # dup after strip
        ]
        doc = _make_doc(texts)
        filtered, diag = filter_docling_document(doc)
        # First one kept (and stripped). Second one blanked.
        assert filtered["texts"][0]["text"]  # non-empty
        assert filtered["texts"][1]["text"] == ""
        assert diag.blanked_dedup == 1

    def test_residue_too_short_after_strip_is_blanked(self):
        from app.services.scoped_docling_document import filter_docling_document
        texts = [_make_text(0, "Audio Coming Soon\n\nShort residue body.")]
        doc = _make_doc(texts)
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == ""
        assert diag.blanked_after_strip == 1


class TestStripInPlace:
    def test_chrome_prefix_text_orig_overridden_with_stripped(self):
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(
            0,
            "Audio Coming Soon\n\nThe S-75 Dvina radar system is documented in this real article passage.",
        )
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        kept = filtered["texts"][0]
        assert "Audio Coming Soon" not in kept["text"]
        assert "Audio Coming Soon" not in kept["orig"]
        assert "S-75 Dvina" in kept["text"]
        assert diag.stripped_in_place == 1

    def test_kept_chunk_with_no_chrome_is_left_alone(self):
        from app.services.scoped_docling_document import filter_docling_document
        text = "The S-75 Dvina radar operates in the C-band frequency range here."
        elem = _make_text(0, text)
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        # Unchanged
        assert filtered["texts"][0]["text"] == text
        assert filtered["texts"][0]["orig"] == text
        assert diag.stripped_in_place == 0


class TestIdempotency:
    def test_running_filter_twice_yields_same_result(self):
        """The filter must be a no-op when run on already-filtered output."""
        from app.services.scoped_docling_document import filter_docling_document
        texts = [
            _make_text(0, "Real radar article paragraph with substantive content here."),
            _make_text(1, "™"),
            _make_text(2, "Audio Coming Soon\n\nThe S-75 radar is documented in this real article passage."),
            _make_text(3, "Real radar article paragraph with substantive content here."),  # dup of 0
        ]
        doc = _make_doc(texts)
        once, diag1 = filter_docling_document(doc)
        twice, diag2 = filter_docling_document(once)
        # texts arrays identical
        for i in range(len(once["texts"])):
            assert once["texts"][i] == twice["texts"][i]
        # Second filter pass produces zero new blanks/strips
        assert diag2.blanked_short == 0
        assert diag2.blanked_dedup == 0
        assert diag2.blanked_after_strip == 0
        assert diag2.stripped_in_place == 0


class TestBodyChildrenAndRefIntegrity:
    def test_body_children_refs_still_resolve(self):
        from app.services.scoped_docling_document import filter_docling_document
        texts = [
            _make_text(0, "Real radar paragraph with substantive content here."),
            _make_text(1, "Audio Coming Soon\nSponsored"),  # all-chrome → blank
            _make_text(2, "Another real radar paragraph with substantive content here."),
        ]
        doc = _make_doc(texts)
        filtered, _ = filter_docling_document(doc)
        # body.children unchanged structurally
        assert filtered["body"]["children"] == doc["body"]["children"]
        # All crefs still resolve to a text entry
        for child in filtered["body"]["children"]:
            ref = child.get("cref")
            idx = int(ref.rsplit("/", 1)[1])
            assert filtered["texts"][idx]["self_ref"] == ref


class TestTextChunksAreNotConsulted:
    """The filter must read text from the texts[] array only — never from any
    external TextChunk / postgres / ArcadeDB source."""

    def test_filter_makes_no_io_calls(self):
        # If filter_docling_document tries to import graph_store / postgres,
        # this test fails because no patches are set up.
        from app.services.scoped_docling_document import filter_docling_document
        doc = _make_doc([_make_text(0, "Real radar article content here for testing purposes.")])
        filtered, _ = filter_docling_document(doc)
        # If we got here without exceptions, no I/O was attempted.
        assert isinstance(filtered, dict)


class TestCaptionProtection:
    """label="caption" entries are protected from blanking + dedup. docling-graph's
    own sanitizer (main.py:482) uses the same protection — match its semantics."""

    def test_short_caption_is_not_blanked(self):
        from app.services.scoped_docling_document import filter_docling_document
        # 8 chars — would normally be blanked as "short"
        elem = _make_text(0, "Fig. 1.", label="caption")
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "Fig. 1."  # untouched
        assert diag.blanked_short == 0

    def test_duplicate_captions_both_kept(self):
        """Two real captions with identical text — both must be preserved."""
        from app.services.scoped_docling_document import filter_docling_document
        c1 = _make_text(0, "Figure: SA-2 launcher schematic.", label="caption")
        c2 = _make_text(1, "Figure: SA-2 launcher schematic.", label="caption")
        doc = _make_doc([c1, c2])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "Figure: SA-2 launcher schematic."
        assert filtered["texts"][1]["text"] == "Figure: SA-2 launcher schematic."
        assert diag.blanked_dedup == 0


class TestHeadingLabelProtection:
    """label values "section_header" / "section-header" / "title" are protected
    from blanking — the chunker uses them as parent-heading context for
    adjacent text chunks. Mirrors _HEADING_LABELS in extraction_chunk_index.py."""

    def test_short_section_header_is_not_blanked(self):
        """A 10-char section heading like 'S-75 DVINA' would normally be
        blanked as short (< 20 chars). With label protection it survives so
        downstream chunks can prefix it as ## S-75 DVINA."""
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "S-75 DVINA", label="section_header")
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "S-75 DVINA"
        assert diag.blanked_short == 0
        assert diag.protected_labels == 1

    def test_hyphenated_section_header_variant_is_also_protected(self):
        """Docling has emitted both 'section_header' and 'section-header' —
        both spellings must be protected to match _HEADING_LABELS in
        extraction_chunk_index.py."""
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "Variants", label="section-header")
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "Variants"
        assert diag.protected_labels == 1

    def test_title_label_is_protected(self):
        """label='title' is the doc-level title and must survive even though
        it's typically short."""
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "SA-2", label="title")
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "SA-2"
        assert diag.protected_labels == 1

    def test_repeated_section_headers_both_kept(self):
        """Two real 'Variants' section headers in different parts of the doc
        — dedup must NOT collapse them away."""
        from app.services.scoped_docling_document import filter_docling_document
        h1 = _make_text(0, "Variants discussion section heading here.", label="section_header")
        h2 = _make_text(1, "Variants discussion section heading here.", label="section_header")
        doc = _make_doc([h1, h2])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == "Variants discussion section heading here."
        assert filtered["texts"][1]["text"] == "Variants discussion section heading here."
        assert diag.blanked_dedup == 0
        assert diag.protected_labels == 2

    def test_page_header_label_is_NOT_protected(self):
        """page_header is webpage-export chrome (date stamps, breadcrumbs)
        and must NOT be in the protected set — it should still be blanked
        when short. Only true heading labels are protected."""
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "10/6/25, 8:33 PM", label="page_header")
        doc = _make_doc([elem])
        filtered, diag = filter_docling_document(doc)
        assert filtered["texts"][0]["text"] == ""  # blanked
        assert diag.blanked_short == 1
        assert diag.protected_labels == 0


class TestDefensiveEdgeCases:
    """Malformed docs must not crash the filter. The worker wraps the call in
    try/except but the function itself should also fail-safe for the most common
    shapes."""

    def test_missing_texts_key(self):
        from app.services.scoped_docling_document import filter_docling_document
        filtered, diag = filter_docling_document({"body": {"children": []}})
        assert diag.texts_in == 0

    def test_texts_is_empty_list(self):
        from app.services.scoped_docling_document import filter_docling_document
        filtered, diag = filter_docling_document({"texts": []})
        assert diag.texts_in == 0

    def test_text_element_is_not_a_dict(self):
        from app.services.scoped_docling_document import filter_docling_document
        doc = {"texts": ["not-a-dict", _make_text(0, "Real radar article content here for indexing purposes.")]}
        filtered, diag = filter_docling_document(doc)
        # Non-dict skipped; real dict processed
        assert filtered["texts"][0] == "not-a-dict"  # untouched
        assert filtered["texts"][1]["text"] == "Real radar article content here for indexing purposes."

    def test_text_field_is_none(self):
        """None text/orig must be skipped without raising AND without modifying
        the entry. The filter's None-guard is a skip path, not a blanking path."""
        from app.services.scoped_docling_document import filter_docling_document
        elem = _make_text(0, "")
        elem["text"] = None  # docling has been observed to emit None
        elem["orig"] = None
        doc = _make_doc([elem])
        # Must not raise on None
        filtered, diag = filter_docling_document(doc)
        assert isinstance(filtered, dict)
        # The None-text entry is skipped, not blanked — text/orig stay None
        assert filtered["texts"][0]["text"] is None
        assert filtered["texts"][0]["orig"] is None
        # No counters incremented for this entry
        assert diag.blanked_short == 0
        assert diag.blanked_dedup == 0
        assert diag.blanked_after_strip == 0
        assert diag.stripped_in_place == 0
