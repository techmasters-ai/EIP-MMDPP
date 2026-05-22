"""Unit tests for app.services.scoped_docling_document.apply_chunk_scope.

VR Phase C.4 — rev 10 M7.

TDD RED→GREEN:
  - These tests were written before the implementation and should pass after it.
  - Run with: pytest tests/unit/test_scoped_docling_document.py -v

Invariants under test:
  1. Preserve array positions (texts[], tables[], pictures[] never modified).
  2. body.children contains only selected refs + their preceding headings.
  3. Unknown self_ref raises ValueError.
  4. Empty self_refs returns body.children=[].
  5. mode != "selected_refs" raises ValueError.
"""
import pytest

from app.services.scoped_docling_document import apply_chunk_scope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(
    texts=None,
    tables=None,
    pictures=None,
    groups=None,
    body_children=None,
):
    """Build a minimal synthetic DoclingDocument JSON for testing."""
    doc = {}
    if texts is not None:
        doc["texts"] = texts
    if tables is not None:
        doc["tables"] = tables
    if pictures is not None:
        doc["pictures"] = pictures
    if groups is not None:
        doc["groups"] = groups
    if body_children is not None:
        doc["body"] = {"children": body_children}
    return doc


def _ref(self_ref: str) -> dict:
    """Synthetic ref dict using $ref key (as in test fixtures)."""
    return {"$ref": self_ref}


# ---------------------------------------------------------------------------
# Test 1: Preserve array positions
# ---------------------------------------------------------------------------


def test_preserve_array_positions_texts():
    """texts[] is NOT modified — no elements removed or reindexed."""
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "Heading", "label": "section_header"},
            {"self_ref": "#/texts/1", "text": "Content A", "label": "paragraph"},
            {"self_ref": "#/texts/2", "text": "Content B", "label": "paragraph"},
        ],
        body_children=[_ref("#/texts/0"), _ref("#/texts/1"), _ref("#/texts/2")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/1"]}
    result = apply_chunk_scope(doc, chunk_scope)

    # texts[] in the original doc must be unmodified
    assert result["texts"] is doc["texts"] or result["texts"] == doc["texts"]
    assert len(result["texts"]) == 3, "texts[] must retain all 3 entries"
    assert result["texts"][2]["text"] == "Content B", (
        "texts[2] must remain unchanged even though it's not in scope"
    )


def test_preserve_array_positions_tables():
    """tables[] is NOT modified."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "Para", "label": "paragraph"}],
        tables=[
            {"self_ref": "#/tables/0", "data": {}},
            {"self_ref": "#/tables/1", "data": {}},
        ],
        body_children=[_ref("#/texts/0"), _ref("#/tables/0"), _ref("#/tables/1")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/tables/0"]}
    result = apply_chunk_scope(doc, chunk_scope)

    assert len(result["tables"]) == 2, "tables[] must retain both entries"
    assert result["tables"][1]["self_ref"] == "#/tables/1", (
        "tables[1] must be preserved even though not in scope"
    )


def test_preserve_array_positions_pictures():
    """pictures[] is NOT modified."""
    doc = _make_doc(
        pictures=[
            {"self_ref": "#/pictures/0", "captions": []},
            {"self_ref": "#/pictures/1", "captions": []},
        ],
        body_children=[_ref("#/pictures/0"), _ref("#/pictures/1")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/pictures/1"]}
    result = apply_chunk_scope(doc, chunk_scope)

    assert len(result["pictures"]) == 2, "pictures[] must retain both entries"


def test_original_doc_not_mutated():
    """apply_chunk_scope must not mutate the input doc_json."""
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "A", "label": "paragraph"},
            {"self_ref": "#/texts/1", "text": "B", "label": "paragraph"},
        ],
        body_children=[_ref("#/texts/0"), _ref("#/texts/1")],
    )
    original_children = list(doc["body"]["children"])
    original_text_count = len(doc["texts"])

    apply_chunk_scope(doc, {"mode": "selected_refs", "self_refs": ["#/texts/0"]})

    assert doc["body"]["children"] == original_children, "Original body must not be mutated"
    assert len(doc["texts"]) == original_text_count, "texts[] must not be mutated"


# ---------------------------------------------------------------------------
# Test 2: body.children contains only selected refs + preceding headings
# ---------------------------------------------------------------------------


def test_body_children_includes_only_selected_refs_and_headings():
    """Scoped body.children: selected ref + nearest preceding heading."""
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "Section 1", "label": "section_header"},
            {"self_ref": "#/texts/1", "text": "Para A", "label": "paragraph"},
            {"self_ref": "#/texts/2", "text": "Para B (not selected)", "label": "paragraph"},
        ],
        body_children=[_ref("#/texts/0"), _ref("#/texts/1"), _ref("#/texts/2")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/1"]}
    result = apply_chunk_scope(doc, chunk_scope)

    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/0" in child_refs, "Preceding heading must be included for context"
    assert "#/texts/1" in child_refs, "Selected ref must be included"
    assert "#/texts/2" not in child_refs, "Non-selected non-heading must be excluded"


def test_body_children_excludes_unselected_content():
    """Non-selected content paragraphs must not appear in scoped body.children."""
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "Section", "label": "section_header"},
            {"self_ref": "#/texts/1", "text": "Selected", "label": "paragraph"},
            {"self_ref": "#/texts/2", "text": "Not selected", "label": "paragraph"},
            {"self_ref": "#/texts/3", "text": "Also selected", "label": "paragraph"},
        ],
        body_children=[
            _ref("#/texts/0"), _ref("#/texts/1"),
            _ref("#/texts/2"), _ref("#/texts/3"),
        ],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/1", "#/texts/3"]}
    result = apply_chunk_scope(doc, chunk_scope)

    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/2" not in child_refs, "texts/2 not in scope — must be absent"
    assert "#/texts/1" in child_refs
    assert "#/texts/3" in child_refs


def test_heading_not_duplicated_when_followed_by_two_selected():
    """A heading preceding two selected elements must appear exactly once."""
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "Heading", "label": "section_header"},
            {"self_ref": "#/texts/1", "text": "Para 1", "label": "paragraph"},
            {"self_ref": "#/texts/2", "text": "Para 2", "label": "paragraph"},
        ],
        body_children=[_ref("#/texts/0"), _ref("#/texts/1"), _ref("#/texts/2")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/1", "#/texts/2"]}
    result = apply_chunk_scope(doc, chunk_scope)

    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert child_refs.count("#/texts/0") == 1, "Heading must appear exactly once"


def test_table_selected():
    """Tables appear in body.children when their self_ref is in scope."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "T heading", "label": "section_header"}],
        tables=[{"self_ref": "#/tables/0", "data": {}}],
        body_children=[_ref("#/texts/0"), _ref("#/tables/0")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/tables/0"]}
    result = apply_chunk_scope(doc, chunk_scope)

    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/tables/0" in child_refs


# ---------------------------------------------------------------------------
# Test 3: Unknown self_ref raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_self_ref_raises_value_error():
    """A self_ref that doesn't exist in the doc should raise ValueError."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "Hello", "label": "paragraph"}],
        body_children=[_ref("#/texts/0")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/99"]}

    with pytest.raises(ValueError, match="not resolvable"):
        apply_chunk_scope(doc, chunk_scope)


def test_unknown_table_ref_raises_value_error():
    """A self_ref pointing to a non-existent table raises ValueError."""
    doc = _make_doc(
        tables=[{"self_ref": "#/tables/0", "data": {}}],
        body_children=[_ref("#/tables/0")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/tables/5"]}

    with pytest.raises(ValueError, match="not resolvable"):
        apply_chunk_scope(doc, chunk_scope)


# ---------------------------------------------------------------------------
# Test 4: Empty self_refs → body.children=[]
# ---------------------------------------------------------------------------


def test_empty_self_refs_returns_empty_body_children():
    """Empty self_refs list → scoped doc has empty body.children."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "Hello", "label": "paragraph"}],
        body_children=[_ref("#/texts/0")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": []}
    result = apply_chunk_scope(doc, chunk_scope)

    assert result["body"]["children"] == [], (
        "Empty self_refs must produce body.children=[]"
    )
    # arrays must still be present
    assert len(result["texts"]) == 1


# ---------------------------------------------------------------------------
# Test 5: mode != "selected_refs" raises ValueError
# ---------------------------------------------------------------------------


def test_mode_full_raises_value_error():
    """mode='full' is not a valid input for apply_chunk_scope — raises ValueError."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "Hello", "label": "paragraph"}],
        body_children=[_ref("#/texts/0")],
    )
    chunk_scope = {"mode": "full", "self_refs": []}

    with pytest.raises(ValueError, match="mode.*selected_refs"):
        apply_chunk_scope(doc, chunk_scope)


def test_mode_would_skip_raises_value_error():
    """mode='would_skip' is not a valid input — raises ValueError."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "Hello", "label": "paragraph"}],
        body_children=[_ref("#/texts/0")],
    )
    with pytest.raises(ValueError, match="mode.*selected_refs"):
        apply_chunk_scope(doc, {"mode": "would_skip", "self_refs": []})


def test_mode_none_raises_value_error():
    """mode=None raises ValueError."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "Hello", "label": "paragraph"}],
        body_children=[_ref("#/texts/0")],
    )
    with pytest.raises(ValueError, match="mode.*selected_refs"):
        apply_chunk_scope(doc, {"mode": None, "self_refs": ["#/texts/0"]})


# ---------------------------------------------------------------------------
# Additional: document order preserved in output
# ---------------------------------------------------------------------------


def test_body_children_order_matches_document_order():
    """The order of refs in output body.children matches document traversal order."""
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "First", "label": "paragraph"},
            {"self_ref": "#/texts/1", "text": "Second", "label": "paragraph"},
        ],
        tables=[{"self_ref": "#/tables/0", "data": {}}],
        body_children=[_ref("#/texts/0"), _ref("#/tables/0"), _ref("#/texts/1")],
    )
    # Select both a text and a table — they should appear in document order
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0", "#/tables/0"]}
    result = apply_chunk_scope(doc, chunk_scope)

    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    # #/texts/0 precedes #/tables/0 in body.children → must be preserved
    idx_0 = child_refs.index("#/texts/0")
    idx_t = child_refs.index("#/tables/0")
    assert idx_0 < idx_t, "Document order must be preserved in scoped body.children"


def test_heading_not_included_when_only_follows_selected():
    """A heading that comes AFTER all selected refs should NOT be included."""
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "Para", "label": "paragraph"},
            {"self_ref": "#/texts/1", "text": "Trailing Heading", "label": "section_header"},
        ],
        body_children=[_ref("#/texts/0"), _ref("#/texts/1")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}
    result = apply_chunk_scope(doc, chunk_scope)

    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/1" not in child_refs, (
        "A heading that follows (but doesn't precede) a selected ref must not be included"
    )
