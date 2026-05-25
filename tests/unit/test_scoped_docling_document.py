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
    """texts[] is not removed or reindexed; retained elements may have parent rewritten.

    Note: C.7 fix — retained elements get parent rewritten to #/body for
    DoclingDocument hierarchy consistency. Length, positions, and content of
    the array entries themselves are preserved.
    """
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

    # No-removal / no-reindex invariant: every position retained, with the same
    # self_ref and content as the input doc.
    assert len(result["texts"]) == 3, "texts[] must retain all 3 entries"
    for i in range(3):
        assert result["texts"][i]["self_ref"] == doc["texts"][i]["self_ref"]
        assert result["texts"][i]["text"] == doc["texts"][i]["text"]
    # Unselected elements pass through unchanged (no parent injection on #/texts/2).
    assert "parent" not in result["texts"][2], (
        "Unselected element must not have a parent rewritten onto it"
    )
    # And the input doc itself must not be mutated.
    assert "parent" not in doc["texts"][1], (
        "Input doc_json must not be mutated by apply_chunk_scope"
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


# ---------------------------------------------------------------------------
# CRITICAL #1: Parent reachability validation (rev 7 H1 + M7)
# ---------------------------------------------------------------------------


def test_apply_chunk_scope_warns_on_dangling_parent(caplog):
    """When a retained element's parent is unreachable from rewritten body.children,
    apply_chunk_scope must emit a logger.warning (NOT raise — narrowing still proceeds).
    """
    import logging

    # #/texts/1 has parent "#/groups/3" which won't be reachable because:
    # - body.children only references #/texts/0 directly (no #/groups/3)
    # - groups[3] is not defined
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "Section", "label": "section_header"},
            {
                "self_ref": "#/texts/1",
                "text": "Para A",
                "label": "paragraph",
                "parent": {"cref": "#/groups/3"},
            },
        ],
        body_children=[_ref("#/texts/0"), _ref("#/texts/1")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/1"]}

    with caplog.at_level(logging.WARNING, logger="app.services.scoped_docling_document"):
        result = apply_chunk_scope(doc, chunk_scope)

    # Must NOT raise — narrowing still proceeds.
    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/1" in child_refs, "Scoping must still proceed despite dangling parent"

    # Must emit a warning about the unreachable parent.
    warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("#/groups/3" in m or "dangling parent" in m.lower() or "unreachable" in m.lower()
               for m in warning_msgs), (
        f"Expected warning about unreachable parent '#/groups/3'; got: {warning_msgs}"
    )


def test_apply_chunk_scope_no_warn_when_parent_reachable(caplog):
    """When a retained element's parent IS reachable, no warning is emitted."""
    import logging

    # #/texts/1 has parent "#/body" — always reachable.
    doc = _make_doc(
        texts=[
            {"self_ref": "#/texts/0", "text": "Section", "label": "section_header"},
            {
                "self_ref": "#/texts/1",
                "text": "Para A",
                "label": "paragraph",
                "parent": {"cref": "#/body"},
            },
        ],
        body_children=[_ref("#/texts/0"), _ref("#/texts/1")],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/1"]}

    with caplog.at_level(logging.WARNING, logger="app.services.scoped_docling_document"):
        result = apply_chunk_scope(doc, chunk_scope)

    parent_warn_msgs = [
        r.message for r in caplog.records
        if r.levelno == logging.WARNING and "unreachable" in r.message.lower()
    ]
    assert parent_warn_msgs == [], (
        f"No reachability warning expected for #/body parent; got: {parent_warn_msgs}"
    )


def test_empty_self_refs_preserves_arrays():
    """test_empty_self_refs array preservation — no accidental mutation in early-return path."""
    doc = _make_doc(
        texts=[{"self_ref": "#/texts/0", "text": "Hello", "label": "paragraph"}],
        body_children=[_ref("#/texts/0")],
    )
    original_texts = list(doc["texts"])  # copy
    chunk_scope = {"mode": "selected_refs", "self_refs": []}
    result = apply_chunk_scope(doc, chunk_scope)

    assert result["body"]["children"] == [], "Empty self_refs must produce body.children=[]"
    assert len(result["texts"]) == 1
    # MINOR #3: lock in the no-mutation invariant on the early-return path.
    assert result["texts"] == original_texts, (
        "texts[] must equal the original in the early-return (empty self_refs) path"
    )


# ---------------------------------------------------------------------------
# IMPORTANT #1 (C.6 BLOCKER) — Reachability walker false-positives on
# group-parented elements (rev 19 round-2 review fix)
# ---------------------------------------------------------------------------


def test_apply_chunk_scope_no_warn_for_group_parented_element_in_scope(caplog):
    """Group-parented element IS in scope — NO spurious warning must be emitted.

    SA-2 has 40/308 texts with parent=#/groups/0 or #/groups/1.  Before the
    fix, _collect_reachable was seeded with unique_scoped_crefs (never
    contains #/groups/N), so groups were never marked reachable.  Every
    group-parented element triggered a spurious WARNING.

    After the fix, the walker traverses the ORIGINAL body.children and marks
    groups reachable iff any (transitively) nested descendant is in scope.
    This topology:
      body.children → #/groups/0
      groups[0].children → #/texts/0, #/texts/1
      texts[0].parent = #/groups/0  (selected)
      texts[1].parent = #/groups/0  (not selected)
    → #/groups/0 IS reachable because texts[0] is in scope.
    → ZERO warnings must be emitted.
    """
    import logging

    doc = _make_doc(
        texts=[
            {
                "self_ref": "#/texts/0",
                "text": "X",
                "label": "list_item",
                "parent": {"cref": "#/groups/0"},
            },
            {
                "self_ref": "#/texts/1",
                "text": "Y",
                "label": "list_item",
                "parent": {"cref": "#/groups/0"},
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}],
            }
        ],
        body_children=[{"$ref": "#/groups/0"}],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}

    with caplog.at_level(logging.WARNING, logger="app.services.scoped_docling_document"):
        result = apply_chunk_scope(doc, chunk_scope)

    # Scoping must still include the selected ref.
    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/0" in child_refs, "Selected ref must be in scoped body.children"

    # ZERO spurious warnings — group IS reachable via its selected descendant.
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    reachability_warns = [m for m in warn_msgs if "unreachable" in m.lower() or "mis-navigate" in m.lower()]
    assert reachability_warns == [], (
        f"Expected ZERO reachability warnings for group-parented element whose group "
        f"contains a selected descendant; got: {reachability_warns}"
    )


def test_apply_chunk_scope_no_warn_for_nested_group_parented_element_in_scope(caplog):
    """Nested groups (body→group0→group1→text) — no warning when text is in scope.

    Verifies that the ancestor-marking logic handles multi-level group nesting.
    """
    import logging

    doc = _make_doc(
        texts=[
            {
                "self_ref": "#/texts/0",
                "text": "Z",
                "label": "paragraph",
                "parent": {"cref": "#/groups/1"},
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "children": [{"$ref": "#/groups/1"}],
            },
            {
                "self_ref": "#/groups/1",
                "children": [{"$ref": "#/texts/0"}],
            },
        ],
        body_children=[{"$ref": "#/groups/0"}],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}

    with caplog.at_level(logging.WARNING, logger="app.services.scoped_docling_document"):
        apply_chunk_scope(doc, chunk_scope)

    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    reachability_warns = [m for m in warn_msgs if "unreachable" in m.lower() or "mis-navigate" in m.lower()]
    assert reachability_warns == [], (
        f"Expected ZERO reachability warnings for nested-group-parented element in scope; "
        f"got: {reachability_warns}"
    )


def test_apply_chunk_scope_warns_for_truly_dangling_group_parent(caplog):
    """TRULY dangling group parent (group not in body topology at all) still warns.

    This confirms the fix didn't accidentally suppress legitimate warnings.
    The group #/groups/3 is NOT referenced anywhere in body.children and
    does NOT appear in groups[], so it is truly dangling.
    """
    import logging

    # Re-use the existing dangling-parent fixture but with a group that is
    # genuinely absent from the original body topology.
    doc = _make_doc(
        texts=[
            {
                "self_ref": "#/texts/0",
                "text": "Para A",
                "label": "paragraph",
                "parent": {"cref": "#/groups/3"},  # #/groups/3 not in body or groups[]
            },
        ],
        groups=[
            {"self_ref": "#/groups/0", "children": []},  # groups[0] exists but not #/groups/3
        ],
        body_children=[{"$ref": "#/texts/0"}],  # body points directly at the text, not via group
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}

    with caplog.at_level(logging.WARNING, logger="app.services.scoped_docling_document"):
        result = apply_chunk_scope(doc, chunk_scope)

    # Scoping still proceeds.
    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/0" in child_refs, "Scoping must proceed despite dangling parent"

    # Must still warn about truly-dangling parent.
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("#/groups/3" in m or "unreachable" in m.lower() for m in warn_msgs), (
        f"Expected warning about truly-dangling '#/groups/3'; got: {warn_msgs}"
    )


def test_apply_chunk_scope_warns_for_orphan_group_present_in_groups_but_absent_from_body(caplog):
    """Rev 20 review Suggestion #2: a group exists in groups[] (so it's a valid
    docling object) BUT is never referenced from body.children. Selected element's
    parent points at this orphan group.

    Expected: the orphan group is unreachable from body (the walker only descends
    into groups it finds in body.children), so the warning fires. This is a real
    coverage gap that the rev-19 walker rewrite handles correctly but had no
    explicit test.
    """
    import logging

    doc = _make_doc(
        texts=[
            {
                "self_ref": "#/texts/0",
                "text": "Para A",
                "label": "paragraph",
                "parent": {"cref": "#/groups/5"},  # orphan group: exists but not in body
            },
        ],
        groups=[
            {"self_ref": "#/groups/0", "children": []},
            {"self_ref": "#/groups/1", "children": []},
            {"self_ref": "#/groups/2", "children": []},
            {"self_ref": "#/groups/3", "children": []},
            {"self_ref": "#/groups/4", "children": []},
            {"self_ref": "#/groups/5", "children": [{"$ref": "#/texts/0"}]},  # exists but...
        ],
        body_children=[{"$ref": "#/texts/0"}],  # ...body points at texts[0] directly, not via #/groups/5
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}

    with caplog.at_level(logging.WARNING, logger="app.services.scoped_docling_document"):
        result = apply_chunk_scope(doc, chunk_scope)

    # Scoping still proceeds.
    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/0" in child_refs

    # The orphan group's parent ref is unreachable because the walker never
    # descends into it (it isn't in body.children). Warning must fire.
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("#/groups/5" in m or "unreachable" in m.lower() for m in warn_msgs), (
        f"Expected warning about orphan group '#/groups/5' (in groups[] but not in body); "
        f"got: {warn_msgs}"
    )


def test_apply_chunk_scope_negative_group_index_in_parent_does_not_silently_resolve(caplog):
    """Rev 20 review Minor #2: a malformed parent ref '#/groups/-1' must NOT
    silently resolve to the last group via Python's negative-indexing.
    The walker rejects negative indices in _group_idx_from_ref, so the group
    is treated as unresolvable → warning fires."""
    import logging

    doc = _make_doc(
        texts=[
            {
                "self_ref": "#/texts/0",
                "text": "Para A",
                "label": "paragraph",
                "parent": {"cref": "#/groups/-1"},  # malformed: negative index
            },
        ],
        groups=[
            {"self_ref": "#/groups/0", "children": [{"$ref": "#/texts/0"}]},
        ],
        body_children=[{"$ref": "#/texts/0"}],
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}

    with caplog.at_level(logging.WARNING, logger="app.services.scoped_docling_document"):
        result = apply_chunk_scope(doc, chunk_scope)

    # Scoping still proceeds.
    child_refs = [c.get("$ref") or c.get("cref") or c.get("$cref") for c in result["body"]["children"]]
    assert "#/texts/0" in child_refs

    # Negative index must NOT silently resolve to groups[-1]; warning fires.
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("#/groups/-1" in m or "unreachable" in m.lower() for m in warn_msgs), (
        f"Expected warning about malformed negative-index group ref; got: {warn_msgs}"
    )


# ---------------------------------------------------------------------------
# C.7 BLOCKER — Scoped doc must validate as a DoclingDocument
#
# `apply_chunk_scope` flattens body.children to a list of (selected + heading)
# refs. Pre-fix, retained text/table/picture elements kept their original
# `parent` (e.g. {"cref": "#/groups/N"}). DoclingDocument's Pydantic model
# rejected the result with:
#   Value error, Document hierarchy is inconsistent.
#   #/body has child #/texts/N with parent #/groups/M
#
# Fix: rewrite retained elements' `parent` to {"cref": "#/body"} and drop them
# from any group's `children` list so no orphan group double-claims them.
# ---------------------------------------------------------------------------


def _docling_doc_envelope(**parts):
    """Minimal DoclingDocument-shaped envelope filled in with the provided parts."""
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "scoped-test",
        "body": parts.get(
            "body",
            {
                "self_ref": "#/body",
                "parent": None,
                "children": [],
                "content_layer": "body",
                "name": "_root_",
                "label": "unspecified",
            },
        ),
        "furniture": {
            "self_ref": "#/furniture",
            "parent": None,
            "children": [],
            "content_layer": "furniture",
            "name": "_root_",
            "label": "unspecified",
        },
        "texts": parts.get("texts", []),
        "tables": parts.get("tables", []),
        "pictures": parts.get("pictures", []),
        "groups": parts.get("groups", []),
        "key_value_items": [],
        "form_items": [],
        "pages": {},
    }


def test_scoped_doc_with_group_parented_text_validates_as_docling_document():
    """A scoped doc with a group-parented selected ref must pass DoclingDocument validation.

    This is the C.7 production blocker reproduction: a list-item text whose
    original parent is #/groups/0 is selected. After apply_chunk_scope, the
    returned doc must be loadable by docling-core's DoclingDocument without
    a "Document hierarchy is inconsistent" error.
    """
    from docling_core.types.doc import DoclingDocument

    doc = _docling_doc_envelope(
        texts=[
            {
                "self_ref": "#/texts/0",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "list_item",
                "prov": [],
                "orig": "Item A",
                "text": "Item A",
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": "#/texts/0"}],
                "content_layer": "body",
                "name": "list-0",
                "label": "list",
            },
        ],
        body={
            "self_ref": "#/body",
            "parent": None,
            "children": [{"cref": "#/groups/0"}],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
    )
    chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}

    result = apply_chunk_scope(doc, chunk_scope)

    # Must validate — raises pydantic.ValidationError on failure.
    DoclingDocument(**result)


def test_scoped_doc_rewrites_retained_non_list_item_parent_to_body():
    """Retained NON-list_item elements with a group parent must have parent rewritten to #/body.

    Note: list_items are handled differently (see test_scoped_doc_keeps_list_group_parent_for_list_item)
    because docling-core's ``validate_misplaced_list_items`` runs on construction and re-groups
    any ListItem whose parent isn't a ListGroup — the re-grouping delete+re-add cycle produces
    duplicate self_refs in texts[] when items unreachable from body.children aren't reindexed.
    For non-list_item labels (paragraph, text, code, ...), no such validator exists, so reparent
    to #/body remains the simple/correct behavior.
    """
    doc = _docling_doc_envelope(
        texts=[
            {
                "self_ref": "#/texts/0",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "paragraph",
                "prov": [],
                "orig": "X",
                "text": "X",
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": "#/texts/0"}],
                "content_layer": "body",
                "name": "wrapper",
                "label": "unspecified",
            },
        ],
        body={
            "self_ref": "#/body",
            "parent": None,
            "children": [{"cref": "#/groups/0"}],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
    )

    result = apply_chunk_scope(doc, {"mode": "selected_refs", "self_refs": ["#/texts/0"]})

    parent = result["texts"][0].get("parent")
    assert isinstance(parent, dict)
    parent_ref = parent.get("cref") or parent.get("$ref") or parent.get("$cref")
    assert parent_ref == "#/body", (
        f"Retained non-list_item element's parent must be rewritten to #/body; got {parent_ref!r}"
    )
    # And the original doc must not be mutated.
    original_parent_ref = doc["texts"][0]["parent"].get("cref")
    assert original_parent_ref == "#/groups/0", (
        "Input doc_json must not be mutated by apply_chunk_scope"
    )


def test_scoped_doc_clears_reparented_non_list_item_refs_from_group_children():
    """Groups must not claim a non-list_item text whose parent was rewritten to #/body."""
    doc = _docling_doc_envelope(
        texts=[
            {
                "self_ref": "#/texts/0",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "paragraph",
                "prov": [],
                "orig": "X",
                "text": "X",
            },
            {
                "self_ref": "#/texts/1",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "paragraph",
                "prov": [],
                "orig": "Y",
                "text": "Y",
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": "#/texts/0"}, {"cref": "#/texts/1"}],
                "content_layer": "body",
                "name": "wrapper",
                "label": "unspecified",
            },
        ],
        body={
            "self_ref": "#/body",
            "parent": None,
            "children": [{"cref": "#/groups/0"}],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
    )

    result = apply_chunk_scope(doc, {"mode": "selected_refs", "self_refs": ["#/texts/0"]})

    grp_children = result["groups"][0].get("children") or []
    crefs = [c.get("cref") or c.get("$ref") or c.get("$cref") for c in grp_children]
    assert "#/texts/0" not in crefs, (
        "Reparented text must be removed from its former group's children"
    )
    # Unselected sibling stays in the group's children list (group still owns it).
    assert "#/texts/1" in crefs, "Unselected sibling must remain in group.children"
    # Original input untouched.
    assert {c.get("cref") for c in doc["groups"][0]["children"]} == {"#/texts/0", "#/texts/1"}


# ---------------------------------------------------------------------------
# C.7r — Duplicate ref bug: ListItems reparented to #/body trigger docling's
# ``validate_misplaced_list_items`` model-validator, whose delete+re-add cycle
# produces DUPLICATE self_refs in texts[] when items unreachable from
# body.children (e.g. picture-children texts) aren't covered by the
# breadth-first reindex.
#
# Fix: for retained list_items whose original parent was a ListGroup
# (group.label in {"list", "ordered_list"}), KEEP the group parent and
# include the group in body.children with the retained list_item in its
# children. Non-list_items in the same group still reparent to #/body.
# ---------------------------------------------------------------------------


def test_scoped_doc_keeps_list_group_parent_for_list_item():
    """A retained list_item with a ListGroup parent must keep that parent.

    Rationale: docling-core's ``validate_misplaced_list_items`` model_validator
    fires when a ListItem's parent isn't a ListGroup. Its delete+re-add cycle
    mis-indexes texts that are unreachable from body.children, producing
    duplicate self_refs that fail ``_validate_unique_refs`` later when the
    HybridChunker wraps the doc in ``ChunkingDocSerializer``.
    """
    doc = _docling_doc_envelope(
        texts=[
            {
                "self_ref": "#/texts/0",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "list_item",
                "prov": [],
                "orig": "Item A",
                "text": "Item A",
                "enumerated": False,
                "marker": "-",
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": "#/texts/0"}],
                "content_layer": "body",
                "name": "list-0",
                "label": "list",
            },
        ],
        body={
            "self_ref": "#/body",
            "parent": None,
            "children": [{"cref": "#/groups/0"}],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
    )

    result = apply_chunk_scope(doc, {"mode": "selected_refs", "self_refs": ["#/texts/0"]})

    parent = result["texts"][0].get("parent")
    assert isinstance(parent, dict)
    parent_ref = parent.get("cref") or parent.get("$ref") or parent.get("$cref")
    assert parent_ref == "#/groups/0", (
        f"Retained list_item must keep its ListGroup parent; got {parent_ref!r}"
    )

    # Group stays in body.children (and the list_item does NOT appear directly there).
    body_refs = [
        c.get("cref") or c.get("$ref") or c.get("$cref")
        for c in result["body"]["children"]
    ]
    assert "#/groups/0" in body_refs, (
        "Retained ListGroup must appear in body.children when it owns a retained list_item"
    )
    assert "#/texts/0" not in body_refs, (
        "Retained list_item must NOT appear directly in body.children — its group represents it"
    )

    # Group's children still includes the retained list_item.
    grp_children = result["groups"][0].get("children") or []
    crefs = [c.get("cref") or c.get("$ref") or c.get("$cref") for c in grp_children]
    assert "#/texts/0" in crefs, (
        "Retained list_item must remain in its ListGroup's children"
    )


def test_scoped_doc_with_list_item_in_list_group_validates_after_chunker_wrap():
    """End-to-end repro of the C.7r ``Duplicate ref`` bug.

    BEFORE the fix:
      - apply_chunk_scope reparents the list_item to #/body.
      - DoclingDocument(**result) constructs OK (initial _validate_unique_refs passes).
      - validate_misplaced_list_items runs on construction, creates a new ListGroup,
        deletes the misplaced list_items, and re-adds new ones. The breadth-first
        reindex only updates self_refs of items REACHABLE from body.children;
        picture-children texts (parent=#/pictures/N, not in body.children) keep
        their old self_refs.
      - When ChunkingDocSerializer(doc=dl_doc) wraps the instance,
        _validate_unique_refs re-fires and detects the duplicate.

    AFTER the fix:
      - List_items keep their ListGroup parent, validate_misplaced_list_items
        finds no misplaced items, no delete/re-add cycle runs, no duplicates.
    """
    from docling_core.types.doc import DoclingDocument
    from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer

    # 5 consecutive list_items in one ListGroup; selected via chunk_scope.
    # Plus a separate text element with parent=#/pictures/0 to mimic the
    # picture-children-reindex gap that surfaces the bug in the SA-2 doc.
    texts = []
    for i in range(5):
        texts.append({
            "self_ref": f"#/texts/{i}",
            "parent": {"cref": "#/groups/0"},
            "children": [],
            "content_layer": "body",
            "label": "list_item",
            "prov": [],
            "orig": f"Item {i}",
            "text": f"Item {i}",
            "enumerated": False,
            "marker": "-",
        })
    # Picture-child text — parent is the picture, NOT in body.children directly.
    # Index 5 picks any number > N-1 so it shifts when items 0-4 are "deleted".
    texts.append({
        "self_ref": "#/texts/5",
        "parent": {"cref": "#/pictures/0"},
        "children": [],
        "content_layer": "body",
        "label": "text",
        "prov": [],
        "orig": "caption",
        "text": "caption",
    })

    doc = _docling_doc_envelope(
        texts=texts,
        pictures=[
            {
                "self_ref": "#/pictures/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": "#/texts/5"}],
                "captions": [{"cref": "#/texts/5"}],
                "content_layer": "body",
                "label": "picture",
                "prov": [],
                "footnotes": [],
                "image": None,
                "annotations": [],
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": f"#/texts/{i}"} for i in range(5)],
                "content_layer": "body",
                "name": "list-0",
                "label": "list",
            },
        ],
        body={
            "self_ref": "#/body",
            "parent": None,
            "children": [{"cref": "#/groups/0"}, {"cref": "#/pictures/0"}],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
    )

    # Sanity: the full unscoped doc itself must validate first.
    DoclingDocument(**doc)

    # Scope: select all list_items (the bug surfaces with multiple consecutive
    # misplaced list_items + an unreachable picture-child text).
    chunk_scope = {"mode": "selected_refs", "self_refs": [f"#/texts/{i}" for i in range(5)]}

    result = apply_chunk_scope(doc, chunk_scope)

    # Construct DoclingDocument — both _validate_unique_refs AND
    # validate_misplaced_list_items run here.
    dl_doc = DoclingDocument(**result)

    # Wrap in ChunkingDocSerializer — _validate_unique_refs re-fires on the
    # mutated instance. THIS is the assertion that protects against the bug.
    ChunkingDocSerializer(doc=dl_doc)


def test_scoped_doc_non_list_item_in_list_group_still_reparents_to_body():
    """A retained NON-list_item child of a ListGroup must still reparent to #/body.

    Only list_items get the special-case treatment (because they trigger
    validate_misplaced_list_items). Other labels reparent as before.
    """
    doc = _docling_doc_envelope(
        texts=[
            {
                "self_ref": "#/texts/0",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "list_item",
                "prov": [],
                "orig": "Item A",
                "text": "Item A",
                "enumerated": False,
                "marker": "-",
            },
            {
                "self_ref": "#/texts/1",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "paragraph",
                "prov": [],
                "orig": "Paragraph in same group",
                "text": "Paragraph in same group",
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": "#/texts/0"}, {"cref": "#/texts/1"}],
                "content_layer": "body",
                "name": "list-0",
                "label": "list",
            },
        ],
        body={
            "self_ref": "#/body",
            "parent": None,
            "children": [{"cref": "#/groups/0"}],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
    )

    result = apply_chunk_scope(
        doc, {"mode": "selected_refs", "self_refs": ["#/texts/0", "#/texts/1"]}
    )

    # The list_item keeps its group parent.
    p0 = result["texts"][0]["parent"]
    p0_ref = p0.get("cref") or p0.get("$ref") or p0.get("$cref")
    assert p0_ref == "#/groups/0", f"list_item parent must stay as group; got {p0_ref!r}"

    # The paragraph reparents to #/body.
    p1 = result["texts"][1]["parent"]
    p1_ref = p1.get("cref") or p1.get("$ref") or p1.get("$cref")
    assert p1_ref == "#/body", f"paragraph parent must rewrite to #/body; got {p1_ref!r}"

    # Group's children: list_item stays (its parent is still group); paragraph removed.
    grp_children = result["groups"][0].get("children") or []
    crefs = [c.get("cref") or c.get("$ref") or c.get("$cref") for c in grp_children]
    assert "#/texts/0" in crefs, "list_item still in group's children"
    assert "#/texts/1" not in crefs, "paragraph was reparented; must drop from group's children"


def test_scoped_doc_list_item_not_in_list_group_still_reparents_to_body():
    """A list_item whose original parent is NOT a ListGroup gets reparented as before.

    The misplaced-list-items rescue logic only triggers for items whose parent
    isn't a ListGroup, so this case never had the duplicate-ref problem and the
    behavior should match the non-list_item path.
    """
    doc = _docling_doc_envelope(
        texts=[
            {
                "self_ref": "#/texts/0",
                "parent": {"cref": "#/groups/0"},
                "children": [],
                "content_layer": "body",
                "label": "list_item",
                "prov": [],
                "orig": "X",
                "text": "X",
                "enumerated": False,
                "marker": "-",
            },
        ],
        groups=[
            {
                "self_ref": "#/groups/0",
                "parent": {"cref": "#/body"},
                "children": [{"cref": "#/texts/0"}],
                "content_layer": "body",
                "name": "wrapper",
                "label": "unspecified",  # NOT a ListGroup
            },
        ],
        body={
            "self_ref": "#/body",
            "parent": None,
            "children": [{"cref": "#/groups/0"}],
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
    )

    result = apply_chunk_scope(doc, {"mode": "selected_refs", "self_refs": ["#/texts/0"]})

    p = result["texts"][0]["parent"]
    p_ref = p.get("cref") or p.get("$ref") or p.get("$cref")
    assert p_ref == "#/body", (
        f"list_item with non-ListGroup parent must still reparent to #/body; got {p_ref!r}"
    )
