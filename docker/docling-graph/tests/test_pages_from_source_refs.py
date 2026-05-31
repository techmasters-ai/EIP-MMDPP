"""Tests for ``ManyToOneStrategy._pages_from_source_refs`` (patch 0005).

The helper is the page-resolution FALLBACK on the pre-built-chunk delta
route: when the worker doesn't supply per-chunk ``page_numbers``, it
resolves each chunk ``source_ref`` against the DoclingDocument and reads
``item.prov[].page_no`` so synthesized provenance still carries a real page
(hard data-lineage requirement).

Regression guard for the 2026-05-30 review finding: the original patch
resolved refs via ``getattr(document, "lookup", None)`` → ``lookup(ref)``,
but ``DoclingDocument`` has NO ``lookup`` method in the pinned docling-core
(verified: ``hasattr(DoclingDocument, "lookup")`` is False). That made the
fallback dead code returning ``[]`` for every chunk. The fix uses
``RefItem(cref=ref).resolve(document)`` — the real API (``RefItem.resolve``
exists) — mirroring ``document_processor._evidence_units_for_chunk``.

The helper lives in the gitignored docling-graph clone
(``docker/docling-graph/repo``) and only exists after patch 0005 is applied
at image-build time. This test adds the clone to ``sys.path`` and asserts
real behavior whenever the patched helper is importable (in-container, or
host after the patch is applied); it skips with a clear rebuild note when
run against the un-patched clean clone on the host.

All fixtures use generic text — never document-specific equipment names.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("docling_core", reason="docling-core not importable in this env")

# Make the gitignored docling-graph clone importable as `docling_graph`.
_REPO = Path(__file__).resolve().parent.parent / "repo"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _make_doc_with_page(page_no: int):
    """Build a minimal REAL DoclingDocument carrying one text item whose
    provenance points at ``page_no``. Returns (document, self_ref)."""
    from docling_core.types.doc import BoundingBox, DocItemLabel
    from docling_core.types.doc.document import DoclingDocument, ProvenanceItem

    doc = DoclingDocument(name="test")
    item = doc.add_text(label=DocItemLabel.TEXT, text="SA-class system specification")
    item.prov.append(
        ProvenanceItem(
            page_no=page_no,
            bbox=BoundingBox(l=0, t=0, r=10, b=10),
            charspan=(0, 5),
        )
    )
    return doc, item.self_ref


def _helper():
    """Return the patched static helper, or skip if the clone isn't patched
    (clean un-patched clone on host — needs the image build / in-container)."""
    from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy

    fn = getattr(ManyToOneStrategy, "_pages_from_source_refs", None)
    if fn is None:
        pytest.skip(
            "_pages_from_source_refs absent — patch 0005 not applied to the clone; "
            "run in-container (eip-mmdpp-docling-graph-1) or after the image build"
        )
    return fn


def test_resolves_real_page_via_refitem_resolve():
    """The fix: resolve self_ref → item.prov[].page_no via RefItem.resolve."""
    fn = _helper()
    doc, ref = _make_doc_with_page(3)
    pages = fn(doc, [ref])
    assert pages == [3], f"expected [3] resolved via RefItem.resolve, got {pages!r}"


def test_lookup_attribute_is_absent_on_docling_document():
    """Pin the root cause: DoclingDocument has no `lookup` — the old
    getattr(document, 'lookup') fallback was guaranteed dead code."""
    from docling_core.types.doc.document import DoclingDocument

    assert not hasattr(DoclingDocument, "lookup"), (
        "DoclingDocument grew a `lookup` method — revisit the helper; the fix "
        "deliberately uses RefItem.resolve because lookup did not exist"
    )


def test_empty_and_pageless_inputs_return_empty_list():
    """Defensive + the legitimate page-less case (never fabricated)."""
    fn = _helper()
    assert fn(None, ["#/texts/0"]) == []          # no document
    doc, _ = _make_doc_with_page(1)
    assert fn(doc, []) == []                       # no refs
    assert fn(doc, ["#/texts/999"]) == []          # unresolvable ref → []
    assert fn(doc, [None, ""]) == []               # junk refs skipped


def test_dedupes_and_sorts_pages():
    """Two refs resolving to pages {5, 2} → sorted unique [2, 5]."""
    from docling_core.types.doc import BoundingBox, DocItemLabel
    from docling_core.types.doc.document import DoclingDocument, ProvenanceItem

    fn = _helper()
    doc = DoclingDocument(name="test")
    refs = []
    for pg in (5, 2, 5):
        it = doc.add_text(label=DocItemLabel.TEXT, text=f"item p{pg}")
        it.prov.append(
            ProvenanceItem(page_no=pg, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))
        )
        refs.append(it.self_ref)
    assert fn(doc, refs) == [2, 5]
