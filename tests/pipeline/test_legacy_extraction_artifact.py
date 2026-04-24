"""Legacy extraction must produce a stub docling_document.json that:
  1. Round-trips through DoclingDocument.model_validate()
  2. Is walkable by docling_anchors.walk()
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_stub_validates_as_docling_document():
    from app.workers.pipeline import _build_legacy_docling_document_json
    from docling_core.types.doc import DoclingDocument

    stub = _build_legacy_docling_document_json(
        document_id="11111111-1111-1111-1111-111111111111",
        text="hello world from a .txt fallback",
    )

    # Contract 1: the stub (a dict) must model_validate as a real DoclingDocument.
    doc = DoclingDocument.model_validate(stub)
    assert doc.name == "11111111-1111-1111-1111-111111111111"
    assert any("hello world" in t.text for t in doc.texts)


def test_stub_walks_via_docling_anchors():
    """walk() accepts a JSON dict (not a validated DoclingDocument) and
    returns a MergedExtraction. It internally calls
    DoclingDocument.model_validate(stub) so a malformed stub would raise
    during validation."""
    from app.workers.pipeline import _build_legacy_docling_document_json
    from app.services.docling_anchors import walk as _walk

    stub = _build_legacy_docling_document_json(
        document_id="22222222-2222-2222-2222-222222222222",
        text="a tiny markdown body",
    )

    result = _walk(
        stub,
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
        {},  # minimal ontology
    )
    assert result is not None
    assert hasattr(result, "entities")


def test_stub_handles_empty_text():
    """Legacy extraction sometimes produces zero-length text; stub must still
    be valid (empty texts list is OK)."""
    from app.workers.pipeline import _build_legacy_docling_document_json
    from docling_core.types.doc import DoclingDocument

    stub = _build_legacy_docling_document_json(
        document_id="33333333-3333-3333-3333-333333333333",
        text="",
    )
    DoclingDocument.model_validate(stub)
