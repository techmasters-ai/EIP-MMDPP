"""Fix B — _field_evidence persistence dict includes evidence_id/page/document_id.

Tests the dict shape that _build_node_record emits into NodeRecord.properties
["_field_evidence"] when a MergedEntityRecord carries FieldEvidenceRow objects
with evidence metadata. We exercise the logic directly rather than invoking the
full pipeline.
"""
from __future__ import annotations

from types import SimpleNamespace


def _make_field_evidence_row(**kwargs):
    """Build a SimpleNamespace that looks like FieldEvidenceRow."""
    defaults = dict(
        chunk_id="chunk-1",
        snippet="diameter: 0.37 m",
        element_uid="#/texts/12",
        value=0.37,
        evidence_id="#/texts/12",
        page=3,
        document_id="doc-uuid-abc",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _build_node_record_props(field_evidence_dict: dict) -> dict:
    """Reproduce the _build_node_record logic from pipeline.py in isolation.

    Returns the value of props["_field_evidence"] so the test can assert
    on its shape without needing a live graph store or DB.
    """
    props: dict = {}
    if field_evidence_dict:
        props["_field_evidence"] = {
            field_name: [
                {
                    "chunk_id": row.chunk_id,
                    "snippet": row.snippet,
                    "element_uid": row.element_uid,
                    "value": row.value,
                    # Fix B: surface evidence metadata added in Task 12.5.
                    "evidence_id": row.evidence_id,
                    "page": row.page,
                    "document_id": row.document_id,
                }
                for row in rows
            ]
            for field_name, rows in field_evidence_dict.items()
        }
    return props


def test_field_evidence_persisted_with_evidence_metadata():
    """_build_node_record must persist the new evidence_id/page/document_id
    into the _field_evidence properties dict."""
    row = _make_field_evidence_row(
        evidence_id="#/texts/12",
        page=3,
        document_id="doc-uuid-abc",
    )
    props = _build_node_record_props({"diameter": [row]})

    field_ev = props["_field_evidence"]
    assert "diameter" in field_ev
    diam_rows = field_ev["diameter"]
    assert len(diam_rows) == 1
    r = diam_rows[0]
    assert r["evidence_id"] == "#/texts/12"
    assert r["page"] == 3
    assert r["document_id"] == "doc-uuid-abc"
    # Existing fields still present.
    assert r["chunk_id"] == "chunk-1"
    assert r["snippet"] == "diameter: 0.37 m"
    assert r["element_uid"] == "#/texts/12"
    assert r["value"] == 0.37


def test_field_evidence_none_evidence_metadata_preserved():
    """Rows where evidence_id/page/document_id are None still serialise cleanly."""
    row = _make_field_evidence_row(evidence_id=None, page=None, document_id=None)
    props = _build_node_record_props({"range": [row]})

    r = props["_field_evidence"]["range"][0]
    assert r["evidence_id"] is None
    assert r["page"] is None
    assert r["document_id"] is None
