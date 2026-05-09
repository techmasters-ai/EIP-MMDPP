"""Unit tests — QueryResultItem provenance metadata fields."""

import uuid

_CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_DOC_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def test_query_result_item_carries_evidence_metadata():
    """QueryResultItem accepts the new provenance fields."""
    from app.schemas.retrieval import QueryResultItem

    item = QueryResultItem(
        chunk_id=_CHUNK_ID,
        document_id=_DOC_ID,
        score=0.9,
        modality="text",
        content_text="hello",
        self_refs=["#/texts/12"],
        evidence_ids=["#/texts/12"],
        page_numbers=[3, 4],
    )
    assert item.self_refs == ["#/texts/12"]
    assert item.evidence_ids == ["#/texts/12"]
    assert item.page_numbers == [3, 4]


def test_query_result_item_backward_compat():
    """Existing callers that don't set provenance fields still work."""
    from app.schemas.retrieval import QueryResultItem

    item = QueryResultItem(
        chunk_id=_CHUNK_ID,
        document_id=_DOC_ID,
        score=0.9,
        modality="text",
        content_text="hello",
    )
    assert item.self_refs == []
    assert item.evidence_ids == []
    assert item.page_numbers == []
