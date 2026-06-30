from __future__ import annotations


def test_request_field_default_none():
    from app.schemas.retrieval import UnifiedQueryRequest
    assert UnifiedQueryRequest(query_text="q").ontology_reserved_slots is None
    assert UnifiedQueryRequest(query_text="q", ontology_reserved_slots=2).ontology_reserved_slots == 2


def test_settings_endpoint_includes_reserved_slots():
    import asyncio
    from app.api.v1.retrieval import get_retrieval_settings
    out = asyncio.get_event_loop().run_until_complete(get_retrieval_settings())
    assert "ontology_reserved_slots" in out
