def test_embedding_chunk_metadata_includes_evidence_fields():
    """Chunks emitted by the native HybridChunker path in pipeline.py
    must carry evidence_ids, self_refs, page_numbers, document_id, text
    in their persisted metadata."""
    from app.workers.pipeline import _build_native_chunk_meta
    from types import SimpleNamespace
    chunk = SimpleNamespace(
        text="hello world",
        meta=SimpleNamespace(doc_items=[
            SimpleNamespace(
                self_ref="#/texts/0",
                prov=[SimpleNamespace(page_no=3)],
            ),
        ]),
    )
    meta = _build_native_chunk_meta(
        chunk_idx=0, chunk=chunk, document_id="doc-uuid-abc",
        model_version="bge-m3:latest",
    )
    assert meta["evidence_ids"] == ["#/texts/0"]
    assert meta["self_refs"] == ["#/texts/0"]
    assert meta["page_numbers"] == [3]
    assert meta["document_id"] == "doc-uuid-abc"
    assert meta["modality"] == "text"


def test_embedding_chunk_metadata_handles_missing_doc_items():
    """When chunk has no doc_items, returns empty self_refs / evidence_ids."""
    from app.workers.pipeline import _build_native_chunk_meta
    from types import SimpleNamespace
    chunk = SimpleNamespace(text="empty", meta=SimpleNamespace(doc_items=[]))
    meta = _build_native_chunk_meta(
        chunk_idx=0, chunk=chunk, document_id="doc-uuid",
        model_version="bge-m3:latest",
    )
    assert meta["self_refs"] == []
    assert meta["evidence_ids"] == []
    assert meta["page_numbers"] == []
    assert meta["page_number"] is None
