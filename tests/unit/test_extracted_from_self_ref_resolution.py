"""Part C / Task 4: a mention's self_refs must resolve to their SPECIFIC chunks
via element_uid_chunk_map / identity_map, falling back to the mention's BATCH
chunk-set (NOT the whole document) when some self_refs are unresolvable.

New signature (Task 4):
    _resolve_mention_chunks(self_refs, element_uid_chunk_map, identity_map,
                            batch_chunk_ids) -> (chunk_ids, is_coarse)

Resolution order:
  (1) resolve EACH self_ref via element_uid_chunk_map (direct hit) or
      identity_map (self_ref -> element_uid -> element_uid_chunk_map) -> UNION;
  (2) if ANY self_ref is unresolved, fall back to batch_chunk_ids (NOT
      all-document);
  (3) only if batch_chunk_ids is ALSO empty, return ([], is_coarse=True) and
      the caller WARNs.
The all-document fan-out has been removed entirely.
"""
from app.workers.pipeline import _resolve_mention_chunks


def test_self_ref_resolves_to_specific_chunk_via_identity_map():
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/5"],
        {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]},
        {"#/texts/5": "p1-2-text-abcd"},
        ["chunkA"],  # batch_chunk_ids — irrelevant here, self_ref resolves
    )
    assert resolved == ["chunkA"]
    assert coarse is False


def test_two_self_refs_both_resolve_union_not_coarse():
    # (a) two self_refs that both resolve -> UNION of their two chunks.
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/5", "#/texts/6"],
        {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]},
        {"#/texts/5": "p1-2-text-abcd", "#/texts/6": "p1-3-text-ef01"},
        ["chunkA", "chunkB"],
    )
    assert set(resolved) == {"chunkA", "chunkB"}
    assert coarse is False


def test_one_resolves_one_misses_falls_back_to_batch_not_all_document():
    # (b) one self_ref resolves + one doesn't -> result includes the resolved
    # chunk and falls back to batch_chunk_ids for the rest, NOT all-document.
    # batch_chunk_ids here is a STRICT SUBSET of all document chunks to prove
    # the fallback is the batch set, not the document.
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/5", "#/texts/99"],  # 99 unmapped
        {"p1-2-text-abcd": ["chunkA"]},
        {"#/texts/5": "p1-2-text-abcd"},
        ["chunkA", "chunkB"],  # batch set (NOT chunkC/D/E of the doc)
    )
    assert "chunkA" in resolved
    # The unresolved self_ref fell back to the BATCH set, never the document.
    assert set(resolved) == {"chunkA", "chunkB"}
    assert coarse is False


def test_all_unresolved_and_empty_batch_returns_empty_coarse():
    # (c) all unresolved + batch_chunk_ids empty -> ([], True), caller WARNs.
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/99"],
        {"p1-2-text-abcd": ["chunkA"]},
        {"#/texts/5": "p1-2-text-abcd"},
        [],  # empty batch set
    )
    assert resolved == []
    assert coarse is True


def test_concrete_element_uid_resolves_directly_no_regression():
    # (d) a concrete {page}-{order}-... element_uid still resolves directly
    # via element_uid_chunk_map (no regression).
    resolved, coarse = _resolve_mention_chunks(
        ["p1-2-text-abcd"],
        {"p1-2-text-abcd": ["chunkA"]},
        {},
        ["chunkA", "chunkB"],
    )
    assert resolved == ["chunkA"]
    assert coarse is False


def test_unknown_concrete_element_uid_falls_back_to_batch():
    # A non-"#/" element_uid absent from the map is unresolved; with a non-empty
    # batch set it falls back to the batch (never all-document), coarse=False.
    resolved, coarse = _resolve_mention_chunks(
        ["p9-9-text-zzzz"],
        {"p1-2-text-abcd": ["chunkA"]},
        {},
        ["chunkA", "chunkB"],
    )
    assert set(resolved) == {"chunkA", "chunkB"}
    assert coarse is False
