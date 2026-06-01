"""Part C: a synthesizer self_ref element_uid must resolve to its SPECIFIC chunk
via identity_map, not fan out to all chunks."""
from app.workers.pipeline import _resolve_mention_chunks


def test_self_ref_resolves_to_specific_chunk_via_identity_map():
    resolved, coarse = _resolve_mention_chunks(
        "#/texts/5",
        {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]},
        {"#/texts/5": "p1-2-text-abcd"},
        ["chunkA", "chunkB", "chunkC"],
    )
    assert resolved == ["chunkA"]
    assert coarse is False


def test_unmapped_self_ref_falls_back_to_all_chunks_flagged_coarse():
    resolved, coarse = _resolve_mention_chunks(
        "#/texts/99", {"x": ["chunkA"]}, {"#/texts/5": "x"}, ["chunkA", "chunkB"],
    )
    assert resolved == ["chunkA", "chunkB"]
    assert coarse is True


def test_concrete_element_uid_resolves_directly_no_regression():
    resolved, coarse = _resolve_mention_chunks(
        "p1-2-text-abcd", {"p1-2-text-abcd": ["chunkA"]}, {}, ["chunkA", "chunkB"],
    )
    assert resolved == ["chunkA"]
    assert coarse is False


def test_unknown_concrete_element_uid_returns_empty_not_coarse():
    # A non-"#/" element_uid absent from the map must NOT fan out to all chunks
    # (only unresolved "#/" self_refs do); it returns ([], False).
    resolved, coarse = _resolve_mention_chunks(
        "p9-9-text-zzzz", {"p1-2-text-abcd": ["chunkA"]}, {}, ["chunkA", "chunkB"],
    )
    assert resolved == []
    assert coarse is False
