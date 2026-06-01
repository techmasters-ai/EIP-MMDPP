"""Part C / Task 4: a mention's self_refs must resolve to their SPECIFIC chunks
via element_uid_chunk_map / identity_map. Unresolvable self_refs contribute
NOTHING — there is no fan-out to the whole document or all-artifact chunks.

Collapsed signature (Task 4 refactor):
    _resolve_mention_chunks(self_refs, element_uid_chunk_map, identity_map)
        -> (chunk_ids, is_coarse)

Behavior:
  (1) resolve EACH self_ref via element_uid_chunk_map (direct hit) or
      identity_map (self_ref -> element_uid -> element_uid_chunk_map) -> UNION,
      dedup preserving order. Unresolvable self_refs add nothing.
  (2) is_coarse is True ONLY when there were self_refs but NONE resolved; the
      caller then WARNs and emits no edge. The all-document fan-out and the
      batch-chunk-ids fallback tier have both been removed.
"""
from app.workers.pipeline import _resolve_mention_chunks


def test_self_ref_resolves_to_specific_chunk_via_identity_map():
    # A real #/texts/N resolves through identity_map -> element_uid -> chunk.
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/5"],
        {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]},
        {"#/texts/5": "p1-2-text-abcd"},
    )
    assert resolved == ["chunkA"]
    assert coarse is False


def test_two_self_refs_both_resolve_union_not_coarse():
    # Two self_refs that both resolve -> UNION of their two chunks (precise).
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/5", "#/texts/6"],
        {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]},
        {"#/texts/5": "p1-2-text-abcd", "#/texts/6": "p1-3-text-ef01"},
    )
    assert set(resolved) == {"chunkA", "chunkB"}
    assert coarse is False


def test_one_resolves_one_misses_returns_only_resolved_no_fan_out():
    # Some self_refs resolve, some don't -> ONLY the resolved chunk(s) come back.
    # The unresolved self_ref contributes nothing; crucially the result NEVER
    # contains unrelated document chunks (chunkB/chunkC exist in the map but
    # belong to other elements and must not leak in).
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/5", "#/texts/99"],  # 99 unmapped
        {
            "p1-2-text-abcd": ["chunkA"],
            "p1-3-text-ef01": ["chunkB"],  # unrelated doc chunk
            "p1-4-text-9999": ["chunkC"],  # unrelated doc chunk
        },
        {"#/texts/5": "p1-2-text-abcd"},
    )
    assert resolved == ["chunkA"]
    # No fan-out: unrelated document chunks must never appear.
    assert "chunkB" not in resolved
    assert "chunkC" not in resolved
    assert coarse is False


def test_all_unresolved_returns_empty_coarse():
    # All self_refs unresolved -> ([], True). Caller WARNs / emits no edge.
    # Even though chunkA exists in the map, it belongs to a different element
    # and must NOT be returned as a fan-out fallback.
    resolved, coarse = _resolve_mention_chunks(
        ["#/texts/99"],
        {"p1-2-text-abcd": ["chunkA"]},
        {"#/texts/5": "p1-2-text-abcd"},
    )
    assert resolved == []
    assert coarse is True


def test_no_self_refs_is_not_coarse():
    # No self_refs at all -> empty result but NOT coarse (nothing to resolve,
    # so there is nothing to WARN about). is_coarse = (not resolved) and bool(self_refs).
    resolved, coarse = _resolve_mention_chunks(
        [],
        {"p1-2-text-abcd": ["chunkA"]},
        {"#/texts/5": "p1-2-text-abcd"},
    )
    assert resolved == []
    assert coarse is False


def test_concrete_element_uid_resolves_directly_no_regression():
    # A concrete {page}-{order}-... element_uid still resolves directly via
    # element_uid_chunk_map (no identity_map bridge needed) — no regression.
    resolved, coarse = _resolve_mention_chunks(
        ["p1-2-text-abcd"],
        {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]},
        {},
    )
    assert resolved == ["chunkA"]
    assert coarse is False


def test_unknown_concrete_element_uid_is_coarse_no_fan_out():
    # A non-"#/" element_uid absent from the map is unresolved. With no other
    # resolvable self_ref the result is ([], True) — never a fan-out to the
    # unrelated chunkA that happens to live in the map.
    resolved, coarse = _resolve_mention_chunks(
        ["p9-9-text-zzzz"],
        {"p1-2-text-abcd": ["chunkA"]},
        {},
    )
    assert resolved == []
    assert "chunkA" not in resolved
    assert coarse is True
