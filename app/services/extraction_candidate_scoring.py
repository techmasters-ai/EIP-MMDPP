"""Task C4 — candidate merging into MergedCandidate.

Aggregates results from all upstream retrieval sources (entity dense,
per-field dense, lexical, pattern, section meta, table meta) by
stable candidate_key and produces a unified MergedCandidate per chunk.

C4 ONLY MERGES — no scoring (C5), no reranking, no endpoint wiring (C6).
content_type stays None (Phase D deferred — no table metadata column yet).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.services.extraction_chunk_index import (
    read_chunk_index,
    read_chunk_source_refs,
    read_chunk_token_count,
)

if TYPE_CHECKING:
    from app.services.graph_store import GraphEntityResult


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MergedCandidate:
    candidate_key: str               # vertex_id preferred; self_ref fallback
    chunk_index: int
    self_ref: str
    chunk_text: str
    source_refs: list[str]
    token_count: int
    page_number: int | None          # lineage — read from properties; do NOT drop
    vector_score: float | None
    field_scores: dict[str, float]   # per-field dense scores (best)
    alias_hits: int
    pattern_hits: int
    negative_hits: int
    section_hits: int
    content_type: str | None         # "table" when Phase D metadata exists (None for now)
    retrieval_sources: set[str]      # {"dense", "field:<field_name>", "lexical", "pattern"}
    supported_field_hints: set[str]  # field_names that contributed signal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _candidate_key(r: "GraphEntityResult") -> str:
    """vertex_id preferred; self_ref fallback — mirrors merged-mode semantics."""
    return r.properties.get("vertex_id") or r.properties.get("self_ref", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_candidates(
    entity_dense: list["GraphEntityResult"],
    field_dense: dict[str, list["GraphEntityResult"]],
    lexical_hits: dict[str, dict],   # from C2: {candidate_key: {alias_hits, negative_hits, supported_fields}}
    pattern_hits: dict[str, dict],   # from C3: {candidate_key: {pattern_hits, supported_fields}}
    section_meta: dict[str, dict],   # may be empty {} for now
    table_meta: dict[str, str],      # content_type keyed by candidate_key; may be {} (Phase D deferred)
) -> list[MergedCandidate]:
    """Merge all retrieval sources by candidate_key.

    Collision guard: merged-mode self_ref can repeat across distinct chunks
    (see extraction_chunk_search.py:385-393). Two results with the SAME
    self_ref but DIFFERENT vertex_id have DIFFERENT candidate_keys
    (vertex_id wins) and remain SEPARATE candidates.
    """
    # --- accumulator keyed by candidate_key ---
    # Each bucket holds the mutable state we reduce into.
    buckets: dict[str, dict] = {}

    def _ensure(key: str, r: "GraphEntityResult") -> dict:
        if key not in buckets:
            buckets[key] = {
                "candidate_key": key,
                "chunk_index": read_chunk_index(r.properties),
                "self_ref": r.properties.get("self_ref", ""),
                "chunk_text": r.properties.get("chunk_text", ""),
                "source_refs": read_chunk_source_refs(r.properties),
                "token_count": read_chunk_token_count(r.properties),
                "page_number": r.properties.get("page_number"),
                "vector_score": None,
                "field_scores": {},
                "retrieval_sources": set(),
                "supported_field_hints": set(),
            }
        return buckets[key]

    # 1. entity_dense — sets vector_score and "dense" tag
    for r in entity_dense:
        key = _candidate_key(r)
        b = _ensure(key, r)
        b["retrieval_sources"].add("dense")
        if r.score is not None:
            # Take the max in case the same chunk appears twice in entity_dense
            if b["vector_score"] is None or r.score > b["vector_score"]:
                b["vector_score"] = r.score

    # 2. field_dense — sets field_scores (max per field) and "field:<name>" tags
    for field_name, results in field_dense.items():
        tag = f"field:{field_name}"
        for r in results:
            key = _candidate_key(r)
            b = _ensure(key, r)
            b["retrieval_sources"].add(tag)
            b["supported_field_hints"].add(field_name)
            if r.score is not None:
                prev = b["field_scores"].get(field_name)
                if prev is None or r.score > prev:
                    b["field_scores"][field_name] = r.score

    # 3. lexical_hits — alias_hits, negative_hits, "lexical" tag, field hints
    for key, lh in lexical_hits.items():
        if key not in buckets:
            # lexical-only candidate: no GER to read chunk fields from.
            # Create a minimal bucket; chunk fields stay at defaults.
            buckets[key] = {
                "candidate_key": key,
                "chunk_index": -1,
                "self_ref": key,
                "chunk_text": "",
                "source_refs": [],
                "token_count": 0,
                "page_number": None,
                "vector_score": None,
                "field_scores": {},
                "retrieval_sources": set(),
                "supported_field_hints": set(),
            }
        b = buckets[key]
        b["retrieval_sources"].add("lexical")
        # lexical_hits values: {alias_hits, negative_hits, supported_fields}
        b.setdefault("_alias_hits", 0)
        b["_alias_hits"] = b.get("_alias_hits", 0) + lh.get("alias_hits", 0)
        b.setdefault("_negative_hits", 0)
        b["_negative_hits"] = b.get("_negative_hits", 0) + lh.get("negative_hits", 0)
        for f in lh.get("supported_fields", set()):
            b["supported_field_hints"].add(f)

    # 4. pattern_hits — pattern_hits count, "pattern" tag, field hints
    for key, ph in pattern_hits.items():
        if key not in buckets:
            buckets[key] = {
                "candidate_key": key,
                "chunk_index": -1,
                "self_ref": key,
                "chunk_text": "",
                "source_refs": [],
                "token_count": 0,
                "page_number": None,
                "vector_score": None,
                "field_scores": {},
                "retrieval_sources": set(),
                "supported_field_hints": set(),
            }
        b = buckets[key]
        b["retrieval_sources"].add("pattern")
        b.setdefault("_pattern_hits", 0)
        b["_pattern_hits"] = b.get("_pattern_hits", 0) + ph.get("pattern_hits", 0)
        for f in ph.get("supported_fields", set()):
            b["supported_field_hints"].add(f)

    # 5. Build final MergedCandidate list
    out: list[MergedCandidate] = []
    for key, b in buckets.items():
        sm = section_meta.get(key, {})
        out.append(
            MergedCandidate(
                candidate_key=key,
                chunk_index=b["chunk_index"],
                self_ref=b["self_ref"],
                chunk_text=b["chunk_text"],
                source_refs=b["source_refs"],
                token_count=b["token_count"],
                page_number=b["page_number"],
                vector_score=b["vector_score"],
                field_scores=b["field_scores"],
                alias_hits=b.get("_alias_hits", 0),
                pattern_hits=b.get("_pattern_hits", 0),
                negative_hits=b.get("_negative_hits", 0),
                section_hits=sm.get("section_hits", 0),
                content_type=table_meta.get(key),
                retrieval_sources=b["retrieval_sources"],
                supported_field_hints=b["supported_field_hints"],
            )
        )

    return out
