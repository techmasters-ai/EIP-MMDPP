"""Task C4 — candidate merging into MergedCandidate.
Task C5 — score_candidates: post-rerank precision scoring.

C4: Aggregates results from all upstream retrieval sources (entity dense,
per-field dense, lexical, pattern, section meta, table meta) by
stable candidate_key and produces a unified MergedCandidate per chunk.
C4 ONLY MERGES — no scoring (C5), no reranking, no endpoint wiring (C6).
content_type stays None (Phase D deferred — no table metadata column yet).

C5: Runs AFTER cross-encoder rerank(). Combines normalized reranker score
(semantic precision) with C2–C4 keyword/pattern/section/negative/table
signals (lexical precision) into a final ordering.

Candidate representation fed to score_candidates (documented here for C6):
  Each input dict carries:
    - "merged_candidate": MergedCandidate   (the C4 object)
    - "content_text": str                   (chunk_text, for reranker compat)
    - "reranker_score": float | absent       (written by rerank(); absent for
                                              unscorable — empty content_text)
  Signals are read directly from merged_candidate.*_hits / .content_type.

Normalization choices (min-max throughout — pinned here for test contract):
  - rerank_norm: min-max over pool; missing reranker_score → 0.0.
  - lexical/pattern/section/negative norms: hit_count / max(1, pool_max).
  Sort: final desc → reranker_score desc (missing = float('-inf')) → candidate_key asc.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.services.extraction_chunk_index import (
    read_chunk_index,
    read_chunk_source_refs,
    read_chunk_token_count,
)

if TYPE_CHECKING:
    from app.services.graph_store import GraphEntityResult
    from app.services.ontology_bundles import RetrievalProfile


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
    # Keyword search is PRECISION not recall: only boost candidates already in
    # the dense pool. Keys absent from buckets are skipped entirely — the recall
    # safety net for "dense missed it" is Phase E's lexical_table fallback.
    for key, lh in lexical_hits.items():
        if key not in buckets:
            continue
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
    # Same precision-not-recall rule as lexical: skip keys absent from dense pool.
    for key, ph in pattern_hits.items():
        if key not in buckets:
            continue
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


# ---------------------------------------------------------------------------
# C5 — post-rerank precision scoring
# ---------------------------------------------------------------------------

_RERANK_FLOOR = float("-inf")   # sentinel for unscorable candidates in sort key


def score_candidates(
    candidates: list[dict[str, Any]],
    cfg: "RetrievalProfile",
) -> list[tuple["MergedCandidate", float]]:
    """Score and sort candidates after cross-encoder reranking.

    Input ``candidates`` is the list produced by C6 (caller), shaped as::

        [
            {
                "merged_candidate": MergedCandidate,   # C4 object
                "content_text":     str,               # chunk_text (reranker compat)
                # "reranker_score": float              # present only on scorable chunks
            },
            ...
        ]

    Returns a list of ``(MergedCandidate, final_score)`` tuples sorted by:
      1. ``final`` descending
      2. ``reranker_score`` descending (missing = ``float('-inf')``)
      3. ``candidate_key`` ascending (lexicographic, stable for equal ties)

    Formula (all weights from ``cfg``)::

        final = cfg.rerank_weight  * rerank_norm
              + cfg.lexical_weight * lexical_norm
              + cfg.pattern_weight * pattern_norm
              + cfg.section_weight * section_norm
              + cfg.table_boost    * is_table
              - cfg.negative_weight * negative_norm
        final = max(final, 0.0)

    Normalisation (min-max for reranker; ratio-max for lexical/pattern signals):
      - ``rerank_norm``: min-max over pool; missing ``reranker_score`` → 0.0.
      - ``lexical_norm``  = alias_hits   / max(1, pool_max_alias_hits)
      - ``pattern_norm``  = pattern_hits / max(1, pool_max_pattern_hits)
      - ``section_norm``  = section_hits / max(1, pool_max_section_hits)
      - ``negative_norm`` = negative_hits / max(1, pool_max_negative_hits)
      - ``is_table``  = 1.0 if content_type == "table" else 0.0

    C5 ONLY SCORES + SORTS — it does NOT apply the top_k cut (C6 does) and
    does NOT call rerank() (C6 does) and does NOT wire the endpoint.
    """
    if not candidates:
        return []

    # ------------------------------------------------------------------
    # 1. Collect raw reranker scores for min-max normalisation
    # ------------------------------------------------------------------
    raw_rr: list[float] = [
        c["reranker_score"]
        for c in candidates
        if "reranker_score" in c
    ]
    if raw_rr and len(raw_rr) > 1:
        rr_min = min(raw_rr)
        rr_max = max(raw_rr)
        rr_span = rr_max - rr_min
    elif raw_rr:          # single scorable candidate
        rr_min = raw_rr[0]
        rr_max = raw_rr[0]
        rr_span = 0.0
    else:                 # all unscorable
        rr_min = rr_max = rr_span = 0.0

    # ------------------------------------------------------------------
    # 2. Pool maxima for hit-count normalisation
    # ------------------------------------------------------------------
    mcs: list[MergedCandidate] = [c["merged_candidate"] for c in candidates]
    max_alias    = max((mc.alias_hits    for mc in mcs), default=0)
    max_pattern  = max((mc.pattern_hits  for mc in mcs), default=0)
    max_section  = max((mc.section_hits  for mc in mcs), default=0)
    max_negative = max((mc.negative_hits for mc in mcs), default=0)

    # ------------------------------------------------------------------
    # 3. Score each candidate
    # ------------------------------------------------------------------
    results: list[tuple[MergedCandidate, float, float]] = []   # (mc, final, raw_rr)

    for c in candidates:
        mc: MergedCandidate = c["merged_candidate"]
        raw = c.get("reranker_score")  # None / absent for unscorable

        # rerank_norm: 0.0 for unscorable; min-max otherwise
        if raw is None:
            rerank_norm = 0.0
            sort_rr = _RERANK_FLOOR
        else:
            if rr_span > 0.0:
                rerank_norm = (raw - rr_min) / rr_span
            else:
                # All scorable candidates have identical scores → normalise to 1.0
                rerank_norm = 1.0 if raw_rr else 0.0
            sort_rr = raw

        lexical_norm  = mc.alias_hits    / max(1, max_alias)
        pattern_norm  = mc.pattern_hits  / max(1, max_pattern)
        section_norm  = mc.section_hits  / max(1, max_section)
        negative_norm = mc.negative_hits / max(1, max_negative)
        is_table      = 1.0 if mc.content_type == "table" else 0.0

        final = (
            cfg.rerank_weight   * rerank_norm
            + cfg.lexical_weight  * lexical_norm
            + cfg.pattern_weight  * pattern_norm
            + cfg.section_weight  * section_norm
            + cfg.table_boost     * is_table
            - cfg.negative_weight * negative_norm
        )
        final = max(final, 0.0)

        results.append((mc, final, sort_rr))

    # ------------------------------------------------------------------
    # 4. Sort: final desc → reranker_score desc → candidate_key asc
    # ------------------------------------------------------------------
    results.sort(key=lambda t: (-t[1], -t[2] if t[2] != _RERANK_FLOOR else math.inf, t[0].candidate_key))

    return [(mc, score) for mc, score, _ in results]


# ---------------------------------------------------------------------------
# E1 — Pure fallback-decision helpers (no DB / LLM / reranker)
# ---------------------------------------------------------------------------
# These helpers let the chunk_scope endpoint (E2) decide whether retrieval
# under-covered a pass WITHOUT calling the reranker or LLM.
#
# "Real retrieval signal" definition (pinned for test contract):
#   A MergedCandidate has real signal when its retrieval_sources is non-empty.
#   retrieval_sources is only populated by the merge process when an actual
#   retrieval channel tagged the candidate ({"dense"}, {"field:<name>"},
#   {"lexical"}, {"pattern"}, {"identity_anchor"}, or any combination).
#   A candidate with retrieval_sources == set() was never touched by any
#   retrieval channel — it is a bare/noise entry and must NOT count toward the
#   enough_candidates threshold so that fallback fires correctly.
# ---------------------------------------------------------------------------


def field_coverage(candidates: list[MergedCandidate]) -> dict[str, int]:
    """Return {field_name: number of candidates whose supported_field_hints include it}.

    Aggregates across the entire pool.  Empty pool → {}.
    """
    counts: dict[str, int] = {}
    for mc in candidates:
        for field in mc.supported_field_hints:
            counts[field] = counts.get(field, 0) + 1
    return counts


def enough_candidates(
    candidates: list[MergedCandidate],
    cfg: "RetrievalProfile",
) -> bool:
    """True iff the number of candidates with REAL retrieval signal >= min(cfg.top_k, 10).

    "Real signal" = retrieval_sources is non-empty (at least one genuine tag
    such as "dense", "field:<name>", "lexical", "pattern", "identity_anchor").
    An all-noise pool (every candidate has retrieval_sources == set()) returns
    False so the caller fires the fallback path rather than suppressing it.

    Does NOT inspect reranker_score or the final post-rerank score — this is a
    pure recall-coverage decision over the merged pool before scoring.
    """
    threshold = min(cfg.top_k, 10)
    real_count = sum(1 for mc in candidates if mc.retrieval_sources)
    return real_count >= threshold


def enough_field_coverage(
    candidates: list[MergedCandidate],
    cfg: "RetrievalProfile",
) -> bool:
    """True iff the number of schema fields covered by retrieved chunks >= cfg.fallback_min_field_coverage.

    A field is "covered" when at least one candidate in the pool has that field
    in its supported_field_hints (i.e. field_coverage count > 0).
    """
    cov = field_coverage(candidates)
    covered = sum(1 for n in cov.values() if n > 0)
    return covered >= cfg.fallback_min_field_coverage
