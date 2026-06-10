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
    # ------------------------------------------------------------------
    # Decomposed-lexical features (router-scoring lexical-decomposition piece).
    # The single conflated ``alias_hits`` is split into FOUR independently
    # weighted lexical features, gated behind ``cfg.lexical_decomposed``.
    # Appended with defaults so existing positional construction is unaffected.
    #
    # LEGACY INVARIANT (preserved): alias_hits == field_label_hits +
    # entity_anchor_text. ``pass_keyword_hits`` is purely additive/new and is
    # NOT folded into alias_hits.
    #   field_label_hits     — pydantic-schema field-alias matches (today's signal)
    #   pass_keyword_hits    — per-pass lexical_keywords matches (NEW)
    #   entity_anchor_text   — committed entity-name matches in chunk TEXT (C8)
    #   entity_anchor_section— entity-name matches in chunk SECTION (= section_hits)
    # ------------------------------------------------------------------
    field_label_hits: int = 0
    pass_keyword_hits: int = 0
    entity_anchor_text: int = 0
    entity_anchor_section: int = 0
    # ------------------------------------------------------------------
    # Per-row dense cosines from the multi-query matmul (Task 6 — capture-only).
    # Computed over ALL valid_rows BEFORE any top-k slice in
    # search_extraction_chunks_dense_multi_query and stamped here via
    # merge_candidates(row_cosines=...).  Default 0.0 is safe for callers
    # that pre-date Task 6 (e.g. build_pool_from_multi_channel_state, which
    # passes row_cosines=None until Task 7 wires the gate union).
    # ------------------------------------------------------------------
    max_field_cosine: float = 0.0
    mean_top3_field_cosine: float = 0.0


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
    row_cosines: dict | None = None, # {candidate_key: {entity_cosine, max_field_cosine, mean_top3_field_cosine}}
                                     # from search_extraction_chunks_dense_multi_query (Task 6).
                                     # None → both cosine fields default to 0.0 (backward-compat).
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
        # and (decomposed-lexical piece) optional {keyword_hits}.
        # alias_hits here is the FIELD-ALIAS count → field_label_hits.
        # keyword_hits is the per-pass keyword count → pass_keyword_hits
        # (a SEPARATE feature; NOT folded into alias_hits).
        b.setdefault("_field_label_hits", 0)
        b["_field_label_hits"] = b.get("_field_label_hits", 0) + lh.get("alias_hits", 0)
        b.setdefault("_pass_keyword_hits", 0)
        b["_pass_keyword_hits"] = b.get("_pass_keyword_hits", 0) + lh.get("keyword_hits", 0)
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
        field_label_hits = b.get("_field_label_hits", 0)
        pass_keyword_hits = b.get("_pass_keyword_hits", 0)
        section_hits = sm.get("section_hits", 0)
        # Decomposed-lexical mapping (router-scoring):
        #   field_label_hits      = field-alias count (from lexical_hits.alias_hits)
        #   pass_keyword_hits     = per-pass keyword count (lexical_hits.keyword_hits)
        #   entity_anchor_text    = anchor-in-TEXT count — 0 at merge time; the C8
        #                           lexical sub-channel (extraction_chunk_search)
        #                           bumps this AND alias_hits post-merge.
        #   entity_anchor_section = section_hits (anchor-in-SECTION count, Part 2)
        # LEGACY INVARIANT: alias_hits == field_label_hits + entity_anchor_text.
        # At merge time entity_anchor_text == 0, so alias_hits == field_label_hits,
        # byte-identical to the pre-decomposition alias_hits value.
        entity_anchor_text = 0
        # Task 6 — stamp per-row dense cosines when provided (0.0 default otherwise).
        rc = (row_cosines or {}).get(key, {})
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
                alias_hits=field_label_hits + entity_anchor_text,
                pattern_hits=b.get("_pattern_hits", 0),
                negative_hits=b.get("_negative_hits", 0),
                section_hits=section_hits,
                content_type=table_meta.get(key),
                retrieval_sources=b["retrieval_sources"],
                supported_field_hints=b["supported_field_hints"],
                field_label_hits=field_label_hits,
                pass_keyword_hits=pass_keyword_hits,
                entity_anchor_text=entity_anchor_text,
                entity_anchor_section=section_hits,
                max_field_cosine=float(rc.get("max_field_cosine", 0.0)),
                mean_top3_field_cosine=float(rc.get("mean_top3_field_cosine", 0.0)),
            )
        )

    return out


# ---------------------------------------------------------------------------
# C5 — post-rerank precision scoring
# ---------------------------------------------------------------------------

_RERANK_FLOOR = float("-inf")   # sentinel for unscorable candidates in sort key


# Keys of the per-candidate component dict surfaced for offline calibration
# (the additive ``score_components_all`` diagnostics field). Pinned here so the
# endpoint, the scoring path, and the tests share one contract. The NAMES are
# chosen to line up with the offline calibration scripts' COMPONENT_FIELDS
# (cosine, rerank_norm, section_norm, is_table, pattern_norm, negative_norm) and
# add the decomposed-lexical features (field_label_*, pass_keyword_*,
# entity_anchor_text/anchor_text_norm, entity_anchor_section/anchor_section_norm).
COMPONENT_KEYS: tuple[str, ...] = (
    "candidate_key",
    "cosine",
    "rerank_norm",
    "field_label_hits",
    "field_label_norm",
    "pass_keyword_hits",
    "pass_keyword_norm",
    "entity_anchor_text",
    "anchor_text_norm",
    "entity_anchor_section",
    "anchor_section_norm",
    "section_norm",
    "is_table",
    "pattern_norm",
    "negative_norm",
    "final_score",
    # Task 6 — per-row dense cosines retained from multi-query matmul (capture-only).
    "max_field_cosine",
    "mean_top3_field_cosine",
)


def score_candidates(
    candidates: list[dict[str, Any]],
    cfg: "RetrievalProfile",
    *,
    return_components: bool = False,
) -> (
    list[tuple["MergedCandidate", float]]
    | list[tuple["MergedCandidate", float, dict[str, Any]]]
):
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
              + lexical_term
              + cfg.pattern_weight * pattern_norm
              + cfg.section_weight * section_norm
              + cfg.table_boost    * is_table
              - cfg.negative_weight * negative_norm
        final = max(final, 0.0)

    The ``lexical_term`` is flag-gated on ``cfg.lexical_decomposed``
    (router-scoring lexical-decomposition piece):

      - ``False`` (DEFAULT) → the LITERAL legacy single-weight term::

            lexical_term = cfg.lexical_weight * lexical_norm

        Byte-identical final_score + ordering to before this piece, even when
        the decomposed features (field_label / pass_keyword / anchor_text /
        anchor_section) carry non-zero values — they are simply not read.

      - ``True`` → the term is replaced by four independently weighted
        sub-terms (each ``hit / max(1, pool_max)`` normalised)::

            lexical_term = cfg.field_label_weight     * field_label_norm
                         + cfg.pass_keyword_weight     * pass_keyword_norm
                         + cfg.anchor_text_weight      * anchor_text_norm
                         + cfg.anchor_section_weight   * anchor_section_norm

      The separate ``cfg.section_weight * section_norm`` term is unchanged in
      both branches (section_weight defaults to 0.0).

    Normalisation (min-max for reranker; ratio-max for lexical/pattern signals):
      - ``rerank_norm``: min-max over pool; missing ``reranker_score`` → 0.0.
      - ``lexical_norm``  = alias_hits   / max(1, pool_max_alias_hits)
      - ``pattern_norm``  = pattern_hits / max(1, pool_max_pattern_hits)
      - ``section_norm``  = section_hits / max(1, pool_max_section_hits)
      - ``negative_norm`` = negative_hits / max(1, pool_max_negative_hits)
      - ``is_table``  = 1.0 if content_type == "table" else 0.0

    C5 ONLY SCORES + SORTS — it does NOT apply the top_k cut (C6 does) and
    does NOT call rerank() (C6 does) and does NOT wire the endpoint.

    Component breakdown (additive, opt-in — ``return_components=True``):
      Returns ``list[(MergedCandidate, final_score, components)]`` where
      ``components`` is the REAL per-candidate dict this call computed (no
      recompute / drift). Keys are :data:`COMPONENT_KEYS`. Scoring, ordering,
      and the default 2-tuple return shape are UNCHANGED — the components are
      simply also emitted. ``final_score`` inside the dict is identical to the
      float in the pair, and ``cosine`` mirrors ``mc.vector_score`` (None → 0.0)
      so the offline calibration can read the absolute dense channel too.
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
    # Decomposed-lexical pool maxima — only consumed when cfg.lexical_decomposed.
    # Computed unconditionally (cheap) but unused on the legacy path.
    max_field_label    = max((mc.field_label_hits      for mc in mcs), default=0)
    max_pass_keyword   = max((mc.pass_keyword_hits      for mc in mcs), default=0)
    max_anchor_text    = max((mc.entity_anchor_text     for mc in mcs), default=0)
    max_anchor_section = max((mc.entity_anchor_section  for mc in mcs), default=0)

    # ------------------------------------------------------------------
    # 3. Score each candidate
    # ------------------------------------------------------------------
    results: list[tuple[MergedCandidate, float, float, dict[str, Any]]] = []
    # (mc, final, raw_rr, components)

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

        # Decomposed-lexical norms. Computed UNCONDITIONALLY (cheap; same
        # ratio-max as everything else) so the component breakdown always
        # carries per-feature signal for offline calibration. They only feed
        # the final_score when cfg.lexical_decomposed (the legacy path ignores
        # them — byte-identical scoring preserved).
        field_label_norm    = mc.field_label_hits      / max(1, max_field_label)
        pass_keyword_norm   = mc.pass_keyword_hits      / max(1, max_pass_keyword)
        anchor_text_norm    = mc.entity_anchor_text     / max(1, max_anchor_text)
        anchor_section_norm = mc.entity_anchor_section  / max(1, max_anchor_section)

        # Lexical term — flag-gated decomposition (router-scoring piece).
        # Default (lexical_decomposed=False): the LITERAL legacy single-weight
        # term, byte-identical to before this piece.
        # When True: replace it with four independently weighted sub-terms.
        if cfg.lexical_decomposed:
            lexical_term = (
                cfg.field_label_weight     * field_label_norm
                + cfg.pass_keyword_weight    * pass_keyword_norm
                + cfg.anchor_text_weight     * anchor_text_norm
                + cfg.anchor_section_weight  * anchor_section_norm
            )
        else:
            lexical_term = cfg.lexical_weight * lexical_norm

        final = (
            cfg.rerank_weight   * rerank_norm
            + lexical_term
            + cfg.pattern_weight  * pattern_norm
            + cfg.section_weight  * section_norm
            + cfg.table_boost     * is_table
            - cfg.negative_weight * negative_norm
        )
        final = max(final, 0.0)

        # Per-candidate component breakdown (additive diagnostics). Built from
        # the SAME values used for final above — no recompute/drift. cosine is
        # the absolute dense channel (mc.vector_score; None → 0.0), not a C5
        # term. Keys match COMPONENT_KEYS.
        components: dict[str, Any] = {
            "candidate_key": mc.candidate_key,
            "cosine": float(mc.vector_score) if mc.vector_score is not None else 0.0,
            "rerank_norm": float(rerank_norm),
            "field_label_hits": mc.field_label_hits,
            "field_label_norm": float(field_label_norm),
            "pass_keyword_hits": mc.pass_keyword_hits,
            "pass_keyword_norm": float(pass_keyword_norm),
            "entity_anchor_text": mc.entity_anchor_text,
            "anchor_text_norm": float(anchor_text_norm),
            "entity_anchor_section": mc.entity_anchor_section,
            "anchor_section_norm": float(anchor_section_norm),
            "section_norm": float(section_norm),
            "is_table": float(is_table),
            "pattern_norm": float(pattern_norm),
            "negative_norm": float(negative_norm),
            "final_score": final,
            # Task 6 — capture-only; not a scoring term.
            "max_field_cosine": mc.max_field_cosine,
            "mean_top3_field_cosine": mc.mean_top3_field_cosine,
        }

        results.append((mc, final, sort_rr, components))

    # ------------------------------------------------------------------
    # 4. Sort: final desc → reranker_score desc → candidate_key asc
    # ------------------------------------------------------------------
    results.sort(key=lambda t: (-t[1], -t[2] if t[2] != _RERANK_FLOOR else math.inf, t[0].candidate_key))

    if return_components:
        return [(mc, score, comps) for mc, score, _, comps in results]
    return [(mc, score) for mc, score, _, _ in results]


def score_components_for_pool(
    candidates: list[dict[str, Any]],
    cfg: "RetrievalProfile",
) -> list[dict[str, Any]]:
    """Return the per-candidate component dict for EVERY candidate in the pool.

    Thin wrapper over ``score_candidates(..., return_components=True)`` that
    drops the (mc, final) prefix and yields just the component dicts (in the
    same final-desc sorted order). This is the function the chunk-scope endpoint
    calls to populate the additive ``score_components_all`` diagnostics field for
    the FULL reranked pool. Keeping it a SEPARATE entry point means endpoint
    tests that monkeypatch ``score_candidates`` to a bare 2-tuple stub are
    unaffected — the full-pool breakdown is computed here on the real signal.

    Purely additive: it does not score, select, sort-differently, or mutate the
    pool. Keys per dict are :data:`COMPONENT_KEYS`.
    """
    scored = score_candidates(candidates, cfg, return_components=True)
    return [comps for _mc, _final, comps in scored]


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


# ---------------------------------------------------------------------------
# F1 — active_fields (§9 subset-schema extraction)
# ---------------------------------------------------------------------------


def active_fields(
    candidates: list[MergedCandidate],
    template_cls: type,
    cfg: "RetrievalProfile",
) -> list[str]:
    """Return the list of field names to include in the LLM extraction prompt.

    Opt-in: when ``cfg.subset_schema_extraction`` is ``False`` (the default),
    returns ALL field names of ``template_cls`` in schema order — byte-identical
    to the current behaviour.

    When ``True``, computes the active set as:

        active = (union of supported_field_hints across all candidates)
               ∪ identity fields   (from template_cls.model_config['graph_id_fields'])
               ∪ required fields   (template_cls.model_fields[name].is_required())

    Only fields with ZERO evidence that are also NOT identity and NOT required
    are dropped.  This is deliberately conservative — it protects recall.

    Returns the active field names **in schema order** (the insertion order of
    ``template_cls.model_fields``).

    Identity fields are derived from ``model_config.get('graph_id_fields', [])``
    — no hardcoded field names.  Required fields are derived from
    ``model_fields[name].is_required()`` — no hardcoded field names.  If
    ``graph_id_fields`` is absent from ``model_config``, no identity protection
    is applied and the function does not raise.
    """
    all_field_names: list[str] = list(template_cls.model_fields.keys())

    # Opt-in guard: off → no-op, return all fields in schema order.
    if not cfg.subset_schema_extraction:
        return all_field_names

    # Build the "always-keep" set from model metadata (no hardcoded names).
    identity_fields: set[str] = set(
        template_cls.model_config.get("graph_id_fields") or []
    )
    required_fields: set[str] = {
        name
        for name, fi in template_cls.model_fields.items()
        if fi.is_required()
    }
    always_keep = identity_fields | required_fields

    # Union of evidenced field names across all candidates.
    evidenced: set[str] = set()
    for mc in candidates:
        evidenced |= mc.supported_field_hints

    # Active = evidenced ∪ always_keep, filtered to fields that exist in the schema.
    active: set[str] = (evidenced | always_keep) & set(all_field_names)

    # Return in schema order.
    return [name for name in all_field_names if name in active]
