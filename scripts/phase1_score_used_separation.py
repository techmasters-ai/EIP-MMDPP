"""Phase-1 go/no-go: router-score (score, used) SEPARATION analysis.

Question answered (OFFLINE, deterministic, read-only):
    For each routed extraction pass, do the router's per-chunk relevance
    scores SEPARATE the chunks that were actually USED (a fact traced to
    them via lineage) from the UNUSED ones — well enough that a score
    THRESHOLD (or dynamic-k) preserves recall?

        YES  → a threshold is viable (GO).  Emit the recommended rule.
        NO   → fall back to tuned per-pass top_k (NO-GO).  Emit per-pass k.

Method
------
From an existing full-doc run we have, per chunk:
  * SCORES — recomputed OFFLINE & deterministically (rerank is order-
    invariant; cosine is a dot product; final via score_candidates over the
    full pool).  Three scores per chunk:
        cosine_score   — dense bge-m3 cosine  ∈ [-1, 1]   (absolute)
        reranker_score — cross-encoder logit              (absolute)
        final_score    — C5 composite (rerank_norm is min-max'd WITHIN the
                         pool → pool-relative, NOT a portable absolute
                         threshold; reported for completeness)
  * USED — from per-pass response provenance (``pipeline_pass_outputs``),
    joined to the ``ExtractionChunk`` index by chunk_index (direct) or
    source_refs ∩ self_refs (fallback).

Recall of any threshold / dynamic-k rule is then a pure SET OPERATION over
the (score, used) table.  All scoring is REUSED from the production services
(``score_candidates``, ``lexical_hit_counts``, ``pattern_hit_counts``,
``build_retrieval_profile``, the embedding + reranker clients) — nothing is
reimplemented.

This module is import-safe (no DB / model work at import time): the pure
analysis functions (``build_used_table``, ``auroc``, ``recall_vs_threshold``,
``analyze_separation``) are unit-tested with synthetic fixtures; the DB /
embedding / rerank I/O is isolated behind thin wrappers (``_fetch_chunks``,
``_embed_query``, ``_rerank_scores``, ``_pass_signals``, ``_fetch_pass_outputs``)
that are only invoked by ``main()``.

Usage (read-only on the live DBs)::

    python scripts/phase1_score_used_separation.py \
        --run-id 5da1a210-0322-401e-a2f2-8a7cab68ca00 \
        --doc-id ddaa9e36-2854-47c3-bc94-ff38d531dafd

Outputs (under reports/collection/):
    phase1_score_used_<run8>.csv   — the (pass, chunk, 3 scores, used*) table
    phase1_separation_<run8>.json  — the full analysis report
    phase1_separation_<run8>.md    — the human-readable RESULT REPORT
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# used_source tags
# ---------------------------------------------------------------------------
USED_SOURCE_DIRECT = "direct"        # chunk_index ∈ provenance.chunk_indexes (in range)
USED_SOURCE_FALLBACK = "fallback"    # rescued via source_refs ∩ self_refs
USED_SOURCE_VALUE_FILL = "value_fill"  # pass had EMPTY provenance (down-weight / flag)

# Default separation floor: a score must reach this AUROC to count as
# "separating used from unused".  Below it the threshold buys nothing.
DEFAULT_AUROC_SEP_FLOOR = 0.75

# rel_max dynamic-k alpha grid (score >= alpha * pool_max).
REL_MAX_ALPHAS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)

SCORE_FIELDS = ("cosine_score", "reranker_score", "final_score")

# Pass-level "did this pass use ANY chunk" flag, populated by build_used_table.
# Exposed module-level so a caller / test can introspect the value-fill passes.
PASS_USED: dict[str, bool] = {}


# ===========================================================================
# Row models
# ===========================================================================

@dataclass
class ScoreUsedRow:
    """One (pass, chunk) row: the three offline scores + the lineage labels."""
    pass_name: str
    chunk_index: int
    cosine_score: float
    reranker_score: float
    final_score: float
    used: bool
    used_field_level: bool
    used_source: str
    self_ref: str = ""
    token_count: int = 0
    page_number: int | None = None


# ===========================================================================
# build_used_table — lineage join (CORRECTNESS CRITICAL)
# ===========================================================================

def _as_int_set(values: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for v in values or ():
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _row_used_chunk_indices(
    prov_row: Mapping[str, Any],
    avail_ci: set[int],
    ref_to_ci: Mapping[str, set[int]],
) -> tuple[set[int], str | None]:
    """Resolve ONE provenance row to ExtractionChunk indices + the source tag.

    direct  : chunk_indexes ∩ available index set (preferred).
    fallback: ONLY when direct yields nothing — source_refs ∩ self_refs via
              the element-ref → chunk_index map.  Applying fallback solely on
              direct-miss keeps the USED set tight (merged chunks SHARE
              source_refs, so unconditional fallback over-expands USED).

    Returns (chunk_index_set, used_source_tag | None).
    """
    cidx = _as_int_set(prov_row.get("chunk_indexes"))
    # back-compat: scalar chunk_index when chunk_indexes absent
    if not cidx and prov_row.get("chunk_index") is not None:
        cidx = _as_int_set([prov_row.get("chunk_index")])

    direct = cidx & avail_ci
    if direct:
        return direct, USED_SOURCE_DIRECT

    # fallback via element-ref overlap (self_refs are element refs that match
    # ExtractionChunk.source_refs entries).
    self_refs = prov_row.get("self_refs") or []
    fb: set[int] = set()
    for ref in self_refs:
        fb |= ref_to_ci.get(ref, set())
    if fb:
        return fb, USED_SOURCE_FALLBACK
    return set(), None


def build_used_table(
    pass_outputs: Mapping[str, Mapping[str, Any]],
    chunks_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[ScoreUsedRow]:
    """Per-(pass, chunk) USED labels from per-pass response provenance.

    Parameters
    ----------
    pass_outputs:
        {pass_name: {"provenance": [...], "field_provenance": [...]}} — the
        parsed ``extract_pass_response_json`` blocks from
        ``pipeline_pass_outputs`` (one row per (run, pass_name)).
    chunks_by_pass:
        {pass_name: [ExtractionChunk row dicts]} with keys
        self_ref / chunk_index / source_refs (+ optional token_count,
        page_number).  Usually the SAME index for every pass of a run.

    Emits exactly one ScoreUsedRow per (pass, chunk) with placeholder scores
    (0.0); the score table is joined in by the runner.  Sets:
        used              — chunk traced to by row-level OR field-level prov.
        used_field_level  — chunk traced to by field_provenance specifically.
        used_source       — direct / fallback (on used rows) or value_fill
                            (every row of an EMPTY-provenance pass).
    Also updates module-level ``PASS_USED[pass_name]``.
    """
    rows: list[ScoreUsedRow] = []

    for pass_name, chunks in chunks_by_pass.items():
        po = pass_outputs.get(pass_name, {}) or {}
        provenance = po.get("provenance") or []
        field_provenance = po.get("field_provenance") or []
        has_any_provenance = bool(provenance) or bool(field_provenance)

        # element-ref → {chunk_index}, and the available index set.
        avail_ci: set[int] = set()
        ref_to_ci: dict[str, set[int]] = {}
        ci_meta: dict[int, Mapping[str, Any]] = {}
        for c in chunks:
            ci = int(c["chunk_index"])
            avail_ci.add(ci)
            ci_meta[ci] = c
            for ref in c.get("source_refs") or []:
                ref_to_ci.setdefault(ref, set()).add(ci)

        # Resolve USED sets.
        used_direct_or_fb: dict[int, str] = {}     # chunk_index -> source tag
        for pr in provenance:
            cis, tag = _row_used_chunk_indices(pr, avail_ci, ref_to_ci)
            for ci in cis:
                # direct wins over fallback if a chunk appears via both.
                if used_direct_or_fb.get(ci) != USED_SOURCE_DIRECT:
                    used_direct_or_fb[ci] = tag

        used_field: dict[int, str] = {}
        for fpr in field_provenance:
            cis, tag = _row_used_chunk_indices(fpr, avail_ci, ref_to_ci)
            for ci in cis:
                if used_field.get(ci) != USED_SOURCE_DIRECT:
                    used_field[ci] = tag

        pass_used_any = bool(used_direct_or_fb) or bool(used_field)
        PASS_USED[pass_name] = pass_used_any

        for ci in sorted(avail_ci):
            meta = ci_meta[ci]
            row_used = ci in used_direct_or_fb
            field_used = ci in used_field
            used = row_used or field_used
            if not has_any_provenance:
                src = USED_SOURCE_VALUE_FILL
            elif used:
                # prefer the row-level source tag, else field-level
                src = used_direct_or_fb.get(ci) or used_field.get(ci) or USED_SOURCE_DIRECT
            else:
                src = USED_SOURCE_DIRECT  # unused rows: tag is irrelevant; keep direct sentinel
            rows.append(ScoreUsedRow(
                pass_name=pass_name,
                chunk_index=ci,
                cosine_score=0.0,
                reranker_score=0.0,
                final_score=0.0,
                used=used,
                used_field_level=field_used,
                used_source=src,
                self_ref=str(meta.get("self_ref", f"chunk_{ci}")),
                token_count=int(meta.get("token_count", 0) or 0),
                page_number=meta.get("page_number"),
            ))

    return rows


# ===========================================================================
# auroc + recall_vs_threshold — primitives
# ===========================================================================

def auroc(scores: Sequence[float], used: Sequence[bool]) -> float:
    """AUROC of (used label, score predictor) via the Mann-Whitney U / midrank
    formula.  Handles ties exactly (matches sklearn.roc_auc_score).

    Returns 0.5 when one class is empty (undefined → no separation).
    """
    pos = [s for s, u in zip(scores, used) if u]
    neg = [s for s, u in zip(scores, used) if not u]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # midranks over the combined sample
    combined = sorted(zip(scores, range(len(scores))), key=lambda t: t[0])
    ranks = [0.0] * len(scores)
    i = 0
    n = len(combined)
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        # ranks i..j (1-based) share the average rank
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[combined[k][1]] = avg
        i = j + 1

    rank_sum_pos = sum(r for r, u in zip(ranks, used) if u)
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)


def recall_vs_threshold(
    scores: Sequence[float], used: Sequence[bool]
) -> list[dict[str, float]]:
    """Sweep tau over the candidate thresholds and report recall + cost.

    A KEEP rule is ``score >= tau``.  For each tau:
        recall(tau) = |{used : score >= tau}| / |used|
        cost(tau)   = |{score >= tau}| / N          (fraction of chunks kept)

    Candidate thresholds = the distinct scores plus a sentinel just below the
    minimum (keep-everything).  Sorted ascending by tau.
    """
    n = len(scores)
    n_used = sum(1 for u in used if u)
    if n == 0:
        return []
    uniq = sorted(set(scores))
    sentinel = (uniq[0] - 1.0) if uniq else 0.0
    taus = [sentinel] + uniq
    out: list[dict[str, float]] = []
    for tau in taus:
        kept = [(s, u) for s, u in zip(scores, used) if s >= tau]
        kept_used = sum(1 for _s, u in kept if u)
        recall = (kept_used / n_used) if n_used else 1.0
        cost = len(kept) / n
        out.append({"threshold": float(tau), "recall": recall, "cost": cost,
                    "kept": float(len(kept)), "kept_used": float(kept_used)})
    return out


# ===========================================================================
# analyze_separation — the GO/NO-GO decision
# ===========================================================================

@dataclass
class Recommendation:
    """A concrete keep-rule: threshold (or dyn-k param) + its outcome."""
    rule: str                       # "threshold" | "rel_max" | "gap" | "knee"
    threshold: float | None
    param: float | None             # alpha for rel_max; None otherwise
    recall_achieved: float
    fraction_kept: float
    chunks_kept: int


@dataclass
class PerPassReport:
    pass_name: str
    routed: bool
    n_chunks: int
    n_used: int
    used_fraction: float
    value_fill: bool                # pass had empty provenance (USED unknown)
    auroc: dict[str, float]
    dist: dict[str, dict[str, float]]              # per-score used/unused mean+median
    recall_curve: dict[str, list[dict[str, float]]]
    recommendation: dict[str, Recommendation | None]
    dynamic_k: dict[str, dict[str, Recommendation]]
    fallback_top_k: int | None      # recall-preserving per-pass k (NO-GO path)
    best_score: str
    best_auroc: float
    separates: bool


@dataclass
class SeparationReport:
    verdict: str                    # "GO" | "NO-GO"
    recommended_rule: dict[str, Any] | None
    auroc_sep_floor: float
    recall_floor: float
    per_pass: dict[str, PerPassReport]
    routed_passes: list[str]
    notes: list[str] = field(default_factory=list)


def _dist_stats(scores: Sequence[float], used: Sequence[bool]) -> dict[str, float]:
    u = sorted(s for s, f in zip(scores, used) if f)
    n = sorted(s for s, f in zip(scores, used) if not f)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def _median(xs):
        if not xs:
            return float("nan")
        m = len(xs)
        return xs[m // 2] if m % 2 else (xs[m // 2 - 1] + xs[m // 2]) / 2.0

    return {
        "used_mean": _mean(u), "used_median": _median(u),
        "used_min": (u[0] if u else float("nan")),
        "unused_mean": _mean(n), "unused_median": _median(n),
        "unused_max": (n[-1] if n else float("nan")),
    }


def _threshold_recommendation(
    curve: list[dict[str, float]], n_chunks: int, recall_floor: float
) -> Recommendation | None:
    """Smallest-fraction-kept threshold whose recall >= floor.

    Tie-break: among thresholds meeting the floor, pick min cost, then max
    recall, then highest threshold (most selective).
    """
    qualifying = [p for p in curve if p["recall"] >= recall_floor - 1e-12]
    if not qualifying:
        return None
    best = min(qualifying, key=lambda p: (p["cost"], -p["recall"], -p["threshold"]))
    return Recommendation(
        rule="threshold",
        threshold=best["threshold"],
        param=None,
        recall_achieved=best["recall"],
        fraction_kept=best["cost"],
        chunks_kept=int(round(best["cost"] * n_chunks)),
    )


def _eval_keepset(
    kept_idx: set[int], used: Sequence[bool], n_chunks: int
) -> tuple[float, float, int]:
    n_used = sum(1 for u in used if u)
    kept_used = sum(1 for i in kept_idx if used[i])
    recall = (kept_used / n_used) if n_used else 1.0
    return recall, (len(kept_idx) / n_chunks if n_chunks else 0.0), len(kept_idx)


def _dynamic_k_rules(
    scores: Sequence[float], used: Sequence[bool]
) -> dict[str, Recommendation]:
    """rel_max / gap / knee dynamic-k rules, each evaluated for recall + cost.

    Operates on the score vector sorted descending.  Each rule selects a KEEP
    prefix of that ordering; recall/cost computed against the used labels.
    """
    n = len(scores)
    out: dict[str, Recommendation] = {}
    if n == 0:
        return out
    order = sorted(range(n), key=lambda i: -scores[i])
    sorted_scores = [scores[i] for i in order]
    smax = sorted_scores[0]
    n_used = sum(1 for u in used if u)

    # --- rel_max: keep score >= alpha*max ; pick the alpha that first reaches
    #     full recall (or best recall), preferring the SMALLEST keepset.
    best_relmax: Recommendation | None = None
    for alpha in REL_MAX_ALPHAS:
        thr = alpha * smax if smax > 0 else smax
        kept_idx = {i for i in range(n) if scores[i] >= thr}
        recall, cost, k = _eval_keepset(kept_idx, used, n)
        cand = Recommendation("rel_max", float(thr), float(alpha), recall, cost, k)
        if best_relmax is None:
            best_relmax = cand
        else:
            # prefer: higher recall, then smaller keepset.
            if (cand.recall_achieved, -cand.fraction_kept) > (
                best_relmax.recall_achieved, -best_relmax.fraction_kept
            ):
                best_relmax = cand
    if best_relmax is not None:
        out["rel_max"] = best_relmax

    # --- gap: cut after the largest successive drop in the sorted-desc scores.
    if n >= 2:
        drops = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
        cut = max(range(len(drops)), key=lambda i: drops[i])  # keep ranks 0..cut
        kept_idx = set(order[: cut + 1])
        recall, cost, k = _eval_keepset(kept_idx, used, n)
        thr = sorted_scores[cut]
        out["gap"] = Recommendation("gap", float(thr), None, recall, cost, k)
    else:
        recall, cost, k = _eval_keepset(set(range(n)), used, n)
        out["gap"] = Recommendation("gap", float(sorted_scores[0]), None, recall, cost, k)

    # --- knee: max perpendicular distance from the chord of the sorted-desc
    #     curve (Kneedle-style).  Keep up to and including the knee rank.
    if n >= 3:
        x0, y0 = 0.0, sorted_scores[0]
        x1, y1 = float(n - 1), sorted_scores[-1]
        dx, dy = (x1 - x0), (y1 - y0)
        denom = math.hypot(dx, dy) or 1.0
        best_i, best_d = 0, -1.0
        for i in range(n):
            # distance from point (i, score_i) to the chord line
            d = abs(dy * i - dx * (sorted_scores[i] - y0)) / denom
            if d > best_d:
                best_d, best_i = d, i
        kept_idx = set(order[: best_i + 1])
        recall, cost, k = _eval_keepset(kept_idx, used, n)
        out["knee"] = Recommendation("knee", float(sorted_scores[best_i]), None, recall, cost, k)
    else:
        recall, cost, k = _eval_keepset(set(range(n)), used, n)
        out["knee"] = Recommendation("knee", float(sorted_scores[0]), None, recall, cost, k)

    return out


def _recall_preserving_top_k(
    scores: Sequence[float], used: Sequence[bool], recall_floor: float
) -> int | None:
    """Smallest k such that the top-k chunks by this score cover recall_floor
    of the used set.  This is the per-pass tuned top_k for the NO-GO fallback.
    """
    n = len(scores)
    n_used = sum(1 for u in used if u)
    if n_used == 0:
        return None
    order = sorted(range(n), key=lambda i: -scores[i])
    target = math.ceil(recall_floor * n_used - 1e-9)
    seen = 0
    for k, i in enumerate(order, start=1):
        if used[i]:
            seen += 1
            if seen >= target:
                return k
    return n


def analyze_separation(
    table: Sequence[ScoreUsedRow],
    *,
    recall_floor: float = 1.0,
    auroc_sep_floor: float = DEFAULT_AUROC_SEP_FLOOR,
    keeps_all_fraction: float = 0.95,
    routed_passes: Sequence[str] | None = None,
) -> SeparationReport:
    """Per-pass separation analysis + the bottom-line GO / NO-GO verdict.

    For each pass, for each of {cosine, reranker, final}:
      (a) used-vs-unused distribution stats
      (b) AUROC(used, score)
      (c) recall-vs-threshold curve
      (d) dynamic-k rules (rel_max / gap / knee)
      (e) the recommendation: min-chunks-kept threshold holding recall >= floor
      (f) the FALLBACK DETECTOR.

    A pass "separates" iff its best AUROC >= auroc_sep_floor AND its best-score
    recall-preserving threshold keeps < keeps_all_fraction of chunks (i.e. the
    threshold actually buys a reduction).  Otherwise → fallback_top_k.

    The DOC-LEVEL verdict is GO iff EVERY routed pass with a known USED set
    (non value-fill) separates; else NO-GO.  Passes whose USED is unknown
    (value_fill / empty provenance) cannot vote GO and are flagged.
    """
    routed = set(routed_passes) if routed_passes is not None else None
    by_pass: dict[str, list[ScoreUsedRow]] = {}
    for r in table:
        by_pass.setdefault(r.pass_name, []).append(r)

    per_pass: dict[str, PerPassReport] = {}
    for pass_name, rows in by_pass.items():
        n_chunks = len(rows)
        used = [r.used for r in rows]
        n_used = sum(1 for u in used if u)
        is_value_fill = all(r.used_source == USED_SOURCE_VALUE_FILL for r in rows)
        is_routed = (pass_name in routed) if routed is not None else True

        auroc_by: dict[str, float] = {}
        dist_by: dict[str, dict[str, float]] = {}
        curve_by: dict[str, list[dict[str, float]]] = {}
        rec_by: dict[str, Recommendation | None] = {}
        dynk_by: dict[str, dict[str, Recommendation]] = {}
        for sf in SCORE_FIELDS:
            scores = [getattr(r, sf) for r in rows]
            auroc_by[sf] = auroc(scores, used)
            dist_by[sf] = _dist_stats(scores, used)
            curve = recall_vs_threshold(scores, used)
            curve_by[sf] = curve
            rec_by[sf] = _threshold_recommendation(curve, n_chunks, recall_floor)
            dynk_by[sf] = _dynamic_k_rules(scores, used)

        # best score = highest AUROC (deterministic tie-break by SCORE_FIELDS order)
        best_score = max(SCORE_FIELDS, key=lambda sf: (auroc_by[sf], -SCORE_FIELDS.index(sf)))
        best_auroc = auroc_by[best_score]

        # does the best score's recall-preserving threshold actually reduce?
        best_rec = rec_by[best_score]
        threshold_reduces = (
            best_rec is not None and best_rec.fraction_kept < keeps_all_fraction
        )
        separates = (
            (not is_value_fill)
            and n_used > 0
            and best_auroc >= auroc_sep_floor
            and threshold_reduces
        )

        # fallback top_k from the best score (always computed; used on NO-GO).
        fb_scores = [getattr(r, best_score) for r in rows]
        fallback_top_k = (
            None if (is_value_fill or n_used == 0)
            else _recall_preserving_top_k(fb_scores, used, recall_floor)
        )

        per_pass[pass_name] = PerPassReport(
            pass_name=pass_name,
            routed=is_routed,
            n_chunks=n_chunks,
            n_used=n_used,
            used_fraction=(n_used / n_chunks if n_chunks else 0.0),
            value_fill=is_value_fill,
            auroc=auroc_by,
            dist=dist_by,
            recall_curve=curve_by,
            recommendation=rec_by,
            dynamic_k=dynk_by,
            fallback_top_k=fallback_top_k,
            best_score=best_score,
            best_auroc=best_auroc,
            separates=separates,
        )

    # ----- doc-level verdict -----
    voting = [
        p for name, p in per_pass.items()
        if p.routed and not p.value_fill and p.n_used > 0
    ]
    notes: list[str] = []
    if not voting:
        verdict = "NO-GO"
        notes.append(
            "No routed pass had a known USED set (all value-fill / empty "
            "provenance); cannot establish separation → default to tuned top_k."
        )
        recommended_rule = None
    elif all(p.separates for p in voting):
        verdict = "GO"
        # recommended rule = the per-pass best-score threshold (absolute scores
        # preferred: cosine/reranker are portable; final_score is pool-relative).
        recommended_rule = {
            "kind": "per_pass_threshold",
            "per_pass": {
                p.pass_name: {
                    "score": _portable_best(p),
                    "threshold": _rec_for(p, _portable_best(p)).threshold,
                    "recall_achieved": _rec_for(p, _portable_best(p)).recall_achieved,
                    "fraction_kept": _rec_for(p, _portable_best(p)).fraction_kept,
                    "chunks_kept": _rec_for(p, _portable_best(p)).chunks_kept,
                }
                for p in voting
            },
        }
    else:
        verdict = "NO-GO"
        notes.append(
            "At least one routed pass's scores do not separate used from "
            "unused (AUROC below floor and/or no recall-preserving threshold "
            "reduces chunk count) → use tuned per-pass top_k."
        )
        recommended_rule = None

    routed_list = sorted(
        name for name, p in per_pass.items() if p.routed
    )
    return SeparationReport(
        verdict=verdict,
        recommended_rule=recommended_rule,
        auroc_sep_floor=auroc_sep_floor,
        recall_floor=recall_floor,
        per_pass=per_pass,
        routed_passes=routed_list,
        notes=notes,
    )


def _portable_best(p: PerPassReport) -> str:
    """Prefer an ABSOLUTE-scored channel (cosine/reranker) for the recommended
    threshold — final_score's rerank_norm is pool-relative and not portable.
    Among the portable channels, pick the higher-AUROC one; only fall back to
    final_score if neither portable channel meets the separation floor.
    """
    portable = ("reranker_score", "cosine_score")
    cand = max(portable, key=lambda sf: p.auroc[sf])
    if p.auroc[cand] >= p.best_auroc - 1e-9 or p.auroc["final_score"] < p.auroc[cand]:
        return cand
    return p.best_score


def _rec_for(p: PerPassReport, score_field: str) -> Recommendation:
    rec = p.recommendation.get(score_field)
    if rec is not None:
        return rec
    # synthesize a keep-all rec if the chosen channel had no qualifying tau
    return Recommendation("threshold", None, None, 1.0, 1.0, p.n_chunks)


# ===========================================================================
# I/O wrappers (only invoked by main(); kept thin + monkeypatchable)
# ===========================================================================

def _normalize_vec(v: list[float]) -> list[float]:
    import numpy as np
    a = np.asarray(v, dtype=np.float32)
    nrm = float(np.linalg.norm(a))
    return (a / (nrm + 1e-12)).tolist()


def _embed_query(text: str) -> list[float]:
    """Embed a query string (bge-m3 via the production embedding client)."""
    from app.services.embedding import embed_texts
    return embed_texts([text], query=True)[0]


def _rerank_scores(query: str, cands: list[dict[str, Any]]) -> dict[str, float]:
    """Full-corpus rerank (order-invariant); return {chunk_index_str: logit}.

    Each candidate must carry ``content_text`` and ``chunk_index``.
    """
    from app.services import reranker as rrk
    scored = rrk.rerank(query=query, candidates=cands, top_k=10 ** 6)
    out: dict[str, float] = {}
    for c in scored:
        ci = c.get("chunk_index")
        if ci is None:
            continue
        out[str(ci)] = float(c.get("reranker_score", 0.0))
    return out


def _pass_signals(pass_name: str):
    """Build PassRetrievalSignals (entity_query + field_queries) for a pass
    from the merged-bundle manifest + pydantic schema.  Reuses the production
    walker; raises if the pass is unknown.
    """
    import importlib
    from app.services.ontology_bundles import load_bundle_manifest
    from app.services.extraction_query_builder import build_retrieval_profile

    bundle_key = os.environ.get("PHASE1_BUNDLE_KEY", "air_defense_v3_merged_v1")
    manifest = load_bundle_manifest(bundle_key)
    by_name = {p.name: p for p in manifest.passes}
    pd = by_name.get(pass_name)
    if pd is None:
        raise KeyError(f"pass {pass_name!r} not in bundle {bundle_key!r}")
    mod = importlib.import_module(f"ontology_bundles.{bundle_key}.{pd.module}")
    tcls = getattr(mod, pd.template_class)
    return build_retrieval_profile(pd, tcls)


def _fetch_chunks(run_id: str) -> list[dict[str, Any]]:
    """Read every embedded ExtractionChunk row for ``run_id`` (read-only)."""
    import asyncio
    from app.db.session import get_graph_store

    async def _q():
        store = get_graph_store()
        rows = await store._client.query(
            store._database, "sql",
            (
                "SELECT self_ref, chunk_text, embedding, chunk_index, "
                "source_refs, token_count, page_number "
                "FROM ExtractionChunk WHERE pipeline_run_id = :run_id "
                "ORDER BY chunk_index ASC"
            ),
            {"run_id": run_id},
        )
        return [dict(r) for r in rows if r.get("embedding")]

    return asyncio.run(_q())


def _sync_database_url() -> str:
    """Resolve the read-only postgres DSN.

    Reuses the app's own ``settings.sync_database_url`` (so it is correct in any
    environment — container hostname/port included). A ``PHASE1_DATABASE_URL``
    env var overrides it (e.g. host-side localhost:5437 when not in-container).
    """
    override = os.environ.get("PHASE1_DATABASE_URL")
    if override:
        return override
    from app.config import get_settings
    return get_settings().sync_database_url


def _fetch_pass_outputs(run_id: str) -> dict[str, dict[str, Any]]:
    """Read per-pass response provenance from ``ingest.pipeline_pass_outputs``
    (read-only).  Returns {pass_name: {"provenance": [...],
    "field_provenance": [...], "primary_entities": int}}.

    Uses the app's configured sync engine (SQLAlchemy) — no hand-rolled DSN.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(_sync_database_url())
    out: dict[str, dict[str, Any]] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT pass_name, "
                    "       COALESCE(extract_pass_response_json->'provenance','[]'::jsonb) AS prov, "
                    "       COALESCE(extract_pass_response_json->'field_provenance','[]'::jsonb) AS fprov, "
                    "       COALESCE(primary_entities_extracted, 0) AS prim "
                    "FROM ingest.pipeline_pass_outputs "
                    "WHERE pipeline_run_id = :run_id "
                    "ORDER BY pass_name, attempt"
                ),
                {"run_id": run_id},
            ).fetchall()
        for pass_name, prov, fprov, prim in rows:
            # latest attempt wins (ORDER BY attempt → last seen). jsonb columns
            # come back as already-parsed python objects under psycopg2.
            out[pass_name] = {
                "provenance": _as_list(prov),
                "field_provenance": _as_list(fprov),
                "primary_entities": int(prim or 0),
            }
    finally:
        engine.dispose()
    return out


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return []
    return list(v) if isinstance(v, (list, tuple)) else []


# ===========================================================================
# build_chunk_score_table — per-(pass, chunk) {cosine, reranker, final}
# ===========================================================================

def _table_content_type(source_refs: Sequence[str]) -> str | None:
    """Mirror the merged-mode table flag: a chunk sourced from a #/tables/*
    element is a table chunk."""
    for r in source_refs or ():
        if "/tables/" in r:
            return "table"
    return None


def build_chunk_score_table(
    *,
    chunks_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    profile: Any = None,
) -> list[ScoreUsedRow]:
    """Per-(pass, chunk) rows with cosine_score / reranker_score / final_score
    for EVERY ExtractionChunk in every routed pass.

    cosine_score   : normalised(chunk_emb) · normalised(query_emb)
    reranker_score : cross-encoder logit (full-corpus rerank; order-invariant)
    final_score    : score_candidates(full pool, profile) — REUSES the C5
                     scorer with offline-computed lexical/pattern hit signals.

    The ``used*`` fields are left as placeholders here (join build_used_table
    in separately).  DB I/O is none — chunks are passed in; embedding / rerank
    / signal builders are the module-level wrappers (monkeypatchable in tests).
    """
    from app.services.extraction_candidate_scoring import MergedCandidate, score_candidates

    if profile is None:
        from app.services.ontology_bundles import RetrievalProfile
        profile = RetrievalProfile()
    pattern_limit = int(getattr(profile, "pattern_hit_limit", 50) or 50)

    from app.services.extraction_lexical_search import lexical_hit_counts, pattern_hit_counts

    out: list[ScoreUsedRow] = []
    for pass_name, chunks in chunks_by_pass.items():
        rows = [dict(c) for c in chunks if c.get("embedding")]
        if not rows:
            continue
        signals = _pass_signals(pass_name)
        query_text = getattr(signals, "entity_query", "") or ""
        field_queries = getattr(signals, "field_queries", ()) or ()

        # 1. cosine
        import numpy as np
        qv = np.asarray(_normalize_vec(_embed_query(query_text)), dtype=np.float32)
        emb = np.asarray([_normalize_vec(r["embedding"]) for r in rows], dtype=np.float32)
        cos = (emb @ qv).astype(float).tolist()

        # 2. reranker (full corpus)
        rer_cands = [
            {"content_text": r.get("chunk_text") or "", "chunk_index": r["chunk_index"]}
            for r in rows
        ]
        rer_map = _rerank_scores(query_text, rer_cands)

        # 3. lexical / pattern hit signals (offline, pure) → final via C5
        lex = lexical_hit_counts(rows, field_queries)
        pat = pattern_hit_counts(rows, field_queries, pattern_limit)

        c5_input: list[dict[str, Any]] = []
        mc_by_key: dict[str, str] = {}   # candidate_key -> chunk_index str
        for r in rows:
            key = str(r.get("vertex_id") or r.get("self_ref") or r["chunk_index"])
            ci = str(r["chunk_index"])
            lh = lex.get(key, {})
            ph = pat.get(key, {})
            mc = MergedCandidate(
                candidate_key=key,
                chunk_index=int(r["chunk_index"]),
                self_ref=str(r.get("self_ref", "")),
                chunk_text=r.get("chunk_text") or "",
                source_refs=list(r.get("source_refs") or []),
                token_count=int(r.get("token_count", 0) or 0),
                page_number=r.get("page_number"),
                # vector_score is NOT a score_candidates input (final_score uses
                # rerank+lexical+pattern+section+table-negative only); leave None.
                vector_score=None,
                field_scores={},
                alias_hits=int(lh.get("alias_hits", 0)),
                pattern_hits=int(ph.get("pattern_hits", 0)),
                negative_hits=int(lh.get("negative_hits", 0)),
                section_hits=0,
                content_type=_table_content_type(r.get("source_refs") or []),
                retrieval_sources=set(),
                supported_field_hints=set(),
            )
            entry: dict[str, Any] = {"merged_candidate": mc, "content_text": mc.chunk_text}
            rr = rer_map.get(ci)
            if rr is not None:
                entry["reranker_score"] = rr
            c5_input.append(entry)
            mc_by_key[key] = ci

        final_pairs = score_candidates(c5_input, profile)
        final_by_ci: dict[str, float] = {}
        for mc, fs in final_pairs:
            final_by_ci[str(mc.chunk_index)] = float(fs)

        cos_by_ci = {str(r["chunk_index"]): c for r, c in zip(rows, cos)}
        for r in rows:
            ci = str(r["chunk_index"])
            out.append(ScoreUsedRow(
                pass_name=pass_name,
                chunk_index=int(r["chunk_index"]),
                cosine_score=float(cos_by_ci.get(ci, 0.0)),
                reranker_score=float(rer_map.get(ci, 0.0)),
                final_score=float(final_by_ci.get(ci, 0.0)),
                used=False,
                used_field_level=False,
                used_source=USED_SOURCE_DIRECT,
                self_ref=str(r.get("self_ref", "")),
                token_count=int(r.get("token_count", 0) or 0),
                page_number=r.get("page_number"),
            ))
    return out


# ===========================================================================
# join + serialization
# ===========================================================================

def join_scores_and_used(
    score_rows: Sequence[ScoreUsedRow], used_rows: Sequence[ScoreUsedRow]
) -> list[ScoreUsedRow]:
    """Merge the score table (3 scores) with the used table (lineage labels)
    on (pass, chunk_index).  Scores come from score_rows; used* from used_rows.
    """
    used_idx = {(r.pass_name, r.chunk_index): r for r in used_rows}
    out: list[ScoreUsedRow] = []
    for s in score_rows:
        u = used_idx.get((s.pass_name, s.chunk_index))
        out.append(ScoreUsedRow(
            pass_name=s.pass_name,
            chunk_index=s.chunk_index,
            cosine_score=s.cosine_score,
            reranker_score=s.reranker_score,
            final_score=s.final_score,
            used=(u.used if u else False),
            used_field_level=(u.used_field_level if u else False),
            used_source=(u.used_source if u else USED_SOURCE_VALUE_FILL),
            self_ref=s.self_ref or (u.self_ref if u else ""),
            token_count=s.token_count or (u.token_count if u else 0),
            page_number=s.page_number,
        ))
    return out


def _write_csv(rows: Sequence[ScoreUsedRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pass_name", "chunk_index", "cosine_score", "reranker_score",
            "final_score", "used", "used_field_level", "used_source",
            "self_ref", "token_count", "page_number",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _report_to_jsonable(rep: SeparationReport) -> dict[str, Any]:
    def _pp(p: PerPassReport) -> dict[str, Any]:
        d = asdict(p)
        return d
    return {
        "verdict": rep.verdict,
        "recommended_rule": rep.recommended_rule,
        "auroc_sep_floor": rep.auroc_sep_floor,
        "recall_floor": rep.recall_floor,
        "routed_passes": rep.routed_passes,
        "notes": rep.notes,
        "per_pass": {k: _pp(v) for k, v in rep.per_pass.items()},
    }


def _render_markdown(rep: SeparationReport, run_id: str, doc_id: str,
                     n_runs: int) -> str:
    L: list[str] = []
    L.append("# Phase-1 go/no-go — router-score (score, used) separation\n")
    L.append(f"Run: `{run_id}`  Doc: `{doc_id}`  (A0 samples: {n_runs})\n")
    L.append(f"recall_floor = {rep.recall_floor}; AUROC separation floor = "
             f"{rep.auroc_sep_floor}\n")
    L.append(f"\n## BOTTOM LINE: **{rep.verdict}**\n")
    for n in rep.notes:
        L.append(f"- {n}\n")
    if rep.recommended_rule:
        L.append("\nRecommended rule (per-pass absolute-score threshold):\n\n")
        L.append("| pass | score | threshold | recall | frac_kept | chunks_kept |\n")
        L.append("|---|---|---|---|---|---|\n")
        for pn, r in rep.recommended_rule["per_pass"].items():
            L.append(f"| {pn} | {r['score']} | {r['threshold']:.4f} | "
                     f"{r['recall_achieved']:.2f} | {r['fraction_kept']:.2f} | "
                     f"{r['chunks_kept']} |\n")

    L.append("\n## Per-pass AUROC (used vs unused) for each score\n\n")
    L.append("| pass | routed | N | used | used% | cosine | reranker | final | best | separates |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for pn in sorted(rep.per_pass):
        p = rep.per_pass[pn]
        vf = " (value_fill)" if p.value_fill else ""
        L.append(
            f"| {pn}{vf} | {'Y' if p.routed else 'n'} | {p.n_chunks} | "
            f"{p.n_used} | {p.used_fraction*100:.0f}% | "
            f"{p.auroc['cosine_score']:.3f} | {p.auroc['reranker_score']:.3f} | "
            f"{p.auroc['final_score']:.3f} | {p.best_score.split('_')[0]} "
            f"({p.best_auroc:.3f}) | {'YES' if p.separates else 'no'} |\n"
        )

    L.append("\n## Recall-preserving threshold per pass (best score)\n\n")
    L.append("| pass | best_score | threshold | recall | frac_kept | "
             "chunks_kept | fallback_top_k |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for pn in sorted(rep.per_pass):
        p = rep.per_pass[pn]
        rec = p.recommendation.get(p.best_score)
        if rec and rec.threshold is not None:
            thr = f"{rec.threshold:.4f}"
            rc = f"{rec.recall_achieved:.2f}"
            fk = f"{rec.fraction_kept:.2f}"
            ck = str(rec.chunks_kept)
        else:
            thr = rc = fk = ck = "—"
        L.append(f"| {pn} | {p.best_score.split('_')[0]} | {thr} | {rc} | "
                 f"{fk} | {ck} | {p.fallback_top_k if p.fallback_top_k is not None else '—'} |\n")

    L.append("\n## Dynamic-k rules (best score per pass)\n\n")
    L.append("| pass | rule | recall | frac_kept | chunks_kept |\n")
    L.append("|---|---|---|---|---|\n")
    for pn in sorted(rep.per_pass):
        p = rep.per_pass[pn]
        for rule, rec in p.dynamic_k.get(p.best_score, {}).items():
            L.append(f"| {pn} | {rule} | {rec.recall_achieved:.2f} | "
                     f"{rec.fraction_kept:.2f} | {rec.chunks_kept} |\n")

    L.append("\n## Distribution (best score): used vs unused\n\n")
    L.append("| pass | used_mean | used_min | unused_mean | unused_max |\n")
    L.append("|---|---|---|---|---|\n")
    for pn in sorted(rep.per_pass):
        p = rep.per_pass[pn]
        d = p.dist[p.best_score]
        L.append(f"| {pn} | {d['used_mean']:.3f} | {d['used_min']:.3f} | "
                 f"{d['unused_mean']:.3f} | {d['unused_max']:.3f} |\n")
    return "".join(L)


# ===========================================================================
# main — the offline pipeline on a real run
# ===========================================================================

def _default_routed_passes(bundle_key: str) -> list[str]:
    """Routed = field_group passes (the only routed phase per the grounding)."""
    from app.services.ontology_bundles import load_bundle_manifest
    manifest = load_bundle_manifest(bundle_key)
    out = []
    for p in manifest.passes:
        phase = str(getattr(p, "phase", "") or "")
        if phase == "field_group":
            out.append(p.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--doc-id", default="")
    ap.add_argument("--bundle-key",
                    default=os.environ.get("PHASE1_BUNDLE_KEY", "air_defense_v3_merged_v1"))
    ap.add_argument("--recall-floor", type=float, default=1.0)
    ap.add_argument("--auroc-floor", type=float, default=DEFAULT_AUROC_SEP_FLOOR)
    ap.add_argument("--include-identity", action="store_true",
                    help="Also analyze identity passes (not routed; extra samples).")
    ap.add_argument("--out-dir", default="reports/collection")
    args = ap.parse_args()

    os.environ.setdefault("PHASE1_BUNDLE_KEY", args.bundle_key)
    run8 = args.run_id.split("-")[0]

    print(f"[phase1] fetching ExtractionChunk index for run {args.run_id} ...")
    chunks = _fetch_chunks(args.run_id)
    print(f"[phase1] chunks={len(chunks)} "
          f"(idx {min((c['chunk_index'] for c in chunks), default=-1)}.."
          f"{max((c['chunk_index'] for c in chunks), default=-1)})")

    print("[phase1] fetching pass outputs (provenance) ...")
    pass_outputs = _fetch_pass_outputs(args.run_id)
    for pn, po in sorted(pass_outputs.items()):
        print(f"   {pn:22} prov={len(po['provenance']):>3} "
              f"field_prov={len(po['field_provenance']):>3} "
              f"prim_entities={po['primary_entities']}")

    routed = _default_routed_passes(args.bundle_key)
    analyze_passes = list(routed)
    if args.include_identity:
        from app.services.ontology_bundles import load_bundle_manifest
        manifest = load_bundle_manifest(args.bundle_key)
        for p in manifest.passes:
            phase = str(getattr(p, "phase", "") or "")
            kind = str(getattr(p, "kind", "") or "")
            if phase == "identity" and p.name not in analyze_passes:
                analyze_passes.append(p.name)
    print(f"[phase1] routed passes: {routed}")
    print(f"[phase1] analyzing: {analyze_passes}")

    # same chunk index for every pass of a run
    chunks_by_pass = {pn: chunks for pn in analyze_passes
                      if pn in pass_outputs}

    print("[phase1] recomputing scores (cosine + reranker + final) ...")
    score_rows = build_chunk_score_table(chunks_by_pass=chunks_by_pass)

    print("[phase1] building USED table from lineage ...")
    used_rows = build_used_table(pass_outputs, chunks_by_pass)

    table = join_scores_and_used(score_rows, used_rows)

    print("[phase1] analyzing separation ...")
    report = analyze_separation(
        table,
        recall_floor=args.recall_floor,
        auroc_sep_floor=args.auroc_floor,
        routed_passes=routed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"phase1_score_used_{run8}.csv")
    json_path = os.path.join(args.out_dir, f"phase1_separation_{run8}.json")
    md_path = os.path.join(args.out_dir, f"phase1_separation_{run8}.md")

    _write_csv(table, csv_path)
    with open(json_path, "w") as f:
        json.dump(_report_to_jsonable(report), f, indent=2, default=str)
    md = _render_markdown(report, args.run_id, args.doc_id, n_runs=1)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n[phase1] wrote {csv_path}")
    print(f"[phase1] wrote {json_path}")
    print(f"[phase1] wrote {md_path}")
    print(f"\n========== VERDICT: {report.verdict} ==========")
    for n in report.notes:
        print(f"  - {n}")
    for pn in sorted(report.per_pass):
        p = report.per_pass[pn]
        print(f"  {pn:22} routed={p.routed} N={p.n_chunks} used={p.n_used} "
              f"AUROC[cos/rer/fin]={p.auroc['cosine_score']:.3f}/"
              f"{p.auroc['reranker_score']:.3f}/{p.auroc['final_score']:.3f} "
              f"best={p.best_score.split('_')[0]} separates={p.separates} "
              f"fallback_top_k={p.fallback_top_k}")


if __name__ == "__main__":
    main()
