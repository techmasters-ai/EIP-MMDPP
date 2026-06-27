"""C.10 / Phase 1 Task 8 — Offline retrieval-calibration sweep on merged chunks.

Mirrors the C.9a methodology (commit 4c7…) but indexes merged chunks via
``build_extraction_index_hybrid`` (Task 3) and sweeps the 9 narrowed passes
from ``ontology_bundles.air_defense_v3`` rather than the C.9a v1 pair.

Sweep dimensions (Task 8 spec):
    min_similarity   ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}      (7)
    top_n_candidates ∈ {25, 50, 75, 100}                               (4)
    top_k            ∈ {5, 10, 15, 20, 30}                             (5)
  → 140 cells per (doc, pass)

Per cell we record:
    - selected_chunk_count
    - expanded_ref_count     (∪ source_refs across selected chunks)
    - selected_chunk_token_estimate
    - gt_coverage            (recall@top_k vs GT_TOP_K reranker top-K)

GROUND-TRUTH: pseudo-GT = ``GT_TOP_K`` (default 30) chunks by reranker score
over the FULL corpus, max-permissive (no min_sim, no top_n cap). This is the
exact methodology used by ``scripts/c9a_calibration_sweep.py`` and
``scripts/c9a_sweep_v2.py`` — full reranker ordering over every embedded
chunk is the "truth" against which any (min_sim, top_n, top_k) config is
scored. Reranker is order-invariant, so we rerank once per (doc, pass) and
filter for each cell — cuts ~140 rerank calls to 1.

Efficiency: full-corpus rerank once per (doc, pass) = 18 rerank passes for
2 docs × 9 passes; each cell becomes a set-operation over the cached scores.

Usage (inside the worker container):
    docker exec eip-mmdpp-worker-1 python scripts/c10_phase1_merged_chunk_calibration.py

Output:
    /tmp/c10_phase1_sweep.csv   — raw per-cell rows
    /tmp/c10_phase1_sweep.md    — per-pass knee tables (for the handoff doc)
"""
from __future__ import annotations

import asyncio
import csv
import os
import time
import uuid
from itertools import product
from typing import Any

import numpy as np

from app.db.session import get_graph_store
from app.services import reranker as rrk
from app.services.embedding import embed_texts
from app.services.extraction_chunk_index import (
    build_extraction_index_hybrid,
    read_chunk_index,
    read_chunk_source_refs,
    read_chunk_token_count,
)
from app.services.extraction_query_builder import build_retrieval_query
from app.services.ontology_bundles import load_bundle_manifest
from app.workers.pipeline import _build_docling_document_json


# Documents under sweep. Dvina + SA-2_SR-71 mirror the C.9a v1 reference pair.
DOCS = [
    ("Dvina", "9c8e09c7-e39f-4359-92c0-46330158c73c"),
    ("SA-2",  "128e48f9-06f9-459a-b2f0-6d42bf62c42d"),
]

BUNDLE_KEY = "air_defense_v3"

# 9 narrowed passes (from the plan / manifest).
NARROWED_PASSES = [
    "radar_power_rf", "radar_antenna", "radar_timing", "radar_modulation",
    "missile_kinematics", "missile_guidance", "missile_airframe",
    "missile_speed_timing", "missile_propulsion",
]

# Task 8 sweep grid.
MIN_SIMS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TOP_NS   = [25, 50, 75, 100]
TOP_KS   = [5, 10, 15, 20, 30]

# Pseudo-GT depth (top-K reranker over full corpus = ground truth).
# C.9a v1 used GT_TOP_K=30 over ~300 per-element chunks (10% of corpus).
# Merged-mode chunks are ~7× denser (45 chunks for Dvina, 42 for SA-2),
# so we scale to GT_TOP_K=15 (~33% of corpus). Same total content-mass
# threshold; preserves the "chunks the LLM would benefit from seeing"
# semantic of the C.9a definition.
GT_TOP_K = 15


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

def index_doc_merged(document_id: str, label: str) -> str:
    """Build the merged-mode ExtractionChunk index for one document.

    Returns the synthetic sweep_run_id (uuid4-based) used as the
    pipeline_run_id key. Uses Task 3's ``build_extraction_index_hybrid``
    directly — bypasses Celery / worker dispatch.
    """
    doc_json = _build_docling_document_json(document_id)
    sweep_run_id = f"c10sweep-{label.lower()}-{uuid.uuid4().hex[:8]}"
    store = get_graph_store()
    diag = build_extraction_index_hybrid(
        doc_json,
        pipeline_run_id=sweep_run_id,
        document_id=document_id,
        store=store,
    )
    print(
        f"[{label}] indexed merged_chunks={diag.chunks_inserted} "
        f"mean_token={diag.mean_token_count} embed_ms={diag.embed_ms} "
        f"insert_ms={diag.insert_ms} run_id={sweep_run_id}"
    )
    return sweep_run_id


async def fetch_chunks(run_id: str) -> list[dict[str, Any]]:
    """Read every ExtractionChunk row for ``run_id`` with merged-mode columns."""
    store = get_graph_store()
    rows = await store._client.query(
        store._database, "sql",
        (
            "SELECT self_ref, chunk_text, embedding, chunk_index, "
            "source_refs, token_count "
            "FROM ExtractionChunk WHERE pipeline_run_id = :run_id "
            "ORDER BY chunk_index ASC, self_ref ASC"
        ),
        {"run_id": run_id},
    )
    return [r for r in rows if r.get("embedding")]


# ---------------------------------------------------------------------------
# Query construction (per pass)
# ---------------------------------------------------------------------------

def _resolve_template_class(bundle_key: str, pass_def) -> type:
    """Mirror app.api.v1.extraction_routing._resolve_template_class."""
    import importlib
    full_module = f"ontology_bundles.{bundle_key}.{pass_def.module}"
    mod = importlib.import_module(full_module)
    return getattr(mod, pass_def.template_class)


def build_pass_queries(bundle_key: str, pass_names: list[str]) -> dict[str, str]:
    """Return {pass_name: rendered_query_text} for the requested passes."""
    manifest = load_bundle_manifest(bundle_key)
    by_name = {p.name: p for p in manifest.passes}
    out: dict[str, str] = {}
    for pn in pass_names:
        pd = by_name.get(pn)
        if pd is None:
            raise KeyError(f"pass {pn!r} not in bundle {bundle_key!r}")
        tcls = _resolve_template_class(bundle_key, pd)
        out[pn] = build_retrieval_query(pd, tcls)
    return out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


async def sweep_one(doc_label: str, run_id: str, pass_name: str, query_text: str) -> list[dict]:
    """Sweep the 140-cell grid for one (doc, pass).

    NOTE: merged-mode rows share ``self_ref`` (the writer sets
    ``self_ref = source_refs[0]`` so legacy NOT NULL constraints hold), so
    chunk identity here is ``chunk_index`` — the only column guaranteed
    unique per merged ExtractionChunk vertex.
    """
    rows = await fetch_chunks(run_id)
    if not rows:
        print(f"  [{doc_label}/{pass_name}] no chunks — skipping")
        return []

    # Build parallel arrays. Use chunk_index as the unique key — self_ref
    # is non-unique across merged rows.
    chunk_indices = [read_chunk_index(r) for r in rows]
    self_refs = [r["self_ref"] for r in rows]
    texts = [r["chunk_text"] or "" for r in rows]
    source_refs_per_chunk = [read_chunk_source_refs(r) for r in rows]
    token_counts = [read_chunk_token_count(r) for r in rows]
    embs = _normalize(np.asarray([r["embedding"] for r in rows], dtype=np.float32))

    # Embed the query
    qv = _normalize(np.asarray(embed_texts([query_text], query=True)[0], dtype=np.float32))
    vec_scores = (embs @ qv).astype(float)

    # Full-corpus rerank (max-permissive, top_k=∞) — order-invariant cache.
    # Key each candidate by chunk_index (str) so the reranker output preserves
    # uniqueness even when self_ref values collide.
    cands = [
        {
            "content_text": t,
            "chunk_key": str(ci),
            "self_ref": sr,
            "vector_score": float(vs),
        }
        for ci, sr, t, vs in zip(chunk_indices, self_refs, texts, vec_scores)
    ]
    t0 = time.monotonic()
    full = rrk.rerank(query=query_text, candidates=cands, top_k=10**6)
    full_ms = (time.monotonic() - t0) * 1000.0

    rer_order = [c["chunk_key"] for c in full]  # chunk_index-keyed
    gt_top = set(rer_order[:GT_TOP_K])

    # Lookups keyed by chunk_index (as str to match chunk_key)
    keys = [str(ci) for ci in chunk_indices]
    token_by_key = dict(zip(keys, token_counts))
    refs_by_key = dict(zip(keys, source_refs_per_chunk))

    print(
        f"  [{doc_label}/{pass_name}] N={len(rows)} "
        f"vec_min={vec_scores.min():.3f} vec_max={vec_scores.max():.3f} "
        f"vec_median={float(np.median(vec_scores)):.3f} "
        f"rerank_ms={full_ms:.0f}"
    )

    results: list[dict] = []
    for min_sim, top_n, top_k in product(MIN_SIMS, TOP_NS, TOP_KS):
        # 1. min_sim filter (vector pre-filter)
        mask = vec_scores >= min_sim
        idx_above = np.where(mask)[0]
        # 2. top_n cap by vector score
        if len(idx_above) > top_n:
            order = np.argsort(-vec_scores[idx_above])[:top_n]
            cand_idx = idx_above[order]
        else:
            cand_idx = idx_above[np.argsort(-vec_scores[idx_above])]
        cand_set = {keys[i] for i in cand_idx}
        # 3. top_k by reranker score within cand_set
        ordered = [k for k in rer_order if k in cand_set]
        selected = ordered[:top_k]
        sel_set = set(selected)

        # Diagnostics (Task 8 spec)
        selected_chunk_count = len(sel_set)
        # expanded_ref_count = | ∪ source_refs of selected chunks |
        expanded = set()
        for k in sel_set:
            expanded.update(refs_by_key.get(k, []))
        expanded_ref_count = len(expanded)
        # selected_chunk_token_estimate = Σ token_count
        selected_chunk_token_estimate = sum(token_by_key.get(k, 0) for k in sel_set)
        # gt_coverage = recall of pseudo-GT top-GT_TOP_K
        cov = len(sel_set & gt_top) / len(gt_top) if gt_top else 0.0

        results.append({
            "doc": doc_label,
            "pass": pass_name,
            "min_similarity": min_sim,
            "top_n_candidates": top_n,
            "top_k": top_k,
            "candidate_count": len(cand_set),
            "selected_chunk_count": selected_chunk_count,
            "expanded_ref_count": expanded_ref_count,
            "selected_chunk_token_estimate": selected_chunk_token_estimate,
            "gt_coverage": cov,
            "full_doc_chunks": len(rows),
            "rerank_ms_full_corpus": full_ms,
        })
    return results


# ---------------------------------------------------------------------------
# Knee selection
# ---------------------------------------------------------------------------

def pick_knee(rows: list[dict]) -> dict | None:
    """Smallest (min_sim, top_n, top_k) config that achieves ≥95% of max cov.

    Tie-break: lower selected_chunk_token_estimate → lower selected_chunk_count
              → higher gt_coverage (more headroom).
    """
    if not rows:
        return None
    max_cov = max(r["gt_coverage"] for r in rows)
    if max_cov <= 0.0:
        return None
    target = 0.95 * max_cov
    qualified = [r for r in rows if r["gt_coverage"] >= target]
    if not qualified:
        return None
    qualified.sort(key=lambda r: (
        r["selected_chunk_token_estimate"],
        r["selected_chunk_count"],
        -r["gt_coverage"],
    ))
    return qualified[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    t_start = time.monotonic()

    # 1. Index both docs in merged mode.
    runs: dict[str, str] = {}
    for label, doc_id in DOCS:
        runs[label] = index_doc_merged(doc_id, label)

    # 2. Build per-pass query text once.
    queries = build_pass_queries(BUNDLE_KEY, NARROWED_PASSES)
    print(f"\nBuilt {len(queries)} query strings; min_len={min(len(q) for q in queries.values())} "
          f"max_len={max(len(q) for q in queries.values())}\n")

    # 3. Sweep every (doc, pass).
    all_rows: list[dict] = []
    for label, _ in DOCS:
        run_id = runs[label]
        for pn in NARROWED_PASSES:
            print(f"--- {label} / {pn} ---")
            rows = await sweep_one(label, run_id, pn, queries[pn])
            all_rows.extend(rows)

    # 4. CSV dump.
    csv_path = "/tmp/c10_phase1_sweep.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {csv_path} ({len(all_rows)} rows)")

    # 5. Per-pass knee selection + markdown.
    md = ["# C.10 Phase 1 — Merged-chunk calibration sweep\n"]
    md.append(f"Bundle: `{BUNDLE_KEY}`. Docs: " + ", ".join(f"`{d[0]}`" for d in DOCS) + ".\n")
    md.append(
        f"Sweep grid: min_similarity={MIN_SIMS}, top_n_candidates={TOP_NS}, "
        f"top_k={TOP_KS} → {len(MIN_SIMS)*len(TOP_NS)*len(TOP_KS)} cells/(doc,pass).\n"
    )
    md.append(f"Ground truth: top-{GT_TOP_K} chunks by full-corpus reranker score (max-permissive).\n")
    md.append("Knee criterion: smallest config (token-then-chunk-count) reaching ≥95% of max coverage.\n\n")

    md.append("## Per-pass knee selection\n\n")
    md.append("| doc | pass | min_sim | top_n | top_k | sel_chunks | exp_refs | sel_tokens | cov | max_cov |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    knees: dict[tuple[str, str], dict | None] = {}
    for label, _ in DOCS:
        for pn in NARROWED_PASSES:
            subset = [r for r in all_rows if r["doc"] == label and r["pass"] == pn]
            knee = pick_knee(subset)
            knees[(label, pn)] = knee
            max_cov = max((r["gt_coverage"] for r in subset), default=0.0)
            if knee is None:
                md.append(f"| {label} | {pn} | — | — | — | — | — | — | — | {max_cov*100:.0f}% |\n")
            else:
                md.append(
                    f"| {label} | {pn} | {knee['min_similarity']:.2f} | "
                    f"{knee['top_n_candidates']} | {knee['top_k']} | "
                    f"{knee['selected_chunk_count']} | {knee['expanded_ref_count']} | "
                    f"{knee['selected_chunk_token_estimate']} | "
                    f"{knee['gt_coverage']*100:.0f}% | {max_cov*100:.0f}% |\n"
                )

    md.append("\n## Per-pass aggregate (mean across docs)\n\n")
    md.append("Recommended values are the mean of the per-doc knee.\n\n")
    md.append("| pass | min_sim | top_n | top_k | mean_sel_chunks | mean_exp_refs | mean_sel_tokens |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    per_pass_recommend: dict[str, dict] = {}
    for pn in NARROWED_PASSES:
        kk = [knees[(d[0], pn)] for d in DOCS if knees[(d[0], pn)] is not None]
        if not kk:
            md.append(f"| {pn} | — | — | — | — | — | — |\n")
            continue
        # Pick the MAX of each retrieval param across docs (cover both docs).
        ms = max(k["min_similarity"] for k in kk)
        tn = max(k["top_n_candidates"] for k in kk)
        tk = max(k["top_k"] for k in kk)
        msc = sum(k["selected_chunk_count"] for k in kk) / len(kk)
        mer = sum(k["expanded_ref_count"] for k in kk) / len(kk)
        mst = sum(k["selected_chunk_token_estimate"] for k in kk) / len(kk)
        per_pass_recommend[pn] = {"min_sim": ms, "top_n": tn, "top_k": tk}
        md.append(
            f"| {pn} | {ms:.2f} | {tn} | {tk} | {msc:.1f} | {mer:.1f} | {mst:.0f} |\n"
        )

    # Diagnostic dump per (doc, pass).
    md.append("\n## Diagnostic: vector-score distribution\n\n")
    md.append("| doc | pass | min_vec | median_vec | max_vec | max_cov |\n")
    md.append("|---|---|---|---|---|---|\n")
    for label, _ in DOCS:
        for pn in NARROWED_PASSES:
            subset = [r for r in all_rows if r["doc"] == label and r["pass"] == pn]
            if not subset:
                continue
            max_cov = max(r["gt_coverage"] for r in subset)
            # Vector-score stats come from the per-cell candidate_count at min_sim
            # extremes; print {pass, doc} once.
            md.append(f"| {label} | {pn} | (see stdout) | (see stdout) | (see stdout) | {max_cov*100:.0f}% |\n")

    out_md = "/tmp/c10_phase1_sweep.md"
    with open(out_md, "w") as f:
        f.writelines(md)
    print(f"Wrote {out_md}")

    wall_min = (time.monotonic() - t_start) / 60.0
    print(f"\nDONE — {len(all_rows)} cells / wall={wall_min:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
