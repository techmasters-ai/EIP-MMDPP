"""C.9a — offline retrieval+rerank calibration sweep.

Replays the C.7g run's chunks + queries through a grid of (min_similarity,
top_n_candidates, top_k) configs and reports per-config metrics. NO LLM call.

Efficiency: cross-encoder is order-invariant, so we rerank the union of every
chunk once per (doc, pass), then derive each config's selected_refs by
filtering the pre-scored list. Cuts ~150 rerank calls to 4.

Usage (inside the api container):
    docker exec eip-mmdpp-api-1 python scripts/c9a_calibration_sweep.py
"""
from __future__ import annotations

import asyncio
import os
import time
from itertools import product
from typing import Any

import numpy as np

from app.db.session import get_graph_store
from app.services import reranker as rrk
from app.services.embedding import embed_texts
from app.services.table_normalization.tokens import count_bge_m3_tokens

DVINA_RUN = os.environ.get("DVINA_RUN", "4ed1c3e6-5976-4b07-9276-c3c23ede5929")
SA2_RUN = os.environ.get("SA2_RUN", "66b4a698-cc1b-439d-bc80-28f5e3309b6d")

GRIDS = {
    "radar_power_rf": {
        "top_n_candidates": [75, 100, 120, 150],
        "top_k": [25, 30, 40, 50],
        "min_similarity": [0.25, 0.30, 0.35],
    },
    "missile_kinematics": {
        "top_n_candidates": [100, 150, 200],
        "top_k": [40, 50, 60],
        "min_similarity": [0.25, 0.30, 0.35],
    },
}

# Ground-truth top-K by cross-encoder score (max-permissive run).
# Reranker scores are uncalibrated; relative ordering is what counts.
GT_TOP_K = 30


async def fetch_chunks(store, run_id: str) -> list[dict[str, Any]]:
    rows = await store._client.query(
        store._database,
        "sql",
        "SELECT self_ref, chunk_text, embedding "
        "FROM ExtractionChunk WHERE pipeline_run_id = :run_id "
        "ORDER BY self_ref ASC",
        {"run_id": run_id},
    )
    return [r for r in rows if r.get("embedding")]


async def fetch_query_text(run_id: str, pass_name: str) -> str:
    """Read the C.7g run's actual rendered query_text from postgres diagnostics."""
    import asyncpg

    dsn = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "").replace("+psycopg2", "")
    if "postgresql" not in dsn:
        dsn = "postgresql://eip:eip_secret@postgres:5432/eip"
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            "SELECT diagnostics_json->'router'->>'query_text' AS q "
            "FROM ingest.pipeline_pass_outputs "
            "WHERE pipeline_run_id=$1 AND pass_name=$2 "
            "ORDER BY attempt DESC LIMIT 1",
            run_id, pass_name,
        )
        return row["q"] if row else ""
    finally:
        await conn.close()


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


async def sweep_one(doc_label: str, run_id: str, pass_name: str, grid: dict) -> list[dict]:
    store = get_graph_store()
    rows = await fetch_chunks(store, run_id)
    if not rows:
        print(f"  [{doc_label}/{pass_name}] no chunks — skipping")
        return []

    self_refs = [r["self_ref"] for r in rows]
    texts = [r["chunk_text"] or "" for r in rows]
    embs = normalize(np.asarray([r["embedding"] for r in rows], dtype=np.float32))

    query_text = await fetch_query_text(run_id, pass_name)
    if not query_text:
        print(f"  [{doc_label}/{pass_name}] no query_text — skipping")
        return []
    qv = normalize(np.asarray(embed_texts([query_text], query=True)[0], dtype=np.float32))

    vec_scores = (embs @ qv).astype(float)  # cosine; (N,)

    # Rerank ALL candidates once (max-permissive: no filter, infinite top_k)
    print(f"  [{doc_label}/{pass_name}] reranking N={len(rows)}...")
    cands = [
        {"content_text": t, "self_ref": sr, "vector_score": float(vs)}
        for sr, t, vs in zip(self_refs, texts, vec_scores)
    ]
    t0 = time.monotonic()
    full = rrk.rerank(query=query_text, candidates=cands, top_k=10**6)
    full_ms = (time.monotonic() - t0) * 1000.0
    print(f"  [{doc_label}/{pass_name}] full rerank: {full_ms:.0f}ms")

    rer_score = {c["self_ref"]: c.get("reranker_score", 0.0) for c in full}
    rer_rank = {c["self_ref"]: i for i, c in enumerate(full)}
    gt_top = {c["self_ref"] for c in full[:GT_TOP_K]}
    text_by_ref = dict(zip(self_refs, texts))

    results = []
    grid_iter = product(grid["top_n_candidates"], grid["top_k"], grid["min_similarity"])
    for top_n, top_k, min_sim in grid_iter:
        # 1. Vector-threshold filter
        mask = vec_scores >= min_sim
        idx_above = np.where(mask)[0]
        # 2. top_n by vector score
        if len(idx_above) > top_n:
            order = np.argsort(-vec_scores[idx_above])[:top_n]
            cand_idx = idx_above[order]
        else:
            cand_idx = idx_above[np.argsort(-vec_scores[idx_above])]
        cand_refs = [self_refs[i] for i in cand_idx]
        # 3. From the pre-scored full rerank, take only cand_refs in their
        #    original rerank order, then top_k.
        cand_set = set(cand_refs)
        ordered = [r for r in (c["self_ref"] for c in full) if r in cand_set]
        selected_refs = ordered[:top_k]
        sel_set = set(selected_refs)
        excl_set = cand_set - sel_set

        sel_text = " ".join(text_by_ref[r] for r in selected_refs)
        sel_tokens = count_bge_m3_tokens(sel_text)

        # Evidence coverage: how many of the GT top-K are in selected_refs
        cov = len(sel_set & gt_top) / len(gt_top) if gt_top else 0.0

        # Score gap: min(rerank in selected) - max(rerank in excluded but in cand_set)
        if sel_set and excl_set:
            gap = min(rer_score[r] for r in sel_set) - max(rer_score[r] for r in excl_set)
        else:
            gap = None

        # Rerank-ms estimate: proportional to candidate-set size
        est_ms = full_ms * (len(cand_refs) / max(len(self_refs), 1))

        results.append({
            "doc": doc_label,
            "pass": pass_name,
            "top_n": top_n,
            "top_k": top_k,
            "min_sim": min_sim,
            "candidate_count": len(cand_refs),
            "selected_ref_count": len(sel_set),
            "selected_tokens": sel_tokens,
            "evidence_cov": cov,
            "score_gap": gap,
            "est_rerank_ms": est_ms,
            "full_doc_chunks": len(rows),
        })
    return results


def fmt(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)\n"
    lines = []
    hdr = ("top_n", "top_k", "min_sim", "cand", "selected", "sel_tok",
           "cov(top30)", "gap", "rerank_ms")
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    # Sort: coverage desc, then selected asc, then est_rerank_ms asc
    rows_sorted = sorted(rows, key=lambda r: (-r["evidence_cov"], r["selected_ref_count"], r["est_rerank_ms"]))
    for r in rows_sorted:
        gap = f"{r['score_gap']:+.4f}" if r["score_gap"] is not None else "—"
        lines.append(
            f"| {r['top_n']} | {r['top_k']} | {r['min_sim']:.2f} | "
            f"{r['candidate_count']} | {r['selected_ref_count']} | {r['selected_tokens']} | "
            f"{r['evidence_cov']*100:.0f}% | {gap} | {r['est_rerank_ms']:.0f} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    all_rows: list[dict] = []
    for doc_label, run_id in [("Dvina", DVINA_RUN), ("SA-2", SA2_RUN)]:
        for pass_name, grid in GRIDS.items():
            print(f"--- {doc_label} / {pass_name} ---")
            rows = await sweep_one(doc_label, run_id, pass_name, grid)
            all_rows.extend(rows)

    out = "# C.9a — Offline calibration sweep\n\n"
    out += f"GT_TOP_K (critical-evidence definition) = top {GT_TOP_K} by reranker score "
    out += "at max-permissive (no threshold, top_k=∞).\n\n"
    out += "Columns:\n"
    out += "- `cand` = candidates surviving min_sim + top_n cap (input to selection)\n"
    out += "- `selected` = chunks the LLM would see (= min(top_k, cand))\n"
    out += "- `sel_tok` = bge-m3 token count of selected chunks\n"
    out += "- `cov(top30)` = % of GT top-30 refs that survive the config\n"
    out += "- `gap` = min(reranker score in selected) − max(reranker score in excluded-but-in-cand)\n"
    out += "- `rerank_ms` = est. (proportional to candidate count)\n\n"
    for doc in ("Dvina", "SA-2"):
        for pass_name in GRIDS:
            out += f"## {doc} / {pass_name}\n\n"
            subset = [r for r in all_rows if r["doc"] == doc and r["pass"] == pass_name]
            if subset:
                out += f"Full-doc chunks: {subset[0]['full_doc_chunks']}\n\n"
            out += fmt(subset)
            out += "\n"

    out_path = "/tmp/c9a_sweep_results.md"
    with open(out_path, "w") as f:
        f.write(out)
    print(f"\nWrote {out_path}")
    print(f"Total configs: {len(all_rows)}")


if __name__ == "__main__":
    asyncio.run(main())
