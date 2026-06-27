"""C.9a v2 — extended SA-2 calibration sweep.

Hypothesis from C.9a v1: vector pre-filter is the weak link, not reranker.
Test: higher top_n_candidates to see where the coverage curve flattens.
Goal: find configs where coverage > 80% AND selected_tokens stays small.

SA-2 only; Dvina is settled at 150/30/0.25 (97% cov).
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

SA2_RUN = "66b4a698-cc1b-439d-bc80-28f5e3309b6d"

GRIDS = {
    "radar_power_rf": {
        # "all" sentinel = no top_n cap (use every chunk above min_sim)
        "top_n_candidates": [200, 250, 300, "all"],
        "top_k": [40, 50, 60, 80],
        "min_similarity": [0.20, 0.25],
    },
    "missile_kinematics": {
        "top_n_candidates": [200, 250, 300, "all"],
        "top_k": [50, 60, 80, 100],
        "min_similarity": [0.20, 0.25],
    },
}

GT_TOP_K = 30


async def fetch_chunks(store, run_id: str):
    rows = await store._client.query(
        store._database, "sql",
        "SELECT self_ref, chunk_text, embedding FROM ExtractionChunk "
        "WHERE pipeline_run_id = :run_id ORDER BY self_ref ASC",
        {"run_id": run_id},
    )
    return [r for r in rows if r.get("embedding")]


async def fetch_query_text(run_id, pass_name):
    import asyncpg
    conn = await asyncpg.connect("postgresql://eip:eip_secret@postgres:5432/eip")
    try:
        row = await conn.fetchrow(
            "SELECT diagnostics_json->'router'->>'query_text' AS q "
            "FROM ingest.pipeline_pass_outputs WHERE pipeline_run_id=$1 AND pass_name=$2 "
            "ORDER BY attempt DESC LIMIT 1",
            run_id, pass_name,
        )
        return row["q"] if row else ""
    finally:
        await conn.close()


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


async def sweep_pass(pass_name, grid):
    store = get_graph_store()
    rows = await fetch_chunks(store, SA2_RUN)
    refs = [r["self_ref"] for r in rows]
    texts = [r["chunk_text"] or "" for r in rows]
    embs = normalize(np.asarray([r["embedding"] for r in rows], dtype=np.float32))

    q = await fetch_query_text(SA2_RUN, pass_name)
    qv = normalize(np.asarray(embed_texts([q], query=True)[0], dtype=np.float32))
    vec = (embs @ qv).astype(float)
    print(f"  [{pass_name}] vec_scores min={vec.min():.4f} max={vec.max():.4f} median={np.median(vec):.4f}")

    cands = [{"content_text": t, "self_ref": r, "vector_score": float(v)}
             for r, t, v in zip(refs, texts, vec)]
    t0 = time.monotonic()
    full = rrk.rerank(query=q, candidates=cands, top_k=10**6)
    full_ms = (time.monotonic() - t0) * 1000.0
    print(f"  [{pass_name}] full rerank N={len(rows)} ms={full_ms:.0f}")

    rer_score = {c["self_ref"]: c.get("reranker_score", 0.0) for c in full}
    full_in_order = [c["self_ref"] for c in full]
    gt_top = set(full_in_order[:GT_TOP_K])
    text_by_ref = dict(zip(refs, texts))

    results = []
    for top_n, top_k, min_sim in product(grid["top_n_candidates"], grid["top_k"], grid["min_similarity"]):
        # 1. min_sim filter
        mask = vec >= min_sim
        idx_above = np.where(mask)[0]
        # 2. top_n cap (or "all" sentinel = no cap)
        cap = len(refs) if top_n == "all" else int(top_n)
        if len(idx_above) > cap:
            order = np.argsort(-vec[idx_above])[:cap]
            cand_idx = idx_above[order]
        else:
            cand_idx = idx_above[np.argsort(-vec[idx_above])]
        cand_set = {refs[i] for i in cand_idx}
        # 3. top_k by reranker_score within cand_set
        ordered = [r for r in full_in_order if r in cand_set]
        selected = ordered[:top_k]
        sel_set = set(selected)
        excl_set = cand_set - sel_set
        sel_tok = count_bge_m3_tokens(" ".join(text_by_ref[r] for r in selected))
        cov = len(sel_set & gt_top) / len(gt_top) if gt_top else 0.0
        gap = (min(rer_score[r] for r in sel_set) - max(rer_score[r] for r in excl_set)) if sel_set and excl_set else None
        est_ms = full_ms * (len(cand_set) / max(len(refs), 1))
        results.append({
            "pass": pass_name,
            "top_n": top_n,
            "top_k": top_k,
            "min_sim": min_sim,
            "cand": len(cand_set),
            "selected": len(sel_set),
            "sel_tok": sel_tok,
            "cov": cov,
            "gap": gap,
            "est_ms": est_ms,
        })
    return results


def fmt(rows):
    hdr = ("top_n", "top_k", "min_sim", "cand", "selected", "sel_tok", "cov(top30)", "gap", "rerank_ms")
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "|".join(["---"] * len(hdr)) + "|"]
    rows_sorted = sorted(rows, key=lambda r: (-r["cov"], r["sel_tok"], r["est_ms"]))
    for r in rows_sorted:
        gap = f"{r['gap']:+.4f}" if r["gap"] is not None else "—"
        tn = "all" if r["top_n"] == "all" else r["top_n"]
        out.append(f"| {tn} | {r['top_k']} | {r['min_sim']:.2f} | "
                   f"{r['cand']} | {r['selected']} | {r['sel_tok']} | "
                   f"{r['cov']*100:.0f}% | {gap} | {r['est_ms']:.0f} |")
    return "\n".join(out) + "\n"


async def main():
    all_rows = []
    for pn, grid in GRIDS.items():
        print(f"--- SA-2 / {pn} ---")
        rows = await sweep_pass(pn, grid)
        all_rows.extend(rows)

    out = "# C.9a v2 — Extended SA-2 sweep\n\n"
    out += "Hypothesis: vector pre-filter is the weak link; higher top_n recovers GT recall.\n\n"
    for pn in GRIDS:
        out += f"## SA-2 / {pn}\n\n"
        out += fmt([r for r in all_rows if r["pass"] == pn])
        out += "\n"
    with open("/tmp/c9a_sweep_v2.md", "w") as f:
        f.write(out)
    print(f"Wrote /tmp/c9a_sweep_v2.md ({len(all_rows)} configs)")


if __name__ == "__main__":
    asyncio.run(main())
