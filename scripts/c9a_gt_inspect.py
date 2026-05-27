"""C.9a follow-up — inspect GT-top-30 reranker score distribution and
overlap with the actual entity-bearing self_refs from C.7g.

Answers: is the 50-63% coverage on SA-2 real recall loss, or an artifact
of clustered reranker scores?
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np

from app.db.session import get_graph_store
from app.services import reranker as rrk
from app.services.embedding import embed_texts

DVINA_RUN = "4ed1c3e6-5976-4b07-9276-c3c23ede5929"
SA2_RUN = "66b4a698-cc1b-439d-bc80-28f5e3309b6d"

PASSES = ["radar_power_rf", "missile_kinematics"]


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


async def fetch_provenance_refs(run_id, pass_name):
    """Return self_refs that the LLM actually anchored an entity to in C.7g."""
    import asyncpg
    conn = await asyncpg.connect("postgresql://eip:eip_secret@postgres:5432/eip")
    try:
        row = await conn.fetchrow(
            "SELECT field_provenance_json FROM ingest.pipeline_pass_outputs "
            "WHERE pipeline_run_id=$1 AND pass_name=$2 ORDER BY attempt DESC LIMIT 1",
            run_id, pass_name,
        )
        if not row or not row["field_provenance_json"]:
            return set()
        prov = row["field_provenance_json"]
        if isinstance(prov, str):
            prov = json.loads(prov)
        refs = set()
        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ("self_ref", "ref", "doc_ref"):
                        if isinstance(v, str):
                            refs.add(v)
                        elif isinstance(v, list):
                            for s in v:
                                if isinstance(s, str): refs.add(s)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for y in x:
                    walk(y)
        walk(prov)
        return refs
    finally:
        await conn.close()


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


async def inspect_one(doc_label, run_id, pass_name):
    store = get_graph_store()
    rows = await fetch_chunks(store, run_id)
    refs = [r["self_ref"] for r in rows]
    texts = [r["chunk_text"] or "" for r in rows]
    embs = normalize(np.asarray([r["embedding"] for r in rows], dtype=np.float32))

    q = await fetch_query_text(run_id, pass_name)
    qv = normalize(np.asarray(embed_texts([q], query=True)[0], dtype=np.float32))
    vec = (embs @ qv).astype(float)

    cands = [{"content_text": t, "self_ref": r, "vector_score": float(v)}
             for r, t, v in zip(refs, texts, vec)]
    full = rrk.rerank(query=q, candidates=cands, top_k=10**6)

    scores = np.array([c["reranker_score"] for c in full])
    top30_scores = scores[:30]
    top50_scores = scores[:50]

    prov_refs = await fetch_provenance_refs(run_id, pass_name)
    full_refs_in_order = [c["self_ref"] for c in full]
    # Where do provenance refs appear in the reranker order?
    rank_of = {r: i for i, r in enumerate(full_refs_in_order)}
    prov_ranks = sorted(rank_of.get(r) for r in prov_refs if r in rank_of)
    # Coverage of provenance refs by various top-K cuts
    by_top = {}
    for k in (25, 30, 40, 50, 60, 75, 100, 150, 200, 314):
        if k <= len(full_refs_in_order):
            covered = len(set(full_refs_in_order[:k]) & prov_refs)
            by_top[k] = (covered, len(prov_refs))

    print(f"\n=== {doc_label} / {pass_name} ===")
    print(f"N chunks: {len(rows)}")
    print(f"Provenance refs (LLM anchored): {len(prov_refs)}")
    if scores.size:
        print(f"Reranker scores — min: {scores.min():.5f}  max: {scores.max():.5f}  median: {np.median(scores):.5f}")
        print(f"GT top-30 score range: [{top30_scores.min():.5f}, {top30_scores.max():.5f}]  median {np.median(top30_scores):.5f}")
        if scores.size >= 50:
            score_at_30 = scores[29]
            score_at_50 = scores[49]
            print(f"Score at rank 30: {score_at_30:.5f}   at rank 50: {score_at_50:.5f}   delta: {score_at_30 - score_at_50:+.5f}")
        # Distinct buckets
        unique = np.unique(np.round(top30_scores, 4))
        print(f"GT top-30 distinct values (rounded 1e-4): {len(unique)}")
    if prov_ranks:
        print(f"Provenance refs ranked at positions: min={prov_ranks[0]}, median={prov_ranks[len(prov_ranks)//2]}, max={prov_ranks[-1]}")
        print(f"Provenance ref position histogram (top-K cumulative coverage):")
        for k, (cov, tot) in by_top.items():
            print(f"  top-{k:3d}: {cov:3d}/{tot} ({100*cov/max(tot,1):.0f}%)")


async def main():
    for doc, run in [("Dvina", DVINA_RUN), ("SA-2", SA2_RUN)]:
        for p in PASSES:
            await inspect_one(doc, run, p)


if __name__ == "__main__":
    asyncio.run(main())
