"""C.9d offline check — replay retrieval+rerank against the new (filtered)
Dvina chunks and verify the top-50 selections are real content, not chrome."""
from __future__ import annotations
import asyncio, re, json
import numpy as np
import asyncpg
from app.db.session import get_graph_store
from app.services import reranker as rrk
from app.services.embedding import embed_texts

RUN = "b7d5332a-a3e1-4f88-a84b-1f2a6211c1e9"

def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

CHROME_KW = ("audio coming soon","subscribe now","sponsored","advertisement",
             "log in","historynet","recommended for you")

def classify(text: str) -> str:
    t = (text or "").strip()
    low = t.lower()
    if len(t) <= 2: return "single_symbol"
    if any(k in low for k in CHROME_KW): return "chrome_residue"
    if len(t) <= 20: return "very_short"
    return "body_text"

async def main():
    pg = await asyncpg.connect("postgresql://eip:eip_secret@postgres:5432/eip")
    q = await pg.fetchval(
        "SELECT diagnostics_json->'router'->>'query_text' "
        "FROM ingest.pipeline_pass_outputs "
        "WHERE pipeline_run_id='66b4a698-cc1b-439d-bc80-28f5e3309b6d' "
        "AND pass_name='radar_power_rf' ORDER BY attempt DESC LIMIT 1"
    )
    await pg.close()

    store = get_graph_store()
    rows = await store._client.query(
        store._database, "sql",
        "SELECT self_ref, chunk_text, embedding FROM ExtractionChunk "
        "WHERE pipeline_run_id = :run_id ORDER BY self_ref ASC",
        {"run_id": RUN},
    )
    rows = [r for r in rows if r.get("embedding")]
    print(f"Indexed chunks for new run: {len(rows)}")

    refs = [r["self_ref"] for r in rows]
    texts = [r.get("chunk_text") or "" for r in rows]
    embs = normalize(np.asarray([r["embedding"] for r in rows], dtype=np.float32))
    qv = normalize(np.asarray(embed_texts([q], query=True)[0], dtype=np.float32))
    vec = (embs @ qv).astype(float)

    # narrowing_v1: min_sim=0.25, top_n=300, top_k=50
    above = np.where(vec >= 0.25)[0]
    order = np.argsort(-vec[above])[:300]
    cand_idx = above[order]
    cands = [{"content_text": texts[i], "self_ref": refs[i], "vector_score": float(vec[i])} for i in cand_idx]
    reranked = rrk.rerank(query=q, candidates=cands, top_k=50)
    sel = [c["self_ref"] for c in reranked]
    sel_text_chars = sum(len(c["content_text"]) for c in reranked)
    sel_tokens = sum(len(c["content_text"].split()) for c in reranked) * 4 // 3  # rough

    print(f"\ncand={len(cands)} selected={len(sel)} sel_chars={sel_text_chars}")
    buckets: dict[str, int] = {}
    snippets: dict[str, list[str]] = {}
    for c in reranked:
        cls = classify(c["content_text"])
        buckets[cls] = buckets.get(cls, 0) + 1
        snippets.setdefault(cls, []).append(c["content_text"][:60].replace("\n", " ⏎ "))

    print("\n=== Selection buckets ===")
    for cls, n in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {cls:<16} n={n:>2}  ex: {snippets[cls][0][:50]!r}")

    print("\n=== Top-10 selected refs ===")
    for c in reranked[:10]:
        snip = c["content_text"][:60].replace("\n", " ⏎ ")
        print(f"  {c['self_ref']:<15} score={c.get('reranker_score',0):.4f} {snip}")

if __name__ == "__main__":
    asyncio.run(main())
