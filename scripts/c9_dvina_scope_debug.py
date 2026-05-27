"""C.9c-blocker — per-ref classification of Dvina radar_power_rf narrowing failure.

For each of the 50 selected refs:
  - self_ref, modality, chunk_text length, first 100 chars
  - whether apply_chunk_scope keeps it (present in scoped_doc.texts/pictures/tables)
  - parent/group chain
  - rendered markdown contribution length

Then bucket each ref so we can decide if the problem is (a) retrieval noise,
(b) scope bug dropping good refs, (c) rendering mismatch, (d) group context missing.
"""
from __future__ import annotations

import asyncio
import json
import re

import numpy as np
import asyncpg

from app.db.session import get_graph_store
from app.services import reranker as rrk
from app.services.embedding import embed_texts
from app.services.scoped_docling_document import apply_chunk_scope
from app.workers.pipeline import _build_docling_document_json

DVINA_DOC = "9c8e09c7-e39f-4359-92c0-46330158c73c"
DVINA_RUN = "3cc31fe0-9080-4e4d-8314-60512c973b23"


async def get_query_text(run_id, pass_name):
    pg = await asyncpg.connect("postgresql://eip:eip_secret@postgres:5432/eip")
    try:
        # Reuse the SA-2 run's query_text — it's identical text-wise for this pass.
        # (Dvina's failed run never persisted diagnostics_json.)
        return await pg.fetchval(
            "SELECT diagnostics_json->'router'->>'query_text' "
            "FROM ingest.pipeline_pass_outputs "
            "WHERE pipeline_run_id='66b4a698-cc1b-439d-bc80-28f5e3309b6d' "
            "AND pass_name='radar_power_rf' ORDER BY attempt DESC LIMIT 1"
        )
    finally:
        await pg.close()


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def classify(chunk_text: str, modality: str) -> str:
    t = (chunk_text or "").strip()
    if not t:
        return "empty"
    if len(t) <= 2:
        return "single_symbol"
    low = t.lower()
    if any(kw in low for kw in ("sponsored", "audio coming soon", "audio", "donald trump", "subscribe", "advertis")):
        return "web_ad_junk"
    if modality == "picture_caption":
        return "picture_caption"
    if len(t) <= 20:
        return "very_short"
    return "body_text"


def find_in_scoped_doc(self_ref: str, scoped: dict) -> tuple[bool, str, str]:
    """Returns (kept, group_chain, where_kind)."""
    m = re.match(r"#/(\w+)/(\d+)", self_ref)
    if not m:
        return False, "—", "unparseable_ref"
    kind, idx_s = m.group(1), m.group(2)
    arr = scoped.get(kind, [])
    if not arr:
        return False, "—", f"no_{kind}_array"
    for item in arr:
        if item.get("self_ref") == self_ref:
            parent_ref = (item.get("parent") or {}).get("$ref") if isinstance(item.get("parent"), dict) else None
            return True, parent_ref or "(no_parent)", kind
    return False, "—", f"absent_from_{kind}"


def markdown_export(doc_json: dict) -> str:
    try:
        from docling_core.types.doc import DoclingDocument
        d = DoclingDocument.model_validate(doc_json)
        return d.export_to_markdown()
    except Exception as exc:
        return f"<<export_failed: {exc!r}>>"


async def main():
    store = get_graph_store()

    # 1. Pull this run's chunks (the same ones the worker just used)
    rows = await store._client.query(
        store._database, "sql",
        "SELECT self_ref, chunk_text, embedding, modality FROM ExtractionChunk "
        "WHERE pipeline_run_id = :run_id ORDER BY self_ref ASC",
        {"run_id": DVINA_RUN},
    )
    rows = [r for r in rows if r.get("embedding")]
    refs = [r["self_ref"] for r in rows]
    texts = [r.get("chunk_text") or "" for r in rows]
    modalities = [r.get("modality") or "?" for r in rows]
    embs = normalize(np.asarray([r["embedding"] for r in rows], dtype=np.float32))

    # 2. Replay the chunk-scope selection
    q = await get_query_text(DVINA_RUN, "radar_power_rf")
    qv = normalize(np.asarray(embed_texts([q], query=True)[0], dtype=np.float32))
    vec = (embs @ qv).astype(float)
    above = np.where(vec >= 0.25)[0]
    order = np.argsort(-vec[above])[:300]
    cand_idx = above[order]
    cands = [{"content_text": texts[i], "self_ref": refs[i], "vector_score": float(vec[i])} for i in cand_idx]
    reranked = rrk.rerank(query=q, candidates=cands, top_k=50)
    selected_refs = [c["self_ref"] for c in reranked]
    selected_set = set(selected_refs)
    print(f"selected {len(selected_refs)} refs, total chunk_text chars="
          f"{sum(len(c['content_text']) for c in reranked)}")

    # 3. Load the original DoclingDocument
    doc_json = _build_docling_document_json(DVINA_DOC)
    print(f"orig doc: texts={len(doc_json.get('texts',[]))}, "
          f"pictures={len(doc_json.get('pictures',[]))}, "
          f"tables={len(doc_json.get('tables',[]))}, "
          f"groups={len(doc_json.get('groups',[]))}")

    # 4. Run apply_chunk_scope
    chunk_scope = {"mode": "selected_refs", "self_refs": selected_refs}
    scoped = apply_chunk_scope(doc_json, chunk_scope)
    print(f"scoped doc: texts={len(scoped.get('texts',[]))}, "
          f"pictures={len(scoped.get('pictures',[]))}, "
          f"tables={len(scoped.get('tables',[]))}, "
          f"groups={len(scoped.get('groups',[]))}")

    # 5. Render to markdown
    md = markdown_export(scoped)
    print(f"scoped markdown: {len(md)} chars")
    print("---scoped markdown (first 800 chars)---")
    print(md[:800])
    print("---end---")
    orig_md = markdown_export(doc_json)
    print(f"orig markdown: {len(orig_md)} chars")

    # 6. Build per-ref table
    chunk_by_ref = {r: t for r, t in zip(refs, texts)}
    mod_by_ref = {r: m for r, m in zip(refs, modalities)}
    rows_out = []
    for sr in selected_refs:
        t = chunk_by_ref.get(sr, "")
        m = mod_by_ref.get(sr, "?")
        kept, parent, where = find_in_scoped_doc(sr, scoped)
        cls = classify(t, m)
        rows_out.append({
            "self_ref": sr, "modality": m, "len": len(t),
            "class": cls, "kept_in_scope": kept,
            "where": where, "parent": parent,
            "snippet": (t[:80] + ("…" if len(t) > 80 else "")).replace("\n", " ⏎ "),
        })

    print("\n=== Per-ref classification ===")
    print(f"{'self_ref':<16} {'mod':<16} {'len':>4} {'kept':>4} {'where':<14} {'class':<16} snippet")
    for r in rows_out:
        print(f"{r['self_ref']:<16} {r['modality']:<16} {r['len']:>4} {str(r['kept_in_scope']):>4} "
              f"{r['where']:<14} {r['class']:<16} {r['snippet']}")

    # 7. Bucket counts
    print("\n=== Bucket counts ===")
    by_class = {}
    for r in rows_out:
        k = (r["class"], r["kept_in_scope"])
        by_class[k] = by_class.get(k, 0) + 1
    for (cls, kept), n in sorted(by_class.items()):
        print(f"  class={cls:<16} kept_in_scope={kept!s:<5}  n={n}")

    # Save full table to JSON for review
    with open("/tmp/c9c_dvina_scope_debug.json", "w") as f:
        json.dump({
            "selected_refs": selected_refs,
            "rows": rows_out,
            "summary": {
                "orig_md_chars": len(orig_md),
                "scoped_md_chars": len(md),
                "orig_texts": len(doc_json.get("texts", [])),
                "scoped_texts": len(scoped.get("texts", [])),
            },
        }, f, indent=2)
    print("\nSaved /tmp/c9c_dvina_scope_debug.json")


if __name__ == "__main__":
    asyncio.run(main())
