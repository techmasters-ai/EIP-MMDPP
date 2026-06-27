#!/usr/bin/env python3
"""Large-scale extraction data collector (see
docs/operational/large-scale-extraction-collection-plan.md).

Collects, per (document × pass), for a set of pipeline runs:
  1. recall        — primary_entities_extracted, relationships_extracted, status
  2. coverage      — field_coverage, channel_counts, min-set-cover, selected refs
                     (via the /v1/extraction/chunk-scope replay; field_group passes only)
  3. chunk stats   — total ExtractionChunk count + tokens for the run
  4. wall/trunc    — pass_wall_ms; truncation events attributed by time window

Read-only against postgres + ArcadeDB + the chunk-scope endpoint — does NOT
trigger extraction. Run it after (or during) ingest to snapshot current state.
Postgres + ArcadeDB are the system of record; this writes a flat CSV + JSONL for
analysis convenience.

Usage:
    python3 scripts/collect_extraction_data.py [--runs <id,id,...>] [--router-mode <tag>]
    # default: all graph_only/full runs since --since (newest per document)

subprocess list-form only (no shell quoting); safe to run detached.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

# --- config / endpoints -----------------------------------------------------
PG = ["docker", "exec", "eip-mmdpp-postgres-1", "psql", "-U", "eip", "-d", "eip",
      "-t", "-A", "-F", "|", "-c"]
ADB_URL = "http://localhost:2480/api/v1/command/eip_knowledge_graph"
ADB_AUTH = "root:eip_arcadedb_secret"
CHUNK_SCOPE_URL = "http://localhost:8005/v1/extraction/chunk-scope"
DG_CONTAINER = "eip-mmdpp-docling-graph-1"

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "collection"
FIELD_GROUP_HINT = ("power_rf", "kinematics", "antenna", "timing", "modulation",
                    "guidance", "airframe", "speed", "propulsion")  # heuristic; refined by router_diagnostics


def sh(args, timeout=60):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception as e:
        print(f"  [warn] subprocess failed: {e}")
        return ""


def psql(sql):
    return [l for l in sh(PG + [sql]).splitlines() if l.strip() and not l.startswith("psql:")]


def adb(sql):
    out = sh(["curl", "-s", "--max-time", "15", "-u", ADB_AUTH, "-X", "POST", ADB_URL,
              "-H", "Content-Type: application/json",
              "-d", json.dumps({"language": "sql", "command": sql})], timeout=25)
    try:
        return json.loads(out).get("result", [])
    except Exception:
        return []


def chunk_scope(run, bundle, pass_name):
    out = sh(["curl", "-s", "--max-time", "120", "-X", "POST", CHUNK_SCOPE_URL,
              "-H", "Content-Type: application/json",
              "-d", json.dumps({"pipeline_run_id": run, "bundle_key": bundle,
                                "pass_name": pass_name})], timeout=150)
    try:
        return json.loads(out)
    except Exception:
        return {}


_IDENTITY_FIELDS = {"system_name", "missile_name", "designation", "identity_name", "name"}


def value_fill_and_provenance(rid, pass_name):
    """Read extract_pass_response_json for the latest attempt and compute the
    REAL recall signal for chunk-minimization:
      - entities, entities_with_value (vs name-only shells)
      - field_values_filled / field_values_possible (the true recall metric —
        NOT entity count, which is inflated by shell entities)
      - provenance_populated: are evidence_ids/self_refs actually recorded per
        entity? (needed to link a value back to a source chunk). If empty,
        chunk-minimization must use ablation, not provenance correlation.
      - evidence_self_refs: union of self_refs cited across the pass's provenance
        (the chunks that actually produced output, when populated).
    """
    rows = psql("SELECT extract_pass_response_json FROM ingest.pipeline_pass_outputs "
                "WHERE pipeline_run_id='" + rid + "' AND pass_name='" + pass_name + "' "
                "ORDER BY attempt DESC LIMIT 1")
    blob = "".join(rows) if rows else ""
    out = {"entities": "", "entities_with_value": "", "values_filled": "",
           "values_possible": "", "value_fill_pct": "", "provenance_populated": "",
           "evidence_self_ref_count": ""}
    if not blob:
        return out
    try:
        d = json.loads(blob)
    except Exception:
        return out
    po = d.get("pass_output") or {}
    key = next(iter(po), None)
    recs = po.get(key) or [] if key else []
    recs = [r for r in recs if isinstance(r, dict)]
    if recs:
        value_fields = [f for f in recs[0].keys() if f not in _IDENTITY_FIELDS]
        empties = (None, "", [], {})
        filled = sum(1 for r in recs for f in value_fields if r.get(f) not in empties)
        poss = len(recs) * max(len(value_fields), 1)
        ent_val = sum(1 for r in recs if any(r.get(f) not in empties for f in value_fields))
        out.update(entities=len(recs), entities_with_value=ent_val,
                   values_filled=filled, values_possible=poss,
                   value_fill_pct=round(100 * filled / max(poss, 1), 1))
    # provenance availability (evidence_ids / self_refs populated?)
    prov = d.get("provenance") or []
    sref = set()
    pop = 0
    for p in prov if isinstance(prov, list) else []:
        eids = (p.get("evidence_ids") or []) + (p.get("self_refs") or [])
        if eids:
            pop += 1
            sref.update(eids)
    out["provenance_populated"] = f"{pop}/{len(prov)}" if prov else "0/0"
    out["evidence_self_ref_count"] = len(sref)
    return out


def greedy_set_cover(chunk_fields):
    """chunk_fields: list[set[str]] (per scored chunk). Return (#chunks, ranks) to
    cover the union of all evidenced fields."""
    all_fields = set().union(*chunk_fields) if chunk_fields else set()
    need = set(all_fields)
    pool = [(i + 1, set(f)) for i, f in enumerate(chunk_fields)]
    picked = []
    while need:
        best = max(pool, key=lambda rf: len(rf[1] & need), default=None)
        if not best or not (best[1] & need):
            break
        picked.append(best[0]); need -= best[1]; pool.remove(best)
    return len(picked), picked, sorted(all_fields)


def discover_runs(since, explicit):
    if explicit:
        ids = explicit.split(",")
        rows = psql("SELECT r.id||'|'||COALESCE(r.ontology_bundle_key,'')||'|'||"
                    "COALESCE((SELECT filename FROM ingest.documents d WHERE d.id=r.document_id),'?') "
                    "FROM ingest.pipeline_runs r WHERE r.id IN ('" + "','".join(ids) + "')")
    else:
        # newest run per document since cutoff
        rows = psql(
            "SELECT DISTINCT ON (r.document_id) r.id||'|'||COALESCE(r.ontology_bundle_key,'')||'|'||"
            "COALESCE(d.filename,'?') FROM ingest.pipeline_runs r "
            "LEFT JOIN ingest.documents d ON d.id=r.document_id "
            "WHERE r.mode IN ('graph_only','full') AND r.started_at > '" + since + "' "
            "ORDER BY r.document_id, r.started_at DESC")
    out = []
    for row in rows:
        parts = row.split("|")
        if len(parts) >= 3:
            out.append({"run_id": parts[0], "bundle": parts[1], "doc": parts[2]})
    return out


def collect_run(run, router_mode):
    rid, bundle, doc = run["run_id"], run["bundle"], run["doc"]
    # --- chunk stats (ArcadeDB) ---
    stat = adb("SELECT count(*) AS n, sum(token_count) AS tok FROM ExtractionChunk "
               "WHERE pipeline_run_id='" + rid + "'")
    total_chunks = stat[0].get("n") if stat else None
    total_tokens = stat[0].get("tok") if stat else None
    # --- per-pass recall + wall (postgres) ---
    rows = psql(
        "SELECT DISTINCT ON (pass_name) pass_name||'%'||execution_status||'%'||"
        "COALESCE(primary_entities_extracted,0)||'%'||COALESCE(relationships_extracted,0)||'%'||"
        "COALESCE((diagnostics_json->>'pass_wall_ms'),'')||'%'||COALESCE(diagnostics_json->>'chunk_count','') "
        "FROM ingest.pipeline_pass_outputs WHERE pipeline_run_id='" + rid + "' "
        "ORDER BY pass_name, attempt DESC")
    results = []
    for r in rows:
        x = r.split("%")
        if len(x) < 6:
            continue
        pass_n, status, ents, rels, wallms, llmc = x[:6]
        wmin = ""
        try:
            wmin = round(float(wallms) / 60000, 1) if wallms else ""
        except Exception:
            pass
        # --- THE REAL recall metric: field-value fill (not entity count) + provenance ---
        vf = value_fill_and_provenance(rid, pass_n)
        rec = {"router_mode": router_mode, "doc": doc, "run_id": rid, "bundle": bundle,
               "pass": pass_n, "status": status, "entities": ents, "rels": rels,
               # value-fill = true recall signal for chunk-minimization:
               "entities_with_value": vf["entities_with_value"],
               "values_filled": vf["values_filled"], "values_possible": vf["values_possible"],
               "value_fill_pct": vf["value_fill_pct"],
               "provenance_populated": vf["provenance_populated"],
               "evidence_self_refs": vf["evidence_self_ref_count"],
               "wall_min": wmin, "llm_chunks": llmc, "total_chunks": total_chunks,
               "total_tokens": total_tokens, "fields_with_evidence": "", "min_set_cover": "",
               # three distinct "chunks to LLM" measures (the chunk-minimization axis):
               "selected_chunk_count": "",   # merged ExtractionChunks chosen (== top_k bound)
               "expanded_refs": "",          # element-level self_refs the LLM actually receives
               "selected_tokens": "", "full_doc_tokens": "", "channel_counts": ""}
        # --- coverage (chunk-scope replay) for field_group passes only ---
        if any(h in pass_n for h in FIELD_GROUP_HINT):
            d = chunk_scope(rid, bundle, pass_n)
            diag = d.get("diagnostics") or {}
            sc = diag.get("score_components") or []
            cf = [{s.split("field:")[1] for s in (c.get("retrieval_sources") or [])
                   if s.startswith("field:")} for c in sc]
            ncov, ranks, allf = greedy_set_cover(cf)
            rec["fields_with_evidence"] = ";".join(allf)
            rec["min_set_cover"] = ncov
            rec["selected_chunk_count"] = diag.get("selected_chunk_count")
            rec["expanded_refs"] = diag.get("expanded_ref_count") or len(d.get("self_refs") or [])
            rec["selected_tokens"] = diag.get("selected_token_estimate")
            rec["full_doc_tokens"] = diag.get("full_doc_token_estimate")
            rec["channel_counts"] = json.dumps(diag.get("channel_counts") or {})
        results.append(rec)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="explicit comma-sep run ids (else newest-per-doc)")
    ap.add_argument("--since", default="2026-05-30 05:00:00")
    ap.add_argument("--router-mode", default="unknown", help="tag for this collection pass (shadow|narrow_only)")
    ap.add_argument("--label", default="collection", help="output file label")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.since, args.runs or None)
    print(f"[collect] {len(runs)} run(s) | router_mode={args.router_mode}")
    cols = ["router_mode", "doc", "run_id", "bundle", "pass", "status", "entities", "rels",
            "entities_with_value", "values_filled", "values_possible", "value_fill_pct",
            "provenance_populated", "evidence_self_refs",
            "wall_min", "llm_chunks", "total_chunks", "total_tokens",
            "fields_with_evidence", "min_set_cover", "selected_chunk_count", "expanded_refs",
            "selected_tokens", "full_doc_tokens", "channel_counts"]
    csv_path = OUT_DIR / f"{args.label}_{args.router_mode}.csv"
    jsonl_path = OUT_DIR / f"{args.label}_{args.router_mode}.jsonl"
    with open(csv_path, "w", newline="") as cf, open(jsonl_path, "w") as jf:
        w = csv.DictWriter(cf, fieldnames=cols)
        w.writeheader()
        for run in runs:
            print(f"  - {run['doc'][:40]} ({run['run_id'][:8]}) bundle={run['bundle']}")
            for rec in collect_run(run, args.router_mode):
                w.writerow(rec)
                jf.write(json.dumps(rec) + "\n")
    print(f"[collect] wrote {csv_path}")
    print(f"[collect] wrote {jsonl_path}")


if __name__ == "__main__":
    main()
