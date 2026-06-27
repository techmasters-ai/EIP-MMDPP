#!/usr/bin/env python3
"""Characterize target noise: grounded vs ungrounded positive (pass,chunk) cells.

A positive cell = (pass P, chunk C) the lineage marked extracted-from. It is
GROUNDED if at least one of P's extracted field VALUES (from pass_output) is
literally present in C's text (string substring or numeric+unit, via the shared
field_value_grounding matcher). UNGROUNDED positives are suspect target noise —
the lineage attributes a field to a chunk whose text doesn't contain the value
(LLM-normalized/hallucinated). These are the cells most likely unrankable by any
chunk feature, pinning the recall-1.0 frontier to 0% savings.

Read-only. Reports per-pass + overall grounded/ungrounded counts and a sample of
ungrounded cells (with the value + where, if anywhere, it actually appears).

    python3 -m scripts.characterize_target_noise --runs <r1,r2,...>
"""
from __future__ import annotations
import argparse, base64, json, os, unicodedata, urllib.request
from collections import defaultdict

from app.services.field_value_grounding import nfc as _nfc, num_variants, units_for, value_in_chunk

_PG = os.environ.get("A0_DATABASE_URL", "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip")
_ARC = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")


def _arc(sql: str):
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = urllib.request.Request(f"{_ARC}/api/v1/command/eip_knowledge_graph",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(urllib.request.urlopen(req, timeout=40)).get("result", [])


def _strnorm(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", _nfc(str(s)).casefold())
                   if not unicodedata.combining(c))


def _grounded(value, txt: str, fname: str) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        units = units_for(fname) or []
        nums = num_variants(value)
        if units and value_in_chunk(nums, units, txt):
            return True
        return any(_strnorm(n) in _strnorm(txt) for n in nums)  # bare number fallback
    sv = _strnorm(value)
    return len(sv) >= 2 and sv in _strnorm(txt)


def _pass_field_values(pass_output) -> dict[str, list]:
    out = defaultdict(list)
    for v in (pass_output or {}).values():
        if isinstance(v, list):
            for rec in v:
                if isinstance(rec, dict):
                    for f, val in rec.items():
                        if val is not None and val != "" and not isinstance(val, (list, dict)):
                            out[f].append(val)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    from sqlalchemy import create_engine, text as _t
    eng = create_engine(_PG)
    per_pass = defaultdict(lambda: {"grounded": 0, "ungrounded": 0})
    samples = []
    tot_g = tot_u = 0
    for run in runs:
        chunks = {int(c["chunk_index"]): _nfc(c.get("chunk_text") or "")
                  for c in _arc(f"SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}'")}
        with eng.connect() as c:
            rows = c.execute(_t("SELECT pass_name, extract_pass_response_json FROM ingest.pipeline_pass_outputs "
                                "WHERE pipeline_run_id=:r ORDER BY pass_name, attempt"), {"r": run}).fetchall()
        latest = {pn: (json.loads(j) if isinstance(j, str) else j) for pn, j in rows}
        for pass_name, j in latest.items():
            if not j:
                continue
            fvals = _pass_field_values(j.get("pass_output"))
            # positive chunks for this pass + which fields cite each chunk
            pos_fields: dict[int, set[str]] = defaultdict(set)
            for fp in (j.get("field_provenance") or []):
                fn = fp.get("field_name")
                cis = list(fp.get("chunk_indexes") or [])
                if isinstance(fp.get("chunk_index"), int):
                    cis.append(fp["chunk_index"])
                for ci in cis:
                    if isinstance(ci, int):
                        pos_fields[ci].add(fn)
            for ci, fields in pos_fields.items():
                txt = chunks.get(ci, "")
                # grounded if any cited field's value(s) present in this chunk
                ok = False
                for fn in fields:
                    if any(_grounded(v, txt, fn) for v in fvals.get(fn, [])):
                        ok = True; break
                if ok:
                    per_pass[pass_name]["grounded"] += 1; tot_g += 1
                else:
                    per_pass[pass_name]["ungrounded"] += 1; tot_u += 1
                    if len(samples) < 20:
                        # where does any cited field value actually appear?
                        where = sorted({d for fn in fields for v in fvals.get(fn, [])
                                        for d, t in chunks.items() if _grounded(v, t, fn)})
                        vals = {fn: fvals.get(fn, [])[:2] for fn in fields}
                        samples.append(f"{run[:8]} {pass_name} chunk{ci}: fields={vals} -> value in chunks {where}")

    tot = tot_g + tot_u
    print(f"=== target-noise characterization ({len(runs)} runs) ===")
    print(f"positive (pass,chunk) cells: {tot}   grounded: {tot_g} ({tot_g/tot:.0%})   "
          f"UNGROUNDED (suspect): {tot_u} ({tot_u/tot:.0%})\n")
    print(f"{'pass':24s}{'grounded':>10s}{'ungrounded':>12s}")
    for p in sorted(per_pass):
        d = per_pass[p]; print(f"{p:24s}{d['grounded']:>10d}{d['ungrounded']:>12d}")
    print("\nsample ungrounded positives (value not found in attributed chunk):")
    for s in samples:
        print(f"  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
