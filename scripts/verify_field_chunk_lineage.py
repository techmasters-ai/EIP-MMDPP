#!/usr/bin/env python3
"""Verify the per-field chunk-origin lineage fix on a finished run.

The delta field_provenance rows carry value=None (the row records field_name +
chunk attribution, not the value), so we JOIN each row's field_name to the
field's actual value(s) from the SAME pass's pass_output, then check whether the
chunk the row was attributed to (chunk_indexes) PHYSICALLY CONTAINS that value
(string substring or numeric+unit). This is the honest test: before the fix
every field collapsed to the entity's first-seen (name/title) chunk; after, each
field inherits the chunk of the batch that emitted it (__property_provenance).

Also reports the per-pass ATTRIBUTION SPREAD (distinct chunks fields point to) —
a collapsed spread (all fields → 1 chunk) is the pre-fix signature.

    python3 -m scripts.verify_field_chunk_lineage --run <run> [--compare <old_run>]
"""
from __future__ import annotations
import argparse, base64, json, os, unicodedata, urllib.request
from collections import defaultdict

from app.services.field_value_grounding import (
    nfc as _nfc, num_variants as _num_variants,
    units_for as _units_for, value_in_chunk as _value_in_chunk,
)

_PG = os.environ.get("A0_DATABASE_URL", "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip")
_ARC = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")


def _arc(sql: str):
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = urllib.request.Request(
        f"{_ARC}/api/v1/command/eip_knowledge_graph",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(urllib.request.urlopen(req, timeout=30)).get("result", [])


def _norm(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", _nfc(str(s)).casefold())
                   if not unicodedata.combining(ch))


def _value_grounded(value, attributed_text: str) -> bool:
    """True if `value` physically appears in attributed_text (numeric+unit or
    string substring, both normalized)."""
    if value is None or value == "":
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # numeric path needs a field-unit; without one fall back to bare-number.
        nums = _num_variants(value)
        txt = _norm(attributed_text)
        return any(_norm(n) in txt for n in nums)
    sval = _norm(value)
    if len(sval) < 2:
        return False
    return sval in _norm(attributed_text)


def _pass_field_values(pass_output) -> dict[str, list]:
    """field_name -> [values] across all entity records in a pass_output."""
    out: dict[str, list] = defaultdict(list)
    po = pass_output or {}
    for v in po.values():
        if not isinstance(v, list):
            continue
        for rec in v:
            if not isinstance(rec, dict):
                continue
            for f, val in rec.items():
                if val is not None and val != "" and not isinstance(val, (list, dict)):
                    out[f].append(val)
    return out


def analyze(run: str, label: str):
    chunks = {int(c["chunk_index"]): _nfc(c.get("chunk_text") or "")
              for c in _arc(f"SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}'")}
    from sqlalchemy import create_engine, text as _t
    eng = create_engine(_PG)
    with eng.connect() as c:
        rows = c.execute(_t(
            "SELECT pass_name, attempt, extract_pass_response_json FROM ingest.pipeline_pass_outputs "
            "WHERE pipeline_run_id=:r ORDER BY pass_name, attempt"), {"r": run}).fetchall()
    latest = {}
    for pn, _a, j in rows:
        latest[pn] = json.loads(j) if isinstance(j, str) else j

    grounded = miss = no_value = 0
    spread: dict[str, set] = defaultdict(set)
    miss_examples, grounded_examples = [], []
    for pn, j in latest.items():
        fvals = _pass_field_values((j or {}).get("pass_output"))
        for fp in (j or {}).get("field_provenance") or []:
            fname = fp.get("field_name")
            cis = fp.get("chunk_indexes") or ([fp["chunk_index"]] if fp.get("chunk_index") is not None else [])
            for x in cis:
                if x is not None:
                    spread[pn].add(int(x))
            vals = fvals.get(fname) or []
            if not vals:
                no_value += 1
                continue
            attributed = " ".join(chunks.get(int(x), "") for x in cis if x is not None)
            if any(_value_grounded(v, attributed) for v in vals):
                grounded += 1
                if len(grounded_examples) < 8:
                    grounded_examples.append(f"{pn}.{fname}={vals[:2]} ∈ chunks {cis} ✓")
            else:
                miss += 1
                where = sorted({x for v in vals for x, txt in chunks.items() if _value_grounded(v, txt)})
                if len(miss_examples) < 12:
                    miss_examples.append(f"{pn}.{fname}={vals[:2]} attributed→{cis}, value in chunks {where}")

    total = grounded + miss
    print(f"\n========== {label}: run {run[:8]} ({len(chunks)} chunks) ==========")
    print(f"per-field value-grounding (rows whose field has an extracted value):")
    print(f"  grounded (attributed chunk CONTAINS value): {grounded}")
    print(f"  mis-attributed:                              {miss}")
    if total:
        print(f"  PRECISION: {grounded}/{total} = {grounded/total:.0%}")
    print(f"  (rows whose field value is null/empty, unverifiable: {no_value})")
    print(f"attribution spread (distinct chunks each pass's fields point to):")
    for pn, s in sorted(spread.items()):
        print(f"  {pn:28s} {sorted(s)}")
    if grounded_examples:
        print("grounded examples:")
        for e in grounded_examples:
            print(f"  {e}")
    if miss_examples:
        print("mis-attribution examples:")
        for e in miss_examples:
            print(f"  {e}")
    return grounded, miss


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", help="optional pre-fix run for before/after")
    args = ap.parse_args(argv)
    analyze(args.run, "POST-FIX")
    if args.compare:
        analyze(args.compare, "PRE-FIX (compare)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
