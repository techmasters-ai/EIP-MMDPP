#!/usr/bin/env python3
"""Clean per-(chunk, pass) 'extracted-from' ground truth via value→chunk matching.

The pipeline's field_provenance is broken (degenerate rows, scope-based chunk
attribution — see the chunk-9 "50 km" diagnosis). This re-derives the CORRECT
target directly: for each field VALUE a pass actually extracted (from its
pass_output), find the chunk(s) whose text physically contains that value next
to its unit, and label (chunk, pass) = extracted_from. That is the honest
answer to "did this pass extract information from this chunk."

Reads pass_output from postgres + chunk text from ArcadeDB. Read-only.
Output: reports/collection/extracted_from_groundtruth.csv (run, chunk, pass,
extracted_from, matched_fields) + a console summary vs the old broken labels.

    python3 -m scripts.build_extracted_from_groundtruth --runs <r1>,<r2>,<r3>
"""
from __future__ import annotations
import argparse, base64, csv, json, os
from collections import defaultdict

# The value→chunk matcher lives in the shared pure module so the production
# worker lineage path (app/workers/pipeline.py) and this ground-truth tooling
# use the IDENTICAL matching logic (no divergence).
from app.services.field_value_grounding import (
    SUFFIX_UNITS,  # noqa: F401  (re-exported for callers/tests that import it here)
    nfc as _nfc,
    num_variants as _num_variants,
    units_for as _units_for,
    value_in_chunk as _value_in_chunk,
)

DEFAULT_RUNS = ("88596dc3-5d6f-4e47-97b4-4ce3d6cd746c",
                "3100b632-c584-4649-a6f2-faa2f5f3a000",
                "2c90bd0b-7d67-4da6-bfd1-1f9924a81284")

_PG = os.environ.get("A0_DATABASE_URL", "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip")
_ARC = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")


def _arc(sql: str):
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = __import__("urllib.request", fromlist=["request"]).Request(
        f"{_ARC}/api/v1/command/eip_knowledge_graph",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(__import__("urllib.request", fromlist=["request"]).urlopen(req, timeout=30)).get("result", [])


def build(runs):
    from sqlalchemy import create_engine, text as _t
    eng = create_engine(_PG)
    # extracted-from[(run, chunk)] = {pass: [field=value,...]}
    ef: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    pass_value_counts: dict[str, int] = defaultdict(int)
    for run in runs:
        chunks = {int(c["chunk_index"]): _nfc(c.get("chunk_text") or "")
                  for c in _arc(f"SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}'")}
        with eng.connect() as c:
            rows = c.execute(_t(
                "SELECT pass_name, extract_pass_response_json FROM ingest.pipeline_pass_outputs "
                "WHERE pipeline_run_id=:r ORDER BY pass_name, attempt"), {"r": run}).fetchall()
        latest = {}
        for pn, j in rows:
            latest[pn] = j  # last attempt wins
        for pn, j in latest.items():
            if isinstance(j, str):
                j = json.loads(j)
            po = (j or {}).get("pass_output") or {}
            records = next((v for v in po.values() if isinstance(v, list)), [])
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                for field, val in rec.items():
                    if val is None or field in ("system_name",) or not isinstance(val, (int, float)):
                        continue
                    units = _units_for(field)
                    if not units:
                        continue
                    nums = _num_variants(val)
                    pass_value_counts[pn] += 1
                    for ci, txt in chunks.items():
                        if _value_in_chunk(nums, units, txt):
                            ef[(run, ci)][pn].append(f"{field}={val}")
    return ef, pass_value_counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--rows", default="reports/collection/a0_fitvalidate_rows_88596dc3.csv",
                    help="old labels for comparison")
    ap.add_argument("--out", default="reports/collection/extracted_from_groundtruth.csv")
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    ef, pvc = build(runs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "chunk", "pass", "extracted_from", "matched_fields"])
        # emit a row per (run, chunk, pass) that matched
        for (run, ci), passes in sorted(ef.items()):
            for pn, fields in passes.items():
                w.writerow([run[:8], ci, pn, True, ";".join(sorted(set(fields)))])

    # summary
    n_chunks_with = len(ef)
    n_links = sum(len(p) for p in ef.values())
    print(f"=== clean 'extracted_from' ground truth ({len(runs)} runs) ===")
    print(f"chunks that a pass extracted a VALUE from: {n_chunks_with}")
    print(f"(chunk,pass) extracted-from links: {n_links}")
    print("\nextracted-VALUE fields scored per pass (how much real data each pass produced):")
    for pn, n in sorted(pvc.items(), key=lambda kv: -kv[1]):
        links = sum(1 for (_, _), ps in ef.items() if pn in ps)
        print(f"  {pn:24s} values={n:3d}  → matched to {links} chunk(s)")

    # compare to old broken used_field_level
    try:
        old = [r for r in csv.DictReader(open(args.rows))]
        oldfld = {(r["run_id"][:8], r["chunk_index"], r["pass_name"]) for r in old if r.get("used_field_level") == "True"}
        newset = {(run[:8], str(ci), pn) for (run, ci), ps in ef.items() for pn in ps}
        print(f"\nvs OLD used_field_level: old={len(oldfld)} links, new(clean)={len(newset)} links, "
              f"overlap={len(oldfld & newset)}  (low overlap ⇒ old labels were wrong)")
    except FileNotFoundError:
        pass
    print(f"\nwrote → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
