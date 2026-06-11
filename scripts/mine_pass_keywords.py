#!/usr/bin/env python3
"""Data-driven mining of prose-friendly, GENERALIZABLE per-pass keywords.

For each field-group pass, find unigrams/bigrams that appear in the pass's
POSITIVE chunks (lineage target) materially more than its NEGATIVE chunks
(within-pass discrimination), AND that recur across MULTIPLE documents (so they
generalize rather than overfit one doc). Excludes numbers/units, stopwords,
short tokens, and instance-designation patterns (digit-bearing, letter-dash-digit)
to honor the no-instance-names guardrail. Output is a REVIEW list — not
auto-committed.

    python3 -m scripts.mine_pass_keywords --runs <r1,r2,...>

Normalisation note: text is normalised with NFC+casefold — identical to the
runtime ``keyword_hit_counts`` function — so lift/pos-fire/neg-fire statistics
predict runtime matching behaviour exactly.  (Prior versions used NFKD+strip-
combining, which differs from the runtime path.)
"""
from __future__ import annotations
import argparse, base64, json, os, re, unicodedata, urllib.request
from collections import defaultdict

_PG = os.environ.get("A0_DATABASE_URL", "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip")
_ARC = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")
PASSES = ["radar_power_rf","radar_antenna","radar_modulation","radar_timing",
          "missile_kinematics","missile_airframe","missile_propulsion",
          "missile_guidance","missile_speed_timing"]

STOP = set("the a an and or of to in for with on at by is are be as it its this that "
           "from into over under per each both all any can may which when where while "
           "such not no only also more most than then they them their these those has "
           "have had was were will would could should one two three first second other "
           "between within about above below up down out off was been being but if".split())
WORD = re.compile(r"[a-z][a-z\-]{2,}")          # alpha tokens, len>=3, allows hyphen
DESIG = re.compile(r"\d|^[a-z]{1,3}-?\d")        # digit-bearing or letter-dash-digit → instance-ish


def _nfc(t: str) -> str:
    """NFC-normalise + casefold — matches runtime ``keyword_hit_counts`` exactly."""
    return unicodedata.normalize("NFC", t).casefold()


def _arc(sql: str):
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = urllib.request.Request(f"{_ARC}/api/v1/command/eip_knowledge_graph",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(urllib.request.urlopen(req, timeout=40)).get("result", [])


def _terms(text: str) -> set[str]:
    toks = [w for w in WORD.findall(text) if w not in STOP and not DESIG.search(w)]
    out = set(toks)
    for a, b in zip(toks, toks[1:]):                 # bigrams
        out.add(f"{a} {b}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--min-lift", type=float, default=0.10)
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--min-posfire", type=float, default=0.15)
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    from sqlalchemy import create_engine, text as _t
    eng = create_engine(_PG)
    # per pass: pos[doc] = list of chunk-term-sets ; neg[doc] = list
    pos: dict[str, dict[str, list[set]]] = {p: defaultdict(list) for p in PASSES}
    neg: dict[str, dict[str, list[set]]] = {p: defaultdict(list) for p in PASSES}
    for run in runs:
        chunks = {int(c["chunk_index"]): _terms(_nfc(c.get("chunk_text") or ""))
                  for c in _arc(f"SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}'")}
        with eng.connect() as c:
            rows = c.execute(_t("SELECT pass_name, extract_pass_response_json FROM ingest.pipeline_pass_outputs "
                                "WHERE pipeline_run_id=:r ORDER BY pass_name, attempt"), {"r": run}).fetchall()
        latest = {pn: (json.loads(j) if isinstance(j, str) else j) for pn, j in rows}
        for p in PASSES:
            j = latest.get(p); posset: set[int] = set()
            if j:
                for fp in (j.get("field_provenance") or []):
                    posset.update(x for x in (fp.get("chunk_indexes") or []) if isinstance(x, int))
                    if isinstance(fp.get("chunk_index"), int):
                        posset.add(fp["chunk_index"])
            for ci, terms in chunks.items():
                (pos[p] if ci in posset else neg[p])[run].append(terms)

    for p in PASSES:
        all_pos = [s for docs in pos[p].values() for s in docs]
        all_neg = [s for docs in neg[p].values() for s in docs]
        nP, nN = len(all_pos), len(all_neg)
        if nP == 0:
            print(f"## {p}: 0 positives — skip\n"); continue
        # candidate terms = those in any positive
        cand = set().union(*all_pos) if all_pos else set()
        scored = []
        for term in cand:
            pf = sum(1 for s in all_pos if term in s) / nP
            nf = (sum(1 for s in all_neg if term in s) / nN) if nN else 0.0
            docspread = sum(1 for run, docs in pos[p].items() if any(term in s for s in docs))
            if pf >= args.min_posfire and (pf - nf) >= args.min_lift and docspread >= args.min_docs:
                scored.append((pf - nf, pf, nf, docspread, term))
        scored.sort(reverse=True)
        print(f"## {p}  ({nP} pos / {nN} neg)  candidates: {len(scored)}")
        print(f"   {'lift':>5s} {'posf':>5s} {'negf':>5s} {'docs':>4s}  term")
        for lift, pf, nf, ds, term in scored[:18]:
            print(f"   {lift:+5.2f} {pf:5.2f} {nf:5.2f} {ds:4d}  {term}")
        print()
    print(f"filters: pos-fire≥{args.min_posfire}, lift≥{args.min_lift}, docs≥{args.min_docs} "
          "(generalizable). Numbers/units/stopwords/designations excluded. REVIEW before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
