#!/usr/bin/env python3
"""Per-pass keyword discrimination check (recall vs negative-fire) on real data.

Objective (user-refined): for each pass P, a keyword is good iff WITHIN P's rows
it fires on P's POSITIVE chunks (maintains recall) without firing on P's NEGATIVE
chunks (lets us drop them). Cross-pass overlap is allowed — the test is purely
within-pass positives-vs-negatives. Positives come from the FIXED lineage target
(field_provenance.chunk_indexes per pass); negatives = all other chunks in the run.

Pure text scan (NFC+casefold substring) over the curated bake-off corpus. No model.

    python3 -m scripts.keyword_discrimination_check --runs <r1,r2,...>

Normalisation note: text and keyword needles are normalised with NFC+casefold —
identical to the runtime ``keyword_hit_counts`` function — so pos-fire/neg-fire
rates predict runtime matching behaviour exactly.  (Prior versions used
NFKD+strip-combining, which differs from the runtime path.)
"""
from __future__ import annotations
import argparse, base64, json, os, unicodedata, urllib.request
from collections import defaultdict

_PG = os.environ.get("A0_DATABASE_URL", "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip")
_ARC = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")

# Proposed signature keywords per field-group pass (instance-free, not aliases).
PROPOSED: dict[str, list[str]] = {
    "radar_power_rf":      ["magnetron","klystron","traveling wave tube","TWT","duplexer",
                            "power amplifier","RF output","average power","duty cycle"],
    "radar_antenna":       ["reflector","feed horn","parabolic","phased array","sidelobe",
                            "boresight","radome","array face","aperture"],
    "radar_modulation":    ["pulse compression","matched filter","chirp","Barker","coherent integration",
                            "frequency agility","phase coding","modulation on pulse","MOP"],
    "radar_timing":        ["antenna rotation","rotation period","scan rate","revisit",
                            "pulse repetition interval","interpulse","rpm"],
    "missile_kinematics":  ["engagement envelope","kill zone","no-escape zone","engagement boundary",
                            "intercept envelope","footprint","slant range"],
    "missile_airframe":    ["fuselage","fins","canard","wingspan","fin span","control surfaces",
                            "nose cone","airframe"],
    "missile_propulsion":  ["rocket motor","propellant","nozzle","specific impulse","solid propellant",
                            "liquid propellant","oxidizer","grain","thrust"],
    "missile_guidance":    ["seeker","homing","midcourse","proportional navigation","command link",
                            "datalink","illuminator","gimbal","terminal phase"],
    "missile_speed_timing":["burnout","boost phase","coast phase","flyout","terminal velocity",
                            "supersonic","hypersonic","mach"],
}


def _nfc(t: str) -> str:
    """NFC-normalise + casefold — matches runtime ``keyword_hit_counts`` exactly."""
    return unicodedata.normalize("NFC", t).casefold()


def _arc(sql: str):
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = urllib.request.Request(f"{_ARC}/api/v1/command/eip_knowledge_graph",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(urllib.request.urlopen(req, timeout=40)).get("result", [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    from sqlalchemy import create_engine, text as _t
    eng = create_engine(_PG)
    # accumulate per pass: list of (chunk_text) for positives and negatives across runs
    pos_texts: dict[str, list[str]] = defaultdict(list)
    neg_texts: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        chunks = {int(c["chunk_index"]): _nfc(c.get("chunk_text") or "")
                  for c in _arc(f"SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}'")}
        with eng.connect() as c:
            rows = c.execute(_t("SELECT pass_name, extract_pass_response_json FROM ingest.pipeline_pass_outputs "
                                "WHERE pipeline_run_id=:r ORDER BY pass_name, attempt"), {"r": run}).fetchall()
        latest = {pn: (json.loads(j) if isinstance(j, str) else j) for pn, j in rows}
        for pass_name in PROPOSED:
            j = latest.get(pass_name)
            pos: set[int] = set()
            if j:
                for fp in (j.get("field_provenance") or []):
                    for ci in (fp.get("chunk_indexes") or []):
                        if isinstance(ci, int):
                            pos.add(ci)
                    ci = fp.get("chunk_index")
                    if isinstance(ci, int):
                        pos.add(ci)
            for ci, txt in chunks.items():
                (pos_texts if ci in pos else neg_texts)[pass_name].append(txt)

    print(f"=== keyword discrimination (within-pass recall vs negative-fire), {len(runs)} runs ===\n")
    for pass_name, kws in PROPOSED.items():
        P = pos_texts[pass_name]; Ng = neg_texts[pass_name]
        nP, nN = len(P), len(Ng)
        if nP == 0:
            print(f"## {pass_name}  (0 positives — skip)\n")
            continue
        norm_k = [(_nfc(k), k) for k in kws]
        print(f"## {pass_name}  ({nP} pos / {nN} neg chunks)")
        print(f"   {'keyword':22s} {'pos-fire':>8s} {'neg-fire':>8s} {'lift':>6s}")
        set_pos_hit = [False]*nP; set_neg_hit = [False]*nN
        for kn, kw in norm_k:
            pf = sum(1 for t in P if kn in t)
            nf = sum(1 for t in Ng if kn in t)
            pfr, nfr = pf/nP, (nf/nN if nN else 0)
            for i, t in enumerate(P):
                if kn in t: set_pos_hit[i] = True
            for i, t in enumerate(Ng):
                if kn in t: set_neg_hit[i] = True
            flag = "  <<" if pfr - nfr >= 0.10 else ("  drop?" if pfr <= nfr else "")
            print(f"   {kw:22s} {pfr:8.2f} {nfr:8.2f} {pfr-nfr:+6.2f}{flag}")
        sr = sum(set_pos_hit)/nP; sn = sum(set_neg_hit)/nN if nN else 0
        print(f"   {'— SET (any keyword)':22s} {sr:8.2f} {sn:8.2f} {sr-sn:+6.2f}   "
              f"=> recall {sr:.0%}, fires on {sn:.0%} of negatives\n")
    print("lift = pos-fire − neg-fire. '<<' = strong discriminator (≥0.10). "
          "'drop?' = fires on negatives ≥ positives (no within-pass value).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
