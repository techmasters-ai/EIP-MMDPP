#!/usr/bin/env python3
"""Survey ALL ingested documents for bake-off usefulness — WITHOUT re-running
extraction.

A doc is useful for the per-(pass,chunk) calibration bake-off iff it yields
many POSITIVE (field-group pass, chunk) cells — i.e. chunks carrying numeric
spec values that a field-group pass would extract. Identity-pass content (entity
names) is NOT counted: those passes capture no score_components, so they can't
contribute bake-off rows.

For each doc we read its TextChunks (ArcadeDB) and, per chunk, detect
value-bearing spec content (number+unit) scored against each FIELD-GROUP pass's
domain vocabulary. We then estimate, per doc:
  - value_chunks         : chunks with >=1 number+unit token AND a field-group topic hit
  - est_positive_cells   : sum over field-group passes of (chunks that hit that pass's
                           domain with a value) — the rough # of positive cells the doc
                           would produce in the bake-off
  - domains_covered      : how many of the 9 field-group passes get >=1 value chunk
Read-only. Ranks docs and prints a recommended subset.

    python3 -m scripts.survey_docs_for_bakeoff
"""
from __future__ import annotations
import base64, json, os, re, urllib.request
from collections import defaultdict

_PG = os.environ.get("A0_DATABASE_URL", "postgresql+psycopg2://eip:eip_secret@localhost:5437/eip")
_ARC = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")

# FIELD-GROUP passes only (the ones that capture score_components → bake-off rows).
# Identity passes (missile_identity/radar_identity) + system_links excluded.
FIELD_GROUP_TOPICS: dict[str, list[str]] = {
    "radar_power_rf":      ["frequency","mhz","ghz","khz","s-band","x-band","c-band","l-band","ku-band",
                            "band","transmitter","erp","eirp","dbw","dbm","peak power","transmit power",
                            "rf","carrier","magnetron","klystron","watt","kw","mw","emission"],
    "radar_antenna":       ["antenna","beamwidth","beam width","gain","aperture","reflector","dish","array",
                            "dipole","sidelobe","side lobe","polariz","boresight","dbi","azimuth beam",
                            "elevation beam","feed horn"],
    "radar_modulation":    ["prf","pulse repetition","pulse width","pulsewidth","waveform","modulation",
                            "chirp","fmcw","duty cycle","compression","coherent","bandwidth","pulse"],
    "radar_timing":        ["pri","scan rate","scan period","dwell","revisit","rotation","rpm","frame time",
                            "update rate","scan time","antenna rotation"],
    "missile_kinematics":  ["velocity","speed","range","altitude","ceiling","trajectory","intercept",
                            "engagement","slant range","max range","min range","km","m/s","mach","envelope"],
    "missile_airframe":    ["length","diameter","span","wingspan","mass","weight","warhead","fins","canard",
                            "airframe","fuselage","body diameter","kg","calibre","caliber"],
    "missile_propulsion":  ["motor","rocket motor","propellant","thrust","booster","sustainer","stage",
                            "two-stage","solid fuel","liquid fuel","burn time","impulse","nozzle","boost"],
    "missile_guidance":    ["guidance","seeker","homing","command","semi-active","sarh","radar homing",
                            "midcourse","terminal","beam riding","proportional navigation","autopilot",
                            "command link","track-via-missile"],
    "missile_speed_timing":["burnout","flight time","time of flight","boost phase","burn duration",
                            "acceleration","mach","seconds of flight"],
}
VALUE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:km|m/s|kg|mhz|ghz|khz|kw|mw|dbw|dbm|deg|°|mm|cm|nmi|mach|rpm|µs|us|ms|sec|s\b|m\b|w\b|%)", re.I)


def _arc(sql: str):
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = urllib.request.Request(
        f"{_ARC}/api/v1/command/eip_knowledge_graph",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(urllib.request.urlopen(req, timeout=40)).get("result", [])


def main() -> int:
    from sqlalchemy import create_engine, text as _t
    eng = create_engine(_PG)
    with eng.connect() as c:
        docs = c.execute(_t(
            "SELECT id, filename FROM ingest.documents WHERE pipeline_status IN "
            "('completed','complete','COMPLETE','succeeded','ready')")).fetchall()
        # latest run per doc (any terminal status), newest first
        run_rows = c.execute(_t(
            "SELECT document_id, id FROM ingest.pipeline_runs ORDER BY started_at DESC")).fetchall()
    names = {str(d[0]): (d[1] or str(d[0])[:8]) for d in docs}

    # TextChunk.chunk_text is empty in ArcadeDB; the real text lives in per-run
    # ExtractionChunk. Map each doc → its LATEST run that has ExtractionChunk text.
    runs_with_text = {str(x["pipeline_run_id"]) for x in
                      _arc("SELECT pipeline_run_id, count(*) AS n FROM ExtractionChunk "
                           "WHERE chunk_text IS NOT NULL GROUP BY pipeline_run_id")}
    doc_to_run: dict[str, str] = {}
    for did, rid in run_rows:
        d, r = str(did), str(rid)
        if d not in doc_to_run and r in runs_with_text:
            doc_to_run[d] = r

    rows = []
    for did, name in names.items():
        run = doc_to_run.get(did)
        if not run:
            continue
        chunks = _arc(f"SELECT chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}'")
        n_chunks = len(chunks)
        if n_chunks == 0:
            continue
        value_chunks = 0
        domain_value_chunks: dict[str, int] = defaultdict(int)
        for ch in chunks:
            txt = (ch.get("chunk_text") or "")
            low = txt.lower()
            has_value = bool(VALUE_RE.search(txt))
            if not has_value:
                continue
            hit_any = False
            for pass_name, terms in FIELD_GROUP_TOPICS.items():
                if any(t in low for t in terms):
                    domain_value_chunks[pass_name] += 1
                    hit_any = True
            if hit_any:
                value_chunks += 1
        est_cells = sum(domain_value_chunks.values())  # (pass,chunk) positive cells ~
        domains = sum(1 for v in domain_value_chunks.values() if v > 0)
        rows.append({
            "doc": did[:8], "name": name[:46], "chunks": n_chunks,
            "value_chunks": value_chunks, "est_positive_cells": est_cells,
            "domains": domains,
            "top_domains": ",".join(f"{k.split('_',1)[1] if '_' in k else k}:{v}"
                                    for k, v in sorted(domain_value_chunks.items(), key=lambda kv: -kv[1])[:4]),
        })

    rows.sort(key=lambda r: (-r["est_positive_cells"], -r["domains"]))
    print(f"=== DOC SURVEY for bake-off usefulness ({len(rows)} docs) ===")
    print(f"{'doc':9s}{'chunks':>7s}{'valchk':>7s}{'cells':>6s}{'doms':>5s}  name / top field-group domains")
    for r in rows:
        print(f"{r['doc']:9s}{r['chunks']:>7d}{r['value_chunks']:>7d}{r['est_positive_cells']:>6d}{r['domains']:>5d}  "
              f"{r['name']:46s} [{r['top_domains']}]")

    # heuristic recommendation: enough value chunks AND domain spread
    useful = [r for r in rows if r["est_positive_cells"] >= 6 and r["domains"] >= 3]
    print(f"\nRECOMMENDED (est_positive_cells>=6 AND domains>=3): {len(useful)} docs")
    for r in useful:
        print(f"  {r['doc']}  {r['name']}")
    weak = [r for r in rows if r not in useful]
    print(f"\nWEAK / SKIP ({len(weak)}): " + ", ".join(f"{r['doc']}({r['est_positive_cells']})" for r in weak))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
