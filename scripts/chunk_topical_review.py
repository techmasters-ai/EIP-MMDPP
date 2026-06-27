#!/usr/bin/env python3
"""Complete topical review of extraction chunks.

For every chunk in the given run(s): classify content-bearing vs
boilerplate/image/prose, count value-tokens (number+unit = extractable spec),
score topical relevance to each field-group pass (domain-general vocab, NOT the
schema's own aliases — an independent read), and cross-reference the lineage
labels (used = any provenance; used_fld = field-VALUE provenance).

Reads chunk text from ArcadeDB (HTTP) and labels from the a0 rows CSV. Pure
analysis, read-only. Outputs a per-chunk CSV + a console summary.

    python3 -m scripts.chunk_topical_review --rows reports/collection/a0_fitvalidate_rows_88596dc3.csv
"""
from __future__ import annotations
import argparse, base64, csv, json, os, re, urllib.request
from collections import defaultdict

# --- domain-general topical vocab per pass (independent of the schema aliases) ---
TOPICS: dict[str, list[str]] = {
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
    "missile_guidance":    ["guidance","seeker","homing","command","semi-active","sarh","radar homing",
                            "midcourse","terminal","beam riding","proportional navigation","autopilot",
                            "command link","track-via-missile"],
    "missile_airframe":    ["length","diameter","span","wingspan","mass","weight","warhead","fins","canard",
                            "airframe","fuselage","body diameter","kg","calibre","caliber"],
    "missile_propulsion":  ["motor","rocket motor","propellant","thrust","booster","sustainer","stage",
                            "two-stage","solid fuel","liquid fuel","burn time","impulse","nozzle","boost"],
    "missile_speed_timing":["burnout","flight time","time of flight","boost phase","burn duration",
                            "acceleration","mach","seconds of flight"],
}
VALUE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:km|m/s|kg|mhz|ghz|khz|kw|mw|dbw|dbm|deg|°|mm|cm|nmi|mach|rpm|µs|us|ms|sec|s\b|m\b|w\b)", re.I)
BOILER_RE = re.compile(r"©|copyright|technical report|all rights reserved|\bby dr\b|\blinks\b|\bpage \d+|table of contents", re.I)
IMAGE_RE  = re.compile(r"this image is classified|image consists of|image depicts|photographic capture|block_diagram|confidence:\s|rendering of", re.I)
HEADER_RE = re.compile(r"Almaz S-75[^\n]{0,120}", re.I)  # the running document header (doc-specific noise)


def _arcadedb(sql: str) -> list[dict]:
    base = os.environ.get("A0_ARCADEDB_URL", "http://localhost:2480").rstrip("/")
    db = os.environ.get("A0_ARCADEDB_DB", "eip_knowledge_graph")
    auth = base64.b64encode(b"root:eip_arcadedb_secret").decode()
    req = urllib.request.Request(f"{base}/api/v1/command/{db}",
        data=json.dumps({"command": sql, "language": "sql"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    return json.load(urllib.request.urlopen(req, timeout=30)).get("result", [])


def classify(text: str) -> dict:
    raw = text or ""
    body = HEADER_RE.sub(" ", raw)          # strip running header before judging
    low = body.lower()
    n_val = len(VALUE_RE.findall(body))
    topic_scores = {p: sum(low.count(t) for t in terms) for p, terms in TOPICS.items()}
    top = max(topic_scores.items(), key=lambda kv: kv[1])
    top_topic, top_score = (top[0], top[1]) if top[1] > 0 else ("none", 0)
    is_image = bool(IMAGE_RE.search(body))
    is_boiler = bool(BOILER_RE.search(raw)) and n_val == 0 and top_score <= 1
    # content class
    if n_val >= 1 and top_score >= 1:
        cls = "value_bearing"          # has a spec value AND topical → the real target
    elif is_image:
        cls = "image_caption"
    elif is_boiler:
        cls = "boilerplate"
    elif top_score >= 2:
        cls = "topical_prose"         # topical language but no extractable value
    else:
        cls = "low_signal"
    return {"len": len(raw), "value_tokens": n_val, "content_class": cls,
            "top_topic": top_topic, "top_topic_score": top_score, "topic_scores": topic_scores}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True, help="a0 rows CSV (per pass/chunk + used/used_field_level)")
    ap.add_argument("--out", default="reports/collection/chunk_topical_review.csv")
    args = ap.parse_args(argv)

    rows = [r for r in csv.DictReader(open(args.rows))]
    runs = sorted({r["run_id"] for r in rows if r.get("run_id")})
    # labels per (run, chunk_index): which passes used it (field-level vs any)
    used_any: dict[tuple, set] = defaultdict(set)
    used_fld: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if r.get("chunk_index") in (None, "", "None"):
            continue
        k = (r["run_id"], int(r["chunk_index"]))
        if r.get("used") == "True": used_any[k].add(r["pass_name"])
        if r.get("used_field_level") == "True": used_fld[k].add(r["pass_name"])

    out_rows = []
    for run in runs:
        chunks = _arcadedb(f"SELECT chunk_index, chunk_text FROM ExtractionChunk WHERE pipeline_run_id='{run}' ORDER BY chunk_index")
        for c in chunks:
            ci = int(c["chunk_index"]); k = (run, ci)
            cl = classify(c.get("chunk_text") or "")
            out_rows.append({"run": run[:8], "chunk": ci, **{x: cl[x] for x in ("len","value_tokens","content_class","top_topic","top_topic_score")},
                             "used_fld_passes": ";".join(sorted(used_fld.get(k, ()))),
                             "used_any_passes": ";".join(sorted(used_any.get(k, ()))),
                             "_topic_scores": cl["topic_scores"], "_run": run})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run","chunk","len","value_tokens","content_class","top_topic","top_topic_score","used_fld_passes","used_any_passes"])
        for o in out_rows:
            w.writerow([o["run"],o["chunk"],o["len"],o["value_tokens"],o["content_class"],o["top_topic"],o["top_topic_score"],o["used_fld_passes"],o["used_any_passes"]])

    # ---- summary ----
    print(f"=== TOPICAL REVIEW: {len(out_rows)} chunks across {len(runs)} docs ===\n")
    print("content-class distribution per doc:")
    by_run = defaultdict(lambda: defaultdict(int))
    for o in out_rows: by_run[o["run"]][o["content_class"]] += 1
    classes = ["value_bearing","topical_prose","image_caption","boilerplate","low_signal"]
    print(f"  {'doc':10s} " + " ".join(f"{c:>14s}" for c in classes) + "  total")
    for run in sorted(by_run):
        d = by_run[run]; print(f"  {run:10s} " + " ".join(f"{d.get(c,0):14d}" for c in classes) + f"  {sum(d.values())}")

    print("\nLABEL CROSS-CHECK:")
    vb = [o for o in out_rows if o["content_class"]=="value_bearing"]
    vb_labeled = [o for o in vb if o["used_fld_passes"]]
    fld_labeled = [o for o in out_rows if o["used_fld_passes"]]
    fld_not_vb = [o for o in fld_labeled if o["content_class"]!="value_bearing"]
    any_not_content = [o for o in out_rows if o["used_any_passes"] and o["content_class"] in ("boilerplate","image_caption","low_signal")]
    print(f"  value_bearing chunks: {len(vb)}   of which field-labeled: {len(vb_labeled)}  (RECALL of real content)")
    print(f"  field-labeled chunks: {len(fld_labeled)}   of which NOT value_bearing: {len(fld_not_vb)}  (label noise)")
    print(f"  'used(any)'-labeled chunks that are boilerplate/image/low-signal: {len(any_not_content)}  (entity-mention pollution)")

    print("\nTOPICAL COVERAGE — value_bearing chunks per top_topic (where the real data is):")
    cov = defaultdict(int)
    for o in vb: cov[o["top_topic"]] += 1
    for t,n in sorted(cov.items(), key=lambda kv:-kv[1]): print(f"  {t:22s} {n}")

    print(f"\nwrote per-chunk table → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
