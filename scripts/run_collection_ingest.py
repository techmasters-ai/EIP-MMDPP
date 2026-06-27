#!/usr/bin/env python3
"""Detached sequential full-ingest driver for the notebooks collection (P1).

Survives session close (launched via setsid/nohup). For each of the 8 fresh docs,
in order, it:
  - skips if already COMPLETE on the notebooks_collection source (RESUMABLE),
  - reuses an in-flight run if the doc is already uploaded (no double-upload),
  - else waits for an idle LLM pool, then uploads to notebooks_collection
    (resolves to air_defense_v3 9-pass),
  - waits until that doc reaches a terminal pipeline_status,
  - runs scripts/collect_extraction_data.py over all collection runs (cumulative
    snapshot to reports/collection/p1_collection_narrow_only.csv).

Sequential by construction (one doc terminal before the next uploads) → no gemma4
pool contention. Read/dispatch only via the API + psql; the docker containers run
the actual pipeline independently of any session. Progress → reports/collection/ingest_driver.log.
"""
import json
import os
import subprocess
import time
from pathlib import Path

BASE = "/home/josh/development/EIP-MMDPP"
WORKTREE = f"{BASE}/.worktrees/walltime-c0-telemetry"
COLL_SOURCE = "ddac23e2-b821-4399-b14a-f2197757fabd"   # notebooks_collection (default=air_defense_v3)
NOTEBOOKS = f"{BASE}/notebooks"
LOG = f"{WORKTREE}/reports/collection/ingest_driver.log"
COLLECTOR = f"{WORKTREE}/scripts/collect_extraction_data.py"
API = "http://localhost:8005"
PER_DOC_TIMEOUT_MIN = 300
TERMINAL = ("COMPLETE", "FAILED", "ERROR", "CANCELLED", "PARTIAL_COMPLETE")

# ALL 21 distinct notebooks/ docs, smallest → largest (30MB stress case last).
# Full-ingest every doc on the collection source so the whole corpus is parsed/
# chunked/extracted by CURRENT code (apples-to-apples) — re-ingesting the ones
# previously ingested on SA-2_Sources/baseline_subset under fresh copies here.
DOCS = [
    "Fan_Song_Radar.jpeg",
    "0005_wildweasels.org_SA-2_and_SR-71_17_Apr_2020.pdf.txt",
    "radar2_waveform1.pdf",                                  # may already be in-flight (6369f186)
    "cw_radar.jpg",
    "Microsoft Word - lf99-4297_final-combineddoc.doc - 20080014230.pdf",
    "Radar Basics.pdf",
    "chinese_handwritten_notes.pdf",
    "SNR-75 - Wikipedia.pdf",
    "chinese_handwritten_notes_2.pdf",
    "Handwritten_Text.pdf",
    "V-75 SA-2 GUIDELINE.pdf",
    "radar_textbook_chapter7.pdf",
    "Images_Demo_Doc.pdf",
    "EWIRDB_Production.pdf",                                  # multi-system — key case
    "SA-2_and_SR-71_17_Apr_2020.pdf",
    "chinese_research_paper.pdf",
    "S-75 Dvina _ Military Wiki _ Fandom.pdf",
    "S-75 Dvina.pdf",
    "SA-2 Surface-to-Air Missile _ National Museum of the United States Air Force™ _ Display.pdf",
    "SA-2 Guideline _ Зенитный Ракетный Комплекс С-75 Двина_Десна_Волхов.pdf",
    "Engagement and Fire Control Radars (S-Band, X-band).pdf",  # 30MB stress case
]


def log(m):
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + m + "\n")


def sh(args, timeout=180):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        log("subprocess err: " + str(e)); return ""


def psql(sql):
    return [l for l in sh(["docker", "exec", "eip-mmdpp-postgres-1", "psql", "-U", "eip",
            "-d", "eip", "-t", "-A", "-F", "|", "-c", sql]).splitlines() if l.strip()]


def doc_for(fname):
    """Most-recent doc + its pipeline_status for this filename on the collection source."""
    q = ("SELECT id||'|'||pipeline_status FROM ingest.documents WHERE source_id='" + COLL_SOURCE +
         "' AND filename='" + fname.replace("'", "''") + "' ORDER BY created_at DESC LIMIT 1")
    rows = psql(q)
    if rows and "|" in rows[0]:
        did, st = rows[0].split("|", 1)
        return did, st
    return None, None


def pool_idle():
    for node in ("10.0.1.121", "10.0.1.109"):
        out = sh(["curl", "-s", "-o", "/dev/null", "-w", "%{time_total}", "--max-time", "30",
                  "http://" + node + ":11434/api/generate", "-d",
                  json.dumps({"model": "gemma4:31b", "prompt": "hi", "stream": False,
                              "options": {"num_predict": 1}})], 40)
        try:
            if float(out) > 10:
                return False
        except Exception:
            return False
    return True


def upload(fname):
    out = sh(["curl", "-s", "--max-time", "300", "-X", "POST",
              API + "/v1/sources/" + COLL_SOURCE + "/documents",
              "-F", "file=@" + NOTEBOOKS + "/" + fname], 360)
    try:
        return json.loads(out)["id"]
    except Exception:
        log("UPLOAD parse failed " + fname + ": " + out[:200]); return None


def wait_terminal(doc_id):
    t0 = time.time()
    while time.time() - t0 < PER_DOC_TIMEOUT_MIN * 60:
        rows = psql("SELECT pipeline_status FROM ingest.documents WHERE id='" + doc_id + "'")
        st = rows[0] if rows else "?"
        if st in TERMINAL:
            return st
        time.sleep(60)
    return "TIMEOUT"


def collect():
    runs = psql("SELECT string_agg(id::text,',') FROM ingest.pipeline_runs "
                "WHERE document_id IN (SELECT id FROM ingest.documents WHERE source_id='" + COLL_SOURCE + "')")
    if runs and runs[0]:
        sh(["python3", COLLECTOR, "--runs", runs[0], "--router-mode", "narrow_only",
            "--label", "p1_collection"], 900)


def main():
    log(f"=== ingest driver started pid {os.getpid()} ===")
    for i, fname in enumerate(DOCS, 1):
        tag = f"[{i}/{len(DOCS)}] {fname[:40]}"
        if not Path(f"{NOTEBOOKS}/{fname}").exists():
            log(f"{tag}: FILE MISSING, skip"); continue
        did, st = doc_for(fname)
        if st == "COMPLETE":
            log(f"{tag}: already COMPLETE ({did[:8]}); collect + skip"); collect(); continue
        if st in ("PROCESSING", "PENDING"):
            log(f"{tag}: in-flight ({did[:8]}); waiting")
        else:
            waited = 0
            while not pool_idle() and waited < 30:
                log(f"{tag}: pool busy, wait 2m"); time.sleep(120); waited += 1
            did = upload(fname)
            if not did:
                log(f"{tag}: UPLOAD FAILED, skip"); continue
            log(f"{tag}: uploaded {did[:8]} (full, air_defense_v3)")
        final = wait_terminal(did)
        log(f"{tag}: {final}")
        collect()
    log("=== ingest driver complete ===")


if __name__ == "__main__":
    main()
