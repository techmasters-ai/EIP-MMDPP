#!/usr/bin/env python3
"""End-to-end lineage verification gate (Task 6 of the precise-lineage fix).

Proves, on a real SA-2 graph_only run on the FIXED build, that THIS run restored
PRECISE, run-attributed entity->chunk lineage. The hard discriminator is the
RUN-SCOPED EXTRACTED_FROM edge count (filtered by pipeline_run_id) — EXTRACTED_FROM
is append-only and never purged, so a GLOBAL count is polluted by prior coarse runs
and pre-existing MENTIONED_IN edges. The checks here close two holes a prior review
found:

  * false-PASS: the old trace + any(edge>0) existence checks were satisfied by
    pre-existing MENTIONED_IN with EXTRACTED_FROM=0. Now the lineage proof is the
    run-scoped EXTRACTED_FROM count (#1) and the trace is EXTRACTED_FROM-only,
    edge-anchored, run-scoped (#4).
  * false-FAIL: the old `post - pre > 0` entity-count delta is 0 on idempotent
    re-runs (upsert). Replaced by an ABSOLUTE global recall sanity check (#2).

Checks:
  1. (HARD) run-scoped EXTRACTED_FROM count > 0 — the lineage discriminator.
  2. (GLOBAL sanity) total entity vertices >= --merged (entity vertices carry no
     run_id, so this can't be run-scoped; it is NOT the lineage proof — #1 is).
  3. fan-out-width precision: NO entity links to >= 80% of the doc's text chunks
     (the coarse all-chunks fan-out signature). Distinct chunks per entity are
     computed in Python (ArcadeDB has no count(DISTINCT)).
  4. EXTRACTED_FROM-only, edge-anchored, run-scoped trace -> chunk + doc + page;
     plus run-windowed provenance-drop / LINEAGE_GATE-rejection warning checks.
  5. the traced page is chunk-sourced (it is the edge target chunk's page_number).

Read-only. Usage:
    python3 scripts/verify_lineage_e2e.py --run <sa2_run_id> [--doc <doc_id>] [--merged N]
"""
import argparse, json, subprocess, sys

ADB_URL = "http://localhost:2480/api/v1/command/eip_knowledge_graph"
ADB_AUTH = "root:eip_arcadedb_secret"
SA2_DOC_ID = "ddaa9e36-2854-47c3-bc94-ff38d531dafd"
ENTITY_TYPES = ["RADAR_SYSTEM", "MISSILE_SYSTEM", "FIRE_CONTROL_SYSTEM", "WEAPON_SYSTEM",
                "EQUIPMENT_SYSTEM", "LAUNCHER_SYSTEM", "ELECTRONIC_WARFARE_SYSTEM",
                "AIR_DEFENSE_ARTILLERY_SYSTEM", "INTEGRATED_AIR_DEFENSE_SYSTEM"]
# fraction of the document's text chunks that, if a single entity links to it,
# signals the old coarse "fan out to all chunks" bug rather than precise lineage.
FANOUT_FAIL_FRACTION = 0.80


def adb(sql):
    out = subprocess.run(
        ["curl", "-s", "--max-time", "15", "-u", ADB_AUTH, "-X", "POST", ADB_URL,
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"language": "sql", "command": sql})],
        capture_output=True, text=True, timeout=25).stdout
    try:
        return json.loads(out).get("result", [])
    except Exception:
        return []


def count(sql_count_expr):
    r = adb(sql_count_expr)
    return r[0].get("count", r[0].get("n", 0)) if r else 0


def pg(sql):
    out = subprocess.run(
        ["docker", "exec", "eip-mmdpp-postgres-1", "psql", "-U", "eip", "-d", "eip",
         "-t", "-A", "-F", "|", "-c", sql], capture_output=True, text=True, timeout=30).stdout
    return [l for l in out.splitlines() if l.strip()]


def _to_rfc3339(ts):
    """Convert a psql timestamp (e.g. '2026-05-31 13:18:32.203173+00') to the
    RFC3339 form `docker logs --since/--until` accepts ('2026-05-31T13:18:32.203173+00:00').

    docker rejects the space-separated psql form and a bare '+00' offset; it wants
    a 'T' separator and a colon in the timezone offset. Returns None when the input
    is empty/null so callers can omit the flag.
    """
    if not ts:
        return None
    ts = ts.strip()
    if not ts:
        return None
    # space -> 'T' (date/time separator)
    ts = ts.replace(" ", "T", 1)
    # '+00' / '-00' style offset (no colon) -> '+00:00'
    for sign in ("+", "-"):
        idx = ts.rfind(sign)
        # only treat as an offset if it's in the time portion (after 'T')
        if idx > ts.find("T") and ":" not in ts[idx:]:
            off = ts[idx + 1:]
            if len(off) <= 2:
                off = off.zfill(2) + "00"
            ts = ts[:idx + 1] + off[:2] + ":" + off[2:4]
            break
    return ts


def run_window(run_id):
    """Return (since, until) RFC3339 strings for the run's Postgres time window.

    started_at / finished_at live in ingest.pipeline_runs. until may be None when
    the run has not finished (caller then omits --until)."""
    rows = pg(f"SELECT started_at, finished_at FROM ingest.pipeline_runs WHERE id = '{run_id}'")
    if not rows:
        return None, None
    parts = rows[0].split("|")
    started = _to_rfc3339(parts[0]) if len(parts) > 0 else None
    finished = _to_rfc3339(parts[1]) if len(parts) > 1 else None
    return started, finished


def resolve_doc(run_id, fallback):
    """Derive the document id from the run; fall back to --doc when unavailable."""
    rows = pg(f"SELECT document_id FROM ingest.pipeline_runs WHERE id = '{run_id}'")
    if rows and rows[0].strip():
        return rows[0].strip(), "derived-from-run"
    return fallback, "fallback --doc"


def _docker_logs(container, since, until):
    cmd = ["docker", "logs", container]
    if since:
        cmd += ["--since", since]
    if until:
        cmd += ["--until", until]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.stdout + r.stderr


def total_entities():
    return sum(count(f"SELECT count(*) AS count FROM `{t}`") for t in ENTITY_TYPES)


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end PRECISE-lineage verification gate. Run-scopes "
                    "EXTRACTED_FROM by pipeline_run_id (the hard discriminator).")
    ap.add_argument("--run", required=True, help="pipeline_run_id of the SA-2 run to verify")
    ap.add_argument("--doc", default=SA2_DOC_ID,
                    help="document id (used for the doc text-chunk count and as a "
                         "fallback); the doc is preferentially DERIVED from --run. "
                         f"Default = SA-2 ({SA2_DOC_ID}).")
    ap.add_argument("--merged", type=int, default=22,
                    help="expected merged-entity count for the GLOBAL recall sanity "
                         "check (from diagnostic; SA-2≈22)")
    args = ap.parse_args()
    results = []

    run = args.run
    doc, doc_src = resolve_doc(run, args.doc)
    since, until = run_window(run)
    window_desc = f"since={since} until={until or '(open)'}"

    # ------------------------------------------------------------------
    # 1. (HARD discriminator) run-scoped EXTRACTED_FROM count > 0.
    #    EXTRACTED_FROM carries pipeline_run_id (set in
    #    batch_create_entity_chunk_edges_sync). It is append-only and never
    #    purged, so a GLOBAL count is polluted by prior coarse runs — must scope
    #    by run. This — not edge-existence or the trace — is the lineage proof.
    # ------------------------------------------------------------------
    ef_run = count(
        f"SELECT count(*) AS count FROM EXTRACTED_FROM WHERE pipeline_run_id = '{run}'")
    results.append(("run-scoped EXTRACTED_FROM count > 0 (HARD lineage discriminator)",
                    ef_run > 0,
                    f"run_scoped_EXTRACTED_FROM={ef_run} for run {run}"))

    # ------------------------------------------------------------------
    # 2. (GLOBAL sanity, NOT the lineage proof) total entity vertices >= merged.
    #    Entity vertices carry NO run_id so this cannot be run-scoped. Replaces
    #    the old `post - pre > 0` delta, which false-FAILED on idempotent re-runs
    #    (upsert => delta 0).
    # ------------------------------------------------------------------
    total = total_entities()
    results.append(("entities present (GLOBAL recall sanity — NOT run-scoped, NOT the lineage proof)",
                    total >= args.merged,
                    f"total_entities={total} (>= --merged={args.merged}); "
                    f"entity vertices carry no run_id, so this is global by necessity"))

    # ------------------------------------------------------------------
    # 3. fan-out-width PRECISION — no entity links to >= 80% of the doc's text
    #    chunks (the coarse all-chunks fan-out signature). An absolute per-entity
    #    cap would false-FAIL legitimately high-frequency entities (SNR-75 spans
    #    ~26 of 102 chunks). Dedup per entity in Python (ArcadeDB has no
    #    count(DISTINCT) — confirmed parse error). N/A when there are 0 run-scoped
    #    edges (then #1 already FAILs).
    # ------------------------------------------------------------------
    doc_chunks = count(
        f"SELECT count(*) AS count FROM TextChunk WHERE document_id = '{doc}'")
    edge_rows = adb(
        f"SELECT @out AS e, @in AS c FROM EXTRACTED_FROM WHERE pipeline_run_id = '{run}'")
    if not edge_rows:
        results.append(("fan-out-width precision (no entity >= 80% of doc text chunks)",
                        False,
                        f"N/A — 0 run-scoped EXTRACTED_FROM edges (see #1); "
                        f"doc={doc} ({doc_src}) text_chunks={doc_chunks}"))
    else:
        per_entity = {}
        for row in edge_rows:
            e, c = row.get("e"), row.get("c")
            if e is None or c is None:
                continue
            per_entity.setdefault(e, set()).add(c)
        dist_counts = sorted((len(s) for s in per_entity.values()), reverse=True)
        threshold = int(doc_chunks * FANOUT_FAIL_FRACTION) if doc_chunks else 0
        worst = dist_counts[0] if dist_counts else 0
        worst_pct = (worst / doc_chunks * 100.0) if doc_chunks else 0.0
        # PASS = no entity at/above the fan-out threshold. If doc_chunks is 0 we
        # cannot compute the fraction reliably -> treat as FAIL (can't prove
        # precision), surfaced in the detail.
        ok = bool(doc_chunks) and worst < threshold
        top = dist_counts[:5]
        results.append(("fan-out-width precision (no entity >= 80% of doc text chunks)",
                        ok,
                        f"doc={doc} ({doc_src}) text_chunks={doc_chunks} "
                        f"entities_with_edges={len(per_entity)} "
                        f"top_distinct_chunk_counts={top} "
                        f"worst={worst} ({worst_pct:.0f}% of doc) "
                        f"fail_threshold={threshold} (>= {int(FANOUT_FAIL_FRACTION*100)}%)"))

    # ------------------------------------------------------------------
    # 4a. EXTRACTED_FROM-only, EDGE-ANCHORED, run-scoped trace -> chunk + doc + page.
    #     A vertex out() projection cannot filter by an edge property, so the trace
    #     is anchored on the edge itself. Drops MENTIONED_IN entirely. PASS = a row
    #     with chunk_id AND doc AND page (page is read as @in.page_number, i.e. the
    #     target chunk's own page — see #5).
    # ------------------------------------------------------------------
    trace_rows = adb(
        f"SELECT @out.system_name AS name, @in.chunk_id AS chunk_id, "
        f"@in.page_number AS page, @in.document_id AS doc "
        f"FROM EXTRACTED_FROM WHERE pipeline_run_id = '{run}' LIMIT 5")
    trace_ok = False
    trace_detail = "no run-scoped EXTRACTED_FROM rows to trace"
    traced_page = None
    for r in trace_rows:
        chunk_id, page, tdoc, name = r.get("chunk_id"), r.get("page"), r.get("doc"), r.get("name")
        if bool(chunk_id) and bool(tdoc) and page is not None:
            trace_ok = True
            traced_page = page
            trace_detail = (f"'{name}' -EXTRACTED_FROM-> chunk_id={chunk_id} "
                            f"page={page} document_id={tdoc}")
            break
    else:
        if trace_rows:
            r = trace_rows[0]
            trace_detail = (f"'{r.get('name')}' -> chunk_id={r.get('chunk_id')} "
                            f"page={r.get('page')} document_id={r.get('doc')} "
                            f"(missing chunk/doc/page)")
    results.append(("trace: entity -EXTRACTED_FROM-> chunk + document + page (run-scoped, edge-anchored)",
                    trace_ok, trace_detail))

    # ------------------------------------------------------------------
    # 4b. run-windowed warning checks: zero provenance-drop warnings + zero
    #     spurious LINEAGE_GATE rejections, scoped to the RUN's Postgres time
    #     window (not a fixed --since 3h, which would catch unrelated runs).
    # ------------------------------------------------------------------
    dl = _docker_logs("eip-mmdpp-docling-graph-1", since, until)
    wl = _docker_logs("eip-mmdpp-worker-graph-1", since, until)
    drops = dl.count("dropping provenance row missing required fields")
    gate_rejects = wl.count("LINEAGE_GATE: rejected")
    results.append(("zero provenance-drop warnings (run window)",
                    drops == 0, f"drops={drops} [{window_desc}]"))
    results.append(("zero LINEAGE_GATE rejections (run window)",
                    gate_rejects == 0,
                    f"gate_rejections={gate_rejects} (nonzero = entities lacked "
                    f"lineage) [{window_desc}]"))

    # ------------------------------------------------------------------
    # 5. page is chunk-sourced. #4a reads page as @in.page_number — the page of
    #    the edge's TARGET TextChunk — so by construction the provenance page IS
    #    the chunk's page, not a batch-level value. Assert it is present and
    #    document that it came from the resolved chunk.
    # ------------------------------------------------------------------
    page_ok = trace_ok and traced_page is not None
    results.append(("page is chunk-sourced (page == edge target TextChunk.page_number, by construction)",
                    page_ok,
                    f"traced page={traced_page} read from EXTRACTED_FROM target "
                    f"chunk (@in.page_number); not batch-sourced"
                    + ("" if trace_ok else " — no successful trace (see trace check)")))

    # ---- report ----
    print(f"\n=== Lineage E2E verification — run {run} ===")
    print(f"  doc: {doc} ({doc_src})    window: {window_desc}")
    all_pass = True
    for name, ok, detail in results:
        all_pass = all_pass and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{'ALL CHECKS PASS' if all_pass else 'FAILURES PRESENT'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
