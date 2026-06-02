#!/usr/bin/env python3
"""End-to-end lineage verification gate (Task 6 of the precise-lineage fix).

Proves, on a real SA-2 graph_only run on the FIXED build, that THIS run restored
PRECISE, run-attributed lineage across ALL THREE targets — entity, field, and
relationship — and that recall did NOT collapse. The hard discriminator is the
RUN-SCOPED EXTRACTED_FROM edge count (filtered by pipeline_run_id) plus a
pre/post DELTA: EXTRACTED_FROM is append-only and never purged, so a GLOBAL count
is polluted by prior coarse runs and pre-existing MENTIONED_IN edges, and an
absolute count alone can be satisfied by a prior run. The run command passes
``--pre-extracted-from 0`` for SA-2 so the delta is the proof that the lineage
edges were built THIS run.

The checks here close the holes a prior review found:

  * false-PASS: the old trace + any(edge>0) existence checks were satisfied by
    pre-existing MENTIONED_IN with EXTRACTED_FROM=0. Now the lineage proof is the
    run-scoped EXTRACTED_FROM count AND a post>pre delta (#1) and the trace is
    EXTRACTED_FROM-only, edge-anchored, run-scoped (#7).
  * false-PASS: a keystone regression that drops ALL committed rows must not pass
    vacuously (0 entities, 0 edges -> "no fan-out, no None chunk_ids"). Per-target
    RECALL baselines (#8) fail-loud when committed==0 while passes extracted >0.
  * false-FAIL: legitimately high-frequency entities (SNR-75 spans ~26/102 chunks)
    must not trip the fan-out cap (#3 uses a >=50% fraction, not an absolute K).

Three precision targets + three recall baselines + the trace, all run-scoped:

  1. (HARD) run-scoped EXTRACTED_FROM count > 0 AND > --pre-extracted-from (the
     pre/post delta discriminator — proves lineage edges were built THIS run).
  2. (GLOBAL sanity) total entity vertices >= --merged (entity vertices carry no
     run_id, so this can't be run-scoped; it is NOT the lineage proof — #1 is).
  3. ENTITY fan-out PRECISION: NO entity links to >= 50% of the doc's text chunks
     (the coarse all-chunks fan-out signature). K-distribution reported.
  4. FIELD PRECISION: 0 ``_field_evidence`` rows with null chunk_id among the
     committed entities (a None means the merge-phase resolution regressed).
  5. RELATIONSHIP PRECISION: committed ontology relationship edges carry a
     non-empty ``source_chunk_ids`` LIST. N/A-not-fail when 0 edges for the run.
  6. ENTITY / FIELD / RELATIONSHIP RECALL baselines: ENTITY uses the DISTINCT
     MERGED-node count (audit blob graph_json->'nodes' length) as the denominator
     — NOT the per-pass instance sum, which double-counts an entity emitted by
     several passes and reports a false low recall. committed>=merged is full
     recall; committed<merged (a merged entity that didn't commit) FAILs;
     committed==0 while merged>0 is a LOUD fail; valid rejection is not loss.
  7. EXTRACTED_FROM-only, edge-anchored, run-scoped trace -> chunk + doc + page;
     plus run-windowed provenance-drop / LINEAGE_GATE-rejection warning checks.
  8. the traced page is chunk-sourced (it is the edge target chunk's page_number).

Read-only — reads ArcadeDB + Postgres + container logs and reports; never mutates.

Usage:
    python3 scripts/verify_lineage_e2e.py --run <sa2_run_id> [--doc <doc_id>] \
        [--merged N] [--pre-extracted-from 0]
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
# The old bug had entities at ~100% (all-document); a legitimately high-frequency
# entity (SNR-75) tops out around 26/102 (~25%), so 50% cleanly separates the two.
FANOUT_FAIL_FRACTION = 0.50

# Structural (infrastructure) edge types — NOT domain ontology relationship
# edges. The relationship-precision check enumerates ArcadeDB edge classes and
# excludes these so it only inspects DOMAIN relationship edges (the LLM /
# system_links-extracted semantic rels — ASSOCIATED_WITH, CUES, DETECTS, …),
# which alone carry source_chunk_ids. Structural edges are built by the chunk /
# provenance / derive-rules path and the docling document-anchor walker via
# create_structural_edge_sync (NOT the ontology relationship path), so they
# legitimately do NOT carry source_chunk_ids and must be excluded here.
#
# AUTHORITATIVE SOURCE for the classification (no guessed/hardcoded list):
#   1. app/services/arcadedb_schema.py::_STRUCTURAL_EDGE_TYPES — the chunk /
#      provenance / derive-rules structural edge classes (CONTAINS_TEXT,
#      SAME_PAGE, HAS_PROVENANCE, EXTRACTED_FROM, MENTIONED_IN-family, plus the
#      HAS_SECTION/HAS_FIGURE/HAS_TABLE/CHILD_OF doc-anchor subset). Imported
#      directly below so this gate can never drift from the schema. A mirrored
#      literal is the fallback for when the app package isn't importable (the
#      gate is meant to run standalone via curl).
#   2. The remaining document-anchor edge types emitted by the
#      `document_anchors` walker (app/services/docling_anchors.py — every edge
#      carries pass_origins={"document_anchors"}, written via
#      create_structural_edge_sync; see also derive_document_anchors in
#      app/workers/pipeline.py). _STRUCTURAL_EDGE_TYPES omits HAS_IMAGE and
#      NEAR_TEXT (and the chunk-containment CONTAINS edge), even though they are
#      structural anchor edges of the SAME family as the HAS_SECTION/HAS_FIGURE
#      entries it DOES list — that omission is exactly what made the old gate
#      over-count (HAS_IMAGE=34 + NEAR_TEXT=136 = the spurious 170). They are
#      added here so the structural set is complete.
#   MENTIONED_IN is in the ontology relationship_types registry but is a
#   derive-rules structural edge (DOCUMENT target, built by
#   derive_structural_edges), so it is also excluded.
try:  # authoritative — keep this gate in lock-step with the schema constant
    from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES as _SCHEMA_STRUCTURAL
except Exception:  # standalone fallback — mirror of arcadedb_schema._STRUCTURAL_EDGE_TYPES
    _SCHEMA_STRUCTURAL = [
        "CONTAINS_TEXT", "CONTAINS_IMAGE", "SAME_PAGE", "SAME_SECTION",
        "SAME_ARTIFACT", "NEXT_CHUNK", "HAS_PROVENANCE", "EXTRACTED_FROM",
        "HAS_ALIAS", "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF",
    ]
# Document-anchor + derive structural edges NOT present in the schema constant
# (docling_anchors.py document_anchors walker + derive_rules structural edges).
_ANCHOR_STRUCTURAL = ["HAS_IMAGE", "NEAR_TEXT", "CONTAINS", "MENTIONED_IN"]
STRUCTURAL_EDGE_TYPES = frozenset(list(_SCHEMA_STRUCTURAL) + _ANCHOR_STRUCTURAL)


# ===========================================================================
# PURE LOGIC (no DB / no subprocess) — directly unit-tested in
# tests/unit/test_verify_lineage_gate.py. Every check's PASS/FAIL decision and
# its human-readable detail are computed here from already-fetched inputs so the
# verdict logic can be proven without a live stack.
# ===========================================================================

def compute_extracted_from_delta(post, pre):
    """HARD lineage discriminator (#1): the run built NEW EXTRACTED_FROM edges.

    PASS iff post > 0 AND post > pre. ``pre`` defaults to 0 for SA-2 (the run
    command passes --pre-extracted-from 0), making this equivalent to "this run
    created run-scoped EXTRACTED_FROM edges". For a non-zero pre (e.g. an
    idempotent re-run baseline) the delta still demands strictly-more edges.
    Returns (ok, detail)."""
    ok = post > 0 and post > pre
    return ok, (
        f"run_scoped_EXTRACTED_FROM post={post} pre={pre} delta={post - pre} "
        f"(PASS requires post>0 AND post>pre)"
    )


def compute_fanout_precision(per_entity_chunk_counts, doc_chunks,
                             fail_fraction=FANOUT_FAIL_FRACTION):
    """ENTITY fan-out PRECISION (#3): no entity links to >= fail_fraction of the
    document's text chunks (the coarse all-chunks fan-out signature).

    ``per_entity_chunk_counts`` is an iterable of per-entity DISTINCT target-chunk
    counts (deduped upstream). PASS iff doc_chunks>0 AND the worst (max) count is
    strictly below fail_fraction*doc_chunks. doc_chunks==0 -> cannot compute the
    fraction -> FAIL (can't prove precision). Empty edge set -> FAIL (no evidence;
    #1 already fails). Returns (ok, detail). Reports the K-distribution
    (min/median/max + top-5) so a human sees precise-vs-coarse at a glance."""
    counts = sorted((int(c) for c in per_entity_chunk_counts), reverse=True)
    n = len(counts)
    if not counts:
        return False, (
            f"N/A — 0 entities with run-scoped EXTRACTED_FROM edges (see #1); "
            f"doc_text_chunks={doc_chunks}"
        )
    worst = counts[0]
    smallest = counts[-1]
    median = counts[n // 2] if n % 2 else (counts[n // 2 - 1] + counts[n // 2]) / 2
    threshold = int(doc_chunks * fail_fraction) if doc_chunks else 0
    worst_pct = (worst / doc_chunks * 100.0) if doc_chunks else 0.0
    ok = bool(doc_chunks) and worst < threshold
    top = counts[:5]
    return ok, (
        f"doc_text_chunks={doc_chunks} entities_with_edges={n} "
        f"distinct_chunks_per_entity[min={smallest} median={median} max={worst}] "
        f"top5={top} worst={worst} ({worst_pct:.0f}% of doc) "
        f"fail_threshold={threshold} (>= {int(fail_fraction * 100)}% of doc)"
    )


def count_field_evidence_nulls(field_evidence_blobs):
    """FIELD PRECISION (#4): count ``_field_evidence`` rows with a null/None
    chunk_id vs total rows, across the committed entities.

    ``field_evidence_blobs`` is an iterable of the raw ``_field_evidence`` JSON
    values read off entity vertices: each is ``{field_name: [ {chunk_id, ...}, ...]}``.
    A blob may be None / a stringified JSON / already a dict — all tolerated.
    Returns (total_rows, null_rows, entities_with_evidence)."""
    total = 0
    nulls = 0
    entities_with_ev = 0
    for blob in field_evidence_blobs:
        if blob is None:
            continue
        if isinstance(blob, str):
            try:
                blob = json.loads(blob)
            except Exception:
                # Unparseable evidence blob = indeterminate -> count its presence
                # as a null row so the check fails loud rather than silently skips.
                total += 1
                nulls += 1
                entities_with_ev += 1
                continue
        if not isinstance(blob, dict):
            continue
        had_row = False
        for rows in blob.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                total += 1
                had_row = True
                cid = row.get("chunk_id")
                if cid is None or (isinstance(cid, str) and not cid.strip()):
                    nulls += 1
        if had_row:
            entities_with_ev += 1
    return total, nulls, entities_with_ev


def evaluate_field_precision(total_rows, null_rows, entities_with_evidence):
    """Decide the FIELD precision check (#4). PASS iff there ARE field-evidence
    rows AND none has a null chunk_id. ZERO rows = no evidence -> FAIL-CLOSED
    (cannot prove the resolution ran). Returns (ok, detail)."""
    if total_rows == 0:
        return False, (
            "0 _field_evidence rows across committed entities — NO EVIDENCE, "
            "cannot confirm chunk_id resolution ran (fail-closed)"
        )
    ok = null_rows == 0
    return ok, (
        f"field_evidence_rows={total_rows} null_chunk_id={null_rows} "
        f"entities_with_evidence={entities_with_evidence} "
        f"(PASS requires 0 null chunk_ids)"
    )


def compute_relationship_precision(rel_edge_rows):
    """RELATIONSHIP PRECISION (#5): committed ontology relationship edges carry a
    non-empty ``source_chunk_ids`` LIST.

    ``rel_edge_rows`` is an iterable of dicts, each with a ``source_chunk_ids``
    key (value may be a list, None, or absent). Returns (status, ok, detail)
    where status is one of 'PASS' | 'FAIL' | 'NA'. 0 edges -> N/A-not-fail
    (relationships depend on the system_links pass extracting any; absence is a
    noted gap, NOT a lineage regression). When edges exist, PASS iff EVERY edge
    has a non-empty source_chunk_ids list."""
    rows = list(rel_edge_rows)
    if not rows:
        return "NA", True, (
            "0 ontology relationship edges for the run — N/A (relationships "
            "depend on the system_links pass extracting any; noted, not a fail)"
        )
    have = 0
    lack = 0
    for r in rows:
        scids = r.get("source_chunk_ids") if isinstance(r, dict) else None
        if isinstance(scids, list) and len(scids) > 0:
            have += 1
        else:
            lack += 1
    ok = lack == 0
    return ("PASS" if ok else "FAIL"), ok, (
        f"relationship_edges={len(rows)} with_source_chunk_ids={have} "
        f"without={lack} (PASS requires non-empty source_chunk_ids on every edge)"
    )


def compute_entity_recall(committed, merged, instance_emissions=None):
    """ENTITY RECALL baseline (#6a): committed entity vertices vs the number of
    DISTINCT MERGED entities the run intended to commit.

    The correct denominator is ``merged`` — the audit blob's nodes[] length
    (DocumentGraphExtraction.graph_json->'nodes'), i.e. the count of distinct
    merged identities AFTER cross-pass dedup. It is NOT the sum of
    pipeline_pass_outputs.primary_entities_extracted: that sum counts per-pass
    entity INSTANCES, where the SAME real entity is emitted by several passes
    (e.g. SNR-75 appears in radar_identity AND radar_power_rf), so it is an
    inflated denominator that reports a false low recall (27 committed vs 133
    instances => spurious "20%"). ``instance_emissions`` (the old per-pass sum)
    is accepted only as a REPORTED diagnostic and never drives the verdict.

    Semantics (fail-closed — a merged entity that did not commit IS lineage loss):
      * committed==0 while merged>0  -> LOUD FAIL (keystone recall collapse).
      * committed >= merged          -> PASS (full recall: every merged entity
        got committed; committed may exceed merged when prior runs' vertices are
        present, since entity vertices carry no run_id).
      * 0 < committed < merged       -> FAIL (a merged entity did not commit =
        real recall loss; the gate is fail-closed).
      * merged unknown (None)        -> require committed>0 so a zeroed commit
        still fails (cannot prove full recall without the denominator).
      * merged==0 and committed==0   -> vacuous-but-honest PASS (nothing to lose).
    Returns (ok, detail)."""
    inst = ("" if instance_emissions is None
            else f" per_pass_instance_emissions={instance_emissions} (NOT distinct)")
    if merged is None:
        ok = committed > 0
        return ok, (
            f"committed_entities={committed} merged_baseline=UNKNOWN "
            f"(audit blob nodes[] unavailable); PASS requires committed>0{inst}"
        )
    if merged > 0 and committed == 0:
        return False, (
            f"committed_entities=0 but {merged} merged entities expected — "
            f"KEYSTONE RECALL COLLAPSE (committed dropped to zero){inst}"
        )
    ratio = (committed / merged * 100.0) if merged else 0.0
    # committed >= merged is full recall; committed < merged loses a merged
    # entity (lineage loss) -> FAIL. merged==0,committed==0 -> vacuous pass.
    ok = committed >= merged
    return ok, (
        f"committed_entities={committed} merged(distinct nodes[])={merged} "
        f"recall={ratio:.0f}% (PASS requires committed>=merged; LOUD fail iff "
        f"committed==0 while merged>0){inst}"
    )


def compute_relationship_recall(committed_edges, accepted, rejected):
    """RELATIONSHIP RECALL baseline (#6c): committed edges vs post-validation
    ACCEPTED rels. Rejected rels are reported SEPARATELY and NOT counted as loss
    (valid rejection is not recall loss). LOUD fail iff accepted>0 but committed
    edges==0. accepted unknown (None) -> report only, don't fail on the baseline.
    Returns (ok, detail)."""
    rej = rejected or 0
    if accepted is None:
        return True, (
            f"committed_edges={committed_edges} accepted_baseline=UNKNOWN "
            f"rejected={rej} (reported; rejection is not loss)"
        )
    if accepted > 0 and committed_edges == 0:
        return False, (
            f"committed_edges=0 but {accepted} rels accepted post-validation — "
            f"edge commit dropped to zero (rejected={rej}, not counted as loss)"
        )
    return True, (
        f"committed_edges={committed_edges} accepted={accepted} rejected={rej} "
        f"(rejection reported separately, NOT counted as recall loss)"
    )


def trace_first_complete(trace_rows):
    """Pick the first EXTRACTED_FROM trace row that carries chunk_id AND doc AND
    a non-None page (#7a). Returns (ok, detail, traced_page). Edge-anchored:
    page is the EDGE TARGET chunk's @in.page_number, so a present page is
    chunk-sourced by construction (#8)."""
    rows = list(trace_rows)
    if not rows:
        return False, "no run-scoped EXTRACTED_FROM rows to trace", None
    for r in rows:
        chunk_id, page, tdoc, name = r.get("chunk_id"), r.get("page"), r.get("doc"), r.get("name")
        if bool(chunk_id) and bool(tdoc) and page is not None:
            return True, (
                f"'{name}' -EXTRACTED_FROM-> chunk_id={chunk_id} "
                f"page={page} document_id={tdoc}"
            ), page
    r = rows[0]
    return False, (
        f"'{r.get('name')}' -> chunk_id={r.get('chunk_id')} "
        f"page={r.get('page')} document_id={r.get('doc')} (missing chunk/doc/page)"
    ), None


# ===========================================================================
# I/O HELPERS (subprocess to curl/docker/psql) — not unit-tested (need a stack).
# ===========================================================================

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


def _pg_int_sum(run_id, column):
    """Return SUM(column) over ingest.pipeline_pass_outputs for the run, or None
    when the query returns nothing (no rows / pg failure) so the caller can mark
    the recall baseline UNKNOWN rather than treat a missing baseline as 0."""
    rows = pg(
        f"SELECT COALESCE(SUM({column}), 0) FROM ingest.pipeline_pass_outputs "
        f"WHERE pipeline_run_id = '{run_id}'"
    )
    if not rows:
        return None
    val = rows[0].strip()
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def merged_node_count(doc):
    """Return the number of DISTINCT MERGED entities for the run's document —
    the audit blob's nodes[] length: jsonb_array_length(graph_json->'nodes') off
    ingest.document_graph_extractions. This is the count of distinct merged
    identities the run intended to commit (post cross-pass dedup), and is the
    CORRECT denominator for entity recall (NOT the per-pass instance sum).

    Returns None when the query yields nothing (no blob row / pg failure) so the
    caller marks the recall baseline UNKNOWN rather than treating a missing
    denominator as 0. (document_id is the table PK — exactly one row per doc.)"""
    rows = pg(
        "SELECT jsonb_array_length(graph_json->'nodes') "
        f"FROM ingest.document_graph_extractions WHERE document_id = '{doc}'"
    )
    if not rows:
        return None
    val = rows[0].strip()
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _docker_logs(container, since, until):
    """Return (text, line_count) for the container's logs in the window.

    line_count lets a caller distinguish "0 matches over N lines" (real evidence)
    from "0 matches over 0 lines" (no evidence — logs rotated/aged out, window
    doesn't overlap retention, or container recreated). A blank trailing line from
    docker is not counted as evidence."""
    cmd = ["docker", "logs", container]
    if since:
        cmd += ["--since", since]
    if until:
        cmd += ["--until", until]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    text = r.stdout + r.stderr
    n_lines = sum(1 for ln in text.splitlines() if ln.strip())
    return text, n_lines


def total_entities():
    return sum(count(f"SELECT count(*) AS count FROM `{t}`") for t in ENTITY_TYPES)


def committed_entity_field_evidence():
    """Read the ``_field_evidence`` JSON property off every committed entity
    vertex. Returns a list of raw _field_evidence values (dict / str / None).

    SCOPE — global entities are NOT doc-scoped by a column. The ontology entity
    types in ENTITY_TYPES (RADAR_SYSTEM / MISSILE_SYSTEM / ...) all declare
    identity_scope='global' (ontology_bundles/air_defense_v3 entity definitions;
    see arcadedb_graph.py global_entity_types filter), so their vertices carry
    NO ``document_id`` property — only document-scoped anchor types (SECTION /
    FIGURE / IMAGE / TEXT_BLOCK) do. The old ``WHERE document_id = '{doc}'``
    filter therefore matched ZERO rows on every global entity, making the
    FIELD-precision check read 0 _field_evidence rows and fail-closed even when
    ~18 entities carry resolved field evidence. This now reads the SAME global
    entity set as total_entities() (no document_id filter) — matching the
    end-of-run spot-check that finds the non-null _field_evidence rows."""
    blobs = []
    for t in ENTITY_TYPES:
        rows = adb(f"SELECT `_field_evidence` AS fe FROM `{t}`")
        for r in rows:
            blobs.append(r.get("fe"))
    return blobs


def _ontology_edge_types():
    """Enumerate ArcadeDB edge classes, excluding the structural infrastructure
    edges. ArcadeDB keeps each edge type as its own top-level class (no shared
    `E` superclass — `SELECT FROM E` raises), so we ask schema:types. Returns the
    domain relationship edge type names (those that carry source_chunk_ids)."""
    rows = adb("SELECT name FROM schema:types WHERE type = 'edge'")
    names = [r.get("name") for r in rows if r.get("name")]
    return [n for n in names if n not in STRUCTURAL_EDGE_TYPES]


def relationship_edge_rows(run):
    """RUN-SCOPED DOMAIN relationship edge rows with their source_chunk_ids.

    RUN-SCOPING IS THE CORRECT DISCRIMINATOR HERE. Domain relationship edges are
    GLOBAL and accumulate across runs — they are append/upsert and never purged,
    so 30+ prior graph_only runs on the same document leave their edges in place.
    Scoping by ``document_ids CONTAINS '<doc>'`` (the old behaviour) therefore
    counts ALL of those cross-run edges (e.g. 32 when only ~3-20 belong to the
    current run) and scores STALE pre-fix edges, which carry pipeline_run_id=NULL
    and an empty source_chunk_ids — producing a false FAIL for the run under test.
    Only edges scoped to THIS run reflect the current build's lineage.

    After the upstream Fix H, every edge a run upserts carries
    ``pipeline_run_id = run``, so ``WHERE pipeline_run_id = '<run>'`` captures
    EXACTLY the edges this run committed. This mirrors the run-scoping the
    HARD EXTRACTED_FROM check uses for the very same reason (EXTRACTED_FROM is
    likewise append-only/global, so its count is run-scoped by pipeline_run_id).

    Enumerate ONLY the non-structural (domain) edge types — the structural /
    derive-rules / doc-anchor edge classes are excluded via STRUCTURAL_EDGE_TYPES
    (see _ontology_edge_types) because they legitimately carry no
    source_chunk_ids — and pull this run's rows from each. Returns a list of
    dicts {type, source_chunk_ids}."""
    out = []
    for etype in _ontology_edge_types():
        rows = adb(
            f"SELECT '{etype}' AS etype, source_chunk_ids AS source_chunk_ids "
            f"FROM `{etype}` WHERE pipeline_run_id = '{run}'"
        )
        for r in rows:
            out.append({"type": etype, "source_chunk_ids": r.get("source_chunk_ids")})
    return out


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end PRECISE-lineage verification gate. Proves "
                    "entity + field + relationship lineage precision AND per-target "
                    "recall, run-scoped by pipeline_run_id (the hard discriminator).")
    ap.add_argument("--run", required=True, help="pipeline_run_id of the SA-2 run to verify")
    ap.add_argument("--doc", default=SA2_DOC_ID,
                    help="document id (used for the doc text-chunk count, the "
                         "field-evidence read, and as a fallback); the doc is "
                         "preferentially DERIVED from --run. "
                         f"Default = SA-2 ({SA2_DOC_ID}).")
    ap.add_argument("--merged", type=int, default=22,
                    help="expected merged-entity count for the GLOBAL recall sanity "
                         "check (from diagnostic; SA-2≈22)")
    ap.add_argument("--pre-extracted-from", type=int, default=0, dest="pre_extracted_from",
                    help="run-attributed EXTRACTED_FROM count BEFORE this run, for "
                         "the pre/post delta discriminator (#1). PASS requires "
                         "post>pre. The run command passes 0 for SA-2 (a clean "
                         "run-scoped count starts at 0).")
    args = ap.parse_args()
    results = []

    run = args.run
    doc, doc_src = resolve_doc(run, args.doc)
    since, until = run_window(run)
    # A (None, None) window means the run was not found in ingest.pipeline_runs
    # (or pg failed). Without a window, _docker_logs would scan the ENTIRE
    # container history and surface unrelated runs' warnings — so we treat the
    # window as DEGRADED and fail-closed on the warning checks (7b).
    window_degraded = since is None and until is None
    window_desc = f"since={since} until={until or '(open)'}"
    if window_degraded:
        window_desc += "  DEGRADED (run not found in pipeline_runs) — warning checks will fail-closed"

    # ------------------------------------------------------------------
    # 1. (HARD discriminator) run-scoped EXTRACTED_FROM count > 0 AND > pre.
    #    EXTRACTED_FROM carries pipeline_run_id (set in
    #    batch_create_entity_chunk_edges_sync). It is append-only and never
    #    purged, so a GLOBAL count is polluted by prior coarse runs — must scope
    #    by run. The pre/post DELTA (post > --pre-extracted-from) proves the
    #    edges were built THIS run. This — not edge-existence or the trace — is
    #    the lineage proof.
    # ------------------------------------------------------------------
    ef_run = count(
        f"SELECT count(*) AS count FROM EXTRACTED_FROM WHERE pipeline_run_id = '{run}'")
    ef_ok, ef_detail = compute_extracted_from_delta(ef_run, args.pre_extracted_from)
    results.append(("run-scoped EXTRACTED_FROM post>pre delta (HARD lineage discriminator)",
                    ef_ok, f"{ef_detail} for run {run}"))

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
    # 3. ENTITY fan-out PRECISION — no entity links to >= 50% of the doc's text
    #    chunks (the coarse all-chunks fan-out signature). An absolute per-entity
    #    cap would false-FAIL legitimately high-frequency entities (SNR-75 spans
    #    ~26 of 102 chunks). Dedup per entity in Python (ArcadeDB has no
    #    count(DISTINCT) — confirmed parse error).
    # ------------------------------------------------------------------
    doc_chunks = count(
        f"SELECT count(*) AS count FROM TextChunk WHERE document_id = '{doc}'")
    edge_rows = adb(
        f"SELECT @out AS e, @in AS c FROM EXTRACTED_FROM WHERE pipeline_run_id = '{run}'")
    per_entity = {}
    for row in edge_rows:
        e, c = row.get("e"), row.get("c")
        if e is None or c is None:
            continue
        per_entity.setdefault(e, set()).add(c)
    fan_ok, fan_detail = compute_fanout_precision(
        (len(s) for s in per_entity.values()), doc_chunks)
    results.append((f"ENTITY fan-out precision (no entity >= {int(FANOUT_FAIL_FRACTION*100)}% of doc text chunks)",
                    fan_ok, f"doc={doc} ({doc_src}) {fan_detail}"))

    # ------------------------------------------------------------------
    # 4. FIELD PRECISION — 0 _field_evidence rows with null chunk_id among the
    #    committed entities. The Task-5 merge resolves a RESOLVED chunk_id onto
    #    every field-evidence row (own self_ref -> chunk, else entity chunk-set,
    #    else None). A None means the resolution regressed. Fail-closed when there
    #    are 0 rows (no evidence the resolver ran).
    # ------------------------------------------------------------------
    fe_blobs = committed_entity_field_evidence()
    fe_total, fe_nulls, fe_entities = count_field_evidence_nulls(fe_blobs)
    field_ok, field_detail = evaluate_field_precision(fe_total, fe_nulls, fe_entities)
    results.append(("FIELD precision (0 _field_evidence rows with null chunk_id)",
                    field_ok, f"doc={doc} {field_detail}"))

    # ------------------------------------------------------------------
    # 5. RELATIONSHIP PRECISION — committed DOMAIN relationship edges carry a
    #    non-empty source_chunk_ids LIST (resolved in the merge phase from the
    #    rel's positional self_refs). Counts ONLY domain (LLM / system_links)
    #    relationship edge types — structural / derive-rules / doc-anchor edges
    #    (HAS_COMPONENT, HAS_SECTION, HAS_FIGURE, HAS_IMAGE, NEAR_TEXT,
    #    EXTRACTED_FROM, MENTIONED_IN, …) are excluded via STRUCTURAL_EDGE_TYPES
    #    because they are built by a different path and legitimately carry no
    #    source_chunk_ids (the old gate counted them, yielding the spurious 170).
    #    RUN-SCOPED by pipeline_run_id (NOT doc-scoped): domain relationship edges
    #    are global and accumulate across runs, so a doc scope counts stale
    #    cross-run edges (32 vs the ~3-20 this run built) and scores pre-fix
    #    NULL-run edges — a false FAIL. Mirrors the EXTRACTED_FROM run-scoping (#1).
    #    N/A-not-fail when there are 0 domain edges for the run: relationships
    #    depend on the system_links pass extracting any, so an absent edge set is
    #    a noted gap, NOT a lineage regression.
    # ------------------------------------------------------------------
    rel_rows = relationship_edge_rows(run)
    rel_status, rel_ok, rel_detail = compute_relationship_precision(rel_rows)
    rel_label = "RELATIONSHIP precision (committed edges carry non-empty source_chunk_ids)"
    if rel_status == "NA":
        rel_label += " [N/A]"
    results.append((rel_label, rel_ok, rel_detail))

    # ------------------------------------------------------------------
    # 6. PER-TARGET RECALL baselines — so valid rejection isn't misread as loss
    #    and a keystone collapse can't pass vacuously.
    #    6a ENTITY: committed entity vertices vs the DISTINCT MERGED entity count
    #       (audit blob graph_json->'nodes' length). NOT the per-pass instance sum
    #       (SUM(primary_entities_extracted)): that double-counts an entity emitted
    #       by multiple passes (SNR-75 in radar_identity AND radar_power_rf) and
    #       reports a false low recall. The old sum is kept as a labeled diagnostic
    #       only. LOUD fail iff committed==0 while merged>0; FAIL iff committed<merged.
    #    6b FIELD: resolved field-evidence rows vs total (the key signal is 0
    #       null chunk_ids from #4); reported here for completeness.
    #    6c RELATIONSHIP: committed edges vs post-validation ACCEPTED; rejected
    #       reported separately (NOT counted as loss).
    # ------------------------------------------------------------------
    merged_entities = merged_node_count(doc)
    # OLD denominator — now a REPORTED diagnostic only (per-pass instance emissions,
    # NOT distinct merged entities); does NOT drive the pass/fail verdict.
    instance_emissions = _pg_int_sum(run, "primary_entities_extracted")
    er_ok, er_detail = compute_entity_recall(total, merged_entities, instance_emissions)
    results.append(("ENTITY recall (committed vs DISTINCT merged nodes[]; LOUD fail on collapse, FAIL on shortfall)",
                    er_ok, er_detail))

    # 6b field recall — report only (the precision check #4 carries the verdict).
    fe_resolved = fe_total - fe_nulls
    results.append(("FIELD recall (resolved field-evidence rows vs total; report)",
                    True,
                    f"resolved={fe_resolved}/{fe_total} field-evidence rows "
                    f"(verdict carried by FIELD precision #4; this is a report row)"))

    accepted_rels = _pg_int_sum(run, "relationships_extracted")
    rejected_rels = _pg_int_sum(run, "relationships_rejected")
    rr_ok, rr_detail = compute_relationship_recall(len(rel_rows), accepted_rels, rejected_rels)
    results.append(("RELATIONSHIP recall (committed edges vs accepted; rejection NOT counted as loss)",
                    rr_ok, rr_detail))

    # ------------------------------------------------------------------
    # 7a. EXTRACTED_FROM-only, EDGE-ANCHORED, run-scoped trace -> chunk + doc + page.
    #     A vertex out() projection cannot filter by an edge property, so the trace
    #     is anchored on the edge itself. Drops MENTIONED_IN entirely (MENTIONED_IN
    #     pre-satisfies the trace and would false-pass). PASS = a row with chunk_id
    #     AND doc AND page (page read as @in.page_number — the target chunk's own
    #     page, see #8).
    # ------------------------------------------------------------------
    trace_rows = adb(
        f"SELECT @out.system_name AS name, @in.chunk_id AS chunk_id, "
        f"@in.page_number AS page, @in.document_id AS doc "
        f"FROM EXTRACTED_FROM WHERE pipeline_run_id = '{run}' LIMIT 5")
    trace_ok, trace_detail, traced_page = trace_first_complete(trace_rows)
    results.append(("trace: entity -EXTRACTED_FROM-> chunk + document + page (run-scoped, edge-anchored)",
                    trace_ok, trace_detail))

    # ------------------------------------------------------------------
    # 7b. run-windowed warning checks: zero provenance-drop warnings + zero
    #     spurious LINEAGE_GATE rejections, scoped to the RUN's Postgres time
    #     window (not a fixed --since 3h, which would catch unrelated runs).
    #
    #     FAIL-CLOSED: a PASS here requires actual log evidence. If the window is
    #     DEGRADED (run not found -> (None, None)) OR the in-window log stream is
    #     EMPTY (0 lines scanned), there is no evidence that zero warnings means
    #     "clean run" rather than "logs rotated/aged out". For a user-ordered
    #     forensic gate, absence-of-evidence must NOT be read as evidence-of-
    #     absence, so the check FAILs and the detail says so. The scanned-line
    #     count is always surfaced so a human can tell "0 drops / 4123 lines"
    #     (trustworthy) from "0 drops / 0 lines — NO LOG EVIDENCE" (not).
    # ------------------------------------------------------------------
    dl, dl_lines = _docker_logs("eip-mmdpp-docling-graph-1", since, until)
    wl, wl_lines = _docker_logs("eip-mmdpp-worker-graph-1", since, until)
    drops = dl.count("dropping provenance row missing required fields")
    gate_rejects = wl.count("LINEAGE_GATE: rejected")

    # provenance-drop warnings (docling-graph)
    dl_no_evidence = window_degraded or dl_lines == 0
    if dl_no_evidence:
        reason = ("DEGRADED window (run not found)" if window_degraded
                  else "in-window log stream EMPTY")
        results.append(("zero provenance-drop warnings (run window)",
                        False,
                        f"drops={drops} / scanned {dl_lines} log lines — "
                        f"NO LOG EVIDENCE, cannot confirm zero warnings "
                        f"({reason}) [{window_desc}]"))
    else:
        results.append(("zero provenance-drop warnings (run window)",
                        drops == 0,
                        f"drops={drops} / scanned {dl_lines} log lines in window "
                        f"[{window_desc}]"))

    # LINEAGE_GATE rejections (worker-graph)
    wl_no_evidence = window_degraded or wl_lines == 0
    if wl_no_evidence:
        reason = ("DEGRADED window (run not found)" if window_degraded
                  else "in-window log stream EMPTY")
        results.append(("zero LINEAGE_GATE rejections (run window)",
                        False,
                        f"gate_rejections={gate_rejects} / scanned {wl_lines} log "
                        f"lines — NO LOG EVIDENCE, cannot confirm zero rejections "
                        f"({reason}) [{window_desc}]"))
    else:
        results.append(("zero LINEAGE_GATE rejections (run window)",
                        gate_rejects == 0,
                        f"gate_rejections={gate_rejects} / scanned {wl_lines} log "
                        f"lines in window (nonzero = entities lacked lineage) "
                        f"[{window_desc}]"))

    # ------------------------------------------------------------------
    # 8. page is chunk-sourced. #7a reads page as @in.page_number — the page of
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
