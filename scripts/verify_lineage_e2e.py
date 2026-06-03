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
  6. ENTITY / FIELD / RELATIONSHIP RECALL baselines: both ENTITY and RELATIONSHIP
     use a DISTINCT MERGED count as the denominator (audit blob
     graph_json->'nodes' length for entities; graph_json->'edges_accepted' ==
     len(merged.edges) after the (from,type,to) dedup for relationships) — NOT the
     pre-dedup per-pass emit sums (primary_entities_extracted / relationships_-
     extracted), which straddle the merge dedup boundary and report a false low
     recall (e.g. 32 committed edges vs 77 emitted ≈ 42%) on a perfectly-healed
     doc. RELATIONSHIP recall measures edge PRESENCE (committed-total vs merged-
     distinct); precision (#5) separately enforces lineage on every edge, so
     presence is NOT redundant. committed>=merged is full recall; committed<merged
     (a merged item that didn't commit) FAILs; committed==0 while merged>0 is a
     LOUD fail; valid rejection is not loss.
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
try:  # authoritative — keep this gate in lock-step with the schema constants
    from app.services.arcadedb_schema import (
        _STRUCTURAL_EDGE_TYPES as _SCHEMA_STRUCTURAL,
        _ANCHOR_DERIVE_STRUCTURAL_EDGE_TYPES as _SCHEMA_ANCHOR_STRUCTURAL,
    )
except Exception:  # standalone fallback — mirror of the arcadedb_schema constants
    _SCHEMA_STRUCTURAL = [
        "CONTAINS_TEXT", "CONTAINS_IMAGE", "SAME_PAGE", "SAME_SECTION",
        "SAME_ARTIFACT", "NEXT_CHUNK", "HAS_PROVENANCE", "EXTRACTED_FROM",
        "HAS_ALIAS", "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF",
    ]
    _SCHEMA_ANCHOR_STRUCTURAL = ("HAS_IMAGE", "NEAR_TEXT", "CONTAINS", "MENTIONED_IN")
# Document-anchor + derive structural edges NOT present in the schema's
# _STRUCTURAL_EDGE_TYPES (docling_anchors.py document_anchors walker + derive_rules
# structural edges). Consumes the SHARED schema constant so this gate and the
# per-run domain-edge retract (pipeline._domain_relationship_edge_types) can never
# diverge on what counts as a domain edge (anti-drift).
_ANCHOR_STRUCTURAL = list(_SCHEMA_ANCHOR_STRUCTURAL)
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


def compute_relationship_precision(rel_edge_rows, accepted_rels=None):
    """RELATIONSHIP PRECISION (#5): EVERY domain relationship edge of the DOCUMENT
    under test carries a non-empty ``source_chunk_ids`` LIST.

    DOC-SCOPED, FAIL-CLOSED (reverts the false-green run-scoping). ``rel_edge_rows``
    is the document's FULL domain-relationship population (every edge whose
    ``document_ids`` CONTAINS the doc — see relationship_edge_rows), NOT just the
    handful a single run re-emitted. The prior fix scoped these by
    ``pipeline_run_id = :run`` and so inspected only the 3 edges THIS run touched,
    going BLIND to 29/32 (91%) of the document's domain edges that carry NULL
    ``source_chunk_ids`` — a false PASS while 91% of the document's relationships
    had no lineage. The document's full relationship population is the goal, so the
    check counts every domain edge and FAILs when ANY counted edge lacks lineage.

    ``accepted_rels`` (the run's accepted/emitted relationship count from the audit
    blob / pipeline_pass_outputs.relationships_extracted) closes the vacuous-pass
    hole: 0 committed domain edges is N/A-not-fail ONLY if the run also accepted 0
    relationships (nothing to lose). If the run accepted >0 relationships but 0 are
    committed-with-lineage, that is a real lineage drop and FAILs, NOT N/A.

    Returns (status, ok, detail), status in 'PASS' | 'FAIL' | 'NA'. When edges
    exist, PASS iff EVERY edge has a non-empty source_chunk_ids list."""
    rows = list(rel_edge_rows)
    if not rows:
        # 0 committed domain edges for the document. Decide NA-vs-FAIL by the
        # run's accepted count (the vacuous-pass hole the review flagged):
        #   accepted == 0 -> nothing accepted, nothing to commit -> honest N/A.
        #   accepted  > 0 -> the run accepted relationships but none is committed
        #                    with lineage -> FAIL (not a vacuous N/A pass).
        #   accepted None -> baseline UNKNOWN; cannot prove the hole, default to
        #                    the historical N/A-not-fail.
        if accepted_rels and accepted_rels > 0:
            return "FAIL", False, (
                f"0 committed-with-lineage domain relationship edges for the "
                f"document, but the run ACCEPTED {accepted_rels} relationships — "
                f"every committed domain edge carries NULL source_chunk_ids "
                f"(lineage drop, NOT a vacuous N/A pass)"
            )
        accepted_note = (" accepted=0 (nothing to lose)" if accepted_rels == 0
                         else " accepted=UNKNOWN")
        return "NA", True, (
            "0 domain relationship edges for the document — N/A (relationships "
            f"depend on the system_links pass extracting any){accepted_note}"
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
        f"domain_relationship_edges(doc-scoped)={len(rows)} "
        f"with_source_chunk_ids={have} without={lack} "
        f"(PASS requires non-empty source_chunk_ids on EVERY domain edge of the "
        f"document)"
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


def compute_relationship_recall(committed_edges, merged_edges, rejected):
    """RELATIONSHIP RECALL baseline (#6c): committed-TOTAL domain relationship
    edges (edge PRESENCE) vs the DISTINCT MERGED-edge count the run intended to
    commit. MIRRORS compute_entity_recall exactly (committed vs distinct merged).

    C1 FIX — the denominator is the POST-DEDUP merged DISTINCT domain-edge count
    (audit blob graph_json->'edges_accepted' == len(merged.edges) after the
    (from_identity, rel_type, to_identity) cross-pass dedup; see
    extraction_merge.derive_ontology_graph_merge + merged_edge_count), NOT the
    PRE-dedup per-pass ``relationships_extracted`` emit sum. The merge collapses
    ~77 EMITTED rels onto ~32 DISTINCT (from,type,to) triples; healing adds
    lineage to EXISTING edges and NEVER creates new ones, so the committed
    numerator can never approach the 77 emit count — comparing the two straddles
    the merge dedup boundary and FALSE-FAILS a perfectly-healed doc (32/77≈42%).
    The distinct merged-edge count is the same side of the dedup boundary as the
    committed total, so committed==merged is exactly full recall.

    H1 — recall measures edge PRESENCE: the numerator is the committed-TOTAL
    domain-edge count, NOT a lineage-bearing subset. Precision (a SEPARATE check)
    already enforces a non-empty source_chunk_ids on EVERY committed domain edge,
    so making recall lineage-bearing would merely duplicate precision. Recall-as-
    presence answers a DISTINCT question (did every merged relationship reach
    ArcadeDB?), so the two are complementary, not redundant.

    Rejected rels are reported SEPARATELY and NOT counted as loss (valid rejection
    is not recall loss).

    Semantics (fail-closed — a merged edge that did not commit IS lineage loss):
      * committed==0 while merged>0  -> LOUD FAIL (keystone edge-commit collapse).
      * committed >= merged          -> PASS (full recall: every merged edge got
        committed; committed may exceed merged when prior runs' edges are present,
        since domain edges carry document_ids but no run_id constraint here).
      * 0 < committed < merged       -> FAIL (a merged edge did not commit = real
        recall loss; the gate is fail-closed).
      * merged unknown (None)        -> require committed>0 so a zeroed commit
        still fails (cannot prove full recall without the denominator).
      * merged==0 and committed==0   -> vacuous-but-honest PASS (nothing to lose).
    Returns (ok, detail)."""
    rej = rejected or 0
    if merged_edges is None:
        ok = committed_edges > 0
        return ok, (
            f"committed_edges={committed_edges} merged_edge_baseline=UNKNOWN "
            f"(audit blob edges_accepted unavailable) rejected={rej}; PASS requires "
            f"committed>0 (rejection reported, NOT counted as recall loss)"
        )
    if merged_edges > 0 and committed_edges == 0:
        return False, (
            f"committed_edges=0 but {merged_edges} merged distinct domain edges "
            f"expected — KEYSTONE EDGE-COMMIT COLLAPSE (committed dropped to zero; "
            f"rejected={rej}, not counted as loss)"
        )
    ratio = (committed_edges / merged_edges * 100.0) if merged_edges else 0.0
    # committed >= merged is full recall; committed < merged loses a merged edge
    # (lineage loss) -> FAIL. merged==0,committed==0 -> vacuous pass.
    ok = committed_edges >= merged_edges
    verdict = "" if ok else (
        f" — SHORTFALL: committed {committed_edges} < merged-distinct "
        f"{merged_edges} (a merged edge did not commit)"
    )
    return ok, (
        f"committed_edges={committed_edges} merged(distinct edges_accepted)="
        f"{merged_edges} recall={ratio:.0f}% rejected={rej} "
        f"(PASS requires committed>=merged-distinct; LOUD fail iff committed==0 "
        f"while merged>0; rejection reported separately, NOT counted as recall "
        f"loss){verdict}"
    )


def check_source_pages_shape(rel_edge_rows, chunk_pages):
    """source_pages SHAPE check: every lineage-bearing domain edge's
    ``source_pages`` must be a FLAT list of ints AND consistent with the resolved
    chunks' pages.

    Committed edges were observed with ``source_pages`` malformed as a
    list-of-lists (e.g. ``[[6, 7]]``) — see the worker-origin note in the gate
    header — and ALSO inconsistent with the resolved chunks (the two resolved
    chunks were both page 7, yet source_pages claimed ``[[6, 7]]``). The gate
    cannot heal this (it originates worker-side), but it MUST DETECT and FAIL on
    it. ``rel_edge_rows`` are the doc-scoped domain edges (each a dict with
    ``source_chunk_ids`` + ``source_pages``); ``chunk_pages`` maps chunk_id ->
    int page for the resolved chunks.

    A row is checked only when it is lineage-bearing (non-empty source_chunk_ids).
    For each such row:
      * source_pages PRESENT must be a list of plain ints (NOT a list-of-lists /
        nested) -> malformed shape FAILs (regardless of the chunks' pages).
      * the SET of source_pages must be a SUBSET of the resolved chunks' pages
        (every source_page must correspond to a resolved chunk's page) -> an
        extra/foreign page FAILs as inconsistent. (Resolved-chunk pages missing
        from chunk_pages are tolerated — the gate may not have every page.)
      * L1 — source_pages MISSING/None is NOT a malformation when the resolved
        chunks genuinely lack a page (after Fix P the worker omits source_pages
        for page-less chunks; see the gate-header note). It is ONLY a real
        inconsistency (-> FAIL) when the resolved chunks DO carry pages but
        source_pages is None (the page was available but dropped).
    No lineage-bearing edges -> N/A (nothing to shape check). Returns (ok, detail)."""
    rows = [r for r in rel_edge_rows if isinstance(r, dict)]
    lineage = [
        r for r in rows
        if isinstance(r.get("source_chunk_ids"), list) and r.get("source_chunk_ids")
    ]
    if not lineage:
        return True, (
            "N/A — no lineage-bearing domain edges to shape-check "
            "(precision/recall carry the verdict)"
        )
    malformed = 0
    inconsistent = 0
    checked = 0
    examples = []
    for r in lineage:
        sp = r.get("source_pages")
        if sp is None:
            # L1 — a MISSING/None source_pages is NOT a malformation when the
            # edge's resolved chunks genuinely lack a page: after Fix P the worker
            # legitimately OMITS source_pages when the resolved chunks have no
            # page_number (it is now derived from the resolved chunks, not written
            # unconditionally). Only when the resolved chunks DO carry pages is a
            # None source_pages a real inconsistency (the page was available but
            # dropped) -> FAIL. (A PRESENT-but-malformed source_pages is handled
            # below regardless of chunk pages.)
            resolved_pages_for_none = {
                chunk_pages[c] for c in r["source_chunk_ids"]
                if c in chunk_pages and isinstance(chunk_pages.get(c), int)
                and not isinstance(chunk_pages.get(c), bool)
            }
            if resolved_pages_for_none:
                inconsistent += 1
                if len(examples) < 3:
                    examples.append(
                        f"source_pages=None but resolved chunk pages exist="
                        f"{sorted(resolved_pages_for_none)} scids="
                        f"{r.get('source_chunk_ids')}")
            # else: pageless resolved chunks -> legitimately omitted, NOT malformed.
            continue
        checked += 1
        if not isinstance(sp, list) or any(not isinstance(p, int) or isinstance(p, bool)
                                           for p in sp):
            # list-of-lists ([[6,7]]) or non-int members -> malformed shape.
            malformed += 1
            if len(examples) < 3:
                examples.append(f"malformed source_pages={sp!r} (not a flat list of ints)")
            continue
        # consistency: source_pages set must be a subset of the resolved chunks'
        # pages. Only the chunk_ids we have a page for constrain the check.
        resolved_pages = {
            chunk_pages[c] for c in r["source_chunk_ids"]
            if c in chunk_pages and isinstance(chunk_pages.get(c), int)
        }
        if resolved_pages and not set(sp).issubset(resolved_pages):
            inconsistent += 1
            if len(examples) < 3:
                examples.append(
                    f"inconsistent source_pages={sp} not subset of resolved chunk "
                    f"pages={sorted(resolved_pages)}"
                )
    ok = malformed == 0 and inconsistent == 0
    ex = ("; ".join(examples)) if examples else "none"
    return ok, (
        f"lineage_bearing_edges={len(lineage)} checked={checked} "
        f"malformed_shape={malformed} inconsistent_with_chunk_pages={inconsistent} "
        f"(PASS requires every source_pages be a FLAT list of ints whose pages are "
        f"a subset of the resolved chunks' pages) examples: {ex}"
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


def merged_edge_count(doc):
    """Return the number of DISTINCT MERGED domain relationship edges for the run's
    document — the audit blob's POST-DEDUP edge count: graph_json->'edges_accepted'
    off ingest.document_graph_extractions. This equals ``len(merged.edges)`` AFTER
    the (from_identity, rel_type, to_identity) cross-pass dedup (see
    extraction_merge.derive_ontology_graph_merge: edge_dedup keyed by that triple,
    then _serialize_for_audit emits edges_accepted = len(merged.edges)). It is the
    count of distinct merged domain relationships the run intended to commit and is
    the CORRECT denominator for relationship recall — exactly analogous to
    merged_node_count's graph_json->'nodes' length for entity recall, and on the
    SAME side of the merge dedup boundary as the committed-edge total.

    NOTE the blob stores the merged edge population as this SCALAR count (there is
    NO graph_json->'edges' array — only nodes[]/mentions[] arrays plus the
    edges_accepted/edges_rejected scalars). merged.edges are DOMAIN relationship
    edges only (the LLM / system_links / walker-reachable rels); structural edges
    are built by a separate derive_structural_edges path and are NOT in this count,
    so this denominator already excludes structural edges (matching the
    STRUCTURAL_EDGE_TYPES exclusion the committed-edge precision query applies).

    Returns None when the query yields nothing (no blob row / pg failure) so the
    caller marks the recall baseline UNKNOWN rather than treating a missing
    denominator as 0. (document_id is the table PK — exactly one row per doc.)"""
    rows = pg(
        "SELECT (graph_json->>'edges_accepted') "
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


def relationship_edge_rows(doc):
    """DOC-SCOPED DOMAIN relationship edge rows — the document's FULL relationship
    population, with their source_chunk_ids + source_pages.

    DOC-SCOPING IS THE HONEST DISCRIMINATOR (reverts the false-green run-scoping).
    The gate must measure the DOCUMENT's full domain-relationship population and
    FAIL while lineage is missing — not just the handful a single run re-emitted.
    The prior fix scoped these by ``pipeline_run_id = :run`` and so inspected only
    the 3 edges THIS run touched, going BLIND to 29/32 (91%) of the document's
    domain edges that carry NULL source_chunk_ids (and pipeline_run_id=NULL — they
    were committed by older runs and were NOT re-emitted by the run under test). A
    run-scoped query simply does not see those un-re-emitted edges, so it passed
    GREEN while 91% of the document's relationships had no lineage. Scoping by
    ``document_ids CONTAINS '<doc>'`` counts EVERY domain relationship of the
    document (domain edges carry ``document_ids`` as a LIST — see
    arcadedb_graph.py upsert_relationships_batch), so the precision check FAILs
    until every one of them is lineage-bearing.

    Enumerate ONLY the non-structural (domain) edge types — the structural /
    derive-rules / doc-anchor edge classes are excluded via STRUCTURAL_EDGE_TYPES
    (see _ontology_edge_types) because they legitimately carry no
    source_chunk_ids. Returns a list of dicts {type, source_chunk_ids,
    source_pages}."""
    out = []
    for etype in _ontology_edge_types():
        rows = adb(
            f"SELECT '{etype}' AS etype, source_chunk_ids AS source_chunk_ids, "
            f"source_pages AS source_pages "
            f"FROM `{etype}` WHERE document_ids CONTAINS '{doc}'"
        )
        for r in rows:
            out.append({
                "type": etype,
                "source_chunk_ids": r.get("source_chunk_ids"),
                "source_pages": r.get("source_pages"),
            })
    return out


def chunk_pages_for(chunk_ids):
    """Return {chunk_id: page_number(int)} for the given resolved chunk ids, read
    off the TextChunk vertices. Used by the source_pages shape/consistency check to
    compare each edge's source_pages against its resolved chunks' actual pages.
    Empty input -> {}. Missing/non-int pages are simply omitted (the shape check
    tolerates chunks it has no page for)."""
    ids = [c for c in (chunk_ids or []) if isinstance(c, str) and c]
    if not ids:
        return {}
    in_list = ", ".join(f"'{c}'" for c in sorted(set(ids)))
    rows = adb(
        f"SELECT chunk_id, page_number FROM TextChunk WHERE chunk_id IN [{in_list}]")
    out = {}
    for r in rows:
        cid, pg = r.get("chunk_id"), r.get("page_number")
        if isinstance(cid, str) and isinstance(pg, int) and not isinstance(pg, bool):
            out[cid] = pg
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
    # 5. RELATIONSHIP PRECISION — every DOMAIN relationship edge OF THE DOCUMENT
    #    carries a non-empty source_chunk_ids LIST (resolved in the merge phase
    #    from the rel's positional self_refs). Counts ONLY domain (LLM /
    #    system_links) relationship edge types — structural / derive-rules /
    #    doc-anchor edges (HAS_COMPONENT, HAS_SECTION, HAS_FIGURE, HAS_IMAGE,
    #    NEAR_TEXT, EXTRACTED_FROM, MENTIONED_IN, …) are excluded via
    #    STRUCTURAL_EDGE_TYPES because they are built by a different path and
    #    legitimately carry no source_chunk_ids (the old gate counted them,
    #    yielding the spurious 170).
    #
    #    DOC-SCOPED by `document_ids CONTAINS '<doc>'` (REVERTS the false-green
    #    run-scoping). The goal is the DOCUMENT's FULL relationship population:
    #    every domain relationship of the document must be lineage-bearing. The
    #    prior fix scoped by pipeline_run_id and so inspected only the 3 edges THIS
    #    run re-emitted, going BLIND to 29/32 (91%) of the document's domain edges
    #    that carry NULL source_chunk_ids (committed by older runs, not re-emitted)
    #    — a false PASS. Doc-scope counts them all so the gate FAILs while lineage
    #    is missing (even if that means it correctly FAILS until healing happens).
    #
    #    NA-vs-FAIL: 0 committed domain edges is N/A ONLY if the run also accepted
    #    0 relationships; if the run accepted >0 but 0 are committed-with-lineage,
    #    that is the vacuous-pass hole the review flagged -> FAIL (accepted_rels).
    # ------------------------------------------------------------------
    accepted_rels = _pg_int_sum(run, "relationships_extracted")
    rejected_rels = _pg_int_sum(run, "relationships_rejected")
    rel_rows = relationship_edge_rows(doc)
    rel_status, rel_ok, rel_detail = compute_relationship_precision(rel_rows, accepted_rels)
    rel_label = ("RELATIONSHIP precision (DOC-scoped: every domain edge of the "
                 "document carries non-empty source_chunk_ids)")
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
    #    6c RELATIONSHIP: committed-TOTAL domain edges (edge PRESENCE) vs the
    #       DISTINCT MERGED-edge count (audit blob edges_accepted == len(merged.edges)
    #       after the (from,type,to) dedup) — MIRRORS 6a's merged nodes[] baseline.
    #       NOT the pre-dedup relationships_extracted emit sum (which straddles the
    #       merge dedup boundary and false-fails a healed doc). rejected reported
    #       separately (NOT counted as loss). LOUD fail iff committed==0 while
    #       merged>0; FAIL iff committed<merged.
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

    # Denominator = the DISTINCT MERGED domain-edge count (audit blob
    # graph_json->'edges_accepted' == len(merged.edges) after the (from,type,to)
    # cross-pass dedup; see merged_edge_count) — the SAME side of the merge dedup
    # boundary as the committed total, EXACTLY as entity recall uses the merged
    # nodes[] count. NOT the PRE-dedup per-pass relationships_extracted emit sum
    # (~77 for SA-2): the merge collapses ~77 emitted rels onto ~32 distinct
    # triples and healing never creates new edges, so committed-vs-emit straddles
    # the dedup boundary and FALSE-FAILS a healed doc (32/77≈42%). Numerator is the
    # committed-TOTAL doc-scoped domain edges (edge PRESENCE) — precision (#5)
    # separately enforces lineage on every edge, so presence is NOT redundant.
    merged_edges = merged_edge_count(doc)
    rr_ok, rr_detail = compute_relationship_recall(len(rel_rows), merged_edges, rejected_rels)
    results.append(("RELATIONSHIP recall (committed-total domain edges vs DISTINCT merged edges_accepted; LOUD fail on collapse, FAIL on shortfall; rejection NOT counted as loss)",
                    rr_ok, rr_detail))

    # ------------------------------------------------------------------
    # 6d. source_pages SHAPE — every lineage-bearing domain edge's source_pages,
    #     WHEN PRESENT, must be a FLAT list of ints AND consistent with the
    #     resolved chunks' pages. Committed edges were observed malformed as a
    #     list-of-lists ([[6,7]]) AND inconsistent (resolved chunks both page 7 yet
    #     source_pages claimed [[6,7]]). source_pages now originates WORKER-SIDE
    #     DERIVED FROM the resolved chunks' page_numbers (Fix P, app/workers/
    #     pipeline.py — it replaced the old unconditional ``props["source_pages"] =
    #     list(pages)`` write), so the worker legitimately OMITS source_pages when
    #     the resolved chunks have no page (L1: a None source_pages on page-less
    #     chunks is NOT a malformation). The host gate cannot heal a malformed
    #     PRESENT value; it DETECTs and FAILs on it here (worker origin reported
    #     separately for a worker-side fix).
    # ------------------------------------------------------------------
    sp_chunk_ids = []
    for r in rel_rows:
        scids = r.get("source_chunk_ids")
        if isinstance(scids, list):
            sp_chunk_ids.extend(c for c in scids if isinstance(c, str))
    sp_chunk_pages = chunk_pages_for(sp_chunk_ids)
    sp_ok, sp_detail = check_source_pages_shape(rel_rows, sp_chunk_pages)
    results.append(("source_pages shape (flat list of ints, consistent with resolved chunk pages)",
                    sp_ok, sp_detail))

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
    # Provenance-DROP warnings come from TWO distinct code paths / containers and
    # the old check looked for the WRONG string in the WRONG container:
    #   * WORKER string "dropping provenance row missing required fields" is
    #     emitted ONLY by the worker (_parse_pass_response in
    #     app/workers/pipeline.py) — NOT by docling-graph. The old gate greped it
    #     against docling-graph, so it could NEVER see a real worker drop.
    #     worker-graph-1 is the dedicated graph worker; worker-1 is the catch-all
    #     that also consumes the graph queue (MEMORY: catch-all worker trap), so
    #     BOTH are scanned for the worker string.
    #   * DOCLING-GRAPH string "build_provenance: dropping node ... no resolvable
    #     element_uid" is emitted by docling-graph (docker/docling-graph/app/
    #     provenance.py build_provenance_from_context). Greped against
    #     docling-graph (its actual emitter).
    # Drops are the SUM across both paths so the check can actually SEE real drops.
    WORKER_DROP = "dropping provenance row missing required fields"
    DG_DROP = "build_provenance: dropping node"  # "... no resolvable element_uid"
    dg, dg_lines = _docker_logs("eip-mmdpp-docling-graph-1", since, until)
    wg, wg_lines = _docker_logs("eip-mmdpp-worker-graph-1", since, until)
    wc, wc_lines = _docker_logs("eip-mmdpp-worker-1", since, until)
    # LINEAGE_GATE rejections are a WORKER string (_apply_strict_lineage_gate in
    # app/workers/pipeline.py). worker-graph-1 is the dedicated graph worker, but
    # worker-1 is the catch-all that ALSO consumes the graph queue (MEMORY:
    # catch-all worker trap), so a merge it grabs emits the rejection there and the
    # old worker-graph-only scan was BLIND to it. Sum across BOTH worker containers
    # — identical to the drop-check container handling above. The scanned-line
    # count is likewise the SUM so the no-evidence guard reflects both streams.
    wl_lines = wg_lines + wc_lines

    worker_drops = wg.count(WORKER_DROP) + wc.count(WORKER_DROP)
    dg_drops = dg.count(DG_DROP)
    drops = worker_drops + dg_drops
    gate_rejects = wg.count("LINEAGE_GATE: rejected") + wc.count("LINEAGE_GATE: rejected")

    # provenance-drop warnings (worker string in worker containers + docling-graph
    # string in docling-graph). FAIL-CLOSED on no log evidence across ALL scanned
    # containers (every relevant stream empty -> cannot confirm zero drops).
    drop_lines = dg_lines + wg_lines + wc_lines
    dl_no_evidence = window_degraded or drop_lines == 0
    if dl_no_evidence:
        reason = ("DEGRADED window (run not found)" if window_degraded
                  else "in-window log streams EMPTY")
        results.append(("zero provenance-drop warnings (run window; worker + docling-graph)",
                        False,
                        f"drops={drops} (worker={worker_drops} docling-graph={dg_drops}) "
                        f"/ scanned {drop_lines} log lines "
                        f"(docling-graph={dg_lines} worker-graph={wg_lines} worker={wc_lines}) "
                        f"— NO LOG EVIDENCE, cannot confirm zero warnings "
                        f"({reason}) [{window_desc}]"))
    else:
        results.append(("zero provenance-drop warnings (run window; worker + docling-graph)",
                        drops == 0,
                        f"drops={drops} (worker '{WORKER_DROP}'={worker_drops} in "
                        f"worker-graph+worker; docling-graph '{DG_DROP} … no "
                        f"resolvable element_uid'={dg_drops}) / scanned {drop_lines} "
                        f"log lines (docling-graph={dg_lines} worker-graph={wg_lines} "
                        f"worker={wc_lines}) in window [{window_desc}]"))

    # LINEAGE_GATE rejections (worker-graph + catch-all worker — both consume the
    # graph queue, so both can emit the worker rejection string).
    wl_no_evidence = window_degraded or wl_lines == 0
    if wl_no_evidence:
        reason = ("DEGRADED window (run not found)" if window_degraded
                  else "in-window log streams EMPTY")
        results.append(("zero LINEAGE_GATE rejections (run window; worker-graph + worker)",
                        False,
                        f"gate_rejections={gate_rejects} / scanned {wl_lines} log "
                        f"lines (worker-graph={wg_lines} worker={wc_lines}) — NO LOG "
                        f"EVIDENCE, cannot confirm zero rejections "
                        f"({reason}) [{window_desc}]"))
    else:
        results.append(("zero LINEAGE_GATE rejections (run window; worker-graph + worker)",
                        gate_rejects == 0,
                        f"gate_rejections={gate_rejects} / scanned {wl_lines} log "
                        f"lines (worker-graph={wg_lines} worker={wc_lines}) in window "
                        f"(nonzero = entities lacked lineage) [{window_desc}]"))

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
