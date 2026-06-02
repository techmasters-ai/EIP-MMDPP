"""Unit tests for the PURE verdict logic of scripts/verify_lineage_e2e.py
(Task 6 — harden the precise-lineage gate).

These exercise the DB-free decision functions the gate's PASS/FAIL hinges on,
with synthetic inputs — NO live ArcadeDB / Postgres / docker. The gate's I/O
helpers (adb/pg/_docker_logs) are intentionally NOT exercised here; they need a
live stack and are the controller's end-to-end step.

Covered:
  * compute_extracted_from_delta   — the HARD pre/post lineage discriminator (#1)
  * compute_fanout_precision       — ENTITY fan-out K-distribution + threshold (#3)
  * count_field_evidence_nulls /
    evaluate_field_precision        — FIELD chunk_id-None counting (#4)
  * compute_relationship_precision — RELATIONSHIP source_chunk_ids presence (#5)
  * compute_entity_recall /
    compute_relationship_recall     — per-target recall, LOUD on collapse (#6);
    ENTITY denominator = DISTINCT merged nodes[] (not the per-pass instance sum)
  * trace_first_complete           — EXTRACTED_FROM edge-anchored trace pick (#7a)

The module imports clean (stdlib only), so a plain import suffices.
"""
import scripts.verify_lineage_e2e as gate


# ---------------------------------------------------------------------------
# #1 compute_extracted_from_delta — HARD lineage discriminator (post>0 AND post>pre)
# ---------------------------------------------------------------------------
def test_delta_passes_when_post_exceeds_zero_pre():
    """SA-2 run command path: pre=0, post>0 -> PASS (edges built this run)."""
    ok, detail = gate.compute_extracted_from_delta(736, 0)
    assert ok is True
    assert "post=736" in detail and "pre=0" in detail and "delta=736" in detail


def test_delta_fails_when_post_zero():
    """No run-scoped edges at all -> FAIL even with pre=0."""
    ok, _ = gate.compute_extracted_from_delta(0, 0)
    assert ok is False


def test_delta_fails_when_post_not_greater_than_pre():
    """post == pre (no new edges this run) -> FAIL; not strictly greater."""
    ok, _ = gate.compute_extracted_from_delta(500, 500)
    assert ok is False
    # and strictly less also fails
    assert gate.compute_extracted_from_delta(400, 500)[0] is False


def test_delta_passes_with_nonzero_pre_when_strictly_greater():
    """Non-zero baseline (re-run): post must be strictly greater than pre."""
    ok, detail = gate.compute_extracted_from_delta(620, 500)
    assert ok is True
    assert "delta=120" in detail


# ---------------------------------------------------------------------------
# #3 compute_fanout_precision — ENTITY fan-out: no entity >= 50% of doc chunks
# ---------------------------------------------------------------------------
def test_fanout_precise_passes_below_threshold():
    """High-frequency-but-precise entity (26/102 ~25%) is BELOW the 50% cap."""
    counts = [26, 10, 4, 2, 1]
    ok, detail = gate.compute_fanout_precision(counts, doc_chunks=102)
    assert ok is True
    assert "max=26" in detail
    assert "fail_threshold=51" in detail  # int(102*0.5)


def test_fanout_coarse_all_document_fails():
    """The old bug: an entity fanned out to ~all 102 chunks -> >=50% -> FAIL."""
    counts = [102, 90, 80]  # worst at 100% of doc
    ok, detail = gate.compute_fanout_precision(counts, doc_chunks=102)
    assert ok is False
    assert "max=102" in detail
    assert "(100% of doc)" in detail


def test_fanout_exactly_at_threshold_fails():
    """Boundary: worst == threshold (51) is NOT strictly below -> FAIL."""
    ok, _ = gate.compute_fanout_precision([51, 3], doc_chunks=102)
    assert ok is False
    # one chunk below the threshold passes
    assert gate.compute_fanout_precision([50, 3], doc_chunks=102)[0] is True


def test_fanout_zero_doc_chunks_fails_cannot_compute():
    """doc_chunks==0 -> cannot compute fraction -> FAIL (can't prove precision)."""
    ok, detail = gate.compute_fanout_precision([5, 2], doc_chunks=0)
    assert ok is False


def test_fanout_empty_edge_set_fails_no_evidence():
    """0 entities with edges -> FAIL (no evidence; #1 already fails)."""
    ok, detail = gate.compute_fanout_precision([], doc_chunks=102)
    assert ok is False
    assert "0 entities" in detail


def test_fanout_reports_full_distribution():
    """Distribution (min/median/max + top5) surfaced for human inspection."""
    counts = [30, 20, 15, 10, 5, 3, 1]
    _, detail = gate.compute_fanout_precision(counts, doc_chunks=200)
    assert "min=1" in detail
    assert "max=30" in detail
    assert "top5=[30, 20, 15, 10, 5]" in detail
    assert "entities_with_edges=7" in detail


# ---------------------------------------------------------------------------
# #4 count_field_evidence_nulls + evaluate_field_precision — FIELD chunk_id None
# ---------------------------------------------------------------------------
def test_field_nulls_all_resolved_passes():
    """Every field-evidence row carries a resolved chunk_id -> 0 nulls -> PASS."""
    blobs = [
        {"range": [{"chunk_id": "c1", "value": "43 km"}],
         "guidance": [{"chunk_id": "c2", "value": "radio"}]},
        {"name": [{"chunk_id": "c3", "value": "SA-2"}]},
    ]
    total, nulls, entities = gate.count_field_evidence_nulls(blobs)
    assert total == 3
    assert nulls == 0
    assert entities == 2
    ok, detail = gate.evaluate_field_precision(total, nulls, entities)
    assert ok is True
    assert "null_chunk_id=0" in detail


def test_field_nulls_one_none_fails():
    """A single None chunk_id (resolution regressed) -> FAIL."""
    blobs = [
        {"range": [{"chunk_id": "c1"}, {"chunk_id": None}]},
    ]
    total, nulls, _ = gate.count_field_evidence_nulls(blobs)
    assert total == 2
    assert nulls == 1
    ok, _ = gate.evaluate_field_precision(total, nulls, 1)
    assert ok is False


def test_field_nulls_empty_string_counts_as_null():
    """An empty/whitespace chunk_id string is treated as unresolved."""
    blobs = [{"f": [{"chunk_id": ""}, {"chunk_id": "   "}, {"chunk_id": "c9"}]}]
    total, nulls, _ = gate.count_field_evidence_nulls(blobs)
    assert total == 3
    assert nulls == 2


def test_field_nulls_handles_stringified_json_blob():
    """ArcadeDB may hand back the JSON property as a string; it is parsed."""
    blobs = ['{"range": [{"chunk_id": "c1"}, {"chunk_id": null}]}']
    total, nulls, _ = gate.count_field_evidence_nulls(blobs)
    assert total == 2
    assert nulls == 1


def test_field_nulls_unparseable_blob_counts_as_null_fail_loud():
    """A non-JSON string blob is indeterminate -> counted as a null row so the
    check fails loud rather than silently skipping evidence."""
    blobs = ["not json at all"]
    total, nulls, _ = gate.count_field_evidence_nulls(blobs)
    assert total == 1
    assert nulls == 1


def test_field_nulls_skips_none_and_nondict_blobs():
    """None / non-dict blobs (entity with no evidence) contribute no rows."""
    blobs = [None, 42, [], {"f": [{"chunk_id": "c1"}]}]
    total, nulls, entities = gate.count_field_evidence_nulls(blobs)
    assert total == 1
    assert nulls == 0
    assert entities == 1


def test_field_precision_zero_rows_fails_closed():
    """0 field-evidence rows -> NO EVIDENCE -> FAIL-CLOSED (resolver may not have run)."""
    ok, detail = gate.evaluate_field_precision(0, 0, 0)
    assert ok is False
    assert "NO EVIDENCE" in detail


# ---------------------------------------------------------------------------
# #5 compute_relationship_precision — non-empty source_chunk_ids on every edge
# ---------------------------------------------------------------------------
def test_relationship_precision_all_have_source_chunks_passes():
    rows = [
        {"type": "GUIDED_BY", "source_chunk_ids": ["c1", "c2"]},
        {"type": "PART_OF", "source_chunk_ids": ["c3"]},
    ]
    status, ok, detail = gate.compute_relationship_precision(rows)
    assert status == "PASS"
    assert ok is True
    assert "with_source_chunk_ids=2" in detail and "without=0" in detail


def test_relationship_precision_missing_source_chunks_fails():
    rows = [
        {"type": "GUIDED_BY", "source_chunk_ids": ["c1"]},
        {"type": "NEAR", "source_chunk_ids": []},          # empty list
        {"type": "PART_OF", "source_chunk_ids": None},     # absent
        {"type": "OTHER"},                                  # key missing
    ]
    status, ok, detail = gate.compute_relationship_precision(rows)
    assert status == "FAIL"
    assert ok is False
    assert "without=3" in detail


def test_relationship_precision_zero_edges_zero_accepted_is_na_not_fail():
    """0 committed domain edges AND 0 accepted rels in the audit blob -> N/A.
    Vacuously honest: the run accepted nothing, so there is nothing to lose."""
    status, ok, detail = gate.compute_relationship_precision([], accepted_rels=0)
    assert status == "NA"
    assert ok is True
    assert "N/A" in detail


def test_relationship_precision_zero_edges_unknown_accepted_is_na():
    """0 committed edges with the accepted baseline UNKNOWN (None) -> N/A. We
    cannot prove the vacuous-pass hole without the denominator, so default to the
    historical N/A-not-fail rather than fail-closed on a missing baseline."""
    status, ok, _ = gate.compute_relationship_precision([])
    assert status == "NA"
    assert ok is True


def test_relationship_precision_zero_committed_but_accepted_positive_FAILS():
    """THE VACUOUS-PASS HOLE the review flagged: the run accepted >0 relationships
    but 0 are committed-with-lineage (every domain edge carries NULL
    source_chunk_ids, so the doc-scoped lineage-bearing set is empty). This MUST
    FAIL, not pass as N/A — the document's relationships exist but none is
    lineage-bearing."""
    status, ok, detail = gate.compute_relationship_precision([], accepted_rels=77)
    assert status == "FAIL"
    assert ok is False
    assert "77" in detail


def test_relationship_precision_is_doc_population_not_run_scoped_concept():
    """Doc-scoped fail-closed: when the document carries 32 domain edges and only
    3 are lineage-bearing (29 NULL source_chunk_ids), precision FAILS — the gate
    measures the DOCUMENT's full relationship population, not just the 3 a run
    re-emitted."""
    rows = ([{"type": "ASSOCIATED_WITH", "source_chunk_ids": ["c1"]}] * 3
            + [{"type": "ASSOCIATED_WITH", "source_chunk_ids": None}] * 29)
    status, ok, detail = gate.compute_relationship_precision(rows, accepted_rels=77)
    assert status == "FAIL"
    assert ok is False
    assert "without=29" in detail


# ---------------------------------------------------------------------------
# #6a compute_entity_recall — denominator is the DISTINCT MERGED-node count
# (audit blob graph_json->'nodes' length), NOT the per-pass instance sum.
# Fail-closed: committed>=merged PASS; 0<committed<merged FAIL; committed==0
# while merged>0 LOUD FAIL. The per-pass instance sum, when supplied, is a
# REPORTED diagnostic only and must never flip the verdict.
# ---------------------------------------------------------------------------
def test_entity_recall_committed_meets_merged_passes():
    """The bug scenario, corrected: 27 committed >= 23 DISTINCT merged -> full
    recall PASS. (The old code compared 27 vs 133 per-pass instances and reported
    a spurious ~20% — that 133 is now only a labeled diagnostic.)"""
    ok, detail = gate.compute_entity_recall(committed=27, merged=23,
                                            instance_emissions=133)
    assert ok is True
    assert "committed_entities=27" in detail
    assert "merged(distinct nodes[])=23" in detail
    assert "recall=117%" in detail  # 27/23
    # the old per-pass sum is reported but explicitly flagged NOT distinct
    assert "per_pass_instance_emissions=133" in detail
    assert "NOT distinct" in detail


def test_entity_recall_committed_equals_merged_passes():
    """Boundary: committed == merged is full recall (every merged node committed)."""
    ok, detail = gate.compute_entity_recall(committed=23, merged=23)
    assert ok is True
    assert "recall=100%" in detail


def test_entity_recall_collapse_to_zero_fails_loud():
    """The recall-gate-collapse scenario: 23 merged entities expected, committed 0
    -> LOUD keystone fail (every merged identity failed to commit = lineage loss)."""
    ok, detail = gate.compute_entity_recall(committed=0, merged=23)
    assert ok is False
    assert "KEYSTONE RECALL COLLAPSE" in detail


def test_entity_recall_shortfall_below_merged_fails():
    """0 < committed < merged -> FAIL. DECISION: a merged entity that did not
    commit IS lineage loss, so the fail-closed gate FAILs (not WARN). e.g. 20
    committed of 23 merged means 3 distinct merged identities never reached
    ArcadeDB — a real recall regression."""
    ok, detail = gate.compute_entity_recall(committed=20, merged=23)
    assert ok is False
    assert "committed_entities=20" in detail
    assert "merged(distinct nodes[])=23" in detail
    assert "recall=87%" in detail  # 20/23


def test_entity_recall_zero_merged_zero_committed_is_honest_pass():
    """Nothing merged, nothing committed -> not a regression (0>=0)."""
    ok, _ = gate.compute_entity_recall(committed=0, merged=0)
    assert ok is True


def test_entity_recall_unknown_baseline_requires_committed():
    """Merged baseline unavailable (audit blob absent) -> still require committed>0
    (no vacuous pass without the denominator)."""
    assert gate.compute_entity_recall(committed=5, merged=None)[0] is True
    assert gate.compute_entity_recall(committed=0, merged=None)[0] is False


def test_entity_recall_instance_emissions_never_flips_verdict():
    """The per-pass instance sum is REPORT-ONLY: a huge instance count must NOT
    turn a full-recall PASS into a fail. committed>=merged still PASSES even when
    instance_emissions dwarfs committed (the exact false-20% bug being fixed)."""
    ok, detail = gate.compute_entity_recall(committed=27, merged=23,
                                            instance_emissions=999)
    assert ok is True
    assert "per_pass_instance_emissions=999" in detail
    # and a real shortfall still FAILs regardless of the diagnostic number
    bad_ok, _ = gate.compute_entity_recall(committed=10, merged=23,
                                           instance_emissions=2)
    assert bad_ok is False


# ---------------------------------------------------------------------------
# #6c compute_relationship_recall — rejection is NOT counted as loss
# ---------------------------------------------------------------------------
def test_relationship_recall_committed_present_passes():
    ok, detail = gate.compute_relationship_recall(committed_edges=8, accepted=8, rejected=3)
    assert ok is True
    assert "rejected=3" in detail


def test_relationship_recall_collapse_fails():
    """Accepted post-validation but 0 committed edges -> commit dropped to zero."""
    ok, detail = gate.compute_relationship_recall(committed_edges=0, accepted=8, rejected=2)
    assert ok is False
    assert "dropped to zero" in detail


def test_relationship_recall_rejection_only_not_loss():
    """All rels rejected, 0 accepted, 0 committed -> NOT a loss (valid rejection)."""
    ok, _ = gate.compute_relationship_recall(committed_edges=0, accepted=0, rejected=12)
    assert ok is True


def test_relationship_recall_unknown_accepted_reports_only():
    ok, detail = gate.compute_relationship_recall(committed_edges=0, accepted=None, rejected=0)
    assert ok is True
    assert "UNKNOWN" in detail


# ---------------------------------------------------------------------------
# #7a trace_first_complete — pick first EXTRACTED_FROM row with chunk+doc+page
# ---------------------------------------------------------------------------
def test_trace_picks_first_complete_row():
    rows = [
        {"name": "X", "chunk_id": None, "page": None, "doc": "d1"},  # incomplete
        {"name": "SA-2", "chunk_id": "c5", "page": 7, "doc": "d1"},  # complete
    ]
    ok, detail, page = gate.trace_first_complete(rows)
    assert ok is True
    assert page == 7
    assert "chunk_id=c5" in detail and "page=7" in detail


def test_trace_no_rows_fails():
    ok, detail, page = gate.trace_first_complete([])
    assert ok is False
    assert page is None
    assert "no run-scoped EXTRACTED_FROM rows" in detail


def test_trace_all_incomplete_fails():
    """Rows present but none has chunk+doc+page -> FAIL, surfaces first row."""
    rows = [{"name": "Y", "chunk_id": "c1", "page": None, "doc": "d1"}]
    ok, detail, page = gate.trace_first_complete(rows)
    assert ok is False
    assert page is None
    assert "missing chunk/doc/page" in detail


def test_trace_page_zero_is_valid():
    """page == 0 is a real page (front matter); not treated as missing."""
    rows = [{"name": "Z", "chunk_id": "c1", "page": 0, "doc": "d1"}]
    ok, _, page = gate.trace_first_complete(rows)
    assert ok is True
    assert page == 0


# ---------------------------------------------------------------------------
# RELATIONSHIP-precision edge classification (Bug 2 fix): the structural set
# must exclude domain semantic rels and INCLUDE the doc-anchor / derive
# structural edges that legitimately carry no source_chunk_ids. The old gate
# omitted HAS_IMAGE / NEAR_TEXT / CONTAINS, so it over-counted them as domain
# edges (HAS_IMAGE 34 + NEAR_TEXT 136 = the spurious 170).
# ---------------------------------------------------------------------------
def test_structural_set_includes_chunk_and_provenance_edges():
    for et in ("CONTAINS_TEXT", "SAME_PAGE", "HAS_PROVENANCE",
               "EXTRACTED_FROM", "NEXT_CHUNK", "HAS_ALIAS"):
        assert et in gate.STRUCTURAL_EDGE_TYPES, et


def test_structural_set_includes_document_anchor_edges():
    """The doc-anchor walker edges are structural (built via
    create_structural_edge_sync, pass_origins={'document_anchors'})."""
    for et in ("HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF",
               "HAS_IMAGE", "NEAR_TEXT"):
        assert et in gate.STRUCTURAL_EDGE_TYPES, et


def test_structural_set_includes_previously_missing_overcounted_edges():
    """Regression for the spurious 170: HAS_IMAGE, NEAR_TEXT, CONTAINS, and the
    MENTIONED_IN derive edge MUST be classified structural (excluded)."""
    for et in ("HAS_IMAGE", "NEAR_TEXT", "CONTAINS", "MENTIONED_IN"):
        assert et in gate.STRUCTURAL_EDGE_TYPES, et


def test_structural_set_excludes_domain_semantic_rels():
    """Domain (LLM / system_links) relationship types must NOT be classified
    structural — they carry source_chunk_ids and are the ones we count."""
    for et in ("ASSOCIATED_WITH", "CUES", "DETECTS", "TRACKS", "GUIDES",
               "LAUNCHES", "ENGAGES", "HAS_COMPONENT", "HAS_SUBSYSTEM"):
        assert et not in gate.STRUCTURAL_EDGE_TYPES, et


def test_ontology_edge_types_filters_structural(monkeypatch):
    """_ontology_edge_types() returns schema edge classes minus the structural
    set — so domain rels survive and structural/anchor edges are dropped."""
    schema_edges = [
        {"name": "ASSOCIATED_WITH"}, {"name": "CUES"},        # domain
        {"name": "HAS_IMAGE"}, {"name": "NEAR_TEXT"},          # anchor (drop)
        {"name": "EXTRACTED_FROM"}, {"name": "MENTIONED_IN"},  # derive  (drop)
        {"name": "CONTAINS_TEXT"},                              # chunk   (drop)
    ]
    monkeypatch.setattr(gate, "adb", lambda sql: schema_edges)
    domain = set(gate._ontology_edge_types())
    assert domain == {"ASSOCIATED_WITH", "CUES"}


def test_relationship_edge_rows_scopes_by_document_ids_not_run(monkeypatch):
    """RELATIONSHIP-precision edges must be DOC-SCOPED by `document_ids CONTAINS
    '<doc>'`, NOT run-scoped by pipeline_run_id (REVERTING the false-green prior
    fix).

    The goal is the DOCUMENT's full relationship population. Run-scoping inspected
    only the 3 edges THIS run re-emitted and was blind to 29/32 (91%) of the
    document's domain edges that carry NULL source_chunk_ids — so it passed green
    while 91% of the document's relationships had no lineage. Doc-scoping counts
    every domain relationship of the document so the gate FAILs while lineage is
    missing. The domain-edge-type restriction (structural edges excluded) is
    preserved."""
    issued = []

    def fake_adb(sql):
        issued.append(sql)
        if "schema:types" in sql:
            return [{"name": "ASSOCIATED_WITH"}, {"name": "CUES"},
                    {"name": "HAS_IMAGE"}, {"name": "EXTRACTED_FROM"}]
        if "ASSOCIATED_WITH" in sql:
            return [{"source_chunk_ids": None}] * 22
        if "CUES" in sql:
            return [{"source_chunk_ids": None}] * 7
        return []

    monkeypatch.setattr(gate, "adb", fake_adb)
    rows = gate.relationship_edge_rows("DOC-XYZ")
    # 22 ASSOCIATED_WITH + 7 CUES = 29 domain edges; structural ones excluded.
    assert len(rows) == 29
    types = {r["type"] for r in rows}
    assert types == {"ASSOCIATED_WITH", "CUES"}
    # every per-type query scopes by the DOCUMENT, never by pipeline_run_id.
    per_type = [s for s in issued if "schema:types" not in s]
    assert per_type, "expected per-edge-type queries"
    for sql in per_type:
        assert "document_ids CONTAINS 'DOC-XYZ'" in sql
        assert "pipeline_run_id =" not in sql


# ---------------------------------------------------------------------------
# RELATIONSHIP recall floor — committed lineage-bearing edges vs accepted rels.
# The old check only failed on committed==0; 3-of-77 passed green. The floor
# fails on a large shortfall so "3 of 77 accepted" is not called full recall.
# ---------------------------------------------------------------------------
def test_relationship_recall_floor_fails_on_large_shortfall():
    """3 committed of 77 accepted (~4%) is a massive shortfall -> FAIL. The old
    check (committed>0) would have passed this green."""
    ok, detail = gate.compute_relationship_recall(committed_edges=3, accepted=77, rejected=0)
    assert ok is False
    assert "3" in detail and "77" in detail


def test_relationship_recall_floor_passes_at_or_above_threshold():
    """Committed >= floor*accepted -> PASS. With a 50% floor, 40 of 77 passes."""
    ok, _ = gate.compute_relationship_recall(committed_edges=40, accepted=77, rejected=0)
    assert ok is True


def test_relationship_recall_floor_reports_committed_vs_accepted():
    """The shortfall report surfaces committed vs accepted prominently."""
    ok, detail = gate.compute_relationship_recall(committed_edges=3, accepted=77, rejected=2)
    assert ok is False
    assert "committed" in detail.lower()
    assert "rejected=2" in detail


# ---------------------------------------------------------------------------
# source_pages shape check — committed edges have source_pages malformed as a
# list-of-lists ([[6,7]]); the gate must DETECT and FAIL on the malformed shape
# AND on inconsistency with the resolved chunks' pages.
# ---------------------------------------------------------------------------
def test_source_pages_flat_and_consistent_passes():
    """Flat list of ints, consistent with the resolved chunks' pages -> PASS."""
    rows = [
        {"type": "ASSOCIATED_WITH", "source_chunk_ids": ["c1", "c2"],
         "source_pages": [6, 7]},
    ]
    chunk_pages = {"c1": 6, "c2": 7}
    ok, detail = gate.check_source_pages_shape(rows, chunk_pages)
    assert ok is True


def test_source_pages_list_of_lists_fails():
    """The committed malformation [[6,7]] (list-of-lists) -> FAIL."""
    rows = [
        {"type": "ASSOCIATED_WITH", "source_chunk_ids": ["c1", "c2"],
         "source_pages": [[6, 7]]},
    ]
    chunk_pages = {"c1": 7, "c2": 7}
    ok, detail = gate.check_source_pages_shape(rows, chunk_pages)
    assert ok is False
    assert "malformed" in detail.lower() or "list-of-lists" in detail.lower()


def test_source_pages_inconsistent_with_chunk_pages_fails():
    """Flat list of ints but inconsistent with the resolved chunks' pages
    (source_pages says page 6 but the resolved chunks are both page 7) -> FAIL."""
    rows = [
        {"type": "ASSOCIATED_WITH", "source_chunk_ids": ["c1", "c2"],
         "source_pages": [6]},
    ]
    chunk_pages = {"c1": 7, "c2": 7}
    ok, detail = gate.check_source_pages_shape(rows, chunk_pages)
    assert ok is False
    assert "inconsistent" in detail.lower()


def test_source_pages_no_lineage_edges_is_na():
    """No lineage-bearing edges (none has source_chunk_ids) -> nothing to shape
    check; N/A-not-fail (the precision/recall checks already carry that verdict)."""
    rows = [{"type": "ASSOCIATED_WITH", "source_chunk_ids": None}]
    ok, detail = gate.check_source_pages_shape(rows, {})
    assert ok is True
    assert "N/A" in detail or "no lineage" in detail.lower()


def test_drop_check_greps_worker_string_in_worker_containers(monkeypatch):
    """#7b drop-check must grep the WORKER container(s) for the WORKER string
    ('dropping provenance row missing required fields') — the old gate greped that
    string against docling-graph (its non-emitter), so it could NEVER see a real
    worker drop. It must ALSO grep docling-graph for its OWN drop string
    ('build_provenance: dropping node … no resolvable element_uid') and SUM both.

    Drive main() with all I/O monkeypatched; assert the worker drop is COUNTED
    (the provenance-drop check FAILs because a worker emitted the worker string),
    and that docling-graph was scanned with its own string."""
    import sys

    # _docker_logs(container, since, until) -> (text, n_lines). Emit the WORKER
    # drop string ONLY from a worker container; emit the docling-graph drop string
    # ONLY from docling-graph. If the gate greps the right string in the right
    # container, drops>0 and the provenance-drop check FAILs.
    seen = {}

    def fake_docker_logs(container, since, until):
        seen[container] = True
        if container == "eip-mmdpp-worker-graph-1":
            return ("INFO start\n_parse_pass_response: dropping provenance row "
                    "missing required fields: {...}\nINFO end\n", 3)
        if container == "eip-mmdpp-worker-1":
            return ("INFO catchall worker idle\n", 1)
        if container == "eip-mmdpp-docling-graph-1":
            return ("INFO dg\nbuild_provenance: dropping node n1 (label=X) — no "
                    "resolvable element_uid\n", 2)
        return ("INFO\n", 1)

    # neutralize every other I/O helper so main() runs purely off our fakes.
    monkeypatch.setattr(gate, "_docker_logs", fake_docker_logs)
    monkeypatch.setattr(gate, "adb", lambda sql: [])
    monkeypatch.setattr(gate, "count", lambda sql: 0)
    monkeypatch.setattr(gate, "pg", lambda sql: [])
    monkeypatch.setattr(gate, "resolve_doc", lambda run, fb: (fb, "test"))
    monkeypatch.setattr(gate, "run_window", lambda run: ("2026-05-31T00:00:00+00:00",
                                                         "2026-05-31T01:00:00+00:00"))
    monkeypatch.setattr(gate, "total_entities", lambda: 0)
    monkeypatch.setattr(gate, "committed_entity_field_evidence", lambda: [])
    monkeypatch.setattr(gate, "merged_node_count", lambda doc: None)
    monkeypatch.setattr(gate, "_pg_int_sum", lambda run, col: None)
    monkeypatch.setattr(gate, "relationship_edge_rows", lambda doc: [])
    monkeypatch.setattr(gate, "chunk_pages_for", lambda ids: {})
    monkeypatch.setattr(sys, "argv", ["prog", "--run", "RUN-1", "--doc", "DOC-1"])

    captured = {}

    def fake_print(*a, **k):
        captured.setdefault("out", []).append(" ".join(str(x) for x in a))

    monkeypatch.setattr("builtins.print", fake_print)

    exited = {}

    def fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)
    try:
        gate.main()
    except SystemExit:
        pass

    out = "\n".join(captured.get("out", []))
    # the worker container(s) AND docling-graph were scanned.
    assert "eip-mmdpp-worker-graph-1" in seen
    assert "eip-mmdpp-worker-1" in seen
    assert "eip-mmdpp-docling-graph-1" in seen
    # the provenance-drop check saw the drops and FAILED (worker=1 + docling-graph=1).
    drop_line = [ln for ln in out.splitlines() if "provenance-drop" in ln]
    assert drop_line, "expected a provenance-drop result line"
    assert "[FAIL]" in drop_line[0]
    assert "drops=2" in drop_line[0]
    assert "worker" in drop_line[0] and "docling-graph" in drop_line[0]


def test_committed_field_evidence_has_no_document_id_filter(monkeypatch):
    """Global entity vertices (RADAR_SYSTEM / MISSILE_SYSTEM / …) carry no
    document_id property, so the field-evidence read must NOT filter on it —
    the old `WHERE document_id = ...` matched zero rows and the check
    fail-closed on 0 evidence. It now reads every entity-type vertex (matching
    total_entities() and the working spot-check)."""
    issued = []

    def fake_adb(sql):
        issued.append(sql)
        return [{"fe": {"nominal_rf_mhz": [{"chunk_id": "c1"}]}}]

    monkeypatch.setattr(gate, "adb", fake_adb)
    blobs = gate.committed_entity_field_evidence()
    assert len(blobs) == len(gate.ENTITY_TYPES)
    assert all("document_id" not in s for s in issued)
    assert all("_field_evidence" in s for s in issued)
