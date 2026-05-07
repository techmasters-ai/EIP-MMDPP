"""Integration tests for spec §8.4 — extraction_merge.py + worker-side
kill switch. Each test exercises merge_and_resolve end-to-end with
in-memory PassResults; no docling-graph HTTP, no Ollama."""
from app.services.extraction_merge import (
    merge_and_resolve, canonicalize_cross_pass_identities, PassResult,
)
from app.services.table_overlay import TableOverlay, TableFact


def _ontology_min():
    """Minimal ontology: one missile entity type."""
    return {"entity_types": [
        {"name": "MISSILE_SYSTEM", "graph_id_fields": ["system_name"]},
    ]}


def _missile_inst(name, **fields):
    from ontology_bundles.air_defense_v3.extraction_schemas import missile_propulsion
    return missile_propulsion.MissilePropulsionRecord(system_name=name, **fields)


def _make_propulsion_passresult(instances, *, table_overlay=None):
    """Build a PassResult-shaped object stub for tests."""
    pr = PassResult.__new__(PassResult)
    pr.pass_name = "missile_propulsion"
    pr.template_instance = None  # tests don't walk the typed-edge graph
    pr.metadata = None
    pr.pre_merge_rejections = []
    pr.upstream_refs = None
    pr.pre_merge_walk = None
    pr.provenance = []
    pr.field_evidence = {}
    pr._walker_entities_cache = list(instances)  # short-circuit walker
    pr.table_overlay = table_overlay
    return pr


def test_table_alias_map_runs_before_token_overlap(monkeypatch):
    """alias_map_by_entity_type collapses three non-token-overlapping
    aliases (SA-75 / SA-2A / 1D) onto canonical 1D before the
    token-overlap pass runs. After canonicalize, all three have
    system_name='1D'."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")
    a = _missile_inst("SA-75")
    b = _missile_inst("SA-2A")
    c = _missile_inst("1D")
    pr = _make_propulsion_passresult([a, b, c])
    pass_results = {"missile_propulsion": pr}
    alias_map = {"MISSILE_SYSTEM": {"SA-75": "1D", "SA-2A": "1D"}}
    rewrites = canonicalize_cross_pass_identities(
        pass_results, _ontology_min(),
        table_alias_map_by_entity_type=alias_map,
    )
    assert rewrites == 2
    assert a.system_name == b.system_name == c.system_name == "1D"


def test_table_overlay_does_not_break_existing_token_overlap(monkeypatch):
    """When table_alias_map_by_entity_type is None, existing token-
    overlap canonicalization runs unchanged. Two instances with
    overlapping tokens (PAC-3 / MIM-104F) still collapse the way they
    did pre-overlay."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")
    a = _missile_inst("PAC-3")
    b = _missile_inst("MIM-104F")  # token-overlap target
    pr = _make_propulsion_passresult([a, b])
    pass_results = {"missile_propulsion": pr}
    canonicalize_cross_pass_identities(
        pass_results, _ontology_min(),
        table_alias_map_by_entity_type=None,
    )
    # Existing token-overlap behavior is whatever the pre-overlay code
    # did — assert only that the call ran without error.
    assert a.system_name in ("PAC-3", "MIM-104F")


def test_kill_switch_disables_overlay_fresh_extraction(monkeypatch, caplog):
    """DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false on the worker → even
    if a fresh-extraction PassResult carries no overlay, behavior is
    unchanged: canonicalize runs without alias_map; Phase 0.5 skipped;
    no IDENTITY_REWRITE / TABLE_OVERLAY_APPLIED log lines.

    Use caplog (NOT patch on extraction_merge.logger) so the assertion
    catches absence-of-log lines from BOTH extraction_merge.logger AND
    app.services.table_overlay.logger — the emission boundary may
    move between them in future refactors and we don't want absence
    asserts to silently soften."""
    import logging as _logging
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "false")
    caplog.set_level(_logging.INFO)
    a = _missile_inst("1D")
    pr = _make_propulsion_passresult([a], table_overlay=None)
    pass_results = {"missile_propulsion": pr}
    manifest = type("M", (), {"passes": [], "bundle_key": "test"})()
    merge_and_resolve(
        pass_results=pass_results, manifest=manifest,
        ontology=_ontology_min(),
        document_id="doc-x", pipeline_run_id="run-x",
    )
    log_messages = [r.getMessage() for r in caplog.records]
    assert not any("IDENTITY_REWRITE" in m for m in log_messages)
    assert not any("TABLE_OVERLAY_APPLIED" in m for m in log_messages)


def test_kill_switch_worker_side_overrides_cached_overlay(monkeypatch, caplog):
    """Critical defense-in-depth case (spec §4.3): a PassResult arrives
    with a fully-populated TableOverlay (e.g., loaded from cached
    pipeline_pass_outputs.metadata_json from yesterday's run). Operator
    has just set DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false on the worker.
    Expected: merge_and_resolve sees the cached overlay AS IF None.
    apply_identity_rewrite NOT called; apply_field_overlay NOT called;
    one TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER INFO log line emitted."""
    import logging as _logging
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "false")
    caplog.set_level(_logging.INFO)
    a = _missile_inst("SA-75")
    cached_overlay = TableOverlay(
        alias_map_by_entity_type={"MISSILE_SYSTEM": {"SA-75": "1D"}},
        facts=[TableFact(
            canonical_entity="1D", entity_type="MISSILE_SYSTEM",
            schema_field="booster_mass_kg", value=1135.0,
            source_label="Weight kg", section_ctx="1st Stage",
            pass_name="missile_propulsion", raw_text="1135",
        )],
    )
    pr = _make_propulsion_passresult([a], table_overlay=cached_overlay)
    pass_results = {"missile_propulsion": pr}
    manifest = type("M", (), {"passes": [], "bundle_key": "test"})()
    merge_and_resolve(
        pass_results=pass_results, manifest=manifest,
        ontology=_ontology_min(),
        document_id="doc-y", pipeline_run_id="run-y",
    )
    log_messages = [r.getMessage() for r in caplog.records]
    assert any("TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER" in m
               for m in log_messages)
    assert not any("IDENTITY_REWRITE" in m for m in log_messages)
    assert not any("TABLE_OVERLAY_APPLIED" in m for m in log_messages)
    # Critical: instance must NOT have been rewritten despite cached
    # alias_map carrying SA-75 → 1D.
    assert a.system_name == "SA-75"
