"""End-to-end fixture for spec §8.5: synthetic DoclingDocument with
SA-2-shaped variants table, 4-pass stub LLM responses encoding the
empirical alias-scatter + wrong-propulsion-value failure modes,
through merge_and_resolve. Validates Mechanism A1 collapses aliases
AND overrides wrong propulsion values."""
from app.services.extraction_merge import merge_and_resolve, PassResult
from app.services.table_overlay import (
    TableOverlay as WorkerTO, TableFact, CrossEntityHint,
)
from ontology_bundles.air_defense_v3.extraction_schemas import (
    missile_propulsion, missile_airframe, missile_kinematics,
    missile_speed_timing,
)


def _build_overlay_for_sa2():
    """Synthetic overlay matching SA-2 column 0 (1D) and column 1 (13D)."""
    alias_map = {"MISSILE_SYSTEM": {
        "SA-75": "1D", "SA-2A": "1D",  # column 0 aliases
        "S-75": "13D", "SA-2C": "13D",  # column 1 aliases
    }}
    facts = [
        # Airframe row (Length mm) for both columns
        TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                  schema_field="body_length_m", value=10.726,
                  source_label="Length mm", section_ctx=None,
                  pass_name="missile_airframe", raw_text="10726"),
        TableFact(canonical_entity="13D", entity_type="MISSILE_SYSTEM",
                  schema_field="body_length_m", value=10.841,
                  source_label="Length mm", section_ctx=None,
                  pass_name="missile_airframe", raw_text="10841"),
        # Propulsion row (1st Stage Weight kg) — these are the
        # acceptance-driving facts.
        TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                  schema_field="booster_mass_kg", value=1135.0,
                  source_label="Weight kg", section_ctx="1st Stage",
                  pass_name="missile_propulsion", raw_text="1135"),
        TableFact(canonical_entity="13D", entity_type="MISSILE_SYSTEM",
                  schema_field="booster_mass_kg", value=1135.0,
                  source_label="Weight kg", section_ctx="1st Stage",
                  pass_name="missile_propulsion", raw_text="1135"),
    ]
    hints = [CrossEntityHint(
        source_canonical="1D", source_entity_type="MISSILE_SYSTEM",
        target_alias="RSNA-75", target_entity_type="RADAR_SYSTEM",
        relationship_kind="associated_with",
    )]
    return WorkerTO(
        alias_map_by_entity_type=alias_map,
        facts=facts,
        cross_entity_hints=hints,
    )


def _build_propulsion_passresult():
    """Stub-LLM propulsion pass: emits 4 instances under different alias
    names, with ONE instance carrying a WRONG booster_mass_kg=970 (the
    empirical failure mode). Overlay must rewrite system_name AND
    override the wrong value."""
    instances = [
        missile_propulsion.MissilePropulsionRecord(
            system_name="SA-75", booster_mass_kg=970.0,  # WRONG
        ),
        missile_propulsion.MissilePropulsionRecord(
            system_name="SA-2A", booster_mass_kg=None,
        ),
        missile_propulsion.MissilePropulsionRecord(
            system_name="S-75", booster_mass_kg=None,
        ),
        missile_propulsion.MissilePropulsionRecord(
            system_name="SA-2C", booster_mass_kg=None,
        ),
    ]
    pr = PassResult.__new__(PassResult)
    pr.pass_name = "missile_propulsion"
    pr.template_instance = None
    pr.metadata = None
    pr.pre_merge_rejections = []
    pr.upstream_refs = None
    pr.pre_merge_walk = None
    pr.provenance = []
    pr.field_evidence = {}
    pr._walker_entities_cache = list(instances)
    pr.table_overlay = _build_overlay_for_sa2()
    return pr, instances


def _build_airframe_passresult():
    instances = [
        missile_airframe.MissileAirframeRecord(
            system_name="SA-75", body_length_m=None,
        ),
        missile_airframe.MissileAirframeRecord(
            system_name="S-75", body_length_m=None,
        ),
    ]
    pr = PassResult.__new__(PassResult)
    pr.pass_name = "missile_airframe"
    pr.template_instance = None
    pr.metadata = None
    pr.pre_merge_rejections = []
    pr.upstream_refs = None
    pr.pre_merge_walk = None
    pr.provenance = []
    pr.field_evidence = {}
    pr._walker_entities_cache = list(instances)
    pr.table_overlay = _build_overlay_for_sa2()
    return pr, instances


def test_end_to_end_sa2_alias_collapse_and_propulsion_override(
    monkeypatch, caplog,
):
    """Mechanism A1 acceptance smoke test:
       - 4 alias instances (SA-75/SA-2A → 1D, S-75/SA-2C → 13D) collapse
         to 2 canonical post-rewrite.
       - Wrong LLM booster_mass_kg=970 on SA-75 is OVERRIDDEN to 1135
         (table fact wins).
       - FIELD_OVERLAY_OVERRIDE log line is emitted for that override.
         IMPORTANT: that log line is emitted from
         app.services.table_overlay.logger (not extraction_merge), so
         we use caplog at INFO level to capture across loggers rather
         than patching one of them — patching extraction_merge.logger
         would miss it.
       - Other instances pick up booster_mass_kg=1135 from null."""
    import logging as _logging
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")
    caplog.set_level(_logging.INFO)

    prop_pr, prop_instances = _build_propulsion_passresult()
    af_pr, af_instances = _build_airframe_passresult()
    pass_results = {
        "missile_propulsion": prop_pr,
        "missile_airframe": af_pr,
    }
    ontology = {"entity_types": [
        {"name": "MISSILE_SYSTEM", "graph_id_fields": ["system_name"]},
    ]}
    manifest = type("M", (), {"passes": [], "bundle_key": "air_defense_v3"})()

    merge_and_resolve(
        pass_results=pass_results, manifest=manifest,
        ontology=ontology,
        document_id="sa2-doc", pipeline_run_id="run-sa2",
    )
    log_messages = [r.getMessage() for r in caplog.records]

    # Alias rewrite happened
    rewritten_names = {inst.system_name for inst in prop_instances}
    assert rewritten_names == {"1D", "13D"}, (
        f"expected alias collapse, got {rewritten_names}"
    )

    # All instances now have booster_mass_kg=1135.0; the LLM=970.0
    # value should have been overridden by the overlay.
    masses = {inst.booster_mass_kg for inst in prop_instances}
    assert masses == {1135.0}, (
        f"expected all instances to carry booster_mass_kg=1135.0, "
        f"got {masses}"
    )

    # FIELD_OVERLAY_OVERRIDE log emitted (from
    # app.services.table_overlay.logger; caplog captures all loggers).
    assert any("FIELD_OVERLAY_OVERRIDE" in m for m in log_messages), (
        "expected FIELD_OVERLAY_OVERRIDE log line for booster_mass_kg override"
    )
    # IDENTITY_REWRITE / TABLE_OVERLAY_APPLIED also emitted (these
    # come from extraction_merge.logger).
    assert any("IDENTITY_REWRITE" in m for m in log_messages)
    assert any("TABLE_OVERLAY_APPLIED" in m for m in log_messages)

    # Airframe instances: body_length_m populated from overlay (was null)
    af_lengths = {inst.body_length_m for inst in af_instances}
    assert af_lengths == {10.726, 10.841}
