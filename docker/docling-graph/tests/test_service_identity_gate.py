import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

# Legacy-loadability check: radar_domain.py is kept in source even
# after the field-group cutover. If this import breaks, that's a
# regression even though the legacy module isn't in the manifest.
from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass
# Legacy-loadability check: missile_domain.py is kept in source even
# after the field-group cutover. If this import breaks, that's a
# regression even though the legacy module isn't in the manifest.
from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import MissileDomainPass
from ontology_bundles.air_defense_v3.extraction_schemas.system_links import SystemLinksPass

_SERVICE_APP_ROOT = Path(__file__).resolve().parent.parent / "app"

# Pre-register the docling-graph-side `app._numeric_evidence` so the file-path
# loaded `evidence_gate.py` can resolve `from app._numeric_evidence import ...`
# without falling back to the (separate) root `app/` package.
_NUM_EV_SPEC = importlib.util.spec_from_file_location(
    "app._numeric_evidence", _SERVICE_APP_ROOT / "_numeric_evidence.py"
)
_NUM_EV_MOD = importlib.util.module_from_spec(_NUM_EV_SPEC)
sys.modules["app._numeric_evidence"] = _NUM_EV_MOD
assert _NUM_EV_SPEC.loader is not None
_NUM_EV_SPEC.loader.exec_module(_NUM_EV_MOD)

_MODULE_PATH = _SERVICE_APP_ROOT / "evidence_gate.py"
_SPEC = importlib.util.spec_from_file_location("docling_graph_evidence_gate", _MODULE_PATH)
_EVIDENCE_GATE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_EVIDENCE_GATE)


def test_filter_pass_output_by_batch_text_drops_prompt_borrowed_missiles():
    evidence_text = _EVIDENCE_GATE.collect_batch_evidence_text({
        "texts": [
            {
                "text": (
                    "SA-2 Surface-to-Air Missile. The missile was better known by the "
                    "NATO designation SA-2 Guideline."
                )
            }
        ]
    })
    pass_output = {
        "missile_systems": [
            {"system_name": "SA-2"},
            {"system_name": "Patriot"},
            {"system_name": "PAC-3"},
        ]
    }

    filtered, stats, _allowed = _EVIDENCE_GATE.filter_pass_output_by_batch_text(
        pass_output,
        MissileDomainPass,
        evidence_text,
    )

    assert [row["system_name"] for row in filtered["missile_systems"]] == ["SA-2"]
    assert stats["dropped_entities_by_field"]["missile_systems"] == 2
    assert stats["dropped_entity_examples"]["missile_systems"] == ["Patriot", "PAC-3"]


def test_system_links_schema_drops_incomplete_relationship_rows():
    model = SystemLinksPass.model_validate({
        "relationships": [
            {"rel_type": "CUES", "from_ref_id": "E001", "to_ref_id": "E002"},
            {"rel_type": None, "from_ref_id": "E001", "to_ref_id": "E003"},
            {"rel_type": "ASSOCIATED_WITH", "from_ref_id": "E002", "to_ref_id": None},
            {"rel_type": "CUES", "from_ref_id": "E004", "to_ref_id": "E004"},
        ]
    })

    assert len(model.relationships) == 1
    assert model.relationships[0].rel_type == "CUES"
    assert model.relationships[0].from_ref_id == "E001"
    assert model.relationships[0].to_ref_id == "E002"


def test_system_links_strips_surrounding_brackets_from_ref_ids():
    """v8a: the LLM has been observed to wrap ref_ids in '[…]' (copied
    from the prompt preamble). The defensive strip should peel exactly
    one surrounding pair so the merge layer sees the clean ref_id."""
    model = SystemLinksPass.model_validate({
        "relationships": [
            {"rel_type": "ASSOCIATED_WITH",
             "from_ref_id": "[RADAR_SYSTEM:Fan Song]",
             "to_ref_id": "[MISSILE_SYSTEM:S-75]"},
        ]
    })
    assert len(model.relationships) == 1
    assert model.relationships[0].from_ref_id == "RADAR_SYSTEM:Fan Song"
    assert model.relationships[0].to_ref_id == "MISSILE_SYSTEM:S-75"


def test_system_links_leaves_bare_ref_ids_alone():
    """Bracket strip must be idempotent: a clean ref_id passes through
    unchanged."""
    model = SystemLinksPass.model_validate({
        "relationships": [
            {"rel_type": "CUES",
             "from_ref_id": "RADAR_SYSTEM:Spoon Rest",
             "to_ref_id": "RADAR_SYSTEM:Fan Song"},
        ]
    })
    assert model.relationships[0].from_ref_id == "RADAR_SYSTEM:Spoon Rest"
    assert model.relationships[0].to_ref_id == "RADAR_SYSTEM:Fan Song"


def test_system_links_dedups_pairs_with_same_from_to():
    """v8a: when the LLM emits two rows with the same (from_ref_id,
    to_ref_id) but different rel_types or confidences, keep only the
    first. Avoids spurious duplicate edge volume."""
    model = SystemLinksPass.model_validate({
        "relationships": [
            {"rel_type": "ASSOCIATED_WITH", "from_ref_id": "A", "to_ref_id": "B", "confidence": 0.9},
            {"rel_type": "CUES", "from_ref_id": "A", "to_ref_id": "B", "confidence": 0.8},
            {"rel_type": "ASSOCIATED_WITH", "from_ref_id": "C", "to_ref_id": "B", "confidence": 1.0},
        ]
    })
    assert len(model.relationships) == 2
    pairs = {(r.from_ref_id, r.to_ref_id) for r in model.relationships}
    assert pairs == {("A", "B"), ("C", "B")}


def test_system_links_does_not_strip_internal_brackets():
    """Only ONE surrounding pair is stripped; internal '[' or ']' in a
    weird ref_id stays put."""
    model = SystemLinksPass.model_validate({
        "relationships": [
            {"rel_type": "CUES",
             "from_ref_id": "[[NESTED:weird]]",
             "to_ref_id": "B"},
        ]
    })
    assert model.relationships[0].from_ref_id == "[NESTED:weird]"


def test_summarize_pass_output_matches_filtered_entity_counts():
    summary = _EVIDENCE_GATE.summarize_pass_output(
        {"missile_systems": [{"system_name": "SA-2"}]},
        MissileDomainPass,
    )

    assert summary["node_count"] == 2
    assert summary["edge_count"] == 1
    assert summary["node_types"] == {"MissileDomainPass": 1, "MissileSystemEntity": 1}
    assert summary["edge_types"] == {"CONTAINS": 1}
    assert summary["path_counts"] == {"": 1, "missile_systems[]": 1}


def test_summarize_pass_output_matches_filtered_radar_counts():
    summary = _EVIDENCE_GATE.summarize_pass_output(
        {"radar_systems": [{"system_name": "Fan Song"}, {"system_name": "Spoon Rest"}]},
        RadarDomainPass,
    )

    assert summary["node_count"] == 3
    assert summary["edge_count"] == 2
    assert summary["node_types"] == {"RadarDomainPass": 1, "RadarSystemEntity": 2}
    assert summary["edge_types"] == {"CONTAINS": 2}
    assert summary["path_counts"] == {"": 1, "radar_systems[]": 2}


def test_summarize_pass_output_counts_system_links_dto_relationships():
    """SystemLinkRelationship is is_entity=False and uses plain Field (not
    edge()), so the entity-path counter ignores it. The DTO-path counter
    must still tally each row's rel_type into edge_count / edge_types,
    otherwise metadata under-reports system_links output as 0 edges."""
    summary = _EVIDENCE_GATE.summarize_pass_output(
        {
            "relationships": [
                {"rel_type": "CUES", "from_ref_id": "E001", "to_ref_id": "E002", "confidence": 0.9},
                {"rel_type": "ASSOCIATED_WITH", "from_ref_id": "E002", "to_ref_id": "E003", "confidence": 0.95},
                {"rel_type": "", "from_ref_id": "E003", "to_ref_id": "E004"},
                {"from_ref_id": "E004", "to_ref_id": "E005"},
            ]
        },
        SystemLinksPass,
    )

    assert summary["node_count"] == 1  # just the pass root
    assert summary["edge_count"] == 2  # two rows with non-empty rel_type
    assert summary["node_types"] == {"SystemLinksPass": 1}
    assert summary["edge_types"] == {"CUES": 1, "ASSOCIATED_WITH": 1}
    assert summary["path_counts"] == {"": 1, "relationships[]": 4}


def test_apply_bundle_postprocessing_rewrites_air_defense_fields_from_evidence():
    evidence_text = _EVIDENCE_GATE.normalize_evidence_text(
        """
        The Spoon Rest radar detected incoming aircraft at long range.
        The Fan Song guidance radar performed two functions: target acquisition and missile guidance.
        TECHNICAL NOTES:
        Range: Minimum 5 miles; maximum effective range about 19 miles; maximum slant range 27 miles
        Ceiling: Up to 60,000 ft.
        """
    )

    radar_output, radar_stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "radar_identity",
        {
            "radar_systems": [
                {"system_name": "Fan Song", "emitter_function": "TRACKING", "system_status": "OPERATIONAL"},
                {"system_name": "Spoon Rest", "emitter_function": None, "system_status": "OPERATIONAL"},
            ]
        },
        evidence_text,
    )
    missile_output, missile_stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "missile_identity",
        {
            "missile_systems": [
                {"system_name": "SA-2", "min_intercept_km": 5.0, "max_intercept_km": 19.0, "system_status": "OPERATIONAL"}
            ]
        },
        evidence_text,
    )

    assert radar_output["radar_systems"][0]["emitter_function"] == "FIRE_CONTROL"
    assert radar_output["radar_systems"][1]["emitter_function"] == "SEARCH"
    assert radar_output["radar_systems"][0]["system_status"] is None
    assert radar_output["radar_systems"][1]["system_status"] is None
    assert radar_stats["emitter_function_overrides"] == {"Fan Song": "FIRE_CONTROL", "Spoon Rest": "SEARCH"}
    assert radar_stats["status_cleared"] == ["Fan Song", "Spoon Rest"]

    missile = missile_output["missile_systems"][0]
    assert missile["min_intercept_km"] == 8.0
    assert missile["max_intercept_km"] == 30.6
    assert missile["max_altitude_km"] == 18.3
    assert missile["system_status"] is None
    assert missile_stats["range_overrides"] == {
        "min_intercept_km": 8.0,
        "max_intercept_km": 30.6,
        "max_altitude_km": 18.3,
    }
    assert missile_stats["status_cleared"] == ["SA-2"]


def test_radar_postprocess_clears_unsupported_specs_and_recovers_spoon_rest(monkeypatch):
    """Documents the QUARANTINED SA-2 radar-name recall behavior
    (`_RADAR_RECALL_PATTERNS` / `_RADAR_NOMENCLATURE_PATTERNS`). Only
    fires when the `DOCLING_GRAPH_LEGACY_SA2_FALLBACKS` env flag is on.
    The default (flag off) is covered by the unsupported-spec clearing
    assertions below — Spoon Rest recovery is the legacy-only part."""
    monkeypatch.setattr(_EVIDENCE_GATE, "_LEGACY_SA2_FALLBACKS_ENABLED", True)
    evidence_text = _EVIDENCE_GATE.normalize_evidence_text(
        """
        RSNA-75/SNR-75 Fan Song Engagement Radar
        The Fan Song is the engagement radar for the S-75/SA-2 family of SAMs.
        The SNR-75 family of radars employ a complex antenna arrangement.

        S-75M Battery Components
        SNR-75 PV Cabin / Fan Song 1 Radar head van
        P-12M/P-18 Spoon Rest 1 Acquisition Radar

        NITEL P-18-2/P-18M Spoon Rest D/E Acquisition Radar
        """
    )

    radar_output, radar_stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "radar_identity",
        {
            "radar_systems": [
                {
                    "system_name": "Fan Song",
                    "nomenclature": "SNR-75",
                    "emitter_function": "FIRE_CONTROL",
                    "system_status": "OPERATIONAL",
                    "responsible_agency": "IWC",
                    "review_cycle": "annual",
                    "next_review_date": "2026-06-30",
                    "erp_dbw": 72.0,
                    "tx_peak_power_kw": 150.0,
                    "gain_dbi": 38.0,
                    "antenna_dim_az_m": 4.5,
                    "antenna_dim_el_m": 2.5,
                    "beamwidth_az_deg": 1.5,
                    "beamwidth_el_deg": 15.0,
                    "spoiled": False,
                    "coverage_limits_el_deg": 45.0,
                    "nominal_rf_mhz": 3000.0,
                    "nominal_pri_usec": 1000.0,
                    "nominal_pd_usec": 0.5,
                    "scan_type": "ELECTRONIC",
                    "scan_period_sec": 4.0,
                    "intra_pulse_mop": "LFM_CHIRP",
                    "frequency_excursion_mhz": 10.0,
                    "inter_pulse": "CONSTANT_PRI",
                    "pulses_per_dwell": 16,
                    "confidence": 0.9,
                }
            ]
        },
        evidence_text,
    )

    rows = {row["system_name"]: row for row in radar_output["radar_systems"]}
    assert set(rows) == {"Fan Song", "Spoon Rest"}

    fan_song = rows["Fan Song"]
    assert fan_song["nomenclature"] == "SNR-75"
    assert fan_song["emitter_function"] == "FIRE_CONTROL"
    assert fan_song["system_status"] is None
    assert fan_song["responsible_agency"] is None
    assert fan_song["review_cycle"] is None
    assert fan_song["next_review_date"] is None
    assert fan_song["erp_dbw"] is None
    assert fan_song["tx_peak_power_kw"] is None
    assert fan_song["gain_dbi"] is None
    assert fan_song["scan_type"] is None
    assert fan_song["confidence"] is None

    spoon_rest = rows["Spoon Rest"]
    assert spoon_rest["system_name"] == "Spoon Rest"
    assert spoon_rest["emitter_function"] == "SEARCH"
    assert spoon_rest["nomenclature"] == "P-18-2/P-18M"

    assert radar_stats["status_cleared"] == ["Fan Song"]
    assert radar_stats["recalled_radars"] == ["Spoon Rest"]
    assert radar_stats["unsupported_properties_cleared"]["Fan Song"] == [
        "antenna_dim_az_m",
        "antenna_dim_el_m",
        "beamwidth_az_deg",
        "beamwidth_el_deg",
        "confidence",
        "coverage_limits_el_deg",
        "erp_dbw",
        "frequency_excursion_mhz",
        "gain_dbi",
        "inter_pulse",
        "intra_pulse_mop",
        "next_review_date",
        "nominal_pd_usec",
        "nominal_pri_usec",
        "nominal_rf_mhz",
        "pulses_per_dwell",
        "responsible_agency",
        "review_cycle",
        "scan_period_sec",
        "scan_type",
        "spoiled",
        "tx_peak_power_kw",
    ]


def test_missile_postprocess_clears_recurring_sa2_hallucinated_properties():
    evidence_text = _EVIDENCE_GATE.normalize_evidence_text(
        """
        Developed in the mid-1950s, the V-750 Dvina was the first effective Soviet surface-to-air missile.
        The missile was better known by the NATO designation SA-2 Guideline.
        TECHNICAL NOTES:
        Range: Minimum 5 miles; maximum effective range about 19 miles; maximum slant range 27 miles
        Ceiling: Up to 60,000 ft.
        Speed: Mach 3.5
        Weight: 4,850 lbs.
        """
    )

    missile_output, missile_stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "missile_identity",
        {
            "missile_systems": [
                {
                    "system_name": "SA-2",
                    "nomenclature": "V-750 Dvina",
                    "guidance_type": "COMMAND",
                    "body_length_m": 7.5,
                    "body_diameter_m": 0.5,
                    "total_mass_kg": 2320.0,
                    "average_speed_mps": 1052.0,
                    "max_speed_mps": 1052.0,
                    "confidence": 0.9,
                }
            ]
        },
        evidence_text,
    )

    missile = missile_output["missile_systems"][0]
    assert missile["nomenclature"] == "V-750 Dvina"
    assert missile["guidance_type"] is None
    assert missile["body_length_m"] is None
    assert missile["body_diameter_m"] is None
    assert missile["total_mass_kg"] == 2199.5
    assert missile["average_speed_mps"] is None
    assert missile["max_speed_mps"] is None
    assert missile["confidence"] is None
    assert missile_stats["unsupported_properties_cleared"] == {
        "SA-2": [
            "average_speed_mps",
            "body_diameter_m",
            "body_length_m",
            "confidence",
            "guidance_type",
            "max_speed_mps",
        ]
    }
    assert missile_stats["range_overrides"] == {
        "min_intercept_km": 8.0,
        "max_intercept_km": 30.6,
        "max_altitude_km": 18.3,
        "total_mass_kg": 2199.5,
    }


def test_apply_bundle_postprocessing_derives_sa2_system_links_from_evidence(monkeypatch):
    """Documents the QUARANTINED SA-2 legacy fallback behavior. Only
    fires when the `DOCLING_GRAPH_LEGACY_SA2_FALLBACKS` env flag is on.
    The default (flag off) is exercised by
    `test_legacy_sa2_fallback_off_by_default` below."""
    monkeypatch.setattr(_EVIDENCE_GATE, "_LEGACY_SA2_FALLBACKS_ENABLED", True)
    evidence_text = _EVIDENCE_GATE.normalize_evidence_text(
        """
        A typical SA-2 site in North Vietnam had six missiles on launchers,
        a Spoon Rest acquisition radar, and a Fan Song guidance radar.
        The Fan Song guidance radar performed two functions: target acquisition and missile guidance.
        After launch, it guided up to three SA-2s against one target.
        """
    )
    upstream_entities = [
        SimpleNamespace(ref_id="E001", identity_values={"system_name": "Fan Song"}, display_label=None),
        SimpleNamespace(ref_id="E002", identity_values={"system_name": "Spoon Rest"}, display_label=None),
        SimpleNamespace(ref_id="E003", identity_values={"system_name": "SA-2"}, display_label=None),
    ]

    pass_output, stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "system_links",
        {"relationships": []},
        evidence_text,
        upstream_entities,
    )

    assert pass_output["relationships"] == [
        {"rel_type": "CUES", "from_ref_id": "E002", "to_ref_id": "E001", "confidence": 0.95},
        {"rel_type": "ASSOCIATED_WITH", "from_ref_id": "E001", "to_ref_id": "E003", "confidence": 0.95},
    ]
    assert stats["derived_relationships"] == pass_output["relationships"]


def test_legacy_sa2_fallback_off_by_default(monkeypatch):
    """The default (flag off) must NOT synthesize legacy SA-2 edges.
    With zero LLM-emitted edges and no hints/upstream resolution, the
    output stays empty — non-SA-2 docs aren't polluted by SA-2-specific
    relationship synthesis."""
    monkeypatch.setattr(_EVIDENCE_GATE, "_LEGACY_SA2_FALLBACKS_ENABLED", False)
    evidence_text = _EVIDENCE_GATE.normalize_evidence_text(
        "A Spoon Rest acquisition radar and a Fan Song guidance radar."
    )
    upstream_entities = [
        SimpleNamespace(ref_id="E001", identity_values={"system_name": "Fan Song"}, display_label=None),
        SimpleNamespace(ref_id="E002", identity_values={"system_name": "Spoon Rest"}, display_label=None),
        SimpleNamespace(ref_id="E003", identity_values={"system_name": "SA-2"}, display_label=None),
    ]
    pass_output, stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "system_links",
        {"relationships": []},
        evidence_text,
        upstream_entities,
    )
    assert pass_output["relationships"] == []
    assert "derived_relationships" not in stats


# --- v8a cross_entity_hints promotion tests ---------------------------------

def test_apply_bundle_postprocessing_promotes_cross_entity_hints():
    """v8a: cross_entity_hints from the table overlay should be promoted
    into ASSOCIATED_WITH edges, resolving source/target by name against
    the upstream catalog. 2026-05-16: resolution is type-segregated, so
    each upstream entity must declare its `entity_type`."""
    upstream_entities = [
        SimpleNamespace(ref_id="RADAR_SYSTEM:Fan Song", entity_type="RADAR_SYSTEM", identity_values={"system_name": "Fan Song"}, display_label="Fan Song"),
        SimpleNamespace(ref_id="RADAR_SYSTEM:RSN-75", entity_type="RADAR_SYSTEM", identity_values={"system_name": "RSN-75"}, display_label="RSN-75"),
        SimpleNamespace(ref_id="MISSILE_SYSTEM:1D", entity_type="MISSILE_SYSTEM", identity_values={"system_name": "1D"}, display_label="1D"),
        SimpleNamespace(ref_id="MISSILE_SYSTEM:13D", entity_type="MISSILE_SYSTEM", identity_values={"system_name": "13D"}, display_label="13D"),
    ]
    hints = [
        SimpleNamespace(source_canonical="1D", source_entity_type="MISSILE_SYSTEM",
                        target_alias="Fan Song", target_entity_type="RADAR_SYSTEM",
                        relationship_kind="associated_with"),
        SimpleNamespace(source_canonical="13D", source_entity_type="MISSILE_SYSTEM",
                        target_alias="RSN-75", target_entity_type="RADAR_SYSTEM",
                        relationship_kind="associated_with"),
    ]

    pass_output, stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "system_links",
        {"relationships": []},
        "",
        upstream_entities,
        hints,
    )

    rels = pass_output["relationships"]
    pair_set = {(r["from_ref_id"], r["to_ref_id"]) for r in rels}
    assert ("MISSILE_SYSTEM:1D", "RADAR_SYSTEM:Fan Song") in pair_set
    assert ("MISSILE_SYSTEM:13D", "RADAR_SYSTEM:RSN-75") in pair_set
    assert "promoted_from_cross_entity_hints" in stats
    assert len(stats["promoted_from_cross_entity_hints"]) == 2


def test_apply_bundle_postprocessing_skips_hints_with_unknown_endpoints():
    """Hints whose source or target name doesn't match any upstream ref
    are silently dropped (no malformed edges with unknown refs)."""
    upstream_entities = [
        SimpleNamespace(ref_id="RADAR_SYSTEM:Fan Song", entity_type="RADAR_SYSTEM", identity_values={"system_name": "Fan Song"}, display_label="Fan Song"),
    ]
    hints = [
        SimpleNamespace(source_canonical="UNKNOWN_MISSILE", source_entity_type="MISSILE_SYSTEM",
                        target_alias="Fan Song", target_entity_type="RADAR_SYSTEM",
                        relationship_kind="associated_with"),
        SimpleNamespace(source_canonical="Fan Song", source_entity_type="RADAR_SYSTEM",
                        target_alias="UNKNOWN_TARGET", target_entity_type="MISSILE_SYSTEM",
                        relationship_kind="associated_with"),
    ]

    pass_output, stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "system_links",
        {"relationships": []},
        "",
        upstream_entities,
        hints,
    )

    assert pass_output["relationships"] == []
    assert "promoted_from_cross_entity_hints" not in stats


def test_apply_bundle_postprocessing_preserves_llm_relationships_when_promoting_hints():
    """LLM-emitted relationships are kept; hints are appended on top."""
    upstream_entities = [
        SimpleNamespace(ref_id="RADAR_SYSTEM:Fan Song", entity_type="RADAR_SYSTEM", identity_values={"system_name": "Fan Song"}, display_label="Fan Song"),
        SimpleNamespace(ref_id="MISSILE_SYSTEM:1D", entity_type="MISSILE_SYSTEM", identity_values={"system_name": "1D"}, display_label="1D"),
        SimpleNamespace(ref_id="RADAR_SYSTEM:Spoon Rest", entity_type="RADAR_SYSTEM", identity_values={"system_name": "Spoon Rest"}, display_label="Spoon Rest"),
    ]
    hints = [
        SimpleNamespace(source_canonical="1D", source_entity_type="MISSILE_SYSTEM",
                        target_alias="Fan Song", target_entity_type="RADAR_SYSTEM",
                        relationship_kind="associated_with"),
    ]
    llm_emitted = {
        "relationships": [
            {"rel_type": "CUES", "from_ref_id": "RADAR_SYSTEM:Spoon Rest",
             "to_ref_id": "RADAR_SYSTEM:Fan Song", "confidence": 0.9},
        ],
    }

    pass_output, stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "system_links",
        llm_emitted,
        "",
        upstream_entities,
        hints,
    )

    assert len(pass_output["relationships"]) == 2
    pair_set = {(r["from_ref_id"], r["to_ref_id"]) for r in pass_output["relationships"]}
    assert ("RADAR_SYSTEM:Spoon Rest", "RADAR_SYSTEM:Fan Song") in pair_set
    assert ("MISSILE_SYSTEM:1D", "RADAR_SYSTEM:Fan Song") in pair_set


def test_apply_bundle_postprocessing_dedupes_when_hint_matches_llm_edge():
    """If a hint resolves to the same (from, to) as an LLM-emitted edge,
    the LLM edge is kept (first-seen wins) and the hint is dropped."""
    upstream_entities = [
        SimpleNamespace(ref_id="RADAR_SYSTEM:Fan Song", entity_type="RADAR_SYSTEM", identity_values={"system_name": "Fan Song"}, display_label="Fan Song"),
        SimpleNamespace(ref_id="MISSILE_SYSTEM:1D", entity_type="MISSILE_SYSTEM", identity_values={"system_name": "1D"}, display_label="1D"),
    ]
    hints = [
        SimpleNamespace(source_canonical="1D", source_entity_type="MISSILE_SYSTEM",
                        target_alias="Fan Song", target_entity_type="RADAR_SYSTEM",
                        relationship_kind="associated_with"),
    ]
    llm_emitted = {
        "relationships": [
            {"rel_type": "CUES", "from_ref_id": "MISSILE_SYSTEM:1D",
             "to_ref_id": "RADAR_SYSTEM:Fan Song", "confidence": 0.85},
        ],
    }

    pass_output, _stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "system_links",
        llm_emitted,
        "",
        upstream_entities,
        hints,
    )

    # Only one edge for that pair, and it's the LLM's CUES edge (kept first).
    assert len(pass_output["relationships"]) == 1
    assert pass_output["relationships"][0]["rel_type"] == "CUES"


# --- #83 Tier A: separator-tolerant identity gate -------------------------
#
# The post-extraction identity gate substring-matched the LLM's emitted
# identity against the batch evidence text. At T>0 gemma4 emits real
# entities with a different separator surface-form than the document —
# e.g. it writes "SA-2 C" (space) while the doc has "SA-2C" (no space) —
# and the literal match dropped them. Verified on the 2026-06-29 fresh
# ingest: 17/19 distinct dropped missile names were recoverable real
# entities, only 2 genuine hallucinations. Tier A tolerates separator/
# spacing/case differences while preserving alphanumeric word boundaries
# (no over-match) and NOT bridging list commas (no new false positives).

def test_identity_gate_recovers_spacing_variant():
    """The dominant failure: LLM 'SA-2 C' vs doc 'SA-2C' must now match."""
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "The SA-2C variant was deployed widely."}]}
    )
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("SA-2 C", evidence) is True


def test_identity_gate_recovers_hyphen_and_slash_variants():
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "Operators fielded the S-75 / SA-2 Guideline system."}]}
    )
    # emitted without the spaces around the slash
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("S-75/SA-2 Guideline", evidence) is True


def test_identity_gate_exact_form_still_matches():
    """Regression: forms that already matched must keep matching."""
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "The SA-2 C was an export variant."}]}
    )
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("SA-2 C", evidence) is True


def test_identity_gate_rejects_absent_hallucination():
    """A name absent from the batch text (hallucination) is still dropped."""
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "The SA-2C and SA-2B variants are described here."}]}
    )
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("HQ-2P", evidence) is False


def test_identity_gate_boundary_no_overmatch():
    """Flexible separators must not let a shorter id match inside a longer
    alphanumeric token: 'S-75' must NOT match 'SA-75'."""
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "The SA-75 Dvina export designation."}]}
    )
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("S-75", evidence) is False


def test_identity_gate_separatorless_identity_unchanged():
    """Identities with no separators behave exactly as before (bounded)."""
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "Designation 5YA23 appears in the table."}]}
    )
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("5Ya23", evidence) is True
    glued = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "token X5YA23Y not a standalone id."}]}
    )
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("5Ya23", glued) is False


def test_identity_gate_does_not_bridge_comma_list():
    """A comma is a list delimiter, not a flexible separator: 'SA-2 C'
    must NOT match across 'SA-2, C-300' (two distinct list items)."""
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "Inventory included SA-2, C-300, and others."}]}
    )
    assert _EVIDENCE_GATE.identity_is_supported_by_batch_text("SA-2 C", evidence) is False


def test_filter_keeps_recoverable_variant_drops_hallucination():
    """End-to-end through the filter: the spacing variant survives, the
    hallucination is dropped."""
    evidence = _EVIDENCE_GATE.collect_batch_evidence_text(
        {"texts": [{"text": "The SA-2C and SA-2B export variants are documented."}]}
    )
    filtered, stats, _allowed = _EVIDENCE_GATE.filter_pass_output_by_batch_text(
        {"missile_systems": [{"system_name": "SA-2 C"}, {"system_name": "HQ-2P"}]},
        MissileDomainPass,
        evidence,
    )
    kept = [row["system_name"] for row in filtered["missile_systems"]]
    assert kept == ["SA-2 C"]
    assert stats["dropped_entity_examples"]["missile_systems"] == ["HQ-2P"]


def test_status_is_not_inferred_from_operation_or_in_use_language():
    evidence_text = _EVIDENCE_GATE.normalize_evidence_text(
        """
        The Soviets began exporting it to many countries worldwide in 1960,
        with many remaining in use into the 21st century.
        North Vietnam began receiving SA-2s shortly after the start of
        Operation Rolling Thunder in the spring of 1965.
        """
    )

    missile_output, missile_stats = _EVIDENCE_GATE.apply_bundle_postprocessing(
        "air_defense_v3",
        "missile_identity",
        {
            "missile_systems": [
                {"system_name": "SA-2", "system_status": "OPERATIONAL"}
            ]
        },
        evidence_text,
    )

    assert missile_output["missile_systems"][0]["system_status"] is None
    assert missile_stats["status_cleared"] == ["SA-2"]
