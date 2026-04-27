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


def test_radar_postprocess_clears_unsupported_specs_and_recovers_spoon_rest():
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


def test_apply_bundle_postprocessing_derives_sa2_system_links_from_evidence():
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
