import importlib.util
from pathlib import Path
from types import SimpleNamespace

from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import MissileDomainPass
from ontology_bundles.air_defense_v3.extraction_schemas.system_links import SystemLinksPass

_MODULE_PATH = Path(__file__).resolve().parent.parent / "app" / "evidence_gate.py"
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
        "radar_domain",
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
        "missile_domain",
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
