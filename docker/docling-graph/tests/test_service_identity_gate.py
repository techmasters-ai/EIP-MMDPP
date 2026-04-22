import importlib.util
from pathlib import Path

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
