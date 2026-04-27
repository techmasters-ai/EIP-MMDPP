"""Tests for LLM-facing schema sanitization used by prompt_rules."""

import importlib.util
from pathlib import Path

from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass


_PROMPT_RULES_PATH = Path(__file__).resolve().parent.parent / "app" / "prompt_rules.py"


def _load_prompt_rules():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_prompt_rules",
        _PROMPT_RULES_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _radar_props(schema):
    return schema["$defs"]["RadarSystemEntity"]["properties"]


def test_numeric_examples_are_removed_from_llm_schema():
    prompt_rules = _load_prompt_rules()
    original = RadarDomainPass.model_json_schema()

    assert _radar_props(original)["gain_dbi"]["examples"] == [38.0, 42.0]
    assert _radar_props(original)["scan_type"]["examples"] == [
        "CIRCULAR",
        "ELECTRONIC",
        "DWELL_AND_SWITCH",
    ]

    sanitized = prompt_rules.sanitize_schema_for_llm(original)

    assert "examples" not in _radar_props(sanitized)["gain_dbi"]
    assert "examples" not in _radar_props(sanitized)["nominal_rf_mhz"]
    assert _radar_props(sanitized)["scan_type"]["examples"] == [
        "CIRCULAR",
        "ELECTRONIC",
        "DWELL_AND_SWITCH",
    ]

    # The source schema is not mutated; validation and downstream model
    # metadata retain their original examples.
    assert _radar_props(original)["gain_dbi"]["examples"] == [38.0, 42.0]


def test_numeric_typical_ranges_are_removed_from_llm_descriptions():
    prompt_rules = _load_prompt_rules()
    sanitized = prompt_rules.sanitize_schema_for_llm(RadarDomainPass.model_json_schema())

    gain_description = _radar_props(sanitized)["gain_dbi"]["description"]
    rf_description = _radar_props(sanitized)["nominal_rf_mhz"]["description"]

    assert "Typical dish" not in gain_description
    assert "Phased arrays" not in gain_description
    assert "Common radar bands" not in rf_description
    assert "If the source gives GHz, multiply by 1000" in rf_description
    assert "mechanically converted value from an explicit source value" in gain_description


def test_long_forbidden_identity_lists_are_compressed_for_llm_schema():
    prompt_rules = _load_prompt_rules()
    original = RadarDomainPass.model_json_schema()
    sanitized = prompt_rules.sanitize_schema_for_llm(original)

    original_description = _radar_props(original)["system_name"]["description"]
    sanitized_description = _radar_props(sanitized)["system_name"]["description"]

    assert "SA-23" in original_description
    assert "MiG-29" in original_description
    assert len(sanitized_description) < len(original_description)
    assert "weapon or missile systems" in sanitized_description
    assert "Reject descriptive phrases" in sanitized_description
