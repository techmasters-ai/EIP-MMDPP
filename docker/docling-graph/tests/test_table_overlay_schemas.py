"""Schema tests for spec §5.4 wire types. Uses the existing
`dg_schemas` fixture from conftest.py — that fixture already handles
the importlib swap so docling-graph schemas don't shadow the worker-
side app/ package."""
import pytest


# Tests use the `dg_schemas` fixture parameter from conftest, which
# returns the loaded docling-graph schemas module. Example:
#
#   def test_X(dg_schemas):
#       fact = dg_schemas.TableFact(...)
#
# We keep a local helper for any tests that don't take the fixture.
def _load_schemas(dg_schemas):
    return dg_schemas


def test_table_fact_required_fields(dg_schemas):
    s = dg_schemas
    fact = s.TableFact(
        canonical_entity="1D",
        entity_type="MISSILE_SYSTEM",
        schema_field="booster_mass_kg",
        value=1135.0,
        source_label="Weight kg",
        section_ctx="1st Stage",
        pass_name="missile_propulsion",
        raw_text="1135",
    )
    assert fact.canonical_entity == "1D"
    assert fact.entity_type == "MISSILE_SYSTEM"


def test_table_fact_frozen(dg_schemas):
    s = dg_schemas
    fact = s.TableFact(
        canonical_entity="1D", entity_type="MISSILE_SYSTEM",
        schema_field="booster_mass_kg", value=1135.0,
        source_label="Weight kg", section_ctx=None,
        pass_name="missile_propulsion", raw_text="1135",
    )
    with pytest.raises(Exception):
        fact.value = 9999.0  # frozen=True must reject


def test_cross_entity_hint_required_fields(dg_schemas):
    s = dg_schemas
    hint = s.CrossEntityHint(
        source_canonical="1D",
        source_entity_type="MISSILE_SYSTEM",
        target_alias="RSNA-75",
        target_entity_type="RADAR_SYSTEM",
        relationship_kind="associated_with",
    )
    assert hint.target_entity_type == "RADAR_SYSTEM"


def test_table_overlay_default_factories_independent(dg_schemas):
    """Mutable defaults bug guard. Two TableOverlay instances must NOT
    share the same dict / list objects."""
    s = dg_schemas
    a = s.TableOverlay()
    b = s.TableOverlay()
    a.alias_map_by_entity_type["MISSILE_SYSTEM"] = {"x": "y"}
    a.facts.append("dummy")  # type: ignore[arg-type]
    a.cross_entity_hints.append("dummy")  # type: ignore[arg-type]
    assert b.alias_map_by_entity_type == {}
    assert b.facts == []
    assert b.cross_entity_hints == []


def test_table_overlay_round_trip(dg_schemas):
    s = dg_schemas
    overlay = s.TableOverlay(
        alias_map_by_entity_type={"MISSILE_SYSTEM": {"SA-75": "1D"}},
        facts=[s.TableFact(
            canonical_entity="1D", entity_type="MISSILE_SYSTEM",
            schema_field="booster_mass_kg", value=1135.0,
            source_label="Weight kg", section_ctx="1st Stage",
            pass_name="missile_propulsion", raw_text="1135",
        )],
        cross_entity_hints=[],
    )
    dumped = overlay.model_dump(mode="json")
    restored = s.TableOverlay.model_validate(dumped)
    assert restored.alias_map_by_entity_type == overlay.alias_map_by_entity_type
    assert len(restored.facts) == 1


def test_extract_pass_response_carries_table_overlay_optional(dg_schemas):
    s = dg_schemas
    # Without overlay
    resp = s.ExtractPassResponse(bundle_key="x", pass_name="y", pass_output={})
    assert resp.table_overlay is None
    # With overlay
    resp2 = s.ExtractPassResponse(
        bundle_key="x", pass_name="y", pass_output={},
        table_overlay=s.TableOverlay(),
    )
    assert resp2.table_overlay is not None
