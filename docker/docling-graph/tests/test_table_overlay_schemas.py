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


def _load_worker_table_overlay_module():
    """Load the worker-side ``app/services/table_overlay.py`` directly via
    importlib.spec_from_file_location, bypassing whatever ``app`` package
    happens to be on sys.path at the moment.

    Why: another test file in this directory
    (``test_table_facts_resolve.py``) prepends the docling-graph service
    root to ``sys.path`` at module import time and never restores it. By
    the time our drift-guard test body runs, a plain ``from
    app.services.table_overlay import ...`` may resolve ``app`` to the
    docling-graph package (which has no ``services`` submodule) instead
    of the worker-side repo-root ``app`` package. We sidestep the
    ambiguity by loading the file at its on-disk path under a stable
    private module name.
    """
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    worker_overlay_path = repo_root / "app" / "services" / "table_overlay.py"
    module_name = "_drift_guard_worker_table_overlay"
    spec = importlib.util.spec_from_file_location(module_name, worker_overlay_path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so @dataclass and
    # other tooling that walks cls.__module__ can resolve the namespace.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parser_and_worker_table_overlay_classes_round_trip(dg_schemas):
    """The parser-side TableOverlay (loaded via dg_schemas fixture from
    docker/docling-graph/app/schemas.py) and the worker-side
    TableOverlay (in app/services/table_overlay.py) declare
    structurally identical Pydantic models. JSON round-trip between
    them must equal element-wise. This test guards against drift: if
    a field is added on one side and not the other, this test fails."""
    # Worker side (loaded via importlib to dodge sys.path pollution
    # from sibling test files — see _load_worker_table_overlay_module).
    worker_mod = _load_worker_table_overlay_module()
    WorkerTO = worker_mod.TableOverlay
    WorkerTF = worker_mod.TableFact

    # Build a parser-side overlay
    parser_ov = dg_schemas.TableOverlay(
        alias_map_by_entity_type={"MISSILE_SYSTEM": {"SA-75": "1D"}},
        facts=[dg_schemas.TableFact(
            canonical_entity="1D", entity_type="MISSILE_SYSTEM",
            schema_field="booster_mass_kg", value=1135.0,
            source_label="Weight kg", section_ctx="1st Stage",
            pass_name="missile_propulsion", raw_text="1135",
        )],
        cross_entity_hints=[],
    )

    # JSON round-trip into worker side
    dumped = parser_ov.model_dump(mode="json")
    worker_ov = WorkerTO.model_validate(dumped)

    assert worker_ov.alias_map_by_entity_type == parser_ov.alias_map_by_entity_type
    assert len(worker_ov.facts) == 1
    assert worker_ov.facts[0].canonical_entity == "1D"
    assert worker_ov.facts[0].value == 1135.0

    # Schema-shape equivalence: same field names.
    parser_fields = set(dg_schemas.TableOverlay.model_fields.keys())
    worker_fields = set(WorkerTO.model_fields.keys())
    assert parser_fields == worker_fields, (
        f"TableOverlay field drift: parser={parser_fields} worker={worker_fields}"
    )
    parser_fact_fields = set(dg_schemas.TableFact.model_fields.keys())
    worker_fact_fields = set(WorkerTF.model_fields.keys())
    assert parser_fact_fields == worker_fact_fields, (
        f"TableFact field drift: parser={parser_fact_fields} worker={worker_fact_fields}"
    )
