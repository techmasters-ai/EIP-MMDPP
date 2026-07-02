"""Unit tests for query profile schemas and ontology caching."""

import pytest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from app.schemas.query_profiles import (
    QueryProfileStep,
    QueryProfileTraversal,
    QueryProfileDefinition,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


class TestQueryProfileStep:
    """Validate QueryProfileStep field constraints."""

    def test_requires_at_least_one_rel_type(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=[])

    def test_rejects_empty_strings_in_rel_types(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["", "   "])

    def test_strips_whitespace_from_rel_types(self):
        step = QueryProfileStep(rel_types=["  HAS_PART  ", "CONTAINS"])
        assert step.rel_types == ["HAS_PART", "CONTAINS"]

    def test_default_direction_is_out(self):
        step = QueryProfileStep(rel_types=["REL"])
        assert step.direction == "out"

    def test_direction_in_accepted(self):
        step = QueryProfileStep(rel_types=["REL"], direction="in")
        assert step.direction == "in"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["REL"], direction="both")

    def test_default_hops(self):
        step = QueryProfileStep(rel_types=["REL"])
        assert step.min_hops == 1
        assert step.max_hops == 1

    def test_max_hops_gte_min_hops(self):
        step = QueryProfileStep(rel_types=["REL"], min_hops=1, max_hops=3)
        assert step.max_hops == 3

    def test_max_hops_lt_min_hops_rejected(self):
        with pytest.raises(ValidationError, match="max_hops"):
            QueryProfileStep(rel_types=["REL"], min_hops=3, max_hops=1)

    def test_hops_equal_is_valid(self):
        step = QueryProfileStep(rel_types=["REL"], min_hops=2, max_hops=2)
        assert step.min_hops == step.max_hops == 2

    def test_min_hops_below_lower_bound_rejected(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["REL"], min_hops=0)

    def test_max_hops_above_upper_bound_rejected(self):
        with pytest.raises(ValidationError):
            QueryProfileStep(rel_types=["REL"], max_hops=5)


class TestQueryProfileTraversal:
    """Validate QueryProfileTraversal constraints."""

    def test_requires_at_least_one_step(self):
        with pytest.raises(ValidationError):
            QueryProfileTraversal(steps=[])

    def test_accepts_single_step(self):
        step = QueryProfileStep(rel_types=["REL"])
        traversal = QueryProfileTraversal(steps=[step])
        assert len(traversal.steps) == 1

    def test_rejects_more_than_three_steps(self):
        steps = [QueryProfileStep(rel_types=["REL"]) for _ in range(4)]
        with pytest.raises(ValidationError):
            QueryProfileTraversal(steps=steps)


class TestQueryProfileDefinition:
    """Validate QueryProfileDefinition shape and validators."""

    @staticmethod
    def _section_profile(**overrides):
        defaults = dict(
            id="test_section",
            label="Test Section",
            kind="section",
            traversals=[
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["HAS_COMPONENT"])]
                )
            ],
        )
        defaults.update(overrides)
        return QueryProfileDefinition(**defaults)

    @staticmethod
    def _dossier_profile(**overrides):
        defaults = dict(
            id="test_dossier",
            label="Test Dossier",
            kind="dossier",
            section_profile_ids=["sec_a"],
        )
        defaults.update(overrides)
        return QueryProfileDefinition(**defaults)

    def test_section_requires_at_least_one_traversal(self):
        with pytest.raises(ValidationError, match="at least one traversal"):
            QueryProfileDefinition(
                id="bad",
                label="Bad Section",
                kind="section",
                traversals=[],
            )

    def test_dossier_requires_at_least_one_section_profile_id(self):
        with pytest.raises(ValidationError, match="at least one section_profile_id"):
            QueryProfileDefinition(
                id="bad",
                label="Bad Dossier",
                kind="dossier",
                section_profile_ids=[],
            )

    def test_valid_section_profile(self):
        p = self._section_profile()
        assert p.kind == "section"
        assert len(p.traversals) == 1

    def test_valid_dossier_profile(self):
        p = self._dossier_profile()
        assert p.kind == "dossier"
        assert p.section_profile_ids == ["sec_a"]

    def test_strip_string_lists_cleans_whitespace(self):
        p = self._section_profile(
            root_entity_types=["  PLATFORM ", " RADAR_SYSTEM  "],
            target_entity_types=["  ANTENNA", ""],
        )
        assert p.root_entity_types == ["PLATFORM", "RADAR_SYSTEM"]
        assert p.target_entity_types == ["ANTENNA"]

    def test_strip_string_lists_removes_blank_entries(self):
        p = self._dossier_profile(
            section_profile_ids=["  sec_a  ", "", "   ", "sec_b"],
        )
        assert p.section_profile_ids == ["sec_a", "sec_b"]

    def test_default_kind_is_section(self):
        # kind defaults to "section", so providing traversals satisfies shape
        p = QueryProfileDefinition(
            id="x",
            label="X",
            traversals=[
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["REL"])]
                )
            ],
        )
        assert p.kind == "section"

    def test_dossier_with_traversals_allowed(self):
        """Dossier may carry traversals, but must have section_profile_ids."""
        p = self._dossier_profile(
            traversals=[
                QueryProfileTraversal(
                    steps=[QueryProfileStep(rel_types=["REL"])]
                )
            ],
        )
        assert p.kind == "dossier"
        assert len(p.traversals) == 1


# ---------------------------------------------------------------------------
# 2. Ontology loader caching
# ---------------------------------------------------------------------------


class TestOntologyCacheInvalidation:
    """Verify cache clear and hook dispatch without touching the database."""

    def test_invalidate_clears_cached_state(self):
        import app.services.ontology_templates as mod

        # Seed the per-bundle cache manually.
        with mod._cache_lock:
            mod._bundle_cache["air_defense_v3"] = (
                {"entity_types": []},
                "test:sig",
                9999999999.0,
            )

        mod.invalidate_ontology_cache()

        with mod._cache_lock:
            assert mod._bundle_cache == {}

    def test_register_invalidation_hook_fires_on_invalidate(self):
        import app.services.ontology_templates as mod

        callback = MagicMock()
        original_hooks = mod._invalidation_hooks[:]
        try:
            mod.register_invalidation_hook(callback)
            mod.invalidate_ontology_cache()
            callback.assert_called_once()
        finally:
            # Clean up so we do not pollute other tests.
            mod._invalidation_hooks[:] = original_hooks

    def test_invalidation_hook_failure_does_not_raise(self):
        import app.services.ontology_templates as mod

        def bad_hook():
            raise RuntimeError("boom")

        original_hooks = mod._invalidation_hooks[:]
        try:
            mod.register_invalidation_hook(bad_hook)
            # Should not propagate.
            mod.invalidate_ontology_cache()
        finally:
            mod._invalidation_hooks[:] = original_hooks



def test_canonical_root_entity_types_in_sync_with_dispatch():
    """Schema layer's _CANONICAL_ROOT_ENTITY_TYPES must match service
    layer's _CANONICAL_BY_ENTITY_TYPE keys. Single source of truth in
    the schema (no circular import); contract test enforces parity."""
    from app.schemas.query_profiles import _CANONICAL_ROOT_ENTITY_TYPES
    from app.services.query_profiles import _CANONICAL_BY_ENTITY_TYPE
    assert set(_CANONICAL_BY_ENTITY_TYPE.keys()) == _CANONICAL_ROOT_ENTITY_TYPES


def test_project_field_groups_groups_by_subgroup():
    """Walks canonical model_fields, picks fields whose
    profile_sections contains the requested section, groups by
    profile_subgroup, sorts deterministically."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {
        "name": "Fan Song",
        "system_name": "Fan Song",
        "gain_dbi": 35.0,
        "beamwidth_az_deg": 1.5,
        "tx_peak_power_kw": 600.0,
        "nominal_rf_mhz": 3000.0,
        "max_speed_mps": None,
    }
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")

    antenna = next(g for g in groups if g.subgroup == "antenna")
    field_names = {f.name for f in antenna.fields}
    assert "gain_dbi" in field_names
    assert "beamwidth_az_deg" in field_names

    transmit = next(g for g in groups if g.subgroup == "transmit")
    assert {f.name for f in transmit.fields} == {"tx_peak_power_kw"}


def test_project_field_groups_skips_none_fields():
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {"name": "X", "gain_dbi": None}
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")

    for g in groups:
        for f in g.fields:
            assert f.value is not None


def test_project_field_groups_only_returns_requested_section():
    """A field tagged ['rf_parameters', 'performance'] appears in both
    sections. A field tagged only ['performance'] appears only in
    performance projections."""
    from ontology_bundles.air_defense_v3.entities import MissileSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {
        "system_name": "SA-2",
        "max_speed_mps": 1100.0,
        "body_length_m": 10.6,
    }
    perf = _project_field_groups(MissileSystemEntity, instance_data, "performance")
    comp = _project_field_groups(MissileSystemEntity, instance_data, "components")

    perf_names = {f.name for g in perf for f in g.fields}
    comp_names = {f.name for g in comp for f in g.fields}

    assert "max_speed_mps" in perf_names
    assert "max_speed_mps" not in comp_names
    assert "body_length_m" in perf_names
    assert "body_length_m" in comp_names


def test_project_field_groups_carries_metadata():
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {"gain_dbi": 35.0}
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")
    antenna = next(g for g in groups if g.subgroup == "antenna")
    gain = next(f for f in antenna.fields if f.name == "gain_dbi")
    assert gain.description and "gain" in gain.description.lower()
    assert gain.label


@pytest.mark.asyncio
async def test_fetch_section_items_property_branch():
    """When profile.kind == section_properties, _fetch_section_items
    returns a list[QueryProfileFieldGroup], not list[GraphEntityResult]."""
    from unittest.mock import AsyncMock
    from app.schemas.graph_store import GraphEntityResult
    from app.schemas.query_profiles import (
        QueryProfileDefinition,
        QueryProfileSearchRequest,
    )
    from app.services.query_profiles import _fetch_section_items

    profile = QueryProfileDefinition(
        id="test_rf", label="Test", kind="section_properties",
        root_entity_types=["RADAR_SYSTEM"],
        profile_sections=["rf_parameters"],
    )
    resolved = GraphEntityResult(
        node_id="#37:0", name="Fan Song", entity_type="RADAR_SYSTEM",
    )
    request = QueryProfileSearchRequest(
        profile_id="test_rf", query_text="Fan Song", top_k=10,
    )
    graph_store = AsyncMock()
    graph_store.get_entity_by_rid = AsyncMock(return_value={
        "system_name": "Fan Song", "gain_dbi": 35.0, "tx_peak_power_kw": 600.0,
    })

    groups = await _fetch_section_items(graph_store, resolved, request, profile)
    assert len(groups) >= 1
    assert any(f.name == "gain_dbi" for g in groups for f in g.fields)


def test_project_field_groups_surfaces_field_evidence():
    """Phase 3 task 35 — when instance_data carries _field_evidence,
    each QueryProfileFieldEntry's evidence list is populated."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {
        "gain_dbi": 35.0,
        "_field_evidence": {
            "gain_dbi": [
                {
                    "chunk_id": None,
                    "element_uid": "#/texts/12",
                    "snippet": "antenna gain measured at 35 dBi",
                    "value": 35.0,
                },
            ],
        },
    }
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")
    antenna = next(g for g in groups if g.subgroup == "antenna")
    gain = next(f for f in antenna.fields if f.name == "gain_dbi")
    assert len(gain.evidence) == 1
    assert gain.evidence[0].supporting_snippet.startswith("antenna gain")
    assert gain.evidence[0].element_uid == "#/texts/12"


def test_project_field_groups_handles_missing_field_evidence():
    """Old data without _field_evidence yields empty evidence lists."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {"gain_dbi": 35.0}
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")
    antenna = next(g for g in groups if g.subgroup == "antenna")
    gain = next(f for f in antenna.fields if f.name == "gain_dbi")
    assert gain.evidence == []


def test_project_field_groups_dossier_catchall_spans_all_sections():
    """The reserved ``"dossier"`` section is a catch-all: it projects EVERY
    field carrying any ``profile_sections`` tag, across all sections, still
    grouped by ``profile_subgroup``, and each field appears exactly ONCE even
    when it carries multiple in-scope section tags."""
    from ontology_bundles.air_defense_v3.entities import MissileSystemEntity
    from app.services.query_profiles import _project_field_groups

    # Values spanning distinct sections:
    #   nomenclature   -> ['identification']              (single-tag)
    #   system_status  -> ['governance']                  (single-tag)
    #   platform       -> ['deployment']                  (single-tag)
    #   emitter_function -> ['performance']               (single-tag)
    #   body_length_m  -> ['components', 'performance']   (multi-tag)
    #   max_speed_mps  -> ['performance', 'engagement_envelope'] (multi-tag)
    instance_data = {
        "system_name": "SA-2",
        "nomenclature": "SA-2 Guideline",
        "system_status": "operational",
        "platform": "towed launcher",
        "emitter_function": "surface-to-air",
        "body_length_m": 10.6,
        "max_speed_mps": 1100.0,
        "min_intercept_km": None,  # None values are still skipped in dossier
    }

    groups = _project_field_groups(MissileSystemEntity, instance_data, "dossier")

    # Every non-None field with a profile tag is projected, spanning
    # identification / governance / deployment / performance / components /
    # engagement_envelope sections.
    all_names = [f.name for g in groups for f in g.fields]
    for expected in (
        "nomenclature",
        "system_status",
        "platform",
        "emitter_function",
        "body_length_m",
        "max_speed_mps",
    ):
        assert expected in all_names, f"{expected} missing from dossier projection"

    # None-valued field is dropped just like in a single-section projection.
    assert "min_intercept_km" not in all_names

    # Grouped by profile_subgroup — the multiple distinct subgroups are present.
    subgroups = {g.subgroup for g in groups}
    for expected_sub in ("designators", "lifecycle", "deployment", "airframe", "kinematics"):
        assert expected_sub in subgroups, f"subgroup {expected_sub} missing"

    # NO duplicates: a field tagged with two in-scope sections (e.g.
    # max_speed_mps is performance + engagement_envelope; body_length_m is
    # components + performance) appears exactly once across all groups.
    assert len(all_names) == len(set(all_names)), (
        f"duplicate field(s) in dossier projection: {all_names}"
    )
    assert all_names.count("max_speed_mps") == 1
    assert all_names.count("body_length_m") == 1
