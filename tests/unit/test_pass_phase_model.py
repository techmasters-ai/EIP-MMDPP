"""C1.6 — PassManifest phase field: schema + inference tests.

Task 1.6.2: verify that:
- A manifest with explicit ``phase:`` per pass loads cleanly.
- A manifest WITHOUT ``phase:`` per pass loads via inference + emits INFO log.
- A manifest with a BAD ``phase:`` value raises ValidationError.
- Inference rules: ``*_identity`` → identity; ``document_plus_entity_refs`` → relationship;
  otherwise → field_group.

No real DB, no Celery. All tests use the pydantic model directly.

Run standalone:
    python3 -m pytest tests/unit/test_pass_phase_model.py -v
"""
from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Helper: build a minimal raw pass dict
# ---------------------------------------------------------------------------

def _raw_pass(
    name: str = "radar_power_rf",
    input_mode: str = "document_only",
    *,
    phase: str | None = None,
) -> dict:
    d = {
        "name": name,
        "required": False,
        "kind": "entities",
        "input_mode": input_mode,
        "module": "extraction_schemas.placeholder",
        "template_class": "PlaceholderPass",
    }
    if phase is not None:
        d["phase"] = phase
    return d


def _raw_manifest(passes: list[dict]) -> dict:
    return {
        "bundle_key": "test_bundle",
        "manifest_schema_version": "1.0.0",
        "ontology_name": "Test Ontology",
        "ontology_version": "1.0.0",
        "extraction_profile_version": "1.0.0",
        "passes": passes,
    }


# ---------------------------------------------------------------------------
# 1. Explicit phase: all three valid values load cleanly
# ---------------------------------------------------------------------------

class TestExplicitPhaseLoadsCleanly:

    def test_identity_phase_explicit(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass("radar_identity", phase="identity"))
        assert p.phase == "identity"

    def test_field_group_phase_explicit(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass("radar_power_rf", phase="field_group"))
        assert p.phase == "field_group"

    def test_relationship_phase_explicit(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(
            _raw_pass("system_links", "document_plus_entity_refs", phase="relationship")
        )
        assert p.phase == "relationship"

    def test_full_bundle_loads_all_passes_with_explicit_phase(self):
        """The real air_defense_v3 manifest (with explicit phase:) loads cleanly."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3")
        for p in m.passes:
            assert p.phase in ("identity", "field_group", "relationship"), (
                f"Pass {p.name!r} has unexpected phase {p.phase!r}"
            )

    def test_subset_bundle_loads_all_passes_with_explicit_phase(self):
        """The real air_defense_v3_baseline_subset manifest loads cleanly."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3_baseline_subset")
        for p in m.passes:
            assert p.phase in ("identity", "field_group", "relationship"), (
                f"Pass {p.name!r} has unexpected phase {p.phase!r}"
            )


# ---------------------------------------------------------------------------
# 2. Bad phase value raises ValidationError
# ---------------------------------------------------------------------------

class TestBadPhaseRaisesValidationError:

    def test_unknown_phase_value_raises(self):
        from app.services.ontology_bundles import PassManifest
        with pytest.raises(ValidationError):
            PassManifest.model_validate(_raw_pass("radar_power_rf", phase="extraction"))

    def test_empty_phase_string_raises(self):
        from app.services.ontology_bundles import PassManifest
        with pytest.raises(ValidationError):
            PassManifest.model_validate(_raw_pass("radar_power_rf", phase=""))

    def test_uppercase_phase_raises(self):
        from app.services.ontology_bundles import PassManifest
        with pytest.raises(ValidationError):
            PassManifest.model_validate(_raw_pass("radar_power_rf", phase="Identity"))


# ---------------------------------------------------------------------------
# 3. Inference when phase: is absent
# ---------------------------------------------------------------------------

class TestPhaseInferenceWhenMissing:

    def test_identity_name_inferred(self):
        """``*_identity`` suffix → phase='identity'."""
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass("radar_identity"))
        assert p.phase == "identity"

    def test_missile_identity_name_inferred(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass("missile_identity"))
        assert p.phase == "identity"

    def test_arbitrary_identity_suffix_inferred(self):
        """Any pass whose name ends with _identity is inferred as identity."""
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass("vehicle_identity"))
        assert p.phase == "identity"

    def test_document_plus_entity_refs_inferred_as_relationship(self):
        """``input_mode=document_plus_entity_refs`` → phase='relationship'."""
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(
            _raw_pass("system_links", "document_plus_entity_refs")
        )
        assert p.phase == "relationship"

    def test_document_only_non_identity_inferred_as_field_group(self):
        """Everything else → phase='field_group'."""
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass("radar_power_rf", "document_only"))
        assert p.phase == "field_group"

    def test_missile_kinematics_inferred_as_field_group(self):
        from app.services.ontology_bundles import PassManifest
        p = PassManifest.model_validate(_raw_pass("missile_kinematics"))
        assert p.phase == "field_group"

    def test_inference_logs_info(self, caplog):
        """Inferred phase emits an INFO-level log message."""
        from app.services.ontology_bundles import PassManifest
        with caplog.at_level(logging.INFO, logger="app.services.ontology_bundles"):
            PassManifest.model_validate(_raw_pass("radar_power_rf"))
        assert any(
            "inferred phase=" in record.message
            and "radar_power_rf" in record.message
            for record in caplog.records
        ), f"Expected INFO inference log; records: {[r.message for r in caplog.records]}"

    def test_explicit_phase_does_not_log(self, caplog):
        """When phase is explicitly declared, no inference log is emitted."""
        from app.services.ontology_bundles import PassManifest
        with caplog.at_level(logging.INFO, logger="app.services.ontology_bundles"):
            PassManifest.model_validate(_raw_pass("radar_power_rf", phase="field_group"))
        assert not any(
            "inferred phase=" in record.message
            for record in caplog.records
        ), "Unexpected inference log for explicitly-declared phase"


# ---------------------------------------------------------------------------
# 4. Phase ordering in actual bundles
# ---------------------------------------------------------------------------

class TestBundlePhaseDistribution:

    def test_air_defense_v3_identity_passes(self):
        """radar_identity + missile_identity must be phase=identity."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3")
        identity_names = {p.name for p in m.passes if p.phase == "identity"}
        assert "radar_identity" in identity_names
        assert "missile_identity" in identity_names

    def test_air_defense_v3_system_links_is_relationship(self):
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3")
        sl = m.find_pass("system_links")
        assert sl.phase == "relationship"

    def test_air_defense_v3_field_group_passes(self):
        """The 9 non-identity, non-system_links passes must be field_group."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3")
        field_group_names = {p.name for p in m.passes if p.phase == "field_group"}
        expected = {
            "radar_power_rf", "radar_antenna", "radar_timing", "radar_modulation",
            "missile_kinematics", "missile_guidance", "missile_airframe",
            "missile_speed_timing", "missile_propulsion",
        }
        assert field_group_names == expected

    def test_subset_bundle_phase_distribution(self):
        """Subset bundle: 2 identity + 2 field_group + 1 relationship."""
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest("air_defense_v3_baseline_subset")
        identity = [p for p in m.passes if p.phase == "identity"]
        field_group = [p for p in m.passes if p.phase == "field_group"]
        relationship = [p for p in m.passes if p.phase == "relationship"]
        assert len(identity) == 2
        assert len(field_group) == 2
        assert len(relationship) == 1
        assert {p.name for p in identity} == {"radar_identity", "missile_identity"}
        assert {p.name for p in field_group} == {"radar_power_rf", "missile_kinematics"}
        assert relationship[0].name == "system_links"

    def test_no_pass_has_none_phase(self):
        """Every pass in every real bundle has a non-None phase."""
        from app.services.ontology_bundles import load_bundle_manifest
        for bundle_key in ("air_defense_v3", "air_defense_v3_baseline_subset"):
            m = load_bundle_manifest(bundle_key)
            for p in m.passes:
                assert p.phase is not None, (
                    f"Pass {p.name!r} in {bundle_key!r} has phase=None"
                )


# ---------------------------------------------------------------------------
# 5. Fix 2: relationship phase restricted to name='system_links' (manifest validator)
# ---------------------------------------------------------------------------

class TestRelationshipPhaseNameValidation:
    """BundleManifest validator: phase=relationship must have name='system_links'."""

    def test_system_links_relationship_loads_ok(self):
        """name=system_links + phase=relationship → valid, loads cleanly."""
        from app.services.ontology_bundles import BundleManifest
        raw = _raw_manifest([
            _raw_pass("system_links", "document_plus_entity_refs", phase="relationship"),
        ])
        m = BundleManifest.model_validate(raw)
        assert m.passes[0].phase == "relationship"
        assert m.passes[0].name == "system_links"

    def test_non_system_links_relationship_raises(self):
        """name=other_links + phase=relationship → ValidationError naming the offending pass."""
        from app.services.ontology_bundles import BundleManifest
        raw = _raw_manifest([
            _raw_pass("other_links", "document_plus_entity_refs", phase="relationship"),
        ])
        with pytest.raises(ValidationError) as exc_info:
            BundleManifest.model_validate(raw)
        assert "other_links" in str(exc_info.value), (
            f"ValidationError message should name the offending pass; got: {exc_info.value}"
        )

    def test_system_links_field_group_loads_ok(self):
        """name=system_links + phase=field_group → valid (no relationship pass, Fix 2 not triggered).

        Fix 3 requires an identity pass whenever field_group is present, so we include one.
        The point of this test is only that Fix 2 doesn't fire for a non-relationship pass.
        """
        from app.services.ontology_bundles import BundleManifest
        raw = _raw_manifest([
            _raw_pass("radar_identity", phase="identity"),
            _raw_pass("system_links", "document_only", phase="field_group"),
        ])
        m = BundleManifest.model_validate(raw)
        assert m.passes[1].phase == "field_group"


# ---------------------------------------------------------------------------
# 6. Fix 3: field_group passes require at least one identity pass (manifest validator)
# ---------------------------------------------------------------------------

class TestFieldGroupRequiresIdentityPass:
    """BundleManifest validator: field_group passes require an identity pass."""

    def test_identity_and_field_group_loads_ok(self):
        """Bundle with identity + field_group → valid."""
        from app.services.ontology_bundles import BundleManifest
        raw = _raw_manifest([
            _raw_pass("radar_identity", phase="identity"),
            _raw_pass("radar_power_rf", phase="field_group"),
        ])
        m = BundleManifest.model_validate(raw)
        assert len(m.passes) == 2

    def test_field_group_only_raises(self):
        """Bundle with field_group but no identity pass → ValidationError."""
        from app.services.ontology_bundles import BundleManifest
        raw = _raw_manifest([
            _raw_pass("radar_power_rf", phase="field_group"),
        ])
        with pytest.raises(ValidationError) as exc_info:
            BundleManifest.model_validate(raw)
        assert "field_group" in str(exc_info.value).lower() or "identity" in str(exc_info.value).lower(), (
            f"ValidationError should mention field_group/identity; got: {exc_info.value}"
        )

    def test_identity_and_relationship_no_field_group_loads_ok(self):
        """Bundle with identity + relationship (no field_group) → valid."""
        from app.services.ontology_bundles import BundleManifest
        raw = _raw_manifest([
            _raw_pass("radar_identity", phase="identity"),
            _raw_pass("system_links", "document_plus_entity_refs", phase="relationship"),
        ])
        m = BundleManifest.model_validate(raw)
        assert len(m.passes) == 2

    def test_relationship_only_loads_ok(self):
        """Bundle with only a relationship pass (no field_group) → valid."""
        from app.services.ontology_bundles import BundleManifest
        raw = _raw_manifest([
            _raw_pass("system_links", "document_plus_entity_refs", phase="relationship"),
        ])
        m = BundleManifest.model_validate(raw)
        assert m.passes[0].phase == "relationship"
