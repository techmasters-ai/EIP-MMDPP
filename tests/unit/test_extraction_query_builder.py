"""B1+B2 — snapshot and structural tests for build_retrieval_profile / build_retrieval_query.

The expected query text is FROZEN at implementation time.  If the schema
descriptions change, these tests fail on purpose — the implementer must
review the new query text and consciously update the expected string.

Run standalone:
    python3 -m pytest tests/unit/test_extraction_query_builder.py -v
"""
from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pass_cls(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# 1. radar_power_rf snapshot
# ---------------------------------------------------------------------------

class TestRadarPowerRfQuerySnapshot:

    def test_radar_power_rf_query_snapshot(self):
        """Snapshot: radar_power_rf query text must match exactly.

        Expected text frozen from the current RadarPowerRfPass / RadarPowerRfRecord
        schema (re-baselined 2026-05-29). Update this string only if you intend to
        change retrieval query content.
        """
        from app.services.extraction_query_builder import build_retrieval_query

        RadarPowerRfPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
        )
        expected = (
            "Radar RF power and carrier-frequency characteristics: ERP, peak "
            "transmitter power, operating frequency, and radar band.\n"
            "Effective Radiated Power in dBW. Relevant source labels include "
            "'ERP', 'EIRP', 'Effective Radiated Power', 'effective radiated "
            "power', 'radiated power', 'effective power', 'dBW', or 'dBm'. "
            "Use only effective radiated power or EIRP values stated with "
            "dBW/dBm units; otherwise null.\n"
            "Transmitter peak power in kilowatts. Relevant source labels include "
            "'Peak Power', 'peak transmitter power', 'Transmitter Power', "
            "'transmitter output power', 'Tx Power', 'Pulse Power', "
            "'peak pulse power', 'magnetron output', or 'klystron output'. "
            "Use peak transmitter or pulse output power, not ERP/EIRP or average power.\n"
            "Nominal RF carrier or operating frequency in MHz. Relevant source "
            "labels include 'Frequency', 'Operating Frequency', 'Carrier "
            "Frequency', 'RF', 'frequency range', 'waveband', 'radar band', "
            "'MHz', 'GHz', 'VHF', 'UHF', 'L-band', 'S-band', 'C-band', "
            "'X-band', or 'Ku-band'. Use radar carrier frequency, not PRF/PRI "
            "or modulation bandwidth."
        )
        result = build_retrieval_query(None, RadarPowerRfPass)
        assert result == expected


# ---------------------------------------------------------------------------
# 2. missile_kinematics snapshot
# ---------------------------------------------------------------------------

class TestMissileKinematicsQuerySnapshot:

    def test_missile_kinematics_query_snapshot(self):
        """Snapshot: missile_kinematics query text must match exactly.

        Expected text frozen from the current MissileKinematicsPass / MissileKinematicsRecord
        schema (re-baselined 2026-05-29). Update this string only if you intend to
        change retrieval query content.
        """
        from app.services.extraction_query_builder import build_retrieval_query

        MissileKinematicsPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics",
            "MissileKinematicsPass",
        )
        expected = (
            "Surface-to-air missile engagement envelope: range, altitude, "
            "ceiling, floor, and launch-angle limits.\n"
            "Minimum intercept range in kilometers. Relevant source labels "
            "include 'minimum effective range', 'minimum range', 'Min Range', "
            "'minimum intercept range', 'inner range', 'range floor', "
            "'near limit', or 'kill zone minimum'. Use missile engagement "
            "envelope limits, not radar range or launcher spacing.\n"
            "Maximum intercept range in kilometers. Relevant source labels "
            "include 'maximum effective range', 'maximum range', 'Max Range', "
            "'Range', 'effective range', 'engagement range', 'intercept range', "
            "'range against targets', 'range limit', or 'kill zone range'. "
            "Use missile engagement envelope limits, not radar range or launcher spacing.\n"
            "Minimum engagement altitude in kilometers. Relevant source labels "
            "include 'minimum effective altitude', 'minimum altitude', "
            "'Min Altitude', 'Min Alt', 'altitude floor', 'lower altitude limit', "
            "'minimum intercept altitude', or 'kill zone floor'.\n"
            "Maximum engagement altitude in kilometers. Relevant source labels "
            "include 'maximum effective altitude', 'maximum altitude', "
            "'Max Altitude', 'Max Alt', 'Altitude', 'altitude ceiling', "
            "'ceiling', 'intercept ceiling', 'engagement altitude', "
            "'launch ceiling', or 'kill zone ceiling'.\n"
            "Maximum launch angle in degrees. Relevant source labels include "
            "'maximum launch angle', 'launch angle', 'elevation launch angle', "
            "'off-boresight launch angle', 'firing angle', or 'canister elevation angle'."
        )
        result = build_retrieval_query(None, MissileKinematicsPass)
        assert result == expected


# ---------------------------------------------------------------------------
# 3. system_name exclusion guard
# ---------------------------------------------------------------------------

class TestSystemNameExclusion:

    def test_query_excludes_system_name_field(self):
        """system_name must NEVER appear in the rendered query — it is an
        identity field, not a field-relevance signal."""
        from app.services.extraction_query_builder import build_retrieval_query

        RadarPowerRfPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
        )
        MissileKinematicsPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics",
            "MissileKinematicsPass",
        )
        for cls in (RadarPowerRfPass, MissileKinematicsPass):
            result = build_retrieval_query(None, cls)
            # "system_name" must not appear anywhere in the query text
            assert "system_name" not in result, (
                f"system_name leaked into query for {cls.__name__}: {result!r}"
            )
            # The identity-field canonical-name description fragment must also
            # not appear (it begins "Canonical designation of the")
            assert "Canonical designation" not in result, (
                f"system_name description leaked into query for {cls.__name__}"
            )


# ---------------------------------------------------------------------------
# 4. Field without description falls back to humanized name
# ---------------------------------------------------------------------------

class TestHumanizedFallback:

    def test_field_without_description_falls_back_to_humanized_name(self):
        """A field with no description renders as humanized snake_case."""
        from app.services.extraction_query_builder import build_retrieval_query

        class _SyntheticRecord(BaseModel):
            """Synthetic record for testing humanized fallback."""

            model_config = ConfigDict(
                extra="ignore",
                ontology_name="TEST_ENTITY",
                graph_id_fields=["system_name"],
                identity_scope="global",
                is_entity=True,
            )

            system_name: str = Field(..., description="identity field — skipped")
            tx_peak_power_kw: Optional[float] = Field(default=None)  # no description
            nominal_rf_mhz: Optional[float] = Field(
                default=None,
                description="Nominal carrier frequency in MHz.",
            )

        class _SyntheticPass(BaseModel):
            """Synthetic pass wrapper."""

            model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

            records: list[_SyntheticRecord] = Field(default_factory=list)

        result = build_retrieval_query(None, _SyntheticPass)

        # Humanized fallback for tx_peak_power_kw
        assert "tx peak power kw" in result, (
            f"humanized field name not found in query: {result!r}"
        )
        # Description for nominal_rf_mhz is present
        assert "Nominal carrier frequency in MHz." in result

    def test_no_description_pure_record_class(self):
        """Works when template_cls IS the record class (is_entity=True,
        graph_id_fields non-empty)."""
        from app.services.extraction_query_builder import build_retrieval_query

        class _DirectRecord(BaseModel):
            """Direct record with no-description field."""

            model_config = ConfigDict(
                extra="ignore",
                ontology_name="TEST",
                graph_id_fields=["system_name"],
                is_entity=True,
            )

            system_name: str = Field(..., description="identity — skipped")
            coast_time_sec: Optional[float] = Field(default=None)  # no description

        result = build_retrieval_query(None, _DirectRecord)
        assert "coast time sec" in result


# ---------------------------------------------------------------------------
# 5. Manifest round-trip — retrieval blocks load and round-trip cleanly
# ---------------------------------------------------------------------------

class TestManifestRetrievalBlock:

    def test_air_defense_v3_field_group_passes_have_retrieval(self):
        """All field_group passes in air_defense_v3 must have a retrieval block."""
        from app.services.ontology_bundles import load_bundle_manifest, RetrievalProfile

        m = load_bundle_manifest("air_defense_v3")
        field_group_passes = [p for p in m.passes if p.phase == "field_group"]
        assert field_group_passes, "Expected at least one field_group pass"

        for p in field_group_passes:
            assert p.retrieval is not None, (
                f"Pass {p.name!r} (field_group) is missing retrieval block"
            )
            assert isinstance(p.retrieval, RetrievalProfile)

    def test_air_defense_v3_non_field_group_passes_have_no_retrieval(self):
        """identity and relationship passes must NOT have a retrieval block."""
        from app.services.ontology_bundles import load_bundle_manifest

        m = load_bundle_manifest("air_defense_v3")
        for p in m.passes:
            if p.phase != "field_group":
                assert p.retrieval is None, (
                    f"Pass {p.name!r} (phase={p.phase}) unexpectedly has a "
                    f"retrieval block — only field_group passes should."
                )

    def test_air_defense_v3_subset_retrieval_blocks(self):
        """Subset bundle: radar_power_rf + missile_kinematics have retrieval;
        identity + relationship do not."""
        from app.services.ontology_bundles import load_bundle_manifest, RetrievalProfile

        m = load_bundle_manifest("air_defense_v3_baseline_subset")
        for p in m.passes:
            if p.phase == "field_group":
                assert p.retrieval is not None and isinstance(
                    p.retrieval, RetrievalProfile
                ), f"Pass {p.name!r} missing RetrievalProfile"
            else:
                assert p.retrieval is None, (
                    f"Non-field_group pass {p.name!r} should not have retrieval"
                )

    def test_conservative_defaults_applied(self):
        """Manifest defaults for air_defense_v3 radar_power_rf must match current values.

        min_similarity was uniformly set to 0.35 across all bundles in commit 48302f2.
        """
        from app.services.ontology_bundles import load_bundle_manifest

        m = load_bundle_manifest("air_defense_v3")
        rp_pass = m.find_pass("radar_power_rf")
        r = rp_pass.retrieval
        assert r is not None
        assert r.min_similarity == pytest.approx(0.35)
        assert r.top_n_candidates == 50
        assert r.top_k == 15
        assert r.fallback_to_full is True


# ---------------------------------------------------------------------------
# 6. RetrievalProfile validation
# ---------------------------------------------------------------------------

class TestRetrievalProfileValidation:

    def test_defaults(self):
        """RetrievalProfile() with no args uses conservative defaults."""
        from app.services.ontology_bundles import RetrievalProfile

        r = RetrievalProfile()
        assert r.min_similarity == pytest.approx(0.45)  # Field-level defaults (no manifest override) — manifest sets 0.35/15
        assert r.top_n_candidates == 50
        assert r.top_k == 20
        assert r.fallback_to_full is True

    def test_min_similarity_below_zero_raises(self):
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(min_similarity=-0.01)

    def test_min_similarity_above_one_raises(self):
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(min_similarity=1.01)

    def test_top_n_candidates_zero_raises(self):
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(top_n_candidates=0)

    def test_top_n_candidates_above_max_raises(self):
        """top_n_candidates has le=2000; values above that must raise."""
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(top_n_candidates=2001)

    def test_top_k_zero_raises(self):
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(top_k=0)

    def test_top_k_above_max_raises(self):
        """top_k has le=2000; values above that must raise."""
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(top_k=2001)

    def test_misspelled_key_raises(self):
        """extra='forbid' must catch misspelled keys."""
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(**{"min_similarity_score": 0.5})

    def test_fallback_to_full_false_accepted(self):
        """fallback_to_full=False is valid — operators may set it post C.6."""
        from app.services.ontology_bundles import RetrievalProfile

        r = RetrievalProfile(fallback_to_full=False)
        assert r.fallback_to_full is False


# ---------------------------------------------------------------------------
# 7. B1 — FieldRetrievalQuery + PassRetrievalSignals importable + constructable
# ---------------------------------------------------------------------------

class TestB1RetrievalSignalTypes:
    """Sanity: both frozen dataclasses are importable and constructable."""

    def test_field_retrieval_query_constructable(self):
        from app.services.extraction_query_builder import FieldRetrievalQuery

        frq = FieldRetrievalQuery(
            field_name="some_field",
            query_text="some query text describing the field",
            aliases=("alias_a", "alias_b"),
            negative_terms=("exclude_this",),
            evidence_patterns=("Table \\d+",),
            likely_sections=("Specifications",),
            units=("km",),
        )
        assert frq.field_name == "some_field"
        assert frq.aliases == ("alias_a", "alias_b")

    def test_field_retrieval_query_is_frozen(self):
        from app.services.extraction_query_builder import FieldRetrievalQuery

        frq = FieldRetrievalQuery(
            field_name="some_field",
            query_text="some query",
            aliases=(),
            negative_terms=(),
            evidence_patterns=(),
            likely_sections=(),
            units=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            frq.field_name = "mutated"  # type: ignore[misc]

    def test_pass_retrieval_signals_constructable(self):
        from app.services.extraction_query_builder import (
            FieldRetrievalQuery,
            PassRetrievalSignals,
        )

        frq = FieldRetrievalQuery(
            field_name="numeric_field",
            query_text="a numeric measurement field",
            aliases=("label_x", "label_y"),
            negative_terms=("not_this",),
            evidence_patterns=(),
            likely_sections=("Performance",),
            units=("m",),
        )
        prs = PassRetrievalSignals(
            pass_name="example_pass",
            entity_doc="Entity documentation string.",
            entity_query="entity level query text",
            field_queries=(frq,),
            lexical_terms=("label_x", "label_y"),
            negative_terms=("not_this",),
            likely_sections=("Performance",),
            evidence_patterns=(),
        )
        assert prs.pass_name == "example_pass"
        assert len(prs.field_queries) == 1
        assert prs.field_queries[0].field_name == "numeric_field"

    def test_pass_retrieval_signals_is_frozen(self):
        from app.services.extraction_query_builder import PassRetrievalSignals

        prs = PassRetrievalSignals(
            pass_name="example_pass",
            entity_doc="",
            entity_query="",
            field_queries=(),
            lexical_terms=(),
            negative_terms=(),
            likely_sections=(),
            evidence_patterns=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            prs.pass_name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. B2 — build_retrieval_profile() returns per-field FieldRetrievalQuery objects
# ---------------------------------------------------------------------------

class TestB2BuildRetrievalProfile:

    def test_build_retrieval_profile_returns_field_queries(self):
        """build_retrieval_profile must return a PassRetrievalSignals with at
        least one FieldRetrievalQuery per non-identity field on
        MissileKinematicsRecord.

        Non-identity fields (system_name is skipped): min_intercept_km,
        max_intercept_km, min_altitude_km, max_altitude_km,
        max_launch_angle_deg.
        """
        from app.services.extraction_query_builder import (
            build_retrieval_profile,
            FieldRetrievalQuery,
            PassRetrievalSignals,
        )

        MissileKinematicsPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics",
            "MissileKinematicsPass",
        )

        result = build_retrieval_profile(None, MissileKinematicsPass)

        # Must return the right container type
        assert isinstance(result, PassRetrievalSignals)

        # Must have field_queries
        assert len(result.field_queries) > 0, "Expected at least one FieldRetrievalQuery"

        # All elements must be FieldRetrievalQuery instances
        for fq in result.field_queries:
            assert isinstance(fq, FieldRetrievalQuery), (
                f"Expected FieldRetrievalQuery, got {type(fq)}"
            )

        # Non-identity fields that must each have a FieldRetrievalQuery
        expected_fields = {
            "min_intercept_km",
            "max_intercept_km",
            "min_altitude_km",
            "max_altitude_km",
            "max_launch_angle_deg",
        }
        field_names_in_result = {fq.field_name for fq in result.field_queries}
        missing = expected_fields - field_names_in_result
        assert not missing, (
            f"Missing FieldRetrievalQuery for fields: {missing}"
        )

        # system_name must NOT appear (identity field — in _SKIP_FIELDS)
        assert "system_name" not in field_names_in_result, (
            "system_name must be excluded from field_queries"
        )

        # entity_query must be non-empty (drives the B0 snapshot)
        assert result.entity_query, "entity_query must be non-empty"

        # Task B4 populated retrieval metadata for all MissileKinematics fields.
        # This guard now asserts non-empty tuples — any regression that clears
        # retrieval blocks will be caught here.
        for fq in result.field_queries:
            assert len(fq.aliases) > 0, (
                f"aliases must be non-empty after B4: {fq.field_name}"
            )
            assert len(fq.likely_sections) > 0, (
                f"likely_sections must be non-empty after B4: {fq.field_name}"
            )

    def test_build_retrieval_query_shim_byte_identical(self):
        """build_retrieval_query (now a shim) must return exactly
        PassRetrievalSignals.entity_query — byte-for-byte equal to the
        B0-re-baselined snapshots that TestRadarPowerRfQuerySnapshot and
        TestMissileKinematicsQuerySnapshot already assert.

        This test proves the shim refactor changed nothing.
        """
        from app.services.extraction_query_builder import (
            build_retrieval_profile,
            build_retrieval_query,
        )

        RadarPowerRfPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
        )
        MissileKinematicsPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics",
            "MissileKinematicsPass",
        )

        for cls in (RadarPowerRfPass, MissileKinematicsPass):
            profile = build_retrieval_profile(None, cls)
            shim_result = build_retrieval_query(None, cls)
            assert shim_result == profile.entity_query, (
                f"Shim result differs from profile.entity_query for {cls.__name__}:\n"
                f"  shim:    {shim_result!r}\n"
                f"  profile: {profile.entity_query!r}"
            )


# ---------------------------------------------------------------------------
# 9. B3 — radar field-group passes have rich retrieval metadata
# ---------------------------------------------------------------------------

class TestB3FieldMetadataLoaded:
    """B3 acceptance: all 4 radar field-group passes have populated retrieval blocks.

    For every non-identity field on every radar Record class:
      - fq.aliases is non-empty
      - PassRetrievalSignals.lexical_terms is non-empty (union of all aliases)
    Also confirms per-pass field counts so schema additions are caught early.
    """

    _RADAR_PASSES = [
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
            # non-identity fields expected
            {"erp_dbw", "tx_peak_power_kw", "nominal_rf_mhz"},
        ),
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_antenna",
            "RadarAntennaPass",
            {
                "antenna_photo", "gain_dbi",
                "antenna_dim_az_m", "antenna_dim_el_m",
                "beamwidth_az_deg", "beamwidth_el_deg",
                "spoiled", "coverage_limits_el_deg",
            },
        ),
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_timing",
            "RadarTimingPass",
            {"nominal_pri_usec", "nominal_pd_usec", "scan_period_sec", "dwell_time"},
        ),
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_modulation",
            "RadarModulationPass",
            {
                "intra_pulse_mop", "inter_pulse",
                "frequency_excursion_mhz", "num_bits_in_code", "pulses_per_dwell",
            },
        ),
    ]

    def test_lexical_terms_non_empty_for_all_radar_passes(self):
        """PassRetrievalSignals.lexical_terms must be non-empty for every radar pass."""
        from app.services.extraction_query_builder import build_retrieval_profile

        for mod_path, cls_name, _ in self._RADAR_PASSES:
            cls = _load_pass_cls(mod_path, cls_name)
            profile = build_retrieval_profile(None, cls)
            assert len(profile.lexical_terms) > 0, (
                f"{cls_name}: lexical_terms is empty — retrieval blocks missing or not loaded"
            )

    def test_all_non_identity_fields_have_aliases(self):
        """Every non-identity field on every radar pass must have at least one alias."""
        from app.services.extraction_query_builder import build_retrieval_profile

        for mod_path, cls_name, expected_fields in self._RADAR_PASSES:
            cls = _load_pass_cls(mod_path, cls_name)
            profile = build_retrieval_profile(None, cls)

            field_query_map = {fq.field_name: fq for fq in profile.field_queries}

            # All expected fields must be present in field_queries
            missing = expected_fields - field_query_map.keys()
            assert not missing, (
                f"{cls_name}: missing FieldRetrievalQuery for fields: {missing}"
            )

            # system_name must be excluded
            assert "system_name" not in field_query_map, (
                f"{cls_name}: system_name (identity field) must not appear in field_queries"
            )

            # Every expected field must have non-empty aliases
            for field_name in expected_fields:
                fq = field_query_map[field_name]
                assert len(fq.aliases) > 0, (
                    f"{cls_name}.{field_name}: aliases is empty — "
                    f"json_schema_extra retrieval block missing or malformed"
                )

    def test_radar_power_rf_aliases_spot_check(self):
        """Spot-check: radar_power_rf erp_dbw has 'ERP' and 'EIRP' aliases;
        tx_peak_power_kw has 'peak power'; nominal_rf_mhz has 'operating frequency'."""
        from app.services.extraction_query_builder import build_retrieval_profile

        RadarPowerRfPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
        )
        profile = build_retrieval_profile(None, RadarPowerRfPass)
        fq_map = {fq.field_name: fq for fq in profile.field_queries}

        erp = fq_map["erp_dbw"]
        assert "ERP" in erp.aliases, f"ERP not found in erp_dbw aliases: {erp.aliases}"
        assert "EIRP" in erp.aliases, f"EIRP not found in erp_dbw aliases: {erp.aliases}"

        tx = fq_map["tx_peak_power_kw"]
        assert "peak power" in tx.aliases, (
            f"'peak power' not in tx_peak_power_kw aliases: {tx.aliases}"
        )

        rf = fq_map["nominal_rf_mhz"]
        assert "operating frequency" in rf.aliases, (
            f"'operating frequency' not in nominal_rf_mhz aliases: {rf.aliases}"
        )

    def test_radar_timing_aliases_spot_check(self):
        """Spot-check: nominal_pri_usec has 'PRI'; nominal_pd_usec has 'pulse width';
        scan_period_sec has 'scan period'; dwell_time has 'dwell time'."""
        from app.services.extraction_query_builder import build_retrieval_profile

        RadarTimingPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_timing",
            "RadarTimingPass",
        )
        profile = build_retrieval_profile(None, RadarTimingPass)
        fq_map = {fq.field_name: fq for fq in profile.field_queries}

        assert "PRI" in fq_map["nominal_pri_usec"].aliases
        assert "pulse width" in fq_map["nominal_pd_usec"].aliases
        assert "scan period" in fq_map["scan_period_sec"].aliases
        assert "dwell time" in fq_map["dwell_time"].aliases

    def test_radar_modulation_aliases_spot_check(self):
        """Spot-check: intra_pulse_mop has 'LFM' and 'chirp'; inter_pulse has
        'frequency hopping'; frequency_excursion_mhz has 'chirp bandwidth';
        num_bits_in_code has 'code length'; pulses_per_dwell has 'pulses per dwell'."""
        from app.services.extraction_query_builder import build_retrieval_profile

        RadarModulationPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_modulation",
            "RadarModulationPass",
        )
        profile = build_retrieval_profile(None, RadarModulationPass)
        fq_map = {fq.field_name: fq for fq in profile.field_queries}

        assert "LFM" in fq_map["intra_pulse_mop"].aliases
        assert "chirp" in fq_map["intra_pulse_mop"].aliases
        assert "frequency hopping" in fq_map["inter_pulse"].aliases
        assert "chirp bandwidth" in fq_map["frequency_excursion_mhz"].aliases
        assert "code length" in fq_map["num_bits_in_code"].aliases
        assert "pulses per dwell" in fq_map["pulses_per_dwell"].aliases

    def test_radar_antenna_aliases_spot_check(self):
        """Spot-check: gain_dbi has 'antenna gain'; beamwidth_az_deg has 'HPBW azimuth';
        spoiled has 'fan beam'; coverage_limits_el_deg has 'elevation coverage'."""
        from app.services.extraction_query_builder import build_retrieval_profile

        RadarAntennaPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_antenna",
            "RadarAntennaPass",
        )
        profile = build_retrieval_profile(None, RadarAntennaPass)
        fq_map = {fq.field_name: fq for fq in profile.field_queries}

        assert "antenna gain" in fq_map["gain_dbi"].aliases
        assert "HPBW azimuth" in fq_map["beamwidth_az_deg"].aliases
        assert "fan beam" in fq_map["spoiled"].aliases
        assert "elevation coverage" in fq_map["coverage_limits_el_deg"].aliases


# ---------------------------------------------------------------------------
# 10. B4 — missile field-group passes have rich retrieval metadata
# ---------------------------------------------------------------------------

class TestB4FieldMetadataLoaded:
    """B4 acceptance: all 5 missile field-group passes have populated retrieval blocks.

    For every non-identity field on every missile Record class:
      - fq.aliases is non-empty
      - PassRetrievalSignals.lexical_terms is non-empty (union of all aliases)
    Also confirms per-pass field counts so schema additions are caught early.
    """

    _MISSILE_PASSES = [
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics",
            "MissileKinematicsPass",
            # non-identity fields expected
            {
                "min_intercept_km", "max_intercept_km",
                "min_altitude_km", "max_altitude_km",
                "max_launch_angle_deg",
            },
        ),
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_guidance",
            "MissileGuidancePass",
            {"guidance_type", "seeker_type", "missile_photo"},
        ),
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_airframe",
            "MissileAirframePass",
            {"body_length_m", "body_diameter_m", "total_mass_kg"},
        ),
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_speed_timing",
            "MissileSpeedTimingPass",
            {
                "average_speed_mps", "max_speed_mps", "max_flyout_time_sec",
                "flight_time_sec", "coast_time_sec", "intra_salvo_time_sec",
                "total_burn_time_sec", "ejector_time_sec",
            },
        ),
        (
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_propulsion",
            "MissilePropulsionPass",
            {
                "ejector_thrust", "ejector_mass_kg",
                "booster_time_sec", "booster_thrust", "booster_mass_kg",
                "sustain_time_sec", "sustain_thrust", "sustain_mass_kg",
            },
        ),
    ]

    def test_lexical_terms_non_empty_for_all_missile_passes(self):
        """PassRetrievalSignals.lexical_terms must be non-empty for every missile pass."""
        from app.services.extraction_query_builder import build_retrieval_profile

        for mod_path, cls_name, _ in self._MISSILE_PASSES:
            cls = _load_pass_cls(mod_path, cls_name)
            profile = build_retrieval_profile(None, cls)
            assert len(profile.lexical_terms) > 0, (
                f"{cls_name}: lexical_terms is empty — retrieval blocks missing or not loaded"
            )

    def test_all_non_identity_fields_have_aliases(self):
        """Every non-identity field on every missile pass must have at least one alias."""
        from app.services.extraction_query_builder import build_retrieval_profile

        for mod_path, cls_name, expected_fields in self._MISSILE_PASSES:
            cls = _load_pass_cls(mod_path, cls_name)
            profile = build_retrieval_profile(None, cls)

            field_query_map = {fq.field_name: fq for fq in profile.field_queries}

            # All expected fields must be present in field_queries
            missing = expected_fields - field_query_map.keys()
            assert not missing, (
                f"{cls_name}: missing FieldRetrievalQuery for fields: {missing}"
            )

            # system_name must be excluded
            assert "system_name" not in field_query_map, (
                f"{cls_name}: system_name (identity field) must not appear in field_queries"
            )

            # Every expected field must have non-empty aliases
            for field_name in expected_fields:
                fq = field_query_map[field_name]
                assert len(fq.aliases) > 0, (
                    f"{cls_name}.{field_name}: aliases is empty — "
                    f"json_schema_extra retrieval block missing or malformed"
                )

    def test_missile_kinematics_aliases_spot_check(self):
        """Spot-check: min_intercept_km has 'minimum range'; max_intercept_km has
        'engagement range'; min_altitude_km has 'altitude floor';
        max_altitude_km has 'ceiling'; max_launch_angle_deg has 'launch angle'."""
        from app.services.extraction_query_builder import build_retrieval_profile

        MissileKinematicsPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics",
            "MissileKinematicsPass",
        )
        profile = build_retrieval_profile(None, MissileKinematicsPass)
        fq_map = {fq.field_name: fq for fq in profile.field_queries}

        assert "minimum range" in fq_map["min_intercept_km"].aliases
        assert "engagement range" in fq_map["max_intercept_km"].aliases
        assert "altitude floor" in fq_map["min_altitude_km"].aliases
        assert "ceiling" in fq_map["max_altitude_km"].aliases
        assert "launch angle" in fq_map["max_launch_angle_deg"].aliases

    def test_missile_guidance_aliases_spot_check(self):
        """Spot-check: guidance_type has 'SARH' and 'command guidance';
        seeker_type has 'homing head'; missile_photo has 'photo'."""
        from app.services.extraction_query_builder import build_retrieval_profile

        MissileGuidancePass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_guidance",
            "MissileGuidancePass",
        )
        profile = build_retrieval_profile(None, MissileGuidancePass)
        fq_map = {fq.field_name: fq for fq in profile.field_queries}

        assert "SARH" in fq_map["guidance_type"].aliases
        assert "command guidance" in fq_map["guidance_type"].aliases
        assert "homing head" in fq_map["seeker_type"].aliases
        assert "photo" in fq_map["missile_photo"].aliases

    def test_missile_propulsion_aliases_spot_check(self):
        """Spot-check: booster_thrust has 'peak thrust'; sustain_thrust has
        'cruise thrust'; ejector_thrust has 'gas generator'."""
        from app.services.extraction_query_builder import build_retrieval_profile

        MissilePropulsionPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_propulsion",
            "MissilePropulsionPass",
        )
        profile = build_retrieval_profile(None, MissilePropulsionPass)
        fq_map = {fq.field_name: fq for fq in profile.field_queries}

        assert "peak thrust" in fq_map["booster_thrust"].aliases
        assert "cruise thrust" in fq_map["sustain_thrust"].aliases
        assert "gas generator" in fq_map["ejector_thrust"].aliases
