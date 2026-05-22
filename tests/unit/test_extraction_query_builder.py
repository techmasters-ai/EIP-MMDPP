"""C.2b — snapshot tests for build_retrieval_query.

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

        Expected text frozen from RadarPowerRfPass / RadarPowerRfRecord at
        commit bdde417 field shapes. Update this string only if you intend to
        change retrieval query content.
        """
        from app.services.extraction_query_builder import build_retrieval_query

        RadarPowerRfPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
        )
        expected = (
            "Subset of RadarSystemEntity covering RF carrier + transmit power.\n"
            "Effective Radiated Power in dBW. Source labels such as "
            "'ERP' or 'Effective Radiated Power' map here only when the "
            "source unit is dBW/dBm. Emit only when the source states "
            "the value with units; otherwise null. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions.\n"
            "Transmitter peak power in kilowatts. Source labels such as "
            "'Peak Power', 'Transmitter Power', 'Tx Power', or 'Pulse "
            "Power' map here when they describe peak transmitter power. See Unit "
            "Policy in DELTA_SYSTEM_PROMPT for conversions.\n"
            "Nominal carrier frequency in MHz. Source labels such as "
            "'Frequency', 'Operating Frequency', 'Carrier Frequency', "
            "or 'RF' map here when they describe the radar carrier. See Unit "
            "Policy in DELTA_SYSTEM_PROMPT for conversions."
        )
        result = build_retrieval_query(None, RadarPowerRfPass)
        assert result == expected


# ---------------------------------------------------------------------------
# 2. missile_kinematics snapshot
# ---------------------------------------------------------------------------

class TestMissileKinematicsQuerySnapshot:

    def test_missile_kinematics_query_snapshot(self):
        """Snapshot: missile_kinematics query text must match exactly.

        Expected text frozen from MissileKinematicsPass / MissileKinematicsRecord
        at commit bdde417 field shapes.
        """
        from app.services.extraction_query_builder import build_retrieval_query

        MissileKinematicsPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics",
            "MissileKinematicsPass",
        )
        expected = (
            "Subset of MissileSystemEntity covering engagement envelope.\n"
            "Minimum intercept range in kilometers. Source labels such as "
            "'Min Range' or 'minimum range' map here when they describe "
            "the missile variant. See Unit Policy in DELTA_SYSTEM_PROMPT "
            "for conversions.\n"
            "Maximum intercept range in kilometers. Source labels such as "
            "'Range', 'Max Range', 'maximum range', 'effective range', "
            "or 'engagement range' map here when they describe the "
            "missile variant. See Unit Policy in DELTA_SYSTEM_PROMPT for "
            "conversions.\n"
            "Minimum engagement altitude in kilometers. Source labels such as "
            "'Min Altitude', 'Min Alt', 'minimum altitude', or 'floor' map here.\n"
            "Maximum engagement altitude in kilometers. Source labels such as "
            "'Altitude', 'Max Altitude', 'Max Alt', 'ceiling', or "
            "'engagement altitude' map here when they describe the missile "
            "variant.\n"
            "Maximum launch angle in degrees."
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
        """Conservative rev-12-H1 defaults must be exactly as specified."""
        from app.services.ontology_bundles import load_bundle_manifest

        m = load_bundle_manifest("air_defense_v3")
        rp_pass = m.find_pass("radar_power_rf")
        r = rp_pass.retrieval
        assert r is not None
        assert r.min_similarity == pytest.approx(0.45)
        assert r.top_n_candidates == 50
        assert r.top_k == 20
        assert r.fallback_to_full is True


# ---------------------------------------------------------------------------
# 6. RetrievalProfile validation
# ---------------------------------------------------------------------------

class TestRetrievalProfileValidation:

    def test_defaults(self):
        """RetrievalProfile() with no args uses conservative defaults."""
        from app.services.ontology_bundles import RetrievalProfile

        r = RetrievalProfile()
        assert r.min_similarity == pytest.approx(0.45)
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
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(top_n_candidates=501)

    def test_top_k_zero_raises(self):
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(top_k=0)

    def test_top_k_above_max_raises(self):
        from app.services.ontology_bundles import RetrievalProfile
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RetrievalProfile(top_k=201)

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
