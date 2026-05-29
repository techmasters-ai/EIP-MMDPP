"""Task B5 — RetrievalProfile multi-channel + post-rerank-boost config fields.

Tests written BEFORE fields were added so they fail first, then green after
the implementation. Config-schema only; no behaviour is wired in this task.

Run standalone:
    python3 -m pytest tests/unit/test_retrieval_profile_config.py -v
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _profile(**kwargs):
    from app.services.ontology_bundles import RetrievalProfile
    return RetrievalProfile(**kwargs)


# ---------------------------------------------------------------------------
# 1. Existing 4 fields are still present and unchanged
# ---------------------------------------------------------------------------

class TestExistingFieldsUnchanged:
    """Guard-rail: the 4 original fields must survive the extension."""

    def test_min_similarity_default(self):
        p = _profile()
        assert p.min_similarity == 0.45

    def test_top_n_candidates_default(self):
        p = _profile()
        assert p.top_n_candidates == 50

    def test_top_k_default(self):
        p = _profile()
        assert p.top_k == 20

    def test_fallback_to_full_default(self):
        p = _profile()
        assert p.fallback_to_full is True

    def test_existing_fields_roundtrip(self):
        p = _profile(
            min_similarity=0.55,
            top_n_candidates=100,
            top_k=10,
            fallback_to_full=False,
        )
        assert p.min_similarity == 0.55
        assert p.top_n_candidates == 100
        assert p.top_k == 10
        assert p.fallback_to_full is False


# ---------------------------------------------------------------------------
# 2. New fields — defaults (construct WITHOUT setting new fields)
# ---------------------------------------------------------------------------

class TestNewFieldDefaults:
    """Every new field must take its documented default when omitted."""

    # channel sizing / limits
    def test_field_query_top_k_default(self):
        assert _profile().field_query_top_k == 8

    def test_lexical_top_k_default(self):
        assert _profile().lexical_top_k == 40

    def test_pattern_hit_limit_default(self):
        assert _profile().pattern_hit_limit == 50

    def test_fallback_min_field_coverage_default(self):
        assert _profile().fallback_min_field_coverage == 1

    def test_fallback_similarity_relaxation_default(self):
        assert _profile().fallback_similarity_relaxation == pytest.approx(0.07)

    # post-rerank weights
    def test_rerank_weight_default(self):
        assert _profile().rerank_weight == pytest.approx(1.0)

    def test_lexical_weight_default(self):
        assert _profile().lexical_weight == pytest.approx(0.20)

    def test_pattern_weight_default(self):
        assert _profile().pattern_weight == pytest.approx(0.15)

    def test_section_weight_default(self):
        assert _profile().section_weight == pytest.approx(0.10)

    def test_table_boost_default(self):
        assert _profile().table_boost == pytest.approx(0.08)

    def test_negative_weight_default(self):
        assert _profile().negative_weight == pytest.approx(0.20)

    # subset-schema extraction (opt-in)
    def test_subset_schema_extraction_default(self):
        assert _profile().subset_schema_extraction is False


# ---------------------------------------------------------------------------
# 3. New fields — round-trip (construct WITH non-default values)
# ---------------------------------------------------------------------------

class TestNewFieldsRoundtrip:
    """Setting a new field must persist to the returned model attribute."""

    def test_channel_sizing_roundtrip(self):
        p = _profile(
            field_query_top_k=12,
            lexical_top_k=60,
            pattern_hit_limit=100,
            fallback_min_field_coverage=2,
            fallback_similarity_relaxation=0.10,
        )
        assert p.field_query_top_k == 12
        assert p.lexical_top_k == 60
        assert p.pattern_hit_limit == 100
        assert p.fallback_min_field_coverage == 2
        assert p.fallback_similarity_relaxation == pytest.approx(0.10)

    def test_rerank_weights_roundtrip(self):
        p = _profile(
            rerank_weight=0.8,
            lexical_weight=0.30,
            pattern_weight=0.25,
            section_weight=0.05,
            table_boost=0.12,
            negative_weight=0.15,
        )
        assert p.rerank_weight == pytest.approx(0.8)
        assert p.lexical_weight == pytest.approx(0.30)
        assert p.pattern_weight == pytest.approx(0.25)
        assert p.section_weight == pytest.approx(0.05)
        assert p.table_boost == pytest.approx(0.12)
        assert p.negative_weight == pytest.approx(0.15)

    def test_subset_schema_extraction_opt_in(self):
        p = _profile(subset_schema_extraction=True)
        assert p.subset_schema_extraction is True

    def test_all_new_fields_together(self):
        """All 13 new fields set simultaneously — full round-trip."""
        p = _profile(
            field_query_top_k=15,
            lexical_top_k=80,
            pattern_hit_limit=75,
            fallback_min_field_coverage=3,
            fallback_similarity_relaxation=0.05,
            rerank_weight=0.9,
            lexical_weight=0.25,
            pattern_weight=0.18,
            section_weight=0.12,
            table_boost=0.09,
            negative_weight=0.22,
            subset_schema_extraction=True,
        )
        assert p.field_query_top_k == 15
        assert p.lexical_top_k == 80
        assert p.pattern_hit_limit == 75
        assert p.fallback_min_field_coverage == 3
        assert p.fallback_similarity_relaxation == pytest.approx(0.05)
        assert p.rerank_weight == pytest.approx(0.9)
        assert p.lexical_weight == pytest.approx(0.25)
        assert p.pattern_weight == pytest.approx(0.18)
        assert p.section_weight == pytest.approx(0.12)
        assert p.table_boost == pytest.approx(0.09)
        assert p.negative_weight == pytest.approx(0.22)
        assert p.subset_schema_extraction is True


# ---------------------------------------------------------------------------
# 4. extra="forbid" is still enforced
# ---------------------------------------------------------------------------

class TestExtraForbidStillEnforced:
    """Misspelled or unknown keys must raise ValidationError."""

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError):
            _profile(totally_unknown_key=99)

    def test_misspelled_existing_field_raises(self):
        """'min_similarty' (typo) must be caught, not silently ignored."""
        with pytest.raises(ValidationError):
            _profile(min_similarty=0.5)

    def test_misspelled_new_field_raises(self):
        """'rerank_wieght' (typo) must be caught."""
        with pytest.raises(ValidationError):
            _profile(rerank_wieght=0.5)


# ---------------------------------------------------------------------------
# 5. Type coercion sanity — ints must be ints, floats floats, bool bool
# ---------------------------------------------------------------------------

class TestTypeCoercion:
    """Pydantic should coerce compatible scalar literals correctly."""

    def test_field_query_top_k_is_int(self):
        p = _profile(field_query_top_k=10)
        assert isinstance(p.field_query_top_k, int)

    def test_lexical_weight_is_float(self):
        p = _profile(lexical_weight=0.5)
        assert isinstance(p.lexical_weight, float)

    def test_subset_schema_extraction_is_bool(self):
        p = _profile(subset_schema_extraction=False)
        assert isinstance(p.subset_schema_extraction, bool)


# ---------------------------------------------------------------------------
# 6. Manifest-level integration: new fields absent from real manifests use defaults
# ---------------------------------------------------------------------------

class TestManifestIntegration:
    """Real bundle manifests don't set the new fields — they must load cleanly
    and yield all-default values for the new fields on every pass that has a
    retrieval_profile block.
    """

    @pytest.mark.parametrize("bundle_key", [
        "air_defense_v3",
        "air_defense_v3_baseline_subset",
    ])
    def test_existing_manifests_load_without_error(self, bundle_key):
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest(bundle_key)
        assert m is not None

    @pytest.mark.parametrize("bundle_key", [
        "air_defense_v3",
        "air_defense_v3_baseline_subset",
    ])
    def test_retrieval_profile_new_fields_take_defaults(self, bundle_key):
        from app.services.ontology_bundles import load_bundle_manifest
        m = load_bundle_manifest(bundle_key)
        for pass_def in m.passes:
            rp = pass_def.retrieval  # field name on PassManifest is 'retrieval'
            if rp is None:
                continue
            # channel sizing
            assert rp.field_query_top_k == 8
            assert rp.lexical_top_k == 40
            assert rp.pattern_hit_limit == 50
            assert rp.fallback_min_field_coverage == 1
            assert rp.fallback_similarity_relaxation == pytest.approx(0.07)
            # post-rerank weights
            assert rp.rerank_weight == pytest.approx(1.0)
            assert rp.lexical_weight == pytest.approx(0.20)
            assert rp.pattern_weight == pytest.approx(0.15)
            assert rp.section_weight == pytest.approx(0.10)
            assert rp.table_boost == pytest.approx(0.08)
            assert rp.negative_weight == pytest.approx(0.20)
            # subset schema
            assert rp.subset_schema_extraction is False
