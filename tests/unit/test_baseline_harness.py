"""Smoke tests for tools/extraction_baseline_harness.py comparison logic."""
import pytest


def test_compare_baseline_met_when_equal():
    from tools.extraction_baseline_harness import compare

    legacy = {
        "aggregate": {
            "entity_counts_by_type": {"RADAR_SYSTEM": 10, "PLATFORM": 5},
            "edge_counts_by_type": {"INSTALLED_ON": 8},
            "yield_distribution": {"radar_domain": {"HIT": 5}},
            "overall_rejection_ratio": 0.1,
        },
    }
    bundle = {
        "aggregate": {
            "entity_counts_by_type": {"RADAR_SYSTEM": 10, "PLATFORM": 5},
            "edge_counts_by_type": {"INSTALLED_ON": 8},
            "yield_distribution": {"radar_domain": {"HIT": 5}},
            "overall_rejection_ratio": 0.1,
        },
    }
    result = compare(legacy, bundle)
    assert result["baseline_met"] is True


def test_compare_baseline_fails_on_entity_regression():
    from tools.extraction_baseline_harness import compare

    legacy = {
        "aggregate": {
            "entity_counts_by_type": {"RADAR_SYSTEM": 100},
            "edge_counts_by_type": {},
            "yield_distribution": {},
            "overall_rejection_ratio": 0.0,
        },
    }
    bundle = {
        "aggregate": {
            "entity_counts_by_type": {"RADAR_SYSTEM": 80},  # 20% drop
            "edge_counts_by_type": {},
            "yield_distribution": {},
            "overall_rejection_ratio": 0.0,
        },
    }
    result = compare(legacy, bundle)
    assert result["baseline_met"] is False
    assert result["criteria"]["entity_extraction_within_10pct"] is False


def test_compare_baseline_fails_on_yield_regression():
    from tools.extraction_baseline_harness import compare

    legacy = {
        "aggregate": {
            "entity_counts_by_type": {},
            "edge_counts_by_type": {},
            "yield_distribution": {"radar_domain": {"HIT": 5}},
            "overall_rejection_ratio": 0.0,
        },
    }
    bundle = {
        "aggregate": {
            "entity_counts_by_type": {},
            "edge_counts_by_type": {},
            "yield_distribution": {"radar_domain": {"EMPTY": 5}},
            "overall_rejection_ratio": 0.0,
        },
    }
    result = compare(legacy, bundle)
    assert result["baseline_met"] is False
    assert len(result["yield_regressions"]) == 1
    assert result["yield_regressions"][0]["pass_name"] == "radar_domain"


def test_compare_baseline_fails_on_rejection_ratio_drift():
    from tools.extraction_baseline_harness import compare

    legacy = {
        "aggregate": {
            "entity_counts_by_type": {},
            "edge_counts_by_type": {},
            "yield_distribution": {},
            "overall_rejection_ratio": 0.05,
        },
    }
    bundle = {
        "aggregate": {
            "entity_counts_by_type": {},
            "edge_counts_by_type": {},
            "yield_distribution": {},
            "overall_rejection_ratio": 0.15,  # 10pp drift > 5pp threshold
        },
    }
    result = compare(legacy, bundle)
    assert result["baseline_met"] is False
    assert result["criteria"]["rejection_ratio_within_5pp"] is False


def test_compare_tolerates_missing_types_in_bundle():
    """A type that exists in legacy but not in bundle is a 100% drop — fails."""
    from tools.extraction_baseline_harness import compare

    legacy = {
        "aggregate": {
            "entity_counts_by_type": {"RADAR_SYSTEM": 10},
            "edge_counts_by_type": {},
            "yield_distribution": {},
            "overall_rejection_ratio": 0.0,
        },
    }
    bundle = {
        "aggregate": {
            "entity_counts_by_type": {},  # missing RADAR_SYSTEM
            "edge_counts_by_type": {},
            "yield_distribution": {},
            "overall_rejection_ratio": 0.0,
        },
    }
    result = compare(legacy, bundle)
    assert result["baseline_met"] is False
