"""Plan Task 52 — worker-side extraction-quality classifier.

Three-state aggregate over per-pass yield_status outcomes:

- ``ok``       — at least one domain pass (radar/missile/other/system_links)
                 achieved HIT.
- ``degraded`` — reference pass HIT (document structure extracted) but
                 every domain pass ended non-HIT (EMPTY / BRIDGES_ONLY /
                 DEGRADED / FAILED / SKIPPED). Signals a real document
                 that simply didn't match the SAM/radar ontology.
- ``anomaly``  — no pass achieved HIT at all. Likely an extraction
                 failure or an entirely off-topic document.
"""
from __future__ import annotations

import pytest

from app.workers.pipeline import _classify_extraction_quality


def _outcome(yield_status: str, execution_status: str = "COMPLETE") -> dict:
    return {
        "execution_status": execution_status,
        "yield_status": yield_status,
        "primary_entities_extracted": 0,
        "bridge_entities_extracted": 0,
        "relationships_extracted": 0,
        "relationships_rejected": 0,
    }


def test_ok_when_any_domain_pass_hits():
    """Any domain pass (radar/missile/other_systems/system_links) in HIT
    state → classification is 'ok' regardless of reference outcome."""
    assert _classify_extraction_quality({
        "reference": _outcome("HIT"),
        "radar_domain": _outcome("HIT"),
        "missile_domain": _outcome("EMPTY"),
        "other_systems": _outcome("EMPTY"),
        "system_links": _outcome("EMPTY"),
    }) == "ok"

    # Reference empty, radar hits — still ok.
    assert _classify_extraction_quality({
        "reference": _outcome("EMPTY"),
        "radar_domain": _outcome("HIT"),
    }) == "ok"


def test_degraded_when_reference_hits_but_domains_empty():
    """Reference pass HIT (structure extracted) but every domain pass is
    non-HIT → classification is 'degraded'. This is the distinctive
    signal that differentiates a real-but-off-topic document from a
    processing failure."""
    assert _classify_extraction_quality({
        "reference": _outcome("HIT"),
        "radar_domain": _outcome("EMPTY"),
        "missile_domain": _outcome("BRIDGES_ONLY"),
        "other_systems": _outcome("EMPTY"),
        "system_links": _outcome("EMPTY"),
    }) == "degraded"

    # DEGRADED domain counts as non-HIT — still 'degraded' aggregate.
    assert _classify_extraction_quality({
        "reference": _outcome("HIT"),
        "radar_domain": _outcome("DEGRADED"),
    }) == "degraded"

    # Failed/Skipped domains also count as non-HIT.
    assert _classify_extraction_quality({
        "reference": _outcome("HIT"),
        "radar_domain": _outcome(None, execution_status="FAILED"),
        "missile_domain": _outcome(None, execution_status="SKIPPED"),
    }) == "degraded"


def test_anomaly_when_no_pass_hits():
    """No pass at HIT at all → 'anomaly'. Reference either failed or
    didn't find document structure; likely a processing break or an
    entirely unsupported document format."""
    assert _classify_extraction_quality({
        "reference": _outcome("EMPTY"),
        "radar_domain": _outcome("EMPTY"),
    }) == "anomaly"

    assert _classify_extraction_quality({
        "reference": _outcome(None, execution_status="FAILED"),
    }) == "anomaly"

    # Empty rollup (no rows) → anomaly (conservative default).
    assert _classify_extraction_quality({}) == "anomaly"


def test_classifier_result_included_in_pipeline_run_metrics_blob(monkeypatch):
    """_write_pipeline_run_metrics surfaces the classifier output under
    metrics['extraction_quality']. Existing signals preserved."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    from app.workers.pipeline import _write_pipeline_run_metrics

    fake_run = MagicMock()
    fake_run.metrics = None
    fake_run.ontology_bundle_key = "air_defense_v3"

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.get.return_value = fake_run

    monkeypatch.setattr(
        "app.workers.pipeline._build_pass_outcomes_rollup",
        lambda *_args, **_kw: {
            "reference": _outcome("HIT"),
            "radar_domain": _outcome("EMPTY"),
            "missile_domain": _outcome("EMPTY"),
            "other_systems": _outcome("EMPTY"),
            "system_links": _outcome("EMPTY"),
        },
    )

    merged = SimpleNamespace(edges=[], rejected_edges=[])
    manifest = SimpleNamespace(bundle_key="air_defense_v3")

    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _write_pipeline_run_metrics("run-1", merged, manifest)

    assert fake_run.metrics["extraction_quality"] == "degraded"
    # Existing keys survive.
    assert "pass_outcomes" in fake_run.metrics
    assert "document_extraction_anomaly" in fake_run.metrics
