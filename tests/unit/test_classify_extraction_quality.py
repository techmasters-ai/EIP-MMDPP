"""E-9 — worker-side extraction-quality classifier.

Post-docs-alignment rewrite (spec §6.8). The LLM `reference` pass was
deleted; the deterministic Docling anchor walker (D-3/D-4) emits
SECTION vertices directly. The classifier now keys "degraded" on the
graph signal (SECTION + TextChunk counts) instead of a reference-pass
HIT.

Three states:

- ``ok``       — at least one domain pass (radar (any sub-pass) /
                 missile (any sub-pass) / system_links) achieved HIT.
- ``degraded`` — no domain HIT, but SECTION vertices and TextChunks
                 both exist. Real document processed through
                 derive_document_anchors and chunking, but nothing
                 matched the SAM/radar ontology.
- ``anomaly``  — no domain HIT and either no SECTION vertices or no
                 TextChunks. Upstream processing failure or
                 unsupported document shape.
"""
from __future__ import annotations

import pytest

from app.workers.pipeline import _classify_extraction_quality


def _outcome(yield_status: str | None, execution_status: str = "COMPLETE") -> dict:
    return {
        "execution_status": execution_status,
        "yield_status": yield_status,
        "primary_entities_extracted": 0,
        "bridge_entities_extracted": 0,
        "relationships_extracted": 0,
        "relationships_rejected": 0,
    }


def test_ok_when_any_domain_pass_hits():
    """Any domain pass in HIT state → 'ok' regardless of graph signals."""
    assert _classify_extraction_quality(
        {
            "radar_identity": _outcome("HIT"),
            "missile_identity": _outcome("EMPTY"),
            "system_links": _outcome("EMPTY"),
        },
        section_count=0,
        text_chunk_count=0,
    ) == "ok"

    # Radar hit dominates even when graph signals are zero.
    assert _classify_extraction_quality(
        {"radar_identity": _outcome("HIT")},
        section_count=0,
        text_chunk_count=0,
    ) == "ok"


def test_degraded_when_sections_and_chunks_present_but_no_domain_hit():
    """Graph signals (SECTION + TextChunk) > 0 AND no domain HIT → 'degraded'."""
    assert _classify_extraction_quality(
        {
            "radar_identity": _outcome("EMPTY"),
            "missile_identity": _outcome("BRIDGES_ONLY"),
            "system_links": _outcome("EMPTY"),
        },
        section_count=5,
        text_chunk_count=42,
    ) == "degraded"

    # Minimum positive signal on both counts is enough.
    assert _classify_extraction_quality(
        {"radar_identity": _outcome("DEGRADED")},
        section_count=1,
        text_chunk_count=1,
    ) == "degraded"

    # FAILED / SKIPPED domain outcomes don't disqualify degraded status.
    assert _classify_extraction_quality(
        {
            "radar_identity": _outcome(None, execution_status="FAILED"),
            "missile_identity": _outcome(None, execution_status="SKIPPED"),
        },
        section_count=3,
        text_chunk_count=10,
    ) == "degraded"


def test_anomaly_when_no_domain_hit_and_no_graph_signal():
    """No domain HIT AND (section_count == 0 OR text_chunk_count == 0) → 'anomaly'."""
    # Both signals zero.
    assert _classify_extraction_quality(
        {"radar_identity": _outcome("EMPTY")},
        section_count=0,
        text_chunk_count=0,
    ) == "anomaly"

    # Only sections present — chunks missing.
    assert _classify_extraction_quality(
        {"radar_identity": _outcome("EMPTY")},
        section_count=5,
        text_chunk_count=0,
    ) == "anomaly"

    # Only chunks present — sections missing.
    assert _classify_extraction_quality(
        {"radar_identity": _outcome("EMPTY")},
        section_count=0,
        text_chunk_count=5,
    ) == "anomaly"

    # Empty outcomes rollup + zero signals → anomaly.
    assert _classify_extraction_quality({}, section_count=0, text_chunk_count=0) == "anomaly"


def test_classifier_result_included_in_pipeline_run_metrics_blob(monkeypatch):
    """_write_pipeline_run_metrics surfaces the classifier output under
    metrics['extraction_quality']. Existing signals preserved; new
    metrics keys section_count / text_chunk_count are populated."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    from app.workers.pipeline import _write_pipeline_run_metrics

    fake_run = MagicMock()
    fake_run.metrics = None
    fake_run.ontology_bundle_key = "air_defense_v3"
    fake_run.document_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.get.return_value = fake_run
    # TextChunk count query returns 12.
    session.execute.return_value.scalar_one.return_value = 12

    monkeypatch.setattr(
        "app.workers.pipeline._build_pass_outcomes_rollup",
        lambda *_args, **_kw: {
            "radar_identity": _outcome("EMPTY"),
            "missile_identity": _outcome("EMPTY"),
            "system_links": _outcome("EMPTY"),
        },
    )
    # Graph store returns 4 SECTION vertices for the run's document.
    graph_store = MagicMock()
    graph_store.count_ontology_nodes_sync = MagicMock(return_value=4)
    monkeypatch.setattr("app.workers.pipeline.get_graph_store", lambda: graph_store)

    merged = SimpleNamespace(edges=[], rejected_edges=[])
    manifest = SimpleNamespace(bundle_key="air_defense_v3")

    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _write_pipeline_run_metrics("run-1", merged, manifest)

    # 4 SECTIONs + 12 TextChunks → degraded (no domain HIT, positive graph).
    assert fake_run.metrics["extraction_quality"] == "degraded"
    assert fake_run.metrics["section_count"] == 4
    assert fake_run.metrics["text_chunk_count"] == 12
    assert "pass_outcomes" in fake_run.metrics
