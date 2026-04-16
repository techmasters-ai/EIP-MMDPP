"""Shape tests for plan Task 36 dataclass changes:
- MergedEdgeRecord.source_pass: str → pass_origins: set[str]
- New PerPassEdgeMetrics dataclass
- MergedExtraction.per_pass_edge_metrics: dict[str, PerPassEdgeMetrics]
"""
from __future__ import annotations

import pytest

from app.services.extraction_merge import (
    LogicalIdentity,
    MergedEdgeRecord,
    MergedExtraction,
    PerPassEdgeMetrics,
)


def _id(entity_type: str, *values: str) -> LogicalIdentity:
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=("name",),
        identity_tuple=tuple(values),
        scope="document",
        document_id="doc-1",
    )


def test_merged_edge_record_carries_pass_origins_set():
    edge = MergedEdgeRecord(
        from_identity=_id("RADAR_SYSTEM", "R1"),
        to_identity=_id("ANTENNA", "A1"),
        rel_type="HAS_ANTENNA",
        confidence=0.9,
        pass_origins={"radar_domain"},
    )
    assert edge.pass_origins == {"radar_domain"}
    assert not hasattr(edge, "source_pass"), "source_pass field was replaced by pass_origins"


def test_merged_edge_record_pass_origins_is_mutable_set():
    """pass_origins is a set so the cross-pass reducer can union contributors."""
    edge = MergedEdgeRecord(
        from_identity=_id("RADAR_SYSTEM", "R1"),
        to_identity=_id("ANTENNA", "A1"),
        rel_type="HAS_ANTENNA",
        confidence=0.9,
        pass_origins={"radar_domain"},
    )
    edge.pass_origins.add("system_links")
    assert edge.pass_origins == {"radar_domain", "system_links"}


def test_per_pass_edge_metrics_has_expected_fields_with_defaults():
    m = PerPassEdgeMetrics()
    assert m.attempted == 0
    assert m.accepted == 0
    assert m.rejected == 0
    assert m.rejection_sample == []
    assert m.rejections_by_reason == {}


def test_per_pass_edge_metrics_populated_fields():
    m = PerPassEdgeMetrics(
        attempted=5,
        accepted=3,
        rejected=2,
        rejection_sample=[{"rel_type": "X"}, {"rel_type": "Y"}],
        rejections_by_reason={"invalid_triple": 2},
    )
    assert m.attempted == 5
    assert m.accepted == 3
    assert m.rejected == 2
    assert len(m.rejection_sample) == 2
    assert m.rejections_by_reason["invalid_triple"] == 2


def test_merged_extraction_has_per_pass_edge_metrics_defaulting_to_empty_dict():
    merged = MergedExtraction(
        entities=[],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
    )
    assert merged.per_pass_edge_metrics == {}


def test_merged_extraction_per_pass_edge_metrics_carries_per_pass_entries():
    merged = MergedExtraction(
        entities=[],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "radar_domain": PerPassEdgeMetrics(attempted=3, accepted=3, rejected=0),
            "system_links": PerPassEdgeMetrics(attempted=2, accepted=1, rejected=1),
        },
    )
    assert merged.per_pass_edge_metrics["radar_domain"].accepted == 3
    assert merged.per_pass_edge_metrics["system_links"].rejected == 1
