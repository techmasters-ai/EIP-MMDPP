"""Plan Task 36 Step 2.authtest — counts_authoritative lifecycle regression.

Locks in the pre-merge → post-merge count lifecycle across BOTH projection
surfaces: the StageRun top-level columns (relationships_extracted,
relationships_rejected) AND the StageRun.metrics JSONB block
(counts_authoritative + relationships_extracted + relationships_rejected +
rejection_sample + rejections_by_reason). An XOR of the two surfaces is a
regression even if each individually matches an expected value.

Also covers the three relationships-only (system_links) sub-cases:
EMPTY promotion (0 accepted, <4 total), DEGRADED promotion (≥4 total with
≥75% rejected), HIT retained (low rejection ratio).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction_merge import (
    LogicalIdentity,
    MergedEdgeRecord,
    MergedExtraction,
    PerPassEdgeMetrics,
)
from app.workers.pipeline import _apply_post_merge_yield_updates


# --- Fixtures --------------------------------------------------------------

def _id(entity_type: str, *values: str) -> LogicalIdentity:
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=("name",),
        identity_tuple=tuple(values),
        scope="document",
        document_id="doc-1",
    )


def _manifest(pass_kind_by_name: dict | None = None):
    mapping = dict(pass_kind_by_name or {})
    def _find(pname):
        kind = mapping.get(pname, "entities_and_relationships")
        return SimpleNamespace(name=pname, kind=kind)
    return SimpleNamespace(find_pass=_find, passes=[])


def _stage_row(
    *,
    pass_name: str,
    yield_status: str = "HIT",
    primary: int = 3,
    bridge: int = 0,
    initial_metrics: dict | None = None,
):
    row = MagicMock()
    row.pass_name = pass_name
    row.yield_status = yield_status
    row.primary_entities_extracted = primary
    row.bridge_entities_extracted = bridge
    row.relationships_extracted = 0
    row.relationships_rejected = 0
    row.metrics = initial_metrics or {
        "counts_authoritative": False,
        "relationships_extracted": 0,
        "relationships_rejected": 0,
        "rejection_sample": [],
        "rejections_by_reason": {},
    }
    row.status = "COMPLETE"
    row.execution_status = "COMPLETE"
    return row


def _session_returning(rows):
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.all.return_value = rows
    return session


# --- Step 2.authtest: typed-edge pass lifecycle ---------------------------

def test_post_merge_writes_all_5_jsonb_keys_and_top_level_columns_in_lockstep():
    """After _apply_post_merge_yield_updates, the StageRun row reflects
    the PerPassEdgeMetrics entry on BOTH surfaces — JSONB and top-level
    columns — in exact lockstep. Row status stays COMPLETE."""
    row = _stage_row(pass_name="radar_domain", yield_status="HIT")
    merged = MergedExtraction(
        entities=[],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "radar_domain": PerPassEdgeMetrics(
                attempted=5,
                accepted=3,
                rejected=2,
                rejection_sample=[{"rel_type": "X"}, {"rel_type": "Y"}],
                rejections_by_reason={"invalid_triple": 2},
            ),
        },
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates("run-1", merged, _manifest())

    # Top-level columns
    assert row.relationships_extracted == 3
    assert row.relationships_rejected == 2

    # JSONB block — all 5 keys authoritative
    assert row.metrics["counts_authoritative"] is True
    assert row.metrics["relationships_extracted"] == 3
    assert row.metrics["relationships_rejected"] == 2
    assert row.metrics["rejection_sample"] == [{"rel_type": "X"}, {"rel_type": "Y"}]
    assert row.metrics["rejections_by_reason"] == {"invalid_triple": 2}

    # Lockstep: the two projections MUST equal each other.
    assert row.relationships_extracted == row.metrics["relationships_extracted"]
    assert row.relationships_rejected == row.metrics["relationships_rejected"]

    # row.status / row.execution_status unchanged.
    assert row.status == "COMPLETE"
    assert row.execution_status == "COMPLETE"


def test_post_merge_is_idempotent_on_second_call():
    """A second _apply_post_merge_yield_updates call against a row with
    counts_authoritative=True already MUST NOT double-count — the writer
    overwrites (not accumulates) all 5 keys."""
    row = _stage_row(pass_name="radar_domain", yield_status="HIT")
    merged = MergedExtraction(
        entities=[],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "radar_domain": PerPassEdgeMetrics(attempted=5, accepted=3, rejected=2),
        },
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates("run-1", merged, _manifest())
        _apply_post_merge_yield_updates("run-1", merged, _manifest())

    assert row.relationships_extracted == 3
    assert row.relationships_rejected == 2
    assert row.metrics["relationships_extracted"] == 3
    assert row.metrics["relationships_rejected"] == 2


# --- Relationships-only branch: EMPTY / DEGRADED / HIT --------------------

def test_relationships_only_empty_promotion():
    """3 DTOs, all rejected (accepted=0, rejected=3, total_rels=3 < 4):
    pre-merge yield_status was HIT (from provisional non-zero edges);
    post-merge authoritative classifier returns EMPTY — yield_status
    overwritten unconditionally for relationships_only passes."""
    row = _stage_row(pass_name="system_links", yield_status="HIT",
                     primary=0, bridge=0)
    merged = MergedExtraction(
        entities=[],
        edges=[],
        rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "system_links": PerPassEdgeMetrics(
                attempted=3, accepted=0, rejected=3,
                rejection_sample=[{"rel_type": "ASSOCIATED_WITH"}] * 3,
                rejections_by_reason={"invalid_triple": 3},
            ),
        },
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates(
            "run-1", merged, _manifest({"system_links": "relationships_only"}),
        )

    assert row.yield_status == "EMPTY"
    assert row.relationships_extracted == 0
    assert row.relationships_rejected == 3
    assert row.metrics["counts_authoritative"] is True
    assert row.metrics["relationships_extracted"] == 0
    assert row.metrics["relationships_rejected"] == 3
    # Lockstep across the two surfaces.
    assert row.relationships_extracted == row.metrics["relationships_extracted"]


def test_relationships_only_degraded_promotion():
    """4 DTOs, all rejected (accepted=0, rejected=4, total_rels=4,
    rejection ratio=1.0 ≥ 0.75): classifier returns DEGRADED."""
    row = _stage_row(pass_name="system_links", yield_status="HIT",
                     primary=0, bridge=0)
    merged = MergedExtraction(
        entities=[], edges=[], rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "system_links": PerPassEdgeMetrics(
                attempted=4, accepted=0, rejected=4,
            ),
        },
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates(
            "run-1", merged, _manifest({"system_links": "relationships_only"}),
        )

    assert row.yield_status == "DEGRADED"
    assert row.relationships_extracted == 0
    assert row.relationships_rejected == 4


def test_relationships_only_hit_retained():
    """4 DTOs, 3 accepted + 1 rejected (total_rels=4, ratio=0.25 < 0.75):
    classifier returns HIT — overwrites unconditionally to the same value."""
    row = _stage_row(pass_name="system_links", yield_status="HIT",
                     primary=0, bridge=0)
    merged = MergedExtraction(
        entities=[], edges=[], rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "system_links": PerPassEdgeMetrics(
                attempted=4, accepted=3, rejected=1,
            ),
        },
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates(
            "run-1", merged, _manifest({"system_links": "relationships_only"}),
        )

    assert row.yield_status == "HIT"
    assert row.relationships_extracted == 3
    assert row.relationships_rejected == 1


# --- Typed-edge pass yield preservation ------------------------------------

def test_entity_bearing_pass_hit_to_degraded_when_ratio_high():
    """Entity-bearing (kind!=relationships_only) passes keep the existing
    HIT → DEGRADED rule guarded on yield_status=='HIT'. Same threshold as
    pre-Phase-5 behavior."""
    row = _stage_row(pass_name="radar_domain", yield_status="HIT",
                     primary=3, bridge=0)
    merged = MergedExtraction(
        entities=[], edges=[], rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "radar_domain": PerPassEdgeMetrics(
                attempted=10, accepted=1, rejected=9,
            ),
        },
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates("run-1", merged, _manifest())

    assert row.yield_status == "DEGRADED"


def test_entity_bearing_pass_empty_is_not_promoted():
    """Entity-bearing passes: EMPTY pre-merge stays EMPTY post-merge
    regardless of per_pass_edge_metrics — only HIT → DEGRADED is allowed."""
    row = _stage_row(pass_name="radar_domain", yield_status="EMPTY",
                     primary=0, bridge=0)
    merged = MergedExtraction(
        entities=[], edges=[], rejected_edges=[],
        rejections_by_pass={},
        pipeline_run_id="run-1",
        document_id="doc-1",
        per_pass_edge_metrics={
            "radar_domain": PerPassEdgeMetrics(attempted=3, accepted=3, rejected=0),
        },
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates("run-1", merged, _manifest())

    assert row.yield_status == "EMPTY"


# --- Fallback: per_pass_edge_metrics empty (interim builds) ----------------

def test_fallback_when_per_pass_edge_metrics_empty_uses_merged_edges():
    """Interim-state / test fixtures without per_pass_edge_metrics: the
    writer derives accepted/rejected from merged.edges / merged.rejected_edges
    grouped by pass name. Still writes all 5 authoritative keys."""
    row = _stage_row(pass_name="radar_domain", yield_status="HIT",
                     primary=3, bridge=0)
    edge = MergedEdgeRecord(
        from_identity=_id("RADAR_SYSTEM", "R1"),
        to_identity=_id("PLATFORM", "P1"),
        rel_type="INSTALLED_ON",
        confidence=0.9,
        pass_origins={"radar_domain"},
    )
    merged = MergedExtraction(
        entities=[],
        edges=[edge],
        rejected_edges=[
            ("radar_domain", SimpleNamespace(),
             SimpleNamespace(value="invalid_triple")),
        ],
        rejections_by_pass={"radar_domain": 1},
        pipeline_run_id="run-1",
        document_id="doc-1",
        # per_pass_edge_metrics intentionally empty.
    )
    session = _session_returning([row])
    with patch("app.workers.pipeline.get_sync_session", return_value=session):
        _apply_post_merge_yield_updates("run-1", merged, _manifest())

    assert row.relationships_extracted == 1
    assert row.relationships_rejected == 1
    assert row.metrics["counts_authoritative"] is True
    assert row.metrics["relationships_extracted"] == 1
    assert row.metrics["relationships_rejected"] == 1
    assert row.metrics["rejection_sample"] == []
    # Fallback path still routes rejections through _build_rejections_by_reason.
    assert row.metrics["rejections_by_reason"].get("invalid_triple") == 1
