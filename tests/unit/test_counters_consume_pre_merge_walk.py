"""Tests for plan Task 35c: classify_yield + _count_pass_output consume
PreMergeWalkSummary; pre-merge relationships_rejected is always 0.

- Nested entities behind edge_label fields are counted via the shared
  summary (by way of iter_entities_of_type's _cached_entities path —
  which prefers pre_merge_walk.entities).
- Walker runs EXACTLY once per PassResult across the full pre-merge
  phase (_build_pre_merge_walk_summary + classify_yield + _count_pass_output).
- relationships_rejected is forced to 0 at pre-merge; post-merge
  _apply_post_merge_yield_updates is the authoritative source (Task 36).
- relationships_extracted at pre-merge is pre_merge_walk.raw_edge_count —
  captures typed-edge walker emissions for entity-bearing passes AND the
  DTO-list length for system_links (both feed the classifier uniformly).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

import app.services.extraction_merge as extraction_merge
import app.workers.pipeline as pipeline
from app.services.extraction_merge import (
    ExtractionMetadata,
    PassResult,
    PreMergeWalkSummary,
    RelationshipRejectionReason,
    YieldStatus,
    classify_yield,
)
from app.workers.pipeline import _build_pre_merge_walk_summary, _count_pass_output


# --- Fixtures --------------------------------------------------------------

def _edge(label: str, **field_kwargs):
    extra = field_kwargs.pop("json_schema_extra", None) or {}
    extra["edge_label"] = label
    return Field(json_schema_extra=extra, **field_kwargs)


class _AntennaEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="ANTENNA",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )
    name: str = Field(...)


class _RadarSystemEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="document",
        is_entity=True,
    )
    system_name: str = Field(...)
    antennas: List[_AntennaEntity] = _edge(label="HAS_ANTENNA", default_factory=list)


class _RadarDomainPass(BaseModel):
    model_config = ConfigDict()
    radar_systems: List[_RadarSystemEntity] = Field(default_factory=list)


ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "document"},
        {"name": "ANTENNA", "identity_fields": ["name"], "identity_scope": "document"},
    ],
}


class _PassDef:
    def __init__(
        self,
        name: str,
        primary: List[str],
        bridge: List[str] | None = None,
        kind: str = "entities_and_relationships",
    ):
        self.name = name
        self.primary_entity_types = primary
        self.bridge_entity_types = bridge or []
        self.kind = kind


def _make_pass_result(template, pass_name="radar_domain", pre_merge_rejections=None):
    return PassResult(
        pass_name=pass_name,
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=pre_merge_rejections or [],
    )


# --- Step 1(a): nested antennas included in primary count -----------------

def test_a_count_pass_output_includes_nested_antennas():
    """_count_pass_output's primary_entities_extracted counts nested antennas
    when ANTENNA is declared as a primary entity type for the pass."""
    template = _RadarDomainPass(
        radar_systems=[
            _RadarSystemEntity(
                system_name="R1",
                antennas=[_AntennaEntity(name="A1"), _AntennaEntity(name="A2")],
            ),
        ],
    )
    pr = _make_pass_result(template)
    pr.pre_merge_walk = _build_pre_merge_walk_summary(
        pr, _PassDef("radar_domain", primary=[]), ONTOLOGY, document_id="doc-1",
    )
    counts = _count_pass_output(
        pr, _PassDef("radar_domain", primary=["RADAR_SYSTEM", "ANTENNA"]), ONTOLOGY,
    )
    assert counts["primary_entities_extracted"] == 3  # 1 radar + 2 antennas


# --- Step 1(b): walker-invocation counter ---------------------------------

def test_b_walker_runs_exactly_once_across_full_pre_merge_phase(monkeypatch):
    """Patched walk_entity_graph increments a counter on each TOP-LEVEL
    invocation (not recursive calls). Run the full pre-merge phase for one
    PassResult: the walker is entered exactly ONCE —
    _build_pre_merge_walk_summary does the one entry; classify_yield and
    _count_pass_output each reuse pre_merge_walk.entities via the
    _cached_entities path, without re-entering the walker."""
    import sys
    counter = {"top_level": 0}
    real_walker = extraction_merge.walk_entity_graph

    def _counting(*args, **kwargs):
        # Count only top-level calls (caller is NOT walk_entity_graph itself).
        caller = sys._getframe(1).f_code.co_name
        if caller != "walk_entity_graph" and caller != "_counting":
            counter["top_level"] += 1
        return real_walker(*args, **kwargs)

    # Patch both bindings — pipeline imports walk_entity_graph into its namespace.
    monkeypatch.setattr(extraction_merge, "walk_entity_graph", _counting)
    monkeypatch.setattr(pipeline, "walk_entity_graph", _counting)

    template = _RadarDomainPass(
        radar_systems=[
            _RadarSystemEntity(system_name="R1", antennas=[_AntennaEntity(name="A1")]),
        ],
    )
    pr = _make_pass_result(template)
    pass_def = _PassDef("radar_domain", primary=["RADAR_SYSTEM", "ANTENNA"])

    # Full pre-merge phase: build summary, classify yield, count output.
    pr.pre_merge_walk = _build_pre_merge_walk_summary(
        pr, pass_def, ONTOLOGY, document_id="doc-1",
    )
    _ = classify_yield(pr, pass_def, ONTOLOGY)
    _ = _count_pass_output(pr, pass_def, ONTOLOGY)

    assert counter["top_level"] == 1, (
        f"walker entered {counter['top_level']} times at top level; "
        "expected exactly 1 (pre_merge_walk.entities should satisfy "
        "classify_yield and _count_pass_output without re-entering)"
    )


# --- Step 1(c): pre-merge rejected always 0 -------------------------------

def test_c_count_pass_output_rejected_forced_to_zero_at_pre_merge():
    """Even if pre_merge_rejections is populated (legacy field still
    carried), the pre-merge row reports relationships_rejected == 0.
    Post-merge _apply_post_merge_yield_updates is the authoritative source."""
    template = _RadarDomainPass(radar_systems=[])
    # Populate pre_merge_rejections to verify the counter ignores it.
    pr = _make_pass_result(
        template,
        pre_merge_rejections=[
            (SimpleNamespace(rel_type="X"), RelationshipRejectionReason.INVALID_TRIPLE),
            (SimpleNamespace(rel_type="Y"), RelationshipRejectionReason.UNKNOWN_REF_ID),
        ],
    )
    pr.pre_merge_walk = PreMergeWalkSummary(entities=[], raw_edge_count=0)
    counts = _count_pass_output(pr, _PassDef("radar_domain", primary=[]), ONTOLOGY)
    assert counts["relationships_rejected"] == 0


def test_c_classify_yield_rejected_forced_to_zero_at_pre_merge():
    """classify_yield does not let pre_merge_rejections influence the
    pre-merge classification. Post-merge classify_yield_from_counts with
    authoritative counts happens in _apply_post_merge_yield_updates (Task 36)."""
    template = _RadarDomainPass(radar_systems=[])
    pr = _make_pass_result(
        template,
        pre_merge_rejections=[
            (SimpleNamespace(rel_type="X"), RelationshipRejectionReason.INVALID_TRIPLE),
        ] * 10,  # enough to trigger DEGRADED if these were counted
    )
    pr.pre_merge_walk = PreMergeWalkSummary(entities=[], raw_edge_count=0)
    status = classify_yield(pr, _PassDef("radar_domain", primary=["RADAR_SYSTEM"]), ONTOLOGY)
    # Zero primary, zero bridge, zero extracted, zero rejected (forced) → EMPTY.
    assert status == YieldStatus.EMPTY


# --- relationships_extracted uses raw_edge_count --------------------------

def test_relationships_extracted_uses_raw_edge_count_from_summary():
    """When pre_merge_walk is set, _count_pass_output's
    relationships_extracted matches raw_edge_count — so typed-edge passes
    report the walker's edge emissions, not the legacy DTO-list length."""
    template = _RadarDomainPass(
        radar_systems=[
            _RadarSystemEntity(
                system_name="R1",
                antennas=[_AntennaEntity(name="A1"), _AntennaEntity(name="A2")],
            ),
        ],
    )
    pr = _make_pass_result(template)
    pass_def = _PassDef("radar_domain", primary=["RADAR_SYSTEM", "ANTENNA"])
    pr.pre_merge_walk = _build_pre_merge_walk_summary(
        pr, pass_def, ONTOLOGY, document_id="doc-1",
    )
    counts = _count_pass_output(pr, pass_def, ONTOLOGY)
    assert counts["relationships_extracted"] == 2  # two HAS_ANTENNA walker emissions


def test_system_links_relationships_extracted_uses_dto_list_length():
    """system_links is the DTO exception. _build_pre_merge_walk_summary
    fills raw_edge_count from len(template.relationships). Round-trip
    through _count_pass_output preserves that value."""
    template = SimpleNamespace(
        relationships=[
            SimpleNamespace(rel_type="ASSOCIATED_WITH"),
            SimpleNamespace(rel_type="MOUNTED_ON"),
            SimpleNamespace(rel_type="CUES"),
        ],
    )
    pr = _make_pass_result(template, pass_name="system_links")
    pass_def = _PassDef("system_links", primary=[], kind="relationships_only")
    pr.pre_merge_walk = _build_pre_merge_walk_summary(
        pr, pass_def, ONTOLOGY, document_id="doc-1",
    )
    counts = _count_pass_output(pr, pass_def, ONTOLOGY)
    assert counts["relationships_extracted"] == 3


# --- Fallback when pre_merge_walk is None (test fixtures) -----------------

def test_classify_yield_fallback_when_pre_merge_walk_is_none():
    """Test-built PassResult without pre_merge_walk still classifies
    meaningfully — falls back to len(result.relationships) for the
    extracted_rels signal, matching legacy pre-Task-34b behavior."""
    template = SimpleNamespace(
        radar_system_list=[SimpleNamespace(system_name="Legacy")],
        relationships=[SimpleNamespace(rel_type="HAS_ANTENNA")],
    )
    pr = _make_pass_result(template)
    pr.pre_merge_walk = None  # explicit — not built by pass loop
    status = classify_yield(pr, _PassDef("radar_domain", primary=["RADAR_SYSTEM"]), ONTOLOGY)
    # Primary=1, bridge=0, rels=1, rejected=0 (forced) → HIT.
    assert status == YieldStatus.HIT


def test_count_pass_output_fallback_when_pre_merge_walk_is_none():
    """_count_pass_output without pre_merge_walk falls back to the legacy
    DTO-list count for relationships_extracted but still forces rejected=0."""
    template = SimpleNamespace(
        radar_system_list=[SimpleNamespace(system_name="Legacy")],
        relationships=[SimpleNamespace(rel_type="HAS_ANTENNA")] * 2,
    )
    pr = _make_pass_result(
        template,
        pre_merge_rejections=[
            (SimpleNamespace(rel_type="X"), RelationshipRejectionReason.INVALID_TRIPLE),
        ],
    )
    pr.pre_merge_walk = None
    counts = _count_pass_output(pr, _PassDef("radar_domain", primary=["RADAR_SYSTEM"]), ONTOLOGY)
    assert counts["relationships_extracted"] == 2  # fallback to DTO list
    assert counts["relationships_rejected"] == 0  # forced
