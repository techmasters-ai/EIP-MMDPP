"""Tests for PreMergeWalkSummary (plan Task 34b).

The pre-merge pass loop needs BOTH entity counts (for
primary_entities_extracted / bridge_entities_extracted) AND raw edge
counts (for provisional relationships_extracted). A single shared
carrier built once per PassResult feeds both downstream consumers —
classify_yield and _count_pass_output — so the walker never runs twice
for the pre-merge phase.

system_links is the documented DTO exception (Decision 4): entities=[],
raw_edge_count=len(template_instance.relationships).
"""
from __future__ import annotations

from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.services.extraction_merge import (
    ExtractionMetadata,
    PassResult,
    PreMergeWalkSummary,
)


# --- Fixture classes -------------------------------------------------------

def _edge(label: str, **field_kwargs):
    extra = field_kwargs.pop("json_schema_extra", None) or {}
    extra["edge_label"] = label
    return Field(json_schema_extra=extra, **field_kwargs)


class AntennaEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="ANTENNA",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )
    name: str = Field(...)


class RadarSystemEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="document",
        is_entity=True,
    )
    system_name: str = Field(...)
    antennas: List[AntennaEntity] = _edge(label="HAS_ANTENNA", default_factory=list)


class RadarDomainPass(BaseModel):
    model_config = ConfigDict()
    radar_systems: List[RadarSystemEntity] = Field(default_factory=list)


class _FakePassDef:
    """Minimal pass_def stub — matches the attributes the builder reads."""
    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind


ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "document"},
        {"name": "ANTENNA", "identity_fields": ["name"], "identity_scope": "document"},
    ],
}


# --- Shape / default tests -------------------------------------------------

def test_pre_merge_walk_summary_has_expected_fields():
    summary = PreMergeWalkSummary(entities=[], raw_edge_count=0)
    assert summary.entities == []
    assert summary.raw_edge_count == 0


def test_pass_result_has_optional_pre_merge_walk_field_defaulting_to_none():
    """PassResult must carry the optional summary; default None preserves
    backward-compat for test-built PassResults (plan 35a fallback path)."""
    pr = PassResult(
        pass_name="radar_domain",
        template_instance=RadarDomainPass(),
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )
    assert pr.pre_merge_walk is None


# --- Builder behavior ------------------------------------------------------

def test_typed_edge_pass_builds_summary_with_entities_and_edge_count():
    """For entity-bearing passes, the builder walks the pass-root and produces
    a summary whose entities list includes nested entities and whose
    raw_edge_count counts edge emissions."""
    from app.workers.pipeline import _build_pre_merge_walk_summary

    template = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="AN/TPY-2",
                antennas=[AntennaEntity(name="A1"), AntennaEntity(name="A2")],
            ),
        ],
    )
    pass_result = PassResult(
        pass_name="radar_domain",
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )
    pass_def = _FakePassDef(name="radar_domain", kind="entities_and_relationships")

    summary = _build_pre_merge_walk_summary(
        pass_result, pass_def, ONTOLOGY, document_id="doc-1",
    )

    assert len(summary.entities) == 3  # 1 radar + 2 antennas
    assert summary.raw_edge_count == 2  # two HAS_ANTENNA edges


def test_typed_edge_pass_builds_summary_empty_when_no_entities():
    """A typed-edge pass whose template contains no entities yields an empty
    summary — not None — so consumers can always read summary.entities."""
    from app.workers.pipeline import _build_pre_merge_walk_summary

    template = RadarDomainPass(radar_systems=[])
    pass_result = PassResult(
        pass_name="radar_domain",
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )
    pass_def = _FakePassDef(name="radar_domain", kind="entities_and_relationships")

    summary = _build_pre_merge_walk_summary(
        pass_result, pass_def, ONTOLOGY, document_id="doc-1",
    )
    assert summary.entities == []
    assert summary.raw_edge_count == 0


def test_system_links_special_case_entities_empty_raw_edge_count_from_dto_list():
    """system_links DTO branch: entities=[] (no entity walker); raw_edge_count
    is the DTO-list length so classify_yield sees non-zero provisional edges
    when the LLM emitted candidates. Decision 4 exception."""
    from types import SimpleNamespace
    from app.workers.pipeline import _build_pre_merge_walk_summary

    template = SimpleNamespace(
        relationships=[
            SimpleNamespace(from_ref_id="a", to_ref_id="b", rel_type="ASSOCIATED_WITH"),
            SimpleNamespace(from_ref_id="c", to_ref_id="d", rel_type="MOUNTED_ON"),
        ],
    )
    pass_result = PassResult(
        pass_name="system_links",
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )
    pass_def = _FakePassDef(name="system_links", kind="relationships_only")

    summary = _build_pre_merge_walk_summary(
        pass_result, pass_def, ONTOLOGY, document_id="doc-1",
    )
    assert summary.entities == []
    assert summary.raw_edge_count == 2


def test_system_links_special_case_empty_dto_list():
    """system_links with zero DTOs → raw_edge_count=0; pre-merge classification
    will see primary=0 bridge=0 edges=0 and produce EMPTY (Task 36 post-merge
    will confirm)."""
    from types import SimpleNamespace
    from app.workers.pipeline import _build_pre_merge_walk_summary

    template = SimpleNamespace(relationships=[])
    pass_result = PassResult(
        pass_name="system_links",
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )
    pass_def = _FakePassDef(name="system_links", kind="relationships_only")

    summary = _build_pre_merge_walk_summary(
        pass_result, pass_def, ONTOLOGY, document_id="doc-1",
    )
    assert summary.entities == []
    assert summary.raw_edge_count == 0


def test_relationships_only_pass_without_relationships_attribute_is_robust():
    """Defensive: a malformed template_instance missing the relationships
    attribute should produce raw_edge_count=0 rather than AttributeError.
    Mirrors PassResult.relationships property's `or []` fallback."""
    from types import SimpleNamespace
    from app.workers.pipeline import _build_pre_merge_walk_summary

    template = SimpleNamespace()  # no relationships attribute
    pass_result = PassResult(
        pass_name="system_links",
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )
    pass_def = _FakePassDef(name="system_links", kind="relationships_only")

    summary = _build_pre_merge_walk_summary(
        pass_result, pass_def, ONTOLOGY, document_id="doc-1",
    )
    assert summary.entities == []
    assert summary.raw_edge_count == 0
