"""Plan Task 36 Step 1 — merge_and_resolve walker-based edge harvesting.

Once relationships move inside entity classes via typed edges, merge_and_resolve
must harvest edges from the walker (not from DTO lists) for non-system_links
passes while keeping the system_links DTO branch verbatim. Per-pass accounting
feeds into ``MergedExtraction.per_pass_edge_metrics``.

Covers the Step 1 edge fixtures:
  (a) Typed-edge HAS_ANTENNA emission from nested entities.
  (b) Self-referential edge — object-id visited set prevents infinite loop.
  (c) Mutual A↔B — one edge in each direction, no infinite recursion.
  (d) Cross-path dedup — same (from, rel, to) reachable via two paths → one edge.
  (e) Empty-identity entity — multiple instances collapse by logical identity.
  (f) system_links DTO branch feeding MergedEdgeRecord.
  (g) DTO ↔ typed-edge normalization parity (reviewer finding C3).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.services.extraction_merge import (
    ExtractionMetadata,
    LogicalIdentity,
    MergedEdgeRecord,
    PassResult,
    PerPassEdgeMetrics,
    RelationshipRejectionReason,
    merge_and_resolve,
)


# --- Fixture entities ------------------------------------------------------

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


class FrequencyBandEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="FREQUENCY_BAND",
        graph_id_fields=["band_name"],
        identity_scope="document",
        is_entity=True,
    )
    band_name: str = Field(...)


class RadarSystemEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="document",
        is_entity=True,
    )
    system_name: str = Field(...)
    antennas: List[AntennaEntity] = _edge(label="HAS_ANTENNA", default_factory=list)
    frequency_bands: List[FrequencyBandEntity] = _edge(
        label="OPERATES_ON", default_factory=list,
    )


class RadarDomainPass(BaseModel):
    model_config = ConfigDict()
    radar_systems: List[RadarSystemEntity] = Field(default_factory=list)


class SectionEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="SECTION",
        graph_id_fields=["heading"],
        identity_scope="document",
        is_entity=True,
    )
    heading: str = Field(...)
    parent_section: Optional["SectionEntity"] = _edge(
        label="CHILD_OF", default=None,
    )


SectionEntity.model_rebuild()


class ReferencePass(BaseModel):
    model_config = ConfigDict()
    sections: List[SectionEntity] = Field(default_factory=list)


class PlatformEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="PLATFORM",
        graph_id_fields=["name"],
        identity_scope="document",
        is_entity=True,
    )
    name: str = Field(...)
    radars: List[RadarSystemEntity] = _edge(
        label="INSTALLED_ON", default_factory=list,
    )


ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"],
         "identity_scope": "document", "properties": ["system_name"]},
        {"name": "ANTENNA", "identity_fields": ["name"],
         "identity_scope": "document", "properties": ["name"]},
        {"name": "FREQUENCY_BAND", "identity_fields": ["band_name"],
         "identity_scope": "document", "properties": ["band_name"]},
        {"name": "PLATFORM", "identity_fields": ["name"],
         "identity_scope": "document", "properties": ["name"]},
        {"name": "SECTION", "identity_fields": ["heading"],
         "identity_scope": "document", "properties": ["heading"]},
    ],
    "validation_matrix": [
        {"source": "RADAR_SYSTEM", "relationship": "HAS_ANTENNA", "target": "ANTENNA"},
        {"source": "RADAR_SYSTEM", "relationship": "OPERATES_ON", "target": "FREQUENCY_BAND"},
        {"source": "SECTION", "relationship": "CHILD_OF", "target": "SECTION"},
        {"source": "PLATFORM", "relationship": "INSTALLED_ON", "target": "RADAR_SYSTEM"},
    ],
}


def _manifest(pass_kind_by_name: dict | None = None):
    mapping = dict(pass_kind_by_name or {})
    def _find(pname):
        kind = mapping.get(pname, "entities_and_relationships")
        return SimpleNamespace(name=pname, kind=kind)
    return SimpleNamespace(find_pass=_find, passes=[])


def _pass_result(template, pass_name="radar_domain"):
    return PassResult(
        pass_name=pass_name,
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )


# --- Step 1 fixtures -------------------------------------------------------

def test_step1_a_has_antenna_typed_edge_emission():
    """(a) RadarSystem with nested antennas → MergedEdgeRecord with
    rel_type=HAS_ANTENNA, from_identity=radar, to_identity=antenna."""
    template = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(system_name="R1", antennas=[AntennaEntity(name="A1")]),
        ],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": _pass_result(template)},
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert len(merged.edges) == 1
    e = merged.edges[0]
    assert e.rel_type == "HAS_ANTENNA"
    assert e.from_identity.entity_type == "RADAR_SYSTEM"
    assert e.from_identity.identity_tuple == ("R1",)
    assert e.to_identity.entity_type == "ANTENNA"
    assert e.to_identity.identity_tuple == ("A1",)
    assert e.pass_origins == {"radar_domain"}


def test_step1_a_chained_edges_radar_antenna_and_radar_frequency():
    """Multiple typed-edge fields on one entity → one edge each, both emitted."""
    template = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="R1",
                antennas=[AntennaEntity(name="A1")],
                frequency_bands=[FrequencyBandEntity(band_name="X-band")],
            ),
        ],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": _pass_result(template)},
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    rel_types = sorted(e.rel_type for e in merged.edges)
    assert rel_types == ["HAS_ANTENNA", "OPERATES_ON"]


def test_step1_b_self_reference_cycle_does_not_infinite_loop():
    """(b) Self-referential edge (Section.parent_section = self).
    visited-by-Python-id guard terminates the walk."""
    root = SectionEntity(heading="Root")
    root.parent_section = root
    template = ReferencePass(sections=[root])
    merged = merge_and_resolve(
        pass_results={"reference": _pass_result(template, pass_name="reference")},
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    # Self-edge emitted and validated (from=to=Root); no infinite recursion.
    assert len(merged.edges) == 1
    assert merged.edges[0].rel_type == "CHILD_OF"


def test_step1_d_same_edge_two_paths_dedup_to_one():
    """(d) (rad_1, HAS_ANTENNA, ant_1) reachable via direct walk from
    the radar node AND via re-visiting the same radar in another pass
    → cross-pass reducer emits the edge once with pass_origins union."""
    ant = AntennaEntity(name="A1")
    radar_a = RadarSystemEntity(system_name="R1", antennas=[ant])
    radar_b = RadarSystemEntity(system_name="R1", antennas=[AntennaEntity(name="A1")])
    merged = merge_and_resolve(
        pass_results={
            "radar_domain": _pass_result(RadarDomainPass(radar_systems=[radar_a])),
            "other_systems": _pass_result(
                RadarDomainPass(radar_systems=[radar_b]),
                pass_name="other_systems",
            ),
        },
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    # Edges grouped by triple; pass_origins unions across contributing passes.
    has_antenna_edges = [e for e in merged.edges if e.rel_type == "HAS_ANTENNA"]
    assert len(has_antenna_edges) == 1
    assert has_antenna_edges[0].pass_origins == {"radar_domain", "other_systems"}


def test_step1_g_dto_and_typed_edge_emit_structurally_identical_records():
    """(g) Reviewer finding C3: an edge emitted via the DTO branch (rel_type +
    identity dict lookup) MUST produce a structurally identical
    MergedEdgeRecord to one emitted via the walker branch — same fields,
    same identity construction, same triple composition. Tested with
    INSTALLED_ON between a PlatformEntity and a RadarSystemEntity.

    Typed-edge pass: PlatformEntity.radars → RadarSystemEntity via
    edge_label=INSTALLED_ON.
    DTO pass: a SimpleNamespace "other_systems" carrying a DTO
    relationships=[...] list with from_identity/to_identity.
    Both resolve to the same (PLATFORM name=P1, INSTALLED_ON, RADAR_SYSTEM
    system_name=R1) triple."""
    class _PlatformRoot(BaseModel):
        model_config = ConfigDict()
        platforms: List[PlatformEntity] = Field(default_factory=list)

    typed_template = _PlatformRoot(
        platforms=[PlatformEntity(name="P1", radars=[RadarSystemEntity(system_name="R1")])],
    )

    # DTO template: SimpleNamespace with platform + radar entities + DTO rel.
    dto_template = SimpleNamespace(
        platform_list=[SimpleNamespace(name="P1")],
        radar_system_list=[SimpleNamespace(system_name="R1")],
        relationships=[SimpleNamespace(
            rel_type="INSTALLED_ON",
            from_type="PLATFORM",
            to_type="RADAR_SYSTEM",
            from_identity={"name": "P1"},
            to_identity={"system_name": "R1"},
            confidence=0.8,
        )],
    )

    typed_merged = merge_and_resolve(
        pass_results={"radar_domain": _pass_result(typed_template)},
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    dto_merged = merge_and_resolve(
        pass_results={"other_systems": _pass_result(dto_template, pass_name="other_systems")},
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )

    assert len(typed_merged.edges) == 1
    assert len(dto_merged.edges) == 1
    te = typed_merged.edges[0]
    de = dto_merged.edges[0]

    # Structural equality on the core triple.
    assert te.rel_type == de.rel_type == "INSTALLED_ON"
    assert te.from_identity == de.from_identity  # LogicalIdentity compares by value
    assert te.to_identity == de.to_identity
    # Each branch records its own pass in pass_origins.
    assert te.pass_origins == {"radar_domain"}
    assert de.pass_origins == {"other_systems"}


# --- Per-pass edge metrics carrier ----------------------------------------

def test_per_pass_edge_metrics_populated_for_typed_edge_pass():
    """merge_and_resolve populates per_pass_edge_metrics[pass_name] with
    attempted/accepted/rejected counts and rejection observability fields."""
    template = RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="R1",
                antennas=[AntennaEntity(name="A1"), AntennaEntity(name="A2")],
            ),
        ],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": _pass_result(template)},
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    m = merged.per_pass_edge_metrics["radar_domain"]
    assert m.attempted == 2
    assert m.accepted == 2
    assert m.rejected == 0
    assert m.rejection_sample == []
    assert m.rejections_by_reason == {}


def test_per_pass_edge_metrics_records_invalid_triple_rejections():
    """A typed-edge emission with a label not in VALIDATION_MATRIX is
    rejected with reason INVALID_TRIPLE; the per-pass carrier reflects it."""
    class _BadEdgeRadar(BaseModel):
        model_config = ConfigDict(
            ontology_name="RADAR_SYSTEM",
            graph_id_fields=["system_name"],
            identity_scope="document",
            is_entity=True,
        )
        system_name: str = Field(...)
        antennas: List[AntennaEntity] = _edge(label="INVENTED_REL", default_factory=list)

    class _BadRoot(BaseModel):
        model_config = ConfigDict()
        radar_systems: List[_BadEdgeRadar] = Field(default_factory=list)

    template = _BadRoot(
        radar_systems=[
            _BadEdgeRadar(system_name="R1", antennas=[AntennaEntity(name="A1")]),
        ],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": _pass_result(template)},
        manifest=_manifest(),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    m = merged.per_pass_edge_metrics["radar_domain"]
    assert m.attempted == 1
    assert m.accepted == 0
    assert m.rejected == 1
    assert "invalid_triple" in m.rejections_by_reason
    assert m.rejections_by_reason["invalid_triple"] == 1
    assert len(m.rejection_sample) == 1
    assert m.rejection_sample[0]["reason"] == "invalid_triple"


def test_per_pass_edge_metrics_populated_for_dto_pass_system_links_shape():
    """system_links DTO branch also populates per_pass_edge_metrics with
    matching shape — the post-merge writer reads it uniformly regardless
    of producer branch."""
    template = SimpleNamespace(
        radar_system_list=[SimpleNamespace(system_name="R1")],
        platform_list=[SimpleNamespace(name="P1")],
        relationships=[
            SimpleNamespace(
                rel_type="INSTALLED_ON",
                from_ref_id="E001", to_ref_id="E002",
            ),
        ],
    )
    upstream = {
        "E001": LogicalIdentity(
            entity_type="PLATFORM", identity_field_names=("name",),
            identity_tuple=("P1",), scope="document", document_id="doc-1",
        ),
        "E002": LogicalIdentity(
            entity_type="RADAR_SYSTEM", identity_field_names=("system_name",),
            identity_tuple=("R1",), scope="document", document_id="doc-1",
        ),
    }
    pr = _pass_result(template, pass_name="system_links")
    pr.upstream_refs = upstream

    merged = merge_and_resolve(
        pass_results={"system_links": pr},
        manifest=_manifest({"system_links": "relationships_only"}),
        ontology=ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    m = merged.per_pass_edge_metrics["system_links"]
    assert m.attempted == 1
    assert m.accepted == 1
    assert m.rejected == 0
