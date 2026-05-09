"""Task 11 — ExtractionRelationshipProvenance schema + builder tests.

Pre-flight finding: delta_trace.json carries stats/diagnostics ONLY (no
nodes/relationships). The normalized IR with per-relationship provenance is
in delta_merged_graph.json. context._delta_merged_graph is the source;
context._delta_trace is NOT used for relationship provenance.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


_DG_SERVICE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dg_provenance(dg_schemas):
    """Load docker/docling-graph/app/provenance.py under an alt module name.
    Returns the provenance module directly (not a tuple)."""
    module_name = "docling_graph_service_provenance_rp"

    if module_name in sys.modules:
        return sys.modules[module_name]

    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    saved_path = list(sys.path)

    sys.path.insert(0, str(_DG_SERVICE_ROOT))
    for key in list(saved.keys()):
        del sys.modules[key]

    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            _DG_SERVICE_ROOT / "app" / "provenance.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    finally:
        for key in list(sys.modules.keys()):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = saved_path

    return mod


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_relationship_provenance_schema_exists(dg_schemas):
    """ExtractionRelationshipProvenance is exported from schemas."""
    assert hasattr(dg_schemas, "ExtractionRelationshipProvenance"), (
        "ExtractionRelationshipProvenance missing from schemas.py"
    )


def test_relationship_provenance_defaults(dg_schemas):
    """All optional fields default to sensible empties."""
    import typing
    dg_schemas.ExtractionRelationshipProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )
    rp = dg_schemas.ExtractionRelationshipProvenance(relationship_type="USES")
    assert rp.relationship_type == "USES"
    assert rp.source_instance_id is None
    assert rp.target_instance_id is None
    assert rp.evidence_ids == []
    assert rp.self_refs == []
    assert rp.page_numbers == []
    assert rp.supporting_snippet is None


def test_relationship_provenance_full_fields(dg_schemas):
    """All fields accept expected types."""
    import typing
    dg_schemas.ExtractionRelationshipProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )
    rp = dg_schemas.ExtractionRelationshipProvenance(
        relationship_type="PART_OF",
        source_instance_id="node-abc",
        target_instance_id="node-xyz",
        evidence_ids=["ev-1", "ev-2"],
        self_refs=["#/texts/3"],
        page_numbers=[1, 2],
        supporting_snippet="some text",
    )
    assert rp.relationship_type == "PART_OF"
    assert rp.source_instance_id == "node-abc"
    assert rp.target_instance_id == "node-xyz"
    assert rp.evidence_ids == ["ev-1", "ev-2"]
    assert rp.self_refs == ["#/texts/3"]
    assert rp.page_numbers == [1, 2]
    assert rp.supporting_snippet == "some text"


def test_extract_pass_response_has_relationship_provenance_field(dg_schemas):
    """ExtractPassResponse exposes relationship_provenance defaulting to []."""
    import typing
    dg_schemas.ExtractPassResponse.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )
    resp = dg_schemas.ExtractPassResponse(
        bundle_key="bk",
        pass_name="pn",
        pass_output={},
    )
    assert hasattr(resp, "relationship_provenance")
    assert resp.relationship_provenance == []


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------

def test_builder_returns_empty_when_no_merged_graph(dg_schemas, dg_provenance):
    """Returns [] when context has no _delta_merged_graph."""
    import typing
    dg_schemas.ExtractionRelationshipProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )

    class _Ctx:
        pass

    result = dg_provenance.build_relationship_provenance_from_delta_trace(
        _Ctx(), dg_schemas.ExtractionRelationshipProvenance
    )
    assert result == []


def test_builder_returns_empty_for_non_dict_merged_graph(dg_schemas, dg_provenance):
    """Returns [] when _delta_merged_graph is not a dict."""
    import typing
    dg_schemas.ExtractionRelationshipProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )

    class _Ctx:
        _delta_merged_graph = "not-a-dict"

    result = dg_provenance.build_relationship_provenance_from_delta_trace(
        _Ctx(), dg_schemas.ExtractionRelationshipProvenance
    )
    assert result == []


def test_builder_emits_one_row_per_relationship(dg_schemas, dg_provenance):
    """One ExtractionRelationshipProvenance row per relationship in merged graph."""
    import typing
    dg_schemas.ExtractionRelationshipProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )

    merged_graph = {
        "nodes": [],
        "relationships": [
            {
                "edge_label": "USES",
                "source_path": "radar_system",
                "source_ids": {"name": "APAR"},
                "target_path": "component",
                "target_ids": {"name": "Antenna"},
                "provenance": {
                    "evidence_ids": ["ev-1", "ev-2"],
                    "self_refs": ["#/texts/5"],
                    "page_numbers": [3],
                },
            },
            {
                "edge_label": "PART_OF",
                "source_path": "component",
                "source_ids": {"name": "Antenna"},
                "target_path": "radar_system",
                "target_ids": {"name": "APAR"},
                "provenance": {
                    "evidence_ids": ["ev-3"],
                    "self_refs": [],
                    "page_numbers": [],
                },
            },
        ],
    }

    class _Ctx:
        _delta_merged_graph = merged_graph

    rows = dg_provenance.build_relationship_provenance_from_delta_trace(
        _Ctx(), dg_schemas.ExtractionRelationshipProvenance
    )
    assert len(rows) == 2
    labels = {r.relationship_type for r in rows}
    assert labels == {"USES", "PART_OF"}

    uses_row = next(r for r in rows if r.relationship_type == "USES")
    assert uses_row.evidence_ids == ["ev-1", "ev-2"]
    assert uses_row.self_refs == ["#/texts/5"]
    assert uses_row.page_numbers == [3]


def test_builder_skips_relationships_without_label(dg_schemas, dg_provenance):
    """Relationships missing edge_label are silently skipped."""
    import typing
    dg_schemas.ExtractionRelationshipProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )

    merged_graph = {
        "nodes": [],
        "relationships": [
            {"source_path": "a", "target_path": "b"},  # no edge_label
            {"edge_label": "VALID", "source_path": "a", "target_path": "b"},
        ],
    }

    class _Ctx:
        _delta_merged_graph = merged_graph

    rows = dg_provenance.build_relationship_provenance_from_delta_trace(
        _Ctx(), dg_schemas.ExtractionRelationshipProvenance
    )
    assert len(rows) == 1
    assert rows[0].relationship_type == "VALID"


def test_builder_resolves_node_instance_ids(dg_schemas, dg_provenance):
    """source_instance_id / target_instance_id resolved from nodes lookup."""
    import typing
    dg_schemas.ExtractionRelationshipProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )

    merged_graph = {
        "nodes": [
            {
                "path": "radar_system",
                "ids": {"name": "APAR"},
                "instance_id": "inst-radar-1",
            },
            {
                "path": "component",
                "ids": {"name": "Antenna"},
                "instance_id": "inst-comp-1",
            },
        ],
        "relationships": [
            {
                "edge_label": "USES",
                "source_path": "radar_system",
                "source_ids": {"name": "APAR"},
                "target_path": "component",
                "target_ids": {"name": "Antenna"},
                "provenance": {},
            }
        ],
    }

    class _Ctx:
        _delta_merged_graph = merged_graph

    rows = dg_provenance.build_relationship_provenance_from_delta_trace(
        _Ctx(), dg_schemas.ExtractionRelationshipProvenance
    )
    assert len(rows) == 1
    assert rows[0].source_instance_id == "inst-radar-1"
    assert rows[0].target_instance_id == "inst-comp-1"
