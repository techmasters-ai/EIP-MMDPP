"""Phase 8 Task 51 — service-side provenance builder tests.

Service-side unit test with a stubbed PipelineContext, per plan Task
51b's scope decision: we do NOT round-trip through the /extract-pass
endpoint (requires vLLM GPU), only exercise the helper on synthetic
knowledge_graph + docling_document shapes.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import networkx as nx
import pytest


@pytest.fixture(scope="module")
def dg_provenance(dg_schemas):
    """Load docker/docling-graph/app/provenance.py under its own
    alt-module name so the repo-root ``app/`` package doesn't shadow.
    Returns (module, ExtractionProvenance) tuple."""
    import importlib.util
    from pathlib import Path

    service_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_provenance",
        service_root / "app" / "provenance.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, dg_schemas.ExtractionProvenance


def _stub_context(
    *,
    nodes: list[tuple[str, dict]],
    texts: list[dict] | None = None,
):
    """Build a stub PipelineContext-shaped object with a knowledge_graph
    + docling_document attribute."""
    graph = nx.DiGraph()
    for node_id, data in nodes:
        graph.add_node(node_id, **data)
    doc = {"texts": texts or []}
    return SimpleNamespace(knowledge_graph=graph, docling_document=doc)


def test_direct_element_uid_attribute_yields_provenance(dg_provenance):
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "RADAR_SYSTEM",
            "element_uid": "#/texts/5",
            "identity_values": {"system_name": "Tombstone"},
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert len(prov) == 1
    assert prov[0].element_uid == "#/texts/5"
    assert prov[0].ontology_name == "RADAR_SYSTEM"
    assert prov[0].identity_values == {"system_name": "Tombstone"}
    assert prov[0].instance_id  # populated


def test_nested_provenance_element_uid(dg_provenance):
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "MISSILE_SYSTEM",
            "provenance": {"element_uid": "#/texts/10"},
            "identity_values": {"system_name": "PAC-3"},
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert prov[0].element_uid == "#/texts/10"


def test_chunk_indexes_fallback_resolves_via_docling_texts(dg_provenance):
    """No direct/nested element_uid → fall back to
    ``provenance.chunk_indexes[0]`` mapped through
    ``docling_document.texts[idx].self_ref``."""
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(
        nodes=[
            ("n1", {
                "label": "SECTION",
                "provenance": {"chunk_indexes": [2]},
                "identity_values": {"section_number": "3.2"},
            }),
        ],
        texts=[
            {"self_ref": "#/texts/0"},
            {"self_ref": "#/texts/1"},
            {"self_ref": "#/texts/2"},
        ],
    )
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert prov[0].element_uid == "#/texts/2"


def test_drops_node_without_resolvable_element_uid_with_warning(dg_provenance, caplog):
    """Node has an entity label but no element_uid anywhere — dropped
    with a WARNING."""
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("n-bad", {"label": "RADAR_SYSTEM"}),  # no provenance, no element_uid
    ])
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert prov == []
    assert any(
        "no resolvable element_uid" in r.message and "n-bad" in r.message
        for r in caplog.records
    )


def test_skips_non_entity_nodes(dg_provenance):
    """Lowercase / helper-node labels are not entities and shouldn't
    produce provenance rows."""
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("help1", {"label": "property", "element_uid": "#/texts/0"}),
        ("help2", {"label": "path", "element_uid": "#/texts/1"}),
        ("help3", {"label": None, "element_uid": "#/texts/2"}),
        ("entity1", {
            "label": "ANTENNA",
            "element_uid": "#/texts/3",
            "identity_values": {"name": "Main Array"},
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert len(prov) == 1
    assert prov[0].ontology_name == "ANTENNA"


def test_accepts_empty_identity_values_for_components(dg_provenance):
    """Components (is_entity=False) may have empty identity tuples —
    the helper must not synthesize a fake identity; it passes {} through."""
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "PROPULSION_STACK",
            "element_uid": "#/texts/4",
            # no identity_values key
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert prov[0].identity_values == {}


def test_page_and_chunk_index_surface_from_provenance_dict(dg_provenance):
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "FIGURE",
            "element_uid": "#/pictures/0",
            "provenance": {
                "page_numbers": [42, 43],
                "chunk_indexes": [7, 8],
            },
            "identity_values": {"figure_ref": "Figure 3-12"},
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert prov[0].page == 42
    assert prov[0].chunk_index == 7


def test_instance_id_distinct_per_node(dg_provenance):
    """Same identity duplicated across two nodes → each gets its own
    instance_id so downstream dedup doesn't collapse them."""
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("A", {
            "label": "PROPULSION_STACK",
            "element_uid": "#/texts/0",
        }),
        ("B", {
            "label": "PROPULSION_STACK",
            "element_uid": "#/texts/1",
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert len(prov) == 2
    assert prov[0].instance_id != prov[1].instance_id


def test_returns_empty_when_context_has_no_graph(dg_provenance):
    mod, ExtractionProvenance = dg_provenance
    ctx = SimpleNamespace(knowledge_graph=None, docling_document=None)
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert prov == []
