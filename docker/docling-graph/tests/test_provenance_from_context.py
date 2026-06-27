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
    Returns (module, ExtractionProvenance) tuple.

    Uses the same sys.path swap pattern as conftest._ensure_dg_app_package
    so that ``from app._numeric_evidence import ...`` inside provenance.py
    resolves to the docling-graph ``app/`` package, not the repo-root one.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    service_root = Path(__file__).resolve().parent.parent
    module_name = "docling_graph_service_provenance"

    if module_name in sys.modules:
        mod = sys.modules[module_name]
        return mod, dg_schemas.ExtractionProvenance

    # Save state
    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    saved_path = list(sys.path)

    sys.path.insert(0, str(service_root))
    for key in list(saved.keys()):
        del sys.modules[key]

    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            service_root / "app" / "provenance.py",
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


# ---------------------------------------------------------------------------
# Task 9 — resolver order + evidence_text population + cap
# ---------------------------------------------------------------------------


def test_resolve_element_uid_uses_self_refs_over_cited_evidence_ids(dg_provenance):
    """Task 2 demotion: the cited-evidence (evidence_ids) preference is GONE.
    self_refs is now the AUTHORITATIVE positional lineage (the delta-IR
    normalizer stamps it per-node); _resolve_element_uid returns self_refs[0]
    regardless of which refs the LLM cited."""
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "RADAR_SYSTEM",
            "identity_values": {"system_name": "Tombstone"},
            "provenance": {
                "self_refs": ["#/texts/99"],
                "evidence_ids": ["#/texts/0"],
            },
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert len(prov) == 1
    assert prov[0].element_uid == "#/texts/99"


def test_resolve_element_uid_does_not_fall_back_to_cited_evidence_ids(dg_provenance):
    """Task 2 demotion: with no direct / nested element_uid / self_refs, a
    '#/'-prefixed evidence_id is NO LONGER consulted (Strategy 3 removed). With
    no chunk_to_self_refs to anchor it, build_provenance_from_context drops the
    node — cited evidence_ids alone can't resolve element_uid anymore."""
    mod, ExtractionProvenance = dg_provenance
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "MISSILE_SYSTEM",
            "identity_values": {"system_name": "PAC-3"},
            "provenance": {
                "evidence_ids": ["#/texts/42"],
            },
        }),
    ])
    prov = mod.build_provenance_from_context(ctx, ExtractionProvenance)
    assert prov == []


def test_resolve_element_uid_skips_non_selfref_evidence(dg_provenance):
    """evidence_ids that don't start with '#/' must not be used as element_uid.
    Falls through to chunk_to_self_refs lookup instead."""
    mod, ExtractionProvenance = dg_provenance
    chunk_to_self_refs = {0: ["#/texts/7"]}
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "SECTION",
            "identity_values": {},
            "provenance": {
                "evidence_ids": ["uuid-abc123"],  # NOT a self_ref
                "chunk_indexes": [0],
            },
        }),
    ])
    prov = mod.build_provenance_from_context(
        ctx, ExtractionProvenance, chunk_to_self_refs=chunk_to_self_refs
    )
    assert len(prov) == 1
    assert prov[0].element_uid == "#/texts/7"


def test_extraction_provenance_evidence_text_is_populated_from_units(dg_provenance):
    """build_provenance_from_context populates evidence_text when
    chunk_to_evidence_units is provided and the node's chunk_indexes match."""
    mod, ExtractionProvenance = dg_provenance
    chunk_to_evidence_units = {
        2: [
            {"evidence_id": "#/texts/10", "text": "The radar range is 400 km."},
            {"evidence_id": "#/texts/11", "text": "Operates at X-band frequency."},
        ]
    }
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "RADAR_SYSTEM",
            "element_uid": "#/texts/10",
            "identity_values": {"system_name": "Tombstone"},
            "provenance": {
                "chunk_indexes": [2],
                "evidence_ids": ["#/texts/10", "#/texts/11"],
            },
        }),
    ])
    prov = mod.build_provenance_from_context(
        ctx, ExtractionProvenance,
        chunk_to_evidence_units=chunk_to_evidence_units,
    )
    assert len(prov) == 1
    assert prov[0].evidence_text is not None
    assert "radar range" in prov[0].evidence_text
    assert "X-band" in prov[0].evidence_text


def test_extraction_provenance_evidence_text_caps_at_500_chars(dg_provenance):
    """evidence_text must be at most 501 chars (500 + ellipsis)."""
    mod, ExtractionProvenance = dg_provenance
    long_text = "A" * 600
    chunk_to_evidence_units = {
        0: [{"evidence_id": "#/texts/0", "text": long_text}]
    }
    ctx = _stub_context(nodes=[
        ("n1", {
            "label": "SECTION",
            "element_uid": "#/texts/0",
            "identity_values": {},
            "provenance": {
                "chunk_indexes": [0],
                "evidence_ids": ["#/texts/0"],
            },
        }),
    ])
    prov = mod.build_provenance_from_context(
        ctx, ExtractionProvenance,
        chunk_to_evidence_units=chunk_to_evidence_units,
    )
    assert len(prov) == 1
    assert prov[0].evidence_text is not None
    assert len(prov[0].evidence_text) <= 501
