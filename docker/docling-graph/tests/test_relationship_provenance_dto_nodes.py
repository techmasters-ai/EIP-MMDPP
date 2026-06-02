"""Bug #59 — relationship provenance for DTO-node relationships (system_links).

The ``system_links`` pass is a relationships-ONLY DTO pass: its
``SystemLinkRelationship`` items are ``is_entity=False`` with DTO fields
(rel_type / from_ref_id / to_ref_id). The delta catalog therefore classifies
each emitted relationship as a *component NODE* (it lands in
``_delta_merged_graph["nodes"]`` carrying ``properties["rel_type"]``), NOT as a
delta-IR graph edge in ``_delta_merged_graph["relationships"]``.

So the relationship-list path of
``build_relationship_provenance_from_delta_trace`` loops an EMPTY list and
returns ``[]`` for this pass — relationships extracted, 0 relationship
provenance, committed edges carry no source_chunk_ids.

The chunk-lineage data DOES exist: those DTO relationship nodes each carry a
``provenance`` dict with ``self_refs`` / ``page_numbers`` / ``evidence_ids``
(positional batch self_refs, attached identically to entity nodes by the IR
normalizer — see ir_normalizer.py ``_attach_evidence_to_prov`` which sets
``node["provenance"]``; ``rel_type`` survives in ``node["properties"]`` because
it is a catalog property_field of the DTO component node). The builder must
recognise DTO rel_type nodes — mirroring the ``summarize_pass_output``
rel_type special-case (evidence_gate.py ~340-346).

Loads provenance.py + schemas.py via the same importlib spec_from_file_location
save/restore dance used by test_entity_provenance_from_delta.py so the host
run resolves ``app._numeric_evidence`` WITHOUT executing app/main.py (which
needs the patched docling_graph fork and fails under the host PYTHONPATH).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent  # docker/docling-graph
_saved_path = list(sys.path)
_saved_modules = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
try:
    sys.path.insert(0, str(_SERVICE_ROOT))
    for _stale in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        del sys.modules[_stale]

    _PROV = _SERVICE_ROOT / "app" / "provenance.py"
    _spec = importlib.util.spec_from_file_location("dg_provenance_under_test_rp_dto", _PROV)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_relationship_provenance_from_delta_trace = (
        _mod.build_relationship_provenance_from_delta_trace
    )

    _SCHEMAS = _SERVICE_ROOT / "app" / "schemas.py"
    _sspec = importlib.util.spec_from_file_location("dg_schemas_under_test_rp_dto", _SCHEMAS)
    _smod = importlib.util.module_from_spec(_sspec)
    _sspec.loader.exec_module(_smod)
    ExtractionRelationshipProvenance = _smod.ExtractionRelationshipProvenance
    # Loaded via spec_from_file_location → forward refs (Any/Optional) resolve
    # only after model_rebuild in the module's namespace.
    ExtractionRelationshipProvenance.model_rebuild(_types_namespace=vars(_smod))
finally:
    sys.path[:] = _saved_path
    for _key in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        del sys.modules[_key]
    sys.modules.update(_saved_modules)


class _Ctx:
    def __init__(self, merged_graph):
        self._delta_merged_graph = merged_graph


def test_dto_node_relationship_emits_provenance():
    """A relationships-only DTO component node (properties.rel_type set) yields
    exactly ONE ExtractionRelationshipProvenance, reading rel_type + self_refs /
    page_numbers / evidence_ids from THAT node's provenance dict. A plain entity
    node WITHOUT rel_type is NOT emitted as a relationship."""
    merged_graph = {
        # system_links produces NO delta-IR graph relationships:
        "relationships": [],
        "nodes": [
            {
                # DTO relationship row (SystemLinkRelationship): is_entity=False,
                # rel_type lives on node["properties"].
                "path": "system_links[]",
                "node_type": "SystemLinkRelationship",
                "ids": {},
                "properties": {
                    "rel_type": "VARIANT_OF",
                    "from_ref_id": "a",
                    "to_ref_id": "b",
                },
                "provenance": {
                    "self_refs": ["#/texts/5"],
                    "page_numbers": [3],
                    "evidence_ids": ["#/texts/5"],
                },
            },
            {
                # A genuine entity node WITHOUT rel_type — must NOT be emitted
                # as a relationship.
                "path": "radar_systems[]",
                "node_type": "RadarSystemEntity",
                "ids": {"name": "APAR"},
                "properties": {"name": "APAR"},
                "provenance": {
                    "self_refs": ["#/texts/9"],
                    "page_numbers": [7],
                    "evidence_ids": ["#/texts/9"],
                },
            },
        ],
    }

    rows = build_relationship_provenance_from_delta_trace(
        _Ctx(merged_graph), ExtractionRelationshipProvenance
    )

    assert len(rows) == 1, (
        f"expected exactly ONE DTO-node relationship row, got {len(rows)}: "
        f"{[r.relationship_type for r in rows]}"
    )
    row = rows[0]
    assert row.relationship_type == "VARIANT_OF"
    assert row.self_refs == ["#/texts/5"]
    assert row.page_numbers == [3]
    assert row.evidence_ids == ["#/texts/5"]
    # source/target instance ids are allowed to be None — the worker keys
    # provenance_by_triple with a __rel_type_fallback__ bucket on rel_type alone.


def test_graph_edge_path_still_emits():
    """Regression: a normal delta-IR graph-edge relationship still emits via the
    existing relationships-list path, and is NOT double-counted by the DTO
    node branch (the relationship's endpoint nodes carry no rel_type)."""
    merged_graph = {
        "nodes": [
            {
                "path": "radar_systems[]",
                "ids": {"name": "APAR"},
                "instance_id": "inst-radar-1",
                "properties": {"name": "APAR"},
                "provenance": {"self_refs": ["#/texts/1"], "page_numbers": [1]},
            },
            {
                "path": "radar_systems[].components[]",
                "ids": {"name": "Antenna"},
                "instance_id": "inst-comp-1",
                "properties": {"name": "Antenna"},
                "provenance": {"self_refs": ["#/texts/2"], "page_numbers": [1]},
            },
        ],
        "relationships": [
            {
                "edge_label": "USES",
                "source_path": "radar_systems[]",
                "source_ids": {"name": "APAR"},
                "target_path": "radar_systems[].components[]",
                "target_ids": {"name": "Antenna"},
                "provenance": {
                    "evidence_ids": ["ev-1"],
                    "self_refs": ["#/texts/5"],
                    "page_numbers": [3],
                },
            }
        ],
    }

    rows = build_relationship_provenance_from_delta_trace(
        _Ctx(merged_graph), ExtractionRelationshipProvenance
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.relationship_type == "USES"
    assert row.source_instance_id == "inst-radar-1"
    assert row.target_instance_id == "inst-comp-1"
    assert row.self_refs == ["#/texts/5"]
    assert row.page_numbers == [3]


def test_dto_node_without_rel_type_not_emitted():
    """A component node whose properties carry NO rel_type is not turned into a
    relationship (guards against treating every property-bearing node as an edge)."""
    merged_graph = {
        "relationships": [],
        "nodes": [
            {
                "path": "radar_systems[].specs[]",
                "node_type": "SpecEntity",
                "ids": {},
                "properties": {"name": "frequency", "value": "X-band"},
                "provenance": {"self_refs": ["#/texts/4"], "page_numbers": [2]},
            }
        ],
    }

    rows = build_relationship_provenance_from_delta_trace(
        _Ctx(merged_graph), ExtractionRelationshipProvenance
    )
    assert rows == []


def test_dto_branch_and_graph_edges_coexist():
    """Defensive: if a single merged graph happened to carry BOTH a graph edge
    AND a DTO rel_type node (not expected for a single pass, but the builder
    must not crash), both are emitted exactly once."""
    merged_graph = {
        "nodes": [
            {
                "path": "system_links[]",
                "node_type": "SystemLinkRelationship",
                "ids": {},
                "properties": {"rel_type": "VARIANT_OF", "from_ref_id": "a", "to_ref_id": "b"},
                "provenance": {"self_refs": ["#/texts/5"], "page_numbers": [3], "evidence_ids": []},
            },
        ],
        "relationships": [
            {
                "edge_label": "USES",
                "source_path": "radar_systems[]",
                "source_ids": {"name": "APAR"},
                "target_path": "radar_systems[].components[]",
                "target_ids": {"name": "Antenna"},
                "provenance": {"self_refs": ["#/texts/1"], "page_numbers": [1]},
            }
        ],
    }

    rows = build_relationship_provenance_from_delta_trace(
        _Ctx(merged_graph), ExtractionRelationshipProvenance
    )
    labels = sorted(r.relationship_type for r in rows)
    assert labels == ["USES", "VARIANT_OF"]
