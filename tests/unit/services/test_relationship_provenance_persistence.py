"""Tests for B.1/B.2: provenance persisted on edge upsert + surfaced on read.

B.1 — _build_upsert_relationship_script must include record.provenance in the
      generated SQL params so per-edge evidence_ids land on the ArcadeDB edge.

B.2 — get_neighborhood_graph must decode the stored JSON string back to a dict
      and include it as 'provenance' on each edge dict.
"""

import json

import pytest


# ---------------------------------------------------------------------------
# B.1 — SQL builder includes provenance
# ---------------------------------------------------------------------------


def _make_record(**kwargs):
    from app.services.graph_store import RelationshipRecord

    defaults = dict(
        from_type="RADAR",
        from_identity={"id": "R1"},
        to_type="MISSILE",
        to_identity={"id": "M1"},
        rel_type="ENGAGES",
        extraction_confidence=0.9,
    )
    defaults.update(kwargs)
    return RelationshipRecord(**defaults)


def test_upsert_relationship_script_includes_provenance():
    """provenance dict must appear in the generated SQL params."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    prov = {
        "evidence_ids": ["#/texts/0"],
        "self_refs": ["#/texts/0"],
        "page_numbers": [3],
    }
    rec = _make_record(provenance=prov)
    script, params = _build_upsert_relationship_script([rec], provenance=None)

    # The provenance must appear in params as a JSON-serialised string
    # keyed provenance_0 (row index 0).
    assert "provenance_0" in params, (
        f"provenance_0 param missing; params keys: {list(params.keys())}"
    )
    stored = json.loads(params["provenance_0"])
    assert stored["evidence_ids"] == ["#/texts/0"]
    assert stored["page_numbers"] == [3]

    # The SET clause in the script must reference the param
    assert ":provenance_0" in script, (
        f"provenance param not referenced in script:\n{script}"
    )


def test_upsert_relationship_script_no_provenance_field_omits_param():
    """When record.provenance is None, no provenance_* param is added."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    rec = _make_record()  # provenance defaults to None
    script, params = _build_upsert_relationship_script([rec], provenance=None)

    provenance_params = [k for k in params if k.startswith("provenance_")]
    assert provenance_params == [], (
        f"unexpected provenance params when provenance=None: {provenance_params}"
    )


def test_upsert_relationship_script_provenance_in_create_and_update():
    """provenance must appear in both the CREATE EDGE branch and UPDATE branch."""
    from app.services.arcadedb_graph import _build_upsert_relationship_script

    prov = {"evidence_ids": ["#/texts/5"], "self_refs": [], "page_numbers": [7]}
    rec = _make_record(provenance=prov)
    script, params = _build_upsert_relationship_script([rec], provenance=None)

    # Both branches use the same :provenance_0 binding
    occurrences = script.count(":provenance_0")
    assert occurrences >= 2, (
        f"Expected :provenance_0 in both CREATE and UPDATE branches, "
        f"found {occurrences} occurrence(s)"
    )


# ---------------------------------------------------------------------------
# B.1 backward compat
# ---------------------------------------------------------------------------


def test_relationship_record_provenance_default_none():
    """Existing constructors without provenance still work."""
    from app.services.graph_store import RelationshipRecord

    rec = RelationshipRecord(
        from_type="A",
        from_identity={"id": "1"},
        to_type="B",
        to_identity={"id": "2"},
        rel_type="REL",
        extraction_confidence=1.0,
    )
    assert rec.provenance is None


# ---------------------------------------------------------------------------
# B.2 — get_neighborhood_graph surfaces provenance from raw DB rows
# ---------------------------------------------------------------------------


def test_get_neighborhood_graph_surfaces_provenance_dict(monkeypatch):
    """Edge dicts returned by get_neighborhood_graph must include 'provenance'
    decoded from the JSON string stored in the DB row."""
    import asyncio

    from app.services.arcadedb_graph import ArcadeDBGraphStore

    prov = {"evidence_ids": ["#/texts/2"], "self_refs": [], "page_numbers": [1]}

    # Fake DB row with provenance stored as JSON string (as written by upsert)
    fake_edge_row = {
        "edge_rid": "#15:0",
        "@type": "ENGAGES",
        "@out": "#10:0",
        "@in": "#11:0",
        "provenance": json.dumps(prov),
    }
    fake_node_row = {
        "node_id": "#11:0",
        "@type": "MISSILE",
        "name": "SA-2",
        "extraction_confidence": 0.8,
    }

    class _FakeClient:
        async def query(self, db, lang, sql, params=None):
            if "bothE" in sql:
                return [fake_edge_row]
            return [fake_node_row]

        async def command(self, *a, **kw):
            return []

    store = object.__new__(ArcadeDBGraphStore)
    store._client = _FakeClient()
    store._database = "test"

    # Stub _resolve_rid to skip actual DB lookup
    async def _fake_resolve_rid(node_id):
        return "#10:0"

    monkeypatch.setattr(store, "_resolve_rid", _fake_resolve_rid)

    # Stub get_neighborhood to avoid separate query
    async def _fake_get_neighborhood(node_id, depth, rel_types):
        return []

    monkeypatch.setattr(store, "get_neighborhood", _fake_get_neighborhood)

    result = asyncio.run(store.get_neighborhood_graph("#10:0", depth=1))

    edges = result["edges"]
    assert len(edges) == 1
    edge = edges[0]
    assert "provenance" in edge, f"'provenance' missing from edge dict: {edge}"
    assert edge["provenance"] == prov, (
        f"provenance mismatch: expected {prov}, got {edge['provenance']}"
    )


def test_get_neighborhood_graph_handles_null_provenance(monkeypatch):
    """Edges without provenance must still return 'provenance': None."""
    import asyncio

    from app.services.arcadedb_graph import ArcadeDBGraphStore

    fake_edge_row = {
        "edge_rid": "#15:1",
        "@type": "SUPPORTS",
        "@out": "#10:1",
        "@in": "#11:1",
        # no 'provenance' key — legacy edge
    }

    class _FakeClient:
        async def query(self, db, lang, sql, params=None):
            if "bothE" in sql:
                return [fake_edge_row]
            return []

        async def command(self, *a, **kw):
            return []

    store = object.__new__(ArcadeDBGraphStore)
    store._client = _FakeClient()
    store._database = "test"

    async def _fake_resolve_rid(node_id):
        return "#10:1"

    monkeypatch.setattr(store, "_resolve_rid", _fake_resolve_rid)

    async def _fake_get_neighborhood(node_id, depth, rel_types):
        return []

    monkeypatch.setattr(store, "get_neighborhood", _fake_get_neighborhood)

    result = asyncio.run(store.get_neighborhood_graph("#10:1", depth=1))
    edge = result["edges"][0]
    assert "provenance" in edge
    assert edge["provenance"] is None
