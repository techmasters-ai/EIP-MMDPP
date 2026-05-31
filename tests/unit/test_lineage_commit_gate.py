import pytest
pytestmark = pytest.mark.unit
from app.workers import pipeline


class _Prov:
    def __init__(self, element_uid, page):
        self.element_uid = element_uid
        self.page = page


class _Ident:
    def __init__(self, n):
        self._n = n
        self.entity_type = "RADAR_SYSTEM"

    def identity_values_dict(self):
        return {"system_name": self._n}

    def as_upsert_identity_dict(self):
        return {"system_name": self._n}

    def __hash__(self):
        return hash(self._n)

    def __eq__(self, o):
        return isinstance(o, _Ident) and o._n == self._n


class _Ent:
    def __init__(self, n, prov):
        self.identity = _Ident(n)
        self.properties = {}
        self.confidence = 0.9
        self.provenance = prov


class _Edge:
    # Match real MergedEdgeRecord field names (extraction_merge.py:366-367).
    def __init__(self, src, dst):
        self.from_identity = src
        self.to_identity = dst


class _Merged:
    def __init__(self, ents, edges):
        self.entities = ents
        self.edges = edges


def test_gate_filters_merged_and_prunes_edges():
    ok = _Ent("Fan Song", [_Prov("#/texts/12", 3)])      # lineage-complete
    bad = _Ent("Ghost", [_Prov("", None)])               # no lineage
    merged = _Merged([ok, bad], [_Edge(ok.identity, bad.identity)])  # edge to rejected
    rejected = pipeline._partition_entities_by_lineage(merged)
    assert [e.identity._n for e in merged.entities] == ["Fan Song"]   # only lineage-ok kept
    assert len(rejected) == 1 and rejected[0].identity._n == "Ghost"
    assert merged.edges == []                                          # dangling edge pruned


def test_gate_noop_when_all_have_lineage():
    a = _Ent("A", [_Prov("#/texts/1", 1)])
    b = _Ent("B", [_Prov("#/texts/2", 2)])
    merged = _Merged([a, b], [_Edge(a.identity, b.identity)])
    rejected = pipeline._partition_entities_by_lineage(merged)
    assert rejected == [] and len(merged.entities) == 2 and len(merged.edges) == 1
