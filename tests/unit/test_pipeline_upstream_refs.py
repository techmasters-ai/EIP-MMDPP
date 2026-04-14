"""Tests for upstream-ref machinery in app/workers/pipeline.py.

The docling-graph service sees these refs on document_plus_entity_refs passes
(system_links), and the merge resolver matches from_ref_id / to_ref_id against
them. Bugs here silently produce UNKNOWN_REF_ID rejections, which is why
these tests exist before the implementation.
"""
from types import SimpleNamespace

from app.workers.pipeline import _extend_upstream_refs


class _FakePassResult:
    def __init__(self, entities_by_type: dict):
        self._by_type = entities_by_type

    def iter_entities_of_type(self, entity_type: str):
        return iter(self._by_type.get(entity_type, []))


ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
        {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
    ],
}


class TestExtendUpstreamRefs:

    def _pass_def(self, primary_types):
        return SimpleNamespace(
            name="radar_domain",
            primary_entity_types=primary_types,
        )

    def test_two_entity_types_produce_unique_ids(self):
        """Bug fix: previous impl reset the counter per entity type, so
        E001/E002 from type A were overwritten by E001/E002 from type B."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [
                SimpleNamespace(system_name="Fan Song", confidence=0.9),
                SimpleNamespace(system_name="Big Bird", confidence=0.8),
            ],
            "MISSILE_SYSTEM": [
                SimpleNamespace(system_name="SA-2", confidence=0.9),
                SimpleNamespace(system_name="SA-3", confidence=0.8),
            ],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["RADAR_SYSTEM", "MISSILE_SYSTEM"]),
            ONTOLOGY,
        )
        assert sorted(refs.keys()) == ["E001", "E002", "E003", "E004"]
        # Ids are unique: each ref points to a distinct entity
        assert len({r.identity_values["system_name"] for r in refs.values()}) == 4

    def test_appending_to_existing_refs_continues_counter(self):
        """A second pass should not clobber refs accumulated from a prior pass."""
        refs: dict = {
            "E001": SimpleNamespace(
                pass_origin="reference",
                entity_type="SECTION",
                identity_values={"heading": "Intro"},
                display_label="Intro",
            ),
        }
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["RADAR_SYSTEM"]),
            ONTOLOGY,
        )
        assert "E001" in refs and refs["E001"].entity_type == "SECTION"
        assert "E002" in refs and refs["E002"].entity_type == "RADAR_SYSTEM"

    def test_identity_values_filter_to_ontology_identity_fields(self):
        """instance.__dict__ may have confidence, nomenclature, etc.; only
        ontology identity_fields belong in identity_values (merge compares
        by identity tuple, so extra keys fragment identity)."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [
                SimpleNamespace(
                    system_name="Fan Song",
                    nomenclature="SA-2-RADAR",
                    confidence=0.9,
                ),
            ],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["RADAR_SYSTEM"]), ONTOLOGY)
        assert list(refs["E001"].identity_values.keys()) == ["system_name"]
        assert refs["E001"].identity_values["system_name"] == "Fan Song"

    def test_display_label_is_populated(self):
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["RADAR_SYSTEM"]), ONTOLOGY)
        assert refs["E001"].display_label == "Fan Song"

    def test_unknown_ontology_type_is_skipped(self):
        """If a primary_entity_type isn't in the ontology, we shouldn't emit a
        ref with a malformed identity (previous impl would have crashed)."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "UNREGISTERED_TYPE": [SimpleNamespace(name="X")],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["UNREGISTERED_TYPE"]), ONTOLOGY)
        assert refs == {}

    def test_empty_primary_types_is_noop(self):
        refs: dict = {}
        pass_result = _FakePassResult({})
        _extend_upstream_refs(refs, pass_result, self._pass_def([]), ONTOLOGY)
        assert refs == {}
