"""Tests for plan Task 35a:

1. PassResult.iter_entities_of_type is walker-driven (recursive via
   walk_entity_graph) for BaseModel templates. Nested entities behind
   typed-edge fields are reachable without relying on pass-root
   attribute-name heuristics.
2. _extend_upstream_refs dedupes yielded duplicates with a detached
   scratch-dict accumulator keyed on ontology-field-ordered identity.
   Live Python instances are never mutated.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.services.extraction_merge import (
    ExtractionMetadata,
    PassResult,
)
from app.workers.pipeline import _extend_upstream_refs


# --- Fixtures --------------------------------------------------------------

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
    gain_dbi: Optional[float] = None
    antenna_type: Optional[str] = None


class RadarSystemEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="document",
        is_entity=True,
    )
    system_name: str = Field(...)
    nomenclature: Optional[str] = None
    antennas: List[AntennaEntity] = _edge(label="HAS_ANTENNA", default_factory=list)


class _Renamed(BaseModel):
    """Pass-root variant where the radar-systems field is renamed. The
    walker-driven iter_entities_of_type must still resolve RADAR_SYSTEM
    via model_config['ontology_name'], not the attribute name."""
    model_config = ConfigDict()
    radars: List[RadarSystemEntity] = Field(default_factory=list)


class _RadarDomainPass(BaseModel):
    model_config = ConfigDict()
    radar_systems: List[RadarSystemEntity] = Field(default_factory=list)


ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "document"},
        {"name": "ANTENNA", "identity_fields": ["name"], "identity_scope": "document"},
        {"name": "FREQ", "identity_fields": ["band_name", "designation"], "identity_scope": "document"},
    ],
}


class _FreqBand(BaseModel):
    model_config = ConfigDict(
        ontology_name="FREQ",
        graph_id_fields=["band_name", "designation"],
        identity_scope="document",
        is_entity=True,
    )
    band_name: str = Field(...)
    designation: str = Field(...)


class _RadarWithFreqs(BaseModel):
    model_config = ConfigDict()
    radars: List[RadarSystemEntity] = Field(default_factory=list)
    freqs: List[_FreqBand] = Field(default_factory=list)


def _make_pass_result(template, pass_name="radar_domain"):
    return PassResult(
        pass_name=pass_name,
        template_instance=template,
        metadata=ExtractionMetadata(schema_size_chars=100, structured_output_mode="strict"),
        pre_merge_rejections=[],
    )


# --- Step 1 + 3: PassResult.iter_entities_of_type (walker-driven) ---------

def test_step1_nested_antenna_discovered_via_walker():
    """Step 1: nested antenna inside radar is yielded by iter_entities_of_type.
    Previously the pass-root-only heuristic would miss nested children."""
    template = _RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="R1",
                antennas=[AntennaEntity(name="A1"), AntennaEntity(name="A2")],
            ),
        ],
    )
    pr = _make_pass_result(template)
    antennas = list(pr.iter_entities_of_type("ANTENNA"))
    assert len(antennas) == 2
    assert {a.name for a in antennas} == {"A1", "A2"}


def test_step2_pass_root_field_rename_still_works():
    """Step 2: rename pass-root field radar_systems → radars. Walker-driven
    iter_entities_of_type resolves RADAR_SYSTEM via model_config, not the
    attribute name, so the rename is invisible to callers."""
    template = _Renamed(
        radars=[RadarSystemEntity(system_name="Rx"), RadarSystemEntity(system_name="Ry")],
    )
    pr = _make_pass_result(template)
    radars = list(pr.iter_entities_of_type("RADAR_SYSTEM"))
    assert [r.system_name for r in radars] == ["Rx", "Ry"]


def test_step3_no_context_pass_result_works():
    """Step 3: test-built PassResult with no ontology/document_id context
    must not crash. iter_entities_of_type uses walker in entity-only mode."""
    template = _RadarDomainPass(
        radar_systems=[RadarSystemEntity(system_name="R1")],
    )
    pr = _make_pass_result(template)
    # No ontology/document_id wiring — entity-only mode of walker tolerates that.
    radars = list(pr.iter_entities_of_type("RADAR_SYSTEM"))
    assert [r.system_name for r in radars] == ["R1"]


def test_simple_namespace_template_heuristic_fallback_preserved():
    """Backward-compat: PassResults built around SimpleNamespace stubs (e.g.
    test_extraction_merge.py's _make_pass_result) must still work. The walker
    is the primary path; the old attribute-name heuristic is the fallback
    when template_instance is not a BaseModel."""
    template = SimpleNamespace(
        radar_system_list=[
            SimpleNamespace(system_name="Legacy", confidence=0.9),
        ],
    )
    pr = _make_pass_result(template)
    radars = list(pr.iter_entities_of_type("RADAR_SYSTEM"))
    assert [r.system_name for r in radars] == ["Legacy"]


def test_iter_entities_cache_walks_only_once_per_pass_result():
    """Memoization: walker runs exactly once per PassResult regardless of how
    many iter_entities_of_type calls fire against it (for different types)."""
    template = _RadarWithFreqs(
        radars=[RadarSystemEntity(system_name="R", antennas=[AntennaEntity(name="A")])],
        freqs=[_FreqBand(band_name="X", designation="lower")],
    )
    pr = _make_pass_result(template)

    # First call: walks
    _ = list(pr.iter_entities_of_type("ANTENNA"))
    cache_after_first = pr._walker_entities_cache
    assert cache_after_first is not None

    # Second call on a different type: reuses the cache — same object identity
    _ = list(pr.iter_entities_of_type("FREQ"))
    assert pr._walker_entities_cache is cache_after_first


# --- Step 2.dedup + union + order (plan 35a) ------------------------------

class _PassDef:
    def __init__(self, name, primary):
        self.name = name
        self.primary_entity_types = primary


def test_step2_dedup_same_antenna_across_two_radars_produces_one_ref():
    """Two RadarSystems each nesting an Antenna with identity {name:'FF-1'} →
    _extend_upstream_refs emits exactly one upstream ref for that antenna.
    Dedup keyed on (entity_type, identity_tuple) in ontology-declared order."""
    template = _RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(system_name="R1", antennas=[AntennaEntity(name="FF-1", gain_dbi=38.0)]),
            RadarSystemEntity(system_name="R2", antennas=[AntennaEntity(name="FF-1", antenna_type="phased-array")]),
        ],
    )
    pr = _make_pass_result(template)
    refs: dict = {}
    _extend_upstream_refs(
        refs, pr,
        _PassDef(name="radar_domain", primary=["RADAR_SYSTEM", "ANTENNA"]),
        ONTOLOGY,
    )
    antenna_refs = [r for r in refs.values() if r.entity_type == "ANTENNA"]
    assert len(antenna_refs) == 1
    assert antenna_refs[0].identity_values == {"name": "FF-1"}


def test_step2_accumulator_merges_complementary_fields(monkeypatch):
    """Accumulator contents: duplicate A has gain_dbi=38.0 and null antenna_type;
    duplicate B has null gain_dbi and antenna_type='phased-array'. The scratch
    dict passed to build_display_label carries BOTH values.

    Pinned to today's code: patches build_display_label with a capturing
    wrapper so we can inspect the properties dict without depending on what
    the builder does with it."""
    captured: list[dict] = []
    import app.workers.pipeline as pipeline_mod
    real_builder = pipeline_mod.build_display_label

    def _capturing(entity_type, identity_values, properties):
        captured.append(dict(properties))
        return real_builder(entity_type, identity_values, properties)

    monkeypatch.setattr(pipeline_mod, "build_display_label", _capturing)

    template = _RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(system_name="R1", antennas=[AntennaEntity(name="FF-1", gain_dbi=38.0)]),
            RadarSystemEntity(system_name="R2", antennas=[AntennaEntity(name="FF-1", antenna_type="phased-array")]),
        ],
    )
    pr = _make_pass_result(template)
    refs: dict = {}
    _extend_upstream_refs(
        refs, pr,
        _PassDef(name="radar_domain", primary=["ANTENNA"]),
        ONTOLOGY,
    )

    antenna_captures = [c for c in captured if "gain_dbi" in c or "antenna_type" in c]
    assert any(
        c.get("gain_dbi") == 38.0 and c.get("antenna_type") == "phased-array"
        for c in antenna_captures
    ), f"expected merged scratch dict, got {antenna_captures}"


def test_step2_no_mutation_of_live_instances():
    """No-mutation: the original Python instances are unchanged after
    _extend_upstream_refs returns. Guards against accidental pre-merge of
    live entity data (which later merge stages will consume)."""
    dup_a = AntennaEntity(name="FF-1", gain_dbi=38.0)
    dup_b = AntennaEntity(name="FF-1", antenna_type="phased-array")
    template = _RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(system_name="R1", antennas=[dup_a]),
            RadarSystemEntity(system_name="R2", antennas=[dup_b]),
        ],
    )
    pr = _make_pass_result(template)
    refs: dict = {}
    _extend_upstream_refs(
        refs, pr,
        _PassDef(name="radar_domain", primary=["ANTENNA"]),
        ONTOLOGY,
    )
    # Live instances unchanged — one still has only gain_dbi, the other only antenna_type.
    assert dup_a.antenna_type is None
    assert dup_a.gain_dbi == 38.0
    assert dup_b.gain_dbi is None
    assert dup_b.antenna_type == "phased-array"


def test_step2_exact_label_step1_hit_identity_name():
    """Exact label, step 1 hit (AntennaEntity with name-like identity):
    identity {'name': 'FF-1'} on both duplicates → display_label == 'FF-1'.
    Step 1 of build_display_label's resolution order returns the name
    identity value before properties are consulted."""
    template = _RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(system_name="R1", antennas=[AntennaEntity(name="FF-1", gain_dbi=38.0)]),
            RadarSystemEntity(system_name="R2", antennas=[AntennaEntity(name="FF-1", antenna_type="phased-array")]),
        ],
    )
    pr = _make_pass_result(template)
    refs: dict = {}
    _extend_upstream_refs(
        refs, pr,
        _PassDef(name="radar_domain", primary=["ANTENNA"]),
        ONTOLOGY,
    )
    antenna_refs = [r for r in refs.values() if r.entity_type == "ANTENNA"]
    assert len(antenna_refs) == 1
    assert antenna_refs[0].display_label == "FF-1"


class _NonNameIdentityEntity(BaseModel):
    """Fixture entity whose identity key is NOT in _NAME_LIKE_KEYS.
    Exercises step 2 of build_display_label's resolution order."""
    model_config = ConfigDict(
        ontology_name="NON_NAME_IDENT",
        graph_id_fields=["label_id"],
        identity_scope="document",
        is_entity=True,
    )
    label_id: str = Field(...)
    # Name-like fallback fields that should NOT be consulted when identity
    # is non-empty (step 3 never reached).
    name: Optional[str] = None
    title: Optional[str] = None


class _NonNameHolder(BaseModel):
    model_config = ConfigDict()
    items: List[_NonNameIdentityEntity] = Field(default_factory=list)


def test_step2_exact_label_step2_hit_non_name_identity():
    """Exact label, step 2 hit (non-name-like identity): entity with
    graph_id_fields=['label_id'] (NOT in _NAME_LIKE_KEYS); identity
    {label_id: 'x-42'}; duplicates carry various name-like properties.
    Assert display_label == 'x-42' — step 2 joins identity values; step 3
    never runs because step 2 returned truthy."""
    ontology = {
        "entity_types": [
            {"name": "NON_NAME_IDENT", "identity_fields": ["label_id"], "identity_scope": "document"},
        ],
    }
    template = _NonNameHolder(
        items=[
            _NonNameIdentityEntity(label_id="x-42", name="Primary"),
            _NonNameIdentityEntity(label_id="x-42", title="Backup"),
        ],
    )
    pr = _make_pass_result(template, pass_name="test_pass")
    refs: dict = {}
    _extend_upstream_refs(
        refs, pr,
        _PassDef(name="test_pass", primary=["NON_NAME_IDENT"]),
        ontology,
    )
    ref = next(iter(refs.values()))
    assert ref.display_label == "x-42"


def test_step2_order_stability_ontology_field_order_wins():
    """Identity-order stability: ontology declares identity_fields=[band_name,
    designation]. Two instances with same values but Pydantic builds them
    independently — the dedup key must follow ontology order, not dict-
    insertion order. _extend_upstream_refs produces exactly one ref."""
    ontology = {
        "entity_types": [
            {"name": "FREQ", "identity_fields": ["band_name", "designation"], "identity_scope": "document"},
        ],
    }

    # Both instances have identical identity values. Pydantic preserves field
    # declaration order in __dict__, so insertion order is the same between
    # the two instances by construction. The real dedup check is that the
    # key tuple is built from ontology order — (band_name, designation) —
    # regardless of what order the dict happens to iterate.
    template = _RadarWithFreqs(
        radars=[],
        freqs=[
            _FreqBand(band_name="X", designation="lower"),
            _FreqBand(band_name="X", designation="lower"),  # same identity
        ],
    )
    pr = _make_pass_result(template)
    refs: dict = {}
    _extend_upstream_refs(
        refs, pr,
        _PassDef(name="radar_domain", primary=["FREQ"]),
        ontology,
    )
    freq_refs = [r for r in refs.values() if r.entity_type == "FREQ"]
    assert len(freq_refs) == 1
    # Identity values preserve ontology order for stable downstream comparison.
    assert list(freq_refs[0].identity_values.keys()) == ["band_name", "designation"]


def test_nested_antenna_appears_in_upstream_refs():
    """Plan Task 35a Step 1 headline: nested antenna inside radar_domain →
    upstream_refs contains an ANTENNA ref with a real ref id. Before this
    task, iter_entities_of_type was pass-root-only and nested antennas
    stayed invisible."""
    template = _RadarDomainPass(
        radar_systems=[
            RadarSystemEntity(
                system_name="R1",
                antennas=[AntennaEntity(name="FF-1")],
            ),
        ],
    )
    pr = _make_pass_result(template)
    refs: dict = {}
    _extend_upstream_refs(
        refs, pr,
        _PassDef(name="radar_domain", primary=["RADAR_SYSTEM", "ANTENNA"]),
        ONTOLOGY,
    )
    antenna_refs = [r for r in refs.values() if r.entity_type == "ANTENNA"]
    assert len(antenna_refs) == 1
    # Ref id follows E001/E002/... format and is distinct from radar's ref.
    radar_refs = [r for r in refs.values() if r.entity_type == "RADAR_SYSTEM"]
    assert len(radar_refs) == 1
    radar_ids = [rid for rid, r in refs.items() if r.entity_type == "RADAR_SYSTEM"]
    antenna_ids = [rid for rid, r in refs.items() if r.entity_type == "ANTENNA"]
    assert radar_ids[0] != antenna_ids[0]
