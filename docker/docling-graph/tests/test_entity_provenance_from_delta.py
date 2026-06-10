"""KEYSTONE: build_entity_provenance_from_delta_graph reads the positional
lineage Task 1 stamped onto ``context._delta_merged_graph["nodes"]`` and emits
PRECISE per-entity ExtractionProvenance + per-field ExtractionFieldProvenance.

The Pydantic→graph converter STRIPS provenance off ``context.knowledge_graph``,
so ``build_provenance_from_context`` returns [] in production and the pipeline
falls to the COARSE chunk-0 synthesizer. Task 1's stamp lands on
``_delta_merged_graph`` — which only the relationship builder read until now.
This builder is the only thing that makes entity+field lineage precise.

CRITICAL TRAP (verified across 3 review rounds): ontology_name MUST be the
``model_config["ontology_name"]`` value (e.g. "RADAR_SYSTEM"), NOT the delta
node's ``node_type`` (the class name e.g. "RadarSystemEntity"). Using the class
name makes the worker's logical_identity_from_dict drop EVERY row → silent total
lineage collapse. Likewise scalar ``page`` MUST be present (page_numbers[0]) —
the worker lineage gate rejects any entity whose provenance has page is None.

Loads provenance.py + schemas.py via the same importlib spec_from_file_location
+ sys.path swap mechanism as test_resolve_element_uid_prefers_evidence.py (the
dg_app_module conftest fixture only loads main.py; a function in provenance.py
needs the direct-import pattern).
"""
import importlib.util
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# provenance.py does `from app._numeric_evidence import ...` at module load, so
# the docling-graph SERVICE dir must be FIRST on sys.path BEFORE exec_module —
# else `app` resolves to the repo-root worker package (no _numeric_evidence).
# This mutation is RESTORED in the finally block. Mirrors the save/restore
# convention in test_resolve_element_uid_prefers_evidence.py + conftest.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent  # docker/docling-graph
_saved_path = list(sys.path)
_saved_modules = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
try:
    sys.path.insert(0, str(_SERVICE_ROOT))
    for _stale in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        del sys.modules[_stale]

    _PROV = _SERVICE_ROOT / "app" / "provenance.py"
    _spec = importlib.util.spec_from_file_location("dg_provenance_under_test_delta", _PROV)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_entity_provenance_from_delta_graph = _mod.build_entity_provenance_from_delta_graph
    _build_path_to_entity_model = _mod._build_path_to_entity_model

    _SCHEMAS = _SERVICE_ROOT / "app" / "schemas.py"
    _sspec = importlib.util.spec_from_file_location("dg_schemas_under_test_delta", _SCHEMAS)
    _smod = importlib.util.module_from_spec(_sspec)
    _sspec.loader.exec_module(_smod)
    ExtractionProvenance = _smod.ExtractionProvenance
    ExtractionFieldProvenance = _smod.ExtractionFieldProvenance
    # Loaded via spec_from_file_location → some fields use `Any` whose forward
    # ref isn't resolved until model_rebuild runs in the module's namespace.
    # The real service triggers this via normal import; do it explicitly here.
    ExtractionProvenance.model_rebuild(_types_namespace=vars(_smod))
    ExtractionFieldProvenance.model_rebuild(_types_namespace=vars(_smod))
finally:
    sys.path[:] = _saved_path
    for _key in [k for k in list(sys.modules) if k == "app" or k.startswith("app.")]:
        del sys.modules[_key]
    sys.modules.update(_saved_modules)


# --- A minimal entity model carrying the production ConfigDict contract.
class RadarSystemEntity(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",   # the value the worker keys on
        graph_id_fields=["system_name"],
        is_entity=True,
    )
    system_name: str
    band: str | None = None


class _Template(BaseModel):
    # Top-level LIST field — the normalizer/catalog stamps this node's `path`
    # WITH the `[]` list-suffix (e.g. ``radar_systems[]``), never the bare
    # field name. The builder's path→model bridge MUST key on that suffixed
    # form or every list-entity node misses → total lineage collapse.
    radar_systems: list[RadarSystemEntity] = Field(default_factory=list)


# The exact `path` the delta catalog/normalizer stamps onto a top-level
# list-entity node. Derived from the production helper itself so the fixture
# can never drift back to the bare (suffix-less) form that silently zeroes
# lineage. _build_path_to_entity_model reproduces the catalog's path shape
# (and intersects with build_delta_node_catalog when docling_graph is on
# path), so its sole key here is the canonical `radar_systems[]`.
_RADAR_PATHS = [p for p, c in _build_path_to_entity_model(_Template).items()
                if c is RadarSystemEntity]
assert _RADAR_PATHS == ["radar_systems[]"], (
    f"expected canonical []-suffixed path, got {_RADAR_PATHS!r}"
)
RADAR_NODE_PATH = _RADAR_PATHS[0]


class _Ctx:
    """Minimal stand-in for the pipeline context: only _delta_merged_graph."""
    def __init__(self, merged_graph):
        self._delta_merged_graph = merged_graph
        self.knowledge_graph = None


# Evidence-unit text source — the SAME shape main.py builds (chunk_index →
# [{text, evidence_id, ...}]) and threads into the delta builder. The field
# row's supporting_snippet is joined from the units the field cites; without a
# non-empty snippet the worker's _parse_pass_response DROPS the row, so this
# map is what lets the precise delta field rows survive into per-field lineage.
_BAND_EVIDENCE_TEXT = "The SNR-75 operates in the S band."
_EVIDENCE_UNITS = {
    0: [
        {"evidence_id": "#/texts/3", "text": _BAND_EVIDENCE_TEXT},
        {"evidence_id": "#/texts/3a", "text": "Unrelated noise that is not cited."},
    ],
    1: [
        {"evidence_id": "#/texts/4", "text": "Continuation paragraph on page 20."},
    ],
}


def _make_ctx():
    return _Ctx({
        "nodes": [
            {
                # Production-shape path: `[]`-suffixed, exactly what the
                # normalizer stamps — NOT the bare "radar_systems".
                "path": RADAR_NODE_PATH,
                "node_type": "RadarSystemEntity",   # class name — MUST NOT become ontology_name
                "ids": {"system_name": "SNR-75"},
                "properties": {"system_name": "SNR-75", "band": "S"},
                "__delta_node_uid": "b0:n0",
                "provenance": {
                    "self_refs": ["#/texts/3", "#/texts/4"],
                    "chunk_indexes": [0, 1],
                    "page_numbers": [19, 20],
                    "cited_refs": ["#/texts/3"],
                    "evidence_ids": ["#/texts/3"],
                    "property_evidence": {"band": ["#/texts/3"]},
                },
            },
        ],
        "relationships": [],
    })


def test_emits_entity_provenance_with_positional_lineage():
    ctx = _make_ctx()
    entity_rows, _field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
    )
    assert len(entity_rows) == 1, entity_rows
    row = entity_rows[0]
    # positional lineage carried verbatim
    assert row.self_refs == ["#/texts/3", "#/texts/4"]
    assert row.cited_refs == ["#/texts/3"]
    assert row.chunk_indexes == [0, 1]
    # element_uid == self_refs[0]
    assert row.element_uid == "#/texts/3"
    # SCALAR page == page_numbers[0] (REQUIRED — worker gate rejects page=None)
    assert row.page == 19
    assert row.page is not None
    # page_numbers list carried too (parity with synth / primary path)
    assert row.page_numbers == [19, 20]
    # ontology_name == the model_config VALUE, NOT the node_type class name
    assert row.ontology_name == "RADAR_SYSTEM"
    assert row.ontology_name != "RadarSystemEntity"
    # identity_values from the node's `ids` dict
    assert row.identity_values == {"system_name": "SNR-75"}


def test_emits_field_provenance_with_positional_self_refs():
    ctx = _make_ctx()
    entity_rows, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
        chunk_to_evidence_units=_EVIDENCE_UNITS,
    )
    assert any(getattr(f, "field_name", None) == "band" for f in field_rows), field_rows
    band_row = next(f for f in field_rows if f.field_name == "band")
    # field row carries the entity's positional self_refs + chunk_indexes
    assert band_row.self_refs == ["#/texts/3", "#/texts/4"]
    assert band_row.chunk_indexes == [0, 1]
    # field row instance_id is the SAME as the parent entity row's instance_id
    assert band_row.instance_id == entity_rows[0].instance_id


def test_field_provenance_supporting_snippet_is_non_empty():
    """REGRESSION GUARD (the bug this fix closes): the worker's
    _parse_pass_response DROPS any field_provenance row whose
    supporting_snippet is falsy. A delta field row emitted with
    supporting_snippet="" is therefore SILENTLY discarded by the worker even
    though it is "emitted" service-side — net-zero per-field lineage. The
    builder must join the field's cited evidence-unit text into a non-empty
    snippet so the row survives. Here `band` cites #/texts/3, whose only unit
    text is _BAND_EVIDENCE_TEXT — so the snippet must equal exactly that
    (the #/texts/3a noise + the chunk-1 #/texts/4 unit are NOT cited)."""
    ctx = _make_ctx()
    _entity_rows, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
        chunk_to_evidence_units=_EVIDENCE_UNITS,
    )
    band_row = next(f for f in field_rows if f.field_name == "band")
    # NON-EMPTY — survives the worker drop-gate.
    assert band_row.supporting_snippet, (
        "supporting_snippet must be non-empty or the worker drops the row "
        "(net-zero field lineage)"
    )
    # Resolved from ONLY the field's cited ref (#/texts/3), not the uncited
    # noise unit or the chunk-1 unit.
    assert band_row.supporting_snippet == _BAND_EVIDENCE_TEXT


def test_returns_entity_and_field_tuple():
    ctx = _make_ctx()
    result = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
    )
    assert isinstance(result, tuple) and len(result) == 2
    entity_rows, field_rows = result
    assert len(entity_rows) == 1
    assert len(field_rows) >= 1


def test_empty_when_no_delta_graph():
    ctx = _Ctx(None)
    entity_rows, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
    )
    assert entity_rows == []
    assert field_rows == []


def test_path_bridge_keys_carry_list_suffix():
    """RELEASE-BLOCKER regression guard: the path→model bridge MUST key on
    the `[]`-suffixed catalog/normalizer path, NOT the bare field name. A
    bridge keyed on "radar_systems" misses every production node whose path
    is "radar_systems[]" → item_cls is None → all rows skipped → the builder
    returns ([], []) → main.py falls to the coarse chunk-0 synth. Pin the
    canonical shape so the helper can't regress to the suffix-less form."""
    mapping = _build_path_to_entity_model(_Template)
    # The production list-entity path is the suffixed form, and it resolves
    # to the entity model whose ontology_name the worker keys on.
    assert "radar_systems[]" in mapping
    assert mapping["radar_systems[]"] is RadarSystemEntity
    # The bare (buggy) key must NOT be present — that is the exact miss the
    # old helper produced against the real delta catalog.
    assert "radar_systems" not in mapping


# --- Per-field chunk origin from __property_provenance (the lineage fix) ------
#
# merge_delta_graphs stamps node["__property_provenance"][field] = [<batch
# provenance dict>, ...] — one stamp per 1-chunk batch that emitted that
# field's value. With 1-chunk batches each stamp's chunk_indexes/self_refs/
# page_numbers point at the EXACT chunk the field's value came from. The
# builder must stamp each field row with ITS field's batch origin, not the
# entity node's aggregate (first-seen) span. This is the "50 km lives in
# chunk 9, not the chunk where the name appears" fix.
_BAND_ORIGIN_TEXT = "The radar operates in the X band."


def _make_ctx_with_property_provenance():
    """Entity SNR-75 is first-seen in chunk 0 (top-level provenance), but its
    `band` field value was emitted in a DIFFERENT batch — chunk 5, page 25 —
    recorded in __property_provenance. The merged top-level provenance keeps
    the first-seen (chunk 0) span; only __property_provenance knows band's true
    origin."""
    return _Ctx({
        "nodes": [
            {
                "path": RADAR_NODE_PATH,
                "node_type": "RadarSystemEntity",
                "ids": {"system_name": "SNR-75"},
                "properties": {"system_name": "SNR-75", "band": "X"},
                "__delta_node_uid": "b0:n0",
                # Top-level (entity) provenance = the FIRST batch the entity was
                # seen in (chunk 0, where the NAME appears) — NOT where band's
                # value lives.
                "provenance": {
                    "self_refs": ["#/texts/3"],
                    "chunk_indexes": [0],
                    "page_numbers": [19],
                    "cited_refs": ["#/texts/3"],
                    "evidence_ids": ["#/texts/3"],
                    "property_evidence": {},
                },
                # Per-field origin recorded by merge_delta_graphs: band came
                # from chunk 5 (page 25), system_name from chunk 0 (page 19).
                "__property_provenance": {
                    "system_name": [{
                        "self_refs": ["#/texts/3"],
                        "chunk_indexes": [0],
                        "page_numbers": [19],
                        "evidence_ids": ["#/texts/3"],
                    }],
                    "band": [{
                        "self_refs": ["#/texts/40"],
                        "chunk_indexes": [5],
                        "page_numbers": [25],
                        "evidence_ids": ["#/texts/40"],
                    }],
                },
            },
        ],
        "relationships": [],
    })


_PROPERTY_PROV_EVIDENCE_UNITS = {
    0: [{"evidence_id": "#/texts/3", "text": "The SNR-75 is a radar system."}],
    5: [{"evidence_id": "#/texts/40", "text": _BAND_ORIGIN_TEXT}],
}


def test_field_provenance_uses_per_field_chunk_origin_not_entity_span():
    """THE FIX: band's value was emitted in chunk 5 (page 25), recorded in
    __property_provenance — even though the entity node's top-level provenance
    is chunk 0 (page 19, where the name appears). The band field row MUST carry
    chunk 5 / page 25 / #/texts/40 — its OWN batch origin — not the entity's
    first-seen chunk 0 span."""
    ctx = _make_ctx_with_property_provenance()
    entity_rows, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
        chunk_to_evidence_units=_PROPERTY_PROV_EVIDENCE_UNITS,
    )
    band_row = next(f for f in field_rows if f.field_name == "band")
    # band's PRECISE per-field origin — chunk 5, NOT the entity's chunk 0.
    assert band_row.chunk_indexes == [5], band_row.chunk_indexes
    assert band_row.chunk_index == 5
    assert band_row.self_refs == ["#/texts/40"]
    assert band_row.element_uid == "#/texts/40"
    assert band_row.page == 25
    # snippet resolved from the field's OWN chunk-5 evidence unit.
    assert band_row.supporting_snippet == _BAND_ORIGIN_TEXT
    # the entity row still reflects the first-seen chunk 0 (unchanged).
    assert entity_rows[0].chunk_indexes == [0]


def test_field_provenance_unions_multiple_batch_origins():
    """A field emitted in TWO batches (chunk 5 and chunk 9) carries the UNION
    of both origins so the field's lineage spans every chunk that produced its
    value."""
    ctx = _make_ctx_with_property_provenance()
    node = ctx._delta_merged_graph["nodes"][0]
    node["__property_provenance"]["band"].append({
        "self_refs": ["#/texts/90"],
        "chunk_indexes": [9],
        "page_numbers": [31],
        "evidence_ids": ["#/texts/90"],
    })
    _entity_rows, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
        chunk_to_evidence_units=_PROPERTY_PROV_EVIDENCE_UNITS,
    )
    band_row = next(f for f in field_rows if f.field_name == "band")
    assert band_row.chunk_indexes == [5, 9], band_row.chunk_indexes
    assert set(band_row.self_refs) == {"#/texts/40", "#/texts/90"}


def test_field_provenance_falls_back_to_entity_span_without_property_provenance():
    """Backward compat: when a node has NO __property_provenance (legacy /
    pre-merge-stamp graphs), field rows inherit the entity's positional span
    exactly as before — no regression for the existing precise path."""
    ctx = _make_ctx()  # the original fixture: no __property_provenance
    _entity_rows, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None,
        chunk_to_evidence_units=_EVIDENCE_UNITS,
    )
    band_row = next(f for f in field_rows if f.field_name == "band")
    assert band_row.chunk_indexes == [0, 1]
    assert band_row.self_refs == ["#/texts/3", "#/texts/4"]


# --- VALUE-GROUNDING: a numeric field's row attributes to the chunk whose text
#     CONTAINS the value, not the LLM's emission chunk (over-emission fix, #67).
def _make_ctx_value_grounding():
    """max_range_km=50 was EMITTED on chunk 0 (prose, no value), but the value
    '50 km' physically lives in chunk 5. The committed lineage must point at
    chunk 5 (where the value is), not chunk 0 (where the LLM emitted it)."""
    return _Ctx({
        "nodes": [
            {
                "path": RADAR_NODE_PATH,
                "node_type": "RadarSystemEntity",
                "ids": {"system_name": "SNR-75"},
                "properties": {"system_name": "SNR-75", "max_range_km": 50},
                "__delta_node_uid": "b0:n0",
                "provenance": {
                    "self_refs": ["#/texts/3"], "chunk_indexes": [0],
                    "page_numbers": [19], "evidence_ids": ["#/texts/3"],
                },
                "__property_provenance": {
                    "max_range_km": [{
                        "self_refs": ["#/texts/3"], "chunk_indexes": [0],   # EMISSION chunk (no value)
                        "page_numbers": [19], "evidence_ids": ["#/texts/3"],
                    }],
                },
            },
        ],
        "relationships": [],
    })


_VG_EVIDENCE_UNITS = {
    0: [{"evidence_id": "#/texts/3", "text": "The SNR-75 is a fire-control radar."}],   # no value
    5: [{"evidence_id": "#/texts/40", "text": "The maximum range is 50 km against targets."}],
}


def test_field_row_value_grounds_to_chunk_containing_the_value():
    ctx = _make_ctx_value_grounding()
    _entity_rows, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None, chunk_to_evidence_units=_VG_EVIDENCE_UNITS,
    )
    row = next(f for f in field_rows if f.field_name == "max_range_km")
    # value '50 km' lives in chunk 5 → row must point there, NOT emission chunk 0
    assert row.chunk_indexes == [5], row.chunk_indexes
    assert row.self_refs == ["#/texts/40"]
    assert row.element_uid == "#/texts/40"


def test_field_row_falls_back_to_emission_when_value_not_groundable():
    """Non-text/null/unitless fields (or value absent from all chunks) keep the
    emission-chunk attribution — value-grounding only overrides on a real hit."""
    ctx = _make_ctx_value_grounding()
    # blank the value text everywhere → nothing to ground
    units = {0: [{"evidence_id": "#/texts/3", "text": "no numbers here"}],
             5: [{"evidence_id": "#/texts/40", "text": "still no numbers"}]}
    _e, field_rows = build_entity_provenance_from_delta_graph(
        ctx, _Template, ExtractionProvenance, ExtractionFieldProvenance,
        chunk_to_self_refs=None, chunk_to_evidence_units=units,
    )
    row = next(f for f in field_rows if f.field_name == "max_range_km")
    assert row.chunk_indexes == [0]  # emission fallback unchanged
