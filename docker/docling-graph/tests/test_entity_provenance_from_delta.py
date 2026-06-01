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
