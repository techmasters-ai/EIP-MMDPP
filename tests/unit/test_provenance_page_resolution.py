import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Load provenance.py directly (docling-graph app dir, not a host package).
_SERVICE_APP_ROOT = Path(__file__).resolve().parents[2] / "docker/docling-graph/app"
_PROV = _SERVICE_APP_ROOT / "provenance.py"


def _load():
    # provenance.py imports ``app._numeric_evidence`` (its sibling in the
    # docling-graph service app dir). On the host the ``app`` package is the
    # worker's, which has no ``_numeric_evidence`` — so pre-register the
    # service module under that name before loading provenance.py (same
    # technique as test_designation_alias_overlay.py).
    if "app._numeric_evidence" not in sys.modules:
        ne_spec = importlib.util.spec_from_file_location(
            "app._numeric_evidence", _SERVICE_APP_ROOT / "_numeric_evidence.py"
        )
        ne_mod = importlib.util.module_from_spec(ne_spec)
        sys.modules["app._numeric_evidence"] = ne_mod
        ne_spec.loader.exec_module(ne_mod)
    spec = importlib.util.spec_from_file_location("dg_provenance", _PROV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _ProvStub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_synthesize_emits_resolved_page_when_map_present():
    # Fixtures MUST be real pydantic models with is_entity config — _find_model_class
    # (provenance.py) returns only BaseModel subclasses, and synthesis skips models
    # without model_config['is_entity']. A bare class would be silently skipped →
    # test would pass for the wrong reason (review finding Medium).
    from typing import List

    from pydantic import BaseModel

    prov = _load()

    class _Rec(BaseModel):
        model_config = {
            "is_entity": True,
            "ontology_name": "RADAR_SYSTEM",
            "graph_id_fields": ["system_name"],
        }
        system_name: str

    class _Tpl(BaseModel):
        radar_systems: List[_Rec] = []

    pass_output = {"radar_systems": [{"system_name": "Fan Song"}, {"system_name": "SNR-75"}]}
    rows = prov.synthesize_provenance_from_pass_output(
        pass_output=pass_output,
        template_cls=_Tpl,
        chunk_to_self_refs={0: ["#/texts/12"]},
        chunk_to_page_numbers={0: [3]},  # NEW arg
        provenance_cls=_ProvStub,
    )
    assert rows, "must synthesize a row per entity"
    assert all(r.element_uid for r in rows), "element_uid must be non-empty"
    assert all(r.page == 3 for r in rows), "page must resolve from chunk_to_page_numbers, not None"


def test_synthesize_page_none_when_no_page_map():
    # Page legitimately None when the source genuinely lacks page data — must not
    # fabricate a page; element_uid still resolves from self_refs.
    from typing import List

    from pydantic import BaseModel

    prov = _load()

    class _Rec(BaseModel):
        model_config = {
            "is_entity": True,
            "ontology_name": "RADAR_SYSTEM",
            "graph_id_fields": ["system_name"],
        }
        system_name: str

    class _Tpl(BaseModel):
        radar_systems: List[_Rec] = []

    rows = prov.synthesize_provenance_from_pass_output(
        pass_output={"radar_systems": [{"system_name": "Fan Song"}]},
        template_cls=_Tpl,
        chunk_to_self_refs={0: ["#/texts/12"]},
        provenance_cls=_ProvStub,
    )
    assert rows and rows[0].element_uid == "#/texts/12"
    assert rows[0].page is None, "page must be None when no page map supplied (no fabrication)"


def test_resolve_page_reads_same_source_as_build_context():
    # _resolve_page and the build_provenance_from_context page-fill must agree:
    # both read provenance.page_numbers[0]. Pin so the two paths can't diverge.
    prov = _load()
    node = {"provenance": {"page_numbers": [7, 9]}}
    assert prov._resolve_page(node) == 7


def test_synthesize_element_uid_and_page_co_sourced_from_same_chunk():
    # Review fix: element_uid and page must come from the SAME chunk id. With
    # chunk 0 ABSENT from both maps, the chosen chunk is the first available key
    # (here 4) — element_uid, page, chunk_index, and page_numbers must all
    # reflect chunk 4, never a mix from different chunks.
    from typing import List

    from pydantic import BaseModel

    prov = _load()

    class _Rec(BaseModel):
        model_config = {
            "is_entity": True,
            "ontology_name": "RADAR_SYSTEM",
            "graph_id_fields": ["system_name"],
        }
        system_name: str

    class _Tpl(BaseModel):
        radar_systems: List[_Rec] = []

    rows = prov.synthesize_provenance_from_pass_output(
        pass_output={"radar_systems": [{"system_name": "Fan Song"}]},
        template_cls=_Tpl,
        chunk_to_self_refs={4: ["#/texts/40"]},   # chunk 0 absent
        chunk_to_page_numbers={4: [8]},           # same key 4
        provenance_cls=_ProvStub,
    )
    assert rows
    r = rows[0]
    assert r.element_uid == "#/texts/40", "element_uid must come from chunk 4"
    assert r.page == 8, "page must come from the SAME chunk (4), not a different one"
    assert r.chunk_index == 4, "chunk_index must report the actually-chosen chunk"
    assert r.page_numbers == [8], "page_numbers must mirror the single resolved page"


def test_synthesize_sets_page_numbers_when_page_resolves():
    # Review fix: synthesized rows populate page_numbers (was always []).
    from typing import List

    from pydantic import BaseModel

    prov = _load()

    class _Rec(BaseModel):
        model_config = {
            "is_entity": True,
            "ontology_name": "RADAR_SYSTEM",
            "graph_id_fields": ["system_name"],
        }
        system_name: str

    class _Tpl(BaseModel):
        radar_systems: List[_Rec] = []

    rows = prov.synthesize_provenance_from_pass_output(
        pass_output={"radar_systems": [{"system_name": "Fan Song"}]},
        template_cls=_Tpl,
        chunk_to_self_refs={0: ["#/texts/12"]},
        chunk_to_page_numbers={0: [3]},
        provenance_cls=_ProvStub,
    )
    assert rows[0].page_numbers == [3]
    # And empty (never [None]) when no page resolves:
    rows2 = prov.synthesize_provenance_from_pass_output(
        pass_output={"radar_systems": [{"system_name": "Fan Song"}]},
        template_cls=_Tpl,
        chunk_to_self_refs={0: ["#/texts/12"]},
        provenance_cls=_ProvStub,
    )
    assert rows2[0].page_numbers == []
