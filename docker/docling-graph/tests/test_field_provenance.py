"""Task 10 — ExtractionFieldProvenance new optional fields + evidence-unit widening."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


_DG_SERVICE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dg_provenance(dg_schemas):
    """Load docker/docling-graph/app/provenance.py under an alt module name.
    Returns the provenance module directly (not a tuple)."""
    module_name = "docling_graph_service_provenance_fp"

    if module_name in sys.modules:
        return sys.modules[module_name]

    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    saved_path = list(sys.path)

    sys.path.insert(0, str(_DG_SERVICE_ROOT))
    for key in list(saved.keys()):
        del sys.modules[key]

    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            _DG_SERVICE_ROOT / "app" / "provenance.py",
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

    return mod


def _rebuild(dg_schemas):
    """model_rebuild with typing namespace so `Any` resolves under
    from __future__ import annotations."""
    import typing
    dg_schemas.ExtractionFieldProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )


def test_field_provenance_backward_compat(dg_schemas):
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="i1",
        field_name="diameter",
        value=0.37,
        supporting_snippet="diameter: 0.37 m",
    )
    assert p.evidence_id is None
    assert p.page is None
    assert p.document_id is None


def test_field_provenance_accepts_new_fields(dg_schemas):
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="i1",
        field_name="diameter",
        value=0.37,
        supporting_snippet="diameter: 0.37 m",
        element_uid="#/texts/12",
        evidence_id="#/texts/12",
        page=3,
        document_id="doc-uuid-abc",
    )
    assert p.evidence_id == "#/texts/12"
    assert p.page == 3
    assert p.document_id == "doc-uuid-abc"


def test_build_auto_field_evidence_accepts_evidence_units(dg_schemas, dg_provenance):
    """build_auto_field_evidence must accept evidence-unit-shaped inputs
    (dicts with evidence_id/text/page_numbers) in addition to the legacy
    (element_uid, text) tuple shape. Extra metadata flows into the emitted
    ExtractionFieldProvenance."""
    from pydantic import BaseModel
    from typing import Optional

    class _StubEntity(BaseModel):
        diameter: Optional[float] = None

    units = [
        {"evidence_id": "#/texts/12", "self_ref": "#/texts/12",
         "text": "diameter: 0.37 m", "page_numbers": [3]},
    ]
    _rebuild(dg_schemas)
    primary_entities = [_StubEntity(diameter=0.37)]
    rows = dg_provenance.build_auto_field_evidence(
        primary_entities=primary_entities,
        instance_ids=["i1"],
        input_chunks=units,
        skip_fields=set(),
        provenance_cls=dg_schemas.ExtractionFieldProvenance,
    )
    assert any(
        r.field_name == "diameter" and r.evidence_id == "#/texts/12" and r.page == 3
        for r in rows
    )


# ---------------------------------------------------------------------------
# Fix D — unified _resolve_instance_id helper
# ---------------------------------------------------------------------------

def test_resolve_instance_id_unified_order(dg_provenance):
    """Both entity and relationship lookup use the same fallback chain."""
    fn = dg_provenance._resolve_instance_id
    assert fn({"instance_id": "i1"}) == "i1"
    assert fn({"node_uid": "u1"}) == "u1"
    assert fn({"__delta_node_uid": "d1"}) == "d1"
    assert fn({}, fallback="nx_node") == "nx_node"
    assert fn({}) is None


def test_resolve_instance_id_instance_id_wins(dg_provenance):
    """instance_id takes priority over node_uid and __delta_node_uid."""
    fn = dg_provenance._resolve_instance_id
    assert fn({"instance_id": "i1", "node_uid": "u1", "__delta_node_uid": "d1"}) == "i1"


def test_resolve_instance_id_fallback_skipped_when_key_present(dg_provenance):
    """Explicit fallback is not used when a key resolves."""
    fn = dg_provenance._resolve_instance_id
    assert fn({"node_uid": "u1"}, fallback="nx_node") == "u1"
