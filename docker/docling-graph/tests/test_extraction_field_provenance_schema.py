"""Task 16: verify chunk_index + cell_refs fields added to ExtractionFieldProvenance.

Per spec rev. 7 §11.6 channel A. Additive, backwards-compatible.

Uses the dg_schemas fixture (defined in conftest.py) so the
docling-graph app/schemas.py loads instead of the repo-root
app/schemas package. Calls _rebuild() per the existing
test_field_provenance.py pattern to resolve forward-ref Any/Optional
under `from __future__ import annotations`.
"""
from __future__ import annotations


def _rebuild(dg_schemas):
    """model_rebuild with typing namespace so `Any` and `Optional`
    resolve under from __future__ import annotations.
    Same shim as test_field_provenance.py:49-55."""
    import typing
    dg_schemas.ExtractionFieldProvenance.model_rebuild(
        _types_namespace={"Any": typing.Any, "Optional": typing.Optional}
    )


def test_chunk_index_field_exists(dg_schemas):
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
        chunk_index=3,
    )
    assert p.chunk_index == 3


def test_chunk_index_defaults_none(dg_schemas):
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
    )
    assert p.chunk_index is None


def test_cell_refs_field_exists_default_empty(dg_schemas):
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
    )
    assert p.cell_refs == []


def test_cell_refs_accepts_table_cell_refs(dg_schemas):
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
        cell_refs=["#/tables/3/data/table_cells/42", "#/tables/3/data/table_cells/43"],
    )
    assert len(p.cell_refs) == 2
    assert p.cell_refs[0].startswith("#/tables/")


def test_serialization_preserves_new_fields(dg_schemas):
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
        chunk_index=3, cell_refs=["#/tables/0/data/table_cells/0"],
    )
    d = p.model_dump()
    assert d["chunk_index"] == 3
    assert d["cell_refs"] == ["#/tables/0/data/table_cells/0"]


def test_backwards_compatibility_existing_fields(dg_schemas):
    """Existing field set still works without the new ones."""
    _rebuild(dg_schemas)
    p = dg_schemas.ExtractionFieldProvenance(
        instance_id="x", field_name="max_range_m",
        supporting_snippet="56000 m",
        evidence_id="#/texts/42",
    )
    assert p.evidence_id == "#/texts/42"
    assert p.chunk_index is None
    assert p.cell_refs == []
