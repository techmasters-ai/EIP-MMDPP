"""D-1: `_to_merged_entity_record` helper tests.

Exercises the walker-side record builder that turns a Pydantic entity
model (as emitted by the upcoming Docling anchor walker) into a
MergedEntityRecord ready for the merge index. Spec §3.4 + §8.2.

All fixture models are is_entity=True entities with model_config
graph_id_fields — the helper must derive identity from model_config
directly (ontology dict is not populated for walker-sourced passes).
"""
from __future__ import annotations

from app.services.extraction_merge import _to_merged_entity_record
from ontology_bundles.air_defense_v3.entities import (
    DocumentEntity,
    SectionEntity,
)


def test_section_merged_record_includes_document_id_and_section_path():
    sec = SectionEntity(
        section_number="1.1",
        heading="Foo",
        section_path="Chapter 1 > Foo",
    )
    rec = _to_merged_entity_record(sec, ontology={}, document_id="doc-uuid-1")
    assert rec.identity.entity_type == "SECTION"
    assert rec.identity.identity_tuple == ("1.1",)
    assert rec.identity.scope == "document"
    assert rec.identity.document_id == "doc-uuid-1"
    assert rec.properties["document_id"] == "doc-uuid-1"
    assert rec.properties["section_path"] == "Chapter 1 > Foo"
    assert rec.properties["heading"] == "Foo"
    assert "section_number" not in rec.properties  # identity is not a property
    assert rec.pass_origins == {"document_anchors"}
    assert rec.confidence == 1.0
    # display_label uses heading (name-like key ranks after system_name/name/title)
    # — section_number is the identity value; heading is in properties only.
    # build_display_label resolves to the first identity-tuple value when no
    # name-like identity key exists, giving "1.1".
    assert rec.display_label == "1.1"


def test_document_merged_record_identity_is_document_number():
    doc = DocumentEntity(document_number="TM 9-1425-386-12", title="Foo Manual")
    rec = _to_merged_entity_record(doc, ontology={}, document_id="doc-uuid-1")
    assert rec.identity.entity_type == "DOCUMENT"
    assert rec.identity.identity_tuple == ("TM 9-1425-386-12",)
    assert rec.identity.scope == "global"
    assert rec.identity.document_id is None  # global scope → no document_id
    assert rec.properties["document_id"] == "doc-uuid-1"
    assert rec.properties["title"] == "Foo Manual"
    # display_label: title is a name-like key in properties; identity has no
    # name-like key — but build_display_label's identity-fallback picks the
    # identity tuple first. That is "TM 9-1425-386-12".
    assert rec.display_label == "TM 9-1425-386-12"


def test_section_sentinel_no_section_path():
    sec = SectionEntity(section_number="0", heading=None, section_path=None)
    rec = _to_merged_entity_record(sec, ontology={}, document_id="doc-uuid-1")
    assert rec.properties.get("section_path") is None
    assert rec.properties.get("heading") is None
    assert rec.properties["document_id"] == "doc-uuid-1"


def test_pass_origin_override():
    sec = SectionEntity(section_number="2.3")
    rec = _to_merged_entity_record(
        sec, ontology={}, document_id="doc-uuid-1", pass_origin="some_other_pass"
    )
    assert rec.pass_origins == {"some_other_pass"}


def test_edge_fields_excluded_from_properties():
    """DocumentEntity declares a `documents: List[DocumentEntity]` edge via
    edge(label='REFERENCES'). That field must not leak into properties."""
    doc = DocumentEntity(document_number="TM X")
    rec = _to_merged_entity_record(doc, ontology={}, document_id="doc-uuid-1")
    assert "documents" not in rec.properties
