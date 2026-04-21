"""Pydantic entity additions for the docling-anchor set.

See docs/plans/2026-04-21-document-structure-pass-design.md §2.
"""
from __future__ import annotations

from ontology_bundles.air_defense_v3.entities import (
    ALL_ENTITIES,
    DocumentEntity,
    FigureEntity,
)


def test_image_entity_imports_and_validates():
    from ontology_bundles.air_defense_v3.entities import ImageEntity
    img = ImageEntity(
        image_ref="#/pictures/7",
        document_id="doc-uuid-1",
        page=3,
        caption=None,
        mime_type="image/png",
        storage_key=None,
        bbox={"l": 0, "t": 0, "r": 100, "b": 100, "page": 3, "coord_origin": "TOPLEFT"},
        image_role="INLINE_IMAGE",
        confidence=1.0,
    )
    assert img.image_ref == "#/pictures/7"
    assert img.model_config["ontology_name"] == "IMAGE"
    assert img.model_config["graph_id_fields"] == ["image_ref"]
    assert img.model_config["identity_scope"] == "document"
    assert img.model_config["is_entity"] is True


def test_text_block_entity_imports_and_validates():
    from ontology_bundles.air_defense_v3.entities import TextBlockEntity
    tb = TextBlockEntity(
        text_ref="#/texts/42",
        document_id="doc-uuid-1",
        text="Some paragraph text",
        label="TEXT",
        page=3,
        confidence=1.0,
    )
    assert tb.text_ref == "#/texts/42"
    assert tb.model_config["ontology_name"] == "TEXT_BLOCK"
    assert tb.model_config["graph_id_fields"] == ["text_ref"]


def test_image_and_text_block_registered_in_ALL_ENTITIES():
    from ontology_bundles.air_defense_v3.entities import ImageEntity, TextBlockEntity
    assert ALL_ENTITIES.get("IMAGE") is ImageEntity
    assert ALL_ENTITIES.get("TEXT_BLOCK") is TextBlockEntity


def test_document_entity_has_storage_key_field():
    doc = DocumentEntity(document_number="TM 9-1425-386-12", storage_key="minio-key-abc")
    assert doc.storage_key == "minio-key-abc"


def test_figure_entity_has_storage_key_field():
    fig = FigureEntity(figure_ref="#/pictures/0", storage_key=None)
    # storage_key always null in this change — see design v2.3 §4.4.
    assert fig.storage_key is None
