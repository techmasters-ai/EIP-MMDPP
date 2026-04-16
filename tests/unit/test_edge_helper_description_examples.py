"""A-1: edge() helper must accept + forward description and examples kwargs.

Test mirrors the contract ``test_descriptions_and_examples_on_extraction_
relevant_fields`` (Task 9c) — edge fields are extraction-relevant and
therefore must carry prose + examples for LLM guidance.
"""
from pydantic import BaseModel

from ontology_bundles.air_defense_v3.entities import edge


def test_edge_helper_accepts_description_and_examples():
    class Parent(BaseModel):
        children: list = edge(
            label="HAS_CHILD",
            description="Children of this parent",
            examples=[[]],
            default_factory=list,
        )

    info = Parent.model_fields["children"]
    assert info.description == "Children of this parent"
    assert info.examples == [[]]
    extra = info.json_schema_extra or {}
    assert extra.get("edge_label") == "HAS_CHILD"


def test_edge_helper_without_description_still_works():
    """Backwards compatibility — description and examples stay optional."""
    class Parent(BaseModel):
        children: list = edge(label="HAS_CHILD", default_factory=list)

    info = Parent.model_fields["children"]
    assert info.description is None
    extra = info.json_schema_extra or {}
    assert extra.get("edge_label") == "HAS_CHILD"


def test_edge_helper_preserves_existing_json_schema_extra():
    """Callers that supply their own json_schema_extra keep those keys."""
    class Parent(BaseModel):
        children: list = edge(
            label="HAS_CHILD",
            default_factory=list,
            json_schema_extra={"extraction_hint": "collect all"},
        )

    info = Parent.model_fields["children"]
    extra = info.json_schema_extra or {}
    assert extra.get("edge_label") == "HAS_CHILD"
    assert extra.get("extraction_hint") == "collect all"
