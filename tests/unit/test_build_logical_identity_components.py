from pydantic import BaseModel, ConfigDict
from app.services.extraction_merge import _build_logical_identity


class SampleComponent(BaseModel):
    model_config = ConfigDict(is_entity=False, ontology_name="SAMPLE_COMP")
    alpha: str | None = None
    beta: int | None = None
    gamma: list[str] = []


def test_component_identity_is_all_fields_canonical():
    """Per docs:17235 — components dedup by ENTIRE content (not non-None subset)."""
    inst_a = SampleComponent(alpha="x", beta=None, gamma=["u", "v"])
    inst_b = SampleComponent(alpha="x", beta=None, gamma=["u", "v"])
    inst_c = SampleComponent(alpha="x", beta=1, gamma=["u", "v"])
    id_a = _build_logical_identity("SAMPLE_COMP", inst_a, {}, "doc-1")
    id_b = _build_logical_identity("SAMPLE_COMP", inst_b, {}, "doc-1")
    id_c = _build_logical_identity("SAMPLE_COMP", inst_c, {}, "doc-1")
    assert id_a == id_b, "identical content → identical identity"
    assert id_a != id_c, "different content (beta) → different identity"


def test_component_identity_includes_none_values():
    """None value is part of the canonical form."""
    inst_null = SampleComponent(alpha="x", beta=None, gamma=[])
    inst_empty = SampleComponent(alpha="x", beta=None, gamma=[])
    id_null = _build_logical_identity("SAMPLE_COMP", inst_null, {}, "doc-1")
    id_empty = _build_logical_identity("SAMPLE_COMP", inst_empty, {}, "doc-1")
    assert id_null == id_empty
