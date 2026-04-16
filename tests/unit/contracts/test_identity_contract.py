"""A-4: Identity contract tests for canonical entities.

7 tests total. 3 already pass (identity fields are required; examples
are short; non-identity fields are Optional) — Phase 5/6/7 satisfied
those invariants before this plan. The remaining 4 are
``@pytest.mark.xfail(strict=True)`` pending Chunk B's canonical
rewrite; F-7 drops the xfail markers after Chunk B lands.

Oracle: ``ontology_bundles.air_defense_v3.entities.ALL_ENTITIES``.
"""
import pytest
from pydantic_core import PydanticUndefined

from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES


@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_entity_has_identity_or_is_component():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        is_entity = cfg.get("is_entity", True)
        id_fields = cfg.get("graph_id_fields") or []
        if is_entity:
            assert id_fields, f"{name}: is_entity=True but graph_id_fields empty"
        else:
            assert not id_fields, f"{name}: is_entity=False but has graph_id_fields"


def test_identity_fields_are_required():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            assert field is not None and field.is_required(), \
                f"{name}.{fname}: identity field must be required"


def test_identity_field_examples_are_short():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            if field and field.examples:
                for ex in field.examples:
                    assert isinstance(ex, (str, int, float, bool))
                    s = str(ex)
                    assert len(s) <= 80, f"{name}.{fname} example '{s[:40]}...' >80 chars"
                    assert "\n" not in s, f"{name}.{fname} example contains newline"


@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_fields_not_named_heading_or_title():
    banned = {"heading", "title", "caption", "description"}
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        for fname in cfg.get("graph_id_fields") or []:
            assert fname not in banned, f"{name}: identity field '{fname}' is a banned name"


@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_examples_are_distinct():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            if field and field.examples:
                assert len(field.examples) == len(set(map(repr, field.examples))), \
                    f"{name}.{fname}: examples contain duplicates"


@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_example_values_populated_for_library_filter():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        if cfg.get("is_entity") is not True:
            continue
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            n_examples = len(field.examples) if field and field.examples else 0
            assert n_examples >= 2, \
                f"{name}.{fname}: needs >=2 examples for library identity_example_values"


def test_non_identity_fields_are_optional():
    """R19: every field NOT in graph_id_fields must be Optional[T] with default=None."""
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        id_fields = set(cfg.get("graph_id_fields") or [])
        for fname, finfo in cls.model_fields.items():
            if fname in id_fields:
                continue
            extra = finfo.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("edge_label"):
                continue  # edges handled separately
            assert finfo.default is not PydanticUndefined or not finfo.is_required(), \
                f"{name}.{fname}: non-identity field is required (should be Optional[T]=None)"
