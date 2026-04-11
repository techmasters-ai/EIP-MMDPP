"""Tests that every extraction schema satisfies the partial-safety
and ontology-subset contracts (spec §3 + checker rules 6, 8, 9)."""
import pytest
from typing import get_args, get_origin, Union
from pydantic import BaseModel

from ontology_bundles.air_defense_v3.extraction_schemas import (
    reference, radar_domain, missile_domain, other_systems, system_links,
)

PASS_MODULES = [
    (reference, "ReferencePass"),
    (radar_domain, "RadarDomainPass"),
    (missile_domain, "MissileDomainPass"),
    (other_systems, "OtherSystemsPass"),
    (system_links, "SystemLinksPass"),
]

SYSTEM_FIELDS = {"confidence"}


def _unwrap_models(annotation):
    if annotation is None:
        return []
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        return []
    results = []
    for arg in get_args(annotation):
        results.extend(_unwrap_models(arg))
    return results


def _iter_nested_models(model_cls: type[BaseModel]):
    """Walk fields recursively, yielding every nested BaseModel."""
    seen = {model_cls}
    stack = [model_cls]
    while stack:
        cls = stack.pop()
        for field_name, field_info in cls.model_fields.items():
            for nested in _unwrap_models(field_info.annotation):
                if nested not in seen:
                    seen.add(nested)
                    stack.append(nested)
                    yield nested


@pytest.mark.parametrize("module,class_name", PASS_MODULES)
def test_top_level_class_exists(module, class_name):
    assert hasattr(module, class_name), f"{module.__name__} missing {class_name}"
    cls = getattr(module, class_name)
    assert issubclass(cls, BaseModel)


@pytest.mark.parametrize("module,class_name", PASS_MODULES)
def test_top_level_instantiates_empty(module, class_name):
    cls = getattr(module, class_name)
    instance = cls()
    assert instance is not None


@pytest.mark.parametrize("module,class_name", PASS_MODULES)
def test_all_fields_optional_or_default_recursive(module, class_name):
    cls = getattr(module, class_name)
    for model in [cls, *_iter_nested_models(cls)]:
        for field_name, field_info in model.model_fields.items():
            if field_name in SYSTEM_FIELDS:
                continue
            is_optional = (
                not field_info.is_required()
                or field_info.default is not None
                or field_info.default_factory is not None
            )
            assert is_optional, (
                f"{model.__name__}.{field_name} is required; "
                f"extraction models must tolerate partial LLM output"
            )


def test_system_links_has_no_entity_fields():
    """Rule from spec §2 manifest self-consistency: input_mode=
    document_plus_entity_refs implies no entity collections."""
    cls = system_links.SystemLinksPass
    for field_name, field_info in cls.model_fields.items():
        if field_name == "relationships":
            continue
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is list:
            inner = get_args(annotation)[0]
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                assert "Relationship" in inner.__name__ or "Link" in inner.__name__, (
                    f"SystemLinksPass.{field_name} is a list of {inner.__name__}, "
                    f"which looks like an entity collection. system_links must "
                    f"have no entity fields (spec §3.5)."
                )
