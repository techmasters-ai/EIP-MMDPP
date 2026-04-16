"""A-5: Extraction-schema contract tests.

3 tests total. 2 already pass (every edge field has edge_label; non-edge
fields carry no nested dict/BaseModel) — Phase 5/6/7 invariants. The
remaining test (heading-style identity examples) xfails pending Chunk B/C.

Oracle: iterate every BaseModel subclass under
``ontology_bundles.air_defense_v3.extraction_schemas.*``.
"""
import importlib
import pkgutil
import re
from typing import get_args

import pytest
from pydantic import BaseModel

from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES


def _extraction_view_entities():
    """Collect all BaseModel subclasses from extraction_schemas/*.py."""
    import ontology_bundles.air_defense_v3.extraction_schemas as ext

    results: list[type[BaseModel]] = []
    for modinfo in pkgutil.iter_modules(ext.__path__):
        mod = importlib.import_module(f"{ext.__name__}.{modinfo.name}")
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            try:
                if isinstance(obj, type) and issubclass(obj, BaseModel):
                    results.append(obj)
            except Exception:
                pass
    return results


_HEADING_STYLE_RE = re.compile(
    r"^(\d+(\.\d+)+|\d+|[IVX]+|[A-Z]|Chapter\s|Section\s|Part\s)$",
    re.IGNORECASE,
)


@pytest.mark.xfail(strict=True, reason="Chunk B/C will fix")
def test_llm_emitted_identity_examples_not_heading_style():
    """R17, docs:18470 — extraction-view identity examples must not look like raw headings."""
    violations: list[str] = []
    for cls in _extraction_view_entities():
        cfg = getattr(cls, "model_config", {}) or {}
        if cfg.get("is_entity") is not True:
            continue
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            for ex in (field.examples or []) if field else []:
                s = str(ex).strip()
                if _HEADING_STYLE_RE.match(s):
                    violations.append(
                        f"{cls.__name__}.{fname}: example '{s}' is heading-style"
                    )
    assert not violations, "\n".join(violations)


def test_edge_fields_have_edge_label():
    """R4 — every List[Entity]/Optional[Entity] field on is_entity=True must carry edge_label."""
    violations: list[str] = []
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        if cfg.get("is_entity") is not True:
            continue
        for fname, finfo in cls.model_fields.items():
            ann = finfo.annotation
            inner_cls: type | None = None
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                inner_cls = ann
            if inner_cls is None:
                for a in get_args(ann):
                    if isinstance(a, type) and issubclass(a, BaseModel):
                        inner_cls = a
                        break
                    for b in get_args(a):
                        if isinstance(b, type) and issubclass(b, BaseModel):
                            inner_cls = b
                            break
                    if inner_cls is not None:
                        break
            if inner_cls is None:
                continue
            extra = finfo.json_schema_extra or {}
            if not (isinstance(extra, dict) and extra.get("edge_label")):
                violations.append(f"{name}.{fname}: BaseModel field missing edge_label")
    assert not violations, "\n".join(violations)


def test_no_nested_property_dicts():
    """R11 — non-edge property fields must be primitive or list[primitive]."""
    violations: list[str] = []

    def _has_nested(a) -> bool:
        if isinstance(a, type):
            return issubclass(a, dict) or issubclass(a, BaseModel)
        return any(_has_nested(x) for x in get_args(a))

    for name, cls in ALL_ENTITIES.items():
        for fname, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("edge_label"):
                continue
            if _has_nested(finfo.annotation):
                violations.append(
                    f"{name}.{fname}: non-edge property has nested dict/BaseModel"
                )
    assert not violations, "\n".join(violations)
