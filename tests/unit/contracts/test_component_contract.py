"""A-5: Component reachability contract (1 xfailed pending Chunk B).

Every ``is_entity=False`` class in ``ALL_ENTITIES`` must be reachable
via an ``edge()`` field from at least one entity — otherwise it would
never become a graph node (the walker only emits components reached
through edge_label, per A0-2).
"""
from typing import get_args

import pytest
from pydantic import BaseModel

from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES


@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_component_fields_attached_via_edge_helper():
    components = {
        name: cls
        for name, cls in ALL_ENTITIES.items()
        if (getattr(cls, "model_config", {}) or {}).get("is_entity") is False
    }
    if not components:
        pytest.skip("No components yet (pre-Chunk-B)")

    def _find(a) -> str | None:
        if isinstance(a, type) and issubclass(a, BaseModel):
            return getattr(a, "__name__", None)
        for x in get_args(a):
            r = _find(x)
            if r:
                return r
        return None

    reachable: set[str] = set()
    component_names = {c.__name__ for c in components.values()}
    for cls in ALL_ENTITIES.values():
        for finfo in cls.model_fields.values():
            extra = finfo.json_schema_extra or {}
            if not (isinstance(extra, dict) and extra.get("edge_label")):
                continue
            target_name = _find(finfo.annotation)
            if target_name in component_names:
                reachable.add(target_name)
    orphans = [cname for cname in component_names if cname not in reachable]
    assert not orphans, f"Components not reachable via edge(): {orphans}"
