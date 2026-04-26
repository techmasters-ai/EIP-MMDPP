"""Phase 3 task 29 — resolves the pass-template's primary entity list
field via Pydantic introspection.

Used by the post-LLM provenance resolver to index entity_index into
the right list when the pass-template uses a custom field name like
``radar_systems`` or ``missile_systems`` (spec §5.1.1).
"""
from __future__ import annotations

from typing import get_args, get_origin

from pydantic import BaseModel


def _primary_list_field_name(template_cls: type[BaseModel], primary_type: str) -> str:
    """Walk template_cls.model_fields, return the first field whose
    annotation is ``list[<class with model_config['ontology_name']
    matching primary_type>]``.

    Raises ValueError when no such field exists (manifest mismatch).
    """
    for fname, finfo in template_cls.model_fields.items():
        ann = finfo.annotation
        if get_origin(ann) is list:
            args = get_args(ann)
            if not args:
                continue
            item_type = args[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                cfg = getattr(item_type, "model_config", None) or {}
                if cfg.get("ontology_name") == primary_type:
                    return fname
    raise ValueError(
        f"No primary list field for ontology_name={primary_type!r} on {template_cls.__name__}"
    )
