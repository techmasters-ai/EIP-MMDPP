from __future__ import annotations

import re
from typing import Any, get_args, get_origin

from pydantic import BaseModel

_EVIDENCE_WS_RE = re.compile(r"\s+")
_EVIDENCE_BOUNDARY_CLASS = r"A-Z0-9"
_EVIDENCE_STRING_KEYS = frozenset({"text", "orig", "content", "caption"})


def normalize_evidence_text(value: str) -> str:
    return _EVIDENCE_WS_RE.sub(
        " ",
        value.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .upper(),
    ).strip()


def collect_batch_evidence_text(docling_document_json: dict[str, Any]) -> str:
    parts: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key in _EVIDENCE_STRING_KEYS and value.strip():
                    parts.append(value)
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(docling_document_json)
    return normalize_evidence_text("\n".join(parts))


def _find_model_class(annotation: Any) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        return None
    for arg in get_args(annotation):
        nested = _find_model_class(arg)
        if nested is not None:
            return nested
    return None


def identity_is_supported_by_batch_text(identity_value: Any, evidence_text: str) -> bool:
    if not isinstance(identity_value, str):
        return False
    normalized = normalize_evidence_text(identity_value)
    if not normalized or not evidence_text:
        return False
    pattern = rf"(?<![{_EVIDENCE_BOUNDARY_CLASS}]){re.escape(normalized)}(?:S)?(?![{_EVIDENCE_BOUNDARY_CLASS}])"
    return re.search(pattern, evidence_text) is not None


def filter_pass_output_by_batch_text(
    pass_output: dict[str, Any],
    template_cls: type[BaseModel],
    evidence_text: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, set[tuple[tuple[str, str], ...]]]]:
    if not evidence_text or not isinstance(pass_output, dict):
        return pass_output, {}, {}

    filtered = dict(pass_output)
    stats: dict[str, Any] = {
        "dropped_entities_by_field": {},
        "dropped_entity_examples": {},
    }
    allowed_identities: dict[str, set[tuple[tuple[str, str], ...]]] = {}

    for field_name, field_info in template_cls.model_fields.items():
        raw_items = filtered.get(field_name)
        if not isinstance(raw_items, list):
            continue
        item_cls = _find_model_class(getattr(field_info, "annotation", None))
        if item_cls is None:
            continue
        model_config = item_cls.model_config or {}
        if not model_config.get("is_entity"):
            continue
        id_fields = tuple(model_config.get("graph_id_fields", []) or [])
        if not id_fields:
            continue

        kept_items: list[Any] = []
        dropped: list[str] = []
        ontology_name = str(model_config.get("ontology_name", item_cls.__name__))
        allowed_for_ontology = allowed_identities.setdefault(ontology_name, set())

        for item in raw_items:
            if not isinstance(item, dict):
                kept_items.append(item)
                continue
            primary_identity = item.get(id_fields[0])
            if identity_is_supported_by_batch_text(primary_identity, evidence_text):
                kept_items.append(item)
                allowed_for_ontology.add(
                    tuple(sorted(
                        (field, str(item.get(field)))
                        for field in id_fields
                        if item.get(field) is not None
                    ))
                )
            else:
                dropped.append(str(primary_identity))

        if dropped:
            stats["dropped_entities_by_field"][field_name] = len(dropped)
            stats["dropped_entity_examples"][field_name] = dropped[:10]
            filtered[field_name] = kept_items

    return filtered, stats, allowed_identities


def filter_provenance_rows_by_allowed_identities(
    provenance_rows: list[Any],
    allowed_identities: dict[str, set[tuple[tuple[str, str], ...]]],
) -> list[Any]:
    if not provenance_rows or not allowed_identities:
        return provenance_rows

    filtered_rows: list[Any] = []
    for row in provenance_rows:
        ontology_name = getattr(row, "ontology_name", None)
        if not ontology_name:
            filtered_rows.append(row)
            continue
        allowed = allowed_identities.get(str(ontology_name))
        if not allowed:
            continue
        identity_values = getattr(row, "identity_values", {}) or {}
        key = tuple(
            sorted(
                (str(k), str(v))
                for k, v in identity_values.items()
                if v is not None
            )
        )
        if key in allowed:
            filtered_rows.append(row)
    return filtered_rows
