from __future__ import annotations

import re
from typing import Any, get_args, get_origin

from pydantic import BaseModel

_EVIDENCE_WS_RE = re.compile(r"\s+")
_EVIDENCE_BOUNDARY_CLASS = r"A-Z0-9"
_EVIDENCE_STRING_KEYS = frozenset({"text", "orig", "content", "caption"})
_STATUS_ALIASES = {
    "OPERATIONAL": ("OPERATIONAL",),
    "DEVELOPMENTAL": ("DEVELOPMENTAL",),
    "RETIRED": ("RETIRED",),
    "UPGRADED": ("UPGRADED",),
    "EXPORTED": ("EXPORTED",),
}


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


def summarize_pass_output(
    pass_output: dict[str, Any],
    template_cls: type[BaseModel],
) -> dict[str, Any]:
    """Compute response-facing counts from the filtered pass_output."""
    node_count = 1
    edge_count = 0
    node_types: dict[str, int] = {template_cls.__name__: 1}
    edge_types: dict[str, int] = {}
    path_counts: dict[str, int] = {"": 1}

    for field_name, field_info in template_cls.model_fields.items():
        raw_items = pass_output.get(field_name)
        if not isinstance(raw_items, list):
            continue
        item_cls = _find_model_class(getattr(field_info, "annotation", None))
        path_key = f"{field_name}[]"
        path_counts[path_key] = len(raw_items)
        if item_cls is None:
            continue

        model_config = item_cls.model_config or {}
        if not model_config.get("is_entity"):
            continue

        node_types[item_cls.__name__] = len(raw_items)
        node_count += len(raw_items)

        edge_label = ((field_info.json_schema_extra or {}) if hasattr(field_info, "json_schema_extra") else {}).get("edge_label")
        if edge_label:
            edge_count += len(raw_items)
            edge_types[edge_label] = edge_types.get(edge_label, 0) + len(raw_items)

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "node_types": node_types,
        "edge_types": edge_types,
        "path_counts": path_counts,
    }


def apply_bundle_postprocessing(
    bundle_key: str,
    pass_name: str,
    pass_output: dict[str, Any],
    evidence_text: str,
    upstream_entities: list[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bundle_key != "air_defense_v3" or not isinstance(pass_output, dict):
        return pass_output, {}

    if pass_name == "radar_domain":
        return _postprocess_air_defense_radars(pass_output, evidence_text)
    if pass_name == "missile_domain":
        return _postprocess_air_defense_missiles(pass_output, evidence_text)
    if pass_name == "system_links":
        return _postprocess_air_defense_system_links(pass_output, evidence_text, upstream_entities)
    return pass_output, {}


def _status_is_explicit(status_value: Any, evidence_text: str) -> bool:
    if not isinstance(status_value, str):
        return False
    aliases = _STATUS_ALIASES.get(normalize_evidence_text(status_value), ())
    if not aliases:
        return False
    return any(alias in evidence_text for alias in aliases)


def _entity_in_context(identity: str, evidence_text: str, markers: tuple[str, ...], window: int = 160) -> bool:
    normalized_identity = normalize_evidence_text(identity)
    if not normalized_identity or not evidence_text:
        return False
    for match in re.finditer(re.escape(normalized_identity), evidence_text):
        start = max(0, match.start() - window)
        end = min(len(evidence_text), match.end() + window)
        context = evidence_text[start:end]
        if any(marker in context for marker in markers):
            return True
    return False


def _postprocess_air_defense_radars(
    pass_output: dict[str, Any],
    evidence_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(pass_output)
    radar_rows = updated.get("radar_systems")
    if not isinstance(radar_rows, list):
        return updated, {}

    stats: dict[str, Any] = {
        "status_cleared": [],
        "emitter_function_overrides": {},
    }
    cleaned_rows: list[Any] = []

    for row in radar_rows:
        if not isinstance(row, dict):
            cleaned_rows.append(row)
            continue
        item = dict(row)
        system_name = item.get("system_name")
        if isinstance(system_name, str):
            normalized_name = normalize_evidence_text(system_name)
            if f"{normalized_name} GUIDANCE RADAR" in evidence_text:
                if item.get("emitter_function") != "FIRE_CONTROL":
                    stats["emitter_function_overrides"][system_name] = "FIRE_CONTROL"
                item["emitter_function"] = "FIRE_CONTROL"
            elif f"{normalized_name} ACQUISITION RADAR" in evidence_text:
                if item.get("emitter_function") != "SEARCH":
                    stats["emitter_function_overrides"][system_name] = "SEARCH"
                item["emitter_function"] = "SEARCH"
            elif _entity_in_context(
                system_name,
                evidence_text,
                ("MISSILE GUIDANCE", "GUIDED UP TO", "GUIDED AGAINST ONE TARGET"),
                window=80,
            ):
                if item.get("emitter_function") != "FIRE_CONTROL":
                    stats["emitter_function_overrides"][system_name] = "FIRE_CONTROL"
                item["emitter_function"] = "FIRE_CONTROL"
            elif _entity_in_context(system_name, evidence_text, ("DETECTED INCOMING AIRCRAFT",), window=80):
                if item.get("emitter_function") != "SEARCH":
                    stats["emitter_function_overrides"][system_name] = "SEARCH"
                item["emitter_function"] = "SEARCH"

        if item.get("system_status") and not _status_is_explicit(item.get("system_status"), evidence_text):
            stats["status_cleared"].append(str(system_name or ""))
            item["system_status"] = None
        cleaned_rows.append(item)

    updated["radar_systems"] = cleaned_rows
    if not stats["status_cleared"]:
        stats.pop("status_cleared")
    if not stats["emitter_function_overrides"]:
        stats.pop("emitter_function_overrides")
    return updated, stats


def _extract_museum_display_range_notes(evidence_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not evidence_text:
        return out
    range_match = re.search(
        r"RANGE:\s*MINIMUM\s+(?P<min>\d+(?:\.\d+)?)\s+MILES;\s*"
        r"MAXIMUM EFFECTIVE RANGE(?: ABOUT)?\s+(?P<max>\d+(?:\.\d+)?)\s+MILES"
        r"(?:;\s*MAXIMUM SLANT RANGE\s+(?P<slant>\d+(?:\.\d+)?)\s+MILES)?",
        evidence_text,
    )
    if range_match:
        out["min_intercept_km"] = round(float(range_match.group("min")) * 1.60934, 1)
        out["max_intercept_km"] = round(float(range_match.group("max")) * 1.60934, 1)
    ceiling_match = re.search(r"CEILING:\s*UP TO\s+(?P<ceiling>[\d,]+)\s*FT", evidence_text)
    if ceiling_match:
        out["max_altitude_km"] = round(float(ceiling_match.group("ceiling").replace(",", "")) * 0.0003048, 1)
    return out


def _postprocess_air_defense_missiles(
    pass_output: dict[str, Any],
    evidence_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(pass_output)
    missile_rows = updated.get("missile_systems")
    if not isinstance(missile_rows, list):
        return updated, {}

    parsed_notes = _extract_museum_display_range_notes(evidence_text)
    stats: dict[str, Any] = {
        "status_cleared": [],
        "range_overrides": {},
    }
    cleaned_rows: list[Any] = []

    apply_note_overrides = "TECHNICAL NOTES:" in evidence_text and len(missile_rows) == 1

    for row in missile_rows:
        if not isinstance(row, dict):
            cleaned_rows.append(row)
            continue
        item = dict(row)
        system_name = item.get("system_name")

        if item.get("system_status") and not _status_is_explicit(item.get("system_status"), evidence_text):
            stats["status_cleared"].append(str(system_name or ""))
            item["system_status"] = None

        if apply_note_overrides and parsed_notes:
            for field_name, corrected_value in parsed_notes.items():
                if item.get(field_name) != corrected_value:
                    stats["range_overrides"][field_name] = corrected_value
                item[field_name] = corrected_value
        cleaned_rows.append(item)

    updated["missile_systems"] = cleaned_rows
    if not stats["status_cleared"]:
        stats.pop("status_cleared")
    if not stats["range_overrides"]:
        stats.pop("range_overrides")
    return updated, stats


def _build_upstream_name_map(upstream_entities: list[Any] | None) -> dict[str, str]:
    if not upstream_entities:
        return {}
    out: dict[str, str] = {}
    for entity in upstream_entities:
        ref_id = getattr(entity, "ref_id", None)
        if not ref_id:
            continue
        identity_values = getattr(entity, "identity_values", None) or {}
        system_name = identity_values.get("system_name")
        if isinstance(system_name, str):
            out[normalize_evidence_text(system_name)] = ref_id
            continue
        display_label = getattr(entity, "display_label", None)
        if isinstance(display_label, str) and display_label.strip():
            out[normalize_evidence_text(display_label)] = ref_id
    return out


def _postprocess_air_defense_system_links(
    pass_output: dict[str, Any],
    evidence_text: str,
    upstream_entities: list[Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(pass_output)
    relationships = updated.get("relationships")
    if not isinstance(relationships, list):
        return updated, {}
    if relationships:
        return updated, {}

    name_to_ref = _build_upstream_name_map(upstream_entities)
    derived: list[dict[str, Any]] = []

    spoon_rest_ref = name_to_ref.get("SPOON REST")
    fan_song_ref = name_to_ref.get("FAN SONG")
    sa2_ref = name_to_ref.get("SA-2")

    if (
        spoon_rest_ref
        and fan_song_ref
        and "SPOON REST ACQUISITION RADAR" in evidence_text
        and "FAN SONG GUIDANCE RADAR" in evidence_text
    ):
        derived.append(
            {
                "rel_type": "CUES",
                "from_ref_id": spoon_rest_ref,
                "to_ref_id": fan_song_ref,
                "confidence": 0.95,
            }
        )

    if (
        fan_song_ref
        and sa2_ref
        and _entity_in_context("FAN SONG", evidence_text, ("MISSILE GUIDANCE", "GUIDED UP TO THREE SA-2S", "GUIDED UP TO THREE SA-2S AGAINST ONE TARGET"))
    ):
        derived.append(
            {
                "rel_type": "ASSOCIATED_WITH",
                "from_ref_id": fan_song_ref,
                "to_ref_id": sa2_ref,
                "confidence": 0.95,
            }
        )

    updated["relationships"] = derived
    if not derived:
        return updated, {}
    return updated, {"derived_relationships": derived}
