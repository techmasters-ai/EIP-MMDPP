from __future__ import annotations

import re
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from app._numeric_evidence import value_is_supported_by_text

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
_STATUS_EXPLICIT_PATTERNS = {
    "OPERATIONAL": (
        r"\bSTATUS\s*:\s*OPERATIONAL\b",
        r"\bIS OPERATIONAL\b",
        r"\bWAS OPERATIONAL\b",
        r"\bWERE OPERATIONAL\b",
        r"\bREMAINS OPERATIONAL\b",
        r"\bREMAINED OPERATIONAL\b",
        r"\bBECAME OPERATIONAL\b",
        r"\bENTERED OPERATIONAL SERVICE\b",
        r"\bIN OPERATIONAL SERVICE\b",
        r"\bACHIEVED OPERATIONAL STATUS\b",
    ),
    "DEVELOPMENTAL": (
        r"\bSTATUS\s*:\s*DEVELOPMENTAL\b",
        r"\bIS DEVELOPMENTAL\b",
        r"\bWAS DEVELOPMENTAL\b",
        r"\bIN DEVELOPMENT\b",
        r"\bUNDER DEVELOPMENT\b",
        r"\bDEVELOPMENTAL STATUS\b",
    ),
    "RETIRED": (
        r"\bSTATUS\s*:\s*RETIRED\b",
        r"\bIS RETIRED\b",
        r"\bWAS RETIRED\b",
        r"\bWERE RETIRED\b",
        r"\bRETIRED FROM SERVICE\b",
        r"\bWITHDRAWN FROM SERVICE\b",
        r"\bNO LONGER IN SERVICE\b",
    ),
    "UPGRADED": (
        r"\bSTATUS\s*:\s*UPGRADED\b",
        r"\bIS UPGRADED\b",
        r"\bWAS UPGRADED\b",
        r"\bUPGRADED VERSION\b",
        r"\bUPGRADED VARIANT\b",
    ),
    "EXPORTED": (
        r"\bSTATUS\s*:\s*EXPORTED\b",
        r"\bIS EXPORTED\b",
        r"\bWAS EXPORTED\b",
        r"\bEXPORTED TO\b",
    ),
}
_GUIDANCE_TYPE_PATTERNS = {
    "COMMAND": (r"\bCOMMAND GUIDANCE\b", r"\bCOMMAND-GUIDED\b"),
    "BEAM_RIDING": (r"\bBEAM RIDING\b", r"\bBEAM-RIDING\b"),
    "SARH": (r"\bSEMI-ACTIVE RADAR HOMING\b",),
    "ARH": (r"\bACTIVE RADAR HOMING\b",),
    "IR": (r"\bINFRARED HOMING\b", r"\bIR GUIDANCE\b"),
    "TVM": (r"\bTRACK-VIA-MISSILE\b",),
    "GPS_INS": (r"\bGPS/INS\b", r"\bGPS INS\b", r"\bINERTIAL NAVIGATION\b"),
    "DUAL_MODE": (r"\bDUAL-MODE\b", r"\bDUAL MODE\b"),
}
_SEEKER_TYPE_PATTERNS = {
    "ACTIVE_RADAR": (r"\bACTIVE RADAR SEEKER\b", r"\bACTIVE RADAR HOMING\b"),
    "SEMI_ACTIVE_RADAR": (r"\bSEMI-ACTIVE RADAR\b",),
    "PASSIVE_RADAR": (r"\bPASSIVE RADAR\b",),
    "IR": (r"\bINFRARED SEEKER\b", r"\bIR SEEKER\b"),
    "DUAL_MODE": (r"\bDUAL-MODE SEEKER\b", r"\bDUAL MODE SEEKER\b"),
    "ARM": (r"\bANTI-RADIATION\b",),
    "GPS_INS": (r"\bGPS/INS\b", r"\bGPS INS\b"),
    "COMMAND": (r"\bCOMMAND GUIDANCE\b", r"\bNO ONBOARD SEEKER\b"),
}
_RADAR_RECALL_PATTERNS = {
    "Fan Song": (
        r"\bFAN SONG\b",
    ),
    "Spoon Rest": (
        r"\bSPOON REST\b",
    ),
}
_RADAR_NOMENCLATURE_PATTERNS = {
    "Fan Song": (
        (r"\bSNR-75\b", "SNR-75"),
        (r"\bRSNA-75\b", "RSNA-75"),
    ),
    "Spoon Rest": (
        (r"\bP-18-2/P-18M\b", "P-18-2/P-18M"),
        (r"\bP-12M/P-18\b", "P-12M/P-18"),
    ),
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


def collect_entity_identity_examples(
    pass_output: dict[str, Any],
    template_cls: type[BaseModel],
    *,
    limit_per_field: int = 30,
) -> dict[str, list[dict[str, str]]]:
    """Return graph identity values from entity lists for diagnostics."""
    if not isinstance(pass_output, dict):
        return {}

    examples: dict[str, list[dict[str, str]]] = {}
    for field_name, field_info in template_cls.model_fields.items():
        raw_items = pass_output.get(field_name)
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

        field_examples: list[dict[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            identity = {
                field: str(item.get(field))
                for field in id_fields
                if item.get(field) not in (None, "")
            }
            if identity:
                field_examples.append(identity)
            if len(field_examples) >= limit_per_field:
                break
        if field_examples:
            examples[field_name] = field_examples
    return examples


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
            # docling-graph-docs § "Template Basics → Edge Helper"
            # prescribes typed `edge(label=...)` fields for graph
            # relationships. The cross-pass-linking DTO pattern
            # (SystemLinkRelationship, with rel_type / from_ref_id /
            # to_ref_id on a plain Field) is a documented exception —
            # see ontology_bundles/air_defense_v3/extraction_schemas/
            # system_links.py module docstring for the rationale.
            # Count each DTO row with a non-empty rel_type so metadata
            # reflects derived relationships instead of reporting 0
            # edges for the whole system_links pass.
            if "rel_type" in item_cls.model_fields:
                for raw in raw_items:
                    rel_type = raw.get("rel_type") if isinstance(raw, dict) else getattr(raw, "rel_type", None)
                    if not isinstance(rel_type, str) or not rel_type:
                        continue
                    edge_count += 1
                    edge_types[rel_type] = edge_types.get(rel_type, 0) + 1
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
    cross_entity_hints: list[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bundle_key != "air_defense_v3" or not isinstance(pass_output, dict):
        return pass_output, {}

    if pass_name in RADAR_PASS_NAMES:
        return _postprocess_air_defense_radars(pass_output, evidence_text)
    if pass_name in MISSILE_PASS_NAMES:
        return _postprocess_air_defense_missiles(pass_output, evidence_text)
    if pass_name == "system_links":
        return _postprocess_air_defense_system_links(
            pass_output, evidence_text, upstream_entities, cross_entity_hints,
        )
    return pass_output, {}


def _status_is_explicit_for_entity(status_value: Any, identity_value: Any, evidence_text: str) -> bool:
    if not isinstance(status_value, str) or not isinstance(identity_value, str):
        return False
    normalized_status = normalize_evidence_text(status_value)
    patterns = _STATUS_EXPLICIT_PATTERNS.get(normalized_status, ())
    if not patterns:
        return False
    normalized_identity = normalize_evidence_text(identity_value)
    if not normalized_identity or not evidence_text:
        return False

    # Require the named entity and the explicit lifecycle-status phrase
    # to occur in the same local statement. This prevents false positives
    # from unrelated words elsewhere on the page such as "Operation
    # Rolling Thunder" or generic narrative text like "remained in use".
    for match in re.finditer(re.escape(normalized_identity), evidence_text):
        start = max(0, match.start() - 120)
        end = min(len(evidence_text), match.end() + 120)
        context = evidence_text[start:end]
        if any(re.search(pattern, context) for pattern in patterns):
            return True
    return False


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


def _infer_radar_emitter_function(system_name: Any, evidence_text: str) -> str | None:
    if not isinstance(system_name, str):
        return None
    normalized_name = normalize_evidence_text(system_name)
    if f"{normalized_name} GUIDANCE RADAR" in evidence_text:
        return "FIRE_CONTROL"
    if f"{normalized_name} ENGAGEMENT RADAR" in evidence_text:
        return "FIRE_CONTROL"
    if f"{normalized_name} ACQUISITION RADAR" in evidence_text:
        return "SEARCH"
    if _entity_in_context(system_name, evidence_text, ("ENGAGEMENT RADAR",), window=80):
        return "FIRE_CONTROL"
    if _entity_in_context(system_name, evidence_text, ("ACQUISITION RADAR", "SEARCH RADAR"), window=80):
        return "SEARCH"
    if _entity_in_context(
        system_name,
        evidence_text,
        ("MISSILE GUIDANCE", "GUIDED UP TO", "GUIDED AGAINST ONE TARGET"),
        window=80,
    ):
        return "FIRE_CONTROL"
    if _entity_in_context(system_name, evidence_text, ("DETECTED INCOMING AIRCRAFT",), window=80):
        return "SEARCH"
    return None


def _find_explicit_radar_nomenclature(system_name: Any, evidence_text: str) -> str | None:
    if not isinstance(system_name, str):
        return None
    for pattern, value in _RADAR_NOMENCLATURE_PATTERNS.get(system_name, ()):
        if re.search(pattern, evidence_text):
            return value
    return None


EVIDENCE_GATE_RADAR_FIELDS: tuple[str, ...] = (
    "erp_dbw", "tx_peak_power_kw", "gain_dbi",
    "antenna_photo", "antenna_dim_az_m", "antenna_dim_el_m",
    "beamwidth_az_deg", "beamwidth_el_deg", "spoiled",
    "coverage_limits_el_deg",
    "nominal_rf_mhz", "nominal_pri_usec", "nominal_pd_usec",
    "scan_period_sec",
    "frequency_excursion_mhz", "num_bits_in_code", "pulses_per_dwell",
    "confidence",
)

RADAR_PASS_NAMES: tuple[str, ...] = (
    "radar_domain",
    "radar_identity",
    "radar_power_rf",
    "radar_antenna",
    "radar_timing",
    "radar_modulation",
)


def _clear_unsupported_radar_properties(
    item: dict[str, Any], evidence_text: str,
) -> list[str]:
    """Null radar properties whose values aren't supported by batch text.

    Spec §4.8 refactor. Previously unconditionally nulled 18 numeric
    fields; now uses value_is_supported_by_text to preserve values
    that appear in evidence_text (with same-unit-suffix variants).
    """
    cleared: list[str] = []

    # Text fields use the existing exact-quote check.
    exact_text_fields = (
        "nomenclature", "elnot", "dieqp", "asrd",
        "responsible_agency", "review_cycle", "next_review_date",
        "dwell_time", "scan_type", "intra_pulse_mop", "inter_pulse",
    )
    for field_name in exact_text_fields:
        value = item.get(field_name)
        if value is not None and not _value_is_quoted_in_text(value, evidence_text):
            item[field_name] = None
            cleared.append(field_name)

    # Numeric (and the bool / coverage-limits) fields are preserved when
    # value_is_supported_by_text accepts them; nulled otherwise. The
    # tuple lives at module scope (above) so the Step 4b drift test
    # can compare it for set equality against the field-group definitions.
    for field_name in EVIDENCE_GATE_RADAR_FIELDS:
        value = item.get(field_name)
        if value is None:
            continue
        if not value_is_supported_by_text(value, field_name, evidence_text):
            item[field_name] = None
            cleared.append(field_name)

    return cleared


def _recover_explicit_radars(
    radar_rows: list[Any],
    evidence_text: str,
) -> tuple[list[Any], list[str]]:
    recovered_rows = list(radar_rows)
    recovered_names: list[str] = []
    existing = {
        normalize_evidence_text(row.get("system_name"))
        for row in radar_rows
        if isinstance(row, dict) and isinstance(row.get("system_name"), str)
    }

    for system_name, patterns in _RADAR_RECALL_PATTERNS.items():
        normalized_name = normalize_evidence_text(system_name)
        if normalized_name in existing:
            continue
        if not any(re.search(pattern, evidence_text) for pattern in patterns):
            continue
        recovered: dict[str, Any] = {"system_name": system_name}
        nomenclature = _find_explicit_radar_nomenclature(system_name, evidence_text)
        if nomenclature is not None:
            recovered["nomenclature"] = nomenclature
        recovered_rows.append(recovered)
        recovered_names.append(system_name)
        existing.add(normalized_name)

    return recovered_rows, recovered_names


def _postprocess_air_defense_radars(
    pass_output: dict[str, Any],
    evidence_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(pass_output)
    radar_rows = updated.get("radar_systems")
    if not isinstance(radar_rows, list):
        return updated, {}

    radar_rows, recovered_names = _recover_explicit_radars(radar_rows, evidence_text)
    stats: dict[str, Any] = {
        "status_cleared": [],
        "emitter_function_overrides": {},
        "recalled_radars": recovered_names,
        "unsupported_properties_cleared": {},
    }
    cleaned_rows: list[Any] = []

    for row in radar_rows:
        if not isinstance(row, dict):
            cleaned_rows.append(row)
            continue
        item = dict(row)
        system_name = item.get("system_name")
        inferred_emitter = _infer_radar_emitter_function(system_name, evidence_text)
        if inferred_emitter is not None:
            if item.get("emitter_function") != inferred_emitter:
                stats["emitter_function_overrides"][str(system_name or "")] = inferred_emitter
            item["emitter_function"] = inferred_emitter
        elif item.get("emitter_function") is not None and not _value_is_quoted_in_text(item.get("emitter_function"), evidence_text):
            item["emitter_function"] = None

        if item.get("system_status") and not _status_is_explicit_for_entity(
            item.get("system_status"),
            system_name,
            evidence_text,
        ):
            stats["status_cleared"].append(str(system_name or ""))
            item["system_status"] = None

        explicit_nomenclature = _find_explicit_radar_nomenclature(system_name, evidence_text)
        if explicit_nomenclature is not None:
            item["nomenclature"] = explicit_nomenclature

        cleared_fields = _clear_unsupported_radar_properties(item, evidence_text)
        if cleared_fields:
            stats["unsupported_properties_cleared"][str(system_name or "")] = sorted(set(cleared_fields))
        cleaned_rows.append(item)

    updated["radar_systems"] = cleaned_rows
    if not stats["status_cleared"]:
        stats.pop("status_cleared")
    if not stats["emitter_function_overrides"]:
        stats.pop("emitter_function_overrides")
    if not stats["recalled_radars"]:
        stats.pop("recalled_radars")
    if not stats["unsupported_properties_cleared"]:
        stats.pop("unsupported_properties_cleared")
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


def _value_is_quoted_in_text(value: Any, evidence_text: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = normalize_evidence_text(value)
    if not normalized or not evidence_text:
        return False
    return normalized in evidence_text


def _enum_is_explicit(enum_value: Any, evidence_text: str, patterns_by_value: dict[str, tuple[str, ...]]) -> bool:
    if not isinstance(enum_value, str):
        return False
    normalized = normalize_evidence_text(enum_value)
    patterns = patterns_by_value.get(normalized, ())
    if not patterns:
        return False
    return any(re.search(pattern, evidence_text) for pattern in patterns)


def _mechanically_supported_missile_fields(evidence_text: str) -> dict[str, float]:
    supported = _extract_museum_display_range_notes(evidence_text)
    weight_match = re.search(r"WEIGHT:\s*(?P<weight>[\d,]+(?:\.\d+)?)\s*LBS?\.?\b", evidence_text)
    if weight_match:
        supported["total_mass_kg"] = round(float(weight_match.group("weight").replace(",", "")) / 2.205, 1)
    return supported


EVIDENCE_GATE_MISSILE_FIELDS: tuple[str, ...] = (
    # missile_airframe numerics
    "body_length_m", "body_diameter_m",
    # missile_speed_timing numerics
    "average_speed_mps", "max_speed_mps",
    "max_flyout_time_sec", "flight_time_sec", "coast_time_sec",
    "intra_salvo_time_sec", "total_burn_time_sec", "ejector_time_sec",
    # missile_propulsion numerics
    "ejector_mass_kg", "booster_time_sec", "booster_mass_kg",
    "sustain_time_sec", "sustain_mass_kg",
    # meta
    "confidence",
)

MISSILE_PASS_NAMES: tuple[str, ...] = (
    "missile_domain",
    "missile_identity",
    "missile_kinematics",
    "missile_guidance",
    "missile_airframe",
    "missile_speed_timing",
    "missile_propulsion",
)


def _clear_unsupported_missile_properties(item: dict[str, Any], evidence_text: str) -> list[str]:
    """Spec §4.8 pattern adapted for missile (mechanical-support path preserved).

    Refactor scope (Session 1):
    - PRESERVE: _mechanically_supported_missile_fields() pattern extraction
      and the override path for min_intercept_km / max_intercept_km /
      max_altitude_km / total_mass_kg (existing behavior).
    - PRESERVE: unconditional-null branch for min_altitude_km /
      max_launch_angle_deg / missile_photo (Session 1 keeps existing
      behavior; relaxing these is Session 2).
    - PRESERVE: exact-text branch for string fields and the
      _enum_is_explicit branches for guidance_type / seeker_type.
    - REPLACE: the strict_null_fields tuple loop. The previous code
      unconditionally nulled these. New code preserves values whose
      stringified form appears in evidence_text via
      value_is_supported_by_text — same predicate the auto-evidence
      resolver uses.
    """
    cleared: list[str] = []
    supported_numeric = _mechanically_supported_missile_fields(evidence_text)

    # PRESERVED: exact-text branch for string fields. String properties
    # must either appear verbatim in the source text or be expressed by
    # an explicit guidance/seeker phrase (handled below). Otherwise they
    # are unsupported and must be null.
    exact_text_fields = (
        "nomenclature",
        "name",
        "dieqp",
        "emitter_function",
        "asrd",
        "responsible_agency",
        "review_cycle",
        "next_review_date",
        "ejector_thrust",
        "booster_thrust",
        "sustain_thrust",
    )
    for field_name in exact_text_fields:
        if item.get(field_name) is not None and not _value_is_quoted_in_text(item.get(field_name), evidence_text):
            item[field_name] = None
            cleared.append(field_name)

    # PRESERVED: enum-explicit branches for guidance_type / seeker_type.
    if item.get("guidance_type") is not None and not _enum_is_explicit(item.get("guidance_type"), evidence_text, _GUIDANCE_TYPE_PATTERNS):
        item["guidance_type"] = None
        cleared.append("guidance_type")

    if item.get("seeker_type") is not None and not _enum_is_explicit(item.get("seeker_type"), evidence_text, _SEEKER_TYPE_PATTERNS):
        item["seeker_type"] = None
        cleared.append("seeker_type")

    # REPLACED (was strict_null_fields tuple loop): preserves values that
    # appear in evidence_text via the shared value_is_supported_by_text
    # predicate. Same predicate the auto-evidence resolver and the radar
    # postprocessor use — single source of truth.
    for field_name in EVIDENCE_GATE_MISSILE_FIELDS:
        value = item.get(field_name)
        if value is None:
            continue
        if not value_is_supported_by_text(value, field_name, evidence_text):
            item[field_name] = None
            cleared.append(field_name)

    # PRESERVED: mechanical-support override for 4 numerics. Mechanical
    # conversion takes priority over the LLM's value when available;
    # otherwise fall back to the same evidence-verification predicate.
    for field_name in ("min_intercept_km", "max_intercept_km", "max_altitude_km", "total_mass_kg"):
        if field_name in supported_numeric:
            item[field_name] = supported_numeric[field_name]
        elif item.get(field_name) is not None:
            value = item[field_name]
            if not value_is_supported_by_text(value, field_name, evidence_text):
                item[field_name] = None
                cleared.append(field_name)

    # PRESERVED: unconditional-null for fields whose Session 1 contract
    # is "always null" (Session 2 may relax).
    if item.get("min_altitude_km") is not None:
        item["min_altitude_km"] = None
        cleared.append("min_altitude_km")
    if item.get("max_launch_angle_deg") is not None:
        item["max_launch_angle_deg"] = None
        cleared.append("max_launch_angle_deg")
    if item.get("missile_photo") is not None:
        item["missile_photo"] = None
        cleared.append("missile_photo")

    return cleared


def _postprocess_air_defense_missiles(
    pass_output: dict[str, Any],
    evidence_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(pass_output)
    missile_rows = updated.get("missile_systems")
    if not isinstance(missile_rows, list):
        return updated, {}

    parsed_notes = _mechanically_supported_missile_fields(evidence_text)
    stats: dict[str, Any] = {
        "status_cleared": [],
        "range_overrides": {},
        "unsupported_properties_cleared": {},
    }
    cleaned_rows: list[Any] = []

    apply_note_overrides = "TECHNICAL NOTES:" in evidence_text and len(missile_rows) == 1

    for row in missile_rows:
        if not isinstance(row, dict):
            cleaned_rows.append(row)
            continue
        item = dict(row)
        original_item = dict(item)
        system_name = item.get("system_name")

        if item.get("system_status") and not _status_is_explicit_for_entity(
            item.get("system_status"),
            system_name,
            evidence_text,
        ):
            stats["status_cleared"].append(str(system_name or ""))
            item["system_status"] = None

        cleared_fields = _clear_unsupported_missile_properties(item, evidence_text)
        if cleared_fields:
            stats["unsupported_properties_cleared"][str(system_name or "")] = sorted(set(cleared_fields))

        if apply_note_overrides and parsed_notes:
            for field_name, corrected_value in parsed_notes.items():
                if original_item.get(field_name) != corrected_value:
                    stats["range_overrides"][field_name] = corrected_value
                item[field_name] = corrected_value
        cleaned_rows.append(item)

    updated["missile_systems"] = cleaned_rows
    if not stats["status_cleared"]:
        stats.pop("status_cleared")
    if not stats["range_overrides"]:
        stats.pop("range_overrides")
    if not stats["unsupported_properties_cleared"]:
        stats.pop("unsupported_properties_cleared")
    return updated, stats


def _build_upstream_name_map(upstream_entities: list[Any] | None) -> dict[str, str]:
    """Build name → ref_id lookup.

    Registers each entity under its primary identity (``system_name``),
    its display_label, AND each entry of ``aliases`` (typically populated
    from upstream schema fields like ``nomenclature`` and ``name``). The
    relationship pass uses this map to resolve chunk-cell names like
    "SA-75" back to a ref_id whose primary identity might be a different
    token ("1D" via Missile Type, etc.). Earlier registrations win on
    conflict so primary identity takes precedence over aliases."""
    if not upstream_entities:
        return {}
    out: dict[str, str] = {}

    def _maybe_register(name: Any, ref_id: str) -> None:
        if not isinstance(name, str):
            return
        key = normalize_evidence_text(name)
        if not key or key in out:
            return
        out[key] = ref_id

    for entity in upstream_entities:
        ref_id = getattr(entity, "ref_id", None)
        if not ref_id:
            continue
        identity_values = getattr(entity, "identity_values", None) or {}
        _maybe_register(identity_values.get("system_name"), ref_id)
        _maybe_register(getattr(entity, "display_label", None), ref_id)
        aliases = getattr(entity, "aliases", None) or []
        for alias in aliases:
            _maybe_register(alias, ref_id)
    return out


def _postprocess_air_defense_system_links(
    pass_output: dict[str, Any],
    evidence_text: str,
    upstream_entities: list[Any] | None,
    cross_entity_hints: list[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Postprocess system_links output.

    - Always promotes ``cross_entity_hints`` from the deterministic table
      overlay into ASSOCIATED_WITH edges (resolving source/target by name
      against the upstream catalog). v8a — closes the relationship-yield
      gap on table-anchored radar↔missile pairings.
    - Falls back to evidence-text heuristics (Spoon Rest → Fan Song,
      Fan Song → SA-2) only when the LLM emitted ZERO relationships AND
      no hint promotions resolved.
    - Dedupes by ``(from_ref_id, to_ref_id)`` canonical-direction; the
      first-emitted wins."""
    updated = dict(pass_output)
    relationships = updated.get("relationships")
    if not isinstance(relationships, list):
        return updated, {}

    name_to_ref = _build_upstream_name_map(upstream_entities)
    seen_pairs: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    # Carry forward LLM-emitted relationships first (they win on (from, to)
    # collisions with derived/promoted edges).
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        f = rel.get("from_ref_id")
        t = rel.get("to_ref_id")
        if not isinstance(f, str) or not isinstance(t, str):
            continue
        if (f, t) in seen_pairs:
            continue
        seen_pairs.add((f, t))
        out.append(rel)

    # v8a: promote cross_entity_hints to ASSOCIATED_WITH edges.
    promoted: list[dict[str, Any]] = []
    if cross_entity_hints:
        for hint in cross_entity_hints:
            source_name = getattr(hint, "source_canonical", None) or (
                hint.get("source_canonical") if isinstance(hint, dict) else None
            )
            target_name = getattr(hint, "target_alias", None) or (
                hint.get("target_alias") if isinstance(hint, dict) else None
            )
            if not isinstance(source_name, str) or not isinstance(target_name, str):
                continue
            source_ref = name_to_ref.get(normalize_evidence_text(source_name))
            target_ref = name_to_ref.get(normalize_evidence_text(target_name))
            if not source_ref or not target_ref:
                continue
            if source_ref == target_ref:
                continue
            if (source_ref, target_ref) in seen_pairs:
                continue
            seen_pairs.add((source_ref, target_ref))
            edge = {
                "rel_type": "ASSOCIATED_WITH",
                "from_ref_id": source_ref,
                "to_ref_id": target_ref,
                "confidence": 1.0,
            }
            promoted.append(edge)
            out.append(edge)
    if promoted:
        stats["promoted_from_cross_entity_hints"] = promoted

    # Legacy evidence-text fallback: only fires when nothing else produced
    # edges (preserves the prior behavior for docs without table hints).
    derived: list[dict[str, Any]] = []
    if not out:
        spoon_rest_ref = name_to_ref.get("SPOON REST")
        fan_song_ref = name_to_ref.get("FAN SONG")
        sa2_ref = name_to_ref.get("SA-2")

        if (
            spoon_rest_ref
            and fan_song_ref
            and "SPOON REST ACQUISITION RADAR" in evidence_text
            and "FAN SONG GUIDANCE RADAR" in evidence_text
            and (spoon_rest_ref, fan_song_ref) not in seen_pairs
        ):
            edge = {
                "rel_type": "CUES",
                "from_ref_id": spoon_rest_ref,
                "to_ref_id": fan_song_ref,
                "confidence": 0.95,
            }
            seen_pairs.add((spoon_rest_ref, fan_song_ref))
            derived.append(edge)
            out.append(edge)

        if (
            fan_song_ref
            and sa2_ref
            and _entity_in_context(
                "FAN SONG", evidence_text,
                ("MISSILE GUIDANCE", "GUIDED UP TO THREE SA-2S",
                 "GUIDED UP TO THREE SA-2S AGAINST ONE TARGET"),
            )
            and (fan_song_ref, sa2_ref) not in seen_pairs
        ):
            edge = {
                "rel_type": "ASSOCIATED_WITH",
                "from_ref_id": fan_song_ref,
                "to_ref_id": sa2_ref,
                "confidence": 0.95,
            }
            seen_pairs.add((fan_song_ref, sa2_ref))
            derived.append(edge)
            out.append(edge)
    if derived:
        stats["derived_relationships"] = derived

    updated["relationships"] = out
    return updated, stats
