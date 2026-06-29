from __future__ import annotations

import os
import re
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from app._numeric_evidence import value_is_supported_by_text


# 2026-05-20: legacy SA-2-specific fallbacks (radar recall by name,
# nomenclature fills by name, system_links Spoon-Rest→Fan-Song→SA-2
# edges) are quarantined behind this env flag. They predate the
# generic structural rules (Step 3 role inference, Step 7 precision
# filter, Step 6 designation alias expansion, hint promotion,
# VARIANT_OF emitter). Default OFF so the pipeline behaves
# document-generically; opt-in with =true / =1 / =yes to restore the
# SA-2 corpus-specific compatibility path.
_LEGACY_SA2_FALLBACKS_ENABLED: bool = os.environ.get(
    "DOCLING_GRAPH_LEGACY_SA2_FALLBACKS", "false",
).strip().lower() in ("true", "1", "yes")

_EVIDENCE_WS_RE = re.compile(r"\s+")
_EVIDENCE_BOUNDARY_CLASS = r"A-Z0-9"
# TODO #83 Tier A: separators that may differ between the LLM's emitted
# identity and the document surface form (e.g. "SA-2 C" vs "SA-2C") are
# treated as interchangeable/optional when matching the identity against
# batch evidence. Whitespace, hyphen, underscore, slash, dot only.
# Comma/semicolon/colon are deliberately EXCLUDED — they delimit list
# items, and bridging them would glue distinct entities ("SA-2, C-300")
# into a false match.
_IDENTITY_FLEX_SEP_RE = re.compile(r"[\s\-_/.]+")
_IDENTITY_FLEX_SEP_CLASS = r"[\s\-_/.]*"
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
    # TODO #83 Tier A: tolerate separator/spacing differences between the
    # emitted identity and the evidence. Split the normalized identity on
    # separator runs and rejoin with an optional-separator class, so
    # "SA-2 C" matches "SA-2C" / "SA-2 C" / "SA 2 C" while the alphanumeric
    # boundary lookarounds still prevent matching inside a larger token
    # ("S-75" must not match "SA-75"), and commas are not bridged. A
    # separator-free identity (e.g. "5YA23") reduces to the original
    # escaped form, so its behavior is unchanged.
    tokens = [re.escape(t) for t in _IDENTITY_FLEX_SEP_RE.split(normalized) if t]
    if not tokens:
        return False
    core = _IDENTITY_FLEX_SEP_CLASS.join(tokens)
    pattern = rf"(?<![{_EVIDENCE_BOUNDARY_CLASS}]){core}(?:S)?(?![{_EVIDENCE_BOUNDARY_CLASS}])"
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


# Bundle keys eligible for the air-defense-specific post-processing
# below. The subset bundle is a test-only sibling of air_defense_v3 with
# identical Pydantic extraction schemas and the same RADAR_/MISSILE_*
# pass names — see ontology_bundles/air_defense_v3_baseline_subset/
# manifest.yaml. A genuinely new ontology (naval, ground systems, …)
# would not belong here because the markers, enum values, list keys,
# and entity types throughout this module are air-defense-specific.
_AIR_DEFENSE_BUNDLE_KEYS: frozenset[str] = frozenset({
    "air_defense_v3",
    "air_defense_v3_baseline_subset",
    "air_defense_v3_narrowing_v1",
    "air_defense_v3_merged_v1",
})


def apply_bundle_postprocessing(
    bundle_key: str,
    pass_name: str,
    pass_output: dict[str, Any],
    evidence_text: str,
    upstream_entities: list[Any] | None = None,
    cross_entity_hints: list[Any] | None = None,
    alias_map_by_entity_type: dict[str, dict[str, str]] | None = None,
    normalized_tables: list[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bundle_key not in _AIR_DEFENSE_BUNDLE_KEYS or not isinstance(pass_output, dict):
        return pass_output, {}

    if pass_name in RADAR_PASS_NAMES:
        return _postprocess_air_defense_radars(pass_output, evidence_text)
    if pass_name in MISSILE_PASS_NAMES:
        # Step 6: designation alias expansion and the predecessor-
        # slash-artifact filter apply ONLY to the identity pass.
        # Numeric passes (missile_kinematics, missile_propulsion, etc.)
        # must not have their merge identity altered by alias
        # enrichment or have partial rows dropped — both would change
        # which round entity their numeric values merge onto.
        is_identity_pass = pass_name == "missile_identity"
        return _postprocess_air_defense_missiles(
            pass_output, evidence_text,
            normalized_tables=normalized_tables if is_identity_pass else None,
            is_identity_pass=is_identity_pass,
        )
    if pass_name == "system_links":
        return _postprocess_air_defense_system_links(
            pass_output, evidence_text, upstream_entities, cross_entity_hints,
            alias_map_by_entity_type=alias_map_by_entity_type,
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


# DIRECT-CONCAT markers — `"{entity_name} {marker}"` appears verbatim in
# evidence. These are unambiguous because the role label is grammatically
# attached to the entity name (no cross-entity ambiguity). Format:
# (role_enum, (marker_text, ...)).
_DIRECT_CONCAT_RADAR_ROLE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HEIGHT_FINDER", (
        "HEIGHTFINDING RADAR", "HEIGHTFINDING RADARS",
        "HEIGHT-FINDING RADAR", "HEIGHT FINDER", "HEIGHTFINDER",
    )),
    ("SEARCH", (
        "ACQUISITION RADAR", "SEARCH RADAR",
        "EARLY WARNING RADAR", "SURVEILLANCE RADAR",
    )),
    ("FIRE_CONTROL", (
        "GUIDANCE RADAR", "ENGAGEMENT RADAR",
        "FIRE CONTROL RADAR", "FIRE-CONTROL RADAR",
        "TRACKING RADAR", "ILLUMINATOR",
    )),
)

# WINDOW markers — for use when the entity name and role label are NOT
# directly adjacent but the role label still binds to this entity (e.g.
# "P-18-2/P-18M Spoon Rest D/E Acquisition Radar" — there's a D/E qualifier
# between the name and the role). EXCLUDES markers that frequently appear
# in cross-entity references (GUIDANCE RADAR is excluded — radars often
# describe other radars' guidance roles, which would mis-bind).
_WINDOW_RADAR_ROLE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HEIGHT_FINDER", (
        "HEIGHTFINDING RADAR", "HEIGHTFINDING RADARS",
        "HEIGHT-FINDING RADAR", "HEIGHT-FINDING",
        "HEIGHT FINDER", "HEIGHTFINDER",
    )),
    ("SEARCH", (
        "ACQUISITION RADAR", "SEARCH RADAR",
    )),
    ("FIRE_CONTROL", (
        "ENGAGEMENT RADAR",
    )),
)

# FALLBACK markers — only consulted when NO direct-concat or explicit
# window marker fires. The prior code unconditionally fired these,
# causing acquisition radars adjacent to "MISSILE GUIDANCE" prose to be
# wrongly force-labeled FIRE_CONTROL (Spoon Rest bug). Now strictly fallback.
_FALLBACK_RADAR_ROLE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FIRE_CONTROL", ("MISSILE GUIDANCE", "GUIDED UP TO", "GUIDED AGAINST")),
    ("SEARCH", ("DETECTED INCOMING AIRCRAFT",)),
)

_RADAR_ROLE_WINDOW_CHARS = 80


# Step 7: precision filter — auxiliary-equipment label markers.
# When an entity emitted into radar_identity is labeled in evidence_text
# as one of these non-radar equipment classes, drop it from the radar
# output. Generic, no equipment names — operates on context labels.
#
# Markers are matched as substring against the ~120-char window
# following the entity name occurrence. "Radar" appearing as part of the
# label (e.g. "Radar head van") is allowed via the explicit radar-
# context whitelist below.
_NON_RADAR_CONTEXT_MARKERS: tuple[str, ...] = (
    "POWER GENERATOR",
    "TRAINING EMULATOR",
    "RADIO RELAY VAN", "RADIO RELAY",
    "DECONTAMINATION VAN", "DECONTAMINATION",
    "FUEL TANK",
    "OXIDISER TANK", "OXIDIZER TANK",  # AmE + BrE
    "OXIDISER", "OXIDIZER",  # standalone tank labels in tables
    "TRANSPORTER/TRANSLOADER", "TRANSPORTER",
    "TRANSLOADER",
    "LAUNCHER, SINGLE RAIL", "SINGLE RAIL LAUNCHER",
    "LAUNCHER,",  # table cell shape "<name> Launcher, <variant>"
)

# Context phrases that CONTAIN words from the non-radar markers above
# but actually describe radar equipment. Whitelist — if any of these
# substrings appear in the window, the entity is NOT auxiliary.
_RADAR_CONTEXT_WHITELIST: tuple[str, ...] = (
    "RADAR HEAD VAN",
    "RADAR OPERATOR VAN",
    "RADAR ELECTRONICS VAN",
    "ACQUISITION RADAR", "ENGAGEMENT RADAR", "GUIDANCE RADAR",
    "HEIGHTFINDING RADAR", "HEIGHT-FINDING RADAR", "HEIGHT FINDER",
    "EARLY WARNING RADAR", "SURVEILLANCE RADAR", "SEARCH RADAR",
    "TRACKING RADAR", "FIRE CONTROL RADAR", "FIRE-CONTROL RADAR",
    "RANGEFINDING RADAR",
)

_NON_RADAR_CONTEXT_WINDOW_CHARS = 120


def _is_non_radar_context(system_name: str, evidence_text: str) -> bool:
    """True when the entity's emitted name appears in evidence_text in a
    context labeled as auxiliary (non-radar) equipment.

    Algorithm — per occurrence, the EARLIEST marker in the post-name
    window decides. A serialized table row places the entity's own role
    label first; any whitelist or non-radar phrase that appears later
    in the window belongs to a different row. Without this proximity
    rule, an auxiliary-equipment row that happens to be followed by a
    real radar row (e.g. 5L62A Oxidiser Tank → RD-75 Amazonka
    Rangefinding Radar in the SA-2 battery-components table) wrongly
    short-circuits to "radar" because the whitelist phrase exists
    *somewhere* in the window.

    Aggregation across multiple occurrences:
      * If any occurrence's earliest marker is a radar-context phrase
        → return False (preserve as radar).
      * Else if any occurrence's earliest marker is a non-radar phrase
        → return True (drop as auxiliary).
      * Else return False (no decision; let LLM emission stand).

    Generic — no equipment names anywhere. Operates on context labels.
    """
    if not isinstance(system_name, str) or not system_name:
        return False
    normalized_name = normalize_evidence_text(system_name)
    if not normalized_name or not evidence_text:
        return False
    saw_non_radar = False
    for m in re.finditer(re.escape(normalized_name), evidence_text):
        end = m.end()
        window = evidence_text[end:end + _NON_RADAR_CONTEXT_WINDOW_CHARS]
        # Earliest marker in this window wins: scan whitelist + non-radar
        # together and pick whichever appears first by character index.
        earliest_pos = len(window) + 1
        earliest_is_radar: bool | None = None
        for phrase in _RADAR_CONTEXT_WHITELIST:
            pos = window.find(phrase)
            if 0 <= pos < earliest_pos:
                earliest_pos = pos
                earliest_is_radar = True
        for phrase in _NON_RADAR_CONTEXT_MARKERS:
            pos = window.find(phrase)
            if 0 <= pos < earliest_pos:
                earliest_pos = pos
                earliest_is_radar = False
        if earliest_is_radar is True:
            return False
        if earliest_is_radar is False:
            saw_non_radar = True
    return saw_non_radar


# Step 4: spec fact overlay bridge — applies parsed prose spec facts to
# the `supported` dict in `_mechanically_supported_missile_fields`.
# Only fills fields that don't already have a value from the synth-block
# or museum-display channels (those are entity-scoped / higher priority).
# Generic — operates purely on canonical labels.
_SPEC_OVERLAY_FIELD_MAP: dict[str, str] = {
    "max_range":     "max_intercept_km",
    "min_range":     "min_intercept_km",
    "max_altitude":  "max_altitude_km",
    "min_altitude":  "min_altitude_km",
    "launch_angle":  "max_launch_angle_deg",
    "weight":        "total_mass_kg",
}


def _apply_spec_overlay_to_supported(supported: dict[str, float], evidence_text: str) -> None:
    """Parse prose spec facts from `evidence_text` and fill `supported`
    fields that don't already have a value. Bridges
    `app.services.spec_overlay` into the missile evidence path. Soft
    import — if spec_overlay isn't available (older container image),
    the function is a no-op."""
    try:
        from app.services.spec_overlay import parse_spec_facts_from_evidence_text
    except Exception:
        return
    if not evidence_text:
        return
    facts = parse_spec_facts_from_evidence_text(evidence_text)
    for fact in facts:
        target_field = _SPEC_OVERLAY_FIELD_MAP.get(fact.label_canonical)
        if not target_field or target_field in supported:
            continue
        # Prefer the metric-converted value matching the field's unit class
        if target_field.endswith("_km") and fact.value_metric_km is not None:
            supported[target_field] = round(fact.value_metric_km, 3)
        elif target_field.endswith("_kg") and fact.value_metric_kg is not None:
            supported[target_field] = round(fact.value_metric_kg, 3)
        elif target_field == "max_launch_angle_deg":
            try:
                supported[target_field] = float(fact.value_raw)
            except (ValueError, TypeError):
                pass


# Step 5: missile launch-angle patterns. Anchored on explicit launch/
# elevation context. Generic — no equipment names anywhere.
# Label form: "MAX LAUNCH ANGLE: 60°" / "LAUNCH ANGLE: 60 DEGREES"
_LAUNCH_ANGLE_LABEL_RE = re.compile(
    r"\b(?:MAX\s+)?LAUNCH\s+ANGLE\s*:?\s*(\d+(?:\.\d+)?)\s*(?:DEGREES?|°)",
    re.IGNORECASE,
)
# Elevation form: "LAUNCH ELEVATION 45°" / "LAUNCH ELEVATION: 45 DEGREES"
_LAUNCH_ELEVATION_RE = re.compile(
    r"\bLAUNCH\s+ELEVATION\s*:?\s*(\d+(?:\.\d+)?)\s*(?:DEGREES?|°)",
    re.IGNORECASE,
)
# Prose form: "launched (the) (missile) at 60 degrees"
_LAUNCHED_AT_RE = re.compile(
    r"\bLAUNCHED?\s+(?:THE\s+)?(?:MISSILE\s+)?AT\s+(\d+(?:\.\d+)?)\s*DEGREES?",
    re.IGNORECASE,
)


def _infer_radar_emitter_function(system_name: Any, evidence_text: str) -> str | None:
    """Infer `emitter_function` from prose context.

    Three phases with strict precedence:
      1. **Direct-concat:** `"{entity_name} {marker}"` substring match.
         Unambiguous: the role label is grammatically attached to this
         entity. Includes broadly-applicable markers like GUIDANCE RADAR
         that would over-trigger in window mode.
      2. **Nearest-window:** for entities where qualifiers sit between
         name and role label ("Spoon Rest D/E Acquisition Radar"), use
         the NEAREST marker by char-distance within ±80 chars. Window
         markers exclude promiscuous phrases (GUIDANCE RADAR) to avoid
         cross-entity bleed.
      3. **Fallback markers:** only when phases 1+2 produce nothing.
         Catches contextual hints like "DETECTED INCOMING AIRCRAFT" or
         "MISSILE GUIDANCE" prose without an explicit role label.

    Returning None lets the LLM-emitted value stand.

    Generic — no equipment names anywhere. Production evidence_text is
    uppercased via `normalize_evidence_text`; entity name is similarly
    normalized.
    """
    if not isinstance(system_name, str) or not system_name:
        return None
    normalized_name = normalize_evidence_text(system_name)
    if not normalized_name or not evidence_text:
        return None

    # Phase 1: direct-concat.
    for role, markers in _DIRECT_CONCAT_RADAR_ROLE_MARKERS:
        for marker in markers:
            if f"{normalized_name} {marker}" in evidence_text:
                return role

    # Phase 2: nearest-window with post-beats-pre precedence.
    #
    # A role marker that appears AFTER the entity (post marker) is taken
    # as binding to this entity (or its slash-group). A role marker that
    # appears BEFORE the entity (pre marker) is generally bound to the
    # entity that immediately preceded it — so it's only a weak fallback
    # signal for the current entity. Within each class (post / pre), the
    # nearest marker wins. Post is checked first; pre is consulted only
    # when no post marker is in the window.
    #
    # This is what allows a slash-group like
    # `... ACQUISITION RADAR PRV-10 KONUS / PRV-11 VERSHINA / SIDE NET
    # HEIGHTFINDING RADARS ...` to bind every entity in the group to
    # `HEIGHTFINDING RADARS` rather than wrongly inheriting the
    # `ACQUISITION RADAR` label that belongs to whichever entity
    # preceded the group (Spoon Rest, in the SA-2 case).
    best_post_distance: int | None = None
    best_post_role: str | None = None
    best_pre_distance: int | None = None
    best_pre_role: str | None = None
    name_pattern = re.compile(re.escape(normalized_name))
    for entity_match in name_pattern.finditer(evidence_text):
        e_start = entity_match.start()
        e_end = entity_match.end()
        win_start = max(0, e_start - _RADAR_ROLE_WINDOW_CHARS)
        win_end = min(len(evidence_text), e_end + _RADAR_ROLE_WINDOW_CHARS)
        context = evidence_text[win_start:win_end]
        for role, markers in _WINDOW_RADAR_ROLE_MARKERS:
            for marker in markers:
                idx = 0
                while True:
                    m_pos = context.find(marker, idx)
                    if m_pos < 0:
                        break
                    abs_pos = win_start + m_pos
                    abs_end = abs_pos + len(marker)
                    if abs_pos >= e_end:
                        distance = abs_pos - e_end
                        if best_post_distance is None or distance < best_post_distance:
                            best_post_distance = distance
                            best_post_role = role
                    elif abs_end <= e_start:
                        distance = e_start - abs_end
                        if best_pre_distance is None or distance < best_pre_distance:
                            best_pre_distance = distance
                            best_pre_role = role
                    # else: marker overlaps the name itself — ignore.
                    idx = m_pos + 1
    if best_post_role is not None:
        return best_post_role
    if best_pre_role is not None:
        return best_pre_role

    # Phase 3: fallback markers — only when no explicit role label found.
    for entity_match in name_pattern.finditer(evidence_text):
        e_start = entity_match.start()
        e_end = entity_match.end()
        win_start = max(0, e_start - _RADAR_ROLE_WINDOW_CHARS)
        win_end = min(len(evidence_text), e_end + _RADAR_ROLE_WINDOW_CHARS)
        context = evidence_text[win_start:win_end]
        for role, markers in _FALLBACK_RADAR_ROLE_MARKERS:
            if any(m in context for m in markers):
                return role
    return None


def _find_explicit_radar_nomenclature(system_name: Any, evidence_text: str) -> str | None:
    # Legacy SA-2 nomenclature backfill is quarantined behind the same
    # env flag as the recall patterns. Generic nomenclature comes via
    # the LLM (and via Step 6 designation alias expansion for missiles).
    if not _LEGACY_SA2_FALLBACKS_ENABLED:
        return None
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
    # Legacy SA-2 name-recall is quarantined: only fires when the env
    # flag explicitly opts in (`DOCLING_GRAPH_LEGACY_SA2_FALLBACKS`).
    # Generic radar recovery now comes from Step 3 role inference + Step
    # 7 precision filter on the LLM's emissions.
    if not _LEGACY_SA2_FALLBACKS_ENABLED:
        return recovered_rows, recovered_names
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


# Identity fields that carry a "canonical display name" for an entity. These
# are subject to Item 5 conservative OCR cleanup post-extraction. Generic —
# applies to any pass that has these fields on its row schema; field set is
# the union of identity-style fields across radar and missile schemas.
_CANONICAL_DISPLAY_NAME_FIELDS: tuple[str, ...] = ("system_name", "nomenclature", "name")


def _canonicalize_display_name(value: Any) -> Any:
    """Conservative OCR-artifact cleanup for entity display names.

    Generic — no domain-specific or equipment-specific rules. Per Item 5:
      * collapse hyphen-space: `RSN- 75M` → `RSN-75M`
      * collapse slash spacing: `RSNA-75 / SNR-75` → `RSNA-75/SNR-75`
      * normalize repeated whitespace and trim

    Does NOT do semantic equivalence: `Fan Song` stays `Fan Song`,
    `S-75` stays distinct from `SA-75`, no synonym expansion.

    Non-string inputs pass through unchanged (defensive — callers iterate
    over arbitrary field values).
    """
    if not isinstance(value, str):
        return value
    if not value:
        return value
    import re as _re
    # 1. hyphen-space: any `-` followed by whitespace → `-` only
    s = _re.sub(r"-\s+", "-", value)
    # 2. slash spacing: whitespace on either side of `/` → `/` only
    s = _re.sub(r"\s*/\s*", "/", s)
    # 3. collapse repeated whitespace and trim
    s = " ".join(s.split())
    return s


def _apply_display_name_canonicalization(
    item: dict[str, Any],
) -> list[dict[str, str]]:
    """Apply `_canonicalize_display_name` to identity-style fields of `item`,
    mutating in place. Returns a list of diagnostic dicts describing each
    rewrite (one per field that actually changed). Generic — operates only
    on the structural _CANONICAL_DISPLAY_NAME_FIELDS list, no domain logic.
    """
    rewrites: list[dict[str, str]] = []
    for field in _CANONICAL_DISPLAY_NAME_FIELDS:
        original = item.get(field)
        if not isinstance(original, str) or not original:
            continue
        canonical = _canonicalize_display_name(original)
        if canonical != original:
            item[field] = canonical
            rewrites.append({
                "field": field,
                "original": original,
                "canonical": canonical,
            })
    return rewrites


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
        "display_name_canonicalized": [],
        "non_radar_dropped": [],
    }
    cleaned_rows: list[Any] = []

    for row in radar_rows:
        if not isinstance(row, dict):
            cleaned_rows.append(row)
            continue
        item = dict(row)
        system_name = item.get("system_name")
        # Step 7: precision filter — drop entities labeled in evidence as
        # auxiliary equipment (power generator, fuel tank, training
        # emulator, radio relay, transloader, launcher, etc.) regardless
        # of what the LLM emitted. Diagnostic records each drop.
        if isinstance(system_name, str) and _is_non_radar_context(system_name, evidence_text):
            stats["non_radar_dropped"].append({
                "system_name": system_name,
                "reason": "auxiliary_equipment_context_in_evidence",
            })
            continue
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
        # Item 5: canonicalize identity-style display names AFTER all
        # evidence-text quoting checks. Running before would have the
        # `_clear_unsupported_*` step null out values that match raw
        # evidence but not the canonicalized form.
        stats["display_name_canonicalized"].extend(
            _apply_display_name_canonicalization(item)
        )
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
    if not stats["display_name_canonicalized"]:
        stats.pop("display_name_canonicalized")
    if not stats["non_radar_dropped"]:
        stats.pop("non_radar_dropped")
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


# 2026-05-16: identity-row labels used to find an entity's synth-block in
# normalized evidence text. Generic across docs — no specific equipment
# names. Same set as IDENTITY_LABELS_BY_ENTITY_TYPE["MISSILE_SYSTEM"] in
# `_pipeline_hooks.py`, plus name/display_name fallbacks.
_MISSILE_IDENTITY_ROW_LABELS_RE = (
    r"(?:MISSILE\s+TYPE|NATO\s+DESIGNATION|MILITARY\s+DESIGNATION|"
    r"INDUSTRY\s+DESIGNATION|MISSILE\s+DESIGNATION|SYSTEM\s+DESIGNATION|"
    r"WEAPON\s+DESIGNATION|DISPLAY\s+NAME|NAME|SYSTEM\s+NAME)"
)

# UNITS preamble marker — set by render_graph.py at the top of every
# graph-side synth table block when unit_convention is metric. Presence in
# a block authorizes bare-number → km/kg/etc. assumption.
_SI_UNIT_HINT_MARKER = "UNITS: NUMERIC VALUES IN THIS BLOCK ARE IN SI BASE UNITS"


def _extract_synth_block_for_entity(
    evidence_text: str,
    *,
    system_name: str | None,
    aliases: list[str] | None = None,
) -> str | None:
    """Find the synth ENTITY block whose identity row matches the given
    `system_name` or any of its `aliases`. Returns the block substring
    (from `ENTITY:` to the next `TABLE:`/`ENTITY:` boundary or EOT), or
    None if no matching block exists.

    Generalized over document layout: scans every ENTITY block, matches
    against any generic missile-identity row label (Missile Type / NATO
    Designation / Military Designation / Industry Designation / System
    Designation / Weapon Designation / Name / Display Name / System Name).
    No specific equipment names anywhere — purely structural matching.

    Used by mechanical numeric support to scope per-row parsing to the
    current entity's block, preventing cross-contamination on multi-entity
    tables.
    """
    if not isinstance(system_name, str) or not system_name.strip():
        return None
    names = [system_name.strip()]
    for a in aliases or ():
        if isinstance(a, str) and a.strip():
            names.append(a.strip())

    entity_starts = [
        m.start() for m in re.finditer(r"\bENTITY:", evidence_text, re.IGNORECASE)
    ]
    if not entity_starts:
        return None
    table_starts = [
        m.start() for m in re.finditer(r"\bTABLE:", evidence_text, re.IGNORECASE)
    ]

    # Return the NARROW entity-scoped portion only: from this ENTITY: to
    # the next ENTITY:/TABLE: boundary (or end of text). Sibling ENTITY
    # blocks are excluded so multi-entity tables don't cross-contaminate.
    # The chunk-level preamble (UNITS: etc.) is found separately by
    # `_evidence_has_si_unit_hint` looking backward through evidence_text.
    for i, ent_start in enumerate(entity_starts):
        eff_end = len(evidence_text)
        for nxt in entity_starts[i + 1:]:
            if nxt > ent_start:
                eff_end = nxt
                break
        for t in table_starts:
            if t > ent_start and t < eff_end:
                eff_end = t
                break
        block = evidence_text[ent_start:eff_end]
        for name in names:
            pattern = (
                _MISSILE_IDENTITY_ROW_LABELS_RE
                + r"\s*:\s*"
                + re.escape(name)
                + r"(?:\s|$)"
            )
            if re.search(pattern, block, re.IGNORECASE):
                return block
    return None


def _evidence_has_si_unit_hint(evidence_text: str, block: str) -> bool:
    """Return True if the synth-chunk that contains `block` has a SI
    UNIT_HINT preamble. Looks at evidence_text from the most recent
    TABLE: or UNITS: marker BEFORE block's start, up to block's start, for
    the UNIT_HINT signature.

    Defined per-chunk (not globally) so a doc with mixed-unit tables can
    have one chunk be metric and another be imperial without
    cross-contamination.
    """
    if _SI_UNIT_HINT_MARKER.lower() in block.lower():
        return True
    # block is a substring of evidence_text — find its start position.
    block_start = evidence_text.find(block)
    if block_start < 0:
        return False
    # Find most recent TABLE: or UNITS: preceding the block.
    preamble_start = 0
    for m in re.finditer(r"\b(?:TABLE|UNITS):", evidence_text[:block_start], re.IGNORECASE):
        preamble_start = m.start()
    preamble = evidence_text[preamble_start:block_start]
    return _SI_UNIT_HINT_MARKER.lower() in preamble.lower()


def _mechanically_supported_missile_fields(
    evidence_text: str,
    *,
    system_name: str | None = None,
    aliases: list[str] | None = None,
) -> dict[str, float]:
    """Extract mechanically-supported missile numeric values from evidence.

    Two channels:
    1. Museum-display range note (entity-agnostic — fine to apply globally).
    2. Synth-table block per-entity numerics (`Min Alt: 1000`, `Max Range:
       45000`, etc.) — ENTITY-SCOPED via `_extract_synth_block_for_entity`
       when `system_name` is provided. Without `system_name`, the synth-
       block channel is skipped to prevent cross-contamination (the first
       `Min Alt` in the doc would otherwise apply to every missile).

    Unit handling per the synth-block channel:
    - Explicit `KM` suffix → value used as-is.
    - Explicit `M` suffix → value / 1000.
    - Bare numeric AND the block contains the SI UNIT_HINT preamble → value
      / 1000 (SI-base metres assumption).
    - Bare numeric AND no UNIT_HINT → skipped (no inference from arbitrary prose).
    """
    supported = _extract_museum_display_range_notes(evidence_text)

    # Existing WEIGHT pattern — kept entity-agnostic for backward compat.
    weight_match = re.search(r"WEIGHT:\s*(?P<weight>[\d,]+(?:\.\d+)?)\s*LBS?\.?\b", evidence_text)
    if weight_match:
        supported["total_mass_kg"] = round(float(weight_match.group("weight").replace(",", "")) / 2.205, 1)

    # Step 5: launch-angle evidence patterns — entity-agnostic, anchored to
    # explicit launch/elevation context so unrelated "N degrees" phrases
    # (e.g. "rotated 360 degrees" referring to launcher azimuth) do not
    # false-trigger. Generic — no equipment names. First match wins.
    angle_match = (
        _LAUNCH_ANGLE_LABEL_RE.search(evidence_text)
        or _LAUNCH_ELEVATION_RE.search(evidence_text)
        or _LAUNCHED_AT_RE.search(evidence_text)
    )
    if angle_match:
        try:
            supported["max_launch_angle_deg"] = float(angle_match.group(1))
        except (ValueError, IndexError):
            pass

    # Step 4: spec fact overlay — parse labeled prose key-value blocks
    # from evidence_text (single-line, bullet-pair, paired max/min forms).
    # Generic, no equipment names; only fills fields the synth-block
    # channel didn't already populate.
    _apply_spec_overlay_to_supported(supported, evidence_text)

    # Synth-table block channel — entity-scoped only.
    block = _extract_synth_block_for_entity(
        evidence_text, system_name=system_name, aliases=aliases,
    )
    if block is None:
        return supported

    has_si_hint = _evidence_has_si_unit_hint(evidence_text, block)

    # Production-shape regex: works on normalized (no-newline, uppercased)
    # evidence AND on raw mixed-case test fixtures. Case-insensitive.
    _LINE = (
        r"(?:^|\s|-\s)"
        r"{label}\s*:\s*"
        r"(?P<v>[\d,]+(?:\.\d+)?)\s*(?P<u>KM|M)?\b"
    )
    _NUMERIC_FROM_SYNTH = (
        (r"MIN\s+ALT(?:ITUDE)?", "min_altitude_km"),
        (r"MAX\s+ALT(?:ITUDE)?", "max_altitude_km"),
        (r"MIN\s+RANGE",          "min_intercept_km"),
        (r"MAX\s+RANGE",          "max_intercept_km"),
    )
    for label_re, field in _NUMERIC_FROM_SYNTH:
        if field in supported:
            continue  # museum-display notes take precedence
        m = re.search(_LINE.format(label=label_re), block, re.IGNORECASE)
        if not m:
            continue
        v = float(m.group("v").replace(",", ""))
        u = (m.group("u") or "").upper()
        if u == "KM":
            supported[field] = round(v, 3)
        elif u == "M":
            supported[field] = round(v / 1000.0, 3)
        elif has_si_hint:
            # SI-base assumption authorized by the block's UNIT_HINT preamble.
            supported[field] = round(v / 1000.0, 3)
        # else: bare numeric with no unit evidence — skip (no inference).

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
    # 2026-05-16: pass system_name so the synth-block channel scopes per-
    # entity and avoids cross-contamination on multi-entity tables.
    # `aliases` is read from the item if present (LLM-emitted aliases) plus
    # nomenclature/name fields that may appear as alternate identifiers in
    # synth blocks. Generic — no equipment-specific names anywhere.
    _aliases: list[str] = []
    for k in ("nomenclature", "name", "display_label", "dieqp"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            _aliases.append(v)
    raw_aliases = item.get("aliases")
    if isinstance(raw_aliases, list):
        for a in raw_aliases:
            if isinstance(a, str) and a.strip():
                _aliases.append(a)
    supported_numeric = _mechanically_supported_missile_fields(
        evidence_text,
        system_name=item.get("system_name") if isinstance(item.get("system_name"), str) else None,
        aliases=_aliases,
    )

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

    # Mechanical-support override for kinematic numerics. Mechanical
    # conversion takes priority over the LLM's value when available;
    # otherwise fall back to the same evidence-verification predicate.
    # 2026-05-16: min_altitude_km added (was previously hard-cleared by the
    # below "unconditional-null" branch — see run history at
    # docs/sa2_extraction_runs.md for the regression that exposed this).
    # 2026-05-19 (Step 5): max_launch_angle_deg added — evidence parser
    # now reads anchored angle phrases (LAUNCH ANGLE / LAUNCHED AT N
    # DEGREES / LAUNCH ELEVATION). Unconditional null branch removed.
    for field_name in (
        "min_intercept_km",
        "max_intercept_km",
        "min_altitude_km",
        "max_altitude_km",
        "max_launch_angle_deg",
        "total_mass_kg",
    ):
        if field_name in supported_numeric:
            item[field_name] = supported_numeric[field_name]
        elif item.get(field_name) is not None:
            value = item[field_name]
            if not value_is_supported_by_text(value, field_name, evidence_text):
                item[field_name] = None
                cleared.append(field_name)

    # PRESERVED: unconditional-null for fields whose contract is "always
    # null" until evidence support is wired up. min_altitude_km removed
    # 2026-05-16; max_launch_angle_deg removed 2026-05-19 (Step 5).
    if item.get("missile_photo") is not None:
        item["missile_photo"] = None
        cleared.append("missile_photo")

    return cleared


# Step 6 follow-up: predecessor-context slash-group artifact filter.
#
# When the LLM extracts every named missile mention, malformed slash-
# group tokens in predecessor-context prose can be promoted to
# standalone `system_name` rows. Generic detection by context language
# and slash adjacency — no equipment names anywhere.
_PREDECESSOR_CONTEXT_MARKERS: tuple[str, ...] = (
    "EARLIER",
    "EVOLVED FROM",
    "PREDECESSOR",
    "PREDECESSORS",
    "PRECURSOR",
    "PRECURSORS",
    "ANCESTOR",
    "ANCESTORS",
    "SUPERSEDED",
)
_PREDECESSOR_CONTEXT_WINDOW_BEFORE_CHARS = 30


def _is_predecessor_slash_artifact(system_name: Any, evidence_text: str) -> bool:
    """True when `system_name` appears in `evidence_text` ONLY as a
    malformed slash-group token preceded by explicit predecessor-context
    language.

    Specifically, ALL of the following must hold:
      * At least one occurrence of the normalized name exists in
        `evidence_text`.
      * EVERY occurrence is immediately followed by `/`.
      * EVERY occurrence has one of `_PREDECESSOR_CONTEXT_MARKERS`
        within `_PREDECESSOR_CONTEXT_WINDOW_BEFORE_CHARS` chars before
        it.

    A real source-supported entity that has at least one legitimate
    (non-predecessor, non-slash-suffix) mention is NOT flagged. The
    "every occurrence" rule lets the docling document store the same
    token in both `text` and `orig` fields (which yields duplicate
    occurrences) without losing the artifact signal.

    Callers should additionally require the entity to have no extracted
    attributes before dropping it — this predicate alone is the shape
    test, not the should-drop decision.
    """
    if not isinstance(system_name, str) or not system_name.strip():
        return False
    if not isinstance(evidence_text, str) or not evidence_text:
        return False
    normalized_name = normalize_evidence_text(system_name)
    if not normalized_name:
        return False
    occurrences = list(re.finditer(re.escape(normalized_name), evidence_text))
    if not occurrences:
        return False
    for m in occurrences:
        end = m.end()
        if end >= len(evidence_text) or evidence_text[end] != "/":
            return False
        win_start = max(0, m.start() - _PREDECESSOR_CONTEXT_WINDOW_BEFORE_CHARS)
        pre_window = evidence_text[win_start:m.start()]
        if not any(marker in pre_window for marker in _PREDECESSOR_CONTEXT_MARKERS):
            return False
    return True


def _apply_designation_alias_overlay_to_missile_systems(
    missile_rows: list[dict[str, Any]],
    normalized_tables: list[Any] | None,
) -> dict[str, dict[str, list[str]]]:
    """Step 6 bridge: project per-column designation aliases onto each
    missile_systems row whose system_name matches the column's canonical
    entity. Returns a diagnostic mapping
    `{system_name: {nomenclature: [aliases], name: [aliases]}}`.

    No-op when normalized_tables is empty or the overlay module isn't
    available (older container image)."""
    if not normalized_tables:
        return {}
    try:
        from app.services.designation_alias_overlay import (
            expand_designation_aliases,
            merge_alias_bags_by_canonical,
        )
    except Exception:
        return {}
    bags = expand_designation_aliases(normalized_tables)
    if not bags:
        return {}
    by_entity = merge_alias_bags_by_canonical(bags)
    diagnostics: dict[str, dict[str, list[str]]] = {}
    for row in missile_rows:
        if not isinstance(row, dict):
            continue
        sysname = row.get("system_name")
        if not isinstance(sysname, str):
            continue
        bag = by_entity.get(sysname)
        if bag is None:
            continue
        added: dict[str, list[str]] = {"nomenclature": [], "name": []}
        for value in bag.nomenclature_aliases:
            current = row.get("nomenclature")
            if current is None:
                row["nomenclature"] = value
                added["nomenclature"].append(value)
            elif value != current and value not in current.split(" / "):
                row["nomenclature"] = f"{current} / {value}"
                added["nomenclature"].append(value)
        for value in bag.name_aliases:
            current = row.get("name")
            if current is None:
                row["name"] = value
                added["name"].append(value)
            elif value != current and value not in current.split(" / "):
                row["name"] = f"{current} / {value}"
                added["name"].append(value)
        if added["nomenclature"] or added["name"]:
            diagnostics[sysname] = {k: v for k, v in added.items() if v}
    return diagnostics


def _postprocess_air_defense_missiles(
    pass_output: dict[str, Any],
    evidence_text: str,
    normalized_tables: list[Any] | None = None,
    is_identity_pass: bool = False,
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
        "display_name_canonicalized": [],
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

        # Step 6 follow-up: drop predecessor-context slash-group artifacts
        # (e.g. SA-25 from "the earlier SA-25/S-25 / SA-1 Guild"). Only
        # applied on missile_identity and only when the row has no
        # extracted attributes beyond system_name — partial-attribute
        # rows are kept so we don't lose real partial extractions.
        if is_identity_pass and isinstance(system_name, str):
            has_any_attr = any(
                item.get(f) not in (None, "")
                for f in (
                    "nomenclature", "dieqp", "name", "emitter_function",
                    "system_status", "asrd", "responsible_agency",
                    "review_cycle", "next_review_date",
                )
            )
            if not has_any_attr and _is_predecessor_slash_artifact(
                system_name, evidence_text,
            ):
                stats.setdefault("predecessor_artifacts_dropped", []).append(
                    str(system_name)
                )
                continue

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
        # Item 5: canonicalize identity-style display names AFTER all
        # evidence-text quoting checks (same ordering rationale as radar).
        stats["display_name_canonicalized"].extend(
            _apply_display_name_canonicalization(item)
        )
        cleaned_rows.append(item)

    # Step 6: deterministic designation alias expansion. Attach Industry/
    # Military designations as `nomenclature` aliases and NATO
    # designations as `name` aliases onto the canonical missile round
    # entities (1D, 13D, 20D, 5Ya23, etc.) defined by the Missile Type
    # row of designation tables. No new entities are created.
    designation_expansion = _apply_designation_alias_overlay_to_missile_systems(
        cleaned_rows, normalized_tables,
    )
    if designation_expansion:
        stats["designation_alias_expansion"] = designation_expansion

    updated["missile_systems"] = cleaned_rows
    if not stats["status_cleared"]:
        stats.pop("status_cleared")
    if not stats["range_overrides"]:
        stats.pop("range_overrides")
    if not stats["unsupported_properties_cleared"]:
        stats.pop("unsupported_properties_cleared")
    if not stats["display_name_canonicalized"]:
        stats.pop("display_name_canonicalized")
    return updated, stats


def _build_upstream_name_map(upstream_entities: list[Any] | None) -> dict[str, str]:
    """Build name → ref_id lookup (TYPE-AGNOSTIC — preserved for backward
    compatibility with callers that don't track entity_type).

    Registers each entity under its primary identity (``system_name``),
    its display_label, AND each entry of ``aliases`` (typically populated
    from upstream schema fields like ``nomenclature`` and ``name``). The
    relationship pass uses this map to resolve chunk-cell names back to a
    ref_id whose primary identity might be a different token. Earlier
    registrations win on conflict so primary identity takes precedence
    over aliases.

    For type-aware resolution (preventing cross-type ref leaks), use
    `_build_upstream_name_map_by_type` instead.
    """
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


def _build_upstream_name_map_by_type(
    upstream_entities: list[Any] | None,
) -> dict[str, dict[str, str]]:
    """Build {entity_type → {normalized_name → ref_id}} lookup.

    Per-type segregation prevents cross-type leaks during cross-entity-
    hint resolution: if a missile name happens to collide with a radar
    name in another part of the catalog, the resolver only sees the
    name under its declared entity_type.

    Returns an empty dict when input is empty. Entries with no `ref_id`
    or no resolvable `entity_type` are skipped. Within a type, earlier
    registrations win on conflict (primary identity beats aliases).
    """
    if not upstream_entities:
        return {}
    out: dict[str, dict[str, str]] = {}

    def _maybe_register(name: Any, ref_id: str, type_map: dict[str, str]) -> None:
        if not isinstance(name, str):
            return
        key = normalize_evidence_text(name)
        if not key or key in type_map:
            return
        type_map[key] = ref_id

    for entity in upstream_entities:
        ref_id = getattr(entity, "ref_id", None)
        entity_type = getattr(entity, "entity_type", None)
        if not ref_id or not isinstance(entity_type, str) or not entity_type:
            continue
        type_map = out.setdefault(entity_type, {})
        identity_values = getattr(entity, "identity_values", None) or {}
        _maybe_register(identity_values.get("system_name"), ref_id, type_map)
        _maybe_register(getattr(entity, "display_label", None), ref_id, type_map)
        aliases = getattr(entity, "aliases", None) or []
        for alias in aliases:
            _maybe_register(alias, ref_id, type_map)
    return out


def _resolve_ref(
    name: str,
    entity_type: str | None,
    upstream_name_to_ref_by_type: dict[str, dict[str, str]],
    alias_map_by_entity_type: dict[str, dict[str, str]] | None,
) -> str | None:
    """Resolve an entity name to an upstream ref_id WITHIN the same entity
    type — never cross-types.

    Strategy:
      1. Direct hit in `upstream_name_to_ref_by_type[entity_type]` (per-type
         segregation prevents a missile name from accidentally resolving to
         a radar ref or vice versa).
      2. If that misses, consult the table-overlay alias map for the same
         entity_type to map `name → canonical_name`, then try the per-type
         upstream lookup again on the canonical.
      3. Return None if neither path resolves OR if `entity_type` is missing.

    Why this matters: per-pass canonical-name cleanup (synth-only chunking)
    can shrink the upstream alias diversity, leaving table-derived hints
    that reference table-local aliases without a direct upstream match.
    The overlay's alias_map_by_entity_type carries those table-local
    aliases mapped to the canonical the upstream catalog knows.

    Type segregation is critical because the same lexical token can denote
    different entities in different domains (e.g. a designation that
    appears in both a missile catalog and a radar catalog).
    """
    if not isinstance(name, str):
        return None
    key = normalize_evidence_text(name)
    if not key:
        return None
    if not isinstance(entity_type, str) or not entity_type:
        return None

    # 1. Direct upstream hit — TYPE-SCOPED.
    type_map = (upstream_name_to_ref_by_type or {}).get(entity_type) or {}
    ref = type_map.get(key)
    if ref:
        return ref

    # 2. Overlay alias fallback for THIS entity type.
    by_type = alias_map_by_entity_type or {}
    alias_map = by_type.get(entity_type) or {}
    if not alias_map:
        return None
    # Try both normalized and original-cased keys — overlay alias maps may
    # be built either way depending on producer.
    canonical = alias_map.get(key) or alias_map.get(name)
    if not isinstance(canonical, str):
        return None
    # Resolve the canonical through the SAME type's upstream map.
    return type_map.get(normalize_evidence_text(canonical))


# Item 3 (role-aware CUES validation). Generic role categories — no
# equipment-specific names. Constants drive the helper below.
#
# Cueing semantics: source detects/tracks ahead; target engages on the
# track handoff. Source-side roles are surveillance / acquisition / early-
# warning class radars. Target-side role is the engagement radar.
#
# We intentionally exclude MULTI_FUNCTION and TRACKING from both sets
# because they can play either role depending on context; treating them
# as ambiguous and leaving the LLM output unchanged is safer.
_CUES_SOURCE_ROLES: frozenset[str] = frozenset({
    "SEARCH", "HEIGHT_FINDER",
})
_CUES_TARGET_ROLES: frozenset[str] = frozenset({
    "FIRE_CONTROL",
})


def _build_role_map_by_ref(upstream_entities: list[Any] | None) -> dict[str, str]:
    """Return ``{ref_id → emitter_function}`` for RADAR_SYSTEM upstream entities
    that carry a role on ``properties.emitter_function``. Missiles and
    role-less entities are omitted. Generic — keys are ref_ids only.
    """
    role_map: dict[str, str] = {}
    if not upstream_entities:
        return role_map
    for ent in upstream_entities:
        entity_type = getattr(ent, "entity_type", None) or (
            ent.get("entity_type") if isinstance(ent, dict) else None
        )
        if entity_type != "RADAR_SYSTEM":
            continue
        ref_id = getattr(ent, "ref_id", None) or (
            ent.get("ref_id") if isinstance(ent, dict) else None
        )
        if not isinstance(ref_id, str) or not ref_id:
            continue
        props = getattr(ent, "properties", None) or (
            ent.get("properties") if isinstance(ent, dict) else None
        )
        if not isinstance(props, dict):
            continue
        role = props.get("emitter_function")
        if isinstance(role, str) and role:
            role_map[ref_id] = role
    return role_map


def _retype_radar_radar_to_cues(
    rels: list[dict[str, Any]],
    role_map: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """In-place: retype/flip RADAR_SYSTEM → RADAR_SYSTEM edges to CUES when
    source/target roles support the cueing direction. Never creates new
    edges. Returns diagnostics: ``{retyped, flipped, skipped}`` lists.

    Rule (generic, no equipment names):
      * source role ∈ _CUES_SOURCE_ROLES AND target role ∈ _CUES_TARGET_ROLES
        → set rel_type = CUES (same direction).
      * source role ∈ _CUES_TARGET_ROLES AND target role ∈ _CUES_SOURCE_ROLES
        → FLIP direction and set rel_type = CUES.
      * Else (missing role, ambiguous role pair, or matching roles on
        both sides) → leave unchanged and emit diagnostic.

    Edges already typed CUES that satisfy the same-direction rule are
    considered correct and not counted as "retyped".

    Cross-type edges (RADAR↔MISSILE) are out-of-scope and not touched.
    """
    diag: dict[str, list[dict[str, Any]]] = {
        "retyped": [], "flipped": [], "skipped": [],
    }
    for rel in rels:
        f = rel.get("from_ref_id") or ""
        t = rel.get("to_ref_id") or ""
        if not (f.startswith("RADAR_SYSTEM:") and t.startswith("RADAR_SYSTEM:")):
            continue
        f_role = role_map.get(f)
        t_role = role_map.get(t)
        if not f_role or not t_role:
            diag["skipped"].append({
                "from_ref_id": f, "to_ref_id": t,
                "rel_type": rel.get("rel_type"),
                "reason": "missing_role",
                "from_role": f_role, "to_role": t_role,
            })
            continue
        # Same-direction match: source is source-side AND target is target-side
        if f_role in _CUES_SOURCE_ROLES and t_role in _CUES_TARGET_ROLES:
            if rel.get("rel_type") != "CUES":
                rel["rel_type"] = "CUES"
                diag["retyped"].append({
                    "from_ref_id": f, "to_ref_id": t,
                    "from_role": f_role, "to_role": t_role,
                    "reason": (
                        "height_finder_to_fire_control"
                        if f_role == "HEIGHT_FINDER"
                        else "search_to_fire_control"
                    ),
                })
            continue
        # Reversed-direction match: should flip + CUES
        if f_role in _CUES_TARGET_ROLES and t_role in _CUES_SOURCE_ROLES:
            rel["from_ref_id"], rel["to_ref_id"] = t, f
            rel["rel_type"] = "CUES"
            diag["flipped"].append({
                "original_from_ref_id": f, "original_to_ref_id": t,
                "new_from_ref_id": t, "new_to_ref_id": f,
                "from_role": f_role, "to_role": t_role,
                "reason": "flipped_fire_control_to_source",
            })
            continue
        # Ambiguous or unsupported role combination
        diag["skipped"].append({
            "from_ref_id": f, "to_ref_id": t,
            "rel_type": rel.get("rel_type"),
            "reason": "roles_ambiguous_or_unsupported",
            "from_role": f_role, "to_role": t_role,
        })
    return diag


# Item 4 (VARIANT_OF emitter) — parent-name eligibility constants.
# Generic guardrails against false parentage from raw substring matching.
#
# SCOPE NOTE: these rules are tuned for designation-style entity families
# (e.g. military systems, model numbers, product SKUs) where canonical
# names mix letters and digits ("S-75", "F-16C", "RX-7"). Pure-letters
# family names ("Dvina", "Apache") are intentionally rejected as parent
# candidates here because they collide too easily with prose tokens
# inside aliases. If a future ontology needs all-letter family names as
# parents, this gate needs ontology-specific tuning rather than the
# generic letter+digit rule.
#
# Within scope: missile/radar SAM-family designations.
# Out of scope (intentional): consumer-product families with all-letter
# names; people; place names.
_VARIANT_OF_MIN_PARENT_NAME_LEN = 3


def _parent_name_is_eligible(parent_sysname: str) -> bool:
    """Generic guardrails for designation-style parent names. See module-
    level SCOPE NOTE above: rejects all-letter or all-digit candidates
    and anything shorter than ``_VARIANT_OF_MIN_PARENT_NAME_LEN``."""
    if len(parent_sysname) < _VARIANT_OF_MIN_PARENT_NAME_LEN:
        return False
    has_letter = any(c.isalpha() for c in parent_sysname)
    has_digit = any(c.isdigit() for c in parent_sysname)
    return has_letter and has_digit


def _find_parent_in_alias(alias: str, parent_sysname: str) -> str | None:
    """Boundary-aware match: parent_sysname appears in alias only when
    preceded/followed by a non-alphanumeric character (or string edge).

    Returns a match-kind string: 'exact' if alias equals parent
    (whitespace-normalized), 'boundary' for a properly-bounded substring,
    or None if no valid match.

    Generic — no equipment-specific logic. Rejects S-75 inside S-750,
    SA-2 inside SA-20, etc.
    """
    if not alias or not parent_sysname:
        return None
    a_norm = " ".join(alias.strip().split())
    p_norm = " ".join(parent_sysname.strip().split())
    if not a_norm or not p_norm:
        return None
    if a_norm.upper() == p_norm.upper():
        return "exact"
    # Substring match with alphanumeric boundary on both sides
    a_upper = a_norm.upper()
    p_upper = p_norm.upper()
    start = 0
    while True:
        idx = a_upper.find(p_upper, start)
        if idx < 0:
            return None
        end = idx + len(p_upper)
        # Character before: must be non-alnum (or string start)
        before_ok = (idx == 0) or (not a_upper[idx - 1].isalnum())
        # Character after: must be non-alnum (or string end)
        after_ok = (end == len(a_upper)) or (not a_upper[end].isalnum())
        if before_ok and after_ok:
            return "boundary"
        start = idx + 1  # try next occurrence


def _emit_variant_of_relationships(
    out: list[dict[str, Any]],
    upstream_entities: list[Any] | None,
    seen_pairs: set[tuple[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Item 4: deterministically emit VARIANT_OF edges from missile-variant
    entities to their parent SAM family entities.

    Generic — operates purely on structural evidence:
      * Source: any MISSILE_SYSTEM entity whose alias/identity field
        contains another MISSILE_SYSTEM entity's ``system_name`` as a
        case-insensitive **boundary-aware** match.
      * Target: that other MISSILE_SYSTEM entity (the parent family).
      * Parent eligibility: ``_parent_name_is_eligible`` requires ≥3
        chars AND mix of letters+digits — rejects too-generic names.
      * Boundary rule: ``_find_parent_in_alias`` rejects substring
        matches where the parent name abuts an alphanumeric character
        (e.g. S-75 inside S-750, SA-2 inside SA-20).
      * If multiple parents match the same alias, prefer **exact alias =
        parent** matches over **boundary substring** matches; tie-break
        by longest system_name.
      * Never creates new entities. Never creates cross-type edges.
      * Skips self-loops. Dedupes against ``seen_pairs``.

    Returns diagnostics: ``{emitted, skipped}`` lists.

    The function mutates ``out`` and ``seen_pairs``. Caller runs this
    AFTER LLM-emitted rels + hint promotion so the dedup check is final.
    """
    diag: dict[str, list[dict[str, Any]]] = {"emitted": [], "skipped": []}
    if not upstream_entities:
        return diag

    # Collect MISSILE_SYSTEM entries + their aliases. Aliases carry the
    # candidate parent surface forms (nomenclature, name, dieqp).
    missile_systems: list[tuple[str, str]] = []
    missile_aliases: dict[str, list[str]] = {}
    for ent in upstream_entities:
        entity_type = getattr(ent, "entity_type", None) or (
            ent.get("entity_type") if isinstance(ent, dict) else None
        )
        if entity_type != "MISSILE_SYSTEM":
            continue
        ref_id = getattr(ent, "ref_id", None) or (
            ent.get("ref_id") if isinstance(ent, dict) else None
        )
        ident = getattr(ent, "identity_values", None) or (
            ent.get("identity_values") if isinstance(ent, dict) else None
        )
        sysname = (ident or {}).get("system_name") if isinstance(ident, dict) else None
        aliases = getattr(ent, "aliases", None) or (
            ent.get("aliases") if isinstance(ent, dict) else None
        )
        if not isinstance(ref_id, str) or not isinstance(sysname, str):
            continue
        missile_systems.append((ref_id, sysname))
        if isinstance(aliases, list):
            missile_aliases[ref_id] = [a for a in aliases if isinstance(a, str) and a]

    if len(missile_systems) < 2:
        return diag

    # Eligible parent candidates only. Sort by length descending so that
    # within the same match_kind, the longer (more specific) parent wins.
    eligible_parents = [
        (ref, sysname) for ref, sysname in missile_systems
        if _parent_name_is_eligible(sysname)
    ]
    eligible_parents.sort(key=lambda pair: len(pair[1]), reverse=True)

    for child_ref, child_sysname in missile_systems:
        aliases = missile_aliases.get(child_ref, [])
        if not aliases:
            continue
        # For each alias, find best match across all eligible parents.
        # Prefer 'exact' over 'boundary'; tie-break by parent length.
        best: tuple[str, str, str, str] | None = None  # (kind, parent_ref, parent_sysname, alias)
        for alias in aliases:
            for parent_ref, parent_sysname in eligible_parents:
                if parent_ref == child_ref:
                    continue  # no self-loop
                kind = _find_parent_in_alias(alias, parent_sysname)
                if kind is None:
                    continue
                # Score: exact (2) beats boundary (1); within same kind,
                # longer parent_sysname wins (eligible_parents is sorted
                # longest-first, so the first hit at each kind is best).
                kind_score = 2 if kind == "exact" else 1
                if best is None:
                    best = (kind, parent_ref, parent_sysname, alias)
                    if kind == "exact":
                        break  # can't do better than exact
                else:
                    best_kind_score = 2 if best[0] == "exact" else 1
                    if kind_score > best_kind_score:
                        best = (kind, parent_ref, parent_sysname, alias)
                        if kind == "exact":
                            break
                    elif (kind_score == best_kind_score
                          and len(parent_sysname) > len(best[2])):
                        best = (kind, parent_ref, parent_sysname, alias)
            if best is not None and best[0] == "exact":
                break
        if best is None:
            continue
        kind, parent_ref, parent_sysname, matched_alias = best
        pair = (child_ref, parent_ref)
        if pair in seen_pairs:
            diag["skipped"].append({
                "child_ref_id": child_ref,
                "parent_ref_id": parent_ref,
                "matched_alias": matched_alias,
                "reason": "duplicate_of_existing_edge",
            })
            continue
        edge = {
            "rel_type": "VARIANT_OF",
            "from_ref_id": child_ref,
            "to_ref_id": parent_ref,
            "confidence": 1.0,
        }
        out.append(edge)
        seen_pairs.add(pair)
        diag["emitted"].append({
            "child_ref_id": child_ref,
            "parent_ref_id": parent_ref,
            "matched_alias": matched_alias,
            "matched_parent_system_name": parent_sysname,
            "match_kind": kind,  # 'exact' or 'boundary'
            "reason": (
                "alias_equals_parent_system_name" if kind == "exact"
                else "alias_contains_parent_system_name_at_boundary"
            ),
        })
    return diag


def _postprocess_air_defense_system_links(
    pass_output: dict[str, Any],
    evidence_text: str,
    upstream_entities: list[Any] | None,
    cross_entity_hints: list[Any] | None = None,
    *,
    alias_map_by_entity_type: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Postprocess system_links output.

    - Always promotes ``cross_entity_hints`` from the deterministic table
      overlay into ASSOCIATED_WITH edges (resolving source/target by name
      against the upstream catalog). v8a — closes the relationship-yield
      gap on table-anchored radar↔missile pairings.
    - 2026-05-16: when a hint's source_name or target_name doesn't directly
      hit the upstream catalog, falls back to the table-overlay
      ``alias_map_by_entity_type`` to map alias → canonical → upstream ref.
      Closes the system_links regression where per-pass canonical-name
      cleanup made table-local aliases (e.g. `20DP`, `RSN- 75V`) stop
      resolving against the smaller upstream alias set.
    - Falls back to evidence-text heuristics (Spoon Rest → Fan Song,
      Fan Song → SA-2) only when the LLM emitted ZERO relationships AND
      no hint promotions resolved.
    - Dedupes by ``(from_ref_id, to_ref_id)`` canonical-direction; the
      first-emitted wins."""
    updated = dict(pass_output)
    relationships = updated.get("relationships")
    if not isinstance(relationships, list):
        return updated, {}

    # Type-agnostic map: still used by the legacy evidence-text fallback
    # heuristics below (Spoon Rest → Fan Song, etc.) which were never
    # type-aware. New cross-entity-hint resolution uses the type-segregated
    # map via _resolve_ref to prevent cross-type leaks.
    name_to_ref = _build_upstream_name_map(upstream_entities)
    name_to_ref_by_type = _build_upstream_name_map_by_type(upstream_entities)
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

    # Item 3 (role-aware CUES validation): retype/flip RADAR_SYSTEM →
    # RADAR_SYSTEM LLM-emitted edges to CUES when source/target roles
    # support the cueing direction. Runs BEFORE hint promotion so the
    # deterministic promoted edges keep their canonical ASSOCIATED_WITH
    # type. Generic — operates only on entity_type + role string.
    role_map = _build_role_map_by_ref(upstream_entities)
    role_aware_diag = _retype_radar_radar_to_cues(out, role_map)
    # Update seen_pairs to reflect any flipped directions so the hint
    # promotion below doesn't add a duplicate.
    if role_aware_diag.get("flipped"):
        seen_pairs.clear()
        for rel in out:
            f = rel.get("from_ref_id"); t = rel.get("to_ref_id")
            if isinstance(f, str) and isinstance(t, str):
                seen_pairs.add((f, t))
    if any(role_aware_diag.get(k) for k in ("retyped", "flipped", "skipped")):
        stats["role_aware_cues"] = role_aware_diag

    # v8a: promote cross_entity_hints to ASSOCIATED_WITH edges.
    # 2026-05-16: use _resolve_ref so unmatched hints fall back to the
    # overlay's alias_map_by_entity_type before being discarded.
    # 2026-05-17 (Rec 1): collect unresolved-hint diagnostics so the
    # postprocess output exposes WHY hints fail to promote.
    promoted: list[dict[str, Any]] = []
    unresolved_samples: list[dict[str, Any]] = []
    unresolved_count = 0
    _UNRESOLVED_SAMPLE_CAP = 20
    if cross_entity_hints:
        for hint in cross_entity_hints:
            source_name = getattr(hint, "source_canonical", None) or (
                hint.get("source_canonical") if isinstance(hint, dict) else None
            )
            target_name = getattr(hint, "target_alias", None) or (
                hint.get("target_alias") if isinstance(hint, dict) else None
            )
            source_entity_type = getattr(hint, "source_entity_type", None) or (
                hint.get("source_entity_type") if isinstance(hint, dict) else None
            )
            target_entity_type = getattr(hint, "target_entity_type", None) or (
                hint.get("target_entity_type") if isinstance(hint, dict) else None
            )
            if not isinstance(source_name, str) or not isinstance(target_name, str):
                continue
            source_ref = _resolve_ref(
                source_name, source_entity_type,
                name_to_ref_by_type, alias_map_by_entity_type,
            )
            target_ref = _resolve_ref(
                target_name, target_entity_type,
                name_to_ref_by_type, alias_map_by_entity_type,
            )
            if not source_ref or not target_ref:
                unresolved_count += 1
                if len(unresolved_samples) < _UNRESOLVED_SAMPLE_CAP:
                    if not source_ref and not target_ref:
                        reason = "both_unresolved"
                    elif not source_ref:
                        reason = "source_unresolved"
                    else:
                        reason = "target_unresolved"
                    unresolved_samples.append({
                        "source_alias": source_name,
                        "source_type": source_entity_type,
                        "target_alias": target_name,
                        "target_type": target_entity_type,
                        "source_resolved": source_ref is not None,
                        "target_resolved": target_ref is not None,
                        "reason": reason,
                    })
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
    # Rec 1: always emit unresolved-hint diagnostics (including the count=0
    # case) so downstream callers can distinguish "no hints were generated"
    # from "hints generated, all resolved" from "hints generated, N failed".
    if cross_entity_hints is not None:
        stats["unresolved_cross_entity_hints"] = {
            "count": unresolved_count,
            "samples": unresolved_samples,
        }

    # Item 4: deterministic VARIANT_OF emitter — runs AFTER hint promotion
    # so the dedup-vs-seen_pairs check covers all prior edges (LLM-emitted
    # + role-retyped + hint-promoted). Only emits MISSILE_SYSTEM →
    # MISSILE_SYSTEM family/variant edges based on alias-substring evidence.
    # Generic — no equipment names anywhere.
    variant_of_diag = _emit_variant_of_relationships(out, upstream_entities, seen_pairs)
    if variant_of_diag.get("emitted") or variant_of_diag.get("skipped"):
        stats["variant_of_emitter"] = variant_of_diag

    # Legacy SA-2 evidence-text fallback (Spoon-Rest→Fan-Song CUES,
    # Fan-Song→SA-2 ASSOCIATED_WITH). Quarantined behind the env flag.
    # The generic VARIANT_OF emitter + hint promotion + role-aware
    # CUES retype now cover the relationship semantics for any corpus
    # whose tables expose them. Default OFF so non-SA-2 docs aren't
    # polluted by SA-2-specific edge synthesis.
    derived: list[dict[str, Any]] = []
    if not out and _LEGACY_SA2_FALLBACKS_ENABLED:
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
