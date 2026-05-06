"""Structured alias map (spec §5.1).

Pairs with the §12b prose in ontology_bundles/_shared/prompt_rules.py
DELTA_SYSTEM_PROMPT. The prose serves the LLM (handles natural-language
conditionals like "only when describing the whole missile"); this module
serves the synthesizer (programmatic lookup with pass- and section-
conditionals as first-class tuple keys).

A drift-guard test (tests/test_alias_map.py) asserts every ALIAS_MAP entry
has a corresponding §12b prose mention (per-token check), and every
SECTION_KEYWORDS entry appears as a contiguous phrase in the prose.

ALIAS_MAP entries are populated in Tasks 4 and 5 of the implementation
plan; this module ships with empty ALIAS_MAP plus the constant tables
that have no per-pass conditionals.
"""
from __future__ import annotations

# AliasKey: tuple[str, SectionContext, str] = (label_normalized, section_ctx, pass_name)
# Value: canonical schema field name (e.g., "booster_mass_kg").
# Populated in Tasks 4 (missile passes) and 5 (radar passes).
ALIAS_MAP: dict[tuple[str, str | None, str], str] = {}

# Section keywords detected by the embedded substring scan in
# detect_section_context (spec §5.4 strategy 1) and the standalone-row
# header path (strategy 2). Extensible per domain; entries here MUST
# also appear verbatim in §12b prose.
SECTION_KEYWORDS: tuple[str, ...] = (
    "1st Stage",
    "2nd Stage",
    "Booster",
    "Sustainer",
    "Sustain",
    "Ejector",
)

# Per-unit-class conversion factors, keyed by unit-class name. Inner dict
# maps cell-extracted unit string (lowercased) to the multiplicative factor
# that converts to the canonical unit (factor 1.0).
#
# Populated for the unit classes used by the four missile passes plus
# the five radar sub-passes. Add new unit classes by extending this dict
# AND adding a corresponding entry in FIELD_SUFFIX_TO_UNIT_CLASS.
UNIT_TABLE: dict[str, dict[str, float]] = {
    "length_m": {
        "m": 1.0,
        "mm": 0.001,
        "cm": 0.01,
        "in": 0.0254,
        "ft": 0.3048,
        "km": 1000.0,
    },
    "length_km": {
        "km": 1.0,
        "m": 0.001,
        "mi": 1.609344,
        "nm": 1.852,
        "nmi": 1.852,
    },
    "mass_kg": {
        "kg": 1.0,
        "g": 0.001,
        "lb": 0.453592,
        "lbs": 0.453592,
        "t": 1000.0,
        "ton": 1000.0,
        "tonne": 1000.0,
    },
    "time_sec": {
        "sec": 1.0,
        "s": 1.0,
        "ms": 0.001,
        "min": 60.0,
        "minutes": 60.0,
    },
    "velocity_mps": {
        "mps": 1.0,
        "m/s": 1.0,
        "kmh": 1.0 / 3.6,
        "km/h": 1.0 / 3.6,
        "mph": 0.44704,
        "knots": 0.514444,
        "kt": 0.514444,
    },
    "frequency_mhz": {
        "mhz": 1.0,
        "ghz": 1000.0,
        "khz": 0.001,
        "hz": 0.000001,
    },
    "gain_dbi": {
        "dbi": 1.0,
    },
    "power_kw": {
        "kw": 1.0,
        "w": 0.001,
        "mw": 1000.0,
    },
    "power_dbw": {
        "dbw": 1.0,
        "dbm": 1.0,
    },
    "angle_deg": {
        "deg": 1.0,
        "°": 1.0,
        "degrees": 1.0,
        "rad": 57.2957795,
        "radians": 57.2957795,
    },
}

# Schema-field-suffix -> unit-class mapping. The synthesizer's coerce_value
# function looks at the schema field name (e.g., "booster_mass_kg"), reads
# the suffix ("_kg"), and selects the unit class ("mass_kg") to coerce
# against. Schema fields whose suffix isn't here go through the string
# passthrough path (e.g., "booster_thrust" has no numeric suffix).
FIELD_SUFFIX_TO_UNIT_CLASS: dict[str, str] = {
    "_m": "length_m",
    "_km": "length_km",
    "_kg": "mass_kg",
    "_sec": "time_sec",
    "_mps": "velocity_mps",
    "_mhz": "frequency_mhz",
    "_dbi": "gain_dbi",
    "_kw": "power_kw",
    "_dbw": "power_dbw",
    "_deg": "angle_deg",
}
