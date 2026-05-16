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
# IMPORTANT: keys are pre-normalized via _table_facts.normalize_label.
# Authors must lowercase, strip punctuation per the normalizer rules, and
# leave hyphens intact. The drift-guard test enforces this match against
# §12b prose tokens in prompt_rules.DELTA_SYSTEM_PROMPT.
ALIAS_MAP: dict[tuple[str, str | None, str], str] = {
    # ============================================================
    # missile_kinematics
    # ============================================================
    # Range -> max_intercept_km
    ("range",            None, "missile_kinematics"): "max_intercept_km",
    ("max range",        None, "missile_kinematics"): "max_intercept_km",
    ("max range km",     None, "missile_kinematics"): "max_intercept_km",
    ("max range m",      None, "missile_kinematics"): "max_intercept_km",
    ("maximum range",    None, "missile_kinematics"): "max_intercept_km",
    ("effective range",  None, "missile_kinematics"): "max_intercept_km",
    ("engagement range", None, "missile_kinematics"): "max_intercept_km",
    # Min Range -> min_intercept_km
    ("min range",        None, "missile_kinematics"): "min_intercept_km",
    ("min range km",     None, "missile_kinematics"): "min_intercept_km",
    ("min range m",      None, "missile_kinematics"): "min_intercept_km",
    ("minimum range",    None, "missile_kinematics"): "min_intercept_km",
    # Altitude -> max_altitude_km. 2026-05-16: "Alt" abbreviations were
    # withheld in the original alias-map design because §12b prose only
    # carried the full word "Altitude" and the drift guard would have failed.
    # §12b was updated to include "Max Alt" / "Min Alt" with an explicit
    # equivalence note, so the abbreviated forms are now registered here.
    ("altitude",            None, "missile_kinematics"): "max_altitude_km",
    ("max altitude",        None, "missile_kinematics"): "max_altitude_km",
    ("max altitude km",     None, "missile_kinematics"): "max_altitude_km",
    ("max altitude m",      None, "missile_kinematics"): "max_altitude_km",
    ("max alt",             None, "missile_kinematics"): "max_altitude_km",
    ("max alt km",          None, "missile_kinematics"): "max_altitude_km",
    ("max alt m",           None, "missile_kinematics"): "max_altitude_km",
    ("ceiling",             None, "missile_kinematics"): "max_altitude_km",
    ("engagement altitude", None, "missile_kinematics"): "max_altitude_km",
    # Min Altitude -> min_altitude_km
    ("min altitude",        None, "missile_kinematics"): "min_altitude_km",
    ("min altitude km",     None, "missile_kinematics"): "min_altitude_km",
    ("min altitude m",      None, "missile_kinematics"): "min_altitude_km",
    ("min alt",             None, "missile_kinematics"): "min_altitude_km",
    ("min alt km",          None, "missile_kinematics"): "min_altitude_km",
    ("min alt m",           None, "missile_kinematics"): "min_altitude_km",

    # ============================================================
    # missile_airframe
    # ============================================================
    ("length",           None, "missile_airframe"): "body_length_m",
    ("length mm",        None, "missile_airframe"): "body_length_m",
    ("length m",         None, "missile_airframe"): "body_length_m",
    ("overall length",   None, "missile_airframe"): "body_length_m",
    ("missile length",   None, "missile_airframe"): "body_length_m",
    ("body length",      None, "missile_airframe"): "body_length_m",

    ("diameter",         None, "missile_airframe"): "body_diameter_m",
    ("diameter mm",      None, "missile_airframe"): "body_diameter_m",
    ("body diameter",    None, "missile_airframe"): "body_diameter_m",
    ("calibre",          None, "missile_airframe"): "body_diameter_m",
    ("caliber",          None, "missile_airframe"): "body_diameter_m",

    # Total mass — when section_ctx is None, "Weight" / "Mass" map to total.
    ("weight",           None, "missile_airframe"): "total_mass_kg",
    ("weight kg",        None, "missile_airframe"): "total_mass_kg",
    ("mass",             None, "missile_airframe"): "total_mass_kg",
    ("mass kg",          None, "missile_airframe"): "total_mass_kg",
    ("total weight",     None, "missile_airframe"): "total_mass_kg",
    ("total weight kg",  None, "missile_airframe"): "total_mass_kg",
    ("launch weight",    None, "missile_airframe"): "total_mass_kg",
    ("launch mass",      None, "missile_airframe"): "total_mass_kg",

    # ============================================================
    # missile_speed_timing
    # ============================================================
    ("speed",            None, "missile_speed_timing"): "max_speed_mps",
    ("max speed",        None, "missile_speed_timing"): "max_speed_mps",
    ("max speed m s",    None, "missile_speed_timing"): "max_speed_mps",
    ("max speed mps",    None, "missile_speed_timing"): "max_speed_mps",
    ("velocity",         None, "missile_speed_timing"): "max_speed_mps",
    ("maximum velocity", None, "missile_speed_timing"): "max_speed_mps",
    ("average speed",    None, "missile_speed_timing"): "average_speed_mps",
    ("flight time",      None, "missile_speed_timing"): "flight_time_sec",
    ("time of flight",   None, "missile_speed_timing"): "flight_time_sec",
    ("flyout time",      None, "missile_speed_timing"): "max_flyout_time_sec",
    ("burn time",        None, "missile_speed_timing"): "total_burn_time_sec",

    # ============================================================
    # missile_propulsion
    # ============================================================
    # Booster (1st Stage) — Weight maps to booster_mass_kg.
    ("weight",      "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("weight kg",   "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("mass",        "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("mass kg",     "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("weight",      "Booster",   "missile_propulsion"): "booster_mass_kg",
    ("weight kg",   "Booster",   "missile_propulsion"): "booster_mass_kg",
    ("mass",        "Booster",   "missile_propulsion"): "booster_mass_kg",
    ("mass kg",     "Booster",   "missile_propulsion"): "booster_mass_kg",
    # Booster — Time maps to booster_time_sec.
    ("time",        "1st Stage", "missile_propulsion"): "booster_time_sec",
    ("time sec",    "1st Stage", "missile_propulsion"): "booster_time_sec",
    ("burn time",   "1st Stage", "missile_propulsion"): "booster_time_sec",
    ("time",        "Booster",   "missile_propulsion"): "booster_time_sec",
    ("time sec",    "Booster",   "missile_propulsion"): "booster_time_sec",
    ("burn time",   "Booster",   "missile_propulsion"): "booster_time_sec",
    # Booster — Thrust (string field; passthrough).
    ("thrust",      "1st Stage", "missile_propulsion"): "booster_thrust",
    ("thrust",      "Booster",   "missile_propulsion"): "booster_thrust",

    # Sustainer (2nd Stage) — Weight maps to sustain_mass_kg.
    ("weight",      "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("weight kg",   "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("mass",        "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("mass kg",     "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("weight",      "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("weight kg",   "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("mass",        "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("mass kg",     "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("weight",      "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    ("weight kg",   "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    ("mass",        "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    ("mass kg",     "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    # Sustainer — Time maps to sustain_time_sec.
    ("time",        "2nd Stage", "missile_propulsion"): "sustain_time_sec",
    ("time sec",    "2nd Stage", "missile_propulsion"): "sustain_time_sec",
    ("burn time",   "2nd Stage", "missile_propulsion"): "sustain_time_sec",
    ("time",        "Sustainer", "missile_propulsion"): "sustain_time_sec",
    ("time sec",    "Sustainer", "missile_propulsion"): "sustain_time_sec",
    ("burn time",   "Sustainer", "missile_propulsion"): "sustain_time_sec",
    ("time",        "Sustain",   "missile_propulsion"): "sustain_time_sec",
    ("time sec",    "Sustain",   "missile_propulsion"): "sustain_time_sec",
    ("burn time",   "Sustain",   "missile_propulsion"): "sustain_time_sec",
    # Sustainer — Thrust (string field; passthrough).
    ("thrust",      "2nd Stage", "missile_propulsion"): "sustain_thrust",
    ("thrust",      "Sustainer", "missile_propulsion"): "sustain_thrust",
    ("thrust",      "Sustain",   "missile_propulsion"): "sustain_thrust",

    # Ejector — Weight / Thrust under Ejector section.
    # Schema has only ejector_thrust + ejector_mass_kg; no ejector_time_sec
    # field exists. "Time"/"Time sec" labels under Ejector simply don't
    # resolve and the synthesizer skips them. The plan's Task 4 originally
    # listed ejector_time_sec; removed by Task 6's schema-side drift guard
    # which correctly flagged the missing field.
    ("weight",      "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("weight kg",   "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("mass",        "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("mass kg",     "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("thrust",      "Ejector",   "missile_propulsion"): "ejector_thrust",

    # ============================================================
    # radar_power_rf
    # ============================================================
    ("frequency",           None, "radar_power_rf"): "nominal_rf_mhz",
    ("frequency mhz",       None, "radar_power_rf"): "nominal_rf_mhz",
    ("operating frequency", None, "radar_power_rf"): "nominal_rf_mhz",
    ("carrier frequency",   None, "radar_power_rf"): "nominal_rf_mhz",
    ("rf",                  None, "radar_power_rf"): "nominal_rf_mhz",

    ("peak power",          None, "radar_power_rf"): "tx_peak_power_kw",
    ("transmitter power",   None, "radar_power_rf"): "tx_peak_power_kw",
    ("tx power",            None, "radar_power_rf"): "tx_peak_power_kw",

    ("erp",                          None, "radar_power_rf"): "erp_dbw",
    ("effective radiated power",     None, "radar_power_rf"): "erp_dbw",

    # ============================================================
    # radar_timing
    # ============================================================
    ("pri",                       None, "radar_timing"): "nominal_pri_usec",
    ("pulse repetition interval", None, "radar_timing"): "nominal_pri_usec",
    ("pulse interval",            None, "radar_timing"): "nominal_pri_usec",

    ("pw",                        None, "radar_timing"): "nominal_pd_usec",
    ("pulse width",               None, "radar_timing"): "nominal_pd_usec",
    ("pulse duration",            None, "radar_timing"): "nominal_pd_usec",

    ("scan period",  None, "radar_timing"): "scan_period_sec",
    ("scan time",    None, "radar_timing"): "scan_period_sec",
    ("rotation period", None, "radar_timing"): "scan_period_sec",

    ("dwell",        None, "radar_timing"): "dwell_time",
    ("dwell time",   None, "radar_timing"): "dwell_time",

    # ============================================================
    # radar_antenna
    # ============================================================
    ("antenna gain",     None, "radar_antenna"): "gain_dbi",

    ("antenna width",       None, "radar_antenna"): "antenna_dim_az_m",
    ("azimuth aperture",    None, "radar_antenna"): "antenna_dim_az_m",
    ("antenna height",      None, "radar_antenna"): "antenna_dim_el_m",
    ("elevation aperture",  None, "radar_antenna"): "antenna_dim_el_m",

    ("azimuth beamwidth",   None, "radar_antenna"): "beamwidth_az_deg",
    ("elevation beamwidth", None, "radar_antenna"): "beamwidth_el_deg",
    ("elevation coverage",  None, "radar_antenna"): "coverage_limits_el_deg",

    # ============================================================
    # radar_modulation
    # ============================================================
    ("chirp bandwidth",     None, "radar_modulation"): "frequency_excursion_mhz",
    ("frequency excursion", None, "radar_modulation"): "frequency_excursion_mhz",
    ("sweep width",         None, "radar_modulation"): "frequency_excursion_mhz",

    ("code length",      None, "radar_modulation"): "num_bits_in_code",
    ("chips",            None, "radar_modulation"): "num_bits_in_code",
    ("bits",             None, "radar_modulation"): "num_bits_in_code",

    ("pulses per dwell", None, "radar_modulation"): "pulses_per_dwell",
}

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
    "time_usec": {
        "usec": 1.0,
        "us": 1.0,
        "µs": 1.0,
        "μs": 1.0,
        "ms": 1000.0,
        "sec": 1000000.0,
        "s": 1000000.0,
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
        # dBm is converted with an offset (dBW = dBm - 30) in coerce_value.
        # The 1.0 factor keeps it discoverable as a valid unit token.
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
    "_usec": "time_usec",
    "_sec": "time_sec",
    "_mps": "velocity_mps",
    "_mhz": "frequency_mhz",
    "_dbi": "gain_dbi",
    "_kw": "power_kw",
    "_dbw": "power_dbw",
    "_deg": "angle_deg",
}

# ----------------------------------------------------------------------
# Mechanism A1 (spec §5.1): identity-row label patterns + cross-entity
# refs + canonical-name priority. Used by extract_table_overlay() in
# _table_facts.py to classify column-0 cells in column-major variants
# tables.
# ----------------------------------------------------------------------

# Row labels that mean "this row holds an identifier for the entity in
# the column above." Bare "variant" and "designation" are DELIBERATELY
# EXCLUDED for v1 — they create false positives via cross-entity-ref
# rows (e.g., "Fan Song Variant" would match "variant", which is wrong).
MISSILE_IDENTITY_LABELS: tuple[str, ...] = (
    "missile type",
    "missile variant",
    "industry designation",
    "military designation",
    "nato designation",
    "system designation",
)

RADAR_IDENTITY_LABELS: tuple[str, ...] = (
    "radar variant",
    "radar designation",
    "radar type",
)

# Cross-entity reference rows: row labels that name a SIBLING entity
# type. When seen in a missile-context table, the row's cells are not
# missile aliases — they're radar aliases attached to the same column's
# missile via a relationship hint. Emitted as CrossEntityHint, not
# folded into the missile alias cluster.
#
# Classification order (enforced in _classify_identity_row):
#   1. Cross-entity-ref check FIRST
#   2. Identity-label check SECOND
#   3. Spec-row check (label-to-schema-field alias) THIRD
#   4. Otherwise: ignored
CROSS_ENTITY_REF_PATTERNS: dict[str, str] = {
    "fan song variant": "RADAR_SYSTEM",
    "spoon rest variant": "RADAR_SYSTEM",
}

# Canonical-name priority per entity type. When a column has aliases
# from multiple identity rows, pick the FIRST priority label that's
# present. Every entry in MISSILE_/RADAR_IDENTITY_LABELS must appear
# (case-insensitive substring) somewhere in this priority tuple,
# OTHERWISE the drift guard
# test_identity_labels_have_canonical_priority_coverage will fail.
# Order is least-specific-last so "Missile Type" wins over "Missile
# Variant" when both are present in a cluster.
CANONICAL_PRIORITY: dict[str, tuple[str, ...]] = {
    "MISSILE_SYSTEM": (
        "Missile Type",
        "Industry Designation",
        "Military Designation",
        "NATO Designation",
        "System Designation",   # fallback for docs that use this label only
        "Missile Variant",      # fallback for docs that use this label only
    ),
    "RADAR_SYSTEM": (
        "Radar Variant",
        "Radar Designation",
        "Radar Type",
    ),
}
