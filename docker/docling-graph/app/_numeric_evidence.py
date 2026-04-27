"""Shared numeric-evidence helpers used by:

- provenance.build_auto_field_evidence (post-extraction evidence-row builder)
- evidence_gate._clear_unsupported_radar_properties (numeric-field
  clearing in the radar postprocessor — refactored per spec §4.8 to
  preserve numeric values that appear in batch evidence text)
- evidence_gate._clear_unsupported_missile_properties (missile equivalent)

All consumers must use the same predicate so a value the resolver
treats as "supported" isn't simultaneously nulled by the postprocessor.

Spec §4.8.

Cross-unit conversion (Option A from cross-unit-conversion-followup-todo.md):
``value_match_candidates`` now emits cross-magnitude candidates derived
from ``_UNIT_CONVERSIONS_BY_SUFFIX``. For value 10000 in field
``nominal_rf_mhz``, candidates include "10 GHz" and "10000000 kHz" in
addition to "10000 MHz". The conversion table only contains scale-based
conversions; logarithmic-domain conversions (dBW <-> dBm, dBi <-> dBd)
remain out of scope because they require offsets, not scale factors.
Mach-number <-> m/s also stays out of scope (depends on altitude/temp).
"""
from __future__ import annotations

import re
from typing import Any

_WS_NORM = re.compile(r"\s+")


# Field-name suffix → list of human-readable unit candidates the
# author may have written FOR THE SAME PHYSICAL MAGNITUDE. NEVER
# include cross-magnitude prefixes here (e.g. "GHz" is NOT a hint for
# "_mhz" — 3000 GHz != 3000 MHz; including it would silently accept
# physically wrong values). Cross-magnitude conversions go in
# _UNIT_CONVERSIONS_BY_SUFFIX (below) where each entry carries its
# scale factor.
_UNIT_HINTS_BY_SUFFIX: dict[str, list[str]] = {
    "_dbi": ["dBi", "dB"],
    "_dbw": ["dBW", "dB"],
    "_mhz": ["MHz", "Mhz", "mhz"],
    "_khz": ["kHz", "khz"],
    "_usec": ["μs", "us", "microseconds"],
    "_sec": ["s", "seconds"],
    "_kw": ["kW"],
    "_mw": ["MW"],
    "_km": ["km", "kilometers"],
    "_m": ["m", "meters"],
    "_kg": ["kg", "kilograms"],
    "_mps": ["m/s", "mps"],
    "_deg": ["°", "deg", "degrees"],
}


# Field-name suffix → list of (other_unit_text, scale_factor) tuples for
# CROSS-MAGNITUDE conversions. Reading guide:
#   ``scale_factor`` is the size of one ``other_unit_text`` in canonical
#   units. To compute the candidate value: ``canonical_value / scale``.
#
# Example for "_mhz" (canonical = MHz):
#   ("GHz", 1000.0)   means 1 GHz = 1000 MHz, so 10000 MHz / 1000 = 10 GHz
#   ("kHz", 0.001)    means 1 kHz = 0.001 MHz, so 10000 MHz / 0.001 = 10000000 kHz
#
# Excluded on purpose:
# - dBi <-> dBd, dBW <-> dBm: logarithmic offsets (not scale factors)
# - Mach <-> m/s: depends on altitude / atmospheric state
# - lbs <-> kg: missile postprocessor's _mechanically_supported_missile_fields()
#   regex already covers this case-specifically; not generalized here yet
_UNIT_CONVERSIONS_BY_SUFFIX: dict[str, list[tuple[str, float]]] = {
    # Frequency
    "_mhz": [
        ("GHz", 1000.0),
        ("Ghz", 1000.0),
        ("ghz", 1000.0),
        ("kHz", 0.001),
    ],
    "_khz": [
        ("MHz", 1000.0),
        ("Hz", 0.001),
    ],
    # Power
    "_kw": [
        ("MW", 1000.0),
        ("megawatts", 1000.0),
        ("megawatt", 1000.0),
        ("W", 0.001),
        ("watts", 0.001),
    ],
    "_mw": [
        ("kW", 0.001),
        ("GW", 1000.0),
    ],
    # Distance / length
    "_km": [
        ("m", 0.001),
        ("meters", 0.001),
        ("metres", 0.001),
        ("nm", 1.852),    # nautical miles → km (1 nm = 1.852 km)
        ("nmi", 1.852),
    ],
    "_m": [
        ("km", 1000.0),
        ("kilometers", 1000.0),
        ("kilometres", 1000.0),
        ("cm", 0.01),
        ("mm", 0.001),
    ],
    # Mass
    "_kg": [
        ("tonnes", 1000.0),
        ("tonne", 1000.0),
        ("metric tons", 1000.0),
        ("metric ton", 1000.0),
        ("g", 0.001),
        ("grams", 0.001),
        # NOTE: lbs <-> kg is handled case-specifically by
        # _mechanically_supported_missile_fields() in evidence_gate.py;
        # not duplicated here.
    ],
    # Speed
    "_mps": [
        ("km/s", 1000.0),
        ("km/h", 1.0 / 3.6),       # 1 km/h = 0.2778 m/s; canonical / scale = km/h value
        ("kph", 1.0 / 3.6),
        ("mph", 0.44704),           # 1 mph = 0.44704 m/s
    ],
    # Time
    "_sec": [
        ("ms", 0.001),
        ("milliseconds", 0.001),
        ("min", 60.0),
        ("minutes", 60.0),
    ],
    "_usec": [
        ("ms", 1000.0),
        ("milliseconds", 1000.0),
        ("ns", 0.001),
        ("nanoseconds", 0.001),
    ],
    # Angle
    "_deg": [
        ("rad", 180.0 / 3.141592653589793),
        ("radians", 180.0 / 3.141592653589793),
    ],
}


def normalize_text(text: str) -> str:
    """Whitespace-collapsed casefold for fuzzy substring matching."""
    return _WS_NORM.sub(" ", text or "").strip().casefold()


def _field_unit_suffix(field_name: str) -> str:
    """Return the longest known unit suffix on a field name, or ''.

    Searches both _UNIT_HINTS_BY_SUFFIX and _UNIT_CONVERSIONS_BY_SUFFIX.
    """
    known = set(_UNIT_HINTS_BY_SUFFIX) | set(_UNIT_CONVERSIONS_BY_SUFFIX)
    for suffix in sorted(known, key=len, reverse=True):
        if field_name.endswith(suffix):
            return suffix
    return ""


def _format_converted(v: float) -> str:
    """Format a converted numeric value without scientific notation or trailing zeros.

    Plain ``str(float)`` introduces "1.0" / "10000000.0" forms; ``:g`` switches
    to scientific notation for large magnitudes. ``:.10g`` keeps 10 significant
    digits before falling back to scientific, which covers any realistic radar
    or missile field magnitude. Whole numbers are stringified as ints.
    """
    if v == int(v):
        return str(int(v))
    return f"{v:.10g}"


def value_match_candidates(value: Any, field_name: str) -> list[str]:
    """Generate likely string forms of a field value for substring matching.

    Numeric values get:
    1. Whole-number and decimal forms ("35", "35.0").
    2. Same-unit-suffix variants from _UNIT_HINTS_BY_SUFFIX (e.g. "35 dBi",
       "35dBi" — case variants of the canonical unit).
    3. Cross-unit converted forms from _UNIT_CONVERSIONS_BY_SUFFIX (e.g.
       value 10000 in nominal_rf_mhz → "10 GHz", "10000000 kHz").

    String values pass through as-is after stripping. Booleans return [].

    Cross-unit conversions only emit values consistent with the canonical
    field. For value 10000 in nominal_rf_mhz:
    - "10 GHz"      — correct (10000 MHz = 10 GHz)
    - "10000000 kHz" — correct (10000 MHz = 10000000 kHz)
    - "10000 GHz"   — NEVER emitted (would be 3 orders of magnitude wrong)
    - "10000 kHz"   — NEVER emitted (also wrong)
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (int, float)):
        forms: list[str] = [str(value)]
        if isinstance(value, float) and value == int(value):
            forms.append(str(int(value)))
        suffix = _field_unit_suffix(field_name)

        # Same-magnitude unit hints (case variants of canonical unit).
        units = _UNIT_HINTS_BY_SUFFIX.get(suffix)
        if units:
            base = forms[-1]
            forms.extend(f"{base} {u}" for u in units)
            forms.extend(f"{base}{u}" for u in units)

        # Cross-magnitude unit conversions (with scale factor).
        conversions = _UNIT_CONVERSIONS_BY_SUFFIX.get(suffix)
        if conversions:
            for unit_text, scale in conversions:
                converted = float(value) / scale
                converted_str = _format_converted(converted)
                forms.append(f"{converted_str} {unit_text}")
                forms.append(f"{converted_str}{unit_text}")

        return forms
    return [str(value)]


def value_is_supported_by_text(
    value: Any, field_name: str, evidence_text: str,
) -> bool:
    """Return True iff *value*'s stringified form appears in *evidence_text*.

    "Stringified form" includes:
    1. The bare numeric form ("35", "35.0").
    2. The field's expected unit suffix appended ("35 dBi" for gain_dbi).
    3. Cross-unit converted forms with the appropriate other-unit text
       ("10 GHz" for value 10000 in nominal_rf_mhz; "1.5 tonnes" for
       value 1500 in total_mass_kg).

    Cross-unit conversion is governed by _UNIT_CONVERSIONS_BY_SUFFIX and
    only emits values consistent with the canonical field. Same-magnitude-
    different-unit matches like "3000 GHz" for value 3000 in field
    nominal_rf_mhz are NEVER generated (would be physically wrong:
    3000 GHz != 3000 MHz).

    Logarithmic conversions (dBW <-> dBm, dBi <-> dBd) and Mach <-> m/s
    remain out of scope — see _UNIT_CONVERSIONS_BY_SUFFIX docstring.

    Whitespace is collapsed and case is folded before comparison.
    None / empty-string values are vacuously supported — the caller's
    null-check happens upstream of this predicate.
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    candidates = value_match_candidates(value, field_name)
    if not candidates:
        return False
    et_norm = normalize_text(evidence_text or "")
    if not et_norm:
        return False
    return any(normalize_text(c) in et_norm for c in candidates)
