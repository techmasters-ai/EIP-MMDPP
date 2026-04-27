"""Shared numeric-evidence helpers used by:

- provenance.build_auto_field_evidence (post-extraction evidence-row builder)
- evidence_gate._clear_unsupported_radar_properties (numeric-field
  clearing in the radar postprocessor — refactored per spec §4.8 to
  preserve numeric values that appear in batch evidence text)

Both consumers must use the same predicate so a value the resolver
treats as "supported" isn't simultaneously nulled by the postprocessor.

Spec §4.8.
"""
from __future__ import annotations

import re
from typing import Any

_WS_NORM = re.compile(r"\s+")


# Field-name suffix → list of human-readable unit candidates the
# author may have written FOR THE SAME PHYSICAL MAGNITUDE. NEVER
# include cross-magnitude prefixes here (e.g. "GHz" is NOT a hint for
# "_mhz" — 3000 GHz != 3000 MHz; including it would silently accept
# physically wrong values). The pre-refactor provenance.py:365 has
# this exact bug — the extraction MUST drop "GHz" from "_mhz" rather
# than preserve it. Same-magnitude case-variant hints are fine.
_UNIT_HINTS_BY_SUFFIX: dict[str, list[str]] = {
    "_dbi": ["dBi", "dB"],
    "_dbw": ["dBW", "dB"],
    "_mhz": ["MHz", "Mhz", "mhz"],   # was ["MHz", "GHz"] — DROP "GHz" (cross-magnitude bug)
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


def normalize_text(text: str) -> str:
    """Whitespace-collapsed casefold for fuzzy substring matching."""
    return _WS_NORM.sub(" ", text or "").strip().casefold()


def _field_unit_suffix(field_name: str) -> str:
    """Return the longest known unit suffix on a field name, or ''."""
    for suffix in sorted(_UNIT_HINTS_BY_SUFFIX, key=len, reverse=True):
        if field_name.endswith(suffix):
            return suffix
    return ""


def value_match_candidates(value: Any, field_name: str) -> list[str]:
    """Generate likely string forms of a field value for substring matching.

    Numeric values get whole-number, decimal, and same-unit-suffix variants
    derived from the field name's suffix convention (e.g. ``gain_dbi`` →
    "35", "35.0", "35 dBi", "35dBi"). String values pass through as-is
    after stripping. Booleans return [] (not useful for substring match).

    IMPORTANT: only same-unit variants are generated — cross-unit
    alternates like "3000 GHz" for value 3000 in field nominal_rf_mhz are
    NEVER emitted, since matching them would silently accept physically
    wrong values. If `_UNIT_HINTS_BY_SUFFIX[<suffix>]` contains a cross-
    unit entry (e.g. "GHz" listed under "_mhz"), audit and remove it
    before this helper is wired into the evidence gate. The contract test
    `test_value_match_candidates_for_int_with_mhz_suffix` enforces this.
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
        units = _UNIT_HINTS_BY_SUFFIX.get(_field_unit_suffix(field_name))
        if units:
            base = forms[-1]
            forms.extend(f"{base} {u}" for u in units)
            forms.extend(f"{base}{u}" for u in units)
        return forms
    return [str(value)]


def value_is_supported_by_text(
    value: Any, field_name: str, evidence_text: str,
) -> bool:
    """Return True iff *value*'s stringified form appears in *evidence_text*.

    "Stringified form" includes the field's expected unit suffix appended
    to the value (e.g. "35 dBi" for value 35.0 in field gain_dbi). It does
    NOT include cross-unit converted forms — "35 dBi" is generated, but
    "37.15 dBd" is NOT. Same-magnitude-different-unit matches like
    "3000 GHz" for value 3000 in field nominal_rf_mhz are explicitly
    rejected (they would be physically wrong: 3000 GHz != 3000 MHz).

    Cross-unit conversion (1.5 tonnes <-> 1500 kg, 43 km <-> 43000 m, etc.)
    is OUT OF SCOPE for Session 1. If a doc states a value in a non-canonical
    unit and the LLM doesn't normalize, this predicate returns False and the
    caller will null the value. Tracked as Session 2 follow-up if false-
    negatives become a real problem in production.

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
