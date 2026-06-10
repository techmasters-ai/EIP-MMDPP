"""Value-grounded field→chunk matching (pure, no DB).

The docling-graph ``field_provenance`` rows are scope-based: their
``self_refs``/``element_uid`` point at the extraction BATCH SCOPE chunks, not
the chunk where the extracted value physically appears (the chunk-9 "50 km"
diagnosis). The extracted VALUES are correct; only the source-CHUNK attribution
is wrong.

This module re-derives the CORRECT source chunk directly from the value: for a
numeric field whose name is unit-suffixed (``max_intercept_km``, ``erp_dbw``,
``nominal_rf_mhz``, ...), find the chunk(s) whose TEXT physically contains that
value next to (or in the same chunk as) its unit. That is the honest answer to
"which chunk did this field's value come from".

It is the shared home for the matcher originally validated in
``scripts/build_extracted_from_groundtruth.py`` (the chunk-9 ground-truth
builder). Both that script and the worker merge phase
(``app/workers/pipeline.py::_resolve_field_evidence_chunk_ids``) import from
here so the matching logic can't diverge between the ground-truth tooling and
the production lineage stamp.

Pure: no DB, no I/O. Callers supply the chunk text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Unit synonyms keyed by the field-name suffix. Schema fields are unit-suffixed:
# max_intercept_km, erp_dbw, nominal_rf_mhz, max_launch_angle_deg, total_mass_kg.
# Longest suffix wins (see ``units_for``) so ``_kmh`` doesn't match as ``_km``.
SUFFIX_UNITS: dict[str, list[str]] = {
    "km": ["km", "км"], "mhz": ["mhz"], "ghz": ["ghz"], "khz": ["khz"],
    "kw": ["kw"], "mw": ["mw"], "dbw": ["dbw"], "dbm": ["dbm"],
    "deg": ["deg", "°", "degree", "degrees", "град"], "kg": ["kg", "кг"],
    "mm": ["mm"], "cm": ["cm"], "kmh": ["km/h", "kph"], "mps": ["m/s"],
    "mach": ["mach", "m="], "kn": ["kn", "knot"], "us": ["µs", "us", "microsec"],
    "ms": ["ms"], "s": ["s", "sec", "second", "seconds"], "m": ["m", "metre", "meter"],
}

# Match tiers returned by ``value_in_chunk`` (highest→lowest confidence).
ADJACENT = "adjacent"      # number immediately followed by a unit ("50 km") — prose
SAME_CHUNK = "same_chunk"  # number present AND a unit elsewhere in the chunk — table


def nfc(s: str | None) -> str:
    """NFC-normalize + casefold for unit/value matching across scripts (Cyrillic,
    micro sign, degree sign) and case. Empty/None → empty string."""
    return unicodedata.normalize("NFC", s or "").casefold()


def units_for(field: str) -> list[str]:
    """Unit synonyms for a unit-suffixed field name (``max_intercept_km`` → km
    variants). Longest suffix wins so ``muzzle_velocity_kmh`` resolves to km/h,
    not km. Returns ``[]`` for unitless / unrecognized fields (caller skips them)."""
    f = (field or "").lower()
    for suf in sorted(SUFFIX_UNITS, key=len, reverse=True):
        if f.endswith("_" + suf):
            return SUFFIX_UNITS[suf]
    return []


def num_variants(value) -> set[str]:
    """String renderings of a numeric value to search for in chunk text.

    Covers the integer form (``50``), the ``%g`` form (``50``/``2391.5``), and
    the raw ``str(value)`` — so ``50.0`` matches the document's ``50``, and
    ``2391.0`` matches ``2391``. Non-numeric values yield an empty set (caller
    falls through to the anchor logic)."""
    out: set[str] = set()
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return out
    if fv == int(fv):
        out.add(str(int(fv)))
    out.add(f"{fv:g}")
    out.add(str(value))
    return {s for s in out if s}


def unit_token_regex(unit_nfc: str) -> str:
    """Token-bounded regex fragment for one already-``nfc()``-folded unit synonym.

    Leading guard (only when the synonym starts with a word char): no LETTER
    immediately before — digit prefixes stay legal so "50km" still carries the
    unit. Trailing guard (only when it ends with a word char): no word char
    immediately after — "50 sites" no longer matches unit "s", "9 months" no
    longer matches unit "m". Unicode-aware so Cyrillic synonyms get the same
    discipline. Symbol-edged synonyms ("°", "m/s", "m=") skip the guard on
    their symbol side (the symbol is its own boundary).
    """
    pat = re.escape(unit_nfc)
    if re.match(r"\w", unit_nfc):
        pat = r"(?<![^\W\d_])" + pat  # not preceded by a unicode letter
    if re.search(r"\w$", unit_nfc):
        pat = pat + r"(?!\w)"  # not followed by a unicode word char
    return pat


def has_unit_token(text_nfc: str, units: Iterable[str]) -> bool:
    """True iff any unit synonym appears as a bounded token in the folded text.

    SHARED by the SAME_CHUNK grounding tier and the G1 selection gate — one
    matcher, so label semantics and gate semantics can never drift.
    """
    return any(re.search(unit_token_regex(nfc(u)), text_nfc) for u in units)


def value_in_chunk(num_strs: set[str], units: Iterable[str], text_nfc: str) -> str | None:
    """Match a numeric value against an NFC-normalized chunk text. Two tiers:

      ``ADJACENT``   — a value string immediately followed by a unit ("50 km",
                       "50-km", "50(km") → prose, high confidence.
      ``SAME_CHUNK`` — a value string present AND a unit appearing anywhere in the
                       chunk → a table with a detached unit header
                       ("Weight, 7 = 2391 … = kg"). Medium confidence; requires
                       >=2 digits in the number to avoid coincidental single-digit
                       hits.

    Returns the tier string or ``None``. ``num_strs`` is the value's string
    renderings (see ``num_variants``); ``units`` the unit synonyms (see
    ``units_for``); ``text_nfc`` the ALREADY ``nfc()``-normalized chunk text.
    """
    units_nfc = [nfc(u) for u in units]
    for ns in num_strs:
        for u in units_nfc:
            if re.search(r"(?<![\d.])" + re.escape(ns) + r"\s*[\(\-–]?\s*" + unit_token_regex(u), text_nfc):
                return ADJACENT
    if has_unit_token(text_nfc, units_nfc):
        for ns in num_strs:
            # require >=2 digits for the table tier to avoid coincidental single-digit hits
            if len(re.sub(r"\D", "", ns)) >= 2 and re.search(r"(?<!\d)" + re.escape(ns) + r"(?!\d)(?!\.\d)", text_nfc):
                return SAME_CHUNK
    return None


def match_value_to_chunks(field_name: str, value, chunk_text_by_id: dict[str, str]) -> list[str]:
    """Public convenience: resolve a numeric, unit-suffixed field VALUE to the
    chunk id(s) whose text physically contains it.

    Returns the matching chunk ids ordered ADJACENT-first (high-confidence prose
    hits before detached-unit table hits), preserving ``chunk_text_by_id``
    iteration order within each tier. Returns ``[]`` when the field is unitless,
    the value is non-numeric, or no chunk text contains the value — the caller
    then falls through to its existing (anchor-based) resolution unchanged.

    Never fans out: only chunks that actually contain the value are returned.
    """
    units = units_for(field_name)
    if not units:
        return []
    nums = num_variants(value)
    if not nums:
        return []
    adjacent: list[str] = []
    same_chunk: list[str] = []
    for cid, text in (chunk_text_by_id or {}).items():
        tier = value_in_chunk(nums, units, nfc(text))
        if tier == ADJACENT:
            adjacent.append(cid)
        elif tier == SAME_CHUNK:
            same_chunk.append(cid)
    return adjacent + same_chunk
