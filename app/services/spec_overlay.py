"""Step 4: spec fact overlay for prose key-value blocks.

Detects labeled spec fragments in docling document texts and normalizes
them into structured facts. Generic — no equipment-specific names anywhere.

Two consumers (wired in main.py + evidence_gate.py):
  1. **Pre-LLM:** emit a SPEC_FACTS preamble into the chunk that contains
     the relevant entity so the LLM sees pre-parsed evidence.
  2. **Post-LLM:** feed evidence_gate so deterministic fields can be
     populated even when the LLM emitted null (especially for prose-only
     documents like Dvina where the chunker fragments the data box).

Patterns covered (all generic):
  * `LABEL: VALUE UNIT`                — `Length: 35 feet`
  * `LABEL: VALUE/VALUE UNIT/UNIT`     — `Max/min effective range: 18 miles/5 miles`
  * Multi-line `LABEL:` then `VALUE UNIT` on the next text item — bullet data box
  * `LABEL VALUE UNIT`                 — `Maximum speed Mach 3` (space-only)
  * `LAUNCHED ... AT N DEGREES` / `MAX LAUNCH ANGLE: N°` — anchored angle

Unit conversion:
  * Imperial → metric where explicit: miles → km, feet → km/m, lbs → kg,
    inches → m.
  * "Mach N" preserved as raw label (no MPS conversion attempt — Mach
    depends on altitude/atmosphere; not a clean mechanical conversion).
  * Degrees: passed through unchanged.

Unit safety: only converts when the unit is EXPLICIT in the matched
text. Unitless values produce a `SpecFact` with `value_raw` populated
and `value_metric_*` left None; consumers may infer convention from
surrounding context (e.g. a UNIT_HINT preamble) but this module never
infers a unit on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SpecFact:
    """One structured spec fact extracted from prose.

    Generic shape — `label_canonical` is the normalized field name (e.g.
    "max_range", "max_altitude", "launch_angle", "length"), independent
    of which extraction pass will consume it.
    """
    label_canonical: str
    value_raw: str
    unit_raw: str | None
    value_metric_m: float | None = None     # populated when label is a length-class
    value_metric_km: float | None = None    # populated when label is a range/altitude-class
    value_metric_kg: float | None = None    # populated when label is a mass-class
    raw_phrase: str = ""
    source_text_idx: int = -1


# ---------------------------------------------------------------------------
# Unit conversions — only triggered when the unit string is explicit.
# Generic — no equipment names.
# ---------------------------------------------------------------------------
_KM_PER_MILE = 1.609344
_M_PER_FOOT = 0.3048
_M_PER_INCH = 0.0254
_KG_PER_LB = 0.453592


# ---------------------------------------------------------------------------
# Label canonicalization. Maps free-form labels to canonical field stems.
# All comparisons are lowercase and whitespace-collapsed. Order matters —
# longer/more-specific labels MUST come first so "max range" wins over
# "range" prefix matching.
# ---------------------------------------------------------------------------
_LABEL_CANONICALS: tuple[tuple[str, str], ...] = (
    # paired/extreme labels (parsed via paired-syntax handler).
    # Longest/most-specific forms come first so the second-pass
    # `endswith(prefix)` fallback in _canonicalize_label cannot match
    # the trailing "<x> effective range" half of a long paired phrase
    # like "Maximum/minimum effective range" and return a single-side
    # canonical (which would emit one fact instead of a max/min pair).
    ("maximum/minimum effective range", "_paired_range"),
    ("maximum/minimum effective altitude", "_paired_altitude"),
    ("maximum/minimum range", "_paired_range"),
    ("maximum/minimum altitude", "_paired_altitude"),
    ("max/min effective range", "_paired_range"),
    ("max/min effective altitude", "_paired_altitude"),
    ("max/min range", "_paired_range"),
    ("max/min altitude", "_paired_altitude"),
    # range
    ("maximum effective range", "max_range"),
    ("max effective range", "max_range"),
    ("maximum range", "max_range"),
    ("max range", "max_range"),
    ("minimum effective range", "min_range"),
    ("min effective range", "min_range"),
    ("minimum range", "min_range"),
    ("min range", "min_range"),
    # altitude
    ("maximum effective altitude", "max_altitude"),
    ("max effective altitude", "max_altitude"),
    ("maximum altitude", "max_altitude"),
    ("max altitude", "max_altitude"),
    ("max alt", "max_altitude"),
    ("ceiling", "max_altitude"),
    ("minimum altitude", "min_altitude"),
    ("min altitude", "min_altitude"),
    ("min alt", "min_altitude"),
    # length / size
    ("body length", "length"),
    ("overall length", "length"),
    ("missile length", "length"),
    ("length", "length"),
    ("booster diameter", "diameter"),
    ("body diameter", "diameter"),
    ("diameter", "diameter"),
    # mass
    ("launch weight", "weight"),
    ("total weight", "weight"),
    ("weight", "weight"),
    # speed (Mach preserved raw)
    ("maximum speed", "max_speed"),
    ("max speed", "max_speed"),
    # launch angle
    ("max launch angle", "launch_angle"),
    ("maximum launch angle", "launch_angle"),
    ("launch angle", "launch_angle"),
    ("launch elevation", "launch_angle"),
)


def _canonicalize_label(label: str) -> str | None:
    """Return the canonical label key for a free-form label string, or
    None if the label doesn't match any known canonical."""
    lowered = " ".join(label.lower().strip().split())
    for prefix, canonical in _LABEL_CANONICALS:
        if lowered == prefix or lowered.startswith(prefix + " ") or lowered.endswith(" " + prefix):
            return canonical
    # try exact match alone too
    for prefix, canonical in _LABEL_CANONICALS:
        if lowered.endswith(prefix):
            return canonical
    return None


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Single label:value with optional unit (case-insensitive). Value captures
# digits + commas + decimals; can be alphanumeric like "Mach 3".
_LABEL_VALUE_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z /]*?)\s*:\s*(?P<value>.+?)$",
    re.IGNORECASE,
)

# Paired value with units like "18 miles/5 miles" or "82,000 feet/1,500 feet".
# Number portion requires a leading digit (`\d[\d,]*`) so comma-only captures
# like "," or ",,," — which `replace(",", "")` would empty out — cannot reach
# float().
_PAIRED_NUM_UNIT_RE = re.compile(
    r"(?P<n1>\d[\d,]*(?:\.\d+)?)\s*(?P<u1>miles?|feet|foot|ft|kilometers?|km|meters?|m|pounds?|lbs?|kilograms?|kg|inches|in)\s*/\s*"
    r"(?P<n2>\d[\d,]*(?:\.\d+)?)\s*(?P<u2>miles?|feet|foot|ft|kilometers?|km|meters?|m|pounds?|lbs?|kilograms?|kg|inches|in)",
    re.IGNORECASE,
)

# Single number + unit. Leading-digit anchor as above.
_NUM_UNIT_RE = re.compile(
    r"(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<u>miles?|feet|foot|ft|kilometers?|km|meters?|m|pounds?|lbs?|kilograms?|kg|inches|in|degrees?|°)",
    re.IGNORECASE,
)

# Mach speed — preserve raw
_MACH_RE = re.compile(r"\bMach\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)

# Launch-angle prose anchored on "launch"/"launched"/"elevation".
_LAUNCH_ANGLE_PROSE_RE = re.compile(
    r"\blaunched?\s+(?:the\s+)?(?:missile\s+)?at\s+(?P<v>\d+(?:\.\d+)?)\s*(?P<u>degrees?|°)",
    re.IGNORECASE,
)


def _convert_to_metric(value: float, unit: str) -> dict[str, float]:
    """Return a dict with the appropriate value_metric_* fields filled in
    based on `unit`. Returns empty dict for non-convertible units."""
    u = unit.lower().rstrip(".")
    out: dict[str, float] = {}
    if u in ("mile", "miles"):
        out["value_metric_km"] = value * _KM_PER_MILE
    elif u in ("foot", "feet", "ft"):
        out["value_metric_m"] = value * _M_PER_FOOT
        out["value_metric_km"] = (value * _M_PER_FOOT) / 1000.0
    elif u in ("inch", "inches", "in"):
        out["value_metric_m"] = value * _M_PER_INCH
    elif u in ("pound", "pounds", "lb", "lbs"):
        out["value_metric_kg"] = value * _KG_PER_LB
    elif u in ("kilometer", "kilometers", "km"):
        out["value_metric_km"] = value
    elif u in ("meter", "meters", "m"):
        out["value_metric_m"] = value
        out["value_metric_km"] = value / 1000.0
    elif u in ("kilogram", "kilograms", "kg"):
        out["value_metric_kg"] = value
    return out


def _parse_paired_value(canonical_pair: str, value_str: str, raw_phrase: str, idx: int) -> list[SpecFact]:
    """Parse `<n1> <u1>/<n2> <u2>` syntax. canonical_pair is `_paired_range`
    or `_paired_altitude`. Returns two SpecFacts (max + min)."""
    m = _PAIRED_NUM_UNIT_RE.search(value_str)
    if not m:
        return []
    n1_raw = m.group("n1")
    n1 = float(n1_raw.replace(",", ""))
    u1 = m.group("u1")
    n2_raw = m.group("n2")
    n2 = float(n2_raw.replace(",", ""))
    u2 = m.group("u2")
    if canonical_pair == "_paired_range":
        max_label, min_label = "max_range", "min_range"
    elif canonical_pair == "_paired_altitude":
        max_label, min_label = "max_altitude", "min_altitude"
    else:
        return []
    out: list[SpecFact] = []
    out.append(SpecFact(
        label_canonical=max_label, value_raw=_format_value_raw(n1_raw), unit_raw=u1,
        raw_phrase=raw_phrase, source_text_idx=idx,
        **_convert_to_metric(n1, u1),
    ))
    out.append(SpecFact(
        label_canonical=min_label, value_raw=_format_value_raw(n2_raw), unit_raw=u2,
        raw_phrase=raw_phrase, source_text_idx=idx,
        **_convert_to_metric(n2, u2),
    ))
    return out


def _format_value_raw(raw_str: str) -> str:
    """Preserve the raw numeric token verbatim (no trailing .0 added)."""
    return raw_str.replace(",", "")


def _parse_single_value(canonical: str, value_str: str, raw_phrase: str, idx: int) -> list[SpecFact]:
    """Parse `<n> <u>` or `Mach <n>` from a single-value spec line."""
    value_str = value_str.strip()
    # Mach speed — preserve raw, no conversion
    mach_m = _MACH_RE.search(value_str)
    if mach_m and canonical == "max_speed":
        return [SpecFact(
            label_canonical=canonical,
            value_raw=f"Mach {mach_m.group(1)}",
            unit_raw="mach",
            raw_phrase=raw_phrase,
            source_text_idx=idx,
        )]
    # Standard number + unit
    nu_m = _NUM_UNIT_RE.search(value_str)
    if nu_m:
        n_raw = nu_m.group("n")
        n = float(n_raw.replace(",", ""))
        u = nu_m.group("u")
        return [SpecFact(
            label_canonical=canonical,
            value_raw=_format_value_raw(n_raw),
            unit_raw=u,
            raw_phrase=raw_phrase,
            source_text_idx=idx,
            **_convert_to_metric(n, u),
        )]
    return []


def _parse_text_item_pair(
    label_text: str, value_text: str, raw_phrase: str, idx: int,
) -> list[SpecFact]:
    """When label and value are in separate text items (bullet data box):
    apply canonicalization to the label, then parse the value."""
    # Strip trailing colon if present
    label = label_text.rstrip(":").strip()
    canonical = _canonicalize_label(label)
    if canonical is None:
        return []
    if canonical.startswith("_paired_"):
        return _parse_paired_value(canonical, value_text, raw_phrase, idx)
    return _parse_single_value(canonical, value_text, raw_phrase, idx)


def _looks_like_label_only(text: str) -> bool:
    """True if the text item is a bare label (ends with ':' and short)."""
    s = text.strip()
    return s.endswith(":") and len(s) <= 60


def parse_spec_facts_from_evidence_text(evidence_text: str) -> list[SpecFact]:
    """Scan a single concatenated evidence string for known LABEL: VALUE
    pairs. Handles the production case where docling-graph's
    `normalize_evidence_text` collapses newlines so multiple label-value
    pairs appear inline ("LENGTH: 35 FEET BOOSTER DIAMETER: 26 INCHES ...").

    For each known canonical label phrase, scans the entire evidence
    text for occurrences followed by a value (possibly paired) and a
    unit. Generic — no equipment names.

    Returns SpecFacts in document order; ranges already matched by a
    longer phrase are skipped so "max/min effective range" wins over
    "range" without producing duplicate facts.
    """
    if not evidence_text:
        return []
    consumed_ranges: list[tuple[int, int]] = []
    out: list[SpecFact] = []
    # Sort label phrases longest-first so more specific labels win.
    sorted_canonicals = sorted(
        _LABEL_CANONICALS, key=lambda x: -len(x[0]),
    )
    for phrase, canonical in sorted_canonicals:
        # Pattern: phrase + (optional colon) + value(+unit) + optional paired
        # Value portion: paired form OR single num+unit OR Mach form
        # Number portion uses `\d[\d,]*` (leading-digit anchor) to mirror
        # _NUM_UNIT_RE / _PAIRED_NUM_UNIT_RE so comma-only fragments such
        # as ", miles" cannot reach float() inside _parse_single_value /
        # _parse_paired_value.
        value_pattern = (
            r"(?:"
            r"(?P<paired>\d[\d,]*(?:\.\d+)?\s*(?:miles?|feet|foot|ft|kilometers?|km|meters?|m|pounds?|lbs?|kilograms?|kg|inches|in)\s*/\s*\d[\d,]*(?:\.\d+)?\s*(?:miles?|feet|foot|ft|kilometers?|km|meters?|m|pounds?|lbs?|kilograms?|kg|inches|in))"
            r"|(?P<single>\d[\d,]*(?:\.\d+)?\s*(?:miles?|feet|foot|ft|kilometers?|km|meters?|m|pounds?|lbs?|kilograms?|kg|inches|in|degrees?|°))"
            r"|(?P<mach>Mach\s*\d+(?:\.\d+)?)"
            r")"
        )
        # Allow word boundary on either side of the label phrase
        regex = re.compile(
            r"\b" + re.escape(phrase) + r"\s*:?\s*" + value_pattern,
            re.IGNORECASE,
        )
        for m in regex.finditer(evidence_text):
            start, end = m.start(), m.end()
            overlapped = False
            for cs, ce in consumed_ranges:
                if not (end <= cs or start >= ce):
                    overlapped = True
                    break
            if overlapped:
                continue
            consumed_ranges.append((start, end))
            raw_phrase = evidence_text[start:end]
            paired_grp = m.group("paired")
            single_grp = m.group("single")
            mach_grp = m.group("mach")
            if canonical.startswith("_paired_") and paired_grp:
                out.extend(_parse_paired_value(canonical, paired_grp, raw_phrase, -1))
            elif paired_grp and not canonical.startswith("_paired_"):
                # Paired value emitted but label is non-paired (e.g. "MAX RANGE: 18 miles/5 miles" —
                # treat as max only, no min). Take first.
                pm = _PAIRED_NUM_UNIT_RE.search(paired_grp)
                if pm:
                    n_raw = pm.group("n1")
                    n = float(n_raw.replace(",", ""))
                    u = pm.group("u1")
                    out.append(SpecFact(
                        label_canonical=canonical,
                        value_raw=_format_value_raw(n_raw), unit_raw=u,
                        raw_phrase=raw_phrase, source_text_idx=-1,
                        **_convert_to_metric(n, u),
                    ))
            elif single_grp:
                facts = _parse_single_value(canonical, single_grp, raw_phrase, -1)
                out.extend(facts)
            elif mach_grp and canonical == "max_speed":
                mm = _MACH_RE.search(mach_grp)
                if mm:
                    out.append(SpecFact(
                        label_canonical=canonical,
                        value_raw=f"Mach {mm.group(1)}",
                        unit_raw="mach",
                        raw_phrase=raw_phrase, source_text_idx=-1,
                    ))
    return out


def parse_spec_facts_from_texts(texts: list[dict] | list[str]) -> list[SpecFact]:
    """Scan docling document text items for labeled spec fragments.

    Accepts either a list of docling text dicts (with `.text` field) or a
    list of plain strings. Returns a list of `SpecFact` instances with
    metric-converted values where unit was explicit.

    Generic — no equipment-specific logic. Handles three textual shapes:
      1. Single-line: `"Length: 35 feet"`
      2. Bullet pair: `["Length:", "35 feet"]` (two separate text items)
      3. Paired:      `"Max/min effective range: 18 miles/5 miles"`
    Plus the launch-angle prose anchor for cases the chunker would split.
    """
    out: list[SpecFact] = []

    # Normalize input to list of (idx, text_str) tuples.
    items: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        if isinstance(t, dict):
            s = t.get("text") or ""
        else:
            s = str(t)
        if isinstance(s, str):
            items.append((i, s))

    # First pass: bullet pair detection — `LABEL:` followed by `VALUE UNIT`
    # on the next text item. Mark which items are part of a pair so the
    # single-line scan doesn't re-process them.
    consumed: set[int] = set()
    for i in range(len(items) - 1):
        idx_lab, text_lab = items[i]
        idx_val, text_val = items[i + 1]
        if not _looks_like_label_only(text_lab):
            continue
        # Skip if next is also label-only (no value to pair with)
        if _looks_like_label_only(text_val):
            continue
        facts = _parse_text_item_pair(
            text_lab, text_val,
            raw_phrase=f"{text_lab} {text_val}", idx=idx_val,
        )
        if facts:
            out.extend(facts)
            consumed.add(idx_lab)
            consumed.add(idx_val)

    # Second pass: single-line label:value (within one text item).
    for idx, text in items:
        if idx in consumed:
            continue
        # Try label:value form first
        m = _LABEL_VALUE_RE.match(text.strip())
        if m:
            label = m.group("label").strip()
            value = m.group("value").strip()
            canonical = _canonicalize_label(label)
            if canonical is not None:
                if canonical.startswith("_paired_"):
                    facts = _parse_paired_value(canonical, value, text, idx)
                else:
                    facts = _parse_single_value(canonical, value, text, idx)
                if facts:
                    out.extend(facts)
                    consumed.add(idx)
                    continue
        # Try launch-angle prose anchor (covers "launched the missile at N degrees")
        la_m = _LAUNCH_ANGLE_PROSE_RE.search(text)
        if la_m:
            out.append(SpecFact(
                label_canonical="launch_angle",
                value_raw=la_m.group("v"),
                unit_raw=la_m.group("u").lower(),
                raw_phrase=text,
                source_text_idx=idx,
            ))

    return out
