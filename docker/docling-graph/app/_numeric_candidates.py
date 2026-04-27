"""Deterministic numeric-with-units candidate extractor.

Runs BEFORE the LLM call. Sweeps the batch text for `<number><unit>`
patterns and returns the matched substrings so we can inject them into
the user prompt as a hint block. The LLM then decides which field each
candidate maps to (a span-classification task rather than a number-
copying task — much easier for general-purpose models).

Phase A item 5 of the post-LLM-extraction-fix work. Pure code; no LLM
involvement; runs in <1ms per batch.
"""
from __future__ import annotations

import re

# Unit alternation ordered longest-first so "GHz" doesn't get split as
# "G" + "Hz". Each entry is the literal unit string the regex matches;
# the unit family is implicit in the LLM's interpretation.
_UNIT_PATTERNS = [
    # Frequency
    r"GHz", r"MHz", r"kHz", r"Hz",
    # Power
    r"MW", r"kW", r"dBW", r"dBm",
    # Gain / loss
    r"dBi", r"dBd", r"dB",
    # Time
    r"µs", r"us", r"ms", r"min", r"sec", r"s",
    # Distance
    r"km", r"nmi", r"NM", r"miles", r"mi", r"ft", r"in", r"cm", r"mm", r"m",
    # Mass
    r"kg", r"lb", r"lbs", r"g", r"tonnes?", r"t",
    # Speed
    r"km/h", r"kph", r"mph", r"m/s", r"Mach",
    # Angle
    r"degrees?", r"deg", r"°", r"radians?", r"rad", r"mil",
    # Pulse counts (no unit but field-relevant patterns)
]

_NUMBER = r"-?\d+(?:\.\d+)?"
_NUM_UNIT = re.compile(
    r"(?<![A-Za-z0-9_])("
    + _NUMBER
    + r")\s*("
    + "|".join(_UNIT_PATTERNS)
    + r")(?![A-Za-z])"
)


def extract_numeric_candidates(text: str, *, max_candidates: int = 40) -> list[str]:
    """Return verbatim "<number> <unit>" substrings found in text.

    Order = order of first appearance. De-duplicates exact repeats.
    Caps at max_candidates to avoid bloating the prompt; in practice
    a single radar parameter table has 10-25 candidates.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _NUM_UNIT.finditer(text):
        num = match.group(1)
        unit = match.group(2)
        candidate = f"{num} {unit}".strip()
        # Normalize spacing around the number so "35dBi" and "35 dBi"
        # collapse to one canonical form.
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
        if len(out) >= max_candidates:
            break
    return out


def render_candidate_block(candidates: list[str]) -> str:
    """Render a candidate list as a user-prompt section. Returns ''
    when no candidates so the caller can skip emitting the section."""
    if not candidates:
        return ""
    bullets = "\n".join(f"- {c}" for c in candidates)
    return (
        "=== NUMERIC CANDIDATES (verbatim spans found in batch document) ===\n"
        + bullets
        + "\n"
        + "Each candidate above appears verbatim in the BATCH DOCUMENT. "
        "When populating a numeric field, prefer assigning a candidate "
        "from this list to a matching field — don't paraphrase, don't "
        "convert unless the Unit Policy permits, and don't invent values "
        "that aren't on this list.\n"
        "=== END NUMERIC CANDIDATES ===\n\n"
    )
