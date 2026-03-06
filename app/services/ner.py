"""Named Entity Recognition for military/defense documents.

Phase 2 implementation: regex + pattern-based NER for structured military data.
Detects part numbers, NSNs, MIL standards, equipment designations, and
measurable specifications without requiring a downloaded ML model.

This is intentionally deterministic and fully offline. A spaCy model fine-tuned
on military corpora can be layered on top in Phase 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Entity candidate
# ---------------------------------------------------------------------------

@dataclass
class EntityCandidate:
    """A candidate entity extracted from text."""

    entity_type: str        # Maps to ontology entity_types.name
    name: str               # Display name
    source_text: str        # The matched text span
    confidence: float       # 0.0 – 1.0
    properties: dict = field(default_factory=dict)
    span_start: int = 0
    span_end: int = 0


@dataclass
class RelationshipCandidate:
    """A candidate relationship between two entities."""

    rel_type: str           # Maps to ontology relationship_types.name
    from_name: str
    to_name: str
    from_type: str          # entity_type of the source entity
    to_type: str            # entity_type of the target entity
    source_text: str
    confidence: float


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# National Stock Number: XXXX-XX-XXX-XXXX
_RE_NSN = re.compile(
    r"\b(?:NSN\s*)?(\d{4}-\d{2}-\d{3}-\d{4})\b",
    re.IGNORECASE,
)

# MIL Standards: MIL-STD-XXXX, MIL-DTL-XXXX, MIL-PRF-XXXX, MIL-SPEC-XXXX, MIL-T-XXXX
_RE_MIL_STD = re.compile(
    r"\b(MIL[-–](STD|DTL|PRF|SPEC|T|C|A|E|R|W|S|F|G|P|H|Q)[-–]\d+[A-Z]?(?:\([A-Z]+\))?)\b",
    re.IGNORECASE,
)

# AN/xxx designations (military electronics designation)
_RE_AN_DESIGNATION = re.compile(
    r"\b(AN/[A-Z]{3}-\d+[A-Z]?(?:\([A-Z]\d?\))?)\b",
    re.IGNORECASE,
)

# General part numbers: letter-digit patterns common in military specs
# e.g. PN-001, GC-MK4-001, A3B7-4492, P/N 1234-5678
_RE_PART_NUMBER = re.compile(
    r"\b(?:P/?N\s*:?\s*|part\s+(?:number|no\.?)\s*:?\s*)"
    r"([A-Z0-9][-A-Z0-9/]{3,20})\b",
    re.IGNORECASE,
)

# CAGE codes: exactly 5 alphanumeric characters (context-dependent)
_RE_CAGE = re.compile(
    r"\b(?:CAGE\s*(?:code)?\s*:?\s*)([A-Z0-9]{5})\b",
    re.IGNORECASE,
)

# Measurable specifications: keyword + optional colon/whitespace + value + unit
_RE_SPEC = re.compile(
    r"(?:range|speed|velocity|altitude|frequency|power|voltage|current|"
    r"temperature|weight|mass|length|diameter|pressure|bandwidth|accuracy|cep|mtbf)"
    r"[:\s]+(?:of\s+)?(-?[0-9]+(?:\.[0-9]+)?(?:\s*[–-]\s*-?[0-9]+(?:\.[0-9]+)?)?)\s*"
    r"(VDC|VAC|km|m\b|cm|mm|kg|lbs?|MHz|GHz|kHz|Hz|kW|W\b|kV|VDC|V\b|A\b|°C|°F|K\b|"
    r"deg|nm|μm|ms|ns|Mach|hours?)\b",
    re.IGNORECASE,
)

# Equipment system designations (e.g. Patriot PAC-3, THAAD, Aegis BMD)
_KNOWN_SYSTEMS = [
    "Patriot PAC-3", "Patriot PAC-2", "Patriot",
    "THAAD", "Terminal High Altitude Area Defense",
    "Aegis BMD", "Aegis", "SM-3", "SM-6", "SM-2",
    "MEADS", "SHORAD", "C-RAM",
    "Arrow-3", "Arrow-2", "Iron Dome",
    "Stinger", "Avenger",
    "Minuteman III", "Minuteman",
    "Trident II", "Trident",
    "ICBM", "IRBM", "SRBM", "MRBM",
]
_RE_KNOWN_SYSTEMS = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _KNOWN_SYSTEMS) + r")\b",
    re.IGNORECASE,
)

# Component function keywords to help classify noun phrases
_COMPONENT_KEYWORDS = [
    "computer", "processor", "controller", "actuator", "servo",
    "sensor", "seeker", "radar", "transponder", "receiver", "transmitter",
    "antenna", "gyroscope", "accelerometer", "altimeter", "fuze",
    "warhead", "motor", "booster", "nozzle", "fin", "canard",
    "battery", "capacitor", "diode", "circuit board", "PCB",
    "valve", "manifold", "pump", "filter", "housing",
]
_RE_COMPONENT = re.compile(
    r"\b([A-Z][a-zA-Z0-9\-]+\s+(?:" + "|".join(_COMPONENT_KEYWORDS) + r"))\b",
    re.IGNORECASE,
)

# Technical procedure keywords
_PROCEDURE_KEYWORDS = ["inspection", "maintenance", "alignment", "calibration",
                        "checkout", "test procedure", "overhaul", "repair"]
_RE_PROCEDURE = re.compile(
    r"\b([A-Z][A-Za-z\s]{3,40}(?:" + "|".join(_PROCEDURE_KEYWORDS) + r"))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# EM/RF domain patterns
# ---------------------------------------------------------------------------

# ELNOT codes: 2-4 uppercase letters (e.g. "BA", "FIRE", "BELL")
# Only match when preceded by "ELNOT" keyword for disambiguation
_RE_ELNOT = re.compile(
    r"\bELNOT\s*:?\s*([A-Z]{2,20}(?:\s+[A-Z]{2,20})*)\b",
)

# DIEQP identifiers: alphanumeric+dash patterns after DIEQP keyword
_RE_DIEQP = re.compile(
    r"\bDIEQP\s*:?\s*([A-Z0-9][-A-Z0-9]{2,20})\b",
    re.IGNORECASE,
)

# Frequency values with band designations
# e.g. "9.4 GHz", "2700-2900 MHz", "S-band", "X-band"
_RE_FREQUENCY = re.compile(
    r"\b(-?[0-9]+(?:\.[0-9]+)?(?:\s*[-–]\s*[0-9]+(?:\.[0-9]+)?)?)\s*"
    r"(GHz|MHz|kHz|Hz)\b",
    re.IGNORECASE,
)

# NATO frequency band letters: S-band, X-band, C-band, L-band, etc.
_RE_FREQ_BAND = re.compile(
    r"\b([A-Z])-?band\b",
    re.IGNORECASE,
)

# PRI/PRF values: "PRI 1500 μs", "PRF 300-500 Hz"
_RE_PRI_PRF = re.compile(
    r"\b(PRI|PRF)\s*:?\s*(-?[0-9]+(?:\.[0-9]+)?(?:\s*[-–]\s*[0-9]+(?:\.[0-9]+)?)?)\s*"
    r"(μs|us|ms|ns|Hz|kHz|pps)\b",
    re.IGNORECASE,
)

# Radar mode keywords
_RADAR_MODES = [
    "search mode", "track mode", "acquisition mode", "burn-through mode",
    "TWS", "track-while-scan", "STT", "single target track",
    "pulse doppler", "pulse compression", "CW", "continuous wave",
    "FMCW", "MTI", "moving target indicator", "ECCM",
    "HPRF", "MPRF", "LPRF",
]
_RE_RADAR_MODE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in _RADAR_MODES) + r")\b",
    re.IGNORECASE,
)

# Waveform families
_WAVEFORM_FAMILIES = [
    "linear FM", "LFM", "NLFM", "Barker", "polyphase",
    "Frank code", "P1", "P2", "P3", "P4",
    "Costas", "frequency hopping", "chirp",
    "pulse burst", "stagger", "jitter",
]
_RE_WAVEFORM = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WAVEFORM_FAMILIES) + r")\b",
    re.IGNORECASE,
)

# Guidance type keywords
_GUIDANCE_TYPES = [
    "semi-active radar", "SARH", "active radar", "ARH",
    "infrared", "IR homing", "command guidance", "CLOS",
    "beam riding", "TVM", "track-via-missile",
    "INS", "inertial navigation", "GPS", "GPS/INS",
    "millimeter wave", "MMW", "imaging infrared", "IIR",
    "dual-mode", "tri-mode",
]
_RE_GUIDANCE = re.compile(
    r"\b(" + "|".join(re.escape(g) for g in _GUIDANCE_TYPES) + r")\b",
    re.IGNORECASE,
)

# AAA caliber patterns: "23mm", "30 mm", "57-mm", "40mm L/70"
_RE_AAA_CALIBER = re.compile(
    r"\b(\d{2,3})\s*[-]?\s*mm\b",
    re.IGNORECASE,
)

# EW technique keywords
_EW_TECHNIQUES = [
    "jamming", "barrage jamming", "spot jamming", "deception jamming",
    "noise jamming", "DRFM", "digital RF memory",
    "chaff", "flare", "decoy", "towed decoy",
    "ESM", "electronic support measures", "ELINT",
    "SIGINT", "COMINT", "MASINT",
    "RWR", "radar warning receiver",
    "ARM", "anti-radiation missile", "HARM",
]
_RE_EW_TECHNIQUE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _EW_TECHNIQUES) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Relationship extraction patterns
# ---------------------------------------------------------------------------

# "X is a subsystem of Y", "X consists of Y", "X contains Y"
_RE_SUBSYSTEM_OF = re.compile(
    r"\b([A-Z][A-Za-z0-9\-\s]{2,40})\s+is\s+a\s+(?:major\s+)?subsystem\s+of\s+"
    r"([A-Z][A-Za-z0-9\-\s]{2,40})",
    re.IGNORECASE,
)
_RE_CONTAINS = re.compile(
    r"\b([A-Z][A-Za-z0-9\-\s]{2,40})\s+(?:consists\s+of|contains|includes|"
    r"incorporates)\s+(?:a\s+|an\s+|the\s+)?([A-Z][A-Za-z0-9\-\s]{2,40})",
    re.IGNORECASE,
)
_RE_MEETS_STANDARD = re.compile(
    r"\b([A-Z][A-Za-z0-9\-\s]{2,40})\s+(?:meets?|complies?\s+with|per|"
    r"in\s+accordance\s+with|IAW)\s+(MIL[-–][A-Z]+-\d+[A-Z]?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> list[EntityCandidate]:
    """Extract military entity candidates from text.

    Returns a list of EntityCandidate objects with entity_type, name,
    confidence, and extracted properties.
    """
    if not text or not text.strip():
        return []

    candidates: list[EntityCandidate] = []

    # NSN — National Stock Number
    for m in _RE_NSN.finditer(text):
        candidates.append(EntityCandidate(
            entity_type="COMPONENT",
            name=f"NSN {m.group(1)}",
            source_text=m.group(0),
            confidence=0.90,
            properties={"nsn": m.group(1)},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # MIL standards
    for m in _RE_MIL_STD.finditer(text):
        std = m.group(1).upper()
        candidates.append(EntityCandidate(
            entity_type="STANDARD",
            name=std,
            source_text=m.group(0),
            confidence=0.95,
            properties={"designation": std},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # AN/ designations
    for m in _RE_AN_DESIGNATION.finditer(text):
        candidates.append(EntityCandidate(
            entity_type="EQUIPMENT_SYSTEM",
            name=m.group(1).upper(),
            source_text=m.group(0),
            confidence=0.88,
            properties={"designation": m.group(1).upper()},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Part numbers
    for m in _RE_PART_NUMBER.finditer(text):
        candidates.append(EntityCandidate(
            entity_type="COMPONENT",
            name=m.group(1).upper(),
            source_text=m.group(0),
            confidence=0.82,
            properties={"part_number": m.group(1).upper()},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # CAGE codes
    for m in _RE_CAGE.finditer(text):
        candidates.append(EntityCandidate(
            entity_type="ORGANIZATION",
            name=f"CAGE {m.group(1)}",
            source_text=m.group(0),
            confidence=0.80,
            properties={"cage_code": m.group(1).upper()},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Known equipment systems
    for m in _RE_KNOWN_SYSTEMS.finditer(text):
        name = m.group(1)
        candidates.append(EntityCandidate(
            entity_type="EQUIPMENT_SYSTEM",
            name=name.title(),
            source_text=m.group(0),
            confidence=0.85,
            properties={"name": name.title()},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Component noun phrases (e.g. "guidance computer", "servo actuator")
    for m in _RE_COMPONENT.finditer(text):
        name = m.group(1).strip()
        if len(name) >= 5:
            candidates.append(EntityCandidate(
                entity_type="COMPONENT",
                name=name.title(),
                source_text=m.group(0),
                confidence=0.70,
                properties={"name": name.title()},
                span_start=m.start(),
                span_end=m.end(),
            ))

    # Measurable specifications
    for m in _RE_SPEC.finditer(text):
        param = text[max(0, m.start()-30):m.start()].split()[-1] if m.start() > 0 else "value"
        value = m.group(1)
        unit = m.group(2)
        candidates.append(EntityCandidate(
            entity_type="SPECIFICATION",
            name=f"{param} {value} {unit}",
            source_text=m.group(0),
            confidence=0.78,
            properties={"parameter": param, "value": value, "unit": unit},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # --- EM/RF domain entities ---

    # ELNOT codes
    for m in _RE_ELNOT.finditer(text):
        elnot = m.group(1).strip()
        candidates.append(EntityCandidate(
            entity_type="RADAR_SYSTEM",
            name=f"ELNOT {elnot}",
            source_text=m.group(0),
            confidence=0.92,
            properties={"elnot": elnot},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # DIEQP identifiers
    for m in _RE_DIEQP.finditer(text):
        dieqp = m.group(1).upper()
        candidates.append(EntityCandidate(
            entity_type="RADAR_SYSTEM",
            name=f"DIEQP {dieqp}",
            source_text=m.group(0),
            confidence=0.90,
            properties={"dieqp": dieqp},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Frequency values
    for m in _RE_FREQUENCY.finditer(text):
        freq_val = m.group(1)
        freq_unit = m.group(2)
        candidates.append(EntityCandidate(
            entity_type="RF_EMISSION",
            name=f"{freq_val} {freq_unit}",
            source_text=m.group(0),
            confidence=0.75,
            properties={"frequency": freq_val, "unit": freq_unit},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Frequency band designations
    for m in _RE_FREQ_BAND.finditer(text):
        band = m.group(1).upper()
        candidates.append(EntityCandidate(
            entity_type="FREQUENCY_BAND",
            name=f"{band}-band",
            source_text=m.group(0),
            confidence=0.85,
            properties={"band_letter": band},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # PRI/PRF values
    for m in _RE_PRI_PRF.finditer(text):
        param_type = m.group(1).upper()
        value = m.group(2)
        unit = m.group(3)
        candidates.append(EntityCandidate(
            entity_type="WAVEFORM",
            name=f"{param_type} {value} {unit}",
            source_text=m.group(0),
            confidence=0.80,
            properties={param_type.lower(): value, "unit": unit},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Radar modes
    for m in _RE_RADAR_MODE.finditer(text):
        mode = m.group(1)
        candidates.append(EntityCandidate(
            entity_type="SCAN_PATTERN",
            name=mode.title(),
            source_text=m.group(0),
            confidence=0.72,
            properties={"mode": mode},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Waveform families
    for m in _RE_WAVEFORM.finditer(text):
        wf = m.group(1)
        candidates.append(EntityCandidate(
            entity_type="WAVEFORM",
            name=wf.upper() if len(wf) <= 4 else wf.title(),
            source_text=m.group(0),
            confidence=0.78,
            properties={"waveform_family": wf},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # Guidance types
    for m in _RE_GUIDANCE.finditer(text):
        guidance = m.group(1)
        candidates.append(EntityCandidate(
            entity_type="GUIDANCE_METHOD",
            name=guidance.upper() if len(guidance) <= 4 else guidance.title(),
            source_text=m.group(0),
            confidence=0.80,
            properties={"guidance_type": guidance},
            span_start=m.start(),
            span_end=m.end(),
        ))

    # EW techniques
    for m in _RE_EW_TECHNIQUE.finditer(text):
        technique = m.group(1)
        candidates.append(EntityCandidate(
            entity_type="ELECTRONIC_WARFARE_SYSTEM",
            name=technique.upper() if len(technique) <= 5 else technique.title(),
            source_text=m.group(0),
            confidence=0.75,
            properties={"technique": technique},
            span_start=m.start(),
            span_end=m.end(),
        ))

    return _deduplicate(candidates)


def extract_relationships(
    text: str,
    entities: list[EntityCandidate],
) -> list[RelationshipCandidate]:
    """Extract relationship candidates between entities found in the text."""
    if not entities:
        return []

    relationships: list[RelationshipCandidate] = []
    # Build lookup: normalised name → entity type
    name_to_type: dict[str, str] = {e.name.lower(): e.entity_type for e in entities}

    def _lookup_type(name: str, default: str) -> str:
        return name_to_type.get(name.lower(), default)

    # Subsystem-of relationships
    for m in _RE_SUBSYSTEM_OF.finditer(text):
        sub = m.group(1).strip()
        parent = m.group(2).strip()
        relationships.append(RelationshipCandidate(
            rel_type="IS_SUBSYSTEM_OF",
            from_name=sub,
            to_name=parent,
            from_type=_lookup_type(sub, "SUBSYSTEM"),
            to_type=_lookup_type(parent, "EQUIPMENT_SYSTEM"),
            source_text=m.group(0),
            confidence=0.80,
        ))

    # Contains relationships
    for m in _RE_CONTAINS.finditer(text):
        container = m.group(1).strip()
        contained = m.group(2).strip()
        relationships.append(RelationshipCandidate(
            rel_type="CONTAINS",
            from_name=container,
            to_name=contained,
            from_type=_lookup_type(container, "EQUIPMENT_SYSTEM"),
            to_type=_lookup_type(contained, "COMPONENT"),
            source_text=m.group(0),
            confidence=0.75,
        ))

    # Meets-standard relationships (component → standard)
    for m in _RE_MEETS_STANDARD.finditer(text):
        component = m.group(1).strip()
        standard = m.group(2).strip().upper()
        relationships.append(RelationshipCandidate(
            rel_type="MEETS_STANDARD",
            from_name=component,
            to_name=standard,
            from_type=_lookup_type(component, "COMPONENT"),
            to_type="STANDARD",
            source_text=m.group(0),
            confidence=0.85,
        ))

    # Co-occurrence: entities within 200 chars of each other.
    # Only emit (component/system → standard) pairs to avoid noise.
    seen_pairs: set[tuple[str, str]] = set()
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            if abs(e1.span_start - e2.span_start) <= 200:
                pair = (min(e1.name, e2.name), max(e1.name, e2.name))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    if (
                        e1.entity_type in ("COMPONENT", "EQUIPMENT_SYSTEM")
                        and e2.entity_type == "STANDARD"
                    ):
                        relationships.append(RelationshipCandidate(
                            rel_type="MEETS_STANDARD",
                            from_name=e1.name,
                            to_name=e2.name,
                            from_type=e1.entity_type,
                            to_type=e2.entity_type,
                            source_text="co-occurrence",
                            confidence=0.50,
                        ))

    return relationships


def _deduplicate(candidates: list[EntityCandidate]) -> list[EntityCandidate]:
    """Remove duplicate entity candidates by name + type, keeping highest confidence."""
    seen: dict[tuple[str, str], EntityCandidate] = {}
    for c in candidates:
        key = (c.entity_type, c.name.lower())
        if key not in seen or c.confidence > seen[key].confidence:
            seen[key] = c
    return list(seen.values())
