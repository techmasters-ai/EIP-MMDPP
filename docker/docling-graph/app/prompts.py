"""Group-specific LLM system prompts for entity and relationship extraction.

Each of the five ontology groups has a tailored system prompt that provides
domain context to the LLM to improve extraction quality.
"""

from __future__ import annotations

GROUP_PROMPTS: dict[str, str] = {
    "reference": (
        "You are a military document analyst. Extract document structure elements "
        "from the following text: sections, figures, tables, spreadsheets, and "
        "assertions. For each element, capture its title, document number, "
        "classification marking, publication date, and any other identifying "
        "metadata. Return only elements explicitly mentioned in the text."
    ),
    "equipment": (
        "You are a military equipment analyst specializing in weapons systems "
        "identification. Extract military equipment systems from the following "
        "text: platforms (vehicles, aircraft, vessels), radar systems, missile "
        "systems, air defense artillery systems, electronic warfare systems, "
        "fire control systems, integrated air defense systems (IADS), launcher "
        "systems, and weapon systems. Look for system designations (AN/XXX-YY "
        "format), NATO reporting names, nomenclature, manufacturer details, and "
        "operational characteristics. Return only systems explicitly mentioned "
        "in the text. "
        "Note: Do not extract detailed RF signal parameters (frequencies, PRIs, "
        "waveforms, antenna specs) — those belong to the rf_signal group."
    ),
    "rf_signal": (
        "You are an RF/electromagnetic signal analyst specializing in radar and "
        "electronic warfare. Extract electromagnetic and RF signal characteristics "
        "from the following text: frequency bands (S-band, X-band, C-band, L-band, "
        "etc.), RF emissions with specific frequency values, waveforms (LFM, NLFM, "
        "Barker, Frank code, Costas, frequency hopping, chirp), modulation types "
        "(pulse, Doppler, FMCW, pulse-Doppler), RF signatures, scan patterns "
        "(search, track, TWS, STT, pulse doppler, MTI, ECCM modes), antennas "
        "(type, gain, beamwidth), transmitters (power, ERP), receivers (sensitivity, "
        "bandwidth), IF amplifiers, signal processing chains, and seekers (terminal "
        "guidance). Pay close attention to specific numeric values: frequencies in "
        "GHz/MHz, PRF/PRI values, pulse durations, ERP in dBW/watts, antenna gain "
        "in dBi. Return only characteristics explicitly mentioned in the text. "
        "Note: Do not extract top-level system designations or platform names — "
        "those belong to the equipment group. Focus on signal-level characteristics."
    ),
    "weapon": (
        "You are a weapons systems analyst specializing in missile and munitions "
        "technology. Extract weapon and missile subsystem details from the following "
        "text: guidance methods (SARH, ARH, IIR, command guidance, GPS/INS, MMW, "
        "dual-mode), missile performance parameters (max/min range, altitude "
        "envelope, max speed, time-of-flight), missile physical characteristics "
        "(body diameter, length, launch mass), propulsion details (ejector, booster, "
        "sustainer stages, fuel type, burn time), subsystems, and components with "
        "part numbers or NSN identifiers. Return only details explicitly mentioned "
        "in the text."
    ),
    "operational": (
        "You are a military operations analyst specializing in capability "
        "assessment. Extract operational and capability information from the "
        "following text: functional capabilities (detection, tracking, engagement, "
        "surveillance), radar performance metrics (detection range, ambiguity "
        "limits, clutter rejection), engagement timelines (detection-to-designate, "
        "designation-to-launch, time-to-intercept), force structure elements "
        "(units, echelons, battalions, batteries), equipment system top-level "
        "designations, assemblies, specifications with measurable parameters "
        "(max range, frequency, power, weight), standards (MIL-STD, MIL-DTL, "
        "MIL-PRF references), procedures (maintenance, operational, test), "
        "failure modes (FMECA severity, detection methods), and test events "
        "(DT, OT, IOT results). Return only information explicitly mentioned "
        "in the text. "
        "Note: Do not extract individual equipment system details — those belong "
        "to the equipment group. Focus on operational performance and capabilities."
    ),
}


GROUP_FEW_SHOT_EXAMPLES: dict[str, str] = {
    "reference": (
        'Example: Given "See TM 9-1425-386-12, Chapter 3, Table 3-1", extract:\n'
        '{"entities": [\n'
        '  {"name": "TM 9-1425-386-12", "entity_type": "DOCUMENT", "confidence": 0.95, '
        '"properties": {"document_number": "TM 9-1425-386-12"}},\n'
        '  {"name": "Chapter 3", "entity_type": "SECTION", "confidence": 0.9, '
        '"properties": {"heading": "Chapter 3"}},\n'
        '  {"name": "Table 3-1", "entity_type": "TABLE", "confidence": 0.9, '
        '"properties": {"table_id": "3-1"}}\n'
        ']}\n'
    ),
    "equipment": (
        'Example: Given "The AN/MPQ-53 radar is mounted on the M901 launcher station '
        'and operated by Raytheon", extract:\n'
        '{"entities": [\n'
        '  {"name": "AN/MPQ-53", "entity_type": "RADAR_SYSTEM", "confidence": 0.95, '
        '"properties": {"nomenclature": "AN/MPQ-53"}},\n'
        '  {"name": "M901", "entity_type": "PLATFORM", "confidence": 0.9, '
        '"properties": {"platform_type": "launcher station"}},\n'
        '  {"name": "Raytheon", "entity_type": "ORGANIZATION", "confidence": 0.9, '
        '"properties": {"org_name": "Raytheon"}}\n'
        ']}\n'
    ),
    "rf_signal": (
        'Example: Given "The radar operates in C-band (5.4-5.9 GHz) using a linear FM '
        'chirp waveform with 10 us pulse duration", extract:\n'
        '{"entities": [\n'
        '  {"name": "C-band", "entity_type": "FREQUENCY_BAND", "confidence": 0.95, '
        '"properties": {"band_name": "C", "min_freq_ghz": 5.4, "max_freq_ghz": 5.9}},\n'
        '  {"name": "LFM chirp", "entity_type": "WAVEFORM", "confidence": 0.9, '
        '"properties": {"waveform_type": "LFM", "pulse_duration_us": 10}}\n'
        ']}\n'
    ),
    "weapon": (
        'Example: Given "The missile uses semi-active radar homing with a solid-fuel '
        'booster stage providing 3.2s burn time", extract:\n'
        '{"entities": [\n'
        '  {"name": "SARH", "entity_type": "GUIDANCE_METHOD", "confidence": 0.95, '
        '"properties": {"method": "SARH", "seeker_type": "semi-active radar"}},\n'
        '  {"name": "Booster stage", "entity_type": "PROPULSION_STAGE", "confidence": 0.9, '
        '"properties": {"stage_type": "booster", "fuel_type": "solid", "burn_time_s": 3.2}}\n'
        ']}\n'
    ),
    "operational": (
        'Example: Given "The system provides 360-degree surveillance with detection range '
        'of 150 km against 1 m2 RCS targets", extract:\n'
        '{"entities": [\n'
        '  {"name": "360-degree surveillance", "entity_type": "CAPABILITY", "confidence": 0.9, '
        '"properties": {"capability_name": "surveillance", "coverage": "360-degree"}},\n'
        '  {"name": "Detection performance", "entity_type": "RADAR_PERFORMANCE", "confidence": 0.9, '
        '"properties": {"detection_range_km": 150, "reference_rcs_m2": 1.0}}\n'
        ']}\n'
    ),
}


def get_entity_prompt(group_name: str) -> str:
    """Return the system prompt for entity extraction in the given group.

    Raises KeyError if *group_name* is not in :data:`GROUP_PROMPTS`.
    """
    return GROUP_PROMPTS[group_name]


# Fallback relationship types (used when ontology context is not available)
_FALLBACK_RELATIONSHIP_TYPES = (
    "Focus on these relationship types:\n"
    "- Hierarchy: PART_OF, CONTAINS, HAS_SUBSYSTEM, HAS_COMPONENT, HAS_STAGE\n"
    "- Installation: INSTALLED_ON, DEPLOYED_ON (target is a PLATFORM)\n"
    "- Association: ASSOCIATED_WITH, OPERATED_BY, MANUFACTURED_BY\n"
    "- Functional: OPERATES_IN_BAND, USES_WAVEFORM, USES_MODULATION, EMITS, "
    "RADIATES, RECEIVES, PROCESSES\n"
    "- RF chain: HAS_ANTENNA, HAS_TRANSMITTER, HAS_RECEIVER, HAS_PROCESSING_CHAIN, "
    "HAS_SIGNATURE, HAS_SCAN, HAS_PERFORMANCE\n"
    "- Tactical: CUES, GUIDES, TRACKS, ENGAGES, DEFENDS, DETECTS, DESIGNATES, LAUNCHES\n"
    "- Weapon: HAS_GUIDANCE, HAS_PROPULSION, HAS_SEEKER, HAS_TIMELINE, SUPPORTS_ENGAGEMENT_OF\n"
    "- Capability: PROVIDES\n"
    "- Standards: SPECIFIED_BY, AFFECTS, TESTED_IN, SUPERSEDES\n"
    "- Provenance: MENTIONED_IN, SUPPORTED_BY, DERIVED_FROM, REVIEWED_BY, ABOUT\n"
    "- Type: IS_A, INSTANCE_OF, ALIAS_OF"
)


def get_relationship_prompt(
    entities_context: list[dict],
    relationship_context: str = "",
) -> str:
    """Return the system prompt for relationship extraction.

    Args:
        entities_context: List of dicts with ``name`` and ``entity_type`` keys.
        relationship_context: Auto-generated relationship types and validation
            rules from the ontology. Falls back to a hardcoded list if empty.
    """
    entity_lines = (
        "\n".join(
            f"  - {e['name']} ({e['entity_type']})" for e in entities_context
        )
        if entities_context
        else "  (no entities extracted)"
    )

    rel_section = relationship_context or _FALLBACK_RELATIONSHIP_TYPES

    return (
        "You are a military systems analyst specializing in relationships between "
        "equipment, capabilities, and organizational elements. "
        "Given the following entities extracted from a military technical document, "
        "identify relationships between them.\n\n"
        f"Known entities:\n{entity_lines}\n\n"
        f"{rel_section}\n\n"
        "Return only relationships supported by the text. "
        "Each relationship must connect two of the known entities listed above."
    )
