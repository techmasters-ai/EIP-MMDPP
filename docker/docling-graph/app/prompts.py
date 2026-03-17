"""Group-specific LLM system prompts for entity and relationship extraction.

Each of the five ontology groups has a tailored system prompt that provides
domain context to the LLM (granite3-dense:8b) to improve extraction quality.
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
        "in the text."
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
        "in dBi. Return only characteristics explicitly mentioned in the text."
    ),
    "weapon": (
        "You are a weapons systems analyst specializing in missile and munitions "
        "technology. Extract weapon and missile subsystem details from the following "
        "text: guidance methods (SARH, ARH, IIR, command guidance, GPS/INS, MMW, "
        "dual-mode), missile performance parameters (max/min range, altitude "
        "envelope, max speed, time-of-flight, single-shot probability of kill), "
        "missile physical characteristics (body diameter, length, wingspan, launch "
        "mass, warhead mass), propulsion details (ejector, booster, sustainer "
        "stages, fuel type, burn time), subsystems, and components with part "
        "numbers or NSN identifiers. Return only details explicitly mentioned "
        "in the text."
    ),
    "operational": (
        "You are a military operations analyst specializing in capability "
        "assessment. Extract operational and capability information from the "
        "following text: functional capabilities (detection, tracking, engagement, "
        "surveillance), radar performance metrics (detection range, velocity "
        "resolution, range resolution, ambiguity limits, clutter rejection), "
        "engagement timelines (detection-to-designate, designation-to-launch, "
        "time-to-intercept), force structure elements (units, echelons, battalions, "
        "batteries), equipment system top-level designations, assemblies, "
        "specifications with measurable parameters (max range, frequency, power, "
        "weight), standards (MIL-STD, MIL-DTL, MIL-PRF references), procedures "
        "(maintenance, operational, test), failure modes (FMECA severity, detection "
        "methods, MTBF), and test events (DT, OT, IOT results). Return only "
        "information explicitly mentioned in the text."
    ),
}


def get_entity_prompt(group_name: str) -> str:
    """Return the system prompt for entity extraction in the given group.

    Raises KeyError if *group_name* is not in :data:`GROUP_PROMPTS`.
    """
    return GROUP_PROMPTS[group_name]


def get_relationship_prompt(entities_context: list[dict]) -> str:
    """Return the system prompt for relationship extraction.

    Args:
        entities_context: List of dicts with ``name`` and ``entity_type`` keys.
    """
    entity_lines = (
        "\n".join(
            f"  - {e['name']} ({e['entity_type']})" for e in entities_context
        )
        if entities_context
        else "  (no entities extracted)"
    )

    return (
        "You are a military systems analyst specializing in relationships between "
        "equipment, capabilities, and organizational elements. "
        "Given the following entities extracted from a military technical document, "
        "identify relationships between them.\n\n"
        f"Known entities:\n{entity_lines}\n\n"
        "Focus on these relationship types:\n"
        "- Hierarchy: PART_OF, CONTAINS, HAS_SUBSYSTEM, HAS_COMPONENT, HAS_STAGE\n"
        "- Installation: INSTALLED_ON, DEPLOYED_ON (target is a PLATFORM)\n"
        "- Association: ASSOCIATED_WITH, OPERATED_BY, MANUFACTURED_BY\n"
        "- Functional: OPERATES_IN_BAND, USES_WAVEFORM, USES_MODULATION, EMITS, "
        "RADIATES, RECEIVES, PROCESSES\n"
        "- RF chain: HAS_ANTENNA, HAS_TRANSMITTER, HAS_RECEIVER, HAS_PROCESSING_CHAIN, "
        "HAS_SIGNATURE, HAS_SCAN, HAS_PERFORMANCE\n"
        "- Tactical: CUES, GUIDES, TRACKS, ENGAGES, DEFENDS, DETECTS, DESIGNATES\n"
        "- Standards: SPECIFIED_BY, COMPLIES_WITH (target is a STANDARD)\n"
        "- Signal chain: FEEDS_INTO, RECEIVES_FROM\n"
        "- Type: IS_A, INSTANCE_OF, ALIAS_OF\n\n"
        "Return only relationships supported by the text. "
        "Each relationship must connect two of the known entities listed above."
    )
