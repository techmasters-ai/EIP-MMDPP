"""radar_modulation extraction pass — pulse modulation + coding.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Group fields: system_name, intra_pulse_mop,
inter_pulse, frequency_excursion_mhz, num_bits_in_code, pulses_per_dwell.

Field descriptions are written as dual-use retrieval and prompt text:
use document-facing anchor terms while preserving extraction constraints. The
byte-equal description-parity check in
test_extraction_views_subset_of_canonical_with_validator_parity was
loosened in commit 20b1a8d to allow this divergence.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float, coerce_optional_int, coerce_optional_text
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_modulation"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class RadarModulationRecord(BaseModel):
    """Radar waveform modulation and coding: intra-pulse modulation, inter-pulse PRI behavior, chirp bandwidth, code length, and pulses per dwell."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR. Accept proper-noun "
            "radar names. Never emit weapon, missile, aircraft, or "
            "platform names — those are filtered deterministically."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    intra_pulse_mop: Optional[str] = Field(
        default=None,
        description=(
            "Intra-pulse modulation type. Accept one of: CW, LFM_CHIRP, "
            "NLFM, BARKER_CODE, POLYPHASE, BIPHASE. "
            "Map prose to enum: 'continuous wave' / 'CW' → CW; "
            "'linear FM' / 'linear frequency modulation' / 'LFM' / "
            "'linear chirp' / 'chirp' → LFM_CHIRP; 'nonlinear FM' / "
            "'nonlinear frequency modulation' / 'NLFM' → NLFM; "
            "'Barker code' / 'Barker-coded pulse' → BARKER_CODE; "
            "'polyphase code' / 'Frank code' / 'P1 code' / 'P2 code' / "
            "'P3 code' / 'P4 code' → POLYPHASE; 'biphase code' / "
            "'phase-shift keying' / 'PSK' / 'BPSK' → BIPHASE. "
            "Emit as uppercase."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "intra-pulse modulation", "modulation type",
                    "pulse modulation", "waveform modulation",
                    "CW", "continuous wave",
                    "LFM", "linear FM", "linear frequency modulation",
                    "linear chirp", "chirp",
                    "NLFM", "nonlinear FM", "nonlinear frequency modulation",
                    "Barker code", "Barker-coded pulse",
                    "polyphase code", "Frank code",
                    "biphase code", "phase-shift keying", "PSK", "BPSK",
                ],
                "negative_terms": [
                    "inter-pulse modulation", "PRI stagger",
                    "frequency agile", "frequency hopping",
                ],
                "evidence_patterns": [
                    r"re:(?:intra|intrapulse)\s+(?:modulation|MOP)",
                    r"re:(?:LFM|NLFM|CW)\b",
                    "Barker code", "polyphase code",
                ],
                "likely_sections": [
                    "waveform", "modulation", "specifications", "radar",
                    "signal processing",
                ],
                "units": [],
            }
        },
    )
    inter_pulse: Optional[str] = Field(
        default=None,
        description=(
            "Inter-pulse modulation type. Accept one of: CONSTANT_PRI, "
            "PRI_STAGGER, PRI_JITTER, FREQ_AGILE. "
            "Map prose to enum: 'constant PRI' / 'fixed PRI' / "
            "'fixed pulse repetition interval' / 'fixed repetition "
            "rate' → CONSTANT_PRI; 'staggered PRI' / 'PRI stagger' / "
            "'pulse stagger' → PRI_STAGGER; 'jittered PRI' / 'PRI "
            "jitter' / 'random PRI' → PRI_JITTER; 'frequency-agile' / "
            "'frequency hopping' / 'frequency-agile pulse' / 'agile "
            "frequency' → FREQ_AGILE. Emit as uppercase."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "inter-pulse modulation", "PRI modulation",
                    "constant PRI", "fixed PRI",
                    "staggered PRI", "PRI stagger", "pulse stagger",
                    "jittered PRI", "PRI jitter", "random PRI",
                    "frequency-agile", "frequency hopping", "agile frequency",
                ],
                "negative_terms": [
                    "intra-pulse modulation", "pulse modulation type",
                    "LFM", "chirp", "Barker code",
                ],
                "evidence_patterns": [
                    r"re:inter(?:-|\s+)pulse\s+modulation",
                    r"re:PRI\s+(?:stagger|jitter)",
                    "frequency-agile", "frequency hopping",
                ],
                "likely_sections": [
                    "waveform", "modulation", "specifications", "radar",
                    "timing",
                ],
                "units": [],
            }
        },
    )
    frequency_excursion_mhz: Optional[float] = Field(
        default=None,
        description=(
            "Frequency excursion or chirp bandwidth in MHz. Relevant source "
            "labels include 'Frequency Excursion', 'Chirp Bandwidth', "
            "'Sweep Width', 'sweep bandwidth', 'FM sweep', 'LFM "
            "bandwidth', 'Modulation Bandwidth', 'frequency deviation', "
            "'MHz', or 'kHz'. Use modulation/chirp bandwidth, not RF "
            "carrier frequency."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "frequency excursion", "chirp bandwidth",
                    "sweep width", "sweep bandwidth",
                    "FM sweep", "LFM bandwidth",
                    "modulation bandwidth", "frequency deviation",
                ],
                "negative_terms": [
                    "carrier frequency", "operating frequency",
                    "RF frequency", "center frequency",
                ],
                "evidence_patterns": [
                    r"re:(?:chirp|sweep|FM)\s+bandwidth",
                    r"re:frequency\s+excursion",
                    r"re:\d+(?:\.\d+)?\s*(?:MHz|kHz)\s+(?:sweep|bandwidth)",
                ],
                "likely_sections": [
                    "waveform", "modulation", "specifications", "radar",
                    "signal processing",
                ],
                "units": ["MHz", "kHz", "GHz"],
            }
        },
    )
    num_bits_in_code: Optional[int] = Field(
        default=None,
        description=(
            "Number of chips or bits in the phase-code sequence. Relevant "
            "source labels include 'Code Length', 'Chips', 'chip count', "
            "'Bits', 'Number of Bits', 'Barker length', 'polyphase "
            "code length', or 'phase-code length'. Integer count; emit "
            "only when the source states it."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "code length", "chips", "chip count",
                    "number of bits", "Barker length",
                    "polyphase code length", "phase-code length",
                    "code bits",
                ],
                "negative_terms": [
                    "pulse width", "pulses per dwell",
                    "pulse duration", "pulse count",
                ],
                "evidence_patterns": [
                    r"re:(?:code|chip)\s+length",
                    r"re:Barker\s*[-\s]*\d+",
                    r"re:\d+\s*(?:chip|bit)\s+code",
                ],
                "likely_sections": [
                    "waveform", "modulation", "specifications", "radar",
                    "coding",
                ],
                "units": ["bits", "chips"],
            }
        },
    )
    pulses_per_dwell: Optional[int] = Field(
        default=None,
        description=(
            "Pulses integrated per beam-position dwell. Relevant source "
            "labels include 'Pulses per Dwell', 'Pulses/Dwell', "
            "'Integrated Pulses', 'pulses integrated', 'pulse count per "
            "look', or 'pulses per look'. Integer count; emit only when "
            "the source states it."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "pulses per dwell", "pulses/dwell",
                    "integrated pulses", "pulses integrated",
                    "pulse count per look", "pulses per look",
                    "pulses per beam position",
                ],
                "negative_terms": [
                    "code length", "chip count", "num bits",
                    "scan period", "dwell time",
                ],
                "evidence_patterns": [
                    r"re:pulses?\s+per\s+(?:dwell|look)",
                    r"re:integrated\s+pulses",
                ],
                "likely_sections": [
                    "waveform", "signal processing", "specifications",
                    "radar", "integration",
                ],
                "units": ["pulses"],
            }
        },
    )

    _v_system_name              = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_intra_pulse_mop          = field_validator("intra_pulse_mop", mode="before")(coerce_optional_text)
    _v_inter_pulse              = field_validator("inter_pulse", mode="before")(coerce_optional_text)
    _v_frequency_excursion_mhz  = field_validator("frequency_excursion_mhz", mode="before")(coerce_optional_float)
    _v_num_bits_in_code         = field_validator("num_bits_in_code", mode="before")(coerce_optional_int)
    _v_pulses_per_dwell         = field_validator("pulses_per_dwell", mode="before")(coerce_optional_int)


class RadarModulationPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarModulationRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with pulse modulation + coding "
            "values extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields={"intra_pulse_mop", "inter_pulse"},
        )
    )
