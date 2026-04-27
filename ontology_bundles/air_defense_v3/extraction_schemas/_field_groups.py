"""Single source of truth for radar extraction field groups.

Each group becomes its own /extract-pass call so the LLM sees a
focused subset of the radar checklist instead of all 30+ fields at
once. Spec §4.2.

Hand-authored — partitioning is a task-fit decision, not derivable
from json_schema_extra.profile_subgroup. Contract-tested in
tests/unit/test_radar_field_groups_contract.py.
"""

RADAR_FIELD_GROUPS: dict[str, list[str]] = {
    "radar_identity": [
        "system_name",
        "nomenclature",
        "elnot",
        "dieqp",
        "emitter_function",
        "system_status",
        "asrd",
        "responsible_agency",
        "review_cycle",
        "next_review_date",
        "scan_type",
    ],
    "radar_power_rf": [
        "system_name",
        "erp_dbw",
        "tx_peak_power_kw",
        "nominal_rf_mhz",
    ],
    "radar_antenna": [
        "system_name",
        "antenna_photo",
        "gain_dbi",
        "antenna_dim_az_m",
        "antenna_dim_el_m",
        "beamwidth_az_deg",
        "beamwidth_el_deg",
        "spoiled",
        "coverage_limits_el_deg",
    ],
    "radar_timing": [
        "system_name",
        "nominal_pri_usec",
        "nominal_pd_usec",
        "scan_period_sec",
        "dwell_time",
    ],
    "radar_modulation": [
        "system_name",
        "intra_pulse_mop",
        "inter_pulse",
        "frequency_excursion_mhz",
        "num_bits_in_code",
        "pulses_per_dwell",
    ],
}
