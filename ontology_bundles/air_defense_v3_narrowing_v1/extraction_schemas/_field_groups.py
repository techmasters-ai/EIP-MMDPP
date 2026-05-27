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

MISSILE_FIELD_GROUPS: dict[str, list[str]] = {
    "missile_identity": [
        "system_name",
        "nomenclature",
        "dieqp",
        "name",
        "emitter_function",
        "system_status",
        "asrd",
        "responsible_agency",
        "review_cycle",
        "next_review_date",
    ],
    "missile_kinematics": [
        "system_name",
        "min_intercept_km",
        "max_intercept_km",
        "min_altitude_km",
        "max_altitude_km",
        "max_launch_angle_deg",
    ],
    "missile_guidance": [
        "system_name",
        "guidance_type",
        "seeker_type",
        "missile_photo",
    ],
    "missile_airframe": [
        "system_name",
        "body_length_m",
        "body_diameter_m",
        "total_mass_kg",
    ],
    "missile_speed_timing": [
        "system_name",
        "average_speed_mps",
        "max_speed_mps",
        "max_flyout_time_sec",
        "flight_time_sec",
        "coast_time_sec",
        "intra_salvo_time_sec",
        "total_burn_time_sec",
        "ejector_time_sec",
    ],
    "missile_propulsion": [
        "system_name",
        "ejector_thrust",
        "ejector_mass_kg",
        "booster_time_sec",
        "booster_thrust",
        "booster_mass_kg",
        "sustain_time_sec",
        "sustain_thrust",
        "sustain_mass_kg",
    ],
}
