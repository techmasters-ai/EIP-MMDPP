"""Shared helpers for the radar sub-pass modules.

Centralizes the items every radar_* sub-pass uses identically:
- edge field decorator
- forbidden-identity set + optional-text-field set
- system_name normalization validator
- root sanitizer factory (sanitize + dedupe)

Spec §4.3.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from ..validators import (
    canonicalize_identity_text,
    dedupe_entities_by_identity,
    sanitize_entity_list,
)

# Re-export the edge field decorator from the legacy radar_domain.py
# (kept in source as a legacy reference per spec §6 step 4). The
# decorator is unchanged — just centralized so sub-passes can import
# from one location.
from .radar_domain import edge as edge   # noqa: F401


RADAR_FORBIDDEN_SYSTEM_NAMES: frozenset[str] = frozenset({
    "SA-2", "SA-3", "SA-5", "SA-6", "SA-10", "SA-12", "SA-15", "SA-17",
    "SA-20", "SA-21", "SA-22", "SA-23", "PATRIOT", "PAC-2", "PAC-3",
    "PAC-3 MSE", "HAWK", "NIKE-HERCULES", "S-75", "S-125", "S-200", "S-300",
    "S-350", "S-400", "S-500", "AEGIS BMD", "SM-2", "SM-3", "SM-6", "THAAD",
    "ARROW", "IRON DOME", "DAVID'S SLING", "U-2", "SR-71", "RF-4C", "F-4",
    "F-15", "F-16", "B-52", "MIG-21", "MIG-23", "MIG-29", "SU-27",
})

# Superset across sub-passes; each make_root_sanitizer call passes only
# the subset its record class declares.
RADAR_OPTIONAL_TEXT_FIELDS: frozenset[str] = frozenset({
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
    "intra_pulse_mop",
    "inter_pulse",
    "dwell_time",
})


def validate_radar_system_name(value: Any) -> Any:
    """field_validator("system_name", mode="before") body.

    Scope: normalization + non-empty-identity check only.
    Does NOT enforce the forbidden-names list — that authority lives
    exclusively in make_root_sanitizer / sanitize_entity_list.
    """
    if value is None:
        raise ValueError("system_name is required and cannot be None")
    text = canonicalize_identity_text(value)
    if not text or not text.strip():
        raise ValueError("system_name cannot be empty / whitespace-only")
    return text.strip()


def make_root_sanitizer(
    *,
    list_field: str,
    optional_text_fields: set[str] | frozenset[str],
):
    """Factory returning a model_validator(mode="before") body.

    The returned validator runs BOTH sanitize_entity_list AND
    dedupe_entities_by_identity, mirroring the legacy
    _sanitize_and_dedupe_root_entities body in radar_domain.py.
    Sanitize-only factories silently break duplicate-emission handling.

    Defaults forbidden_identities to RADAR_FORBIDDEN_SYSTEM_NAMES so
    sub-pass modules don't have to import the constant directly. This
    is the SINGLE authority for forbidden-name enforcement; the
    field_validator on system_name only normalizes.
    """
    def _sanitize_and_dedupe(cls, values: Any) -> Any:
        values = sanitize_entity_list(
            cls,
            values,
            list_field=list_field,
            identity_field="system_name",
            optional_text_fields=set(optional_text_fields),
            forbidden_identities=RADAR_FORBIDDEN_SYSTEM_NAMES,
        )
        return dedupe_entities_by_identity(cls, values)

    return _sanitize_and_dedupe
