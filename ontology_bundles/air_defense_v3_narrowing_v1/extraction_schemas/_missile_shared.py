"""Shared utilities for missile sub-pass extraction schemas.

Mirrors _radar_shared.py exactly. The 6 missile sub-pass modules all
import from this file rather than copy-pasting forbidden sets, sanitizer
factories, or validator bodies.

Single authority for forbidden-name enforcement: make_missile_root_sanitizer.
Single authority for missile identity normalization: validate_missile_system_name.

Spec §4.3.
"""
from __future__ import annotations

from typing import Any

from ..validators import (
    canonicalize_identity_text,
    dedupe_entities_by_identity,
    sanitize_entity_list,
)
from .missile_domain import _MISSILE_FORBIDDEN_SYSTEM_NAMES
from .radar_domain import edge as edge  # noqa: F401  re-export for sub-pass modules


# Frozen so accidental mutation in a sub-pass module fails loudly.
MISSILE_FORBIDDEN_SYSTEM_NAMES: frozenset[str] = frozenset(_MISSILE_FORBIDDEN_SYSTEM_NAMES)


# Superset across all missile sub-passes. Used by sub-pass sanitizer
# wiring to decide which fields qualify for optional-text coercion.
MISSILE_OPTIONAL_TEXT_FIELDS: frozenset[str] = frozenset({
    "nomenclature",
    "dieqp",
    "name",
    "emitter_function",
    "system_status",
    "asrd",
    "responsible_agency",
    "review_cycle",
    "next_review_date",
    "guidance_type",
    "seeker_type",
    "ejector_thrust",
    "booster_thrust",
    "sustain_thrust",
})


def validate_missile_system_name(value: Any) -> Any:
    """field_validator("system_name", mode="before") body for missile passes.

    Scope: normalization + non-empty-identity check only.
    Does NOT enforce the forbidden-names list — that authority lives
    exclusively in make_missile_root_sanitizer / sanitize_entity_list.
    """
    if value is None:
        raise ValueError("system_name is required and cannot be None")
    text = canonicalize_identity_text(value)
    if not text or not text.strip():
        raise ValueError("system_name cannot be empty / whitespace-only")
    return text.strip()


def make_missile_root_sanitizer(
    *,
    list_field: str,
    optional_text_fields: set[str] | frozenset[str],
):
    """Factory returning a model_validator(mode="before") body.

    The returned validator runs BOTH sanitize_entity_list AND
    dedupe_entities_by_identity, mirroring make_root_sanitizer in
    _radar_shared.py. Sanitize-only factories silently break
    duplicate-emission handling.

    Defaults forbidden_identities to MISSILE_FORBIDDEN_SYSTEM_NAMES so
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
            forbidden_identities=MISSILE_FORBIDDEN_SYSTEM_NAMES,
        )
        return dedupe_entities_by_identity(cls, values)

    return _sanitize_and_dedupe
