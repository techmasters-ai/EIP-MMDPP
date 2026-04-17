"""Missile domain pass: missile systems, launcher, guidance, and seekers.

Narrow extraction views of canonical missile-chain entities.

Key entities (ontology_name):
- ``MISSILE_SYSTEM`` — guided missile weapon
- ``LAUNCHER_SYSTEM`` — launcher platform (TEL, VLS, etc.)
- ``GUIDANCE_METHOD`` — guidance scheme (command, SARH, ARH, IR, …)
- ``SEEKER`` — terminal guidance seeker head
- ``PROPULSION_STACK`` — full propulsion subsystem
- ``PLATFORM`` — host platform
- ``SPECIFICATION`` — measurable parameter/value/unit

Key relationships (docs-valid RelationshipType labels):
- ``MISSILE_SYSTEM HAS_GUIDANCE GUIDANCE_METHOD``
- ``MISSILE_SYSTEM HAS_SEEKER SEEKER``
- ``MISSILE_SYSTEM HAS_PROPULSION PROPULSION_STACK``
- ``MISSILE_SYSTEM INSTALLED_ON PLATFORM``
- ``LAUNCHER_SYSTEM LAUNCHES MISSILE_SYSTEM``
- ``LAUNCHER_SYSTEM INSTALLED_ON PLATFORM``

Plan v32 Task 47 (Phase 5 docs-compliance rewrite + typed-edge migration).
Intra-pass ``MissileRelationship`` DTO retained until Task 64 deletes it.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import (
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_confidence,
    coerce_optional_text,
)


def edge(
    label: str,
    *,
    description: str | None = None,
    examples: list | None = None,
    **field_kwargs: Any,
) -> Any:
    """Helper: declare a typed entity-to-entity edge field.

    Per docs "Template Basics → Edge Helper Function → Required Definition":
    this function must be defined identically in every template.
    """
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    if description is not None:
        field_kwargs["description"] = description
    if examples is not None:
        field_kwargs["examples"] = examples
    return Field(json_schema_extra=existing_extra, **field_kwargs)


def _dedupe_entities_by_identity(cls, values):
    """Merge-preserving dedup for pass-root entity lists (plan Task 9d)."""
    if not isinstance(values, dict):
        return values
    for fname, finfo in cls.model_fields.items():
        raw = values.get(fname)
        if not isinstance(raw, list) or not raw:
            continue
        from typing import get_args as _ga
        def _find_basemodel(ann):
            if ann is None:
                return None
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                return ann
            for a in _ga(ann):
                r = _find_basemodel(a)
                if r is not None:
                    return r
            return None
        inner_cls = _find_basemodel(getattr(finfo, "annotation", None))
        if inner_cls is None:
            continue
        id_fields = list((inner_cls.model_config or {}).get("graph_id_fields", []) or [])
        if not id_fields:
            continue
        accum: dict[tuple, dict] = {}
        order: list[tuple] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = tuple(item.get(k) for k in id_fields)
            if key not in accum:
                accum[key] = dict(item)
                order.append(key)
            else:
                merged = accum[key]
                for k, v in item.items():
                    if k in id_fields or v is None:
                        continue
                    if k not in merged or merged[k] is None:
                        merged[k] = v
        if len(accum) != len(raw):
            values[fname] = [accum[k] for k in order]
    return values


# ----------------------------------------------------------------------
# Entities
# ----------------------------------------------------------------------

class GuidanceMethodEntity(BaseModel):
    """Guidance Method — narrow view of canonical GUIDANCE_METHOD."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="GUIDANCE_METHOD",
        graph_id_fields=["guidance_type"],
        identity_scope="global",
        is_entity=True,
    )

    guidance_type: str = Field(
        ...,
        description="Type of guidance method",
        examples=["COMMAND", "SARH"],
        json_schema_extra={
            "enum": ["COMMAND", "SARH", "ARH", "IR", "BEAM_RIDING",
                     "TVM", "GPS_INS", "DUAL_MODE"],
        },
    )
    firing_doctrine: Optional[str] = Field(
        default=None,
        description="Firing doctrine (shoot-look-shoot, ripple, etc.)",
        examples=["Shoot-look-shoot"],
    )
    track_quality: Optional[str] = Field(
        default=None,
        description="Required track quality for guidance",
        examples=["Fire-control quality track required"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SeekerEntity(BaseModel):
    """Seeker — narrow view of canonical SEEKER."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="SEEKER",
        graph_id_fields=["seeker_nomenclature"],
        identity_scope="document",
        is_entity=True,
    )

    seeker_nomenclature: str = Field(
        ...,
        description="Designation or nomenclature of the seeker",
        examples=["Ka-band active seeker", "Ka-band active seeker"],
    )
    seeker_type: Optional[str] = Field(
        default=None,
        description="Guidance technology used by the seeker",
        json_schema_extra={
            "enum": ["ACTIVE_RADAR", "SEMI_ACTIVE_RADAR", "IR", "DUAL_MODE",
                     "ARM", "GPS_INS", "COMMAND"],
        },
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class PropulsionStackEntity(BaseModel):
    """Propulsion Stack — narrow view of canonical PROPULSION_STACK.

    B-26: value-object component (is_entity=False) per spec §4.8 — the
    prior graph_id_fields=[] anti-pattern resolves via content-based
    identity through the walker (A0-1).
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="PROPULSION_STACK",
        graph_id_fields=[],
        identity_scope="document",
        is_entity=False,
    )

    total_burntime_s: Optional[float] = Field(
        default=None,
        description="Total burn time across all propulsion stages in seconds",
        examples=[25],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_total_burntime_s = field_validator("total_burntime_s", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class PlatformEntity(BaseModel):
    """Platform — narrow view of canonical PLATFORM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="PLATFORM",
        graph_id_fields=["name"],
        identity_scope="global",
        is_entity=True,
    )

    name: str = Field(
        ...,
        description="Common name of the platform",
        examples=["SA-20 TEL", "SA-20 TEL"],
    )
    platform_type: Optional[str] = Field(
        default=None,
        description="Category of the platform",
        json_schema_extra={
            "enum": ["AIRCRAFT", "SHIP", "GROUND_VEHICLE", "FIXED_SITE",
                     "SPACE", "MOBILE_LAUNCHER", "AIR_DEFENSE_BATTERY"],
        },
    )
    service_branch: Optional[str] = Field(
        default=None,
        description="Military branch operating the platform",
        examples=["Russian Aerospace Forces"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class MissileSystemEntity(BaseModel):
    """Missile System — narrow view of canonical MISSILE_SYSTEM with
    typed edges to guidance, seeker, propulsion, and host platform.
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description="Common name of the missile system",
        examples=["PAC-3 MSE", "PAC-3 MSE"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description="Military designation or NATO reporting name",
        examples=["MIM-104F"],
    )

    guidance_method: Optional[GuidanceMethodEntity] = edge(label="HAS_GUIDANCE", default=None)
    seeker: Optional[SeekerEntity] = edge(label="HAS_SEEKER", default=None)
    propulsion_stacks: List[PropulsionStackEntity] = edge(
        label="HAS_PROPULSION", default_factory=list,
    )
    platform: Optional[PlatformEntity] = edge(label="INSTALLED_ON", default=None)

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class LauncherSystemEntity(BaseModel):
    """Launcher System — narrow view of canonical LAUNCHER_SYSTEM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="LAUNCHER_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description="Common name of the launcher system",
        examples=["M903 Launching Station", "M903 Launching Station"],
    )
    launcher_type: Optional[str] = Field(
        default=None,
        description="Category of launcher mechanism",
        examples=["Vertical cold-launch canister"],
    )
    capacity: Optional[int] = Field(
        default=None,
        description="Number of missiles the launcher can hold",
        examples=[16],
    )

    missile_systems: List[MissileSystemEntity] = edge(
        label="LAUNCHES", default_factory=list,
    )
    platform: Optional[PlatformEntity] = edge(label="INSTALLED_ON", default=None)

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_capacity = field_validator("capacity", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SpecificationEntity(BaseModel):
    """Specification — narrow view of canonical SPECIFICATION.

    Value-object component (is_entity=False) per spec §4.8 — content-based
    identity via the walker (A0-1); all fields optional per R19.
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="SPECIFICATION",
        graph_id_fields=[],
        identity_scope="document",
        is_entity=False,
    )

    parameter: Optional[str] = Field(
        default=None,
        description="Name of the measured parameter (e.g. max_range, operating_temperature)",
        examples=["max_range"],
    )
    value: Optional[str] = Field(
        default=None,
        description="Numeric value or range of the measurement",
        examples=["150"],
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement (SI or military standard)",
        examples=["km"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_parameter = field_validator("parameter", mode="before")(coerce_optional_text)
    _v_value = field_validator("value", mode="before")(coerce_optional_text)
    _v_unit = field_validator("unit", mode="before")(coerce_optional_text)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


# ----------------------------------------------------------------------
# Pass-root container — is_entity=False (Decision 4a).
# Intra-pass MissileRelationship DTO removed in Task 64 (plan Task 37);
# typed edges on MissileSystemEntity and LauncherSystemEntity replace it.
# ----------------------------------------------------------------------

class MissileDomainPass(BaseModel):
    """Missile-domain pass root. Guidance / seeker / propulsion / platform
    are reached via typed edges on MissileSystemEntity; launcher_systems
    stay at pass root. Specifications at pass root as bridge entity."""
    model_config = ConfigDict(extra="ignore", is_entity=False)

    missile_systems: List[MissileSystemEntity] = Field(default_factory=list)
    launcher_systems: List[LauncherSystemEntity] = Field(default_factory=list)
    specifications: List[SpecificationEntity] = Field(default_factory=list)

    _dedupe_root_entities = model_validator(mode="before")(_dedupe_entities_by_identity)
