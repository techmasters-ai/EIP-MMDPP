"""Missile domain pass: missile systems and all sub-components."""
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..validators import (
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_confidence,
    coerce_optional_text,
    normalize_enum,
)


class MissileSystemEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [system_name]
    system_name: Optional[str] = None
    nomenclature: Optional[str] = None
    guidance_type: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class LauncherSystemEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [system_name]
    system_name: Optional[str] = None
    launcher_type: Optional[str] = None
    capacity: Optional[int] = None
    confidence: Optional[float] = None

    _v_capacity = field_validator("capacity", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class GuidanceMethodEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [guidance_type]
    guidance_type: Optional[str] = None
    firing_doctrine: Optional[str] = None
    track_quality: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SeekerEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [seeker_nomenclature]
    seeker_nomenclature: Optional[str] = None
    seeker_ELNOT: Optional[str] = None
    seeker_type: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class PropulsionStackEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [] — content-hash fallback
    total_burntime_s: Optional[float] = None
    confidence: Optional[float] = None

    _v_total_burntime_s = field_validator("total_burntime_s", mode="before")(coerce_optional_float)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class PlatformEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [name]
    name: Optional[str] = None
    platform_type: Optional[str] = None
    service_branch: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SpecificationEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [parameter, value]
    parameter: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    confidence: Optional[float] = None

    # See radar_domain.SpecificationEntity for rationale.
    _v_parameter = field_validator("parameter", mode="before")(coerce_optional_text)
    _v_value = field_validator("value", mode="before")(coerce_optional_text)
    _v_unit = field_validator("unit", mode="before")(coerce_optional_text)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class MissileRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rel_type: Optional[str] = None
    from_type: Optional[str] = None
    from_identity: Optional[dict[str, Any]] = None
    to_type: Optional[str] = None
    to_identity: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None

    _v_rel_type = field_validator("rel_type", mode="before")(
        normalize_enum({
            "INSTALLED_ON", "HAS_GUIDANCE", "HAS_SEEKER", "HAS_PROPULSION", "LAUNCHES", "SPECIFIED_BY",
        })
    )
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class MissileDomainPass(BaseModel):
    model_config = ConfigDict(extra="ignore")

    missile_systems: list[MissileSystemEntity] = Field(default_factory=list)
    launcher_systems: list[LauncherSystemEntity] = Field(default_factory=list)
    guidance_methods: list[GuidanceMethodEntity] = Field(default_factory=list)
    seekers: list[SeekerEntity] = Field(default_factory=list)
    propulsion_stacks: list[PropulsionStackEntity] = Field(default_factory=list)
    platforms: list[PlatformEntity] = Field(default_factory=list)
    specifications: list[SpecificationEntity] = Field(default_factory=list)
    relationships: list[MissileRelationship] = Field(default_factory=list)
