"""Other systems pass: ADA, EW, fire control, weapon systems, and IADS."""
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..validators import (
    coerce_optional_confidence,
    coerce_optional_text,
    normalize_enum,
)


class ADAEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [system_name]
    system_name: Optional[str] = None
    caliber: Optional[str] = None
    max_tactical_range: Optional[str] = None
    maximum_rate_of_fire: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class EWSystemEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [system_name]
    system_name: Optional[str] = None
    nomenclature: Optional[str] = None
    ew_role: Optional[str] = None
    coverage: Optional[str] = None
    power_output: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class FireControlSystemEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [system_name]
    system_name: Optional[str] = None
    nomenclature: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class WeaponSystemEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [system_name]
    system_name: Optional[str] = None
    nomenclature: Optional[str] = None
    weapon_type: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class IADSEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [name]
    name: Optional[str] = None
    status: Optional[str] = None
    doctrine: Optional[str] = None
    confidence: Optional[float] = None

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


class OtherSystemsRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rel_type: Optional[str] = None
    from_type: Optional[str] = None
    from_identity: Optional[dict[str, Any]] = None
    to_type: Optional[str] = None
    to_identity: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None

    _v_rel_type = field_validator("rel_type", mode="before")(
        normalize_enum({"INSTALLED_ON", "SPECIFIED_BY"})
    )
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class OtherSystemsPass(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ada_systems: list[ADAEntity] = Field(default_factory=list)
    ew_systems: list[EWSystemEntity] = Field(default_factory=list)
    fire_control_systems: list[FireControlSystemEntity] = Field(default_factory=list)
    weapon_systems: list[WeaponSystemEntity] = Field(default_factory=list)
    iads_systems: list[IADSEntity] = Field(default_factory=list)
    platforms: list[PlatformEntity] = Field(default_factory=list)
    specifications: list[SpecificationEntity] = Field(default_factory=list)
    relationships: list[OtherSystemsRelationship] = Field(default_factory=list)
