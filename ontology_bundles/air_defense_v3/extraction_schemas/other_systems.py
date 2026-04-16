"""Other systems pass: ADA, EW, fire control, weapon systems, and IADS.

Narrow extraction views of canonical system-level entities.

Key entities (ontology_name):
- ``AIR_DEFENSE_ARTILLERY_SYSTEM`` — gun-based air defense (AAA)
- ``ELECTRONIC_WARFARE_SYSTEM`` — EA/ES/EP system (jammer, receiver, …)
- ``FIRE_CONTROL_SYSTEM`` — targeting and engagement control
- ``WEAPON_SYSTEM`` — generic weapon system
- ``INTEGRATED_AIR_DEFENSE_SYSTEM`` — system-of-systems
- ``PLATFORM`` — host platform
- ``SPECIFICATION`` — measurable parameter/value/unit

Key relationships (docs-valid RelationshipType labels):
- ``INSTALLED_ON`` (system → platform)
- ``SPECIFIED_BY`` (system → specification, handled by merge layer)

Plan v32 Task 48 (Phase 5 docs-compliance rewrite + typed-edge migration).
Intra-pass ``OtherSystemsRelationship`` DTO retained until Task 64.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import (
    coerce_optional_confidence,
    coerce_optional_text,
    normalize_enum,
)


def edge(label: str, **field_kwargs: Any) -> Any:
    """Helper: declare a typed entity-to-entity edge field."""
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
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


class AirDefenseArtillerySystemEntity(BaseModel):
    """Air Defense Artillery System — narrow view of canonical
    AIR_DEFENSE_ARTILLERY_SYSTEM (AAA)."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="AIR_DEFENSE_ARTILLERY_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description="Common name of the AAA system",
        examples=["ZSU-23-4 Shilka", "ZSU-23-4 Shilka"],
    )
    caliber: Optional[str] = Field(
        default=None,
        description="Gun barrel caliber",
        examples=["23 mm"],
    )
    max_tactical_range: Optional[str] = Field(
        default=None,
        description="Maximum tactical engagement range",
        examples=["2.5 km"],
    )
    maximum_rate_of_fire: Optional[str] = Field(
        default=None,
        description="Maximum cyclic rate of fire",
        examples=["3400 rounds/min"],
    )

    platform: Optional[PlatformEntity] = edge(label="INSTALLED_ON", default=None)

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


# Preserve legacy short name as alias (tests import ADAEntity).
ADAEntity = AirDefenseArtillerySystemEntity


class ElectronicWarfareSystemEntity(BaseModel):
    """Electronic Warfare System — narrow view of canonical
    ELECTRONIC_WARFARE_SYSTEM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="ELECTRONIC_WARFARE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description="Common name of the EW system",
        examples=["AN/ALQ-99", "AN/ALQ-99"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description="Military AN/ designation",
        examples=["AN/ALQ-99"],
    )
    ew_role: Optional[str] = Field(
        default=None,
        description="Electronic warfare functional role",
        json_schema_extra={
            "enum": ["EA", "ES", "EP", "SIGINT", "ELINT", "COMINT"],
        },
    )
    coverage: Optional[str] = Field(
        default=None,
        description="Frequency or angular coverage range",
        examples=["64 MHz - 40 GHz"],
    )
    power_output: Optional[str] = Field(
        default=None,
        description="Maximum effective radiated power output",
        examples=["10 kW"],
    )

    platform: Optional[PlatformEntity] = edge(label="INSTALLED_ON", default=None)

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


EWSystemEntity = ElectronicWarfareSystemEntity


class FireControlSystemEntity(BaseModel):
    """Fire Control System — narrow view of canonical FIRE_CONTROL_SYSTEM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="FIRE_CONTROL_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description="Common name of the fire control system",
        examples=["AN/MPQ-65 Radar Set", "AN/MPQ-65 Radar Set"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description="Military AN/ designation",
        examples=["AN/MPQ-65"],
    )

    platform: Optional[PlatformEntity] = edge(label="INSTALLED_ON", default=None)

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class WeaponSystemEntity(BaseModel):
    """Weapon System — narrow view of canonical WEAPON_SYSTEM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="WEAPON_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description="Common name of the weapon system",
        examples=["S-400 Triumf", "S-400 Triumf"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description="Military designation",
        examples=["40R6"],
    )
    weapon_type: Optional[str] = Field(
        default=None,
        description="Category of weapon system",
        examples=["Surface-to-air missile"],
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class IntegratedAirDefenseSystemEntity(BaseModel):
    """IADS — narrow view of canonical INTEGRATED_AIR_DEFENSE_SYSTEM."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="INTEGRATED_AIR_DEFENSE_SYSTEM",
        graph_id_fields=["name"],
        identity_scope="global",
        is_entity=True,
    )

    name: str = Field(
        ...,
        description="Common or NATO reporting name of the IADS",
        examples=["S-400 Triumf", "S-400 Triumf"],
    )
    status: Optional[str] = Field(
        default=None,
        description="Current lifecycle status",
        json_schema_extra={
            "enum": ["DEVELOPMENTAL", "OPERATIONAL", "RETIRED"],
        },
    )
    doctrine: Optional[str] = Field(
        default=None,
        description="Employment doctrine or concept of operations",
        examples=["Layered defense with multi-range engagement"],
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


IADSEntity = IntegratedAirDefenseSystemEntity


class SpecificationEntity(BaseModel):
    """Specification — narrow view of canonical SPECIFICATION."""
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="SPECIFICATION",
        graph_id_fields=["parameter", "value"],
        identity_scope="document",
        is_entity=True,
    )

    parameter: str = Field(
        ...,
        description="Name of the measured parameter (e.g. max_range, operating_temperature)",
        examples=["max_range", "max_range"],
    )
    value: str = Field(
        ...,
        description="Numeric value or range of the measurement",
        examples=["150", "150"],
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
# Legacy intra-pass DTO — retained until Task 64 removes.
# ----------------------------------------------------------------------

class OtherSystemsRelationship(BaseModel):
    """Intra-pass relationship DTO — DEPRECATED.

    Typed edges on the entity classes replace this in Phase 5.
    """
    model_config = ConfigDict(extra="ignore", is_entity=False)

    rel_type: Optional[str] = None
    from_type: Optional[str] = None
    from_identity: Optional[dict[str, Any]] = None
    to_type: Optional[str] = None
    to_identity: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None

    _v_rel_type = field_validator("rel_type", mode="before")(
        normalize_enum({"INSTALLED_ON", "SPECIFIED_BY"}),
    )
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


# ----------------------------------------------------------------------
# Pass-root container — is_entity=False (Decision 4a).
# ----------------------------------------------------------------------

class OtherSystemsPass(BaseModel):
    """Other-systems pass root. Each system carries a typed
    INSTALLED_ON edge to its host PLATFORM when known; SPECIFICATION
    linking is handled post-merge. IADS is a top-level system-of-systems."""
    model_config = ConfigDict(extra="ignore", is_entity=False)

    ada_systems: List[AirDefenseArtillerySystemEntity] = Field(default_factory=list)
    ew_systems: List[ElectronicWarfareSystemEntity] = Field(default_factory=list)
    fire_control_systems: List[FireControlSystemEntity] = Field(default_factory=list)
    weapon_systems: List[WeaponSystemEntity] = Field(default_factory=list)
    iads_systems: List[IntegratedAirDefenseSystemEntity] = Field(default_factory=list)
    specifications: List[SpecificationEntity] = Field(default_factory=list)
    relationships: List[OtherSystemsRelationship] = Field(default_factory=list)

    _dedupe_root_entities = model_validator(mode="before")(_dedupe_entities_by_identity)
