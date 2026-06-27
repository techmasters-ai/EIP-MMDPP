# app/services/extraction_pass_signal_config.py
"""Derive per-pass selection-signal config from a bundle's extraction schema.
Single source of truth = the schema field names + enum descriptions; no literals."""
from __future__ import annotations
import re
from dataclasses import dataclass, field as dc_field
from functools import lru_cache

# unit-suffix -> physical dimension
SUFFIX_DIMENSION = {
    "km": "length", "m": "length", "mm": "length", "cm": "length",
    "deg": "angle", "rad": "angle",
    "sec": "time", "usec": "time", "ms": "time", "ns": "time",
    "mhz": "frequency", "ghz": "frequency", "khz": "frequency", "hz": "frequency",
    "mps": "velocity",
    "kg": "mass", "g": "mass",
    "dbi": "gain",
    "kw": "power", "w": "power",
}
# enum field -> matchable categorical phrases (enum values + schema prose mappings)
CATEGORICAL_PHRASE_FIELDS = {"scan_type", "emitter_function", "system_status",
                             "guidance_type", "seeker_type"}

@dataclass
class PassSignalConfig:
    pass_name: str
    dimensions: set[str] = dc_field(default_factory=set)
    categorical_fields: set[str] = dc_field(default_factory=set)
    has_image_field: bool = False


def _suffix_dimension(field_name: str) -> str | None:
    m = re.search(r"_([a-z]+)$", field_name)
    return SUFFIX_DIMENSION.get(m.group(1)) if m else None

@lru_cache(maxsize=8)
def derive_pass_signal_config(bundle_key: str) -> dict[str, "PassSignalConfig"]:
    from app.services.ontology_bundles import iter_routable_pass_fields  # (pass_name, [field_names])
    out: dict[str, PassSignalConfig] = {}
    for pass_name, field_names in iter_routable_pass_fields(bundle_key):
        c = PassSignalConfig(pass_name=pass_name)
        for fn in field_names:
            dim = _suffix_dimension(fn)
            if dim:
                c.dimensions.add(dim)
            if fn in CATEGORICAL_PHRASE_FIELDS:
                c.categorical_fields.add(fn)
            if fn.endswith("_photo") or "photo" in fn or "image" in fn:
                c.has_image_field = True
        out[pass_name] = c
    return out
