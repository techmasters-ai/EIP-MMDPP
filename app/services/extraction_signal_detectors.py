# app/services/extraction_signal_detectors.py
"""Pure per-chunk selection signals. Reuses the bounded unit matcher so short
units (m/s/g/w) never match inside designators (S-75M, 9 months)."""
from __future__ import annotations
from app.services.field_value_grounding import has_unit_token, nfc

# Dimension -> expanded unit surface forms (abbrev + spelled-out + plural + imperial).
# bare single-char units (m,g,t,s,w) and ambiguous "us"/"hr" omitted: they
# false-match designators (S-75M, Yak-130G, T-90) or common prose ("us", "HR").
DIMENSION_UNITS: dict[str, list[str]] = {
    "length": ["km","mm","cm","nmi","ft","yd","kilometers","kilometres","kilometer","kilometre",
               "meters","metres","meter","metre","millimeters","millimeter","centimeters","centimeter",
               "miles","mile","feet","foot","yards","yard","inches","nautical miles","nautical mile"],
    "mass": ["kg","mg","lb","lbs","kilograms","kilogram","grams","gram","tonnes","tonne",
             "tons","ton","pounds","pound"],
    "time": ["sec","secs","ms","ns","µs","hrs","min","mins","seconds","second",
             "milliseconds","millisecond","microseconds","microsecond","nanoseconds","nanosecond",
             "minutes","minute","hours","hour"],
    "frequency": ["hz","khz","mhz","ghz","hertz","kilohertz","megahertz","gigahertz"],
    "velocity": ["m/s","km/s","km/h","kph","mph","mps","kt","kts","knots","knot","mach",
                 "meters per second","metres per second","kilometers per second"],
    "angle": ["deg","rad","°","degrees","degree","radians","radian","mrad","mils","mil"],
    "gain": ["db","dbi","dbm","dbw","decibels","decibel"],
    "power": ["kw","mw","watts","watt","kilowatts","kilowatt","megawatts","megawatt",
              "milliwatts","milliwatt"],
}

# Categorical enum field -> matchable phrases (enum values + schema prose-mapping phrases).
# (Lifted from the schema field descriptions; keep in sync if descriptions change.)
# Source: ontology_bundles/air_defense_v3/extraction_schemas/ (missile_guidance.py,
# _radar_shared.py field descriptions).
CATEGORICAL_PHRASES: dict[str, list[str]] = {
    "scan_type": ["rotating antenna","mechanical rotation","360-degree scan","rotating dish",
                  "sector scan","raster scan","electronically scanned","phased array","phased-array",
                  "dwell-and-switch","helical scan","spiral scan","conical scan","circular scan",
                  "aesa","pesa"],
    "emitter_function": ["search radar","early warning","acquisition radar","tracking radar",
                         "fire-control radar","fire control radar","engagement radar","illuminator",
                         "multi-function radar","multifunction","height finder","navigation radar",
                         "mfr","amdr"],
    "system_status": ["operational","in service","deployed","fielded","developmental","prototype",
                      "decommissioned","modernized","retired","exported","fms"],
    "guidance_type": ["command guidance","command-to-line-of-sight","semi-active radar homing",
                      "sarh","active radar homing","track-via-missile","tvm","inertial guidance",
                      "beam-rider","beam riding","infrared homing","ir homing","imaging infrared","iir",
                      "passive radar homing","prh","home-on-jam","hoj","homing","guidance"],
    "seeker_type": ["sarh seeker","arh seeker","ir seeker","iir seeker","eo seeker","mmw seeker",
                    "millimeter-wave seeker","dual-mode","electro-optical","seeker"],
}

def measurement_present(dimensions: set[str], text: str) -> bool:
    if not dimensions or not text:
        return False
    units: list[str] = []
    for d in dimensions:
        units.extend(DIMENSION_UNITS.get(d, ()))
    return has_unit_token(nfc(text), units)


def categorical_present(categorical_fields: set[str], text: str) -> bool:
    if not categorical_fields or not text:
        return False
    t = text.lower()
    for fn in categorical_fields:
        for phrase in CATEGORICAL_PHRASES.get(fn, ()):
            if phrase in t:
                return True
    return False


def image_present(source_refs: list[str] | None) -> bool:
    return any(str(r).startswith("#/pictures/") for r in (source_refs or []))
