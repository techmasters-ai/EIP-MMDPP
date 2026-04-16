"""Shared Pydantic field validators for the air_defense_v3 extraction
schemas. Handle messy LLM output (empty strings, embedded numbers,
text confidence levels) that would otherwise fail Pydantic coercion.

Spec §3.2. Used by every extraction schema module under
ontology_bundles/air_defense_v3/extraction_schemas/."""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _log_unrecoverable(fn_name: str, value: Any) -> None:
    """Warn that ``value`` could not be coerced by ``fn_name`` → None.

    Kept as a single choke-point so the wording stays stable for the
    A-2 contract test (``"unrecoverable" in message``).
    """
    logger.warning(
        "%s: unrecoverable input %r (type=%s) -> None",
        fn_name,
        value,
        type(value).__name__,
    )


def coerce_optional_int(value: Any) -> int | None:
    """Return an int, or None. Accepts None, int, numeric strings,
    and strings with an embedded int ('page 5 of 10' -> 5). Empty/
    whitespace strings and unparseable values become None.

    Logs a WARNING when the input is non-None, non-empty, but
    unrecoverable (A-2). Empty strings and None are silent (those
    are normal "field absent" signals, not bad LLM output).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        _log_unrecoverable("coerce_optional_int", value)
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            pass
        match = _INT_RE.search(stripped)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                _log_unrecoverable("coerce_optional_int", value)
                return None
        _log_unrecoverable("coerce_optional_int", value)
        return None
    _log_unrecoverable("coerce_optional_int", value)
    return None


def coerce_optional_float(value: Any) -> float | None:
    """Return a float, or None. Accepts None, int, float, and parseable
    decimal strings. Unparseable values become None.

    Logs a WARNING when the input is non-None, non-empty, but
    unrecoverable (A-2). None and empty strings are silent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        _log_unrecoverable("coerce_optional_float", value)
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            match = _FLOAT_RE.search(stripped)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    _log_unrecoverable("coerce_optional_float", value)
                    return None
            _log_unrecoverable("coerce_optional_float", value)
            return None
    _log_unrecoverable("coerce_optional_float", value)
    return None


def coerce_optional_text(value: Any) -> str | None:
    """Return a stripped string, or None. Used for SpecificationEntity
    fields (parameter, value, unit) where the LLM frequently emits the
    numeric part of a specification as a raw int/float instead of a
    string, which Pydantic would otherwise reject as ``Input should be
    a valid string``.

    Rules:
    - None / empty / whitespace-only string -> None.
    - str -> stripped str (empty-after-strip collapses to None).
    - int / float -> ``str(value)`` so 150 and '150' agree on identity.
    - bool -> None: True/False for a SPECIFICATION.value is almost
      always an LLM mistake; surfacing it as 'True' would pollute the
      graph and destabilize SPECIFICATION identity (``[parameter, value]``).
    - dict / list / any other type -> None: there is no stable stringify
      rule (``str({'a': 1}) == "{'a': 1}"`` depends on dict iteration
      order), so coerce to None rather than fragment the graph.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


# Text-confidence mappings used when the LLM returns "high"/"medium"/"low"
# instead of a numeric value. These are arbitrary bucket midpoints chosen
# to roughly agree with the extraction calibration in spec §6.6.
_TEXT_CONFIDENCE = {
    "high": 0.9,
    "medium": 0.6,
    "med": 0.6,
    "low": 0.3,
}


def coerce_optional_confidence(value: Any) -> float | None:
    """Return a confidence float in [0.0, 1.0], or None.

    - None -> None.
    - float/int in [0.0, 1.0] -> pass through as float.
    - int/float > 1 -> divided by 100 (percent).
    - 'high' / 'medium' / 'low' (case-insensitive) -> bucket midpoints.
    - Unparseable strings -> None.

    IMPORTANT: explicit 0.0 MUST be returned as 0.0, not coerced to a
    default. Do not use 'x or default' anywhere in this function — that
    is the bug this validator exists to protect against.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        _log_unrecoverable("coerce_optional_confidence", value)
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if f > 1.0:
            return f / 100.0
        return f
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return None
        if stripped in _TEXT_CONFIDENCE:
            return _TEXT_CONFIDENCE[stripped]
        try:
            f = float(stripped)
        except ValueError:
            _log_unrecoverable("coerce_optional_confidence", value)
            return None
        if f > 1.0:
            return f / 100.0
        return f
    _log_unrecoverable("coerce_optional_confidence", value)
    return None


def _normalize_enum(enum_cls: type[Enum], v: Any) -> str | None:
    """Normalize a value to one of ``enum_cls``'s string values, or None.

    Docs-signature form of the enum normalization helper — takes the Enum
    class directly. Intended for use with
    ``field_validator("field", mode="before")(partial(_normalize_enum, MyEnum))``
    or wrapped in a closure. Keeps ``normalize_enum(set[str])`` around for
    back-compat with existing call sites.

    Normalization rules match the set-based ``normalize_enum``:
    - None -> None
    - Enum member (of ``enum_cls`` or any) -> its ``value`` attribute
    - Non-string, non-Enum -> None
    - Empty / whitespace-only string -> None
    - Exact match against any member's ``value`` -> that value
    - Case-insensitive + space-to-underscore match -> the canonical value
    - No match -> None
    """
    if v is None:
        return None
    if isinstance(v, Enum):
        return v.value
    if not isinstance(v, str):
        return None
    stripped = v.strip()
    if not stripped:
        return None
    canonical = {m.value.upper(): m.value for m in enum_cls}
    normalized = stripped.upper().replace(" ", "_")
    return canonical.get(normalized)


def normalize_enum(allowed: set[str]) -> Callable[[Any], str | None]:
    """Build a validator that normalizes an input to one of `allowed` or None.

    Normalization:
    - None -> None
    - Non-string -> None
    - Exact match -> the allowed value itself
    - Case-insensitive match -> the allowed value in its canonical form
    - Spaces replaced with underscores before matching
    - No match -> None

    Used by extraction schemas to let the LLM return 'radar' or 'fan
    song' and still get 'RADAR' or 'FAN_SONG' in the final output."""
    canonical = {a.upper(): a for a in allowed}

    def validator(value: Any) -> str | None:
        if value is None or not isinstance(value, str):
            return None
        normalized = value.strip().upper().replace(" ", "_")
        return canonical.get(normalized)

    return validator
