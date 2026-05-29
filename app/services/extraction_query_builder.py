"""Per-pass vector-retrieval query builder.

Walks a pydantic extraction-schema Record class's model_fields to produce
natural-language query text that the C.3 embed endpoint will embed at
retrieval time.

VR Phase C.2b deliverable — pure function, no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.services.ontology_bundles import PassManifest


# ---------------------------------------------------------------------------
# Typed retrieval-signal containers (B1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldRetrievalQuery:
    """A single field's retrieval signal — used in multi-channel candidate gen.
    All values originate from the schema field's json_schema_extra['retrieval'];
    this type contains NO literal domain terms."""
    field_name: str                      # snake_case field name from the schema
    query_text: str                      # doc + field label + description + aliases
    aliases: tuple[str, ...]             # lexical hits
    negative_terms: tuple[str, ...]
    evidence_patterns: tuple[str, ...]
    likely_sections: tuple[str, ...]
    units: tuple[str, ...]


@dataclass(frozen=True)
class PassRetrievalSignals:
    """Structured retrieval signals for a pass (distinct from the pydantic
    RetrievalProfile *config* in ontology_bundles.py)."""
    pass_name:          str
    entity_doc:         str
    entity_query:       str                           # backward-compat single-string query
    field_queries:      tuple[FieldRetrievalQuery, ...]
    lexical_terms:      tuple[str, ...]               # union of all field aliases (dedup)
    negative_terms:     tuple[str, ...]               # union of all field negative_terms (dedup)
    likely_sections:    tuple[str, ...]               # union (dedup)
    evidence_patterns:  tuple[str, ...]               # union (dedup)

# Fields that are identity, not field-relevance — never included in queries.
_SKIP_FIELDS: frozenset[str] = frozenset({"system_name"})

# Description prefix that marks internal/meta fields to be excluded.
_SKIP_DESC_PREFIXES: tuple[str, ...] = ("INTERNAL",)


def _humanize(field_name: str) -> str:
    """Convert a snake_case field name to a readable fallback label.

    Example: ``tx_peak_power_kw`` → ``"tx peak power kw"``
    """
    return field_name.replace("_", " ")


def _record_cls_from_pass_cls(pass_cls: type[BaseModel]) -> type[BaseModel] | None:
    """Extract the Record sub-class from a Pass class.

    Pass classes follow the pattern ``radar_systems: List[RadarPowerRfRecord]``.
    We look for the first list-typed field whose item type is a BaseModel subclass
    and return that item type.  Returns None if the pattern is not found.
    """
    import typing

    for _field_name, field_info in pass_cls.model_fields.items():
        annotation = field_info.annotation
        if annotation is None:
            continue
        # Unwrap Optional / List generics
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            args = getattr(annotation, "__args__", ())
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                return args[0]
    return None


def build_retrieval_query(pass_def: Any, template_cls: type[BaseModel]) -> str:
    """Build the per-pass retrieval query from the pydantic extraction schema.

    Strategy:
      1. Resolve the Record class from the Pass class (e.g. RadarPowerRfRecord
         from RadarPowerRfPass).  If template_cls IS the record class (is_entity
         is true and graph_id_fields is non-empty), use it directly.
      2. Use the Record class's ``__doc__`` (if present) as a prefix.
      3. For each model_field in model_fields, append its description (or
         a humanized version of the field name if no description).
      4. Skip ``system_name`` (identity field, not field-relevance).
      5. Skip fields whose description starts with an INTERNAL marker.
      6. Join with newlines as natural-language text.

    Returns a string that will be embedded by the C.3 endpoint at retrieval
    time.  Pure function — no side effects.

    Example output for radar_power_rf:
      "Subset of RadarSystemEntity covering RF carrier + transmit power.
      Effective Radiated Power in dBW. ...
      Transmitter peak power in kilowatts. ...
      Nominal carrier frequency in MHz. ..."

    Args:
        pass_def:      PassManifest instance (or any object; unused beyond
                       type annotation — the query is derived entirely from
                       the schema class).
        template_cls:  The Pass-level pydantic class (e.g. RadarPowerRfPass).
                       The builder resolves the Record class from this.
                       If template_cls itself is an entity record class, it is
                       used directly.
    """
    # Determine record class
    cfg = getattr(template_cls, "model_config", {}) or {}
    is_record = cfg.get("is_entity") and cfg.get("graph_id_fields")

    if is_record:
        record_cls: type[BaseModel] = template_cls
    else:
        record_cls = _record_cls_from_pass_cls(template_cls)
        if record_cls is None:
            # Fallback: use template_cls fields directly
            record_cls = template_cls

    parts: list[str] = []

    # 1. Class docstring as topic prefix.
    doc = record_cls.__doc__
    if doc and doc.strip():
        parts.append(doc.strip())

    # 2. Walk fields.
    for field_name, field_info in record_cls.model_fields.items():
        if field_name in _SKIP_FIELDS:
            continue
        desc = field_info.description
        if desc:
            stripped = desc.strip()
            if any(stripped.startswith(prefix) for prefix in _SKIP_DESC_PREFIXES):
                continue
            parts.append(stripped)
        else:
            parts.append(_humanize(field_name))

    return "\n".join(parts)
