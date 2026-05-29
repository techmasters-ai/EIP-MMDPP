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


def build_retrieval_profile(pass_def: Any, template_cls: type[BaseModel]) -> PassRetrievalSignals:
    """Build structured retrieval signals from the pydantic extraction schema.

    Strategy:
      1. Resolve the Record class (existing logic). Tolerates pass_def=None.
      2. Build entity_query via the existing walker (= prior build_retrieval_query output).
      3. For each non-skipped, non-INTERNAL field with a description:
         - Read (model_fields[name].json_schema_extra or {}).get("retrieval", {}).
           json_schema_extra is None until B3/B4 populate it, so the ``or {}``
           guard is required.
         - Construct a FieldRetrievalQuery from description + retrieval block.
      4. Union all field-level aliases/negative_terms/etc. into the top-level tuples.

    Returns a PassRetrievalSignals with entity_query byte-identical to the
    prior build_retrieval_query output (backward-compat contract).  Pure
    function — no side effects, no literal domain terms.

    Args:
        pass_def:      PassManifest instance or None. Tolerated but unused;
                       all signals are derived entirely from the schema class.
        template_cls:  The Pass-level pydantic class (e.g. RadarPowerRfPass),
                       or the Record class directly (is_entity + graph_id_fields).
    """
    # ------------------------------------------------------------------
    # 1. Resolve Record class (mirrors the pre-existing walker logic)
    # ------------------------------------------------------------------
    cfg = getattr(template_cls, "model_config", {}) or {}
    is_record = cfg.get("is_entity") and cfg.get("graph_id_fields")

    if is_record:
        record_cls: type[BaseModel] = template_cls
    else:
        record_cls = _record_cls_from_pass_cls(template_cls)
        if record_cls is None:
            record_cls = template_cls

    # ------------------------------------------------------------------
    # 2. Build entity_query (exact same logic as the old walker)
    # ------------------------------------------------------------------
    entity_parts: list[str] = []

    doc = record_cls.__doc__
    entity_doc = doc.strip() if (doc and doc.strip()) else ""
    if entity_doc:
        entity_parts.append(entity_doc)

    for field_name, field_info in record_cls.model_fields.items():
        if field_name in _SKIP_FIELDS:
            continue
        desc = field_info.description
        if desc:
            stripped = desc.strip()
            if any(stripped.startswith(prefix) for prefix in _SKIP_DESC_PREFIXES):
                continue
            entity_parts.append(stripped)
        else:
            entity_parts.append(_humanize(field_name))

    entity_query = "\n".join(entity_parts)

    # ------------------------------------------------------------------
    # 3. Build per-field FieldRetrievalQuery objects
    # ------------------------------------------------------------------
    field_queries: list[FieldRetrievalQuery] = []

    for field_name, field_info in record_cls.model_fields.items():
        if field_name in _SKIP_FIELDS:
            continue
        desc = field_info.description
        if desc:
            stripped = desc.strip()
            if any(stripped.startswith(prefix) for prefix in _SKIP_DESC_PREFIXES):
                continue
            query_text = stripped
        else:
            query_text = _humanize(field_name)

        # json_schema_extra is None until B3/B4 populate retrieval blocks.
        extra = field_info.json_schema_extra or {}
        retrieval = extra.get("retrieval", {}) if isinstance(extra, dict) else {}

        field_queries.append(FieldRetrievalQuery(
            field_name=field_name,
            query_text=query_text,
            aliases=tuple(retrieval.get("aliases", ())),
            negative_terms=tuple(retrieval.get("negative_terms", ())),
            evidence_patterns=tuple(retrieval.get("evidence_patterns", ())),
            likely_sections=tuple(retrieval.get("likely_sections", ())),
            units=tuple(retrieval.get("units", ())),
        ))

    # ------------------------------------------------------------------
    # 4. Union all field-level signals (deduped, insertion-ordered)
    # ------------------------------------------------------------------
    def _union(*iterables: tuple[str, ...]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for it in iterables:
            for v in it:
                seen[v] = None
        return tuple(seen)

    all_aliases          = _union(*(fq.aliases          for fq in field_queries))
    all_negative_terms   = _union(*(fq.negative_terms   for fq in field_queries))
    all_likely_sections  = _union(*(fq.likely_sections  for fq in field_queries))
    all_evidence_patterns = _union(*(fq.evidence_patterns for fq in field_queries))

    # pass_name: derive from the Pass class name (no literal domain terms)
    pass_name = template_cls.__name__

    return PassRetrievalSignals(
        pass_name=pass_name,
        entity_doc=entity_doc,
        entity_query=entity_query,
        field_queries=tuple(field_queries),
        lexical_terms=all_aliases,
        negative_terms=all_negative_terms,
        likely_sections=all_likely_sections,
        evidence_patterns=all_evidence_patterns,
    )


def build_retrieval_query(pass_def: Any, template_cls: type[BaseModel]) -> str:
    """Backward-compat shim — returns PassRetrievalSignals.entity_query (str).

    All callers that previously depended on the raw string continue to work
    without change.  The full structured profile is available via
    build_retrieval_profile().
    """
    return build_retrieval_profile(pass_def, template_cls).entity_query
