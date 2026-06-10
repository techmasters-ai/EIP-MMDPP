"""Per-pass vector-retrieval query builder.

Walks a pydantic extraction-schema Record class's model_fields to produce
natural-language query text and per-field retrieval signals.

B1+B2 deliverables: FieldRetrievalQuery, PassRetrievalSignals,
build_retrieval_profile (structured) + build_retrieval_query shim (str).
Pure functions, no side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.services.ontology_bundles import PassManifest

from app.services.extraction_unit_gate import signature_for_fields


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
    unit_signature:     tuple[str, ...] = ()          # G1 gate: sorted union of unit synonyms

# Fields that are identity, not field-relevance — never included in queries.
_SKIP_FIELDS: frozenset[str] = frozenset({"system_name"})

# Description prefix that marks internal/meta fields to be excluded.
_SKIP_DESC_PREFIXES: tuple[str, ...] = ("INTERNAL",)


def _humanize(field_name: str) -> str:
    """Convert a snake_case field name to a readable fallback label.

    Example: ``tx_peak_power_kw`` → ``"tx peak power kw"``
    """
    return field_name.replace("_", " ")


def _union(*iterables: tuple[str, ...]) -> tuple[str, ...]:
    """Insertion-order-preserving dedup across multiple string tuples."""
    seen: dict[str, None] = {}
    for it in iterables:
        for v in it:
            seen[v] = None
    return tuple(seen)


def _field_query_text(field_name: str, field_info: Any) -> str | None:
    """Return the query_text for a field, or None if the field should be skipped.

    Keeps the entity_query walker and the field_queries walker synchronised —
    both loops use this function so their skip logic can never drift.
    """
    if field_name in _SKIP_FIELDS:
        return None
    desc = field_info.description
    if desc:
        stripped = desc.strip()
        if any(stripped.startswith(p) for p in _SKIP_DESC_PREFIXES):
            return None
        return stripped
    return _humanize(field_name)


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
        qt = _field_query_text(field_name, field_info)
        if qt is not None:
            entity_parts.append(qt)

    entity_query = "\n".join(entity_parts)

    # ------------------------------------------------------------------
    # 3. Build per-field FieldRetrievalQuery objects
    # ------------------------------------------------------------------
    field_queries: list[FieldRetrievalQuery] = []

    for field_name, field_info in record_cls.model_fields.items():
        query_text = _field_query_text(field_name, field_info)
        if query_text is None:
            continue

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
    all_aliases          = _union(*(fq.aliases          for fq in field_queries))
    all_negative_terms   = _union(*(fq.negative_terms   for fq in field_queries))
    all_likely_sections  = _union(*(fq.likely_sections  for fq in field_queries))
    all_evidence_patterns = _union(*(fq.evidence_patterns for fq in field_queries))

    # pass_name: prefer pass_def.name when provided; fall back to class name.
    pass_name = getattr(pass_def, "name", None) or template_cls.__name__

    # G1 gate: derive unit signature from the FULL record_cls.model_fields
    # (not field_queries, which excludes system_name + INTERNAL fields).
    # units_for() returns [] for non-unit fields, so over-inclusion is harmless.
    unit_sig = signature_for_fields(record_cls.model_fields.keys()) if record_cls is not None else ()

    return PassRetrievalSignals(
        pass_name=pass_name,
        entity_doc=entity_doc,
        entity_query=entity_query,
        field_queries=tuple(field_queries),
        lexical_terms=all_aliases,
        negative_terms=all_negative_terms,
        likely_sections=all_likely_sections,
        evidence_patterns=all_evidence_patterns,
        unit_signature=unit_sig,
    )


def derive_pass_keywords(signals: PassRetrievalSignals) -> list[str]:
    """Derive a generalizable per-pass keyword list from the schema ``units``.

    Each extraction field carries ``json_schema_extra["retrieval"]["units"]``
    (e.g. ``("dBW", "dBm", "W", "kW")`` for power fields), already parsed into
    ``FieldRetrievalQuery.units`` by ``build_retrieval_profile``.  Units are
    per-pass-distinctive, domain-general, and contain ZERO instance names —
    making them ideal generalizable keyword needles for the lexical router
    channel (``RetrievalProfile.lexical_keywords`` → ``keyword_hit_counts``),
    which is otherwise empty everywhere (dead signal).

    Strategy:
      1. Union all ``fq.units`` across ``signals.field_queries``
         (insertion-ordered dedup via the shared ``_union`` helper).
      2. Drop any unit that is ALREADY in the alias vocabulary counted as
         field_label hits (``signals.lexical_terms``).  Dedup is NFC +
         casefold so case-insensitive overlap is removed.  The result therefore
         contains ONLY terms NOT already counted as aliases — it adds NEW
         signal rather than double-counting.

    Guardrail: this mines ``units`` ONLY.  It never touches field ``examples``,
    pass-edge ``examples``, or ``description`` token-splits — those carry
    literal equipment designations and would violate the no-instance-names
    rule.  ``units`` and ``aliases`` are both curated, instance-free.

    Pure function — no side effects, no literal domain terms.

    Args:
        signals: The pass's structured retrieval signals.

    Returns:
        Insertion-ordered list of unit keywords not already present in the
        alias vocabulary.  Empty list when no units are declared.
    """
    all_units = _union(*(fq.units for fq in signals.field_queries))

    # NFC + casefold alias vocabulary for instance-free dedup.
    import unicodedata  # noqa: PLC0415 (module-local; mirrors lexical-search normalisation)

    alias_cf = {
        unicodedata.normalize("NFC", term).casefold()
        for term in signals.lexical_terms
    }

    return [
        unit
        for unit in all_units
        if unicodedata.normalize("NFC", unit).casefold() not in alias_cf
    ]


def build_retrieval_query(pass_def: Any, template_cls: type[BaseModel]) -> str:
    """Backward-compat shim — returns PassRetrievalSignals.entity_query (str).

    All callers that previously depended on the raw string continue to work
    without change.  The full structured profile is available via
    build_retrieval_profile().
    """
    return build_retrieval_profile(pass_def, template_cls).entity_query
