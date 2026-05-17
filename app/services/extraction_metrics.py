"""Categorized fill metrics for /extract-pass outputs.

Existing metric scripts compute a single "total fills" number per pass.
That conflates three categorically different things:

  * **Schema fields** declared in the pass's `*_FIELD_GROUPS` entry.
    These are the metric the pass is being judged on.
  * **Identity fields** (always `system_name`). Always present when an
    entity is emitted — counting them as a "fill" is misleading.
  * **Extra/out-of-schema fields** that leak through Pydantic's
    `extra="ignore"` filter (typically merge-layer artifacts inherited
    from sibling passes). Counting these as "the pass extracted RF
    data" is the false-positive Recommendation 3 calls out.

This module is the single source of truth for that categorization. It
imports the ontology-bundle field groups so a schema change in one
place automatically updates the metric.
"""
from __future__ import annotations

from typing import Any, Iterable


# Always-identity fields. Currently uniform across all passes in the
# air_defense_v3 bundle. If a future pass adds compound identities, add
# the relevant token here.
IDENTITY_FIELDS: frozenset[str] = frozenset({"system_name"})


def _load_field_groups(bundle_key: str) -> dict[str, list[str]]:
    """Return the merged radar+missile field-groups dict for `bundle_key`.

    Air-defense bundles expose two `*_FIELD_GROUPS` dicts whose keys are
    disjoint pass names. We merge them at the call site so callers don't
    need to know which group a pass belongs to.
    """
    if bundle_key != "air_defense_v3":
        return {}
    # Lazy import — keeps app/services free of ontology_bundles import at module load.
    from ontology_bundles.air_defense_v3.extraction_schemas._field_groups import (  # noqa: PLC0415
        MISSILE_FIELD_GROUPS,
        RADAR_FIELD_GROUPS,
    )
    out: dict[str, list[str]] = {}
    out.update(RADAR_FIELD_GROUPS)
    out.update(MISSILE_FIELD_GROUPS)
    return out


def declared_schema_fields(
    pass_name: str, bundle_key: str = "air_defense_v3",
) -> tuple[frozenset[str], frozenset[str]]:
    """Return (schema_non_identity_fields, identity_fields) for a pass.

    Returns two disjoint frozensets:
      * non-identity schema fields the pass should fill
      * identity fields (always at least `system_name`)

    Empty schema set when the pass isn't recognized in the bundle's
    `*_FIELD_GROUPS` — caller should treat that as "categorization unavailable".
    """
    groups = _load_field_groups(bundle_key)
    declared = set(groups.get(pass_name, ()))
    identity = declared & IDENTITY_FIELDS
    non_identity = frozenset(declared - IDENTITY_FIELDS)
    return non_identity, frozenset(identity or IDENTITY_FIELDS)


def categorize_entity_fills(
    entity: dict[str, Any],
    pass_name: str,
    bundle_key: str = "air_defense_v3",
) -> dict[str, int]:
    """Return per-entity fill counts split into three categories.

    Output shape:
        {
          "schema_fills": int,   # filled non-identity fields declared in pass schema
          "identity_fills": int, # filled identity fields (typically system_name)
          "extra_fills": int,    # filled fields NOT in the pass schema
          "schema_total_possible": int,  # how many schema non-identity fields exist
        }
    """
    schema_non_id, identity = declared_schema_fields(pass_name, bundle_key)

    schema_fills = 0
    identity_fills = 0
    extra_fills = 0
    for field, value in entity.items():
        if field.startswith("_"):
            continue
        if value in (None, "", [], {}):
            continue
        if field in identity:
            identity_fills += 1
        elif field in schema_non_id:
            schema_fills += 1
        else:
            extra_fills += 1
    return {
        "schema_fills": schema_fills,
        "identity_fills": identity_fills,
        "extra_fills": extra_fills,
        "schema_total_possible": len(schema_non_id),
    }


def summarize_pass_metrics(
    entities: Iterable[dict[str, Any]],
    pass_name: str,
    bundle_key: str = "air_defense_v3",
) -> dict[str, Any]:
    """Aggregate categorized fill metrics across all entities for a pass.

    Output shape:
        {
          "entity_count": int,
          "schema_fills_total": int,
          "identity_fills_total": int,
          "extra_fills_total": int,
          "schema_fields_per_entity_possible": int,
          "schema_fill_rate": float,  # schema_fills_total / (entity_count * schema_fields_per_entity_possible)
          "per_field": {field_name: filled_count},  # for schema fields only
        }

    Use this when comparing runs — gives a stable RF-only number for
    radar_power_rf rather than the polluted total that includes
    out-of-schema merge crumbs.
    """
    entity_list = list(entities)
    schema_non_id, _ = declared_schema_fields(pass_name, bundle_key)

    schema_total = identity_total = extra_total = 0
    per_field: dict[str, int] = {f: 0 for f in schema_non_id}
    for ent in entity_list:
        cats = categorize_entity_fills(ent, pass_name, bundle_key)
        schema_total += cats["schema_fills"]
        identity_total += cats["identity_fills"]
        extra_total += cats["extra_fills"]
        for f in schema_non_id:
            if ent.get(f) not in (None, "", [], {}):
                per_field[f] += 1

    n_entities = len(entity_list)
    schema_per_entity = len(schema_non_id)
    total_possible = n_entities * schema_per_entity
    fill_rate = (schema_total / total_possible) if total_possible else 0.0
    return {
        "entity_count": n_entities,
        "schema_fills_total": schema_total,
        "identity_fills_total": identity_total,
        "extra_fills_total": extra_total,
        "schema_fields_per_entity_possible": schema_per_entity,
        "schema_fill_rate": round(fill_rate, 4),
        "per_field": per_field,
    }
