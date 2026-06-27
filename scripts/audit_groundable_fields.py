#!/usr/bin/env python3
"""Groundable-fields audit — verify every numeric field in the production bundle
has unit synonyms via ``units_for`` (or is explicitly allowlisted as unitless).

A numeric field with no unit suffix means ``value_in_chunk`` can never produce a
label positive for that field, silently starving the grounding signal.

Usage::

    python3 -m scripts.audit_groundable_fields

Exit 1 if any numeric field is groundable-gap (numeric, no units, not allowlisted).
"""
from __future__ import annotations

import importlib
import types
import typing
from dataclasses import dataclass, field

from app.services.field_value_grounding import SUFFIX_UNITS, units_for
from app.services.ontology_bundles import load_bundle_manifest
from app.services.extraction_query_builder import _record_cls_from_pass_cls

# ---------------------------------------------------------------------------
# Fields that are unitless by design — counts and a unitless ratio/score.
# Adding a field here suppresses the GAP exit-code; every addition needs a
# justifying comment.
# ---------------------------------------------------------------------------
UNITLESS_OK: frozenset[str] = frozenset({
    "confidence",       # unitless 0-1 extraction confidence score
    "num_bits_in_code", # count of chips/bits in phase-code sequence (dimensionless)
    "pulses_per_dwell", # count of integrated pulses per beam dwell (dimensionless)
})


@dataclass
class AuditRow:
    pass_name: str
    field: str
    numeric: bool
    suffix: str | None
    units: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AuditRow(pass={self.pass_name!r}, field={self.field!r}, "
            f"numeric={self.numeric}, suffix={self.suffix!r}, units={self.units!r})"
        )


def _is_numeric_annotation(annotation) -> bool:
    """Return True iff the (possibly Optional-wrapped) annotation is int or float.

    Handles ``Optional[float]`` (= ``float | None`` = ``Union[float, None]``),
    plain ``float``, and plain ``int``.  Supports both ``typing.Union`` and the
    PEP 604 ``types.UnionType`` (``float | None`` syntax, Python ≥ 3.10).
    """
    if annotation is None:
        return False
    # Unwrap PEP 604 union (float | None, Python ≥ 3.10)
    if isinstance(annotation, types.UnionType):
        args = typing.get_args(annotation)
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return inner[0] in (int, float)
        return False
    # Unwrap typing.Union / Optional
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        args = typing.get_args(annotation)
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return inner[0] in (int, float)
        return False
    return annotation in (int, float)


def _detect_suffix(field_name: str) -> str | None:
    """Return the matched SUFFIX_UNITS key for this field name, or None."""
    f = (field_name or "").lower()
    for suf in sorted(SUFFIX_UNITS, key=len, reverse=True):
        if f.endswith("_" + suf):
            return suf
    return None


def _resolve_template_class(bundle_key: str, pass_def) -> type:
    """Mirror of ``_resolve_template_class`` in app/api/v1/extraction_routing.py."""
    full_module = f"ontology_bundles.{bundle_key}.{pass_def.module}"
    mod = importlib.import_module(full_module)
    return getattr(mod, pass_def.template_class)


def audit_bundle(bundle_key: str) -> list[AuditRow]:
    """Walk every pass in *bundle_key*'s manifest and return one AuditRow per
    model_field of its Record class.

    Only passes with both ``module`` and ``template_class`` set are walked
    (relationship-only passes that carry no record fields are skipped).
    Identity passes are included — they tend to have no numeric fields, which is
    correct, but we want the audit to confirm that.
    """
    manifest = load_bundle_manifest(bundle_key)
    rows: list[AuditRow] = []

    for pass_def in manifest.passes:
        if not (pass_def.module and pass_def.template_class):
            continue

        try:
            pass_cls = _resolve_template_class(bundle_key, pass_def)
        except (ImportError, AttributeError) as exc:
            # Emit a sentinel row so the caller can detect resolution failures.
            rows.append(AuditRow(
                pass_name=pass_def.name,
                field="<resolution_error>",
                numeric=False,
                suffix=None,
                units=[str(exc)],
            ))
            continue

        record_cls = _record_cls_from_pass_cls(pass_cls)
        if record_cls is None:
            # Relationship-only pass or unusual shape — skip field walk.
            continue

        for fname, finfo in record_cls.model_fields.items():
            annotation = finfo.annotation
            numeric = _is_numeric_annotation(annotation)
            suf = _detect_suffix(fname) if numeric else None
            unit_list = units_for(fname) if numeric else []
            rows.append(AuditRow(
                pass_name=pass_def.name,
                field=fname,
                numeric=numeric,
                suffix=suf,
                units=unit_list,
            ))

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    rows = audit_bundle("air_defense_v3")

    # Collect resolution errors first — these mean the audit is incomplete.
    resolution_errors = [r for r in rows if r.field == "<resolution_error>"]

    # Group by pass for table display (non-error rows only).
    by_pass: dict[str, list[AuditRow]] = {}
    for r in rows:
        if r.field == "<resolution_error>":
            continue
        by_pass.setdefault(r.pass_name, []).append(r)

    gaps: list[AuditRow] = []

    for pass_name, pass_rows in by_pass.items():
        numeric_rows = [r for r in pass_rows if r.numeric]
        if not numeric_rows:
            continue
        print(f"\n## {pass_name}")
        print(f"  {'field':35s} {'type':8s} {'suffix':8s} {'units / status'}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*30}")
        for r in numeric_rows:
            if r.units:
                status = f"→ {r.units}"
            elif r.field in UNITLESS_OK:
                status = "UNITLESS_OK"
            else:
                status = "GAP ← no units + not allowlisted"
                gaps.append(r)
            print(f"  {r.field:35s} {'numeric':8s} {r.suffix or '—':8s} {status}")

    print()

    if resolution_errors:
        print("RESOLUTION ERRORS — the following passes could not be imported; "
              "the audit is incomplete:")
        for r in resolution_errors:
            print(f"  {r.pass_name}: {r.units[0] if r.units else '(unknown error)'}")

    if gaps:
        print(f"FAIL — {len(gaps)} numeric field(s) with no unit suffix and not in UNITLESS_OK:")
        for r in gaps:
            print(f"  {r.pass_name}.{r.field}")

    if resolution_errors or gaps:
        return 1

    print(f"PASS — all numeric fields are groundable or explicitly allowlisted "
          f"({len(UNITLESS_OK)} allowlisted: {sorted(UNITLESS_OK)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
