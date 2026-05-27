"""Audit pydantic field descriptions across all field-group extraction schemas.

For each schema model in ontology_bundles/air_defense_v3/extraction_schemas/,
report per-field description quality so we know which fields need text
improvements before the vector-routing query construction reads them.

Usage:
    python3 scripts/audit_schema_descriptions.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


# Field-group schema modules to audit. Excludes identity schemas (never routed)
# and system_links (required, never routed).
FIELD_GROUP_SCHEMAS = [
    "extraction_schemas.radar_power_rf",
    "extraction_schemas.radar_antenna",
    "extraction_schemas.radar_timing",
    "extraction_schemas.radar_modulation",
    "extraction_schemas.missile_kinematics",
    "extraction_schemas.missile_guidance",
    "extraction_schemas.missile_airframe",
    "extraction_schemas.missile_speed_timing",
    "extraction_schemas.missile_propulsion",
]


def _quality_band(desc_len: int) -> str:
    if desc_len == 0:
        return "EMPTY"
    if desc_len < 30:
        return "SPARSE"
    if desc_len < 80:
        return "OK"
    return "RICH"


def _find_record_class(module):
    """Locate the pydantic record class in the schema module."""
    from pydantic import BaseModel

    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and name.endswith("Record")
        ):
            return obj
    return None


def audit_schema(module_path: str) -> dict:
    """Audit one schema module."""
    pkg = "ontology_bundles.air_defense_v3"
    try:
        module = importlib.import_module(f"{pkg}.{module_path}")
    except Exception as exc:
        return {
            "module": module_path,
            "error": f"import failed: {exc}",
            "fields": [],
        }

    record_cls = _find_record_class(module)
    if record_cls is None:
        return {
            "module": module_path,
            "error": "no Record class found",
            "fields": [],
        }

    fields = []
    for field_name, field_info in record_cls.model_fields.items():
        desc = (field_info.description or "").strip()
        examples = field_info.examples or []
        fields.append({
            "name": field_name,
            "desc_len": len(desc),
            "desc_band": _quality_band(len(desc)),
            "has_examples": len(examples) > 0,
            "n_examples": len(examples),
            "desc_preview": desc[:60] + ("…" if len(desc) > 60 else "") if desc else "",
        })

    return {
        "module": module_path,
        "record_class": record_cls.__name__,
        "n_fields": len(fields),
        "fields": fields,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    overall_band_counts = {"EMPTY": 0, "SPARSE": 0, "OK": 0, "RICH": 0}
    schemas_with_issues = []

    print("=" * 90)
    print("SCHEMA FIELD DESCRIPTION AUDIT")
    print("=" * 90)
    print()

    for module_path in FIELD_GROUP_SCHEMAS:
        result = audit_schema(module_path)
        pass_name = module_path.split(".")[-1]
        if result.get("error"):
            print(f"⚠️  {pass_name}: {result['error']}")
            continue

        band_counts = {"EMPTY": 0, "SPARSE": 0, "OK": 0, "RICH": 0}
        for f in result["fields"]:
            band_counts[f["desc_band"]] += 1
            overall_band_counts[f["desc_band"]] += 1

        has_issues = band_counts["EMPTY"] > 0 or band_counts["SPARSE"] > 0
        marker = "⚠️ " if has_issues else "✓ "
        if has_issues:
            schemas_with_issues.append(pass_name)

        print(f"{marker}{pass_name}  ({result['record_class']}, {result['n_fields']} fields)")
        print(f"    EMPTY: {band_counts['EMPTY']}  SPARSE: {band_counts['SPARSE']}  "
              f"OK: {band_counts['OK']}  RICH: {band_counts['RICH']}")

        for f in result["fields"]:
            if f["desc_band"] in ("EMPTY", "SPARSE"):
                ex = f" (+{f['n_examples']} examples)" if f["has_examples"] else ""
                print(f"    [{f['desc_band']:6}] {f['name']:30} "
                      f"{f['desc_len']:>3} chars  {f['desc_preview']!r}{ex}")
        print()

    print("=" * 90)
    print("OVERALL")
    print("=" * 90)
    total = sum(overall_band_counts.values())
    for band in ("EMPTY", "SPARSE", "OK", "RICH"):
        n = overall_band_counts[band]
        pct = (n / total * 100) if total else 0
        print(f"  {band:6}: {n:>3} fields ({pct:5.1f}%)")
    print(f"  TOTAL:  {total:>3} fields")
    print()
    print(f"Schemas needing improvement: {len(schemas_with_issues)} / {len(FIELD_GROUP_SCHEMAS)}")
    if schemas_with_issues:
        for s in schemas_with_issues:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
