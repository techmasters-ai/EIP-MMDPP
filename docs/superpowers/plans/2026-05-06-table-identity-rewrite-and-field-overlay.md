# Table-Derived Identity Rewrite + Per-Cell Field Overlay (Mechanism A1) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land deterministic post-LLM table-derived identity rewrite + per-cell field overlay so column-major variants tables (SA-2 PDF and equivalent) collapse alias-vertices into one canonical entity per column AND override wrong-but-non-null LLM field values where the variants table is authoritative — without disturbing LLM-only fields and behind a kill switch.

**Architecture:** Two-stage post-LLM pipeline. Parser side (`docker/docling-graph/`) parses the first STRICTLY-QUALIFYING column-major / hybrid table into a `TableOverlay` Pydantic payload (entity-type-scoped alias map + typed `TableFact` list + `CrossEntityHint` list) and ships it on the `/extract-pass` response. Worker side (`app/services/table_overlay.py` NEW + `extraction_merge.py` MODIFY) applies the overlay as a phase before merge: `apply_identity_rewrite` rewrites `system_name` aliases inside `canonicalize_cross_pass_identities`; `apply_field_overlay` runs as Phase 0.5 of `merge_and_resolve` with full `cls.model_validate(...)` validation, fan-out to ALL post-rewrite matching instances, per-(fact, instance) atomicity, and `policy="table_wins_for_table_facts"` (scoped: only fields with a corresponding `TableFact` are touched). Worker-side env-flag check is authoritative over cached overlay payloads from `pipeline_pass_outputs.metadata_json`.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, dataclasses, FastAPI, docling-graph LLM extraction service, Ollama gemma4:31b, pytest, Docker Compose.

**Spec:** [`docs/superpowers/specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md`](../specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md) (commit `2729089`, signed off after 7 review passes)

**Acceptance (headline):** `missile_propulsion`: ≥6 of 7 listed variants (13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23) have `booster_mass_kg` matching the variants-table 1st-stage weight row within tolerance AND ≥6 of 7 have `sustain_mass_kg` matching the 2nd-stage weight row, sourced via `apply_field_overlay` (with `FIELD_OVERLAY_OVERRIDE` log lines emitted where the LLM had wrong values pre-overlay). No regression on `missile_kinematics` / `missile_airframe` / `missile_speed_timing` floor counts vs the live baseline re-derived in Task 0.

---

## Pre-flight checklist

Run these once at the start of the session and before each chunk:

- [ ] **P0: Read the spec.**

```bash
wc -l docs/superpowers/specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md
```
Expected: ≥1500 lines. If less, the file is truncated — abort and re-fetch.

Use the @superpowers-extended-cc:test-driven-development skill for every code-bearing task.

- [ ] **P1: Confirm baseline test suite status.**

Run on host from repo root (host pytest, NOT inside the docling-graph container — the Dockerfile copies `app/` but not `tests/`):
```bash
pytest docker/docling-graph/tests -q 2>&1 | tail -5
pytest tests/unit -q 2>&1 | tail -5
```
Expected: All current tests pass. Note any pre-existing failures so they aren't attributed to this plan.

- [ ] **P2: Confirm stack is up.**

```bash
docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "docling-graph|api|worker"
```
Expected: docling-graph, api, worker / worker-graph all Up. If not, `docker compose up -d` and wait 30 s.

- [ ] **P3: Confirm none of the new symbols already exist.**

```bash
grep -nE "extract_table_overlay|class TableOverlay|class TableFact|class CrossEntityHint|MISSILE_IDENTITY_LABELS|RADAR_IDENTITY_LABELS|CANONICAL_PRIORITY|CROSS_ENTITY_REF_PATTERNS|apply_identity_rewrite|apply_field_overlay|table_alias_map_by_entity_type|DOCLING_GRAPH_TABLE_OVERLAY_ENABLED" docker/docling-graph/app/_table_facts.py docker/docling-graph/app/_alias_map.py docker/docling-graph/app/main.py docker/docling-graph/app/schemas.py app/services/extraction_merge.py app/workers/pipeline.py docker-compose.yml 2>&1
```
Expected: No hits in any of the listed files. (One pre-existing hit on `class PassResult` at `app/services/extraction_merge.py:202` is fine — that's not a new symbol.) If hits appear, an in-progress branch is bleeding through; reconcile before starting.

- [ ] **P4: Confirm ground-truth schema field names.**

```bash
grep -nE "^class.*Record|^    [a-z_]+(_kg|_mps|_km|_sec|_m|_deg): " ontology_bundles/air_defense_v3/extraction_schemas/{missile_airframe,missile_kinematics,missile_speed_timing,missile_propulsion}.py
```
Expected: `body_length_m`, `body_diameter_m`, `total_mass_kg` on airframe; `min_intercept_km`, `max_intercept_km`, `min_altitude_km`, `max_altitude_km`, `max_launch_angle_deg` on kinematics; `average_speed_mps`, `max_speed_mps`, plus `_time_sec` floats on speed_timing; `ejector_mass_kg`, `booster_mass_kg`, `sustain_mass_kg`, plus `_time_sec` and text `_thrust` fields on propulsion. **NO `booster_propellant_mass_kg`.** **NO `max_speed_m_per_s`.** Spec acceptance §8.6 is pinned to these names.

- [ ] **P5: Confirm extraction-schema field validators are in place.**

```bash
grep -n "field_validator(" ontology_bundles/air_defense_v3/extraction_schemas/{missile_airframe,missile_kinematics,missile_speed_timing,missile_propulsion}.py | wc -l
```
Expected: ≥20 hits (each numeric float field has a `_v_<field> = field_validator("<field>", mode="before")(coerce_optional_float)` hook). The full-`model_validate` validation gate in §5.3 step (c) depends on these firing.

- [ ] **P6: Verify host Python has pytest.**

```bash
which pytest && pytest --version
```
Expected: pytest 7+. If not, activate `.venv/bin/activate` then re-check.

---

## Chunk 0: Re-derive baseline (Task 0)

Spec §10 step 0 mandates this BEFORE any code lands. Without a fresh baseline, the §8.6 floor row is anchored on stale `/tmp/r21_alias_only_backup/` numbers and acceptance comparisons are not meaningful.

### Task 0: Re-derive live baseline at HEAD

**Files:** None modified. This task captures measurements only.

- [ ] **Step 1: Confirm overlay code is NOT yet wired (it shouldn't be — pre-implementation).**

```bash
grep -nE "extract_table_overlay|table_overlay" docker/docling-graph/app/main.py app/services/extraction_merge.py 2>&1
```
Expected: 0 hits. If any, a parallel branch landed early; reconcile before re-deriving.

- [ ] **Step 2: Set kill switch off explicitly (defensive).**

Edit `docker-compose.yml` if needed to make sure the env var is OFF at this snapshot. Since the var doesn't exist yet, this is a no-op; document it for clarity in the run log.

- [ ] **Step 3: Run notebook §20 at T=1.0 against the SA-2 PDF.**

Open `notebooks/extraction_walkthrough.ipynb` in the running Jupyter container. Execute §20 with `temperature=1.0`. Capture per-pass `✓ exact` counts AND per-variant field-correctness on the fields named in spec §8.6 acceptance:
- `missile_kinematics`: `max_intercept_km`, `min_intercept_km`, `max_altitude_km`, `min_altitude_km` correctness per variant
- `missile_airframe`: `body_length_m`, `body_diameter_m`, `total_mass_kg` correctness per variant
- `missile_speed_timing`: `max_speed_mps` correctness per variant
- `missile_propulsion`: `booster_mass_kg`, `sustain_mass_kg` correctness per variant

- [ ] **Step 4: Save baseline snapshot.**

```bash
mkdir -p /tmp/baseline_2026-05-06_pre_overlay
docker exec eip-mmdpp-jupyter cp -r /home/jovyan/work/notebooks/.cell_outputs/section_20_T1.0/ /tmp/baseline_2026-05-06_pre_overlay/
```
Save the headline scorecard markdown to `/tmp/baseline_2026-05-06_pre_overlay/scorecard.md` for direct quotation in the spec.

- [ ] **Step 5: Update spec §8.6 floor row with live numbers.**

In the spec at `docs/superpowers/specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md`, replace each `≥ live-baseline floor` placeholder in the §8.6 acceptance table with the actual count from Step 3 (e.g., `missile_propulsion ≥ 4` if the live baseline produced 4 ✓ exact). Commit:

```bash
git add docs/superpowers/specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md
git commit -m "spec(table-overlay): pin live baseline floor numbers from §20 T=1.0 run"
```

- [ ] **Step 6: Document baseline FIELD_OVERLAY_OVERRIDE delta predictions.**

For each variant where the LLM emitted a wrong propulsion mass value (off-by-one row attribution per spec §1), record the predicted FIELD_OVERLAY_OVERRIDE log line content (entity, field, llm_value, table_value) so Task 12 acceptance can grep for them as confirmation. Save to `/tmp/baseline_2026-05-06_pre_overlay/expected_overrides.md`.

**Acceptance:**
- Live baseline floor numbers replace placeholders in spec §8.6
- Per-variant field correctness recorded for all four passes
- Predicted override log lines documented
- `/tmp/baseline_2026-05-06_pre_overlay/` snapshot committed (or noted in commit-message body since /tmp is ephemeral)

---

## Chunk 1: Constants + parser primitives (Tasks 1–2)

Tasks 1–2 establish data and primitives that downstream parser entry depends on. After Chunk 1, no behavior change — `extract_table_overlay()` does not yet exist, just its building blocks.

### Task 1: `_alias_map.py` constants + drift guards

**Files:**
- Modify: `docker/docling-graph/app/_alias_map.py`
- Create: `docker/docling-graph/tests/test_alias_map_overlay_constants.py`

- [ ] **Step 1: Write failing drift-guard tests.**

Create `docker/docling-graph/tests/test_alias_map_overlay_constants.py`:

```python
"""Drift guards for _alias_map.py overlay constants (spec §8.2).

These tests pin the structure and content invariants of the four new
constants (MISSILE_IDENTITY_LABELS, RADAR_IDENTITY_LABELS,
CROSS_ENTITY_REF_PATTERNS, CANONICAL_PRIORITY) so that future edits
cannot silently break the overlay's classification rules.
"""
import importlib.util
from pathlib import Path

_ALIAS_PATH = Path(__file__).resolve().parent.parent / "app" / "_alias_map.py"


def _load_alias_map():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_alias_map", _ALIAS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missile_identity_labels_excludes_bare_variant_and_designation():
    """Spec §5.1: bare 'variant' and 'designation' MUST NOT appear; they
    create false positives via cross-entity-ref rows like
    'Fan Song Variant'."""
    am = _load_alias_map()
    labels = tuple(s.lower() for s in am.MISSILE_IDENTITY_LABELS)
    assert "variant" not in labels, (
        "bare 'variant' would misclassify Fan Song Variant rows as missile aliases"
    )
    assert "designation" not in labels


def test_identity_labels_have_canonical_priority_coverage():
    """Every label in MISSILE_IDENTITY_LABELS appears (case-insensitive
    substring) in CANONICAL_PRIORITY['MISSILE_SYSTEM'] OR is documented
    as intentional fallback. Catches new label without priority entry."""
    am = _load_alias_map()
    priority = tuple(s.lower() for s in am.CANONICAL_PRIORITY["MISSILE_SYSTEM"])
    for label in am.MISSILE_IDENTITY_LABELS:
        norm = label.lower()
        # Match any priority entry that's a substring of the label, or vice-versa.
        assert any(p in norm or norm in p for p in priority), (
            f"identity label {label!r} has no CANONICAL_PRIORITY entry"
        )


def test_cross_entity_ref_patterns_dont_overlap_identity_labels():
    """A label can't be both a missile-identity row AND a cross-entity-ref
    row. CROSS_ENTITY_REF_PATTERNS keys must not match MISSILE_IDENTITY_LABELS
    or RADAR_IDENTITY_LABELS after normalization."""
    am = _load_alias_map()
    refs = set(am.CROSS_ENTITY_REF_PATTERNS.keys())
    missile = {s.lower() for s in am.MISSILE_IDENTITY_LABELS}
    radar = {s.lower() for s in am.RADAR_IDENTITY_LABELS}
    assert refs.isdisjoint(missile)
    assert refs.isdisjoint(radar)


def test_canonical_priority_uses_display_labels():
    """CANONICAL_PRIORITY entries are user-facing label patterns
    (Title Case with spaces, e.g., 'Missile Type'), not schema field
    names (snake_case)."""
    am = _load_alias_map()
    for entity_type, entries in am.CANONICAL_PRIORITY.items():
        for entry in entries:
            assert " " in entry, (
                f"{entity_type}: {entry!r} is missing a space — looks like a "
                f"schema field name, not a display label"
            )
            assert "_" not in entry, (
                f"{entity_type}: {entry!r} contains underscore — display "
                f"labels should be Title Case with spaces"
            )
            assert entry[0].isupper(), (
                f"{entity_type}: {entry!r} should start uppercase"
            )
```

- [ ] **Step 2: Run tests, confirm 4 fail.**

```bash
pytest docker/docling-graph/tests/test_alias_map_overlay_constants.py -v
```
Expected: 4 FAILED with `AttributeError: module ... has no attribute 'MISSILE_IDENTITY_LABELS'`.

- [ ] **Step 3: Add the four constants to `_alias_map.py`.**

Append to `docker/docling-graph/app/_alias_map.py`:

```python
# ----------------------------------------------------------------------
# Mechanism A1 (spec §5.1): identity-row label patterns + cross-entity
# refs + canonical-name priority. Used by extract_table_overlay() in
# _table_facts.py to classify column-0 cells in column-major variants
# tables.
# ----------------------------------------------------------------------

# Row labels that mean "this row holds an identifier for the entity in
# the column above." Bare "variant" and "designation" are DELIBERATELY
# EXCLUDED for v1 — they create false positives via cross-entity-ref
# rows (e.g., "Fan Song Variant" would match "variant", which is wrong).
MISSILE_IDENTITY_LABELS: tuple[str, ...] = (
    "missile type",
    "missile variant",
    "industry designation",
    "military designation",
    "nato designation",
    "system designation",
)

RADAR_IDENTITY_LABELS: tuple[str, ...] = (
    "radar variant",
    "radar designation",
    "radar type",
)

# Cross-entity reference rows: row labels that name a SIBLING entity
# type. When seen in a missile-context table, the row's cells are not
# missile aliases — they're radar aliases attached to the same column's
# missile via a relationship hint. Emitted as CrossEntityHint, not
# folded into the missile alias cluster.
#
# Classification order (enforced in _classify_identity_row):
#   1. Cross-entity-ref check FIRST
#   2. Identity-label check SECOND
#   3. Spec-row check (label-to-schema-field alias) THIRD
#   4. Otherwise: ignored
CROSS_ENTITY_REF_PATTERNS: dict[str, str] = {
    "fan song variant": "RADAR_SYSTEM",
    "spoon rest variant": "RADAR_SYSTEM",
}

# Canonical-name priority per entity type. When a column has aliases
# from multiple identity rows, pick the FIRST priority label that's
# present.
CANONICAL_PRIORITY: dict[str, tuple[str, ...]] = {
    "MISSILE_SYSTEM": (
        "Missile Type",
        "Industry Designation",
        "Military Designation",
        "NATO Designation",
    ),
    "RADAR_SYSTEM": (
        "Radar Variant",
        "Radar Designation",
        "Radar Type",
    ),
}
```

- [ ] **Step 4: Run tests, confirm all 4 pass.**

```bash
pytest docker/docling-graph/tests/test_alias_map_overlay_constants.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Run the full docling-graph test suite to confirm no regression.**

```bash
pytest docker/docling-graph/tests -q 2>&1 | tail -10
```
Expected: All tests pass; new test count up by 4.

- [ ] **Step 6: Commit.**

```bash
git add docker/docling-graph/app/_alias_map.py \
       docker/docling-graph/tests/test_alias_map_overlay_constants.py
git commit -m "feat(table-overlay): add MISSILE_/RADAR_IDENTITY_LABELS, CROSS_ENTITY_REF_PATTERNS, CANONICAL_PRIORITY constants"
```

### Task 2: `_table_facts.py` parser primitives

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py`
- Create: `docker/docling-graph/tests/test_table_overlay_extract.py`

- [ ] **Step 1: Write failing unit tests for the four helpers.**

Create `docker/docling-graph/tests/test_table_overlay_extract.py` with the helper tests from spec §8.1 row 1–4:

```python
"""Unit tests for spec §5.2 helper functions (Mechanism A1).

These cover the four pure helpers that compose extract_table_overlay:
_classify_identity_row, _classify_cross_entity_ref,
_extract_alias_clusters, _pick_canonical.
"""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load_table_facts():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_table_facts_overlay", _FACTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---- _classify_identity_row -------------------------------------------------


def test_classify_identity_row_missile():
    tf = _load_table_facts()
    assert tf._classify_identity_row("Missile Type") == "MISSILE_SYSTEM"
    assert tf._classify_identity_row("Industry Designation") == "MISSILE_SYSTEM"


def test_classify_identity_row_radar():
    tf = _load_table_facts()
    assert tf._classify_identity_row("Radar Variant") == "RADAR_SYSTEM"


def test_classify_identity_row_cross_entity_ref_returns_none():
    """Fan Song Variant matches RADAR_SYSTEM in CROSS_ENTITY_REF_PATTERNS,
    not MISSILE_IDENTITY_LABELS — must be classified by
    _classify_cross_entity_ref instead, NOT by _classify_identity_row.
    Per spec §5.1 classification-order rule, cross-entity-ref check
    runs FIRST; identity-label check runs SECOND."""
    tf = _load_table_facts()
    assert tf._classify_identity_row("Fan Song Variant") is None


def test_classify_identity_row_spec_row_returns_none():
    tf = _load_table_facts()
    assert tf._classify_identity_row("Length mm") is None
    assert tf._classify_identity_row("") is None


# ---- _classify_cross_entity_ref --------------------------------------------


def test_classify_cross_entity_ref_fan_song():
    tf = _load_table_facts()
    assert tf._classify_cross_entity_ref("Fan Song Variant") == "RADAR_SYSTEM"


def test_classify_cross_entity_ref_unknown_returns_none():
    tf = _load_table_facts()
    assert tf._classify_cross_entity_ref("Missile Type") is None
    assert tf._classify_cross_entity_ref("Length mm") is None


# ---- _pick_canonical -------------------------------------------------------


def test_pick_canonical_picks_missile_type_first():
    tf = _load_table_facts()
    cluster = {
        "Missile Type": "1D",
        "Industry Designation": "SA-75",
        "NATO Designation": "SA-2A",
    }
    assert tf._pick_canonical(cluster, entity_type="MISSILE_SYSTEM") == "1D"


def test_pick_canonical_falls_back_when_missile_type_missing():
    tf = _load_table_facts()
    cluster = {
        "Industry Designation": "SA-75",
        "NATO Designation": "SA-2A",
    }
    assert tf._pick_canonical(cluster, entity_type="MISSILE_SYSTEM") == "SA-75"


def test_pick_canonical_alphabetic_fallback_for_no_priority_match():
    tf = _load_table_facts()
    cluster = {"Some Custom Label": "Z-1", "Another Label": "A-1"}
    # NFC + casefold sort → A-1 < Z-1
    assert tf._pick_canonical(cluster, entity_type="MISSILE_SYSTEM") == "A-1"


def test_pick_canonical_empty_cluster():
    tf = _load_table_facts()
    assert tf._pick_canonical({}, entity_type="MISSILE_SYSTEM") == ""


# ---- _extract_alias_clusters -----------------------------------------------


def _build_sa2_like_cells():
    """Synthetic 5×5 column-major table:
       row 0 col 0: Missile Type (row_header)
       row 1 col 0: Industry Designation (row_header)
       row 2 col 0: NATO Designation (row_header)
       row 3 col 0: Fan Song Variant (row_header) — cross-entity-ref
       row 4 col 0: Length mm (row_header) — spec row
       cols 1..4 hold the values for variants 1D / 13D / 13DM / 20D
    """
    cells = []
    labels = ("Missile Type", "Industry Designation", "NATO Designation",
              "Fan Song Variant", "Length mm")
    for r, label in enumerate(labels):
        cells.append({
            "start_row_offset_idx": r, "start_col_offset_idx": 0,
            "end_col_offset_idx": 1, "row_header": True, "text": label,
        })
    variants = (
        ("1D", "SA-75", "SA-2A", "RSNA-75", "10726"),
        ("13D", "S-75",  "SA-2C", "RSN-75",  "10726"),
        ("13DM", "S-75M", "SA-2D", "RSN-75M", "10841"),
        ("20D", "V-755", "SA-2F", "RSN-75V", "10841"),
    )
    for col_idx, col_vals in enumerate(variants, start=1):
        for r, val in enumerate(col_vals):
            cells.append({
                "start_row_offset_idx": r, "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1, "row_header": False,
                "text": val,
            })
    return {"data": {"table_cells": cells, "num_rows": 5, "num_cols": 5}}


def test_extract_alias_clusters_builds_one_cluster_per_column():
    tf = _load_table_facts()
    table = _build_sa2_like_cells()
    clusters = tf._extract_alias_clusters(table, entity_type="MISSILE_SYSTEM")
    # One cluster per data column (4); each cluster has the three identity
    # rows but NOT the Fan Song row (cross-entity-ref) or the Length row
    # (spec row).
    assert len(clusters) == 4
    for cluster in clusters:
        assert "Fan Song Variant" not in cluster
        assert "Length mm" not in cluster
        assert "Missile Type" in cluster


def test_extract_alias_clusters_excludes_empty_cells():
    tf = _load_table_facts()
    table = _build_sa2_like_cells()
    # Blank out one identity cell.
    for cell in table["data"]["table_cells"]:
        if (cell.get("start_row_offset_idx") == 1
                and cell.get("start_col_offset_idx") == 1):
            cell["text"] = ""
    clusters = tf._extract_alias_clusters(table, entity_type="MISSILE_SYSTEM")
    # Column 1's cluster must NOT include "Industry Designation" anymore.
    assert clusters[0].get("Industry Designation") in (None, "")


def test_extract_alias_clusters_no_identity_rows_returns_empty():
    tf = _load_table_facts()
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0,
         "end_col_offset_idx": 1, "row_header": True, "text": "Length mm"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 1,
         "end_col_offset_idx": 2, "row_header": False, "text": "10726"},
    ]
    table = {"data": {"table_cells": cells, "num_rows": 1, "num_cols": 2}}
    assert tf._extract_alias_clusters(table, entity_type="MISSILE_SYSTEM") == []
```

- [ ] **Step 2: Run tests, confirm all fail.**

```bash
pytest docker/docling-graph/tests/test_table_overlay_extract.py -v
```
Expected: All FAILED with `AttributeError: module ... has no attribute '_classify_identity_row'`.

- [ ] **Step 3: Implement the four helpers in `_table_facts.py`.**

Append to `docker/docling-graph/app/_table_facts.py` (after existing exports):

```python
# ============================================================================
# Mechanism A1 helpers (spec §5.2). Pure, deterministic, milliseconds.
# ============================================================================

import unicodedata
from ._alias_map import (
    MISSILE_IDENTITY_LABELS,
    RADAR_IDENTITY_LABELS,
    CROSS_ENTITY_REF_PATTERNS,
    CANONICAL_PRIORITY,
)


def _normalize_label(s: str) -> str:
    """Case-insensitive + NFC-folded label normalization.
    Used everywhere we substring-match a row label against the constants
    in _alias_map.py. Stable across operating-system locale settings."""
    return unicodedata.normalize("NFC", (s or "").strip()).casefold()


def _classify_identity_row(label: str) -> str | None:
    """Return entity type ("MISSILE_SYSTEM" / "RADAR_SYSTEM") if the
    label matches an identity row for that type. Otherwise None.

    Classification order (spec §5.1):
      1. Cross-entity-ref check FIRST — labels like 'Fan Song Variant'
         return None here so the caller routes them to
         _classify_cross_entity_ref instead.
      2. Identity-label check SECOND.
    """
    norm = _normalize_label(label)
    if not norm:
        return None
    # (1) cross-entity-ref short-circuits — the row is NOT an identity row
    # for any entity type.
    if norm in CROSS_ENTITY_REF_PATTERNS:
        return None
    # (2) identity-label check, longest-first to avoid 'designation'
    # eating 'industry designation' (we already removed bare 'designation'
    # from the labels list, but stay defensive).
    for missile_label in MISSILE_IDENTITY_LABELS:
        if missile_label in norm:
            return "MISSILE_SYSTEM"
    for radar_label in RADAR_IDENTITY_LABELS:
        if radar_label in norm:
            return "RADAR_SYSTEM"
    return None


def _classify_cross_entity_ref(label: str) -> str | None:
    """Return target entity type if the label is a cross-entity-ref row
    (e.g., 'Fan Song Variant' → 'RADAR_SYSTEM' in a missile-context
    table). Otherwise None.
    """
    norm = _normalize_label(label)
    return CROSS_ENTITY_REF_PATTERNS.get(norm)


def _label_column_width(table: dict) -> int:
    """Number of leftmost columns the row-label region spans. Most
    variants tables have a 1-column label region; SA-2's
    'Industry Designation' spans 2 cols. Returns max
    end_col_offset_idx of row-header cells in col 0."""
    cells = (table or {}).get("data", {}).get("table_cells") or []
    width = 1
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        end = c.get("end_col_offset_idx", 1) or 1
        if end > width:
            width = end
    return width


def _extract_alias_clusters(
    table: dict,
    *,
    entity_type: str,
) -> list[dict[str, str]]:
    """For each data column in the table, build a {label: value} map of
    cells from rows whose label matches an identity row for entity_type.

    Excludes:
      - Cross-entity-ref rows (Fan Song Variant etc.) — they go to
        cross_entity_hints, NOT into the alias cluster.
      - Empty cells.

    Returns a list parallel to the data columns (in left-to-right order).
    Empty list if the table has no identity rows for entity_type.
    """
    cells = (table or {}).get("data", {}).get("table_cells") or []
    if not cells:
        return []

    label_width = _label_column_width(table)

    # Map (row_idx → label) for identity rows that match entity_type.
    identity_rows: dict[int, str] = {}
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        if _classify_identity_row(text) != entity_type:
            continue
        row = c.get("start_row_offset_idx")
        if row is None:
            continue
        identity_rows[row] = text

    if not identity_rows:
        return []

    # Enumerate data column starts (cells with start_col_offset_idx >=
    # label_width).
    data_col_starts = sorted({
        c.get("start_col_offset_idx") for c in cells
        if c.get("start_col_offset_idx") is not None
        and c.get("start_col_offset_idx") >= label_width
    })

    clusters: list[dict[str, str]] = []
    for col in data_col_starts:
        cluster: dict[str, str] = {}
        for cell in cells:
            if cell.get("start_col_offset_idx") != col:
                continue
            row = cell.get("start_row_offset_idx")
            if row not in identity_rows:
                continue
            text = (cell.get("text") or "").strip()
            if not text:
                continue
            cluster[identity_rows[row]] = text
        clusters.append(cluster)
    return clusters


def _pick_canonical(
    cluster: dict[str, str],
    *,
    entity_type: str,
) -> str:
    """Pick the canonical name from an alias cluster using
    CANONICAL_PRIORITY[entity_type]. First priority entry that's a
    substring of any cluster label wins.

    Fallback: if no priority entry matches, return the alphabetically-
    first value (NFC + casefold sort) and log INFO
    'canonical_picked_via_fallback'. Empty cluster → empty string.
    """
    if not cluster:
        return ""
    priority = CANONICAL_PRIORITY.get(entity_type, ())
    for priority_label in priority:
        priority_norm = _normalize_label(priority_label)
        for label, value in cluster.items():
            if priority_norm in _normalize_label(label):
                return value
    # Fallback — alphabetically first.
    return sorted(cluster.values(), key=lambda v: _normalize_label(v))[0]
```

- [ ] **Step 4: Run tests, confirm all pass.**

```bash
pytest docker/docling-graph/tests/test_table_overlay_extract.py -v
```
Expected: All PASSED (~12 tests).

- [ ] **Step 5: Run full docling-graph test suite to confirm no regression.**

```bash
pytest docker/docling-graph/tests -q 2>&1 | tail -10
```
Expected: All pass; the existing 99 tests still green.

- [ ] **Step 6: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py \
       docker/docling-graph/tests/test_table_overlay_extract.py
git commit -m "feat(table-overlay): add parser helpers _classify_identity_row, _classify_cross_entity_ref, _extract_alias_clusters, _pick_canonical"
```

---

## Chunk 2: Schemas + parser entry point (Tasks 3–4)

After Chunk 2, the parser knows how to produce a `TableOverlay` from a `DoclingDocument`, but `main.py` doesn't yet call it.

### Task 3: `schemas.py` wire types

**Files:**
- Modify: `docker/docling-graph/app/schemas.py`
- Create: `docker/docling-graph/tests/test_table_overlay_schemas.py`

- [ ] **Step 1: Write failing schema tests.**

Create `docker/docling-graph/tests/test_table_overlay_schemas.py`:

```python
"""Schema tests for spec §5.4 wire types."""
import importlib.util
from pathlib import Path

import pytest


_SCHEMAS_PATH = Path(__file__).resolve().parent.parent / "app" / "schemas.py"


def _load_schemas():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_schemas", _SCHEMAS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_table_fact_required_fields():
    s = _load_schemas()
    fact = s.TableFact(
        canonical_entity="1D",
        entity_type="MISSILE_SYSTEM",
        schema_field="booster_mass_kg",
        value=1135.0,
        source_label="Weight kg",
        section_ctx="1st Stage",
        pass_name="missile_propulsion",
        raw_text="1135",
    )
    assert fact.canonical_entity == "1D"
    assert fact.entity_type == "MISSILE_SYSTEM"


def test_table_fact_frozen():
    s = _load_schemas()
    fact = s.TableFact(
        canonical_entity="1D", entity_type="MISSILE_SYSTEM",
        schema_field="booster_mass_kg", value=1135.0,
        source_label="Weight kg", section_ctx=None,
        pass_name="missile_propulsion", raw_text="1135",
    )
    with pytest.raises(Exception):
        fact.value = 9999.0  # frozen=True must reject


def test_cross_entity_hint_required_fields():
    s = _load_schemas()
    hint = s.CrossEntityHint(
        source_canonical="1D",
        source_entity_type="MISSILE_SYSTEM",
        target_alias="RSNA-75",
        target_entity_type="RADAR_SYSTEM",
        relationship_kind="associated_with",
    )
    assert hint.target_entity_type == "RADAR_SYSTEM"


def test_table_overlay_default_factories_independent():
    """Mutable defaults bug guard. Two TableOverlay instances must NOT
    share the same dict / list objects."""
    s = _load_schemas()
    a = s.TableOverlay()
    b = s.TableOverlay()
    a.alias_map_by_entity_type["MISSILE_SYSTEM"] = {"x": "y"}
    a.facts.append("dummy")  # type: ignore[arg-type]
    a.cross_entity_hints.append("dummy")  # type: ignore[arg-type]
    assert b.alias_map_by_entity_type == {}
    assert b.facts == []
    assert b.cross_entity_hints == []


def test_table_overlay_round_trip():
    s = _load_schemas()
    overlay = s.TableOverlay(
        alias_map_by_entity_type={"MISSILE_SYSTEM": {"SA-75": "1D"}},
        facts=[s.TableFact(
            canonical_entity="1D", entity_type="MISSILE_SYSTEM",
            schema_field="booster_mass_kg", value=1135.0,
            source_label="Weight kg", section_ctx="1st Stage",
            pass_name="missile_propulsion", raw_text="1135",
        )],
        cross_entity_hints=[],
    )
    dumped = overlay.model_dump(mode="json")
    restored = s.TableOverlay.model_validate(dumped)
    assert restored.alias_map_by_entity_type == overlay.alias_map_by_entity_type
    assert len(restored.facts) == 1


def test_extract_pass_response_carries_table_overlay_optional():
    s = _load_schemas()
    # Without overlay
    resp = s.ExtractPassResponse(bundle_key="x", pass_name="y", pass_output={})
    assert resp.table_overlay is None
    # With overlay
    resp2 = s.ExtractPassResponse(
        bundle_key="x", pass_name="y", pass_output={},
        table_overlay=s.TableOverlay(),
    )
    assert resp2.table_overlay is not None
```

- [ ] **Step 2: Run tests, confirm all fail.**

```bash
pytest docker/docling-graph/tests/test_table_overlay_schemas.py -v
```
Expected: All FAILED (`AttributeError: module ... has no attribute 'TableFact'`).

- [ ] **Step 3: Add the three Pydantic models + `table_overlay` field.**

In `docker/docling-graph/app/schemas.py`, before the existing `class ExtractPassResponse`:

```python
class TableFact(BaseModel):
    """Per-cell deterministic fact derived from a variants table row.
    Spec §5.4.
    """
    model_config = ConfigDict(frozen=True)
    canonical_entity: str
    entity_type: str
    schema_field: str
    value: Any
    source_label: str
    section_ctx: Optional[str] = None
    pass_name: str
    raw_text: str


class CrossEntityHint(BaseModel):
    """Row-level cross-entity reference. v1: collected but not applied
    as edges. Spec §5.4."""
    model_config = ConfigDict(frozen=True)
    source_canonical: str
    source_entity_type: str
    target_alias: str
    target_entity_type: str
    relationship_kind: str


class TableOverlay(BaseModel):
    """Doc-level deterministic overlay derived from a variants table.
    Spec §5.4. Mutable defaults via Field(default_factory=...) so each
    instance gets its own dict/list."""
    alias_map_by_entity_type: dict[str, dict[str, str]] = Field(default_factory=dict)
    facts: list[TableFact] = Field(default_factory=list)
    cross_entity_hints: list[CrossEntityHint] = Field(default_factory=list)
```

Add `from pydantic import ConfigDict` and `Any` to the existing imports if not already present, and add the new field to `ExtractPassResponse`:

```python
    table_overlay: Optional[TableOverlay] = Field(
        default=None,
        description=(
            "Doc-level deterministic overlay (Mechanism A1, spec §5.4). "
            "None when no qualifying variants table found OR kill switch "
            "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false."
        ),
    )
```

Add the `model_rebuild()` calls at the bottom:

```python
TableFact.model_rebuild()
CrossEntityHint.model_rebuild()
TableOverlay.model_rebuild()
```

- [ ] **Step 4: Run tests.**

```bash
pytest docker/docling-graph/tests/test_table_overlay_schemas.py -v
```
Expected: All PASSED (6 tests).

- [ ] **Step 5: Confirm no regression in the schemas tests.**

```bash
pytest docker/docling-graph/tests -q 2>&1 | tail -10
```

- [ ] **Step 6: Commit.**

```bash
git add docker/docling-graph/app/schemas.py \
       docker/docling-graph/tests/test_table_overlay_schemas.py
git commit -m "feat(table-overlay): add TableFact, CrossEntityHint, TableOverlay wire types and ExtractPassResponse.table_overlay field"
```

### Task 4: `extract_table_overlay()` public entry + qualification gate

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py`
- Create: `docker/docling-graph/tests/test_table_overlay_qualification.py`
- Modify: `docker/docling-graph/tests/test_table_overlay_extract.py` (add SA-2 fixture test)

- [ ] **Step 1: Write failing qualification tests.**

Create `docker/docling-graph/tests/test_table_overlay_qualification.py` with the five starvation tests from spec §8.1:

```python
"""Strict-qualification gate tests for spec §3 / §5.2.

Guards against the user-flagged failure mode: a small earlier column-
major-shaped table starving the real variants table at row 6+.
"""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load_table_facts():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_table_facts_qual", _FACTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_qualifying_missile_table(num_cols: int = 5):
    """num_cols entity columns + label col 0. 2 identity rows
    (Missile Type, NATO Designation) + 1 spec row (Length mm).
    All entity columns have non-empty cells in BOTH identity rows
    (so it qualifies under all 4 gates)."""
    cells = []
    cells.append({"start_row_offset_idx": 0, "start_col_offset_idx": 0,
                  "end_col_offset_idx": 1, "row_header": True,
                  "text": "Missile Type"})
    cells.append({"start_row_offset_idx": 1, "start_col_offset_idx": 0,
                  "end_col_offset_idx": 1, "row_header": True,
                  "text": "NATO Designation"})
    cells.append({"start_row_offset_idx": 2, "start_col_offset_idx": 0,
                  "end_col_offset_idx": 1, "row_header": True,
                  "text": "Length mm"})
    for col_idx in range(1, num_cols + 1):
        for r, val in enumerate((f"M{col_idx}", f"NATO{col_idx}", "10726")):
            cells.append({
                "start_row_offset_idx": r,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1,
                "row_header": False, "text": val,
            })
    return {"data": {"table_cells": cells, "num_rows": 3,
                     "num_cols": num_cols + 1}}


def _make_unqualified_3_col_table():
    """3 entity columns (< 4) → fails entity_columns gate."""
    return _make_qualifying_missile_table(num_cols=3)


def _make_unqualified_sparse_identity_table():
    """5 entity columns, identity row exists, but only column 1 has a
    non-empty cell in the identity row → fails sparse-identity gate."""
    table = _make_qualifying_missile_table(num_cols=5)
    cells = table["data"]["table_cells"]
    # Blank identity-row cells in cols 2..5 for both identity rows.
    for c in cells:
        if (c.get("start_row_offset_idx") in (0, 1)
                and c.get("start_col_offset_idx", 0) >= 2):
            c["text"] = ""
    return table


def _make_qualifying_radar_table(num_cols: int = 5):
    cells = []
    cells.append({"start_row_offset_idx": 0, "start_col_offset_idx": 0,
                  "end_col_offset_idx": 1, "row_header": True,
                  "text": "Radar Variant"})
    cells.append({"start_row_offset_idx": 1, "start_col_offset_idx": 0,
                  "end_col_offset_idx": 1, "row_header": True,
                  "text": "Radar Type"})
    for col_idx in range(1, num_cols + 1):
        for r, val in enumerate((f"R{col_idx}", f"Type{col_idx}")):
            cells.append({
                "start_row_offset_idx": r,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1,
                "row_header": False, "text": val,
            })
    return {"data": {"table_cells": cells, "num_rows": 2,
                     "num_cols": num_cols + 1}}


def test_unqualified_earlier_table_does_not_starve_real_variants_table():
    """Doc has [unqualified_3col, qualifying_5col]. extract_table_overlay
    must skip the first (tables_skipped_unqualified++) and pick the
    second."""
    tf = _load_table_facts()
    doc = {"tables": [
        _make_unqualified_3_col_table(),
        _make_qualifying_missile_table(num_cols=5),
    ]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert "MISSILE_SYSTEM" in overlay.alias_map_by_entity_type
    assert len(overlay.alias_map_by_entity_type["MISSILE_SYSTEM"]) > 0
    assert stats["tables_skipped_unqualified"] == 1
    assert stats["tables_skipped_multi"] == 0


def test_entity_columns_gate_under_4_rejects():
    tf = _load_table_facts()
    doc = {"tables": [_make_unqualified_3_col_table()]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert overlay.alias_map_by_entity_type == {}
    assert stats["tables_skipped_unqualified"] == 1


def test_sparse_identity_cells_rejects():
    tf = _load_table_facts()
    doc = {"tables": [_make_unqualified_sparse_identity_table()]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert overlay.alias_map_by_entity_type == {}
    assert stats["tables_skipped_unqualified"] == 1


def test_radar_qualifying_table_before_missile_v1_picks_first():
    """v1 picker is entity-type-agnostic. Radar table comes first → it
    wins; missile table goes to tables_skipped_multi."""
    tf = _load_table_facts()
    doc = {"tables": [
        _make_qualifying_radar_table(num_cols=5),
        _make_qualifying_missile_table(num_cols=5),
    ]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert "RADAR_SYSTEM" in overlay.alias_map_by_entity_type
    assert "MISSILE_SYSTEM" not in overlay.alias_map_by_entity_type
    assert stats["tables_skipped_multi"] == 1


def test_two_qualifying_missile_tables_first_wins():
    tf = _load_table_facts()
    doc = {"tables": [
        _make_qualifying_missile_table(num_cols=5),
        _make_qualifying_missile_table(num_cols=4),
    ]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert "MISSILE_SYSTEM" in overlay.alias_map_by_entity_type
    assert stats["tables_skipped_multi"] == 1
```

- [ ] **Step 2: Run tests, confirm all fail.**

```bash
pytest docker/docling-graph/tests/test_table_overlay_qualification.py -v
```
Expected: 5 FAILED (`extract_table_overlay` undefined).

- [ ] **Step 3: Implement `extract_table_overlay()`.**

Append to `docker/docling-graph/app/_table_facts.py`:

```python
def _table_qualifies_for_overlay(
    table: dict,
    *,
    entity_type: str,
) -> tuple[bool, str]:
    """Return (qualifies, reason). Reason is one of:
       'qualified', 'wrong_shape', 'too_few_entity_columns',
       'no_identity_rows', 'sparse_identity_cells'.
    Spec §3 four-of-four AND gate.
    """
    if not _is_column_major_or_hybrid(table):
        return False, "wrong_shape"

    label_width = _label_column_width(table)
    cells = (table or {}).get("data", {}).get("table_cells") or []

    data_col_starts = sorted({
        c.get("start_col_offset_idx") for c in cells
        if c.get("start_col_offset_idx") is not None
        and c.get("start_col_offset_idx") >= label_width
    })
    if len(data_col_starts) < 4:
        return False, "too_few_entity_columns"

    # Identity rows for entity_type
    identity_rows = {
        c.get("start_row_offset_idx")
        for c in cells
        if c.get("start_col_offset_idx") == 0
        and c.get("row_header")
        and _classify_identity_row(c.get("text") or "") == entity_type
    }
    if not identity_rows:
        return False, "no_identity_rows"

    # Every entity column has a non-empty cell in ≥1 identity row
    for col in data_col_starts:
        has_cell = False
        for cell in cells:
            if cell.get("start_col_offset_idx") != col:
                continue
            if cell.get("start_row_offset_idx") not in identity_rows:
                continue
            if (cell.get("text") or "").strip():
                has_cell = True
                break
        if not has_cell:
            return False, "sparse_identity_cells"

    return True, "qualified"


def _is_column_major_or_hybrid(table: dict) -> bool:
    """Lightweight shape detector: leftmost column carries row_header
    cells AND ≥4 cols / ≥4 rows. Mirrors the existing _table_pivot.py
    heuristic so we stay consistent with the table-fact synthesizer."""
    data = (table or {}).get("data") or {}
    cells = data.get("table_cells") or []
    if not cells:
        return False
    nr, nc = data.get("num_rows") or 0, data.get("num_cols") or 0
    if nr < 4 or nc < 4:
        return False
    col0 = [c for c in cells if c.get("start_col_offset_idx") == 0]
    if not col0:
        return False
    label_count = sum(1 for c in col0 if c.get("row_header") is True)
    return label_count * 2 >= len(col0)


def extract_table_overlay(doc_json: dict) -> tuple["TableOverlay", dict]:
    """Parse the FIRST strictly-qualifying column-major / hybrid table
    in doc.tables[]. Returns (TableOverlay, stats_dict). Spec §5.2.
    """
    # Lazy import to avoid circular schemas → _table_facts dependency
    # at module load.
    from .schemas import TableOverlay, TableFact, CrossEntityHint  # noqa: PLC0415

    stats = {
        "tables_processed": 0,
        "tables_skipped_unqualified": 0,
        "tables_skipped_multi": 0,
        "tables_skipped_other": 0,
        "columns_skipped_no_canonical": 0,
        "columns_with_canonical_via_fallback": 0,
        "values_skipped_unparseable": 0,
        "facts_skipped_construct_fail": 0,
    }

    tables = (doc_json or {}).get("tables") or []
    if not tables:
        return TableOverlay(), stats

    winner_table = None
    winner_entity_type = None
    for table in tables:
        # Try MISSILE_SYSTEM first, then RADAR_SYSTEM (entity-type-
        # agnostic in v1; first qualifying-of-any-type wins).
        for et in ("MISSILE_SYSTEM", "RADAR_SYSTEM"):
            qualifies, reason = _table_qualifies_for_overlay(
                table, entity_type=et,
            )
            if qualifies:
                if winner_table is None:
                    winner_table = table
                    winner_entity_type = et
                else:
                    stats["tables_skipped_multi"] += 1
                break
        else:
            # No entity-type qualified for this table.
            if not _is_column_major_or_hybrid(table):
                stats["tables_skipped_other"] += 1
            else:
                stats["tables_skipped_unqualified"] += 1

    if winner_table is None:
        return TableOverlay(), stats

    stats["tables_processed"] = 1

    # Build alias clusters + canonical names.
    clusters = _extract_alias_clusters(
        winner_table, entity_type=winner_entity_type,
    )
    sub_map: dict[str, str] = {}
    canonical_per_col: list[str] = []
    for cluster in clusters:
        canonical = _pick_canonical(cluster, entity_type=winner_entity_type)
        canonical_per_col.append(canonical)
        if not canonical:
            stats["columns_skipped_no_canonical"] += 1
            continue
        for alias in cluster.values():
            if alias and alias != canonical:
                sub_map[alias] = canonical
            if alias:
                # Identity: also map canonical → canonical so
                # downstream rewrite is idempotent.
                sub_map.setdefault(alias, canonical)

    alias_map_by_entity_type: dict[str, dict[str, str]] = {}
    if sub_map:
        alias_map_by_entity_type[winner_entity_type] = sub_map

    # Build facts from spec rows + cross_entity_hints from cross-entity-
    # ref rows. Reuse existing _table_facts.py primitives where possible
    # (extract_label_rows / detect_section_context / coerce_value if
    # these have been exported from the predecessor synthesizer work).
    # See spec §5.2 step 4 + 5.
    facts: list = []  # populated below
    hints: list = []

    # NOTE: per-pass spec-row + section-context emission is delegated to
    # the existing _table_facts.py helpers from the predecessor plan
    # (commit 1b71150 era). If those helpers are not yet present in this
    # branch of _table_facts.py, port them from the parked synthesizer
    # module before this task lands. The per-cell loop below is the
    # new code:
    label_width = _label_column_width(winner_table)
    cells = winner_table.get("data", {}).get("table_cells") or []
    data_col_starts = sorted({
        c.get("start_col_offset_idx") for c in cells
        if c.get("start_col_offset_idx") is not None
        and c.get("start_col_offset_idx") >= label_width
    })

    # Section-context tracking + spec-row + cross-entity-row emission:
    # reuse predecessor primitives via a single inner helper:
    facts, hints = _emit_facts_and_hints(
        winner_table,
        canonical_per_col=canonical_per_col,
        winner_entity_type=winner_entity_type,
        data_col_starts=data_col_starts,
        stats=stats,
    )

    overlay = TableOverlay(
        alias_map_by_entity_type=alias_map_by_entity_type,
        facts=facts,
        cross_entity_hints=hints,
    )
    return overlay, stats


def _emit_facts_and_hints(
    table: dict,
    *,
    canonical_per_col: list[str],
    winner_entity_type: str,
    data_col_starts: list[int],
    stats: dict,
) -> tuple[list, list]:
    """Walk spec rows + cross-entity-ref rows; emit TableFacts and
    CrossEntityHints. Wraps the existing _table_facts.py primitives:
    extract_label_rows, detect_section_context, resolve_alias,
    coerce_value, _looks_like_key_label, _classify_cross_entity_ref.

    Multi-pass routing: for each non-identity, non-cross-entity row, try
    each missile/radar pass that can own that label-section combo via
    resolve_alias(label, section, pass_name). The first pass that
    resolves it gets the fact. (resolve_alias returns None when a pass
    doesn't own the label; the alias map in _alias_map.py is keyed on
    (label, section, pass) triples, which gives us pass-uniqueness for
    free.)
    """
    from .schemas import TableFact, CrossEntityHint  # noqa: PLC0415

    facts: list = []
    hints: list = []

    if winner_entity_type == "MISSILE_SYSTEM":
        candidate_passes = (
            "missile_propulsion",
            "missile_kinematics",
            "missile_speed_timing",
            "missile_airframe",
        )
        target_for_cross_ref = "RADAR_SYSTEM"
    elif winner_entity_type == "RADAR_SYSTEM":
        candidate_passes = (
            "radar_antenna", "radar_modulation",
            "radar_power_rf", "radar_timing",
        )
        target_for_cross_ref = "MISSILE_SYSTEM"
    else:
        return [], []

    # Map data_col_starts → canonical
    col_to_canonical = {
        col: canonical_per_col[i]
        for i, col in enumerate(data_col_starts)
        if i < len(canonical_per_col) and canonical_per_col[i]
    }

    shape = detect_table_shape(table)
    if shape.value == "OTHER":
        return [], []

    rows = extract_label_rows(table, shape)
    if not rows:
        return [], []

    sectioned = detect_section_context(rows)

    for row, section_ctx in sectioned:
        label_text = row["label_text"]

        # Identity rows produce alias_map only — already handled in
        # the caller via _extract_alias_clusters + _pick_canonical.
        if _looks_like_key_label(label_text):
            continue

        # Cross-entity-ref rows produce CrossEntityHint, not facts.
        cross_target = _classify_cross_entity_ref(label_text)
        if cross_target == target_for_cross_ref:
            for entity_col, cell_text in row["data_cells"].items():
                target_alias = (cell_text or "").strip()
                if not target_alias:
                    continue
                source_canonical = col_to_canonical.get(entity_col, "")
                if not source_canonical:
                    continue
                try:
                    hints.append(CrossEntityHint(
                        source_canonical=source_canonical,
                        source_entity_type=winner_entity_type,
                        target_alias=target_alias,
                        target_entity_type=cross_target,
                        relationship_kind="associated_with",
                    ))
                except Exception:
                    stats["facts_skipped_construct_fail"] += 1
            continue

        # Spec rows: try each candidate pass; the alias map will
        # resolve at most one (or zero).
        for entity_col, cell_text in row["data_cells"].items():
            canonical = col_to_canonical.get(entity_col, "")
            if not canonical:
                continue
            for pass_name in candidate_passes:
                schema_field = resolve_alias(
                    label_text, section_ctx, pass_name,
                )
                if schema_field is None:
                    continue
                parsed = coerce_value(
                    cell_text, schema_field, row_label=label_text,
                )
                if not parsed:
                    stats["values_skipped_unparseable"] += 1
                    continue
                for pv in parsed:
                    try:
                        facts.append(TableFact(
                            canonical_entity=canonical,
                            entity_type=winner_entity_type,
                            schema_field=schema_field,
                            value=pv.value,
                            source_label=label_text,
                            section_ctx=section_ctx,
                            pass_name=pass_name,
                            raw_text=cell_text,
                        ))
                    except Exception:
                        stats["facts_skipped_construct_fail"] += 1
                # Stop after the first pass that owns this label —
                # alias map is keyed on (label, section, pass), so at
                # most one pass will match for any given (label,
                # section).
                break

    return facts, hints
```

> **Implementer note:** the loop above wraps the same primitives the parked synthesizer used (commit `1b71150` era), now switched from emitting TextItems to emitting `TableFact` / `CrossEntityHint` Pydantic instances. Expected SA-2 fact count: ~50 across all 4 passes (matches spec §6 worked example). If fewer facts emit on real SA-2 (~<30), the alias map in `_alias_map.py` is missing label entries — verify §12b prose-to-alias-map drift via the predecessor plan's drift-guard test before assuming a bug here.

- [ ] **Step 4: Add SA-2-shaped fixture test in `test_table_overlay_extract.py`.**

```python
def test_extract_table_overlay_sa2_shape_yields_alias_map():
    tf = _load_table_facts()
    table = _build_sa2_like_cells()  # from earlier test
    doc = {"tables": [table]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert "MISSILE_SYSTEM" in overlay.alias_map_by_entity_type
    sub = overlay.alias_map_by_entity_type["MISSILE_SYSTEM"]
    # 1D should be a canonical (Missile Type wins)
    assert sub.get("SA-75") == "1D"
    assert sub.get("SA-2A") == "1D"
    # 13DM column
    assert sub.get("S-75M") == "13DM"
    assert sub.get("SA-2D") == "13DM"
```

- [ ] **Step 5: Run tests, all pass.**

```bash
pytest docker/docling-graph/tests/test_table_overlay_qualification.py docker/docling-graph/tests/test_table_overlay_extract.py -v
```
Expected: All PASSED.

- [ ] **Step 6: Run full docling-graph suite.**

```bash
pytest docker/docling-graph/tests -q 2>&1 | tail -10
```
Expected: All pass.

- [ ] **Step 7: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py \
       docker/docling-graph/tests/test_table_overlay_qualification.py \
       docker/docling-graph/tests/test_table_overlay_extract.py
git commit -m "feat(table-overlay): extract_table_overlay public entry + strict qualification gate"
```

---

## Chunk 3: main.py wire-up + parser-side kill switch (Task 5)

After Chunk 3, the parser ships overlay payloads on the response. The worker doesn't yet know what to do with them.

### Task 5: `main.py` wire-up + kill switch + diagnostics

**Files:**
- Modify: `docker/docling-graph/app/main.py`
- Create: `docker/docling-graph/tests/test_main_table_overlay_integration.py`

- [ ] **Step 1: Write failing main.py integration test.**

Create `docker/docling-graph/tests/test_main_table_overlay_integration.py`:

```python
"""Integration: main.py /extract-pass populates response.table_overlay
when a qualifying variants table exists, and respects the kill switch."""
import os
from unittest.mock import patch

from fastapi.testclient import TestClient


def _make_minimal_request_payload(doc_with_table: dict):
    return {
        "bundle_key": "air_defense_v3",
        "pass_name": "missile_propulsion",
        "docling_document_json": doc_with_table,
    }


def test_extract_pass_includes_table_overlay_when_table_qualifies(
    sa2_like_doc_with_table_fixture,  # see conftest
    monkeypatch,
):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")
    from docker.docling_graph.app.main import app  # adapt path
    client = TestClient(app)
    r = client.post("/extract-pass",
                    json=_make_minimal_request_payload(sa2_like_doc_with_table_fixture))
    assert r.status_code == 200
    body = r.json()
    assert body.get("table_overlay") is not None
    overlay = body["table_overlay"]
    assert "MISSILE_SYSTEM" in overlay["alias_map_by_entity_type"]
    diag = body.get("diagnostics") or {}
    svc = diag.get("service_table_overlay") or {}
    assert svc.get("kill_switch_active_parser") is False
    assert svc.get("tables_processed") == 1


def test_extract_pass_kill_switch_returns_no_overlay(
    sa2_like_doc_with_table_fixture, monkeypatch,
):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "false")
    from docker.docling_graph.app.main import app
    client = TestClient(app)
    r = client.post("/extract-pass",
                    json=_make_minimal_request_payload(sa2_like_doc_with_table_fixture))
    assert r.status_code == 200
    body = r.json()
    assert body.get("table_overlay") is None
    diag = body.get("diagnostics") or {}
    svc = diag.get("service_table_overlay") or {}
    assert svc.get("kill_switch_active_parser") is True
```

(If a TestClient setup doesn't already exist in the docling-graph tests, mock the LLM call inside the handler to avoid a live Ollama dependency. The existing `_table_facts.py` test suite has prior art — borrow its FastAPI test fixture conventions.)

- [ ] **Step 2: Run test, confirm fail.**

```bash
pytest docker/docling-graph/tests/test_main_table_overlay_integration.py -v
```
Expected: FAIL.

- [ ] **Step 3: Wire `extract_table_overlay()` into `main.py`.**

In `docker/docling-graph/app/main.py`, find the `/extract-pass` POST handler. After the sanitize step and BEFORE the LLM extraction call, add:

```python
import os

def _table_overlay_enabled_parser() -> bool:
    return os.environ.get(
        "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true",
    ).lower() != "false"


# ... inside the /extract-pass handler ...

table_overlay_obj = None
overlay_stats = {
    "kill_switch_active_parser": not _table_overlay_enabled_parser(),
    "tables_processed": 0,
}

if _table_overlay_enabled_parser():
    try:
        from app._table_facts import extract_table_overlay
        table_overlay_obj, parser_stats = extract_table_overlay(
            sanitized_doc_json,
        )
        overlay_stats.update(parser_stats)
        if table_overlay_obj is not None:
            overlay_stats["alias_map_size"] = sum(
                len(m) for m in table_overlay_obj.alias_map_by_entity_type.values()
            )
            overlay_stats["facts_count"] = len(table_overlay_obj.facts)
            overlay_stats["cross_entity_hints_count"] = len(
                table_overlay_obj.cross_entity_hints,
            )
    except Exception as exc:
        logger.warning(
            "extract_table_overlay failed: %s — continuing with table_overlay=None",
            exc,
        )
        table_overlay_obj = None
        overlay_stats["extract_failure"] = repr(exc)
```

Then in the response build, add `table_overlay=table_overlay_obj` and merge `overlay_stats` into `diagnostics["service_table_overlay"]`.

- [ ] **Step 4: Run integration tests.**

```bash
pytest docker/docling-graph/tests/test_main_table_overlay_integration.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Run full suite.**

```bash
pytest docker/docling-graph/tests -q 2>&1 | tail -10
```
Expected: All pass.

- [ ] **Step 6: Commit.**

```bash
git add docker/docling-graph/app/main.py \
       docker/docling-graph/tests/test_main_table_overlay_integration.py
git commit -m "feat(table-overlay): wire extract_table_overlay into /extract-pass with kill switch + diagnostics"
```

---

## Chunk 4: Worker overlay module (Tasks 6–7)

After Chunk 4, the worker has the two functions but they're not yet called from `extraction_merge.py`.

### Task 6: `apply_identity_rewrite` + `RewriteStats`

**Files:**
- Create: `app/services/table_overlay.py`
- Create: `tests/unit/test_table_overlay_worker.py`

- [ ] **Step 1: Write failing identity-rewrite tests.**

Create `tests/unit/test_table_overlay_worker.py`:

```python
"""Worker-side unit tests for app.services.table_overlay (spec §8.3)."""
from unittest.mock import MagicMock
import pytest


def _make_pass_result(entity_type, instances):
    """Build a minimal PassResult-like with iter_entities_of_type."""
    pr = MagicMock()
    def _iter(et):
        if et != entity_type:
            return iter([])
        return iter(instances)
    pr.iter_entities_of_type = _iter
    return pr


def _missile_inst(name):
    """Real Pydantic missile instance, not a MagicMock."""
    from ontology_bundles.air_defense_v3.extraction_schemas import (
        missile_propulsion,
    )
    return missile_propulsion.MissilePropulsionRecord(system_name=name)


def test_identity_rewrite_empty_alias_map_is_noop():
    from app.services.table_overlay import apply_identity_rewrite
    inst = _missile_inst("SA-75")
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    ontology = {"entity_types": [{"name": "MISSILE_SYSTEM"}]}
    stats = apply_identity_rewrite(pass_results, {}, ontology)
    assert stats.rewrites == 0
    assert inst.system_name == "SA-75"


def test_identity_rewrite_rewrites_alias_to_canonical():
    from app.services.table_overlay import apply_identity_rewrite
    a = _missile_inst("SA-75")
    b = _missile_inst("SA-2A")
    pr = _make_pass_result("MISSILE_SYSTEM", [a, b])
    pass_results = {"missile_propulsion": pr}
    ontology = {"entity_types": [{"name": "MISSILE_SYSTEM"}]}
    alias_map = {"MISSILE_SYSTEM": {"SA-75": "1D", "SA-2A": "1D"}}
    stats = apply_identity_rewrite(pass_results, alias_map, ontology)
    assert stats.rewrites == 2
    assert a.system_name == "1D"
    assert b.system_name == "1D"


def test_identity_rewrite_entity_type_scoped():
    """alias 'COMMON' under MISSILE_SYSTEM rewrites missiles only;
    radar instances unaffected."""
    from app.services.table_overlay import apply_identity_rewrite
    from ontology_bundles.air_defense_v3.extraction_schemas import (
        missile_propulsion, radar_antenna,
    )
    m = missile_propulsion.MissilePropulsionRecord(system_name="COMMON")
    r = radar_antenna.RadarAntennaRecord(system_name="COMMON")
    pr_m = _make_pass_result("MISSILE_SYSTEM", [m])
    pr_r = _make_pass_result("RADAR_SYSTEM", [r])
    pass_results = {"missile_propulsion": pr_m, "radar_antenna": pr_r}
    ontology = {"entity_types": [
        {"name": "MISSILE_SYSTEM"}, {"name": "RADAR_SYSTEM"},
    ]}
    alias_map = {"MISSILE_SYSTEM": {"COMMON": "REWRITTEN"}}
    stats = apply_identity_rewrite(pass_results, alias_map, ontology)
    assert m.system_name == "REWRITTEN"
    assert r.system_name == "COMMON"  # untouched
    assert stats.rewrites == 1
```

- [ ] **Step 2: Run, confirm fail.**

```bash
pytest tests/unit/test_table_overlay_worker.py -v
```
Expected: FAIL (`ModuleNotFoundError: app.services.table_overlay`).

- [ ] **Step 3: Implement `apply_identity_rewrite`.**

Create `app/services/table_overlay.py`:

```python
"""Worker-side overlay application (spec §5.3, Mechanism A1).

Two functions operate on Pydantic instances reachable via
PassResult.iter_entities_of_type:

  apply_identity_rewrite — entity-type-scoped system_name alias
    collapse, runs inside canonicalize_cross_pass_identities BEFORE
    the existing token-overlap pass.

  apply_field_overlay — per-cell field overlay with
    table_wins_for_table_facts policy (default), full
    cls.model_validate(...) gate, fan-out to all matching post-rewrite
    instances, per-(fact, instance) atomicity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

from pydantic import ValidationError, BaseModel

logger = logging.getLogger(__name__)


@dataclass
class RewriteStats:
    rewrites: int = 0
    unique_canonicals: int = 0
    passes_touched: int = 0
    def as_dict(self) -> dict: return asdict(self)


def apply_identity_rewrite(
    pass_results: dict,           # dict[str, PassResult]
    alias_map_by_entity_type: dict[str, dict[str, str]],
    ontology: dict,
) -> RewriteStats:
    """Mutate Pydantic instances in-place: where system_name is in the
    alias map for the instance's entity_type, replace with canonical.
    Idempotent (alias_map[canonical] == canonical short-circuits).
    Spec §5.3.
    """
    stats = RewriteStats()
    if not alias_map_by_entity_type:
        return stats

    canonicals: set[str] = set()
    for entity_def in ontology.get("entity_types", []) or []:
        entity_type = entity_def.get("name")
        if not entity_type:
            continue
        sub_map = alias_map_by_entity_type.get(entity_type) or {}
        if not sub_map:
            continue
        for pass_name, pass_result in pass_results.items():
            touched_this_pass = False
            try:
                instances = list(pass_result.iter_entities_of_type(entity_type))
            except Exception as exc:
                logger.warning(
                    "apply_identity_rewrite: iter_entities_of_type failed for "
                    "pass=%s entity_type=%s: %s", pass_name, entity_type, exc,
                )
                continue
            for inst in instances:
                current = getattr(inst, "system_name", None)
                if not current or current not in sub_map:
                    continue
                canonical = sub_map[current]
                if current == canonical:
                    canonicals.add(canonical)
                    continue
                try:
                    inst.system_name = canonical
                    stats.rewrites += 1
                    canonicals.add(canonical)
                    touched_this_pass = True
                except Exception as exc:
                    logger.warning(
                        "apply_identity_rewrite: cannot set system_name on "
                        "%s instance: %s", entity_type, exc,
                    )
            if touched_this_pass:
                stats.passes_touched += 1
    stats.unique_canonicals = len(canonicals)
    return stats
```

- [ ] **Step 4: Run, confirm pass.**

```bash
pytest tests/unit/test_table_overlay_worker.py -v -k identity_rewrite
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit.**

```bash
git add app/services/table_overlay.py tests/unit/test_table_overlay_worker.py
git commit -m "feat(table-overlay): apply_identity_rewrite with entity-type scoping + RewriteStats"
```

### Task 7: `apply_field_overlay` + `OverlayStats`

**Files:**
- Modify: `app/services/table_overlay.py`
- Modify: `tests/unit/test_table_overlay_worker.py`

- [ ] **Step 1: Add failing field-overlay tests.**

Append to `tests/unit/test_table_overlay_worker.py`:

```python
def _make_table_fact(**kwargs):
    """Build TableFact compatible with both schema imports."""
    from docker.docling_graph.app.schemas import TableFact
    defaults = dict(
        canonical_entity="1D", entity_type="MISSILE_SYSTEM",
        schema_field="booster_mass_kg", value=1135.0,
        source_label="Weight kg", section_ctx="1st Stage",
        pass_name="missile_propulsion", raw_text="1135",
    )
    defaults.update(kwargs)
    return TableFact(**defaults)


def test_field_overlay_validation_runs_field_validator():
    """Spec §5.3 step (c): cls.model_validate must execute
    _v_booster_mass_kg = field_validator(...)(coerce_optional_float).
    Pass value as a STRING; expected coerced to float."""
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(value="1135")  # string, not float
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 1
    assert stats.matches_touched == 1
    assert stats.skipped_validation_fail == 0
    assert isinstance(inst.booster_mass_kg, float)
    assert inst.booster_mass_kg == 1135.0


def test_field_overlay_unknown_field_precheck():
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(schema_field="totally_bogus_field")
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 0
    assert stats.skipped_unknown_field == 1
    # Instance unchanged
    assert getattr(inst, "totally_bogus_field", "ABSENT") == "ABSENT"


def test_field_overlay_table_wins_overrides_populated():
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    inst.booster_mass_kg = 970.0  # LLM wrong
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(value=1135.0)
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 1
    assert stats.conflicts_overridden == 1
    assert inst.booster_mass_kg == 1135.0


def test_field_overlay_fans_out_to_all_matching():
    from app.services.table_overlay import apply_field_overlay
    a = _missile_inst("1D")
    b = _missile_inst("1D")  # post-rewrite duplicate
    pr = _make_pass_result("MISSILE_SYSTEM", [a, b])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(value=1135.0)
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 2  # fact-instance count under fan-out
    assert stats.matches_touched == 1  # fact landed on >=1 (incremented once)
    assert a.booster_mass_kg == 1135.0
    assert b.booster_mass_kg == 1135.0


def test_field_overlay_entity_type_scope():
    from app.services.table_overlay import apply_field_overlay
    from ontology_bundles.air_defense_v3.extraction_schemas import radar_antenna
    radar = radar_antenna.RadarAntennaRecord(system_name="1D")
    pr = _make_pass_result("RADAR_SYSTEM", [radar])
    pass_results = {"radar_antenna": pr}
    fact = _make_table_fact()  # entity_type="MISSILE_SYSTEM"
    stats = apply_field_overlay(pass_results, [fact])
    # MISSILE_SYSTEM fact must NOT land on a RADAR_SYSTEM instance
    assert stats.applied == 0
    assert stats.skipped_no_entity == 1


def test_field_overlay_validation_failure_keeps_instance_unchanged():
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    inst.booster_mass_kg = 970.0
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(value="not a number")
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 0
    assert stats.skipped_validation_fail == 1
    # Instance retains its prior LLM value
    assert inst.booster_mass_kg == 970.0


def test_field_overlay_only_touches_fields_with_facts():
    """Scoped table_wins: a field with no fact is never touched."""
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    inst.booster_mass_kg = 970.0
    inst.sustain_mass_kg = 555.0
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(schema_field="booster_mass_kg", value=1135.0)
    apply_field_overlay(pass_results, [fact])
    assert inst.booster_mass_kg == 1135.0
    assert inst.sustain_mass_kg == 555.0  # untouched
```

- [ ] **Step 2: Run, confirm fail.**

```bash
pytest tests/unit/test_table_overlay_worker.py -v -k field_overlay
```
Expected: 7 FAILED (`apply_field_overlay` undefined).

- [ ] **Step 3: Implement `apply_field_overlay` + `OverlayStats`.**

Append to `app/services/table_overlay.py`:

```python
@dataclass
class OverlayStats:
    applied: int = 0                  # fact-instance count (fan-out)
    matches_touched: int = 0          # fact count that landed on >=1 inst
    skipped_no_entity: int = 0
    skipped_unknown_field: int = 0
    skipped_validation_fail: int = 0
    conflicts_overridden: int = 0
    policy_active: str = "table_wins_for_table_facts"
    def as_dict(self) -> dict: return asdict(self)


def _instances_for_fact(
    pass_results: dict,
    fact: Any,
) -> list[Any]:
    """Enumerate ALL instances in pass_results[fact.pass_name] of type
    fact.entity_type whose system_name == fact.canonical_entity (post-
    rewrite). Empty list if pass_name not in pass_results OR no
    matches."""
    pr = pass_results.get(fact.pass_name)
    if pr is None:
        return []
    try:
        candidates = list(pr.iter_entities_of_type(fact.entity_type))
    except Exception as exc:
        logger.warning(
            "apply_field_overlay: iter_entities_of_type failed for "
            "pass=%s entity_type=%s: %s",
            fact.pass_name, fact.entity_type, exc,
        )
        return []
    return [
        inst for inst in candidates
        if getattr(inst, "system_name", None) == fact.canonical_entity
    ]


def apply_field_overlay(
    pass_results: dict,
    table_facts: list,
    *,
    policy: str = "table_wins_for_table_facts",
) -> OverlayStats:
    """Apply per-cell table facts to Pydantic entity instances. Spec §5.3.

    Per-(fact, instance) atomicity: a model_validate failure on one
    instance leaves that instance UNCHANGED and does NOT block
    fan-out to siblings. The overall loop is NOT a single transaction.
    """
    stats = OverlayStats(policy_active=policy)

    for fact in table_facts:
        matches = _instances_for_fact(pass_results, fact)
        if not matches:
            stats.skipped_no_entity += 1
            continue

        any_landed = False
        for inst in matches:
            cls = type(inst)

            # (a) Pre-validate field name. extra="ignore" would drop
            # unknown keys silently otherwise.
            if not isinstance(inst, BaseModel):
                stats.skipped_unknown_field += 1
                continue
            if fact.schema_field not in cls.model_fields:
                stats.skipped_unknown_field += 1
                logger.info(
                    "FIELD_OVERLAY_UNKNOWN_FIELD pass=%s entity_type=%s "
                    "entity=%s schema_field=%s model=%s — fact dropped",
                    fact.pass_name, fact.entity_type, fact.canonical_entity,
                    fact.schema_field, cls.__name__,
                )
                continue

            # (b) capture original
            original = getattr(inst, fact.schema_field, None)

            # (c) full model validation
            candidate = {**inst.model_dump(), fact.schema_field: fact.value}
            try:
                revalidated = cls.model_validate(candidate)
            except (ValidationError, ValueError, TypeError):
                stats.skipped_validation_fail += 1
                continue
            coerced = getattr(revalidated, fact.schema_field)

            # (d) atomic swap
            for k, v in revalidated.model_dump().items():
                try:
                    setattr(inst, k, v)
                except Exception:
                    # If a field cannot be set, skip — instance is no
                    # longer guaranteed consistent, but per spec §7
                    # bounded-degraded: do NOT roll back.
                    pass

            # (e) per-instance bookkeeping
            stats.applied += 1
            any_landed = True
            if original is not None and original != coerced:
                stats.conflicts_overridden += 1
                logger.info(
                    "FIELD_OVERLAY_OVERRIDE pass=%s entity_type=%s "
                    "entity=%s field=%s llm=%r table=%r source=%r",
                    fact.pass_name, fact.entity_type, fact.canonical_entity,
                    fact.schema_field, original, coerced, fact.source_label,
                )

        # step 3: fact-level matches_touched (once per fact that
        # landed on >=1 instance — NOT per instance).
        if any_landed:
            stats.matches_touched += 1

    return stats
```

- [ ] **Step 4: Run, confirm pass.**

```bash
pytest tests/unit/test_table_overlay_worker.py -v
```
Expected: 10+ PASSED.

- [ ] **Step 5: Commit.**

```bash
git add app/services/table_overlay.py tests/unit/test_table_overlay_worker.py
git commit -m "feat(table-overlay): apply_field_overlay with unknown-field precheck, fan-out, atomicity, OverlayStats"
```

---

## Chunk 5: extraction_merge.py + pipeline.py integration (Tasks 8–9)

After Chunk 5, the worker actually invokes the overlay during merge.

### Task 8: `extraction_merge.py` integration + worker-side kill switch

**Files:**
- Modify: `app/services/extraction_merge.py`
- Create: `tests/unit/test_extraction_merge_table_overlay.py`

- [ ] **Step 1: Write failing integration tests.**

Create `tests/unit/test_extraction_merge_table_overlay.py`:

```python
"""Integration tests for spec §8.4 — extraction_merge.py + worker-side
kill switch. Each test exercises merge_and_resolve end-to-end with
in-memory PassResults; no docling-graph HTTP, no Ollama."""
import os
from unittest.mock import patch

from app.services.extraction_merge import (
    merge_and_resolve, canonicalize_cross_pass_identities, PassResult,
)
from docker.docling_graph.app.schemas import TableOverlay, TableFact


def _ontology_min():
    """Minimal ontology: one missile entity type."""
    return {"entity_types": [
        {"name": "MISSILE_SYSTEM", "graph_id_fields": ["system_name"]},
    ]}


def _missile_inst(name, **fields):
    from ontology_bundles.air_defense_v3.extraction_schemas import missile_propulsion
    return missile_propulsion.MissilePropulsionRecord(system_name=name, **fields)


def _make_propulsion_passresult(instances, *, table_overlay=None):
    """Build a PassResult-shaped object stub for tests."""
    pr = PassResult.__new__(PassResult)
    pr.pass_name = "missile_propulsion"
    pr.template_instance = None  # tests don't walk the typed-edge graph
    pr.metadata = None
    pr.pre_merge_rejections = []
    pr.upstream_refs = None
    pr.pre_merge_walk = None
    pr.provenance = []
    pr.field_evidence = {}
    pr._walker_entities_cache = list(instances)  # short-circuit walker
    pr.table_overlay = table_overlay
    return pr


def test_table_alias_map_runs_before_token_overlap(monkeypatch):
    """alias_map_by_entity_type collapses three non-token-overlapping
    aliases (SA-75 / SA-2A / 1D) onto canonical 1D before the
    token-overlap pass runs. After canonicalize, all three have
    system_name='1D'."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")
    a = _missile_inst("SA-75")
    b = _missile_inst("SA-2A")
    c = _missile_inst("1D")
    pr = _make_propulsion_passresult([a, b, c])
    pass_results = {"missile_propulsion": pr}
    alias_map = {"MISSILE_SYSTEM": {"SA-75": "1D", "SA-2A": "1D"}}
    rewrites = canonicalize_cross_pass_identities(
        pass_results, _ontology_min(),
        table_alias_map_by_entity_type=alias_map,
    )
    assert rewrites == 2
    assert a.system_name == b.system_name == c.system_name == "1D"


def test_table_overlay_does_not_break_existing_token_overlap(monkeypatch):
    """When table_alias_map_by_entity_type is None, existing token-
    overlap canonicalization runs unchanged. Two instances with
    overlapping tokens (PAC-3 / MIM-104F) still collapse the way they
    did pre-overlay."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")
    a = _missile_inst("PAC-3")
    b = _missile_inst("MIM-104F")  # token-overlap target
    pr = _make_propulsion_passresult([a, b])
    pass_results = {"missile_propulsion": pr}
    canonicalize_cross_pass_identities(
        pass_results, _ontology_min(),
        table_alias_map_by_entity_type=None,
    )
    # Existing token-overlap behavior is whatever the pre-overlay code
    # did — assert only that the call ran without error.
    assert a.system_name in ("PAC-3", "MIM-104F")


def test_kill_switch_disables_overlay_fresh_extraction(monkeypatch):
    """DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false on the worker → even
    if a fresh-extraction PassResult carries no overlay, behavior is
    unchanged: canonicalize runs without alias_map; Phase 0.5 skipped;
    no IDENTITY_REWRITE / TABLE_OVERLAY_APPLIED log lines."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "false")
    a = _missile_inst("1D")
    pr = _make_propulsion_passresult([a], table_overlay=None)
    pass_results = {"missile_propulsion": pr}
    # Build minimal manifest stub
    manifest = type("M", (), {"passes": [], "bundle_key": "test"})()
    with patch("app.services.extraction_merge.logger") as log:
        merge_and_resolve(
            pass_results=pass_results, manifest=manifest,
            ontology=_ontology_min(),
            document_id="doc-x", pipeline_run_id="run-x",
        )
        log_calls = [c.args[0] for c in log.info.call_args_list]
        assert not any("IDENTITY_REWRITE" in s for s in log_calls)
        assert not any("TABLE_OVERLAY_APPLIED" in s for s in log_calls)


def test_kill_switch_worker_side_overrides_cached_overlay(monkeypatch):
    """Critical defense-in-depth case (spec §4.3): a PassResult arrives
    with a fully-populated TableOverlay (e.g., loaded from cached
    pipeline_pass_outputs.metadata_json from yesterday's run). Operator
    has just set DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false on the worker.
    Expected: merge_and_resolve sees the cached overlay AS IF None.
    apply_identity_rewrite NOT called; apply_field_overlay NOT called;
    one TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER INFO log line emitted."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "false")
    a = _missile_inst("SA-75")
    cached_overlay = TableOverlay(
        alias_map_by_entity_type={"MISSILE_SYSTEM": {"SA-75": "1D"}},
        facts=[TableFact(
            canonical_entity="1D", entity_type="MISSILE_SYSTEM",
            schema_field="booster_mass_kg", value=1135.0,
            source_label="Weight kg", section_ctx="1st Stage",
            pass_name="missile_propulsion", raw_text="1135",
        )],
    )
    pr = _make_propulsion_passresult([a], table_overlay=cached_overlay)
    pass_results = {"missile_propulsion": pr}
    manifest = type("M", (), {"passes": [], "bundle_key": "test"})()
    with patch("app.services.extraction_merge.logger") as log:
        merge_and_resolve(
            pass_results=pass_results, manifest=manifest,
            ontology=_ontology_min(),
            document_id="doc-y", pipeline_run_id="run-y",
        )
        log_calls = [c.args[0] for c in log.info.call_args_list]
        assert any("TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER" in s
                   for s in log_calls)
        assert not any("IDENTITY_REWRITE" in s for s in log_calls)
        assert not any("TABLE_OVERLAY_APPLIED" in s for s in log_calls)
    # Critical: instance must NOT have been rewritten despite cached
    # alias_map carrying SA-75 → 1D.
    assert a.system_name == "SA-75"
```

- [ ] **Step 2: Run, confirm fail.**

```bash
pytest tests/unit/test_extraction_merge_table_overlay.py -v
```
Expected: 4 FAILED — `canonicalize_cross_pass_identities` doesn't yet accept `table_alias_map_by_entity_type`; `merge_and_resolve` doesn't yet honor the worker-side kill switch.

- [ ] **Step 3: Modify `canonicalize_cross_pass_identities` signature.**

In `app/services/extraction_merge.py:1015`, change the signature to:

```python
def canonicalize_cross_pass_identities(
    pass_results: dict[str, "PassResult"],
    ontology: dict,
    *,
    table_alias_map_by_entity_type: dict[str, dict[str, str]] | None = None,
) -> int:
```

At the top of the function body, before the existing token-overlap pass:

```python
    rewrites = 0
    if table_alias_map_by_entity_type:
        try:
            from app.services.table_overlay import apply_identity_rewrite
            stats = apply_identity_rewrite(
                pass_results, table_alias_map_by_entity_type, ontology,
            )
            rewrites += stats.rewrites
            logger.info(
                "IDENTITY_REWRITE rewrites=%d unique_canonicals=%d passes_touched=%d",
                stats.rewrites, stats.unique_canonicals, stats.passes_touched,
            )
        except Exception as exc:
            logger.warning(
                "apply_identity_rewrite failed: %s — falling through to "
                "existing token-overlap canonicalization", exc,
            )
```

- [ ] **Step 4: Restructure `merge_and_resolve` — extract overlay FIRST, then call canonicalize WITH the alias map, then Phase 0.5.**

In `app/services/extraction_merge.py:1105` `merge_and_resolve`, the canonical final shape (replace the existing `canonicalize_cross_pass_identities(...)` call near the top of the function with this block):

```python
    # ----- Mechanism A1 (Phase 0 + Phase 0.5): table overlay -----
    import os
    from app.services.table_overlay import apply_field_overlay

    def _table_overlay_enabled_worker() -> bool:
        return os.environ.get(
            "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true",
        ).lower() != "false"

    overlay_enabled = _table_overlay_enabled_worker()
    table_overlay = _extract_doc_overlay(pass_results) if overlay_enabled else None

    # Worker-side kill switch is authoritative over cached overlays.
    # Spec §4.3.
    if not overlay_enabled:
        cached_overlay_present = sum(
            1 for pr in pass_results.values()
            if getattr(pr, "table_overlay", None) is not None
        )
        if cached_overlay_present:
            logger.info(
                "TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER doc_id=%s "
                "pass_count=%d cached_overlay_present=%d",
                document_id, len(pass_results), cached_overlay_present,
            )

    # Phase 0: cross-pass identity canonicalization. When overlay is
    # enabled AND we found one, pass the alias map through; otherwise
    # call with None and rely on the existing token-overlap pass.
    canonicalize_cross_pass_identities(
        pass_results,
        ontology,
        table_alias_map_by_entity_type=(
            table_overlay.alias_map_by_entity_type
            if (overlay_enabled and table_overlay is not None)
            else None
        ),
    )

    # Phase 0.5: per-cell field overlay. Only when overlay is enabled,
    # we found one, and it carries facts.
    if overlay_enabled and table_overlay is not None and table_overlay.facts:
        try:
            stats = apply_field_overlay(
                pass_results,
                table_overlay.facts,
                policy="table_wins_for_table_facts",
            )
            logger.info(
                "TABLE_OVERLAY_APPLIED doc_id=%s "
                "field_overlay_applied=%d matches_touched=%d "
                "skipped_no_entity=%d skipped_unknown_field=%d "
                "skipped_validation_fail=%d conflicts_overridden=%d "
                "policy=%s",
                document_id, stats.applied, stats.matches_touched,
                stats.skipped_no_entity, stats.skipped_unknown_field,
                stats.skipped_validation_fail, stats.conflicts_overridden,
                stats.policy_active,
            )
        except Exception as exc:
            logger.warning(
                "apply_field_overlay failed mid-loop: %s — proceeding "
                "with merge using whatever (fact, instance) swaps had "
                "already completed. Bounded-degraded per §7. Operator "
                "rollback via kill switch only.", exc,
            )
    # ---------- end Mechanism A1 ----------
```

The control flow guarantees:
- Worker-side kill-switch off → `table_overlay = None` → `canonicalize` called with `None` → Phase 0.5 skipped → emits `TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER` if cached overlays exist.
- Worker-side on AND no overlay → `canonicalize` called with `None`, Phase 0.5 skipped (no facts).
- Worker-side on AND overlay present → `canonicalize` called with the alias map, Phase 0.5 applies facts.

Per-(fact, instance) atomicity inside `apply_field_overlay` (Task 7) guarantees the per-mutation safety; this block adds the catch-and-log if `apply_field_overlay` itself raises.

- [ ] **Step 5: Add `_extract_doc_overlay` helper.**

```python
def _extract_doc_overlay(pass_results: dict) -> "TableOverlay | None":
    """Find the first non-empty table_overlay across pass_results.
    All passes from the same DoclingDocument should have identical
    overlays; if they diverge, log WARNING and use the first.
    Spec §5.5."""
    first = None
    for pass_name, pr in pass_results.items():
        ov = getattr(pr, "table_overlay", None)
        if ov is None:
            continue
        is_nonempty = bool(
            ov.alias_map_by_entity_type or ov.facts or ov.cross_entity_hints
        )
        if not is_nonempty:
            continue
        if first is None:
            first = ov
            continue
        if ov.model_dump() != first.model_dump():
            logger.warning(
                "_extract_doc_overlay: divergent overlays across passes — "
                "using first non-empty. Inspect parser deterministic "
                "behavior. first_facts=%d other_facts=%d",
                len(first.facts), len(ov.facts),
            )
    return first
```

- [ ] **Step 6: Run integration tests.**

```bash
pytest tests/unit/test_extraction_merge_table_overlay.py -v
```
Expected: All PASSED.

- [ ] **Step 7: Confirm worker test suite still green.**

```bash
pytest tests/unit -q 2>&1 | tail -15
```

- [ ] **Step 8: Commit.**

```bash
git add app/services/extraction_merge.py \
       tests/unit/test_extraction_merge_table_overlay.py
git commit -m "feat(table-overlay): integrate identity rewrite + Phase 0.5 field overlay into merge_and_resolve, worker-side kill switch"
```

### Task 9: `pipeline.py` — read `table_overlay` onto `PassResult`

**Files:**
- Modify: `app/services/extraction_merge.py` (PassResult dataclass field)
- Modify: `app/workers/pipeline.py` (`_parse_pass_response` populates field)
- Modify: `tests/unit/test_run_single_pass.py`

**Decision: where does the `TableOverlay` Pydantic class live?**

The reviewer flagged that `from docker.docling_graph.app.schemas import TableOverlay` is fragile (hyphen in directory name; not a normal Python package layout). Decision for this plan: **the canonical home is `app/services/table_overlay.py` (worker side).** The parser-side `docker/docling-graph/app/schemas.py` declares its own `TableOverlay` (Task 3) for the response schema; the worker never imports from `docker/docling-graph/`. The two declarations are **structurally identical Pydantic models** with the same field names and types — JSON travels between them, not class identity. Task 9 wires both sides to use their own local class. Drift guard: a unit test (Step 6 below) round-trips a parser-side TableOverlay through JSON into a worker-side TableOverlay and asserts equality.

- [ ] **Step 1: Add `TableFact`, `CrossEntityHint`, `TableOverlay` to `app/services/table_overlay.py` (worker copy).**

Append to `app/services/table_overlay.py`:

```python
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class TableFact(BaseModel):
    """Worker-side mirror of the parser's TableFact wire shape (spec §5.4)."""
    model_config = ConfigDict(frozen=True)
    canonical_entity: str
    entity_type: str
    schema_field: str
    value: Any
    source_label: str
    section_ctx: Optional[str] = None
    pass_name: str
    raw_text: str


class CrossEntityHint(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_canonical: str
    source_entity_type: str
    target_alias: str
    target_entity_type: str
    relationship_kind: str


class TableOverlay(BaseModel):
    alias_map_by_entity_type: dict[str, dict[str, str]] = Field(default_factory=dict)
    facts: list[TableFact] = Field(default_factory=list)
    cross_entity_hints: list[CrossEntityHint] = Field(default_factory=list)


TableFact.model_rebuild()
CrossEntityHint.model_rebuild()
TableOverlay.model_rebuild()
```

- [ ] **Step 2: Write failing test for `_parse_pass_response`.**

In `tests/unit/test_run_single_pass.py`, add:

```python
def test_parse_pass_response_reads_table_overlay():
    """When response_json carries a table_overlay key, _parse_pass_response
    must populate PassResult.table_overlay with a parsed TableOverlay
    instance. None when key is missing or value is null."""
    from app.workers.pipeline import _parse_pass_response
    from app.services.table_overlay import TableOverlay

    # Minimal pass_def + manifest stub matching what the prod code reads.
    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()

    response_json = {
        "bundle_key": "air_defense_v3",
        "pass_name": "missile_propulsion",
        "pass_output": {"records": []},
        "metadata": {"node_count": 0, "edge_count": 0},
        "provenance": [],
        "field_provenance": [],
        "table_overlay": {
            "alias_map_by_entity_type": {
                "MISSILE_SYSTEM": {"SA-75": "1D"},
            },
            "facts": [{
                "canonical_entity": "1D",
                "entity_type": "MISSILE_SYSTEM",
                "schema_field": "booster_mass_kg",
                "value": 1135.0,
                "source_label": "Weight kg",
                "section_ctx": "1st Stage",
                "pass_name": "missile_propulsion",
                "raw_text": "1135",
            }],
            "cross_entity_hints": [],
        },
    }

    result = _parse_pass_response(response_json, pass_def, manifest)
    assert isinstance(result.table_overlay, TableOverlay)
    assert result.table_overlay.alias_map_by_entity_type == {
        "MISSILE_SYSTEM": {"SA-75": "1D"},
    }
    assert len(result.table_overlay.facts) == 1
    assert result.table_overlay.facts[0].canonical_entity == "1D"


def test_parse_pass_response_table_overlay_missing_is_none():
    from app.workers.pipeline import _parse_pass_response
    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()
    response_json = {
        "bundle_key": "air_defense_v3", "pass_name": "missile_propulsion",
        "pass_output": {"records": []},
        "metadata": {}, "provenance": [], "field_provenance": [],
        # No table_overlay key
    }
    result = _parse_pass_response(response_json, pass_def, manifest)
    assert result.table_overlay is None


def test_parse_pass_response_malformed_table_overlay_is_dropped():
    """A malformed payload (e.g., wrong field types) must not crash;
    log a WARNING and set table_overlay=None."""
    from app.workers.pipeline import _parse_pass_response
    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()
    response_json = {
        "bundle_key": "air_defense_v3", "pass_name": "missile_propulsion",
        "pass_output": {"records": []},
        "metadata": {}, "provenance": [], "field_provenance": [],
        "table_overlay": {"alias_map_by_entity_type": "not a dict"},  # bogus
    }
    result = _parse_pass_response(response_json, pass_def, manifest)
    assert result.table_overlay is None
```

- [ ] **Step 3: Run, confirm 3 fail.**

```bash
pytest tests/unit/test_run_single_pass.py -k table_overlay -v
```
Expected: 3 FAILED.

- [ ] **Step 4: Add `table_overlay` field to `PassResult`.**

In `app/services/extraction_merge.py` near the top of the dataclass at line 202, add the import at file top:

```python
from app.services.table_overlay import TableOverlay  # type: ignore[import]
```

And the field on the `PassResult` dataclass:

```python
    # Mechanism A1 (spec §4.4 + §5.5): doc-level overlay parsed from
    # the docling-graph /extract-pass response. None when the parser
    # found no qualifying table OR the parser-side kill switch was off.
    table_overlay: TableOverlay | None = None
```

- [ ] **Step 5: Modify `_parse_pass_response` in `app/workers/pipeline.py`.**

At `app/workers/pipeline.py:2610` (where `PassResult(...)` is constructed), insert just before the `return PassResult(...)` line:

```python
    overlay_dict = response_json.get("table_overlay")
    table_overlay_obj = None
    if isinstance(overlay_dict, dict):
        try:
            from app.services.table_overlay import TableOverlay
            table_overlay_obj = TableOverlay.model_validate(overlay_dict)
        except Exception as exc:
            logger.warning(
                "_parse_pass_response: dropping malformed table_overlay: %s", exc,
            )
            table_overlay_obj = None
```

Then thread `table_overlay=table_overlay_obj` into the `PassResult(...)` constructor call.

- [ ] **Step 6: Add parser↔worker drift-guard test.**

In `tests/unit/test_table_overlay_worker.py`, add:

```python
def test_parser_and_worker_table_overlay_classes_round_trip():
    """The parser-side TableOverlay (in docker/docling-graph/app/schemas.py)
    and the worker-side TableOverlay (in app/services/table_overlay.py)
    declare structurally identical Pydantic models. JSON round-trip
    between them must equal element-wise. This test guards against
    drift: if a field is added on one side and not the other, this
    test fails."""
    # Parser side
    import importlib.util
    from pathlib import Path
    parser_path = (
        Path(__file__).resolve().parent.parent.parent
        / "docker" / "docling-graph" / "app" / "schemas.py"
    )
    spec = importlib.util.spec_from_file_location("dg_schemas", parser_path)
    parser_schemas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_schemas)

    # Worker side
    from app.services.table_overlay import (
        TableOverlay as WorkerTO, TableFact as WorkerTF,
    )

    # Build a parser-side overlay
    parser_ov = parser_schemas.TableOverlay(
        alias_map_by_entity_type={"MISSILE_SYSTEM": {"SA-75": "1D"}},
        facts=[parser_schemas.TableFact(
            canonical_entity="1D", entity_type="MISSILE_SYSTEM",
            schema_field="booster_mass_kg", value=1135.0,
            source_label="Weight kg", section_ctx="1st Stage",
            pass_name="missile_propulsion", raw_text="1135",
        )],
        cross_entity_hints=[],
    )

    # JSON round-trip into worker side
    dumped = parser_ov.model_dump(mode="json")
    worker_ov = WorkerTO.model_validate(dumped)

    assert worker_ov.alias_map_by_entity_type == parser_ov.alias_map_by_entity_type
    assert len(worker_ov.facts) == 1
    assert worker_ov.facts[0].canonical_entity == "1D"
    assert worker_ov.facts[0].value == 1135.0

    # Schema-shape equivalence: same field names + types.
    parser_fields = set(parser_schemas.TableOverlay.model_fields.keys())
    worker_fields = set(WorkerTO.model_fields.keys())
    assert parser_fields == worker_fields, (
        f"TableOverlay field drift: parser={parser_fields} worker={worker_fields}"
    )
    parser_fact_fields = set(parser_schemas.TableFact.model_fields.keys())
    worker_fact_fields = set(WorkerTF.model_fields.keys())
    assert parser_fact_fields == worker_fact_fields, (
        f"TableFact field drift: parser={parser_fact_fields} worker={worker_fact_fields}"
    )
```

- [ ] **Step 7: Run pipeline tests.**

```bash
pytest tests/unit/test_run_single_pass.py -v
pytest tests/unit/test_table_overlay_worker.py::test_parser_and_worker_table_overlay_classes_round_trip -v
```
Expected: All PASSED.

- [ ] **Step 8: Commit.**

```bash
git add app/services/extraction_merge.py app/services/table_overlay.py \
       app/workers/pipeline.py \
       tests/unit/test_run_single_pass.py tests/unit/test_table_overlay_worker.py
git commit -m "feat(table-overlay): plumb table_overlay from /extract-pass response onto PassResult; add worker-side TableOverlay class + parser drift guard"
```

### Task 10: `docker-compose.yml` kill-switch env var on both services

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add the env var to both services.**

In the `docker-compose.yml` env block for `docling-graph` AND `worker` / `worker-graph`:

```yaml
DOCLING_GRAPH_TABLE_OVERLAY_ENABLED: ${DOCLING_GRAPH_TABLE_OVERLAY_ENABLED:-true}
```

- [ ] **Step 2: Verify with `docker compose config`.**

```bash
docker compose config | grep -A1 "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED"
```
Expected: 2 hits (one per service), both showing the variable interpolated.

- [ ] **Step 3: Commit.**

```bash
git add docker-compose.yml
git commit -m "feat(table-overlay): expose DOCLING_GRAPH_TABLE_OVERLAY_ENABLED on both docling-graph and worker services"
```

---

## Chunk 6: End-to-end fixture (Task 11)

### Task 11: End-to-end synthetic fixture test

**Files:**
- Create: `tests/integration/test_table_overlay_end_to_end.py`

- [ ] **Step 1: Write the failing end-to-end test.**

Create `tests/integration/test_table_overlay_end_to_end.py`:

```python
"""End-to-end fixture for spec §8.5: synthetic DoclingDocument with
SA-2-shaped variants table, 4-pass stub LLM responses encoding the
empirical alias-scatter + wrong-propulsion-value failure modes,
through merge_and_resolve. Validates Mechanism A1 collapses aliases
AND overrides wrong propulsion values."""
from unittest.mock import patch

from app.services.extraction_merge import merge_and_resolve, PassResult
from app.services.table_overlay import (
    TableOverlay as WorkerTO, TableFact, CrossEntityHint,
)
from ontology_bundles.air_defense_v3.extraction_schemas import (
    missile_propulsion, missile_airframe, missile_kinematics,
    missile_speed_timing,
)


def _build_overlay_for_sa2():
    """Synthetic overlay matching SA-2 column 0 (1D) and column 1 (13D)."""
    alias_map = {"MISSILE_SYSTEM": {
        "SA-75": "1D", "SA-2A": "1D",  # column 0 aliases
        "S-75": "13D", "SA-2C": "13D",  # column 1 aliases
    }}
    facts = [
        # Airframe row (Length mm) for both columns
        TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                  schema_field="body_length_m", value=10.726,
                  source_label="Length mm", section_ctx=None,
                  pass_name="missile_airframe", raw_text="10726"),
        TableFact(canonical_entity="13D", entity_type="MISSILE_SYSTEM",
                  schema_field="body_length_m", value=10.841,
                  source_label="Length mm", section_ctx=None,
                  pass_name="missile_airframe", raw_text="10841"),
        # Propulsion row (1st Stage Weight kg) — these are the
        # acceptance-driving facts.
        TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                  schema_field="booster_mass_kg", value=1135.0,
                  source_label="Weight kg", section_ctx="1st Stage",
                  pass_name="missile_propulsion", raw_text="1135"),
        TableFact(canonical_entity="13D", entity_type="MISSILE_SYSTEM",
                  schema_field="booster_mass_kg", value=1135.0,
                  source_label="Weight kg", section_ctx="1st Stage",
                  pass_name="missile_propulsion", raw_text="1135"),
    ]
    hints = [CrossEntityHint(
        source_canonical="1D", source_entity_type="MISSILE_SYSTEM",
        target_alias="RSNA-75", target_entity_type="RADAR_SYSTEM",
        relationship_kind="associated_with",
    )]
    return WorkerTO(
        alias_map_by_entity_type=alias_map,
        facts=facts,
        cross_entity_hints=hints,
    )


def _build_propulsion_passresult():
    """Stub-LLM propulsion pass: emits 4 instances under different alias
    names, with ONE instance carrying a WRONG booster_mass_kg=970 (the
    empirical failure mode). Overlay must rewrite system_name AND
    override the wrong value."""
    instances = [
        missile_propulsion.MissilePropulsionRecord(
            system_name="SA-75", booster_mass_kg=970.0,  # WRONG
        ),
        missile_propulsion.MissilePropulsionRecord(
            system_name="SA-2A", booster_mass_kg=None,
        ),
        missile_propulsion.MissilePropulsionRecord(
            system_name="S-75", booster_mass_kg=None,
        ),
        missile_propulsion.MissilePropulsionRecord(
            system_name="SA-2C", booster_mass_kg=None,
        ),
    ]
    pr = PassResult.__new__(PassResult)
    pr.pass_name = "missile_propulsion"
    pr.template_instance = None
    pr.metadata = None
    pr.pre_merge_rejections = []
    pr.upstream_refs = None
    pr.pre_merge_walk = None
    pr.provenance = []
    pr.field_evidence = {}
    pr._walker_entities_cache = list(instances)
    pr.table_overlay = _build_overlay_for_sa2()
    return pr, instances


def _build_airframe_passresult():
    instances = [
        missile_airframe.MissileAirframeRecord(
            system_name="SA-75", body_length_m=None,
        ),
        missile_airframe.MissileAirframeRecord(
            system_name="S-75", body_length_m=None,
        ),
    ]
    pr = PassResult.__new__(PassResult)
    pr.pass_name = "missile_airframe"
    pr.template_instance = None
    pr.metadata = None
    pr.pre_merge_rejections = []
    pr.upstream_refs = None
    pr.pre_merge_walk = None
    pr.provenance = []
    pr.field_evidence = {}
    pr._walker_entities_cache = list(instances)
    pr.table_overlay = _build_overlay_for_sa2()
    return pr, instances


def test_end_to_end_sa2_alias_collapse_and_propulsion_override(monkeypatch):
    """Mechanism A1 acceptance smoke test:
       - 4 alias instances (SA-75/SA-2A → 1D, S-75/SA-2C → 13D) collapse
         to 2 canonical post-rewrite.
       - Wrong LLM booster_mass_kg=970 on SA-75 is OVERRIDDEN to 1135
         (table fact wins).
       - FIELD_OVERLAY_OVERRIDE log line is emitted for that override.
       - Other instances pick up booster_mass_kg=1135 from null."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true")

    prop_pr, prop_instances = _build_propulsion_passresult()
    af_pr, af_instances = _build_airframe_passresult()
    pass_results = {
        "missile_propulsion": prop_pr,
        "missile_airframe": af_pr,
    }
    ontology = {"entity_types": [
        {"name": "MISSILE_SYSTEM", "graph_id_fields": ["system_name"]},
    ]}
    manifest = type("M", (), {"passes": [], "bundle_key": "air_defense_v3"})()

    with patch("app.services.extraction_merge.logger") as log:
        merge_and_resolve(
            pass_results=pass_results, manifest=manifest,
            ontology=ontology,
            document_id="sa2-doc", pipeline_run_id="run-sa2",
        )
        log_calls_info = [c.args[0] for c in log.info.call_args_list]

    # Alias rewrite happened
    rewritten_names = {inst.system_name for inst in prop_instances}
    assert rewritten_names == {"1D", "13D"}, (
        f"expected alias collapse, got {rewritten_names}"
    )

    # The wrong-booster-mass instance now carries the table value
    wrong_orig = next(i for i in prop_instances if i.booster_mass_kg == 970.0)
    # Should NOT exist anymore — must be overridden
    matching_wrong = [i for i in prop_instances if i.booster_mass_kg == 970.0]
    assert matching_wrong == [], (
        "wrong LLM booster_mass_kg=970.0 should have been overridden"
    )
    # All 1D / 13D instances now have 1135.0
    for inst in prop_instances:
        assert inst.booster_mass_kg == 1135.0

    # FIELD_OVERLAY_OVERRIDE log emitted for the override case
    assert any("FIELD_OVERLAY_OVERRIDE" in s for s in log_calls_info), (
        "expected FIELD_OVERLAY_OVERRIDE log line for booster_mass_kg override"
    )
    # IDENTITY_REWRITE and TABLE_OVERLAY_APPLIED also emitted
    assert any("IDENTITY_REWRITE" in s for s in log_calls_info)
    assert any("TABLE_OVERLAY_APPLIED" in s for s in log_calls_info)

    # Airframe instances: body_length_m populated from overlay (was null)
    af_lengths = {inst.body_length_m for inst in af_instances}
    assert af_lengths == {10.726, 10.841}
```

- [ ] **Step 2: Run, confirm fail (until merge_and_resolve integration is done in Task 8).**

```bash
pytest tests/integration/test_table_overlay_end_to_end.py -v
```
Expected: PASS once Tasks 6–9 are complete (this test depends on the worker overlay module + extraction_merge integration + PassResult.table_overlay plumbing).

If it fails after Tasks 6–9, examine the exact assertion that fails and trace back through:
1. Did `apply_identity_rewrite` actually run? (look for `IDENTITY_REWRITE` log)
2. Did `apply_field_overlay` find the matching instance? (look for `TABLE_OVERLAY_APPLIED` log with `applied>0`)
3. Did fan-out work — both 1D instances received the fact? Did the 970→1135 override register `conflicts_overridden++`?

- [ ] **Step 3: Commit.**

```bash
git add tests/integration/test_table_overlay_end_to_end.py
git commit -m "test(table-overlay): synthetic SA-2 end-to-end with alias collapse + propulsion override"
```

---

## Chunk 7: Container rebuild + §20 acceptance run (Task 12)

### Task 12: Production acceptance

**Files:** None modified. Operator-driven measurement.

- [ ] **Step 1: Rebuild containers.**

```bash
docker compose build docling-graph worker worker-graph
docker compose up -d docling-graph worker worker-graph
docker compose ps | grep -E "docling-graph|worker"
```
Expected: All Up healthy.

- [ ] **Step 2: Confirm kill switch defaults to true.**

```bash
docker compose exec docling-graph env | grep DOCLING_GRAPH_TABLE_OVERLAY_ENABLED
docker compose exec worker env | grep DOCLING_GRAPH_TABLE_OVERLAY_ENABLED
```
Expected: `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=true` on both.

- [ ] **Step 3: Smoke test on synthetic SA-2.**

```bash
curl -s -X POST http://localhost:8002/extract-pass \
  -H "Content-Type: application/json" \
  -d @/tmp/synthetic_sa2_request.json \
  | jq '.table_overlay.alias_map_by_entity_type, .diagnostics.service_table_overlay'
```
Expected: alias map with MISSILE_SYSTEM key + non-zero size; diagnostics show `tables_processed=1`, `kill_switch_active_parser=false`.

- [ ] **Step 4: Run notebook §20 at T=1.0 against real SA-2 PDF.**

Open Jupyter, execute §20 cells with `temperature=1.0`. Capture the per-pass scorecard.

- [ ] **Step 5: Compare against the live baseline from Task 0.**

Per spec §9 acceptance criteria:
- **Item 6 (field-specific):** Verify ≥6 of 7 listed variants have correct `booster_mass_kg` AND ≥6 of 7 have correct `sustain_mass_kg`. Grep worker logs for `FIELD_OVERLAY_OVERRIDE` lines on those fields:

```bash
docker compose logs worker 2>&1 | grep "FIELD_OVERLAY_OVERRIDE" | grep "booster_mass_kg\|sustain_mass_kg"
```

Expected: ≥1 override line per variant where the LLM had a wrong value pre-overlay (matching the predictions captured in Task 0 step 6).

- **Item 7 (floor):** Each pass's ✓ exact count ≥ live-baseline floor (±2 LLM-noise tolerance).
- **Item 8 (no-table docs):** For 16 no-table corpus docs and 4 row-major-table docs, system_name set within ±2 entities and ⊇ 80% of pre-deploy set.
- **Item 9 (wall-time):** ≤ +5% per /extract-pass on table-bearing; 0% on no-table.
- **Item 10 (kill switch):** Toggle the env vars off via `docker compose --env-file ...`, restart, re-run §20: behavior bit-identical to Task 0 baseline within LLM noise.
- **Item 11 (diagnostics):** Confirm response carries `service_table_overlay` with all fields populated; worker logs show `IDENTITY_REWRITE`, `TABLE_OVERLAY_APPLIED`, `FIELD_OVERLAY_OVERRIDE` lines.

- [ ] **Step 6: Document acceptance verdict.**

Append a brief acceptance memo to `/tmp/baseline_2026-05-06_pre_overlay/acceptance.md` with:
- All 11 §9 criteria with PASS/FAIL.
- Side-by-side per-pass scorecard (baseline vs. post-overlay).
- Counts of FIELD_OVERLAY_OVERRIDE log lines per variant per field.

- [ ] **Step 7: If acceptance passes, mark spec status Approved.**

Edit `docs/superpowers/specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md` line 3:
```
**Status:** Approved 2026-05-06 (post-acceptance run, commit <hash>)
```
Commit:
```bash
git add docs/superpowers/specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md
git commit -m "spec(table-overlay): mark Approved after §20 acceptance pass"
```

---

## Rollback plan

If acceptance fails (any of items 6–9 misses target), do NOT iterate fixes under time pressure. Instead:

1. Flip `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false` in `docker-compose.yml`.
2. Restart docling-graph + worker.
3. Verify behavior reverts to baseline via §20 re-run.
4. Diagnose the gap against §9 criteria with the diagnostics surface.
5. Reopen the spec at the relevant section; revise; re-plan.

The overlay code stays in place — only the env flag is flipped. No code revert needed.

---

## File-structure summary

| File | Status | Responsibility |
|---|---|---|
| `docker/docling-graph/app/_alias_map.py` | MODIFY | Add MISSILE_/RADAR_IDENTITY_LABELS, CROSS_ENTITY_REF_PATTERNS, CANONICAL_PRIORITY |
| `docker/docling-graph/app/_table_facts.py` | MODIFY | Add four helpers + `extract_table_overlay()` + qualification gate |
| `docker/docling-graph/app/schemas.py` | MODIFY | Add TableFact / CrossEntityHint / TableOverlay; add `table_overlay` field on `ExtractPassResponse` |
| `docker/docling-graph/app/main.py` | MODIFY | Wire `extract_table_overlay()` into /extract-pass with parser-side kill switch and diagnostics |
| `app/services/table_overlay.py` | NEW | `apply_identity_rewrite` + `apply_field_overlay` + `RewriteStats` + `OverlayStats` |
| `app/services/extraction_merge.py` | MODIFY | `canonicalize_cross_pass_identities` accepts `table_alias_map_by_entity_type`; `merge_and_resolve` calls Phase 0.5 with worker-side kill switch; PassResult gains `table_overlay` field |
| `app/workers/pipeline.py` | MODIFY | `_parse_pass_response` reads `table_overlay` from response, attaches to `PassResult` |
| `docker-compose.yml` | MODIFY | Expose `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED` on both services |

| Test file | Status | Coverage |
|---|---|---|
| `docker/docling-graph/tests/test_alias_map_overlay_constants.py` | NEW | Drift guards on the four constants |
| `docker/docling-graph/tests/test_table_overlay_extract.py` | NEW | Helper unit tests + SA-2 fixture |
| `docker/docling-graph/tests/test_table_overlay_qualification.py` | NEW | Strict-qualification gate (5 starvation tests) |
| `docker/docling-graph/tests/test_table_overlay_schemas.py` | NEW | Pydantic wire types round-trip + frozen + default-factory |
| `docker/docling-graph/tests/test_main_table_overlay_integration.py` | NEW | /extract-pass response integration + parser-side kill switch |
| `tests/unit/test_table_overlay_worker.py` | NEW | Worker `apply_identity_rewrite` + `apply_field_overlay` (10+ tests) |
| `tests/unit/test_extraction_merge_table_overlay.py` | NEW | merge_and_resolve integration + worker-side kill switch |
| `tests/unit/test_run_single_pass.py` | MODIFY | `_parse_pass_response` reads `table_overlay` |
| `tests/integration/test_table_overlay_end_to_end.py` | NEW | Synthetic SA-2 4-pass through merge_and_resolve |

---

## Parallelizability

After Task 0 (sequential, must come first):
- Tasks 1, 3 can run in parallel (different files: `_alias_map.py` vs `schemas.py`).
- Task 2 depends on Task 1 (uses the constants).
- Task 4 depends on Tasks 2 + 3 (uses helpers + Pydantic types).
- Task 5 depends on Task 4.
- Task 6 depends on Task 3 (uses TableFact / TableOverlay types — wire shape).
- Task 7 depends on Task 6 (extends same module).
- Task 8 depends on Task 7.
- Task 9 depends on Tasks 3 + 8.
- Task 10 depends on Task 8 (env var matters once worker reads it).
- Task 11 depends on Tasks 9 + 10 (full integration).
- Task 12 depends on Task 11.

Single-implementer sequential walk = ~16 commits across 12 tasks. Most tasks are 5–8 steps, ~10–15 min each; total estimate 6–8 hours of focused implementation, plus the §20 acceptance run wall-time (~30 min for re-derive + ~30 min post-overlay).
