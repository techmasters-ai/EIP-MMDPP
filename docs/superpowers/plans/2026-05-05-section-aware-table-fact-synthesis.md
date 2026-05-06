# Section-Aware Per-Cell Table-Fact Synthesis Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the operationally-disabled B1+B2 column-pivot (`_table_pivot.py`) with a section-aware per-cell table-fact synthesizer that emits one TextItem per `(entity, schema_field, value)` triple from column-major tables in `DoclingDocument.tables[]`, with row labels resolved deterministically via a structured Python alias map.

**Architecture:** Pipeline of pure functions in `docker/docling-graph/app/_table_facts.py`. Six steps: `detect_table_shape` → `extract_label_rows` → `derive_entity_ids` → `detect_section_context` → `resolve_alias` → `coerce_value` → `emit_fact`. Pass-aware (same DoclingDocument fed to four passes produces four different fact sets). The structured alias map lives in `_alias_map.py`, paired with §12b prose in `prompt_rules.py`; a drift-guard test asserts every entry has prose backing.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, dataclasses, FastAPI, docling-graph LLM extraction service, Ollama gemma4:31b, pytest.

**Spec:** [`docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md`](../specs/2026-05-05-section-aware-table-fact-synthesis-design.md) (commit `00c307c`, signed off after 3 review passes)

**Acceptance:** `missile_propulsion` ✓ exact ≥ 6 of the 7 listed variants (13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23) on the operator-driven §20 GT scorecard at T=1.0 against the SA-2 PDF, with no regression on `missile_kinematics ≥ 4`, `missile_airframe ≥ 8`, `missile_speed_timing ≥ 6` ✓ exact counts (alias-only T=1.0 baseline cited in spec §8.5).

---

## Pre-flight checklist

Run these once at the start of the session and before each chunk to confirm baseline:

- [ ] **P0: Read the spec.**

Run: `wc -l docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md`
Expected: ≥ 700 lines. If less, the file is truncated — abort and re-fetch.

Use the @superpowers-extended-cc:test-driven-development skill for every code-bearing task.

- [ ] **P1: Confirm baseline test suite status.**

Run on host from repo root (tests are not packaged in the docling-graph
container image — see P6 below):
```bash
pytest docker/docling-graph/tests -q 2>&1 | tail -5
```
Expected: All current tests pass (test_sanitizer, test_numeric_candidates, test_table_pivot all green). Document any failure as a pre-existing issue not caused by this plan.

- [ ] **P2: Confirm stack is up.**

Run: `docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "docling-graph|api"`
Expected: docling-graph and api both Up. If not, `docker compose up -d` and wait 30 s.

- [ ] **P3: Confirm B1+B2 hook state.**

Run: `grep -n "synthesize_pivoted_table_texts" docker/docling-graph/app/main.py`
Expected: 2 hits (one import at line ~120, one call at line ~561). If hits = 0, the rollback already happened; if hits ≥ 3, an unexpected duplicate is present.

- [ ] **P4: Verify §12b prose is intact.**

Run: `grep -n "Source-label to schema-field aliases" ontology_bundles/_shared/prompt_rules.py`
Expected: One hit (the §12b heading). If 0, the prose alias map has been edited; reconcile before proceeding because it is the SSoT for the drift-guard test.

- [ ] **P5: Confirm §20 baseline cache exists.**

Run inside the Jupyter container:
```bash
docker exec eip-mmdpp-jupyter ls /tmp/r21_alias_only_backup/
```
Expected: 4 files, one per missile pass (kinematics/airframe/speed_timing/propulsion at T=1.0). Required for the post-deploy delta comparison in Task 18. If missing, re-derive baseline before judging the synthesizer's deltas.

- [ ] **P6: Confirm test invocation environment.**

This plan's tests are NOT inside the docling-graph container image (the
Dockerfile copies `app/`, `repo/`, `patches/`, and `ontology_bundles/` but
not `tests/`). All `pytest` commands in this plan run on the host from the
repo root. The existing `docker/docling-graph/tests/conftest.py` adds the
service root to `sys.path` so tests' lazy `from app.X import Y` calls
resolve correctly when invoked as `pytest docker/docling-graph/tests/...`.

Verify host Python has pytest available:
```bash
which pytest && pytest --version
```
Expected: pytest 7+ available. If not, activate the project's venv:
`.venv/bin/activate` then re-check.

---

## Chunk 1: Types + alias map foundation

Tasks 1–6 establish the data structures and alias data that every downstream
function depends on. After Chunk 1, no behavior change in production — the
synthesizer is not yet invoked from `main.py`. Tests are unit-only.

### Task 1: Create `_table_facts.py` types module

**Files:**
- Create: `docker/docling-graph/app/_table_facts.py`
- Create: `docker/docling-graph/tests/test_table_facts_types.py`

- [ ] **Step 1: Write failing tests for the type shapes.**

Create `docker/docling-graph/tests/test_table_facts_types.py`:

```python
"""Type-shape tests for _table_facts.py.

Verifies the core data types declared in spec §5.0 — Shape enum, LabelRow
TypedDict, ParsedValue and FactStats dataclasses. These are the contract every
other component in the synthesizer depends on; if they drift the rest of the
pipeline silently misbehaves.
"""
import importlib.util
from dataclasses import is_dataclass, fields as dataclass_fields
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load_table_facts():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_table_facts", _FACTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shape_enum_has_required_members():
    tf = _load_table_facts()
    assert tf.Shape.COLUMN_MAJOR.value == "column_major"
    assert tf.Shape.ROW_MAJOR.value == "row_major"
    assert tf.Shape.HYBRID.value == "hybrid"
    assert tf.Shape.OTHER.value == "other"


def test_label_row_typed_dict_keys():
    tf = _load_table_facts()
    # TypedDict's __annotations__ enumerates the declared keys.
    expected = {"row_idx", "label_text", "label_col_span", "data_cells"}
    assert set(tf.LabelRow.__annotations__.keys()) == expected


def test_parsed_value_is_frozen_dataclass():
    tf = _load_table_facts()
    assert is_dataclass(tf.ParsedValue)
    field_names = {f.name for f in dataclass_fields(tf.ParsedValue)}
    assert field_names == {"value", "unit_inferred", "conversion_factor", "raw_text"}
    # Frozen check: assigning to an instance must raise.
    pv = tf.ParsedValue(value=1135.0, unit_inferred="kg", conversion_factor=1.0, raw_text="1135")
    try:
        pv.value = 0
    except Exception:
        return
    raise AssertionError("ParsedValue must be frozen — assignment should raise")


def test_parsed_value_supports_positional_construction():
    """§6 worked example uses ParsedValue(1135, 'kg', 1.0, '1135') positionally."""
    tf = _load_table_facts()
    pv = tf.ParsedValue(1135.0, "kg", 1.0, "1135")
    assert pv.value == 1135.0
    assert pv.unit_inferred == "kg"


def test_fact_stats_is_mutable_dataclass_with_defaults():
    tf = _load_table_facts()
    assert is_dataclass(tf.FactStats)
    fs = tf.FactStats()  # All defaults
    assert fs.tables_seen == 0
    assert fs.facts_emitted == 0
    assert fs.tables_by_shape == {}
    assert fs.hybrid_collisions == 0
    assert fs.truncated_at_cap is False
    assert fs.idempotent_skip is False
    # Mutability — counter increments must work.
    fs.facts_emitted += 1
    assert fs.facts_emitted == 1


def test_fact_stats_empty_classmethod():
    tf = _load_table_facts()
    fs = tf.FactStats.empty()
    assert isinstance(fs, tf.FactStats)
    assert fs.facts_emitted == 0


def test_fact_stats_as_dict():
    tf = _load_table_facts()
    fs = tf.FactStats(tables_seen=3, facts_emitted=33)
    fs.tables_by_shape["column_major"] = 1
    d = fs.as_dict()
    assert isinstance(d, dict)
    assert d["tables_seen"] == 3
    assert d["facts_emitted"] == 33
    assert d["tables_by_shape"] == {"column_major": 1}


def test_fact_stats_default_factory_isolates_instances():
    """Each FactStats instance must get its own tables_by_shape dict."""
    tf = _load_table_facts()
    a = tf.FactStats()
    b = tf.FactStats()
    a.tables_by_shape["column_major"] = 1
    assert "column_major" not in b.tables_by_shape


def test_alias_key_typealias_exists():
    tf = _load_table_facts()
    # TypeAlias is just an annotation; we verify import surface.
    assert hasattr(tf, "AliasKey")
    assert hasattr(tf, "SectionContext")
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `pytest docker/docling-graph/tests/test_table_facts_types.py -v 2>&1 | tail -20`

Expected: ALL fail with `ModuleNotFoundError` or attribute errors — file doesn't exist yet. The pytest invocation is host-side; the docling-graph container does not contain the tests/ directory.

- [ ] **Step 3: Create the types module.**

Create `docker/docling-graph/app/_table_facts.py`:

```python
"""Section-aware per-cell table-fact synthesis (spec
2026-05-05-section-aware-table-fact-synthesis-design.md).

Replaces _table_pivot.py operationally. Emits one TextItem per
(entity, schema_field, value) triple drawn from column-major tables in
DoclingDocument.tables[]. Pass-aware — same document fed to four passes
produces four different fact sets, each scoped to that pass's schema fields.

This module declares only the types and the public synthesize_table_facts
entry-point skeleton; the pipeline's pure functions live in this same file
and are added in subsequent tasks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TypeAlias, TypedDict


class Shape(str, Enum):
    """Detected shape of a DoclingDocument table.

    COLUMN_MAJOR: leftmost column(s) hold row labels; remaining columns
        hold per-entity values. Variant-specs tables are typically this.
    ROW_MAJOR: top row(s) hold column labels; remaining rows hold per-
        entity values. Financial/comparative tables are typically this.
    HYBRID: column-major with multi-row identity (e.g., row 0 "Industry
        Designation" + row 1 "Missile Type" both labeling each column).
    OTHER: skip — synthesis not applicable (below 4×4 floor, or shape
        signals match neither pattern).
    """

    COLUMN_MAJOR = "column_major"
    ROW_MAJOR = "row_major"
    HYBRID = "hybrid"
    OTHER = "other"


# Section keyword (e.g., "1st Stage") or None when no section context applies.
SectionContext: TypeAlias = str | None

# Key into ALIAS_MAP: (label_normalized, section_ctx, pass_name).
AliasKey: TypeAlias = tuple[str, SectionContext, str]


class LabelRow(TypedDict):
    """Normalized representation of a label-bearing table row.

    Both column-major and row-major paths produce LabelRow records; the
    pipeline downstream of extract_label_rows is shape-agnostic.
    """

    row_idx: int
    label_text: str  # raw, pre-normalization
    label_col_span: int
    data_cells: dict[int, str]  # entity_col -> cell text (raw)


@dataclass(frozen=True)
class ParsedValue:
    """One value extracted from a cell after coercion.

    A single cell may produce multiple ParsedValues (discrete alternatives
    like "1135/1028") or one (single value, range collapsed to midpoint).
    Frozen because instances are value-typed and shared across emit_fact.
    """

    value: float | str
    unit_inferred: str | None
    conversion_factor: float  # 1.0 if no conversion applied
    raw_text: str


@dataclass
class FactStats:
    """Per-call synthesis stats. Mutable so the orchestrator increments
    counters in place. Surfaces in diagnostics["service_table_facts"]."""

    tables_seen: int = 0
    tables_by_shape: dict[str, int] = field(default_factory=dict)
    sections_detected: int = 0  # distinct sections matched (embedded only)
    facts_emitted: int = 0
    rows_skipped_unresolvable: int = 0
    values_skipped_unparseable: int = 0
    multi_value_emissions: int = 0  # cells producing ≥2 facts (alternatives, not ranges)
    hybrid_collisions: int = 0  # composite-id collisions; last-write-wins
    truncated_at_cap: bool = False
    idempotent_skip: bool = False

    @classmethod
    def empty(cls) -> "FactStats":
        return cls()

    def as_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_types.py -v 2>&1 | tail -20`

Expected: 9/9 pass. (If the container needs a rebuild for the test file to be accessible, build first: `docker compose build docling-graph && docker compose up -d docling-graph`. Tests can also run on host with the loader pattern from `_load_table_facts()`.)

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_types.py
git commit -m "$(cat <<'EOF'
feat(extraction): table-fact synthesis types module

Adds _table_facts.py with Shape enum, LabelRow TypedDict, ParsedValue
(frozen dataclass), FactStats (mutable dataclass) per spec §5.0. Tests
verify dataclass shape, mutability invariants (frozen ParsedValue,
mutable FactStats), default_factory isolation, and method surfaces
(.empty()/.as_dict()).

Refs: docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md
EOF
)"
```

---

### Task 2: Add `normalize_label` function + tests

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py` (append `normalize_label` near top)
- Modify: `docker/docling-graph/tests/test_table_facts_types.py` (rename to `test_table_facts_normalize.py` OR add a new file)
- Create: `docker/docling-graph/tests/test_table_facts_normalize.py`

- [ ] **Step 1: Write failing tests for normalize_label.**

Create `docker/docling-graph/tests/test_table_facts_normalize.py`:

```python
"""Tests for normalize_label (spec §5.5).

This function is the single source of truth for label normalization across
the resolver and the §8.3 drift guard — both must use it so they assert the
same equality."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_table_facts_norm", _FACTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lowercase_and_whitespace_collapse():
    tf = _load()
    assert tf.normalize_label("  Length  mm  ") == "length mm"
    assert tf.normalize_label("Total Weight kg") == "total weight kg"


def test_punctuation_stripped_but_hyphens_preserved():
    """Hyphens distinguish 'SA-2' from 'SA 2'. Other punctuation goes."""
    tf = _load()
    assert tf.normalize_label("SA-2") == "sa-2"
    assert tf.normalize_label("SA-2 Guideline") == "sa-2 guideline"
    # Comma, period, parens, slashes stripped.
    assert tf.normalize_label("Weight, kg") == "weight kg"
    assert tf.normalize_label("Weight (kg)") == "weight kg"
    assert tf.normalize_label("Weight/kg") == "weight kg"
    assert tf.normalize_label("Mass.") == "mass"


def test_dash_class_collapsed_to_ascii_hyphen():
    """En-dash, em-dash, figure-dash all map to ASCII hyphen so '13D-A',
    '13D–A', '13D—A' compare equal."""
    tf = _load()
    assert tf.normalize_label("13D–A") == tf.normalize_label("13D-A")
    assert tf.normalize_label("13D—A") == tf.normalize_label("13D-A")
    assert tf.normalize_label("13D‒A") == tf.normalize_label("13D-A")


def test_nfkc_fold_collapses_full_width_and_compatibility_chars():
    """Full-width digits (e.g., '１') and compatibility characters fold to
    ASCII so OCR-extracted CJK-context tables still match."""
    tf = _load()
    assert tf.normalize_label("１D") == tf.normalize_label("1D")
    # Compatibility ligatures, fancy quotes, etc.
    assert tf.normalize_label("ª") == "a"  # Feminine ordinal indicator -> a


def test_idempotent():
    tf = _load()
    once = tf.normalize_label("1st Stage Weight kg")
    twice = tf.normalize_label(once)
    assert once == twice


def test_empty_string():
    tf = _load()
    assert tf.normalize_label("") == ""
    assert tf.normalize_label("   ") == ""
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `pytest docker/docling-graph/tests/test_table_facts_normalize.py -v 2>&1 | tail -10`
Expected: FAIL — `normalize_label` not yet defined.

- [ ] **Step 3: Add normalize_label to _table_facts.py.**

Append to `docker/docling-graph/app/_table_facts.py`, after the type declarations:

```python
import re
import unicodedata

# Dash-class characters mapped to single ASCII hyphen for stable matching.
# Covers: hyphen, non-breaking hyphen, figure dash, en-dash, em-dash,
# horizontal bar, minus sign, hyphen bullet, two-em / three-em dash,
# small em-dash, small hyphen-minus, fullwidth hyphen-minus.
_DASH_CLASS = re.compile(r"[‐‑‒–—―−⁃﹘﹣－]")

# Punctuation to strip after dash normalization. Keeps ASCII alphanumerics,
# whitespace, and hyphens (dashes already collapsed). Strips:
# . , ; : ! ? ' " ` ( ) [ ] { } / \ | _ * + = & % @ # ^ ~ < >
_PUNCT_TO_STRIP = re.compile(
    r"[\.\,\;\:\!\?\'\"\`\(\)\[\]\{\}\/\\\|_\*\+\=\&\%\@\#\^\~\<\>]"
)


def normalize_label(text: str) -> str:
    """Normalize a label string for ALIAS_MAP lookup and §8.3 drift-guard.

    Steps (spec §5.5):
    1. Unicode NFKC fold (collapses fancy quotes, full-width digits, etc.).
    2. Collapse all dash variants (en/em/figure/etc.) -> ASCII hyphen.
    3. Strip punctuation per _PUNCT_TO_STRIP (hyphens preserved).
    4. Lowercase.
    5. Collapse whitespace runs to single space; strip leading/trailing.

    The same function is used by resolve_alias and the §8.3 drift-guard
    test so prose-side and label-side normalization always agree.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DASH_CLASS.sub("-", text)
    text = _PUNCT_TO_STRIP.sub(" ", text)
    text = text.lower()
    text = " ".join(text.split())
    return text
```

- [ ] **Step 4: Run tests to verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_normalize.py -v 2>&1 | tail -10`
Expected: 6/6 pass.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_normalize.py
git commit -m "feat(extraction): normalize_label for ALIAS_MAP keying and drift-guard"
```

---

### Task 3: Create `_alias_map.py` skeleton + drift guard

**Files:**
- Create: `docker/docling-graph/app/_alias_map.py`
- Create: `docker/docling-graph/tests/test_alias_map.py`

- [ ] **Step 1: Write failing drift-guard tests.**

Create `docker/docling-graph/tests/test_alias_map.py`:

```python
"""Drift guard for _alias_map.py (spec §8.3).

The structured ALIAS_MAP in _alias_map.py and the §12b prose in
prompt_rules.DELTA_SYSTEM_PROMPT are paired SSoTs. These tests catch
drift in either direction: a new alias added to the map without a prose
mention, or a renamed schema field that the alias map still points at.
"""
import importlib.util
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _facts():
    return _load("dg_tf", _APP_DIR / "_table_facts.py")


def _aliases():
    return _load("dg_am", _APP_DIR / "_alias_map.py")


def _delta_prompt() -> str:
    from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT
    return DELTA_SYSTEM_PROMPT


def test_alias_map_labels_appear_in_prompt_rule():
    """Every ALIAS_MAP key's label_normalized appears as a token in §12b prose."""
    tf = _facts()
    am = _aliases()
    prose_normalized = tf.normalize_label(_delta_prompt())
    prose_tokens = set(prose_normalized.split())

    for (label_norm, _section, _pass), _field in am.ALIAS_MAP.items():
        for token in label_norm.split():
            assert token in prose_tokens, (
                f"Token {token!r} from ALIAS_MAP label {label_norm!r} "
                f"missing from §12b prose tokens. Either add it to §12b "
                f"or drop it from ALIAS_MAP."
            )


def test_section_keywords_appear_in_prompt_rule():
    """Every SECTION_KEYWORDS entry appears as a contiguous phrase in §12b prose."""
    tf = _facts()
    am = _aliases()
    prose_normalized = tf.normalize_label(_delta_prompt())
    for keyword in am.SECTION_KEYWORDS:
        keyword_norm = tf.normalize_label(keyword)
        assert keyword_norm in prose_normalized, (
            f"Section keyword {keyword!r} (normalized {keyword_norm!r}) "
            f"missing from §12b prose. Add to §12b before adding to "
            f"SECTION_KEYWORDS."
        )


def test_unit_table_keys_match_field_suffix_classes():
    """FIELD_SUFFIX_TO_UNIT_CLASS values must all be UNIT_TABLE keys."""
    am = _aliases()
    unit_classes = set(am.UNIT_TABLE.keys())
    referenced = set(am.FIELD_SUFFIX_TO_UNIT_CLASS.values())
    missing = referenced - unit_classes
    assert not missing, (
        f"FIELD_SUFFIX_TO_UNIT_CLASS references unit classes not in "
        f"UNIT_TABLE: {missing}"
    )


def test_unit_table_includes_canonical_unit():
    """Each unit class's table must include the canonical unit at factor 1.0
    (e.g., length_m table must include 'm' -> 1.0)."""
    am = _aliases()
    canonical_units = {
        "length_m": "m",
        "length_km": "km",
        "mass_kg": "kg",
        "time_sec": "sec",
        "velocity_mps": "mps",
        "frequency_mhz": "mhz",
        "gain_dbi": "dbi",
        "power_kw": "kw",
        "power_dbw": "dbw",
        "angle_deg": "deg",
    }
    for cls, unit in canonical_units.items():
        if cls not in am.UNIT_TABLE:
            continue  # Optional unit class
        assert unit in am.UNIT_TABLE[cls], f"{cls} missing canonical unit {unit!r}"
        assert am.UNIT_TABLE[cls][unit] == 1.0, f"{cls}[{unit!r}] should be 1.0"
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `pytest docker/docling-graph/tests/test_alias_map.py -v 2>&1 | tail -10`
Expected: FAIL — `_alias_map.py` doesn't exist.

- [ ] **Step 3: Create _alias_map.py with empty ALIAS_MAP + populated SECTION_KEYWORDS + UNIT_TABLE + FIELD_SUFFIX_TO_UNIT_CLASS.**

Create `docker/docling-graph/app/_alias_map.py`:

```python
"""Structured alias map (spec §5.1).

Pairs with the §12b prose in ontology_bundles/_shared/prompt_rules.py
DELTA_SYSTEM_PROMPT. The prose serves the LLM (handles natural-language
conditionals like "only when describing the whole missile"); this module
serves the synthesizer (programmatic lookup with pass- and section-
conditionals as first-class tuple keys).

A drift-guard test (tests/test_alias_map.py) asserts every ALIAS_MAP entry
has a corresponding §12b prose mention (per-token check), and every
SECTION_KEYWORDS entry appears as a contiguous phrase in the prose.

ALIAS_MAP entries are populated in Tasks 4 and 5 of the implementation
plan; this module ships with empty ALIAS_MAP plus the constant tables
that have no per-pass conditionals.
"""
from __future__ import annotations

# AliasKey: tuple[str, SectionContext, str] = (label_normalized, section_ctx, pass_name)
# Value: canonical schema field name (e.g., "booster_mass_kg").
# Populated in Tasks 4 (missile passes) and 5 (radar passes).
ALIAS_MAP: dict[tuple[str, str | None, str], str] = {}

# Section keywords detected by the embedded substring scan in
# detect_section_context (spec §5.4 strategy 1) and the standalone-row
# header path (strategy 2). Extensible per domain; entries here MUST
# also appear verbatim in §12b prose.
SECTION_KEYWORDS: tuple[str, ...] = (
    "1st Stage",
    "2nd Stage",
    "Booster",
    "Sustainer",
    "Sustain",
    "Ejector",
)

# Per-unit-class conversion factors, keyed by unit-class name. Inner dict
# maps cell-extracted unit string (lowercased) to the multiplicative factor
# that converts to the canonical unit (factor 1.0).
#
# Populated for the unit classes used by the four missile passes plus
# the five radar sub-passes. Add new unit classes by extending this dict
# AND adding a corresponding entry in FIELD_SUFFIX_TO_UNIT_CLASS.
UNIT_TABLE: dict[str, dict[str, float]] = {
    "length_m": {
        "m": 1.0,
        "mm": 0.001,
        "cm": 0.01,
        "in": 0.0254,
        "ft": 0.3048,
        "km": 1000.0,
    },
    "length_km": {
        "km": 1.0,
        "m": 0.001,
        "mi": 1.609344,
        "nm": 1.852,  # nautical miles
        "nmi": 1.852,
    },
    "mass_kg": {
        "kg": 1.0,
        "g": 0.001,
        "lb": 0.453592,
        "lbs": 0.453592,
        "t": 1000.0,  # metric tonne
        "ton": 1000.0,
        "tonne": 1000.0,
    },
    "time_sec": {
        "sec": 1.0,
        "s": 1.0,
        "ms": 0.001,
        "min": 60.0,
        "minutes": 60.0,
    },
    "velocity_mps": {
        "mps": 1.0,
        "m/s": 1.0,
        "kmh": 1.0 / 3.6,
        "km/h": 1.0 / 3.6,
        "mph": 0.44704,
        "knots": 0.514444,
        "kt": 0.514444,
    },
    "frequency_mhz": {
        "mhz": 1.0,
        "ghz": 1000.0,
        "khz": 0.001,
        "hz": 0.000001,
    },
    "gain_dbi": {
        "dbi": 1.0,
    },
    "power_kw": {
        "kw": 1.0,
        "w": 0.001,
        "mw": 1000.0,  # megawatt; lowercase clash with milliwatt is a real ambiguity — see coerce_value §5.6 unit-disambiguation
    },
    "power_dbw": {
        "dbw": 1.0,
        "dbm": 1.0,  # caller is responsible for sign offset; we don't synthesize ERP here
    },
    "angle_deg": {
        "deg": 1.0,
        "°": 1.0,
        "degrees": 1.0,
        "rad": 57.2957795,
        "radians": 57.2957795,
    },
}

# Schema-field-suffix -> unit-class mapping. The synthesizer's coerce_value
# function looks at the schema field name (e.g., "booster_mass_kg"), reads
# the suffix ("_kg"), and selects the unit class ("mass_kg") to coerce
# against. Schema fields whose suffix isn't here go through the string
# passthrough path (e.g., "booster_thrust" has no numeric suffix).
FIELD_SUFFIX_TO_UNIT_CLASS: dict[str, str] = {
    "_m": "length_m",
    "_km": "length_km",
    "_kg": "mass_kg",
    "_sec": "time_sec",
    "_mps": "velocity_mps",
    "_mhz": "frequency_mhz",
    "_dbi": "gain_dbi",
    "_kw": "power_kw",
    "_dbw": "power_dbw",
    "_deg": "angle_deg",
}
```

- [ ] **Step 4: Run tests to verify they pass.**

Run: `pytest docker/docling-graph/tests/test_alias_map.py -v 2>&1 | tail -10`

Expected: All 4 tests pass. The drift-guard tests trivially pass on empty `ALIAS_MAP` (the for-loop never executes); they will start enforcing real constraints in Tasks 4 and 5 when the dict is populated.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_alias_map.py docker/docling-graph/tests/test_alias_map.py
git commit -m "$(cat <<'EOF'
feat(extraction): _alias_map.py skeleton with SECTION_KEYWORDS, UNIT_TABLE

ALIAS_MAP starts empty; populated for missile passes in Task 4 and radar
passes in Task 5. SECTION_KEYWORDS, UNIT_TABLE, FIELD_SUFFIX_TO_UNIT_CLASS
populated for the 10 unit classes covering all four missile + five radar
sub-passes per spec §5.1. Drift-guard tests trivially pass on empty map
and tighten as entries land.

Refs: docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md §5.1
EOF
)"
```

---

### Task 4: Populate ALIAS_MAP for missile passes

**Files:**
- Modify: `docker/docling-graph/app/_alias_map.py` (populate ALIAS_MAP missile entries)
- Create: `docker/docling-graph/tests/test_alias_map_missile.py`

- [ ] **Step 1: Write failing tests asserting specific missile aliases resolve.**

Create `docker/docling-graph/tests/test_alias_map_missile.py`:

```python
"""Per-pass alias-resolution tests for missile passes (spec §5.5).

Verifies that the headline aliases for each missile sub-pass are populated
correctly. These tests anchor the production behavior — if any of these
break, the synthesizer cannot recover the GT scorecard targets.
"""
import importlib.util
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _APP_DIR / fname)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _aliases():
    return _load("dg_am", "_alias_map.py")


def _facts():
    return _load("dg_tf", "_table_facts.py")


def _resolve(label: str, section: str | None, pass_name: str) -> str | None:
    """Helper mirroring the resolve_alias contract."""
    am = _aliases()
    tf = _facts()
    return am.ALIAS_MAP.get((tf.normalize_label(label), section, pass_name))


# --- missile_kinematics ----------------------------------------------------

def test_kinematics_max_range_km_aliases():
    assert _resolve("Max Range km", None, "missile_kinematics") == "max_intercept_km"
    assert _resolve("Max Range m", None, "missile_kinematics") == "max_intercept_km"
    # Range alone (when source label is just "Range") also maps.
    assert _resolve("Range", None, "missile_kinematics") == "max_intercept_km"


def test_kinematics_min_range_aliases():
    assert _resolve("Min Range", None, "missile_kinematics") == "min_intercept_km"
    assert _resolve("Min Range km", None, "missile_kinematics") == "min_intercept_km"


def test_kinematics_altitude_aliases():
    """Only the full word 'Altitude' resolves — 'Alt' abbreviation is not in §12b prose."""
    assert _resolve("Max Altitude", None, "missile_kinematics") == "max_altitude_km"
    assert _resolve("Max Altitude km", None, "missile_kinematics") == "max_altitude_km"
    assert _resolve("Min Altitude", None, "missile_kinematics") == "min_altitude_km"
    # "Max Alt" deliberately does NOT resolve (would require §12b prose update).
    assert _resolve("Max Alt", None, "missile_kinematics") is None


def test_kinematics_pass_isolation():
    """Range labels do NOT resolve for other passes."""
    assert _resolve("Max Range km", None, "missile_propulsion") is None
    assert _resolve("Max Range km", None, "missile_airframe") is None


# --- missile_airframe ------------------------------------------------------

def test_airframe_length_aliases():
    assert _resolve("Length", None, "missile_airframe") == "body_length_m"
    assert _resolve("Length mm", None, "missile_airframe") == "body_length_m"
    assert _resolve("Body Length", None, "missile_airframe") == "body_length_m"


def test_airframe_diameter_aliases():
    assert _resolve("Diameter", None, "missile_airframe") == "body_diameter_m"
    assert _resolve("Diameter mm", None, "missile_airframe") == "body_diameter_m"
    assert _resolve("Body Diameter", None, "missile_airframe") == "body_diameter_m"


def test_airframe_total_mass_aliases():
    """Total Weight kg / Weight kg without section context maps to total_mass_kg."""
    assert _resolve("Total Weight kg", None, "missile_airframe") == "total_mass_kg"
    assert _resolve("Weight kg", None, "missile_airframe") == "total_mass_kg"
    assert _resolve("Mass kg", None, "missile_airframe") == "total_mass_kg"


# --- missile_speed_timing --------------------------------------------------

def test_speed_timing_max_speed_aliases():
    assert _resolve("Max Speed m/s", None, "missile_speed_timing") == "max_speed_mps"
    assert _resolve("Max Speed", None, "missile_speed_timing") == "max_speed_mps"


# --- missile_propulsion ----------------------------------------------------

def test_propulsion_booster_mass_kg_under_1st_stage():
    """The headline acceptance case — Weight kg under 1st Stage maps to
    booster_mass_kg only when active pass is missile_propulsion."""
    assert _resolve("Weight kg", "1st Stage", "missile_propulsion") == "booster_mass_kg"
    assert _resolve("Weight kg", "Booster", "missile_propulsion") == "booster_mass_kg"


def test_propulsion_sustain_mass_kg_under_2nd_stage():
    assert _resolve("Weight kg", "2nd Stage", "missile_propulsion") == "sustain_mass_kg"
    assert _resolve("Weight kg", "Sustainer", "missile_propulsion") == "sustain_mass_kg"
    assert _resolve("Weight kg", "Sustain", "missile_propulsion") == "sustain_mass_kg"


def test_propulsion_section_isolation():
    """Weight kg without section context does NOT resolve in propulsion pass —
    must have explicit stage section."""
    assert _resolve("Weight kg", None, "missile_propulsion") is None


def test_propulsion_embedded_label_resolves():
    """The SA-2 PDF puts the section in the label itself: '1st Stage Weight kg'.
    With embedded-section detection (Task 8), the label_text is left as-is and
    section_ctx is set separately. resolve_alias is keyed on the bare label."""
    # The bare label "Weight kg" with section_ctx="1st Stage" is the key.
    # The embedded label "1st Stage Weight kg" must NOT be a separate key —
    # detect_section_context strips the section keyword (or doesn't; the
    # split-vs-keep convention is fixed here for the resolver to be
    # deterministic). Convention: detect_section_context REMOVES the section
    # prefix from label_text before resolution, so the resolver sees only
    # "Weight kg". This must be tested by detect_section_context (Task 8);
    # here we verify the resolver works with the post-strip label.
    assert _resolve("Weight kg", "1st Stage", "missile_propulsion") == "booster_mass_kg"


def test_propulsion_burn_time_aliases():
    assert _resolve("Time sec", "1st Stage", "missile_propulsion") == "booster_time_sec"
    assert _resolve("Burn Time", "1st Stage", "missile_propulsion") == "booster_time_sec"
    assert _resolve("Time sec", "2nd Stage", "missile_propulsion") == "sustain_time_sec"


def test_propulsion_thrust_aliases():
    assert _resolve("Thrust", "1st Stage", "missile_propulsion") == "booster_thrust"
    assert _resolve("Thrust", "2nd Stage", "missile_propulsion") == "sustain_thrust"
    assert _resolve("Thrust", "Ejector", "missile_propulsion") == "ejector_thrust"
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `pytest docker/docling-graph/tests/test_alias_map_missile.py -v 2>&1 | tail -15`
Expected: ALL fail — ALIAS_MAP is empty.

- [ ] **Step 3: Populate ALIAS_MAP missile entries.**

Edit `docker/docling-graph/app/_alias_map.py`. Replace the empty `ALIAS_MAP: dict[...] = {}` declaration with:

```python
# IMPORTANT: keys are pre-normalized via _table_facts.normalize_label.
# Authors must lowercase, strip punctuation per the normalizer rules, and
# leave hyphens intact. The drift-guard test enforces this match against
# §12b prose tokens in prompt_rules.DELTA_SYSTEM_PROMPT.
ALIAS_MAP: dict[tuple[str, str | None, str], str] = {
    # ============================================================
    # missile_kinematics
    # ============================================================
    # Range -> max_intercept_km
    ("range",            None, "missile_kinematics"): "max_intercept_km",
    ("max range",        None, "missile_kinematics"): "max_intercept_km",
    ("max range km",     None, "missile_kinematics"): "max_intercept_km",
    ("max range m",      None, "missile_kinematics"): "max_intercept_km",
    ("maximum range",    None, "missile_kinematics"): "max_intercept_km",
    ("effective range",  None, "missile_kinematics"): "max_intercept_km",
    ("engagement range", None, "missile_kinematics"): "max_intercept_km",
    # Min Range -> min_intercept_km
    ("min range",        None, "missile_kinematics"): "min_intercept_km",
    ("min range km",     None, "missile_kinematics"): "min_intercept_km",
    ("min range m",      None, "missile_kinematics"): "min_intercept_km",
    ("minimum range",    None, "missile_kinematics"): "min_intercept_km",
    # Altitude -> max_altitude_km. NOTE: §12b prose uses only the full word
    # "Altitude" — the "Alt" abbreviation is NOT in §12b prose and would fail
    # the drift guard. We only register the full forms. If real documents use
    # "Max Alt km" labels we extend the alias map AFTER adding "(Alt is short
    # for Altitude)" to §12b prose so the drift guard still passes.
    ("altitude",            None, "missile_kinematics"): "max_altitude_km",
    ("max altitude",        None, "missile_kinematics"): "max_altitude_km",
    ("max altitude km",     None, "missile_kinematics"): "max_altitude_km",
    ("ceiling",             None, "missile_kinematics"): "max_altitude_km",
    ("engagement altitude", None, "missile_kinematics"): "max_altitude_km",
    # Min Altitude -> min_altitude_km
    ("min altitude",        None, "missile_kinematics"): "min_altitude_km",
    ("min altitude km",     None, "missile_kinematics"): "min_altitude_km",

    # ============================================================
    # missile_airframe
    # ============================================================
    ("length",           None, "missile_airframe"): "body_length_m",
    ("length mm",        None, "missile_airframe"): "body_length_m",
    ("length m",         None, "missile_airframe"): "body_length_m",
    ("overall length",   None, "missile_airframe"): "body_length_m",
    ("missile length",   None, "missile_airframe"): "body_length_m",
    ("body length",      None, "missile_airframe"): "body_length_m",

    ("diameter",         None, "missile_airframe"): "body_diameter_m",
    ("diameter mm",      None, "missile_airframe"): "body_diameter_m",
    ("body diameter",    None, "missile_airframe"): "body_diameter_m",
    ("calibre",          None, "missile_airframe"): "body_diameter_m",
    ("caliber",          None, "missile_airframe"): "body_diameter_m",

    # Total mass — when section_ctx is None, "Weight" / "Mass" map to total.
    ("weight",           None, "missile_airframe"): "total_mass_kg",
    ("weight kg",        None, "missile_airframe"): "total_mass_kg",
    ("mass",             None, "missile_airframe"): "total_mass_kg",
    ("mass kg",          None, "missile_airframe"): "total_mass_kg",
    ("total weight",     None, "missile_airframe"): "total_mass_kg",
    ("total weight kg",  None, "missile_airframe"): "total_mass_kg",
    ("launch weight",    None, "missile_airframe"): "total_mass_kg",
    ("launch mass",      None, "missile_airframe"): "total_mass_kg",

    # ============================================================
    # missile_speed_timing
    # ============================================================
    ("speed",            None, "missile_speed_timing"): "max_speed_mps",
    ("max speed",        None, "missile_speed_timing"): "max_speed_mps",
    ("max speed m s",    None, "missile_speed_timing"): "max_speed_mps",
    ("max speed mps",    None, "missile_speed_timing"): "max_speed_mps",
    ("velocity",         None, "missile_speed_timing"): "max_speed_mps",
    ("maximum velocity", None, "missile_speed_timing"): "max_speed_mps",
    ("average speed",    None, "missile_speed_timing"): "average_speed_mps",
    ("flight time",      None, "missile_speed_timing"): "flight_time_sec",
    ("time of flight",   None, "missile_speed_timing"): "flight_time_sec",
    ("flyout time",      None, "missile_speed_timing"): "max_flyout_time_sec",
    ("burn time",        None, "missile_speed_timing"): "total_burn_time_sec",

    # ============================================================
    # missile_propulsion
    # ============================================================
    # Booster (1st Stage) — Weight maps to booster_mass_kg.
    ("weight",      "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("weight kg",   "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("mass",        "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("mass kg",     "1st Stage", "missile_propulsion"): "booster_mass_kg",
    ("weight",      "Booster",   "missile_propulsion"): "booster_mass_kg",
    ("weight kg",   "Booster",   "missile_propulsion"): "booster_mass_kg",
    ("mass",        "Booster",   "missile_propulsion"): "booster_mass_kg",
    ("mass kg",     "Booster",   "missile_propulsion"): "booster_mass_kg",
    # Booster — Time maps to booster_time_sec.
    ("time",        "1st Stage", "missile_propulsion"): "booster_time_sec",
    ("time sec",    "1st Stage", "missile_propulsion"): "booster_time_sec",
    ("burn time",   "1st Stage", "missile_propulsion"): "booster_time_sec",
    ("time",        "Booster",   "missile_propulsion"): "booster_time_sec",
    ("time sec",    "Booster",   "missile_propulsion"): "booster_time_sec",
    ("burn time",   "Booster",   "missile_propulsion"): "booster_time_sec",
    # Booster — Thrust (string field; passthrough).
    ("thrust",      "1st Stage", "missile_propulsion"): "booster_thrust",
    ("thrust",      "Booster",   "missile_propulsion"): "booster_thrust",

    # Sustainer (2nd Stage) — Weight maps to sustain_mass_kg.
    ("weight",      "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("weight kg",   "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("mass",        "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("mass kg",     "2nd Stage", "missile_propulsion"): "sustain_mass_kg",
    ("weight",      "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("weight kg",   "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("mass",        "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("mass kg",     "Sustainer", "missile_propulsion"): "sustain_mass_kg",
    ("weight",      "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    ("weight kg",   "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    ("mass",        "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    ("mass kg",     "Sustain",   "missile_propulsion"): "sustain_mass_kg",
    # Sustainer — Time maps to sustain_time_sec.
    ("time",        "2nd Stage", "missile_propulsion"): "sustain_time_sec",
    ("time sec",    "2nd Stage", "missile_propulsion"): "sustain_time_sec",
    ("burn time",   "2nd Stage", "missile_propulsion"): "sustain_time_sec",
    ("time",        "Sustainer", "missile_propulsion"): "sustain_time_sec",
    ("time sec",    "Sustainer", "missile_propulsion"): "sustain_time_sec",
    ("burn time",   "Sustainer", "missile_propulsion"): "sustain_time_sec",
    ("time",        "Sustain",   "missile_propulsion"): "sustain_time_sec",
    ("time sec",    "Sustain",   "missile_propulsion"): "sustain_time_sec",
    ("burn time",   "Sustain",   "missile_propulsion"): "sustain_time_sec",
    # Sustainer — Thrust (string field; passthrough).
    ("thrust",      "2nd Stage", "missile_propulsion"): "sustain_thrust",
    ("thrust",      "Sustainer", "missile_propulsion"): "sustain_thrust",
    ("thrust",      "Sustain",   "missile_propulsion"): "sustain_thrust",

    # Ejector — Weight / Time / Thrust under Ejector section.
    ("weight",      "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("weight kg",   "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("mass",        "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("mass kg",     "Ejector",   "missile_propulsion"): "ejector_mass_kg",
    ("time",        "Ejector",   "missile_propulsion"): "ejector_time_sec",
    ("time sec",    "Ejector",   "missile_propulsion"): "ejector_time_sec",
    ("thrust",      "Ejector",   "missile_propulsion"): "ejector_thrust",
}
```

- [ ] **Step 4: Run all alias-map tests.**

Run:
```bash
pytest docker/docling-graph/tests/test_alias_map.py docker/docling-graph/tests/test_alias_map_missile.py -v 2>&1 | tail -25
```

Expected: All tests pass — drift-guard tests AND missile-specific resolution tests both green.

If drift-guard fails (e.g., `Token "alt" missing from §12b prose`), reconcile by checking what §12b actually says — `alt` may need to be added as a prose mention OR `("max alt", None, ...)` may need to be removed from ALIAS_MAP. The §12b prose IS the LLM-facing SSoT and takes precedence.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_alias_map.py docker/docling-graph/tests/test_alias_map_missile.py
git commit -m "$(cat <<'EOF'
feat(extraction): populate ALIAS_MAP missile entries (4 passes)

Adds entries for missile_kinematics, missile_airframe, missile_speed_timing,
missile_propulsion. Section-conditional entries for propulsion (1st Stage /
Booster / 2nd Stage / Sustainer / Sustain / Ejector) anchor the headline
acceptance case. Drift-guard tests enforce per-token §12b prose mentions.

Refs: docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md §5.1
EOF
)"
```

---

### Task 5: Populate ALIAS_MAP for radar passes

**Files:**
- Modify: `docker/docling-graph/app/_alias_map.py` (extend ALIAS_MAP with radar entries)
- Create: `docker/docling-graph/tests/test_alias_map_radar.py`

- [ ] **Step 1: Write failing tests for radar aliases.**

Create `docker/docling-graph/tests/test_alias_map_radar.py`:

```python
"""Per-pass alias-resolution tests for radar passes (spec §5.5)."""
import importlib.util
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _APP_DIR / fname)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def _resolve(label, section, pass_name):
    am = _load("am", "_alias_map.py")
    tf = _load("tf", "_table_facts.py")
    return am.ALIAS_MAP.get((tf.normalize_label(label), section, pass_name))


def test_radar_power_rf_frequency():
    assert _resolve("Frequency MHz", None, "radar_power_rf") == "nominal_rf_mhz"
    assert _resolve("Operating Frequency", None, "radar_power_rf") == "nominal_rf_mhz"
    assert _resolve("Carrier Frequency", None, "radar_power_rf") == "nominal_rf_mhz"


def test_radar_power_rf_peak_power():
    assert _resolve("Peak Power", None, "radar_power_rf") == "tx_peak_power_kw"
    assert _resolve("Tx Power", None, "radar_power_rf") == "tx_peak_power_kw"


def test_radar_timing_pri_aliases():
    assert _resolve("PRI", None, "radar_timing") == "nominal_pri_usec"
    assert _resolve("Pulse Repetition Interval", None, "radar_timing") == "nominal_pri_usec"


def test_radar_timing_pulse_width():
    assert _resolve("PW", None, "radar_timing") == "nominal_pd_usec"
    assert _resolve("Pulse Width", None, "radar_timing") == "nominal_pd_usec"
    assert _resolve("Pulse Duration", None, "radar_timing") == "nominal_pd_usec"


def test_radar_antenna_gain():
    assert _resolve("Antenna Gain", None, "radar_antenna") == "gain_dbi"


def test_radar_antenna_beamwidth():
    assert _resolve("Azimuth Beamwidth", None, "radar_antenna") == "beamwidth_az_deg"
    assert _resolve("Elevation Beamwidth", None, "radar_antenna") == "beamwidth_el_deg"


def test_radar_modulation_chirp_bandwidth():
    assert _resolve("Chirp Bandwidth", None, "radar_modulation") == "frequency_excursion_mhz"


def test_radar_pass_isolation():
    """Radar labels do NOT resolve in missile passes."""
    assert _resolve("PRI", None, "missile_propulsion") is None
    assert _resolve("Antenna Gain", None, "missile_kinematics") is None
```

- [ ] **Step 2: Run tests, verify they fail.**

Run: `pytest docker/docling-graph/tests/test_alias_map_radar.py -v 2>&1 | tail -10`
Expected: ALL fail.

- [ ] **Step 3: Extend ALIAS_MAP with radar entries.**

Edit `docker/docling-graph/app/_alias_map.py`. Append BEFORE the closing `}` of the existing ALIAS_MAP (after the missile_propulsion ejector entries):

```python
    # ============================================================
    # radar_power_rf
    # ============================================================
    ("frequency",           None, "radar_power_rf"): "nominal_rf_mhz",
    ("frequency mhz",       None, "radar_power_rf"): "nominal_rf_mhz",
    ("operating frequency", None, "radar_power_rf"): "nominal_rf_mhz",
    ("carrier frequency",   None, "radar_power_rf"): "nominal_rf_mhz",
    ("rf",                  None, "radar_power_rf"): "nominal_rf_mhz",

    ("peak power",          None, "radar_power_rf"): "tx_peak_power_kw",
    ("transmitter power",   None, "radar_power_rf"): "tx_peak_power_kw",
    ("tx power",            None, "radar_power_rf"): "tx_peak_power_kw",

    ("erp",                          None, "radar_power_rf"): "erp_dbw",
    ("effective radiated power",     None, "radar_power_rf"): "erp_dbw",

    # ============================================================
    # radar_timing
    # ============================================================
    ("pri",                       None, "radar_timing"): "nominal_pri_usec",
    ("pulse repetition interval", None, "radar_timing"): "nominal_pri_usec",
    ("pulse interval",            None, "radar_timing"): "nominal_pri_usec",

    ("pw",                        None, "radar_timing"): "nominal_pd_usec",
    ("pulse width",               None, "radar_timing"): "nominal_pd_usec",
    ("pulse duration",            None, "radar_timing"): "nominal_pd_usec",

    ("scan period",  None, "radar_timing"): "scan_period_sec",
    ("scan time",    None, "radar_timing"): "scan_period_sec",
    ("rotation period", None, "radar_timing"): "scan_period_sec",

    ("dwell",        None, "radar_timing"): "dwell_time",
    ("dwell time",   None, "radar_timing"): "dwell_time",

    # ============================================================
    # radar_antenna
    # ============================================================
    ("antenna gain",     None, "radar_antenna"): "gain_dbi",

    ("antenna width",       None, "radar_antenna"): "antenna_dim_az_m",
    ("azimuth aperture",    None, "radar_antenna"): "antenna_dim_az_m",
    ("antenna height",      None, "radar_antenna"): "antenna_dim_el_m",
    ("elevation aperture",  None, "radar_antenna"): "antenna_dim_el_m",

    ("azimuth beamwidth",   None, "radar_antenna"): "beamwidth_az_deg",
    ("elevation beamwidth", None, "radar_antenna"): "beamwidth_el_deg",
    ("elevation coverage",  None, "radar_antenna"): "coverage_limits_el_deg",

    # ============================================================
    # radar_modulation
    # ============================================================
    ("chirp bandwidth",     None, "radar_modulation"): "frequency_excursion_mhz",
    ("frequency excursion", None, "radar_modulation"): "frequency_excursion_mhz",
    ("sweep width",         None, "radar_modulation"): "frequency_excursion_mhz",

    ("code length",      None, "radar_modulation"): "num_bits_in_code",
    ("chips",            None, "radar_modulation"): "num_bits_in_code",
    ("bits",             None, "radar_modulation"): "num_bits_in_code",

    ("pulses per dwell", None, "radar_modulation"): "pulses_per_dwell",
```

- [ ] **Step 4: Run all alias-map tests (drift guard + missile + radar).**

Run:
```bash
pytest docker/docling-graph/tests/test_alias_map.py docker/docling-graph/tests/test_alias_map_missile.py docker/docling-graph/tests/test_alias_map_radar.py -v 2>&1 | tail -25
```

Expected: All pass — drift guard accepts every new label-token has a §12b mention; missile + radar resolution tests green.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_alias_map.py docker/docling-graph/tests/test_alias_map_radar.py
git commit -m "feat(extraction): populate ALIAS_MAP radar entries (5 sub-passes)"
```

---

### Task 6: Add `_collect_all_field_names` helper + schema-side drift-guard test

**Files:**
- Modify: `docker/docling-graph/tests/test_alias_map.py` (add the third drift-guard test)

- [ ] **Step 1: Add the third drift-guard test for schema-side drift.**

Append to `docker/docling-graph/tests/test_alias_map.py`:

```python
def _collect_all_field_names(template_cls) -> set[str]:
    """Two-level walk: template_cls -> each entity item class -> model_fields.
    Mirrors the pattern in app/_field_provenance_helpers.py and
    docling-graph's provenance walker. Catches schema-side drift if a field
    is renamed or removed."""
    all_fields = set(template_cls.model_fields.keys())
    for fname, finfo in template_cls.model_fields.items():
        # If this field's annotation is a list[ItemClass] or ItemClass,
        # introspect the item class's fields too.
        annotation = finfo.annotation
        item_cls = None
        # list[X] case
        if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            args = getattr(annotation, "__args__", ())
            if args and hasattr(args[0], "model_fields"):
                item_cls = args[0]
        # Direct BaseModel case
        elif hasattr(annotation, "model_fields"):
            item_cls = annotation
        if item_cls is not None:
            all_fields.update(item_cls.model_fields.keys())
    return all_fields


def test_alias_map_target_fields_exist_on_schemas():
    """Every ALIAS_MAP value (target schema field) must exist as a field on
    the schema for the corresponding pass. Catches drift where a schema is
    refactored and the alias map still points at a renamed/removed field."""
    am = _aliases()
    # Group ALIAS_MAP entries by pass.
    by_pass: dict[str, set[str]] = {}
    for (_label, _section, pass_name), schema_field in am.ALIAS_MAP.items():
        by_pass.setdefault(pass_name, set()).add(schema_field)

    # Resolve template classes via the bundle loader. Mirrors how main.py
    # loads them at extract-pass time.
    from app.bundles import load_pass_template
    for pass_name, fields in by_pass.items():
        template_cls = load_pass_template("air_defense_v3", pass_name)
        actual_fields = _collect_all_field_names(template_cls)
        missing = fields - actual_fields
        assert not missing, (
            f"ALIAS_MAP entries for pass {pass_name!r} reference fields "
            f"{missing!r} that do not exist on the schema. The schema may "
            f"have been refactored; reconcile the alias map."
        )
```

- [ ] **Step 2: Run tests.**

Run: `pytest docker/docling-graph/tests/test_alias_map.py -v 2>&1 | tail -15`

Expected: All 5 tests pass (4 prior + 1 new). If the new test fails with `'sustain_thrust' missing` or similar, the schema field doesn't exist as named — reconcile by either renaming the alias-map target OR adding the missing schema field.

- [ ] **Step 3: Commit.**

```bash
git add docker/docling-graph/tests/test_alias_map.py
git commit -m "test(extraction): schema-side drift guard for ALIAS_MAP target fields"
```

---

## Chunk 2: Pure functions

Tasks 7–13 add the six pipeline functions plus their unit tests. After Chunk 2,
all building blocks exist but the orchestrator does not yet wire them
together — `synthesize_table_facts` is still un-callable. Chunk 2 tasks are
parallelizable: 7 (shape) and 8 (label rows) and 9 (entity ids) only depend on
Chunk 1 outputs; 10 (sections) depends on 8; 11 (resolve) depends on 3 + 4 + 5;
12 (coerce) depends on 3; 13 (emit) depends on 1 only.

### Task 7: `detect_table_shape`

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py` (append function)
- Create: `docker/docling-graph/tests/test_table_facts_shape.py`

- [ ] **Step 1: Write failing tests.**

Create `docker/docling-graph/tests/test_table_facts_shape.py`:

```python
"""Tests for detect_table_shape (spec §5.2)."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def _cell(text, row, col, *, row_header=False, col_header=False, row_span=1, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + row_span,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": row_span,
        "col_span": col_span,
        "row_header": row_header,
        "column_header": col_header,
    }


def test_below_4x4_floor_returns_other():
    """Tables smaller than 4 rows × 4 cols are skipped."""
    tf = _load()
    table = {"data": {"table_cells": [_cell("a", 0, 0)], "num_rows": 1, "num_cols": 1}}
    assert tf.detect_table_shape(table) == tf.Shape.OTHER


def test_column_major_detection():
    """Leftmost col has row_header=True majority."""
    tf = _load()
    cells = []
    # Col 0: 4 row_header cells (the labels)
    for r in range(4):
        cells.append(_cell(f"label{r}", r, 0, row_header=True))
    # Cols 1-3: data cells
    for r in range(4):
        for c in range(1, 4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    assert tf.detect_table_shape(table) == tf.Shape.COLUMN_MAJOR


def test_row_major_detection():
    """Top row has column_header=True majority."""
    tf = _load()
    cells = []
    # Row 0: 4 column_header cells
    for c in range(4):
        cells.append(_cell(f"hdr{c}", 0, c, col_header=True))
    # Rows 1-3: data cells
    for r in range(1, 4):
        for c in range(4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    assert tf.detect_table_shape(table) == tf.Shape.ROW_MAJOR


def test_hybrid_multi_row_left_labels():
    """Multiple rows have row_header=True at col 0 with no data — composite identity."""
    tf = _load()
    cells = []
    # Rows 0 and 1: identity labels in col 0 (multi-row identity region)
    cells.append(_cell("Industry Designation", 0, 0, row_header=True))
    cells.append(_cell("Missile Type", 1, 0, row_header=True))
    # Row 2-3: spec rows
    cells.append(_cell("Length mm", 2, 0, row_header=True))
    cells.append(_cell("Weight kg", 3, 0, row_header=True))
    # Cols 1-3: data
    for r in range(4):
        for c in range(1, 4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    # All col 0 cells are row_header — count is 4. With ≥ 2 of them at the
    # top being identity-pattern labels (Industry/Missile/Variant/etc),
    # detect HYBRID.
    shape = tf.detect_table_shape(table)
    assert shape in (tf.Shape.HYBRID, tf.Shape.COLUMN_MAJOR)
    # The HYBRID-vs-COLUMN_MAJOR distinction is decided in derive_entity_ids
    # (Task 9) based on whether multiple key-label rows are present.
    # detect_table_shape can return either — both paths produce the same
    # downstream behavior for the column-major-with-multi-row-id case.


def test_other_shape_when_neither_pattern_matches():
    tf = _load()
    cells = []
    for r in range(4):
        for c in range(4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    assert tf.detect_table_shape(table) == tf.Shape.OTHER
```

- [ ] **Step 2: Run tests, verify they fail.**

Run: `pytest docker/docling-graph/tests/test_table_facts_shape.py -v 2>&1 | tail -10`

- [ ] **Step 3: Add detect_table_shape to _table_facts.py.**

Append:

```python
# ============================================================
# Pipeline step 1: detect_table_shape (spec §5.2 / D1)
# ============================================================

def detect_table_shape(table: dict) -> Shape:
    """Classify a DoclingDocument table cell-shape into one of four buckets.

    COLUMN_MAJOR: ≥50% of leftmost-col cells flagged row_header=True.
    ROW_MAJOR: ≥50% of top-row cells flagged column_header=True.
    HYBRID: column-major with multiple identity rows (left col has more
        than one row_header cell whose value is a key-label pattern).
    OTHER: below 4×4 floor, or neither pattern fires.
    """
    data = (table or {}).get("data") or {}
    cells = data.get("table_cells") or []
    num_rows = data.get("num_rows") or 0
    num_cols = data.get("num_cols") or 0

    if num_rows < 4 or num_cols < 4 or not cells:
        return Shape.OTHER

    col0_cells = [c for c in cells if c.get("start_col_offset_idx") == 0]
    row0_cells = [c for c in cells if c.get("start_row_offset_idx") == 0]

    col0_rh = sum(1 for c in col0_cells if c.get("row_header") is True)
    row0_ch = sum(1 for c in row0_cells if c.get("column_header") is True)

    is_col_major = col0_cells and col0_rh * 2 >= len(col0_cells)
    is_row_major = row0_cells and row0_ch * 2 >= len(row0_cells)

    if is_col_major and not is_row_major:
        # Distinguish HYBRID by counting row_header cells in col 0 that
        # match identity patterns. The patterns are intentionally local to
        # detect_table_shape (single-purpose); derive_entity_ids has its
        # own list. Both are kept in sync via a constant defined below.
        identity_count = sum(
            1 for c in col0_cells
            if c.get("row_header") is True
            and _looks_like_key_label((c.get("text") or "").strip())
        )
        return Shape.HYBRID if identity_count >= 2 else Shape.COLUMN_MAJOR
    if is_row_major and not is_col_major:
        return Shape.ROW_MAJOR
    return Shape.OTHER


# Identity-row label patterns. Cells matching any of these (case-insensitive
# substring) are treated as entity-naming labels, not spec labels. Shared by
# detect_table_shape (HYBRID detection) and derive_entity_ids (Task 9).
_KEY_LABEL_PATTERNS = (
    "missile type",
    "missile variant",
    "industry designation",
    "military designation",
    "nato designation",
    "fan song variant",
    "radar variant",
    "system name",
    "system designation",
    "designation",
    "variant",
)


def _looks_like_key_label(label: str) -> bool:
    if not label:
        return False
    norm = label.strip().lower()
    return any(pat in norm for pat in _KEY_LABEL_PATTERNS)
```

- [ ] **Step 4: Run tests.**

Expected: 5/5 pass.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_shape.py
git commit -m "feat(extraction): detect_table_shape — D1 column-major/row-major/hybrid/other classifier"
```

---

### Task 8: `extract_label_rows`

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py` (append)
- Create: `docker/docling-graph/tests/test_table_facts_extract.py`

- [ ] **Step 1: Write failing tests.**

Create `docker/docling-graph/tests/test_table_facts_extract.py`:

```python
"""Tests for extract_label_rows (spec §5.3)."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def _cell(text, row, col, *, row_header=False, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + 1,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": 1, "col_span": col_span,
        "row_header": row_header, "column_header": False,
    }


def _column_major_table_3rows_3entities():
    """3 rows × 4 cols (1 label + 3 entity columns)."""
    cells = [
        _cell("Length mm", 0, 0, row_header=True),
        _cell("10726", 0, 1), _cell("10841", 0, 2), _cell("10778", 0, 3),
        _cell("Diameter mm", 1, 0, row_header=True),
        _cell("654", 1, 1), _cell("654", 1, 2), _cell("654", 1, 3),
        _cell("Weight kg", 2, 0, row_header=True),
        _cell("2163", 2, 1), _cell("2283", 2, 2), _cell("2391", 2, 3),
    ]
    return {"data": {"table_cells": cells, "num_rows": 3, "num_cols": 4}}


def test_column_major_extraction_basic():
    tf = _load()
    table = _column_major_table_3rows_3entities()
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)
    assert len(rows) == 3
    assert rows[0]["label_text"] == "Length mm"
    assert rows[0]["data_cells"] == {1: "10726", 2: "10841", 3: "10778"}
    assert rows[1]["label_text"] == "Diameter mm"
    assert rows[2]["label_text"] == "Weight kg"


def test_column_major_label_col_span_carried_through():
    tf = _load()
    cells = [
        _cell("Industry Designation", 0, 0, row_header=True, col_span=2),
        _cell("SA-75", 0, 2), _cell("S-75", 0, 3), _cell("S-75M", 0, 4),
        _cell("Length mm", 1, 0, row_header=True, col_span=2),
        _cell("10726", 1, 2), _cell("10841", 1, 3), _cell("10778", 1, 4),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 2, "num_cols": 5}}
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)
    assert rows[0]["label_text"] == "Industry Designation"
    assert rows[0]["label_col_span"] == 2
    # Data cells are at col 2, 3, 4 — col_span 2 means label spans 0-1.
    assert rows[0]["data_cells"] == {2: "SA-75", 3: "S-75", 4: "S-75M"}


def test_row_major_transposition():
    """Top row holds labels; each subsequent row is one entity."""
    tf = _load()
    cells = [
        # Row 0: column headers (treated as labels in row-major mode)
        {**_cell("System", 0, 0), "column_header": True},
        {**_cell("Length mm", 0, 1), "column_header": True},
        {**_cell("Weight kg", 0, 2), "column_header": True},
        # Rows 1-3: entities
        _cell("1D",  1, 0), _cell("10726", 1, 1), _cell("2163", 1, 2),
        _cell("13D", 2, 0), _cell("10841", 2, 1), _cell("2283", 2, 2),
        _cell("20D", 3, 0), _cell("10778", 3, 1), _cell("2391", 3, 2),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 3}}
    rows = tf.extract_label_rows(table, tf.Shape.ROW_MAJOR)
    # In row-major, each label-column becomes a virtual row — but we
    # transpose so consumers can use the same downstream pipeline.
    # Convention: each column header (excluding the first identity column)
    # becomes one LabelRow with data_cells keyed by entity-row-idx.
    assert len(rows) == 2  # Length mm + Weight kg (System is identity, skipped)
    by_label = {r["label_text"]: r for r in rows}
    assert by_label["Length mm"]["data_cells"] == {1: "10726", 2: "10841", 3: "10778"}
    assert by_label["Weight kg"]["data_cells"] == {1: "2163", 2: "2283", 3: "2391"}


def test_skips_empty_label_rows():
    tf = _load()
    cells = [
        _cell("Length mm", 0, 0, row_header=True),
        _cell("10726", 0, 1), _cell("10841", 0, 2), _cell("10778", 0, 3),
        _cell("", 1, 0, row_header=True),  # blank label row — skipped
        _cell("654", 1, 1), _cell("654", 1, 2), _cell("654", 1, 3),
        _cell("Weight kg", 2, 0, row_header=True),
        _cell("2163", 2, 1), _cell("2283", 2, 2), _cell("2391", 2, 3),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 3, "num_cols": 4}}
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)
    assert len(rows) == 2
    assert {r["label_text"] for r in rows} == {"Length mm", "Weight kg"}
```

- [ ] **Step 2: Run tests, verify they fail.**

- [ ] **Step 3: Implement extract_label_rows.**

Append to `_table_facts.py`:

```python
# ============================================================
# Pipeline step 2: extract_label_rows (spec §5.3)
# ============================================================

def extract_label_rows(table: dict, shape: Shape) -> list[LabelRow]:
    """Normalize column-major and row-major into a unified LabelRow stream.

    Column-major: leftmost label column(s) -> LabelRow.label_text; remaining
        cols are data_cells keyed by entity_col.
    Row-major: top header row labels -> one LabelRow per non-identity column;
        rows below provide data_cells keyed by entity row_idx (transposed).
    HYBRID: same as COLUMN_MAJOR; the multi-row identity region is just a
        sequence of identity-label rows that derive_entity_ids consumes.
    OTHER: returns [].
    """
    if shape == Shape.OTHER:
        return []

    cells = (table or {}).get("data", {}).get("table_cells") or []
    if not cells:
        return []

    if shape in (Shape.COLUMN_MAJOR, Shape.HYBRID):
        return _extract_column_major(cells)
    if shape == Shape.ROW_MAJOR:
        return _extract_row_major(cells)
    return []


def _label_column_width(cells: list[dict]) -> int:
    """Return how many leftmost columns the row-label cells span."""
    width = 1
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        end_col = c.get("end_col_offset_idx", 1) or 1
        if end_col > width:
            width = end_col
    return width


def _extract_column_major(cells: list[dict]) -> list[LabelRow]:
    label_width = _label_column_width(cells)

    rows_by_idx: dict[int, LabelRow] = {}
    # First pass: collect labels (row_header cells in label region).
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        row_idx = c.get("start_row_offset_idx")
        if row_idx is None:
            continue
        rows_by_idx[row_idx] = LabelRow(
            row_idx=row_idx,
            label_text=text,
            label_col_span=c.get("col_span", 1) or 1,
            data_cells={},
        )

    # Second pass: collect data cells (col >= label_width).
    for c in cells:
        col = c.get("start_col_offset_idx")
        if col is None or col < label_width:
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        row_idx = c.get("start_row_offset_idx")
        if row_idx is None or row_idx not in rows_by_idx:
            continue
        rows_by_idx[row_idx]["data_cells"][col] = text

    return [rows_by_idx[k] for k in sorted(rows_by_idx)]


def _extract_row_major(cells: list[dict]) -> list[LabelRow]:
    # Top-row column headers; assume col 0 is identity, cols 1+ are spec labels.
    header_cells = [c for c in cells if c.get("start_row_offset_idx") == 0
                    and c.get("column_header") is True]
    if not header_cells:
        return []
    # Identity column is the leftmost column header; skip it.
    sorted_headers = sorted(header_cells, key=lambda c: c.get("start_col_offset_idx", 0))
    if not sorted_headers:
        return []
    spec_headers = sorted_headers[1:]  # skip identity column

    rows_by_label: dict[str, LabelRow] = {}
    for header in spec_headers:
        col = header.get("start_col_offset_idx")
        text = (header.get("text") or "").strip()
        if not text or col is None:
            continue
        rows_by_label[text] = LabelRow(
            row_idx=col,  # in row-major, "row_idx" of the synthetic LabelRow is the source col
            label_text=text,
            label_col_span=1,
            data_cells={},
        )

    # Collect data cells (rows below 0, at the columns we care about).
    label_cols = {h["row_idx"]: h["label_text"] for h in rows_by_label.values()}
    for c in cells:
        row = c.get("start_row_offset_idx")
        col = c.get("start_col_offset_idx")
        if row is None or row == 0:
            continue
        if col not in label_cols:
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        rows_by_label[label_cols[col]]["data_cells"][row] = text

    return list(rows_by_label.values())
```

- [ ] **Step 4: Run tests, verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_extract.py -v 2>&1 | tail -10`
Expected: 4/4 pass.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_extract.py
git commit -m "feat(extraction): extract_label_rows — D1 shape-agnostic LabelRow extraction"
```

---

### Task 9: `derive_entity_ids`

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py` (append)
- Create: `docker/docling-graph/tests/test_table_facts_entity_ids.py`

- [ ] **Step 1: Write failing tests.**

```python
"""Tests for derive_entity_ids (spec §5.3.5)."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def _row(idx, label, data):
    return {"row_idx": idx, "label_text": label, "label_col_span": 1, "data_cells": data}


def test_single_key_label_row():
    tf = _load()
    rows = [
        _row(0, "Missile Type", {1: "1D", 2: "13D", 3: "13DM"}),
        _row(1, "Length mm",    {1: "10726", 2: "10841", 3: "10841"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.COLUMN_MAJOR)
    assert ids == {1: "1D", 2: "13D", 3: "13DM"}


def test_hybrid_composite_identity():
    tf = _load()
    rows = [
        _row(0, "Industry Designation", {1: "SA-75", 2: "S-75", 3: "S-75M"}),
        _row(1, "Missile Type",         {1: "1D",    2: "13D",  3: "13DM"}),
        _row(2, "Length mm",            {1: "10726", 2: "10841", 3: "10841"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.HYBRID)
    assert ids == {1: "SA-75 1D", 2: "S-75 13D", 3: "S-75M 13DM"}


def test_no_key_label_row_returns_empty():
    """If no row matches _KEY_LABEL_PATTERNS, derive_entity_ids returns {}."""
    tf = _load()
    rows = [
        _row(0, "Length mm", {1: "10726", 2: "10841"}),
        _row(1, "Weight kg", {1: "2163", 2: "2283"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.COLUMN_MAJOR)
    assert ids == {}


def test_empty_data_cell_excluded_from_composite():
    """Hybrid composite with one column missing the upper id — only complete columns appear."""
    tf = _load()
    rows = [
        _row(0, "Industry Designation", {1: "SA-75", 2: "", 3: "S-75M"}),
        _row(1, "Missile Type",         {1: "1D",    2: "13D",  3: "13DM"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.HYBRID)
    # Col 2 has no Industry Designation; composite skips that token but keeps the rest.
    # Implementation choice: concatenate non-empty cells only.
    assert ids[1] == "SA-75 1D"
    assert ids[2] == "13D"  # only Missile Type populated for col 2
    assert ids[3] == "S-75M 13DM"


def test_collision_last_write_wins():
    """Two columns producing the same composite — last one wins.
    derive_entity_ids deduplicates so only one entry per unique composite
    appears in the result. The orchestrator detects collision counts by
    comparing all_cols size to returned ids size."""
    tf = _load()
    rows = [
        _row(0, "Missile Type", {1: "1D", 2: "1D", 3: "13DM"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.COLUMN_MAJOR)
    # "1D" appeared in cols 1 and 2; only col 2 (last) should appear.
    # "13DM" appeared in col 3; appears in result.
    assert ids == {2: "1D", 3: "13DM"}
    # Total source cols: 3. Returned ids: 2. Collision count = 1.
```

- [ ] **Step 2: Run tests, verify they fail.**

Run: `pytest docker/docling-graph/tests/$TEST_FILE -v 2>&1 | tail -10`
Expected: FAIL — function not yet defined (NameError or AttributeError).

- [ ] **Step 3: Implement.**

```python
# ============================================================
# Pipeline step 3: derive_entity_ids (spec §5.3.5)
# ============================================================

def derive_entity_ids(rows: list[LabelRow], shape: Shape) -> dict[int, str]:
    """Map entity_col -> entity_id from key-label rows.

    For COLUMN_MAJOR: single key-label row's data_cells become entity_ids.
    For HYBRID: multiple key-label rows produce composite identities by
        concatenating non-empty cells in row order.

    Composite collisions (two columns producing the same entity_id) are
    resolved last-write-wins: only the latest column with that identity
    appears in the returned dict. The orchestrator detects collisions by
    comparing the count of source columns to the count of returned ids
    (incrementing FactStats.hybrid_collisions for the difference).
    """
    key_rows = [r for r in rows if _looks_like_key_label(r["label_text"])]
    if not key_rows:
        return {}

    # Collect all entity_cols seen across key rows.
    all_cols: set[int] = set()
    for kr in key_rows:
        all_cols.update(kr["data_cells"].keys())

    # Build (col -> composite_id) preserving column iteration order.
    raw: dict[int, str] = {}
    for col in sorted(all_cols):
        parts = []
        for kr in key_rows:  # rows already sorted by row_idx
            cell = kr["data_cells"].get(col, "").strip()
            if cell:
                parts.append(cell)
        if parts:
            raw[col] = " ".join(parts)

    # Apply last-write-wins on duplicate composites: track which
    # composite_id last appeared at which column, then keep only those cols.
    last_col_for_id: dict[str, int] = {}
    for col in sorted(raw):
        last_col_for_id[raw[col]] = col

    return {col: composite for composite, col in last_col_for_id.items()}
```

- [ ] **Step 4: Run tests, verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_entity_ids.py -v 2>&1 | tail -10`
Expected: 5/5 pass.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_entity_ids.py
git commit -m "feat(extraction): derive_entity_ids — column-to-entity mapping with HYBRID composite"
```

---

### Task 10: `detect_section_context`

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py`
- Create: `docker/docling-graph/tests/test_table_facts_sections.py`

- [ ] **Step 1: Write failing tests.**

```python
"""Tests for detect_section_context (spec §5.4)."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def _row(idx, label, data):
    return {"row_idx": idx, "label_text": label, "label_col_span": 1, "data_cells": data}


def test_embedded_section_keyword_in_label():
    """SA-2 PDF style: '1st Stage Weight kg' has section embedded in label."""
    tf = _load()
    rows = [
        _row(0, "Length mm",            {1: "10726"}),
        _row(1, "1st Stage Weight kg",  {1: "1135"}),
        _row(2, "1st Stage Time sec",   {1: "4.0"}),
        _row(3, "2nd Stage Weight kg",  {1: "1028"}),
    ]
    result = tf.detect_section_context(rows)
    contexts = {r["row_idx"]: ctx for r, ctx in result}
    assert contexts[0] is None
    assert contexts[1] == "1st Stage"
    assert contexts[2] == "1st Stage"
    assert contexts[3] == "2nd Stage"


def test_embedded_strips_section_from_label_text():
    """When a section keyword is embedded, the returned LabelRow's label_text
    has the section keyword removed so resolve_alias can look up the bare
    label without the section prefix."""
    tf = _load()
    rows = [
        _row(1, "1st Stage Weight kg",  {1: "1135"}),
    ]
    result = tf.detect_section_context(rows)
    new_row, ctx = result[0]
    assert ctx == "1st Stage"
    assert new_row["label_text"] == "Weight kg"  # section prefix stripped


def test_header_row_strategy():
    """Header-row marker rows propagate context to subsequent rows."""
    tf = _load()
    rows = [
        _row(0, "Missile Type", {1: "1D", 2: "13D"}),
        _row(1, "Total Weight kg", {1: "2163", 2: "2283"}),
        _row(2, "1st Stage", {}),  # header-row marker (no data cells)
        _row(3, "Weight kg",  {1: "1135", 2: "1032"}),
        _row(4, "Time sec",   {1: "4.0", 2: "4.0"}),
        _row(5, "2nd Stage", {}),
        _row(6, "Weight kg",  {1: "1028", 2: "1251"}),
    ]
    result = tf.detect_section_context(rows)
    contexts = {r["row_idx"]: ctx for r, ctx in result if r["row_idx"] not in (2, 5)}
    assert contexts[0] is None
    assert contexts[1] is None
    assert contexts[3] == "1st Stage"
    assert contexts[4] == "1st Stage"
    assert contexts[6] == "2nd Stage"
    # The header-row markers (rows 2 and 5) themselves are dropped from
    # the result (they have no data cells anyway).
    out_rows = [r for r, _ in result]
    out_idxs = {r["row_idx"] for r in out_rows}
    assert 2 not in out_idxs
    assert 5 not in out_idxs


def test_embedded_wins_over_header_row():
    """If a row has both an embedded keyword and a propagating header-row
    context that would assign a different keyword, embedded wins."""
    tf = _load()
    rows = [
        _row(0, "1st Stage", {}),  # header-row marker
        _row(1, "2nd Stage Weight kg", {1: "1028"}),  # embedded "2nd Stage"
    ]
    result = tf.detect_section_context(rows)
    contexts = {r["row_idx"]: ctx for r, ctx in result if r["row_idx"] != 0}
    assert contexts[1] == "2nd Stage"  # embedded wins
```

- [ ] **Step 2: Run tests, verify they fail.**

Run: `pytest docker/docling-graph/tests/$TEST_FILE -v 2>&1 | tail -10`
Expected: FAIL — function not yet defined (NameError or AttributeError).

- [ ] **Step 3: Implement.**

```python
# ============================================================
# Pipeline step 4: detect_section_context (spec §5.4 / D2)
# ============================================================

def detect_section_context(
    rows: list[LabelRow],
) -> list[tuple[LabelRow, SectionContext]]:
    """Pair each LabelRow with its section_ctx using a two-strategy chain.

    Strategy 1 (embedded): substring scan of label_text against
        SECTION_KEYWORDS. If matched, that row's section_ctx is the matched
        keyword AND the keyword is stripped from label_text in the returned
        row (so resolve_alias keys on the bare label).
    Strategy 2 (header-row): track most recent row whose label_text equals
        a section keyword AND whose data_cells are empty/header-like.
        Subsequent rows inherit until the next header-row or end-of-table.
        Header-row marker rows themselves are dropped from the result.

    Conflict resolution: embedded wins.
    """
    # Lazy import to avoid circular reference if tests load standalone.
    from app._alias_map import SECTION_KEYWORDS

    out: list[tuple[LabelRow, SectionContext]] = []
    current_header_section: str | None = None

    for row in rows:
        label = row["label_text"]
        # Strategy 1: embedded keyword scan (case-insensitive).
        embedded = _find_embedded_keyword(label, SECTION_KEYWORDS)
        if embedded is not None:
            new_label = _strip_keyword(label, embedded)
            new_row: LabelRow = {
                "row_idx": row["row_idx"],
                "label_text": new_label,
                "label_col_span": row["label_col_span"],
                "data_cells": row["data_cells"],
            }
            out.append((new_row, embedded))
            continue

        # Strategy 2: header-row marker. A row whose label IS a section
        # keyword (after normalize) AND whose data_cells are empty/header-like
        # acts as a context propagator. Drop the marker itself.
        if _is_header_row_marker(row, SECTION_KEYWORDS):
            current_header_section = _matching_keyword(label, SECTION_KEYWORDS)
            continue

        out.append((row, current_header_section))

    return out


def _find_embedded_keyword(label: str, keywords: tuple[str, ...]) -> str | None:
    label_lower = label.lower()
    # Iterate longest-first so "1st Stage" matches before "Stage".
    for kw in sorted(keywords, key=len, reverse=True):
        if kw.lower() in label_lower:
            return kw
    return None


def _strip_keyword(label: str, keyword: str) -> str:
    """Remove a case-insensitive occurrence of keyword from label, collapsing
    whitespace. Example: '1st Stage Weight kg' + '1st Stage' -> 'Weight kg'."""
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    stripped = pattern.sub("", label, count=1)
    return " ".join(stripped.split())


def _is_header_row_marker(row: LabelRow, keywords: tuple[str, ...]) -> bool:
    """Marker rows have a section keyword as their label AND no real data."""
    matching = _matching_keyword(row["label_text"], keywords)
    if matching is None:
        return False
    # Empty data_cells, OR all data_cells values are empty/whitespace.
    cells = row["data_cells"]
    if not cells:
        return True
    return all(not (v or "").strip() for v in cells.values())


def _matching_keyword(label: str, keywords: tuple[str, ...]) -> str | None:
    """Return the keyword IF the label IS exactly that keyword (after normalize)."""
    label_norm = " ".join(label.lower().split())
    for kw in keywords:
        if kw.lower() == label_norm:
            return kw
    return None
```

- [ ] **Step 4: Run tests, verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_sections.py -v 2>&1 | tail -10`
Expected: 4/4 pass.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_sections.py
git commit -m "feat(extraction): detect_section_context — D2 embedded + header-row strategies"
```

---

### Task 11: `resolve_alias`

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py`
- Create: `docker/docling-graph/tests/test_table_facts_resolve.py`

- [ ] **Step 1: Write failing tests.**

```python
"""Tests for resolve_alias (spec §5.5).

The actual data-driven resolution is exhaustively tested in
test_alias_map_missile.py and test_alias_map_radar.py via direct dict
lookups; these tests verify the resolve_alias wrapper handles
normalization and the None-fallback path.
"""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def test_resolve_uses_normalize_label():
    """Caller passes raw label; resolve_alias normalizes internally."""
    tf = _load()
    # "Length mm" with various punctuation/case variations should all resolve.
    assert tf.resolve_alias("Length mm", None, "missile_airframe") == "body_length_m"
    assert tf.resolve_alias("LENGTH MM", None, "missile_airframe") == "body_length_m"
    assert tf.resolve_alias("  length, mm  ", None, "missile_airframe") == "body_length_m"


def test_resolve_returns_none_for_unknown_label():
    tf = _load()
    assert tf.resolve_alias("Mystery Field", None, "missile_airframe") is None


def test_resolve_section_conditional():
    tf = _load()
    assert tf.resolve_alias("Weight kg", "1st Stage", "missile_propulsion") == "booster_mass_kg"
    assert tf.resolve_alias("Weight kg", None, "missile_propulsion") is None
    assert tf.resolve_alias("Weight kg", None, "missile_airframe") == "total_mass_kg"


def test_resolve_pass_conditional():
    tf = _load()
    assert tf.resolve_alias("Length mm", None, "missile_airframe") == "body_length_m"
    assert tf.resolve_alias("Length mm", None, "missile_propulsion") is None
```

- [ ] **Step 2: Run tests, verify they fail.**

Run: `pytest docker/docling-graph/tests/test_table_facts_resolve.py -v 2>&1 | tail -10`
Expected: FAIL — `resolve_alias` not yet defined.

- [ ] **Step 3: Implement.**

```python
# ============================================================
# Pipeline step 5: resolve_alias (spec §5.5)
# ============================================================

def resolve_alias(
    label: str, section_ctx: SectionContext, active_pass: str
) -> str | None:
    """Look up the schema field for (label, section, pass). Returns None
    when no entry exists; the synthesizer skips that row silently."""
    from app._alias_map import ALIAS_MAP

    key: AliasKey = (normalize_label(label), section_ctx, active_pass)
    return ALIAS_MAP.get(key)
```

- [ ] **Step 4: Run tests, verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_resolve.py -v 2>&1 | tail -10`
Expected: 4/4 pass.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_resolve.py
git commit -m "feat(extraction): resolve_alias — pass+section-conditional ALIAS_MAP lookup"
```

---

### Task 12: `coerce_value`

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py`
- Create: `docker/docling-graph/tests/test_table_facts_coerce.py`

This is the largest pure function — multi-value detection + unit conversion +
stop-words + numeric vs string fields. Tests in §8.1 fully enumerate cases.

- [ ] **Step 1: Write failing tests.**

```python
"""Tests for coerce_value (spec §5.6 / D3 + D4)."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


# --- numeric, single value -------------------------------------------------

def test_numeric_single_value_explicit_unit():
    tf = _load()
    out = tf.coerce_value("1135 kg", "booster_mass_kg")
    assert len(out) == 1
    assert out[0].value == 1135.0
    assert out[0].unit_inferred == "kg"
    assert out[0].conversion_factor == 1.0


def test_numeric_single_value_implied_unit_from_field_suffix():
    """No unit in cell; field suffix _kg implies mass_kg unit class."""
    tf = _load()
    out = tf.coerce_value("1135", "booster_mass_kg")
    assert len(out) == 1
    assert out[0].value == 1135.0


def test_numeric_unit_conversion_mm_to_m_explicit():
    """Cell has explicit unit '10726 mm' -> 10.726 m via mm conversion."""
    tf = _load()
    out = tf.coerce_value("10726 mm", "body_length_m")
    assert len(out) == 1
    assert abs(out[0].value - 10.726) < 1e-6
    assert out[0].conversion_factor == 0.001


def test_numeric_unit_conversion_mm_to_m_via_row_label():
    """Cell '10726' (no unit) + row_label 'Length mm' -> mm implied -> 10.726 m.
    This is the SA-2 PDF case: spec values are stored as bare numbers in the
    cell with the unit declared in the row label."""
    tf = _load()
    out = tf.coerce_value("10726", "body_length_m", row_label="Length mm")
    assert len(out) == 1
    assert abs(out[0].value - 10.726) < 1e-6


def test_numeric_unit_from_row_label_kg():
    """Cell '1135' + row_label '1st Stage Weight kg' -> kg implied."""
    tf = _load()
    out = tf.coerce_value("1135", "booster_mass_kg", row_label="1st Stage Weight kg")
    assert len(out) == 1
    assert out[0].value == 1135.0
    assert out[0].unit_inferred == "kg"


def test_numeric_no_unit_anywhere_falls_back_to_canonical():
    """Cell '1135' + no row_label -> assumed canonical unit (kg) since the
    field is *_kg. The coerce succeeds because there's nothing to convert."""
    tf = _load()
    out = tf.coerce_value("1135", "booster_mass_kg")
    assert len(out) == 1
    assert out[0].value == 1135.0


def test_multi_value_alternatives_slash():
    """'X/Y' -> two facts, multi_value emitted."""
    tf = _load()
    out = tf.coerce_value("1135/1028", "booster_mass_kg")
    assert len(out) == 2
    values = sorted(p.value for p in out)
    assert values == [1028.0, 1135.0]


def test_multi_value_range_endash_collapses_to_midpoint():
    """'4–6 sec' (en-dash range) -> ONE fact at midpoint 5.0."""
    tf = _load()
    out = tf.coerce_value("4–6 sec", "booster_time_sec")
    assert len(out) == 1
    assert out[0].value == 5.0


def test_multi_value_range_to_word_collapses_to_midpoint():
    tf = _load()
    out = tf.coerce_value("4 to 6 sec", "booster_time_sec")
    assert len(out) == 1
    assert out[0].value == 5.0


def test_ambiguous_hyphen_range_via_xLessThanY():
    """'29-34' with X<Y -> range, midpoint."""
    tf = _load()
    out = tf.coerce_value("29-34", "max_intercept_km")
    assert len(out) == 1
    assert out[0].value == 31.5


def test_ambiguous_hyphen_alternatives_via_xGreaterThanY():
    """'1135-1028' with X>Y -> alternatives, two facts."""
    tf = _load()
    out = tf.coerce_value("1135-1028", "booster_mass_kg")
    assert len(out) == 2


def test_string_field_passthrough():
    """String-typed schema fields pass cell text through verbatim."""
    tf = _load()
    out = tf.coerce_value("dual-pulse Mark 104 sustainer", "sustain_thrust")
    assert len(out) == 1
    assert out[0].value == "dual-pulse Mark 104 sustainer"
    assert out[0].unit_inferred is None


def test_stop_words_return_empty_list():
    tf = _load()
    for stop in ["", "TBD", "—", "N/A", "unknown", "?", "—", "-", "--"]:
        assert tf.coerce_value(stop, "booster_mass_kg") == [], f"{stop!r} not stopped"


def test_unparseable_returns_empty():
    tf = _load()
    assert tf.coerce_value("not a number", "booster_mass_kg") == []


def test_unknown_unit_returns_empty():
    """'1135 furlongs' has no entry in mass_kg unit table -> skip."""
    tf = _load()
    assert tf.coerce_value("1135 furlongs", "booster_mass_kg") == []
```

- [ ] **Step 2: Run tests, verify they fail.**

Run: `pytest docker/docling-graph/tests/test_table_facts_coerce.py -v 2>&1 | tail -10`
Expected: FAIL — `coerce_value` not defined.

- [ ] **Step 3: Implement coerce_value with multi-value parsing + unit conversion.**

```python
# ============================================================
# Pipeline step 6: coerce_value (spec §5.6 / D3 + D4)
# ============================================================

# Stop-words returning [] from coerce_value, normalized lowercased.
_STOP_WORDS = frozenset({
    "", "tbd", "n/a", "na", "unknown", "unk", "none",
    "?", "??", "???",
    "-", "--",
    "–", "—", "―",  # en/em/horizontal-bar dashes
})

# Range separators (after dash-class normalization to '-'): treat '-' here
# as ASCII hyphen-minus. The range vs alternatives disambiguation for ASCII
# hyphen falls back to numeric ordering.
_RANGE_SEPARATORS_TEXT = (" to ", " - ", "-")  # handled in _split_values
_ALTERNATIVE_SEPARATOR = "/"


def coerce_value(
    cell_text: str,
    schema_field: str,
    *,
    row_label: str = "",
) -> list[ParsedValue]:
    """Parse cell into 0+ ParsedValues. See spec §5.6 for the policy.

    `row_label` is the row's label text (e.g., "Length mm"), used as a
    fallback unit hint when the cell value itself has no explicit unit.
    SA-2-style tables store bare numbers in cells with units declared in
    the row label — without this hint, '10726' for body_length_m would
    coerce to 10726 metres instead of 10.726.
    """
    raw = cell_text
    # Stop-word check on stripped/lowered raw text (with dash collapse).
    stop_check = _DASH_CLASS.sub("-", raw.strip()).lower()
    if stop_check in _STOP_WORDS:
        return []

    # Determine if this is a string-typed field (no numeric suffix match).
    from app._alias_map import FIELD_SUFFIX_TO_UNIT_CLASS, UNIT_TABLE
    unit_class = _field_unit_class(schema_field, FIELD_SUFFIX_TO_UNIT_CLASS)
    if unit_class is None:
        # String passthrough.
        return [ParsedValue(value=raw.strip(), unit_inferred=None,
                            conversion_factor=1.0, raw_text=raw)]

    # Numeric path. Normalize dash characters first.
    normalized = _DASH_CLASS.sub("-", raw)
    # Split into value-fragments.
    fragments, is_range = _split_values(normalized)
    if not fragments:
        return []

    # Parse each fragment (number + unit).
    parsed: list[tuple[float, str | None]] = []
    for frag in fragments:
        num_unit = _parse_number_and_unit(frag)
        if num_unit is None:
            continue
        parsed.append(num_unit)
    if not parsed:
        return []

    # Determine the implied unit fallback chain:
    # 1. Explicit unit in the cell fragment (preferred).
    # 2. Implied unit from the row label (e.g., "Length mm" -> "mm").
    # 3. Canonical unit for the field's unit class (factor 1.0).
    label_implied_unit = _extract_unit_from_label(row_label, unit_class, UNIT_TABLE)

    # Apply unit conversion using the schema field's unit class.
    out: list[ParsedValue] = []
    for value, unit_str in parsed:
        if unit_str is None:
            # No explicit unit in cell fragment — try row label, then canonical.
            unit_str = label_implied_unit or _canonical_unit_for_class(unit_class, UNIT_TABLE)
        unit_norm = unit_str.lower()
        unit_table = UNIT_TABLE.get(unit_class) or {}
        factor = unit_table.get(unit_norm)
        if factor is None:
            # Unknown unit -> skip this value.
            continue
        out.append(ParsedValue(
            value=value * factor,
            unit_inferred=unit_norm,
            conversion_factor=factor,
            raw_text=raw,
        ))

    # Range collapse: if input was detected as a range and we ended up with
    # exactly two parsed values, return the midpoint as a single ParsedValue.
    if is_range and len(out) == 2:
        midpoint = (out[0].value + out[1].value) / 2
        return [ParsedValue(
            value=midpoint,
            unit_inferred=out[0].unit_inferred,
            conversion_factor=out[0].conversion_factor,
            raw_text=raw,
        )]

    return out


def _field_unit_class(schema_field: str, suffix_map: dict[str, str]) -> str | None:
    """Find the unit class by checking each registered suffix (longest first)."""
    for suffix in sorted(suffix_map, key=len, reverse=True):
        if schema_field.endswith(suffix):
            return suffix_map[suffix]
    return None


def _canonical_unit_for_class(unit_class: str, unit_table: dict) -> str:
    """Find a unit with factor 1.0 in the class — that's the canonical."""
    table = unit_table.get(unit_class) or {}
    for unit, factor in table.items():
        if factor == 1.0:
            return unit
    return ""


def _extract_unit_from_label(
    row_label: str, unit_class: str, unit_table: dict
) -> str | None:
    """Scan the row label for any token matching a unit in the field's unit
    class. Used when the cell has no explicit unit but the label does
    (e.g., "Length mm", "Weight kg").

    Returns the matched unit (lowercased) or None. Longest-token-first match
    so 'kg' beats 'g' on labels like 'Weight kg'.

    Uses word-boundary-style matching for ALL unit lengths to prevent false
    matches inside larger words. Real failure modes this guards against:
    - 'min' (factor 60 for time_sec) embedded inside 'minimum'
    - 'rad' (factor 57.29 for angle_deg) embedded inside 'radar'
    - 'ton' (factor 1000 for mass_kg) embedded inside 'stone'
    Word-boundary chars: start/end of label, whitespace, or punctuation
    [,.;:/\\-]. Using a uniform check is cheaper to reason about than a
    length-dependent rule and has the same correctness on the canonical
    'Length mm' case.
    """
    if not row_label:
        return None
    table = unit_table.get(unit_class) or {}
    label_lower = row_label.lower()
    # Sort by descending length so 'kg' wins over 'g' on 'Weight kg'.
    for unit in sorted(table.keys(), key=len, reverse=True):
        pattern = re.compile(
            rf"(?:^|[\s,.;:/\\\-])({re.escape(unit)})(?:$|[\s,.;:/\\\-])"
        )
        if pattern.search(label_lower):
            return unit
    return None


def _split_values(cell_text: str) -> tuple[list[str], bool]:
    """Split a cell into value fragments and report whether it was a range.

    Returns (fragments, is_range). is_range=True means the orchestrator
    should collapse the parsed values to their midpoint.
    """
    text = cell_text.strip()
    # Discrete alternatives via slash.
    if _ALTERNATIVE_SEPARATOR in text:
        parts = [p.strip() for p in text.split(_ALTERNATIVE_SEPARATOR) if p.strip()]
        if len(parts) >= 2:
            return parts, False

    # Range via " to " word separator.
    if " to " in text.lower():
        parts = re.split(r"\s+to\s+", text, flags=re.IGNORECASE)
        if len(parts) == 2:
            return [p.strip() for p in parts], True

    # Range or alternatives via ASCII hyphen-minus (ambiguous).
    # Avoid splitting negative numbers ("-5") or things like "X-band".
    hyphen_split = re.split(r"\s*-\s*", text)
    if len(hyphen_split) == 2:
        # Try to parse both halves as numbers; if both succeed, decide
        # range-vs-alternatives by ordering.
        a = _parse_number_and_unit(hyphen_split[0])
        b = _parse_number_and_unit(hyphen_split[1])
        if a is not None and b is not None:
            is_range = a[0] < b[0]  # X<Y -> range
            return [hyphen_split[0].strip(), hyphen_split[1].strip()], is_range

    # Single value.
    return [text], False


# Number + optional unit parser. Accepts "1135", "1135 kg", "1.5e3", "1,135".
_NUMBER_UNIT_PATTERN = re.compile(
    r"^\s*([+-]?\d+(?:[,\.]\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-z°/]+)?\s*$"
)


def _parse_number_and_unit(text: str) -> tuple[float, str | None] | None:
    if not text:
        return None
    # Replace thousands separators (commas immediately before 3 digits).
    cleaned = re.sub(r",(\d{3})", r"\1", text)
    m = _NUMBER_UNIT_PATTERN.match(cleaned)
    if not m:
        return None
    num_str, unit_str = m.group(1), m.group(2)
    try:
        value = float(num_str)
    except ValueError:
        return None
    return value, unit_str
```

- [ ] **Step 4: Run tests, verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_coerce.py -v 2>&1 | tail -20`
Expected: 15/15 pass (covers stop-words, single-value with explicit/implied/row-label/canonical units, multi-value alternatives, range collapse via en-dash and "to" word, ambiguous-hyphen with X<Y vs X>Y, string passthrough, unparseable, unknown unit).

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_coerce.py
git commit -m "feat(extraction): coerce_value — D3 multi-value + D4 unit conversion"
```

---

### Task 13: `emit_fact`

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py`
- Create: `docker/docling-graph/tests/test_table_facts_emit.py`

- [ ] **Step 1: Write failing tests.**

```python
"""Tests for emit_fact (spec §5.7)."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def test_emit_fact_textitem_schema_completeness():
    """Returned dict must satisfy DoclingDocument's TextItem union variant."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_mass_kg",
        value=1135.0,
        source_label="1st Stage Weight kg",
        text_idx=42,
    )
    assert item["self_ref"] == "#/texts/42"
    assert item["parent"] == {"$ref": "#/body"}
    assert item["children"] == []
    assert item["content_layer"] == "body"
    assert item["label"] == "text"
    assert item["prov"] == []
    assert item["orig"] == item["text"]


def test_emit_fact_text_format_integer_valued_float():
    """1135.0 is an integer-valued float; _format_value trims trailing .0."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_mass_kg",
        value=1135.0,
        source_label="1st Stage Weight kg",
        text_idx=42,
    )
    assert item["text"] == "1D — booster_mass_kg = 1135 [source: 1st Stage Weight kg row of variants table]"


def test_emit_fact_text_format_decimal_float():
    """10.726 has a non-zero fractional part; _format_value preserves it."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="body_length_m",
        value=10.726,
        source_label="Length mm",
        text_idx=42,
    )
    assert item["text"] == "1D — body_length_m = 10.726 [source: Length mm row of variants table]"


def test_emit_fact_string_value():
    """String values render verbatim without trailing .0."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_thrust",
        value="dual-pulse Mark 104",
        source_label="1st Stage Thrust",
        text_idx=43,
    )
    assert "1D — booster_thrust = dual-pulse Mark 104 [source: 1st Stage Thrust row of variants table]" == item["text"]


def test_emit_fact_int_value_formatted_without_decimal():
    """Integer values should not show a trailing .0 — '1135' not '1135.0'."""
    tf = _load()
    item = tf.emit_fact(
        entity_id="1D",
        schema_field="booster_mass_kg",
        value=1135,  # int
        source_label="1st Stage Weight kg",
        text_idx=44,
    )
    assert " = 1135 " in item["text"]
```

- [ ] **Step 2: Run tests, verify they fail.**

Run: `pytest docker/docling-graph/tests/test_table_facts_emit.py -v 2>&1 | tail -10`
Expected: FAIL — `emit_fact` not defined.

- [ ] **Step 3: Implement emit_fact.**

```python
# ============================================================
# Pipeline step 7: emit_fact (spec §5.7)
# ============================================================

def emit_fact(
    entity_id: str,
    schema_field: str,
    value: float | int | str,
    source_label: str,
    text_idx: int,
) -> dict:
    """Render a (entity, field, value) triple as a TextItem-shaped dict.

    Format:
        "{entity_id} — {schema_field} = {value} [source: {source_label} row of variants table]"

    The em-dash separator is the schema-keyed prefix; the bracketed source
    preserves traceability without forcing the LLM to re-derive it.

    The TextItem skeleton mirrors the b9fe407 schema-validation fix used by
    _table_pivot.py — same shape so DoclingDocument's Pydantic union
    validates correctly.
    """
    formatted_value = _format_value(value)
    text = (
        f"{entity_id} — {schema_field} = {formatted_value} "
        f"[source: {source_label} row of variants table]"
    )
    return {
        "self_ref": f"#/texts/{text_idx}",
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": "text",
        "prov": [],
        "orig": text,
        "text": text,
    }


def _format_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Trim trailing .0 for integer-valued floats; retain decimals otherwise.
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)
```

- [ ] **Step 4: Run tests, verify they pass.**

Run: `pytest docker/docling-graph/tests/test_table_facts_emit.py -v 2>&1 | tail -10`
Expected: 4/4 pass.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_emit.py
git commit -m "feat(extraction): emit_fact — TextItem-shaped (entity, field, value) renderer"
```

---

## Chunk 3: Orchestrator + main.py wire-up

Tasks 14–15 connect the six pipeline functions into the public
`synthesize_table_facts` entry point and wire it into `main.py`. After
Chunk 3, the synthesizer is live in production behavior on the next
container rebuild — but B1+B2's old import is still present (see Task 16
for the clean-up).

### Task 14: `synthesize_table_facts` orchestrator + integration tests

**Files:**
- Modify: `docker/docling-graph/app/_table_facts.py` (append public entry point)
- Create: `docker/docling-graph/tests/test_table_facts_integration.py`

- [ ] **Step 1: Write failing integration tests.**

Create `docker/docling-graph/tests/test_table_facts_integration.py`:

```python
"""Integration tests for synthesize_table_facts (spec §6 worked example).

Synthetic SA-2-shaped column-major table; verifies end-to-end emission for
each of the four missile passes, idempotence flag, max_synthesized cap, and
graceful error handling for malformed input.
"""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(m); return m


def _cell(text, row, col, *, row_header=False, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + 1,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": 1, "col_span": col_span,
        "row_header": row_header, "column_header": False,
    }


def _sa2_shaped_doc():
    """SA-2-style variants table: 3 missile columns × 9 spec rows with
    embedded section keywords on rows 6-8."""
    cells = [
        _cell("Missile Type",        0, 0, row_header=True),
        _cell("1D",  0, 1), _cell("13D", 0, 2), _cell("13DM", 0, 3),

        _cell("Max Range km",        1, 0, row_header=True),
        _cell("29",  1, 1), _cell("34",  1, 2), _cell("43",   1, 3),

        _cell("Max Altitude km",     2, 0, row_header=True),
        _cell("22",  2, 1), _cell("27",  2, 2), _cell("30",   2, 3),

        _cell("Length mm",           3, 0, row_header=True),
        _cell("10726", 3, 1), _cell("10841", 3, 2), _cell("10841", 3, 3),

        _cell("Total Weight kg",     4, 0, row_header=True),
        _cell("2163",  4, 1), _cell("2283",  4, 2), _cell("2283",  4, 3),

        _cell("Max Speed m/s",       5, 0, row_header=True),
        _cell("",      5, 1), _cell("650",   5, 2), _cell("650",   5, 3),

        _cell("1st Stage Weight kg", 6, 0, row_header=True),
        _cell("1135",  6, 1), _cell("1032",  6, 2), _cell("1032",  6, 3),

        _cell("1st Stage Time sec",  7, 0, row_header=True),
        _cell("4.0",   7, 1), _cell("4.0",   7, 2), _cell("4.0",   7, 3),

        _cell("2nd Stage Weight kg", 8, 0, row_header=True),
        _cell("1028",  8, 1), _cell("1251",  8, 2), _cell("1251",  8, 3),
    ]
    return {
        "tables": [
            {
                "self_ref": "#/tables/0",
                "data": {"table_cells": cells, "num_rows": 9, "num_cols": 4},
                "prov": [{"page_no": 6}],
            }
        ],
        "texts": [],
        "body": {"children": []},
    }


def test_synthesizes_propulsion_facts():
    """missile_propulsion pass on SA-2 doc emits booster + sustain mass facts."""
    tf = _load()
    doc = _sa2_shaped_doc()
    out_doc, stats = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    assert stats.facts_emitted >= 6  # 3 entities × (booster_mass + sustain_mass) = 6
    text_set = {t["text"] for t in out_doc["texts"]}
    assert any("1D — booster_mass_kg = 1135" in t for t in text_set)
    assert any("13D — booster_mass_kg = 1032" in t for t in text_set)
    assert any("13DM — booster_mass_kg = 1032" in t for t in text_set)
    assert any("1D — sustain_mass_kg = 1028" in t for t in text_set)
    # No kinematics fields leak into propulsion output.
    assert not any("max_intercept_km" in t for t in text_set)


def test_synthesizes_kinematics_facts():
    tf = _load()
    doc = _sa2_shaped_doc()
    out_doc, stats = tf.synthesize_table_facts(doc, active_pass="missile_kinematics")
    assert stats.facts_emitted >= 6  # 3 entities × (max_range + max_alt) = 6
    text_set = {t["text"] for t in out_doc["texts"]}
    assert any("1D — max_intercept_km = 29" in t for t in text_set)
    assert any("1D — max_altitude_km = 22" in t for t in text_set)
    assert not any("booster_mass_kg" in t for t in text_set)


def test_synthesizes_airframe_facts():
    tf = _load()
    doc = _sa2_shaped_doc()
    out_doc, stats = tf.synthesize_table_facts(doc, active_pass="missile_airframe")
    assert stats.facts_emitted >= 6  # 3 entities × (length + total_weight) = 6
    text_set = {t["text"] for t in out_doc["texts"]}
    # body_length_m is converted from mm: 10726 -> 10.726
    assert any("1D — body_length_m = 10.726" in t for t in text_set)
    assert any("1D — total_mass_kg = 2163" in t for t in text_set)


def test_idempotence_skips_second_call():
    tf = _load()
    doc = _sa2_shaped_doc()
    out1, stats1 = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    out2, stats2 = tf.synthesize_table_facts(out1, active_pass="missile_propulsion")
    assert stats1.idempotent_skip is False
    assert stats2.idempotent_skip is True
    assert stats2.facts_emitted == 0
    # Second call returns the doc with same texts count as first call.
    assert len(out2["texts"]) == len(out1["texts"])


def test_max_synthesized_cap():
    """max_synthesized=5 caps emission at 5 even when more would fire."""
    tf = _load()
    doc = _sa2_shaped_doc()
    out, stats = tf.synthesize_table_facts(
        doc, active_pass="missile_propulsion", max_synthesized=5,
    )
    assert stats.facts_emitted == 5
    assert stats.truncated_at_cap is True


def test_appends_to_body_children_so_chunker_walks_them():
    tf = _load()
    doc = _sa2_shaped_doc()
    out, _ = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    refs = {c.get("$ref") for c in out["body"]["children"]}
    for i in range(len(out["texts"])):
        assert f"#/texts/{i}" in refs


def test_handles_doc_with_no_tables():
    tf = _load()
    doc = {"tables": [], "texts": [], "body": {"children": []}}
    out, stats = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    assert stats.facts_emitted == 0
    assert stats.tables_seen == 0


def test_handles_malformed_doc():
    tf = _load()
    out, stats = tf.synthesize_table_facts({}, active_pass="missile_propulsion")
    assert stats.facts_emitted == 0


def test_unknown_pass_skips_with_warning():
    """active_pass='nonexistent_pass' emits no facts; ALIAS_MAP won't match anything."""
    tf = _load()
    doc = _sa2_shaped_doc()
    out, stats = tf.synthesize_table_facts(doc, active_pass="nonexistent_pass")
    assert stats.facts_emitted == 0
    # Tables were still inspected.
    assert stats.tables_seen >= 1


def test_stats_counters_increment_correctly():
    tf = _load()
    doc = _sa2_shaped_doc()
    _, stats = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")
    assert stats.tables_seen == 1
    assert stats.tables_by_shape == {"column_major": 1}
    assert stats.sections_detected >= 2  # "1st Stage" + "2nd Stage"
    # Skipped rows (kinematics labels in propulsion pass): Max Range km,
    # Max Alt km, Length mm, Total Weight kg, Max Speed m/s = 5 rows × 3 cols
    assert stats.rows_skipped_unresolvable >= 5 * 3
```

- [ ] **Step 2: Run tests, verify they fail.**

- [ ] **Step 3: Implement synthesize_table_facts orchestrator.**

Append to `_table_facts.py`:

```python
import logging

logger = logging.getLogger(__name__)

_IDEMPOTENCE_FLAG = "__synthesized_table_facts__"


def synthesize_table_facts(
    doc_json: dict,
    *,
    active_pass: str,
    max_synthesized: int = 256,
) -> tuple[dict, FactStats]:
    """Append section-aware per-cell table-fact TextItems to doc_json.

    Spec §4.2 / §6 entry point. Pass-aware — same DoclingDocument fed to
    four different passes produces four different fact sets, each scoped
    to that pass's schema fields via ALIAS_MAP[(label, section, pass)].

    Returns (mutated_doc_json, FactStats). Mutates doc_json in place but
    also returns it for chaining. Sets the idempotence flag on first run;
    second call short-circuits with idempotent_skip=True.
    """
    stats = FactStats.empty()

    if not isinstance(doc_json, dict):
        return doc_json, stats

    if doc_json.get(_IDEMPOTENCE_FLAG) is True:
        stats.idempotent_skip = True
        return doc_json, stats

    tables = doc_json.get("tables") or []
    if not tables:
        doc_json[_IDEMPOTENCE_FLAG] = True
        return doc_json, stats

    texts = doc_json.setdefault("texts", [])
    body = doc_json.setdefault("body", {})
    body_children = body.setdefault("children", [])

    sections_seen: set[str] = set()

    for table in tables:
        stats.tables_seen += 1
        shape = detect_table_shape(table)
        stats.tables_by_shape[shape.value] = (
            stats.tables_by_shape.get(shape.value, 0) + 1
        )

        if shape == Shape.OTHER:
            continue

        rows = extract_label_rows(table, shape)
        if not rows:
            continue

        entity_ids = derive_entity_ids(rows, shape)
        if not entity_ids:
            # No identifiable entities — skip the whole table.
            continue

        # Detect collisions: derive_entity_ids deduplicates composites
        # (last-write-wins), so any difference between source-column count
        # and returned-id count indicates collisions. Applies to all shapes
        # (HYBRID is most common but column-major can collide too if two
        # columns happen to share the same identity row value).
        key_rows = [r for r in rows if _looks_like_key_label(r["label_text"])]
        source_cols: set[int] = set()
        for kr in key_rows:
            source_cols.update(
                col for col, cell in kr["data_cells"].items() if cell.strip()
            )
        stats.hybrid_collisions += max(0, len(source_cols) - len(entity_ids))

        sectioned = detect_section_context(rows)

        for row, section_ctx in sectioned:
            # Skip the rows that are themselves identity rows; they don't
            # produce facts (they produce entity_ids).
            if _looks_like_key_label(row["label_text"]):
                continue

            if section_ctx is not None:
                sections_seen.add(section_ctx)

            for entity_col, cell_text in row["data_cells"].items():
                entity_id = entity_ids.get(entity_col)
                if entity_id is None:
                    continue

                schema_field = resolve_alias(
                    row["label_text"], section_ctx, active_pass
                )
                if schema_field is None:
                    stats.rows_skipped_unresolvable += 1
                    continue

                # Pass row_label so coerce_value can extract implied units
                # (e.g., "Length mm" -> mm conversion for body_length_m).
                # detect_section_context strips the section keyword from
                # label_text but leaves the unit token intact ("1st Stage
                # Weight kg" -> "Weight kg"), so the post-strip label still
                # carries the unit hint we need.
                parsed = coerce_value(
                    cell_text, schema_field, row_label=row["label_text"]
                )
                if not parsed:
                    stats.values_skipped_unparseable += 1
                    continue

                if len(parsed) >= 2:
                    stats.multi_value_emissions += 1

                for pv in parsed:
                    if stats.facts_emitted >= max_synthesized:
                        stats.truncated_at_cap = True
                        doc_json[_IDEMPOTENCE_FLAG] = True
                        return doc_json, stats

                    text_idx = len(texts)
                    item = emit_fact(
                        entity_id=entity_id,
                        schema_field=schema_field,
                        value=pv.value,
                        source_label=row["label_text"],
                        text_idx=text_idx,
                    )
                    texts.append(item)
                    body_children.append({"$ref": f"#/texts/{text_idx}"})
                    stats.facts_emitted += 1

    stats.sections_detected = len(sections_seen)
    doc_json[_IDEMPOTENCE_FLAG] = True
    return doc_json, stats
```

- [ ] **Step 4: Run tests, expect 10/10 pass.**

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/_table_facts.py docker/docling-graph/tests/test_table_facts_integration.py
git commit -m "$(cat <<'EOF'
feat(extraction): synthesize_table_facts orchestrator + integration tests

Wires the six pure functions into the public synthesize_table_facts entry
point per spec §4.2. Pass-aware (same doc → 4 different fact sets across
passes), idempotent (top-level flag short-circuits second call), capped
at max_synthesized to bound runaway emission. Stats mutated through the
loop; sections_seen deduplicated.

Acceptance: integration tests cover synthetic SA-2-shaped fixture for all
four missile passes, idempotence, cap, malformed input, and pass isolation.

Refs: docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md §4.2 §6
EOF
)"
```

---

### Task 15: Wire `synthesize_table_facts` into `main.py` + prompt-content test

**Files:**
- Modify: `docker/docling-graph/app/main.py` (replace B1+B2 hook)
- Create: `docker/docling-graph/tests/test_table_facts_prompt_content.py`

- [ ] **Step 1: Write failing prompt-content test.**

Create `docker/docling-graph/tests/test_table_facts_prompt_content.py`:

```python
"""Prompt-content test (spec §8.4) — synthesizer's facts land in the LLM
user message in the exact emit_fact format. CI proxy for the full §20
end-to-end run."""
from unittest.mock import patch
import importlib.util
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def test_prompt_contains_synthesized_facts_in_emit_format():
    """Run /extract-pass-equivalent against the SA-2 synthetic fixture;
    assert specific emit_fact lines appear in the captured user message."""
    # Use the synthesizer directly + chunker simulation. The full
    # /extract-pass dispatch is covered by §8.5 operator-driven runs.

    facts_path = _REPO_ROOT / "docker/docling-graph/app/_table_facts.py"
    spec = importlib.util.spec_from_file_location("dg_tf", facts_path)
    tf = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tf)

    # Reuse the same fixture as integration tests.
    integration_path = _REPO_ROOT / "docker/docling-graph/tests/test_table_facts_integration.py"
    spec2 = importlib.util.spec_from_file_location("dg_int", integration_path)
    int_mod = importlib.util.module_from_spec(spec2)
    assert spec2.loader is not None
    spec2.loader.exec_module(int_mod)

    doc = int_mod._sa2_shaped_doc()
    out, _ = tf.synthesize_table_facts(doc, active_pass="missile_propulsion")

    # Concatenate all synthesized text — what the LLM ultimately sees in
    # whatever chunk the chunker assigns these to.
    rendered = "\n".join(t["text"] for t in out["texts"])

    # The exact-string assertions catch both presence AND format drift.
    assert (
        "1D — booster_mass_kg = 1135 [source: 1st Stage Weight kg row of variants table]"
        in rendered
    )
    assert (
        "13DM — sustain_mass_kg = 1251 [source: 2nd Stage Weight kg row of variants table]"
        in rendered
    )
    assert (
        "13DM — booster_mass_kg = 1032 [source: 1st Stage Weight kg row of variants table]"
        in rendered
    )
```

- [ ] **Step 2: Run, verify it passes (it doesn't depend on main.py changes — it's testing the synthesizer's output format).**

Run: `pytest docker/docling-graph/tests/test_table_facts_prompt_content.py -v 2>&1 | tail -10`
Expected: 1/1 pass.

- [ ] **Step 3: Modify main.py — replace B1+B2 hook with synthesize_table_facts.**

Edit `docker/docling-graph/app/main.py`:

Replace the import at line ~120:
```python
from app._table_pivot import synthesize_pivoted_table_texts
```
with:
```python
from app._table_facts import synthesize_table_facts, FactStats
```

First locate the actual line range of the B1+B2 block (line numbers
drift across PRs):

```bash
grep -n "synthesize_pivoted_table_texts\|GRAPH_EXTRACTION_PIVOTED" docker/docling-graph/app/main.py
```

Expected: 2 hits (one call site, one log statement). Note the line numbers
of the call statement and the closing `)` of its `if _pivoted_count > 0:`
block — those bound the replacement range.

Replace the B1+B2 block at the located lines (the block currently
re-enabled for the diagnostic test in this session):
```python
    docling_document_json, _pivoted_count = synthesize_pivoted_table_texts(
        docling_document_json
    )
    if _pivoted_count > 0:
        logger.info(
            "GRAPH_EXTRACTION_PIVOTED pass=%s synthesized=%d "
            "(per-column row-major summaries appended for column-major tables)",
            pass_name, _pivoted_count,
        )
```
with:
```python
    fact_stats: FactStats
    try:
        docling_document_json, fact_stats = synthesize_table_facts(
            docling_document_json,
            active_pass=pass_name,
        )
        if fact_stats.facts_emitted > 0:
            logger.info(
                "GRAPH_EXTRACTION_FACTS pass=%s facts=%d tables=%d sections=%d "
                "skipped_unresolvable=%d unparseable=%d shapes=%s "
                "hybrid_collisions=%d truncated=%s",
                pass_name, fact_stats.facts_emitted, fact_stats.tables_seen,
                fact_stats.sections_detected, fact_stats.rows_skipped_unresolvable,
                fact_stats.values_skipped_unparseable, fact_stats.tables_by_shape,
                fact_stats.hybrid_collisions, fact_stats.truncated_at_cap,
            )
    except Exception as exc:
        logger.warning(
            "synthesize_table_facts failed pass=%s: %s — continuing with original doc",
            pass_name, exc,
        )
        fact_stats = FactStats.empty()
```

Locate the diagnostics block — first verify the actual line number with grep:

```bash
grep -n 'diagnostics\["service_identity_gate"\]' docker/docling-graph/app/main.py
```

Expected: one hit (current state). Append the new line directly after the
`service_identity_gate` assignment:

```python
        diagnostics["service_table_facts"] = fact_stats.as_dict()
```

The fact_stats variable is in scope from the synthesizer call above. Verify
after editing: `grep -n 'service_table_facts' docker/docling-graph/app/main.py`
should produce one hit.

- [ ] **Step 4: Rebuild container, then run all tests on host.**

```bash
docker compose build docling-graph && docker compose up -d docling-graph
sleep 10
pytest docker/docling-graph/tests -v 2>&1 | tail -20
```

Expected: All tests green. The `_table_pivot.py` import is removed from
main.py; existing test_table_pivot.py still imports `_table_pivot.py`
directly via `importlib.util` (loadable standalone), so that test keeps
passing as a regression for the deprecated path.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/tests/test_table_facts_prompt_content.py
git commit -m "$(cat <<'EOF'
feat(extraction): wire synthesize_table_facts into main.py extract-pass

Replaces _table_pivot import + B1+B2 hook with the section-aware per-cell
synthesizer. Catches synthesizer exceptions and continues with original doc
(non-critical guard). Surfaces FactStats in diagnostics["service_table_facts"]
alongside existing service_identity_gate / service_postprocess plumbing.

Refs: docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md §9.1
EOF
)"
```

---

## Chunk 4: Deprecation + deployment + acceptance

Tasks 16–18 finalize the cutover: deprecate `_table_pivot.py`, rebuild the
container with all changes live, and run the §20 GT scorecard at T=1.0 to
verify the headline acceptance criterion.

### Task 16: Mark `_table_pivot.py` DEPRECATED

**Files:**
- Modify: `docker/docling-graph/app/_table_pivot.py` (docstring update)

- [ ] **Step 1: Update module docstring.**

Edit `docker/docling-graph/app/_table_pivot.py` — replace the existing
module docstring's first paragraph with:

```python
"""DEPRECATED — replaced by _table_facts.py operationally on 2026-05-05.

This module's synthesize_pivoted_table_texts emitted one prose summary per
column; empirical R21 measurements showed it was neutral (no field-fill
gain) on the SA-2 propulsion pass. The replacement (_table_facts.py) emits
one TextItem per (entity, schema_field, value) triple, resolving row labels
to schema fields deterministically via _alias_map.py.

This module is preserved one cycle as regression scaffolding for the
deprecated path (test_table_pivot.py still imports it). Both this file and
its test will be deleted in the next maintenance cycle once the new
synthesizer has had at least one round of green production telemetry.

Original docstring follows.
========================================================================
[original docstring here, unchanged]
"""
```

- [ ] **Step 2: Verify the test file still loads.**

Run: `pytest docker/docling-graph/tests/test_table_pivot.py -v 2>&1 | tail -5`
Expected: All test_table_pivot tests still pass — module is still importable
even after main.py drops the import. The test file uses `importlib.util` to
load it standalone, independent of the rest of the service.

- [ ] **Step 3: Commit.**

```bash
git add docker/docling-graph/app/_table_pivot.py
git commit -m "$(cat <<'EOF'
chore(extraction): mark _table_pivot.py DEPRECATED

Replaced operationally by _table_facts.py. Module preserved one cycle as
regression scaffolding for the deprecated path; both this file and its test
will be deleted in the next maintenance cycle.

Refs: docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md §4.1
EOF
)"
```

---

### Task 17: Container rebuild + log verification + smoke run

**Files:**
- (none — operational task)

- [ ] **Step 1: Rebuild docling-graph and verify it starts cleanly.**

```bash
docker compose build docling-graph 2>&1 | tail -10
docker compose up -d docling-graph
sleep 8
docker logs --tail 30 eip-mmdpp-docling-graph-1 2>&1 | grep -E "Application startup|prompt_rules|Preloaded|ERROR"
```

Expected:
- "prompt_rules: installed delta system-prompt rewrite + semantic-guide budget expansion ..." (existing log, unchanged)
- "Preloaded all bundle extraction schemas" (existing log)
- "Application startup complete." (uvicorn confirmation)
- No ERROR / Traceback lines

- [ ] **Step 2: Verify the new module is in the running image.**

```bash
docker exec eip-mmdpp-docling-graph-1 grep -n "synthesize_table_facts" /app/app/main.py
docker exec eip-mmdpp-docling-graph-1 ls /app/app/_table_facts.py /app/app/_alias_map.py
```

Expected: 2-3 hits in main.py; both files present.

- [ ] **Step 3: Smoke test — hit /health and verify schemas loaded.**

```bash
docker exec eip-mmdpp-docling-graph-1 curl -s http://localhost:8002/health
```

Expected: 200 OK response.

- [ ] **Step 4: Run all tests on host.**

```bash
pytest docker/docling-graph/tests -v 2>&1 | tail -25
```

Expected: All test files green (test_alias_map, test_alias_map_missile,
test_alias_map_radar, test_table_facts_*, test_table_pivot, test_sanitizer,
test_numeric_candidates).

If any test fails, do not proceed to Task 18. Reconcile first.

- [ ] **Step 5: Commit any final fixups (if needed).**

If steps 1-4 surfaced minor issues (e.g., import error, missing env var),
commit the fix and re-run the chunk.

---

### Task 18: §20 acceptance run

**Files:**
- Modify: `notebooks/extraction_walkthrough.ipynb` §20 cell — set `TEMP_RUNS = [1.0]` and clear cache for fresh run.
- Modify: TODO.md — mark related items resolved if applicable.

This is operator-driven; run after Task 17 completes cleanly.

- [ ] **Step 1: Back up and clear the alias-only T=1.0 baseline cache (the new run will replace it; baseline must be preserved).**

Run inside the Jupyter container:
```bash
docker exec eip-mmdpp-jupyter sh -c '
mkdir -p /tmp/r22_synthesizer_T1.0_pre_run_backup
# (Cache should already be empty from Chunk 4 sequencing; this is a
# defense-in-depth step.)
ls /tmp/field_ab_missile_*_T1.0.json 2>/dev/null && \
  mv /tmp/field_ab_missile_*_T1.0.json /tmp/r22_synthesizer_T1.0_pre_run_backup/
echo "Cache cleared"
'
```

Expected: Either "Cache cleared" with files moved, OR "No such file or directory" (already empty — equally fine).

- [ ] **Step 2: Restart Jupyter kernel and re-run §20.**

Operator action:
1. Open `notebooks/extraction_walkthrough.ipynb`.
2. Restart kernel (Kernel → Restart).
3. Run cells from top through §20.
4. Wait for §20 to complete — 4 missile passes at T=1.0 with the synthesizer
   active. Approximate wall: 75-90 minutes on a 2-host Ollama pool.

- [ ] **Step 3: Capture output and compare against acceptance criteria.**

Expected GT scorecard at T=1.0 (from spec §10):

| Pass | ✓ exact (synth) | Baseline | Acceptance |
|---|---|---|---|
| `missile_kinematics` | ≥ 4 | 4 | no regression |
| `missile_airframe` | ≥ 8 | 8 | no regression |
| `missile_speed_timing` | ≥ 6 | 6 | no regression |
| `missile_propulsion` | **≥ 6 of 7** | 0 | **must move from 0 → ≥ 6** |

The 7 listed propulsion variants: 13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23.

Also verify:
- Wall-time delta per pass ≤ +20% vs alias-only T=1.0 baseline (subject to
  Ollama pool topology — re-measure baseline if pool size changed).
- `service_table_facts.facts_emitted` per pass logged in
  `docker logs eip-mmdpp-docling-graph-1 | grep GRAPH_EXTRACTION_FACTS` —
  should be ≥30 for missile_propulsion (10 entities × 3+ propulsion fields).
- `IDENTITY_FILTER` drop counts remain 1-5 per pass (no spurious noise).

- [ ] **Step 4: If acceptance fails, decide rollback vs forward.**

If `missile_propulsion ✓ exact < 6`:
- Inspect specific failure modes in the GT scorecard. Common cases:
  - Synthesized facts missing (check `GRAPH_EXTRACTION_FACTS facts=0` log) → §6 detect_section_context likely failing on the actual SA-2 table shape; verify the docling-emitted table has section keywords as embedded labels.
  - Synthesized facts present but LLM still attributes wrong values (check that emit_fact strings appear in the prompt by manually inspecting captured `/extract-pass` request body) → format issue or chunker is splitting facts away from entity_id; consider widening the chunker max-tokens for that pass.
- If diagnosis points to an additive fix, address as a follow-up task and
  re-run §20.
- If diagnosis points to a fundamental design issue, **rollback**:
  ```bash
  git revert <main.py-wire-up commit SHA>
  docker compose build docling-graph && docker compose up -d docling-graph
  ```
  Then file a follow-up plan with the new findings.

- [ ] **Step 5: On acceptance pass — final commit + close-out.**

Update TODO.md if any related items are now resolved. Specifically, the
"Section-aware table-fact synthesis" line item (if filed) gets a DONE marker
with this date. TODO #83 (IDENTITY_FILTER relaxation) stays as filed —
unrelated to this plan.

```bash
git add TODO.md  # if changes made
git commit -m "$(cat <<'EOF'
chore(extraction): mark section-aware table-fact synthesis DONE 2026-05-05

§20 GT scorecard at T=1.0 with synthesizer active:
- missile_kinematics ✓ exact: <result> (baseline: 4)
- missile_airframe ✓ exact: <result> (baseline: 8)
- missile_speed_timing ✓ exact: <result> (baseline: 6)
- missile_propulsion ✓ exact: <result> of 7 listed variants (baseline: 0)

[Operator: fill in the actual numbers from §20 output before committing.]

Refs: docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md §10
EOF
)"
```

- [ ] **Step 6: Optional notebook tracker update (deferred, separate PR).**

Per spec §3 non-goals, the notebook outcome tracker `facts/pass` column is
a follow-up. File a follow-up task and don't block this plan's close-out
on it.

---

## Plan Summary

**Total tasks:** 18 (6 in Chunk 1 + 7 in Chunk 2 + 2 in Chunk 3 + 3 in Chunk 4)

**Total commits expected:** ~18 (one per task; some tasks combine multiple changes into a single commit).

**Critical dependencies:**
- Tasks 1, 2 must complete before any other Chunk 1 task.
- Chunk 1 must complete before Chunk 2.
- Tasks 7-13 (Chunk 2) can be parallelized once Chunk 1 is done — each is independently testable.
- Chunk 3 must complete after Chunk 2.
- Chunk 4 Tasks 16-17 must complete before Task 18.

**Acceptance criteria (spec §10):** verified in Task 18.
