# Guarded-Ranker Chunk Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-pass chunk selection with recall 1.0 by construction (label-aligned OR-gates) and maximal pruning (per-(doc,pass) quantile ranker), per the approved spec `docs/superpowers/specs/2026-06-10-chunk-selection-guarded-ranker-design.md`.

**Architecture:** Phase 0 fixes the ground-truth label (unit-matcher hardening in BOTH the app and the docling-graph mirror, then missing `SUFFIX_UNITS` suffixes, then dataset re-export). Phases 1–2 add capture-side features and gates (gate union into the retrieval pool BEFORE the cap, max_field_cosine retention, structural features, is_table wiring, keyword-channel revival) — all flag-gated or diagnostics-only, production `final_score` byte-identical by default. Phase 3 re-collects the 8-doc corpus and evaluates. Phase 4 ships the flag-gated selection plumbing and the calibration script.

**Tech Stack:** Python 3 / FastAPI app services, ArcadeDB (schemaless vertex props), Pydantic ontology bundles, sklearn offline scripts, pytest.

**Worktree:** `/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry` (branch `walltime/c0-telemetry`). All paths below are relative to it.

---

## STANDING RULES (read before any task)

1. **PHASE GATES:** Each phase below has a discussion gate. Do NOT start a phase's tasks without the user's explicit go-ahead for that phase (standing user rule).
2. **Generalization guardrail:** no equipment/instance names in any rule, keyword, regex, or config. Operate on schema, units, modality, structure only.
3. **Lineage:** raw `chunk_text`, `source_refs`, `page_number` are never modified or substituted. New haystacks are NEW fields.
4. **Byte-identical default:** every production-code change must leave default-config `final_score` and selection output unchanged. Flag/default flips that keep behavior identical (e.g. `table_boost` 0.08→0.0 while `is_table` was constantly 0.0) must be proven by test.
5. **Bundle propagation:** manifest/schema edits land in `ontology_bundles/air_defense_v3` FIRST, then `air_defense_v3_baseline_subset`, `air_defense_v3_narrowing_v1`, `air_defense_v3_merged_v1`.
6. **docling-graph mirror:** `docker/docling-graph/**` is COPY'd into its image — code changes there require a container rebuild to take effect (worker/api bind-mount `app/` and only need restarts).
7. After each task: run the task's Verify command(s); after each phase: full unit suite + `VERIFICATION_CHECKLIST.md`.

## File structure (what gets touched where)

| File | Role in this plan |
|---|---|
| `app/services/field_value_grounding.py` | unit-token matcher hardening (T1); SUFFIX_UNITS additions (T3); single-source matcher consumed by label AND gate |
| `docker/docling-graph/app/provenance.py` | `_vg_value_in_chunk` mirror hardening (T2) |
| `docker/docling-graph/tests/fixtures/unit_matcher_cases.json` | ONE fixture file pinning both matcher copies (T1/T2) |
| `scripts/audit_groundable_fields.py` + `tests/unit/test_groundable_fields_audit.py` | suffix-gap audit (T3) |
| `app/services/extraction_unit_gate.py` (new) | per-pass unit signatures + G1/G2 gate predicates (T5, T10) |
| `app/services/extraction_chunk_search.py` | gate union before cap (T6); max_field_cosine retention (T7); table_meta wiring (T9) |
| `app/services/extraction_candidate_scoring.py` | new COMPONENT_KEYS features (T7/T8); structural features (T8); cut helper (T18) |
| `app/services/hybrid_chunking.py` + `app/services/extraction_chunk_index.py` + `app/services/arcadedb_schema.py` | is_table persistence (T9); match_text header projection (T14) |
| `app/services/extraction_lexical_search.py` | word-boundary matching + weighted keywords (T12/T13); match_text haystack (T14) |
| `app/api/v1/extraction_routing.py` | keyword union (T11); content_type from row (T9); cut-helper call sites (T18) |
| `app/services/ontology_bundles.py` | RetrievalProfile fields: gate flag, selection_mode, q/k_min/k_max, ranker weights (T5/T18); table_boost default flip (T9) |
| `app/services/reranker.py` | match_text as content_text (T14) |
| `scripts/mine_pass_keywords.py` | re-mine against fixed label (T15) |
| `scripts/export_bakeoff_dataset.py` + `scripts/a0_captured_separation.py` | new feature columns, eval updates (T16) |
| `scripts/fit_guarded_ranker.py` (new) | calibration: sign-constrained LogReg + quantile q + margin (T19) |
| `ontology_bundles/air_defense_v3*/manifest.yaml` | gate/selection config + keyword list updates (T20) |

## Phase map

- **Phase 0 (Tasks 1–4):** ground truth — matcher hardening, mirror, suffixes+audit, re-export/re-baseline. **GATE: user go-ahead obtained for Phase 0 via spec approval; confirm before T4's DB-touching re-export.**
- **Phase 1 (Tasks 5–8):** gates + capture features. **GATE: discuss Phase 0 re-baseline results first.**
- **Phase 2 (Tasks 9–15):** is_table, G2, keyword revival, header projection, re-mine. **GATE.**
- **Phase 3 (Tasks 16–17):** harness updates, deploy + re-collect 8 docs + evaluate. **GATE; T17 is a USER GATE.**
- **Phase 4 (Tasks 18–20):** selection plumbing, calibration script, bundle/docs propagation. **GATE: requires T17 results.**
- Phase 5 (21-doc collection) is OUT of this plan.

---

# Phase 0 — Ground truth

### Task 1: Harden the unit-token matcher in `field_value_grounding`

**Goal:** Unit synonyms match as bounded tokens (no `"50 sites"` ⇒ unit `s`; no `"9 months"` ⇒ unit `m`), via ONE reusable helper that the label and the future G1 gate both import.

**Files:**
- Modify: `app/services/field_value_grounding.py:98-108` (`value_in_chunk`) + add two helpers above it
- Create: `docker/docling-graph/tests/fixtures/unit_matcher_cases.json` (shared fixture, lives under docling-graph so the container test env can read it; app tests read it by repo-relative path)
- Test: `tests/unit/test_field_value_grounding.py` (extend)

**Acceptance Criteria:**
- [ ] `unit_token_regex` / `has_unit_token` exported from `field_value_grounding`
- [ ] ADJACENT tier rejects `"50 sites"` for unit `s`; still accepts `"50 s"`, `"50s"`, `"50 km."`, `"50 км"`, `"45°"`
- [ ] SAME_CHUNK tier uses `has_unit_token` (no plain substring); `"background radiation measured 2391 times"` no longer counts unit `kg`
- [ ] ≥2-digit SAME_CHUNK rule unchanged; all existing tests still pass
- [ ] Every case in `unit_matcher_cases.json` passes

**Verify:** `python3 -m pytest tests/unit/test_field_value_grounding.py -v` → all PASS (new cases included)

**Steps:**

- [ ] **Step 1: Write the shared fixture file** `docker/docling-graph/tests/fixtures/unit_matcher_cases.json`:

```json
{
  "comment": "Single source of truth for unit-token matcher behavior. tests/unit/test_field_value_grounding.py (app) and docker/docling-graph/tests/test_value_grounding_mirror.py (mirror) BOTH run every case. Do not fork.",
  "unit_token_cases": [
    {"text": "the 50 sites were observed", "units": ["s"], "expect": false},
    {"text": "elapsed 50 s before launch", "units": ["s"], "expect": true},
    {"text": "background radiation measured 2391 times", "units": ["kg", "кг"], "expect": false},
    {"text": "weight , 7 = 2391 , = kg", "units": ["kg", "кг"], "expect": true},
    {"text": "over 9 months of trials", "units": ["m", "metre", "meter"], "expect": false},
    {"text": "max speed in m/s shown in table", "units": ["m/s"], "expect": true},
    {"text": "дальность 50 км", "units": ["km", "км"], "expect": true},
    {"text": "elevation 45° max", "units": ["deg", "°"], "expect": true},
    {"text": "the masts were 9 metres tall", "units": ["m", "metre", "meter"], "expect": true},
    {"text": "pulse width 2.8 µs nominal", "units": ["µs", "us", "microsec"], "expect": true},
    {"text": "tell us about the radar", "units": ["µs", "us", "microsec"], "expect": false}
  ],
  "value_in_chunk_cases": [
    {"text": "the 50 sites were observed", "num_strs": ["50"], "units": ["s"], "expect": null},
    {"text": "elapsed 50 s before launch", "num_strs": ["50"], "units": ["s"], "expect": "adjacent"},
    {"text": "burn time 50s total", "num_strs": ["50"], "units": ["s", "sec", "second", "seconds"], "expect": "adjacent"},
    {"text": "range 50 km. then home", "num_strs": ["50"], "units": ["km", "км"], "expect": "adjacent"},
    {"text": "antenna is 9 m wide", "num_strs": ["9"], "units": ["m", "metre", "meter"], "expect": "adjacent"},
    {"text": "over 9 months of trials", "num_strs": ["9"], "units": ["m", "metre", "meter"], "expect": null},
    {"text": "weight , 7 = 2391 ... = kg", "num_strs": ["2391"], "units": ["kg", "кг"], "expect": "same_chunk"},
    {"text": "weight , = 7 ... = kg", "num_strs": ["7"], "units": ["kg", "кг"], "expect": null},
    {"text": "elevation 45° max", "num_strs": ["45"], "units": ["deg", "°"], "expect": "adjacent"},
    {"text": "peak power [kw] 180.0 listed", "num_strs": ["180.0", "180"], "units": ["kw"], "expect": "same_chunk"}
  ]
}
```

Note: every `text` is written pre-casefolded (the matcher receives `nfc()`-folded text; `nfc()` casefolds — `field_value_grounding.py:47-50`).

- [ ] **Step 2: Write the failing tests** — append to `tests/unit/test_field_value_grounding.py`:

```python
import json
from pathlib import Path

import pytest

from app.services.field_value_grounding import (
    ADJACENT,
    SAME_CHUNK,
    has_unit_token,
    nfc,
    value_in_chunk,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "docker" / "docling-graph" / "tests" / "fixtures" / "unit_matcher_cases.json"
)
_CASES = json.loads(_FIXTURE.read_text())


@pytest.mark.parametrize("case", _CASES["unit_token_cases"])
def test_unit_token_cases(case):
    assert has_unit_token(nfc(case["text"]), case["units"]) is case["expect"]


@pytest.mark.parametrize("case", _CASES["value_in_chunk_cases"])
def test_value_in_chunk_cases(case):
    got = value_in_chunk(case["num_strs"], case["units"], nfc(case["text"]))
    assert got == case["expect"]
```

- [ ] **Step 3: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_field_value_grounding.py -v -k "unit_token or value_in_chunk_cases"`
Expected: FAIL — `ImportError: cannot import name 'has_unit_token'`

- [ ] **Step 4: Implement** in `app/services/field_value_grounding.py`, above `value_in_chunk`:

```python
def unit_token_regex(unit_nfc: str) -> str:
    """Token-bounded regex fragment for one already-``nfc()``-folded unit synonym.

    Leading guard (only when the synonym starts with a word char): no LETTER
    immediately before — digit prefixes stay legal so "50km" still carries the
    unit. Trailing guard (only when it ends with a word char): no word char
    immediately after — "50 sites" no longer matches unit "s", "9 months" no
    longer matches unit "m". Unicode-aware so Cyrillic synonyms get the same
    discipline. Symbol-edged synonyms ("°", "m/s", "m=") skip the guard on
    their symbol side (the symbol is its own boundary).
    """
    pat = re.escape(unit_nfc)
    if re.match(r"\w", unit_nfc):
        pat = r"(?<![^\W\d_])" + pat  # not preceded by a unicode letter
    if re.search(r"\w$", unit_nfc):
        pat = pat + r"(?!\w)"  # not followed by a unicode word char
    return pat


def has_unit_token(text_nfc: str, units: Iterable[str]) -> bool:
    """True iff any unit synonym appears as a bounded token in the folded text.

    SHARED by the SAME_CHUNK grounding tier and the G1 selection gate — one
    matcher, so label semantics and gate semantics can never drift.
    """
    return any(re.search(unit_token_regex(nfc(u)), text_nfc) for u in units)
```

Then inside `value_in_chunk` replace the two matching sites:

```python
    units_nfc = [nfc(u) for u in units]
    for ns in num_strs:
        for u in units_nfc:
            if re.search(
                r"(?<![\d.])" + re.escape(ns) + r"\s*[\(\-–]?\s*" + unit_token_regex(u),
                text_nfc,
            ):
                return ADJACENT
    if has_unit_token(text_nfc, units_nfc):
        for ns in num_strs:
            # require >=2 digits for the table tier to avoid coincidental single-digit hits
            if len(re.sub(r"\D", "", ns)) >= 2 and re.search(
                r"(?<!\d)" + re.escape(ns) + r"(?!\d)(?!\.\d)", text_nfc
            ):
                return SAME_CHUNK
    return None
```

- [ ] **Step 5: Run the full module test**

Run: `python3 -m pytest tests/unit/test_field_value_grounding.py -v`
Expected: ALL PASS (pre-existing cases too — if a pre-existing case fails, the case encoded the old substring bug; inspect and update it ONLY with a comment citing this task)

- [ ] **Step 6: Commit**

```bash
git add app/services/field_value_grounding.py tests/unit/test_field_value_grounding.py docker/docling-graph/tests/fixtures/unit_matcher_cases.json
git commit -m "fix(grounding): token-bounded unit matching in value_in_chunk + shared has_unit_token helper"
```

### Task 2: Mirror the matcher hardening in docling-graph `_vg_value_in_chunk`

**Goal:** The docling-graph provenance mirror (`_vg_value_in_chunk`) gets byte-equivalent matching semantics, pinned to the SAME fixture file, so committed lineage and the app label can't drift.

**Files:**
- Modify: `docker/docling-graph/app/provenance.py:1050-1067` (`_vg_value_in_chunk`; add `_vg_unit_token_regex`, `_vg_has_unit_token` next to `_vg_nfc`)
- Create: `docker/docling-graph/tests/test_value_grounding_mirror.py`

**Acceptance Criteria:**
- [ ] `_vg_value_in_chunk` returns True/False consistent with app `value_in_chunk` truthiness on EVERY fixture case (mirror returns bool, app returns tier-or-None — compare truthiness)
- [ ] Helper code is a verbatim copy modulo the `_vg_` prefix (comment in both files cross-references the other)
- [ ] Mirror test loads `tests/fixtures/unit_matcher_cases.json` relative to its own directory

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_value_grounding_mirror.py -v` → all PASS (run host-side; the file has no docling_graph package imports — it imports `provenance` via its path. If host import fails due to module deps, run in-container per Step 4.)

**Steps:**

- [ ] **Step 1: Write the failing mirror test** `docker/docling-graph/tests/test_value_grounding_mirror.py`:

```python
"""Pins docker/docling-graph/app/provenance.py value-grounding helpers to the
shared fixture (tests/fixtures/unit_matcher_cases.json) so the mirror can never
drift from app/services/field_value_grounding.py. Update BOTH copies + the
fixture together, always."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_CASES = json.loads((_HERE / "fixtures" / "unit_matcher_cases.json").read_text())


def _load_provenance():
    spec = importlib.util.spec_from_file_location(
        "dg_provenance", _HERE.parent / "app" / "provenance.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prov = _load_provenance()


@pytest.mark.parametrize("case", _CASES["unit_token_cases"])
def test_mirror_unit_token(case):
    text = prov._vg_nfc(case["text"])
    assert prov._vg_has_unit_token(text, case["units"]) is case["expect"]


@pytest.mark.parametrize("case", _CASES["value_in_chunk_cases"])
def test_mirror_value_in_chunk(case):
    text = prov._vg_nfc(case["text"])
    got = prov._vg_value_in_chunk(set(case["num_strs"]), case["units"], text)
    assert got is (case["expect"] is not None)
```

(If `provenance.py` cannot be loaded standalone because of module-level imports, fall back to `sys.path.insert(0, str(_HERE.parent))` + `from app.provenance import ...` and note which approach worked in the commit message.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest docker/docling-graph/tests/test_value_grounding_mirror.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_vg_has_unit_token'`

- [ ] **Step 3: Implement the mirror** in `docker/docling-graph/app/provenance.py`, next to `_vg_nfc` (verbatim copies of Task 1's helpers, `_vg_` prefix, docstring line: `Mirrors app/services/field_value_grounding.unit_token_regex — update together (see tests/fixtures/unit_matcher_cases.json).`):

```python
def _vg_unit_token_regex(unit_nfc: str) -> str:
    pat = re.escape(unit_nfc)
    if re.match(r"\w", unit_nfc):
        pat = r"(?<![^\W\d_])" + pat
    if re.search(r"\w$", unit_nfc):
        pat = pat + r"(?!\w)"
    return pat


def _vg_has_unit_token(text_nfc: str, units) -> bool:
    return any(re.search(_vg_unit_token_regex(_vg_nfc(u)), text_nfc) for u in units)
```

Then in `_vg_value_in_chunk` replace `re.escape(u)` in the ADJACENT regex with `_vg_unit_token_regex(u)` and replace `if any(u in text_nfc for u in units_nfc):` with `if _vg_has_unit_token(text_nfc, units_nfc):` — keeping everything else identical to the current body.

- [ ] **Step 4: Run mirror tests + the existing provenance delta tests**

Run: `PYTHONPATH="docker/docling-graph/repo:$PYTHONPATH" python3 -m pytest docker/docling-graph/tests/test_value_grounding_mirror.py docker/docling-graph/tests/test_entity_provenance_from_delta.py -v`
Expected: ALL PASS. (The full `docker/docling-graph/tests/` dir is container-only — never collect it as a directory; name files explicitly, the `run_dg_lineage` pattern in `scripts/run_tests.sh:140-158`.)

- [ ] **Step 4b: Register the mirror test in the host-safe DG list** — add `docker/docling-graph/tests/test_value_grounding_mirror.py` to the explicit file list in `scripts/run_tests.sh` `run_dg_lineage()` (around line 153, next to `test_entity_provenance_from_delta.py`).

- [ ] **Step 5: Commit** (note in the body: image rebuild required at deploy time — COPY semantics)

```bash
git add docker/docling-graph/app/provenance.py docker/docling-graph/tests/test_value_grounding_mirror.py
git commit -m "fix(docling-graph): mirror token-bounded unit matching in _vg_value_in_chunk (rebuild required at deploy)"
```

### Task 3: Add missing SUFFIX_UNITS + groundable-fields audit

**Goal:** `_sec`/`_usec`/`_dbi` fields become groundable; an audit test makes any future suffix gap a hard failure instead of a silent label hole.

**Files:**
- Modify: `app/services/field_value_grounding.py:32-40` (`SUFFIX_UNITS`)
- Create: `scripts/audit_groundable_fields.py`
- Create: `tests/unit/test_groundable_fields_audit.py`

**Acceptance Criteria:**
- [ ] `units_for("scan_period_sec") == ["s", "sec", "second", "seconds"]`, `units_for("nominal_pri_usec") == ["µs", "us", "microsec"]`, `units_for("gain_dbi") == ["dbi"]`
- [ ] Audit walks EVERY pass's schema fields in `air_defense_v3`; every numeric (int/float incl. Optional) field is either groundable (`units_for` non-empty) or on the explicit `UNITLESS_OK` allowlist — anything else fails the test
- [ ] `python3 -m scripts.audit_groundable_fields` prints a per-pass table (field, type, suffix, units or UNITLESS/GAP) usable in reports

**Verify:** `python3 -m pytest tests/unit/test_groundable_fields_audit.py -v` → PASS; `python3 -m scripts.audit_groundable_fields` → table with zero `GAP` rows

**Steps:**

- [ ] **Step 1: Write the failing test** `tests/unit/test_groundable_fields_audit.py`:

```python
from app.services.field_value_grounding import units_for
from scripts.audit_groundable_fields import audit_bundle, UNITLESS_OK


def test_new_suffixes_ground():
    assert units_for("scan_period_sec") == ["s", "sec", "second", "seconds"]
    assert units_for("nominal_pri_usec") == ["µs", "us", "microsec"]
    assert units_for("gain_dbi") == ["dbi"]
    # longest-suffix-wins must keep working
    assert units_for("muzzle_velocity_kmh") == ["km/h", "kph"]


def test_no_silent_suffix_gaps():
    rows = audit_bundle("air_defense_v3")
    gaps = [r for r in rows if r.numeric and not r.units and r.field not in UNITLESS_OK]
    assert gaps == [], f"numeric fields with no unit suffix and not allowlisted: {gaps}"
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/unit/test_groundable_fields_audit.py -v` → FAIL (no script module / suffixes missing)

- [ ] **Step 3: Add the suffixes** to `SUFFIX_UNITS` in `app/services/field_value_grounding.py` (inside the existing dict literal — keep the comment about longest-suffix-wins):

```python
    "sec": ["s", "sec", "second", "seconds"], "usec": ["µs", "us", "microsec"],
    "dbi": ["dbi"],
```

- [ ] **Step 4: Write `scripts/audit_groundable_fields.py`** — walk the production bundle's pass schemas via the router's own resolution path: `load_bundle_manifest("air_defense_v3")` (`app/services/ontology_bundles.py:422-432`) → for each pass with `module`/`template_class`, resolve the Pass class exactly like `_resolve_template_class` (`app/api/v1/extraction_routing.py:116-131`: `importlib.import_module(f"ontology_bundles.air_defense_v3.{pass_def.module}")` + `getattr`) → record class via `_record_cls_from_pass_cls` (`app/services/extraction_query_builder.py:93-112`) → enumerate `record_cls.model_fields`. Classify each field: numeric iff annotation (Optional-unwrapped) is `int`/`float`; look up `units_for(name)`. NOTE: use `model_fields`, NOT `signals.field_queries` — field_queries skips `system_name` and INTERNAL-prefixed fields, but the grounding loop iterates raw record fields. Emit `AuditRow(pass_name, field, numeric, suffix, units)` rows; `UNITLESS_OK = {"confidence", "num_bits_in_code", "pulses_per_dwell"}` (extend ONLY with a comment justifying each entry — these are unitless by design). `main()` prints the table and exits 1 if gaps exist. ~60 lines; follow the CLI shape of `scripts/keyword_discrimination_check.py`. Mark the test file with the same pytest marker convention as its neighbors in `tests/unit/` (check one — `run_tests.sh` selects `-m "unit"`).

- [ ] **Step 5: Run** — `python3 -m pytest tests/unit/test_groundable_fields_audit.py -v` → PASS; `python3 -m scripts.audit_groundable_fields` → zero GAP rows (the three previously-gapped groups now show units)

- [ ] **Step 6: Commit**

```bash
git add app/services/field_value_grounding.py scripts/audit_groundable_fields.py tests/unit/test_groundable_fields_audit.py
git commit -m "fix(grounding): add sec/usec/dbi unit suffixes + groundable-fields audit (closes radar_timing label blind spot)"
```

### Task 4: Suspect-label audit + dataset re-export + re-baseline report

**Goal:** Regenerate the 8-doc ground truth under the hardened matcher + new suffixes; audit the suspect positive; produce the Phase-0 re-baseline report all later work compares against.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Run (not modify): `scripts/export_bakeoff_dataset.py`, `scripts/a0_captured_separation.py`, `scripts/per_metric_signal.py`
- Create: `reports/dataset_v1_relabel/` (new export dir — do NOT overwrite `reports/dataset/`)
- Create: `docs/operational/phase0-relabel-report-2026-06.md`

**Acceptance Criteria:**
- [ ] Dataset re-exported for the SAME 8 run IDs (`reports/dataset/dataset_meta.json` holds them) with `--target lineage_grounded` into `reports/dataset_v1_relabel/`
- [ ] Report documents: old vs new positive counts per (doc, pass); which previously-blind passes gained positives; whether the SA2-SR71 `radar_antenna` chunk-16 positive survived the hardened matcher, and if it survived, the exact (field, value, tier) that grounds it plus a keep/principled-fix decision recorded
- [ ] Per-feature pooled + LODO AUROC re-run on the new labels (both conventions, labeled as such) in the report
- [ ] DB access confirmed via `A0_DATABASE_URL=postgresql+psycopg2://eip:eip_secret@localhost:5437/eip`; NOTHING in the pipeline DB is mutated (export + read-only scripts only)

**Verify:** `ls reports/dataset_v1_relabel/bakeoff_dataset.csv docs/operational/phase0-relabel-report-2026-06.md && python3 -c "import csv; n=sum(int(r['used']) for r in csv.DictReader(open('reports/dataset_v1_relabel/bakeoff_dataset.csv'))); print('new positives:', n)"` → both files exist; printed count matches the report's table

**Steps:**

- [ ] **Step 1:** Read run IDs from `reports/dataset/dataset_meta.json`. Re-export: `A0_DATABASE_URL=... python3 -m scripts.export_bakeoff_dataset --runs <8 ids> --target lineage_grounded --out-dir reports/dataset_v1_relabel` (check the script's actual `--out-dir` flag name first; if absent, copy outputs manually — never overwrite `reports/dataset/`).
- [ ] **Step 2:** Diff positives old vs new: pandas groupby (run_id, pass_name) on `used` for both CSVs; table into the report.
- [ ] **Step 3:** Suspect audit: locate the SA2-SR71 radar_antenna chunk-16 row in the new export. If still positive, reproduce its grounding by running `match_value_to_chunks`/`value_in_chunk` over that chunk's text with each radar_antenna field's extracted values (one-off snippet; paste into the report). Decide: legitimate (keep) or coincidental (then the principled fix is a follow-on task proposal in the report — NOT an ad-hoc row deletion).
- [ ] **Step 4:** Re-baseline: `A0_DATABASE_URL=... python3 -m scripts.a0_captured_separation --fit-runs <8 ids> --target lineage_grounded --out-dir reports/dataset_v1_relabel` + `python3 -m scripts.per_metric_signal ...` (same args pattern); copy headline numbers (per-feature AUROC pooled/LODO-both-conventions, frontier recall-1.0 savings) into the report.
- [ ] **Step 5:** Write `docs/operational/phase0-relabel-report-2026-06.md` (old/new counts, suspect verdict, new AUROC table, implications for Phase 1).
- [ ] **Step 6:** Commit report + new dataset dir. `git add reports/dataset_v1_relabel docs/operational/phase0-relabel-report-2026-06.md && git commit -m "data(phase0): re-export 8-doc dataset under hardened matcher + new suffixes; re-baseline report"`
- [ ] **Step 7: PHASE GATE** — present the report to the user; get explicit go-ahead before any Phase 1 task.

```json:metadata
{"userGate": true, "tags": ["user-gate"], "verifyCommand": "ls reports/dataset_v1_relabel/bakeoff_dataset.csv docs/operational/phase0-relabel-report-2026-06.md && python3 -c \"import csv; n=sum(int(r['used']) for r in csv.DictReader(open('reports/dataset_v1_relabel/bakeoff_dataset.csv'))); print('new positives:', n)\"", "acceptanceCriteria": ["re-export for the same 8 run IDs into reports/dataset_v1_relabel (never overwrite reports/dataset)", "report has old-vs-new positive counts per (doc,pass) + activated passes", "suspect SA2-SR71 radar_antenna c16 audited with (field,value,tier) evidence + keep/fix decision recorded", "per-feature AUROC re-run on new labels, both LODO conventions labeled", "no pipeline DB mutation", "user go-ahead recorded before Phase 1"], "requireEvidenceTokens": [["old", "dataset_v1", "35 positives"], ["new", "dataset_v1_relabel", "relabel"]]}
```

---

# Phase 1 — Gates + capture features (all default-off / diagnostics-only)

**Verified context for this phase (from code review 2026-06-10):**
- The retrieval pool is built in `search_extraction_chunks_multi_channel_full` (`app/services/extraction_chunk_search.py:901-1193`); merge at `:1055-1062`; C8 re-merge at `:1139-1149`; sort+cap at `:1176-1179` (`capped_pool = merged_pool[: cfg.top_n_candidates]`). There is NO rerank in this file — rerank happens in `extraction_routing.py:601-650` over the returned pool (`pool_dicts` built at `:574-593` with `content_text` + `merged_candidate` back-ref), then `score_candidates` at `:648`, slice at `:650`.
- `merge_candidates` is precision-only for lexical/pattern (skips keys not in dense buckets, `extraction_candidate_scoring.py:165-167,186-188`). The pattern for building pool-external candidates is `_build_lexical_table_candidates` (`extraction_routing.py:164-252`).
- `MergedCandidate` is at `extraction_candidate_scoring.py:49-84`; `COMPONENT_KEYS` at `:256-273` (16 keys); norm computations `:358-452`; components dict `:458-475` (`"cosine"` = `mc.vector_score or 0.0`); `score_components_for_pool` `:489-507`.
- `search_extraction_chunks_dense_multi_query` (`:1227-1386`) computes ONE matmul `scores = chunk_matrix @ query_matrix.T` of shape `(N_chunks, N_queries)` (col 0 = entity query, col i+1 = field i), then keeps only per-column top-k (`_top_k_results`); the matrix is discarded.
- `RetrievalProfile` (`app/services/ontology_bundles.py:67-287`) is `extra="forbid"` — every new knob must be declared. `PassRetrievalSignals`/`FieldRetrievalQuery` are frozen dataclasses in `extraction_query_builder.py:25-56`; `signals.field_queries` EXCLUDES `system_name` + INTERNAL fields.
- Byte-identical-default test pattern: `TestScoreCandidatesDecomposedFlagOff::test_byte_identical_legacy_with_nonzero_decomposed_hits` (`tests/unit/test_extraction_candidate_scoring.py:1801-1865`) — same pool with hot new features + hot weights, flag off, assert exact equality of orderings AND score lists.

### Task 5: Unit-signature derivation + G1 gate predicate module

**Goal:** Every routed pass gets a schema-derived unit signature; a pure module exposes the G1 predicate (digit present AND signature-unit token present) using the Task-1 hardened matcher.

**Files:**
- Create: `app/services/extraction_unit_gate.py`
- Modify: `app/services/extraction_query_builder.py` (add `unit_signature` to `PassRetrievalSignals`, populate in `build_retrieval_profile`)
- Modify: `app/services/ontology_bundles.py` (new `RetrievalProfile.unit_gate: bool = Field(default=False, ...)`)
- Test: `tests/unit/test_extraction_unit_gate.py`

**Acceptance Criteria:**
- [ ] `PassRetrievalSignals.unit_signature: tuple[str, ...]` = sorted union of `units_for(field)` over the pass's **record-class `model_fields`** (NOT `field_queries` — must include every groundable field; `system_name` contributes nothing since `units_for` returns `[]`)
- [ ] `chunk_passes_unit_gate(text_nfc, signature) -> bool` is True iff text contains a digit AND `has_unit_token(text_nfc, signature)` (imports `has_unit_token` from `field_value_grounding` — single-source rule)
- [ ] Gate is a SUPERSET of the label on the fixture: for every `value_in_chunk_cases` fixture entry with non-null `expect`, the gate fires on that text given the case's units
- [ ] `RetrievalProfile(unit_gate=True)` parses; default False; `extra="forbid"` still passes existing manifest loads

**Verify:** `python3 -m pytest tests/unit/test_extraction_unit_gate.py tests/unit/test_extraction_query_builder.py -v` → PASS

**Steps:**

- [ ] **Step 1: Failing tests** `tests/unit/test_extraction_unit_gate.py`:

```python
import json
from pathlib import Path

import pytest

from app.services.extraction_unit_gate import chunk_passes_unit_gate, signature_for_fields
from app.services.field_value_grounding import nfc

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "docker" / "docling-graph" / "tests" / "fixtures" / "unit_matcher_cases.json"
)
_CASES = json.loads(_FIXTURE.read_text())


def test_signature_for_fields_unions_units():
    sig = signature_for_fields(["max_intercept_km", "max_launch_angle_deg", "system_name"])
    assert "km" in sig and "км" in sig and "deg" in sig and "°" in sig
    assert "kw" not in sig  # not this pass's unit


def test_gate_requires_digit_and_unit():
    sig = signature_for_fields(["max_intercept_km"])
    assert chunk_passes_unit_gate(nfc("range is 50 km"), sig) is True
    assert chunk_passes_unit_gate(nfc("range in km is unknown"), sig) is False  # no digit
    assert chunk_passes_unit_gate(nfc("range is 50 miles"), sig) is False       # no signature unit


@pytest.mark.parametrize(
    "case", [c for c in _CASES["value_in_chunk_cases"] if c["expect"] is not None]
)
def test_gate_superset_of_label(case):
    """By construction: any text the label can ground must fire the gate."""
    assert chunk_passes_unit_gate(nfc(case["text"]), tuple(case["units"])) is True
```

- [ ] **Step 2: Run to verify failure** — module missing.

- [ ] **Step 3: Implement** `app/services/extraction_unit_gate.py` (~40 lines):

```python
"""G1 recall-floor gate (guarded-ranker spec §3).

The gate mirrors the value-grounding LABEL's form without knowing values:
force-keep a chunk for pass P iff it contains a digit AND a unit token from
P's unit signature. Token matching is imported from field_value_grounding —
the SAME matcher the label uses; gate and label must never drift.
No equipment names, no config: the signature derives from schema field-name
suffixes only.
"""
import re

from app.services.field_value_grounding import has_unit_token, units_for

_DIGIT = re.compile(r"\d")


def signature_for_fields(field_names) -> tuple[str, ...]:
    """Sorted union of unit synonyms over the pass's field names."""
    units: set[str] = set()
    for name in field_names:
        units.update(units_for(name))
    return tuple(sorted(units))


def chunk_passes_unit_gate(text_nfc: str, signature) -> bool:
    """True iff the (already nfc()-folded) chunk text could ground ANY
    numeric+unit value of this pass — digit present + signature unit token."""
    if not signature or not text_nfc:
        return False
    return bool(_DIGIT.search(text_nfc)) and has_unit_token(text_nfc, signature)
```

- [ ] **Step 4: Thread the signature** — in `extraction_query_builder.py`: add `unit_signature: tuple[str, ...] = ()` to `PassRetrievalSignals` (frozen dataclass, `:38-47` region); in `build_retrieval_profile`, after `record_cls` is resolved, set `unit_signature=signature_for_fields(record_cls.model_fields.keys())`. Add `unit_gate: bool = Field(default=False, description="Enable the G1 unit-signature recall gate: chunks containing a digit + a pass-unit token join the candidate pool exempt from the top_n cap (guarded-ranker spec §3). Default False (byte-identical legacy pool).")` to `RetrievalProfile` next to `lexical_decomposed` (`ontology_bundles.py:266-275` region).

- [ ] **Step 5: Run** Verify command → PASS (including existing query-builder tests — `PassRetrievalSignals` gains a defaulted field, constructor call sites unaffected).

- [ ] **Step 6: Commit** — `feat(router): G1 unit-signature gate module + per-pass unit signatures (inert; flag default off)`

### Task 6: Retain per-row dense cosines (max_field_cosine) in the multi-query scorer

**Goal:** The full cosine matrix already computed in `search_extraction_chunks_dense_multi_query` stops being discarded: every row gets `entity_cosine`, `max_field_cosine`, `mean_top3_field_cosine`, available to merge/gate/capture.

**Files:**
- Modify: `app/services/extraction_chunk_search.py:1227-1386` (return per-row stats), `:984-986` (unpack), `:1055-1062` + `:1139-1149` (pass into merge via new arg) — see Step 3 for the chosen plumbing
- Modify: `app/services/extraction_candidate_scoring.py` (MergedCandidate fields + COMPONENT_KEYS + components dict)
- Test: `tests/unit/test_extraction_chunk_search_dense_multi.py` (extend), `tests/unit/test_extraction_candidate_scoring.py` (extend)

**Acceptance Criteria:**
- [ ] `search_extraction_chunks_dense_multi_query` returns a third value: `row_cosines: dict[str, dict[str, float]]` keyed by the same candidate key as `_top_k_results` properties (`vertex_id` fallback `self_ref`), each `{"entity_cosine": float, "max_field_cosine": float, "mean_top3_field_cosine": float}` computed over ALL rows (not just top-k) BEFORE the per-column slice; passes with zero field queries yield 0.0 for the field stats
- [ ] `MergedCandidate` gains `max_field_cosine: float = 0.0`, `mean_top3_field_cosine: float = 0.0`; `merge_candidates` takes `row_cosines: dict | None = None` and stamps them for every candidate
- [ ] `COMPONENT_KEYS` gains `"max_field_cosine"`, `"mean_top3_field_cosine"` (appended at the END, before nothing — order is the capture contract; a0 reads keys by name so append-only is safe); components dict emits them
- [ ] `final_score` byte-identical (new fields carry NO weight anywhere)

**Verify:** `python3 -m pytest tests/unit/test_extraction_chunk_search_dense_multi.py tests/unit/test_extraction_candidate_scoring.py -v` → PASS

**Steps:**

- [ ] **Step 1: Failing test** (extend `test_extraction_chunk_search_dense_multi.py`): build 3 rows with known embeddings + 2 field queries such that row B's best field cosine ranks OUTSIDE `field_query_top_k=1` for every field; assert the third return value contains B with the hand-computed `max_field_cosine` (cosines are exact for unit vectors — use orthogonal/parallel vectors for clean values 0.0/1.0).
- [ ] **Step 2:** Run → FAIL (2-tuple return).
- [ ] **Step 3: Implement.** In `search_extraction_chunks_dense_multi_query`, after the matmul (`:1323`): if `field_queries` is non-empty, `field_block = scores[:, 1:]`; per row compute max and mean-of-top-3 (`np.sort(field_block, axis=1)[:, -3:].mean(axis=1)` guarding `n_fields < 3` with what exists); build `row_cosines[key] = {...}` for ALL `valid_rows` (key = `row.get("vertex_id") or row["self_ref"]` — matches `_candidate_key`). Return 3-tuple; update the single call site (`:984-986`) and the thin wrapper `search_extraction_chunks_multi_channel` if it touches the return. Pass `row_cosines` through both `merge_candidates` calls (new keyword arg, default None for compat); in `merge_candidates` step 5 stamp the two floats (0.0 when absent). Append the two keys to `COMPONENT_KEYS` and the components dict (`extraction_candidate_scoring.py:458-475`): `"max_field_cosine": mc.max_field_cosine, "mean_top3_field_cosine": mc.mean_top3_field_cosine`.
- [ ] **Step 4: Byte-identical guard** (extend the `:1801-1865` pattern): same pool with hot `max_field_cosine` values → identical orderings + scores.
- [ ] **Step 5:** Run Verify → PASS. **Commit:** `feat(router): retain per-row entity/max-field/mean-top3 cosines from the multi-query matmul (capture-only)`

### Task 7: Gate union into the candidate pool (exempt from the cap) + diagnostics

**Goal:** When `unit_gate=True`, G1 runs over ALL rows; gated chunks join the pool past `top_n_candidates`, flow through rerank/C5/capture, and are flagged.

**Files:**
- Modify: `app/services/extraction_chunk_search.py:1153-1192` (gate scan + union after C8, around the sort/cap), `MultiChannelDiagnostics` (same file, add counts)
- Modify: `app/services/extraction_candidate_scoring.py` (`MergedCandidate.gate_flags: set[str]`, `COMPONENT_KEYS` += `"unit_gate"`, components dict)
- Modify: `app/api/v1/extraction_routing.py:1490-1500` (capture slice covers the enlarged pool)
- Test: `tests/unit/test_extraction_unit_gate.py` (extend), `tests/unit/test_v1_extraction_routing.py` (extend)

**Acceptance Criteria:**
- [ ] With `unit_gate=True`: a row whose text passes G1 but which is NOT in the dense-capped pool appears in the returned pool with `retrieval_sources ⊇ {"unit_gate"}`, `gate_flags == {"unit"}`, cosines stamped from Task 6's `row_cosines`
- [ ] Gated chunks are EXEMPT from the `top_n_candidates` cap; non-gated pool unchanged (the first `top_n` by dense sort are the same objects)
- [ ] With `unit_gate=False` (default): returned pool byte-identical to today (count, order, keys)
- [ ] `MultiChannelDiagnostics` carries `unit_gate_total` (rows passing G1) and `unit_gate_added` (gated rows not already in the capped pool)
- [ ] Components include `"unit_gate"`: 1.0/0.0; the endpoint's `score_components_all` includes EVERY pool member incl. gate-added (the `[: profile.top_n_candidates]` truncation at `extraction_routing.py:1494-1495` is widened to the full mc-bearing pool)

**Verify:** `python3 -m pytest tests/unit/test_extraction_unit_gate.py tests/unit/test_v1_extraction_routing.py tests/unit/test_extraction_routing_fallback.py -v` → PASS

**Steps:**

- [ ] **Step 1: Failing test:** in the multi-channel-full unit test setup (copy the fixture style of `test_extraction_chunk_search_dense_multi.py` / `test_c8_identity_anchor_channel.py`): 60 rows where row 55 ("…spec block 180 kw…") is dense-ranked below `top_n_candidates=50` for every query; `cfg = RetrievalProfile(top_n_candidates=50, unit_gate=True)`; signals with `unit_signature=("kw",)`. Assert row 55's key in returned pool keys with `gate_flags == {"unit"}` and `len(pool) == 51`. Mirror test with `unit_gate=False` → 50, row 55 absent.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** in `search_extraction_chunks_multi_channel_full` immediately before the sort/cap (`:1176`): when `cfg.unit_gate and retrieval_signals.unit_signature`: fold the text once per row (the C8 lexical sub-channel at `:1159-1171` already folds `mc.chunk_text` — gate needs ALL rows, so fold `row.get("chunk_text")` via the same `unicodedata.normalize("NFC", …).casefold()`); `gated_keys = {key(row) for row passing chunk_passes_unit_gate}` (key = `vertex_id or self_ref`, the `_candidate_key` convention). After `capped_pool = merged_pool[: cfg.top_n_candidates]`: mark `mc.gate_flags.add("unit")` + `mc.retrieval_sources.add("unit_gate")` for capped-pool members in `gated_keys`; for gated keys NOT in `merged_pool` at all, build MergedCandidates from rows following the `_build_lexical_table_candidates` field-by-field pattern (`extraction_routing.py:238-257`) with `vector_score=row_cosines.get(key, {}).get("entity_cosine")`, `retrieval_sources={"unit_gate"}`, `gate_flags={"unit"}`; for gated members of `merged_pool` beyond the cap, append the existing objects. `capped_pool = capped_pool + gate_extras`. Add the two counters to `MultiChannelDiagnostics`. `MergedCandidate.gate_flags: set[str] = dataclasses.field(default_factory=set)`. `COMPONENT_KEYS` += `"unit_gate"`; components dict: `"unit_gate": 1.0 if "unit" in mc.gate_flags else 0.0`. Widen the `_sca_pool` slice (`extraction_routing.py:1494-1495`) from `[: profile.top_n_candidates]` to the full `[d for d in reranked_pool if "merged_candidate" in d]` (the cap was defensive; pool growth is bounded by gate selectivity — log a warning if `len > 4 * profile.top_n_candidates`).
- [ ] **Step 4: Byte-identical default test:** default-profile multi-channel run twice (flag off / flag absent) → identical pool keys and order; plus the score_candidates hot-fields equality test extended with hot `gate_flags`.
- [ ] **Step 5:** Run Verify → PASS. **Commit:** `feat(router): G1 gate union — gated chunks join the pool exempt from the top_n cap (flag-gated, default off)`

### Task 8: Structural text features (digit_density, label_value_lines, unit_token_count)

**Goal:** Three cheap, name-free, pass-aware-where-it-matters text statistics enter the capture for every candidate.

**Files:**
- Modify: `app/services/extraction_candidate_scoring.py` (compute in `score_candidates` per candidate from `mc.chunk_text` + `cfg`/signals-independent helpers; COMPONENT_KEYS + components)
- Modify: `app/services/extraction_unit_gate.py` (add `count_unit_tokens(text_nfc, signature) -> int`)
- Modify: `app/api/v1/extraction_routing.py` (thread `unit_signature` into scoring — `score_candidates` gains optional `unit_signature: tuple[str, ...] = ()` parameter; pass `signals.unit_signature` at all 5 `score_candidates`/`score_components_for_pool` call sites)
- Test: `tests/unit/test_extraction_candidate_scoring.py` (extend)

**Acceptance Criteria:**
- [ ] `digit_density` = digit chars / max(1, len(text)); `label_value_lines` = count of lines matching `^\s*[-•]?\s*[^:\n]{2,40}:\s*\S` (the `- <label>: <value>` shape `render_graph.py` emits) capped at 20; `unit_token_count` = bounded-token unit hits against the pass signature (0 when signature empty), capped at 20
- [ ] All three appear in COMPONENT_KEYS + components for every candidate; `final_score` byte-identical (no weights)
- [ ] Computed on raw `mc.chunk_text` folded once — no per-candidate recompiling of the unit regexes (compile per call from the signature ONCE)

**Verify:** `python3 -m pytest tests/unit/test_extraction_candidate_scoring.py -v` → PASS

**Steps:**

- [ ] **Step 1: Failing tests:** hand-built candidates — a spec-block text (`"- Peak Power: 180 kW\n- PRF: 2.8 kHz"`) vs a prose text; assert `digit_density` ordering, `label_value_lines == 2` vs `0`, `unit_token_count == 2` vs `0` with signature `("kw","khz")`; byte-identical-default equality test extended.
- [ ] **Step 2:** FAIL. **Step 3: Implement** (components-only; helpers at module top of `extraction_candidate_scoring.py`; `count_unit_tokens` in the gate module reusing `unit_token_regex`). **Step 4:** PASS. **Commit:** `feat(router): structural text features in capture (digit_density, label_value_lines, unit_token_count)`

**PHASE 1 GATE: present capture-feature smoke results (one local multi-channel test run) to the user before Phase 2.**

---

# Phase 2 — is_table, G2, keyword revival

**Verified context:** merged-mode `modality` is hard-coded `'merged'` (`extraction_chunk_index.py:1115`); table identity at index time = `MergedChunk.source_refs` ∩ ({raw `#/tables/N` refs} ∪ {synth text refs in `_synth_only_table_refs` values, dict at `:1279`, populated `:1281-1308`}). With upstream table-norm + suppress_raw (production), normalized tables are synthetic TextItems — `DocItemLabel.TABLE` does NOT mark them. Schemaless persistence precedent: `section_path`/`headings` (SET column + `read_chunk_*` accessor with default, `extraction_chunk_index.py:162-290`); declared-property precedent: `chunk_index`/`source_refs`/`token_count` (`arcadedb_schema.py:38-57` + Phase 3b backfill `:247-309`). Merged table-chunk text ALREADY carries `TABLE: {caption}` + ENTITY identity block (column headers) + `- label: value unit` rows (`render_graph.py` via `:1284-1300`, units preamble suppressed by `emit_unit_hint=False` at `:1299`) + the contextualize() heading prefix.

### Task 9: Persist + wire `is_table` (task #70) with a score-neutral default

**Goal:** Table identity survives merged-mode indexing, reaches `table_meta` at both merge sites, and `is_table` becomes a live capture feature — while production `final_score` stays byte-identical via `table_boost` 0.08 → 0.0.

**Files:**
- Modify: `app/services/hybrid_chunking.py:126-172` (`MergedChunk.is_table: bool = False`) and the build loop (raw `#/tables/` detection)
- Modify: `app/services/extraction_chunk_index.py` (`_INSERT_MERGED_SQL` + `_insert_merged_chunk_row` SET column; insert loop ORs in synth-ref membership from `_synth_only_table_refs`; new `read_chunk_is_table` accessor next to `:224-247`)
- Modify: `app/services/extraction_chunk_search.py` (project `is_table` in the per-run SELECT; build `table_meta={key: "table" ...}` at `:1055-1062`, `:1139-1149`, and `build_pool_from_multi_channel_state` `:879`)
- Modify: `app/api/v1/extraction_routing.py:244` (`content_type=read_chunk_is_table(row) and "table" or None` — keep the `_build_lexical_table_candidates` candidates consistent)
- Modify: `app/services/ontology_bundles.py:194` (`table_boost` default 0.08 → 0.0, comment citing the section_weight precedent at `:180-193`)
- Test: `tests/unit/test_extraction_candidate_scoring.py`, plus the chunk-index unit tests' file (extend whichever covers `_insert_merged_chunk_row`)

**Acceptance Criteria:**
- [ ] A MergedChunk whose `source_refs` contain a raw `#/tables/N` ref OR any synth ref from `_synth_only_table_refs` persists `is_table=true`; all others `false`
- [ ] `read_chunk_is_table(row)` defaults False for legacy rows (no column)
- [ ] Both merge sites + the state-pool builder pass real `table_meta`; `content_type == "table"` → `is_table` component 1.0
- [ ] `table_boost` default 0.0; explicit-manifest `table_boost: 0.08` still works; default-config `final_score` byte-identical on pools containing table candidates (test proves the flip compensates the wiring)
- [ ] Old captured runs unaffected (accessor default) — documented in the commit body

**Verify:** `python3 -m pytest tests/unit/test_extraction_candidate_scoring.py tests/unit/test_extraction_chunk_search_direct.py -v` + the chunk-index test file → PASS

**Steps:** (TDD per site)

- [ ] **Step 1:** Failing accessor + insert tests (legacy row → False; synth-ref chunk → True; raw-ref chunk → True).
- [ ] **Step 2:** `MergedChunk.is_table` + build-loop raw-ref detection (`any(ref.startswith("#/tables/") for ref in source_refs)`); insert loop ORs `bool(set(c.source_refs) & synth_ref_union)` where `synth_ref_union = {r for refs in _synth_only_table_refs.values() for r in refs}` (empty when norm off — raw refs cover that path). SET column via the section_path precedent (no `arcadedb_schema.py` declaration; schemaless).
- [ ] **Step 3:** SELECT projection + `table_meta` construction at the 3 sites: `table_meta = {(r.get("vertex_id") or r.get("self_ref")): "table" for r in rows if read_chunk_is_table(r)}`.
- [ ] **Step 4:** `table_boost` default flip + byte-identical test (pool with `content_type="table"` candidates: default cfg score == pre-change score; `RetrievalProfile(table_boost=0.08)` reproduces the old boost).
- [ ] **Step 5:** Verify → PASS. **Commit:** `feat(router): wire is_table end-to-end (index-time persistence, table_meta at all merge sites) with score-neutral table_boost=0.0 default`

### Task 10: G2 table gate

**Goal:** `is_table` chunks that are unit-bearing join the gate union (flag `"table"`), covering table positives whose serialized text might fail G1's digit/unit co-occurrence only marginally — and giving the eval a separable gate channel.

**Files:**
- Modify: `app/services/extraction_chunk_search.py` (gate scan: add G2 condition), `app/services/extraction_candidate_scoring.py` (components `"table_gate"`), `MultiChannelDiagnostics` (`table_gate_added`)
- Test: `tests/unit/test_extraction_unit_gate.py` (extend)

**Acceptance Criteria:**
- [ ] With `unit_gate=True`: a row with `is_table` true AND `has_unit_token(text, signature)` true (digit not required) gets `gate_flags ⊇ {"table"}` and joins the union exactly like G1 members
- [ ] `"table_gate"` component emitted (1.0/0.0); default-off byte-identical preserved
- [ ] G1 ∪ G2 keep-set reported separately in diagnostics (`unit_gate_added`, `table_gate_added`)

**Verify:** `python3 -m pytest tests/unit/test_extraction_unit_gate.py -v` → PASS

**Steps:** failing test (table row, unit token, NO digit → gated with `{"table"}`) → implement inside the Task-7 scan (`read_chunk_is_table(row) and has_unit_token(...)`) → byte-identical default → commit `feat(router): G2 table gate joins the gate union`.

### Task 11: Union schema-derived unit keywords into manifest keyword lists

**Goal:** `inject_pass_keywords` merges `derive_pass_keywords(signals)` INTO non-empty manifest lists (dedup, manifest order first) instead of being shadowed — unit vocabulary reaches the keyword channel on every pass.

**Files:**
- Modify: `app/api/v1/extraction_routing.py:80-113`
- Test: `tests/unit/test_v1_extraction_routing.py` or the existing injector tests (extend where `inject_pass_keywords` is covered)

**Acceptance Criteria:**
- [ ] Non-empty manifest list + derived units → union (manifest entries first, derived appended, NFC+casefold dedup — reuse the dedup shape in `derive_pass_keywords`, `extraction_query_builder.py:276-285`)
- [ ] Empty manifest list → derived only (current behavior)
- [ ] `final_score` byte-identical (keyword hits feed `pass_keyword_hits` → diagnostics-only; `alias_hits` untouched) — asserted by test
- [ ] Docstring updated (the "manifest override wins" contract changes — say so loudly)

**Verify:** `python3 -m pytest tests/unit/test_v1_extraction_routing.py -v` → PASS

**Steps:** failing test (profile with `["magnetron"]` + signals deriving `["kW","MHz"]` → `["magnetron","kw-or-kW per dedup rule","MHz"]`-shaped union, exact casing preserved from source list) → implement → byte-identical-final_score test → commit `feat(router): union derived unit keywords into manifest lexical_keywords (was: shadowed)`.

### Task 12: Word-boundary keyword matching + mining/runtime normalization unification

**Goal:** Single-token pass-keywords match with word boundaries (kills `mach`→`machine`); the offline mining/check scripts use the SAME normalization as runtime so their stats predict runtime behavior.

**Files:**
- Modify: `app/services/extraction_lexical_search.py:161-222` (`keyword_hit_counts` ONLY — the alias channel `lexical_hit_counts` is LIVE in `final_score` and stays byte-identical)
- Modify: `scripts/mine_pass_keywords.py:33-35`, `scripts/keyword_discrimination_check.py:44-46` (NFKD+strip-accents → NFC+casefold, matching `extraction_lexical_search._nfc`)
- Test: `tests/unit/test_extraction_lexical_search.py` (extend)

**Acceptance Criteria:**
- [ ] In `keyword_hit_counts`: single-token needles (no whitespace) match via `re.search(rf"(?<![^\W\d_]){re.escape(kw)}(?!\w)", haystack)`; multi-word phrases keep substring semantics
- [ ] `"mach"` no longer hits `"machinery"`; `"fins"` no longer hits `"muffins"`; `"mach 2"` still hits `"at mach 2 the"`
- [ ] `lexical_hit_counts` (alias channel) behavior unchanged — explicit regression test
- [ ] Mining scripts normalize identically to runtime (`unicodedata.normalize("NFC", s).casefold()`)

**Verify:** `python3 -m pytest tests/unit/test_extraction_lexical_search.py -v` → PASS

**Steps:** failing tests → implement (compile needle patterns once per call) → commit `fix(lexical): word-boundary single-token pass-keyword matching + NFC-unify mining scripts`.

### Task 13: Per-keyword runtime weights

**Goal:** Keywords can carry mined-lift weights at runtime; hit counting becomes a weighted sum (default weight 1.0 — unweighted lists byte-identical).

**Files:**
- Modify: `app/services/ontology_bundles.py` (`RetrievalProfile.lexical_keyword_weights: dict[str, float] = Field(default_factory=dict, ...)`)
- Modify: `app/services/extraction_lexical_search.py` (`keyword_hit_counts(rows, keywords, weights=None)` — each matched needle contributes `weights.get(kw, 1.0)`), `app/services/extraction_chunk_search.py:1006-1008` (pass `cfg.lexical_keyword_weights`)
- Test: `tests/unit/test_extraction_lexical_search.py`

**Acceptance Criteria:**
- [ ] Empty weights dict → counts byte-identical to today (ints preserved as floats equal in value)
- [ ] `{"kw_a": 2.0}` → a chunk matching kw_a scores `keyword_hits == 2.0`
- [ ] `pass_keyword_norm` ratio-max normalization unchanged (floats flow through `max(1, pool_max)` — adjust to `max(1.0, pool_max)`)

**Verify:** `python3 -m pytest tests/unit/test_extraction_lexical_search.py tests/unit/test_extraction_candidate_scoring.py -v` → PASS

**Steps:** failing test → implement → byte-identical default → commit `feat(lexical): optional per-keyword weights in the pass-keyword channel`.

### Task 14: Header-projection diagnostic → conditional `match_text`

**Goal:** Determine empirically whether table-derived chunks' MATCHING haystack is actually missing header/caption vocabulary (the gather suggests `render_graph` text already carries `TABLE: {caption}` + ENTITY header block + `- label: value` rows); implement the separate `match_text` haystack ONLY if the diagnostic shows a real gap (YAGNI otherwise).

**Files:**
- Create: `scripts/diagnose_table_haystack.py` (read 3 known Engagement table-positive chunks' `chunk_text` from ArcadeDB for run `1329caf5…`; report which pass keywords / field aliases / unit tokens appear; compare against the doc's raw table captions)
- Conditional (only on confirmed gap): `app/services/hybrid_chunking.py` + `extraction_chunk_index.py` (nullable `match_text` SET column, embed `match_text or text` at `:1358-1361`), `extraction_lexical_search.py` (haystack = `match_text or chunk_text`), `extraction_routing.py:574-593` (`content_text = match_text or chunk_text`), accessor `read_chunk_match_text`
- Test: conditional — consumer-matrix tests per the spec (`match_text` feeds embed/lexical/rerank; `chunk_text`/lineage/grounding untouched)

**Acceptance Criteria:**
- [ ] Diagnostic report committed (`docs/operational/table-haystack-diagnostic-2026-06.md`): per-chunk, the EXACT tokens present/absent for each lexical channel, and a clear go/no-go for `match_text`
- [ ] If GO: `match_text` never read by `build_extracted_from_groundtruth` / value-grounding / provenance paths (grep-verified list in the report); raw `chunk_text` byte-identical in storage; embedding input switches only for rows WITH `match_text`
- [ ] If NO-GO: task closes with the diagnostic alone; spec §5.2 marked superseded-by-evidence in the report

**Verify:** diagnostic: `A0_ARCADEDB_URL=http://localhost:2480 python3 -m scripts.diagnose_table_haystack --run 1329caf5-d57b-403d-96ea-1399a7d3d67f --chunks 93,94,124` → report rows for all 3 chunks. Conditional impl: consumer-matrix pytest file → PASS

**Steps:** diagnostic script → report → **STOP and present to user (mini-gate)** → conditional implementation per spec §5.2 with TDD → commit(s).

### Task 15: Re-mine keywords against the value-grounded label

**Goal:** The miner's positives come from the CLEAN label (value-grounded CSV from Task 4's re-export), Engagement included, with the digit/unit exclusion relaxed so unit-bearing tokens are minable; output stays a human-review list.

**Files:**
- Modify: `scripts/mine_pass_keywords.py` (`--positives-csv` flag: posset per (run, pass) from the re-exported `bakeoff_dataset.csv` rows where `used==1`, replacing the field_provenance source at `:75-83` when the flag is set; `--allow-units` flag: skip the DESIG/digit drop for tokens that are unit synonyms — import the unit lexicon from `field_value_grounding.SUFFIX_UNITS` values)
- Test: none beyond `--help` smoke (offline analysis tool); Verify is the run itself

**Acceptance Criteria:**
- [ ] `python3 -m scripts.mine_pass_keywords --runs <8 ids> --positives-csv reports/dataset_v1_relabel/bakeoff_dataset.csv --allow-units` produces per-pass candidate lists with lift/posfire/docspread stats
- [ ] Output explicitly labeled REVIEW-ONLY; no manifest writes
- [ ] Equipment-designation guardrail intact for non-unit tokens (DESIG still applied to everything else)

**Verify:** the command above runs and prints per-pass tables (requires DB; run alongside Task 16 work)

**Steps:** implement flags → run → attach output to the Phase-2 gate discussion → commit `feat(mining): mine against value-grounded labels with unit tokens allowed (review-only)`.

**PHASE 2 GATE: present is_table/G2/keyword-revival test results + the re-mine review list + Task 14 diagnostic to the user before Phase 3.**

---

# Phase 3 — Harness + re-collect + evaluate

### Task 16: Export/eval harness for the new features + guarded-ranker frontier

**Goal:** Offline tooling understands the new capture features and can answer: "savings at recall 1.0 for gates ∪ quantile-cut ranker vs calibrated-score-only vs production final_score," with the literal gate-coverage acceptance check.

**Files:**
- Modify: `scripts/a0_captured_separation.py` (`NEW_FEATURES: tuple = ("max_field_cosine", "mean_top3_field_cosine", "unit_gate", "table_gate", "digit_density", "label_value_lines", "unit_token_count")` appended to a combined `ALL_FEATURES`; `Row` dataclass + row parsing default missing keys to 0.0 so OLD captures still load; `is_table` stays in FEATURES and goes live)
- Modify: `scripts/export_bakeoff_dataset.py` (columns = old + NEW_FEATURES; backward-compatible)
- Create: `scripts/eval_guarded_ranker.py` — loads a dataset CSV; CLI `--features`, `--quantile-grid`, `--k-min`, `--k-max`, `--gate-cols unit_gate,table_gate`; computes per-(run,pass) pools; for each q: keep = gate-members ∪ (score ≥ pool-quantile(q), clipped k_min/k_max); reports recall/kept-fraction frontier pooled-OOF AND mean-per-fold (GroupKFold by run_id for the LogReg score; sign-constraint = drop-and-refit on wrong-signed coefficients, document each drop); baselines: final_score topk sweep + calibrated-score-only quantile sweep
- Create: `scripts/check_gate_coverage.py` — exits 1 unless EVERY `used==1` row has `unit_gate==1 or table_gate==1` (the literal spec §3 acceptance), prints the misses
- Test: `tests/unit/test_eval_guarded_ranker.py` — synthetic 2-doc dataframe with known frontier; gate-coverage pass/fail cases

**Acceptance Criteria:**
- [ ] Old `reports/dataset/bakeoff_dataset.csv` still loads through the updated a0/export code paths (missing new columns → 0.0)
- [ ] `eval_guarded_ranker` reproduces hand-computable results on the synthetic fixture (e.g. q=0.5 on a 4-candidate pool with 1 gated member → exact keep set)
- [ ] `check_gate_coverage` returns 0 on a fixture where gates cover positives, 1 with a named miss otherwise
- [ ] Both LODO conventions printed with explicit labels (`pooled-OOF` / `mean-per-fold`)

**Verify:** `python3 -m pytest tests/unit/test_eval_guarded_ranker.py -v` → PASS

**Steps:** TDD on the synthetic fixture → implement the three scripts → commit `feat(calibration): guarded-ranker frontier eval + literal gate-coverage check + NEW_FEATURES plumbing`.

### Task 17: Deploy + re-collect the 8 docs + evaluate (USER GATE)

**Goal:** Fresh capture of the 8-doc corpus with the full Phase 0–2 feature set live, then the guarded-ranker evaluation report that decides Phase 4.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:** none modified — operational. Output: `reports/dataset_v2/`, `docs/operational/phase3-guarded-ranker-eval-2026-06.md`

**Acceptance Criteria:**
- [ ] Deploy verified: docling-graph image rebuilt (`cd /home/josh/development/EIP-MMDPP && docker compose -p eip-mmdpp build docling-graph && docker compose -p eip-mmdpp up -d --force-recreate docling-graph`) AND both workers restarted (`docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1`) with `docker inspect -f '{{.State.StartedAt}}'` captured AFTER the code landed (catch-all-worker trap); test bundle manifest sets `unit_gate: true` on its 2 field-group passes; `VECTOR_ROUTER_MODE=shadow`
- [ ] Idle-pool check passes before runs (active-inference probe, expect ~sub-second response) — wall-clock numbers from a contended pool are invalid
- [ ] All 8 docs re-ingested/re-run; 8 fresh run IDs recorded; `score_components_all` non-empty for every field-group pass (spot-check SQL)
- [ ] `reports/dataset_v2/` exported with NEW_FEATURES populated (non-constant `is_table`, `unit_gate` where expected)
- [ ] `python3 -m scripts.check_gate_coverage reports/dataset_v2/bakeoff_dataset.csv` → exit 0 (recall floor literal check; any miss = unit-lexicon bug → fix and re-check before closing)
- [ ] Eval report: guarded frontier vs calibrated-only vs final_score baselines, both LODO conventions, per-doc breakdown, gate_keeps/ranker_keeps cost table → committed
- [ ] PAUSE: user go-ahead recorded before any Phase 4 task

**Verify:** `python3 -m scripts.check_gate_coverage reports/dataset_v2/bakeoff_dataset.csv && ls docs/operational/phase3-guarded-ranker-eval-2026-06.md` → exit 0, file exists

**Steps:** deploy + verify StartedAt → idle probe → re-run 8 docs (same upload procedure as the prior collection; SA-2_Sources auto-runs the subset bundle) → export v2 → `check_gate_coverage` → `eval_guarded_ranker` → write + commit report → **PHASE GATE: user decision on Phase 4 with the report in hand**.

```json:metadata
{"userGate": true, "tags": ["user-gate"], "verifyCommand": "python3 -m scripts.check_gate_coverage reports/dataset_v2/bakeoff_dataset.csv && ls docs/operational/phase3-guarded-ranker-eval-2026-06.md", "acceptanceCriteria": ["deploy verified with fresh StartedAt on both workers + rebuilt docling-graph", "idle-pool probe before runs", "8 fresh runs with non-empty score_components_all", "dataset_v2 exported with new features non-constant", "check_gate_coverage exit 0", "eval report committed with both LODO conventions", "user go-ahead recorded before Phase 4"], "requireEvidenceTokens": [["dataset_v1", "old", "baseline"], ["dataset_v2", "new", "re-collect"]]}
```

---

# Phase 4 — Selection plumbing + calibration (requires Task 17 results)

### Task 18: `selection_mode` + shared cut helper + selection diagnostics

**Goal:** The four `[:profile.top_k]` slices become one helper that supports `topk` (default, byte-identical) and `guarded_quantile` (gate-keeps ∪ quantile cut with k_min floor / k_max cap, gates exempt from k_max).

**Files:**
- Modify: `app/services/ontology_bundles.py` (RetrievalProfile: `selection_mode: Literal["topk","guarded_quantile"]="topk"`, `quantile_q: float = Field(default=0.8, ge=0.0, le=1.0)`, `k_min: int = Field(default=3, gt=0)`, `k_max: int = Field(default=0, ge=0, description="0 = uncapped")`, `ranker_weights: dict[str, float] = Field(default_factory=dict, description="component-name -> weight; empty = rank by final_score")`)
- Modify: `app/services/extraction_candidate_scoring.py` (new `select_candidates(c5_scored, components, cfg) -> list[tuple[MergedCandidate, float]]`)
- Modify: `app/api/v1/extraction_routing.py:650, :766, :833, :891` (call the helper; each site already has `c5_scored` — fetch components via `score_candidates(..., return_components=True)` once and reuse for both selection and capture)
- Modify: `app/schemas/extraction_routing.py` (ChunkScopeDiagnostics: `gate_unit_keeps: int | None = None`, `gate_table_keeps: int | None = None`, `ranker_keeps: int | None = None`, `selection_threshold: float | None = None`, `selection_k: int | None = None`)
- Test: `tests/unit/test_extraction_candidate_scoring.py` (helper), `tests/unit/test_v1_extraction_routing.py` + `tests/unit/test_extraction_routing_fallback.py` (sites)

**Acceptance Criteria:**
- [ ] `selection_mode="topk"` → helper returns EXACTLY `c5_scored[: cfg.top_k]` (object-identical slice) at all four sites — proven byte-identical
- [ ] `guarded_quantile`: ranker score = `final_score` when `ranker_weights` empty, else `Σ ranker_weights[k] * components[k]` (unknown keys → hard error at profile validation, not silent 0); threshold = the `quantile_q` quantile of ranker scores within THIS pool; ranker-keeps = scores ≥ threshold clipped to `[k_min, k_max if k_max>0 else len(pool)]`; final = dedup(gate-flagged ∪ ranker-keeps) preserving rank order; **gate-flagged members are ALWAYS kept and never count against k_max**
- [ ] Diagnostics populated in guarded mode; None in topk mode (schema-stable)
- [ ] Worker side untouched (it consumes self_refs opaquely)

**Verify:** `python3 -m pytest tests/unit/test_extraction_candidate_scoring.py tests/unit/test_v1_extraction_routing.py tests/unit/test_extraction_routing_fallback.py -v` → PASS

**Steps:** TDD the helper on hand-built pools (gated member below threshold stays; k_max smaller than gate count → all gates still kept; empty pool; all-tied scores) → replace the 4 sites → endpoint tests incl. byte-identical default → commit `feat(router): selection_mode topk|guarded_quantile with shared cut helper (default byte-identical)`.

### Task 19: Calibration script (sign-constrained LogReg + quantile + margin → manifest numbers)

**Goal:** One command turns `reports/dataset_v2/` into deployable `ranker_weights` + `quantile_q` (+ per-pass overrides if warranted), with a conformal-style margin and honest LODO reporting.

**Files:**
- Create: `scripts/fit_guarded_ranker.py`
- Test: `tests/unit/test_fit_guarded_ranker.py` (synthetic fixture: known separable data → expected sign pattern, expected chosen q)

**Acceptance Criteria:**
- [ ] Fits L2 LogReg (class_weight='balanced', C from `--c-grid` via nested GroupKFold) on `--features` (default: cosine, max_field_cosine, mean_top3_field_cosine, rerank_norm, is_table, unit_token_count, digit_density, label_value_lines, negative_norm); wrong-signed coefficients (sign spec: all non-negative except negative_norm ≤ 0) → drop feature, refit, log
- [ ] Chooses the smallest `quantile_q` whose LODO pooled-OOF recall (gates ∪ cut) = 1.0 on every fold, then applies the margin: step q down by `--margin-steps` (default 1) grid steps (the finite-sample pad; document that a guarantee is NOT claimed — spec §4)
- [ ] Emits `reports/dataset_v2/guarded_ranker_fit.json` (weights, q, per-fold recall/kept, both LODO conventions) + a ready-to-paste manifest YAML snippet (`selection_mode/quantile_q/k_min/k_max/ranker_weights`)
- [ ] Refuses to emit if `check_gate_coverage` fails on the input CSV (recall floor precondition)

**Verify:** `python3 -m pytest tests/unit/test_fit_guarded_ranker.py -v` → PASS

**Steps:** TDD on synthetic data → implement → run on dataset_v2 (attach to Phase-4 gate discussion) → commit `feat(calibration): fit_guarded_ranker — weights + quantile + margin from the v2 dataset`.

### Task 20: Bundle propagation + docs + suite

**Goal:** Config lands per the propagation rule; repo docs and checks reflect the new machinery; full suite green.

**Files:**
- Modify: `ontology_bundles/air_defense_v3/manifest.yaml` + the 3 sibling bundle manifests (add `unit_gate: true` where the user enables it; selection_mode stays `topk` until the user flips it — Phase-4 gate decision)
- Modify: `VERIFICATION_CHECKLIST.md` (gate-coverage check, byte-identical default checks, mirror-fixture rule), `README.md` (guarded-ranker section: flags, scripts, the two LODO conventions)
- Check: `.env`/`.env.example` — NO new env vars were introduced by this plan (everything is RetrievalProfile/manifest config); verify with `git diff --stat` and state it in the commit
- Test: full suite

**Acceptance Criteria:**
- [ ] All 4 bundles parse (`python3 -c "from app.services.ontology_bundles import load_bundle_manifest; [load_bundle_manifest(b) for b in ('air_defense_v3','air_defense_v3_baseline_subset','air_defense_v3_narrowing_v1','air_defense_v3_merged_v1')]"`)
- [ ] `bash scripts/run_tests.sh unit` green — AND junit XML inspected (`reports/junit_unit.xml` failure/error counts == 0; do NOT trust the banner — known false-green)
- [ ] `python3 -m pytest tests/integration/test_sa2_dvina_extraction_acceptance.py -q` → 33 passed (run DIRECTLY — the file is unmarked and `-m integration` deselects it)
- [ ] VERIFICATION_CHECKLIST.md + README updated

**Verify:** the three commands above, captured

**Steps:** propagate → docs → suite (+ junit inspection) → commit `chore(bundles+docs): guarded-ranker config propagation + verification checklist/README`.

**PHASE 4 GATE: present calibration numbers + the proposed manifest flip to the user. Production `narrow_only` remains OUT of scope (spec §8).**

