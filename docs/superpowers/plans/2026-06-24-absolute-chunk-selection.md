# Absolute Chunk-Selection (signal-union) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the relative `guarded_quantile` chunk selector with a per-chunk **absolute signal-union** (`measurement OR categorical OR image-presence OR cosine≥τ`) that returns 0-to-all chunks based on content, applied to the 9 routable extraction passes.

**Architecture:** New pure signal-detector module + a schema-derived per-pass signal config; a new `absolute_union` branch in `select_candidates`; and a three-layer empty-selection contract (chunk-scope endpoint emits a new `empty_selection` mode → worker maps it to "extract nothing" → pass finalizes as ZERO_YIELD/COMPLETE, never full-doc/FAILED). Opt-in per manifest; `narrow_min_doc_tokens` size-gate stays as a safety net until online validation.

**Tech Stack:** Python 3.12, Pydantic (`RetrievalProfile`), FastAPI (chunk-scope endpoint), Celery worker, ArcadeDB (`ExtractionChunk`), pytest. Reuses `field_value_grounding.has_unit_token`/`unit_token_regex` (bounded unit matcher) and `extraction_unit_gate.signature_for_fields`.

**User decisions (already made):**
- "I want an absolute scoring mechanism" — 0 chunks when none appropriate, all when all relevant; no fixed count/percentage.
- "Keep things the way they were" — the union is **all-OR**; cosine is the single tunable τ knob. AND-combinations were tested and rejected (trade large recall for negligible precision).
- Reject C-lexical (worse than quantile). Use measurement + categorical + image + cosine.
- Measurement is **pass-specific** (only the pass's dimensions) with a **dimension-grouped** vocabulary (a length field accepts any length unit) including spelled-out + plural + imperial forms ("meters", "kilometers", "feet", "miles").
- Identity passes (`missile_identity`, `radar_identity`) are **not routable** — leave them full-doc, untouched.
- **Retain `narrow_min_doc_tokens`** as a safety net; remove only after online validation passes (§7.4 of spec).
- Derive per-pass config (dimensions/categorical/image) from the bundle schema (single source of truth), not hardcoded literals.
- Default `τ` = **0.55** global, with optional per-pass override.
- Online validation (NMUSAF/SA-2/Engagement under `absolute_union` vs full-doc baseline) is **required before flipping any production default** — Task 7 (user-ordered gate).

**Spec:** `docs/superpowers/specs/2026-06-24-absolute-chunk-selection-design.md` (commit `71115fb`).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `app/services/extraction_signal_detectors.py` | Pure per-chunk signal detectors (measurement, categorical, image) + the dimension→units expanded vocab + categorical phrase vocab | **Create** |
| `app/services/extraction_pass_signal_config.py` | Derive `{pass: PassSignalConfig(dimensions, categorical_phrases, has_image_field)}` from a bundle's schema (field suffixes + enum descriptions + `_photo` fields) | **Create** |
| `app/services/extraction_candidate_scoring.py` | Add `absolute_union` branch to `select_candidates` (combine the 4 signals from `MergedCandidate`); returns selected subset (possibly empty) | Modify (`select_candidates` ~line 724) |
| `app/services/ontology_bundles.py` | Add `cosine_tau` field + `"absolute_union"` to `RetrievalProfile.selection_mode` Literal | Modify (`RetrievalProfile`) |
| `app/api/v1/extraction_routing.py` | Emit new `mode="empty_selection"` when `absolute_union` selects 0; thread through `ChunkScopeResponse` | Modify (`select_candidates` call site + the `mode="full"` paths ~710/1279 + response model) |
| `app/workers/pipeline.py` | `_compute_effective_chunk_scope`: map `mode="empty_selection"` → sentinel "extract nothing" scope (not `None`); finalization → ZERO_YIELD/COMPLETE | Modify (`_compute_effective_chunk_scope` ~8543; `_call_extract_pass`/`_is_clean_empty_pipeline_error` ~4480) |
| `ontology_bundles/air_defense_v3/manifest.yaml` (+ `air_defense_v3_narrowing_v1`, `air_defense_v3_merged_v1`) | Flip the 9 routable passes `selection_mode: absolute_union` + `cosine_tau` | Modify |
| `tests/unit/test_extraction_signal_detectors.py`, `test_extraction_pass_signal_config.py`, `test_absolute_union_selection.py`, `test_empty_selection_contract.py` | Unit tests | **Create** |

Pass→signal config (derived in Task 0, used everywhere):
| pass | dimensions | categorical fields | image field |
|---|---|---|---|
| missile_airframe | length, mass | — | — |
| missile_propulsion | time, mass | — | — |
| missile_kinematics | length, angle | — | — |
| missile_speed_timing | velocity, time | — | — |
| radar_antenna | length, angle, gain | — | antenna_photo |
| radar_power_rf | frequency, power | — | — |
| radar_modulation | frequency, time | — | — |
| radar_timing | time, length | — | — |
| missile_guidance | — | guidance_type, seeker_type | missile_photo |

---

## Task 0: Per-pass signal config (schema-derived)

**Goal:** Derive, from a bundle's extraction schema, the per-pass `{dimensions, categorical_phrases, has_image_field}` config — no hardcoded equipment names.

**Files:**
- Create: `app/services/extraction_pass_signal_config.py`
- Test: `tests/unit/test_extraction_pass_signal_config.py`

**Acceptance Criteria:**
- [ ] `derive_pass_signal_config(bundle_key)` returns a `dict[str, PassSignalConfig]` keyed by pass_name.
- [ ] `missile_kinematics.dimensions == {"length","angle"}` (from `_km`/`_deg` field suffixes).
- [ ] `missile_guidance.categorical_fields == {"guidance_type","seeker_type"}` and `has_image_field is True` (from `missile_photo`).
- [ ] `radar_antenna.has_image_field is True` (`antenna_photo`); `dimensions == {"length","angle","gain"}`.
- [ ] Non-routable passes (`missile_identity`, `radar_identity`) are absent or flagged non-routable (not consumed by the selector).

**Verify:** `pytest tests/unit/test_extraction_pass_signal_config.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_extraction_pass_signal_config.py
from app.services.extraction_pass_signal_config import derive_pass_signal_config

def test_kinematics_dimensions():
    cfg = derive_pass_signal_config("air_defense_v3")
    assert cfg["missile_kinematics"].dimensions == {"length", "angle"}

def test_guidance_categorical_and_image():
    cfg = derive_pass_signal_config("air_defense_v3")
    g = cfg["missile_guidance"]
    assert g.categorical_fields == {"guidance_type", "seeker_type"}
    assert g.has_image_field is True

def test_antenna_image_and_dims():
    cfg = derive_pass_signal_config("air_defense_v3")
    a = cfg["radar_antenna"]
    assert a.has_image_field is True
    assert a.dimensions == {"length", "angle", "gain"}
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError: extraction_pass_signal_config`).

- [ ] **Step 3: Implement**

```python
# app/services/extraction_pass_signal_config.py
"""Derive per-pass selection-signal config from a bundle's extraction schema.
Single source of truth = the schema field names + enum descriptions; no literals."""
from __future__ import annotations
import re
from dataclasses import dataclass, field as dc_field
from functools import lru_cache

# unit-suffix -> physical dimension
SUFFIX_DIMENSION = {
    "km": "length", "m": "length", "mm": "length", "cm": "length",
    "deg": "angle", "rad": "angle",
    "sec": "time", "usec": "time", "ms": "time", "ns": "time",
    "mhz": "frequency", "ghz": "frequency", "khz": "frequency", "hz": "frequency",
    "mps": "velocity",
    "kg": "mass", "g": "mass",
    "dbi": "gain", "db": "gain",
    "kw": "power", "w": "power",
}
# enum field -> matchable categorical phrases (enum values + schema prose mappings)
CATEGORICAL_PHRASE_FIELDS = {"scan_type", "emitter_function", "system_status",
                             "guidance_type", "seeker_type"}

@dataclass
class PassSignalConfig:
    pass_name: str
    dimensions: set[str] = dc_field(default_factory=set)
    categorical_fields: set[str] = dc_field(default_factory=set)
    has_image_field: bool = False

def _suffix_dimension(field_name: str) -> str | None:
    m = re.search(r"_([a-z]+)$", field_name)
    return SUFFIX_DIMENSION.get(m.group(1)) if m else None

@lru_cache(maxsize=8)
def derive_pass_signal_config(bundle_key: str) -> dict[str, "PassSignalConfig"]:
    # introspect.py already enumerates routable passes + their pydantic field names.
    from app.services.ontology_bundles import iter_routable_pass_fields  # (pass_name, [field_names])
    out: dict[str, PassSignalConfig] = {}
    for pass_name, field_names in iter_routable_pass_fields(bundle_key):
        c = PassSignalConfig(pass_name=pass_name)
        for fn in field_names:
            dim = _suffix_dimension(fn)
            if dim:
                c.dimensions.add(dim)
            if fn in CATEGORICAL_PHRASE_FIELDS:
                c.categorical_fields.add(fn)
            if fn.endswith("_photo") or "photo" in fn or "image" in fn:
                c.has_image_field = True
        out[pass_name] = c
    return out
```

> Implementer note: `iter_routable_pass_fields(bundle_key)` must yield `(pass_name, field_names)` for the 9 routable passes only (those with a `RetrievalProfile`/`selection_mode` in the manifest). If a helper does not already exist in `ontology_bundles.py`/`introspect.py`, add a thin one that reads the manifest pass list + the pydantic template class fields (`template_class.model_fields`). Keep it schema-driven.

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git add app/services/extraction_pass_signal_config.py tests/unit/test_extraction_pass_signal_config.py && git commit -m "feat(selection): schema-derived per-pass signal config"`

---

## Task 1: Signal-detector primitives (measurement / categorical / image)

**Goal:** Pure per-chunk detectors: pass-specific measurement (bounded matcher + dimension-grouped vocab), categorical (enum phrases), image-presence (`#/pictures/` in source_refs).

**Files:**
- Create: `app/services/extraction_signal_detectors.py`
- Test: `tests/unit/test_extraction_signal_detectors.py`

**Acceptance Criteria:**
- [ ] `measurement_present({"length","angle"}, "max range 2500 km")` is True; `measurement_present({"mass"}, "2500 km")` is False.
- [ ] **Designators do NOT false-match:** `measurement_present({"length"}, "S-75M and V-88, 5Ya23")` is False (bounded matcher; short-unit guard).
- [ ] Spelled-out/imperial: `measurement_present({"length"}, "about 40 feet")` and `("...4500 meters per second")` for velocity are True.
- [ ] `categorical_present({"guidance_type"}, "uses semi-active radar homing")` is True; `categorical_present({"guidance_type"}, "the warhead weighs 200 kg")` is False.
- [ ] `image_present(["#/pictures/2", "#/texts/9"])` is True; `image_present(["#/texts/9"])` is False.

**Verify:** `pytest tests/unit/test_extraction_signal_detectors.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_extraction_signal_detectors.py
from app.services.extraction_signal_detectors import (
    measurement_present, categorical_present, image_present)

def test_measurement_pass_specific():
    assert measurement_present({"length", "angle"}, "max range 2500 km") is True
    assert measurement_present({"mass"}, "2500 km") is False

def test_measurement_rejects_designators():
    assert measurement_present({"length"}, "S-75M and V-88, 5Ya23 variant") is False

def test_measurement_spelled_and_imperial():
    assert measurement_present({"length"}, "about 40 feet tall") is True
    assert measurement_present({"velocity"}, "4500 meters per second") is True

def test_categorical():
    assert categorical_present({"guidance_type"}, "uses semi-active radar homing") is True
    assert categorical_present({"guidance_type"}, "the warhead weighs 200 kg") is False

def test_image():
    assert image_present(["#/pictures/2", "#/texts/9"]) is True
    assert image_present(["#/texts/9"]) is False
    assert image_present(None) is False
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** (reuse the bounded matcher; expanded dimension vocab)

```python
# app/services/extraction_signal_detectors.py
"""Pure per-chunk selection signals. Reuses the bounded unit matcher so short
units (m/s/g/w) never match inside designators (S-75M, 9 months)."""
from __future__ import annotations
from app.services.field_value_grounding import has_unit_token, nfc

# Dimension -> expanded unit surface forms (abbrev + spelled-out + plural + imperial).
DIMENSION_UNITS: dict[str, list[str]] = {
    "length": ["km","m","mm","cm","nmi","ft","yd","kilometers","kilometres","kilometer","kilometre",
               "meters","metres","meter","metre","millimeters","millimeter","centimeters","centimeter",
               "miles","mile","feet","foot","yards","yard","inches","nautical miles","nautical mile"],
    "mass": ["kg","g","mg","t","lb","lbs","kilograms","kilogram","grams","gram","tonnes","tonne",
             "tons","ton","pounds","pound"],
    "time": ["s","sec","secs","ms","ns","µs","us","hr","hrs","min","mins","seconds","second",
             "milliseconds","millisecond","microseconds","microsecond","nanoseconds","nanosecond",
             "minutes","minute","hours","hour"],
    "frequency": ["hz","khz","mhz","ghz","hertz","kilohertz","megahertz","gigahertz"],
    "velocity": ["m/s","km/s","km/h","kph","mph","mps","kt","kts","knots","knot","mach",
                 "meters per second","metres per second","kilometers per second"],
    "angle": ["deg","rad","°","degrees","degree","radians","radian","mrad","mils","mil"],
    "gain": ["db","dbi","dbm","dbw","decibels","decibel"],
    "power": ["w","kw","mw","watts","watt","kilowatts","kilowatt","megawatts","megawatt",
              "milliwatts","milliwatt"],
}

# Categorical enum field -> matchable phrases (enum values + schema prose-mapping phrases).
# (Lifted from the schema field descriptions; keep in sync if descriptions change.)
CATEGORICAL_PHRASES: dict[str, list[str]] = {
    "scan_type": ["rotating antenna","mechanical rotation","360-degree scan","rotating dish",
                  "sector scan","raster scan","electronically scanned","phased array","phased-array",
                  "dwell-and-switch","helical scan","spiral scan","conical scan","circular scan",
                  "aesa","pesa"],
    "emitter_function": ["search radar","early warning","acquisition radar","tracking radar",
                         "fire-control radar","fire control radar","engagement radar","illuminator",
                         "multi-function radar","multifunction","height finder","navigation radar",
                         "mfr","amdr"],
    "system_status": ["operational","in service","deployed","fielded","developmental","prototype",
                      "decommissioned","modernized","retired","exported","fms"],
    "guidance_type": ["command guidance","command-to-line-of-sight","clos","semi-active radar homing",
                      "sarh","active radar homing","arh","track-via-missile","tvm","inertial guidance",
                      "beam-rider","beam riding","infrared homing","ir homing","imaging infrared","iir",
                      "passive radar homing","prh","home-on-jam","hoj","homing","guidance"],
    "seeker_type": ["sarh seeker","arh seeker","ir seeker","iir seeker","eo seeker","mmw seeker",
                    "millimeter-wave seeker","dual-mode","electro-optical","seeker"],
}

def measurement_present(dimensions: set[str], text: str) -> bool:
    if not dimensions or not text:
        return False
    units: list[str] = []
    for d in dimensions:
        units.extend(DIMENSION_UNITS.get(d, ()))
    return has_unit_token(nfc(text), units)

def categorical_present(categorical_fields: set[str], text: str) -> bool:
    if not categorical_fields or not text:
        return False
    t = text.lower()
    for fn in categorical_fields:
        for phrase in CATEGORICAL_PHRASES.get(fn, ()):
            if phrase in t:
                return True
    return False

def image_present(source_refs) -> bool:
    return any(str(r).startswith("#/pictures/") for r in (source_refs or []))
```

> Designator safety (Acceptance #2): `has_unit_token` uses `unit_token_regex`'s leading/trailing word-char guards, so `"m"` will not match inside `S-75M`/`5Ya23`. Confirm with the designator test; if any short unit leaks, drop the bare single-letter form for that dimension and keep the spelled-out form.

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git add app/services/extraction_signal_detectors.py tests/unit/test_extraction_signal_detectors.py && git commit -m "feat(selection): pass-specific measurement/categorical/image signal detectors"`

---

## Task 2: `absolute_union` selection mode

**Goal:** Add `absolute_union` to `select_candidates` + `cosine_tau`/Literal to `RetrievalProfile`. Selects each candidate iff any of the four signals fires; returns the subset (possibly empty).

**Files:**
- Modify: `app/services/ontology_bundles.py` (`RetrievalProfile`)
- Modify: `app/services/extraction_candidate_scoring.py` (`select_candidates` ~724)
- Test: `tests/unit/test_absolute_union_selection.py`

**Acceptance Criteria:**
- [ ] `RetrievalProfile.selection_mode` accepts `"absolute_union"`; `cosine_tau: float = 0.55` exists (per-pass override allowed).
- [ ] In `absolute_union`, a candidate with a pass-measurement is kept; one with neither signal nor `max_field_cosine ≥ cosine_tau` is dropped.
- [ ] When NO candidate fires any signal, `select_candidates` returns `[]` (empty) — no `k_min` floor.
- [ ] `diag_out` records `selection_mode="absolute_union"`, `selection_k`, and per-signal keep counts (`measurement_keeps`, `categorical_keeps`, `image_keeps`, `cosine_keeps`).
- [ ] `topk`/`guarded_quantile` paths unchanged (regression).

**Verify:** `pytest tests/unit/test_absolute_union_selection.py tests/unit/test_dispatcher_vr_wiring.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_absolute_union_selection.py
from types import SimpleNamespace
from app.services.extraction_candidate_scoring import select_candidates, MergedCandidate

def _mc(idx, text, refs, cos):
    return MergedCandidate(candidate_key=f"r:chunk_{idx}", chunk_index=idx, self_ref=f"chunk_{idx}",
        chunk_text=text, source_refs=refs, token_count=10, page_number=1, vector_score=0.0,
        field_scores={}, alias_hits=0, pattern_hits=0, negative_hits=0, section_hits=0,
        content_type=None, retrieval_sources=set(), supported_field_hints=set(), max_field_cosine=cos)

def _cfg(tau=0.55):
    return SimpleNamespace(selection_mode="absolute_union", cosine_tau=tau, top_k=15,
        signal_dimensions={"length","angle"}, signal_categorical=set(), signal_has_image=False)

def test_keeps_measurement_and_cosine_drops_rest():
    cands = [(_mc(0,"max range 2500 km",[], 0.1), 0.9),    # measurement
             (_mc(1,"the colonel said hello",[], 0.9), 0.4), # cosine>=tau
             (_mc(2,"unrelated prose",[], 0.1), 0.3)]         # nothing
    diag = {}
    out = select_candidates(cands, [{}]*3, _cfg(), diag_out=diag)
    kept = {c.chunk_index for c,_ in out}
    assert kept == {0, 1}
    assert diag["selection_k"] == 2

def test_empty_when_nothing_fires():
    cands = [(_mc(0,"prose",[], 0.1), 0.3), (_mc(1,"more prose",[], 0.2), 0.2)]
    out = select_candidates(cands, [{}]*2, _cfg(), diag_out={})
    assert out == []  # no k_min floor — genuinely 0
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3a: Add config to `RetrievalProfile`** (`ontology_bundles.py`): extend the `selection_mode` Literal to include `"absolute_union"`, and add `cosine_tau: float = Field(default=0.55)`. The per-pass `signal_dimensions`/`signal_categorical`/`signal_has_image` are injected by the endpoint from Task 0's config (see Task 3); in `select_candidates` read them via `getattr(cfg, "signal_dimensions", set())` so non-absolute modes are unaffected.

- [ ] **Step 3b: Add the branch in `select_candidates`** (after the `topk` early-return, before/alongside the `guarded_quantile` branch):

```python
    if cfg.selection_mode == "absolute_union":
        from app.services.extraction_signal_detectors import (
            measurement_present, categorical_present, image_present)
        dims = getattr(cfg, "signal_dimensions", set()) or set()
        cats = getattr(cfg, "signal_categorical", set()) or set()
        has_img = bool(getattr(cfg, "signal_has_image", False))
        tau = float(getattr(cfg, "cosine_tau", 0.55))
        out, mk, ck, ik, kk = [], 0, 0, 0, 0
        for mc, score in c5_scored:
            m = measurement_present(dims, mc.chunk_text)
            c = categorical_present(cats, mc.chunk_text)
            i = has_img and image_present(mc.source_refs)
            k = float(getattr(mc, "max_field_cosine", 0.0) or 0.0) >= tau
            if m or c or i or k:
                out.append((mc, score))
                mk += m; ck += c; ik += i; kk += k
        if diag_out is not None:
            diag_out["selection_mode"] = "absolute_union"
            diag_out["selection_k"] = len(out)
            diag_out["measurement_keeps"] = mk
            diag_out["categorical_keeps"] = ck
            diag_out["image_keeps"] = ik
            diag_out["cosine_keeps"] = kk
        return out
```

- [ ] **Step 4: Run → PASS** (and `test_dispatcher_vr_wiring.py` still green — no regression to topk/guarded paths).
- [ ] **Step 5: Commit** `git add app/services/ontology_bundles.py app/services/extraction_candidate_scoring.py tests/unit/test_absolute_union_selection.py && git commit -m "feat(selection): absolute_union selection mode"`

---

## Task 3: Endpoint — inject per-pass config + emit `empty_selection`

**Goal:** The chunk-scope endpoint injects the pass's signal config into the profile, and when `absolute_union` selects 0, returns a NEW `mode="empty_selection"` (distinct from `full`/`would_skip`/`selected_refs`), instead of falling open to `mode="full"`.

**Files:**
- Modify: `app/api/v1/extraction_routing.py` (the `select_candidates` call site; the `mode="full"` empty paths ~710, ~1279; `ChunkScopeResponse`/`ChunkScopeDiagnostics`)
- Test: `tests/unit/test_empty_selection_contract.py` (endpoint half)

**Acceptance Criteria:**
- [ ] For an `absolute_union` pass, the endpoint sets `profile.signal_dimensions/signal_categorical/signal_has_image` from `derive_pass_signal_config(bundle_key)[pass_name]` before calling `select_candidates`.
- [ ] When `absolute_union` returns 0 candidates, the response is `ChunkScopeResponse(mode="empty_selection", self_refs=[], diagnostics=...)` — NOT `mode="full"`.
- [ ] The selected (non-empty) case still returns `mode="selected_refs"` with `self_refs` populated as today.
- [ ] `topk`/`guarded_quantile` empty handling is unchanged (still `mode="full"`/`would_skip`).

**Verify:** `pytest tests/unit/test_empty_selection_contract.py -k endpoint -v` → pass.

**Steps:**

- [ ] **Step 1: Write failing test** — call the selection-assembly path with an `absolute_union` profile and candidates where none fire; assert response `mode == "empty_selection"`. (Mirror existing endpoint unit-test setup in the repo's `tests/unit` for the chunk-scope path; mock retrieval to return a fixed candidate pool.)

```python
# tests/unit/test_empty_selection_contract.py  (endpoint half)
def test_absolute_union_zero_selection_returns_empty_mode(absolute_union_pool_no_signal):
    resp = run_chunk_scope(absolute_union_pool_no_signal)   # helper builds the endpoint call
    assert resp.mode == "empty_selection"
    assert resp.self_refs == []
    assert resp.diagnostics.selection_mode == "absolute_union"
```

- [ ] **Step 2: Run → FAIL** (endpoint returns `mode="full"`).

- [ ] **Step 3a: Inject config before `select_candidates`** at the endpoint call site:

```python
from app.services.extraction_pass_signal_config import derive_pass_signal_config
if profile.selection_mode == "absolute_union":
    sc = derive_pass_signal_config(bundle_key).get(pass_name)
    if sc:
        profile = profile.model_copy(update={
            "signal_dimensions": sc.dimensions,
            "signal_categorical": sc.categorical_fields,
            "signal_has_image": sc.has_image_field})
```

- [ ] **Step 3b: Branch the empty result** — where the selected list is turned into a response, add: if `profile.selection_mode == "absolute_union" and not selected`, return `ChunkScopeResponse(mode="empty_selection", self_refs=[], diagnostics=ChunkScopeDiagnostics(mode="empty_selection", selection_mode="absolute_union", fallback_reason="absolute_union_no_signal", candidate_count=len(pool), ...))`. Add `"empty_selection"` to the `mode` Literal on `ChunkScopeResponse`/`ChunkScopeDiagnostics`. Do **not** route it through the existing `mode="full"` paths (~710, ~1279) — those remain for the topk/guarded fall-opens.

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git add app/api/v1/extraction_routing.py tests/unit/test_empty_selection_contract.py && git commit -m "feat(selection): endpoint emits empty_selection mode for absolute_union zero-select"`

---

## Task 4: Worker — map `empty_selection` to "extract nothing"

**Goal:** `_compute_effective_chunk_scope` maps `mode="empty_selection"` to a sentinel scope meaning "run this pass on zero chunks", NOT `None` (which means full-doc).

**Files:**
- Modify: `app/workers/pipeline.py` (`_compute_effective_chunk_scope` ~8543)
- Test: `tests/unit/test_dispatcher_vr_wiring.py` (add cases)

**Acceptance Criteria:**
- [ ] `mode="empty_selection"` → `effective_chunk_scope == {"mode": "empty_selection", "self_refs": []}` (a distinct sentinel), with `diag["empty_selection"] is True`.
- [ ] It is NOT mapped to `None` (full-doc) and NOT to `selected_refs`.
- [ ] Existing `narrow_only`/`shadow`/`degraded`/size-gate cases unchanged.

**Verify:** `pytest tests/unit/test_dispatcher_vr_wiring.py -k empty_selection -v` → pass.

**Steps:**

- [ ] **Step 1: Write failing test**

```python
def test_narrow_only_empty_selection_maps_to_zero_scope(self):
    resp = self._make_response("empty_selection", self_refs=[], diag_extra={"selection_mode":"absolute_union"})
    eff, diag = _compute_effective_chunk_scope(resp, "narrow_only")
    assert eff == {"mode": "empty_selection", "self_refs": []}
    assert diag.get("empty_selection") is True
```

- [ ] **Step 2: Run → FAIL** (current code returns `None` for non-`selected_refs` modes → full-doc).

- [ ] **Step 3: Implement** — in the `narrow_only` branch, before the `resp_mode == "selected_refs"` handling:

```python
        if resp_mode == "empty_selection":
            effective_chunk_scope = {"mode": "empty_selection", "self_refs": []}
            diag["empty_selection"] = True
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git add app/workers/pipeline.py tests/unit/test_dispatcher_vr_wiring.py && git commit -m "feat(selection): worker maps empty_selection to zero-chunk scope (not full-doc)"`

---

## Task 5: Finalization — `empty_selection` pass → ZERO_YIELD / COMPLETE

**Goal:** A pass dispatched with the `empty_selection` scope extracts nothing and terminates as ZERO_YIELD → COMPLETE/EMPTY (never full-doc, never FAILED).

**Files:**
- Modify: `app/workers/pipeline.py` (extract-pass dispatch / `_call_extract_pass` ~4480; the scope→request mapping)
- Test: `tests/unit/test_empty_selection_contract.py` (worker half)

**Acceptance Criteria:**
- [ ] When `effective_chunk_scope["mode"] == "empty_selection"`, the worker does NOT call docling-graph extraction for that pass; it records the pass as ZERO_YIELD (`execution_status` COMPLETE, `primary_entities_extracted=0`) with a diagnostic `zero_yield_reason="empty_selection"`.
- [ ] The run does not escalate to FAILED or PARTIAL because of an `empty_selection` pass.
- [ ] (Reuse, don't duplicate, the existing ZERO_YIELD path that `_is_clean_empty_pipeline_error` already feeds.)

**Verify:** `pytest tests/unit/test_empty_selection_contract.py -k worker -v` → pass.

**Steps:**

- [ ] **Step 1: Write failing test** — dispatch a pass with `effective_chunk_scope={"mode":"empty_selection","self_refs":[]}`; assert the docling-graph HTTP call is NOT made and the pass result is `execution_status="COMPLETE"`, `primary_entities_extracted=0`, `zero_yield_reason="empty_selection"`.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — at the point where `effective_chunk_scope` is turned into the extract-pass request (just before the `_call_extract_pass` HTTP call), short-circuit:

```python
        if isinstance(effective_chunk_scope, dict) and effective_chunk_scope.get("mode") == "empty_selection":
            logger.info("EXTRACT_PASS_EMPTY_SELECTION pass=%s document_id=%s — absolute_union "
                        "selected 0 chunks; recording ZERO_YIELD (COMPLETE/EMPTY), no extraction call.",
                        pass_name, document_id)
            return _zero_yield_payload(pass_name, reason="empty_selection")  # COMPLETE/EMPTY stub
```

> `_zero_yield_payload` should produce the same COMPLETE/EMPTY shape the existing clean-empty path returns (node_count=0, edge_count=0, `diagnostics.zero_yield_reason="empty_selection"`), so downstream gating treats it identically to an off-domain clean-empty. Extract a shared helper if one does not already exist around the `_is_clean_empty_pipeline_error` return.

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git add app/workers/pipeline.py tests/unit/test_empty_selection_contract.py && git commit -m "feat(selection): empty_selection pass finalizes as ZERO_YIELD/COMPLETE"`

---

## Task 6: Manifests — opt the 9 routable passes into `absolute_union`

**Goal:** Flip the 9 routable passes in the production bundle (and the two narrowed siblings) to `selection_mode: absolute_union` + `cosine_tau: 0.55`. `narrow_min_doc_tokens` stays active.

**Files:**
- Modify: `ontology_bundles/air_defense_v3/manifest.yaml`
- Modify: `ontology_bundles/air_defense_v3_narrowing_v1/manifest.yaml`
- Modify: `ontology_bundles/air_defense_v3_merged_v1/manifest.yaml`

**Acceptance Criteria:**
- [ ] Each of the 9 routable passes (radar_power_rf, radar_antenna, radar_timing, radar_modulation, missile_kinematics, missile_guidance, missile_airframe, missile_speed_timing, missile_propulsion) has `selection_mode: absolute_union` and `cosine_tau: 0.55`.
- [ ] No `quantile_q`/`ranker_weights`/`k_min` required for these passes (left inert or removed).
- [ ] `air_defense_v3` edited first, then the two siblings mirror it (per the schema-changes-hit-production rule).
- [ ] Bundle loads without validation error: `python -c "from app.services.ontology_bundles import load_bundle; load_bundle('air_defense_v3')"` → no error.

**Verify:** `python -c "from app.services.ontology_bundles import load_bundle; load_bundle('air_defense_v3')"` → exit 0; `grep -c 'selection_mode: absolute_union' ontology_bundles/air_defense_v3/manifest.yaml` → 9.

**Steps:**

- [ ] **Step 1:** In `air_defense_v3/manifest.yaml`, for each of the 9 routable passes, set:
```yaml
      selection_mode: absolute_union
      cosine_tau: 0.55
```
(replacing the `guarded_quantile` + `quantile_q`/`ranker_weights` block for those passes).
- [ ] **Step 2:** Mirror the same edits into `air_defense_v3_narrowing_v1/manifest.yaml` and `air_defense_v3_merged_v1/manifest.yaml`.
- [ ] **Step 3:** Run the load check (Verify) → exit 0 and grep count 9.
- [ ] **Step 4: Commit** `git add ontology_bundles/air_defense_v3*/manifest.yaml && git commit -m "feat(selection): opt routable passes into absolute_union (cosine_tau=0.55)"`

---

## Task 7: Online validation (USER-ORDERED GATE)

**Goal:** Prove, end-to-end on real extraction, that `absolute_union` holds entity-recall vs. the full-doc baseline on the dense docs before any production default flip or removal of `narrow_min_doc_tokens`.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- (No source files — operational validation. Uses the existing reingest/driver harness in `/home/josh/.guardrank_eval_state/`.)

**Acceptance Criteria:**
- [ ] Deploy the change live (docling-graph rebuilt if needed; workers force-recreated; verify `selection_mode=absolute_union` live in-container).
- [ ] Run **NMUSAF**, **SA-2/SR-71 PDF**, and **Engagement** end-to-end under `absolute_union` (reingest graph_only) AND a full-doc (shadow) baseline run of each, on the same healthy/idle LLM pool.
- [ ] For each doc, **narrowed entity count ≥ 0.9 × full-doc baseline** (recall held), and each run terminates `COMPLETE` (off-domain passes show ZERO_YIELD, not FAILED).
- [ ] Capture the per-doc full-doc-vs-absolute_union entity table to the verdict file; report it.

**Verify:** `cat /home/josh/.guardrank_eval_state/absolute_union_validation_verdict.txt` shows, per doc, `absolute_union_entities ≥ 0.9 × fulldoc_entities` AND `status=COMPLETE` for all three docs.

**Steps:**

- [ ] **Step 1:** Deploy (rebuild docling-graph only if its image changed; `docker compose -p eip-mmdpp ... up -d --force-recreate --no-deps worker worker-graph`; verify live config in-container).
- [ ] **Step 2:** Session-proof driver: for each of {NMUSAF, SA-2/SR-71 PDF, Engagement}, run a full-doc (shadow) baseline + an absolute_union run via reingest graph_only; record entity counts + terminal status to the verdict file.
- [ ] **Step 3:** Compare per doc; PASS iff all three hold ≥0.9× recall and COMPLETE.
- [ ] **Step 4 (gate):** If PASS → the design is validated; a FOLLOW-UP (not this plan) may then remove `narrow_min_doc_tokens`. If FAIL → hold; do not flip any production default; tune `cosine_tau` / signals and re-validate.

---

## Self-Review

**Spec coverage:** §3 four signals → Tasks 1+2. §3.1 bounded matcher (Finding 4) → Task 1 (reuses `has_unit_token`; designator test). §3.5 empty-selection contract (Finding 1) → Tasks 3+4+5 (endpoint mode + worker map + ZERO_YIELD). §4 routing/per-pass config → Task 0 + Task 6. §6 `select_candidates` uses MergedCandidate directly (Finding 3) → Task 2. §6 retain `narrow_min_doc_tokens` (Finding 2) → not dropped; Task 7 gates eventual removal. §7.1 τ default 0.55 → Task 2/6. §7.2 schema-derived vocab → Task 0. §7.3 0→ZERO_YIELD → Task 5. §7.4 online validation → Task 7. All covered.

**Placeholder scan:** code shown in every code step; the one external dependency (`iter_routable_pass_fields`) is called out with an explicit "add a thin helper if absent" note, not left as TBD.

**Type consistency:** `PassSignalConfig.{dimensions,categorical_fields,has_image_field}` used identically in Tasks 0/2/3; signal functions `measurement_present(set,str)`, `categorical_present(set,str)`, `image_present(list)` consistent across Tasks 1/2; `mode="empty_selection"` sentinel consistent across Tasks 3/4/5; `cosine_tau` consistent across Tasks 2/6.
