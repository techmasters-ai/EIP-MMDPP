# Remove Runtime Ontology→Template Generation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace runtime Pydantic template generation in the extraction hot path with hand-authored fixed schemas under `ontology_bundles/air_defense_v3/`, threading bundle selection through Source/PipelineRun/DocumentGraphExtraction, and delivering the refactor in three PRs (scaffolding → switchover behind flag → deletion after soak).

**Architecture:** The spec at `docs/superpowers/specs/2026-04-10-remove-runtime-template-generation-design.md` (committed at `6d14eaf` on `feature/arcadedb`) is the single source of truth. This plan is an execution ordering; it does NOT restate design decisions. Every task either cites a specific spec section for detail or links to an existing file in the repo. The plan is organized around the spec's §7 rollout sequence: pre-branch stabilization on `feature/arcadedb`, then PR 1 (scaffolding, strictly additive), PR 2 (switchover behind a feature flag), PR 3 (deletion after 7-day soak).

**Tech Stack:** Python 3.12, SQLAlchemy 2.x + Alembic, FastAPI, Celery, Pydantic v2, pytest, Docker Compose. Graph: ArcadeDB (via the `GraphStore` protocol + `ArcadeDBGraphStore` implementation). Extraction: Docling-Graph library running in a sidecar service, invoked over HTTP from the worker. Test framework: pytest with fixtures in `tests/conftest.py`.

**Spec reference:** Throughout this plan, section references like `§4.4` or `§7.3` refer to sections in the committed spec. Open it in a second editor window and refer to it often.

---

## Chunk 0: Pre-work — Stabilization on `feature/arcadedb`

Before any refactor work, the 4 uncommitted files on `feature/arcadedb` must be committed as two stabilization commits, then `feature/extraction-refactor` branched off the stabilized `feature/arcadedb`. See spec §7.1.

### Task 0.1: Verify starting state

**Files:** none (inspection only)

- [x] **Step 1: Check current branch and uncommitted files**

Run:
```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/arcadedb
git status --short
git log --oneline -5
```

Expected: on `feature/arcadedb`, with 4 modified files:
- `app/services/arcadedb_client.py`
- `app/services/arcadedb_graph.py`
- `app/workers/pipeline.py`
- `docker/docling-graph/app/template_builder.py`

You may also see an untracked `?? "ArcadeDB Manual.pdf"` entry. That's fine; leave it untracked. Latest commit should be `6d14eaf` (the twelfth-round spec fix) or more recent. If anything ELSE is uncommitted or untracked that you don't recognize, stop and investigate before proceeding.

- [x] **Step 2: Skim the uncommitted diffs**

Run:
```bash
git diff app/services/arcadedb_client.py app/services/arcadedb_graph.py
git diff app/workers/pipeline.py docker/docling-graph/app/template_builder.py
```

The first two are durable ArcadeDB fixes: event-loop-aware httpx client + Cypher label predicates. The second two are legacy extraction-path stabilizers: None-safe confidence + fallback `id_fields` Optional. Both sets are described in spec §7.1.

### Task 0.2: Commit A — durable ArcadeDB fixes

**Files:** `app/services/arcadedb_client.py`, `app/services/arcadedb_graph.py`

- [x] **Step 1: Stage the two ArcadeDB files only**

Run:
```bash
git add app/services/arcadedb_client.py app/services/arcadedb_graph.py
git status --short
```

Expected: only those two files staged, the pipeline.py and template_builder.py files still unstaged.

- [x] **Step 2: Commit A**

Run:
```bash
git commit -m "$(cat <<'EOF'
fix(arcadedb): event-loop-aware httpx client + cypher label predicates

- ArcadeDBClient: rebuild httpx client when the asyncio event loop
  changes. Celery tasks using asyncio.run() create a fresh loop per
  invocation, leaving the old client's transport bound to a closed
  loop and raising "Event loop is closed" on reuse.
- ArcadeDBGraphStore.run_community_algorithm: rewrite the Cypher
  WHERE clause to use label negation (NOT node:Foo) instead of
  @class, which is SQL-only syntax and invalid in Cypher. Use
  id(node) and labels(node)[0] for projection.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

Expected: new commit on `feature/arcadedb` ahead of `6d14eaf` (or whatever the prior head was).

### Task 0.3: Commit B — legacy extraction-path stabilizers

**Files:** `app/workers/pipeline.py`, `docker/docling-graph/app/template_builder.py`

- [x] **Step 1: Stage and commit B**

Run:
```bash
git add app/workers/pipeline.py docker/docling-graph/app/template_builder.py
git status --short
git commit -m "$(cat <<'EOF'
fix(extraction): None-safe confidence + fallback id_field Optional

- pipeline.py: defensive confidence defaulting in the node/edge
  ingest filter. node.get("confidence", 0.8) returns None when the
  key exists with a None value, and None < node_min_conf raises
  TypeError. Use explicit None check.
- template_builder.py: fallback id_field (first-property heuristic)
  now produces Optional Pydantic fields, so extractions don't fail
  Pydantic validation when the LLM omits an arbitrarily-chosen
  identifier. This is a band-aid on the code path being deleted in
  PR 3; it prevents bleed-through failures in the legacy path
  during the PR 1/PR 2 soak window.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -4
git status --short
```

Expected: working tree clean, two new commits on `feature/arcadedb`.

### Task 0.4: Cut `feature/extraction-refactor` branch

**Files:** none (git operation)

- [x] **Step 1: Branch from stabilized feature/arcadedb**

Run:
```bash
git checkout -b feature/extraction-refactor
git branch --show-current
git log --oneline -5
```

Expected: on `feature/extraction-refactor`, with the two stabilization commits and the spec commit visible in the log. All subsequent tasks in this plan run on this branch.

- [x] **Step 2: Create TaskCreate tracking entries for the remaining chunks**

This is a planning checkpoint. Tasks for PR 1 will be created at the start of Chunk 1, PR 2 at Chunk 2, PR 3 at Chunk 3.

---

**END OF CHUNK 0.** Proceed to Chunk 1 (PR 1: Baseline + Scaffolding).

## Chunk 1: PR 1 — Baseline + Scaffolding (strictly additive)

**Invariant for every task in this chunk:** the legacy extraction path must still run production documents end-to-end. The new path exists, loads cleanly, passes unit tests, and is testable directly — but no caller has switched to it. Spec §7.3.

### Task 1.1: Add identity_fields/identity_scope to ontology.yaml

**Files:**
- Modify: `ontology/ontology.yaml`

Spec §3.6 + §4.3 identity scope rules.

- [x] **Step 1: Read the current ontology.yaml to understand its shape**

Run:
```bash
wc -l ontology/ontology.yaml
head -40 ontology/ontology.yaml
grep -n "^  - name:" ontology/ontology.yaml | head -50
```

Confirm: top-level entries under `entity_types:`, each with `name`, `label`, `parent`, `properties`.

- [x] **Step 2: For each of the 23 extract-bucket entity types, add identity_fields and identity_scope**

**Authoritative values for this PR.** Spec §3.6 enumerates the rules and shows seven examples; the table below is the authoritative set for all 23 extract-bucket entity types and must be used as-is. Every `identity_fields` entry below is verified against the actual property list in `ontology/ontology.yaml` as of commit `6d14eaf` — do NOT re-derive. Scope choices follow spec §3.6: `global` for real-world-shared equipment identified by a stable system designation, `document` for document-local anchors and subordinate components that belong to a specific document's context.

| Entity | identity_fields | identity_scope | Notes |
|---|---|---|---|
| SECTION | `[heading, page_start]` | `document` | |
| FIGURE | `[figure_id, page]` | `document` | |
| TABLE | `[table_id, page]` | `document` | |
| ASSERTION | `[assertion_text]` | `document` | |
| SPECIFICATION | `[parameter, value]` | `document` | |
| PLATFORM | `[name]` | `global` | bridge |
| RADAR_SYSTEM | `[system_name]` | `global` | |
| ANTENNA | `[name]` | `document` | |
| RECEIVER | `[name]` | `document` | |
| TRANSMITTER | `[name]` | `document` | |
| SIGNAL_PROCESSING_CHAIN | `[name]` | `document` | |
| FREQUENCY_BAND | `[band_name]` | `global` | |
| WAVEFORM | `[waveform_name]` | `document` | |
| MISSILE_SYSTEM | `[system_name]` | `global` | |
| LAUNCHER_SYSTEM | `[system_name]` | `global` | |
| GUIDANCE_METHOD | `[guidance_type]` | `global` | |
| SEEKER | `[seeker_nomenclature]` | `document` | |
| PROPULSION_STACK | `[]` | `document` | content-hash fallback — no natural identifier; only `total_burntime_s` exists as a property |
| AIR_DEFENSE_ARTILLERY_SYSTEM | `[system_name]` | `global` | |
| ELECTRONIC_WARFARE_SYSTEM | `[system_name]` | `global` | |
| FIRE_CONTROL_SYSTEM | `[system_name]` | `global` | |
| WEAPON_SYSTEM | `[system_name]` | `global` | |
| INTEGRATED_AIR_DEFENSE_SYSTEM | `[name]` | `global` | |

Edit each entity in-place. Example:

```yaml
  - name: RADAR_SYSTEM
    label: "Radar System"
    identity_fields: [system_name]   # ADD
    identity_scope: global            # ADD
    parent: MilitarySystem
    properties:
      ...
```

- [x] **Step 3: Validate YAML syntax**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('ontology/ontology.yaml'))" && echo "YAML OK"
```

Expected: `YAML OK`.

- [x] **Step 4: Commit**

```bash
git add ontology/ontology.yaml
git commit -m "feat(ontology): add identity_fields and identity_scope to extract-bucket entities

Part of PR 1 scaffolding. Each extract-bucket entity type now declares
its identity_fields and identity_scope explicitly per spec §3.6. This
is a prerequisite for the hand-authored extraction schemas in
ontology_bundles/air_defense_v3/.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 1.2: Create bundle directory skeleton and move ontology.yaml via symlink

**Files:**
- Create: `ontology_bundles/__init__.py`
- Create: `ontology_bundles/air_defense_v3/__init__.py`
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/__init__.py`
- Move: `ontology/ontology.yaml` → `ontology_bundles/air_defense_v3/ontology.yaml`
- Create (symlink): `ontology/ontology.yaml` → `../ontology_bundles/air_defense_v3/ontology.yaml`

Spec §2 directory layout + §7.3 symlink compatibility.

- [x] **Step 1: Create directories and __init__.py files**

Run:
```bash
mkdir -p ontology_bundles/air_defense_v3/extraction_schemas
touch ontology_bundles/__init__.py
touch ontology_bundles/air_defense_v3/__init__.py
touch ontology_bundles/air_defense_v3/extraction_schemas/__init__.py
ls -la ontology_bundles/air_defense_v3/
```

Expected: empty `__init__.py` files present in the three directories.

- [x] **Step 2: Move ontology.yaml into the bundle**

Run:
```bash
git mv ontology/ontology.yaml ontology_bundles/air_defense_v3/ontology.yaml
ls -la ontology/ ontology_bundles/air_defense_v3/
```

Expected: `ontology/` directory may still exist (contains other files); `ontology.yaml` is now only in the bundle.

- [x] **Step 3: Create symlink for legacy compatibility**

Run:
```bash
cd ontology
ln -s ../ontology_bundles/air_defense_v3/ontology.yaml ontology.yaml
cd ..
ls -la ontology/ontology.yaml
cat ontology/ontology.yaml | head -20
```

Expected: `ls -la` shows a symlink arrow pointing to `../ontology_bundles/air_defense_v3/ontology.yaml`; `cat` resolves through the symlink and dumps the same content as the bundle file.

- [x] **Step 4: Verify the symlink is tracked by git as a symlink**

Run:
```bash
git add ontology/ontology.yaml
git ls-files -s ontology/ontology.yaml
```

Expected: mode `120000` (symlink) in the git index. If it shows `100644` (regular file), `core.symlinks` is off for this clone. Fix via:

```bash
git rm --cached ontology/ontology.yaml
git config core.symlinks true
SHA=$(printf '../ontology_bundles/air_defense_v3/ontology.yaml' | git hash-object -w --stdin)
git update-index --add --cacheinfo 120000,$SHA,ontology/ontology.yaml
git ls-files -s ontology/ontology.yaml   # should now show 120000
```

This is Linux-only-friendly; the deploy target is Linux so the mode should stick. If the clone is on Windows, the symlink will need to be recreated in each checkout.

- [x] **Step 5: Commit**

```bash
git add ontology_bundles/ ontology/ontology.yaml
git commit -m "feat(bundle): scaffold ontology_bundles/air_defense_v3 with symlink compat

Creates the ontology_bundles/ directory tree, moves ontology.yaml into
air_defense_v3/, and adds a git-tracked symlink at the old location
(ontology/ontology.yaml) so the legacy docling-graph bind mount keeps
working unchanged. See spec §2 directory layout and §7.3 symlink
compatibility. PR 3 removes the symlink and the mount together.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 1.3: Write manifest.yaml and coverage.yaml

**Files:**
- Create: `ontology_bundles/air_defense_v3/manifest.yaml`
- Create: `ontology_bundles/air_defense_v3/coverage.yaml`

Spec §2 manifest schema + coverage schema. Copy the full content from the spec's §2 code blocks.

- [x] **Step 1: Write manifest.yaml**

Create `ontology_bundles/air_defense_v3/manifest.yaml` with the full content from spec §2 "manifest.yaml schema" code block. All 5 passes (reference, radar_domain, missile_domain, other_systems, system_links) with their `name`, `required`, `kind`, `input_mode`, `module`, `template_class`, `primary_entity_types`, `bridge_entity_types`, `extracted_relationship_types`, `depends_on`, and `skip_if_no_upstream_endpoints` (where applicable). Top-level keys: `bundle_key`, `manifest_schema_version`, `ontology_name`, `ontology_version`, `extraction_profile_version`.

- [x] **Step 2: Validate manifest.yaml is parseable**

Run:
```bash
python -c "import yaml; m = yaml.safe_load(open('ontology_bundles/air_defense_v3/manifest.yaml')); print(len(m['passes']), 'passes')"
```

Expected: `5 passes`.

- [x] **Step 3: Write coverage.yaml**

Create `ontology_bundles/air_defense_v3/coverage.yaml` with the full content from spec §2 "coverage.yaml schema" code block. `entity_types.extract` has 23 entries; `entity_types.derive` is `[]`; `relationship_types.extract` has 14 entries; `relationship_types.derive` has `HAS_PROVENANCE`, `MENTIONED_IN`, `CONTAINS_TEXT`, `CONTAINS_IMAGE`.

- [x] **Step 4: Validate coverage.yaml**

Run:
```bash
python -c "import yaml; c = yaml.safe_load(open('ontology_bundles/air_defense_v3/coverage.yaml')); print(len(c['entity_types']['extract']), 'extract entities')"
```

Expected: `23 extract entities`.

- [x] **Step 5: Commit**

```bash
git add ontology_bundles/air_defense_v3/manifest.yaml ontology_bundles/air_defense_v3/coverage.yaml
git commit -m "feat(bundle): add manifest.yaml and coverage.yaml for air_defense_v3

5 passes declared in manifest.yaml per spec §2. coverage.yaml
enumerates the 23 extract-bucket entity types and 14 extracted
relationship types. Derive bucket covers structural edges
(HAS_PROVENANCE, MENTIONED_IN, CONTAINS_TEXT, CONTAINS_IMAGE).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 1.4: Write shared validators.py

**Files:**
- Create: `ontology_bundles/air_defense_v3/validators.py`
- Create: `tests/unit/test_bundle_validators.py` (new test file)

Spec §3.2 shared validators.

- [x] **Step 1: Write the failing tests first**

Create `tests/unit/test_bundle_validators.py`:

```python
"""Tests for ontology_bundles.air_defense_v3.validators."""
import pytest

from ontology_bundles.air_defense_v3.validators import (
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_confidence,
    normalize_enum,
)


class TestCoerceOptionalInt:
    def test_none_returns_none(self):
        assert coerce_optional_int(None) is None

    def test_int_returns_int(self):
        assert coerce_optional_int(5) == 5

    def test_numeric_string(self):
        assert coerce_optional_int("5") == 5

    def test_negative_string(self):
        assert coerce_optional_int("-12") == -12

    def test_empty_string_returns_none(self):
        assert coerce_optional_int("") is None

    def test_whitespace_string_returns_none(self):
        assert coerce_optional_int("   ") is None

    def test_embedded_number(self):
        assert coerce_optional_int("page 5 of 10") == 5

    def test_unparseable_returns_none(self):
        assert coerce_optional_int("unknown") is None


class TestCoerceOptionalFloat:
    def test_none_returns_none(self):
        assert coerce_optional_float(None) is None

    def test_float_returns_float(self):
        assert coerce_optional_float(3.14) == 3.14

    def test_int_coerces_to_float(self):
        assert coerce_optional_float(5) == 5.0

    def test_decimal_string(self):
        assert coerce_optional_float("3.14") == 3.14

    def test_unparseable_returns_none(self):
        assert coerce_optional_float("abc") is None


class TestCoerceOptionalConfidence:
    def test_none_returns_none(self):
        assert coerce_optional_confidence(None) is None

    def test_valid_float(self):
        assert coerce_optional_confidence(0.75) == 0.75

    def test_percentage_over_one(self):
        assert coerce_optional_confidence(85) == 0.85

    def test_text_high(self):
        assert coerce_optional_confidence("high") == 0.9

    def test_text_medium(self):
        assert coerce_optional_confidence("medium") == 0.6

    def test_text_low(self):
        assert coerce_optional_confidence("low") == 0.3

    def test_unparseable_returns_none(self):
        assert coerce_optional_confidence("vague") is None

    def test_explicit_zero_preserved(self):
        # Regression: the 'or 0.8' bug would have defaulted this.
        assert coerce_optional_confidence(0.0) == 0.0


class TestNormalizeEnum:
    def test_exact_match(self):
        validator = normalize_enum({"RADAR", "SONAR"})
        assert validator("RADAR") == "RADAR"

    def test_case_insensitive(self):
        validator = normalize_enum({"RADAR"})
        assert validator("radar") == "RADAR"

    def test_space_to_underscore(self):
        validator = normalize_enum({"FOO_BAR"})
        assert validator("foo bar") == "FOO_BAR"

    def test_unknown_returns_none(self):
        validator = normalize_enum({"RADAR"})
        assert validator("UNKNOWN") is None

    def test_none_returns_none(self):
        validator = normalize_enum({"RADAR"})
        assert validator(None) is None
```

- [x] **Step 2: Run tests — expect ImportError**

Run:
```bash
pytest tests/unit/test_bundle_validators.py -v 2>&1 | tail -20
```

Expected: collection error / import error because `validators.py` doesn't exist yet.

- [x] **Step 3: Write validators.py**

Create `ontology_bundles/air_defense_v3/validators.py` with the content from spec §3.2, including `coerce_optional_int`, `coerce_optional_float`, `coerce_optional_confidence`, and `normalize_enum`.

- [x] **Step 4: Run tests — expect pass**

Run:
```bash
pytest tests/unit/test_bundle_validators.py -v
```

Expected: all 22+ test cases pass.

- [x] **Step 5: Commit**

```bash
git add ontology_bundles/air_defense_v3/validators.py tests/unit/test_bundle_validators.py
git commit -m "feat(bundle): add shared validators for extraction schemas

coerce_optional_int/float/confidence handle messy LLM output that
the salvage layer currently papers over. normalize_enum is a factory
for field-validator closures. Includes regression test for the
'confidence or 0.8' bug (explicit 0.0 must be preserved).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 1.5: Write the five extraction schema modules

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/reference.py`
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/radar_domain.py`
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/missile_domain.py`
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/other_systems.py`
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/system_links.py`
- Create: `tests/unit/test_extraction_schemas.py`

Spec §3.3–§3.5.

- [x] **Step 0: Lock the curated property sets per entity class**

The spec fully shows `SectionEntity` (§3.3) and `RadarSystemEntity` + `RadarRelationship` + `RadarDomainPass` top-level (§3.4). For the ~25 other entity classes across the four entity-bearing passes, the plan enumerates the curated property list below. These are **prescriptive for this plan** and every listed property name has been verified against `ontology/ontology.yaml` as of commit `6d14eaf`. Do NOT re-derive during implementation; the coverage checker's rule 8 (extraction ⊆ ontology) will reject any drift.

**Confidence exemption.** Every entity class gets a `confidence: Optional[float] = None` Pydantic field with `coerce_optional_confidence` validator. Of the 23 extract-bucket entities, only `ASSERTION` has a `confidence` property in `ontology.yaml` itself. For the other 22 entities, `confidence` is treated as an **extraction system field** exempt from checker rule 8. The coverage checker (Task 1.11) must explicitly list `confidence` in its `SYSTEM_FIELDS` exemption set alongside the partial-safety test helper at `tests/unit/test_extraction_schemas.py`. Do NOT add `confidence` as a property to each entity in `ontology.yaml`; doing so would pollute the domain contract with an extraction artifact.

**General construction rule.** For each class below, the field order is: (1) all `identity_fields` from Task 1.1 for that entity, (2) the listed curated properties, (3) `confidence: Optional[float] = None`. Every field uses `Optional[T] = None` or `Optional[T] = Field(default=None, ...)`. Int-typed fields get `coerce_optional_int`; float-typed fields get `coerce_optional_float`; `confidence` gets `coerce_optional_confidence`.

**reference.py (4 classes):**

| Class | Fields (in order, excluding confidence) |
|---|---|
| `SectionEntity` | `heading`, `page_start`, `page_end` |
| `FigureEntity` | `figure_id`, `page`, `caption`, `figure_type` |
| `TableEntity` | `table_id`, `page`, `caption` |
| `AssertionEntity` | `assertion_text`, `extraction_method`, `review_status` (confidence is a real ontology property for ASSERTION — still included via the construction rule) |

**radar_domain.py (9 classes — 7 primary + 2 bridges):**

| Class | Fields (in order, excluding confidence) |
|---|---|
| `RadarSystemEntity` | `system_name`, `nomenclature`, `radar_type`, `nominal_frequency`, `tx_peak_power` |
| `AntennaEntity` | `name`, `antenna_type`, `gain_dbi`, `beamwidth_az_deg`, `beamwidth_el_deg` |
| `ReceiverEntity` | `name`, `noise_figure_db`, `minimum_discernible_signal_dbm` |
| `TransmitterEntity` | `name`, `peak_power_at_transmitter_kw`, `duty_cycle` |
| `SPCEntity` (SIGNAL_PROCESSING_CHAIN) | `name`, `matched_filter_detection_loss_db`, `filter_response_type` |
| `FrequencyBandEntity` | `band_name`, `designation`, `freq_min_mhz`, `freq_max_mhz` |
| `WaveformEntity` | `waveform_name`, `waveform_family`, `nominal_pulse_duration_us`, `nominal_PRI_us`, `duty_cycle` |
| `PlatformEntity` (bridge) | `name`, `platform_type`, `service_branch` |
| `SpecificationEntity` (bridge) | `parameter`, `value`, `unit` |

**missile_domain.py (7 classes — 5 primary + 2 bridges):**

| Class | Fields (in order, excluding confidence) |
|---|---|
| `MissileSystemEntity` | `system_name`, `nomenclature`, `guidance_type` |
| `LauncherSystemEntity` | `system_name`, `launcher_type`, `capacity` |
| `GuidanceMethodEntity` | `guidance_type`, `firing_doctrine`, `track_quality` |
| `SeekerEntity` | `seeker_nomenclature`, `seeker_ELNOT`, `seeker_type` |
| `PropulsionStackEntity` | `total_burntime_s` (identity_fields=[]; content-hash fallback per Task 1.1) |
| `PlatformEntity` (bridge) | `name`, `platform_type`, `service_branch` |
| `SpecificationEntity` (bridge) | `parameter`, `value`, `unit` |

**other_systems.py (7 classes — 5 primary + 2 bridges):**

| Class | Fields (in order, excluding confidence) |
|---|---|
| `ADAEntity` (AIR_DEFENSE_ARTILLERY_SYSTEM) | `system_name`, `caliber`, `max_tactical_range`, `maximum_rate_of_fire` |
| `EWSystemEntity` (ELECTRONIC_WARFARE_SYSTEM) | `system_name`, `nomenclature`, `ew_role`, `coverage`, `power_output` |
| `FireControlSystemEntity` | `system_name`, `nomenclature` |
| `WeaponSystemEntity` | `system_name`, `nomenclature`, `weapon_type` |
| `IADSEntity` (INTEGRATED_AIR_DEFENSE_SYSTEM) | `name`, `status`, `doctrine` |
| `PlatformEntity` (bridge) | `name`, `platform_type`, `service_branch` |
| `SpecificationEntity` (bridge) | `parameter`, `value`, `unit` |

**Bridge consistency invariant.** `PlatformEntity` and `SpecificationEntity` appear identically in radar_domain, missile_domain, and other_systems — same field lists, same identity. This is required by checker rule 13 (bridge scope consistency) and enforced in Chunk 2's coverage checker.

**system_links.py (no entity classes):**

Only `SystemLinkRelationship` and `SystemLinksPass`. See Step 7 for the relationship class shape.

**Relationship class fields** (same-pass relationships in radar/missile/other_systems):

| Field | Type | Notes |
|---|---|---|
| `rel_type` | `Optional[str]` | `normalize_enum(...)` with the pass's allowed rel_types set |
| `from_type` | `Optional[str]` | e.g. "RADAR_SYSTEM" |
| `from_identity` | `Optional[dict[str, Any]]` | e.g. `{"system_name": "AN/MPQ-53"}` |
| `to_type` | `Optional[str]` | |
| `to_identity` | `Optional[dict[str, Any]]` | |
| `confidence` | `Optional[float]` | `coerce_optional_confidence` |

**`SystemLinkRelationship` fields** (cross-pass, ref_id-based per spec §3.5):

| Field | Type |
|---|---|
| `rel_type` | `Optional[str]` — `normalize_enum({"ASSOCIATED_WITH", "CUES"})` |
| `from_ref_id` | `Optional[str]` |
| `to_ref_id` | `Optional[str]` |
| `confidence` | `Optional[float]` |

**Pass-level `rel_type` enum sets for `normalize_enum`** (the allowed_rels per pass, matching manifest.yaml):

- `RadarDomainPass`: `{"INSTALLED_ON", "HAS_ANTENNA", "HAS_RECEIVER", "HAS_TRANSMITTER", "HAS_PROCESSING_CHAIN", "OPERATES_IN_BAND", "USES_WAVEFORM", "SPECIFIED_BY"}`
- `MissileDomainPass`: `{"INSTALLED_ON", "HAS_GUIDANCE", "HAS_SEEKER", "HAS_PROPULSION", "LAUNCHES", "SPECIFIED_BY"}`
- `OtherSystemsPass`: `{"INSTALLED_ON", "SPECIFIED_BY"}`
- `SystemLinksPass`: `{"ASSOCIATED_WITH", "CUES"}`

- [x] **Step 1: Write the failing test for structural invariants**

Create `tests/unit/test_extraction_schemas.py`:

```python
"""Tests that every extraction schema satisfies the partial-safety
and ontology-subset contracts (spec §3 + checker rules 6, 8, 9)."""
import pytest
from typing import get_args, get_origin, Union, Optional, Literal
from pydantic import BaseModel

from ontology_bundles.air_defense_v3.extraction_schemas import (
    reference, radar_domain, missile_domain, other_systems, system_links,
)

PASS_MODULES = [
    (reference, "ReferencePass"),
    (radar_domain, "RadarDomainPass"),
    (missile_domain, "MissileDomainPass"),
    (other_systems, "OtherSystemsPass"),
    (system_links, "SystemLinksPass"),
]

SYSTEM_FIELDS = {"confidence"}


def _is_optional(annotation) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def _iter_nested_models(model_cls: type[BaseModel]):
    """Walk fields recursively, yielding every nested BaseModel."""
    seen = {model_cls}
    stack = [model_cls]
    while stack:
        cls = stack.pop()
        for field_name, field_info in cls.model_fields.items():
            ann = field_info.annotation
            for nested in _unwrap_models(ann):
                if nested not in seen:
                    seen.add(nested)
                    stack.append(nested)
                    yield nested


def _unwrap_models(annotation):
    if annotation is None:
        return []
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        return []
    results = []
    for arg in get_args(annotation):
        results.extend(_unwrap_models(arg))
    return results


@pytest.mark.parametrize("module,class_name", PASS_MODULES)
def test_top_level_class_exists(module, class_name):
    assert hasattr(module, class_name), f"{module.__name__} missing {class_name}"
    cls = getattr(module, class_name)
    assert issubclass(cls, BaseModel)


@pytest.mark.parametrize("module,class_name", PASS_MODULES)
def test_top_level_instantiates_empty(module, class_name):
    cls = getattr(module, class_name)
    instance = cls()
    assert instance is not None


@pytest.mark.parametrize("module,class_name", PASS_MODULES)
def test_all_fields_optional_or_default_recursive(module, class_name):
    cls = getattr(module, class_name)
    for model in [cls, *_iter_nested_models(cls)]:
        for field_name, field_info in model.model_fields.items():
            if field_name in SYSTEM_FIELDS:
                continue
            is_optional = (
                not field_info.is_required()
                or field_info.default is not None
                or field_info.default_factory is not None
            )
            assert is_optional, (
                f"{model.__name__}.{field_name} is required; "
                f"extraction models must tolerate partial LLM output"
            )


def test_system_links_has_no_entity_fields():
    """Rule from spec §2 manifest self-consistency: input_mode=
    document_plus_entity_refs implies no entity collections."""
    cls = system_links.SystemLinksPass
    for field_name, field_info in cls.model_fields.items():
        if field_name == "relationships":
            continue
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is list:
            # This field is a list — make sure it's NOT a list of entity models.
            # Relationships are allowed; entities are not.
            inner = get_args(annotation)[0]
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                # Acceptable only if it's a Relationship-style class
                assert "Relationship" in inner.__name__ or "Link" in inner.__name__, (
                    f"SystemLinksPass.{field_name} is a list of {inner.__name__}, "
                    f"which looks like an entity collection. system_links must "
                    f"have no entity fields (spec §3.5)."
                )
```

- [x] **Step 2: Run tests — expect ImportError**

Run:
```bash
pytest tests/unit/test_extraction_schemas.py -v 2>&1 | tail -15
```

Expected: collection error because the schema modules don't exist yet.

- [x] **Step 3: Write reference.py**

Create `ontology_bundles/air_defense_v3/extraction_schemas/reference.py` with:
- Module docstring clarifying that HAS_PROVENANCE is auto-created by upsert_nodes_batch_sync (not derive_rules) per spec §3.3
- `SectionEntity`, `FigureEntity`, `TableEntity`, `AssertionEntity` classes. Every field Optional with default.
- Top-level `ReferencePass(BaseModel)` with `sections`, `figures`, `tables`, `assertions` fields (all `list[X] = Field(default_factory=list)`). NO `documents` field (spec §3.5).
- Field validators from `..validators` for page numbers, confidence.

Use spec §3.3 as the template.

- [x] **Step 4: Write radar_domain.py**

Create `ontology_bundles/air_defense_v3/extraction_schemas/radar_domain.py` per spec §3.4. Entity classes: `RadarSystemEntity`, `AntennaEntity`, `ReceiverEntity`, `TransmitterEntity`, `SPCEntity` (signal processing chain), `FrequencyBandEntity`, `WaveformEntity`, `PlatformEntity`, `SpecificationEntity`. All identity_fields from the ontology must be Pydantic fields (checker rule 9). Use a curated subset of properties, not the exhaustive list.

`RadarRelationship` class with `rel_type`, `from_type`, `from_identity: dict`, `to_type`, `to_identity: dict`, `confidence`. Enum normalization via `normalize_enum({...})`.

Top-level `RadarDomainPass` with all entity lists + `relationships: list[RadarRelationship]`.

- [x] **Step 5: Write missile_domain.py**

Same pattern. Entity classes: `MissileSystemEntity`, `LauncherSystemEntity`, `GuidanceMethodEntity`, `SeekerEntity`, `PropulsionStackEntity`, `PlatformEntity`, `SpecificationEntity` (bridges). `MissileRelationship` + `MissileDomainPass`.

- [x] **Step 6: Write other_systems.py**

Entity classes: `ADAEntity`, `EWSystemEntity`, `FireControlSystemEntity`, `WeaponSystemEntity`, `IADSEntity`, `PlatformEntity`, `SpecificationEntity`. `OtherSystemsRelationship` + `OtherSystemsPass`.

- [x] **Step 7: Write system_links.py**

Per spec §3.5: NO entity fields. Only a `SystemLinkRelationship` model and a `SystemLinksPass` with `relationships: list[SystemLinkRelationship]`. `SystemLinkRelationship` uses `from_ref_id` / `to_ref_id` (not `from_identity` / `to_identity`) because it operates on pre-extracted entity refs per spec §3.5 wire contract.

- [x] **Step 8: Run tests — expect pass**

Run:
```bash
pytest tests/unit/test_extraction_schemas.py -v
```

Expected: all pass. If any nested model fails the partial-safety check, find the required field and make it Optional with `default=None`.

- [x] **Step 9: Commit**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/ tests/unit/test_extraction_schemas.py
git commit -m "feat(bundle): add 5 hand-authored extraction schema modules

reference.py, radar_domain.py, missile_domain.py, other_systems.py,
system_links.py per spec §3.3–§3.5. Every field is Optional with
defaults (checker rule 6: partial-safety). system_links has no
entity fields because its input_mode is document_plus_entity_refs.
All modules share ../validators.py for @field_validator helpers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 1.6: Write derive_rules.py

**Files:**
- Create: `ontology_bundles/air_defense_v3/derive_rules.py`
- Create: `tests/unit/test_derive_rules.py`

Spec §3.8. HAS_PROVENANCE is NOT created here — it's auto-created by `upsert_nodes_batch_sync` in phase 2. Only `MENTIONED_IN` is produced by derive_rules.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_derive_rules.py`:

```python
"""Tests for derive_rules.derive_structural_edges.
HAS_PROVENANCE is NOT created here — see spec §3.8."""
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from ontology_bundles.air_defense_v3 import derive_rules
from ontology_bundles.air_defense_v3.derive_rules import (
    ChunkForDerivation, DerivedEdge,
)


@dataclass
class FakeIdentity:
    entity_type: str
    identity_field_names: tuple = ("name",)
    identity_tuple: tuple = ("Test",)
    scope: str = "global"
    document_id: str | None = None


@dataclass
class FakeMergedEntity:
    identity: FakeIdentity
    properties: dict = field(default_factory=dict)
    confidence: float = 0.9
    pass_origins: set = field(default_factory=set)
    display_label: str = "Test"


@dataclass
class FakeMerged:
    entities: list
    edges: list = field(default_factory=list)
    rejected_edges: list = field(default_factory=list)


def test_derive_structural_edges_no_has_provenance():
    """HAS_PROVENANCE must be handled by upsert_nodes_batch_sync,
    NOT by derive_rules. See spec §3.8."""
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="Fan Song",
    )
    merged = FakeMerged(entities=[entity])
    identity_to_rid = {entity.identity: "#10:1"}
    chunks = []

    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid="#11:1",
    )

    rel_types = [e.rel_type for e in edges]
    assert "HAS_PROVENANCE" not in rel_types, (
        "HAS_PROVENANCE must come from upsert_nodes_batch_sync, not derive_rules"
    )


def test_derive_structural_edges_mentioned_in():
    """MENTIONED_IN edges are created from entity display labels to
    chunks that contain them."""
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="Fan Song",
    )
    merged = FakeMerged(entities=[entity])
    identity_to_rid = {entity.identity: "#10:1"}
    chunks = [
        ChunkForDerivation(rid="#5:1", text_normalized="the fan song radar system"),
        ChunkForDerivation(rid="#5:2", text_normalized="unrelated text"),
    ]

    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid="#11:1",
    )

    mentioned = [e for e in edges if e.rel_type == "MENTIONED_IN"]
    assert len(mentioned) == 1
    assert mentioned[0].from_id == "#10:1"
    assert mentioned[0].to_id == "#5:1"


def test_derive_structural_edges_skips_entities_without_rid():
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="Fan Song",
    )
    merged = FakeMerged(entities=[entity])
    # Empty identity_to_rid — simulates a merged entity that wasn't upserted
    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid={},
        chunks=[ChunkForDerivation(rid="#5:1", text_normalized="fan song")],
        document_rid="#11:1",
    )
    assert edges == []


def test_derive_structural_edges_skips_empty_display_label():
    entity = FakeMergedEntity(
        identity=FakeIdentity(entity_type="RADAR_SYSTEM"),
        display_label="",
    )
    merged = FakeMerged(entities=[entity])
    identity_to_rid = {entity.identity: "#10:1"}
    edges = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=[ChunkForDerivation(rid="#5:1", text_normalized="some text")],
        document_rid="#11:1",
    )
    # No edges because canonical label is empty
    assert edges == []
```

- [x] **Step 2: Run tests — expect ImportError**

Run:
```bash
pytest tests/unit/test_derive_rules.py -v 2>&1 | tail -10
```

- [x] **Step 3: Write derive_rules.py**

Create `ontology_bundles/air_defense_v3/derive_rules.py` per spec §3.8. Include `ChunkForDerivation` dataclass, `DerivedEdge` dataclass, and `derive_structural_edges()` function. The function walks `merged.entities`, looks up each entity's RID, and emits MENTIONED_IN edges to chunks whose `text_normalized` contains the canonical label. Do NOT create HAS_PROVENANCE.

Include `normalize_name()` as a private module-level helper:

```python
import re

_WHITESPACE_RE = re.compile(r"\s+")

def normalize_name(name: str | None) -> str:
    """Canonicalize a display label for substring matching against chunk text.
    Returns '' for None or empty input so callers can skip falsy results."""
    if not name:
        return ""
    return _WHITESPACE_RE.sub(" ", name.strip().lower())
```

- [x] **Step 4: Run tests — expect pass**

Run:
```bash
pytest tests/unit/test_derive_rules.py -v
```

- [x] **Step 5: Commit**

```bash
git add ontology_bundles/air_defense_v3/derive_rules.py tests/unit/test_derive_rules.py
git commit -m "feat(bundle): add derive_rules.py for post-merge structural edges

Creates MENTIONED_IN edges from extracted entities to TextChunks
whose normalized text contains the entity's display label.
HAS_PROVENANCE is explicitly NOT created here — see spec §3.8.
It's auto-created by graph_store.upsert_nodes_batch_sync during
phase 2 via _create_provenance_edges_batch_sync.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```


---

**END OF CHUNK 1.** Chunk 1 covers the bundle directory skeleton and the core hand-authored content: manifest, coverage, validators, extraction schemas, derive_rules. The remaining PR 1 work (load_ontology refactor, alembic migration, new docling-graph endpoint, docker packaging, PR 0 honest-failures changes, and the PR 1 integration test) lives in Chunk 2 because Chunk 1 would exceed 1000 lines otherwise. Both chunks together form PR 1.


## Chunk 2: PR 1 — Scaffolding continued + honest-failures + integration test

This chunk completes PR 1. It covers: `load_ontology` refactor with caller audit, Alembic migration for all data-model additions, the coverage checker tool, the new docling-graph `POST /extract-pass` endpoint, docker packaging updates, PR 0-equivalent honest-failures changes on the legacy path, and the PR 1 integration test. Everything in this chunk is still strictly additive — the legacy extraction path must continue to run production documents end-to-end after Chunk 2 merges. Spec §7.3.

### Task 2.1: Refactor `load_ontology` + write caller audit

**Files:**
- Modify: `app/services/ontology_templates.py`
- Create: `tests/unit/test_load_ontology_refactor.py`
- Create: `docs/superpowers/plans/2026-04-10-load-ontology-caller-audit.md`

Spec §7.3 caller audit + §4.6 legacy NULL semantics.

- [x] **Step 1: Run the caller audit grep**

Run:
```bash
grep -rn "load_ontology\|get_ontology_cache_signature\|prefer_active" app/ tests/ docker/ \
  | grep -v "^Binary" | tee /tmp/load_ontology_callers.txt
wc -l /tmp/load_ontology_callers.txt
```

Expected: several dozen hits. Capture the list verbatim in `docs/superpowers/plans/2026-04-10-load-ontology-caller-audit.md`.

- [x] **Step 2: Classify each caller into one of three buckets**

Create `docs/superpowers/plans/2026-04-10-load-ontology-caller-audit.md` with a table:

```markdown
# load_ontology caller audit (PR 1)

| File:line | Caller | Bucket | Action |
|---|---|---|---|
| app/workers/pipeline.py:NNNN | ... | extraction | migrate to load_ontology(bundle_key=run.ontology_bundle_key) |
| app/api/v1/query_profiles.py:NNNN | ... | registry | migrate to load_registry_ontology(version_id=...) — add regression test |
| app/services/arcadedb_schema.py:NNNN | ... | default | no-change — drop prefer_active=True if present |
```

Three buckets (spec §7.3):
1. **extraction** — extraction hot path → `load_ontology(bundle_key=...)`
2. **registry** — version-pinned consumers → `load_registry_ontology(version_id=...)` + regression test
3. **default** — no-arg callers that are safe with system-default bundle → drop `prefer_active=True` if present, no other change

Every caller must be classified. If you can't decide, mark it `TBD` and flag for human review.

- [x] **Step 3: Commit the audit document**

```bash
git add docs/superpowers/plans/2026-04-10-load-ontology-caller-audit.md
git commit -m "docs(plan): load_ontology caller audit for PR 1 refactor

Every current caller of load_ontology / get_ontology_cache_signature /
prefer_active classified into extraction / registry / default buckets.
Bucket-2 (registry) callers get regression tests when the refactor
lands. See spec §7.3 and plan Chunk 2 Task 2.1.
"
```

- [x] **Step 4: Write the failing tests for the new load_ontology signatures**

Create `tests/unit/test_load_ontology_refactor.py`:

```python
"""Tests for the refactored load_ontology / load_registry_ontology split.
Spec §7.3 + §4.6."""
import pytest
from pathlib import Path
from unittest.mock import patch

from app.services.ontology_templates import (
    load_ontology, load_registry_ontology,
)


def test_load_ontology_no_args_uses_system_default_bundle():
    """With no args, load_ontology returns the system default bundle's ontology."""
    ont = load_ontology()
    assert "entity_types" in ont
    # Air-defense bundle has RADAR_SYSTEM
    assert any(e.get("name") == "RADAR_SYSTEM" for e in ont["entity_types"])


def test_load_ontology_with_bundle_key():
    ont = load_ontology(bundle_key="air_defense_v3")
    assert "entity_types" in ont


def test_load_ontology_unknown_bundle_raises():
    with pytest.raises(Exception):  # UnknownBundleError or similar
        load_ontology(bundle_key="does_not_exist")


def test_load_ontology_with_path(tmp_path):
    """Explicit path still works for tests / admin tools."""
    import yaml
    p = tmp_path / "fake.yaml"
    p.write_text(yaml.safe_dump({
        "entity_types": [{"name": "FOO"}],
        "relationship_types": [],
        "validation_matrix": [],
    }))
    ont = load_ontology(path=p)
    assert ont["entity_types"][0]["name"] == "FOO"


def test_load_ontology_no_longer_accepts_prefer_active():
    """prefer_active was dropped from the public signature."""
    with pytest.raises(TypeError):
        load_ontology(prefer_active=True)


def test_load_registry_ontology_exists():
    """load_registry_ontology is a separate function for version-pinned loads."""
    from app.services.ontology_templates import load_registry_ontology
    # The function exists and takes a version_id argument
    import inspect
    sig = inspect.signature(load_registry_ontology)
    assert "version_id" in sig.parameters
```

- [x] **Step 5: Run the tests — expect failures**

Run:
```bash
pytest tests/unit/test_load_ontology_refactor.py -v 2>&1 | tail -30
```

Expected: several failures because load_registry_ontology doesn't exist yet and load_ontology still accepts `prefer_active`.

- [x] **Step 6: Refactor app/services/ontology_templates.py**

Per spec §2 "Load ontology split":

```python
def load_ontology(
    *,
    bundle_key: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load an ontology definition.

    Resolution order (exactly one applies):
    1. If `path` is given, load directly from that file.
       (Used by tests and admin tools pointing at arbitrary files.)
    2. Else if `bundle_key` is given, load that bundle's ontology.yaml.
    3. Else load the system default bundle's ontology.yaml
       (currently `air_defense_v3`).

    This function never consults the registry/version-pinning store.
    For version-pinned loads, call load_registry_ontology() explicitly.
    """
    # Implementation:
    # - If path: read file directly
    # - Else: resolve bundle_key (or system default), read
    #   ontology_bundles/<bundle_key>/ontology.yaml

def load_registry_ontology(version_id: str) -> dict[str, Any]:
    """Load a version-pinned ontology snapshot from the
    registry.ontology_versions table.
    Used only by audit/historical-reproduction paths."""
    # Implementation: query registry.ontology_versions WHERE version_id=...
```

Update `_ensure_cache_populated` and any helper functions. Cache per-bundle via a dict keyed on `bundle_key`, not a global "active" cache.

Also update `get_ontology_cache_signature()` similarly if it still exists; split into `get_bundle_ontology_signature(bundle_key)` and `get_registry_ontology_signature(version_id)`.

- [x] **Step 7: Run tests — expect pass**

Run:
```bash
pytest tests/unit/test_load_ontology_refactor.py -v
```

Expected: all pass.

- [x] **Step 8: Run the full test suite to catch caller regressions**

Run:
```bash
pytest tests/ -x 2>&1 | tail -30
```

Expected: the audit document's bucket-2 callers may fail (they're using the old registry-lookup path). Fix them per the audit: replace `load_ontology(prefer_active=True)` calls with `load_registry_ontology(version_id=...)` using an explicit version id. Re-run until green.

Bucket-1 (extraction) callers don't exist yet at this point — the legacy path still uses the old dispatch code in docker/docling-graph. Those stay unchanged.

Bucket-3 (default) callers may need `prefer_active=True` stripped; that's a mechanical `sed`-level change.

- [x] **Step 9: Commit**

List every file you actually modified during Steps 6–8 (the refactor plus the caller fixes you had to make to get the suite green) and stage them explicitly. Do NOT use `git add -A`.

```bash
# Example — replace with the real list from your working tree:
# git status --short
git add app/services/ontology_templates.py tests/unit/test_load_ontology_refactor.py
# Plus whichever caller files Step 8 forced you to touch, e.g.:
# git add app/api/v1/query_profiles.py app/services/arcadedb_schema.py ...
git commit -m "refactor(ontology): split load_ontology and load_registry_ontology

Per spec §2 load_ontology split. load_ontology() is now a bundle+path
loader with no registry side effects; load_registry_ontology() is a
new explicit function for version-pinned historical loads. prefer_active
parameter dropped from the public signature — callers that need
registry lookups must call load_registry_ontology(version_id=...) with
an explicit version. Per-bundle caching replaces the global active cache.

Caller audit committed separately at
docs/superpowers/plans/2026-04-10-load-ontology-caller-audit.md per
spec §7.3 PR 1 prep step.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.2: Write Alembic migration for all data-model additions

**Files:**
- Create: `alembic/versions/0015_bundle_and_per_pass_extraction.py`
- Create: `tests/unit/test_migration_0015.py`

Spec §4.8 full migration. All new columns and indexes, including the explicit drop of the existing `uq_stage_run` uniqueness constraint.

- [x] **Step 1: Verify the existing uq_stage_run constraint name**

The verification MUST run against a DB that is at Alembic revision 0014 (pre-migration) — otherwise an earlier local attempt may have already dropped the constraint, giving a false negative.

```bash
# Confirm the DB is at 0014 before querying the constraint:
alembic current
# Expected output: "0014 (head)" or similar. If not, run:
#   alembic downgrade 0014
# before proceeding.

docker compose exec postgres psql -U postgres -d eip_arcadedb \
  -c "SELECT conname FROM pg_constraint
      WHERE conrelid = 'ingest.stage_runs'::regclass AND contype = 'u';"
```

Expected: `uq_stage_run`. Residual check #2 in the spec. If the name differs, record it — the migration hardcodes the exact name.

- [x] **Step 2: Check existing data for partial-index pre-flight**

Run against the same DB:

```bash
docker compose exec postgres psql -U postgres -d eip_arcadedb \
  -c "SELECT pipeline_run_id, attempt, count(*)
      FROM ingest.stage_runs
      WHERE pass_name IS NULL AND stage_name = 'derive_ontology_graph'
      GROUP BY 1, 2 HAVING count(*) > 1;"
```

Expected: empty result. If duplicates exist, write a dedupe migration step BEFORE the index creation. Residual check #3 in the spec.

- [x] **Step 3: Generate the migration skeleton**

Run:
```bash
alembic revision -m "bundle and per-pass extraction"
ls alembic/versions/ | tail -3
```

Rename the generated file to `0015_bundle_and_per_pass_extraction.py` if needed to keep the numeric prefix pattern.

- [x] **Step 4: Write the migration body**

Use spec §4.8 as the column/index/constraint checklist, but source the actual view SQL from spec §4.4 (§4.8 uses a `...` placeholder for that block). Key sections:

1. `op.add_column` for Source (2 cols), PipelineRun (6 cols including `metrics` JSONB and `mode`), DocumentGraphExtraction (1 FK + 5 audit), StageRun (12 cols including `rollback_executed`).
2. `op.create_check_constraint('chk_pipeline_run_mode', ...)` on PipelineRun.
3. `op.create_foreign_key('fk_dge_pipeline_run', ...)` on DocumentGraphExtraction.
4. **`op.drop_constraint('uq_stage_run', 'stage_runs', schema='ingest', type_='unique')`** — verified name from Step 1.
5. `op.create_index('uq_stage_runs_run_pass_attempt', ..., unique=True, postgresql_where=sa.text('pass_name IS NOT NULL'))`
6. `op.create_index('uq_stage_runs_summary_row', ..., unique=True, postgresql_where=sa.text("pass_name IS NULL AND stage_name = 'derive_ontology_graph'"))` — narrowed scope per spec §4.4.
7. `op.create_index('ix_stage_runs_extraction_pass', ..., postgresql_where=...)`.
8. `op.create_index('ix_stage_runs_execution_status', ..., postgresql_where=...)`.
9. `op.execute("""CREATE VIEW ingest.v_latest_pass_attempts AS ...""")` — **the literal view body lives in spec §4.4, not §4.8.** Open §4.4 ("Per-pass StageRun rows"), find the SQL block that starts with `CREATE VIEW ingest.v_latest_pass_attempts AS`, and copy it verbatim into the `op.execute("""...""")` call. Include the `DISTINCT ON (pipeline_run_id, stage_name, pass_name)` projection and the `ORDER BY ..., attempt DESC` ordering.

The `downgrade()` function must be authored from scratch in strict reverse order — spec §4.8 does not provide a drop-in downgrade body:

1. `op.execute("DROP VIEW IF EXISTS ingest.v_latest_pass_attempts")`
2. `op.drop_index('ix_stage_runs_execution_status', table_name='stage_runs', schema='ingest')`
3. `op.drop_index('ix_stage_runs_extraction_pass', table_name='stage_runs', schema='ingest')`
4. `op.drop_index('uq_stage_runs_summary_row', table_name='stage_runs', schema='ingest')`
5. `op.drop_index('uq_stage_runs_run_pass_attempt', table_name='stage_runs', schema='ingest')`
6. `op.create_unique_constraint('uq_stage_run', 'stage_runs', ['pipeline_run_id', 'stage_name', 'attempt'], schema='ingest')` — restore the original constraint.
7. `op.drop_constraint('fk_dge_pipeline_run', 'document_graph_extractions', schema='retrieval', type_='foreignkey')`
8. `op.drop_constraint('chk_pipeline_run_mode', 'pipeline_runs', schema='ingest', type_='check')`
9. `op.drop_column(...)` for every column added in `upgrade()`, in reverse order of addition.

- [x] **Step 5: Write a migration smoke test**

Create `tests/unit/test_migration_0015.py`:

```python
"""Smoke test for Alembic migration 0015."""
import subprocess

def test_migration_applies_cleanly():
    """Upgrade head → verify all new columns present."""
    result = subprocess.run(
        ["alembic", "upgrade", "0015"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"Migration failed: {result.stderr}"

def test_migration_downgrades_cleanly():
    """Downgrade head → 0014 → verify new columns removed."""
    result = subprocess.run(
        ["alembic", "downgrade", "0014"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"Downgrade failed: {result.stderr}"
    # Re-apply so other tests can use the head schema
    subprocess.run(["alembic", "upgrade", "head"], check=True)
```

- [x] **Step 6: Run the migration on a fresh DB**

Run:
```bash
docker compose exec postgres psql -U postgres -d eip_arcadedb \
  -c "\d ingest.stage_runs" | grep -E "pass_name|execution_status|rollback_executed|metrics"
alembic upgrade head
docker compose exec postgres psql -U postgres -d eip_arcadedb \
  -c "\d ingest.stage_runs" | grep -E "pass_name|execution_status|rollback_executed"
docker compose exec postgres psql -U postgres -d eip_arcadedb \
  -c "\dv ingest.v_latest_pass_attempts"
```

Expected: before upgrade, no new columns. After upgrade, all new columns present and the view exists.

- [x] **Step 7: Run downgrade and re-upgrade**

Run:
```bash
alembic downgrade 0014
alembic upgrade head
pytest tests/unit/test_migration_0015.py -v
```

Expected: both operations succeed, test passes.

- [x] **Step 8: Update SQLAlchemy models to match the new schema**

Modify `app/models/ingest.py`:
- `Source`: add `default_ontology_bundle_key`, `default_use_case_key` per spec §4.1.
- `PipelineRun`: add `mode`, `ontology_bundle_key`, `ontology_name`, `ontology_version`, `use_case_key`, `extraction_profile_version`, `metrics` per spec §4.2.
- `DocumentGraphExtraction`: add `pipeline_run_id` FK and 5 audit columns per spec §4.3.
- `StageRun`: add `pass_name`, `execution_status`, `yield_status`, `skip_reason`, `primary_entities_extracted`, `bridge_entities_extracted`, `relationships_extracted`, `relationships_rejected`, `salvaged`, `schema_size_chars`, `structured_output_mode`, `rollback_executed` per spec §4.4.
- Remove the `UniqueConstraint("pipeline_run_id", "stage_name", "attempt", name="uq_stage_run")` from `__table_args__` and leave the partial indexes to be declared in the migration.

Run:
```bash
pytest tests/ -x 2>&1 | tail -30
```

Expected: all existing tests still pass. Model changes are additive + one constraint removal.

- [x] **Step 9: Commit**

```bash
git add alembic/versions/0015_bundle_and_per_pass_extraction.py \
       app/models/ingest.py tests/unit/test_migration_0015.py
git commit -m "feat(db): alembic migration 0015 — bundle fields + per-pass StageRun

Adds every column, index, constraint, and view from spec §4.8:
- Source.default_ontology_bundle_key, default_use_case_key
- PipelineRun.mode, ontology_bundle_key, ontology_name, ontology_version,
  use_case_key, extraction_profile_version, metrics
- DocumentGraphExtraction.pipeline_run_id FK + 5 audit cols
- StageRun.pass_name, execution_status, yield_status, skip_reason,
  primary/bridge/relationships_extracted/_rejected, salvaged,
  schema_size_chars, structured_output_mode, rollback_executed
- Drops old uq_stage_run constraint
- Creates uq_stage_runs_run_pass_attempt and uq_stage_runs_summary_row
  partial unique indexes, ix_stage_runs_extraction_pass,
  ix_stage_runs_execution_status, and the v_latest_pass_attempts view

All columns nullable (mode has server_default='full'). No backfill —
NULL means legacy/unknown per spec §4.6.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.3: Write ontology_bundles loader (worker side) and bundles loader (service side)

**Files:**
- Create: `app/services/ontology_bundles.py`
- Create: `docker/docling-graph/app/bundles.py`
- Create: `tests/unit/test_ontology_bundles.py`

Spec §2 Bundle loader API.

- [x] **Step 1: Write failing tests**

Create `tests/unit/test_ontology_bundles.py` with the five test cases from spec §8.4 `test_ontology_bundles.py`:

1. `test_load_bundle_manifest` — loader happy path, parses `manifest.yaml` into a `BundleManifest` with five passes.
2. `test_resolve_bundle_key_precedence` — run → source → system-default, each taking over when the previous tier is `None`.
3. `test_resolve_bundle_key_raises_when_all_tiers_none` — `BundleResolutionError` when every tier is `None`.
4. `test_load_ontology_bundle_key_none_round_trip` — `load_ontology(bundle_key=None)` returns the system default bundle's ontology dict and is equivalent to `load_ontology(bundle_key=<system_default>)`.
5. `test_load_ontology_prefer_active_regression` — `load_ontology(prefer_active=True)` raises `TypeError`.

- [x] **Step 2: Run tests — expect failures**

```bash
pytest tests/unit/test_ontology_bundles.py -v 2>&1 | tail -15
```

- [x] **Step 3: Implement worker-side `app/services/ontology_bundles.py`**

Per spec §2 Bundle loader API. In this chunk, declare only:

- Classes: `PassManifest`, `BundleManifest` (Pydantic v2 models matching the `manifest.yaml` shape from Chunk 1 Task 1.3).
- Functions: `load_bundle_manifest(bundle_key)`, `list_available_bundles()`, `load_bundle_ontology(bundle_key)`, `resolve_bundle_key(*, run_key, source_key, system_default)`, `resolve_bundle_key_for_graph_only(*, run_key)`, `describe_bundle_for_display(bundle_key)`.
- Constant: `LEGACY_BUNDLE_LABEL = "legacy/unknown"` per spec §4.6.
- Exception: `BundleResolutionError(ValueError)` for the all-tiers-None case.

**Defer `StatusSignals` to Chunk 3 — do not create it here.** That class belongs to the orchestrator rewrite (Chunk 3 Task covering `extraction_merge.py` and the status roll-up), so declaring it now would leave an unused import surface.

- [x] **Step 4: Implement service-side `docker/docling-graph/app/bundles.py`**

Per spec §2:

```python
def load_bundle_manifest(bundle_key: str) -> BundleManifest: ...
def load_pass_template(bundle_key: str, pass_name: str) -> type[BaseModel]:
    """Import the module and return the template class declared in manifest.
    All bundles are pre-imported on service startup for fast per-request dispatch."""
def preload_all_templates() -> None:
    """Iterate every bundle and force-import every declared extraction
    schema module, populating the module-level cache used by
    load_pass_template. Called once at service startup."""
```

The `preload_all_templates()` call is wired into the FastAPI lifespan in Task 2.5 Step 4 when `main.py` is edited — do NOT modify `main.py` in this task. Task 2.5 owns that edit and will import from `app.bundles`.

- [x] **Step 5: Run tests**

```bash
pytest tests/unit/test_ontology_bundles.py -v
```

Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add app/services/ontology_bundles.py docker/docling-graph/app/bundles.py \
       tests/unit/test_ontology_bundles.py
git commit -m "feat(bundle): worker and service bundle loaders

app/services/ontology_bundles.py: worker-side loader that reads
manifest.yaml metadata without importing extraction_schemas modules.
Exposes BundleManifest, PassManifest, resolve_bundle_key,
resolve_bundle_key_for_graph_only, load_bundle_ontology,
describe_bundle_for_display, LEGACY_BUNDLE_LABEL.

docker/docling-graph/app/bundles.py: service-side loader that also
imports the extraction_schemas modules and returns Pydantic template
classes. Pre-imports all bundles at service startup.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.4: Write the coverage checker (tools/check_extraction_coverage.py)

**Files:**
- Create: `tools/check_extraction_coverage.py`
- Create: `tools/extraction_coverage/__init__.py`
- Create: `tools/extraction_coverage/rules.py`
- Create: `tools/extraction_coverage/manifest_consistency.py`
- Create: `ontology_bundles/_shared/__init__.py`
- Create: `ontology_bundles/_shared/limits.py` — shared schema-size constant used by both the CI checker and the docling-graph service ServiceSettings, so the two cannot drift (spec §409)
- Create: `tests/unit/test_coverage_checker.py`

Spec §2 checker rules 1-14 (rule 7 removed per spec §7.10) and manifest self-consistency sub-checks. The `_shared` subdirectory is intentionally skipped by the CLI loop because it starts with `_`.

- [x] **Step 1: Write failing tests that drive each checker rule**

Create `tests/unit/test_coverage_checker.py` with unit tests for each rule in spec §2. Use small in-memory fixtures rather than mutating the real bundle. Each test sets up a synthetic manifest/coverage/ontology and calls the checker function for that rule in isolation.

- [x] **Step 2a: Create the shared schema-size limit module**

Spec §409 requires the CI checker and the runtime monkey-patch to read the same `structured_output_threshold_chars` value. Put the default in a shared module that both the worker-side checker and the docling-graph service can import:

```python
# ontology_bundles/_shared/limits.py
"""Shared numeric limits used by both the CI coverage checker and the
docling-graph service runtime. Spec §409: the structured-output
threshold must be a single source of truth so CI and runtime cannot drift."""

#: Default schema-size ceiling. Both sides may override at runtime by reading
#: env var DOCLING_GRAPH_STRUCTURED_OUTPUT_THRESHOLD_CHARS, but the default
#: MUST be defined here.
DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS = 8000
```

And an empty `ontology_bundles/_shared/__init__.py` so it imports cleanly.

Both images already COPY `ontology_bundles/` per Task 2.6, so both sides can `from ontology_bundles._shared.limits import DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS`.

- [x] **Step 2: Implement `tools/check_extraction_coverage.py` as a small package**

Because the checker has 13 rules + manifest self-consistency + advisory warnings, keep it split into three modules so no file exceeds ~300 lines:

```
tools/
  check_extraction_coverage.py   # thin CLI entry point; iterates bundles
  extraction_coverage/
    __init__.py
    rules.py                     # one function per rule (1-14, no 7)
    manifest_consistency.py      # manifest-self-consistency sub-checks
```

`check_extraction_coverage.py` CLI entry point:

```python
#!/usr/bin/env python3
"""Coverage checker for ontology_bundles/*. Runs in CI.
Spec §2 checker rules 1-14 (rule 7 removed) + manifest self-consistency."""
from pathlib import Path
import sys
import yaml

from tools.extraction_coverage.rules import check_bundle

def main() -> int:
    bundles_dir = Path("ontology_bundles")
    exit_code = 0
    for bundle_dir in sorted(bundles_dir.iterdir()):
        if not bundle_dir.is_dir() or bundle_dir.name.startswith("_"):
            continue
        errors, warnings = check_bundle(bundle_dir)
        for w in warnings:
            print(f"WARN {bundle_dir.name}: {w}")
        if errors:
            exit_code = 1
            print(f"FAIL {bundle_dir.name}:")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"PASS {bundle_dir.name}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
```

`tools/extraction_coverage/rules.py` skeleton — one private function per rule, and a `check_bundle` that composes them:

```python
"""Coverage rule implementations. Spec §2 checker rules 1-14 (7 removed)."""
import importlib
import os
from pathlib import Path
from typing import get_args, get_origin, Union
import yaml
from pydantic import BaseModel

from ontology_bundles._shared.limits import DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS
from app.services.ontology_bundles import load_bundle_manifest
from tools.extraction_coverage.manifest_consistency import check_manifest_self_consistency

# Exempt from rule 8 per Chunk 1 Task 1.5 Step 0.
SYSTEM_FIELDS = {"confidence"}

# Rule 5: resolve the schema-size threshold exactly like the runtime
# monkey-patch does — env override if present, else the shared default.
# Spec §409 requires CI and runtime to agree.
SCHEMA_SIZE_THRESHOLD_CHARS = int(
    os.environ.get(
        "DOCLING_GRAPH_STRUCTURED_OUTPUT_THRESHOLD_CHARS",
        DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS,
    )
)


def _check_coverage_subset(ontology, coverage) -> list[str]:
    """Rule 1: coverage.extract ∪ coverage.derive ⊆ ontology.entity_types."""
    ...

def _check_manifest_entities_in_coverage(manifest, coverage) -> list[str]:
    """Rule 2: every entity declared in a manifest pass appears in coverage.extract."""
    ...

def _check_manifest_relationships_in_coverage(manifest, coverage) -> list[str]:
    """Rule 3: every relationship declared in a manifest pass appears in coverage.extract."""
    ...

def _check_relationships_in_validation_matrix(manifest, ontology) -> list[str]:
    """Rule 4: every extracted relationship has a row in ontology.validation_matrix."""
    ...

def _check_schema_size(template_cls) -> list[str]:
    """Rule 5: rendered JSON schema for each pass template is under SCHEMA_SIZE_THRESHOLD_CHARS."""
    ...

def _check_recursive_partial_safety(template_cls) -> list[str]:
    """Rule 6: every field reachable from the pass template that Pydantic
    could serialize as None is either Optional[...] or has a default."""
    ...

# Rule 7 is intentionally removed — see spec §2.

def _check_extraction_subset_of_ontology(template_cls, ontology) -> list[str]:
    """Rule 8: every field name on every nested entity model in the pass
    template corresponds to a property on the matching ontology entity
    (SYSTEM_FIELDS are exempt)."""
    ...

def _check_identity_fields_completeness(ontology, coverage) -> list[str]:
    """Rule 9: every entity in coverage.extract has a non-empty identity_fields
    list OR is explicitly marked as allowing content-hash fallback."""
    ...

def _check_display_label_present(ontology, coverage) -> list[str]:
    """Rule 10: every entity in coverage.extract has a non-empty display_label field."""
    ...

def _check_identity_scope_required(ontology, coverage) -> list[str]:
    """Rule 11: every entity in coverage.extract declares identity_scope ∈ {'document','global'}."""
    ...

def _check_empty_identity_global_warning(ontology, coverage) -> list[str]:
    """Rule 12: WARNING (not error) on identity_fields=[] combined with identity_scope='global'."""
    # returns warnings, not errors — handled specially in check_bundle
    ...

def _check_bridge_scope_consistency(ontology, coverage, manifest) -> list[str]:
    """Rule 13: bridge entities appearing in multiple passes have identical
    identity_fields and identity_scope across those declarations."""
    ...

# Rule 14 (rejection-reason coverage) is enforced by a separate pytest test,
# not by this checker — see Chunk 3 test tasks.

def _load_pass_template_by_manifest(pass_def: dict) -> type[BaseModel]:
    """Resolve `module_path` + `template_class` from manifest.yaml into a
    Pydantic class. Works for any bundle — never hardcodes the bundle key."""
    module = importlib.import_module(pass_def["module_path"])
    return getattr(module, pass_def["template_class"])


def check_bundle(bundle_path: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Empty errors list = checker passed."""
    errors: list[str] = []
    warnings: list[str] = []

    ontology = yaml.safe_load((bundle_path / "ontology.yaml").read_text())
    manifest = yaml.safe_load((bundle_path / "manifest.yaml").read_text())
    coverage = yaml.safe_load((bundle_path / "coverage.yaml").read_text())
    bundle_key = bundle_path.name

    errors += _check_coverage_subset(ontology, coverage)
    errors += _check_manifest_entities_in_coverage(manifest, coverage)
    errors += _check_manifest_relationships_in_coverage(manifest, coverage)
    errors += _check_relationships_in_validation_matrix(manifest, ontology)
    errors += _check_identity_fields_completeness(ontology, coverage)
    errors += _check_display_label_present(ontology, coverage)
    errors += _check_identity_scope_required(ontology, coverage)
    errors += _check_bridge_scope_consistency(ontology, coverage, manifest)
    warnings += _check_empty_identity_global_warning(ontology, coverage)

    for pass_def in manifest.get("passes", []):
        try:
            template_cls = _load_pass_template_by_manifest(pass_def)
        except (ImportError, AttributeError) as exc:
            errors.append(
                f"pass {pass_def.get('name', '?')}: "
                f"cannot import {pass_def.get('module_path')}."
                f"{pass_def.get('template_class')}: {exc}"
            )
            continue
        errors += _check_schema_size(template_cls)
        errors += _check_recursive_partial_safety(template_cls)
        errors += _check_extraction_subset_of_ontology(template_cls, ontology)

    errors += check_manifest_self_consistency(manifest, coverage, bundle_key=bundle_key)

    return errors, warnings
```

`tools/extraction_coverage/manifest_consistency.py` — one function covering all manifest self-consistency sub-checks from spec §2:

```python
"""Manifest self-consistency sub-checks — structural validation of
manifest.yaml independent of ontology and pydantic templates."""
import importlib

def check_manifest_self_consistency(
    manifest: dict,
    coverage: dict,
    *,
    bundle_key: str,
) -> list[str]:
    """All checks from spec §2 'manifest self-consistency' section:
    - unique pass names
    - depends_on references only earlier passes
    - document_only vs document_plus_entity_refs usage
    - entities and relationships sets are disjoint across passes
      EXCEPT for declared bridge entities
    - every template_class / module path is importable (smoke import)

    `bundle_key` is passed in to allow future per-bundle structural checks
    and to produce error messages that name the offending bundle.
    """
    errors: list[str] = []
    ...
    return errors
```

Implement every `...` body by reading the matching rule text in spec §2 and translating it into Python. Each rule function must return a list of human-readable error strings; callers accumulate them.

- [x] **Step 3: Run unit tests**

```bash
pytest tests/unit/test_coverage_checker.py -v
```

Expected: all pass.

- [x] **Step 4: Run the checker against the real bundle**

```bash
python tools/check_extraction_coverage.py
```

Expected: `PASS air_defense_v3`. If any rules fail, fix the bundle content (not the checker) and re-run.

- [x] **Step 5: Commit**

```bash
git add tools/check_extraction_coverage.py tests/unit/test_coverage_checker.py
git commit -m "feat(tools): coverage checker for ontology_bundles

tools/check_extraction_coverage.py implements all 13 active checker
rules from spec §2 (rule 7 is a removed placeholder) plus manifest
self-consistency sub-checks. Runs in CI and against the local bundle
during development.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.5: Add POST /extract-pass endpoint to docling-graph service

**Files:**
- Modify: `docker/docling-graph/app/main.py`
- Modify: `docker/docling-graph/app/schemas.py`
- Create: `docker/docling-graph/app/config.py` (new `ServiceSettings`)
- Create: `docker/docling-graph/tests/test_extract_pass_endpoint.py`

Spec §5.9 wire contract + §2 packaging §2.9 + §7.3 PR 1 deliverables.

- [x] **Step 1: Write failing endpoint tests**

Create `docker/docling-graph/tests/test_extract_pass_endpoint.py` with the 7 test cases from spec §8.6:
- document_only pass with unexpected upstream_entities → 400
- document_plus_entity_refs pass with missing upstream_entities → 400
- document_plus_entity_refs pass with empty upstream_entities → 400
- unknown bundle_key → 404
- unknown pass_name → 404
- valid document_only request → 200
- valid document_plus_entity_refs request → 200

- [x] **Step 2: Create `docker/docling-graph/app/config.py`**

Scope this PR-1 change to ONE setting only: the structured-output threshold currently hardcoded as `8000` inside `_patched_build_request`. Other scattered config stays where it is; later PRs may consolidate more, but that is out of scope for PR 1.

```python
from pydantic_settings import BaseSettings
from ontology_bundles._shared.limits import DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS


class ServiceSettings(BaseSettings):
    """Docling-graph service configuration. Loaded at startup.
    PR 1 scope: only structured_output_threshold_chars.
    Do NOT add more keys in this PR — other scattered config stays in place.
    """
    structured_output_threshold_chars: int = DEFAULT_STRUCTURED_OUTPUT_THRESHOLD_CHARS

    class Config:
        env_prefix = "DOCLING_GRAPH_"

settings = ServiceSettings()
```

The default is sourced from `ontology_bundles/_shared/limits.py` (created in Task 2.4 Step 2a) so the CI checker and the runtime monkey-patch cannot drift per spec §409.

- [x] **Step 3: Add ExtractPassRequest / ExtractPassResponse / EntityRef to schemas.py**

Per spec §5.9 wire contract. Do NOT remove `ExtractAllRequest` or `ontology_definition` yet — PR 1 is additive.

- [x] **Step 4: Add POST /extract-pass to main.py**

Before writing new code, read the existing `POST /extract-all` handler in `docker/docling-graph/app/main.py` end-to-end. The new handler MUST invoke `docling_graph.run_pipeline` (or whichever function the existing handler calls) with the exact same shape of arguments, except that `template_cls` is passed as the single fixed pass template from `load_pass_template(bundle_key, pass_name)` instead of the runtime-generated template the legacy path builds. Nothing else about the pipeline call — logger, error translation, response body assembly — should differ from the legacy handler. Reuse, do not reinvent.

Also wire the lifespan startup hook that `preload_all_templates()` from Task 2.3 Step 4 expects:

```python
from contextlib import asynccontextmanager
from app.bundles import preload_all_templates

@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_all_templates()
    yield

app = FastAPI(lifespan=lifespan, ...)  # preserve existing args
```

If the service already defines a lifespan context, extend it rather than replacing it.

The new handler skeleton:

```python
@app.post("/extract-pass", response_model=ExtractPassResponse)
async def extract_pass(body: ExtractPassRequest, request: Request):
    # 1. Resolve bundle + pass
    try:
        manifest = load_bundle_manifest(body.bundle_key)
    except UnknownBundleError:
        raise HTTPException(status_code=404, detail=f"Unknown bundle_key: {body.bundle_key}")
    try:
        pass_def = manifest.find_pass(body.pass_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown pass_name: {body.pass_name}")

    # 2. Validate input_mode compatibility
    if pass_def.input_mode == "document_only" and body.upstream_entities:
        raise HTTPException(status_code=400, detail="document_only pass received upstream_entities")
    if pass_def.input_mode == "document_plus_entity_refs" and not body.upstream_entities:
        raise HTTPException(status_code=400, detail="document_plus_entity_refs pass missing upstream_entities")

    # 3. Load the pre-imported fixed template
    template_cls = load_pass_template(body.bundle_key, body.pass_name)

    # 4. Run the pipeline — mirror the existing /extract-all handler's
    #    docling_graph.run_pipeline invocation exactly, substituting
    #    template_cls and passing body.upstream_entities when present.
    #    Response body assembly should match ExtractPassResponse shape
    #    defined in Step 3.
```

Finally, replace the hardcoded `8000` threshold in the existing `_patched_build_request` monkey-patch with `settings.structured_output_threshold_chars`. Import `settings` from `app.config`.

- [x] **Step 5: Run tests**

The endpoint tests are contract/protocol tests per spec §8.6 — they do NOT run the real pipeline. Use `pytest` with `fastapi.testclient.TestClient` and `unittest.mock.patch` to replace `docling_graph.run_pipeline` (or whichever helper the legacy handler calls) with a stub that returns a canned response shape. The tests assert request-validation behavior (400/404/200 status codes and error detail strings), not extraction correctness.

```bash
pytest docker/docling-graph/tests/test_extract_pass_endpoint.py -v
```

Expected: all 7 tests pass without a running docling-graph service.

- [x] **Step 6: Verify legacy /extract-all still responds**

Rebuild and restart the docling-graph service (the image COPY uses baked-in code, not a volume mount — stored preference in MEMORY.md):

```bash
docker compose build docling-graph
docker compose up -d docling-graph
docker compose logs --tail=20 docling-graph   # confirm clean startup, lifespan preload_all_templates ran
```

Then hit the legacy endpoint:

```bash
curl -X POST http://localhost:8002/extract-all \
  -H "Content-Type: application/json" \
  -d '{"docling_document_json": {"...": "..."}}'
```

Expected: 200 OK (or whatever the existing endpoint returned pre-change). The old path must still work after all PR 1 scaffolding landed.

- [x] **Step 7: Commit**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/app/schemas.py \
       docker/docling-graph/app/config.py \
       docker/docling-graph/tests/test_extract_pass_endpoint.py
git commit -m "feat(docling-graph): new POST /extract-pass endpoint (additive)

Per spec §5.9 wire contract. Accepts {bundle_key, pass_name,
docling_document_json, upstream_entities?}, loads the fixed pass
template from the bundle via bundles.load_pass_template, and runs
the docling_graph pipeline. Legacy POST /extract-all endpoint is
untouched. Introduces docker/docling-graph/app/config.py with
ServiceSettings; the 8000-char threshold in _patched_build_request
now reads from settings.structured_output_threshold_chars.

Protocol tests per spec §8.6: input_mode validation (400), unknown
bundle/pass (404), happy-path 200 for both input_mode values.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.6: Update Docker packaging for ontology_bundles

**Files:**
- Modify: `docker-compose.yml`
- Modify: `docker/docling-graph/Dockerfile`
- Modify: `docker/worker/Dockerfile`
- Create: `scripts/smoke_test_bundle_import.sh`

Spec §2 packaging section + §7.3 Docker packaging.

- [x] **Step 1: Update docker-compose.yml for docling-graph build context**

Edit the `docling-graph` service:
```yaml
docling-graph:
  build:
    context: .                              # changed from docker/docling-graph
    dockerfile: docker/docling-graph/Dockerfile
```

Keep the existing `./ontology:/app/ontology:ro` bind mount — the symlink from Task 1.2 makes it continue to work.

- [x] **Step 2: Update docker/docling-graph/Dockerfile**

Change COPY paths to be relative to repo root:
```dockerfile
COPY docker/docling-graph/app /app/app
COPY ontology_bundles /app/ontology_bundles
ENV PYTHONPATH=/app
```

- [x] **Step 3: Update docker/worker/Dockerfile**

Add:
```dockerfile
COPY ontology_bundles /app/ontology_bundles
```

- [x] **Step 4: Create the smoke-test script**

Create `scripts/smoke_test_bundle_import.sh`:
```bash
#!/bin/bash
set -euo pipefail
# Smoke test: verify ontology_bundles is importable inside both
# worker and docling-graph images. Run during CI after builds.
docker compose build worker docling-graph
docker compose run --rm worker python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass
print('worker OK:', RadarDomainPass)
"
docker compose run --rm docling-graph python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass
print('docling-graph OK:', RadarDomainPass)
"
```

- [x] **Step 5: Run the smoke test**

```bash
chmod +x scripts/smoke_test_bundle_import.sh
./scripts/smoke_test_bundle_import.sh
```

Expected: both services print `OK` and the class repr.

- [x] **Step 6: Commit**

```bash
git add docker-compose.yml docker/docling-graph/Dockerfile docker/worker/Dockerfile \
       scripts/smoke_test_bundle_import.sh
git commit -m "feat(docker): package ontology_bundles into worker and docling-graph images

- docker-compose.yml: docling-graph build.context = . (repo root)
- docker/docling-graph/Dockerfile: COPY docker/docling-graph/app and
  COPY ontology_bundles, PYTHONPATH=/app
- docker/worker/Dockerfile: COPY ontology_bundles
- scripts/smoke_test_bundle_import.sh verifies importability

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.7: PR 0 equivalent — honest failures on the legacy path

**Files:**
- Modify: `app/config.py`
- Modify: `app/services/layered_extraction.py`
- Modify: `app/workers/pipeline.py`

Spec §7.3 PR 0 equivalent. This change establishes a truthful baseline on the legacy path before switchover, so the §8.3 baseline harness in Chunk 3 has honest numbers to compare against.

- [x] **Step 1: Flip default flags in app/config.py**

```python
# Before:
graph_layered_shadow_mode: bool = True
graph_layered_fail_open_to_single_pass: bool = True

# After:
graph_layered_shadow_mode: bool = False
graph_layered_fail_open_to_single_pass: bool = False
```

Leave the settings as env-var-overridable so ops can re-enable if production needs it.

- [x] **Step 2: Remove error-swallowing in layered_extraction.py**

Locate the per-pass try/except that catches exceptions and converts them to empty-result sentinels. Find it with:

```bash
grep -n "except" app/services/layered_extraction.py | head -20
```

The target is the `try/except Exception` block(s) inside the per-pass runner function — typically named something like `_run_single_pass`, `_execute_pass`, or `run_layered_extraction`'s inner loop. Each such block currently:

1. Catches any exception
2. Logs it
3. Returns an empty result dict (or appends an empty result and continues)

Replace each block with: log at ERROR level, then `raise`. The outer caller (`derive_ontology_graph` in `app/workers/pipeline.py`) already translates exceptions into `IngestFailed` correctly, so the rethrow propagates to the right handler.

If there are multiple error-swallowing sites, fix every one — any one remaining swallower defeats the PR 0 baseline honesty goal.

- [x] **Step 3: Persist attempted_mode / completed_mode on the legacy stage metrics**

This step writes to `PipelineRun.metrics` (the new JSONB column added in migration 0015 Task 2.2 Step 8). `StageRun` does NOT have a `metrics` column — its per-run fields are first-class columns only. So the "metrics blob" here means the `PipelineRun.metrics` dict, which the legacy `derive_ontology_graph` stage updates in-memory and commits alongside its StageRun row.

In `app/workers/pipeline.py` `derive_ontology_graph` (legacy branch), find the point where the stage handler assembles its metrics dict before writing the StageRun. Add these two keys to `pipeline_run.metrics` under a `legacy_extraction` subkey so Chunk 3's baseline harness can distinguish "layered attempted, single-pass completed via fail-open" from "single-pass only":

```python
# Inside the legacy branch of derive_ontology_graph, after the extraction
# path has decided which mode it used and which mode it actually finished in:
run_metrics = dict(pipeline_run.metrics or {})
legacy_block = dict(run_metrics.get("legacy_extraction", {}))
legacy_block["attempted_mode"] = attempted_mode  # "layered" | "single_pass"
legacy_block["completed_mode"] = completed_mode  # "layered" | "single_pass"
run_metrics["legacy_extraction"] = legacy_block
pipeline_run.metrics = run_metrics
# Mark dirty if your ORM session needs it, then commit as part of the
# existing stage-finalization logic.
```

If the existing code path does not flow through an in-memory `pipeline_run.metrics` update, use whichever persistence helper the legacy branch already uses for its stats — do NOT introduce a new write path. The point is that the data lands on `PipelineRun.metrics`, not how.

- [x] **Step 4: Run the existing extraction tests**

```bash
pytest tests/unit/test_layered_extraction.py tests/unit/test_pipeline.py -v
```

Expected: some tests may fail because they assumed silent fail-open. Fix the tests to assert the loud-failure behavior or mark them xfail with a tracking note.

- [x] **Step 5: Commit**

List every file you actually modified in Steps 1–4 (including the specific test files Step 4 forced you to touch) and stage them explicitly. Do NOT stage `tests/` as a directory — that would pick up unrelated in-flight files.

```bash
# Example — replace with the real list from your working tree:
# git status --short
git add app/config.py app/services/layered_extraction.py app/workers/pipeline.py \
       tests/unit/test_layered_extraction.py tests/unit/test_pipeline.py
# Plus any other test files you had to update in Step 4.
git commit -m "feat(extraction): honest failures on the legacy path (PR 0 equivalent)

- Default graph_layered_shadow_mode=False and
  graph_layered_fail_open_to_single_pass=False per spec §7.3.
- Remove per-pass error swallowing in layered_extraction.py.
- Persist attempted_mode and completed_mode in the stage metrics
  blob so the Chunk 3 baseline harness can detect silent mode drift.

This runs on the LEGACY path (still active after PR 1) and will be
deleted wholesale by PR 3 along with layered_extraction.py. It
establishes a truthful baseline before the switchover per spec §8.3.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.8: PR 1 integration smoke test + final PR 1 exit criteria verification

**Files:**
- Create: `tests/integration/test_pr1_scaffolding_smoke.py`

- [x] **Step 1: Write the PR 1 smoke test**

Create `tests/integration/test_pr1_scaffolding_smoke.py`:

```python
"""PR 1 smoke test: verify all scaffolding is in place and the legacy
path still works end-to-end. Spec §7.3 PR 1 exit criteria."""
import subprocess
from pathlib import Path

def test_bundle_directory_exists():
    p = Path("ontology_bundles/air_defense_v3")
    assert p.is_dir()
    for required in ["ontology.yaml", "manifest.yaml", "coverage.yaml",
                     "validators.py", "derive_rules.py"]:
        assert (p / required).exists()
    for pass_file in ["reference", "radar_domain", "missile_domain",
                      "other_systems", "system_links"]:
        assert (p / "extraction_schemas" / f"{pass_file}.py").exists()

def test_symlink_resolves():
    p = Path("ontology/ontology.yaml")
    assert p.is_symlink()
    assert p.resolve().name == "ontology.yaml"
    assert "ontology_bundles/air_defense_v3" in str(p.resolve())

def test_coverage_checker_passes():
    result = subprocess.run(
        ["python", "tools/check_extraction_coverage.py"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS air_defense_v3" in result.stdout

def test_load_ontology_returns_bundle_ontology():
    from app.services.ontology_templates import load_ontology
    ont = load_ontology()
    assert any(e.get("name") == "RADAR_SYSTEM" for e in ont["entity_types"])

def test_load_bundle_manifest():
    from app.services.ontology_bundles import load_bundle_manifest
    m = load_bundle_manifest("air_defense_v3")
    assert len(m.passes) == 5
    assert {p.name for p in m.passes} == {
        "reference", "radar_domain", "missile_domain",
        "other_systems", "system_links",
    }

def test_migration_applied():
    """v_latest_pass_attempts view and rollback_executed column exist."""
    # Query information_schema via SQLAlchemy
    from sqlalchemy import create_engine, text
    from app.config import get_settings
    engine = create_engine(get_settings().database_url)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'ingest'
              AND table_name = 'stage_runs'
              AND column_name IN ('pass_name', 'execution_status',
                                  'yield_status', 'rollback_executed')
        """))
        cols = {row[0] for row in result}
    assert cols == {"pass_name", "execution_status", "yield_status", "rollback_executed"}
```

- [x] **Step 2: Run the smoke test**

```bash
pytest tests/integration/test_pr1_scaffolding_smoke.py -v
```

Expected: all tests pass.

- [x] **Step 3: Run the full test suite**

```bash
pytest tests/ 2>&1 | tail -30
```

Expected: green. If any failures, fix them before PR 1 review.

- [x] **Step 4: Verify legacy path end-to-end with a canary document**

The existing e2e harness at `tests/e2e/test_full_pipeline.py` already runs a real ingest against the compose stack. Re-run it explicitly in PR 1 mode to prove the legacy path still produces a graph after all Chunk 1 + Chunk 2 scaffolding lands:

```bash
# 1. Bring up the full stack (or restart if already up) on this branch
docker compose up -d postgres redis arcadedb worker api docling-graph

# 2. Apply migration 0015 if it is not already applied
alembic upgrade head

# 3. Run the existing e2e test — it ingests a minimal PDF and asserts
#    the graph is produced
pytest tests/e2e/test_full_pipeline.py -v
```

Expected: test passes. Because `graph_extraction_engine` has not been introduced yet (PR 2), the pipeline takes the legacy branch automatically. Capture the PipelineRun id from the test output (or from `SELECT id, mode FROM ingest.pipeline_runs ORDER BY started_at DESC LIMIT 1;`) and manually confirm:

```bash
docker compose exec postgres psql -U postgres -d eip_arcadedb -c "
  SELECT pr.id, pr.mode, pr.metrics->'legacy_extraction' AS legacy,
         COUNT(dge.id) AS doc_extraction_count
  FROM ingest.pipeline_runs pr
  LEFT JOIN retrieval.document_graph_extractions dge
    ON dge.pipeline_run_id = pr.id
  WHERE pr.started_at > NOW() - INTERVAL '10 minutes'
  GROUP BY pr.id, pr.mode, pr.metrics
  ORDER BY pr.started_at DESC LIMIT 5;
"
```

Expected row: `mode='full'` (or the default the test uses), `legacy` JSON contains `attempted_mode` and `completed_mode` keys from Task 2.7, and `doc_extraction_count >= 1`. Record the output in the PR 1 description so reviewers can reproduce.

Only AFTER this canary passes are the `[x]` items in Task 2.9's PR body allowed to be pre-checked.

- [x] **Step 5: Commit the smoke test**

```bash
git add tests/integration/test_pr1_scaffolding_smoke.py
git commit -m "test(pr1): scaffolding smoke test

Verifies all PR 1 exit criteria from spec §7.3: bundle directory
exists, symlink resolves, coverage checker passes, load_ontology
and load_bundle_manifest work, migration applied. Legacy path
canary verification is manual and logged in the PR description.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 2.9: Open PR 1 against main

- [x] **Step 1: Push the branch**

```bash
git push -u origin feature/extraction-refactor
```

- [x] **Step 2: Open PR 1 via gh cli**

```bash
gh pr create --title "PR 1: Extraction refactor — baseline + scaffolding (strictly additive)" \
  --body "$(cat <<'EOF'
## Summary

PR 1 of the three-PR extraction refactor described in spec
`docs/superpowers/specs/2026-04-10-remove-runtime-template-generation-design.md`.
Strictly additive: the legacy path still runs production documents
end-to-end. No caller has switched to the new code.

Changes:
- ontology_bundles/air_defense_v3/ scaffolding: manifest, coverage,
  validators, 5 extraction schema modules, derive_rules
- identity_fields / identity_scope added to 23 extract-bucket entity
  types in ontology.yaml
- ontology/ontology.yaml now a git-tracked symlink to the bundle path
- load_ontology / load_registry_ontology split per spec §7.3
- Alembic 0015: bundle fields on Source/PipelineRun/DocumentGraphExtraction,
  new StageRun columns, partial unique indexes, v_latest_pass_attempts view
- Coverage checker tool running in CI
- New POST /extract-pass endpoint on docling-graph service (legacy
  /extract-all endpoint is untouched)
- Docker packaging updated to include ontology_bundles
- Honest failures on the legacy path (shadow/fail-open flags default False)
- PR 1 smoke test

Exit criteria per spec §7.3:
- [x] Full test suite passes
- [x] Coverage checker passes on air_defense_v3
- [x] Both images build; smoke import test passes
- [x] Migration applies and downgrades cleanly
- [x] Old /extract-all endpoint still responds
- [x] New /extract-pass endpoint responds for all 5 passes
- [x] End-to-end ingest via legacy path still produces a graph

## Test plan
- [x] Reviewer runs ./scripts/smoke_test_bundle_import.sh locally
- [x] Reviewer runs pytest tests/ and confirms green
- [x] Reviewer ingests one canary document and verifies legacy path
      still produces a graph

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [x] **Step 3: Wait for CI and review**

Once CI is green and the PR is approved, merge.

---

**END OF CHUNK 2.** PR 1 is complete once Task 2.9 merges. Chunk 3 begins PR 2 (switchover behind feature flag).


## Chunk 3: PR 2 — Switchover foundations (new graph primitive, merge module, feature flag)

This chunk is the first half of PR 2. It lands the foundation pieces that the orchestrator rewrite in Chunk 4 will depend on: the new narrower graph-store rollback primitive (residual check #1), the additive `ProvenanceMetadata.pipeline_run_id` field, the full `app/services/extraction_merge.py` module, the `StatusSignals` dataclass that was deferred from Chunk 2, the `graph_extraction_engine` feature flag (defaulting to `legacy`), and the new `IngestDispatchResult` dataclass. Nothing in this chunk actually switches traffic — the feature flag defaults to `legacy` so the legacy path still runs every production ingest after Chunk 3 merges. The orchestrator rewrite itself lives in Chunk 4. Spec §5.4 + §5.6 + §5.7 + §7.4 + residual check #1.

### Task 3.1: Add `delete_extraction_layer_graph_sync` to `GraphStore` protocol

**Files:**
- Modify: `app/services/graph_store.py`
- Create: `tests/unit/test_graph_store_protocol.py`

Residual check #1 protocol half. Adds the method to the backend-agnostic Protocol declaration. Implementation lands in Task 3.2.

- [x] **Step 1: Locate the existing `GraphStore` protocol**

```bash
grep -n "class GraphStore\|delete_document_graph_sync" app/services/graph_store.py
```

Capture the file layout (class definition, existing method signatures). The new method goes after `delete_document_graph_sync`.

- [x] **Step 2: Write the failing contract test**

Create `tests/unit/test_graph_store_protocol.py`:

```python
"""Protocol-level assertions for the GraphStore interface.
Spec residual check #1."""
import inspect
from app.services.graph_store import GraphStore


def test_delete_extraction_layer_graph_sync_is_declared():
    """The new narrower rollback primitive must be part of the protocol."""
    assert hasattr(GraphStore, "delete_extraction_layer_graph_sync")
    sig = inspect.signature(GraphStore.delete_extraction_layer_graph_sync)
    params = list(sig.parameters)
    assert params == ["self", "document_id"], (
        f"Unexpected signature: {params}"
    )
    # Return type matches the existing delete_document_graph_sync for
    # logging parity — both return int (count of deletions).
    assert sig.return_annotation is int, (
        f"Expected -> int, got {sig.return_annotation}"
    )


def test_delete_document_graph_sync_unchanged():
    """The existing broader primitive is kept unchanged for purge callers."""
    assert hasattr(GraphStore, "delete_document_graph_sync")
    sig = inspect.signature(GraphStore.delete_document_graph_sync)
    params = list(sig.parameters)
    assert params == ["self", "document_id"], (
        f"Unexpected signature: {params}"
    )
    assert sig.return_annotation is int, (
        f"delete_document_graph_sync regression: return type changed "
        f"from int to {sig.return_annotation}"
    )
```

Run:
```bash
pytest tests/unit/test_graph_store_protocol.py -v
```

Expected: `test_delete_extraction_layer_graph_sync_is_declared` FAILS (method does not yet exist).

- [x] **Step 3: Add the method to the protocol**

Insert after the existing `delete_document_graph_sync` declaration at `app/services/graph_store.py:614`. The existing primitive returns `int` (deletion count); the new primitive returns `int` for logging parity so both code paths can emit counts uniformly.

```python
def delete_extraction_layer_graph_sync(self, document_id: str) -> int:
    """Delete only this document's extraction-layer graph state.
    Returns the total count of vertices + edges deleted (for logging
    parity with delete_document_graph_sync).

    MUST delete:
      - document-scoped extracted entity vertices (identity includes document_id)
      - domain edges tagged with document_id in provenance metadata
      - HAS_PROVENANCE edges whose target is the structural Document
        vertex with this document_id, REGARDLESS of whether the source
        vertex is document-scoped or global-scoped. The edges are
        deleted; global source vertices are preserved.
      - structural edges produced by derive_rules in phase 4
        (MENTIONED_IN from extracted entities to TextChunks, tagged
        with document_id / source=derive_rules)

    MUST NOT delete:
      - chunks (TextChunk, ImageChunk)
      - the structural Document vertex itself
      - global-scoped entity vertices (PLATFORM, RADAR_SYSTEM, etc.)
      - HAS_PROVENANCE edges from those global vertices to OTHER documents

    Any alternative backend implementing GraphStore must honor this
    contract. This is narrower than delete_document_graph_sync, which
    is reserved for purge/full-document-delete callers.

    Spec §6.8 + residual check #1.
    """
    ...
```

- [x] **Step 4: Run tests — expect pass**

```bash
pytest tests/unit/test_graph_store_protocol.py -v
```

Expected: both tests pass. The concrete `ArcadeDBGraphStore` implementation is in Task 3.2 — the Protocol declaration alone is enough for this test.

- [x] **Step 5: Commit**

```bash
git add app/services/graph_store.py tests/unit/test_graph_store_protocol.py
git commit -m "feat(graph_store): add delete_extraction_layer_graph_sync to protocol

Per spec residual check #1. The existing delete_document_graph_sync
over-deletes for extraction-stage rollback (it removes chunks, image
chunks, and the structural Document vertex, which must be preserved).
This commit adds the narrower primitive to the backend-agnostic
GraphStore Protocol with the full contract docstring. The concrete
ArcadeDB implementation lands in the next commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 3.2: Implement `delete_extraction_layer_graph_sync` in `ArcadeDBGraphStore`

**Files:**
- Modify: `app/services/arcadedb_graph.py`
- Create: `tests/integration/test_arcadedb_extraction_rollback.py`

Residual check #1 implementation half. This is the ArcadeDB-specific SQL for the narrower rollback primitive.

- [x] **Step 1: Read the existing `delete_document_graph_sync` impl as a style reference**

```bash
grep -n "delete_document_graph_sync\|_build_delete_document_graph_sql" app/services/arcadedb_graph.py
```

Read the block starting at line ~1975 (actual location — spec reference is §4.6 of its own prose). Study how it:
- Binds parameters with `:name` syntax via `self._client.command_sync(db, "sql", query, params={...})`.
- Handles RID lookups (they're generally not passed as typed RID bindings — the existing helper issues SQL that resolves them inline).
- Iterates vertex/edge classes via a list.
- Returns the deletion count.

Your new method must match these idioms exactly. Do NOT reuse this method's SQL — it over-deletes for rollback's scope — but DO reuse its parameter-binding and class-iteration style. Mismatches between the two methods will make future maintenance harder.

- [x] **Step 2: Inventory current entity and edge class names in ArcadeDB**

```bash
docker compose exec arcadedb /opt/arcadedb/bin/console.sh -c "
  connect remote:localhost/eip_graph root PASSWORD;
  SELECT name FROM schema:types WHERE type = 'vertex' ORDER BY name;
  SELECT name FROM schema:types WHERE type = 'edge' ORDER BY name;
"
```

Capture the vertex and edge class lists. Identify:
- **Document-scoped vertex classes** (those whose identity includes `document_id`): typically `SECTION`, `FIGURE`, `TABLE`, `ASSERTION`, and similar structural-domain entities. Cross-reference with the `identity_scope: document` entries in `ontology_bundles/air_defense_v3/ontology.yaml` from Chunk 1 Task 1.1.
- **Global-scoped vertex classes**: `RADAR_SYSTEM`, `MISSILE_SYSTEM`, `PLATFORM`, etc. — these are NEVER deleted by rollback.
- **Domain edge classes**: anything that carries `document_id` in its properties via `ProvenanceMetadata`.
- **Structural edge classes**: `HAS_PROVENANCE`, `MENTIONED_IN`, `CONTAINS_TEXT`, `CONTAINS_IMAGE`, `NEXT_CHUNK`. Rollback touches only `HAS_PROVENANCE` (target-scoped) and `MENTIONED_IN` (where `source='derive_rules'` and `document_id` matches).

Write the lists into a comment block at the top of the new method so future maintainers can cross-check.

- [x] **Step 3a: Locate or create the integration-test ArcadeDB fixture**

```bash
grep -rn "ArcadeDBGraphStore\|ArcadeDBClient\|arcadedb_graph_store" tests/ | head -20
find tests/ -name conftest.py
```

Capture the result. If a usable `ArcadeDBGraphStore` fixture already exists in some `conftest.py`, import it in Step 3b. If not (likely for `tests/integration/`), create `tests/integration/conftest.py` with a minimal connected-store fixture along these lines:

```python
# tests/integration/conftest.py (only if a fixture doesn't already exist)
import os
import pytest
from app.config import get_settings
from app.services.arcadedb_client import ArcadeDBClient
from app.services.arcadedb_graph import ArcadeDBGraphStore


@pytest.fixture(scope="module")
def arcadedb_store():
    s = get_settings()
    client = ArcadeDBClient(
        base_url=s.arcadedb_url,
        username=s.arcadedb_username,
        password=s.arcadedb_password,
    )
    store = ArcadeDBGraphStore(client=client, database=s.arcadedb_database)
    yield store
    # No teardown: each test isolates itself by using a unique document_id
    # (e.g., UUID4 per test) and cleans up after itself via the rollback
    # primitive under test.
```

If the connection helper constructor signatures differ from what's shown, read `app/services/arcadedb_graph.py` at the class definition and mirror what it already expects. Do NOT invent environment variable names — reuse what `get_settings()` already exposes.

- [x] **Step 3b: Write the integration test (it must fail)**

Create `tests/integration/test_arcadedb_extraction_rollback.py`:

```python
"""Integration test for the narrower rollback primitive.
Spec §6.8 + residual check #1 test setup."""
import uuid
import pytest

from app.services.graph_store import (
    NodeRecord, RelationshipRecord, ProvenanceMetadata,
)


def test_delete_extraction_layer_graph_sync(arcadedb_store):
    """Set up a mixed-scope document, call the rollback primitive, assert
    the exact set of things that must / must not survive."""
    store = arcadedb_store
    doc_id = f"rollback-test-{uuid.uuid4().hex}"
    other_doc_id = f"rollback-test-{uuid.uuid4().hex}"

    _seed_rollback_fixture(store, doc_id, other_doc_id)

    # Act
    deleted_count = store.delete_extraction_layer_graph_sync(doc_id)
    assert deleted_count > 0, "rollback primitive must report what it deleted"

    # Assert — the MUST-delete set
    assert _count_vertices(store, "SECTION", document_id=doc_id) == 0
    assert _count_has_provenance_to_doc(store, doc_id) == 0
    assert _count_edges(store, "INSTALLED_ON", document_id=doc_id) == 0
    assert _count_edges(
        store, "MENTIONED_IN",
        document_id=doc_id, source="derive_rules",
    ) == 0

    # Assert — the MUST-NOT-delete set
    assert _count_vertices(store, "TextChunk", document_id=doc_id) > 0
    assert _count_vertices(store, "ImageChunk", document_id=doc_id) > 0
    assert _count_vertices(store, "Document", document_id=doc_id) == 1
    assert _count_vertices(store, "RADAR_SYSTEM") > 0  # global — never deleted
    # cross-document preservation: the global RADAR_SYSTEM still has a
    # HAS_PROVENANCE edge to the OTHER Document vertex.
    assert _count_has_provenance_to_doc(store, other_doc_id) > 0

    # Cleanup: remove the other_doc_id fixture so repeated runs don't
    # accumulate test data.
    store.delete_document_graph_sync(other_doc_id)


def _seed_rollback_fixture(store, doc_id, other_doc_id):
    """Populate the graph with the mixed-scope fixture. Uses the real
    upsert_* paths so the fixture matches writer behavior exactly."""
    # 1. Create two structural Document vertices via whatever helper the
    #    codebase already uses. If a helper doesn't exist, issue a direct
    #    SQL command through store._client — Document vertex creation is
    #    a one-liner in ArcadeDB SQL. The existing pipeline's
    #    prepare_document stage already does this; mimic that call.
    store._client.command_sync(
        store._database, "sql",
        "CREATE VERTEX Document SET document_id = :doc_id",
        params={"doc_id": doc_id},
    )
    store._client.command_sync(
        store._database, "sql",
        "CREATE VERTEX Document SET document_id = :doc_id",
        params={"doc_id": other_doc_id},
    )

    # 2. Create a TextChunk and an ImageChunk for this document
    store._client.command_sync(
        store._database, "sql",
        "CREATE VERTEX TextChunk SET document_id = :doc_id, text = 'hello world'",
        params={"doc_id": doc_id},
    )
    store._client.command_sync(
        store._database, "sql",
        "CREATE VERTEX ImageChunk SET document_id = :doc_id",
        params={"doc_id": doc_id},
    )

    # 3. Create a document-scoped extracted entity (SECTION) via the real
    #    writer path so the HAS_PROVENANCE edge gets auto-created.
    section_node = NodeRecord(
        entity_type="SECTION",
        identity_fields={"heading": "Intro", "document_id": doc_id},
        name="Intro",
        properties={"heading": "Intro", "page_start": 1, "page_end": 2},
    )
    store.upsert_nodes_batch_sync(
        [section_node],
        ProvenanceMetadata(document_id=doc_id),
    )

    # 4. Create a global RADAR_SYSTEM with HAS_PROVENANCE edges to BOTH
    #    Document vertices. Upsert via provenance=doc_id first, then
    #    again with provenance=other_doc_id so the existing writer
    #    creates both HAS_PROVENANCE edges.
    radar_node = NodeRecord(
        entity_type="RADAR_SYSTEM",
        identity_fields={"system_name": "TestRadar-" + doc_id[-6:]},
        name="TestRadar",
        properties={"system_name": "TestRadar-" + doc_id[-6:]},
    )
    store.upsert_nodes_batch_sync([radar_node], ProvenanceMetadata(document_id=doc_id))
    store.upsert_nodes_batch_sync([radar_node], ProvenanceMetadata(document_id=other_doc_id))

    # 5. Domain edge (INSTALLED_ON) between RADAR_SYSTEM and another
    #    global (create a PLATFORM first). Tagged with provenance doc_id.
    platform_node = NodeRecord(
        entity_type="PLATFORM",
        identity_fields={"name": "TestPlatform-" + doc_id[-6:]},
        name="TestPlatform",
        properties={"name": "TestPlatform-" + doc_id[-6:]},
    )
    store.upsert_nodes_batch_sync([platform_node], ProvenanceMetadata(document_id=doc_id))
    installed_on = RelationshipRecord(
        from_type="RADAR_SYSTEM",
        from_identity={"system_name": "TestRadar-" + doc_id[-6:]},
        to_type="PLATFORM",
        to_identity={"name": "TestPlatform-" + doc_id[-6:]},
        rel_type="INSTALLED_ON",
    )
    store.upsert_relationships_batch_sync(
        [installed_on],
        ProvenanceMetadata(document_id=doc_id),
    )

    # 6. MENTIONED_IN edge from RADAR_SYSTEM to the TextChunk, tagged
    #    source='derive_rules' and document_id=doc_id. Use the existing
    #    structural-edge helper at app/services/graph_store.py:528.
    #    (Look up the RIDs via a short SELECT since create_structural_edge_sync
    #    uses from_id / to_id, not identity dicts.)
    radar_rid = store._client.query_sync(
        store._database, "sql",
        "SELECT @rid AS rid FROM RADAR_SYSTEM WHERE system_name = :nm",
        params={"nm": "TestRadar-" + doc_id[-6:]},
    )[0]["rid"]
    chunk_rid = store._client.query_sync(
        store._database, "sql",
        "SELECT @rid AS rid FROM TextChunk WHERE document_id = :doc_id LIMIT 1",
        params={"doc_id": doc_id},
    )[0]["rid"]
    store.create_structural_edge_sync(
        from_id=radar_rid,
        to_id=chunk_rid,
        rel_type="MENTIONED_IN",
        properties={"document_id": doc_id, "source": "derive_rules"},
    )


def _count_vertices(store, class_name, **filters):
    """Issue a COUNT query scoped by class_name and any equality filters."""
    where = " AND ".join(f"{k} = :{k}" for k in filters)
    where_clause = f" WHERE {where}" if where else ""
    rows = store._client.query_sync(
        store._database, "sql",
        f"SELECT count(*) AS n FROM {class_name}{where_clause}",
        params=filters or None,
    )
    return int(rows[0]["n"]) if rows else 0


def _count_edges(store, class_name, **filters):
    """Issue a COUNT query scoped by edge class and any equality filters."""
    where = " AND ".join(f"{k} = :{k}" for k in filters)
    where_clause = f" WHERE {where}" if where else ""
    rows = store._client.query_sync(
        store._database, "sql",
        f"SELECT count(*) AS n FROM {class_name}{where_clause}",
        params=filters or None,
    )
    return int(rows[0]["n"]) if rows else 0


def _count_has_provenance_to_doc(store, doc_id):
    """Count HAS_PROVENANCE edges whose target is the Document vertex with
    the given document_id. Mirrors the scoping of the rollback primitive's
    deletion query so we assert exactly what was deleted."""
    rows = store._client.query_sync(
        store._database, "sql",
        "SELECT count(*) AS n FROM HAS_PROVENANCE "
        "WHERE in.document_id = :doc_id",
        params={"doc_id": doc_id},
    )
    return int(rows[0]["n"]) if rows else 0
```

The fixture uses real writer paths (`upsert_nodes_batch_sync`, `upsert_relationships_batch_sync`, `create_structural_edge_sync`) deliberately — seeding with hand-rolled SQL would diverge from what the rollback primitive needs to undo. If any of the existing writer calls raise because the vertex/edge class does not exist in the test DB, create it via `CREATE VERTEX TYPE <name>` / `CREATE EDGE TYPE <name>` in a per-module setup fixture. The ontology-schema-sync stage handles this in production; the test fixture may need to invoke it explicitly.

Run:
```bash
pytest tests/integration/test_arcadedb_extraction_rollback.py -v
```

Expected: test fails with `AttributeError: 'ArcadeDBGraphStore' object has no attribute 'delete_extraction_layer_graph_sync'`.

- [x] **Step 4: Implement `delete_extraction_layer_graph_sync` in `ArcadeDBGraphStore`**

Add the method to `app/services/arcadedb_graph.py`. **The SQL shown below is a starting point, not drop-in code** — ArcadeDB's SQL dialect has quirks (RID binding, `DELETE VERTEX` vs `DELETE FROM`, parameter syntax) that must be verified against the ArcadeDB manual (`ArcadeDB Manual.pdf` in the repo root) and against the existing `_build_delete_document_graph_sql` helper's actual SQL. Read that helper before writing this one — it shows the idioms the team already relies on.

Skeleton based on spec residual check #1:

```python
def delete_extraction_layer_graph_sync(self, document_id: str) -> None:
    """Narrower rollback primitive. Spec §6.8 + residual check #1.

    Deleted: document-scoped extracted entity vertices, domain edges
    tagged with document_id, HAS_PROVENANCE edges targeting this
    document's Document vertex (including edges from global sources),
    MENTIONED_IN edges from derive_rules tagged with document_id.

    Preserved: chunks, the structural Document vertex, global vertices,
    HAS_PROVENANCE edges to other documents.

    Vertex / edge class inventory captured in Task 3.2 Step 2:
    # document-scoped vertex classes: SECTION, FIGURE, TABLE, ASSERTION
    # global-scoped vertex classes:   RADAR_SYSTEM, MISSILE_SYSTEM, PLATFORM, ...
    # domain edge classes:            HAS_ANTENNA, INSTALLED_ON, FIRES, ...
    # touched structural edges:       HAS_PROVENANCE, MENTIONED_IN
    """
    # 1. Look up the structural Document RID
    doc_rid_rows = self._client.query_sync(
        self._database, "sql",
        "SELECT @rid AS rid FROM Document WHERE document_id = :doc_id",
        params={"doc_id": document_id},
    )
    if not doc_rid_rows:
        # No Document vertex for this id — nothing to clean up.
        return
    doc_rid = doc_rid_rows[0]["rid"]

    # 2. Delete HAS_PROVENANCE edges incident to this Document vertex.
    #    Scoped by the edge's target (the Document RID), not by
    #    edge-local document_id, because current provenance writers do
    #    not tag the edge itself with document_id. This removes edges
    #    from BOTH document-scoped AND global-scoped sources. Global
    #    source vertices are untouched.
    self._client.command_sync(
        self._database, "sql",
        "DELETE EDGE HAS_PROVENANCE WHERE in = :doc_rid",
        params={"doc_rid": doc_rid},
    )

    # 3. Delete document-scoped extracted entity vertices. Iterate the
    #    list from Step 2 inventory. Only classes whose identity_scope
    #    is 'document' appear here.
    for vertex_class in _DOCUMENT_SCOPED_ENTITY_CLASSES:
        self._client.command_sync(
            self._database, "sql",
            f"DELETE VERTEX FROM {vertex_class} WHERE document_id = :doc_id",
            params={"doc_id": document_id},
        )

    # 4. Delete domain edges tagged with this document_id. These edges
    #    carry document_id via ProvenanceMetadata. Iterate each domain
    #    edge class from the inventory.
    for edge_class in _DOMAIN_EDGE_CLASSES:
        self._client.command_sync(
            self._database, "sql",
            f"DELETE EDGE FROM {edge_class} WHERE document_id = :doc_id",
            params={"doc_id": document_id},
        )

    # 5. Delete derive_rules-originated MENTIONED_IN structural edges.
    self._client.command_sync(
        self._database, "sql",
        "DELETE EDGE MENTIONED_IN "
        "WHERE document_id = :doc_id AND source = 'derive_rules'",
        params={"doc_id": document_id},
    )


# Module-level class inventories. These MUST come from two sources of
# truth that agree:
#   1. ontology_bundles/air_defense_v3/ontology.yaml (the entity and
#      relationship declarations with identity_scope: document and the
#      validation_matrix rows for domain edges)
#   2. The live `schema:types` query from Step 2
# If the two disagree, fix the gap in ontology.yaml or the schema sync
# stage — do NOT paper over it here with a hardcoded list.
_DOCUMENT_SCOPED_ENTITY_CLASSES: tuple[str, ...] = (
    "SECTION", "FIGURE", "TABLE", "ASSERTION",
    # Plus every other entity whose identity_scope is 'document' per
    # ontology.yaml Chunk 1 Task 1.1. Read ontology.yaml to enumerate.
)
_DOMAIN_EDGE_CLASSES: tuple[str, ...] = (
    # Every edge class that carries document_id in its properties via
    # ProvenanceMetadata. Enumerate from the validation_matrix in
    # ontology.yaml — these are the relationships extracted by the
    # domain passes (radar_domain, missile_domain, other_systems,
    # system_links). Example: HAS_ANTENNA, USES_WAVEFORM, FIRES,
    # INSTALLED_ON, PAIRED_WITH, etc. Do not include structural edge
    # classes (HAS_PROVENANCE, MENTIONED_IN, CONTAINS_TEXT) here —
    # those have their own deletion paths above.
)
```

Wire the class-name lists to the actual inventory from Step 2. Do NOT hardcode names not confirmed by BOTH ontology.yaml AND the live inventory query. An inventory-check assertion at module import time (raising if the two disagree) is a reasonable belt-and-suspenders addition, but not required for the first commit.

- [x] **Step 5: Run the integration test**

```bash
docker compose up -d arcadedb postgres
pytest tests/integration/test_arcadedb_extraction_rollback.py -v
```

Expected: test passes. If any assertion fails, fix the SQL — do NOT relax the test.

- [x] **Step 6: Commit**

```bash
git add app/services/arcadedb_graph.py tests/integration/test_arcadedb_extraction_rollback.py
git commit -m "feat(arcadedb): implement delete_extraction_layer_graph_sync

Per spec §6.8 + residual check #1. Narrower rollback primitive scoped
to extraction-layer state only: document-scoped entities, domain edges
tagged with document_id, HAS_PROVENANCE edges targeted at this
document's Document vertex (including from global sources), and
derive_rules-originated MENTIONED_IN edges.

Preserves chunks, ImageChunks, the structural Document vertex itself,
global entity vertices, and cross-document HAS_PROVENANCE. The existing
delete_document_graph_sync at arcadedb_graph.py:401 is unchanged and
remains reserved for purge / full-document-delete callers.

Integration test covers the mixed-scope fixture from residual check #1
Step 4.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 3.3: Extend `ProvenanceMetadata` with `pipeline_run_id`

**Files:**
- Modify: `app/services/graph_store.py`
- Modify: `tests/unit/test_graph_store_protocol.py`

Additive change per spec §5.6 ProvenanceMetadata extension. The new field is optional — existing callers remain unchanged.

- [x] **Step 1: Write the failing field-presence test**

Append to `tests/unit/test_graph_store_protocol.py`:

```python
def test_provenance_metadata_has_pipeline_run_id():
    """Spec §5.6: ProvenanceMetadata gains an optional pipeline_run_id field."""
    from dataclasses import fields
    from app.services.graph_store import ProvenanceMetadata
    field_names = {f.name for f in fields(ProvenanceMetadata)}
    assert "pipeline_run_id" in field_names

    # Optional — default None, no positional requirement
    meta = ProvenanceMetadata(document_id="doc-1")
    assert meta.pipeline_run_id is None

    # Explicit value still works
    meta2 = ProvenanceMetadata(document_id="doc-1", pipeline_run_id="run-1")
    assert meta2.pipeline_run_id == "run-1"


def test_provenance_metadata_existing_fields_unchanged():
    """The pipeline_run_id addition must not disturb existing defaults."""
    from app.services.graph_store import ProvenanceMetadata
    meta = ProvenanceMetadata(document_id="doc-1")
    # Existing defaults per spec §5.6 note: page_numbers default_factory=list.
    assert meta.page_numbers == []
```

Run:
```bash
pytest tests/unit/test_graph_store_protocol.py::test_provenance_metadata_has_pipeline_run_id -v
```

Expected: FAIL — `pipeline_run_id` is not a field of `ProvenanceMetadata` yet.

- [x] **Step 2: Add the field**

Open `app/services/graph_store.py` at the `ProvenanceMetadata` dataclass (lines 19-26 in the current file). The existing fields are:

```python
@dataclass
class ProvenanceMetadata:
    document_id: str
    page_numbers: list[int] = field(default_factory=list)
    upload_datetime: str | None = None       # NOTE: str, not datetime
    document_datetime: str | None = None     # NOTE: str, not datetime
```

Append ONE new field at the end, preserving every other field's existing type (`str | None`, NOT `datetime | None` — the current storage is already ISO strings):

```python
@dataclass
class ProvenanceMetadata:
    document_id: str
    page_numbers: list[int] = field(default_factory=list)
    upload_datetime: str | None = None
    document_datetime: str | None = None
    pipeline_run_id: str | None = None       # NEW in PR 2 — optional, additive
```

Do NOT touch the existing field types or defaults. Do NOT touch `_create_provenance_edges_batch_sync` — this field is not required by the current auto-creation path (per §5.6 note).

- [x] **Step 3: Run the tests**

```bash
pytest tests/unit/test_graph_store_protocol.py -v
```

Expected: all pass.

- [x] **Step 4: Run the full suite to catch existing callers**

```bash
pytest tests/ -x 2>&1 | tail -20
```

Expected: green. Additive changes to an Optional field with default `None` cannot break existing positional or keyword callers.

- [x] **Step 5: Commit**

```bash
git add app/services/graph_store.py tests/unit/test_graph_store_protocol.py
git commit -m "feat(graph_store): extend ProvenanceMetadata with pipeline_run_id

Per spec §5.6 ProvenanceMetadata extension. Additive only — new field
is Optional[str] with default None. Existing callers are untouched
because they do not need to pass this field. Used by the new
orchestrator in Chunk 4 Task 4.x to correlate provenance writes
with the PipelineRun that produced them.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 3.4: Write `app/services/extraction_merge.py` (TDD)

**Files:**
- Create: `app/services/extraction_merge.py`
- Create: `tests/unit/test_extraction_merge.py`

Spec §3.6 + §3.7 + §3.9 + §6.2 + §6.7. This is the biggest single task in the chunk — the full merge module with every dataclass, enum, and function the orchestrator will consume. Use TDD discipline: write tests first for each function, watch them fail, implement, watch pass.

This task owns `LogicalIdentity`, `PassResult`, `MergedEntityRecord`, `MergedEdgeRecord`, `MergedExtraction`, `ChunkForDerivation`, `DerivedEdge`, `RelationshipRejectionReason`, `merge_and_resolve`, `build_display_label`, `classify_yield`, `classify_yield_from_counts`, and `YieldStatus`.

- [x] **Step 1: Write failing tests for `RelationshipRejectionReason` (rule 14 coverage)**

Create `tests/unit/test_extraction_merge.py` with the section that satisfies spec §2 checker rule 14:

```python
"""Tests for app/services/extraction_merge.py.
Spec §3.6 + §3.7 + §3.9 + §6.2 + §6.7, and satisfies checker rule 14
(every RelationshipRejectionReason enum value has at least one test fixture)."""
import pytest

from app.services.extraction_merge import (
    RelationshipRejectionReason,
    LogicalIdentity,
    PassResult,
    MergedEntityRecord,
    MergedEdgeRecord,
    MergedExtraction,
    merge_and_resolve,
    build_display_label,
    classify_yield,
    classify_yield_from_counts,
    YieldStatus,
)


# --- Shared fixtures -------------------------------------------------------

# Minimal slice of ontology.yaml sufficient to run merge_and_resolve.
# Only the entity_types, identity_fields/identity_scope, and
# validation_matrix rows relevant to the tests below need to be present.
MINIMAL_ONTOLOGY = {
    "entity_types": [
        {
            "name": "RADAR_SYSTEM",
            "identity_fields": ["system_name"],
            "identity_scope": "global",
            "properties": ["system_name", "nomenclature"],
        },
        {
            "name": "PLATFORM",
            "identity_fields": ["name"],
            "identity_scope": "global",
            "properties": ["name"],
        },
        {
            "name": "SECTION",
            "identity_fields": ["heading", "page_start"],
            "identity_scope": "document",
            "properties": ["heading", "page_start", "page_end"],
        },
    ],
    "validation_matrix": [
        {"source": "RADAR_SYSTEM", "relationship": "INSTALLED_ON", "target": "PLATFORM"},
        # Deliberately omit (RADAR_SYSTEM, LOVES, PLATFORM) — used to test INVALID_TRIPLE.
    ],
}


def _make_pass_result(pass_name, entities, relationships, rejections=None):
    """Build a PassResult-like stub that satisfies the merge consumer's
    interface without requiring a real generated Pydantic template.

    `entities` is a list of (type_name, identity_dict, properties_dict).
    `relationships` is a list of dicts with rel_type, from_identity,
    to_identity (and optionally from_ref_id / to_ref_id / confidence).
    """
    from types import SimpleNamespace
    from app.services.extraction_merge import (
        PassResult, ExtractionMetadata,
    )

    # Stub template_instance exposing iter_entities_of_type via SimpleNamespace
    entities_by_type: dict[str, list[SimpleNamespace]] = {}
    for ent_type, identity, props in entities:
        entities_by_type.setdefault(ent_type, []).append(
            SimpleNamespace(**{**identity, **props})
        )

    template = SimpleNamespace(
        **{f"{k.lower()}_list": v for k, v in entities_by_type.items()},
        relationships=[SimpleNamespace(**r) for r in relationships],
    )

    return PassResult(
        pass_name=pass_name,
        template_instance=template,
        metadata=ExtractionMetadata(
            schema_size_chars=1000,
            structured_output_mode="strict",
        ),
        pre_merge_rejections=rejections or [],
    )


# --- Rule 14 coverage: one test per enum value -----------------------------

def test_rejection_invalid_triple():
    """(from_type, rel_type, to_type) not in validation_matrix → INVALID_TRIPLE.

    Worked example. Other rejection tests follow the same structure — copy
    this test as a template and tweak the input to trigger a different reason.
    """
    from app.services.extraction_merge import merge_and_resolve, RelationshipRejectionReason

    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {"nomenclature": "Triumf"}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[
            {
                "rel_type": "LOVES",  # not in validation_matrix
                "from_identity": {"system_name": "S-400"},
                "to_identity": {"name": "Truck-1"},
                "from_type": "RADAR_SYSTEM",
                "to_type": "PLATFORM",
                "confidence": 0.9,
            },
        ],
    )

    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )

    # The invalid triple must be rejected, not merged.
    assert len(merged.edges) == 0
    assert any(
        reason == RelationshipRejectionReason.INVALID_TRIPLE
        for _, _, reason in merged.rejected_edges
    )


def test_rejection_missing_rel_type():
    """rel_type is None or not in pass's allowed enum → MISSING_REL_TYPE.
    Mirror test_rejection_invalid_triple above; set relationships[0]['rel_type']=None."""
    ...


def test_rejection_invalid_identity_payload():
    """from_identity / to_identity missing required keys or type-incompatible.
    Template: omit a required identity field (e.g. drop 'system_name') from from_identity."""
    ...


def test_rejection_unknown_ref_id():
    """from_ref_id / to_ref_id doesn't match any upstream_entities entry (system_links only).
    Template: use a system_links-pass fixture with a ref_id that is not in upstream refs."""
    ...


def test_rejection_from_endpoint_not_found():
    """Resolved from-side LogicalIdentity doesn't match any merged entity.
    Template: emit a relationship whose from_identity is a RADAR_SYSTEM that was never emitted as an entity in any pass."""
    ...


def test_rejection_to_endpoint_not_found():
    """Same for target side.
    Template: mirror test_rejection_from_endpoint_not_found but on the to side."""
    ...


def _fake_manifest(pass_names):
    """Build a minimal BundleManifest stub for merge consumption.
    Only fields accessed by merge_and_resolve need to be populated."""
    from types import SimpleNamespace
    return SimpleNamespace(
        bundle_key="air_defense_v3",
        passes=[
            SimpleNamespace(
                name=name,
                entities=[],
                bridge_entities=[],
                relationships=[],
                kind="source_centric",
            )
            for name in pass_names
        ],
    )


def test_every_rejection_reason_has_a_test():
    """Meta-check: every enum value has at least one test function above."""
    tested = {
        RelationshipRejectionReason.MISSING_REL_TYPE,
        RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD,
        RelationshipRejectionReason.UNKNOWN_REF_ID,
        RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND,
        RelationshipRejectionReason.TO_ENDPOINT_NOT_FOUND,
        RelationshipRejectionReason.INVALID_TRIPLE,
    }
    assert tested == set(RelationshipRejectionReason), (
        "RelationshipRejectionReason has values without test fixtures; "
        "add a test for each new value to satisfy checker rule 14"
    )
```

Each `test_rejection_*` must build a minimal fixture that triggers that specific reason when fed into `merge_and_resolve`, and assert the rejection appears in `merged.rejected_edges` with the right reason.

- [x] **Step 2: Write failing tests for `LogicalIdentity` identity dict helpers**

Append to `tests/unit/test_extraction_merge.py`:

```python
def test_logical_identity_values_dict_excludes_document_id():
    """identity_values_dict returns only ontology-declared identity fields."""
    li = LogicalIdentity(
        entity_type="SECTION",
        identity_field_names=("heading", "page_start"),
        identity_tuple=("Executive Summary", 1),
        scope="document",
        document_id="doc-123",
    )
    vals = li.identity_values_dict()
    assert vals == {"heading": "Executive Summary", "page_start": 1}
    assert "document_id" not in vals


def test_logical_identity_upsert_dict_adds_document_id_for_document_scope():
    li = LogicalIdentity(
        entity_type="SECTION",
        identity_field_names=("heading", "page_start"),
        identity_tuple=("Executive Summary", 1),
        scope="document",
        document_id="doc-123",
    )
    d = li.as_upsert_identity_dict()
    assert d == {"heading": "Executive Summary", "page_start": 1, "document_id": "doc-123"}


def test_logical_identity_upsert_dict_omits_document_id_for_global_scope():
    li = LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name",),
        identity_tuple=("S-400",),
        scope="global",
        document_id=None,
    )
    d = li.as_upsert_identity_dict()
    assert d == {"system_name": "S-400"}
    assert "document_id" not in d


def test_logical_identity_document_scope_requires_document_id():
    """document-scoped identity without document_id is a bug."""
    li = LogicalIdentity(
        entity_type="SECTION",
        identity_field_names=("heading",),
        identity_tuple=("Intro",),
        scope="document",
        document_id=None,
    )
    with pytest.raises(AssertionError):
        li.as_upsert_identity_dict()
```

- [x] **Step 3: Write failing tests for `build_display_label`**

```python
def test_build_display_label_prefers_system_name_from_identity():
    label = build_display_label(
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "S-400"},
        properties={"nomenclature": "Triumf"},
    )
    assert label == "S-400"


def test_build_display_label_joins_multiple_identity_values():
    label = build_display_label(
        entity_type="SPECIFICATION",
        identity_values={"parameter": "frequency", "value": "2.4GHz"},
        properties={},
    )
    assert label == "frequency / 2.4GHz"


def test_build_display_label_falls_back_to_properties():
    label = build_display_label(
        entity_type="SECTION",
        identity_values={},
        properties={"heading": "Intro"},
    )
    assert label == "Intro"


def test_build_display_label_deterministic_hash_fallback():
    """No name-like key, no identity values, no properties → hash fallback."""
    label1 = build_display_label(
        entity_type="PROPULSION_STACK",
        identity_values={"total_burntime_s": 12.5},
        properties={"propellant_type": "solid"},
    )
    # Same inputs must produce same output
    label2 = build_display_label(
        entity_type="PROPULSION_STACK",
        identity_values={"total_burntime_s": 12.5},
        properties={"propellant_type": "solid"},
    )
    assert label1 == label2
    # When no name-like key exists in identity but identity has values,
    # the helper joins them — per spec §3.9 step 2.
    assert "12.5" in label1
```

- [x] **Step 4: Write failing tests for `classify_yield_from_counts`**

```python
# Per spec §6.2 precedence (top wins)

def test_classify_yield_degraded_when_rejection_ratio_high():
    y = classify_yield_from_counts(primary=2, bridge=0, extracted_rels=1, rejected_rels=3)
    assert y == YieldStatus.DEGRADED


def test_classify_yield_degraded_requires_min_total_rels():
    """total_rels < 4 cannot be DEGRADED regardless of ratio."""
    y = classify_yield_from_counts(primary=0, bridge=0, extracted_rels=0, rejected_rels=3)
    # Below threshold — falls through to EMPTY
    assert y == YieldStatus.EMPTY


def test_classify_yield_empty_when_nothing_extracted():
    y = classify_yield_from_counts(primary=0, bridge=0, extracted_rels=0, rejected_rels=0)
    assert y == YieldStatus.EMPTY


def test_classify_yield_bridges_only_when_primary_zero_and_bridge_present():
    y = classify_yield_from_counts(primary=0, bridge=2, extracted_rels=0, rejected_rels=0)
    assert y == YieldStatus.BRIDGES_ONLY


def test_classify_yield_hit_fallthrough():
    y = classify_yield_from_counts(primary=3, bridge=0, extracted_rels=2, rejected_rels=0)
    assert y == YieldStatus.HIT
```

- [x] **Step 5: Write failing tests for `merge_and_resolve` key properties**

```python
def test_merge_collapses_bridge_entity_across_passes():
    """Same LogicalIdentity from two passes produces ONE MergedEntityRecord
    with both pass names in pass_origins. Worked example."""
    pass_a = _make_pass_result(
        pass_name="radar_domain",
        entities=[("PLATFORM", {"name": "Truck-1"}, {"name": "Truck-1"})],
        relationships=[],
    )
    pass_b = _make_pass_result(
        pass_name="missile_domain",
        entities=[("PLATFORM", {"name": "Truck-1"}, {"name": "Truck-1"})],
        relationships=[],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_a, "missile_domain": pass_b},
        manifest=_fake_manifest(["radar_domain", "missile_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    platforms = [e for e in merged.entities if e.identity.entity_type == "PLATFORM"]
    assert len(platforms) == 1
    assert platforms[0].pass_origins == {"radar_domain", "missile_domain"}


def test_merge_confidence_uses_highest_across_passes():
    """Multiple passes producing same entity → merged confidence = max().
    Template: copy test_merge_collapses_bridge_entity_across_passes, set
    pass_a entity confidence=0.5 and pass_b entity confidence=0.9, assert
    merged.confidence == 0.9."""
    ...


def test_merge_defaults_relationship_confidence_to_point_eight_when_none():
    """Spec §3.7 key property 5: confidence = 0.8 if rel.confidence is None."""
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "from_identity": {"system_name": "S-400"},
            "to_identity": {"name": "Truck-1"},
            "confidence": None,  # expected to be defaulted
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert len(merged.edges) == 1
    assert merged.edges[0].confidence == 0.8


def test_merge_preserves_explicit_zero_confidence():
    """Explicit 0.0 must NOT be coerced to 0.8.
    Template: copy test_merge_defaults_..., set confidence=0.0, assert
    merged.edges[0].confidence == 0.0."""
    ...


def test_merge_rejects_edge_when_from_endpoint_missing():
    """Edge pointing to a LogicalIdentity with no matching entity → rejection.
    Template: emit only a PLATFORM entity; emit an edge whose from_identity
    names a RADAR_SYSTEM that was never emitted. Assert
    FROM_ENDPOINT_NOT_FOUND in merged.rejected_edges."""
    ...


def test_merge_same_pass_resolves_by_identity_dict():
    """Same-pass relationships look up endpoints by identity_dict, not ref_id.
    Template: one pass emits entity + relationship; assert merged.edges has
    exactly 1 entry and from_identity matches."""
    ...


def test_merge_cross_pass_resolves_by_ref_id():
    """system_links-style relationships resolve endpoints via the upstream ref_id.
    Template: fixture needs pass_results for 'radar_domain' (producer) and
    'system_links' (consumer via ref_id). system_links pass consumes
    upstream_entities which the _make_pass_result helper will need to
    accommodate — extend the helper if needed."""
    ...
```

The first, third, and fourth tests are fully worked examples. The remaining tests follow the same structure — copy a worked example, mutate the input, assert the new invariant. Do NOT leave `...` in the committed test file; fill each one in before running Step 6.

Each test constructs a minimal `pass_results` dict + manifest + ontology fixture and asserts a specific merge behavior. Use pytest fixtures to share the minimum ontology slice.

- [x] **Step 6: Run all tests — expect failures**

```bash
pytest tests/unit/test_extraction_merge.py -v 2>&1 | tail -40
```

Expected: every test fails because `app/services/extraction_merge.py` doesn't exist yet.

- [x] **Step 7: Implement `app/services/extraction_merge.py` in two passes**

Per spec §3.6 through §3.9 and §6.2 + §6.7. Implement in two checkpoints so the red→green test loop stays tight:

**Checkpoint A (simple building blocks):** implement `RelationshipRejectionReason`, `YieldStatus`, `LogicalIdentity` (both helper methods), `ExtractionMetadata`, `PassResult` skeleton (defer `iter_entities_of_type` to Checkpoint B), `MergedEntityRecord`, `MergedEdgeRecord`, `MergedExtraction`, `ChunkForDerivation`, `DerivedEdge`, `build_display_label`, `classify_yield_from_counts`. Stop, run tests — expect the `build_display_label`, `classify_yield_*`, and `LogicalIdentity` tests to pass; the merge and rejection tests still fail because `merge_and_resolve` is not yet written.

```bash
pytest tests/unit/test_extraction_merge.py::test_build_display_label_prefers_system_name_from_identity tests/unit/test_extraction_merge.py::test_classify_yield_hit_fallthrough -v
```

**Checkpoint B (merge core):** implement `iter_entities_of_type`, `_count_primary_entities`, `_count_bridge_entities`, `classify_yield`, `merge_and_resolve`. Run the full test file — every test should pass.

The reason for the split: merge is the hardest part, and having 8 passing simple tests before you touch merge makes it much easier to isolate merge bugs from plumbing bugs.

Structure:

```python
"""Merge and resolve logic for the bundle-passes extraction path.
Spec §3.6 + §3.7 + §3.9 + §6.2 + §6.7."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel


# --- Enums -----------------------------------------------------------------

class YieldStatus(str, Enum):
    HIT          = "HIT"
    EMPTY        = "EMPTY"
    BRIDGES_ONLY = "BRIDGES_ONLY"
    DEGRADED     = "DEGRADED"


class RelationshipRejectionReason(str, Enum):
    MISSING_REL_TYPE         = "missing_rel_type"
    INVALID_IDENTITY_PAYLOAD = "invalid_identity_payload"
    UNKNOWN_REF_ID           = "unknown_ref_id"
    FROM_ENDPOINT_NOT_FOUND  = "from_endpoint_not_found"
    TO_ENDPOINT_NOT_FOUND    = "to_endpoint_not_found"
    INVALID_TRIPLE           = "invalid_triple"


# --- Logical identity ------------------------------------------------------

@dataclass(frozen=True)
class LogicalIdentity:
    entity_type: str
    identity_field_names: tuple[str, ...]
    identity_tuple: tuple[Any, ...]
    scope: Literal["document", "global"]
    document_id: str | None  # populated iff scope == "document"

    def identity_values_dict(self) -> dict[str, Any]:
        return dict(zip(self.identity_field_names, self.identity_tuple, strict=True))

    def as_upsert_identity_dict(self) -> dict[str, Any]:
        d = self.identity_values_dict()
        if self.scope == "document":
            assert self.document_id is not None, "document_id required for scope=document"
            d["document_id"] = self.document_id
        return d


# --- Handoff and merged types ---------------------------------------------

@dataclass
class ExtractionMetadata:
    schema_size_chars: int
    structured_output_mode: Literal["strict", "fallback_json"]


@dataclass
class PassResult:
    """Normalized output of one /extract-pass call. Handoff type between
    _run_single_pass (producer) and merge_and_resolve (consumer)."""
    pass_name: str
    template_instance: BaseModel
    metadata: ExtractionMetadata
    pre_merge_rejections: list[tuple[Any, RelationshipRejectionReason]]

    def iter_entities_of_type(self, entity_type: str) -> Iterable[Any]:
        """Return the nested entity model instances matching the given type.

        Convention: the fixed extraction schema modules in
        ontology_bundles/<bundle>/extraction_schemas/*.py declare their
        pass template with an attribute per entity type, named with the
        lowercase entity type + '_list' suffix (e.g., RADAR_SYSTEM is
        listed on `radar_system_list`). This method reads that attribute
        by convention. The test fixtures in Task 3.4 Step 1 use the same
        naming via SimpleNamespace.
        """
        attr = f"{entity_type.lower()}_list"
        return list(getattr(self.template_instance, attr, []) or [])

    @property
    def relationships(self) -> list[BaseModel]:
        """Return the relationships field (empty list for entities-only passes)."""
        val = getattr(self.template_instance, "relationships", None)
        return list(val) if val is not None else []


@dataclass
class MergedEntityRecord:
    identity: LogicalIdentity
    properties: dict[str, Any]
    confidence: float
    pass_origins: set[str]
    display_label: str


@dataclass
class MergedEdgeRecord:
    from_identity: LogicalIdentity
    to_identity: LogicalIdentity
    rel_type: str
    confidence: float
    source_pass: str


@dataclass
class MergedExtraction:
    entities: list[MergedEntityRecord]
    edges: list[MergedEdgeRecord]
    rejected_edges: list[tuple[str, Any, RelationshipRejectionReason]]
    rejections_by_pass: dict[str, int]
    pipeline_run_id: str
    document_id: str


# --- derive_rules DTOs (mirrored from spec §3.8 — used by derive_rules) ---

@dataclass
class ChunkForDerivation:
    rid: str
    text_normalized: str


@dataclass
class DerivedEdge:
    from_id: str
    to_id: str
    rel_type: str
    confidence: float | None


# --- Core merge function ---------------------------------------------------

def merge_and_resolve(
    pass_results: dict[str, PassResult],
    manifest,  # BundleManifest — avoid circular import via TYPE_CHECKING if needed
    ontology: dict,
    document_id: str,
    pipeline_run_id: str,
) -> MergedExtraction:
    """Phase 1: merge entities; resolve edges against logical identity.
    No backend RIDs involved. Pure logical-identity IR.

    Spec §3.7 key properties (each is enforced by a test in
    test_extraction_merge.py):
      1. Entities keyed by LogicalIdentity. Bridge entities with identical
         identity across passes collapse into one MergedEntityRecord with
         both pass names in pass_origins.
      2. Relationships resolved post-merge by LogicalIdentity lookup.
      3. Same-pass edges use identity-dict lookup; cross-pass edges use
         ref_id lookup against the upstream set.
      4. Rejections are counted per pass and per reason.
      5. confidence = 0.8 if rel.confidence is None. Explicit 0.0 preserved.
    """
    # Index ontology once for fast lookups
    entity_defs = {e["name"]: e for e in ontology.get("entity_types", [])}
    valid_triples = {
        (row["source"], row["relationship"], row["target"])
        for row in ontology.get("validation_matrix", [])
    }

    # Step 1: collect and merge entities keyed by LogicalIdentity
    entity_index: dict[LogicalIdentity, MergedEntityRecord] = {}
    rejected_edges: list[tuple[str, Any, RelationshipRejectionReason]] = []
    rejections_by_pass: dict[str, int] = {p: 0 for p in pass_results}

    for pass_name, pass_result in pass_results.items():
        for entity_type, entity_def in entity_defs.items():
            for inst in pass_result.iter_entities_of_type(entity_type):
                identity = _build_logical_identity(
                    entity_type=entity_type,
                    entity_def=entity_def,
                    instance=inst,
                    document_id=document_id,
                )
                props = _extract_properties(inst, entity_def)
                confidence = getattr(inst, "confidence", None) or 1.0
                existing = entity_index.get(identity)
                if existing:
                    # Merge: max confidence, union pass_origins, merge props
                    existing.confidence = max(existing.confidence, confidence)
                    existing.pass_origins.add(pass_name)
                    existing.properties.update(
                        {k: v for k, v in props.items() if v is not None}
                    )
                else:
                    entity_index[identity] = MergedEntityRecord(
                        identity=identity,
                        properties=props,
                        confidence=confidence,
                        pass_origins={pass_name},
                        display_label=build_display_label(
                            entity_type,
                            identity.identity_values_dict(),
                            props,
                        ),
                    )

    # Step 2: collect and resolve relationships
    edges: list[MergedEdgeRecord] = []
    for pass_name, pass_result in pass_results.items():
        for rel in pass_result.relationships:
            outcome = _resolve_relationship(
                rel=rel,
                source_pass=pass_name,
                entity_index=entity_index,
                valid_triples=valid_triples,
                entity_defs=entity_defs,
                document_id=document_id,
            )
            if isinstance(outcome, MergedEdgeRecord):
                edges.append(outcome)
            else:
                # outcome is a rejection reason
                rejected_edges.append((pass_name, rel, outcome))
                rejections_by_pass[pass_name] += 1

    return MergedExtraction(
        entities=list(entity_index.values()),
        edges=edges,
        rejected_edges=rejected_edges,
        rejections_by_pass=rejections_by_pass,
        pipeline_run_id=pipeline_run_id,
        document_id=document_id,
    )


def _build_logical_identity(
    *, entity_type: str, entity_def: dict, instance, document_id: str,
) -> LogicalIdentity:
    """Extract identity tuple from instance using entity_def.identity_fields.
    Scope comes from entity_def.identity_scope."""
    field_names = tuple(entity_def.get("identity_fields") or ())
    values = tuple(getattr(instance, name, None) for name in field_names)
    scope = entity_def.get("identity_scope", "document")
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=field_names,
        identity_tuple=values,
        scope=scope,
        document_id=document_id if scope == "document" else None,
    )


def _extract_properties(instance, entity_def: dict) -> dict[str, Any]:
    """Return {property_name: value} for every property in the entity def
    whose value on the instance is non-None."""
    props = {}
    for name in entity_def.get("properties", []):
        v = getattr(instance, name, None)
        if v is not None:
            props[name] = v
    return props


def _resolve_relationship(
    *,
    rel,
    source_pass: str,
    entity_index: dict[LogicalIdentity, MergedEntityRecord],
    valid_triples: set[tuple[str, str, str]],
    entity_defs: dict[str, dict],
    document_id: str,
) -> MergedEdgeRecord | RelationshipRejectionReason:
    """Resolve one relationship against the merged entity index.

    Returns a MergedEdgeRecord on success, or a RelationshipRejectionReason
    enum value on failure. The caller appends the reason to rejected_edges."""
    rel_type = getattr(rel, "rel_type", None)
    if rel_type is None:
        return RelationshipRejectionReason.MISSING_REL_TYPE

    from_type = getattr(rel, "from_type", None)
    to_type = getattr(rel, "to_type", None)
    from_identity_dict = getattr(rel, "from_identity", None)
    to_identity_dict = getattr(rel, "to_identity", None)

    # Identity payload sanity check
    if (not from_identity_dict or not to_identity_dict
            or from_type not in entity_defs or to_type not in entity_defs):
        return RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD

    # Validation matrix check
    if (from_type, rel_type, to_type) not in valid_triples:
        return RelationshipRejectionReason.INVALID_TRIPLE

    # Build the from/to LogicalIdentity to look up in the entity index
    from_li = _identity_from_dict(from_type, from_identity_dict, entity_defs, document_id)
    to_li = _identity_from_dict(to_type, to_identity_dict, entity_defs, document_id)

    if from_li not in entity_index:
        return RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND
    if to_li not in entity_index:
        return RelationshipRejectionReason.TO_ENDPOINT_NOT_FOUND

    confidence = getattr(rel, "confidence", None)
    if confidence is None:
        confidence = 0.8  # spec §3.7 key property 5

    return MergedEdgeRecord(
        from_identity=from_li,
        to_identity=to_li,
        rel_type=rel_type,
        confidence=confidence,
        source_pass=source_pass,
    )


def _identity_from_dict(
    entity_type: str,
    identity_dict: dict,
    entity_defs: dict[str, dict],
    document_id: str,
) -> LogicalIdentity:
    entity_def = entity_defs[entity_type]
    field_names = tuple(entity_def.get("identity_fields") or ())
    values = tuple(identity_dict.get(k) for k in field_names)
    scope = entity_def.get("identity_scope", "document")
    return LogicalIdentity(
        entity_type=entity_type,
        identity_field_names=field_names,
        identity_tuple=values,
        scope=scope,
        document_id=document_id if scope == "document" else None,
    )


def build_display_label(
    entity_type: str,
    identity_values: dict[str, Any],
    properties: dict[str, Any],
) -> str:
    """Spec §3.9 resolution order."""
    NAME_LIKE_KEYS = ("system_name", "name", "title", "heading", "document_id")

    for key in NAME_LIKE_KEYS:
        v = identity_values.get(key)
        if v:
            return str(v)

    non_empty = [str(v) for v in identity_values.values() if v]
    if non_empty:
        return " / ".join(non_empty)

    for key in NAME_LIKE_KEYS:
        v = properties.get(key)
        if v:
            return str(v)

    identity_hash = hashlib.sha1(
        json.dumps(identity_values, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{entity_type}_{identity_hash}"


# --- Yield classification --------------------------------------------------

def classify_yield_from_counts(
    *,
    primary: int,
    bridge: int,
    extracted_rels: int,
    rejected_rels: int,
) -> YieldStatus:
    """Spec §6.2 precedence. Pure function of counts."""
    total_rels = extracted_rels + rejected_rels
    if total_rels >= 4 and rejected_rels / total_rels >= 0.75:
        return YieldStatus.DEGRADED
    if primary == 0 and bridge == 0 and extracted_rels == 0:
        return YieldStatus.EMPTY
    if primary == 0 and bridge > 0:
        return YieldStatus.BRIDGES_ONLY
    return YieldStatus.HIT


def classify_yield(
    result: PassResult,
    pass_def,  # PassManifest
    ontology: dict,
) -> YieldStatus:
    """Convenience wrapper used inside _run_single_pass.
    Extracts counts from the PassResult and delegates."""
    primary = _count_primary_entities(result, pass_def)
    bridge = _count_bridge_entities(result, pass_def)
    extracted_rels = len(result.relationships)
    rejected_pre_merge = len(result.pre_merge_rejections)
    return classify_yield_from_counts(
        primary=primary,
        bridge=bridge,
        extracted_rels=extracted_rels,
        rejected_rels=rejected_pre_merge,
    )


def _count_primary_entities(result: PassResult, pass_def) -> int:
    """Count entities in the pass whose entity_type is in
    pass_def.entities MINUS the bridge-entity subset."""
    bridge_set = set(getattr(pass_def, "bridge_entities", []) or [])
    primary_types = [
        t for t in (getattr(pass_def, "entities", []) or [])
        if t not in bridge_set
    ]
    return sum(len(list(result.iter_entities_of_type(t))) for t in primary_types)


def _count_bridge_entities(result: PassResult, pass_def) -> int:
    """Count entities in the pass whose entity_type is in
    pass_def.bridge_entities."""
    return sum(
        len(list(result.iter_entities_of_type(t)))
        for t in (getattr(pass_def, "bridge_entities", []) or [])
    )
```

All implementation bodies are provided above. Cross-reference spec §3.7 for edge cases that the fixtures in Step 5 do not cover (e.g., ref_id resolution for cross-pass relationships — the `_resolve_relationship` skeleton does not handle `from_ref_id` / `to_ref_id` yet; extend it when you implement `test_rejection_unknown_ref_id` and `test_merge_cross_pass_resolves_by_ref_id`).

- [x] **Step 8: Run tests iteratively until green**

```bash
pytest tests/unit/test_extraction_merge.py -v 2>&1 | tail -40
```

Fix failures one at a time. Do NOT modify tests to match implementation — modify implementation to match tests. Commit is gated on green suite.

- [x] **Step 9: Run the coverage checker rule 14 locally**

The checker (Task 2.4) has rule 14 tied to `tests/unit/test_extraction_merge.py`. Re-run it now that the file exists:

```bash
python tools/check_extraction_coverage.py
```

Expected: `PASS air_defense_v3`. (Rule 14 itself is enforced by pytest, not by the coverage-checker binary — but the checker should not regress.)

- [x] **Step 10: Commit**

```bash
git add app/services/extraction_merge.py tests/unit/test_extraction_merge.py
git commit -m "feat(extraction): app/services/extraction_merge.py

Per spec §3.6 + §3.7 + §3.9 + §6.2 + §6.7. The full merge and resolve
module used by the new bundle-passes orchestrator (Chunk 4 Task 4.x).
Exposes LogicalIdentity, PassResult, MergedEntityRecord,
MergedEdgeRecord, MergedExtraction, ChunkForDerivation, DerivedEdge,
RelationshipRejectionReason, YieldStatus, merge_and_resolve,
build_display_label, classify_yield, classify_yield_from_counts.

Test module covers:
- Every RelationshipRejectionReason enum value (satisfies checker rule 14)
- LogicalIdentity identity-dict and upsert-dict semantics, including
  the document-scope document_id assertion
- build_display_label resolution order (identity → join → properties → hash)
- classify_yield_from_counts precedence rules from §6.2
- merge_and_resolve key properties (bridge collapse, confidence=max,
  rejection taxonomy, 0.8 default, explicit 0.0 preservation, same-pass
  vs cross-pass endpoint resolution)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 3.5: Add `StatusSignals` to `app/services/ontology_bundles.py`

**Files:**
- Modify: `app/services/ontology_bundles.py`
- Modify: `tests/unit/test_ontology_bundles.py`

Chunk 2 Task 2.3 deferred `StatusSignals` to Chunk 3. This task adds the dataclass next to the bundle loader so the status API in Chunk 4 can import it. Spec §7.10 `StatusSignals` declaration.

- [x] **Step 1: Write the failing test**

Append to `tests/unit/test_ontology_bundles.py`:

```python
def test_status_signals_shape():
    """StatusSignals is a dataclass with snapshot, is_stale, graph_queryable."""
    from dataclasses import fields
    from app.services.ontology_bundles import StatusSignals
    field_names = {f.name for f in fields(StatusSignals)}
    assert field_names == {"snapshot", "is_stale", "graph_queryable"}
```

Run:
```bash
pytest tests/unit/test_ontology_bundles.py::test_status_signals_shape -v
```

Expected: ImportError or AttributeError.

- [x] **Step 2: Add `StatusSignals` to `app/services/ontology_bundles.py`**

Append to the module:

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.ingest import DocumentGraphExtraction


@dataclass
class StatusSignals:
    """Status API roll-up per spec §7.10.

    snapshot:        the DocumentGraphExtraction row or None
    is_stale:        meaningful iff snapshot is not None; otherwise unused
    graph_queryable: top-level — always meaningful, computed via the
                     cross-run rollback query from §7.10
    """
    snapshot: "DocumentGraphExtraction | None"
    is_stale: bool
    graph_queryable: bool
```

Use `TYPE_CHECKING` to avoid a circular import between `app.services.ontology_bundles` and `app.models.ingest`.

- [x] **Step 3: Run tests — expect pass**

```bash
pytest tests/unit/test_ontology_bundles.py -v
```

Expected: all pass, including the new one.

- [x] **Step 4: Commit**

```bash
git add app/services/ontology_bundles.py tests/unit/test_ontology_bundles.py
git commit -m "feat(status): add StatusSignals dataclass to ontology_bundles

Per spec §7.10 status-signals declaration. Deferred from Chunk 2 Task
2.3 per that task's own guidance. The status API in Chunk 4 Task 4.x
consumes this type via compute_status_signals. Kept in ontology_bundles
rather than a new module because it is the natural home for
bundle-and-run roll-ups.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 3.6: Add `graph_extraction_engine` feature flag + `IngestDispatchResult`

**Files:**
- Modify: `app/config.py`
- Create: `app/workers/dispatch_types.py`
- Modify: `tests/unit/test_config.py` (or create if none exists)

Spec §7.4 feature flag + §5.2 IngestDispatchResult. The flag defaults to `legacy` and is read via `get_settings()` only — no uncached per-task reader — because flipping it requires a worker+beat restart (§7.6).

- [x] **Step 1: Write the failing config test**

Create or append `tests/unit/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_graph_extraction_engine_default_is_legacy():
    """Per spec §7.4, new flag defaults to 'legacy' so PR 2 merge does
    not auto-switch production traffic."""
    s = get_settings()
    assert s.graph_extraction_engine == "legacy"


def test_graph_extraction_engine_accepts_bundle_passes():
    """Only 'legacy' and 'bundle_passes' are valid values."""
    s = Settings(graph_extraction_engine="bundle_passes")
    assert s.graph_extraction_engine == "bundle_passes"


def test_graph_extraction_engine_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Settings(graph_extraction_engine="experimental")
```

- [x] **Step 2: Add the flag to `app/config.py`**

Find the `Settings` class and add:

```python
from typing import Literal

class Settings(BaseSettings):
    # ... existing fields ...

    # Spec §7.4 — feature flag for the bundle-passes extraction path.
    # Defaults to 'legacy' so PR 2 merge does not auto-switch production
    # traffic. Flipping to 'bundle_passes' requires a worker + beat
    # restart (§7.6). Read only via get_settings() — no uncached reader.
    graph_extraction_engine: Literal["legacy", "bundle_passes"] = "legacy"
```

Do NOT add a per-task reader helper. `get_settings()` is the only access point.

- [x] **Step 3: Run the config tests**

```bash
pytest tests/unit/test_config.py -v
```

Expected: all pass.

- [x] **Step 4: Create `IngestDispatchResult` dataclass**

Create `app/workers/dispatch_types.py`:

```python
"""Worker-facing dispatch return types.

Spec §5.2: start_ingest_pipeline returns IngestDispatchResult so
callers can access both the new pipeline_run_id and the Celery task
id without depending on the order they became available."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestDispatchResult:
    pipeline_run_id: str
    celery_task_id: str
```

A standalone module is used so both `app/workers/pipeline.py` and `app/api/v1/sources.py` can import it without creating a circular dependency between workers and API.

- [x] **Step 5: Write a smoke test for the dataclass**

Append to `tests/unit/test_config.py` (which already imports `pytest` at the top):

```python
from dataclasses import FrozenInstanceError

from app.workers.dispatch_types import IngestDispatchResult


def test_ingest_dispatch_result_is_frozen():
    r = IngestDispatchResult(pipeline_run_id="run-1", celery_task_id="task-1")
    assert r.pipeline_run_id == "run-1"
    assert r.celery_task_id == "task-1"
    with pytest.raises(FrozenInstanceError):
        r.pipeline_run_id = "run-2"  # type: ignore[misc]
```

- [x] **Step 6: Run the tests and commit**

```bash
pytest tests/unit/test_config.py -v
```

Expected: all pass.

```bash
git add app/config.py app/workers/dispatch_types.py tests/unit/test_config.py
git commit -m "feat(config): add graph_extraction_engine flag + IngestDispatchResult

Per spec §7.4 (feature flag) + §5.2 (dispatch return type).

graph_extraction_engine is a Literal['legacy','bundle_passes'] setting
on the existing Settings class, defaulting to 'legacy'. Read only via
get_settings() — no uncached per-task reader. Flipping requires a
worker + beat restart per §7.6.

IngestDispatchResult lives in a new app/workers/dispatch_types.py
module so both the worker and the upload/reingest API routes can
import it without creating a circular dependency.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 3.7: Bundle threading on `Source` schemas + new `ReingestRequest`

**Files:**
- Modify: `app/schemas/sources.py`
- Modify: `app/api/v1/sources.py` (route signature update only — start_ingest_pipeline rewrite stays in Chunk 4)
- Create: `tests/unit/test_source_schemas.py` (if absent)
- Create: `tests/unit/test_reingest_request.py`

Spec §7.4 bundle-threading subsection. Two concerns are bundled in this task because they touch the same route file:

1. Add `default_ontology_bundle_key` + `default_use_case_key` to the existing `SourceCreate` / `SourceResponse` at `app/schemas/sources.py:16-21`. These are purely additive (optional, default None).
2. Introduce a brand-new `ReingestRequest` Pydantic model and wire it into the `POST /documents/{document_id}/reingest` route at `app/api/v1/sources.py:291`, which currently uses `body: dict = None`. The route's behavior must be unchanged except that it now accepts the new bundle fields (which are forwarded to `start_ingest_pipeline` — that forwarding lands in Chunk 4's orchestrator rewrite; in this task the fields are accepted but not yet acted on).

- [x] **Step 1: Inspect the current schemas and route**

```bash
grep -n "class SourceCreate\|class SourceResponse\|class SourceUpdate" app/schemas/sources.py
grep -n "reingest\|class Reingest" app/api/v1/sources.py
```

Capture the current `SourceCreate` / `SourceResponse` field lists. Confirm there is no existing `ReingestRequest`. Confirm the reingest route at `app/api/v1/sources.py:291` takes `body: dict = None` today.

- [x] **Step 2: Write failing schema tests**

Create `tests/unit/test_source_schemas.py`:

```python
import pytest
from pydantic import TypeAdapter

from app.schemas.sources import SourceCreate, SourceResponse


def test_source_create_accepts_default_ontology_bundle_key():
    s = SourceCreate(
        name="test",
        default_ontology_bundle_key="air_defense_v3",
    )
    assert s.default_ontology_bundle_key == "air_defense_v3"


def test_source_create_defaults_bundle_key_to_none():
    """Existing callers that don't set the field must keep working."""
    s = SourceCreate(name="test")
    assert s.default_ontology_bundle_key is None


def test_source_create_accepts_default_use_case_key():
    s = SourceCreate(name="test", default_use_case_key="air_defense_v3_use_case")
    assert s.default_use_case_key == "air_defense_v3_use_case"


def test_source_response_exposes_default_ontology_bundle_key():
    """The field appears on the response so UIs can display it."""
    schema = TypeAdapter(SourceResponse).json_schema()
    assert "default_ontology_bundle_key" in schema.get("properties", {})
```

Create `tests/unit/test_reingest_request.py`:

```python
import pytest

from app.schemas.sources import ReingestRequest


def test_reingest_request_defaults_mode_to_full():
    """Behavior-preserving default: existing clients sending no body
    (or just {\"mode\": \"full\"}) must still get mode=\"full\"."""
    r = ReingestRequest()
    assert r.mode == "full"


def test_reingest_request_accepts_mode_values():
    for m in ("full", "embeddings_only", "graph_only"):
        r = ReingestRequest(mode=m)
        assert r.mode == m


def test_reingest_request_rejects_unknown_mode():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ReingestRequest(mode="experimental")


def test_reingest_request_accepts_optional_bundle_override():
    r = ReingestRequest(ontology_bundle_key="air_defense_v3")
    assert r.ontology_bundle_key == "air_defense_v3"
    assert r.mode == "full"


def test_reingest_request_bundle_override_defaults_to_none():
    r = ReingestRequest()
    assert r.ontology_bundle_key is None
    assert r.use_case_key is None
```

Run:
```bash
pytest tests/unit/test_source_schemas.py tests/unit/test_reingest_request.py -v
```

Expected: every test fails (new fields + new class don't exist yet).

- [x] **Step 3: Extend `SourceCreate` / `SourceResponse`**

Open `app/schemas/sources.py` at line 16. Add the two new optional fields to `SourceCreate` (and any other sibling schemas already there):

```python
class SourceCreate(APIModel):
    # ... existing fields ...
    default_ontology_bundle_key: str | None = None
    default_use_case_key: str | None = None


class SourceResponse(APIModel):
    # ... existing fields ...
    default_ontology_bundle_key: str | None = None
    default_use_case_key: str | None = None
```

If a `SourceUpdate` schema exists, add the same two fields there too. Keep the additions purely additive — do not change existing field defaults or required-ness.

- [x] **Step 4: Introduce `ReingestRequest` in the same file**

Append to `app/schemas/sources.py`:

```python
from typing import Literal


class ReingestRequest(APIModel):
    """Body for POST /documents/{document_id}/reingest.

    All fields optional. Introduced in PR 2 to thread bundle overrides
    through the reingest path. The current route accepts `body: dict`;
    this schema replaces that in Task 3.7 Step 5. The bundle fields are
    consumed by the new orchestrator in Chunk 4 Task 4.x.
    """
    mode: Literal["full", "embeddings_only", "graph_only"] = "full"
    ontology_bundle_key: str | None = None
    use_case_key: str | None = None
```

- [x] **Step 5: Update the reingest route signature to accept `ReingestRequest`**

Open `app/api/v1/sources.py` at line 291. Replace the current signature:

```python
# Before:
async def reingest_document(
    document_id: uuid.UUID,
    body: dict = None,
    db: AsyncSession = Depends(get_async_session),
):
    # ...
    mode = (body or {}).get("mode", "full")
```

With:

```python
# After:
from app.schemas.sources import ReingestRequest

async def reingest_document(
    document_id: uuid.UUID,
    body: ReingestRequest | None = None,
    db: AsyncSession = Depends(get_async_session),
):
    # ...
    body = body or ReingestRequest()
    mode = body.mode
    # ontology_bundle_key and use_case_key on `body` are READ here but
    # not yet forwarded to start_ingest_pipeline — that wiring lands in
    # Chunk 4 Task 4.x when the orchestrator rewrite accepts bundle
    # overrides. Accepting the fields now (without acting on them) keeps
    # the API surface forward-compatible.
```

Do NOT change how the route dispatches to `start_ingest_pipeline` or the other mode-specific code paths. The only behavior change is: clients can now send `{"mode": "full", "ontology_bundle_key": "..."}` and the schema will validate and ignore the bundle key (for now).

- [x] **Step 6: Run tests**

```bash
pytest tests/unit/test_source_schemas.py tests/unit/test_reingest_request.py -v
```

Expected: all pass.

- [x] **Step 7: Run the full suite**

```bash
pytest tests/ -x 2>&1 | tail -20
```

Expected: green. Existing reingest-route tests that send `{"mode": "graph_only"}` as raw dicts still work because FastAPI validates the dict into `ReingestRequest` transparently.

- [x] **Step 8: Commit**

```bash
git add app/schemas/sources.py app/api/v1/sources.py \
       tests/unit/test_source_schemas.py tests/unit/test_reingest_request.py
git commit -m "feat(api): bundle threading on Source + new ReingestRequest

Per spec §7.4 bundle-threading subsection. SourceCreate / SourceResponse
gain default_ontology_bundle_key + default_use_case_key (both
Optional[str] = None). Introduces ReingestRequest Pydantic model
(mode: Literal, ontology_bundle_key: Optional, use_case_key: Optional)
in the same module, replacing the route's previous 'body: dict = None'
shape. Existing clients sending {\"mode\": \"...\"} still validate.

Bundle override fields are accepted by the route but not yet forwarded
to start_ingest_pipeline — that wiring lands in Chunk 4 alongside the
orchestrator rewrite. This keeps PR 2 foundations additive while
establishing the forward-compatible API surface.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 3.8: Chunk 3 exit-gate test

**Files:**
- Create: `tests/integration/test_chunk3_foundations_smoke.py`

A small integration test that imports every new symbol introduced in Chunk 3 and smoke-tests that the Chunk 3 slice hasn't broken the legacy path. Written to give Chunk 4 a clean baseline.

- [x] **Step 1: Write the smoke test**

Create `tests/integration/test_chunk3_foundations_smoke.py`:

```python
"""Chunk 3 foundations smoke test. Verifies every new symbol is importable
and that the Chunk 3 invariants (feature flag default, new methods on
the graph store, ProvenanceMetadata extension) hold. The legacy e2e run
is a separate CI step — see Step 2 — not nested inside this test module."""


def test_imports_all_chunk3_symbols():
    from app.services.graph_store import ProvenanceMetadata, GraphStore  # noqa
    from app.services.arcadedb_graph import ArcadeDBGraphStore  # noqa
    from app.services.extraction_merge import (  # noqa
        LogicalIdentity, PassResult, MergedEntityRecord, MergedEdgeRecord,
        MergedExtraction, ChunkForDerivation, DerivedEdge,
        RelationshipRejectionReason, YieldStatus,
        merge_and_resolve, build_display_label,
        classify_yield, classify_yield_from_counts,
    )
    from app.services.ontology_bundles import StatusSignals  # noqa
    from app.workers.dispatch_types import IngestDispatchResult  # noqa


def test_feature_flag_defaults_to_legacy():
    """The 'nothing switched traffic' invariant for Chunk 3."""
    from app.config import get_settings
    assert get_settings().graph_extraction_engine == "legacy"


def test_delete_extraction_layer_graph_sync_present_on_arcadedb_store():
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    assert hasattr(ArcadeDBGraphStore, "delete_extraction_layer_graph_sync")


def test_provenance_metadata_accepts_pipeline_run_id():
    from app.services.graph_store import ProvenanceMetadata
    m = ProvenanceMetadata(document_id="doc-1", pipeline_run_id="run-1")
    assert m.pipeline_run_id == "run-1"
```

- [x] **Step 2: Run the smoke test AND the legacy e2e test as two separate invocations**

Nesting `pytest` inside `pytest` via `subprocess.run` is an anti-pattern — conftest search paths, working directory, and fixture state all get confused. Run them as two separate top-level invocations:

```bash
docker compose up -d postgres redis arcadedb worker api docling-graph
pytest tests/integration/test_chunk3_foundations_smoke.py -v
pytest tests/e2e/test_full_pipeline.py -v
```

Expected: both suites pass. The smoke test covers the Chunk 3 invariants; the e2e test covers the legacy-path end-to-end assertion. In CI, wire these as two sequential steps in the pipeline YAML — do NOT combine them into one test module.

- [x] **Step 3: Commit**

```bash
git add tests/integration/test_chunk3_foundations_smoke.py
git commit -m "test(chunk3): foundations smoke test

Asserts every new Chunk 3 symbol is importable, the rollback primitive
is present on ArcadeDBGraphStore, ProvenanceMetadata accepts the new
pipeline_run_id field, and the legacy path still runs the full e2e
pipeline test. Establishes a green baseline for Chunk 4 to build on.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

---

**END OF CHUNK 3.** PR 2 foundations are landed. The new rollback primitive, ProvenanceMetadata extension, extraction_merge module, StatusSignals, feature flag, IngestDispatchResult, and bundle-threading schemas are all present. Nothing has switched traffic yet — `graph_extraction_engine` defaults to `legacy`. Chunk 4 rewrites `derive_ontology_graph`, refactors `start_ingest_pipeline` / `reingest_graph_only`, adds the status API, builds the baseline harness, runs the comparison, and opens PR 2.

---

## Chunk 4: PR 2 — Orchestrator rewrite, status API, baseline harness, PR 2 open

This chunk is the back half of PR 2. Everything in Chunk 3 was strictly foundations — nothing switched traffic. Chunk 4 rewrites `derive_ontology_graph` behind the `graph_extraction_engine` feature flag, refactors `start_ingest_pipeline` / `reingest_graph_only`, wires the new status API (three-concept split with the cross-run `graph_queryable` query), builds the baseline harness, runs the baseline comparison, and opens PR 2. At the end of Chunk 4 the feature flag still defaults to `legacy` — the flip to `bundle_passes` happens during the soak procedure, not as part of PR 2 merge. Spec §5.1–§5.7 + §7.4 + §7.10 + §8.3.

### Task 4.1: Write `GraphWriteTracker` and orchestrator helper skeleton

**Files:**
- Modify: `app/workers/pipeline.py`
- Create: `tests/unit/test_graph_write_tracker.py`

Spec §5.4 tracker gate. Starts with the tracker dataclass and the skeleton of the helper functions the new orchestrator branch will call. Nothing wires to the live extraction path yet — all under the `graph_extraction_engine == "bundle_passes"` branch.

- [x] **Step 1: Write failing tracker tests**

Create `tests/unit/test_graph_write_tracker.py`:

```python
"""Spec §5.4 GraphWriteTracker — worker-local rollback gate."""
from app.workers.pipeline import GraphWriteTracker


def test_tracker_default_is_false():
    t = GraphWriteTracker()
    assert t.any_mutation_attempted is False


def test_tracker_mark_flips_true():
    t = GraphWriteTracker()
    t.mark()
    assert t.any_mutation_attempted is True


def test_tracker_mark_is_idempotent():
    t = GraphWriteTracker()
    t.mark()
    t.mark()
    t.mark()
    assert t.any_mutation_attempted is True
```

Run:
```bash
pytest tests/unit/test_graph_write_tracker.py -v
```

Expected: fails on import.

- [x] **Step 2: Add `GraphWriteTracker` to `app/workers/pipeline.py`**

Near the top of the file (after existing imports, before the Celery tasks):

```python
from dataclasses import dataclass


@dataclass
class GraphWriteTracker:
    """Worker-local rollback gate per spec §5.4. Phase helpers call
    .mark() immediately before the first graph_store mutation in each
    phase. Failures before .mark() leave the flag False so rollback
    is skipped."""
    any_mutation_attempted: bool = False

    def mark(self) -> None:
        self.any_mutation_attempted = True
```

Run tests — expect pass.

- [x] **Step 3: Add helper function stubs that raise NotImplementedError**

Near `GraphWriteTracker`, add stubs for every helper the new branch needs. Each raises `NotImplementedError` until its specific task lands. This lets dependent tasks reference the names without breaking imports.

```python
def _attempt_rollback(document_id: str) -> str:
    raise NotImplementedError("Task 4.6")

def _delete_extraction_layer_graph(document_id: str) -> None:
    raise NotImplementedError("Task 4.6")

def _write_pipeline_run_metrics(pipeline_run_id, merged, manifest) -> None:
    raise NotImplementedError("Task 4.5")

def _run_single_pass(**kwargs) -> None:
    raise NotImplementedError("Task 4.3")

def _should_skip(pass_def, upstream_refs, ontology) -> bool:
    raise NotImplementedError("Task 4.3")

def _apply_post_merge_yield_updates(pipeline_run_id, merged) -> None:
    raise NotImplementedError("Task 4.5")

def _import_graph_phase_nodes(merged, ontology, document_id, tracker):
    raise NotImplementedError("Task 4.4")

def _import_graph_phase_domain_edges(merged, ontology, tracker) -> None:
    raise NotImplementedError("Task 4.4")

def _import_graph_phase_structural_edges(merged, identity_to_rid, document_id, pipeline_run_id, tracker) -> None:
    raise NotImplementedError("Task 4.4")

def _update_document_pipeline_status(document_id: str, new_status: str) -> None:
    raise NotImplementedError("Task 4.7")

def check_required_pass_gate(pipeline_run_id):
    raise NotImplementedError("Task 4.3")

def _build_docling_document_json(document_id: str) -> dict:
    raise NotImplementedError("existing helper — no change required; just imported")

def _upsert_document_graph_extraction(**kwargs) -> None:
    raise NotImplementedError("Task 4.5")
```

- [x] **Step 4: Commit**

```bash
git add app/workers/pipeline.py tests/unit/test_graph_write_tracker.py
git commit -m "feat(pipeline): GraphWriteTracker dataclass + orchestrator helper stubs

Per spec §5.4 tracker gate. Adds the rollback-gate dataclass and
NotImplementedError stubs for every helper the new derive_ontology_graph
branch in Chunk 4 will call. Stubs let downstream tasks reference
these names without breaking imports — each stub carries a task-ID
back-pointer so the implementer knows which task will fill it in.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.2: Refactor `start_ingest_pipeline` + `reingest_graph_only`

**Files:**
- Modify: `app/workers/pipeline.py`
- Modify: `app/api/v1/sources.py`
- Create: `tests/unit/test_start_ingest_pipeline.py`

Spec §5.2 + §5.3. `start_ingest_pipeline` returns `IngestDispatchResult` and snapshots the bundle on `PipelineRun`. `reingest_graph_only` uses `resolve_bundle_key_for_graph_only` with inheritance precedence.

- [x] **Step 1: Write failing tests**

Create `tests/unit/test_start_ingest_pipeline.py`:

```python
"""Spec §5.2 / §5.3 — ingest entry points and bundle resolution."""
import pytest
from unittest.mock import patch, MagicMock


def test_start_ingest_pipeline_returns_dispatch_result():
    from app.workers.pipeline import start_ingest_pipeline
    from app.workers.dispatch_types import IngestDispatchResult
    # ... fixture setup: a test Document and Source
    result = start_ingest_pipeline("doc-1")
    assert isinstance(result, IngestDispatchResult)
    assert result.pipeline_run_id
    assert result.celery_task_id


def test_start_ingest_pipeline_snapshots_bundle_on_pipeline_run():
    """PipelineRun row gets ontology_bundle_key / ontology_name / version."""
    ...


def test_start_ingest_pipeline_explicit_override_wins():
    """Caller-provided ontology_bundle_key beats source default."""
    ...


def test_start_ingest_pipeline_source_default_beats_system_default():
    ...


def test_start_ingest_pipeline_system_default_as_final_fallback():
    ...


def test_reingest_graph_only_inherits_bundle_from_latest_run():
    """Spec §5.3: latest run's bundle is the first-priority inheritance source."""
    ...


def test_reingest_graph_only_falls_back_to_source_when_latest_legacy():
    """Legacy latest run (ontology_bundle_key IS NULL) → fall back to source default."""
    ...


def test_reingest_graph_only_logs_legacy_inference():
    """When inferring from source because latest run is legacy, emit a log."""
    ...
```

Run:
```bash
pytest tests/unit/test_start_ingest_pipeline.py -v
```

Expected: tests fail (functions don't exist or have old signatures).

- [x] **Step 2: Refactor `start_ingest_pipeline` per spec §5.2**

In `app/workers/pipeline.py`, find the current `start_ingest_pipeline` and replace with:

```python
def start_ingest_pipeline(
    document_id: str,
    *,
    ontology_bundle_key: str | None = None,
    use_case_key: str | None = None,
) -> IngestDispatchResult:
    """Kick off the full ingest pipeline for a document.
    Spec §5.2. Snapshots the resolved bundle onto the PipelineRun row
    so downstream stages see it as immutable per-run metadata."""
    from app.workers.dispatch_types import IngestDispatchResult
    from app.services.ontology_bundles import resolve_bundle_key, load_bundle_manifest
    from app.db.session import get_sync_session
    from app.models.ingest import Document, PipelineRun
    import uuid

    settings = get_settings()

    with get_sync_session() as session:
        document = session.get(Document, document_id)
        source = document.source if document else None

        resolved_key = resolve_bundle_key(
            explicit_override=ontology_bundle_key,
            source_default=(source.default_ontology_bundle_key if source else None),
            system_default=settings.default_ontology_bundle_key,
        )
        manifest = load_bundle_manifest(resolved_key)

        run = PipelineRun(
            id=uuid.uuid4(),
            document_id=document_id,
            mode="full",
            status="PROCESSING",
            ontology_bundle_key=resolved_key,
            ontology_name=manifest.ontology_name,
            ontology_version=manifest.ontology_version,
            use_case_key=use_case_key or (source.default_use_case_key if source else None),
            extraction_profile_version=manifest.extraction_profile_version,
            started_at=datetime.utcnow(),
        )
        session.add(run)
        session.commit()
        run_id = str(run.id)

    async_result = dispatch_ingest_chain(run_id)
    return IngestDispatchResult(
        pipeline_run_id=run_id,
        celery_task_id=async_result.id,
    )
```

`dispatch_ingest_chain(run_id)` is the existing Celery chain builder — if the current file uses a different name, preserve that name and only change the input argument shape. The goal is to return `IngestDispatchResult`, not rename infrastructure.

- [x] **Step 3: Add `reingest_graph_only` per spec §5.3**

Add the new function to `app/workers/pipeline.py`:

```python
def reingest_graph_only(doc_id, request):
    """Dispatch a graph-only reingest. Spec §5.3."""
    from app.services.ontology_bundles import (
        resolve_bundle_key_for_graph_only, load_bundle_manifest,
    )
    from app.db.session import get_sync_session
    from app.models.ingest import Document, PipelineRun
    import uuid
    import logging
    logger = logging.getLogger(__name__)

    settings = get_settings()

    with get_sync_session() as session:
        document = session.get(Document, doc_id)
        latest_run = (
            session.query(PipelineRun)
            .filter_by(document_id=doc_id)
            .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
            .first()
        )
        inherited_bundle = (
            latest_run.ontology_bundle_key
            if latest_run and latest_run.ontology_bundle_key else None
        )

        resolved_key = resolve_bundle_key_for_graph_only(
            explicit_override=getattr(request, "ontology_bundle_key", None),
            inherited_from_run=inherited_bundle,
            source_default=(
                document.source.default_ontology_bundle_key
                if document and document.source else None
            ),
            system_default=settings.default_ontology_bundle_key,
        )

        if inherited_bundle is None and latest_run is not None:
            logger.info(
                "reingest_graph_only: latest run for document %s is legacy "
                "(ontology_bundle_key NULL); bundle inferred from source/system "
                "default (%s)",
                doc_id, resolved_key,
            )

        manifest = load_bundle_manifest(resolved_key)
        new_run = PipelineRun(
            id=uuid.uuid4(),
            document_id=doc_id,
            mode="graph_only",
            status="PROCESSING",
            ontology_bundle_key=resolved_key,
            ontology_name=manifest.ontology_name,
            ontology_version=manifest.ontology_version,
            use_case_key=(
                getattr(request, "use_case_key", None)
                or (latest_run.use_case_key if latest_run else None)
            ),
            extraction_profile_version=manifest.extraction_profile_version,
            started_at=datetime.utcnow(),
        )
        session.add(new_run)
        session.commit()
        run_id = str(new_run.id)

    async_result = derive_ontology_graph.delay(pipeline_run_id=run_id)
    return {
        "pipeline_run_id": run_id,
        "celery_task_id": async_result.id,
        "ontology_bundle_key": resolved_key,
    }
```

- [x] **Step 4: Update `POST /documents/{id}/reingest` route to consume `ReingestRequest`**

Open `app/api/v1/sources.py` line 291. The route already accepts `ReingestRequest` as of Chunk 3 Task 3.7. In this task, make it forward the bundle override to the new entry points:

```python
if mode == "full":
    from app.workers.pipeline import start_ingest_pipeline
    result = start_ingest_pipeline(
        str(document_id),
        ontology_bundle_key=body.ontology_bundle_key,
        use_case_key=body.use_case_key,
    )
    task_id = result.celery_task_id
    pipeline_run_id = result.pipeline_run_id
elif mode == "graph_only":
    from app.workers.pipeline import reingest_graph_only
    result = reingest_graph_only(document_id, body)
    task_id = result["celery_task_id"]
    pipeline_run_id = result["pipeline_run_id"]
elif mode == "embeddings_only":
    # Unchanged path — embeddings_only does not touch ontology.
    ...
```

Update the response body to include `pipeline_run_id` where newly available.

- [x] **Step 5: Run tests**

```bash
pytest tests/unit/test_start_ingest_pipeline.py -v
pytest tests/ -x 2>&1 | tail -20
```

Expected: green. If the existing upload route depends on the old `start_ingest_pipeline` return type (it used to return a bare Celery task id), update its caller at `app/api/v1/sources.py` per spec §5.2's caller-update example. Mechanical 3-line change.

- [x] **Step 6: Commit**

```bash
git add app/workers/pipeline.py app/api/v1/sources.py tests/unit/test_start_ingest_pipeline.py
git commit -m "feat(pipeline): start_ingest_pipeline + reingest_graph_only bundle threading

Per spec §5.2 / §5.3. start_ingest_pipeline accepts explicit
ontology_bundle_key / use_case_key overrides, resolves via
resolve_bundle_key with precedence (explicit > source > system
default), snapshots manifest metadata onto PipelineRun, and returns
IngestDispatchResult. reingest_graph_only is new — uses
resolve_bundle_key_for_graph_only with latest-run inheritance and a
logged fallback when the latest run is legacy.

The reingest route at app/api/v1/sources.py:291 now forwards
body.ontology_bundle_key / body.use_case_key to the new entry points.
Upload route updated to unwrap IngestDispatchResult.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.3: Implement `_run_single_pass` + `_should_skip` + `check_required_pass_gate`

**Files:**
- Modify: `app/workers/pipeline.py`
- Create: `tests/unit/test_run_single_pass.py`

Spec §5.5 + §6.4. Per-pass dispatcher with retry, skip, gate logic.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_run_single_pass.py` with TDD coverage:
- `_should_skip` returns True for relationships_only pass with no satisfiable triple
- `_should_skip` returns False for entities pass
- `_should_skip` returns False when a satisfiable triple exists in validation_matrix
- `_run_single_pass` writes a StageRun row with `execution_status="SKIPPED"` and `skip_reason="NO_UPSTREAM_ENDPOINTS"` when `_should_skip` returns True
- `_run_single_pass` retries on PassRetryable and writes a new StageRun row with `attempt+1` each time
- `_run_single_pass` stops retrying on `PassTerminal` and raises `IngestFailed` if pass is required
- `_run_single_pass` populates `pass_results[pass_def.name]` on success
- `check_required_pass_gate` passes when all required passes are `COMPLETE`
- `check_required_pass_gate` fails when a required pass is `FAILED`
- `check_required_pass_gate` passes when a required pass is `SKIPPED` with authorized skip_reason
- `check_required_pass_gate` fails when a required pass is `SKIPPED` with unauthorized skip_reason
- `check_required_pass_gate` raises `WorkerInvariantError` when a required pass has NO StageRun

Use mocked `_call_extract_pass` (the HTTP helper) so tests don't hit the real docling-graph service.

- [ ] **Step 2: Implement `_should_skip`**

Per spec §5.5 skip logic block:

```python
def _should_skip(pass_def, upstream_refs: dict, ontology: dict) -> bool:
    if pass_def.kind != "relationships_only":
        return False
    if not getattr(pass_def, "skip_if_no_upstream_endpoints", False):
        return False

    declared_deps = set(pass_def.depends_on)
    available_types: set[str] = {
        ref.entity_type
        for ref in upstream_refs.values()
        if ref.pass_origin in declared_deps
    }
    if not available_types:
        return True

    allowed_rels = set(pass_def.extracted_relationship_types)
    for row in ontology.get("validation_matrix", []):
        if row.get("relationship") not in allowed_rels:
            continue
        if (row.get("source") in available_types
                and row.get("target") in available_types):
            return False

    return True
```

- [ ] **Step 3: Implement `_run_single_pass`**

Per spec §5.5. Skeleton:

```python
class PassRetryable(Exception): pass
class PassTerminal(Exception): pass
class IngestFailed(Exception): pass
class WorkerInvariantError(Exception): pass


def _run_single_pass(
    pipeline_run_id,
    pass_def,
    manifest,
    ontology: dict,
    bundle_key: str,
    doc_json: dict,
    pass_results: dict,
    upstream_refs: dict,
    document_id: str,
) -> None:
    settings = get_settings()
    attempt = 1
    while True:
        if _should_skip(pass_def, upstream_refs, ontology):
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="SKIPPED",
                yield_status=None,
                skip_reason="NO_UPSTREAM_ENDPOINTS",
                counts=None,
                error=None,
            )
            return

        try:
            request_body = _build_extract_pass_request(
                bundle_key=bundle_key,
                pass_def=pass_def,
                doc_json=doc_json,
                upstream_refs=(
                    upstream_refs
                    if pass_def.input_mode == "document_plus_entity_refs"
                    else None
                ),
            )
            response = _call_extract_pass(
                request_body,
                timeout=settings.docling_graph_timeout,
            )
            pass_result = _parse_pass_response(response, pass_def, manifest)

        except PassRetryable as exc:
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="FAILED",
                yield_status=None,
                skip_reason=None,
                counts=None,
                error=str(exc),
            )
            if attempt >= settings.pass_max_retries:
                if pass_def.required:
                    raise IngestFailed(
                        f"Required pass {pass_def.name} exhausted retries"
                    ) from exc
                return
            _backoff(attempt)
            attempt += 1
            continue

        except PassTerminal as exc:
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="FAILED",
                yield_status=None,
                skip_reason=None,
                counts=None,
                error=str(exc),
            )
            if pass_def.required:
                raise IngestFailed(
                    f"Required pass {pass_def.name} terminal failure"
                ) from exc
            return

        yield_status = classify_yield(pass_result, pass_def, ontology)
        counts = _count_pass_output(pass_result, pass_def, ontology)
        _write_stage_run(
            pipeline_run_id=pipeline_run_id,
            pass_def=pass_def,
            attempt=attempt,
            execution_status="COMPLETE",
            yield_status=yield_status,
            skip_reason=None,
            counts=counts,
            error=None,
        )
        pass_results[pass_def.name] = pass_result

        if _any_downstream_pass_depends_on(manifest, pass_def.name):
            _extend_upstream_refs(upstream_refs, pass_result, pass_def, ontology)
        return
```

Implement the private helpers (`_build_extract_pass_request`, `_call_extract_pass`, `_parse_pass_response`, `_backoff`, `_count_pass_output`, `_any_downstream_pass_depends_on`, `_extend_upstream_refs`, `_write_stage_run`) inline in the same file. Keep each ≤20 lines. Use `httpx.Client` for `_call_extract_pass`, targeting `settings.docling_graph_url + "/extract-pass"`.

`_backoff(attempt)` uses the spec-declared backoff: `min(30 * 2**(attempt-1), 300)` seconds.

- [ ] **Step 4: Implement `check_required_pass_gate`**

Per spec §6.4:

```python
from dataclasses import dataclass

@dataclass
class GateResult:
    passed: bool
    failures: list[tuple[str, str]]


def check_required_pass_gate(pipeline_run_id) -> GateResult:
    from app.db.session import get_sync_session
    from app.models.ingest import PipelineRun, StageRun
    from app.services.ontology_bundles import load_bundle_manifest

    with get_sync_session() as session:
        run = session.get(PipelineRun, pipeline_run_id)
        manifest = load_bundle_manifest(run.ontology_bundle_key)
        required_passes = [p.name for p in manifest.passes if p.required]
        failures: list[tuple[str, str]] = []

        for pass_name in required_passes:
            latest = (
                session.query(StageRun)
                .filter(
                    StageRun.pipeline_run_id == pipeline_run_id,
                    StageRun.stage_name == "derive_ontology_graph",
                    StageRun.pass_name == pass_name,
                )
                .order_by(StageRun.attempt.desc())
                .first()
            )
            if latest is None:
                raise WorkerInvariantError(
                    f"Required pass {pass_name} has no StageRun"
                )

            if latest.execution_status == "COMPLETE":
                continue
            if latest.execution_status == "FAILED":
                failures.append((pass_name, f"FAILED: {latest.error_message}"))
                continue
            if latest.execution_status == "SKIPPED":
                if latest.skip_reason in {"NO_UPSTREAM_ENDPOINTS"}:
                    continue
                failures.append(
                    (pass_name, f"unauthorized skip: {latest.skip_reason}")
                )
                continue

        return GateResult(passed=(not failures), failures=failures)
```

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/unit/test_run_single_pass.py -v
```

Fix failures until green, then:

```bash
git add app/workers/pipeline.py tests/unit/test_run_single_pass.py
git commit -m "feat(pipeline): _run_single_pass + _should_skip + check_required_pass_gate

Per spec §5.5 + §6.4. Per-pass dispatcher with retry, skip, and
required-pass gate logic. Writes a new StageRun row per attempt so
previous FAILED rows are preserved. Retryable vs terminal exceptions
are distinguished by type. Skip logic consumes the actual upstream
refs and queries the validation_matrix, not yield buckets. Gate
treats COMPLETE and authorized SKIPPED as passing.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.4: Implement three-phase graph import helpers

**Files:**
- Modify: `app/workers/pipeline.py`
- Create: `tests/unit/test_graph_import_phases.py`

Spec §5.6. Three phase helpers: nodes, domain edges, structural edges. Each calls `tracker.mark()` immediately before its first graph_store mutation.

- [ ] **Step 1: Write failing tests for tracker semantics**

Create `tests/unit/test_graph_import_phases.py`:

```python
"""Spec §5.6 three-phase import. Focus on tracker semantics so the
rollback gate is correctly wired."""
from unittest.mock import MagicMock, patch

from app.workers.pipeline import (
    GraphWriteTracker,
    _import_graph_phase_nodes,
    _import_graph_phase_domain_edges,
    _import_graph_phase_structural_edges,
)


def test_phase_nodes_marks_tracker_before_mutation():
    """Tracker flips to True the moment upsert_nodes_batch_sync is called."""
    ...


def test_phase_nodes_leaves_tracker_false_on_pre_mutation_failure():
    """If building NodeRecord list raises, tracker stays False."""
    ...


def test_phase_domain_edges_uses_existing_tracker_state():
    """Phase 3 does not reset the tracker — phase 2 may have already marked it."""
    ...


def test_phase_structural_edges_is_noop_for_empty_derived_list():
    """Empty derived list → no tracker changes, no graph_store calls."""
    ...
```

Tests use `MagicMock()` for `graph_store` and `tracker` — this is unit-level; real GraphStore wiring is covered by the Chunk 3 rollback integration test.

- [ ] **Step 2: Implement phase 2 — node upsert**

Per spec §5.6 phase 2:

```python
def _import_graph_phase_nodes(
    merged,
    ontology: dict,
    document_id: str,
    tracker: GraphWriteTracker,
):
    from app.services.graph_store import NodeRecord, ProvenanceMetadata
    from app.services.extraction_merge import build_display_label
    from app.db.session import get_graph_store

    provenance = ProvenanceMetadata(
        document_id=document_id,
        pipeline_run_id=merged.pipeline_run_id,
    )
    node_records = [
        NodeRecord(
            entity_type=e.identity.entity_type,
            identity_fields=e.identity.as_upsert_identity_dict(),
            name=build_display_label(
                e.identity.entity_type,
                e.identity.identity_values_dict(),
                e.properties,
            ),
            properties=e.properties,
            extraction_confidence=e.confidence,
        )
        for e in merged.entities
    ]

    tracker.mark()
    graph_store = get_graph_store()
    node_rids: list[str] = graph_store.upsert_nodes_batch_sync(
        node_records, provenance,
    )

    identity_to_rid = dict(
        zip(
            (e.identity for e in merged.entities),
            node_rids,
            strict=True,
        )
    )
    return identity_to_rid
```

- [ ] **Step 3: Implement phase 3 — domain edges**

```python
def _import_graph_phase_domain_edges(merged, ontology: dict, tracker: GraphWriteTracker) -> None:
    from app.services.graph_store import RelationshipRecord, ProvenanceMetadata
    from app.db.session import get_graph_store

    provenance = ProvenanceMetadata(
        document_id=merged.document_id,
        pipeline_run_id=merged.pipeline_run_id,
    )
    rel_records = [
        RelationshipRecord(
            from_type=e.from_identity.entity_type,
            from_identity=e.from_identity.as_upsert_identity_dict(),
            to_type=e.to_identity.entity_type,
            to_identity=e.to_identity.as_upsert_identity_dict(),
            rel_type=e.rel_type,
            extraction_confidence=e.confidence,
        )
        for e in merged.edges
    ]

    tracker.mark()  # idempotent — phase 2 already marked
    graph_store = get_graph_store()
    graph_store.upsert_relationships_batch_sync(rel_records, provenance)
```

- [ ] **Step 4: Implement phase 4 — derived structural edges**

```python
def _import_graph_phase_structural_edges(
    merged,
    identity_to_rid,
    document_id: str,
    pipeline_run_id: str,
    tracker: GraphWriteTracker,
) -> None:
    from ontology_bundles.air_defense_v3 import derive_rules
    from app.db.session import get_graph_store

    chunks = _load_chunks_for_derivation(document_id)
    document_rid = _get_structural_document_rid(document_id)

    derived = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid=document_rid,
    )

    graph_store = get_graph_store()
    for edge in derived:
        tracker.mark()  # idempotent
        graph_store.create_structural_edge_sync(
            from_id=edge.from_id,
            to_id=edge.to_id,
            rel_type=edge.rel_type,
            properties={
                "document_id": document_id,
                "pipeline_run_id": pipeline_run_id,
                "extraction_confidence": edge.confidence,
                "source": "derive_rules",
            },
        )


def _load_chunks_for_derivation(document_id: str) -> list:
    """Load TextChunk rows and convert to ChunkForDerivation DTOs.
    This helper may already exist in the current codebase under a
    different name — reuse it if so, don't duplicate."""
    from app.services.extraction_merge import ChunkForDerivation
    from app.db.session import get_sync_session
    from sqlalchemy import text
    with get_sync_session() as session:
        rows = session.execute(
            text(
                "SELECT rid, text_normalized FROM retrieval.text_chunks "
                "WHERE document_id = :doc_id"
            ),
            {"doc_id": document_id},
        ).all()
    return [ChunkForDerivation(rid=r.rid, text_normalized=r.text_normalized) for r in rows]


def _get_structural_document_rid(document_id: str) -> str:
    """Look up the ArcadeDB @rid of the structural Document vertex."""
    from app.db.session import get_graph_store
    graph_store = get_graph_store()
    rows = graph_store._client.query_sync(
        graph_store._database, "sql",
        "SELECT @rid AS rid FROM Document WHERE document_id = :doc_id",
        params={"doc_id": document_id},
    )
    if not rows:
        raise WorkerInvariantError(
            f"No structural Document vertex found for {document_id}"
        )
    return rows[0]["rid"]
```

Verify `retrieval.text_chunks` column names against the actual ingest schema before committing — the `rid` column may be named `arcadedb_rid` or similar. Grep: `grep -n "class TextChunk\|text_chunks" app/models/`.

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/unit/test_graph_import_phases.py -v
```

Fix any failures, then commit.

```bash
git add app/workers/pipeline.py tests/unit/test_graph_import_phases.py
git commit -m "feat(pipeline): three-phase graph import with tracker gating

Per spec §5.6. Phase 2 (node upsert), phase 3 (domain edges), phase 4
(derived structural edges). Each phase builds its record list in pure
Python before calling tracker.mark() immediately prior to the first
graph_store mutation. Failures during record construction leave the
tracker False so rollback is skipped. HAS_PROVENANCE edges are
auto-created by upsert_nodes_batch_sync via ProvenanceMetadata; phase
4 does NOT create them.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.5: Implement merge post-processing + metrics + snapshot write

**Files:**
- Modify: `app/workers/pipeline.py`
- Create: `tests/unit/test_pipeline_metrics.py`

Spec §5.4 `_apply_post_merge_yield_updates`, `_write_pipeline_run_metrics`, `_upsert_document_graph_extraction`. All three run after merge but before the rollback-gated import phases.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_pipeline_metrics.py` covering:
- `_write_pipeline_run_metrics` populates `PipelineRun.metrics` with `pass_outcomes`, `document_extraction_anomaly`, `pass_degraded_count`, `overall_relationship_rejection_ratio`, `rejected_relationships_sample`, `bundle_legacy`, `bundle_key_display`
- `document_extraction_anomaly` is True when `radar_domain`, `missile_domain`, `other_systems` all end EMPTY/BRIDGES_ONLY
- `document_extraction_anomaly` is False when any core pass ends HIT
- `bundle_legacy` is False for runs with a bundle_key
- `_apply_post_merge_yield_updates` can move HIT → DEGRADED but never the other direction
- `_upsert_document_graph_extraction` inserts on first write, updates on subsequent

- [ ] **Step 2: Implement `_apply_post_merge_yield_updates`**

```python
def _apply_post_merge_yield_updates(pipeline_run_id, merged) -> None:
    """After merge_and_resolve, some passes may have a different yield
    status because post-merge rejections changed the rejection ratio.
    Recompute yield from updated totals via classify_yield_from_counts
    and update the StageRun row iff the new status is DEGRADED (HIT
    cannot regress to DEGRADED but DEGRADED is sticky).

    Only HIT → DEGRADED transitions are applied. Other directions are
    no-ops to preserve pre-merge classification."""
    from app.db.session import get_sync_session
    from app.models.ingest import StageRun
    from app.services.extraction_merge import classify_yield_from_counts, YieldStatus

    # Group merged edges and rejections by source pass
    from collections import defaultdict
    by_pass_accepted = defaultdict(int)
    by_pass_rejected = defaultdict(int)
    for edge in merged.edges:
        by_pass_accepted[edge.source_pass] += 1
    for src_pass, _rel, _reason in merged.rejected_edges:
        by_pass_rejected[src_pass] += 1

    with get_sync_session() as session:
        all_pass_rows = (
            session.query(StageRun)
            .filter(
                StageRun.pipeline_run_id == pipeline_run_id,
                StageRun.stage_name == "derive_ontology_graph",
                StageRun.pass_name.isnot(None),
                StageRun.execution_status == "COMPLETE",
            )
            .all()
        )
        for row in all_pass_rows:
            extracted = by_pass_accepted.get(row.pass_name, 0)
            rejected = by_pass_rejected.get(row.pass_name, 0)
            row.relationships_extracted = extracted
            row.relationships_rejected = rejected
            new_yield = classify_yield_from_counts(
                primary=row.primary_entities_extracted or 0,
                bridge=row.bridge_entities_extracted or 0,
                extracted_rels=extracted,
                rejected_rels=rejected,
            )
            # Only apply HIT → DEGRADED transitions.
            if row.yield_status == "HIT" and new_yield == YieldStatus.DEGRADED:
                row.yield_status = "DEGRADED"
        session.commit()
```

- [ ] **Step 3: Implement `_write_pipeline_run_metrics`**

```python
def _write_pipeline_run_metrics(pipeline_run_id, merged, manifest) -> None:
    """Populate PipelineRun.metrics per spec §5.4 + §6.6."""
    from app.db.session import get_sync_session
    from app.models.ingest import PipelineRun

    pass_outcomes = _build_pass_outcomes_rollup(pipeline_run_id)
    total_extracted = len(merged.edges)
    total_rejected = len(merged.rejected_edges)
    ratio = (
        total_rejected / (total_extracted + total_rejected)
        if (total_extracted + total_rejected) > 0 else 0.0
    )
    rejection_sample = _build_rejection_sample(merged)

    with get_sync_session() as session:
        run = session.get(PipelineRun, pipeline_run_id)
        run.metrics = {
            "pass_outcomes": pass_outcomes,
            "document_extraction_anomaly": all(
                pass_outcomes.get(name, {}).get("yield") in ("EMPTY", "BRIDGES_ONLY")
                for name in ("radar_domain", "missile_domain", "other_systems")
                if pass_outcomes.get(name, {}).get("execution") == "COMPLETE"
            ),
            "pass_degraded_count": sum(
                1 for outcome in pass_outcomes.values()
                if outcome.get("yield") == "DEGRADED"
            ),
            "overall_relationship_rejection_ratio": ratio,
            "rejected_relationships_sample": rejection_sample,
            "bundle_legacy": False,
            "bundle_key_display": manifest.bundle_key,
        }
        session.commit()


def _build_pass_outcomes_rollup(pipeline_run_id) -> dict:
    """Query v_latest_pass_attempts view for this run, roll up by pass name."""
    from app.db.session import get_sync_session
    from sqlalchemy import text
    with get_sync_session() as session:
        rows = session.execute(
            text(
                "SELECT pass_name, execution_status, yield_status, attempt, "
                "primary_entities_extracted, bridge_entities_extracted, "
                "relationships_extracted, relationships_rejected "
                "FROM ingest.v_latest_pass_attempts "
                "WHERE pipeline_run_id = :run_id"
            ),
            {"run_id": pipeline_run_id},
        ).all()
    return {
        row.pass_name: {
            "execution": row.execution_status,
            "yield": row.yield_status,
            "attempt": row.attempt,
            "primary": row.primary_entities_extracted,
            "bridge": row.bridge_entities_extracted,
            "extracted": row.relationships_extracted,
            "rejected": row.relationships_rejected,
        }
        for row in rows
    }


def _build_rejection_sample(merged) -> dict:
    """Up to 20 rejections per pass per reason for diagnostics."""
    from collections import defaultdict
    sample = defaultdict(lambda: defaultdict(list))
    for src_pass, rel, reason in merged.rejected_edges:
        bucket = sample[src_pass][reason.value]
        if len(bucket) < 20:
            bucket.append(_rel_to_dict(rel))
    return {k: dict(v) for k, v in sample.items()}


def _rel_to_dict(rel) -> dict:
    """Convert a raw rejection source object to a JSON-serializable dict."""
    if hasattr(rel, "model_dump"):
        return rel.model_dump()
    if hasattr(rel, "__dict__"):
        return {k: v for k, v in rel.__dict__.items() if not k.startswith("_")}
    return {"repr": repr(rel)}
```

- [ ] **Step 4: Implement `_upsert_document_graph_extraction`**

Per spec §5.7:

```python
def _upsert_document_graph_extraction(*, document_id, pipeline_run_id, run, merged) -> None:
    from app.db.session import get_sync_session
    from app.models.ingest import DocumentGraphExtraction
    from datetime import datetime

    values = dict(
        pipeline_run_id=pipeline_run_id,
        ontology_bundle_key=run.ontology_bundle_key,
        ontology_name=run.ontology_name,
        ontology_version=run.ontology_version,
        use_case_key=run.use_case_key,
        extraction_profile_version=run.extraction_profile_version,
        graph_json=_serialize_for_audit(merged),
        updated_at=datetime.utcnow(),
    )

    with get_sync_session() as session:
        existing = (
            session.query(DocumentGraphExtraction)
            .filter_by(document_id=document_id)
            .first()
        )
        if existing:
            for k, v in values.items():
                setattr(existing, k, v)
        else:
            session.add(DocumentGraphExtraction(document_id=document_id, **values))
        session.commit()


def _serialize_for_audit(merged) -> dict:
    """Audit blob — counts and samples, not full graph. Spec §5.7."""
    from collections import Counter
    entity_count_by_type: dict[str, int] = Counter()
    edge_count_by_type: dict[str, int] = Counter()
    rejection_reasons: dict[str, int] = Counter()
    primary_total = 0
    bridge_total = 0

    for e in merged.entities:
        entity_count_by_type[e.identity.entity_type] += 1
        # Assume a helper distinguishes primary vs bridge; if none exists
        # yet, infer from manifest lookups in a follow-up.
    for edge in merged.edges:
        edge_count_by_type[edge.rel_type] += 1
    for _src, _rel, reason in merged.rejected_edges:
        rejection_reasons[reason.value] += 1

    return {
        "entity_count_by_type": dict(entity_count_by_type),
        "edge_count_by_type": dict(edge_count_by_type),
        "primary_entities_total": primary_total,
        "bridge_entities_total": bridge_total,
        "edges_accepted": len(merged.edges),
        "edges_rejected": len(merged.rejected_edges),
        "rejection_reasons": dict(rejection_reasons),
        "pass_summaries": [],  # populated from _build_pass_outcomes_rollup if desired
    }
```

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/unit/test_pipeline_metrics.py -v
```

Fix, then commit.

```bash
git add app/workers/pipeline.py tests/unit/test_pipeline_metrics.py
git commit -m "feat(pipeline): merge post-processing, metrics, snapshot write

Per spec §5.4 + §5.7 + §6.6. _apply_post_merge_yield_updates recomputes
yield_status for each pass after merge rejects edges (HIT→DEGRADED is
the only allowed transition). _write_pipeline_run_metrics populates
PipelineRun.metrics with pass outcomes, anomaly flag, rejection ratio,
and sample. _upsert_document_graph_extraction writes the
audit-blob-shaped graph_json per spec §5.7.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.6: Wire rollback helpers + implement new `derive_ontology_graph` branch

**Files:**
- Modify: `app/workers/pipeline.py`
- Create: `tests/unit/test_derive_ontology_graph_bundle_passes.py`

Spec §5.4 orchestrator. This is the task that actually puts the new branch behind the feature flag. After this commit, setting `graph_extraction_engine=bundle_passes` and restarting workers routes traffic to the new path.

- [ ] **Step 1: Wire `_delete_extraction_layer_graph` + `_attempt_rollback`**

Replace the stubs from Task 4.1:

```python
def _delete_extraction_layer_graph(document_id: str) -> None:
    """Abstract helper; wired to the narrower graph-store primitive
    from Chunk 3 Task 3.2. Spec §5.4 + residual check #1."""
    from app.db.session import get_graph_store
    graph_store = get_graph_store()
    graph_store.delete_extraction_layer_graph_sync(document_id)


def _attempt_rollback(document_id: str) -> str:
    """Best-effort rollback. Returns a diagnostic suffix for the stage
    error_message — empty on success, '; ROLLBACK_ALSO_FAILED: <detail>'
    on failure. Spec §5.4."""
    try:
        _delete_extraction_layer_graph(document_id)
        return ""
    except Exception as rollback_exc:
        logger.error(
            "rollback during failure handling also failed: %s", rollback_exc,
        )
        return f"; ROLLBACK_ALSO_FAILED: {rollback_exc}"
```

- [ ] **Step 2: Implement `_update_document_pipeline_status`**

```python
def _update_document_pipeline_status(document_id: str, new_status: str) -> None:
    """Update Document.pipeline_status using the EXISTING vocabulary
    from app/models/ingest.py:60. Spec §5.4."""
    from app.db.session import get_sync_session
    from app.models.ingest import Document
    with get_sync_session() as session:
        doc = session.get(Document, document_id)
        if doc:
            doc.pipeline_status = new_status
            session.commit()
```

- [ ] **Step 3: Add the new branch to `derive_ontology_graph`**

Find the existing `derive_ontology_graph` Celery task. Wrap it so the feature flag dispatches:

```python
@celery_app.task(bind=True, name="derive_ontology_graph", queue="graph")
def derive_ontology_graph(self, pipeline_run_id: str) -> dict:
    """Orchestrator entry point. Dispatches to legacy or bundle_passes
    branch based on settings.graph_extraction_engine. Spec §5.4 +
    §7.4 feature flag."""
    settings = get_settings()
    if settings.graph_extraction_engine == "bundle_passes":
        return _derive_ontology_graph_bundle_passes(self, pipeline_run_id)
    # Legacy branch: preserved unchanged
    return _derive_ontology_graph_legacy(self, pipeline_run_id)
```

Rename the existing body to `_derive_ontology_graph_legacy(self, pipeline_run_id)`. Touch nothing else about its behavior.

Add the new branch implementing the full §5.4 orchestrator:

```python
def _derive_ontology_graph_bundle_passes(self, pipeline_run_id: str) -> dict:
    """New path: fixed per-pass templates, merge, import, rollback. Spec §5.4."""
    from app.db.session import get_sync_session
    from app.models.ingest import PipelineRun, StageRun
    from app.services.extraction_merge import merge_and_resolve
    from app.services.ontology_bundles import load_bundle_manifest
    from app.services.ontology_templates import load_ontology
    from datetime import datetime

    # --- Stage-summary row bookkeeping ---
    with get_sync_session() as session:
        run = session.get(PipelineRun, pipeline_run_id)
        stage_summary = StageRun(
            pipeline_run_id=run.id,
            stage_name="derive_ontology_graph",
            pass_name=None,
            attempt=self.request.retries + 1,
            status="RUNNING",
            started_at=datetime.utcnow(),
        )
        session.add(stage_summary)
        session.commit()
        stage_summary_id = stage_summary.id
        run_document_id = run.document_id
        run_mode = run.mode

    tracker = GraphWriteTracker()

    def _terminalize_failure(exc_type: str, error_msg: str, should_rollback: bool):
        """Shared bookkeeping for the three failure branches below."""
        rollback_note = _attempt_rollback(run_document_id) if should_rollback else ""
        try:
            with get_sync_session() as session:
                row = session.get(StageRun, stage_summary_id)
                row.status = "FAILED"
                row.execution_status = "FAILED"
                row.rollback_executed = should_rollback
                row.error_message = f"{exc_type}: {error_msg}{rollback_note}"
                row.finished_at = datetime.utcnow()
                run_row = session.get(PipelineRun, pipeline_run_id)
                run_row.status = "FAILED"
                run_row.finished_at = datetime.utcnow()
                session.commit()
            _update_document_pipeline_status(run_document_id, "PARTIAL_COMPLETE")
        except Exception as bookkeeping_exc:
            logger.error(
                "derive_ontology_graph: bookkeeping update also failed: %s",
                bookkeeping_exc,
            )

    try:
        # Load bundle + ontology + docling document
        with get_sync_session() as session:
            run = session.get(PipelineRun, pipeline_run_id)
            bundle_key = run.ontology_bundle_key
        manifest = load_bundle_manifest(bundle_key)
        ontology = load_ontology(bundle_key=bundle_key)
        doc_json = _build_docling_document_json(run_document_id)

        pass_results: dict = {}
        upstream_refs: dict = {}

        for pass_def in manifest.passes:
            _run_single_pass(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                manifest=manifest,
                ontology=ontology,
                bundle_key=bundle_key,
                doc_json=doc_json,
                pass_results=pass_results,
                upstream_refs=upstream_refs,
                document_id=run_document_id,
            )

        # Required-pass gate
        gate = check_required_pass_gate(pipeline_run_id)
        if not gate.passed:
            raise IngestFailed(f"Required passes failed: {gate.failures}")

        # Merge and resolve
        merged = merge_and_resolve(
            pass_results=pass_results,
            manifest=manifest,
            ontology=ontology,
            document_id=run_document_id,
            pipeline_run_id=str(pipeline_run_id),
        )

        _apply_post_merge_yield_updates(pipeline_run_id, merged)
        _write_pipeline_run_metrics(pipeline_run_id, merged, manifest)

        # Three-phase import
        identity_to_rid = _import_graph_phase_nodes(
            merged, ontology, run_document_id, tracker,
        )
        _import_graph_phase_domain_edges(merged, ontology, tracker)
        _import_graph_phase_structural_edges(
            merged, identity_to_rid, run_document_id, str(pipeline_run_id), tracker,
        )

        # Snapshot write (PostgreSQL only — tracker stays unchanged)
        with get_sync_session() as session:
            run_row = session.get(PipelineRun, pipeline_run_id)
            _upsert_document_graph_extraction(
                document_id=run_document_id,
                pipeline_run_id=pipeline_run_id,
                run=run_row,
                merged=merged,
            )

        # Success terminalization — mode-conditional for PipelineRun
        with get_sync_session() as session:
            row = session.get(StageRun, stage_summary_id)
            row.status = "COMPLETE"
            row.execution_status = "COMPLETE"
            row.rollback_executed = False
            row.finished_at = datetime.utcnow()

            run_row = session.get(PipelineRun, pipeline_run_id)
            if run_mode == "graph_only":
                run_row.status = "COMPLETE"
                run_row.finished_at = datetime.utcnow()
            session.commit()

        if run_mode == "graph_only":
            _update_document_pipeline_status(run_document_id, "COMPLETE")

        return {
            "stage": "derive_ontology_graph",
            "status": "ok",
            "entities": len(merged.entities),
            "edges": len(merged.edges),
        }

    except IngestFailed as exc:
        # Gate failure — no graph writes occurred, no rollback needed.
        _terminalize_failure("gate_failed", str(exc), should_rollback=False)
        raise
    except (MergeError, GraphImportError) as exc:
        logger.exception("derive_ontology_graph merge/import failure")
        _terminalize_failure(
            "merge_or_import_failed", str(exc),
            should_rollback=tracker.any_mutation_attempted,
        )
        raise
    except Exception as exc:
        logger.exception("derive_ontology_graph unexpected failure")
        _terminalize_failure(
            "unexpected_failure", str(exc),
            should_rollback=tracker.any_mutation_attempted,
        )
        raise
```

Define `MergeError` and `GraphImportError` exception classes alongside `IngestFailed` — they can be simple `Exception` subclasses; the spec treats them the same.

- [ ] **Step 4: Write orchestrator-branch tests**

Create `tests/unit/test_derive_ontology_graph_bundle_passes.py`:

- Happy path: mocked `_run_single_pass` populates `pass_results`, mocked `merge_and_resolve` returns a populated `MergedExtraction`, mocked import phases flip the tracker; assert stage_summary row ends `COMPLETE`, `run.status == "COMPLETE"` for graph_only, `PROCESSING` for full.
- Gate failure path: `check_required_pass_gate` returns `GateResult(passed=False, ...)`; assert `_attempt_rollback` NOT called, `rollback_executed=False`, `run.status == "FAILED"`, `Document.pipeline_status == "PARTIAL_COMPLETE"`.
- Merge failure pre-mutation: `merge_and_resolve` raises `MergeError`; tracker never marked; `_attempt_rollback` NOT called; `rollback_executed=False`.
- Import failure mutation-time: phase 2 calls `tracker.mark()` then raises `GraphImportError`; `_attempt_rollback` called; `rollback_executed=True`.
- Unexpected failure post-mutation: phase 3 raises; tracker already True from phase 2; rollback runs.
- Bookkeeping exception swallowed: when bookkeeping update raises AFTER the original exception, the original exception re-raises correctly (no mask).

- [ ] **Step 5: Run tests — feature flag OFF in most tests**

The unit tests above set `graph_extraction_engine=bundle_passes` via a monkeypatch fixture. The wider pytest suite should still default to `legacy` so unrelated tests don't drift.

```bash
pytest tests/unit/test_derive_ontology_graph_bundle_passes.py -v
pytest tests/ 2>&1 | tail -20
```

Expected: new tests green, full suite green.

- [ ] **Step 6: Commit**

```bash
git add app/workers/pipeline.py tests/unit/test_derive_ontology_graph_bundle_passes.py
git commit -m "feat(pipeline): new derive_ontology_graph branch behind feature flag

Per spec §5.4 orchestrator. Adds the bundle_passes branch which
wires through manifest load, per-pass loop, required-pass gate,
merge_and_resolve, post-merge yield updates, run metrics write,
three-phase graph import, snapshot write, and mode-conditional
terminalization. Three failure branches (gate / merge-or-import /
unexpected) share a _terminalize_failure helper and all honor the
GraphWriteTracker gate for rollback.

Routed via graph_extraction_engine feature flag (default 'legacy').
Legacy path renamed to _derive_ontology_graph_legacy and is
untouched. Nothing has switched traffic — the flag flip happens
during soak (spec §7.4).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.7: Implement the status API (spec §7.10)

**Files:**
- Modify: `app/api/v1/sources.py` (status endpoint)
- Create: `app/services/status_signals.py`
- Create: `tests/unit/test_compute_status_signals.py`
- Create: `tests/integration/test_status_api_shape.py`

Spec §7.10 three-concept split + cross-run `graph_queryable` query.

- [ ] **Step 1: Write failing tests for `compute_status_signals`**

Create `tests/unit/test_compute_status_signals.py` covering every row of the case tables in spec §7.10:

- Fresh successful extraction: snapshot present, no rollback → `graph_queryable=True`, `is_stale=False`
- `graph_only` gate failure with preserved prior snapshot → `graph_queryable=True`, `is_stale=True`
- `graph_only` mutation-time failure with `rollback_executed=True` → `graph_queryable=False`
- Run B rolled back, run C in pre-purge → `graph_queryable=False` (cross-run query catches B's rollback)
- `full` pre-purge failure → prior snapshot intact, `graph_queryable=True`
- `full` post-purge pre-success → `graph_snapshot=None`, `graph_queryable=False`
- No successful extraction ever → `graph_snapshot=None`, `graph_queryable=False`
- Snapshot orphan (snapshot_run deleted) → legacy safety behavior: conservative `graph_queryable` via "any rollback" fallback

- [ ] **Step 2: Implement `compute_status_signals`**

Create `app/services/status_signals.py`:

```python
"""Spec §7.10 status signals computation.

`graph_queryable` uses a CROSS-RUN query, not a latest-run query.
The composite (started_at, id) ordering ensures a deterministic total
order even when two PipelineRuns for the same document share a
started_at timestamp."""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models.ingest import (
    DocumentGraphExtraction, PipelineRun, StageRun,
)
from app.services.ontology_bundles import StatusSignals


def compute_status_signals(document_id: str, session: Session) -> StatusSignals:
    """Spec §7.10 authoritative rule. See that section for the full
    derivation and case tables."""
    snapshot = (
        session.query(DocumentGraphExtraction)
        .filter_by(document_id=document_id)
        .first()
    )

    latest_run = (
        session.query(PipelineRun)
        .filter_by(document_id=document_id)
        .order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
        .first()
    )

    is_stale = False
    if snapshot is not None:
        is_stale = (
            latest_run is None
            or latest_run.id != snapshot.pipeline_run_id
            or latest_run.status != "COMPLETE"
        )

    if snapshot is None:
        graph_queryable = False
    else:
        snapshot_run = (
            session.query(PipelineRun)
            .filter_by(id=snapshot.pipeline_run_id)
            .first()
        )
        q = (
            session.query(StageRun)
            .join(PipelineRun, StageRun.pipeline_run_id == PipelineRun.id)
            .filter(
                PipelineRun.document_id == document_id,
                StageRun.stage_name == "derive_ontology_graph",
                StageRun.pass_name.is_(None),
                StageRun.rollback_executed.is_(True),
            )
        )
        if snapshot_run is not None:
            # Composite row-constructor comparison for "strictly newer"
            q = q.filter(
                sa.tuple_(PipelineRun.started_at, PipelineRun.id)
                > sa.tuple_(snapshot_run.started_at, snapshot_run.id)
            )
        graph_invalidated = q.first() is not None
        graph_queryable = not graph_invalidated

    return StatusSignals(
        snapshot=snapshot,
        is_stale=is_stale,
        graph_queryable=graph_queryable,
    )
```

- [ ] **Step 3: Update the status endpoint response shape**

Find the document-status endpoint in `app/api/v1/sources.py` (search for `@router.get.*status` or the existing status return shape). Rewrite its response shaper to match spec §7.10:

```python
# Inside the status endpoint:
from app.services.status_signals import compute_status_signals

def _shape_status_response(document, latest_run, signals):
    """Spec §7.10 response shape."""
    # latest_run block (empty passes / null stage_summary when
    # derive_ontology_graph hasn't started yet)
    pass_rows = session.query(StageRun).filter(
        StageRun.pipeline_run_id == latest_run.id,
        StageRun.stage_name == "derive_ontology_graph",
        StageRun.pass_name.isnot(None),
    ).order_by(StageRun.started_at).all() if latest_run else []
    stage_summary_row = session.query(StageRun).filter(
        StageRun.pipeline_run_id == latest_run.id,
        StageRun.stage_name == "derive_ontology_graph",
        StageRun.pass_name.is_(None),
    ).first() if latest_run else None

    latest_run_block = None
    if latest_run is not None:
        latest_run_block = {
            "pipeline_run_id": str(latest_run.id),
            "status": latest_run.status,
            "mode": latest_run.mode,
            "started_at": latest_run.started_at.isoformat() if latest_run.started_at else None,
            "finished_at": latest_run.finished_at.isoformat() if latest_run.finished_at else None,
            "ontology_bundle_key": latest_run.ontology_bundle_key,
            "ontology_bundle_label": latest_run.ontology_bundle_key or "legacy/unknown",
            "ontology_name": latest_run.ontology_name,
            "ontology_version": latest_run.ontology_version,
            "passes": [_row_to_pass(r) for r in pass_rows],
            "stage_summary": _row_to_summary(stage_summary_row) if stage_summary_row else None,
        }

    snapshot_block = None
    if signals.snapshot is not None:
        snapshot_block = {
            "pipeline_run_id": str(signals.snapshot.pipeline_run_id),
            "ontology_bundle_key": signals.snapshot.ontology_bundle_key,
            "ontology_bundle_label": signals.snapshot.ontology_bundle_key or "legacy/unknown",
            "ontology_version": signals.snapshot.ontology_version,
            "entity_count": _count_entities(signals.snapshot),
            "edge_count": _count_edges_snapshot(signals.snapshot),
            "updated_at": signals.snapshot.updated_at.isoformat(),
            "is_stale": signals.is_stale,
        }

    return {
        "document_id": str(document.id),
        "document_status": document.pipeline_status,
        "latest_run": latest_run_block,
        "graph_snapshot": snapshot_block,
        "graph_queryable": signals.graph_queryable,  # TOP-LEVEL — always meaningful
    }
```

`_row_to_pass` and `_row_to_summary` map a StageRun row to the JSON shape shown in spec §7.10. Keep them as local helper functions in the same module.

- [ ] **Step 4: Integration test — response shape**

Create `tests/integration/test_status_api_shape.py` that drives the real endpoint with seeded fixtures and asserts:
- Top-level keys: `document_id`, `document_status`, `latest_run`, `graph_snapshot`, `graph_queryable`
- `graph_queryable` is present regardless of `graph_snapshot` being null or populated
- `is_stale` nested inside `graph_snapshot` only when `graph_snapshot is not None`
- `latest_run.passes == []` and `latest_run.stage_summary is None` when `derive_ontology_graph` hasn't started

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/unit/test_compute_status_signals.py tests/integration/test_status_api_shape.py -v
```

```bash
git add app/api/v1/sources.py app/services/status_signals.py \
       tests/unit/test_compute_status_signals.py tests/integration/test_status_api_shape.py
git commit -m "feat(api): status API with cross-run graph_queryable rule (§7.10)

Per spec §7.10. compute_status_signals implements the authoritative
rule: snapshot from DocumentGraphExtraction, is_stale from latest-run
equivalence, graph_queryable from a CROSS-RUN query that checks
whether any run strictly newer than the snapshot's run has a
derive_ontology_graph summary with rollback_executed=True. Composite
(started_at, id) ordering for strict-newer-than comparison.

Status endpoint rewritten to return the three-concept split:
document_status, latest_run (with passes[] and stage_summary),
graph_snapshot (nullable with nested is_stale), and top-level
graph_queryable that remains meaningful even when graph_snapshot is null.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.8: Baseline harness + run comparison

**Files:**
- Create: `tools/extraction_baseline_harness.py`
- Create: `tests/integration/test_baseline_comparison.py`

Spec §8.3 baseline harness. Runs the same corpus through both paths (by flipping the flag) and compares per-entity-type and per-edge-type counts + rejection rates.

- [ ] **Step 1: Write the harness**

Create `tools/extraction_baseline_harness.py`:

```python
#!/usr/bin/env python3
"""Baseline harness for PR 2 comparison. Spec §8.3.

Runs a fixed corpus through both the legacy and bundle_passes paths
and reports per-entity-type and per-edge-type counts plus rejection
rates. Does NOT flip the global feature flag — that must be done at
the infrastructure level (env var + worker restart). Instead, the
harness assumes the caller has ALREADY deployed two instances (or
runs sequential invocations with the flag set differently in each)."""
import argparse
import json
import sys
from pathlib import Path


def run_corpus(corpus_dir: Path, engine_label: str) -> dict:
    """Ingest each document in corpus_dir and collect the resulting
    pipeline_run_id + PipelineRun.metrics + per-pass yield stats."""
    ...


def compare(legacy: dict, bundle: dict) -> dict:
    """Compare two run manifests. Returns a diff dict with:
      - per_entity_type_counts: {type: (legacy, bundle, delta)}
      - per_edge_type_counts: {type: (legacy, bundle, delta)}
      - overall_rejection_ratio: (legacy, bundle, delta)
      - pass_yield_distribution: (legacy, bundle)
      - baseline_met: bool — true iff all criteria in spec §8.3 pass"""
    ...


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--legacy-results", type=Path)
    parser.add_argument("--bundle-results", type=Path)
    parser.add_argument("--collect", choices=["legacy", "bundle_passes"])
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.collect:
        result = run_corpus(args.corpus, args.collect)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.compare:
        legacy = json.loads(args.legacy_results.read_text())
        bundle = json.loads(args.bundle_results.read_text())
        diff = compare(legacy, bundle)
        print(json.dumps(diff, indent=2, default=str))
        return 0 if diff["baseline_met"] else 1

    parser.error("provide either --collect or --compare")


if __name__ == "__main__":
    sys.exit(main())
```

Implement `run_corpus` (iterate files, call the upload API, poll for completion, read PipelineRun.metrics) and `compare` (field-by-field comparison with the spec §8.3 thresholds). Baseline criteria per spec §8.3: extraction rates within 10% of legacy, rejection ratio within 5 percentage points, no pass regresses from HIT to EMPTY/DEGRADED.

- [ ] **Step 2: Run the harness in dev**

Per the soak procedure (spec §7.4):

```bash
# Collect legacy baseline
DOCLING_GRAPH_EXTRACTION_ENGINE=legacy docker compose up -d worker
python tools/extraction_baseline_harness.py --corpus corpus/ --collect legacy > legacy.json

# Collect new baseline
DOCLING_GRAPH_EXTRACTION_ENGINE=bundle_passes docker compose up -d worker
python tools/extraction_baseline_harness.py --corpus corpus/ --collect bundle_passes > bundle.json

# Compare
python tools/extraction_baseline_harness.py --compare \
    --legacy-results legacy.json --bundle-results bundle.json
```

Expected: `baseline_met=true` in the diff output, OR a clear per-field breakdown of where the new path diverges. If divergences exist that aren't in the "acceptable drift" bucket, investigate before proceeding with soak.

- [ ] **Step 3: Commit**

```bash
git add tools/extraction_baseline_harness.py
git commit -m "feat(tools): extraction baseline harness for PR 2 comparison

Per spec §8.3. Collects per-entity-type and per-edge-type counts
plus yield/rejection stats from a fixed corpus and compares the
legacy and bundle_passes paths. Exit code 0 when baseline criteria
are met (extraction rates within 10%, rejection ratio within 5pp,
no HIT→EMPTY/DEGRADED regressions). Run during the soak procedure
per spec §7.4.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 4.9: PR 2 integration smoke test + open PR

**Files:**
- Create: `tests/integration/test_pr2_switchover_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
"""PR 2 smoke: with graph_extraction_engine=bundle_passes, the full
ingest chain runs end-to-end and produces a queryable graph."""
import pytest
from unittest.mock import patch


def test_bundle_passes_end_to_end(monkeypatch):
    """Flip the flag for one test only, ingest a small canary, assert:
    - PipelineRun row exists with mode=full and the bundle snapshotted
    - StageRun rows: one per pass + one stage summary
    - DocumentGraphExtraction row has the audit blob shape
    - graph_queryable=True via compute_status_signals
    - PipelineRun.metrics has pass_outcomes, document_extraction_anomaly,
      bundle_key_display, bundle_legacy=False"""
    monkeypatch.setenv("GRAPH_EXTRACTION_ENGINE", "bundle_passes")
    from app.config import get_settings
    get_settings.cache_clear()  # force re-read after env change
    assert get_settings().graph_extraction_engine == "bundle_passes"
    # ... the rest of the test drives an actual ingest and asserts
    ...


def test_legacy_still_works_with_default_flag():
    """Default flag (legacy) still produces a graph end-to-end."""
    ...
```

- [ ] **Step 2: Run the smoke test**

```bash
docker compose up -d postgres redis arcadedb worker api docling-graph
pytest tests/integration/test_pr2_switchover_smoke.py -v
```

Expected: both pass.

- [ ] **Step 3: Commit and open PR 2**

```bash
git add tests/integration/test_pr2_switchover_smoke.py
git commit -m "test(pr2): switchover smoke

Verifies the bundle_passes branch runs end-to-end when the flag is
flipped per-test, and that the default (legacy) path is still intact.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"

git push origin feature/extraction-refactor

gh pr create --title "PR 2: Extraction refactor — switchover behind feature flag" --body "$(cat <<'EOF'
## Summary

PR 2 of the three-PR extraction refactor. Lands everything needed to
switch the extraction hot path to the new bundle-passes engine behind
the graph_extraction_engine feature flag. Flag defaults to legacy —
the flip happens during the soak procedure per spec §7.4, not at
merge time.

Foundations (Chunk 3):
- New narrower rollback primitive delete_extraction_layer_graph_sync
  on GraphStore protocol + ArcadeDBGraphStore
- ProvenanceMetadata.pipeline_run_id optional field (additive)
- Full app/services/extraction_merge.py module (LogicalIdentity,
  PassResult, MergedEntityRecord, MergedEdgeRecord, MergedExtraction,
  RelationshipRejectionReason, YieldStatus, merge_and_resolve,
  build_display_label, classify_yield*)
- StatusSignals dataclass in app/services/ontology_bundles.py
- graph_extraction_engine feature flag (Literal['legacy','bundle_passes'],
  default 'legacy')
- IngestDispatchResult frozen dataclass in app/workers/dispatch_types.py
- Bundle threading on SourceCreate/SourceResponse and new
  ReingestRequest schema

Orchestrator (Chunk 4):
- derive_ontology_graph dispatches on the flag; legacy branch renamed
  to _derive_ontology_graph_legacy (untouched)
- New _derive_ontology_graph_bundle_passes branch implements the full
  §5.4 orchestrator with per-pass loop, required-pass gate, merge,
  three-phase import, mode-conditional terminalization, and
  tracker-gated rollback
- start_ingest_pipeline returns IngestDispatchResult and snapshots
  bundle metadata onto PipelineRun
- reingest_graph_only uses resolve_bundle_key_for_graph_only with
  latest-run inheritance
- Status API rewritten to spec §7.10 three-concept split with
  cross-run graph_queryable query
- Baseline harness (tools/extraction_baseline_harness.py) for soak

## Feature flag

After merge, the flag defaults to legacy. To switch:
1. Set env DOCLING_GRAPH_EXTRACTION_ENGINE=bundle_passes
2. Restart worker and beat
3. Run the baseline harness against the soak corpus
4. Monitor metrics per spec §7.8 for 7 days

## Test plan
- [ ] Full pytest suite passes with default flag
- [ ] test_derive_ontology_graph_bundle_passes.py passes with the flag
      flipped per-test
- [ ] Legacy e2e still produces a graph end-to-end
- [ ] Status API returns the §7.10 shape for both snapshot-present and
      snapshot-null cases
- [ ] Baseline harness reports baseline_met=true in dev/staging
- [ ] Rollback integration test still passes after any graph_store changes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Begin soak procedure**

Per spec §7.4 soak procedure (operational, not a plan task):

1. Merge PR 2.
2. Flip flag in dev; run baseline harness; verify baseline_met.
3. Flip flag in staging; run baseline harness; verify.
4. Flip flag in prod with monitoring ready.
5. Soak for at least 7 days per spec §7.4.
6. Rerun baseline at least twice during soak.
7. Verify no metric alerts fired (spec §7.8).
8. If all checks pass, proceed to Chunk 5 (PR 3 deletion).

---

**END OF CHUNK 4.** PR 2 is complete once Task 4.9 merges. The feature flag is still `legacy` by default — the soak procedure is operational work that happens outside this plan. Chunk 5 begins PR 3 (deletion + hardening) after the 7-day soak gate is satisfied.

---

## Chunk 5: PR 3 — Deletion and hardening

This chunk is PR 3. It MUST NOT start until the 7-day soak gate is satisfied (spec §7.5 pre-merge gates). After soak is green, this chunk deletes every legacy artifact, adds contract tests and CI lints to prevent regression, and renames `HealthResponse.template_count` → `schema_count`. All changes are within the same feature branch `feature/extraction-refactor`. Spec §7.5 + §7.9.

### Task 5.1: Verify pre-merge gates

**Files:** none (procedural)

Spec §7.5 pre-merge gates. This is a go/no-go checkpoint — if ANY of these is unmet, DO NOT proceed to the deletion tasks.

- [ ] **Step 1: Verify production has been running `bundle_passes` for ≥7 days**

```bash
# Check the deploy timeline against your ops system
# Expected: >= 168 hours of continuous bundle_passes in prod
```

If less than 7 days, stop and wait. No partial soak.

- [ ] **Step 2: Verify no incidents attributed to the new path**

```bash
# Check the incident tracker for the 7-day window
```

If ANY production incident is attributed to the bundle_passes path, stop, open a bug, and delay PR 3 until the issue is resolved and the soak clock restarts.

- [ ] **Step 3: Verify metric alerts never fired**

Query Grafana / alerting backend for the 7-day window. Per spec §7.8:
- `silent_format_json_degradations > 0` — must be 0
- `hidden_mode_drift_incidents > 0` — must be 0
- `required_pass_failure_rate > 10%` in any 1-hour window — must be 0
- `domain_entity_extraction_rate < baseline × 0.75` — must be 0
- Per-pass FAILED rate > 20% for any required pass — must be 0

If any alert fired, stop and investigate.

- [ ] **Step 4: Rerun baseline harness twice**

```bash
python tools/extraction_baseline_harness.py --corpus corpus/ --collect bundle_passes > bundle_$(date +%s).json
python tools/extraction_baseline_harness.py --compare --legacy-results legacy.json --bundle-results bundle_$(date +%s).json
```

Run the comparison at least twice during the 7-day window with consistent results. If results vary, extend the soak.

- [ ] **Step 5: At least one operator-verified reingest**

Pick a previously-ingested document, run `graph_only` reingest via the API, and verify with an operator that the output graph is what they expect. This is qualitative — spec §7.5 requirement.

- [ ] **Step 6: Document the gate evaluation in the PR 3 opening comment**

Copy the answers from Steps 1–5 into a text file you'll paste into the PR description. An auditable gate record matters.

### Task 5.2: Delete legacy extraction code

**Files:**
- Delete: `docker/docling-graph/app/template_builder.py`
- Delete: `docker/docling-graph/tests/test_template_builder.py`
- Delete: `app/services/layered_extraction.py`
- Delete: `tests/unit/test_layered_extraction.py`
- Delete: `app/services/ontology_layers.py`
- Delete: `tests/unit/test_ontology_layers.py`
- Delete: `ontology/layer_map.yaml`
- Delete: `ontology/ontology.yaml` (the symlink)
- Modify: `app/workers/pipeline.py`
- Modify: `docker-compose.yml`

Spec §7.5 deletions.

- [ ] **Step 1: Remove the legacy branch of `derive_ontology_graph`**

Open `app/workers/pipeline.py`. The current dispatcher is:

```python
def derive_ontology_graph(self, pipeline_run_id: str) -> dict:
    settings = get_settings()
    if settings.graph_extraction_engine == "bundle_passes":
        return _derive_ontology_graph_bundle_passes(self, pipeline_run_id)
    return _derive_ontology_graph_legacy(self, pipeline_run_id)
```

Replace with:

```python
def derive_ontology_graph(self, pipeline_run_id: str) -> dict:
    return _derive_ontology_graph_bundle_passes(self, pipeline_run_id)
```

Then delete `_derive_ontology_graph_legacy` entirely from the same file. Also delete:
- Any legacy helper functions it called that are not used by the new branch
- `graph_layered_*` related code (config reads, helpers)

- [ ] **Step 2: Delete the legacy modules**

```bash
git rm docker/docling-graph/app/template_builder.py \
       docker/docling-graph/tests/test_template_builder.py \
       app/services/layered_extraction.py \
       tests/unit/test_layered_extraction.py \
       app/services/ontology_layers.py \
       tests/unit/test_ontology_layers.py \
       ontology/layer_map.yaml
```

- [ ] **Step 3: Delete the `ontology/ontology.yaml` symlink**

```bash
# Verify it's still a symlink (not a real file)
ls -la ontology/ontology.yaml
# Expected: "ontology/ontology.yaml -> ../ontology_bundles/air_defense_v3/ontology.yaml"

git rm ontology/ontology.yaml
```

Nothing in the new path reads from `ontology/ontology.yaml` — the bundle loader uses the bundle's own path. The symlink was a PR 1/PR 2 compatibility shim for code that still assumed the legacy path existed.

- [ ] **Step 4: Remove the `./ontology` bind mount from `docker-compose.yml`**

Open `docker-compose.yml`, find the `docling-graph` service, and delete the `./ontology:/app/ontology:ro` volume line. With the symlink gone, this mount has nothing meaningful inside.

- [ ] **Step 5: Delete `ExtractAllRequest`, `ontology_definition`, and POST /extract-all**

In `docker/docling-graph/app/schemas.py`:
- Delete the `ExtractAllRequest` class entirely
- Delete `ontology_definition` field from any remaining schemas
- Delete any `ExtractAllResponse` if it only fed `/extract-all`

In `docker/docling-graph/app/main.py`:
- Delete the `@app.post("/extract-all")` handler entirely
- Delete `docling_graph_service.extract_graph_all()` if it's no longer called (verify with a grep first)
- Remove `ontology_definition` from any remaining request handlers

In `app/services/docling_graph_service.py` (or wherever the worker-side HTTP client lives):
- Delete `extract_graph_all` method
- Delete `ontology_definition` parameter from any helper

- [ ] **Step 6: Delete the feature flag itself**

Open `app/config.py`. Remove `graph_extraction_engine` field from `Settings`. Remove `graph_layered_shadow_mode` and `graph_layered_fail_open_to_single_pass` fields (they were Task 2.7 honest-failure stabilizers that are no longer needed — the new path doesn't use them).

- [ ] **Step 7: Run the full test suite**

```bash
pytest tests/ 2>&1 | tail -30
```

Expected: every test touched by the deletions either (a) is itself deleted, (b) updates because it was asserting against removed behavior, or (c) still passes because it was only referencing shared helpers. If any test fails because it imports a deleted module, update its imports. If a test was asserting on legacy behavior only, delete it.

- [ ] **Step 8: Rebuild both images and re-run the e2e test**

```bash
docker compose build worker docling-graph
docker compose up -d
pytest tests/e2e/test_full_pipeline.py -v
```

Expected: green end-to-end. The new path is now the ONLY path.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor(extraction): delete legacy path (PR 3)

Spec §7.5 deletions after 7-day soak:
- docker/docling-graph/app/template_builder.py + test
- app/services/layered_extraction.py + test
- app/services/ontology_layers.py + test
- ontology/layer_map.yaml
- ontology/ontology.yaml symlink
- docling-graph service /extract-all endpoint, ExtractAllRequest,
  ontology_definition parameter
- Legacy branch of derive_ontology_graph + _derive_ontology_graph_legacy
- settings.graph_extraction_engine feature flag
- settings.graph_layered_* config keys
- ./ontology bind mount from docker-compose.yml docling-graph service

derive_ontology_graph now dispatches unconditionally to the
bundle_passes path. All legacy references verified removed via the
CI lints landing in Task 5.4.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 5.3: Rename `HealthResponse.template_count` → `schema_count`

**Files:**
- Modify: `docker/docling-graph/app/schemas.py`
- Modify: `docker/docling-graph/app/main.py` (health endpoint)
- Modify: any worker-side code that reads `template_count` from the health response
- Modify: tests that reference the old name

Spec §7.9 rename.

- [ ] **Step 1: Grep for the old name**

```bash
grep -rn "template_count" docker/ app/ tests/
```

Capture every site.

- [ ] **Step 2: Rename**

Replace `template_count` with `schema_count` at every captured site. The `HealthResponse` model gets the renamed field; the health endpoint handler builds the response with `schema_count` keyed off the current bundle's template count.

- [ ] **Step 3: Run tests**

```bash
pytest tests/ 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add docker/docling-graph/app/ app/ tests/
git commit -m "refactor(health): rename template_count → schema_count

Per spec §7.9 intentional-naming carry-forwards. The old name was
actively misleading (the new path uses pre-loaded schemas, not
runtime-generated templates). This is the only rename in PR 3; all
other names are kept on purpose.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 5.4: CI lints for legacy-reference prevention

**Files:**
- Create: `tools/ci_lints.sh`
- Modify: `.github/workflows/ci.yml` (or your CI config equivalent)

Spec §7.5 CI additions. Each lint returns zero hits or fails CI.

- [ ] **Step 1: Write the lint script**

Create `tools/ci_lints.sh`:

```bash
#!/usr/bin/env bash
# CI lints enforcing the PR 3 deletions stay deleted.
# Spec §7.5 CI additions. Each lint exits nonzero on any hit.
set -euo pipefail

FAILED=0

check() {
    local description="$1"
    local pattern="$2"
    local scope="${3:-app/}"
    if grep -rn "$pattern" $scope > /tmp/lint_hits 2>/dev/null; then
        echo "FAIL: $description"
        cat /tmp/lint_hits
        FAILED=1
    else
        echo "PASS: $description"
    fi
}

# Import scope lint: service-side schemas never imported by worker
check "worker does not import service-side extraction_schemas" \
    "from ontology_bundles\..*\.extraction_schemas" \
    "app/"

# Unsafe confidence defaulting lint
check "no naked 'confidence or 0.0' defaulting" \
    "confidence or 0\.0" \
    "app/ docker/"

# prefer_active resurrection
check "no resurrected prefer_active kwarg" \
    "prefer_active" \
    "app/ docker/"

# ontology_definition absence
check "no ontology_definition references" \
    "ontology_definition" \
    "app/ docker/"

# /app/ontology/ path absence
check "no /app/ontology path references" \
    "/app/ontology" \
    "app/ docker/"

# graph_extraction_engine absence
check "no graph_extraction_engine references" \
    "graph_extraction_engine" \
    "app/"

# graph_layered_ absence
check "no graph_layered_ references" \
    "graph_layered_" \
    "app/"

# template_count absence (renamed in Task 5.3)
check "no template_count field references" \
    "template_count" \
    "app/ docker/"

if [ $FAILED -ne 0 ]; then
    exit 1
fi
echo "All CI lints passed."
```

Make it executable:
```bash
chmod +x tools/ci_lints.sh
```

- [ ] **Step 2: Wire into CI**

Add a step to the CI workflow that runs `./tools/ci_lints.sh` after the test suite. Exact CI config depends on the runner (GitHub Actions vs GitLab vs self-hosted) — add the step in the same stage as the coverage checker.

- [ ] **Step 3: Run the script locally**

```bash
./tools/ci_lints.sh
```

Expected: every check prints `PASS`. If any FAIL, fix the remaining references in the codebase before committing.

- [ ] **Step 4: Commit**

```bash
git add tools/ci_lints.sh .github/workflows/ci.yml
git commit -m "ci: legacy-reference lints (spec §7.5)

Eight grep-based lints wired into CI to prevent legacy references
from creeping back in: extraction_schemas worker imports, naked
confidence defaulting, prefer_active, ontology_definition, /app/ontology
paths, graph_extraction_engine, graph_layered_, template_count.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 5.5: Monkey-patch contract tests

**Files:**
- Create: `docker/docling-graph/tests/test_monkey_patches.py`

Spec §7.5 new contract tests. Detects upstream LiteLLM / docling-graph drift by asserting the three monkey-patch sites still have the shape the patches assume.

- [ ] **Step 1: Identify the three monkey-patch sites**

```bash
grep -n "_patched_" docker/docling-graph/app/main.py
```

There should be three patches (per spec). For each, capture:
- The target class and method name
- The version of the upstream library the patch assumes
- What the patch does

- [ ] **Step 2: Write one test per patch**

Create `docker/docling-graph/tests/test_monkey_patches.py`:

```python
"""Contract tests for the three docling-graph monkey patches. Spec §7.5.

Each test asserts the upstream target still has the shape the patch
expects. If an upstream dependency changes, the test fails loudly
instead of the patch silently no-op'ing or producing wrong output."""
import inspect


def test_build_request_patch_target_signature():
    """Patch 1: _patched_build_request — patches <module.class.method>.
    Asserts the target method still has the expected parameter list."""
    from <upstream_module> import <TargetClass>
    sig = inspect.signature(<TargetClass>.<target_method>)
    params = list(sig.parameters)
    # Expected by the patch: [...specific params it relies on...]
    assert "messages" in params or "prompt" in params  # Example — fill in actual
    ...


def test_patch_2_target():
    """Patch 2 contract check — fill in real signature assertion."""
    ...


def test_patch_3_target():
    """Patch 3 contract check — fill in real signature assertion."""
    ...
```

Each test name and body must be filled in after reading the actual monkey-patch sources in `main.py`. The test is intentionally NOT asserting the patch produces correct OUTPUT — it asserts the patch's ASSUMPTIONS about the upstream interface are still true. If an upstream update breaks an assumption, these tests fire before the patch has a chance to silently fail.

- [ ] **Step 3: Run the tests**

```bash
pytest docker/docling-graph/tests/test_monkey_patches.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add docker/docling-graph/tests/test_monkey_patches.py
git commit -m "test(docling-graph): monkey-patch contract tests

Per spec §7.5. Three tests, one per patch in main.py. Each asserts
the upstream target's signature still matches what the patch assumes,
so an upstream LiteLLM or docling-graph update that changes the
surface fires these tests instead of silently breaking the patch.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 5.6: Update `DocumentGraphExtraction.graph_json` docstring

**Files:**
- Modify: `app/models/ingest.py`

Spec §7.5 column docstring update.

- [ ] **Step 1: Find the column definition**

```bash
grep -n "graph_json" app/models/ingest.py
```

- [ ] **Step 2: Update the docstring / comment**

Replace the existing docstring or inline comment with:

```python
graph_json = Column(
    JSONB,
    nullable=True,
    doc="Audit blob (entity/edge counts, rejection reasons, pass summaries). "
        "NOT a serialized graph — the prior NetworkX node-link payload shape "
        "was replaced in PR 2. Read spec §5.7 serialize_for_audit for the "
        "authoritative shape.",
)
```

- [ ] **Step 3: Commit**

```bash
git add app/models/ingest.py
git commit -m "docs(model): DocumentGraphExtraction.graph_json is an audit blob

Per spec §7.5 + §7.9. The column name is deliberately kept, but the
semantics shifted from 'serialized graph payload' to 'extraction
audit blob'. Docstring now reflects that.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"
```

### Task 5.7: PR 3 exit verification + open PR

**Files:**
- Create: `tests/integration/test_pr3_post_deletion_smoke.py`

Spec §7.5 post-merge verification.

- [ ] **Step 1: Write a post-deletion smoke test**

```python
"""PR 3 post-deletion smoke test. Verifies all exit criteria from spec §7.5."""
import subprocess
from pathlib import Path


def test_deleted_files_really_gone():
    deleted = [
        "docker/docling-graph/app/template_builder.py",
        "app/services/layered_extraction.py",
        "app/services/ontology_layers.py",
        "ontology/layer_map.yaml",
        "ontology/ontology.yaml",
    ]
    for path in deleted:
        assert not Path(path).exists(), f"{path} should have been deleted"


def test_ci_lints_all_pass():
    result = subprocess.run(
        ["./tools/ci_lints.sh"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All CI lints passed." in result.stdout


def test_only_new_path_exists():
    """The feature-flag dispatcher should be gone — derive_ontology_graph
    must call _derive_ontology_graph_bundle_passes unconditionally."""
    from app.workers import pipeline
    import inspect
    src = inspect.getsource(pipeline.derive_ontology_graph)
    assert "_derive_ontology_graph_legacy" not in src
    assert "_derive_ontology_graph_bundle_passes" in src
    assert "graph_extraction_engine" not in src


def test_settings_no_legacy_flags():
    from app.config import Settings
    fields = Settings.model_fields
    assert "graph_extraction_engine" not in fields
    assert "graph_layered_shadow_mode" not in fields
    assert "graph_layered_fail_open_to_single_pass" not in fields
```

- [ ] **Step 2: Run the smoke test**

```bash
pytest tests/integration/test_pr3_post_deletion_smoke.py -v
```

Expected: all pass.

- [ ] **Step 3: Run the full suite one more time**

```bash
pytest tests/ 2>&1 | tail -30
./tools/ci_lints.sh
python tools/check_extraction_coverage.py
```

Expected: every test passes, every lint passes, coverage checker passes. If ANY fail, stop and fix before opening the PR.

- [ ] **Step 4: Run the e2e ingest one more time**

```bash
docker compose build worker docling-graph
docker compose up -d
pytest tests/e2e/test_full_pipeline.py -v
```

Expected: green. This is the final check that the deletion didn't leave a stale import or a missing helper.

- [ ] **Step 5: Commit and open PR 3**

```bash
git add tests/integration/test_pr3_post_deletion_smoke.py
git commit -m "test(pr3): post-deletion smoke

Verifies every legacy file is deleted, every CI lint passes,
derive_ontology_graph has no feature-flag branch, and Settings has
no legacy flag fields.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
"

git push origin feature/extraction-refactor

gh pr create --title "PR 3: Extraction refactor — legacy deletion + hardening" --body "$(cat <<'EOF'
## Summary

PR 3 of the three-PR extraction refactor. Delete the legacy extraction
path now that the 7-day soak on bundle_passes is complete (see the
gate evaluation below). After this PR, the only extraction path is the
fixed-template bundle_passes path.

Deletions (spec §7.5):
- docker/docling-graph/app/template_builder.py + test
- app/services/layered_extraction.py + test
- app/services/ontology_layers.py + test
- ontology/layer_map.yaml
- ontology/ontology.yaml symlink
- docling-graph /extract-all endpoint, ExtractAllRequest, ontology_definition
- Legacy branch of derive_ontology_graph
- settings.graph_extraction_engine flag
- settings.graph_layered_* config keys
- ./ontology bind mount from docker-compose

Renames:
- HealthResponse.template_count → schema_count (the only rename)

Added hardening:
- CI lints preventing 8 classes of legacy reference regression
- Three monkey-patch contract tests (docling-graph/tests/test_monkey_patches.py)
- Updated DocumentGraphExtraction.graph_json docstring (audit blob semantics)
- PR 3 post-deletion smoke test

## Pre-merge gate evaluation

<fill in with the Task 5.1 results — date of soak start, date window,
baseline harness results, incident check results, metric alert
evaluations, and operator reingest notes>

## Test plan
- [ ] Full pytest suite passes
- [ ] ./tools/ci_lints.sh returns zero
- [ ] Monkey-patch contract tests pass
- [ ] End-to-end ingest produces a graph
- [ ] Coverage checker passes

## Post-merge verification (first 24 hours)
- [ ] Production ingests still produce graphs
- [ ] No extraction regressions in Grafana
- [ ] No 500s on /extract-pass
- [ ] Status API shape unchanged

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Wait for CI and merge**

Once CI is green and the PR is approved, merge. Post-merge, the team should watch the extraction metric dashboards for 24 hours per spec §7.5 post-merge exit criteria. If any regression appears in that window, git revert PR 3, redeploy with the (now-gone) flag temporarily re-introduced, investigate, and re-land.

---

**END OF CHUNK 5.** PR 3 is complete. The three-PR extraction refactor is done. After PR 3 merges and the 24-hour post-merge window passes clean, the refactor is closed out. Rollback from this point requires a git revert + redeploy (spec §7.7 after-PR-3 rollback).
