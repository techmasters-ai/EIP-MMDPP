# Docling-Graph Docs-Alignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realign the `air_defense_v3` canonical ontology and extraction schemas with `docling-graph-docs.md` — fix the 12 `graph_id_fields=[]` anti-patterns, adopt components-as-first-class-vertices per docs, replace the LLM `reference` pass with deterministic Docling-derived anchors, and unblock the live-extraction gate of the existing schema-compliance plan.

**Architecture:** Three architectural shifts: (1) document structure (SECTION/FIGURE/TABLE/DOCUMENT) comes from a deterministic Docling walker, not the LLM; (2) 12 value-object entities demote to `is_entity=False` components with content-based identity, requiring a walker change in `extraction_merge.py`; (3) structural-layout edges (`HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF`) live in `_STRUCTURAL_EDGE_TYPES`, not the ontology relationship matrix. Migration wipes derived state and re-ingests all 21 docs.

**Tech Stack:** Python (Pydantic v2), ArcadeDB, Docling/docling-graph, Celery, PostgreSQL, pytest. See `docling-graph-docs.md` (authoritative) and the approved design spec `docs/superpowers/specs/2026-04-16-docling-graph-docs-alignment-design.md`.

**Authoritative reference:** `docling-graph-docs.md` at repo root. Every identity / component / edge decision in this plan traces to an R-rule in that file (see spec §11).

**Blocks:** `docs/superpowers/plans/2026-04-14-docling-graph-schema-compliance.md` (that plan's Phase 7 Task 53 resumes only after Chunk G of this plan passes its acceptance gate).

---

## Conventions for every task in this plan

- **TDD:** every task writes a failing test first, then the implementation.
- **Commits:** every task ends with one commit. Commit message footer on every commit: `Plan: docs-alignment` (so `git log --grep` can scope to this plan).
- **Pre-commit hooks:** do NOT bypass with `--no-verify`. If a hook fails, fix the underlying issue.
- **File paths:** always absolute-from-repo-root (e.g. `app/services/extraction_merge.py`), never relative.
- **Test discovery:** `.venv/bin/python -m pytest <path> -q` is the standard run command.
- **Docs citations:** R-rules (R1…R22) reference `docling-graph-docs.md`; see spec §11 for the rule index.

---

## Chunk A0: Walker + schema prereqs (6 tasks)

These changes MUST land before Chunk B's canonical rewrites — Chunk B demotes 12 entities to `is_entity=False`, and without A0 the walker would silently drop them.

### Task A0-1: Content-based identity for components in `_build_logical_identity`

**Files:**
- Modify: `app/services/extraction_merge.py` — `_build_logical_identity` at `:416`
- Test: `tests/unit/test_build_logical_identity_components.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_build_logical_identity_components.py
from pydantic import BaseModel, ConfigDict
from app.services.extraction_merge import _build_logical_identity

class SampleComponent(BaseModel):
    model_config = ConfigDict(is_entity=False, ontology_name="SAMPLE_COMP")
    alpha: str | None = None
    beta: int | None = None
    gamma: list[str] = []

def test_component_identity_is_all_fields_canonical():
    """Per docs:17235 — components dedup by ENTIRE content (not non-None subset)."""
    inst_a = SampleComponent(alpha="x", beta=None, gamma=["u", "v"])
    inst_b = SampleComponent(alpha="x", beta=None, gamma=["u", "v"])
    inst_c = SampleComponent(alpha="x", beta=1, gamma=["u", "v"])
    id_a = _build_logical_identity("SAMPLE_COMP", inst_a, {}, "doc-1")
    id_b = _build_logical_identity("SAMPLE_COMP", inst_b, {}, "doc-1")
    id_c = _build_logical_identity("SAMPLE_COMP", inst_c, {}, "doc-1")
    assert id_a == id_b, "identical content → identical identity"
    assert id_a != id_c, "different content (beta) → different identity"

def test_component_identity_includes_none_values():
    """None value is part of the canonical form."""
    inst_null = SampleComponent(alpha="x", beta=None, gamma=[])
    inst_empty = SampleComponent(alpha="x", beta=None, gamma=[])
    # Even all-None except alpha should produce stable identity
    id_null = _build_logical_identity("SAMPLE_COMP", inst_null, {}, "doc-1")
    id_empty = _build_logical_identity("SAMPLE_COMP", inst_empty, {}, "doc-1")
    assert id_null == id_empty
```

- [ ] **Step 2: Run test to verify it fails**
  `.venv/bin/python -m pytest tests/unit/test_build_logical_identity_components.py -v` → FAIL (component branch not implemented)

- [ ] **Step 3: Implement component branch in `_build_logical_identity`**

Locate the function at `app/services/extraction_merge.py:416`. Add an early branch:

```python
def _build_logical_identity(entity_type, node, ontology, document_id):
    cfg = getattr(node, "model_config", {}) or {}
    if cfg.get("is_entity") is False:
        # Content-based identity — all fields in declaration order.
        # Per docs:17235: "All fields are used for deduplication."
        values = []
        for fname in node.__class__.model_fields:
            raw = getattr(node, fname, None)
            # Canonicalize: tuple for lists (hashable); leave None/scalar as-is
            if isinstance(raw, list):
                values.append(tuple(raw))
            else:
                values.append(raw)
        scope = cfg.get("identity_scope", "document")
        return LogicalIdentity(
            ontology_name=entity_type,
            identity_values=tuple(values),
            identity_scope=scope,
            document_id=document_id if scope == "document" else None,
        )
    # ... existing entity branch unchanged
```

Also update `NodeRecord` construction for components (wherever records are built from `MergedEntityRecord`): populate `name` from a content fingerprint so `_upsert_node_impl_sync` at `arcadedb_graph.py:1722` keeps writing uniformly. See `_to_merged_entity_record` spec §8.2 — component case sets `display_label` from the content digest.

- [ ] **Step 4: Run tests to verify they pass + full suite**
  `.venv/bin/python -m pytest tests/unit/test_build_logical_identity_components.py -v` → PASS
  `.venv/bin/python -m pytest tests/unit/ -q` → no new failures

- [ ] **Step 5: Commit**

```bash
git add app/services/extraction_merge.py tests/unit/test_build_logical_identity_components.py
git commit -m "$(cat <<'EOF'
feat(merge): content-based identity for is_entity=False components (A0-1)

_build_logical_identity gains an is_entity=False branch producing a
LogicalIdentity whose identity_values is the canonical tuple of all
fields in declaration order (None preserved; lists → tuples).

Aligns with docs:17235 "All fields are used for deduplication".
Prereq for Chunk B component demotions.

Plan: docs-alignment
EOF
)"
```

### Task A0-2: Walker — allow component emission via edge_label

**Files:**
- Modify: `app/services/extraction_merge.py` — `walk_entity_graph` at `:564` + `:593-601`
- Test: `tests/unit/test_walk_entity_graph_components.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_walk_entity_graph_components.py
from typing import List
from pydantic import BaseModel, ConfigDict, Field
from app.services.extraction_merge import walk_entity_graph, edge

class TinyComp(BaseModel):
    model_config = ConfigDict(is_entity=False, ontology_name="TINY_COMP")
    value: str

class TinyEntity(BaseModel):
    model_config = ConfigDict(is_entity=True, ontology_name="TINY", graph_id_fields=["name"])
    name: str
    comps: List[TinyComp] = edge(label="HAS_COMP", default_factory=list)

def test_walker_emits_component_via_edge_label():
    """Per docs:17500-17509 — components reachable via edge() must become graph nodes."""
    entity = TinyEntity(name="e1", comps=[TinyComp(value="x"), TinyComp(value="y")])
    entities = []
    edges = []
    walk_entity_graph(
        entity,
        on_entity=lambda n: entities.append(n),
        on_edge=lambda parent_id, label, child: edges.append((label, type(child).__name__)),
        ontology={}, document_id="doc-1",
    )
    comp_names = [type(e).__name__ for e in entities]
    assert "TinyEntity" in comp_names
    assert comp_names.count("TinyComp") == 2  # both components emitted
    assert ("HAS_COMP", "TinyComp") in edges
```

- [ ] **Step 2: Run test to verify it fails**
  Expected: `TinyComp` missing from `comp_names` (walker currently skips components at :564/:593).

- [ ] **Step 3: Update the walker**

At `extraction_merge.py:564`, remove the unconditional `return` for components that were reached via `edge_label` (top-level walk retains the skip for components at pass root). At `:593-601`, change the "contract violation" branch to an emit path:

```python
if child_cfg.get("is_entity") is not True:
    # Per §4.8: components reached via edge() become graph nodes
    # with content-based identity. Do NOT recurse (components can't
    # have edge_label fields — enforced by test_components_have_no_edge_label_fields).
    on_entity(child)
    if full_mode and on_edge is not None and parent_identity is not None:
        on_edge(parent_identity, edge_label, child)
    continue
# existing is_entity=True path continues here
```

Remove the `logger.warning` for "contract violation" — this is now the normal path for components.

- [ ] **Step 4: Run tests to verify — watch for regressions in existing walker tests**
  `.venv/bin/python -m pytest tests/unit/test_walk_entity_graph.py tests/unit/test_walk_entity_graph_components.py -v` → PASS
  `.venv/bin/python -m pytest tests/unit/ -q` → no new failures

- [ ] **Step 5: Commit**

```bash
git add app/services/extraction_merge.py tests/unit/test_walk_entity_graph_components.py
git commit -m "$(cat <<'EOF'
feat(merge): walker emits is_entity=False components via edge_label (A0-2)

walk_entity_graph at :564 + :593-601 now emits components reached
through edge(label=...) fields as graph nodes. Components do NOT
recurse (enforced by the new contract test in A0-5).

Aligns with docs:17500-17509 "Same address node is shared across
multiple people/organizations".

Plan: docs-alignment
EOF
)"
```

### Task A0-3: Add 4 structural edge types

**Files:**
- Modify: `app/services/arcadedb_schema.py:79` — `_STRUCTURAL_EDGE_TYPES`
- Test: `tests/unit/test_arcadedb_schema_structural_edges.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_arcadedb_schema_structural_edges.py
from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES

def test_anchor_structural_edges_declared():
    """HAS_SECTION/HAS_FIGURE/HAS_TABLE/CHILD_OF must be declared."""
    for label in ("HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF"):
        assert label in _STRUCTURAL_EDGE_TYPES, f"{label} missing from _STRUCTURAL_EDGE_TYPES"
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Add the 4 labels**

```python
# arcadedb_schema.py:79
_STRUCTURAL_EDGE_TYPES = [
    "CONTAINS_TEXT",
    "CONTAINS_IMAGE",
    "SAME_PAGE",
    "SAME_SECTION",
    "SAME_ARTIFACT",
    "NEXT_CHUNK",
    "HAS_PROVENANCE",
    "EXTRACTED_FROM",
    "HAS_ALIAS",
    # Document-anchor edges (per spec §3.5a).
    # Intentionally OUTSIDE ontology validation (RelationshipType/VALIDATION_MATRIX).
    "HAS_SECTION",
    "HAS_FIGURE",
    "HAS_TABLE",
    "CHILD_OF",
]
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add app/services/arcadedb_schema.py tests/unit/test_arcadedb_schema_structural_edges.py
git commit -m "$(cat <<'EOF'
feat(schema): HAS_SECTION/HAS_FIGURE/HAS_TABLE/CHILD_OF structural edges (A0-3)

Added to _STRUCTURAL_EDGE_TYPES. Intentionally not in RelationshipType
or VALIDATION_MATRIX — these are document-layout edges, written via
create_structural_edge_sync (not upsert_relationships_batch_sync).

Plan: docs-alignment
EOF
)"
```

### Task A0-4: Relax contract test 9e (edge targets can be components)

**Files:**
- Modify: `tests/unit/test_docs_compliance_contracts.py` — rename + update `test_edge_label_targets_are_is_entity_true`

- [ ] **Step 1: Locate the existing test**
  `grep -n "test_edge_label_targets_are_is_entity_true" tests/unit/test_docs_compliance_contracts.py` — should be around line 466.

- [ ] **Step 2: Rename + update assertion**

```python
def test_edge_label_targets_are_is_entity_true_or_is_component(all_models):
    """Per §4.8: edge() fields may target is_entity=True entities OR is_entity=False components.
    Pure components (value objects) became valid edge targets in A0-2."""
    violations = []
    for cls in all_models:
        for fname, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra or {}
            if not (isinstance(extra, dict) and extra.get("edge_label")):
                continue
            target = _resolve_edge_target(finfo.annotation)
            if target is None:
                continue
            target_cfg = getattr(target, "model_config", {}) or {}
            is_entity_flag = target_cfg.get("is_entity")
            if is_entity_flag is None:
                violations.append(f"{cls.__name__}.{fname} → {target.__name__}: no is_entity declared")
            # Both True (entity) and False (component) are now allowed.
    assert not violations, "\n".join(violations)
```

- [ ] **Step 3: Run tests** — the rename means old references break; grep and fix anywhere the old name is cited.
  `grep -rn "test_edge_label_targets_are_is_entity_true" tests/ docs/` — update each hit.

- [ ] **Step 4: Run full contract-test module** — `.venv/bin/python -m pytest tests/unit/test_docs_compliance_contracts.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_docs_compliance_contracts.py
git commit -m "feat(tests): relax contract 9e — components are valid edge() targets (A0-4)

Plan: docs-alignment"
```

### Task A0-5: Add `test_components_have_no_edge_label_fields` contract

**Files:**
- Modify: `tests/unit/test_docs_compliance_contracts.py` (add new test)

- [ ] **Step 1: Add the test**

```python
def test_components_have_no_edge_label_fields(all_models):
    """Per §4.8 + docs:17517 — components (is_entity=False) cannot carry
    edge(label=...) fields. Keeps the 'no-recurse' walker policy safe;
    if component→component edges are ever needed, promote to entity."""
    violations = []
    for cls in all_models:
        cfg = getattr(cls, "model_config", {}) or {}
        if cfg.get("is_entity") is not False:
            continue  # entities can have edges
        for fname, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("edge_label"):
                violations.append(f"{cls.__name__}.{fname} has edge_label (component→edge not allowed)")
    assert not violations, "\n".join(violations)
```

- [ ] **Step 2: Run test** — expected PASS since no current class violates (Chunk B will keep it passing through demotions).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_docs_compliance_contracts.py
git commit -m "test(contracts): components cannot carry edge() fields (A0-5)

Plan: docs-alignment"
```

### Task A0-6: New `test_arcadedb_schema_introspection.py`

**Files:**
- Create: `tests/unit/test_arcadedb_schema_introspection.py` (replacement for the snapshot-driven `test_arcadedb_schema.py` that gets deleted in Chunk F)

- [ ] **Step 1: Write the test**

```python
"""Schema-creation coverage driven by Pydantic introspection.

Replaces the YAML-snapshot-driven test_arcadedb_schema.py.
Oracle: ontology_bundles.air_defense_v3.introspect.build_entity_types_list
+ introspect.build_relationship_types_list.
"""
from unittest.mock import MagicMock, patch
from ontology_bundles.air_defense_v3.introspect import (
    build_entity_types_list,
    build_relationship_types_list,
)
from app.services.arcadedb_schema import ensure_schema

def test_entity_vertex_classes_cover_ALL_ENTITIES():
    entity_types = build_entity_types_list()
    names = {e["name"] for e in entity_types}
    # Every canonical entity and component class produces a vertex class entry.
    # (Components remain in ALL_ENTITIES even when is_entity=False.)
    assert "PLATFORM" in names
    assert "RADAR_SYSTEM" in names
    assert "MODULATION" in names  # demoted-to-component in Chunk B, still present

def test_structural_edges_include_anchor_labels():
    from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES
    for label in ("HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF"):
        assert label in _STRUCTURAL_EDGE_TYPES

def test_ensure_schema_runs_without_yaml_dependency(tmp_path):
    """ensure_schema drives ArcadeDB DDL purely from introspection — no YAML file read."""
    # Mock the ArcadeDB client; assert ensure_schema completes and emits DDL for
    # each entity_type and structural_edge declared by introspection.
    client = MagicMock()
    import asyncio
    asyncio.run(ensure_schema(client, database="test_db"))
    # Verify command was called (DDL emitted); exact count is brittle,
    # just assert it ran.
    assert client.command.called or client.command_sync.called
```

- [ ] **Step 2: Run test** — expected PASS after A0-3 (structural edges) landed.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_arcadedb_schema_introspection.py
git commit -m "test(schema): introspection-based schema coverage (A0-6)

Replaces snapshot-driven test_arcadedb_schema.py deleted in Chunk F.
Oracle is Pydantic introspection, not a YAML fixture.

Plan: docs-alignment"
```

---

## Chunk A: Contract tests + helper extensions (6 tasks)

### Task A-1: Extend `edge()` helper to accept `description` + `examples`

**Files:**
- Modify: `ontology_bundles/air_defense_v3/entities.py` — `edge()` at ~line 32
- Modify: `ontology_bundles/air_defense_v3/extraction_schemas/*.py` — each file's local `edge()` copy
- Test: `tests/unit/test_edge_helper_description_examples.py` (new)

- [ ] **Step 1: Failing test**

```python
def test_edge_helper_accepts_description_and_examples():
    from ontology_bundles.air_defense_v3.entities import edge
    from pydantic import BaseModel
    class Parent(BaseModel):
        children: list = edge(label="HAS_CHILD", description="Children of this parent", examples=[[]], default_factory=list)
    info = Parent.model_fields["children"]
    assert info.description == "Children of this parent"
    assert info.examples == [[]]
    extra = info.json_schema_extra or {}
    assert extra.get("edge_label") == "HAS_CHILD"
```

- [ ] **Step 2: Run → FAIL** (current `edge()` doesn't forward description/examples to Field).

- [ ] **Step 3: Update `edge()`**

```python
def edge(label: str, *, description: str | None = None, examples: list | None = None, **field_kwargs):
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    if description is not None:
        field_kwargs["description"] = description
    if examples is not None:
        field_kwargs["examples"] = examples
    return Field(json_schema_extra=existing_extra, **field_kwargs)
```

Mirror the change in each `extraction_schemas/*.py` that has a local `edge()` copy (radar_domain, missile_domain, other_systems, system_links, reference before it's deleted).

- [ ] **Step 4: Run test → PASS** + full suite clean.

- [ ] **Step 5: Commit**

```bash
git add ontology_bundles/air_defense_v3/entities.py ontology_bundles/air_defense_v3/extraction_schemas/*.py tests/unit/test_edge_helper_description_examples.py
git commit -m "feat(ontology): edge() accepts description+examples kwargs (A-1)

Forwards to Field(); json_schema_extra.edge_label preserved. Enables
Chunk B's xfail-resolution for test_descriptions_and_examples_on_
extraction_relevant_fields.

Plan: docs-alignment"
```

### Task A-2: Lenient coercer logging

**Files:**
- Modify: `ontology_bundles/air_defense_v3/validators.py` — `coerce_optional_int/float/confidence`
- Test: `tests/unit/test_validators_lenient_logging.py` (new)

- [ ] **Step 1: Failing test**

```python
import logging
from ontology_bundles.air_defense_v3.validators import coerce_optional_int

def test_coerce_optional_int_logs_on_unrecoverable(caplog):
    with caplog.at_level(logging.WARNING, logger="ontology_bundles.air_defense_v3.validators"):
        result = coerce_optional_int("not a number")
    assert result is None
    assert any("unrecoverable" in rec.message.lower() for rec in caplog.records)

def test_coerce_optional_int_no_log_on_none(caplog):
    with caplog.at_level(logging.WARNING, logger="ontology_bundles.air_defense_v3.validators"):
        assert coerce_optional_int(None) is None
    assert not caplog.records
```

- [ ] **Step 2: Run → FAIL** (current coercer is silent).

- [ ] **Step 3: Add logger + warnings**

```python
# validators.py
import logging
logger = logging.getLogger(__name__)

def coerce_optional_int(value):
    if value is None: return None
    # ... existing attempts
    if _still_unrecoverable:
        logger.warning("coerce_optional_int: unrecoverable input %r → None", value)
        return None
```

Same pattern for `coerce_optional_float` and `coerce_optional_confidence`.

- [ ] **Step 4: Run → PASS**.

- [ ] **Step 5: Commit**

```bash
git add ontology_bundles/air_defense_v3/validators.py tests/unit/test_validators_lenient_logging.py
git commit -m "feat(validators): lenient coercers log on unrecoverable input (A-2)

Plan: docs-alignment"
```

### Task A-3: Pipeline config quality-gate knob

**Files:**
- Modify: `docker/docling-graph/app/config_builder.py:108`
- Modify: `docker-compose.yml` — add `DOCLING_GRAPH_QUALITY_MIN_INSTANCES` env
- Modify: `app/config.py` — add `docling_graph_quality_min_instances` setting default (on worker side, used when building per-pass config)
- Test: `tests/unit/test_docling_graph_quality_config.py` (new)

- [ ] **Step 1: Failing test** — assert `config_builder.build_pipeline_config(...)` returns a dict where `delta_quality_min_instances` reflects the env var.

- [ ] **Step 2: Run → FAIL** if unset.

- [ ] **Step 3: Set default 3 in compose + Settings; override to 1 per-pass for system_links in whatever per-pass config injection the worker uses** (search `grep -n "pass_name.*system_links" app/workers/pipeline.py` for the injection point).

- [ ] **Step 4: Run → PASS**.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(config): DOCLING_GRAPH_QUALITY_MIN_INSTANCES env with per-pass override (A-3)

Default 3 for domain passes (radar/missile/other/system_links=1).
Pinned by settings.docling_graph_quality_min_instances (spec §4.6).

Plan: docs-alignment"
```

### Task A-4: Contract tests (batch 1 — identity rules)

**Files:**
- Create: `tests/unit/contracts/__init__.py`
- Create: `tests/unit/contracts/test_identity_contract.py`

Add 7 tests (all initially `@pytest.mark.xfail(strict=True)` since Chunk B hasn't run yet):

- [ ] **Step 1: Write all 7 tests with xfail markers**

```python
import pytest
from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_entity_has_identity_or_is_component():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        is_entity = cfg.get("is_entity", True)
        id_fields = cfg.get("graph_id_fields") or []
        if is_entity:
            assert id_fields, f"{name}: is_entity=True but graph_id_fields empty"
        else:
            assert not id_fields, f"{name}: is_entity=False but has graph_id_fields"

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_fields_are_required():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            assert field is not None and field.is_required(), \
                f"{name}.{fname}: identity field must be required"

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_field_examples_are_short():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            if field and field.examples:
                for ex in field.examples:
                    assert isinstance(ex, (str, int, float, bool))
                    s = str(ex)
                    assert len(s) <= 80, f"{name}.{fname} example '{s[:40]}...' >80 chars"
                    assert "\n" not in s, f"{name}.{fname} example contains newline"

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_fields_not_named_heading_or_title():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        banned = {"heading", "title", "caption", "description"}
        for fname in cfg.get("graph_id_fields") or []:
            assert fname not in banned, f"{name}: identity field '{fname}' is a banned name"

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_examples_are_distinct():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            if field and field.examples:
                assert len(field.examples) == len(set(map(repr, field.examples))), \
                    f"{name}.{fname}: examples contain duplicates"

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_identity_example_values_populated_for_library_filter():
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        if cfg.get("is_entity") is not True:
            continue
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            n_examples = len(field.examples) if field and field.examples else 0
            assert n_examples >= 2, \
                f"{name}.{fname}: needs ≥2 examples for library identity_example_values"

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_non_identity_fields_are_optional():
    """R19: every field NOT in graph_id_fields must be Optional[T] with default=None."""
    from pydantic_core import PydanticUndefined
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        id_fields = set(cfg.get("graph_id_fields") or [])
        for fname, finfo in cls.model_fields.items():
            if fname in id_fields:
                continue
            extra = finfo.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("edge_label"):
                continue  # edges handled separately
            assert finfo.default is not PydanticUndefined or not finfo.is_required(), \
                f"{name}.{fname}: non-identity field is required (should be Optional[T]=None)"
```

- [ ] **Step 2: Run** — expected 7 XFAIL (strict, currently failing).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/
git commit -m "test(contracts): identity contract tests (xfailed pending Chunk B) (A-4)

Plan: docs-alignment"
```

### Task A-5: Contract tests (batch 2 — LLM-style + component)

**Files:**
- Create: `tests/unit/contracts/test_extraction_schema_contract.py`
- Create: `tests/unit/contracts/test_component_contract.py`

- [ ] **Step 1: Write 5 tests (xfailed)**

```python
# tests/unit/contracts/test_extraction_schema_contract.py
import re
import pytest
from pathlib import Path
import importlib
import pkgutil

def _extraction_view_entities():
    """Collect all BaseModel subclasses from extraction_schemas/*.py"""
    import ontology_bundles.air_defense_v3.extraction_schemas as ext
    results = []
    for modinfo in pkgutil.iter_modules(ext.__path__):
        mod = importlib.import_module(f"{ext.__name__}.{modinfo.name}")
        for name in dir(mod):
            obj = getattr(mod, name)
            try:
                from pydantic import BaseModel
                if isinstance(obj, type) and issubclass(obj, BaseModel):
                    results.append(obj)
            except Exception:
                pass
    return results

_HEADING_STYLE_RE = re.compile(
    r"^(\d+(\.\d+)+|\d+|[IVX]+|[A-Z]|Chapter\s|Section\s|Part\s)$",
    re.IGNORECASE,
)

@pytest.mark.xfail(strict=True, reason="Chunk B/C will fix")
def test_llm_emitted_identity_examples_not_heading_style():
    """R17, docs:18470 — extraction-view identity examples must not look like raw headings."""
    violations = []
    for cls in _extraction_view_entities():
        cfg = getattr(cls, "model_config", {}) or {}
        if cfg.get("is_entity") is not True:
            continue
        for fname in cfg.get("graph_id_fields") or []:
            field = cls.model_fields.get(fname)
            for ex in (field.examples or []) if field else []:
                s = str(ex).strip()
                if _HEADING_STYLE_RE.match(s):
                    violations.append(f"{cls.__name__}.{fname}: example '{s}' is heading-style")
    assert not violations, "\n".join(violations)

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_edge_fields_have_edge_label():
    """R4 — every List[Entity]/Optional[Entity] field on is_entity=True must carry edge_label."""
    from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES
    from pydantic import BaseModel
    from typing import get_args
    violations = []
    for name, cls in ALL_ENTITIES.items():
        cfg = getattr(cls, "model_config", {}) or {}
        if cfg.get("is_entity") is not True:
            continue
        for fname, finfo in cls.model_fields.items():
            # Check if annotation is a BaseModel or List[BaseModel]
            ann = finfo.annotation
            inner_cls = None
            for a in get_args(ann):
                if isinstance(a, type) and issubclass(a, BaseModel):
                    inner_cls = a; break
                for b in get_args(a):
                    if isinstance(b, type) and issubclass(b, BaseModel):
                        inner_cls = b; break
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                inner_cls = ann
            if inner_cls is None:
                continue
            extra = finfo.json_schema_extra or {}
            if not (isinstance(extra, dict) and extra.get("edge_label")):
                violations.append(f"{name}.{fname}: BaseModel field missing edge_label")
    assert not violations, "\n".join(violations)

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_no_nested_property_dicts():
    """R11 — non-edge property fields must be primitive or list[primitive]."""
    from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES
    from pydantic import BaseModel
    from typing import get_args
    violations = []
    primitives = (str, int, float, bool)
    for name, cls in ALL_ENTITIES.items():
        for fname, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("edge_label"):
                continue
            ann = finfo.annotation
            # Walk through Optional, List, etc., look for dict or BaseModel
            def _has_nested(a):
                if isinstance(a, type):
                    return issubclass(a, (dict,)) or (issubclass(a, BaseModel))
                return any(_has_nested(x) for x in get_args(a))
            if _has_nested(ann):
                violations.append(f"{name}.{fname}: non-edge property has nested dict/BaseModel")
    assert not violations, "\n".join(violations)
```

```python
# tests/unit/contracts/test_component_contract.py
import pytest
from ontology_bundles.air_defense_v3.entities import ALL_ENTITIES

@pytest.mark.xfail(strict=True, reason="Chunk B will fix")
def test_component_fields_attached_via_edge_helper():
    """Components appearing in ALL_ENTITIES must be reachable via edge() from at least one entity."""
    from pydantic import BaseModel
    from typing import get_args
    components = {name: cls for name, cls in ALL_ENTITIES.items()
                  if (getattr(cls, "model_config", {}) or {}).get("is_entity") is False}
    if not components:
        pytest.skip("No components yet (pre-Chunk-B)")
    reachable = set()
    for name, cls in ALL_ENTITIES.items():
        for fname, finfo in cls.model_fields.items():
            extra = finfo.json_schema_extra or {}
            if not (isinstance(extra, dict) and extra.get("edge_label")):
                continue
            def _find(a):
                if isinstance(a, type) and issubclass(a, BaseModel):
                    return getattr(a, "__name__", None)
                for x in get_args(a):
                    r = _find(x)
                    if r: return r
                return None
            target_name = _find(finfo.annotation)
            if target_name in {c.__name__ for c in components.values()}:
                reachable.add(target_name)
    orphans = [c.__name__ for c in components.values() if c.__name__ not in reachable]
    assert not orphans, f"Components not reachable via edge(): {orphans}"
```

- [ ] **Step 2: Run → xfail expected.**

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/
git commit -m "test(contracts): LLM-style + component contract tests (xfailed) (A-5)

Plan: docs-alignment"
```

### Task A-6: Extend `MergedEntityRecord` + NodeRecord for component stamping (if needed)

**Files:**
- Modify: `app/services/extraction_merge.py` — add `is_component: bool = False` flag on `MergedEntityRecord` if downstream needs to branch
- Test: `tests/unit/test_merged_entity_record_shape.py` (new)

- [ ] **Step 1: Audit** — grep call sites that read `MergedEntityRecord.properties` and decide whether a flag is needed. If downstream can distinguish by `model_config["is_entity"]` via identity, skip this task (no-op).

- [ ] **Step 2: If needed**, add the flag with a minimal test and commit. Else commit a note-only change (empty task OK).

- [ ] **Step 3: Commit if applicable**

```bash
git commit -m "chore(merge): MergedEntityRecord component-flag (if needed) (A-6)

Plan: docs-alignment"
```

---

## Chunk B: Canonical `entities.py` rewrite (32 tasks)

**Important for every task in B:** after each per-entity commit, run `.venv/bin/python -m pytest tests/unit/ -q` and ensure no NEW failures beyond the already-xfailed contract tests.

### Tasks B-1, B-2: Drop ASSERTION + SPREADSHEET (see spec §2.6)

- [ ] **B-1** Delete `AssertionEntity` from `entities.py` + remove from `ALL_ENTITIES` + delete from `extraction_schemas/reference.py` (note: reference.py is fully deleted in C-1 but for orderly B-sequence, just remove the entity class here). Remove ASSERTION from any hardcoded frontend/backend entity-type list you find via `grep -rn "ASSERTION"` within `app/` (leave deleted-file references alone). Commit: `refactor(entities): drop AssertionEntity (B-1)`

- [ ] **B-2** Delete `SpreadsheetEntity` from `entities.py` + add `"SPREADSHEET"` to `DocumentEntity.source_type` enum + move `workbook_name`/`sheet_name` to DocumentEntity as optional properties. Commit: `refactor(entities): merge SPREADSHEET into DOCUMENT.source_type (B-2)`

### Tasks B-3 through B-17: Give-identity batch

Each task edits only one entity's `model_config.graph_id_fields`, updates the field to required, adds R16-compliant examples, then commits. Template:

```python
# Example for B-3 DocumentEntity
class DocumentEntity(BaseModel):
    model_config = ConfigDict(
        ontology_name="DOCUMENT",
        graph_id_fields=["document_number"],  # was []
        identity_scope="global",
        is_entity=True,
    )
    document_number: str = Field(  # was Optional[str] = None
        ...,
        description="Official document designator",
        examples=["TM 9-1425-386-12", "MIL-STD-1553B", "MIL-DTL-31000G"],
    )
    # ... all other fields become Optional[T] = None per R19 ...
```

Run applicable contract tests after each: once all B-3..B-17 land, `test_entity_has_identity_or_is_component` + `test_identity_fields_are_required` + `test_identity_field_examples_are_short` should gradually un-xfail. **Do NOT drop xfail markers until Chunk F — let them accumulate.**

- [ ] **B-3** DocumentEntity: `graph_id_fields=["document_number"]`. Also: `document_id` field (internal UUID) STAYS as non-identity property (preserves structural-vs-ontology distinction from spec §2.2 thesis 2). Commit: `refactor(entities): DOCUMENT identity=document_number (B-3)`
- [ ] **B-4** SectionEntity: `graph_id_fields=["section_number"]` + add `heading`, `section_path`, `document_id` as Optional[str]=None non-identity properties (spec §2.2). Commit: `refactor(entities): SECTION identity=section_number (B-4)`
- [ ] **B-5** FigureEntity: `graph_id_fields=["figure_ref"]` + `figure_label`, `document_id` Optional. Commit: `refactor(entities): FIGURE identity=figure_ref (B-5)`
- [ ] **B-6** TableEntity: `graph_id_fields=["table_ref"]` + `table_label`, `document_id` Optional. Commit: `refactor(entities): TABLE identity=table_ref (B-6)`
- [ ] **B-7** OrganizationEntity: `graph_id_fields=["name"]`. Commit: `refactor(entities): ORGANIZATION identity=name (B-7)`
- [ ] **B-8** StandardEntity: `graph_id_fields=["designation"]`. Commit: `refactor(entities): STANDARD identity=designation (B-8)`
- [ ] **B-9** EquipmentSystemEntity: `graph_id_fields=["name"]`. Commit: `refactor(entities): EQUIPMENT_SYSTEM identity=name (B-9)`
- [ ] **B-10** ComponentEntity: `graph_id_fields=["part_number"]`. Commit: `refactor(entities): COMPONENT identity=part_number (B-10)`
- [ ] **B-11** AssemblyEntity: `graph_id_fields=["assembly_number"]`. Commit: `refactor(entities): ASSEMBLY identity=assembly_number (B-11)`
- [ ] **B-12** CapabilityEntity: `graph_id_fields=["capability_name"]`. Commit: `refactor(entities): CAPABILITY identity=capability_name (B-12)`
- [ ] **B-13** ProcedureEntity: `graph_id_fields=["name"]`, scope=document. Commit.
- [ ] **B-14** FailureModeEntity: `graph_id_fields=["name"]`, scope=document. Commit.
- [ ] **B-15** TestEventEntity: `graph_id_fields=["name"]`, scope=global. Commit.
- [ ] **B-16** ForceStructureEntity: `graph_id_fields=["name"]`, scope=global. Commit.
- [ ] **B-17** SubsystemEntity: `graph_id_fields=["name"]`, scope=document. Commit.

### Tasks B-18 through B-29: Demote batch

Each flip: set `is_entity=False`, remove `graph_id_fields` (empty tuple), flip any previously-required non-identity fields to `Optional[T] = None`. Per spec §4.8 A0 is now complete, walker handles these.

- [ ] **B-18** SpecificationEntity → component. Also flip `parameter: str` → `Optional[str] = None`, same for `value: str`. Commit: `refactor(entities): SPECIFICATION → is_entity=False component (B-18)`
- [ ] **B-19** ModulationEntity → component. Commit.
- [ ] **B-20** RfSignatureEntity → component. Commit.
- [ ] **B-21** RfEmissionEntity → component. Commit.
- [ ] **B-22** ScanPatternEntity → component. Commit.
- [ ] **B-23** IfAmplifierEntity → component. Commit.
- [ ] **B-24** MissilePerformanceEntity → component. Commit.
- [ ] **B-25** MissilePhysicalCharacteristicsEntity → component. Commit.
- [ ] **B-26** PropulsionStackEntity → component. Commit.
- [ ] **B-27** PropulsionStageEntity → component. Commit.
- [ ] **B-28** RadarPerformanceEntity → component. Commit.
- [ ] **B-29** EngagementTimelineEntity → component. Commit.

### Tasks B-30, B-31, B-32: Batched touch-ups

- [ ] **B-30** All 17 unchanged entities: apply R16/R17 example-list cleanup (no duplicates, distinct values, heading-style examples removed from LLM-emitted identities). One commit across the batch. See spec §4.3 and §2.4 for which entities. Commit: `refactor(entities): R16/R17 example cleanup for 17 unchanged entities (B-30)`
- [ ] **B-31** All entities: audit and flip any remaining required non-identity fields to `Optional[T] = None`. Commit: `refactor(entities): flip remaining required non-identity fields to Optional (B-31)`
- [ ] **B-32** Update every `edge(label=...)` call site in `entities.py` to add `description` + `examples` per A-1's extended helper. The 10 new contract tests in A-4/A-5 should now all un-xfail. **DO NOT drop xfail markers here — that's Chunk F.** Commit: `refactor(entities): edge() calls gain description+examples (B-32)`

---

## Chunk C: Extraction schemas rewrite + manifest change (5 tasks)

### Task C-1: Delete `extraction_schemas/reference.py`

- [ ] Delete `ontology_bundles/air_defense_v3/extraction_schemas/reference.py`
- [ ] Remove `from .reference import ...` from `extraction_schemas/__init__.py` if present
- [ ] Run `.venv/bin/python -m pytest tests/unit/ -q` — expect new failures in `test_coverage_checker.py`, `test_extraction_schemas.py`, `test_ontology_bundles.py` (fixed in Chunk E)
- [ ] Commit: `refactor(schemas): delete extraction_schemas/reference.py (C-1)`

### Task C-2: Update manifest.yaml

- [ ] Remove the `reference` pass block from `ontology_bundles/air_defense_v3/manifest.yaml`
- [ ] Verify the 4 remaining passes parse via `load_bundle_manifest`
- [ ] Commit: `refactor(bundle): delete reference pass from manifest (C-2)`

### Task C-3: Rewrite `extraction_schemas/radar_domain.py`

- [ ] Match new canonical identities (Chunk B) — `RadarSystemEntity.graph_id_fields=["system_name"]` already, just verify.
- [ ] Import shared `_deduplicate_by_identity` helper from `validators.py` (task depends on moving from per-file copies) — if not already consolidated, consolidate here.
- [ ] Remove local dedup code.
- [ ] Flip non-identity fields to Optional per R19.
- [ ] Demoted components (MODULATION, RF_SIGNATURE, etc.) carry `is_entity=False`, no `graph_id_fields`.
- [ ] Identity examples cleaned per R17 (no heading-style).
- [ ] Commit: `refactor(schemas): radar_domain identity + R17 examples + Optional (C-3)`

### Task C-4: Rewrite `missile_domain.py` and `other_systems.py`

- [ ] Same treatment as C-3.
- [ ] Commit per-file: `refactor(schemas): missile_domain alignment (C-4a)` and `refactor(schemas): other_systems alignment (C-4b)` (treat as 2 sub-tasks).

### Task C-5: Minimal touch to `system_links.py`

- [ ] Decision-4 DTO pattern preserved.
- [ ] Examples cleaned per R17.
- [ ] Commit: `refactor(schemas): system_links R17 cleanup (C-5)`

---

## Chunk D: Docling anchor walker + new worker task (6 tasks)

### Task D-1: `_to_merged_entity_record` helper (spec §8.2)

**Files:**
- Modify: `app/services/extraction_merge.py` — add helper
- Test: `tests/unit/test_to_merged_entity_record.py`

- [ ] **Step 1: Failing test**

```python
from ontology_bundles.air_defense_v3.entities import SectionEntity, DocumentEntity
from app.services.extraction_merge import _to_merged_entity_record

def test_section_merged_record_includes_document_id_and_section_path():
    sec = SectionEntity(section_number="1.1", heading="Foo", section_path="Chapter 1 > Foo")
    rec = _to_merged_entity_record(sec, ontology={}, document_id="doc-uuid-1")
    assert rec.identity.ontology_name == "SECTION"
    assert rec.identity.identity_values == ("1.1",)
    assert rec.properties["document_id"] == "doc-uuid-1"
    assert rec.properties["section_path"] == "Chapter 1 > Foo"
    assert "section_number" not in rec.properties  # identity ≠ property
    assert rec.pass_origins == {"document_anchors"}

def test_document_merged_record_identity_is_document_number():
    doc = DocumentEntity(document_number="TM 9-1425-386-12")
    rec = _to_merged_entity_record(doc, ontology={}, document_id="doc-uuid-1")
    assert rec.identity.identity_values == ("TM 9-1425-386-12",)
    assert rec.properties["document_id"] == "doc-uuid-1"

def test_section_sentinel_no_section_path():
    sec = SectionEntity(section_number="0", heading=None, section_path=None)
    rec = _to_merged_entity_record(sec, ontology={}, document_id="doc-uuid-1")
    assert rec.properties.get("section_path") is None
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
def _to_merged_entity_record(
    model: BaseModel,
    ontology: dict,
    document_id: str,
    pass_origin: str = "document_anchors",
) -> MergedEntityRecord:
    cfg = getattr(model, "model_config", {}) or {}
    ontology_name = cfg["ontology_name"]
    identity = _build_logical_identity(ontology_name, model, ontology, document_id)
    id_fields = set(cfg.get("graph_id_fields") or [])
    dumped = model.model_dump(mode="json")
    properties = {}
    for fname, value in dumped.items():
        if fname in id_fields: continue
        finfo = type(model).model_fields.get(fname)
        extra = finfo.json_schema_extra if finfo else None
        if isinstance(extra, dict) and extra.get("edge_label"):
            continue
        properties[fname] = value
    # Stamp document_id (if model has such a field, model_dump already included it;
    # if the model doesn't declare document_id, add it explicitly).
    properties.setdefault("document_id", document_id)
    return MergedEntityRecord(
        identity=identity,
        properties=properties,
        confidence=1.0,
        pass_origins={pass_origin},
        display_label=build_display_label(model, ontology_name),
    )
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add app/services/extraction_merge.py tests/unit/test_to_merged_entity_record.py
git commit -m "feat(merge): _to_merged_entity_record helper (D-1)

Converts Pydantic model → MergedEntityRecord with document_id stamped
as non-identity property for TextChunk joins (spec §3.4 + §8.2).

Plan: docs-alignment"
```

### Task D-2: Document-number extraction helper

**Files:**
- Create: `app/services/docling_anchors.py` — module housing the walker + helpers
- Test: `tests/unit/test_docling_anchor_walker.py` (partial — just doc-number heuristic)

- [ ] **Step 1: Failing test** — fixture DoclingDocument containing a title `"TM 9-1425-386-12 Operator Manual"` → helper returns `"TM 9-1425-386-12"`. Counter-test with a title `"Introduction"` → returns `None`.

- [ ] **Step 2: Implement regex**

```python
_DOC_NUMBER_RE = re.compile(
    r"\b(TM|MIL-STD|MIL-DTL|MIL-HDBK|MIL-PRF|ANSI/IEEE|ISO|DoD)\s*[-\s]?[A-Z0-9][\w.-]+",
    re.IGNORECASE,
)

def _extract_document_number_from_front_matter(docling_doc) -> str | None:
    """Scan first N titles/section headers for a MIL-STD/TM-style designator."""
    count = 0
    for item, level in docling_doc.iterate_items():
        label = getattr(item, "label", None)
        if label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            text = getattr(item, "text", None) or ""
            m = _DOC_NUMBER_RE.search(text)
            if m:
                return m.group(0).strip()
        count += 1
        if count > 30:  # bail after first 30 items (front matter only)
            break
    return None
```

- [ ] **Step 3: Run → PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(anchors): _extract_document_number_from_front_matter (D-2)

Regex-matches MIL-STD/TM/ISO/ANSI designators in first 30 items.
Returns None when no match — caller skips ontology DOCUMENT.

Plan: docs-alignment"
```

### Task D-3: Anchor walker implementation

**Files:**
- Modify: `app/services/docling_anchors.py` — add `walk()` function
- Test: `tests/unit/test_docling_anchor_walker.py` (main battery)

- [ ] **Step 1: Failing tests** — 6-8 fixtures exercising: empty-structure fallback, 3-level hierarchy, figure+table counts from pictures/tables arrays, CHILD_OF edges by path prefix, HAS_* skipped when document_number absent.

- [ ] **Step 2: Implement `walk()` per spec §3.3 pseudocode** — full function with section_stack, `_register_section`, `_caption_label`, conditional DocumentEntity, edge construction using `MergedEdgeRecord(from_identity=, to_identity=, rel_type=, confidence=1.0, pass_origins={"document_anchors"})`, return `MergedExtraction(entities=, edges=, rejected_edges=[], rejections_by_pass={}, pipeline_run_id=, document_id=)`.

- [ ] **Step 3: Run → PASS**

- [ ] **Step 4: Commit**

```bash
git add app/services/docling_anchors.py tests/unit/test_docling_anchor_walker.py
git commit -m "feat(anchors): Docling-derived SECTION/FIGURE/TABLE walker (D-3)

Deterministic structural anchor emission with section_stack mirroring
docker/docling/app/converter.py:296-318. Fallback SECTION(section_number='0')
when no headings. Ontology DOCUMENT conditional on document_number
extractability.

Plan: docs-alignment"
```

### Task D-4: New Celery task `derive_document_anchors`

**Files:**
- Modify: `app/workers/pipeline.py` — add `derive_document_anchors` task
- Modify: `app/workers/pipeline.py` — insert task into pipeline chain between `derive_image_embeddings` and `derive_ontology_graph`
- Modify: `app/workers/celery_app.py:50+` — register task route to `graph` queue
- Test: `tests/unit/test_derive_document_anchors.py`

- [ ] **Step 1: Failing test** — mock `_build_docling_document_json` + `graph_store`; verify the task creates a StageRun with `stage_name="derive_document_anchors"`, calls `upsert_nodes_batch_sync` once with the walker's records, then iterates `create_structural_edge_sync` for each edge.

- [ ] **Step 2: Implement**

```python
@celery_app.task(bind=True, max_retries=1, default_retry_delay=30, queue="graph",
                 soft_time_limit=settings.finalize_soft_time_limit,
                 time_limit=settings.finalize_time_limit)
def derive_document_anchors(self, document_id: str, run_id: str | None = None) -> dict:
    """Emit ontology DOCUMENT/SECTION/FIGURE/TABLE from DoclingDocument.
    Spec §3.1–§3.5."""
    from app.services.docling_anchors import walk
    from app.services.ontology_templates import load_ontology
    logger.info("derive_document_anchors: document_id=%s", document_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_document_anchors")
    db = _get_db()
    try:
        if run_id is None:
            run_id = _get_pipeline_run_id(db, document_id)
        _update_stage_run(db, run_id, "derive_document_anchors", "RUNNING", attempt=1)
        db.commit()
        doc_json = _build_docling_document_json(document_id)
        ontology = load_ontology()
        merged = walk(doc_json, document_id, run_id, ontology)
        # Step 1: upsert vertices
        from app.services.graph_store import NodeRecord as _NR
        records = [_merged_entity_to_node_record(e) for e in merged.entities]
        graph_store = get_graph_store()
        rids = graph_store.upsert_nodes_batch_sync(records)
        # Step 2: identity → RID bridge
        rid_by_identity = {e.identity: rid for e, rid in zip(merged.entities, rids)}
        # Step 3: structural edges
        for edge in merged.edges:
            from_rid = rid_by_identity[edge.from_identity]
            to_rid = rid_by_identity[edge.to_identity]
            graph_store.create_structural_edge_sync(from_rid, to_rid, edge.rel_type)
        metrics = {
            "section_count": sum(1 for e in merged.entities if e.identity.ontology_name == "SECTION"),
            "figure_count": sum(1 for e in merged.entities if e.identity.ontology_name == "FIGURE"),
            "table_count": sum(1 for e in merged.entities if e.identity.ontology_name == "TABLE"),
            "document_ontology_emitted": any(e.identity.ontology_name == "DOCUMENT" for e in merged.entities),
            "fallback_fired": any(
                e.identity.identity_values == ("0",) for e in merged.entities
                if e.identity.ontology_name == "SECTION"
            ),
        }
        _update_stage_run(db, run_id, "derive_document_anchors", "COMPLETE", attempt=1, metrics=metrics)
        db.commit()
        return {"stage": "derive_document_anchors", "status": "ok", **metrics}
    except Exception as exc:
        db.rollback()
        _update_stage_run(db, run_id, "derive_document_anchors", "FAILED", attempt=1, error=str(exc))
        db.commit()
        raise
    finally:
        db.close()
```

Insert into the pipeline chain at the documented position:

```python
# Around line 1524 in pipeline.py where the chain is built
pipeline = chain(
    prepare_document.si(document_id, run_id),
    detect_and_translate.si(document_id, run_id),
    derive_document_metadata.si(document_id, run_id),
    purge_document_derivations.si(document_id, run_id),
    derive_picture_descriptions.si(document_id, run_id),
    derive_text_chunks_and_embeddings.si(document_id, run_id),
    derive_image_embeddings.si(document_id, run_id),
    derive_document_anchors.si(document_id, run_id),  # NEW
    derive_ontology_graph.si(document_id, run_id),
    collect_derivations.si(document_id, run_id),
    derive_structure_links.si(document_id, run_id),
    derive_canonicalization.si(document_id, run_id),
    finalize_document.si(document_id, run_id),
)
```

Register the task route in `celery_app.py:50`:
```python
"app.workers.pipeline.derive_document_anchors": {"queue": "graph"},
```

- [ ] **Step 3: Run → PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(worker): derive_document_anchors Celery task (D-4)

Runs between image embeddings and derive_ontology_graph. Uses the
structural-edge write path (create_structural_edge_sync), NOT the
ontology-relationship path. StageRun audit channel.

Plan: docs-alignment"
```

### Task D-5: Fixture `docling_document.json` for walker tests

**Files:**
- Create: `tests/fixtures/docling_anchors/sa2_minimal.json` (small synthetic 3-section DoclingDocument)
- Create: `tests/fixtures/docling_anchors/empty_structure.json` (no headings)
- Create: `tests/fixtures/docling_anchors/with_figures_tables.json`
- Create: `tests/fixtures/docling_anchors/with_document_number.json` (title contains "MIL-STD-1553B")

Each fixture is a minimal JSON matching the `DoclingDocument.model_dump()` shape. Content-wise ~50-200 lines each.

- [ ] Add fixture files
- [ ] Update `tests/unit/test_docling_anchor_walker.py` to load from these fixtures
- [ ] Commit: `test(anchors): DoclingDocument fixtures for walker (D-5)`

### Task D-6: Real-document smoke test

**Files:**
- Test: `tests/integration/test_docling_anchors_smoke.py` (marked `@pytest.mark.integration`)

- [ ] **Step 1: Test loads a real `docling_document.json` from `tests/fixtures/real_docs/` (if available; otherwise skip). Runs `walk()` + asserts non-zero SECTION count, non-zero FIGURE count if pictures in fixture.**
- [ ] **Step 2: Run** — PASS or SKIP.
- [ ] **Step 3: Commit**: `test(anchors): real-doc integration smoke (D-6)`

---

## Chunk E: Consumer updates (11 tasks)

### Task E-1: `ontology_bundles/air_defense_v3/derive_rules.py` identity lookup update

- [ ] Update `derive_structural_edges` to look up SECTION by `section_number`, FIGURE by `figure_ref`, TABLE by `table_ref`. Commit: `refactor(bundle): derive_rules identity lookups (E-1)`

### Task E-2: `pipeline.py:derive_structure_links` — no behavior change, verify SAME_SECTION stays chunk-to-chunk

- [ ] Read current implementation, verify `SAME_SECTION` is chunk-to-chunk by `section_path` string — no change needed per spec §6.4.
- [ ] Add a comment at `pipeline.py:4376` explicitly noting: "SAME_SECTION is chunk-to-chunk; SECTION vertices are written by derive_document_anchors (spec §3.4)." Commit: `docs(worker): annotate SAME_SECTION scope (E-2)`

### Task E-3: `pipeline.py:finalize_document` REQUIRED_STAGES

- [ ] Add `"derive_document_anchors"` to the `REQUIRED_STAGES` set at `app/workers/pipeline.py:4799`.
- [ ] Update any unit test fixture that enumerates REQUIRED_STAGES.
- [ ] Commit: `refactor(worker): finalize REQUIRED_STAGES includes anchors (E-3)`

### Task E-4: `app/services/arcadedb_graph.py` docstring

- [ ] Update the docstring at `:2020-2026` to drop ASSERTION from the document-scoped entity list; update per new classification.
- [ ] Commit: `docs(graph): refresh delete_extraction_layer docstring (E-4)`

### Task E-5: Frontend `GraphExplorer.tsx` + `entityTypes.ts`

- [ ] `frontend/src/components/GraphExplorer.tsx:33` — remove `"ASSERTION"`.
- [ ] `frontend/src/components/GraphExplorer.tsx:29` — keep `"SPECIFICATION"` (stays as first-class component vertex per §4.8).
- [ ] `frontend/src/constants/entityTypes.ts:23` — remove `"Assertion"` from `REFERENCE_TYPES`.
- [ ] Audit the rest of entityTypes.ts: keep demoted components in their arrays (they're still renderable).
- [ ] Commit: `refactor(frontend): drop ASSERTION from filter lists (E-5)`

### Task E-6: `app/services/extraction_merge.py` `_NAME_LIKE_KEYS`

- [ ] Change `"heading"` → `"section_number"` in the tuple at line 305. Commit: `refactor(merge): _NAME_LIKE_KEYS uses section_number (E-6)`

### Task E-7: `dossier_service.py` + `query_profiles.py` filter-list audit

- [ ] Audit `app/services/dossier_service.py:38-68` lists + `app/services/query_profiles.py:49-79` lists.
- [ ] No removals needed per spec §6.1 — demoted components retain their ontology_name. Add integration test that issues a dossier query for each demoted-to-component entity type and asserts non-zero results when the data exists.
- [ ] Commit: `test(retrieval): dossier filter-list component-vertex coverage (E-7)`

### Task E-8: `canonicalization.py` verification

- [ ] Verify no SPECIFICATION (or other demoted) hardcoded references. Per spec §6.5 audit: code is generic.
- [ ] Commit a note-only doc comment: `docs(canonicalization): confirm component-demotion is no-op (E-8)`

### Task E-9: `_classify_extraction_quality` rewrite (spec §6.8)

**Files:**
- Modify: `app/workers/pipeline.py:282-317`
- Modify: `app/workers/pipeline.py:320+` — `_write_pipeline_run_metrics` to pass new args
- Test: `tests/unit/test_classify_extraction_quality.py` (rewrite)

- [ ] **Step 1: Failing tests** — the 3 new states (ok/degraded/anomaly) per new signature.

- [ ] **Step 2: Implement**

```python
_DOMAIN_PASS_NAMES = frozenset({
    "radar_domain", "missile_domain", "other_systems", "system_links",
})  # NOTE: reference removed

def _classify_extraction_quality(
    pass_outcomes: dict,
    section_count: int,
    text_chunk_count: int,
) -> str:
    domain_hit = any(
        v.get("yield_status") == "HIT"
        for k, v in pass_outcomes.items()
        if k in _DOMAIN_PASS_NAMES
    )
    if domain_hit:
        return "ok"
    if section_count > 0 and text_chunk_count > 0:
        return "degraded"
    return "anomaly"
```

- [ ] **Step 3: Update caller `_write_pipeline_run_metrics`** to fetch `section_count` via `graph_store.count_ontology_nodes_sync("SECTION", document_id)` (add that helper if missing — separate sub-task E-10) and `text_chunk_count` via SQL on `retrieval.text_chunks`.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(worker): _classify_extraction_quality uses graph counts (E-9)

New signature (pass_outcomes, section_count, text_chunk_count). degraded
now means 'SECTION+TextChunk exist but no domain HIT' (spec §6.8).

Plan: docs-alignment"
```

### Task E-10: `graph_store.count_ontology_nodes_sync` helper if missing

- [ ] Grep `grep -n "def count.*nodes\|def count_" app/services/graph_store.py` — add `count_ontology_nodes_sync(entity_type, document_id) -> int` only if absent.
- [ ] If added, include a unit test.
- [ ] Commit: `feat(graph): count_ontology_nodes_sync helper (E-10)` — or skip this task if helper exists.

### Task E-11: Update reference-pass-assuming tests

- [ ] `tests/unit/test_coverage_checker.py:68` — `module="extraction_schemas.reference"` → either delete the case or re-target to remaining passes.
- [ ] `tests/unit/test_extraction_schemas.py:8` — update 5-pass assumption.
- [ ] `tests/unit/test_ontology_bundles.py:11` — update 5-pass assumption.
- [ ] Run `.venv/bin/python -m pytest tests/unit/ -q` clean.
- [ ] Commit: `test(pipeline): update tests for 4-pass reality (E-11)`

---

## Chunk F: Test cleanup (7 tasks)

### Task F-1 through F-6: Delete parity test files (one commit per file — grouped 3 per task for efficiency)

- [ ] **F-1** Delete `test_arcadedb_schema_ontology_source_parity.py`, `test_canonicalization_ontology_source_parity.py`, `test_dossier_service_ontology_source_parity.py`. Run suite. Commit: `test(cleanup): delete 3 parity tests (F-1)`
- [ ] **F-2** Delete `test_extraction_merge_ontology_source_parity.py`, `test_graph_store_ontology_source_parity.py`, `test_main_api_ontology_source_parity.py`. Commit: `test(cleanup): delete 3 parity tests (F-2)`
- [ ] **F-3** Delete `test_pipeline_ontology_source_parity.py`, `test_query_profiles_ontology_source_parity.py`, `test_ontology_templates_internals_parity.py`. Commit: `test(cleanup): delete 3 parity tests (F-3)`
- [ ] **F-4** Delete `test_relationships_parity.py`, `test_validation_matrix_parity.py`. Commit: `test(cleanup): delete 2 relationship-parity tests (F-4)`
- [ ] **F-5** Delete `test_introspect_entity_types.py`, `test_introspect_ontology_dict.py`, `test_introspect_relationship_types.py`, `test_introspect_validation_and_weights.py`. Commit: `test(cleanup): delete 4 introspect-parity tests (F-5)`
- [ ] **F-6** Delete `test_ontology_source_flag.py` + `test_arcadedb_schema.py` + `tests/fixtures/ontology/air_defense_v3_snapshot.yaml`. Commit: `test(cleanup): delete ontology_source_flag + snapshot fixture (F-6)`

### Task F-7: Drop xfail markers + verify 21/21

- [ ] Remove `@pytest.mark.xfail(strict=True, reason="Chunk B will fix")` from the 12 new contract tests in `tests/unit/contracts/*`
- [ ] Remove the 2 remaining xfails from `tests/unit/test_docs_compliance_contracts.py` (`test_identity_fields_have_examples`, `test_descriptions_and_examples_on_extraction_relevant_fields`)
- [ ] Run `.venv/bin/python -m pytest tests/unit/ -q` — expect: **21 contract tests green, 0 xfails remain**.
- [ ] Commit: `test(contracts): drop xfails, all 21 contract tests green (F-7)`

---

## Chunk G: Migration + validation (4 tasks)

### Task G-1: Write `scripts/full_purge_and_reingest.py`

**Files:**
- Create: `scripts/full_purge_and_reingest.py`
- Test: `tests/unit/test_full_purge_script.py` (dry-run coverage)

- [ ] **Step 1: Implement the script per spec §5.1 + §5.2** — argparse with `--dry-run` and `--i-understand-this-deletes-derived-data` flags. Functions: `stop_workers()`, `truncate_postgres(dry_run)`, `reset_arcadedb()`, `empty_minio_derived()`, `flush_redis()`, `apply_migrations()`, `restart_workers()`, `reset_document_statuses()`, `enqueue_pipeline()`, `poll_until_complete(timeout)`, `emit_report(path)`.

- [ ] **Step 2: Dry-run test** — verify `--dry-run` prints truncate plan without executing. Asserts exact per-table list from §5.1.

- [ ] **Step 3: Run test → PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(scripts): full_purge_and_reingest.py (G-1)

Dry-run prints the truncate list. Real run wipes derived state
(Postgres, ArcadeDB, MinIO derived/*, Redis) and enqueues pipeline
for every doc in ingest.documents. Report written to /tmp/
migration-report-{timestamp}.md.

Plan: docs-alignment"
```

### Task G-2: Dry-run on the 21-doc corpus

- [ ] Execute `.venv/bin/python scripts/full_purge_and_reingest.py --dry-run`. Confirm the truncate list matches §5.1.
- [ ] No commit (observational).

### Task G-3: Real run — execute migration

- [ ] Execute `.venv/bin/python scripts/full_purge_and_reingest.py --i-understand-this-deletes-derived-data`.
- [ ] Wait 3-6 hours for completion.
- [ ] Report lands at `/tmp/migration-report-{timestamp}.md`.
- [ ] No commit (observational).

### Task G-4: Acceptance gate

- [ ] Read the migration report. Verify every item in §5.4:
    - All 21 docs reach `COMPLETE` or `PARTIAL_COMPLETE` (not `FAILED`).
    - ≥3 radar/missile-heavy docs produce `extraction_quality="ok"`.
    - Every doc produces ≥1 SECTION (or sentinel `"0"`), ≥1 TextChunk, and a structural `Document` vertex.
    - ≥50% of the corpus emits an ontology DOCUMENT (heuristic sanity check).
    - Zero `Input should be a valid integer` errors in docling-graph logs.
- [ ] Commit the report artifact into the repo (as `docs/migration-reports/2026-04-16-docs-alignment.md`):

```bash
git add docs/migration-reports/2026-04-16-docs-alignment.md
git commit -m "docs(migration): acceptance gate passed, docs-alignment complete (G-4)

All 21 docs re-ingested on new schemas. Gate signals documented in
spec §5.4. Current plan (2026-04-14-docling-graph-schema-compliance.md)
is now unblocked; resume from Phase 7 Task 53.

Plan: docs-alignment"
```

---

## Post-plan: handoff to current plan

Once G-4 commits the acceptance gate:

- [ ] Open `docs/superpowers/plans/2026-04-14-docling-graph-schema-compliance.md.tasks.json` and mark Task 53 as unblocked.
- [ ] Resume via `/superpowers-extended-cc:executing-plans docs/superpowers/plans/2026-04-14-docling-graph-schema-compliance.md`.
- [ ] Update `/home/josh/.claude/projects/-home-josh-development-EIP-MMDPP/memory/MEMORY.md` — add a project memory pointing at this plan's commit SHA as the canonical reset point.
