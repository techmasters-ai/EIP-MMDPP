# Upstream Refs — Plan 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `document_plus_entity_refs` passes (specifically `system_links`) actually receive, prompt-inject, and resolve upstream entity references, so cross-pass relationships (`ASSOCIATED_WITH`, `CUES`) become merged edges instead of being rejected as `UNKNOWN_REF_ID`.

**Architecture:** Fix the three places in the chain where upstream refs silently degrade: (1) the worker generates colliding `E###` ids with unfiltered identity payloads and null display labels — rewrite `_extend_upstream_refs` with a single monotonic counter, ontology-filtered identities, and real display labels, *plus* reject any ref whose ontology identity fields are missing or empty (one shared validity rule applied at ref-emit, request-build, and merge-attach time); (2) the worker never attaches `PassResult.upstream_refs` for `document_plus_entity_refs` passes — attach after `_parse_pass_response` using a newly-public `logical_identity_from_dict` helper, filtered to `pass_def.depends_on`, passed through the same validity rule, and deterministically ordered; (3) the docling-graph service logs upstream entities but never threads them into the model input — add `_render_upstream_entities_preamble()` that consumes the existing `EntityRef` Pydantic model (not dicts), a discovery sub-step to verify/choose the actual library hook (prepend-to-user-document is the fallback path), a feature flag, and `upstream_ref_count` + `upstream_preamble_applied` response metadata for observability. Also extend `_write_stage_run`'s `counts` JSONB with per-reason rejection breakdown so "UNKNOWN_REF_ID drops to near zero" is directly measurable. No new passes, no ExtractPassRequest wire-contract changes (`document_id` already present), no Postgres migrations; response metadata fields added are additive defaults so existing callers keep working.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI, Celery, httpx, pytest, Ollama (gpt-oss:120b), Docker Compose.

---

## File Structure

### Production

| File | Responsibility | Change type |
|---|---|---|
| `app/services/extraction_merge.py` | Rename `_identity_from_dict` → `logical_identity_from_dict` (public); update internal caller; keep existing behaviour. | Modify (~5 lines) |
| `app/workers/pipeline.py` | Fix `_extend_upstream_refs` (monotonic counter, ontology-filtered identity, populated `display_label`). Add `_is_valid_upstream_ref(ref, ontology)` — the single shared validity rule. Add `_select_upstream_refs_for_pass` (filters by `depends_on`, drops invalid refs, sorts deterministically). `_build_extract_pass_request` already has `document_id` parameter today? No — call signature must add it (request body field `document_id` already exists in the service schema). After parse, attach `pass_result.upstream_refs` for `document_plus_entity_refs` passes, using the same validity rule. Extend `_write_stage_run`'s `counts` dict with `rejections_by_reason` so UNKNOWN_REF_ID trend is measurable. | Modify (~100 lines, 3 new functions) |
| `docker/docling-graph/app/schemas.py` | Add `upstream_ref_count: int = 0` and `upstream_preamble_applied: bool = False` to `ExtractionMetadata`. **No change to `ExtractPassRequest`** — `document_id: Optional[str]` is already present (schemas.py:60-63). | Modify (~2 lines) |
| `docker/docling-graph/app/main.py` | New `_render_upstream_entities_preamble(entities)` helper consuming `list[EntityRef]` Pydantic models (not dicts). Inject preamble via the chosen library hook discovered in Task 5a (fallback: prepend preamble text to the `docling_document_json` body before writing the tmp file). Gate behind `DOCLING_GRAPH_UPSTREAM_PREAMBLE` env flag (default on). `run_extraction_pass` keeps returning just `context`; set `context._upstream_preamble_applied` as an attribute so existing test mocks don't break. **Keep `from docling_graph import run_pipeline` local** — hoisting it would break the test shim on host envs without `docling_graph` installed. Expand START/END logs with `upstream_ref_count`, `input_mode`, `node_count`, `edge_count`. Populate the two new metadata fields on every response. | Modify (~80 lines, 1 new function) |
| `docker/docling-graph/app/config_builder.py` | **Path A only** (Task 5a chose a library kwarg hook): widen `build_pipeline_config(source, template_class)` to accept `extra_prompt_preamble: str \| None = None` and forward it onto the `PipelineConfig(...)` kwarg that Task 5a identified (exact kwarg name comes from the discovery output). Default of `None` keeps existing callers unchanged. **Path B does NOT edit this file** — the body mutation happens inside `run_extraction_pass` before `build_pipeline_config` is called. | Modify (~3 lines, Path A only) |

### Tests

| File | Responsibility | Change type |
|---|---|---|
| `tests/unit/test_extraction_merge.py` | New `TestLogicalIdentityFromDict` class. New `test_system_links_resolves_valid_ref_ids` success path (using the existing `_make_pass_result(pass_name, entities, relationships)` helper where `entities` is `(type, identity_dict, properties_dict)` tuples). | Modify (~70 lines) |
| `tests/unit/test_pipeline_upstream_refs.py` | New file. `_extend_upstream_refs` correctness + validity rule. `_is_valid_upstream_ref` truth table (missing/None/empty/present). `_select_upstream_refs_for_pass` filtering + validity + ordering. `_build_extract_pass_request` `document_id` propagation. `_run_single_pass` attaches validated `pass_result.upstream_refs` for `document_plus_entity_refs`, none for `document_only`. Invalid refs never reach the request body or the merge attach. | Create |
| `tests/unit/test_run_single_pass.py` | Extend — follow existing fixture pattern in that file; two cases: document_plus_entity_refs attaches upstream_refs, document_only does not. | Modify (~40 lines) |
| `docker/docling-graph/tests/test_extract_pass_endpoint.py` | Extend using the existing harness — `TestClient(app)` + `patch(f"{_DG_MODULE_NAME}.run_extraction_pass")` with `_mock_run_pipeline_return()` pattern from line 87. New cases assert `upstream_ref_count` in response metadata and `upstream_preamble_applied` flips with input_mode + env flag. Existing `_mock_run_pipeline_return()` helper gains a `preamble_applied=False` keyword so tests that don't care keep passing; new tests set it explicitly. | Modify (~60 lines) |
| `docker/docling-graph/tests/conftest.py` | **Task 5b sub-step:** move `_ensure_dg_app_package()` + `dg_app_module` fixture out of `test_extract_pass_endpoint.py:13-70` into here, so any test file in `docker/docling-graph/tests/` (not just the sibling where they were originally declared) can request the fixture. Without this move, `test_upstream_entities_preamble.py` fails with `fixture 'dg_app_module' not found` — pytest does not share fixtures between test modules. Existing `test_extract_pass_endpoint.py` uses the fixture unchanged after the move. | Modify (~60 lines moved, no behavioural change) |
| `docker/docling-graph/tests/test_upstream_entities_preamble.py` | New file. Pure formatting tests — exact line shape, deterministic ordering, empty → empty, instruction footer, env-flag off → returns empty string. **All inputs are `EntityRef` Pydantic models**, not dicts, to match the runtime type. Reuses the `dg_app_module` fixture now defined in `conftest.py`. | Create |

### Config

| File | Change type |
|---|---|
| `docker-compose.yml` (docling-graph environment block) | Add `DOCLING_GRAPH_UPSTREAM_PREAMBLE: ${DOCLING_GRAPH_UPSTREAM_PREAMBLE:-true}` so the flag is surfaced but defaults on. Modify (~1 line). |

### No changes

- `ontology_bundles/` (manifest, schemas) — passes and `required` markers untouched.
- Postgres schema, migrations — none.
- ArcadeDB schema — none.

---

## Rollout ordering

Tasks must run in order. Each task ends with a green test run and a commit; a mid-task failure reverts just that task.

| Order | Task | Why it's here |
|---|---|---|
| 1 | Service logging + metadata | Shippable alone. Gives the next upload batch real diagnostic data whether or not the rest lands. |
| 2 | Promote `logical_identity_from_dict` | Prerequisite for Task 4. Zero behaviour change. |
| 3 | Fix `_extend_upstream_refs` + `_is_valid_upstream_ref` | Prerequisite for Task 4. Unique ids, ontology-filtered identity, real display labels, single validity rule. |
| 4 | Validity filter + `_select_upstream_refs_for_pass` + `pass_result.upstream_refs` + `rejections_by_reason` metrics | The merge-side fix — `UNKNOWN_REF_ID` stops firing on legitimate refs, and the trend is now measurable via `stage_runs.metrics.rejections_by_reason`. |
| 5a | DISCOVERY — locate prompt hook in docling_graph library | Task 5b's implementation depends on this finding. No production code written here; decision recorded on the PR. |
| 5b | Preamble injection using the chosen hook | Gives the model the ref list. Fallback path prepends to document body; env flag `DOCLING_GRAPH_UPSTREAM_PREAMBLE` makes it trivially rollback-able. Attribute (`context._upstream_preamble_applied`) avoids a tuple-return refactor that would break existing endpoint-test mocks. |
| 6 | Verify, rebuild, recreate, live smoke | Full suite + container recreate + log spot-check. |

---

## Task 1: Service logging + ExtractionMetadata observability

**Files:**
- Modify: `docker/docling-graph/app/schemas.py`
- Modify: `docker/docling-graph/app/main.py:386-470` (the `extract_pass` handler)
- Test: `docker/docling-graph/tests/test_extract_pass_endpoint.py`

### Step 1: Write failing test — metadata carries `upstream_ref_count` for document_plus_entity_refs

- [ ] **Step 1: Write the failing test**

Append to `docker/docling-graph/tests/test_extract_pass_endpoint.py`. Use the **existing** `client` fixture (TestClient-based, defined in `tests/conftest.py`) and the existing `patch(f"{_DG_MODULE_NAME}.run_extraction_pass")` + `_mock_run_pipeline_return()` pattern from lines 86-99 and 163-192 — the live path is deliberately patched out because real calls would hit an LLM:

```python
def test_metadata_reports_upstream_ref_count_for_document_plus_entity_refs(client):
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "system_links",
            "docling_document_json": {"name": "test"},
            "upstream_entities": [
                {"ref_id": "E001", "entity_type": "RADAR_SYSTEM",
                 "identity_values": {"system_name": "Fan Song"},
                 "display_label": "Fan Song"},
                {"ref_id": "E002", "entity_type": "MISSILE_SYSTEM",
                 "identity_values": {"system_name": "SA-2"},
                 "display_label": "SA-2"},
            ],
        })
    assert resp.status_code == 200, resp.text
    meta = resp.json()["metadata"]
    assert meta["upstream_ref_count"] == 2
    # Task 1 only adds the field; Task 5b flips it to True when preamble is injected.
    assert meta["upstream_preamble_applied"] is False


def test_metadata_reports_zero_refs_for_document_only(client):
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return()
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "reference",
            "docling_document_json": {"name": "test"},
        })
    assert resp.status_code == 200, resp.text
    meta = resp.json()["metadata"]
    assert meta["upstream_ref_count"] == 0
    assert meta["upstream_preamble_applied"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd docker/docling-graph && pytest tests/test_extract_pass_endpoint.py::test_metadata_reports_upstream_ref_count_for_document_plus_entity_refs tests/test_extract_pass_endpoint.py::test_metadata_reports_zero_refs_for_document_only -v`
Expected: FAIL — `KeyError: 'upstream_ref_count'` (field not in schema yet).

- [ ] **Step 3: APPEND two fields to ExtractionMetadata — do NOT replace the class**

**This is an additive edit.** `docker/docling-graph/app/schemas.py:10-21` already defines these fields in order:

```python
class ExtractionMetadata(BaseModel):
    node_count: int = 0
    edge_count: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)
    extraction_contract: str = "delta"
    gleaning_passes: int = 0
    resolvers_applied: bool = False
    quality_gate_passed: bool = True
    validation_pass_applied: bool = False
    validation_pass_edges_added: int = 0
```

Leave every existing field — including the `Field(default_factory=dict)` declarations for `node_types`/`edge_types` and the five validation-pass-related fields — exactly as-is. Append the two new fields **after** the last existing field:

```python
    # --- Plan 1 observability — append AFTER existing fields. -------
    # Populated by the extract_pass handler; see main.py's metadata
    # construction for how these are filled in.
    upstream_ref_count: int = 0
    upstream_preamble_applied: bool = False
```

The diff you should produce is two added lines at the bottom of the class body, not a rewrite of the whole class. If pytest shows any existing response field is missing or `node_types == {}` stops being a fresh dict per instance, you accidentally replaced the class — revert and re-apply additively.

- [ ] **Step 4: Populate the new fields in the endpoint — ADDITIVE edit**

In `docker/docling-graph/app/main.py`, find the `ExtractionMetadata(...)` construction inside the `extract_pass` handler (around `main.py:462`). The current call passes exactly five kwargs:

```python
metadata = ExtractionMetadata(
    node_count=getattr(meta, "node_count", graph.number_of_nodes()),
    edge_count=getattr(meta, "edge_count", graph.number_of_edges()),
    node_types=getattr(meta, "node_types", {}),
    edge_types=getattr(meta, "edge_types", {}),
    extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
)
```

The other fields on the model (`gleaning_passes`, `resolvers_applied`, `quality_gate_passed`, `validation_pass_applied`, `validation_pass_edges_added`) are **defined in `schemas.py:10-21` with defaults** but the handler does not pass them today — they fall back to those defaults. Do NOT add kwargs for them in this task.

Two additive edits:

1. **Compute `upstream_ref_count` just above the existing metadata call** so both the log line and the metadata share the same value:

   ```python
   upstream_ref_count = len(body.upstream_entities) if body.upstream_entities else 0
   ```

2. **Append two kwargs** to the existing five-kwarg `ExtractionMetadata(...)` call — leave the existing five exactly in place:

   ```python
   metadata = ExtractionMetadata(
       node_count=getattr(meta, "node_count", graph.number_of_nodes()),
       edge_count=getattr(meta, "edge_count", graph.number_of_edges()),
       node_types=getattr(meta, "node_types", {}),
       edge_types=getattr(meta, "edge_types", {}),
       extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
       # --- Plan 1 — appended below. -----------------------------------
       upstream_ref_count=upstream_ref_count,
       upstream_preamble_applied=False,  # flipped to True in Task 5b
   )
   ```

The diff for this step is exactly two new kwarg lines inside the existing `ExtractionMetadata(...)` call, plus the `upstream_ref_count = ...` assignment above it. Nothing else in that handler should change in this task.

Also expand the existing START / END logs (from the earlier salvage-logging change) to include counts:

```python
    logger.info(
        "extract-pass: START bundle=%s pass=%s input_mode=%s document_id=%s upstream_ref_count=%d",
        body.bundle_key, body.pass_name, pass_def.get("input_mode"),
        body.document_id, upstream_ref_count,
    )
    # ... existing run_extraction_pass call ...
    logger.info(
        "extract-pass: END bundle=%s pass=%s document_id=%s node_count=%d edge_count=%d",
        body.bundle_key, body.pass_name, body.document_id,
        metadata.node_count, metadata.edge_count,
    )
```

Move `upstream_ref_count = len(...)` calculation above the START log so both the log line and the metadata use the same value.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd docker/docling-graph && pytest tests/test_extract_pass_endpoint.py -v`
Expected: all existing tests still PASS; both new tests PASS.

- [ ] **Step 6: Commit**

```bash
git add docker/docling-graph/app/schemas.py docker/docling-graph/app/main.py docker/docling-graph/tests/test_extract_pass_endpoint.py
git commit -m "feat(docling-graph): surface upstream_ref_count + pass-level logs on /extract-pass

Adds upstream_ref_count and upstream_preamble_applied to ExtractionMetadata so
operators can confirm the worker is wiring upstream entities into the service.
Expands START/END logs with input_mode, upstream_ref_count, node_count, and
edge_count. upstream_preamble_applied stays False until the preamble is
actually injected (Plan 1 / Task 5)."
```

---

## Task 2: Promote `logical_identity_from_dict` to public helper

**Files:**
- Modify: `app/services/extraction_merge.py:316-343` (`_identity_from_dict`) and `:399-403` (internal caller)
- Test: `tests/unit/test_extraction_merge.py`

### Step 1: Write failing tests for the public helper

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_extraction_merge.py` (near the other identity-related tests):

```python
class TestLogicalIdentityFromDict:
    """logical_identity_from_dict (renamed from the private helper) is the
    canonical way the worker converts upstream entity refs into
    LogicalIdentity objects for PassResult.upstream_refs."""

    ONTOLOGY = {
        "entity_types": [
            {
                "name": "RADAR_SYSTEM",
                "identity_fields": ["system_name"],
                "identity_scope": "global",
            },
            {
                "name": "SPECIFICATION",
                "identity_fields": ["parameter", "value"],
                "identity_scope": "document",
            },
        ],
    }

    def test_happy_path_global_scope(self):
        from app.services.extraction_merge import logical_identity_from_dict
        identity = logical_identity_from_dict(
            "RADAR_SYSTEM", {"system_name": "Fan Song"}, self.ONTOLOGY, "doc-1",
        )
        assert identity is not None
        assert identity.entity_type == "RADAR_SYSTEM"
        assert identity.identity_tuple == ("Fan Song",)
        assert identity.scope == "global"
        assert identity.document_id is None  # global scope drops document_id

    def test_happy_path_document_scope(self):
        from app.services.extraction_merge import logical_identity_from_dict
        identity = logical_identity_from_dict(
            "SPECIFICATION",
            {"parameter": "range", "value": "150"},
            self.ONTOLOGY, "doc-7",
        )
        assert identity is not None
        assert identity.document_id == "doc-7"
        assert identity.identity_tuple == ("range", "150")

    def test_missing_identity_key_returns_none(self):
        from app.services.extraction_merge import logical_identity_from_dict
        identity = logical_identity_from_dict(
            "SPECIFICATION", {"parameter": "range"}, self.ONTOLOGY, "doc-1",
        )
        assert identity is None

    def test_unknown_entity_type_returns_none(self):
        from app.services.extraction_merge import logical_identity_from_dict
        identity = logical_identity_from_dict(
            "UNKNOWN_TYPE", {"system_name": "X"}, self.ONTOLOGY, "doc-1",
        )
        assert identity is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_extraction_merge.py::TestLogicalIdentityFromDict -v`
Expected: FAIL — `ImportError: cannot import name 'logical_identity_from_dict'`.

- [ ] **Step 3: Rename + expose the helper**

In `app/services/extraction_merge.py`, rename the function at line 316 and update the one caller at line ~399-403:

```python
def logical_identity_from_dict(
    entity_type: str,
    identity_dict: dict,
    ontology: dict,
    document_id: str,
) -> LogicalIdentity | None:
    """Build a LogicalIdentity from a raw identity dict.

    This is the canonical way the worker converts an upstream entity ref's
    ``identity_values`` into a ``LogicalIdentity`` suitable for
    ``PassResult.upstream_refs``. The merge resolver compares these objects
    by value (``@dataclass(frozen=True)``) against the merged entity index,
    so the identity tuple must come straight from the ontology's
    ``identity_fields`` list in declared order.

    Returns None if the entity_type is unknown or the payload is missing a
    required identity key — in that case the caller should drop the ref.
    """
    # (body unchanged)
```

Update the internal caller (inside `_resolve_relationship` or whichever function previously called `_identity_from_dict`):

```python
        from_identity = logical_identity_from_dict(
            from_type, from_identity_dict, ontology, document_id
        )
        to_identity = logical_identity_from_dict(
            to_type, to_identity_dict, ontology, document_id
        )
```

- [ ] **Step 4: Run the full extraction_merge test module to verify no regressions**

Run: `.venv/bin/pytest tests/unit/test_extraction_merge.py -v`
Expected: all existing tests PASS plus the 4 new `TestLogicalIdentityFromDict` cases PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/extraction_merge.py tests/unit/test_extraction_merge.py
git commit -m "refactor(extraction_merge): promote _identity_from_dict to public API

Renames the private helper to logical_identity_from_dict so the worker can
use it to convert upstream entity refs into LogicalIdentity objects for
PassResult.upstream_refs. Behaviour unchanged; docstring clarified."
```

---

## Task 3: Fix `_extend_upstream_refs` — unique ids, ontology-filtered identity, display_label

**Files:**
- Modify: `app/workers/pipeline.py:1733-1761` (`_extend_upstream_refs`)
- Test: `tests/unit/test_pipeline_upstream_refs.py` (new)

### Step 1: Write failing tests

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pipeline_upstream_refs.py`:

```python
"""Tests for upstream-ref machinery in app/workers/pipeline.py.

The docling-graph service sees these refs on document_plus_entity_refs passes
(system_links), and the merge resolver matches from_ref_id / to_ref_id against
them. Bugs here silently produce UNKNOWN_REF_ID rejections, which is why
these tests exist before the implementation.
"""
from types import SimpleNamespace

from app.workers.pipeline import _extend_upstream_refs


class _FakePassResult:
    def __init__(self, entities_by_type: dict):
        self._by_type = entities_by_type

    def iter_entities_of_type(self, entity_type: str):
        return iter(self._by_type.get(entity_type, []))


ONTOLOGY = {
    "entity_types": [
        {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
        {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
    ],
}


class TestExtendUpstreamRefs:

    def _pass_def(self, primary_types):
        return SimpleNamespace(
            name="radar_domain",
            primary_entity_types=primary_types,
        )

    def test_two_entity_types_produce_unique_ids(self):
        """Bug fix: previous impl reset the counter per entity type, so
        E001/E002 from type A were overwritten by E001/E002 from type B."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [
                SimpleNamespace(system_name="Fan Song", confidence=0.9),
                SimpleNamespace(system_name="Big Bird", confidence=0.8),
            ],
            "MISSILE_SYSTEM": [
                SimpleNamespace(system_name="SA-2", confidence=0.9),
                SimpleNamespace(system_name="SA-3", confidence=0.8),
            ],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["RADAR_SYSTEM", "MISSILE_SYSTEM"]),
            ONTOLOGY,
        )
        assert sorted(refs.keys()) == ["E001", "E002", "E003", "E004"]
        # Ids are unique: each ref points to a distinct entity
        assert len({r.identity_values["system_name"] for r in refs.values()}) == 4

    def test_appending_to_existing_refs_continues_counter(self):
        """A second pass should not clobber refs accumulated from a prior pass."""
        refs: dict = {
            "E001": SimpleNamespace(
                pass_origin="reference",
                entity_type="SECTION",
                identity_values={"heading": "Intro"},
                display_label="Intro",
            ),
        }
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
        })
        _extend_upstream_refs(
            refs, pass_result,
            self._pass_def(["RADAR_SYSTEM"]),
            ONTOLOGY,
        )
        assert "E001" in refs and refs["E001"].entity_type == "SECTION"
        assert "E002" in refs and refs["E002"].entity_type == "RADAR_SYSTEM"

    def test_identity_values_filter_to_ontology_identity_fields(self):
        """instance.__dict__ may have confidence, nomenclature, etc.; only
        ontology identity_fields belong in identity_values (merge compares
        by identity tuple, so extra keys fragment identity)."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [
                SimpleNamespace(
                    system_name="Fan Song",
                    nomenclature="SA-2-RADAR",
                    confidence=0.9,
                ),
            ],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["RADAR_SYSTEM"]), ONTOLOGY)
        assert list(refs["E001"].identity_values.keys()) == ["system_name"]
        assert refs["E001"].identity_values["system_name"] == "Fan Song"

    def test_display_label_is_populated(self):
        refs: dict = {}
        pass_result = _FakePassResult({
            "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["RADAR_SYSTEM"]), ONTOLOGY)
        assert refs["E001"].display_label == "Fan Song"

    def test_unknown_ontology_type_is_skipped(self):
        """If a primary_entity_type isn't in the ontology, we shouldn't emit a
        ref with a malformed identity (previous impl would have crashed)."""
        refs: dict = {}
        pass_result = _FakePassResult({
            "UNREGISTERED_TYPE": [SimpleNamespace(name="X")],
        })
        _extend_upstream_refs(refs, pass_result, self._pass_def(["UNREGISTERED_TYPE"]), ONTOLOGY)
        assert refs == {}

    def test_empty_primary_types_is_noop(self):
        refs: dict = {}
        pass_result = _FakePassResult({})
        _extend_upstream_refs(refs, pass_result, self._pass_def([]), ONTOLOGY)
        assert refs == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py -v`
Expected: `test_two_entity_types_produce_unique_ids` FAIL (collisions), `test_identity_values_filter_to_ontology_identity_fields` FAIL (extra keys present), `test_display_label_is_populated` FAIL (None), `test_unknown_ontology_type_is_skipped` may FAIL or pass by accident. That's the target.

- [ ] **Step 3: Rewrite `_extend_upstream_refs`**

In `app/workers/pipeline.py`, replace lines 1733-1761:

```python
def _extend_upstream_refs(
    upstream_refs: dict, pass_result, pass_def, ontology
) -> None:
    """Add ref_id → ref entries to upstream_refs for every primary entity
    produced by this pass so downstream passes can reference them.

    Uses a SINGLE monotonic counter across all primary_entity_types so
    ids don't collide (previous impl restarted the enumerate() counter per
    entity type). identity_values is filtered to the ontology's
    identity_fields only — merge_and_resolve compares refs by identity
    tuple, so extra keys would fragment identity. display_label is
    populated via build_display_label so downstream prompts and UIs get a
    meaningful name.
    """
    from types import SimpleNamespace
    from app.services.extraction_merge import build_display_label

    ontology_by_type = {
        e["name"]: e for e in ontology.get("entity_types", [])
    }

    counter = len(upstream_refs) + 1
    if not hasattr(pass_result, "iter_entities_of_type"):
        return

    for entity_type in pass_def.primary_entity_types:
        entity_def = ontology_by_type.get(entity_type)
        if entity_def is None:
            # Not in ontology — skip rather than emit a malformed ref.
            continue
        identity_fields = list(entity_def.get("identity_fields") or ())

        for instance in pass_result.iter_entities_of_type(entity_type):
            instance_dict = (
                instance.__dict__
                if hasattr(instance, "__dict__")
                else {}
            )
            identity_values = {
                k: instance_dict.get(k) for k in identity_fields
            }
            # Everything non-identity that isn't private goes into
            # properties so build_display_label can use it as a fallback.
            properties = {
                k: v
                for k, v in instance_dict.items()
                if not k.startswith("_") and k not in identity_values
            }
            display_label = build_display_label(
                entity_type, identity_values, properties,
            )

            ref_id = f"E{counter:03d}"
            upstream_refs[ref_id] = SimpleNamespace(
                pass_origin=pass_def.name,
                entity_type=entity_type,
                identity_values=identity_values,
                display_label=display_label,
            )
            counter += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py -v`
Expected: all 6 cases PASS.

Also run the existing pass-flow tests to check nothing regressed:
Run: `.venv/bin/pytest tests/unit/test_derive_ontology_graph_bundle_passes.py tests/unit/test_run_single_pass.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/workers/pipeline.py tests/unit/test_pipeline_upstream_refs.py
git commit -m "fix(pipeline): _extend_upstream_refs emits unique ids, ontology identity, real display labels

Previously: (a) the enumerate() counter restarted at the same start value for
every entity_type so multiple types collided on the same E### ids; (b)
identity_values included every non-underscore attribute of the instance
(confidence, nomenclature, …) which fragmented LogicalIdentity comparisons in
merge_and_resolve; (c) display_label was hardcoded to None.

Now uses a single monotonic counter, filters identity_values to the
ontology's identity_fields, and populates display_label via
build_display_label()."
```

---

## Task 4: Validity rule + `PassResult.upstream_refs` attach + `_select_upstream_refs_for_pass` + `document_id` in request

**Single validity rule** (used at three sites — ref emission, request build, merge attach): a ref is valid iff **(a)** its `entity_type` is known to the ontology, **(b)** that entity type declares **at least one** `identity_field` in the ontology, **(c)** every declared `identity_field` is present as a key in `identity_values`, AND **(d)** every such value is truthy (not `None`, not empty after `str.strip()`). Rule (b) explicitly rejects anchor-less types like `PROPULSION_STACK` (`ontology.yaml:579` declares `identity_fields: []`) — with no identity there is no anchor to hand the LLM and no way to round-trip to a `LogicalIdentity` that merge could use. Invalid refs are dropped silently at each stage; they never appear in the request body, never appear in `pass_result.upstream_refs`, and never generate an `UNKNOWN_REF_ID` rejection because they never existed.

**Files:**
- Modify: `app/workers/pipeline.py` — add `_is_valid_upstream_ref(ref, ontology)` near `_extend_upstream_refs`; use it inside `_extend_upstream_refs`, `_select_upstream_refs_for_pass`, and the new merge-attach block in `_run_single_pass`. `_build_extract_pass_request` signature gains `document_id: str`. The request body already accepts `document_id` (service schema unchanged).
- Test: `tests/unit/test_pipeline_upstream_refs.py` (extend), `tests/unit/test_run_single_pass.py` (extend), `tests/unit/test_extraction_merge.py` (extend using the existing `_make_pass_result(pass_name, entities, relationships)` tuple-entities helper).

**Skipped from the earlier draft:** no change to `docker/docling-graph/app/schemas.py:ExtractPassRequest`. `document_id` is already a field there (line 60-63, optional). The earlier plan's step that added it was wrong.

### Step 0: Write failing tests for `_is_valid_upstream_ref` (the shared validity rule)

- [ ] **Step 0.1: Write the failing test**

Append to `tests/unit/test_pipeline_upstream_refs.py`:

```python
from app.workers.pipeline import _is_valid_upstream_ref


class TestIsValidUpstreamRef:
    ONTOLOGY = {
        "entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
            {"name": "SPECIFICATION", "identity_fields": ["parameter", "value"], "identity_scope": "document"},
            # Some ontology entity types have no identity fields (e.g.
            # PROPULSION_STACK). They should not produce upstream refs.
            {"name": "PROPULSION_STACK", "identity_fields": [], "identity_scope": "global"},
        ],
    }

    def _ref(self, entity_type, identity_values):
        return SimpleNamespace(
            pass_origin="radar_domain",
            entity_type=entity_type,
            identity_values=identity_values,
            display_label="x",
        )

    def test_all_identity_fields_present_and_truthy_is_valid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": "Fan Song"}),
            self.ONTOLOGY,
        ) is True

    def test_unknown_entity_type_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("BOGUS", {"system_name": "X"}),
            self.ONTOLOGY,
        ) is False

    def test_missing_identity_key_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("SPECIFICATION", {"parameter": "range"}),  # missing value
            self.ONTOLOGY,
        ) is False

    def test_none_identity_value_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": None}),
            self.ONTOLOGY,
        ) is False

    def test_empty_string_identity_value_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": ""}),
            self.ONTOLOGY,
        ) is False

    def test_whitespace_identity_value_invalid(self):
        assert _is_valid_upstream_ref(
            self._ref("RADAR_SYSTEM", {"system_name": "   "}),
            self.ONTOLOGY,
        ) is False

    def test_all_fields_truthy_multifield_valid(self):
        assert _is_valid_upstream_ref(
            self._ref("SPECIFICATION", {"parameter": "range", "value": "150"}),
            self.ONTOLOGY,
        ) is True

    def test_zero_identity_fields_invalid(self):
        """Rule (b): an entity type with no identity anchors can't be a
        useful upstream ref — nothing to hand the LLM and nothing merge
        can resolve."""
        assert _is_valid_upstream_ref(
            self._ref("PROPULSION_STACK", {}),
            self.ONTOLOGY,
        ) is False
```

- [ ] **Step 0.2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py::TestIsValidUpstreamRef -v`
Expected: FAIL — `ImportError: cannot import name '_is_valid_upstream_ref'`.

- [ ] **Step 0.3: Implement the validity rule**

Add above `_extend_upstream_refs` in `app/workers/pipeline.py`:

```python
def _is_valid_upstream_ref(ref, ontology: dict) -> bool:
    """Single shared validity rule used at ref emission, request build, and
    merge attachment. A ref is valid iff:
      (a) its entity_type is in the ontology,
      (b) every ontology identity_field for that type is present as a key
          in identity_values,
      (c) every such value is truthy after ``str.strip()`` for strings
          (None, "", "   " all reject).

    Applied at three sites so invalid refs cannot leak into the request
    body, the prompt preamble, or ``PassResult.upstream_refs``. A ref
    that fails this check simply never existed as far as the rest of the
    pipeline is concerned — no UNKNOWN_REF_ID rejection, no polluted
    LogicalIdentity.
    """
    entity_type = getattr(ref, "entity_type", None)
    identity_values = getattr(ref, "identity_values", None) or {}
    entity_def = next(
        (e for e in ontology.get("entity_types", []) if e["name"] == entity_type),
        None,
    )
    if entity_def is None:
        return False
    identity_fields = list(entity_def.get("identity_fields") or ())
    if not identity_fields:
        # Rule (b): no anchors → not usable as an upstream ref.
        return False
    for field in identity_fields:
        if field not in identity_values:
            return False
        val = identity_values[field]
        if val is None:
            return False
        if isinstance(val, str) and not val.strip():
            return False
    return True
```

Also update `_extend_upstream_refs` (from Task 3) to skip invalid refs by wrapping the emission step:

```python
            ref = SimpleNamespace(
                pass_origin=pass_def.name,
                entity_type=entity_type,
                identity_values=identity_values,
                display_label=display_label,
            )
            if not _is_valid_upstream_ref(ref, ontology):
                continue  # Drop refs with missing/empty identity; see _is_valid_upstream_ref.
            upstream_refs[f"E{counter:03d}"] = ref
            counter += 1
```

- [ ] **Step 0.4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py::TestIsValidUpstreamRef -v`
Expected: all 7 PASS.

### Step 1: Write failing tests for `_select_upstream_refs_for_pass`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pipeline_upstream_refs.py`:

```python
from app.workers.pipeline import _select_upstream_refs_for_pass


class TestSelectUpstreamRefsForPass:

    def _ref(self, pass_origin, entity_type, identity_value):
        return SimpleNamespace(
            pass_origin=pass_origin,
            entity_type=entity_type,
            identity_values={"system_name": identity_value},
            display_label=identity_value,
        )

    ONTOLOGY = {
        "entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
            {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
            {"name": "IADS", "identity_fields": ["system_name"], "identity_scope": "global"},
        ],
    }

    def test_filters_by_depends_on(self):
        all_refs = {
            "E001": self._ref("radar_domain", "RADAR_SYSTEM", "Fan Song"),
            "E002": self._ref("missile_domain", "MISSILE_SYSTEM", "SA-2"),
            "E003": self._ref("other_systems", "IADS", "System-X"),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain", "missile_domain"],
            extracted_relationship_types=[],  # no rel-type narrowing
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY)
        assert set(selected.keys()) == {"E001", "E002"}

    def test_empty_depends_on_selects_nothing(self):
        all_refs = {"E001": self._ref("radar_domain", "RADAR_SYSTEM", "X")}
        pass_def = SimpleNamespace(
            name="reference", depends_on=[], extracted_relationship_types=[],
        )
        assert _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY) == {}

    def test_invalid_refs_are_dropped(self):
        """Shared validity rule: a ref whose identity values are None/empty is
        filtered out before the service ever sees it."""
        all_refs = {
            "E001": self._ref("radar_domain", "RADAR_SYSTEM", "Fan Song"),
            "E002": self._ref("radar_domain", "RADAR_SYSTEM", None),   # invalid
            "E003": self._ref("radar_domain", "RADAR_SYSTEM", ""),     # invalid
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain"],
            extracted_relationship_types=[],  # no narrowing → all valid refs pass
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY)
        assert set(selected.keys()) == {"E001"}

    def test_sort_uses_ontology_identity_field_order_not_alphabetical(self):
        """Multi-field identities must sort by ontology-declared order to
        match LogicalIdentity's canonical identity_tuple. Alphabetical
        dict-key order would put 'parameter' after 'value' in a field list
        declared as [parameter, value], which would make the prompt
        preamble and the merge identity disagree on the first value."""
        ontology = {
            "entity_types": [
                # Declared order matters: parameter first, value second.
                {"name": "SPECIFICATION",
                 "identity_fields": ["parameter", "value"],
                 "identity_scope": "document"},
            ],
            "validation_matrix": [],
        }
        refs = {
            "E002": SimpleNamespace(
                pass_origin="radar_domain", entity_type="SPECIFICATION",
                identity_values={"parameter": "B", "value": "1"},
                display_label="B=1",
            ),
            "E001": SimpleNamespace(
                pass_origin="radar_domain", entity_type="SPECIFICATION",
                identity_values={"parameter": "A", "value": "2"},
                display_label="A=2",
            ),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain"],
            extracted_relationship_types=[],
        )
        selected = _select_upstream_refs_for_pass(pass_def, refs, ontology)
        # Sorted by (pass_origin, entity_type, (parameter, value))
        # in ontology-declared order:
        #   ("radar_domain", "SPECIFICATION", ("A", "2")) → A=2
        #   ("radar_domain", "SPECIFICATION", ("B", "1")) → B=1
        assert [r.display_label for r in selected.values()] == ["A=2", "B=1"]

    def test_narrows_to_validation_matrix_endpoint_types(self):
        """system_links extracts ASSOCIATED_WITH / CUES, which only connect
        system-level entities (validation_matrix rows 1118-1217). Upstream
        refs of types that can't source OR target any of those relationships
        (e.g. GUIDANCE_METHOD from missile_domain) must be dropped here,
        before the LLM ever sees them."""
        ontology = {
            "entity_types": [
                {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
                {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"], "identity_scope": "global"},
                {"name": "GUIDANCE_METHOD", "identity_fields": ["name"], "identity_scope": "global"},
            ],
            "validation_matrix": [
                {"source": "RADAR_SYSTEM", "relationship": "ASSOCIATED_WITH",
                 "target": "MISSILE_SYSTEM"},
                {"source": "RADAR_SYSTEM", "relationship": "CUES",
                 "target": "MISSILE_SYSTEM"},
                # GUIDANCE_METHOD is not on either side of any ASSOCIATED_WITH/CUES row.
            ],
        }
        all_refs = {
            "E001": SimpleNamespace(
                pass_origin="radar_domain",
                entity_type="RADAR_SYSTEM",
                identity_values={"system_name": "Fan Song"},
                display_label="Fan Song",
            ),
            "E002": SimpleNamespace(
                pass_origin="missile_domain",
                entity_type="MISSILE_SYSTEM",
                identity_values={"system_name": "SA-2"},
                display_label="SA-2",
            ),
            "E003": SimpleNamespace(  # legally valid ref, but wrong type for system_links
                pass_origin="missile_domain",
                entity_type="GUIDANCE_METHOD",
                identity_values={"name": "Command"},
                display_label="Command",
            ),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain", "missile_domain"],
            extracted_relationship_types=["ASSOCIATED_WITH", "CUES"],
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, ontology)
        assert set(selected.keys()) == {"E001", "E002"}  # GUIDANCE_METHOD dropped

    def test_deterministic_order_under_shuffled_input(self):
        """Ordering is (pass_origin, entity_type, identity tuple) so
        repeat runs of the same extraction produce the same preamble."""
        all_refs = {
            "E005": self._ref("radar_domain", "RADAR_SYSTEM", "Zebra"),
            "E001": self._ref("radar_domain", "RADAR_SYSTEM", "Alpha"),
            "E003": self._ref("missile_domain", "MISSILE_SYSTEM", "Bravo"),
        }
        pass_def = SimpleNamespace(
            name="system_links",
            depends_on=["radar_domain", "missile_domain"],
            extracted_relationship_types=[],
        )
        selected = _select_upstream_refs_for_pass(pass_def, all_refs, self.ONTOLOGY)
        ordered = list(selected.values())
        # Sorted by (pass_origin, entity_type, identity tuple)
        # → missile_domain/MISSILE_SYSTEM/Bravo, radar_domain/RADAR_SYSTEM/Alpha,
        #   radar_domain/RADAR_SYSTEM/Zebra
        assert [r.display_label for r in ordered] == ["Bravo", "Alpha", "Zebra"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py::TestSelectUpstreamRefsForPass -v`
Expected: FAIL — `ImportError: cannot import name '_select_upstream_refs_for_pass'`.

- [ ] **Step 3: Implement `_select_upstream_refs_for_pass`**

**Narrow by validation_matrix, not just `depends_on`.** `system_links` declares `extracted_relationship_types: [ASSOCIATED_WITH, CUES]`. Those validation rows (`ontology.yaml:1118-1217`) only connect system-level entities (`RADAR_SYSTEM`, `MISSILE_SYSTEM`, `AIR_DEFENSE_ARTILLERY_SYSTEM`, `ELECTRONIC_WARFARE_SYSTEM`, `FIRE_CONTROL_SYSTEM`). Shipping `GUIDANCE_METHOD`, `SEEKER`, `PROPULSION_STACK`, etc. as upstream refs is noise that the LLM cannot legally connect. Filter to the set of types that appear as `source` or `target` for any of `pass_def.extracted_relationship_types`.

Add just below `_extend_upstream_refs` in `app/workers/pipeline.py`:

```python
def _endpoint_types_for_rel_types(
    ontology: dict, rel_types: list[str],
) -> set[str]:
    """Return the set of entity types that appear as source or target for
    any of the given relationship types in the ontology validation_matrix.

    Used by _select_upstream_refs_for_pass to drop upstream refs whose
    entity_type cannot legally participate in any relationship the
    downstream pass extracts. For system_links (ASSOCIATED_WITH, CUES)
    this resolves to the system-level entity types only."""
    if not rel_types:
        return set()
    wanted = set(rel_types)
    endpoint_types: set[str] = set()
    for row in ontology.get("validation_matrix", []):
        if row.get("relationship") in wanted:
            src = row.get("source")
            tgt = row.get("target")
            if src:
                endpoint_types.add(src)
            if tgt:
                endpoint_types.add(tgt)
    return endpoint_types


def _select_upstream_refs_for_pass(
    pass_def, upstream_refs: dict, ontology: dict,
) -> dict:
    """Filter upstream_refs so the downstream pass only sees refs it can
    legally use: (1) pass_origin in pass_def.depends_on, (2) the ref is
    valid (see _is_valid_upstream_ref), and (3) the ref's entity_type is
    a valid source or target for at least one of
    pass_def.extracted_relationship_types in the ontology validation_matrix.
    Returns a dict ordered by (pass_origin, entity_type, identity) so
    repeat runs produce the same preamble."""
    depends_on = set(getattr(pass_def, "depends_on", None) or [])
    if not depends_on:
        return {}

    rel_types = list(getattr(pass_def, "extracted_relationship_types", None) or [])
    endpoint_types = _endpoint_types_for_rel_types(ontology, rel_types)

    # Precompute the ontology-declared identity_fields order per type so
    # the sort key matches LogicalIdentity's canonical ordering
    # (extraction_merge.py:43-ish: identity_field_names comes straight
    # from entity_def["identity_fields"]). Sorting by sorted(dict.keys())
    # would diverge from that canonical order on any multi-field
    # identity, which means the LLM preamble and the merge identity
    # tuple could disagree on which value goes first.
    identity_fields_by_type = {
        e["name"]: tuple(e.get("identity_fields") or ())
        for e in ontology.get("entity_types", [])
    }

    eligible = []
    for ref_id, ref in upstream_refs.items():
        if getattr(ref, "pass_origin", None) not in depends_on:
            continue
        if not _is_valid_upstream_ref(ref, ontology):
            continue
        # When the downstream pass extracts relationships, the ref's type
        # must be legal for at least one of them. If the pass declares
        # no extracted_relationship_types, keep all depends_on refs.
        if endpoint_types and ref.entity_type not in endpoint_types:
            continue
        eligible.append((ref_id, ref))

    def _sort_key(item):
        _ref_id, ref = item
        identity_values = getattr(ref, "identity_values", {}) or {}
        # Use ontology-declared identity_fields order (same as
        # LogicalIdentity.identity_tuple), NOT sorted(dict.keys()).
        fields = identity_fields_by_type.get(ref.entity_type, ())
        identity_tuple = tuple(identity_values.get(k) for k in fields)
        return (ref.pass_origin, ref.entity_type, identity_tuple)

    eligible.sort(key=_sort_key)
    return {ref_id: ref for ref_id, ref in eligible}
```

- [ ] **Step 4: Verify the helper tests pass**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py::TestSelectUpstreamRefsForPass -v`
Expected: all 3 PASS.

### Step 2: Write failing test — request carries `document_id`

- [ ] **Step 5: Write the failing test**

Append to `tests/unit/test_pipeline_upstream_refs.py`:

```python
from app.workers.pipeline import _build_extract_pass_request


class TestBuildExtractPassRequest:

    def test_document_id_included_in_body(self):
        pass_def = SimpleNamespace(name="reference", primary_entity_types=[])
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=pass_def,
            doc_json={"stub": True},
            upstream_refs=None,
            document_id="doc-42",
        )
        assert body["document_id"] == "doc-42"

    def test_upstream_entities_carry_all_fields(self):
        ref = SimpleNamespace(
            pass_origin="radar_domain",
            entity_type="RADAR_SYSTEM",
            identity_values={"system_name": "Fan Song"},
            display_label="Fan Song",
        )
        pass_def = SimpleNamespace(name="system_links", primary_entity_types=[])
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=pass_def,
            doc_json={"stub": True},
            upstream_refs={"E001": ref},
            document_id="doc-42",
        )
        assert body["upstream_entities"] == [
            {
                "ref_id": "E001",
                "entity_type": "RADAR_SYSTEM",
                "identity_values": {"system_name": "Fan Song"},
                "display_label": "Fan Song",
            },
        ]
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py::TestBuildExtractPassRequest -v`
Expected: FAIL — `TypeError: _build_extract_pass_request() got an unexpected keyword argument 'document_id'`.

- [ ] **Step 7: Extend `_build_extract_pass_request`**

Modify at `app/workers/pipeline.py:1607-1626`:

```python
def _build_extract_pass_request(
    *, bundle_key: str, pass_def, doc_json: dict,
    upstream_refs: dict | None, document_id: str,
) -> dict:
    """Assemble the POST body for /extract-pass.

    document_id is always included so the service can log and attribute
    extraction runs to a specific document (useful when correlating
    salvage warnings and timeout retries across the batch).
    """
    body: dict = {
        "bundle_key": bundle_key,
        "pass_name": pass_def.name,
        "document_id": document_id,
        "docling_document_json": doc_json,
    }
    if upstream_refs:
        body["upstream_entities"] = [
            {
                "ref_id": ref_id,
                "entity_type": getattr(ref, "entity_type", None),
                "identity_values": getattr(ref, "identity_values", {}) or {},
                "display_label": getattr(ref, "display_label", None),
            }
            for ref_id, ref in upstream_refs.items()
        ]
    return body
```

**No service-schema edit required.** `docker/docling-graph/app/schemas.py:60-63` already declares `document_id: Optional[str]` on `ExtractPassRequest`, so the body field is accepted as-is.

Update the **one** live caller of `_build_extract_pass_request` — inside `_run_single_pass` at `pipeline.py:364` — to pass `document_id=...` (the helper's `document_id: str` parameter). A grep for `_build_extract_pass_request(` confirms there is only one call site today; no other callers need updating.

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py -v`
Expected: all PASS.
Run also: `cd docker/docling-graph && pytest tests/test_extract_pass_endpoint.py -v`
Expected: all PASS (new `document_id` field is optional and defaults to None).

### Step 3: Write failing test — `_run_single_pass` attaches `pass_result.upstream_refs`

- [ ] **Step 9: Write the failing test**

Append to `tests/unit/test_run_single_pass.py`. Use the existing local helpers `_fake_pass_def` and `_fake_manifest` (verified at `test_run_single_pass.py:16-49`) and the `TestRunSinglePass`-style patch pattern:

```python
class TestPassResultUpstreamRefsAttachment:
    """After Task 4, document_plus_entity_refs passes must have
    pass_result.upstream_refs populated with LogicalIdentity values so
    merge_and_resolve can match from_ref_id / to_ref_id. document_only
    passes must NOT have upstream_refs attached."""

    ONTOLOGY = {
        "entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"],
             "identity_scope": "global"},
            {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"],
             "identity_scope": "global"},
        ],
        "validation_matrix": [
            {"source": "RADAR_SYSTEM", "relationship": "ASSOCIATED_WITH",
             "target": "MISSILE_SYSTEM"},
        ],
    }

    def _fake_pass_result(self):
        return SimpleNamespace(
            pass_name="system_links",
            template_instance=SimpleNamespace(),
            metadata=SimpleNamespace(schema_size_chars=500,
                                     structured_output_mode="strict"),
            pre_merge_rejections=[],
            relationships=[],
            upstream_refs=None,  # _run_single_pass should populate this
        )

    def test_document_plus_entity_refs_pass_attaches_upstream_refs(self):
        """system_links is document_plus_entity_refs; after the parsed
        result returns, pass_result.upstream_refs must be a dict of
        LogicalIdentity objects keyed by ref_id (filtered through
        _select_upstream_refs_for_pass)."""
        from app.workers.pipeline import _run_single_pass
        from app.services.extraction_merge import LogicalIdentity

        system_links_def = _fake_pass_def(
            name="system_links",
            kind="relationships_only",
            input_mode="document_plus_entity_refs",
            required=True,
            depends_on=["radar_domain", "missile_domain"],
            primary=(),  # relationships_only — no primary entity output
            bridge=(),
            rels=("ASSOCIATED_WITH",),
        )
        manifest = _fake_manifest([system_links_def])
        pass_results: dict = {}

        # Upstream refs accumulated by earlier (mocked) passes.
        accumulated_refs = {
            "E001": SimpleNamespace(
                pass_origin="radar_domain",
                entity_type="RADAR_SYSTEM",
                identity_values={"system_name": "Fan Song"},
                display_label="Fan Song",
            ),
            "E002": SimpleNamespace(
                pass_origin="missile_domain",
                entity_type="MISSILE_SYSTEM",
                identity_values={"system_name": "SA-2"},
                display_label="SA-2",
            ),
        }

        fake_pass_result = self._fake_pass_result()

        with patch("app.workers.pipeline._call_extract_pass") as mock_call, \
             patch("app.workers.pipeline._parse_pass_response",
                   return_value=fake_pass_result), \
             patch("app.workers.pipeline._write_stage_run"), \
             patch("app.workers.pipeline._count_pass_output", return_value={
                 "primary_entities_extracted": 0,
                 "bridge_entities_extracted": 0,
                 "relationships_extracted": 0,
                 "relationships_rejected": 0,
                 "schema_size_chars": 500,
                 "structured_output_mode": "strict",
                 "salvaged": False,
             }), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            mock_call.return_value = {"pass_output": {}, "metadata": {}}
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=system_links_def,
                manifest=manifest,
                ontology=self.ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={},
                pass_results=pass_results,
                upstream_refs=accumulated_refs,
                document_id="doc-1",
            )

        # The pass was recorded.
        assert "system_links" in pass_results
        result = pass_results["system_links"]

        # Both refs attached as LogicalIdentity (both pass the shared
        # validity rule AND are valid endpoints for ASSOCIATED_WITH).
        assert result.upstream_refs is not None
        assert set(result.upstream_refs.keys()) == {"E001", "E002"}
        assert all(
            isinstance(v, LogicalIdentity)
            for v in result.upstream_refs.values()
        )
        assert result.upstream_refs["E001"].entity_type == "RADAR_SYSTEM"
        assert result.upstream_refs["E002"].entity_type == "MISSILE_SYSTEM"

    def test_document_only_pass_does_not_attach_upstream_refs(self):
        """radar_domain is document_only — pass_result.upstream_refs must
        remain None/absent even when accumulated upstream refs exist."""
        from app.workers.pipeline import _run_single_pass

        radar_def = _fake_pass_def(
            name="radar_domain",
            kind="entities_and_relationships",
            input_mode="document_only",
            primary=("RADAR_SYSTEM",),
            rels=("INSTALLED_ON",),
        )
        manifest = _fake_manifest([radar_def])
        pass_results: dict = {}

        accumulated_refs = {
            "E001": SimpleNamespace(
                pass_origin="reference",
                entity_type="SECTION",
                identity_values={"heading": "Intro"},
                display_label="Intro",
            ),
        }

        fake_pass_result = self._fake_pass_result()
        fake_pass_result.pass_name = "radar_domain"

        with patch("app.workers.pipeline._call_extract_pass") as mock_call, \
             patch("app.workers.pipeline._parse_pass_response",
                   return_value=fake_pass_result), \
             patch("app.workers.pipeline._write_stage_run"), \
             patch("app.workers.pipeline._count_pass_output", return_value={
                 "primary_entities_extracted": 0,
                 "bridge_entities_extracted": 0,
                 "relationships_extracted": 0,
                 "relationships_rejected": 0,
                 "schema_size_chars": 500,
                 "structured_output_mode": "strict",
                 "salvaged": False,
             }), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            mock_call.return_value = {"pass_output": {}, "metadata": {}}
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=radar_def,
                manifest=manifest,
                ontology=self.ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={},
                pass_results=pass_results,
                upstream_refs=accumulated_refs,
                document_id="doc-1",
            )

        result = pass_results["radar_domain"]
        # document_only passes do not consume upstream refs, so the post-
        # parse attach block must not populate this field.
        assert result.upstream_refs is None
```

(The fixture structure already used by `test_run_single_pass.py` is preserved — follow it; the behavioural asserts are the part that is new.)

- [ ] **Step 10: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_run_single_pass.py::TestPassResultUpstreamRefsAttachment -v`
Expected: FAIL — `pass_result.upstream_refs is None`.

- [ ] **Step 11: Attach `upstream_refs` in `_run_single_pass`**

In `app/workers/pipeline.py`, just after the call to `_parse_pass_response` at line 378, add:

```python
            pass_result = _parse_pass_response(response, pass_def, manifest)

            # Attach the filtered, ordered upstream refs AS LogicalIdentity objects
            # so merge_and_resolve can resolve from_ref_id / to_ref_id directly
            # (extraction_merge.py:384). Only document_plus_entity_refs passes use
            # this — document_only passes do not consume upstream refs.
            if pass_def.input_mode == "document_plus_entity_refs":
                from app.services.extraction_merge import logical_identity_from_dict
                # Use the SAME selection + validity filter that built the
                # request body, so the merge side sees exactly the refs the
                # LLM was told about. Invalid refs were already dropped by
                # _is_valid_upstream_ref inside _select_upstream_refs_for_pass.
                selected = _select_upstream_refs_for_pass(
                    pass_def, upstream_refs, ontology,
                )
                pass_result.upstream_refs = {}
                for ref_id, ref in selected.items():
                    identity = logical_identity_from_dict(
                        ref.entity_type,
                        ref.identity_values or {},
                        ontology,
                        document_id,
                    )
                    if identity is not None:
                        pass_result.upstream_refs[ref_id] = identity
```

Also update the `_build_extract_pass_request` call at line 364 to use the filtered selection AND pass `document_id`:

```python
            selected_refs = (
                _select_upstream_refs_for_pass(pass_def, upstream_refs, ontology)
                if pass_def.input_mode == "document_plus_entity_refs"
                else None
            )
            request_body = _build_extract_pass_request(
                bundle_key=bundle_key,
                pass_def=pass_def,
                doc_json=doc_json,
                upstream_refs=selected_refs,
                document_id=document_id,
            )
```

### Step 4: Write failing test — merge success path

- [ ] **Step 12: Write the failing test**

Append to `tests/unit/test_extraction_merge.py` (near the existing `test_rejection_unknown_ref_id`):

```python
def test_system_links_resolves_valid_ref_ids():
    """Plan 1 success path: when pass_result.upstream_refs is populated with
    real LogicalIdentity values AND those identities exist in the merged
    entity_index, system_links emits a merged edge (not UNKNOWN_REF_ID).

    Note: _make_pass_result's real signature (test_extraction_merge.py:53)
    takes entities as a list of (entity_type, identity_dict, properties_dict)
    tuples — not dicts. The local ontology below extends MINIMAL_ONTOLOGY
    with MISSILE_SYSTEM and an ASSOCIATED_WITH validation-matrix row; adjust
    the top-of-file MINIMAL_ONTOLOGY if those are missing.
    """
    from app.services.extraction_merge import LogicalIdentity

    local_ontology = {
        "entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"],
             "identity_scope": "global", "properties": ["system_name"]},
            {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"],
             "identity_scope": "global", "properties": ["system_name"]},
        ],
        "validation_matrix": [
            {"source": "RADAR_SYSTEM", "relationship": "ASSOCIATED_WITH",
             "target": "MISSILE_SYSTEM"},
        ],
    }

    radar_identity = LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name",),
        identity_tuple=("Fan Song",),
        scope="global",
        document_id=None,
    )
    missile_identity = LogicalIdentity(
        entity_type="MISSILE_SYSTEM",
        identity_field_names=("system_name",),
        identity_tuple=("SA-2",),
        scope="global",
        document_id=None,
    )

    # Pre-seed entity extractions so merge already has both endpoints.
    # _make_pass_result expects (entity_type, identity_dict, properties_dict).
    radar_pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[("RADAR_SYSTEM", {"system_name": "Fan Song"}, {})],
        relationships=[],
    )
    missile_pass_result = _make_pass_result(
        pass_name="missile_domain",
        entities=[("MISSILE_SYSTEM", {"system_name": "SA-2"}, {})],
        relationships=[],
    )
    # system_links: the relationship uses ref_ids; upstream_refs resolves them.
    links_pass_result = _make_pass_result(
        pass_name="system_links",
        entities=[],
        relationships=[{
            "rel_type": "ASSOCIATED_WITH",
            "from_ref_id": "E001",
            "to_ref_id": "E002",
            "confidence": 0.9,
        }],
    )
    links_pass_result.upstream_refs = {
        "E001": radar_identity,
        "E002": missile_identity,
    }

    merged = merge_and_resolve(
        pass_results={
            "radar_domain": radar_pass_result,
            "missile_domain": missile_pass_result,
            "system_links": links_pass_result,
        },
        manifest=_fake_manifest(["radar_domain", "missile_domain", "system_links"]),
        ontology=local_ontology,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )

    assert len(merged.rejected_edges) == 0
    # One ASSOCIATED_WITH edge between the two systems.
    # NOTE: MergedEdgeRecord field is `rel_type`, NOT `relationship_type`
    # (extraction_merge.py:150).
    assert any(
        e.rel_type == "ASSOCIATED_WITH"
        and e.from_identity == radar_identity
        and e.to_identity == missile_identity
        for e in merged.edges
    )
```

- [ ] **Step 13: Run tests to verify — passes if Task 4 wiring is correct**

Run: `.venv/bin/pytest tests/unit/test_extraction_merge.py::test_system_links_resolves_valid_ref_ids -v`
Expected: PASS (the merge code already handles this path; this test locks in that the full pipeline now supplies the data).

### Step 5: Make UNKNOWN_REF_ID measurable (post-merge authoritative path)

**Key facts verified against the code:**

- `RelationshipRejectionReason` values are **lowercase** (`extraction_merge.py:35-42`): `unknown_ref_id`, `invalid_triple`, `from_endpoint_not_found`, etc. Use those exact values — not uppercase.
- `UNKNOWN_REF_ID` is emitted by `_resolve_relationship` during `merge_and_resolve` (`extraction_merge.py:384`), **not** from `pass_result.pre_merge_rejections`. A pre-merge-only counter would always read zero for this reason.
- `_write_stage_run` (`pipeline.py:1528-1576`) whitelists a fixed set of `counts` keys and ignores everything else, including writing nothing to the `metrics` JSONB column. The existing shape does not surface extra keys.

**Plan — authoritative site is per-pass, not summary.** `_run_single_pass` writes per-pass rows BEFORE merge (`pipeline.py:327-440`), so they cannot contain UNKNOWN_REF_ID (which is generated during merge). The existing post-merge hook `_apply_post_merge_yield_updates` (`pipeline.py:477+`) already walks the per-pass `StageRun` rows and recomputes counts from `merged.rejected_edges` grouped by `source_pass` — extend it to also write `metrics["rejections_by_reason"]` onto each per-pass row. This makes queries like `WHERE pass_name='system_links'` work and keeps per-pass attribution intact. The summary row optionally gets the aggregate too, but per-pass is the source of truth for the success criterion.

**Two input tuple shapes exist in the codebase and the helper must accept both:**

- `pass_result.pre_merge_rejections` is `list[tuple[rel, reason]]` (2-tuples).
- `MergedExtraction.rejected_edges` is `list[tuple[source_pass, raw_rel, reason]]` (3-tuples — verified at `extraction_merge.py:159`).

The helper treats the **last element** of each tuple as the reason, which works cleanly for both shapes without a separate pre/post helper. A separate helper at each call site would create awkward conditionals for a five-line counting loop.

- [ ] **Step 14a: Write the failing tests**

Append to `tests/unit/test_pipeline_upstream_refs.py`:

```python
def test_build_rejections_by_reason_uses_lowercase_enum_values():
    """RelationshipRejectionReason values are lowercase in the enum
    (extraction_merge.py:35-42). The persisted key MUST match so
    downstream queries don't need to case-normalise."""
    from app.workers.pipeline import _build_rejections_by_reason
    from app.services.extraction_merge import RelationshipRejectionReason

    result = _build_rejections_by_reason([
        (object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        (object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        (object(), RelationshipRejectionReason.INVALID_TRIPLE),
    ])
    assert result == {
        "unknown_ref_id": 2,
        "invalid_triple": 1,
    }


def test_build_rejections_by_reason_accepts_pre_merge_tuples():
    """pass_result.pre_merge_rejections shape: (rel, reason)."""
    from app.workers.pipeline import _build_rejections_by_reason
    from app.services.extraction_merge import RelationshipRejectionReason
    result = _build_rejections_by_reason([
        (object(), RelationshipRejectionReason.MISSING_REL_TYPE),
    ])
    assert result == {"missing_rel_type": 1}


def test_build_rejections_by_reason_accepts_merged_rejected_edges_tuples():
    """MergedExtraction.rejected_edges shape: (source_pass, raw_rel, reason)
    (extraction_merge.py:159). The helper must handle this 3-tuple shape
    alongside the 2-tuple pre_merge_rejections shape without a separate
    caller-side conditional."""
    from app.workers.pipeline import _build_rejections_by_reason
    from app.services.extraction_merge import RelationshipRejectionReason
    result = _build_rejections_by_reason([
        ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
        ("radar_domain", object(), RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND),
    ])
    assert result == {
        "unknown_ref_id": 2,
        "from_endpoint_not_found": 1,
    }


def test_build_rejections_by_reason_empty_list_returns_empty_dict():
    from app.workers.pipeline import _build_rejections_by_reason
    assert _build_rejections_by_reason([]) == {}
    assert _build_rejections_by_reason(None) == {}
```

And a targeted test that the metrics actually land in `StageRun.metrics` (mock the DB session; follow the existing `_write_stage_run` test pattern in this file if one exists, otherwise a freshly-mocked session):

```python
def test_write_stage_run_persists_metrics_dict_into_jsonb(monkeypatch):
    """When counts includes a 'metrics' key with rejections_by_reason,
    _write_stage_run MUST include that dict in the values dict it passes
    to pg_insert(...).values(...), so it lands in the StageRun.metrics
    JSONB column.

    Asserting against the compiled SQL is brittle — JSONB/UUID rendering
    depends on the dialect and SA version. Instead, intercept the
    Insert statement before execute() and inspect the .values() dict
    directly.
    """
    from app.workers import pipeline as _pipeline
    from sqlalchemy.dialects.postgresql import Insert as _PgInsert

    captured = {}

    # The real _write_stage_run chains:
    #   pg_insert(StageRun).values(**values).on_conflict_do_update(...)
    # We wrap .values() so we can snapshot the dict that was passed in.
    orig_values = _PgInsert.values

    def _spy_values(self, **kwargs):
        captured.setdefault("values_calls", []).append(kwargs)
        return orig_values(self, **kwargs)

    monkeypatch.setattr(_PgInsert, "values", _spy_values)

    class _FakeDB:
        def execute(self, stmt): pass
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr(_pipeline, "_get_db", lambda: _FakeDB())
    _pipeline._write_stage_run(
        pipeline_run_id="00000000-0000-0000-0000-000000000000",
        pass_def=SimpleNamespace(name="system_links"),
        attempt=1,
        execution_status="COMPLETE",
        yield_status="HIT",
        skip_reason=None,
        counts={
            "primary_entities_extracted": 0,
            "relationships_extracted": 1,
            "relationships_rejected": 2,
            "metrics": {"rejections_by_reason": {"unknown_ref_id": 2}},
        },
        error=None,
    )

    # At least one .values() call must carry the metrics dict.
    assert captured["values_calls"], "pg_insert(...).values(...) was never called"
    metrics_values = [
        v["metrics"] for v in captured["values_calls"] if "metrics" in v
    ]
    assert metrics_values, "metrics key was not forwarded into the Insert values dict"
    assert metrics_values[-1] == {"rejections_by_reason": {"unknown_ref_id": 2}}
```

- [ ] **Step 14b: Run to verify they fail**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py::test_build_rejections_by_reason_uses_lowercase_enum_values tests/unit/test_pipeline_upstream_refs.py::test_write_stage_run_persists_metrics_dict_into_jsonb -v`
Expected: FAIL — `ImportError: cannot import name '_build_rejections_by_reason'` and/or the metrics key is not in the rendered SQL.

- [ ] **Step 14c: Add the helper + extend `_write_stage_run`**

Add a small pure helper near `_extend_upstream_refs` in `app/workers/pipeline.py`:

```python
def _build_rejections_by_reason(
    rejections: list | None,
) -> dict[str, int]:
    """Bucket rejection tuples by the reason enum's ``.value``
    (lowercase, e.g. 'unknown_ref_id'). Accepts both tuple shapes:

    * ``(rel, reason)`` — ``pass_result.pre_merge_rejections``
    * ``(source_pass, raw_rel, reason)`` — ``MergedExtraction.rejected_edges``
      (see extraction_merge.py:159)

    The helper treats the **last** element of each tuple as the reason,
    which works for both shapes without the caller needing a conditional.
    Used to persist per-reason counts into ``StageRun.metrics`` JSONB so
    UNKNOWN_REF_ID trends are queryable from the DB without reprocessing
    passes."""
    result: dict[str, int] = {}
    for tup in rejections or []:
        if not tup:
            continue
        reason = tup[-1]
        key = reason.value if hasattr(reason, "value") else str(reason)
        result[key] = result.get(key, 0) + 1
    return result
```

Extend `_write_stage_run` (`pipeline.py:1528-1576`) to forward an opaque `metrics` dict into the `StageRun.metrics` JSONB column — still a fixed contract, but one the caller can populate with `rejections_by_reason` (or any future observability key) without widening the whitelist every time:

```python
    if counts:
        values.update({
            "primary_entities_extracted": counts.get("primary_entities_extracted"),
            "bridge_entities_extracted": counts.get("bridge_entities_extracted"),
            "relationships_extracted": counts.get("relationships_extracted"),
            "relationships_rejected": counts.get("relationships_rejected"),
            "schema_size_chars": counts.get("schema_size_chars"),
            "structured_output_mode": counts.get("structured_output_mode"),
            "salvaged": counts.get("salvaged"),
        })
        if counts.get("metrics"):
            values["metrics"] = counts["metrics"]
```

Now wire it into two sites — **per-pass is authoritative**, summary is optional aggregate.

1. **Per-pass pre-merge counts (inside `_run_single_pass`).** After `pass_result` is built and before the `_write_stage_run(... execution_status="COMPLETE" ...)` call, seed the metrics dict with pre-merge rejections (if any). These are intentionally overwritten in step 2 once merge has run.

   ```python
   counts = _count_pass_output(pass_result, pass_def, ontology)
   counts["metrics"] = {
       "rejections_by_reason": _build_rejections_by_reason(
           getattr(pass_result, "pre_merge_rejections", None),
       ),
   }
   _write_stage_run(..., counts=counts, ...)
   ```

2. **Per-pass post-merge counts (inside `_apply_post_merge_yield_updates`).** `MergedExtraction.rejected_edges` is `list[tuple[source_pass, raw_rel, reason]]` (`extraction_merge.py:159`), so grouping by `source_pass` gives a per-pass breakdown. The function already walks `StageRun` rows filtered to `pass_name.isnot(None)` and updates count columns — extend it to also set `row.metrics["rejections_by_reason"]` for each row.

   Find the loop at `pipeline.py:511` and extend it:

   ```python
   # Group merged.rejected_edges by source_pass for per-pass metrics.
   rejections_by_pass: dict[str, list] = {}
   for tup in merged.rejected_edges:
       source_pass = tup[0]
       rejections_by_pass.setdefault(source_pass, []).append(tup)

   for row in rows:
       pass_name = row.pass_name
       extracted = extracted_by_pass.get(pass_name, 0)
       rejected = rejected_by_pass.get(pass_name, 0)
       # ... existing count updates ...

       # Plan 1: per-pass post-merge rejection breakdown into metrics JSONB.
       # Merge with any pre-merge counts already written in step 1 — post-merge
       # wins on conflict because it is authoritative for resolve-stage reasons
       # like UNKNOWN_REF_ID.
       post_merge = _build_rejections_by_reason(rejections_by_pass.get(pass_name, []))
       merged_metrics = dict(row.metrics or {})
       existing = dict(merged_metrics.get("rejections_by_reason") or {})
       existing.update(post_merge)  # post-merge values take precedence
       merged_metrics["rejections_by_reason"] = existing
       row.metrics = merged_metrics
   ```

   Because `_apply_post_merge_yield_updates` already runs on the success path inside `_derive_ontology_graph_bundle_passes`, no new call site is needed.

3. **(Optional) Summary-row aggregate (inside `_derive_ontology_graph_bundle_passes`).** If a batch-level single number is useful for dashboards, also write `row.metrics["rejections_by_reason"] = _build_rejections_by_reason(merged.rejected_edges)` onto the summary `StageRun` row (`pass_name IS NULL`) constructed at `pipeline.py:3602-3616`. This is a nice-to-have — the per-pass query below does not depend on it.

- [ ] **Step 14d: Verify the tests pass**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py -v`
Expected: all PASS.

Add a test asserting `_apply_post_merge_yield_updates` writes the per-pass metrics:

```python
def test_apply_post_merge_yield_updates_writes_rejections_by_reason_per_pass(monkeypatch):
    """_apply_post_merge_yield_updates is the single post-merge hook that
    touches per-pass StageRun rows (pipeline.py:477+). It must also write
    rejections_by_reason into row.metrics so the per-pass success query
    in Task 6 finds unknown_ref_id counts.

    The helper also reads row.yield_status, row.primary_entities_extracted,
    and row.bridge_entities_extracted (pipeline.py:521-527) when
    recomputing the HIT → DEGRADED transition. Fake rows must supply those
    attributes, otherwise the test fails on AttributeError before it
    checks the new metrics write."""
    from app.workers import pipeline as _pipeline
    from app.services.extraction_merge import (
        MergedExtraction, RelationshipRejectionReason,
    )

    def _fake_row(pass_name, *, metrics=None, yield_status="HIT",
                  primary=0, bridge=0):
        return SimpleNamespace(
            pass_name=pass_name,
            metrics=metrics,
            yield_status=yield_status,
            primary_entities_extracted=primary,
            bridge_entities_extracted=bridge,
            relationships_extracted=0,
            relationships_rejected=0,
        )

    # Fake StageRun rows the post-merge hook should update.
    row_links = _fake_row("system_links", metrics=None, yield_status="HIT",
                          primary=0, bridge=0)
    row_radar = _fake_row("radar_domain", metrics={"preexisting": 1},
                          yield_status="HIT", primary=2, bridge=0)

    # Fake session returning those rows. Matches the `with get_sync_session()
    # as session:` context-manager shape used at pipeline.py:499.
    class _FakeQuery:
        def filter(self, *a, **k): return self
        def all(self): return [row_links, row_radar]
    class _FakeSession:
        def query(self, _cls): return _FakeQuery()
        def commit(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(_pipeline, "get_sync_session", lambda: _FakeSession())

    merged = MergedExtraction(
        entities=[], edges=[],
        rejected_edges=[
            ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
            ("system_links", object(), RelationshipRejectionReason.UNKNOWN_REF_ID),
            ("radar_domain", object(), RelationshipRejectionReason.INVALID_TRIPLE),
        ],
        rejections_by_pass={"system_links": 2, "radar_domain": 1},
        pipeline_run_id="run-1", document_id="doc-1",
    )
    _pipeline._apply_post_merge_yield_updates("run-1", merged)

    assert row_links.metrics["rejections_by_reason"] == {"unknown_ref_id": 2}
    # Pre-existing metrics keys survive; rejections_by_reason is added.
    assert row_radar.metrics["preexisting"] == 1
    assert row_radar.metrics["rejections_by_reason"] == {"invalid_triple": 1}
    # Sanity: the existing relationship-count update still fires.
    assert row_links.relationships_rejected == 2
    assert row_radar.relationships_rejected == 1
```

**Query convention** for success criteria in Task 6 — queries **per-pass rows**, not the summary:

```sql
SELECT sr.pass_name,
       sr.metrics->'rejections_by_reason'->>'unknown_ref_id' AS unknown_ref_id_count,
       sr.finished_at
FROM ingest.stage_runs sr
WHERE sr.stage_name = 'derive_ontology_graph'
  AND sr.pass_name = 'system_links'
ORDER BY sr.finished_at DESC LIMIT 20;
```

### Step 6: Run full affected suite

- [ ] **Step 14: Run full affected suite**

Run: `.venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py tests/unit/test_run_single_pass.py tests/unit/test_extraction_merge.py tests/unit/test_derive_ontology_graph_bundle_passes.py -v`
Expected: all PASS.

- [ ] **Step 15: Commit**

```bash
git add app/workers/pipeline.py app/services/extraction_merge.py docker/docling-graph/app/schemas.py tests/unit/test_pipeline_upstream_refs.py tests/unit/test_run_single_pass.py tests/unit/test_extraction_merge.py
git commit -m "feat(pipeline): attach PassResult.upstream_refs for document_plus_entity_refs passes

Wires up three previously-missing pieces of the upstream-ref chain:

1. _select_upstream_refs_for_pass filters refs by pass_def.depends_on and
   returns them in deterministic (pass_origin, entity_type, identity)
   order.
2. _build_extract_pass_request now carries document_id and the filtered
   ref selection. ExtractPassRequest schema accepts document_id
   optionally.
3. _run_single_pass attaches pass_result.upstream_refs as a dict of
   LogicalIdentity objects (built via the public
   logical_identity_from_dict helper) so merge_and_resolve can match
   from_ref_id / to_ref_id directly — UNKNOWN_REF_ID rejections will
   stop firing on legitimate cross-pass refs."
```

---

## Task 5a: DISCOVERY — locate the prompt-injection hook in the shipped `docling_graph` library

**Why separate:** `docker/docling-graph/app/config_builder.py:81-84` only passes `source` and `template_class` to `PipelineConfig(...)`. `PipelineConfig` is a Pydantic `BaseModel` in `docker/docling-graph/repo/docling_graph/config.py:102`. There is no `extra_prompt_preamble=` field today, and `run_extraction_pass` explicitly says `"NOT threaded into docling_graph.run_pipeline in PR 1"`. Before writing code for Task 5b we need to KNOW which hook the shipped library actually exposes.

**Method:** run this inside the deployed `docling-graph` container and record findings on a scratch doc in the plan's PR body:

```bash
docker compose exec -T docling-graph python3 - <<'PY'
from docling_graph import PipelineConfig
import inspect
# 1. Public config surface
print("PipelineConfig fields:", sorted(PipelineConfig.model_fields.keys()))
# 2. Any "prompt" or "preamble" hook in the runtime call
from docling_graph import run_pipeline
print(inspect.signature(run_pipeline))
# 3. Prompt assembly files (already grep'd once at plan-writing time)
import pathlib, os
root = pathlib.Path(os.environ.get('DG_REPO', '/usr/local/lib/python3.11/site-packages/docling_graph'))
for path in sorted(root.glob("core/extractors/contracts/*/prompts.py")):
    print("---", path)
    print(path.read_text()[:600])
PY
```

**Expected outcomes and the decision rule:**

| Discovery outcome | Chosen implementation in Task 5b |
|---|---|
| `PipelineConfig` has a prompt-override field (the exact name is part of the discovery output), OR `run_pipeline` accepts a prompt-override kwarg | Thread preamble through that documented hook by widening `build_pipeline_config(source, template_class, extra_prompt_preamble=None)` to forward it — keep the public parameter name `extra_prompt_preamble` regardless of what the underlying library calls it. Lowest risk. |
| No hook exists but the library reads the document body verbatim into the user prompt | **Fallback path (default):** in `run_extraction_pass`, prepend the preamble text to `docling_document_json` — either as a synthetic top-of-body text segment or as a `"_upstream_entities_preamble"` metadata key the system prompt in the repo's `delta/prompts.py` already interpolates. Verify which by reading the prompt templates. |
| Neither hook nor document-body injection is viable | Document the blocker and STOP. Do not land a monkey-patch; surface the finding and decide with the team whether to vendor a docling-graph fork. |

- [ ] **Step A: Run the discovery snippet and record output**
- [ ] **Step B: Match output to the table above and choose path A / B / C**
- [ ] **Step C: Record the decision in a follow-up comment on the plan PR before writing Task 5b tests**

Output of Task 5a is a one-paragraph finding + the exact library symbol/path to be used. **No production code changes in this task.**

---

## Task 5b: Inject upstream-entity preamble using the chosen hook

**Prerequisite:** Task 5a decision recorded. The file set depends on which path Task 5a chose.

**Decision: Path A widens `build_pipeline_config` — it does NOT bypass it.** Path A's intent is to route the preamble through the central config builder (`docker/docling-graph/app/config_builder.py:81`) by adding an optional `extra_prompt_preamble: str | None = None` parameter that forwards to whichever `PipelineConfig(...)` kwarg Task 5a identified. Bypassing `build_pipeline_config` would fork service configuration and lose env-var-driven settings — not acceptable. If Task 5a discovers that `PipelineConfig` has no suitable kwarg at all, that's a Path-C outcome (STOP and surface to team), NOT a reason to bypass the helper.

**Files:**

| Path | Files to touch |
|---|---|
| **A (library hook)** | `docker/docling-graph/app/main.py` (`run_extraction_pass` + `extract_pass` handler + `_render_upstream_entities_preamble`; **keep `run_pipeline` import local**); `docker/docling-graph/app/config_builder.py:81` (add `extra_prompt_preamble` parameter, forward to `PipelineConfig`); `docker-compose.yml` (env flag); preamble + endpoint test files. |
| **B (body prepend fallback)** | Same as Path A **minus** `config_builder.py`. The body mutation lives in `run_extraction_pass`; `build_pipeline_config`'s signature does not change. |

- Modify: `docker/docling-graph/app/main.py:324-356` (`run_extraction_pass`), `:386-470` (`extract_pass` handler). **Keep `from docling_graph import run_pipeline` LOCAL** inside `run_extraction_pass` — do not hoist to module scope (breaks the test shim on host envs without `docling_graph` installed; see `_validate_library_surface` at `main.py:292` for the host-safe pattern the service preserves).
- Modify: `docker/docling-graph/app/config_builder.py:81-84` (**Path A only**)
- Modify: `docker-compose.yml` (docling-graph service env)
- Test: `docker/docling-graph/tests/test_upstream_entities_preamble.py` (new), `docker/docling-graph/tests/test_extract_pass_endpoint.py` (extend)

### Step 0: Move the service-app shim + fixture into `conftest.py` (enables fixture reuse)

`test_extract_pass_endpoint.py:13-70` currently owns `_ensure_dg_app_package()` and the `dg_app_module` fixture. Pytest does **not** inject fixtures from one test module into another, so `test_upstream_entities_preamble.py` can't just request `dg_app_module`. Move the shim + fixture into `docker/docling-graph/tests/conftest.py` so every test file in that directory shares them.

- [ ] **Step 0a: Cut from `test_extract_pass_endpoint.py`, paste into `conftest.py`**

Move the following block from `test_extract_pass_endpoint.py` (lines 13-70, adjust to actual line numbers) into `docker/docling-graph/tests/conftest.py`, keeping the append-path append that already lives there:

```python
# Add below the existing sys.path.append stanza:

_DG_MODULE_NAME = "docling_graph_service_main"
_DG_SERVICE_ROOT = Path(__file__).resolve().parent.parent


def _ensure_dg_app_package() -> None:
    """Ensure the docling-graph `app.*` sub-modules are importable as `app.*`.

    When the combined test suite runs from repo root, the repo-root `app/`
    package is already in sys.modules['app']. We temporarily swap it out
    for the docling-graph `app/` package so that `from app.config_builder
    import ...` in main.py resolves to the docling-graph package.

    This is called once; subsequent calls are no-ops.
    """
    import importlib
    import importlib.util

    if _DG_MODULE_NAME in sys.modules:
        return

    service_root = _DG_SERVICE_ROOT
    dg_app_path = service_root / "app"

    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    saved_path = list(sys.path)

    sys.path.insert(0, str(service_root))
    for key in list(saved.keys()):
        del sys.modules[key]

    try:
        spec = importlib.util.spec_from_file_location(
            _DG_MODULE_NAME,
            service_root / "app" / "main.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_DG_MODULE_NAME] = mod
        spec.loader.exec_module(mod)
    finally:
        for key in list(sys.modules.keys()):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = saved_path


@pytest.fixture(scope="module")
def dg_app_module():
    """Load the docling-graph app module once per test module."""
    _ensure_dg_app_package()
    return sys.modules[_DG_MODULE_NAME]
```

Add `import pytest` at the top of `conftest.py` if not already present.

- [ ] **Step 0b: Update `test_extract_pass_endpoint.py` — remove ONLY the fixture + shim function, keep the module-name constants local**

Delete **only** `_ensure_dg_app_package()` and the `dg_app_module` fixture from `test_extract_pass_endpoint.py`. **Keep `_DG_MODULE_NAME` and `_DG_SERVICE_ROOT` exactly where they are.** Those constants are consumed by in-file `patch(f"{_DG_MODULE_NAME}.run_extraction_pass")` call sites (the reviewer flagged the existing ones around lines 165 and 181; the new tests added later in Task 5b also use the constant). Removing them would turn every such `patch(...)` into a `NameError`. Fixtures flow through conftest automatically, but plain module-level globals do not — they have to stay visible to the code that names them.

The duplication is negligible (`_DG_MODULE_NAME = "docling_graph_service_main"` — a single string literal) and keeps each file self-contained. The conftest copy exists so `_ensure_dg_app_package()` knows which name to register in `sys.modules`; the test-file copy exists so `patch(f"{_DG_MODULE_NAME}...")` resolves at collection time. The two constants must hold the same string — if you change one, change both.

The `client` fixture in `test_extract_pass_endpoint.py` still takes `dg_app_module` as a parameter and continues to work because conftest fixtures are auto-discovered. No other test body changes.

- [ ] **Step 0c: Run the existing endpoint tests to prove no regression**

Run: `cd docker/docling-graph && pytest tests/test_extract_pass_endpoint.py -v`
Expected: all existing tests still PASS. If any fail with `fixture 'dg_app_module' not found`, the move wasn't completed or `import pytest` is missing in `conftest.py`.

### Step 1: Write failing tests for the preamble builder

- [ ] **Step 1: Write the failing test**

Create `docker/docling-graph/tests/test_upstream_entities_preamble.py`.

**Import discipline** — `docker/docling-graph/tests/conftest.py:13` *appends* the service root to `sys.path`, which means the repo-root `app/` package resolves FIRST. `from app.schemas import EntityRef` would therefore hit the repo-root `app` (which has no `EntityRef`), not the service's. Use the `dg_app_module` fixture now defined in `conftest.py` (Step 0) and pull symbols off the loaded module:

```python
"""Pure-formatting tests for _render_upstream_entities_preamble().

No network, no model call — lock down exact string shape + deterministic
ordering so ref-id / display-label regressions surface immediately.

Inputs are `EntityRef` (Pydantic) because that is what the runtime hands
the helper (see main.py:extract_pass). Using dicts here would hide
production bugs.

Import discipline (conftest.py:13 appends service root, so the repo-root
`app/` package resolves first): this file uses the dg_app_module fixture
defined in test_extract_pass_endpoint.py. No direct `from app.main import
...` or `from app.schemas import ...` — those would resolve to the wrong
package under the combined suite.
"""
import pytest


def _ent(dg_app_module, ref_id, entity_type, display_label=None):
    EntityRef = dg_app_module.EntityRef
    return EntityRef(
        ref_id=ref_id,
        entity_type=entity_type,
        identity_values={},
        display_label=display_label,
    )


def test_empty_list_returns_empty_string(dg_app_module):
    assert dg_app_module._render_upstream_entities_preamble([]) == ""


def test_none_returns_empty_string(dg_app_module):
    assert dg_app_module._render_upstream_entities_preamble(None) == ""


def test_single_entity_shape(dg_app_module):
    text = dg_app_module._render_upstream_entities_preamble([
        _ent(dg_app_module, "E001", "RADAR_SYSTEM", "Fan Song"),
    ])
    assert "Upstream entities:" in text
    assert "[E001] RADAR_SYSTEM — Fan Song" in text
    assert "Only emit from_ref_id and to_ref_id values from the list above" in text


def test_multiple_entities_preserve_input_order(dg_app_module):
    """Caller (worker) supplies the order; the preamble renders it verbatim
    so _select_upstream_refs_for_pass's deterministic sort determines the
    on-wire order."""
    text = dg_app_module._render_upstream_entities_preamble([
        _ent(dg_app_module, "E002", "MISSILE_SYSTEM", "SA-2"),
        _ent(dg_app_module, "E001", "RADAR_SYSTEM", "Fan Song"),
    ])
    e2_idx = text.index("[E002] MISSILE_SYSTEM")
    e1_idx = text.index("[E001] RADAR_SYSTEM")
    assert e2_idx < e1_idx


def test_missing_display_label_falls_back_to_entity_type(dg_app_module):
    text = dg_app_module._render_upstream_entities_preamble([
        _ent(dg_app_module, "E001", "RADAR_SYSTEM", None),
    ])
    assert "[E001] RADAR_SYSTEM" in text
    assert "—" not in text  # no trailing em-dash when label is absent


def test_env_flag_disables_preamble(dg_app_module, monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "false")
    text = dg_app_module._render_upstream_entities_preamble([
        _ent(dg_app_module, "E001", "RADAR_SYSTEM", "Fan Song"),
    ])
    assert text == ""
```

If you prefer the helpers next to `EntityRef` in test_extract_pass_endpoint.py, another valid option is to fold these cases into that file — both files then share one import boundary (`dg_app_module`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd docker/docling-graph && pytest tests/test_upstream_entities_preamble.py -v`
Expected: FAIL — `ImportError: cannot import name '_render_upstream_entities_preamble'`.

- [ ] **Step 3: Implement the preamble builder**

In `docker/docling-graph/app/main.py`, add near the top of the file (after imports):

```python
_UPSTREAM_PREAMBLE_ENABLED_ENV = "DOCLING_GRAPH_UPSTREAM_PREAMBLE"


def _render_upstream_entities_preamble(
    upstream_entities: "list[EntityRef] | None",
) -> str:
    """Render upstream refs as a compact prompt preamble.

    Output shape:

        Upstream entities:
          [E001] RADAR_SYSTEM — Fan Song
          [E002] MISSILE_SYSTEM — SA-2

        Only emit from_ref_id and to_ref_id values from the list above …

    Input is ``list[EntityRef]`` (Pydantic) — the runtime shape from
    ``ExtractPassRequest.upstream_entities``. Accessing fields via
    attribute (``ent.ref_id``) keeps the helper honest against the
    runtime contract; test coverage uses real ``EntityRef`` instances.

    Returns an empty string when there are no entities, or when the
    env-flag ``DOCLING_GRAPH_UPSTREAM_PREAMBLE`` is set to ``false``.
    Order is preserved verbatim — callers supply a stable order (the
    worker sorts by pass_origin, entity_type, identity tuple).
    """
    if os.environ.get(_UPSTREAM_PREAMBLE_ENABLED_ENV, "true").lower() in (
        "false", "0", "no",
    ):
        return ""
    if not upstream_entities:
        return ""

    lines = ["Upstream entities:"]
    for ent in upstream_entities:
        ref_id = ent.ref_id
        entity_type = ent.entity_type
        display = ent.display_label
        if display:
            lines.append(f"  [{ref_id}] {entity_type} — {display}")
        else:
            lines.append(f"  [{ref_id}] {entity_type}")
    lines.append("")
    lines.append(
        "Only emit from_ref_id and to_ref_id values from the list above. "
        "Do not invent new ref_ids."
    )
    return "\n".join(lines)
```

Add the import near the top of `docker/docling-graph/app/main.py`:

```python
from app.schemas import EntityRef  # for typing only; runtime access is by attribute
```

- [ ] **Step 4: Run preamble tests to verify they pass**

Run: `cd docker/docling-graph && pytest tests/test_upstream_entities_preamble.py -v`
Expected: all 6 PASS.

### Step 2: Thread preamble into `run_extraction_pass`

- [ ] **Step 5: Write failing integration test**

Append to `docker/docling-graph/tests/test_extract_pass_endpoint.py`. Use `_mock_run_pipeline_return(preamble_applied=...)` — the mock now accepts an explicit flag because `run_extraction_pass` attaches `_upstream_preamble_applied` to the context; in these tests we patch `run_extraction_pass` itself so the test controls the flag directly:

```python
def test_preamble_applied_flag_flips_true_for_document_plus_entity_refs(client, monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "true")
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        # Simulate Task 5b's behaviour: when real code runs the preamble,
        # it attaches _upstream_preamble_applied=True. The mock models that.
        mock_run.return_value = _mock_run_pipeline_return(preamble_applied=True)
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "system_links",
            "docling_document_json": {"name": "test"},
            "upstream_entities": [
                {"ref_id": "E001", "entity_type": "RADAR_SYSTEM",
                 "identity_values": {"system_name": "Fan Song"},
                 "display_label": "Fan Song"},
            ],
        })
    assert resp.status_code == 200, resp.text
    assert resp.json()["metadata"]["upstream_preamble_applied"] is True


def test_preamble_applied_flag_false_for_document_only(client):
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return(preamble_applied=False)
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "reference",
            "docling_document_json": {"name": "test"},
        })
    assert resp.status_code == 200
    assert resp.json()["metadata"]["upstream_preamble_applied"] is False


def test_preamble_applied_flag_false_when_env_disables(client, monkeypatch):
    """Direct unit coverage of the env-flag path is in
    test_upstream_entities_preamble.py; this test locks in that when
    run_extraction_pass returns preamble_applied=False (which it will
    when the env flag is off), the metadata reflects that."""
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "false")
    with patch(f"{_DG_MODULE_NAME}.run_extraction_pass") as mock_run:
        mock_run.return_value = _mock_run_pipeline_return(preamble_applied=False)
        resp = client.post("/extract-pass", json={
            "bundle_key": "air_defense_v3",
            "pass_name": "system_links",
            "docling_document_json": {"name": "test"},
            "upstream_entities": [
                {"ref_id": "E001", "entity_type": "RADAR_SYSTEM",
                 "identity_values": {"system_name": "X"}, "display_label": "X"},
            ],
        })
    assert resp.status_code == 200
    assert resp.json()["metadata"]["upstream_preamble_applied"] is False
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd docker/docling-graph && pytest tests/test_extract_pass_endpoint.py::test_preamble_applied_flag_flips_true_for_document_plus_entity_refs -v`
Expected: FAIL — flag still False because preamble isn't wired yet.

- [ ] **Step 7: Inject preamble using the Task 5a hook**

The *mechanism* of injection is whichever hook Task 5a identified and recorded on the PR. What this step specifies is the contract Task 5b's code must meet regardless of which hook wins:

1. **Function signature** — `run_extraction_pass` returns `context` as before. Do NOT change to a tuple; existing mocks (`_mock_run_pipeline_return()`) cannot absorb a signature change without edits to unrelated tests.
2. **Flag propagation** — attach `context._upstream_preamble_applied: bool` as the authoritative way the `extract_pass` handler learns whether the preamble reached the model. The flag is `True` iff the preamble string was non-empty **and** the chosen hook actually ran.
3. **Env flag** — `DOCLING_GRAPH_UPSTREAM_PREAMBLE=false` short-circuits `_render_upstream_entities_preamble` to return `""`, which in turn makes `preamble_applied=False`. This gives live rollback without a redeploy.
4. **Idempotency** — the helper must never mutate its input `docling_document_json` dict; use copy-on-write if the chosen hook is a body-mutation fallback.
5. **Leave the `run_pipeline` import local — stub `sys.modules["docling_graph"]` in tests, don't patch library attributes.** The existing `test_extract_pass_endpoint.py:13-63` shim loads `main.py` at module import time. Hoisting `from docling_graph import run_pipeline` to module scope would execute that import eagerly under the shim; on any host test env where `docling_graph` is not installed (the worker test env, for example), the whole service-test module would fail to load before any patching can happen — contradicting the host-safe pattern `_validate_library_surface` (`main.py:292`) deliberately establishes. Keep the import local. The Step 9a unit test preserves that property by assigning `sys.modules["docling_graph"] = types.ModuleType("docling_graph")` with a `run_pipeline` attribute — the local `from docling_graph import run_pipeline` at call time resolves through the stub without the real library ever loading.

Skeleton (**fill in the `# INJECT:` block per the Task 5a decision** — don't land this skeleton with a placeholder comment in main):

```python
def run_extraction_pass(
    docling_document_json: dict[str, Any],
    template_cls: type,
    upstream_entities: "list[EntityRef] | None" = None,
) -> Any:
    """Run docling-graph pipeline for a SINGLE fixed-template pass.

    When upstream_entities is non-empty AND the env flag is on, the
    rendered preamble is injected via the hook chosen in Task 5a.
    Attaches ``context._upstream_preamble_applied`` so the extract_pass
    handler can populate ExtractionMetadata without changing this
    function's return contract.
    """
    import tempfile
    from docling_graph import run_pipeline
    # NOTE: keep this import LOCAL. Hoisting it to module scope would
    # cause `import docling_graph` to run eagerly when the service test
    # shim loads main.py — and on host test envs where docling_graph
    # isn't installed, that kills the whole test module. The Step 9a
    # unit test stubs `sys.modules["docling_graph"]` with a
    # types.ModuleType carrying a run_pipeline attribute; this local
    # `from docling_graph import run_pipeline` resolves the stubbed
    # module at call time, so the real library never has to load.

    preamble = _render_upstream_entities_preamble(upstream_entities)
    preamble_applied = False

    # --- INJECT (Path B: body prepend) -------------------------------
    # If Task 5a chose Path B, mutate docling_document_json here BEFORE
    # writing the tmp file. If Task 5a chose Path A, leave this block
    # empty and modify the build_pipeline_config(...) call below instead.
    # Path C → STOP (do not land; do not bypass build_pipeline_config).
    # -----------------------------------------------------------------
    if preamble:
        # Placeholder — Task 5b replaces exactly ONE of the two INJECT
        # sites (this one OR the build_pipeline_config call below) per
        # the Task 5a decision. Do not land with either NotImplementedError
        # still reachable.
        raise NotImplementedError(
            "Task 5b: replace this INJECT site (Path B) OR the "
            "build_pipeline_config INJECT site below (Path A) per "
            "Task 5a decision before shipping",
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(docling_document_json, tmp, ensure_ascii=False, default=str)
        tmp_path = tmp.name

    try:
        # --- INJECT (Path A: library kwarg) --------------------------
        # If Task 5a chose Path A, forward `preamble` into the widened
        # build_pipeline_config signature here:
        #   config = build_pipeline_config(
        #       source=tmp_path,
        #       template_class=template_cls,
        #       extra_prompt_preamble=preamble,  # empty string when flag off
        #   )
        # If Task 5a chose Path B, keep the call unchanged (preamble was
        # already baked into docling_document_json above).
        # -------------------------------------------------------------
        config = build_pipeline_config(source=tmp_path, template_class=template_cls)
        context = run_pipeline(config)
        if preamble:
            preamble_applied = True
    finally:
        os.unlink(tmp_path)

    try:
        context._upstream_preamble_applied = preamble_applied
    except Exception:
        pass  # best-effort; handler falls back to False when attr missing
    return context
```

**Do not merge Task 5b with the `NotImplementedError` still in place.** The discovery output from Task 5a must be converted into a concrete `# INJECT:` implementation, and the integration test `test_preamble_applied_flag_flips_true_for_document_plus_entity_refs` must pass against that concrete implementation — not against a MagicMock that fakes the flag.

**Quick Path-A sketch** (library kwarg). The `build_pipeline_config` call in `run_extraction_pass` happens AFTER the tmp file is written (the `try:` block at the bottom of the function). Path A modifies **that** call, not the earlier Path-B INJECT site.

1. In `docker/docling-graph/app/config_builder.py`, widen the signature:

   ```python
   def build_pipeline_config(
       source: str,
       template_class: Type[BaseModel] | None,
       extra_prompt_preamble: str | None = None,
   ) -> Any:
       """Build a PipelineConfig from environment variables.

       extra_prompt_preamble, when non-empty, is forwarded to
       PipelineConfig's prompt-override kwarg (exact kwarg name
       discovered in Task 5a). Default of None preserves prior behaviour
       for every existing caller."""
       # ... existing body builds config_kwargs ...
       if extra_prompt_preamble:
           config_kwargs["<kwarg-name-from-task-5a>"] = extra_prompt_preamble
       return PipelineConfig(**config_kwargs)
   ```

2. Modify the existing `build_pipeline_config(...)` call **already in** `run_extraction_pass`'s `try:` block (AFTER `tmp_path` is assigned) to forward the preamble:

   ```python
   try:
       config = build_pipeline_config(
           source=tmp_path,
           template_class=template_cls,
           extra_prompt_preamble=preamble,  # "" when flag off or no entities
       )
       context = run_pipeline(config)
       if preamble:
           preamble_applied = True
   finally:
       os.unlink(tmp_path)
   ```

The kwarg name is `extra_prompt_preamble` everywhere — that's the single name used in the config-builder signature, the function skeleton's comment, and the Step 9a unit test's assertion. Don't introduce a shorter alias like `extra_prompt` anywhere.

**Quick Path-B sketch** (body prepend — no `config_builder.py` change):

```python
# Path B body mutation — copy-on-write. No edit to config_builder.py.
new_doc = dict(docling_document_json)
body_text = new_doc.get("body_text") or ""
new_doc["body_text"] = f"{preamble}\n\n{body_text}" if body_text else preamble
docling_document_json = new_doc
preamble_applied = True
```

The exact dict key (`body_text` vs something else) depends on what Task 5a finds in `docling_graph/core/extractors/contracts/*/prompts.py`. Do not land it without verification.

**Path C** (no viable hook): STOP. Do not bypass `build_pipeline_config` and construct `PipelineConfig` directly — that would fork service configuration. Surface the blocker and decide with the team whether to vendor a docling-graph fork.

In the `extract_pass` handler, read the attribute (no tuple unpacking so existing mocks don't break):

```python
    async with semaphore:
        try:
            context = await asyncio.to_thread(
                run_extraction_pass,
                body.docling_document_json,
                template_cls,
                body.upstream_entities,
            )
        except Exception as exc:
            logger.exception(...)
            raise HTTPException(...)

    preamble_applied = bool(getattr(context, "_upstream_preamble_applied", False))

    # Task 5b only CHANGES the value of upstream_preamble_applied in the
    # existing ExtractionMetadata(...) call (placed by Task 1 Step 4).
    # All other kwargs — including the existing gleaning_passes /
    # resolvers_applied / quality_gate_passed / validation_pass_applied /
    # validation_pass_edges_added fields — stay exactly as they are.
    #
    # The only diff against Task 1's metadata block is flipping False
    # to `preamble_applied` on this one kwarg:
    #   -     upstream_preamble_applied=False,  # flipped to True in Task 5b
    #   +     upstream_preamble_applied=preamble_applied,
```

**Existing test impact:** `_mock_run_pipeline_return()` in `test_extract_pass_endpoint.py:87` returns a `MagicMock()` — attribute access on `context._upstream_preamble_applied` yields another MagicMock, which is truthy and would pollute the flag. Fix the mock once:

```python
def _mock_run_pipeline_return(preamble_applied: bool = False):
    ctx = MagicMock()
    ctx.knowledge_graph.number_of_nodes.return_value = 0
    ctx.knowledge_graph.number_of_edges.return_value = 0
    ctx.graph_metadata = MagicMock(node_count=0, edge_count=0, node_types={}, edge_types={})
    ctx.template_instance.model_dump.return_value = {}
    ctx._upstream_preamble_applied = preamble_applied  # explicit; avoids auto-MagicMock
    return ctx
```

All existing call sites (`test_extract_pass_valid_document_only_returns_200`, `test_extract_pass_valid_document_plus_entity_refs_returns_200`) work unchanged because the default is `False`.

- [ ] **Step 8: Expose the env flag in docker-compose.yml**

In `docker-compose.yml`, under the `docling-graph` service's `environment:` block, add:

```yaml
      DOCLING_GRAPH_UPSTREAM_PREAMBLE: ${DOCLING_GRAPH_UPSTREAM_PREAMBLE:-true}
```

- [ ] **Step 9: Run endpoint tests to verify they pass**

Run: `cd docker/docling-graph && pytest tests/test_extract_pass_endpoint.py tests/test_upstream_entities_preamble.py -v`
Expected: all PASS.

### Step 3: Direct unit test of `run_extraction_pass` — proves the INJECT hook actually runs

The endpoint tests above only prove metadata propagation — they patch `run_extraction_pass` itself, so they cannot catch the case where the INJECT block is wrong (e.g. wrong dict key, wrong `PipelineConfig` kwarg). Add one unit test that calls `run_extraction_pass` directly with `run_pipeline` and `build_pipeline_config` mocked, and asserts the observable side-effect of the chosen injection path.

**Two concrete traps the reviewer flagged that this step must handle:**

1. **Module import shim** — `docker/docling-graph/tests/test_extract_pass_endpoint.py:13-63` loads the service's `main.py` under the module name `docling_graph_service_main` via `importlib`, because the repo-root `app/` package conflicts with the service's `app/` package at import time. `from app import main as _main` in a test is not reliable. Use the existing `dg_app_module` fixture.
2. **Local import inside `run_extraction_pass`, and the library may not even be installed on the host** — `main.py:340` does `from docling_graph import run_pipeline` **inside the function body**, and `_validate_library_surface` (`main.py:292`) explicitly tolerates the library being absent on import. Host test envs commonly don't have `docling_graph` installed. **Do not** use `monkeypatch.setattr("docling_graph.run_pipeline", _fake)` — that form does `import docling_graph` eagerly and tears down the host-safe property the service is built around. Instead, stub `sys.modules["docling_graph"]` with a `types.ModuleType` carrying a `run_pipeline` attribute before calling `run_extraction_pass`. The local `from docling_graph import run_pipeline` resolves `sys.modules['docling_graph']` at call time, so the stub is picked up without the real library loading.

- [ ] **Step 9a: Write the failing tests** — append to `docker/docling-graph/tests/test_extract_pass_endpoint.py` so they reuse the existing `dg_app_module` fixture and shim machinery:

```python
def test_run_extraction_pass_exercises_chosen_injection_hook(dg_app_module, monkeypatch):
    """Prove run_extraction_pass actually invokes the hook Task 5a chose.

    This is the test that fails if Task 5b lands with the
    NotImplementedError placeholder OR with the wrong hook path. It
    uses the dg_app_module fixture (see _ensure_dg_app_package at line
    17) instead of `from app import main` because the repo-root and
    service `app/` packages collide under the combined test suite.

    The import of `run_pipeline` stays LOCAL inside run_extraction_pass
    — do NOT hoist it. The test below stubs `sys.modules["docling_graph"]`
    with a types.ModuleType carrying a `run_pipeline` attribute; the local
    `from docling_graph import run_pipeline` inside run_extraction_pass
    resolves `sys.modules['docling_graph']` at call time, so the stub is
    picked up without the real library ever loading. This preserves the
    host-safe property main.py:292's _validate_library_surface
    deliberately establishes.

    Adjust the final assertion to whichever side-effect Task 5a identified:
      - Path A (library kwarg): preamble arrived in build_pipeline_config kwargs
      - Path B (body prepend):  preamble arrived in the dict passed to json.dump
    """
    import json as _json
    dg_app_module.__dict__["EntityRef"]  # sanity: symbol is resolvable under shim
    EntityRef = dg_app_module.EntityRef

    captured = {}

    def _fake_build_pipeline_config(*args, **kwargs):
        captured.setdefault("build_config_calls", []).append({"args": args, "kwargs": kwargs})
        return object()  # opaque config — run_pipeline is mocked too

    def _fake_run_pipeline(config):
        captured["config"] = config
        ctx = MagicMock()
        ctx.knowledge_graph.number_of_nodes.return_value = 0
        ctx.knowledge_graph.number_of_edges.return_value = 0
        ctx.graph_metadata = MagicMock(node_count=0, edge_count=0, node_types={}, edge_types={})
        ctx.template_instance.model_dump.return_value = {}
        return ctx

    # Spy on json.dump so Path-B tests see EXACTLY what was written to the
    # temp file, without scanning /tmp/*.json (which races other tests).
    real_dump = _json.dump
    def _spy_dump(obj, fp, *args, **kwargs):
        captured["tmp_body_dict"] = obj
        return real_dump(obj, fp, *args, **kwargs)
    monkeypatch.setattr(_json, "dump", _spy_dump)

    monkeypatch.setattr(dg_app_module, "build_pipeline_config", _fake_build_pipeline_config)
    # Stub docling_graph in sys.modules so run_extraction_pass's LOCAL
    # `from docling_graph import run_pipeline` resolves to our fake.
    #
    # Why not `monkeypatch.setattr("docling_graph.run_pipeline", ...)`?
    # That form requires `import docling_graph` to succeed first, which
    # forces the real library to load. The service is deliberately written
    # to tolerate docling_graph being absent at module-import time
    # (_validate_library_surface at main.py:292 catches ImportError), and
    # on host test envs the library often IS absent. Stubbing sys.modules
    # preserves that host-safe property and still makes the local
    # `from docling_graph import run_pipeline` inside run_extraction_pass
    # pick up our fake.
    import sys, types
    fake_dg_module = types.ModuleType("docling_graph")
    fake_dg_module.run_pipeline = _fake_run_pipeline
    monkeypatch.setitem(sys.modules, "docling_graph", fake_dg_module)
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "true")

    entities = [EntityRef(
        ref_id="E001",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
    )]
    context = dg_app_module.run_extraction_pass(
        docling_document_json={"name": "test", "body_text": "original body"},
        template_cls=type("T", (), {}),
        upstream_entities=entities,
    )

    # Flag propagation (independent of path chosen).
    assert context._upstream_preamble_applied is True

    # === Path-specific assertion — pick ONE per Task 5a decision ===
    # Path A (kwarg name must match config_builder.py's widened signature —
    # `extra_prompt_preamble`):
    #   last = captured["build_config_calls"][-1]
    #   assert "Upstream entities:" in (last["kwargs"].get("extra_prompt_preamble") or "")
    # Path B:
    #   body = captured["tmp_body_dict"]
    #   assert body["body_text"].startswith("Upstream entities:")
    #   # Copy-on-write: the caller's dict must NOT be mutated.
    #   assert body is not original_input  # capture the input dict above if needed
    #
    # Task 5b MUST replace the below AssertionError with the concrete
    # assertion above — leaving the placeholder means the INJECT block
    # wasn't verified.
    raise AssertionError(
        "Task 5b: replace with the concrete Path-A or Path-B assertion "
        "per the Task 5a decision before committing."
    )


def test_run_extraction_pass_env_flag_off_does_not_invoke_hook(dg_app_module, monkeypatch):
    """With the preamble env flag off, the INJECT block must not run even
    when upstream_entities is populated. Same shim/patch discipline as
    the positive test."""
    import json as _json
    EntityRef = dg_app_module.EntityRef

    captured = {}

    def _fake_build_pipeline_config(*args, **kwargs):
        captured["build_config_kwargs"] = kwargs
        return object()

    def _fake_run_pipeline(config):
        ctx = MagicMock()
        ctx.knowledge_graph.number_of_nodes.return_value = 0
        ctx.knowledge_graph.number_of_edges.return_value = 0
        ctx.graph_metadata = MagicMock(node_count=0, edge_count=0, node_types={}, edge_types={})
        ctx.template_instance.model_dump.return_value = {}
        return ctx

    real_dump = _json.dump
    def _spy_dump(obj, fp, *args, **kwargs):
        captured["tmp_body_dict"] = obj
        return real_dump(obj, fp, *args, **kwargs)
    monkeypatch.setattr(_json, "dump", _spy_dump)

    monkeypatch.setattr(dg_app_module, "build_pipeline_config", _fake_build_pipeline_config)
    # Same host-safe stub as the positive test — see its comment.
    import sys, types
    fake_dg_module = types.ModuleType("docling_graph")
    fake_dg_module.run_pipeline = _fake_run_pipeline
    monkeypatch.setitem(sys.modules, "docling_graph", fake_dg_module)
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "false")

    entities = [EntityRef(
        ref_id="E001", entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"}, display_label="Fan Song",
    )]
    context = dg_app_module.run_extraction_pass(
        docling_document_json={"name": "test", "body_text": "original body"},
        template_cls=type("T", (), {}),
        upstream_entities=entities,
    )
    assert context._upstream_preamble_applied is False

    # === Path-specific NEGATIVE assertion — pick ONE per Task 5a decision ===
    # The flag being False is not enough: a buggy INJECT block could set
    # _upstream_preamble_applied = False while still forwarding the kwarg OR
    # mutating the body. Assert the side-effect is absent.
    #
    # Path A: assert not captured["build_config_kwargs"].get("extra_prompt_preamble")
    # Path B: assert captured["tmp_body_dict"]["body_text"] == "original body"
    #
    # This AssertionError is a tripwire — Task 5b MUST replace it with the
    # concrete negative assertion for the chosen path before the task is
    # considered done. Leaving the tripwire in = the env-flag path was not
    # actually verified.
    raise AssertionError(
        "Task 5b: replace with the concrete Path-A or Path-B NEGATIVE "
        "assertion (side-effect absent) per the Task 5a decision before committing."
    )
```

- [ ] **Step 9b: Confirm BOTH assertions are the right shape for the Task 5a path**

Before running the suite, replace the `AssertionError("Task 5b: replace with the concrete Path-A or Path-B …")` placeholders in **both** tests — `test_run_extraction_pass_exercises_chosen_injection_hook` (positive assertion: the side-effect was applied) AND `test_run_extraction_pass_env_flag_off_does_not_invoke_hook` (negative assertion: the side-effect is absent). If the Task 5a decision is still "unknown", these two tripwires are what keep the task from being declared done.

- [ ] **Step 9c: Run the preamble-injection tests**

Run: `cd docker/docling-graph && pytest tests/test_upstream_entities_preamble.py tests/test_extract_pass_endpoint.py -v -k "run_extraction_pass or preamble"`
Expected: all PASS with the real injection path exercised. If only the endpoint tests pass but `test_run_extraction_pass_exercises_chosen_injection_hook` fails, the INJECT block is still wrong.

- [ ] **Step 10: Commit**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/app/config_builder.py docker/docling-graph/tests/test_upstream_entities_preamble.py docker/docling-graph/tests/test_extract_pass_endpoint.py docker-compose.yml
git commit -m "feat(docling-graph): inject upstream-entity preamble for document_plus_entity_refs passes

Adds _render_upstream_entities_preamble which emits a compact
'[E###] ENTITY_TYPE — display_label' list plus an instruction to the
model to emit from_ref_id/to_ref_id only from that list. run_extraction_pass
threads the preamble into the pipeline config's prompt; run without entities
or with DOCLING_GRAPH_UPSTREAM_PREAMBLE=false behaves as before.
upstream_preamble_applied in ExtractionMetadata reflects whether the
preamble actually reached the model."
```

---

## Task 6: Final verification + rebuild + recreate

**Files:** none (scripts + compose)

- [ ] **Step 1: Run the full affected suite on the host**

Run: `.venv/bin/pytest tests/unit/test_bundle_validators.py tests/unit/test_specification_entity_validation.py tests/unit/test_ontology_bundles.py tests/unit/test_extraction_merge.py tests/unit/test_extraction_schemas.py tests/unit/test_derive_ontology_graph_bundle_passes.py tests/unit/test_arcadedb_graph.py tests/unit/test_pipeline_upstream_refs.py tests/unit/test_run_single_pass.py`
Expected: all PASS, zero warnings about deprecated helpers.

- [ ] **Step 2: Run the docling-graph suite**

Run: `cd docker/docling-graph && pytest -v`
Expected: all PASS.

- [ ] **Step 3: Rebuild worker + docling-graph images**

Run: `docker compose build worker api beat docling-graph`
Expected: all four images "Built" with no errors. Timing ~2-5 min on a warm cache.

- [ ] **Step 4: Recreate containers**

Run: `docker compose up -d worker beat api docling-graph`
Expected: services "Started", none "Restarting", health probes passing within 60s.

- [ ] **Step 5: Smoke-check the live stack**

Run: `docker compose exec -T worker python3 -c "from app.workers.pipeline import _select_upstream_refs_for_pass, _extend_upstream_refs; from app.services.extraction_merge import logical_identity_from_dict; print('worker OK')"`
Expected output: `worker OK`

Run:
```bash
docker compose exec -T docling-graph python3 -c "
from app.schemas import EntityRef
from app.main import _render_upstream_entities_preamble
print(_render_upstream_entities_preamble([
    EntityRef(ref_id='E001', entity_type='RADAR_SYSTEM',
              identity_values={'system_name': 'Fan Song'},
              display_label='Fan Song'),
]))
"
```

Expected output begins:

```
Upstream entities:
  [E001] RADAR_SYSTEM — Fan Song

Only emit from_ref_id and to_ref_id values from the list above. …
```

- [ ] **Step 6: Watch logs on the next upload batch**

After the user triggers a reingest, tail the docling-graph service:

Run: `docker compose logs --tail=50 -f docling-graph 2>&1 | grep -E "START|END|preamble"`
Expected lines for a system_links call:
- `extract-pass: START bundle=air_defense_v3 pass=system_links input_mode=document_plus_entity_refs document_id=<uuid> upstream_ref_count=N` (N > 0 when prior passes produced entities)
- `extract-pass: injecting upstream-entity preamble (<chars> chars, N refs)`
- `extract-pass: END bundle=air_defense_v3 pass=system_links … node_count=X edge_count=Y`

And check the DB per-pass metrics for rejection reasons. The authoritative key is `stage_runs.metrics['rejections_by_reason']['unknown_ref_id']` (lowercase, under the `rejections_by_reason` sub-object — NOT a top-level `unknown_ref_id_count`). Run:

```sql
SELECT pass_name,
       metrics->'rejections_by_reason'->>'unknown_ref_id' AS unknown_ref_id,
       metrics->'rejections_by_reason'->>'invalid_triple' AS invalid_triple,
       finished_at
FROM ingest.stage_runs
WHERE stage_name='derive_ontology_graph' AND pass_name='system_links'
ORDER BY finished_at DESC LIMIT 20;
```

Batch-over-batch, `unknown_ref_id` for `pass_name='system_links'` should trend down. If the column reads `NULL`, Task 4's `_apply_post_merge_yield_updates` wiring didn't land — go back to that task.

- [ ] **Step 7: Final commit**

No code changes in this task; if any ad-hoc fixes were needed they should be squashed into the task that introduced them.

---

## Success criteria (measurable)

- ✅ `extract-pass: START` log line includes `pass_name`, `input_mode`, `document_id`, `upstream_ref_count`; `END` line includes `node_count`, `edge_count`.
- ✅ `ExtractionMetadata.upstream_ref_count` and `upstream_preamble_applied` present on every response (Task 1).
- ✅ Per-pass `stage_runs.metrics->'rejections_by_reason'->>'unknown_ref_id'` drops materially batch-over-batch for `pass_name='system_links'`. Values are lowercase (matching `RelationshipRejectionReason` enum). Query: `SELECT metrics->'rejections_by_reason'->>'unknown_ref_id' FROM ingest.stage_runs WHERE stage_name='derive_ontology_graph' AND pass_name='system_links' ORDER BY finished_at DESC LIMIT 20;`
- ✅ Retained `ASSOCIATED_WITH` / `CUES` edge counts increase batch-over-batch (query ArcadeDB for edges of those types per pipeline_run).
- ✅ Per-pass entity counts do not regress for the same corpus.
- ✅ Total number of passes unchanged (still 5; `system_links` still conditionally skipped via `skip_if_no_upstream_endpoints`).
- ✅ `DOCLING_GRAPH_UPSTREAM_PREAMBLE=false` fully disables the injection (rollback flag verified live).
- ✅ No ref with missing/None/empty identity fields ever appears in request bodies (Task 4 validity-rule tests lock this in).

## Rollback plan

- Any step can be reverted with a single `git revert` — tasks are independent commits.
- Preamble injection alone (Task 5) can be disabled live via `DOCLING_GRAPH_UPSTREAM_PREAMBLE=false` without redeploying.
- `document_id` in the extract-pass request body is backward-compatible (the service accepts the field but didn't require it previously).

## Out of scope for this plan

- Surfacing salvage metadata in `ExtractionMetadata` (would require changes to the docling-graph library; tracked separately).
- A spec-heavy corpus eval harness (separate tooling workflow).
- Schema redesign for `SPECIFICATION` / other entity types.
- Any change to the number of passes, their `required` flags, or their `depends_on`.

---
