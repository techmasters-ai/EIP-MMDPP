# Table-Aware Chunking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a table-aware chunking layer that normalizes Docling tables into a shared model with two renderers (graph extraction + embedding retrieval), preserving cell-level provenance without regressing today's behavior.

**Architecture:** Single source of truth `NormalizedTable` model in `app/services/table_normalization/`, COPY'd into the docling-graph image at build. Master switches default OFF — code merges with zero behavior change; behavior changes are a separate Phase 2 flag flip gated on regression checks against a captured baseline.

**Tech Stack:** Python 3.12, SQLAlchemy 2 + Alembic, ArcadeDB, Docling/HybridChunker, pytest, Postgres JSONB.

**Spec:** `docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md` (rev. 7, 1,412 lines).

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `app/services/table_normalization/__init__.py` | Public API exports: `normalize_tables`, `render_for_graph`, `render_for_embedding`, the `is_*_enabled` flag helpers. |
| `app/services/table_normalization/models.py` | `Shape`, `ChunkKind` enums; `CellRef`, `NormalizedCell`, `NormalizedRow`, `NormalizedColumn`, `TableSection`, `NormalizedTable` frozen dataclasses; `GraphTableChunk`, `EmbeddingTableChunk`. |
| `app/services/table_normalization/detect.py` | Pure `detect_shape(table_cells, table_data) -> Shape`. Closed keyword frozensets (SPEC_ROW_KEYWORDS, SECTION_KEYWORDS, IDENTITY_LABEL_KEYWORDS). |
| `app/services/table_normalization/normalize.py` | `normalize_tables(doc_json) -> list[NormalizedTable]`. Per-table pipeline; idempotent. |
| `app/services/table_normalization/tokens.py` | Lazy-loaded `count_bge_m3_tokens(text) -> int`. Module-level tokenizer cache. |
| `app/services/table_normalization/render_graph.py` | `render_for_graph(table, token_limit_whole, token_limit_column) -> list[GraphTableChunk]`. Internal `_render_column_as_text` shared with embedding renderer. |
| `app/services/table_normalization/render_embedding.py` | `render_for_embedding(table, token_limit, summary_limit) -> list[EmbeddingTableChunk]`. Reuses `_render_column_as_text`. |
| `app/services/table_normalization/config.py` | Env-flag reading: `is_table_normalization_enabled_graph()`, `is_table_normalization_enabled_embedding()`, `is_experimental_table_facts_enabled()`, threshold getters. |
| `app/services/table_normalization/_provenance_bridge.py` | `record_text_idx_cell_refs`, `cell_refs_for_text_idx`, `reset` — process-local `_TEXT_IDX_TO_CELL_REFS: dict[int, list[str]]`. |
| `app/services/table_normalization/_pipeline_hooks.py` | `_substitute_table_chunks`, `_classify_native_chunk`, `_normalized_table_size_tokens`, `_NormalizedTableChunkAdapter` — the HybridChunker post-processing logic. |
| `alembic/versions/0021_chunk_metadata.py` | Migration adding `chunk_metadata JSONB` column + partial expression index on `chunk_kind`. |
| `tests/unit/test_table_normalization_models.py` | Frozen dataclass invariants. |
| `tests/unit/test_table_normalization_detect.py` | Shape-detection heuristics. |
| `tests/unit/test_table_normalization_normalize.py` | End-to-end normalization on fixture documents. |
| `tests/unit/test_table_normalization_tokens.py` | Tokenizer accuracy. |
| `tests/unit/test_table_normalization_render_graph.py` | Graph renderer snapshots. |
| `tests/unit/test_table_normalization_render_embedding.py` | Embedding renderer snapshots. |
| `tests/unit/test_render_column_byte_equality.py` | `_render_column_as_text` produces byte-identical output called from both renderers. |
| `tests/unit/test_suppress_raw_table_texts_invariant.py` | Blank-in-place; `tables[]` untouched. |
| `tests/unit/test_hybrid_chunker_substitution.py` | `_substitute_table_chunks` + classification + threshold + adapter ducktyping. |
| `tests/unit/test_normalized_table_chunk_adapter.py` | Adapter conforms to native-chunk read interface. |
| `tests/unit/test_table_size_threshold.py` | `_normalized_table_size_tokens` boundary behavior at 256 tokens. |
| `tests/integration/test_chunk_metadata_persistence.py` | Postgres write/read; partial index returns matching rows. |
| `tests/integration/test_retrieval_table_chunk_surfacing.py` | `/v1/retrieval/query` response envelope includes `table_chunk` block. |
| `tests/integration/test_graph_provenance_cell_refs.py` | `ExtractionFieldProvenance.cell_refs` populated correctly. |
| `tests/integration/test_master_kill_switch_byte_equality.py` | All flags off → byte-identical chunks vs §19 baseline (both paths). |
| `tests/integration/test_legacy_table_facts_drift.py` | Experimental `_table_facts.py` path still produces non-empty output (count ± 10%). |
| `tests/integration/test_disallowed_combination_fallback.py` | Both `*_ENABLED` master + experimental flag → fallback to today's behavior + ERROR log. |
| `tests/integration/test_sa2_table_pipeline_e2e.py` | Full ingest → extract → score → cell_refs traceable from extracted field. |
| `tests/spike/test_provenance_e2e.py` | Step 0b: end-to-end channel-A provenance verification on minimal fixture. |
| `tests/spike/test_legacy_element_bridge.py` | Step 0b optional: `DocumentElement.element_metadata.self_ref` presence for table elements. |
| `tests/fixtures/sa2/baseline.meta.json` | Captured at Step 0a: main_sha, docling_graph_image_id, corpus_files, runs_per_doc, comparison_mode. |
| `tests/fixtures/sa2/<docid>_texts_today.json` | Captured at Step 0a: full `doc_json["texts"]` after sanitization. |
| `tests/fixtures/sa2/<docid>_extraction_counts_today.json` | Captured at Step 0a: per-pass `{exact, wrong, null}` counts × 3 runs. |
| `tests/fixtures/sa2_sample_table.json` | Snapshot fixture of one SA-2 table for renderer/normalizer tests. |

### Modified files

| Path | Lines | Change |
|---|---|---|
| `app/workers/pipeline.py` | ~5399–5664 (HybridChunker + legacy paths) | Add `_substitute_table_chunks` call, thread `chunk_metadata`, add `chunk_index` + `page_number` to upsert `set_`, add `chunk_kind` to ArcadeDB write. |
| `app/services/chunking.py` | 36, 113–126 | Add `normalized_tables` kwarg; integrate `normalized_table_for` lookup; add optional `metadata` to `StructuredChunk`. |
| `app/models/retrieval.py` | 11–57 | Add `chunk_metadata: Mapped[Optional[dict]]` to `TextChunk`. |
| `docker/docling-graph/app/schemas.py` | 195–233 | Add `chunk_index: Optional[int]` and `cell_refs: list[str]` to `ExtractionFieldProvenance`. |
| `docker/docling-graph/app/main.py` | 564 (insertion), 793–797 (enrichment call) | Insert `normalize_tables` + integration block after sanitization; call `_enrich_field_provenance_with_cell_refs` after `chunk_to_self_refs` built. |
| `app/services/arcadedb_schema.py` | 31 | Add `("chunk_kind", "STRING")` to `_STRUCTURAL_VERTEX_TYPES["TextChunk"]`; add index on `chunk_kind`. |
| `docker/docling-graph/Dockerfile` | end-of-file COPY block | `COPY app/services/table_normalization /app/app/services/table_normalization`. |
| `app/config.py` | (new vars region) | Add new settings fields for env vars per §12. |
| `.env`, `.env.example` | (new vars region) | Add the 8 new env vars with defaults + one-line comments each. |
| `VERIFICATION_CHECKLIST.md` | (end) | Add the 8 verification rows per §14. |

### Deleted files

| Path | Reason |
|---|---|
| `docker/docling-graph/app/_table_pivot.py` | Already deprecated; sole consumer is its own test (grep verified). |
| `docker/docling-graph/tests/test_table_pivot.py` | Tests the deleted module. |

### Preserved unchanged (verify by grep — no implementation changes)

- `docker/docling-graph/app/_table_facts.py` (production-live for `extract_table_overlay` at `main.py:1185`).
- `docker/docling-graph/app/_alias_map.py` (consumed only by `_table_facts.py`).
- `app/services/table_overlay.py` (orthogonal — Phase 0/0.5 merge-time machinery).

---

## Chunk 1: Phase 0 (Baseline + Spike) and Phase 1 (Normalization Library)

## Phase 0 — Baseline capture + implementation spike

Both Phase 0 tasks must complete BEFORE any feature work lands. Phase 0 outputs are committed fixtures that downstream tasks depend on.

### Task 0a: Capture today's baseline (§19)

**Files:**
- Create: `tests/fixtures/sa2/baseline.meta.json`
- Create: `tests/fixtures/sa2/<docid>_texts_today.json` (one per SA-2 corpus document)
- Create: `tests/fixtures/sa2/<docid>_extraction_counts_today.json` (one per SA-2 corpus document)
- Modify (temporary): `docker/docling-graph/app/main.py` around line 564

**Why:** all subsequent merge gates compare against this baseline. Without it captured at the pre-rewrite SHA, the master kill-switch byte-equality test (Task 19) and the Phase 2 flip gate (Task 21) have nothing to assert against.

- [ ] **Step 1: Confirm the corpus and ingest mechanism**

The user runs `think_true_*.csv` extraction on the SA-2 corpus. Identify:
- The exact list of SA-2 corpus documents (their `document_id`s or filenames).
- The exact ingest command the user runs to process them on `main`.
- The exact scoring command that produces `{exact, wrong, null}` counts per pass.

Record these in `tests/fixtures/sa2/baseline.meta.json` (`corpus_files`, plus a comment field documenting the ingest/scoring commands).

If you can't determine these, STOP and ask the user — Task 0a cannot be completed without them.

- [ ] **Step 2: Add temporary instrumentation hook to capture `doc_json["texts"]`**

Edit `docker/docling-graph/app/main.py` immediately after the sanitization block (around line 564, before the reverted-block comment at 566):

```python
# TEMPORARY: baseline capture hook for Task 0a. Remove before Phase 1 merges.
if os.environ.get("CAPTURE_BASELINE_TEXTS"):
    import json as _json
    _baseline_dir = os.environ["CAPTURE_BASELINE_TEXTS"]
    _doc_id = body.document_id
    with open(f"{_baseline_dir}/{_doc_id}_texts_today.json", "w") as f:
        _json.dump(body.docling_document_json["texts"], f, indent=2)
```

- [ ] **Step 3: Capture `doc_json["texts"]` for every SA-2 doc on current `main`**

Verify you are on `main` (not the feature branch):
```bash
git status   # should show clean tree; current branch can be feat/table-aware-chunking
git rev-parse main   # record this SHA
```

For each SA-2 doc, set `CAPTURE_BASELINE_TEXTS=$(pwd)/tests/fixtures/sa2` and run the ingest command. Confirm one JSON file per doc lands at `tests/fixtures/sa2/<docid>_texts_today.json`.

These captures are deterministic (sanitization is a pure function of input + sanitizer code at the recorded SHA). Single run suffices.

- [ ] **Step 4: Run extraction N=3 times per doc on current `main`**

For each SA-2 doc, run extraction 3 times. Use temperature=0 (or whichever temperature the user's `think_true_*.csv` runs use; record in meta).

For each run, capture per-pass `{exact, wrong, null}` counts via the user's existing scoring script. Aggregate into one file per doc:

```json
{
  "missile_propulsion": {"runs": [{"exact": 3, "wrong": 12, "null": 5}, {"exact": 4, "wrong": 11, "null": 5}, {"exact": 3, "wrong": 12, "null": 5}]},
  "kinematics":         {"runs": [...]},
  "speed_timing":       {"runs": [...]},
  "airframe":           {"runs": [...]}
}
```

Commit one file per doc as `tests/fixtures/sa2/<docid>_extraction_counts_today.json`.

- [ ] **Step 5: Determine comparison mode (strict / median) per the §19 decision rule**

Compute per-pass per-doc `max(exact) − min(exact)` across the 3 runs:
- If ALL (pass, doc) pairs satisfy `max − min ≤ 1`: comparison_mode = `"strict"`. Use run-0 counts.
- If ANY (pass, doc) pair has `max − min ≥ 2`: comparison_mode = `"median"`. Use median-of-3.

- [ ] **Step 6: Write `tests/fixtures/sa2/baseline.meta.json`**

```json
{
  "captured_at": "<ISO 8601 timestamp>",
  "main_sha": "<git rev-parse main>",
  "docling_graph_image_id": "<docker compose images docling-graph --format json | jq -r '.[].ID'>",
  "corpus_files": ["doc1.pdf", "doc2.pdf", ...],
  "temperature": 0,
  "runs_per_doc": 3,
  "comparison_mode": "strict",
  "ingest_command": "<exact command used in Step 1>",
  "scoring_command": "<exact command used in Step 4>"
}
```

- [ ] **Step 7: Remove the temporary instrumentation hook**

Revert the change to `docker/docling-graph/app/main.py` from Step 2.

If the hook was NEVER committed (preferred — keep it as a working-tree change throughout):
```bash
git checkout HEAD -- docker/docling-graph/app/main.py
```

If the hook WAS committed in error:
```bash
git revert <hook-commit-sha>   # or manually remove the lines and commit
```

Verify the file matches `main`'s version:
```bash
git diff main -- docker/docling-graph/app/main.py
```
Expected: no diff.

- [ ] **Step 8: Commit baseline fixtures**

```bash
git add tests/fixtures/sa2/
git commit -m "test(baseline): capture today's SA-2 production behavior pre-rewrite

main_sha: <SHA recorded in baseline.meta.json>
Captured per §19 procedure: doc_json[\"texts\"] per doc (deterministic) +
per-pass {exact, wrong, null} counts × 3 runs + meta with comparison_mode.

This baseline is the merge-gate target for Task 19 (byte-equality) and
the Phase 2 flip-gate target for Task 21 (no-regression check).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 0b: Implementation spike — channel-A provenance verification (§20.1)

**Files:**
- Create: `tests/spike/test_provenance_e2e.py`
- Create: `tests/spike/test_legacy_element_bridge.py` (optional / deferred)
- Possibly modify: `docker/docling-graph/app/main.py:765-781` and `:952-975` (only if spike reveals the bridge mechanism needs adjustment)

**Why:** the spec's channel-A mechanism (§11.6) assumes `chunk_to_self_refs` contains `#/texts/N` refs that our synthesized TextItems' `self_ref` matches. The spike verifies this end-to-end on a minimal fixture before Phase 1 feature work begins.

- [ ] **Step 1: Build a minimal fixture document**

Create `tests/fixtures/spike/minimal_doc_with_table.json` — a tiny `DoclingDocument` JSON with:
- Two `texts[]` entries (one prose paragraph, one synthesized TextItem with `self_ref="#/texts/1"` and `prov=[{"$ref": "#/tables/0/data/table_cells/0"}]`).
- One `tables[]` entry (the table the prov entry references).
- Minimal sanitizer-friendly shape.

The synthesized TextItem represents what `_text_item_from_chunk` would produce in the real flow. We're testing the downstream provenance flow, not the renderer.

- [ ] **Step 2: Write `tests/spike/test_provenance_e2e.py`**

```python
"""Spike (§20.1): verify channel-A provenance flow end-to-end.

Constructs a minimal doc with a synthesized TextItem whose self_ref is
#/texts/1 and whose prov references a cell in #/tables/0. Runs one
extraction pass. Asserts:
  1. chunk_to_self_refs[chunk_id] contains '#/texts/1' for the chunk
     that consumed the synthesized TextItem.
  2. The provenance bridge map (populated by record_text_idx_cell_refs)
     is read correctly by _enrich_field_provenance_with_cell_refs.
  3. The resulting ExtractionFieldProvenance row carries non-empty
     cell_refs.
  4. Per-pass reset clears the bridge.
"""
import pytest
from pathlib import Path

# This spike runs end-to-end through extract-pass. It requires the
# docling-graph stack to be running locally (docker compose up).

FIXTURE = Path("tests/fixtures/spike/minimal_doc_with_table.json")

@pytest.mark.spike
def test_channel_a_e2e():
    # 1. Load minimal doc.
    # 2. Manually populate _TEXT_IDX_TO_CELL_REFS via record_text_idx_cell_refs(1, ["#/tables/0/data/table_cells/0"])
    # 3. Call extract-pass against a minimal pass template that extracts an entity.
    # 4. Read context._chunk_to_self_refs at the end of the pass.
    # 5. Assert: at least one chunk's self_refs contains "#/texts/1".
    # 6. Assert: the response's ExtractionFieldProvenance rows include cell_refs containing "#/tables/0/data/table_cells/0".
    # 7. Call reset(); confirm a second pass on a different doc has no leakage.
    pass  # Implementation in Step 3
```

- [ ] **Step 3: Implement the spike test body**

Implement the steps in the docstring. Use whatever harness exists for running `extract-pass` (FastAPI TestClient, direct function call, or the existing `tests/integration/` patterns — pick what's cheapest and most direct).

- [ ] **Step 4: Run the spike test**

Run: `pytest tests/spike/test_provenance_e2e.py -v -m spike`

**Pass criterion (decisions tree):**
- ✅ Test passes → spec §11.6 mechanism works as designed. Phase 1 proceeds.
- ❌ `chunk_to_self_refs` does not contain `#/texts/1` → the synthesized TextItem's self_ref didn't flow into the chunk-creation trace. Likely cause: docling-graph library doesn't populate `self_refs` from text items that weren't part of the original `texts[]` array (i.e., it caches an earlier snapshot). **Fix:** extend `main.py:765-774` to also scan `body.docling_document_json["texts"]` for synthesized text items (matching by `self_ref` shape `#/texts/N`) and add their refs to `chunk_to_self_refs[chunk_id]` where the chunk text contains them. ~15 LOC.
- ❌ `cell_refs` are not populated in `ExtractionFieldProvenance` → the bridge map was populated but `_enrich_field_provenance_with_cell_refs` either wasn't called or returned empty. Debug by adding logging at the enrichment site; verify `chunk_to_self_refs` has the right shape; verify the regex `#/texts/(\d+)$` matches.
- ❌ Bridge leakage across passes → `reset()` wasn't called. Trace the call site and fix.

For any ❌, fix the spec + the underlying code, re-run, repeat until ✅. Estimated budget: 1 hour for ✅, 3-4 hours if any ❌ (per §20 caveat).

- [ ] **Step 5: Run the legacy bridge spike (optional)**

`tests/spike/test_legacy_element_bridge.py` — ingest one document via the legacy path (force by setting `enrichments.version = None` in the doc); query `SELECT id, element_type, metadata FROM ingest.document_elements WHERE element_type = 'table' LIMIT 5`. Assert each table element's `metadata.self_ref` matches `#/tables/N`.

If absent: add `"self_ref": chunk.self_ref` to the chunk-metadata-building site in `pipeline.py` (locate via `grep -n 'chunk.metadata' app/workers/pipeline.py`). One-line change. Skip this spike if the legacy path won't be exercised in your environment.

- [ ] **Step 6: Commit spike output**

```bash
git add tests/spike/ tests/fixtures/spike/
# Plus any code changes from the ❌ paths in Step 4.
git commit -m "test(spike): verify channel-A provenance flow end-to-end

Step 0b per spec §20.1. Confirms cell_refs flow into
ExtractionFieldProvenance via the two-hop lookup (chunk_index →
chunk_to_self_refs → first #/texts/N → bridge → cell_refs).

[Outcome paragraph: 'spec mechanism worked unchanged' OR 'required
fix at main.py:XXX with X LOC change']

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 — Build the normalization library (TDD)

Each task builds one module of `app/services/table_normalization/`. TDD: write the failing test first, then minimum implementation, then commit. Files stay focused (<300 LOC each per the spec's module decomposition).

### Task 1: NormalizedTable data model + ChunkKind enum

**Files:**
- Create: `app/services/table_normalization/models.py`
- Create: `app/services/table_normalization/__init__.py` (skeleton exports)
- Create: `tests/unit/test_table_normalization_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_table_normalization_models.py
import pytest
from app.services.table_normalization.models import (
    Shape, ChunkKind, CellRef, NormalizedCell, NormalizedRow,
    NormalizedColumn, TableSection, NormalizedTable,
    GraphTableChunk, EmbeddingTableChunk,
)


def test_shape_enum_values():
    assert Shape.COLUMN_MAJOR.value == "column_major"
    assert Shape.ROW_MAJOR.value == "row_major"
    assert Shape.HYBRID.value == "hybrid"
    assert Shape.OTHER.value == "other"


def test_chunk_kind_enum_values():
    assert ChunkKind.TABLE_SUMMARY.value == "table_summary"
    assert ChunkKind.TABLE_WHOLE.value == "table_whole"
    assert ChunkKind.TABLE_ENTITY_COLUMN.value == "table_entity_column"
    assert ChunkKind.TABLE_ENTITY_SECTION.value == "table_entity_section"


def test_normalized_table_is_frozen():
    nt = NormalizedTable(
        table_index=0, self_ref="#/tables/0", caption=None,
        page_numbers=(1,), shape=Shape.OTHER, rows=(), columns=(),
        sections=(), cells=(), raw_markdown="",
    )
    with pytest.raises((AttributeError, Exception)):  # frozen dataclass raises FrozenInstanceError
        nt.caption = "mutated"


def test_cell_ref_self_ref_format():
    cr = CellRef(table_index=3, row_idx=5, col_idx=2, self_ref="#/tables/3/data/table_cells/17")
    assert cr.self_ref.startswith("#/tables/")
    assert cr.self_ref.endswith("/17")


def test_graph_table_chunk_carries_chunk_kind():
    gtc = GraphTableChunk(
        text="...", table_ref="#/tables/3", page_numbers=(6,),
        chunk_kind=ChunkKind.TABLE_ENTITY_COLUMN,
        entity_display_name="S-75M2", section=None, column_index=7,
        cell_refs=("#/tables/3/data/table_cells/42",),
        row_labels=("Max Range",),
    )
    assert gtc.chunk_kind == ChunkKind.TABLE_ENTITY_COLUMN
    assert "S-75M2" in gtc.entity_display_name
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_models.py -v
```

Expected: `ImportError` / `ModuleNotFoundError` because `app.services.table_normalization.models` doesn't exist.

- [ ] **Step 3: Implement `app/services/table_normalization/models.py`**

```python
"""Frozen dataclasses and enums for the table normalization layer.

The NormalizedTable model is the only contract between normalization
(normalize.py) and the renderers (render_graph.py, render_embedding.py).
All types are frozen — immutable post-construction — so renderers can't
mutate state across calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Shape(str, Enum):
    COLUMN_MAJOR = "column_major"
    ROW_MAJOR = "row_major"
    HYBRID = "hybrid"          # column-major + multiple identity rows
    OTHER = "other"            # skip; fall back to raw rendering


class ChunkKind(str, Enum):
    TABLE_SUMMARY = "table_summary"
    TABLE_WHOLE = "table_whole"
    TABLE_ENTITY_COLUMN = "table_entity_column"
    TABLE_ENTITY_SECTION = "table_entity_section"


@dataclass(frozen=True)
class CellRef:
    table_index: int
    row_idx: int
    col_idx: int
    self_ref: str              # "#/tables/3/data/table_cells/42"


@dataclass(frozen=True)
class NormalizedCell:
    row_idx: int
    col_idx: int
    row_label: str | None
    column_identity: dict[str, str]
    section: str | None
    value: str                 # raw text; no numeric coercion
    unit: str | None
    cell_ref: CellRef


@dataclass(frozen=True)
class NormalizedRow:
    row_idx: int
    label: str
    is_identity_row: bool
    is_section_header: bool
    section: str | None
    unit: str | None


@dataclass(frozen=True)
class NormalizedColumn:
    col_idx: int
    identity: dict[str, str]
    display_name: str          # heuristic: Industry → Military → NATO → Missile Type; fallback "col-{n}"


@dataclass(frozen=True)
class TableSection:
    name: str
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class NormalizedTable:
    table_index: int
    self_ref: str
    caption: str | None
    page_numbers: tuple[int, ...]
    shape: Shape
    rows: tuple[NormalizedRow, ...]
    columns: tuple[NormalizedColumn, ...]
    sections: tuple[TableSection, ...]
    cells: tuple[NormalizedCell, ...]
    raw_markdown: str          # source resolution per §8 step 7


@dataclass(frozen=True)
class GraphTableChunk:
    text: str
    table_ref: str
    page_numbers: tuple[int, ...]
    chunk_kind: ChunkKind
    entity_display_name: str | None
    section: str | None
    column_index: int | None
    cell_refs: tuple[str, ...]
    row_labels: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingTableChunk:
    text: str
    table_ref: str
    page_numbers: tuple[int, ...]
    chunk_kind: ChunkKind
    entity_display_name: str | None
    section: str | None
    column_index: int | None
    cell_refs: tuple[str, ...]
    row_labels: tuple[str, ...]
```

- [ ] **Step 4: Add skeleton `__init__.py`**

```python
# app/services/table_normalization/__init__.py
"""Table normalization layer — see docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md."""
from app.services.table_normalization.models import (
    Shape,
    ChunkKind,
    CellRef,
    NormalizedCell,
    NormalizedRow,
    NormalizedColumn,
    TableSection,
    NormalizedTable,
    GraphTableChunk,
    EmbeddingTableChunk,
)

__all__ = [
    "Shape",
    "ChunkKind",
    "CellRef",
    "NormalizedCell",
    "NormalizedRow",
    "NormalizedColumn",
    "TableSection",
    "NormalizedTable",
    "GraphTableChunk",
    "EmbeddingTableChunk",
]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_models.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/table_normalization/ tests/unit/test_table_normalization_models.py
git commit -m "feat(table-norm): models — frozen dataclasses + enums

NormalizedTable model is the contract between normalization and the two
renderers. All types frozen (immutable). Empty __init__ exports for
downstream tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2: Shape detection (`detect.py`)

**Files:**
- Create: `app/services/table_normalization/detect.py`
- Create: `tests/unit/test_table_normalization_detect.py`
- Create: `tests/fixtures/sa2_sample_table.json` (small SA-2-shaped table for snapshot tests)

- [ ] **Step 1: Build the test fixture `tests/fixtures/sa2_sample_table.json`**

A minimal column-major hybrid table mirroring the SA-2 variants structure: 8 rows × 5 cols (cuts down from the real ~23×12). Includes:
- Row 0–2: identity rows (`Industry Designation`, `Military Designation`, `NATO Designation`)
- Row 3: section header `1st Stage` (full-width span)
- Row 4–5: spec rows under section
- Row 6: section header `2nd Stage`
- Row 7: spec row

Each cell has the Docling shape: `{"text": "...", "row_header": bool, "column_header": bool, "start_row_offset_idx": N, "end_row_offset_idx": N, "start_col_offset_idx": M, "end_col_offset_idx": M}`.

Save under `tests/fixtures/sa2_sample_table.json`.

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_table_normalization_detect.py
import json
import pytest
from pathlib import Path
from app.services.table_normalization.detect import (
    detect_shape, SPEC_ROW_KEYWORDS, SECTION_KEYWORDS, IDENTITY_LABEL_KEYWORDS,
)
from app.services.table_normalization.models import Shape


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def test_sa2_fixture_detected_as_hybrid():
    shape = detect_shape(SA2_FIXTURE["table_cells"], SA2_FIXTURE)
    assert shape == Shape.HYBRID


def test_undersized_table_returns_other():
    tiny = {"table_cells": [
        {"text": "a", "row_header": True, "start_row_offset_idx": 0, "end_row_offset_idx": 0, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
    ]}
    assert detect_shape(tiny["table_cells"], tiny) == Shape.OTHER


def test_plain_column_major_table():
    # 5x5 with first column row_header, contains spec keyword
    cells = [
        {"text": "Max Range", "row_header": True, "start_row_offset_idx": 0, "end_row_offset_idx": 0, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Weight", "row_header": True, "start_row_offset_idx": 1, "end_row_offset_idx": 1, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Length", "row_header": True, "start_row_offset_idx": 2, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Diameter", "row_header": True, "start_row_offset_idx": 3, "end_row_offset_idx": 3, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Speed", "row_header": True, "start_row_offset_idx": 4, "end_row_offset_idx": 4, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
    ]
    for col in range(1, 5):
        for row in range(5):
            cells.append({
                "text": f"v{row}{col}", "row_header": False,
                "start_row_offset_idx": row, "end_row_offset_idx": row,
                "start_col_offset_idx": col, "end_col_offset_idx": col,
            })
    table_data = {"num_rows": 5, "num_cols": 5, "table_cells": cells}
    assert detect_shape(cells, table_data) == Shape.COLUMN_MAJOR


def test_row_major_table():
    # 5x5 with first row column_header, contains spec keyword in row 0
    cells = []
    for col_idx, label in enumerate(["Variant", "Max Range", "Weight", "Length", "Speed"]):
        cells.append({
            "text": label, "column_header": True,
            "start_row_offset_idx": 0, "end_row_offset_idx": 0,
            "start_col_offset_idx": col_idx, "end_col_offset_idx": col_idx,
        })
    for row in range(1, 5):
        for col in range(5):
            cells.append({
                "text": f"v{row}{col}", "column_header": False, "row_header": False,
                "start_row_offset_idx": row, "end_row_offset_idx": row,
                "start_col_offset_idx": col, "end_col_offset_idx": col,
            })
    table_data = {"num_rows": 5, "num_cols": 5, "table_cells": cells}
    assert detect_shape(cells, table_data) == Shape.ROW_MAJOR


def test_keyword_lists_are_frozensets():
    assert isinstance(SPEC_ROW_KEYWORDS, frozenset)
    assert isinstance(SECTION_KEYWORDS, frozenset)
    assert isinstance(IDENTITY_LABEL_KEYWORDS, frozenset)
    assert "max range" in SPEC_ROW_KEYWORDS
    assert "1st stage" in SECTION_KEYWORDS
    assert "nato designation" in IDENTITY_LABEL_KEYWORDS


def test_malformed_cells_return_other_not_crash():
    bad = {"table_cells": [{"text": "x"}]}  # missing all offset fields
    # Must not raise; return OTHER
    assert detect_shape(bad["table_cells"], bad) == Shape.OTHER
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_detect.py -v
```

Expected: ImportError on `app.services.table_normalization.detect`.

- [ ] **Step 4: Implement `app/services/table_normalization/detect.py`**

```python
"""Shape detection — pure heuristic over Docling table_cells.

Implements the rules per §7 of the design spec. No LLM, no external state.
Returns Shape; emits diagnostics dict separately for caller logging."""
from __future__ import annotations

import logging
from typing import Any
from app.services.table_normalization.models import Shape

logger = logging.getLogger(__name__)


SPEC_ROW_KEYWORDS: frozenset[str] = frozenset({
    "max range", "min range", "range", "max altitude", "min altitude", "altitude",
    "max speed", "min speed", "speed", "velocity", "vmax", "vmin",
    "weight", "mass", "total weight", "warhead weight",
    "length", "width", "diameter", "span", "height",
    "max alt", "min alt",
    "missile type", "missile variant",
    "frequency", "wavelength", "power",
    "thrust", "burn time", "stage",
})

SECTION_KEYWORDS: frozenset[str] = frozenset({
    "missile", "1st stage", "2nd stage", "first stage", "second stage",
    "booster", "sustainer", "propulsion",
    "radar", "launcher", "guidance",
    "warhead", "fuze",
    "system performance", "performance",
})

IDENTITY_LABEL_KEYWORDS: frozenset[str] = frozenset({
    "designation", "variant", "type", "name",
    "industry designation", "military designation", "nato designation",
    "fan song variant", "radar variant",
    "system name", "system designation",
})


_MIN_DIM = 4   # floor: <4 rows or <4 cols → OTHER


def _safe_get_int(d: dict, key: str, default: int = -1) -> int:
    v = d.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _cells_at_col_zero(cells: list[dict]) -> list[dict]:
    return [c for c in cells if _safe_get_int(c, "start_col_offset_idx") == 0 and (c.get("text") or "").strip()]


def _cells_at_row_zero(cells: list[dict]) -> list[dict]:
    return [c for c in cells if _safe_get_int(c, "start_row_offset_idx") == 0 and (c.get("text") or "").strip()]


def _num_rows(cells: list[dict]) -> int:
    if not cells:
        return 0
    return max(_safe_get_int(c, "end_row_offset_idx", 0) for c in cells) + 1


def _num_cols(cells: list[dict]) -> int:
    if not cells:
        return 0
    return max(_safe_get_int(c, "end_col_offset_idx", 0) for c in cells) + 1


def _has_spec_keyword(texts: list[str]) -> bool:
    for t in texts:
        if any(kw in t.lower() for kw in SPEC_ROW_KEYWORDS):
            return True
    return False


def _is_identity_shaped(text: str) -> bool:
    """Short, non-blank, non-numeric — looks like a variant designation."""
    text = text.strip()
    if not text or len(text) >= 40:
        return False
    try:
        float(text.replace(",", "").replace(" ", ""))
        return False  # numeric
    except ValueError:
        return True


def detect_shape(table_cells: list[dict], table_data: dict) -> Shape:
    """Return the Shape classification of a Docling table.

    See §7 of the design spec for the decision rules.
    """
    try:
        if not table_cells:
            return Shape.OTHER
        if _num_rows(table_cells) < _MIN_DIM or _num_cols(table_cells) < _MIN_DIM:
            return Shape.OTHER

        # Test 2: COLUMN_MAJOR
        col0 = _cells_at_col_zero(table_cells)
        if col0:
            row_header_share = sum(1 for c in col0 if c.get("row_header")) / len(col0)
            if row_header_share >= 0.5 and _has_spec_keyword([c.get("text") or "" for c in col0]):
                # Test 3: HYBRID upgrade — count identity rows at top
                identity_row_count = 0
                for row_idx in range(_num_rows(table_cells)):
                    data_cells_in_row = [
                        c for c in table_cells
                        if _safe_get_int(c, "start_row_offset_idx") == row_idx
                        and _safe_get_int(c, "start_col_offset_idx") > 0
                    ]
                    if not data_cells_in_row:
                        break
                    if all(_is_identity_shaped(c.get("text") or "") for c in data_cells_in_row):
                        identity_row_count += 1
                    else:
                        break
                if identity_row_count >= 2:
                    return Shape.HYBRID
                return Shape.COLUMN_MAJOR

        # Test 4: ROW_MAJOR
        row0 = _cells_at_row_zero(table_cells)
        if row0:
            col_header_share = sum(1 for c in row0 if c.get("column_header")) / len(row0)
            if col_header_share >= 0.5 and _has_spec_keyword([c.get("text") or "" for c in row0]):
                return Shape.ROW_MAJOR

        # Operational signal: ≥4×4 table that didn't classify
        rows, cols = _num_rows(table_cells), _num_cols(table_cells)
        if rows >= _MIN_DIM and cols >= _MIN_DIM:
            logger.warning(
                "table_normalization.detect: %dx%d table fell to OTHER. "
                "Row-0 headers=%d, col-0 row_headers=%d. Consider adding row labels "
                "to SPEC_ROW_KEYWORDS or column headers to row-major detection.",
                rows, cols, len(row0), sum(1 for c in col0 if c.get("row_header")) if col0 else 0,
            )
        return Shape.OTHER

    except Exception as exc:
        logger.warning("table_normalization.detect: exception during shape detection: %s; returning OTHER", exc)
        return Shape.OTHER
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_detect.py -v
```

Expected: all 6 tests PASS. If any fail, iterate the implementation until green.

- [ ] **Step 6: Commit**

```bash
git add app/services/table_normalization/detect.py tests/unit/test_table_normalization_detect.py tests/fixtures/sa2_sample_table.json
git commit -m "feat(table-norm): detect — shape classification heuristic

Pure function detect_shape(table_cells, table_data) -> Shape. Closed
keyword frozensets (no 'etc.' placeholders). 4x4 floor; OTHER fallback
is operational signal at WARNING level for tables ≥4x4 that didn't
classify. SA-2 hybrid fixture committed for snapshot regression.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3: Normalization pipeline (`normalize.py`)

**Files:**
- Create: `app/services/table_normalization/normalize.py`
- Create: `tests/unit/test_table_normalization_normalize.py`
- Modify: `app/services/table_normalization/__init__.py` (add `normalize_tables` export)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_table_normalization_normalize.py
import json
import pytest
from pathlib import Path
from app.services.table_normalization import normalize_tables
from app.services.table_normalization.models import Shape


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _wrap_in_doc_json(table_fixture: dict, table_idx: int = 0) -> dict:
    """Wrap a single-table fixture as a minimal doc_json."""
    return {
        "tables": [table_fixture],
        "texts": [],
    }


def test_normalize_sa2_returns_hybrid():
    doc = _wrap_in_doc_json(SA2_FIXTURE)
    result = normalize_tables(doc)
    assert len(result) == 1
    nt = result[0]
    assert nt.shape == Shape.HYBRID
    assert nt.table_index == 0
    assert nt.self_ref == "#/tables/0"


def test_normalize_other_table_carries_empty_cells():
    doc = _wrap_in_doc_json({"table_cells": [{"text": "x"}], "text": "raw"})
    result = normalize_tables(doc)
    assert len(result) == 1
    nt = result[0]
    assert nt.shape == Shape.OTHER
    assert nt.cells == ()
    assert nt.raw_markdown == "raw"


def test_normalize_does_not_mutate_doc_json():
    doc = _wrap_in_doc_json(SA2_FIXTURE)
    snapshot = json.dumps(doc, sort_keys=True)
    normalize_tables(doc)
    after = json.dumps(doc, sort_keys=True)
    assert snapshot == after


def test_normalize_skips_empty_cells():
    """Per §8 step 5 — empty cell values are dropped from NormalizedCells."""
    doc = _wrap_in_doc_json(SA2_FIXTURE)
    result = normalize_tables(doc)
    nt = result[0]
    assert all(c.value.strip() for c in nt.cells)


def test_normalize_extracts_units_from_row_labels():
    """A row labeled 'Max Range (m)' should yield NormalizedRow.unit == 'm'."""
    # Build a fixture with explicit (m) suffix
    cells = []
    cells.append({
        "text": "Max Range (m)", "row_header": True,
        "start_row_offset_idx": 0, "end_row_offset_idx": 0,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
    })
    # Fill more rows + cols to clear the 4x4 floor
    for r in range(4):
        cells.append({
            "text": f"Row{r}", "row_header": True,
            "start_row_offset_idx": r, "end_row_offset_idx": r,
            "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        })
    for r in range(4):
        for c in range(1, 5):
            cells.append({
                "text": f"v{r}{c}", "row_header": False,
                "start_row_offset_idx": r, "end_row_offset_idx": r,
                "start_col_offset_idx": c, "end_col_offset_idx": c,
            })
    table = {"table_cells": cells, "num_rows": 4, "num_cols": 5}
    doc = _wrap_in_doc_json(table)
    result = normalize_tables(doc)
    nt = result[0]
    # Find the Max Range row
    matches = [r for r in nt.rows if "Max Range" in (r.label or "")]
    assert matches, f"no 'Max Range' row found; rows={[r.label for r in nt.rows]}"
    assert matches[0].unit == "m"


def test_normalize_continues_on_per_table_failure():
    """One bad table doesn't stop other tables from being normalized."""
    doc = {
        "tables": [
            SA2_FIXTURE,                  # good
            None,                          # corrupt — will raise inside per-table loop
            SA2_FIXTURE,                  # good again
        ],
        "texts": [],
    }
    result = normalize_tables(doc)
    assert len(result) == 3
    assert result[0].shape == Shape.HYBRID
    assert result[1].shape == Shape.OTHER
    assert result[2].shape == Shape.HYBRID
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_normalize.py -v
```

Expected: ImportError (`normalize_tables` not exported yet).

- [ ] **Step 3: Implement `app/services/table_normalization/normalize.py`**

Implement the per-table pipeline per §8 of the spec. Key functions:
- `normalize_tables(doc_json) -> list[NormalizedTable]` — public entry.
- `_per_table(doc_json, table_index) -> NormalizedTable` — single-table pipeline, wrapped in try/except.
- `_build_rows(cells, shape) -> tuple[NormalizedRow, ...]`
- `_build_columns(cells, rows, shape) -> tuple[NormalizedColumn, ...]`
- `_assign_sections(rows) -> tuple[NormalizedRow, ...]` (returns updated rows with `section` set)
- `_build_cells(cells, rows, columns) -> tuple[NormalizedCell, ...]`
- `_extract_unit(label) -> str | None` (regex `r"\(\s*([a-zA-Z/°²³]+)\s*\)\s*$"`)
- `_resolve_raw_markdown(doc_json, table_index) -> str` (per §8 step 7 lookup rule)
- `_display_name_for_column(identity_dict, col_idx) -> str` (heuristic chain)

```python
# app/services/table_normalization/normalize.py
"""Normalize Docling tables into the shared NormalizedTable model.

Pure function: reads doc_json['tables']; never writes doc_json.
Per-table exceptions are caught; one bad table doesn't break others.
See §8 of the design spec."""
from __future__ import annotations

import logging
import re
from typing import Any
from app.services.table_normalization.detect import (
    detect_shape, IDENTITY_LABEL_KEYWORDS, SECTION_KEYWORDS, SPEC_ROW_KEYWORDS,
)
from app.services.table_normalization.models import (
    Shape, CellRef, NormalizedCell, NormalizedRow, NormalizedColumn,
    TableSection, NormalizedTable,
)

logger = logging.getLogger(__name__)


_UNIT_RE = re.compile(r"\(\s*([a-zA-Z/°²³]+)\s*\)\s*$")
_DISPLAY_NAME_PREFERENCE = (
    "industry designation", "military designation",
    "nato designation", "missile type",
)


def normalize_tables(doc_json: dict) -> list[NormalizedTable]:
    """Public entry. Returns one NormalizedTable per doc_json['tables'] entry."""
    tables = (doc_json or {}).get("tables") or []
    return [_per_table_safe(doc_json, i) for i in range(len(tables))]


def _per_table_safe(doc_json: dict, table_index: int) -> NormalizedTable:
    try:
        return _per_table(doc_json, table_index)
    except Exception as exc:
        logger.warning(
            "table_normalization.normalize: table %d failed (%s); returning OTHER",
            table_index, exc,
        )
        return _empty_normalized(doc_json, table_index)


def _per_table(doc_json: dict, table_index: int) -> NormalizedTable:
    table = doc_json["tables"][table_index]
    if not isinstance(table, dict):
        raise ValueError(f"table {table_index} is not a dict")
    cells = table.get("data", {}).get("table_cells") or table.get("table_cells") or []
    raw_md = _resolve_raw_markdown(doc_json, table_index, table)
    page_numbers = _resolve_page_numbers(table)
    caption = table.get("caption") or table.get("data", {}).get("caption")
    self_ref = f"#/tables/{table_index}"

    shape = detect_shape(cells, table)
    if shape == Shape.OTHER:
        return NormalizedTable(
            table_index=table_index, self_ref=self_ref, caption=caption,
            page_numbers=page_numbers, shape=Shape.OTHER,
            rows=(), columns=(), sections=(), cells=(), raw_markdown=raw_md,
        )

    rows = _build_rows(cells, shape)
    rows = _assign_sections(rows)
    columns = _build_columns(cells, rows, shape, table_index)
    sections = _build_sections(rows)
    norm_cells = _build_cells(cells, rows, columns, table_index, sections)

    return NormalizedTable(
        table_index=table_index, self_ref=self_ref, caption=caption,
        page_numbers=page_numbers, shape=shape, rows=rows, columns=columns,
        sections=sections, cells=norm_cells, raw_markdown=raw_md,
    )


def _empty_normalized(doc_json: dict, table_index: int) -> NormalizedTable:
    raw_md = ""
    try:
        table = doc_json["tables"][table_index] if isinstance(doc_json.get("tables", [None] * (table_index + 1))[table_index], dict) else None
        if table:
            raw_md = _resolve_raw_markdown(doc_json, table_index, table)
    except Exception:
        pass
    return NormalizedTable(
        table_index=table_index, self_ref=f"#/tables/{table_index}",
        caption=None, page_numbers=(), shape=Shape.OTHER,
        rows=(), columns=(), sections=(), cells=(), raw_markdown=raw_md,
    )


def _resolve_raw_markdown(doc_json: dict, table_index: int, table: dict) -> str:
    """Per §8 step 7 lookup rule."""
    # First preference: text item where prov[0].$ref matches this table
    target_ref = f"#/tables/{table_index}"
    for t in doc_json.get("texts") or []:
        prov = t.get("prov") or []
        if prov and isinstance(prov, list):
            first = prov[0] if isinstance(prov[0], dict) else None
            if first and first.get("$ref") == target_ref:
                txt = t.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt
    # Second preference
    txt = table.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt
    # Final fallback
    md = (table.get("data") or {}).get("table_markdown") or ""
    if md.strip():
        return md
    logger.debug("table_normalization.normalize: no raw_markdown source found for table %d", table_index)
    return ""


def _resolve_page_numbers(table: dict) -> tuple[int, ...]:
    pages: set[int] = set()
    for p in (table.get("prov") or []):
        page = p.get("page_no") if isinstance(p, dict) else None
        if isinstance(page, int):
            pages.add(page)
    return tuple(sorted(pages))


def _extract_unit(label: str) -> str | None:
    if not label:
        return None
    m = _UNIT_RE.search(label)
    return m.group(1) if m else None


def _is_identity_label(label: str) -> bool:
    norm = (label or "").strip().lower()
    return any(kw in norm for kw in IDENTITY_LABEL_KEYWORDS)


def _is_section_header_cell(cell: dict, num_cols: int) -> bool:
    span = (cell.get("end_col_offset_idx", 0) - cell.get("start_col_offset_idx", 0)) + 1
    text = (cell.get("text") or "").strip().lower()
    if span < max(2, num_cols - 1):  # must span most of the table
        return False
    return any(kw in text for kw in SECTION_KEYWORDS)


def _build_rows(cells: list[dict], shape: Shape) -> tuple[NormalizedRow, ...]:
    """Return per-row metadata. Section context is assigned in _assign_sections."""
    if not cells:
        return ()
    num_rows = max((c.get("end_row_offset_idx", 0) for c in cells), default=-1) + 1
    num_cols = max((c.get("end_col_offset_idx", 0) for c in cells), default=-1) + 1

    rows: list[NormalizedRow] = []
    for r in range(num_rows):
        # The label cell is at (row=r, col=0)
        label_cell = next(
            (c for c in cells if c.get("start_row_offset_idx") == r and c.get("start_col_offset_idx") == 0),
            None,
        )
        label = (label_cell.get("text") if label_cell else "") or ""
        is_section = bool(label_cell and _is_section_header_cell(label_cell, num_cols))
        is_identity = (not is_section) and _is_identity_label(label)
        unit = _extract_unit(label)
        rows.append(NormalizedRow(
            row_idx=r, label=label.strip(),
            is_identity_row=is_identity, is_section_header=is_section,
            section=None, unit=unit,
        ))
    return tuple(rows)


def _assign_sections(rows: tuple[NormalizedRow, ...]) -> tuple[NormalizedRow, ...]:
    """Walk top-to-bottom; section-header rows reset the section context."""
    current: str | None = None
    out: list[NormalizedRow] = []
    for r in rows:
        if r.is_section_header:
            current = r.label
            out.append(r)  # section_header itself has no section
        else:
            out.append(NormalizedRow(
                row_idx=r.row_idx, label=r.label,
                is_identity_row=r.is_identity_row, is_section_header=False,
                section=current, unit=r.unit,
            ))
    return tuple(out)


def _build_sections(rows: tuple[NormalizedRow, ...]) -> tuple[TableSection, ...]:
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for r in rows:
        if r.section is None or r.is_section_header:
            continue
        if r.section not in grouped:
            grouped[r.section] = []
            order.append(r.section)
        grouped[r.section].append(r.row_idx)
    return tuple(TableSection(name=name, row_indices=tuple(grouped[name])) for name in order)


def _build_columns(
    cells: list[dict], rows: tuple[NormalizedRow, ...], shape: Shape, table_index: int
) -> tuple[NormalizedColumn, ...]:
    """For column-major/hybrid: each non-label column is one entity.
    Identity is the dict of (identity_row_label, cell_value) at that column.
    """
    num_cols = max((c.get("end_col_offset_idx", 0) for c in cells), default=-1) + 1
    identity_rows = [r for r in rows if r.is_identity_row]
    columns: list[NormalizedColumn] = []
    for col_idx in range(1, num_cols):
        identity: dict[str, str] = {}
        for irow in identity_rows:
            cell = next(
                (c for c in cells
                 if c.get("start_row_offset_idx") == irow.row_idx
                 and c.get("start_col_offset_idx") <= col_idx <= c.get("end_col_offset_idx", col_idx)),
                None,
            )
            if cell:
                val = (cell.get("text") or "").strip()
                if val:
                    identity[irow.label] = val
        display = _display_name_for_column(identity, col_idx)
        columns.append(NormalizedColumn(col_idx=col_idx, identity=identity, display_name=display))
    return tuple(columns)


def _display_name_for_column(identity: dict[str, str], col_idx: int) -> str:
    norm = {k.lower(): v for k, v in identity.items()}
    for pref in _DISPLAY_NAME_PREFERENCE:
        for k, v in norm.items():
            if pref in k:
                return v
    return f"col-{col_idx}"


def _build_cells(
    cells: list[dict], rows: tuple[NormalizedRow, ...],
    columns: tuple[NormalizedColumn, ...], table_index: int,
    sections: tuple[TableSection, ...],
) -> tuple[NormalizedCell, ...]:
    """One cell per (spec row × entity column) pair. Empties skipped."""
    out: list[NormalizedCell] = []
    spec_rows = [r for r in rows if not r.is_identity_row and not r.is_section_header]
    for row in spec_rows:
        for col in columns:
            cell = next(
                (c for c in cells
                 if c.get("start_row_offset_idx") == row.row_idx
                 and c.get("start_col_offset_idx") <= col.col_idx <= c.get("end_col_offset_idx", col.col_idx)),
                None,
            )
            if not cell:
                continue
            value = (cell.get("text") or "").strip()
            if not value:
                continue
            cell_pos = cells.index(cell)
            ref = CellRef(
                table_index=table_index, row_idx=row.row_idx, col_idx=col.col_idx,
                self_ref=f"#/tables/{table_index}/data/table_cells/{cell_pos}",
            )
            out.append(NormalizedCell(
                row_idx=row.row_idx, col_idx=col.col_idx,
                row_label=row.label, column_identity=col.identity,
                section=row.section, value=value, unit=row.unit, cell_ref=ref,
            ))
    return tuple(out)
```

Add `from app.services.table_normalization.normalize import normalize_tables` to `__init__.py` and append `"normalize_tables"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_normalize.py -v
```

Expected: all 6 tests PASS. Iterate the implementation if any fail.

- [ ] **Step 5: Commit**

```bash
git add app/services/table_normalization/normalize.py app/services/table_normalization/__init__.py tests/unit/test_table_normalization_normalize.py
git commit -m "feat(table-norm): normalize_tables — Docling cells → NormalizedTable

Pure function; reads doc_json['tables'] without mutation. Per-table
try/except guard — one bad table doesn't break the pipeline. Empty
cells skipped; units extracted from label suffix regex; raw_markdown
resolved per §8 step 7 lookup rule.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 4: Tokenizer (`tokens.py`)

**Files:**
- Create: `app/services/table_normalization/tokens.py`
- Create: `tests/unit/test_table_normalization_tokens.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_table_normalization_tokens.py
import pytest
from app.services.table_normalization.tokens import count_bge_m3_tokens


def test_empty_string_zero_tokens():
    assert count_bge_m3_tokens("") == 0


def test_short_text_is_under_limit():
    n = count_bge_m3_tokens("Hello world")
    assert 0 < n < 10


def test_long_text_exceeds_512():
    long = ("Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 200)
    n = count_bge_m3_tokens(long)
    assert n > 512


def test_repeated_calls_use_cached_tokenizer():
    """Second call should not re-load the tokenizer from disk."""
    count_bge_m3_tokens("warmup")
    # If we didn't crash, the cached path is reached.
    n1 = count_bge_m3_tokens("once more")
    n2 = count_bge_m3_tokens("once more")
    assert n1 == n2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_tokens.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `tokens.py`**

```python
"""bge-m3 tokenizer wrapper — lazy load, module-level cache.

The chunker in pipeline.py:5524-5525 uses bge-m3 with max_tokens=512.
We use the same tokenizer for size measurements so render-side budgeting
matches chunker-side budgeting."""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_TOKENIZER_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _tokenizer():
    """Lazy-load + cache the HF tokenizer."""
    from transformers import AutoTokenizer
    logger.info("table_normalization.tokens: loading tokenizer %s (first call only)", _TOKENIZER_NAME)
    return AutoTokenizer.from_pretrained(_TOKENIZER_NAME)


def count_bge_m3_tokens(text: str) -> int:
    """Return the bge-m3 token count of `text`. Empty string → 0."""
    if not text:
        return 0
    tok = _tokenizer()
    # Use encode without special tokens to get a clean count
    return len(tok.encode(text, add_special_tokens=False))
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_tokens.py -v
```

Expected: all 4 PASS. First call loads the tokenizer (slow); subsequent calls are fast.

- [ ] **Step 5: Commit**

```bash
git add app/services/table_normalization/tokens.py tests/unit/test_table_normalization_tokens.py
git commit -m "feat(table-norm): tokens — bge-m3 tokenizer wrapper

Module-level @lru_cache(maxsize=1) ensures the tokenizer loads once
per process. Matches the chunker's tokenizer choice at pipeline.py:5524
so render-side budgets align with chunker-side budgets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 5: Configuration (`config.py` + .env files + app/config.py)

**Files:**
- Create: `app/services/table_normalization/config.py`
- Modify: `app/config.py` (add settings fields for new env vars)
- Modify: `.env` (add 8 new vars with defaults)
- Modify: `.env.example` (add 8 new vars with defaults + comments)
- Create: `tests/unit/test_table_normalization_config.py`

- [ ] **Step 1: Inventory the 8 new env vars**

From spec §12:
1. `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED` (default `false`)
2. `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS` (default `false`)
3. `DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN` (default `true`)
4. `DOCLING_GRAPH_TABLE_WHOLE_LIMIT` (default `1500`)
5. `DOCLING_GRAPH_TABLE_COLUMN_LIMIT` (default `1200`)
6. `EMBEDDING_TABLE_NORMALIZATION_ENABLED` (default `false`)
7. `EMBEDDING_TABLE_SUMMARY_MAX_TOKENS` (default `300`)
8. `MIN_TABLE_NORMALIZATION_TOKENS` (default `256`)

`EMBEDDING_CHUNK_MAX_TOKENS` already exists at `app/config.py:400` (reused, not new).

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_table_normalization_config.py
import os
import pytest
from app.services.table_normalization.config import (
    is_table_normalization_enabled_graph,
    is_table_normalization_enabled_embedding,
    is_experimental_table_facts_enabled,
    is_suppress_raw_table_markdown_enabled,
    table_whole_limit,
    table_column_limit,
    min_table_normalization_tokens,
    embedding_chunk_max_tokens,
    embedding_table_summary_max_tokens,
)


def test_defaults_are_off_for_master_switches(monkeypatch):
    for v in (
        "DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED",
        "DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS",
        "EMBEDDING_TABLE_NORMALIZATION_ENABLED",
    ):
        monkeypatch.delenv(v, raising=False)
    assert is_table_normalization_enabled_graph() is False
    assert is_experimental_table_facts_enabled() is False
    assert is_table_normalization_enabled_embedding() is False


def test_suppress_default_true(monkeypatch):
    monkeypatch.delenv("DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN", raising=False)
    assert is_suppress_raw_table_markdown_enabled() is True


def test_threshold_defaults(monkeypatch):
    for v in (
        "DOCLING_GRAPH_TABLE_WHOLE_LIMIT",
        "DOCLING_GRAPH_TABLE_COLUMN_LIMIT",
        "MIN_TABLE_NORMALIZATION_TOKENS",
        "EMBEDDING_TABLE_SUMMARY_MAX_TOKENS",
    ):
        monkeypatch.delenv(v, raising=False)
    assert table_whole_limit() == 1500
    assert table_column_limit() == 1200
    assert min_table_normalization_tokens() == 256
    assert embedding_table_summary_max_tokens() == 300


def test_flags_respond_to_env(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "true")
    assert is_table_normalization_enabled_graph() is True
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "false")
    assert is_table_normalization_enabled_graph() is False


def test_embedding_chunk_max_tokens_reuses_existing(monkeypatch):
    # Confirms we read from the same settings field as the rest of the pipeline.
    monkeypatch.setenv("EMBEDDING_CHUNK_MAX_TOKENS", "999")
    # Need to force settings reload — but the test mainly asserts the function
    # exists and returns an int. The actual reload mechanism is test-env dependent.
    n = embedding_chunk_max_tokens()
    assert isinstance(n, int)
    assert n > 0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_config.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `app/services/table_normalization/config.py`**

```python
"""Env-var reading for the table normalization layer.

Single source of truth for flag names and defaults. All readers check
the env var on every call — no module-level caching of values — so
runtime flag flips (per the §13 rollout) take effect immediately."""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("table_normalization.config: %s=%r is not an int; using default %d", name, raw, default)
        return default


def is_table_normalization_enabled_graph() -> bool:
    return _env_bool("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", False)


def is_table_normalization_enabled_embedding() -> bool:
    return _env_bool("EMBEDDING_TABLE_NORMALIZATION_ENABLED", False)


def is_experimental_table_facts_enabled() -> bool:
    return _env_bool("DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS", False)


def is_suppress_raw_table_markdown_enabled() -> bool:
    return _env_bool("DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN", True)


def table_whole_limit() -> int:
    return _env_int("DOCLING_GRAPH_TABLE_WHOLE_LIMIT", 1500)


def table_column_limit() -> int:
    return _env_int("DOCLING_GRAPH_TABLE_COLUMN_LIMIT", 1200)


def min_table_normalization_tokens() -> int:
    return _env_int("MIN_TABLE_NORMALIZATION_TOKENS", 256)


def embedding_chunk_max_tokens() -> int:
    # Reuses the existing field at app/config.py:400 (embedding_chunk_max_tokens=512).
    # We re-read the env var directly here for consistency with our other flag helpers.
    return _env_int("EMBEDDING_CHUNK_MAX_TOKENS", 512)


def embedding_table_summary_max_tokens() -> int:
    return _env_int("EMBEDDING_TABLE_SUMMARY_MAX_TOKENS", 300)
```

- [ ] **Step 5: Add settings fields to `app/config.py`**

After the existing `embedding_chunk_max_tokens: int = 512` (line 400), append:

```python
# Table normalization (spec 2026-05-11-table-aware-chunking).
# Master switches default FALSE — code merges without changing behavior;
# Phase 2 flip activates the new path.
docling_graph_table_normalization_enabled: bool = False
docling_graph_use_experimental_table_facts: bool = False
docling_graph_suppress_raw_table_markdown: bool = True
docling_graph_table_whole_limit: int = 1500
docling_graph_table_column_limit: int = 1200
embedding_table_normalization_enabled: bool = False
embedding_table_summary_max_tokens: int = 300
min_table_normalization_tokens: int = 256
```

(These settings entries are **documentation-only** for pydantic-settings discovery; they are NOT the runtime read path. All runtime flag checks go through `app/services/table_normalization/config.py`, which reads `os.environ.get` directly so values stay fresh across the rollout flag flip without a process restart. Do not change the runtime read path to use `settings.*`.)

- [ ] **Step 6: Add 8 new vars to `.env`**

Append to `.env`:

```bash
# --- Table normalization (spec 2026-05-11) ---
# Master switches default FALSE. Phase 2 flip activates the new path.
DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=false
DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=false
DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN=true
DOCLING_GRAPH_TABLE_WHOLE_LIMIT=1500
DOCLING_GRAPH_TABLE_COLUMN_LIMIT=1200
EMBEDDING_TABLE_NORMALIZATION_ENABLED=false
EMBEDDING_TABLE_SUMMARY_MAX_TOKENS=300
MIN_TABLE_NORMALIZATION_TOKENS=256
```

- [ ] **Step 7: Add 8 new vars to `.env.example`**

Append to `.env.example`:

```bash
# --- Table normalization (spec 2026-05-11) ---
# Graph-side master switch. Default false ships code without changing behavior; flip to true to activate per-entity-column chunks for the LLM.
DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=false
# Experimental: re-enable the reverted _table_facts.py path (mutually exclusive with normalization). For A/B comparison only.
DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=false
# When graph normalization is on, strip the raw flattened table text from texts[] so the LLM sees only normalized chunks.
DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN=true
# Token threshold; tables whose whole rendering is ≤ this emit one TABLE_WHOLE chunk; above, per-entity-column.
DOCLING_GRAPH_TABLE_WHOLE_LIMIT=1500
# Per-column rendering above this size splits by section, repeating the identity header.
DOCLING_GRAPH_TABLE_COLUMN_LIMIT=1200
# Embedding-side master switch. Default false ships code without changing behavior.
EMBEDDING_TABLE_NORMALIZATION_ENABLED=false
# Cap on the always-emitted TABLE_SUMMARY chunk size (bge-m3 tokens).
EMBEDDING_TABLE_SUMMARY_MAX_TOKENS=300
# Minimum bge-m3 token count for a normalized table to be substituted in the chunker; below this, native chunk passes through unchanged.
MIN_TABLE_NORMALIZATION_TOKENS=256
```

- [ ] **Step 8: Run tests**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_config.py -v
```

Expected: all 5 PASS.

- [ ] **Step 9: Verify env-var presence in both files**

```bash
for v in DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN DOCLING_GRAPH_TABLE_WHOLE_LIMIT DOCLING_GRAPH_TABLE_COLUMN_LIMIT EMBEDDING_TABLE_NORMALIZATION_ENABLED EMBEDDING_TABLE_SUMMARY_MAX_TOKENS MIN_TABLE_NORMALIZATION_TOKENS; do
    grep -q "^$v=" .env || echo "MISSING from .env: $v"
    grep -q "^$v=" .env.example || echo "MISSING from .env.example: $v"
done
```

Expected: no output (all 8 vars present in both files).

- [ ] **Step 10: Commit**

```bash
git add app/services/table_normalization/config.py app/config.py .env .env.example tests/unit/test_table_normalization_config.py
git commit -m "feat(table-norm): config — env-var helpers + .env / .env.example

8 new env vars (master switches default false). config.py reads env on
every call so runtime flips take effect. app/config.py adds matching
pydantic-settings fields for documentation. .env + .env.example
populated per feedback_env_vars_must_appear_in_dotenv_files.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 6: Shared column renderer (`_render_column_as_text`)

**Files:**
- Create stub: `app/services/table_normalization/render_graph.py` (export `_render_column_as_text` only at this point)
- Create: `tests/unit/test_render_column_byte_equality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_render_column_byte_equality.py
import json
from pathlib import Path
from app.services.table_normalization import normalize_tables


def test_render_column_as_text_returns_expected_format():
    """Snapshot test: SA-2 column 1 renders to a known string."""
    from app.services.table_normalization.render_graph import _render_column_as_text

    fixture = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    doc = {"tables": [fixture], "texts": []}
    nt = normalize_tables(doc)[0]
    assert len(nt.columns) >= 1

    text = _render_column_as_text(nt.columns[0], nt, nt.sections)

    # Format invariants per §9 of the spec
    assert "TABLE:" in text
    assert "ENTITY:" in text
    # Section names uppercased
    for s in nt.sections:
        assert s.name.upper() in text


def test_render_column_byte_identical_across_renderers():
    """Both renderers must produce byte-identical output for the same column."""
    from app.services.table_normalization.render_graph import _render_column_as_text as render_g
    from app.services.table_normalization.render_embedding import _render_column_as_text as render_e

    assert render_g is render_e  # same function object — single source of truth
```

- [ ] **Step 2: Run tests to verify failure**

Expected: ImportError on `render_graph` or `render_embedding`.

- [ ] **Step 3: Implement `_render_column_as_text` in `render_graph.py`**

```python
# app/services/table_normalization/render_graph.py
"""Graph-side renderer. Also exports _render_column_as_text as the
shared column-rendering helper used by both renderers (single source
of truth for chunk text format).
"""
from __future__ import annotations

from app.services.table_normalization.models import (
    NormalizedTable, NormalizedColumn, TableSection,
)


def _render_column_as_text(
    column: NormalizedColumn,
    table: NormalizedTable,
    sections: tuple[TableSection, ...],
) -> str:
    """Produce the identity+sections+rows block for one entity column.

    Both the graph and embedding renderers call this helper; their outputs
    differ only by what they wrap around this block. See §9 of the spec
    for the exact text format.
    """
    parts: list[str] = []

    # TABLE header
    caption = table.caption or table.self_ref
    parts.append(f"TABLE: {caption}")
    if table.page_numbers:
        parts.append(f"SOURCE: page {' '.join(str(p) for p in table.page_numbers)}")
    parts.append("")

    # ENTITY block — full identity dict
    parts.append("ENTITY:")
    for k, v in column.identity.items():
        parts.append(f"- {k}: {v}")
    parts.append("")

    # Section blocks
    spec_cells_by_section: dict[str | None, list[tuple[str, str, str | None]]] = {}
    for cell in table.cells:
        if cell.col_idx != column.col_idx:
            continue
        bucket = cell.section  # None == GENERAL
        spec_cells_by_section.setdefault(bucket, []).append(
            (cell.row_label or "", cell.value, cell.unit)
        )

    # Render GENERAL first, then named sections in document order
    if None in spec_cells_by_section:
        parts.append("GENERAL:")
        for label, value, unit in spec_cells_by_section[None]:
            parts.append(_render_row_line(label, value, unit))
        parts.append("")

    for section in sections:
        rows_for_section = spec_cells_by_section.get(section.name)
        if not rows_for_section:
            continue
        parts.append(f"{section.name.upper()}:")
        for label, value, unit in rows_for_section:
            parts.append(_render_row_line(label, value, unit))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _render_row_line(label: str, value: str, unit: str | None) -> str:
    if unit:
        return f"- {label}: {value} {unit}"
    return f"- {label}: {value}"
```

- [ ] **Step 4: Create stub `render_embedding.py` that re-imports `_render_column_as_text`**

```python
# app/services/table_normalization/render_embedding.py
"""Embedding-side renderer. Will be filled in Task 8.

Re-exports _render_column_as_text from render_graph so callers
on the embedding side import from the right module without coupling."""
from app.services.table_normalization.render_graph import _render_column_as_text

__all__ = ["_render_column_as_text"]
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/unit/test_render_column_byte_equality.py -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/table_normalization/render_graph.py app/services/table_normalization/render_embedding.py tests/unit/test_render_column_byte_equality.py
git commit -m "feat(table-norm): _render_column_as_text — shared chunk helper

Single source of truth for the identity+sections+rows block used by
both the graph and embedding renderers. render_embedding.py re-exports
it so consumers on each side import from their natural module; both
references resolve to the same function object (asserted by test).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 7: Graph renderer (`render_for_graph`)

**Files:**
- Modify: `app/services/table_normalization/render_graph.py` (add `render_for_graph`)
- Modify: `app/services/table_normalization/__init__.py` (export)
- Create: `tests/unit/test_table_normalization_render_graph.py`
- Create: `tests/fixtures/sa2_graph_chunks_expected.json` (snapshot)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_table_normalization_render_graph.py
import json
import pytest
from pathlib import Path
from app.services.table_normalization import normalize_tables, render_for_graph
from app.services.table_normalization.models import ChunkKind, Shape


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _doc(fixture):
    return {"tables": [fixture], "texts": []}


def test_other_table_emits_one_table_whole_chunk():
    nt_list = normalize_tables(_doc({"table_cells": [{"text": "x"}], "text": "raw stuff"}))
    chunks = render_for_graph(nt_list[0], token_limit_whole=1500, token_limit_column=1200)
    assert len(chunks) == 1
    assert chunks[0].chunk_kind == ChunkKind.TABLE_WHOLE
    assert chunks[0].text == "raw stuff" or "raw" in chunks[0].text


def test_sa2_emits_one_chunk_per_column():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)
    # SA-2 fixture has multiple entity columns; expect at least one chunk per column
    column_chunks = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_ENTITY_COLUMN]
    section_chunks = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_ENTITY_SECTION]
    # Either pattern is acceptable; total chunks ≥ #columns
    assert len(column_chunks) + len(section_chunks) >= len(nt.columns)


def test_chunk_cell_refs_point_into_source_table():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)
    for c in chunks:
        if c.chunk_kind in (ChunkKind.TABLE_ENTITY_COLUMN, ChunkKind.TABLE_ENTITY_SECTION):
            assert all(ref.startswith(f"#/tables/{nt.table_index}/data/table_cells/") for ref in c.cell_refs)


def test_small_table_below_whole_limit_emits_one_chunk():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    # Pass a very generous whole limit
    chunks = render_for_graph(nt, token_limit_whole=100000, token_limit_column=1200)
    table_whole = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_WHOLE]
    assert len(table_whole) == 1


def test_chunk_format_contains_entity_block():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)
    for c in chunks:
        if c.chunk_kind == ChunkKind.TABLE_ENTITY_COLUMN:
            assert "ENTITY:" in c.text
            assert "TABLE:" in c.text
```

- [ ] **Step 2: Run tests to verify failure**

Expected: ImportError on `render_for_graph`.

- [ ] **Step 3: Implement `render_for_graph`**

Append to `app/services/table_normalization/render_graph.py`:

```python
from app.services.table_normalization.models import (
    GraphTableChunk, ChunkKind,
)
from app.services.table_normalization.tokens import count_bge_m3_tokens


def render_for_graph(
    table: NormalizedTable,
    token_limit_whole: int,
    token_limit_column: int,
) -> list[GraphTableChunk]:
    """Per §9 of the design spec."""
    # Shape.OTHER → one TABLE_WHOLE chunk with raw_markdown
    if table.shape.value == "other":
        return [GraphTableChunk(
            text=table.raw_markdown,
            table_ref=table.self_ref,
            page_numbers=table.page_numbers,
            chunk_kind=ChunkKind.TABLE_WHOLE,
            entity_display_name=None,
            section=None,
            column_index=None,
            cell_refs=(),
            row_labels=(),
        )]

    # Whole-table rendering
    whole_text = _render_whole_table(table)
    whole_tokens = count_bge_m3_tokens(whole_text)
    if whole_tokens <= token_limit_whole:
        return [GraphTableChunk(
            text=whole_text,
            table_ref=table.self_ref,
            page_numbers=table.page_numbers,
            chunk_kind=ChunkKind.TABLE_WHOLE,
            entity_display_name=None,
            section=None,
            column_index=None,
            cell_refs=tuple(c.cell_ref.self_ref for c in table.cells),
            row_labels=tuple(sorted({c.row_label or "" for c in table.cells})),
        )]

    # Per-column emission
    out: list[GraphTableChunk] = []
    for col in table.columns:
        col_text = _render_column_as_text(col, table, table.sections)
        col_tokens = count_bge_m3_tokens(col_text)
        col_cells = [c for c in table.cells if c.col_idx == col.col_idx]
        col_refs = tuple(c.cell_ref.self_ref for c in col_cells)
        col_row_labels = tuple(sorted({c.row_label or "" for c in col_cells}))

        if col_tokens <= token_limit_column:
            out.append(GraphTableChunk(
                text=col_text,
                table_ref=table.self_ref,
                page_numbers=table.page_numbers,
                chunk_kind=ChunkKind.TABLE_ENTITY_COLUMN,
                entity_display_name=col.display_name,
                section=None,
                column_index=col.col_idx,
                cell_refs=col_refs,
                row_labels=col_row_labels,
            ))
        else:
            # Split by section, repeating identity header
            for section in table.sections:
                sec_text = _render_column_section(col, table, section)
                sec_cells = [c for c in col_cells if c.section == section.name]
                if not sec_cells:
                    continue
                out.append(GraphTableChunk(
                    text=sec_text,
                    table_ref=table.self_ref,
                    page_numbers=table.page_numbers,
                    chunk_kind=ChunkKind.TABLE_ENTITY_SECTION,
                    entity_display_name=col.display_name,
                    section=section.name,
                    column_index=col.col_idx,
                    cell_refs=tuple(c.cell_ref.self_ref for c in sec_cells),
                    row_labels=tuple(sorted({c.row_label or "" for c in sec_cells})),
                ))
    return out


def _render_whole_table(table: NormalizedTable) -> str:
    """Whole-table rendering: identity-rows header + each column block."""
    parts: list[str] = []
    caption = table.caption or table.self_ref
    parts.append(f"TABLE: {caption}")
    if table.page_numbers:
        parts.append(f"SOURCE: page {' '.join(str(p) for p in table.page_numbers)}")
    parts.append("")
    for col in table.columns:
        parts.append(_render_column_as_text(col, table, table.sections).rstrip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_column_section(
    column: NormalizedColumn,
    table: NormalizedTable,
    section: TableSection,
) -> str:
    """Single section of one column, with identity header repeated."""
    parts: list[str] = []
    caption = table.caption or table.self_ref
    parts.append(f"TABLE: {caption}")
    if table.page_numbers:
        parts.append(f"SOURCE: page {' '.join(str(p) for p in table.page_numbers)}")
    parts.append("")
    parts.append("ENTITY:")
    for k, v in column.identity.items():
        parts.append(f"- {k}: {v}")
    parts.append("")
    parts.append(f"{section.name.upper()}:")
    for c in table.cells:
        if c.col_idx != column.col_idx or c.section != section.name:
            continue
        parts.append(_render_row_line(c.row_label or "", c.value, c.unit))
    return "\n".join(parts).rstrip() + "\n"
```

Add `render_for_graph` to `__init__.py` exports.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_render_graph.py -v
```

Expected: all 5 PASS. Iterate if not.

- [ ] **Step 5: Capture snapshot fixture**

After tests pass, generate `tests/fixtures/sa2_graph_chunks_expected.json` with the actual output. Future runs assert no drift:

```python
# Run once after implementation to capture:
import json
from pathlib import Path
from dataclasses import asdict
from app.services.table_normalization import normalize_tables, render_for_graph

fixture = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
doc = {"tables": [fixture], "texts": []}
nt = normalize_tables(doc)[0]
chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)
# Convert to dicts and persist
serializable = [{**asdict(c), "chunk_kind": c.chunk_kind.value} for c in chunks]
Path("tests/fixtures/sa2_graph_chunks_expected.json").write_text(json.dumps(serializable, indent=2))
```

Add a snapshot assertion to the test file:

```python
def test_sa2_graph_chunks_match_snapshot():
    expected = json.loads(Path("tests/fixtures/sa2_graph_chunks_expected.json").read_text())
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)
    actual = [{**c.__dict__, "chunk_kind": c.chunk_kind.value, "page_numbers": list(c.page_numbers), "cell_refs": list(c.cell_refs), "row_labels": list(c.row_labels)} for c in chunks]
    assert actual == expected
```

- [ ] **Step 6: Commit**

```bash
git add app/services/table_normalization/render_graph.py app/services/table_normalization/__init__.py tests/unit/test_table_normalization_render_graph.py tests/fixtures/sa2_graph_chunks_expected.json
git commit -m "feat(table-norm): render_for_graph — graph renderer + snapshot

Per-entity-column emission with section-split fallback for oversized
columns. Small tables (≤ token_limit_whole) get one TABLE_WHOLE chunk.
Shape.OTHER falls back to raw_markdown passthrough. SA-2 snapshot
committed for drift detection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8: Embedding renderer (`render_for_embedding`)

**Files:**
- Modify: `app/services/table_normalization/render_embedding.py`
- Modify: `app/services/table_normalization/__init__.py` (export)
- Create: `tests/unit/test_table_normalization_render_embedding.py`
- Create: `tests/fixtures/sa2_embedding_chunks_expected.json` (snapshot)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_table_normalization_render_embedding.py
import json
import pytest
from pathlib import Path
from app.services.table_normalization import normalize_tables, render_for_embedding
from app.services.table_normalization.models import ChunkKind


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _doc(fixture):
    return {"tables": [fixture], "texts": []}


def test_always_emits_summary():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    summaries = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_SUMMARY]
    assert len(summaries) == 1


def test_small_table_emits_summary_plus_whole():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    # Generous limit ensures the table fits
    chunks = render_for_embedding(nt, token_limit=100000, summary_limit=300)
    kinds = [c.chunk_kind for c in chunks]
    assert ChunkKind.TABLE_SUMMARY in kinds
    assert ChunkKind.TABLE_WHOLE in kinds


def test_large_table_emits_summary_plus_columns():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_embedding(nt, token_limit=10, summary_limit=300)  # force splitting
    kinds = {c.chunk_kind for c in chunks}
    assert ChunkKind.TABLE_SUMMARY in kinds
    assert ChunkKind.TABLE_WHOLE not in kinds  # didn't fit, must not emit


def test_summary_chunk_capped_at_summary_limit():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=50)
    summary = next(c for c in chunks if c.chunk_kind == ChunkKind.TABLE_SUMMARY)
    from app.services.table_normalization.tokens import count_bge_m3_tokens
    assert count_bge_m3_tokens(summary.text) <= 60   # slack for tokenizer variance
```

- [ ] **Step 2: Run tests to verify failure**

Expected: ImportError on `render_for_embedding`.

- [ ] **Step 3: Implement `render_for_embedding`**

```python
# app/services/table_normalization/render_embedding.py — complete file
"""Embedding-side renderer.

Always emits a TABLE_SUMMARY chunk. Emits a TABLE_WHOLE when the table
fits within token_limit; otherwise emits per-entity-column chunks
(with section-split for oversized columns).
"""
from __future__ import annotations

from app.services.table_normalization.models import (
    NormalizedTable, NormalizedColumn, TableSection,
    EmbeddingTableChunk, ChunkKind,
)
from app.services.table_normalization.tokens import count_bge_m3_tokens
from app.services.table_normalization.render_graph import (
    _render_column_as_text,
    _render_whole_table,
    _render_column_section,
    _render_row_line,
)

__all__ = ["_render_column_as_text", "render_for_embedding"]


def render_for_embedding(
    table: NormalizedTable,
    token_limit: int,
    summary_limit: int,
) -> list[EmbeddingTableChunk]:
    """Per §10 of the spec."""
    out: list[EmbeddingTableChunk] = []

    # Shape.OTHER → one TABLE_WHOLE with raw_markdown
    if table.shape.value == "other":
        return [EmbeddingTableChunk(
            text=table.raw_markdown,
            table_ref=table.self_ref,
            page_numbers=table.page_numbers,
            chunk_kind=ChunkKind.TABLE_WHOLE,
            entity_display_name=None,
            section=None,
            column_index=None,
            cell_refs=(),
            row_labels=(),
        )]

    # Always emit summary
    summary_text = _render_summary(table, summary_limit)
    out.append(EmbeddingTableChunk(
        text=summary_text,
        table_ref=table.self_ref,
        page_numbers=table.page_numbers,
        chunk_kind=ChunkKind.TABLE_SUMMARY,
        entity_display_name=None,
        section=None,
        column_index=None,
        cell_refs=tuple(c.cell_ref.self_ref for c in table.cells),
        row_labels=tuple(sorted({c.row_label or "" for c in table.cells})),
    ))

    # Whole-table check
    whole_text = _render_whole_table(table)
    if count_bge_m3_tokens(whole_text) <= token_limit:
        out.append(EmbeddingTableChunk(
            text=whole_text,
            table_ref=table.self_ref,
            page_numbers=table.page_numbers,
            chunk_kind=ChunkKind.TABLE_WHOLE,
            entity_display_name=None,
            section=None,
            column_index=None,
            cell_refs=tuple(c.cell_ref.self_ref for c in table.cells),
            row_labels=tuple(sorted({c.row_label or "" for c in table.cells})),
        ))
        return out

    # Per-column emission (with section-split for oversized columns)
    for col in table.columns:
        col_text = _render_column_as_text(col, table, table.sections)
        col_cells = [c for c in table.cells if c.col_idx == col.col_idx]
        col_refs = tuple(c.cell_ref.self_ref for c in col_cells)
        col_row_labels = tuple(sorted({c.row_label or "" for c in col_cells}))

        if count_bge_m3_tokens(col_text) <= token_limit:
            out.append(EmbeddingTableChunk(
                text=col_text,
                table_ref=table.self_ref,
                page_numbers=table.page_numbers,
                chunk_kind=ChunkKind.TABLE_ENTITY_COLUMN,
                entity_display_name=col.display_name,
                section=None,
                column_index=col.col_idx,
                cell_refs=col_refs,
                row_labels=col_row_labels,
            ))
        else:
            for section in table.sections:
                sec_text = _render_column_section(col, table, section)
                sec_cells = [c for c in col_cells if c.section == section.name]
                if not sec_cells:
                    continue
                out.append(EmbeddingTableChunk(
                    text=sec_text,
                    table_ref=table.self_ref,
                    page_numbers=table.page_numbers,
                    chunk_kind=ChunkKind.TABLE_ENTITY_SECTION,
                    entity_display_name=col.display_name,
                    section=section.name,
                    column_index=col.col_idx,
                    cell_refs=tuple(c.cell_ref.self_ref for c in sec_cells),
                    row_labels=tuple(sorted({c.row_label or "" for c in sec_cells})),
                ))
    return out


def _render_summary(table: NormalizedTable, summary_limit: int) -> str:
    """Per §10 emission rule 2."""
    caption = table.caption or table.self_ref
    pages = " ".join(str(p) for p in table.page_numbers) if table.page_numbers else ""
    variants = ", ".join(c.display_name for c in table.columns)
    spec_labels = sorted({c.row_label for c in table.cells if c.row_label})
    props = ", ".join(spec_labels)

    parts = [f"TABLE: {caption}"]
    if pages:
        parts.append(f"SOURCE: page {pages}; ref {table.self_ref}")
    else:
        parts.append(f"SOURCE: ref {table.self_ref}")
    parts.append(f"VARIANTS: {variants}")
    parts.append(f"PROPERTIES: {props}")
    text = "\n".join(parts)

    # Truncate VARIANTS / PROPERTIES if over limit
    while count_bge_m3_tokens(text) > summary_limit and variants:
        variants_list = variants.rsplit(", ", 1)
        if len(variants_list) == 1:
            break
        variants = variants_list[0] + ", ..."
        text = "\n".join([
            f"TABLE: {caption}",
            f"SOURCE: page {pages}; ref {table.self_ref}" if pages else f"SOURCE: ref {table.self_ref}",
            f"VARIANTS: {variants}",
            f"PROPERTIES: {props}",
        ])
        # Also truncate props if still too long
        while count_bge_m3_tokens(text) > summary_limit and props:
            props_list = props.rsplit(", ", 1)
            if len(props_list) == 1:
                break
            props = props_list[0] + ", ..."
            text = "\n".join([
                f"TABLE: {caption}",
                f"SOURCE: page {pages}; ref {table.self_ref}" if pages else f"SOURCE: ref {table.self_ref}",
                f"VARIANTS: {variants}",
                f"PROPERTIES: {props}",
            ])
    return text
```

Add `render_for_embedding` to `__init__.py`.

- [ ] **Step 4: Run tests + capture snapshot**

```bash
.venv/bin/pytest tests/unit/test_table_normalization_render_embedding.py -v
```

Expected: all 4 PASS.

After they pass, capture snapshot fixture `tests/fixtures/sa2_embedding_chunks_expected.json` and add snapshot assertion (same pattern as Task 7 Step 5).

- [ ] **Step 5: Commit**

```bash
git add app/services/table_normalization/render_embedding.py app/services/table_normalization/__init__.py tests/unit/test_table_normalization_render_embedding.py tests/fixtures/sa2_embedding_chunks_expected.json
git commit -m "feat(table-norm): render_for_embedding — always-summary + size-aware

Always emits TABLE_SUMMARY. Adds TABLE_WHOLE if table fits; otherwise
emits per-entity-column chunks (with section-split for oversized
columns). Shares _render_column_as_text with the graph renderer so
chunk format is byte-identical for the same column input.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 9: Provenance bridge module

**Files:**
- Create: `app/services/table_normalization/_provenance_bridge.py`
- Create: `tests/unit/test_provenance_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_provenance_bridge.py
from app.services.table_normalization import _provenance_bridge as bridge


def test_record_and_lookup():
    bridge.reset()
    bridge.record_text_idx_cell_refs(142, ["#/tables/3/data/table_cells/42"])
    assert bridge.cell_refs_for_text_idx(142) == ["#/tables/3/data/table_cells/42"]


def test_lookup_unknown_returns_empty():
    bridge.reset()
    assert bridge.cell_refs_for_text_idx(999) == []


def test_reset_clears_state():
    bridge.record_text_idx_cell_refs(1, ["#/tables/0/data/table_cells/0"])
    bridge.reset()
    assert bridge.cell_refs_for_text_idx(1) == []


def test_empty_list_not_recorded():
    bridge.reset()
    bridge.record_text_idx_cell_refs(5, [])
    assert bridge.cell_refs_for_text_idx(5) == []


def test_returned_list_is_a_copy():
    """Mutating the returned list must not corrupt the stored value."""
    bridge.reset()
    bridge.record_text_idx_cell_refs(7, ["#/tables/0/data/table_cells/1"])
    got = bridge.cell_refs_for_text_idx(7)
    got.append("MALICIOUS")
    assert bridge.cell_refs_for_text_idx(7) == ["#/tables/0/data/table_cells/1"]
```

- [ ] **Step 2: Run tests to verify failure**

Expected: ImportError.

- [ ] **Step 3: Implement `_provenance_bridge.py`**

```python
# app/services/table_normalization/_provenance_bridge.py
"""Process-local map: text_idx (docling #/texts/N) → list of cell_refs.

Populated by _text_item_from_chunk at TextItem-creation time. Read by
the field-provenance enrichment wrapper that fills
ExtractionFieldProvenance.cell_refs after extraction.

Per-pass reset() prevents cross-pass leakage. Module-level state is
safe in the single-process docling-graph FastAPI worker; multi-process
deployments maintain per-process maps.
"""
from __future__ import annotations

_TEXT_IDX_TO_CELL_REFS: dict[int, list[str]] = {}


def record_text_idx_cell_refs(text_idx: int, cell_refs: list[str]) -> None:
    """Record cell_refs at TextItem-creation time.

    Empty/None lists are not stored (saves memory and makes
    cell_refs_for_text_idx() return [] cleanly).
    """
    if cell_refs:
        _TEXT_IDX_TO_CELL_REFS[int(text_idx)] = list(cell_refs)


def cell_refs_for_text_idx(text_idx: int) -> list[str]:
    """Return a COPY of the cell_refs for text_idx; [] if not recorded."""
    return list(_TEXT_IDX_TO_CELL_REFS.get(int(text_idx), ()))


def reset() -> None:
    """Clear the bridge map. Called at the start of each
    run_extraction_pass to prevent cross-pass leakage."""
    _TEXT_IDX_TO_CELL_REFS.clear()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_provenance_bridge.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/table_normalization/_provenance_bridge.py tests/unit/test_provenance_bridge.py
git commit -m "feat(table-norm): _provenance_bridge — text_idx → cell_refs map

Process-local dict populated by _text_item_from_chunk at TextItem-
creation time; read by the field-provenance enrichment wrapper. Per-
pass reset() prevents cross-pass leakage. cell_refs_for_text_idx
returns a COPY so callers can't corrupt stored state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## End of Chunk 1

Chunk 1 builds the entire normalization library (Phase 0 setup + Phase 1 tasks 1–9). Every component is TDD-built with snapshot fixtures committed for drift detection.

---

## Chunk 2: Phase 2 (Schema) and Phase 3 (Integration)

## Phase 2 — Schema migrations

### Task 10: Alembic migration — `chunk_metadata` column + partial index

**Files:**
- Create: `alembic/versions/0021_chunk_metadata.py`
- Modify: `app/models/retrieval.py:11-57`
- Create: `tests/integration/test_chunk_metadata_persistence.py`

- [ ] **Step 1: Determine the next alembic revision number**

```bash
ls alembic/versions/ | sort | tail -3
```

Expected: shows recent migration files. The next number should be `0021_*`. If a higher number exists, adjust.

- [ ] **Step 2: Add `chunk_metadata` to the `TextChunk` SQLAlchemy model**

Edit `app/models/retrieval.py` — after line 47 (`classification: ...`), add:

```python
chunk_metadata: Mapped[Optional[dict]] = mapped_column(
    JSONB, nullable=True,
    doc=(
        "Optional structured metadata for normalized table chunks "
        "(chunk_kind, table_ref, entity_display_name, cell_refs, etc.). "
        "NULL for prose / heading / equation / image chunks. "
        "See docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md §11.2."
    ),
)
```

- [ ] **Step 3: Write the failing integration test**

```python
# tests/integration/test_chunk_metadata_persistence.py
import json
import uuid
import pytest
from sqlalchemy import text
from app.db.session import get_db_sync
from app.models.retrieval import TextChunk


@pytest.fixture
def db():
    with get_db_sync() as session:
        yield session


def test_text_chunk_has_chunk_metadata_column(db):
    """Schema column exists and accepts JSONB."""
    result = db.execute(text(
        "SELECT column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_schema='retrieval' "
        "AND table_name='text_chunks' "
        "AND column_name='chunk_metadata'"
    )).fetchone()
    assert result is not None, "chunk_metadata column missing — migration 0021 not applied"
    assert result[1] == "jsonb"


def test_partial_index_exists(db):
    """The partial expression index ix_text_chunks_chunk_kind exists."""
    result = db.execute(text(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname='retrieval' "
        "AND tablename='text_chunks' "
        "AND indexname='ix_text_chunks_chunk_kind'"
    )).fetchone()
    assert result is not None, "ix_text_chunks_chunk_kind index missing"


def test_write_and_read_chunk_metadata(db):
    """Round-trip a TextChunk row with chunk_metadata."""
    # This test requires a Document already to exist; integration test
    # harness or fixture must provide one.
    pass  # Filled in Step 6 after migration applied
```

- [ ] **Step 4: Determine the previous revision id**

```bash
PREV_REV=$(grep -E "^revision = " alembic/versions/$(ls -1 alembic/versions/ | sort | tail -1) | head -1 | sed -E "s/^revision = ['\"]([^'\"]+)['\"].*/\\1/")
echo "down_revision should be: $PREV_REV"
```

- [ ] **Step 5: Write the migration**

```python
# alembic/versions/0021_chunk_metadata.py
"""Add chunk_metadata JSONB to retrieval.text_chunks for table-aware chunking.

Spec: docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md §11.1.

WARNING: downgrade drops the chunk_metadata column. Run ./manage.sh --blow-away
before downgrading to avoid silent data loss in retrieval responses that
rely on the table_chunk block.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0021_chunk_metadata"
down_revision = "<previous revision id — fill from ls alembic/versions/>"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "text_chunks",
        sa.Column("chunk_metadata", postgresql.JSONB(), nullable=True),
        schema="retrieval",
    )
    op.create_index(
        "ix_text_chunks_chunk_kind",
        "text_chunks",
        [sa.text("(chunk_metadata->>'chunk_kind')")],
        schema="retrieval",
        postgresql_where=sa.text("chunk_metadata IS NOT NULL"),
    )


def downgrade():
    op.drop_index(
        "ix_text_chunks_chunk_kind",
        table_name="text_chunks",
        schema="retrieval",
    )
    op.drop_column(
        "text_chunks", "chunk_metadata", schema="retrieval",
    )
```

Replace `<previous revision id>` with the actual previous revision (look up via `alembic history | head -3`).

- [ ] **Step 5: Apply the migration**

```bash
.venv/bin/alembic upgrade head
```

Expected: migration 0021 applied; no errors.

- [ ] **Step 6: Complete the round-trip test body**

```python
def test_write_and_read_chunk_metadata(db):
    # Create a minimal Document + TextChunk
    from app.models.ingest import Document
    doc = Document(id=uuid.uuid4(), document_metadata={})
    db.add(doc)
    db.flush()

    chunk = TextChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=0,
        chunk_text="test",
        modality="table",
        chunk_metadata={
            "chunk_kind": "table_entity_column",
            "table_ref": "#/tables/3",
            "entity_display_name": "S-75M2",
            "cell_refs": ["#/tables/3/data/table_cells/42"],
        },
    )
    db.add(chunk)
    db.commit()

    fetched = db.query(TextChunk).filter_by(id=chunk.id).one()
    assert fetched.chunk_metadata["chunk_kind"] == "table_entity_column"
    assert fetched.chunk_metadata["cell_refs"] == ["#/tables/3/data/table_cells/42"]


def test_partial_index_returns_matching_rows(db):
    """The partial expression index returns rows where chunk_metadata->>'chunk_kind' matches."""
    from app.models.ingest import Document
    doc = Document(id=uuid.uuid4(), document_metadata={})
    db.add(doc)
    db.flush()

    for kind in ["table_entity_column", "table_entity_section", None]:
        meta = {"chunk_kind": kind} if kind else None
        db.add(TextChunk(
            id=uuid.uuid4(), document_id=doc.id,
            chunk_index=0, chunk_text="x", modality="table",
            chunk_metadata=meta,
        ))
    db.commit()

    matched = db.execute(text(
        "SELECT COUNT(*) FROM retrieval.text_chunks "
        "WHERE chunk_metadata->>'chunk_kind' = 'table_entity_column'"
    )).scalar()
    assert matched == 1
```

- [ ] **Step 7: Run the integration tests**

```bash
.venv/bin/pytest tests/integration/test_chunk_metadata_persistence.py -v
```

Expected: all 4 PASS.

- [ ] **Step 8: Commit**

```bash
git add alembic/versions/0021_chunk_metadata.py app/models/retrieval.py tests/integration/test_chunk_metadata_persistence.py
git commit -m "feat(table-norm): alembic 0021 — chunk_metadata column + partial index

Additive nullable JSONB column on retrieval.text_chunks. Partial
expression index ix_text_chunks_chunk_kind (WHERE chunk_metadata IS
NOT NULL) prepares for future retrieval-side filtering by chunk_kind.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 11: ArcadeDB schema — `chunk_kind` property on `TextChunk` vertex

**Files:**
- Modify: `app/services/arcadedb_schema.py:31`
- Modify: `app/services/arcadedb_schema.py` (add index in the index-creation loop, around line 280-290)
- Create: `tests/integration/test_arcadedb_chunk_kind_schema.py`

- [ ] **Step 1: Add `chunk_kind` to the `TextChunk` property list**

Edit `app/services/arcadedb_schema.py:31` — `_STRUCTURAL_VERTEX_TYPES["TextChunk"]`:

```python
"TextChunk": [
    ("chunk_id", "STRING"),
    ("document_id", "STRING"),
    ("page_number", "INTEGER"),
    ("modality", "STRING"),
    ("classification", "STRING"),
    ("text_embedding", "ARRAY_OF_FLOATS"),
    ("chunk_kind", "STRING"),   # NEW — populated from chunk_metadata.chunk_kind
],
```

- [ ] **Step 2: Add `chunk_kind` index**

Find the existing index-creation loop in `arcadedb_schema.py` (around line 280–290 per the spec). Patterns to look for: a function like `_create_indexes` or a loop iterating `_STRUCTURAL_VERTEX_TYPES`. Add:

```python
# After existing TextChunk index declarations
self._exec(f"CREATE INDEX IF NOT EXISTS ON TextChunk (chunk_kind) NOTUNIQUE")
```

- [ ] **Step 3: Write the integration test**

```python
# tests/integration/test_arcadedb_chunk_kind_schema.py
import pytest
from app.db.session import get_graph_store


def test_text_chunk_has_chunk_kind_property():
    """ArcadeDB TextChunk schema includes chunk_kind property after bootstrap."""
    gs = get_graph_store()
    gs.ensure_ready_sync()
    # Query schema metadata — ArcadeDB SQL: SELECT FROM schema:types
    result = gs.execute_sync(
        "SELECT properties FROM schema:types WHERE name = 'TextChunk'"
    )
    assert result, "TextChunk type not found in ArcadeDB schema"
    props = result[0].get("properties", [])
    prop_names = [p.get("name") for p in props]
    assert "chunk_kind" in prop_names


def test_chunk_kind_index_exists():
    """ArcadeDB has an index on TextChunk.chunk_kind."""
    gs = get_graph_store()
    result = gs.execute_sync("SELECT name FROM schema:indexes WHERE typeName = 'TextChunk'")
    index_names = [r.get("name", "") for r in result]
    assert any("chunk_kind" in n for n in index_names), f"chunk_kind index not found in {index_names}"


def test_write_vertex_with_chunk_kind():
    """Writing a TextChunk vertex with chunk_kind succeeds."""
    gs = get_graph_store()
    # Use the existing TextChunkRecord write path
    from app.services.graph_store import TextChunkRecord
    record = TextChunkRecord(
        chunk_id="test-chunk-kind",
        text="sample",
        document_id="test-doc",
        properties={
            "modality": "table",
            "page_number": 1,
            "classification": "UNCLASSIFIED",
            "chunk_kind": "table_entity_column",
        },
        embedding=[0.0] * 1024,
    )
    gs.create_text_chunks_batch_sync([record])
    # Read back
    result = gs.execute_sync(
        "SELECT chunk_kind FROM TextChunk WHERE chunk_id = ?", ["test-chunk-kind"]
    )
    assert result
    assert result[0]["chunk_kind"] == "table_entity_column"
```

(If the ArcadeDB read/write API differs from what's shown, adjust to match existing patterns in `app/services/arcadedb_*.py`. The test asserts the same intent regardless.)

- [ ] **Step 4: Apply ArcadeDB schema changes via `./manage.sh --blow-away` OR restart**

```bash
# Either:
./manage.sh --blow-away   # full reset
# OR
docker compose restart arcadedb docling-graph   # in-place; CREATE PROPERTY IF NOT EXISTS is idempotent
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/integration/test_arcadedb_chunk_kind_schema.py -v
```

Expected: all 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/arcadedb_schema.py tests/integration/test_arcadedb_chunk_kind_schema.py
git commit -m "feat(table-norm): ArcadeDB chunk_kind property + index on TextChunk

Single-line addition to _STRUCTURAL_VERTEX_TYPES['TextChunk'] tuple
list. CREATE PROPERTY IF NOT EXISTS is idempotent on the existing
bootstrap path so either --blow-away or in-place restart picks up
the new schema.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Pipeline integration

### Task 12: `_pipeline_hooks.py` — substitution helper + adapter + size-threshold

**Files:**
- Create: `app/services/table_normalization/_pipeline_hooks.py`
- Create: `tests/unit/test_hybrid_chunker_substitution.py`
- Create: `tests/unit/test_normalized_table_chunk_adapter.py`
- Create: `tests/unit/test_table_size_threshold.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_table_size_threshold.py
import json
from pathlib import Path
from app.services.table_normalization import normalize_tables
from app.services.table_normalization._pipeline_hooks import _normalized_table_size_tokens


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def test_size_function_returns_positive_for_real_table():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    n = _normalized_table_size_tokens(nt)
    assert n > 0


def test_size_function_sums_column_renderings():
    """Canonical contract per spec rev. 7 §10.1: sum across columns."""
    from app.services.table_normalization.tokens import count_bge_m3_tokens
    from app.services.table_normalization.render_graph import _render_column_as_text

    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    expected = sum(
        count_bge_m3_tokens(_render_column_as_text(col, nt, nt.sections))
        for col in nt.columns
    )
    assert _normalized_table_size_tokens(nt) == expected
```

```python
# tests/unit/test_normalized_table_chunk_adapter.py
import json
from pathlib import Path
from app.services.table_normalization import normalize_tables, render_for_embedding
from app.services.table_normalization._pipeline_hooks import _NormalizedTableChunkAdapter


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def test_adapter_ducktypes_native_chunk_interface():
    """Adapter exposes .text, .meta.doc_items[].self_ref, .meta.doc_items[].prov[].page_no, .meta.headings."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    etc = chunks[0]
    adapter = _NormalizedTableChunkAdapter(
        etc=etc, parent_headings=("Chapter 4", "Variants"),
        parent_table_ref="#/tables/0",
    )

    # The interface that _build_native_chunk_meta reads:
    assert adapter.text == etc.text
    assert hasattr(adapter, "meta")
    items = adapter.meta.doc_items
    assert len(items) == 1
    assert items[0].self_ref == "#/tables/0"  # NOT a cell ref — today-shape


def test_adapter_preserves_parent_headings():
    """Section_path regression: native chunk's parent_headings preserved."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    etc = chunks[0]
    adapter = _NormalizedTableChunkAdapter(
        etc=etc, parent_headings=("Section A",),
        parent_table_ref="#/tables/0",
    )
    assert adapter.meta.headings == ("Section A",)


def test_adapter_exposes_extra_metadata():
    """extra_metadata carries chunk_kind + cell_refs for the metadata column."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    etc = chunks[1]  # not the summary (which has no cell_refs likely set differently)
    adapter = _NormalizedTableChunkAdapter(
        etc=etc, parent_headings=(),
        parent_table_ref="#/tables/0",
    )
    em = adapter.extra_metadata
    assert em["chunk_kind"] == etc.chunk_kind.value
    assert "cell_refs" in em
    assert "row_labels" in em
```

```python
# tests/unit/test_hybrid_chunker_substitution.py
import json
from pathlib import Path
from app.services.table_normalization import normalize_tables, render_for_embedding
from app.services.table_normalization._pipeline_hooks import _substitute_table_chunks


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _fake_native_chunk(text, self_refs, headings=()):
    """Build a duck-typed object matching what HybridChunker emits."""
    class _Prov:
        def __init__(self, page_no): self.page_no = page_no
    class _Item:
        def __init__(self, self_ref):
            self.self_ref = self_ref
            self.prov = [_Prov(1)]
    class _Meta:
        def __init__(self, items, headings):
            self.doc_items = items
            self.headings = list(headings)
    class _Chunk:
        def __init__(self, text, items, headings):
            self.text = text
            self.meta = _Meta(items, headings)
    items = [_Item(ref) for ref in self_refs]
    return _Chunk(text, items, headings)


def test_table_dominant_substitution():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    # Native chunk where 100% of doc_items reference the table
    native = _fake_native_chunk("raw table text", ["#/tables/0", "#/tables/0", "#/tables/0"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    # Native chunk gone, normalized chunks present
    assert all(getattr(c, "text", "") != "raw table text" for c in out)
    assert len(out) > 0


def test_non_table_passthrough():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    native = _fake_native_chunk("prose", ["#/texts/100"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    assert out == [native]


def test_small_table_passes_through():
    """Tables below MIN_TABLE_NORMALIZATION_TOKENS pass through unchanged."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    native = _fake_native_chunk("small table text", ["#/tables/0"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=999999,
    )
    assert out == [native]


def test_subsequent_native_chunks_for_same_table_dropped():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    n1 = _fake_native_chunk("first", ["#/tables/0", "#/tables/0"])
    n2 = _fake_native_chunk("second", ["#/tables/0"])
    out = _substitute_table_chunks(
        [n1, n2], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    # No adapter should have the raw text from either native chunk
    assert all(getattr(c, "text", "") not in ("first", "second") for c in out)


def test_content_coverage_assertion():
    """Union of substituted chunks' cell_refs covers all cells of the normalized table."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    native = _fake_native_chunk("dominant", ["#/tables/0", "#/tables/0", "#/tables/0"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    # Collect all cell_refs from substituted adapters
    collected_refs: set[str] = set()
    for c in out:
        em = getattr(c, "extra_metadata", None) or {}
        for ref in em.get("cell_refs", []):
            collected_refs.add(ref)
    expected_refs = {c.cell_ref.self_ref for c in nt.cells}
    # Substituted chunks must cover all cell refs of the normalized table
    assert expected_refs.issubset(collected_refs), \
        f"missing cells: {expected_refs - collected_refs}"
```

- [ ] **Step 2: Run tests to verify failure**

Expected: ImportError on `_pipeline_hooks`.

- [ ] **Step 3: Implement `_pipeline_hooks.py`**

```python
# app/services/table_normalization/_pipeline_hooks.py
"""HybridChunker integration: post-process native chunks to substitute
normalized table chunks where appropriate.

See §10.1 of the design spec. The functions here are pure (no I/O);
they are invoked from app/workers/pipeline.py between the native
chunker call and the chunk-iteration loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from app.services.table_normalization.models import (
    NormalizedTable, NormalizedColumn, TableSection, Shape,
    EmbeddingTableChunk, GraphTableChunk,
)
from app.services.table_normalization.tokens import count_bge_m3_tokens
from app.services.table_normalization.render_graph import _render_column_as_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AdapterProv:
    page_no: int


@dataclass(frozen=True)
class _AdapterDocItem:
    self_ref: str
    prov: tuple[_AdapterProv, ...]


@dataclass(frozen=True)
class _AdapterMeta:
    doc_items: tuple[_AdapterDocItem, ...]
    headings: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedTableChunkAdapter:
    """Ducktypes the HybridChunker chunk interface.

    Read interface that the downstream pipeline.py:5559-5623 loop uses:
    - .text
    - .meta.doc_items[].self_ref
    - .meta.doc_items[].prov[].page_no
    - .meta.headings
    Plus a new .extra_metadata for chunk_metadata column population.

    CRITICAL: the single synthetic doc_item's self_ref is the TABLE-LEVEL
    ref ("#/tables/{N}"), never a cell ref. Cell refs flow through
    .extra_metadata.cell_refs only — keeps TextChunk.self_refs in today's
    shape (#/texts/N or #/tables/N), so provenance.py:_resolve_element_uid
    and the retrieval response surface stay backwards compatible.
    """
    etc: Any                                # EmbeddingTableChunk or GraphTableChunk
    parent_headings: tuple[str, ...]
    parent_table_ref: str

    @property
    def text(self) -> str:
        return self.etc.text

    @property
    def meta(self) -> _AdapterMeta:
        prov_tuple = tuple(_AdapterProv(page_no=p) for p in self.etc.page_numbers)
        item = _AdapterDocItem(self_ref=self.parent_table_ref, prov=prov_tuple)
        return _AdapterMeta(doc_items=(item,), headings=self.parent_headings)

    @property
    def extra_metadata(self) -> dict:
        return {
            "chunk_kind": self.etc.chunk_kind.value,
            "table_ref": self.etc.table_ref,
            "entity_display_name": self.etc.entity_display_name,
            "section": self.etc.section,
            "column_index": self.etc.column_index,
            "cell_refs": list(self.etc.cell_refs),
            "row_labels": list(self.etc.row_labels),
            "page_numbers": list(self.etc.page_numbers),
        }


def _normalized_table_size_tokens(nt: NormalizedTable) -> int:
    """Canonical size function per spec rev. 7 §10.1.

    Sum of bge-m3 tokens across rendered columns. Single contract; no
    cheap fallback (boundary behavior at MIN_TABLE_NORMALIZATION_TOKENS
    must be deterministic).
    """
    return sum(
        count_bge_m3_tokens(_render_column_as_text(col, nt, nt.sections))
        for col in nt.columns
    )


def _classify_native_chunk(
    nc: Any, normalized_by_table_idx: dict[int, NormalizedTable],
) -> tuple[str, int | None]:
    """Classify a native chunk as table_dominant / table_mixed / non_table.

    Returns (classification, dominant_table_idx_or_None).
    """
    items = getattr(getattr(nc, "meta", None), "doc_items", None) or []
    if not items:
        return ("non_table", None)
    table_idx_counts: dict[int, int] = {}
    for item in items:
        ref = getattr(item, "self_ref", None) or ""
        if not ref.startswith("#/tables/"):
            continue
        try:
            idx = int(ref.split("/")[-1])
        except (ValueError, IndexError):
            continue
        if idx in normalized_by_table_idx and normalized_by_table_idx[idx].shape != Shape.OTHER:
            table_idx_counts[idx] = table_idx_counts.get(idx, 0) + 1
    if not table_idx_counts:
        return ("non_table", None)
    dominant_idx = max(table_idx_counts, key=table_idx_counts.get)
    dominant_share = table_idx_counts[dominant_idx] / len(items)
    return (("table_dominant" if dominant_share >= 0.8 else "table_mixed"), dominant_idx)


def _substitute_table_chunks(
    native_chunks: list[Any],
    normalized_by_table_idx: dict[int, NormalizedTable],
    render_fn: Callable[..., list[Any]],
    *,
    token_limit: int,
    summary_limit: int,
    min_table_tokens: int,
) -> list[Any]:
    """Per §10.1 of the design spec.

    Substitution decision tree:
    - non_table: pass through unchanged.
    - normalized table below min_table_tokens: pass through unchanged.
    - table_dominant: substitute entirely. Subsequent natives for same
      table_idx are dropped (NormalizedTable.cells covers all content).
    - table_mixed above threshold: emit normalized chunks AND keep native
      (defensive — wide tables shouldn't reach this branch).
    """
    seen_table_idx: set[int] = set()
    out: list[Any] = []
    for nc in native_chunks:
        cls, table_idx = _classify_native_chunk(nc, normalized_by_table_idx)

        if cls == "non_table":
            out.append(nc)
            continue

        nt = normalized_by_table_idx[table_idx]
        size = _normalized_table_size_tokens(nt)
        if size < min_table_tokens:
            out.append(nc)
            continue

        parent_headings = tuple(getattr(getattr(nc, "meta", None), "headings", None) or [])
        parent_table_ref = f"#/tables/{table_idx}"

        if cls == "table_dominant":
            if table_idx in seen_table_idx:
                continue
            seen_table_idx.add(table_idx)
            for etc in render_fn(nt, token_limit=token_limit, summary_limit=summary_limit):
                out.append(_NormalizedTableChunkAdapter(
                    etc=etc, parent_headings=parent_headings,
                    parent_table_ref=parent_table_ref,
                ))
            continue

        # cls == "table_mixed" (anomalous for wide tables per spike findings)
        logger.warning(
            "_substitute_table_chunks: table_mixed classification fired (table_idx=%d, "
            "dominant_share<0.8). HybridChunker merged this table with prose; "
            "emitting normalized chunks AND keeping native (degraded path).",
            table_idx,
        )
        if table_idx not in seen_table_idx:
            seen_table_idx.add(table_idx)
            for etc in render_fn(nt, token_limit=token_limit, summary_limit=summary_limit):
                out.append(_NormalizedTableChunkAdapter(
                    etc=etc, parent_headings=parent_headings,
                    parent_table_ref=parent_table_ref,
                ))
        out.append(nc)
    return out
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_hybrid_chunker_substitution.py tests/unit/test_normalized_table_chunk_adapter.py tests/unit/test_table_size_threshold.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/table_normalization/_pipeline_hooks.py tests/unit/test_hybrid_chunker_substitution.py tests/unit/test_normalized_table_chunk_adapter.py tests/unit/test_table_size_threshold.py
git commit -m "feat(table-norm): _pipeline_hooks — substitution + adapter + size

Pure functions for HybridChunker post-processing. _substitute_table_chunks
implements the 3-way classification (dominant / mixed / non-table) with
the 256-token size threshold gating substitution. Adapter ducktypes
the native-chunk interface; cell_refs route through extra_metadata,
keeping TextChunk.self_refs in today's #/texts or #/tables shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 13: Integrate HybridChunker primary path (`pipeline.py:5500-5634`)

**Files:**
- Modify: `app/workers/pipeline.py:5399-5448` (`_build_native_chunk_meta`)
- Modify: `app/workers/pipeline.py:5500-5534` (insertion: substitution call)
- Modify: `app/workers/pipeline.py:5559-5623` (write path: `chunk_metadata`, `chunk_kind`)
- Modify: `app/services/table_normalization/__init__.py` (export `_pipeline_hooks`)

- [ ] **Step 1: Add `extra_metadata` parameter to `_build_native_chunk_meta`**

In `app/workers/pipeline.py`, modify `_build_native_chunk_meta` at line 5399. Read the existing function first (`app/workers/pipeline.py:5399-5448`). The change is:

```python
def _build_native_chunk_meta(
    chunk_idx: int, chunk, document_id: str, model_version: str,
) -> dict:
    # ... existing logic unchanged ...
    extra = getattr(chunk, "extra_metadata", None)   # NEW: ducktyped read; None for native chunks
    return {
        "chunk_id": chunk_id,
        "chunk_index": chunk_idx,
        "page_number": min(page_numbers) if page_numbers else None,
        "page_numbers": sorted(page_numbers),
        "modality": "text",
        "self_refs": self_refs,
        "evidence_ids": list(self_refs),
        "document_id": document_id,
        "section_path": section_path,
        "headings": headings,
        "chunk_metadata": extra,                     # NEW
    }
```

- [ ] **Step 2: Insert substitution call between native_chunks creation and the loop**

In `app/workers/pipeline.py` around line 5534 (just after `native_chunks = list(chunker.chunk(doc_obj_dl))`), insert the block below. **Variable names in the existing code:** the loaded JSON is `doc_dict` (line 5511); the parsed DoclingDocument is `doc_obj_dl` (line 5533). The block below uses `doc_dict` (the JSON form, which is what `normalize_tables` expects). Read the surrounding code first to confirm and adjust if names have drifted.

```python
# Table normalization substitution (spec 2026-05-11 §10.1).
# Only fires when EMBEDDING_TABLE_NORMALIZATION_ENABLED=true.
from app.services.table_normalization.config import is_table_normalization_enabled_embedding
if is_table_normalization_enabled_embedding():
    from app.services.table_normalization import normalize_tables, render_for_embedding
    from app.services.table_normalization.config import (
        embedding_chunk_max_tokens, embedding_table_summary_max_tokens,
        min_table_normalization_tokens,
    )
    from app.services.table_normalization._pipeline_hooks import _substitute_table_chunks

    normalized_by_table_idx = {
        nt.table_index: nt for nt in normalize_tables(doc_dict)
    }
    native_chunks = _substitute_table_chunks(
        native_chunks,
        normalized_by_table_idx,
        render_for_embedding,
        token_limit=embedding_chunk_max_tokens(),
        summary_limit=embedding_table_summary_max_tokens(),
        min_table_tokens=min_table_normalization_tokens(),
    )
```

- [ ] **Step 3: Add `chunk_metadata` to Postgres write at line 5594, update `set_` clause**

Modify the existing `chunk_values` dict + `on_conflict_do_update` at `app/workers/pipeline.py:5594`:

```python
chunk_values = {
    "id": meta["chunk_id"],
    "artifact_id": None,
    "document_id": uuid.UUID(document_id),
    "chunk_index": meta["chunk_index"],
    "chunk_text": text,
    "modality": meta["modality"],
    "page_number": meta["page_number"],
    "bounding_box": None,
    "chunk_metadata": meta.get("chunk_metadata"),     # NEW
}

stmt = pg_insert(TextChunk).values(**chunk_values).on_conflict_do_update(
    index_elements=["id"],
    set_={
        "chunk_text": chunk_values["chunk_text"],
        "modality": chunk_values["modality"],
        "chunk_index": chunk_values["chunk_index"],   # NEW — re-run safety
        "page_number": chunk_values["page_number"],   # NEW — re-run safety
        "chunk_metadata": chunk_values["chunk_metadata"],  # NEW
    },
)
```

- [ ] **Step 4: Add `chunk_kind` to ArcadeDB write at line 5603-5618**

Modify the existing `properties` dict on `TextChunkRecord`:

```python
text_chunk_records.append(_TCR(
    chunk_id=str(meta["chunk_id"]),
    text=text,
    document_id=document_id,
    properties={
        "artifact_id": None,
        "modality": meta["modality"],
        "page_number": meta["page_number"],
        "classification": doc_classification,
        "page_numbers": meta["page_numbers"],
        "self_refs": meta["self_refs"],
        "evidence_ids": meta["evidence_ids"],
        "section_path": meta.get("section_path"),
        "headings": meta.get("headings", []),
        "chunk_kind": (meta.get("chunk_metadata") or {}).get("chunk_kind"),  # NEW
    },
    embedding=embedding,
))
```

- [ ] **Step 5: Run existing pipeline tests to verify nothing breaks with flag off**

```bash
.venv/bin/pytest tests/integration/ -k embedding -v
```

Expected: existing embedding tests still pass (since `EMBEDDING_TABLE_NORMALIZATION_ENABLED=false` by default, substitution doesn't fire).

- [ ] **Step 6: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "feat(table-norm): integrate HybridChunker substitution (pipeline.py)

Inserted _substitute_table_chunks call between chunker.chunk() and the
chunk-iteration loop (only fires when EMBEDDING_TABLE_NORMALIZATION_ENABLED
=true). _build_native_chunk_meta reads chunk.extra_metadata duck-typedly;
populates the new chunk_metadata return key. Postgres upsert set_ clause
now includes chunk_index + page_number + chunk_metadata for re-run safety.
ArcadeDB write adds chunk_kind property.

With master switch off (default): behavior is byte-identical to today
on the primary embedding path. Verified by test_master_kill_switch_byte
_equality (Task 19) and the existing embedding test suite.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 14: Integrate legacy `structure_aware_chunk` path

**Files:**
- Modify: `app/services/chunking.py:36, 113-126`
- Modify: `app/workers/pipeline.py:5636-5664` (legacy branch — thread `element_metadata`, normalized_tables, pass to chunker)

- [ ] **Step 1: Add optional `metadata` field to `StructuredChunk`**

Edit `app/services/chunking.py` around line 22 (the `@dataclass` declaration). Add:

```python
@dataclass
class StructuredChunk:
    """A structure-aware text chunk with document provenance."""
    text: str
    chunk_index: int
    modality: str
    page_number: Optional[int] = None
    section_path: Optional[str] = None
    element_uids: list[str] = field(default_factory=list)
    heading_text: Optional[str] = None
    metadata: Optional[dict] = None     # NEW — populated for normalized table chunks
```

- [ ] **Step 2: Modify `structure_aware_chunk` signature**

Add the `normalized_tables` keyword-only arg with empty-tuple default:

```python
def structure_aware_chunk(
    elements: list[dict],
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    *,
    normalized_tables: list | tuple = (),
) -> list[StructuredChunk]:
```

- [ ] **Step 3: Add `normalized_table_for` lookup**

Inside `structure_aware_chunk`, before the main loop:

```python
def normalized_table_for(elem: dict) -> "NormalizedTable | None":
    from app.services.table_normalization.models import Shape
    if not normalized_tables:
        return None
    ref = (elem.get("element_metadata") or {}).get("self_ref", "")
    if not ref.startswith("#/tables/"):
        return None
    try:
        idx = int(ref.split("/")[-1])
    except (ValueError, IndexError):
        return None
    nt = next((nt for nt in normalized_tables if nt.table_index == idx), None)
    if nt is None or nt.shape == Shape.OTHER:
        return None
    return nt
```

- [ ] **Step 4: Replace the `elif etype == "table":` branch (lines 113-126)**

```python
elif etype == "table":
    # Tables are always their own chunk — never split (or normalized).
    current_heading = buffer_heading
    _flush_buffer()

    from app.services.table_normalization.config import is_table_normalization_enabled_embedding
    if not is_table_normalization_enabled_embedding():
        # Master kill-switch: today's behavior unchanged.
        chunks.append(StructuredChunk(
            text=content, chunk_index=chunk_index,
            modality="table", page_number=page,
            section_path=section, element_uids=[uid],
            heading_text=current_heading,
        ))
        chunk_index += 1
    else:
        nt = normalized_table_for(elem)
        if nt is None:
            # Fall back to today's behavior (Shape.OTHER, missing self_ref, etc.)
            chunks.append(StructuredChunk(
                text=content, chunk_index=chunk_index,
                modality="table", page_number=page,
                section_path=section, element_uids=[uid],
                heading_text=current_heading,
            ))
            chunk_index += 1
        else:
            from app.services.table_normalization import render_for_embedding
            from app.services.table_normalization.config import (
                embedding_chunk_max_tokens, embedding_table_summary_max_tokens,
            )
            for etc in render_for_embedding(
                nt,
                token_limit=embedding_chunk_max_tokens(),
                summary_limit=embedding_table_summary_max_tokens(),
            ):
                chunks.append(StructuredChunk(
                    text=etc.text,
                    chunk_index=chunk_index,
                    modality="table",
                    page_number=etc.page_numbers[0] if etc.page_numbers else page,
                    section_path=section,
                    element_uids=[uid],
                    heading_text=current_heading,
                    metadata={
                        "chunk_kind": etc.chunk_kind.value,
                        "table_ref": etc.table_ref,
                        "entity_display_name": etc.entity_display_name,
                        "section": etc.section,
                        "column_index": etc.column_index,
                        "cell_refs": list(etc.cell_refs),
                        "row_labels": list(etc.row_labels),
                    },
                ))
                chunk_index += 1
```

- [ ] **Step 5: Thread `element_metadata` into `element_dicts` (pipeline.py:5647-5658)**

```python
element_dicts = [
    {
        "element_type": elem.element_type,
        "content_text": elem.translated_text or elem.content_text,
        "page_number": elem.page_number,
        "section_path": elem.section_path,
        "element_uid": str(elem.element_uid) if elem.element_uid else "",
        "element_order": elem.element_order,
        "heading_level": elem.heading_level,
        "element_metadata": elem.element_metadata or {},   # NEW
    }
    for elem in elements
    if (elem.translated_text or elem.content_text)
]
```

- [ ] **Step 6: Compute `normalized_tables` in legacy branch + pass to chunker**

Before the `structure_aware_chunk(...)` call at line 5660:

```python
# Best-effort: load docling_document.json for normalization. If unreadable,
# normalized_tables stays empty → today's behavior preserved on this path.
normalized_tables: list = []
if is_table_normalization_enabled_embedding():
    try:
        _raw = download_bytes_sync(
            settings.minio_bucket_derived,
            f"artifacts/{document_id}/docling_document.json",
        )
        from app.services.table_normalization import normalize_tables as _normalize
        normalized_tables = _normalize(_json_mod.loads(_raw))
    except Exception as exc:
        logger.debug(
            "Legacy path: docling_document.json unavailable for normalization (%s); "
            "tables pass through as today's opaque chunks.", exc,
        )

structured_chunks = structure_aware_chunk(
    element_dicts,
    max_chunk_tokens=settings.embedding_chunk_max_tokens,
    overlap_tokens=settings.embedding_chunk_overlap_tokens,
    normalized_tables=normalized_tables,   # NEW
)
```

- [ ] **Step 7: Update the legacy-path TextChunk write to include `chunk_metadata`**

After `structured_chunks = ...`, find where `StructuredChunk` rows write to `TextChunk` (look around line 5680+ for the `pg_insert(TextChunk).values(...)` pattern). Add `chunk_metadata` from `sc.metadata` to the insert values dict and the `set_` clause (mirror Task 13 Step 3).

Then update the ArcadeDB write similarly.

- [ ] **Step 8: Verify the legacy path with flag off doesn't regress**

```bash
# Force the legacy path by disabling enrichments on one test doc
# OR run an existing test that exercises the legacy path
.venv/bin/pytest tests/integration/ -k "legacy or fallback" -v
```

Expected: no regressions; existing tests pass.

- [ ] **Step 9: Commit**

```bash
git add app/services/chunking.py app/workers/pipeline.py
git commit -m "feat(table-norm): integrate legacy structure_aware_chunk path

- StructuredChunk gains optional metadata field.
- structure_aware_chunk gains normalized_tables kwarg.
- normalized_table_for resolves element → NormalizedTable via element_metadata.self_ref.
- pipeline.py legacy branch threads element_metadata + computes
  normalized_tables (best-effort load of docling_document.json).
- Legacy-path TextChunk write mirrors primary path's chunk_metadata
  handling.

With master switch off: byte-identical to today on the legacy path
(asserted by Task 19 test).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 15: Suppression helper — `_suppress_raw_table_texts`

**Files:**
- Modify: `app/services/table_normalization/__init__.py` (add `_suppress_raw_table_texts` from `_pipeline_hooks` or a new module)
- Create: `tests/unit/test_suppress_raw_table_texts_invariant.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_suppress_raw_table_texts_invariant.py
import json
from pathlib import Path
from app.services.table_normalization import normalize_tables
from app.services.table_normalization._pipeline_hooks import _suppress_raw_table_texts


def test_blanks_in_place_does_not_remove_entries():
    doc = {
        "tables": [{"table_cells": [], "text": "raw"}],  # OTHER table — not suppressed
        "texts": [
            {"self_ref": "#/texts/0", "text": "prose 1"},
            {"self_ref": "#/texts/1", "text": "prose 2", "orig": "prose 2 orig"},
        ],
    }
    normalized = normalize_tables(doc)
    initial_len = len(doc["texts"])
    _suppress_raw_table_texts(doc, normalized)
    # No entries removed (Shape.OTHER table; nothing suppressed)
    assert len(doc["texts"]) == initial_len


def test_blanks_only_target_self_refs():
    # Build a doc with a normalizable table + a flat-text mirror
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    doc = {
        "tables": [sa2],
        "texts": [
            {"self_ref": "#/texts/0", "text": "prose", "orig": "prose"},
            {"self_ref": "#/tables/0", "text": "the flattened table text", "orig": "the flattened table text"},
            {"self_ref": "#/texts/1", "text": "more prose", "orig": "more prose"},
        ],
    }
    normalized = normalize_tables(doc)
    _suppress_raw_table_texts(doc, normalized)

    # texts[0] and texts[2] (prose) unchanged
    assert doc["texts"][0]["text"] == "prose"
    assert doc["texts"][2]["text"] == "more prose"

    # texts[1] (the table mirror) blanked
    assert doc["texts"][1]["text"] == ""
    assert doc["texts"][1]["orig"] == ""

    # All entries still present (no reindexing)
    assert len(doc["texts"]) == 3


def test_tables_array_not_mutated():
    """CRITICAL: doc_json['tables'] is byte-identical pre/post."""
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    doc = {
        "tables": [sa2],
        "texts": [{"self_ref": "#/tables/0", "text": "x", "orig": "x"}],
    }
    before = json.dumps(doc["tables"], sort_keys=True)
    normalized = normalize_tables(doc)
    _suppress_raw_table_texts(doc, normalized)
    after = json.dumps(doc["tables"], sort_keys=True)
    assert before == after, "tables[] was mutated — overlay path will break"


def test_other_shape_tables_preserved():
    """Tables with Shape.OTHER keep their flat-text mirror."""
    doc = {
        "tables": [{"table_cells": [], "text": "raw"}],  # OTHER
        "texts": [{"self_ref": "#/tables/0", "text": "raw flat", "orig": "raw flat"}],
    }
    normalized = normalize_tables(doc)
    _suppress_raw_table_texts(doc, normalized)
    # OTHER table → not suppressed; flat text preserved
    assert doc["texts"][0]["text"] == "raw flat"
```

- [ ] **Step 2: Run tests to verify failure**

Expected: ImportError on `_suppress_raw_table_texts`.

- [ ] **Step 3: Add to `_pipeline_hooks.py`**

```python
# Append to app/services/table_normalization/_pipeline_hooks.py
def _suppress_raw_table_texts(
    doc_json: dict,
    normalized: list[NormalizedTable],
) -> None:
    """Blank the flat-text mirrors of normalized non-OTHER tables in-place.

    Per §9.2 invariant:
    - len(doc_json['texts']) is UNCHANGED. No element is removed; no
      index shifts. This preserves self_ref stability for any code that
      references texts by index (children refs, prov entries, etc.).
    - doc_json['tables'] is NOT touched. The Phase 0/0.5 overlay
      machinery reads tables[] directly and must remain functional.
    - Tables with shape == OTHER keep their flat text (the OTHER
      fallback depends on it).
    """
    non_other = {nt.table_index for nt in normalized if nt.shape != Shape.OTHER}
    if not non_other:
        return
    target_refs = {f"#/tables/{i}" for i in non_other}
    for t in doc_json.get("texts") or []:
        if t.get("self_ref") in target_refs:
            t["text"] = ""
            t["orig"] = ""
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_suppress_raw_table_texts_invariant.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/table_normalization/_pipeline_hooks.py tests/unit/test_suppress_raw_table_texts_invariant.py
git commit -m "feat(table-norm): _suppress_raw_table_texts — blank in place

Per §9.2 invariant: blanks text + orig fields in-place rather than
removing list entries (matches existing sanitizer pattern at
main.py:379). Preserves self_ref stability across doc_json. tables[]
itself is not touched — Phase 0/0.5 overlay machinery keeps working.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 16: Add `chunk_index` + `cell_refs` to `ExtractionFieldProvenance`

**Files:**
- Modify: `docker/docling-graph/app/schemas.py:195-233`
- Create: `tests/unit/test_extraction_field_provenance_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_extraction_field_provenance_schema.py
import pytest
from app.schemas import ExtractionFieldProvenance  # adjusted import path for docling-graph


def test_chunk_index_field_exists():
    p = ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
        chunk_index=3,
    )
    assert p.chunk_index == 3


def test_cell_refs_field_exists_default_empty():
    p = ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
    )
    assert p.cell_refs == []


def test_cell_refs_accepts_table_cell_refs():
    p = ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
        cell_refs=["#/tables/3/data/table_cells/42", "#/tables/3/data/table_cells/43"],
    )
    assert len(p.cell_refs) == 2
    assert p.cell_refs[0].startswith("#/tables/")


def test_serialization_preserves_new_fields():
    p = ExtractionFieldProvenance(
        instance_id="x", field_name="f", supporting_snippet="s",
        chunk_index=3, cell_refs=["#/tables/0/data/table_cells/0"],
    )
    d = p.model_dump()
    assert d["chunk_index"] == 3
    assert d["cell_refs"] == ["#/tables/0/data/table_cells/0"]
```

- [ ] **Step 2: Run tests to verify failure**

The test runs in the docling-graph environment, not the main app. Adjust the test runner appropriately:

```bash
cd docker/docling-graph && .venv/bin/pytest tests/unit/test_extraction_field_provenance_schema.py -v
```

Expected: AttributeError or validation error on missing fields.

- [ ] **Step 3: Modify `docker/docling-graph/app/schemas.py:195-233`**

Add two new fields to `ExtractionFieldProvenance`:

```python
class ExtractionFieldProvenance(BaseModel):
    """Wire-shape per-field provenance row (spec §5.3, §5.1.1).
    ...existing docstring..."""
    instance_id: str = Field(..., description="...")
    field_name: str = Field(..., description="...")
    value: Any = Field(None, description="...")
    supporting_snippet: str = Field(..., description="...")
    element_uid: Optional[str] = Field(None, description="...")
    evidence_id: Optional[str] = Field(default=None, description="...")
    page: Optional[int] = Field(default=None, description="...")
    document_id: Optional[str] = Field(default=None, description="...")

    # NEW (spec 2026-05-11 §11.6 — channel A cell-ref provenance):
    chunk_index: Optional[int] = Field(
        default=None,
        description=(
            "Library chunk_id (0..N-1, sequential per pass) the field was "
            "extracted from. Required for the cell_refs lookup. None for "
            "rows where chunk_index cannot be determined."
        ),
    )
    cell_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Cell-level self_refs of the form '#/tables/{N}/data/table_cells/{M}' "
            "when the field was extracted from a chunk synthesized from a "
            "NormalizedTable. Empty for prose chunks. Populated post-construction "
            "by _enrich_field_provenance_with_cell_refs via the two-hop lookup "
            "(chunk_index → chunk_to_self_refs → first #/texts/N → bridge → cell_refs)."
        ),
    )
```

- [ ] **Step 4: Run tests**

```bash
cd docker/docling-graph && .venv/bin/pytest tests/unit/test_extraction_field_provenance_schema.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add docker/docling-graph/app/schemas.py docker/docling-graph/tests/unit/test_extraction_field_provenance_schema.py
git commit -m "feat(table-norm): ExtractionFieldProvenance — chunk_index + cell_refs

Two new optional fields added to the wire-shape per-field provenance
row (additive, backwards-compatible). chunk_index is the library
chunk_id; cell_refs lists '#/tables/{N}/data/table_cells/{M}' for
fields extracted from normalized table chunks.

Populated post-construction via the two-hop lookup at the
field-provenance enrichment site (Task 17).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 17: Graph-side integration in docling-graph + field-provenance enrichment

**Files:**
- Create: `app/services/table_normalization/_text_item.py` (the `_text_item_from_chunk` helper)
- Modify: `docker/docling-graph/app/main.py:564` (insertion: normalize + integration block)
- Modify: `docker/docling-graph/app/main.py:794` (insertion: enrichment call)
- Modify: `docker/docling-graph/Dockerfile` (COPY directive)
- Create: `tests/unit/test_text_item_from_chunk.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_text_item_from_chunk.py
import json
from pathlib import Path
from app.services.table_normalization import normalize_tables, render_for_graph
from app.services.table_normalization._text_item import _text_item_from_chunk
from app.services.table_normalization import _provenance_bridge as bridge


def test_text_item_assigns_self_ref_and_records_bridge():
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    nt = normalize_tables({"tables": [sa2], "texts": []})[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)

    bridge.reset()
    next_text_idx = 100
    items: list = []
    for c in chunks:
        item, next_text_idx = _text_item_from_chunk(c, next_text_idx=next_text_idx)
        items.append(item)

    # Each item has a #/texts/N self_ref
    for i, item in enumerate(items):
        assert item["self_ref"] == f"#/texts/{100 + i}"

    # Bridge map contains entries for each chunk with cell_refs
    for i, c in enumerate(chunks):
        if c.cell_refs:
            assert bridge.cell_refs_for_text_idx(100 + i) == list(c.cell_refs)


def test_text_item_empty_prov_for_table_whole_other_fallback():
    """Shape.OTHER chunks have empty cell_refs; bridge not populated."""
    nt = normalize_tables({"tables": [{"table_cells": [], "text": "raw"}], "texts": []})[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)
    bridge.reset()
    item, _ = _text_item_from_chunk(chunks[0], next_text_idx=42)
    assert bridge.cell_refs_for_text_idx(42) == []
```

- [ ] **Step 2: Run tests to verify failure**

Expected: ImportError on `_text_item`.

- [ ] **Step 3: Implement `_text_item.py`**

```python
# app/services/table_normalization/_text_item.py
"""Convert a GraphTableChunk into a docling TextItem dict for texts[].

Mirrors the pattern at _table_facts.py:818-826: self_ref is hand-rolled
as f"#/texts/{next_text_idx}"; the caller threads next_text_idx and
bumps it after each call.

Records (text_idx, cell_refs) in _provenance_bridge for downstream
field-provenance enrichment.
"""
from __future__ import annotations

from typing import Tuple
from app.services.table_normalization.models import GraphTableChunk
from app.services.table_normalization._provenance_bridge import record_text_idx_cell_refs


def _text_item_from_chunk(
    gtc: GraphTableChunk,
    *,
    next_text_idx: int,
) -> Tuple[dict, int]:
    """Build a docling TextItem dict for a GraphTableChunk.

    Returns (text_item, next_text_idx + 1). Caller must thread the
    returned next_text_idx into subsequent calls to avoid collisions
    with existing #/texts/N entries.
    """
    record_text_idx_cell_refs(next_text_idx, list(gtc.cell_refs))

    return ({
        "self_ref": f"#/texts/{next_text_idx}",
        "label": "text",
        "prov": [],  # empty — cell refs flow through the bridge, not prov[].$ref
        "orig": gtc.text,
        "text": gtc.text,
    }, next_text_idx + 1)
```

- [ ] **Step 4: Run unit tests**

```bash
.venv/bin/pytest tests/unit/test_text_item_from_chunk.py -v
```

Expected: both PASS.

- [ ] **Step 5: Add the Dockerfile COPY directive**

Edit `docker/docling-graph/Dockerfile`. Find the section that copies app code (likely a `COPY app/services` or similar). Add:

```dockerfile
COPY app/services/table_normalization /app/app/services/table_normalization
```

If `app/services/` is already COPYd wholesale, this directive is redundant — verify by reading the existing Dockerfile and only add if not covered.

- [ ] **Step 6: Insert normalize + integration block in `main.py` at line ~564**

In `docker/docling-graph/app/main.py`, between line 564 (after sanitization) and line 566 (the reverted-block comment), insert:

```python
# Table normalization (spec 2026-05-11 §9.1). Master switch defaults off
# at app config; flag matrix at §9.1 controls the four combinations.
from app.services.table_normalization import normalize_tables, render_for_graph
from app.services.table_normalization.config import (
    is_table_normalization_enabled_graph,
    is_experimental_table_facts_enabled,
    is_suppress_raw_table_markdown_enabled,
    table_whole_limit,
    table_column_limit,
)
from app.services.table_normalization._pipeline_hooks import _suppress_raw_table_texts
from app.services.table_normalization._text_item import _text_item_from_chunk
from app.services.table_normalization._provenance_bridge import reset as _bridge_reset

# Reset per-pass bridge state to prevent cross-pass leakage.
_bridge_reset()

_norm_on = is_table_normalization_enabled_graph()
_exp_on = is_experimental_table_facts_enabled()

if _norm_on and _exp_on:
    logger.error(
        "Both DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED and "
        "DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS are true; falling back to "
        "off+off (today's production behavior). Set only one."
    )
elif _norm_on:
    # New path: normalize + render + append + suppress
    normalized = normalize_tables(docling_document_json)
    next_text_idx = len(docling_document_json.get("texts") or [])
    for nt in normalized:
        for gtc in render_for_graph(
            nt,
            token_limit_whole=table_whole_limit(),
            token_limit_column=table_column_limit(),
        ):
            text_item, next_text_idx = _text_item_from_chunk(gtc, next_text_idx=next_text_idx)
            docling_document_json.setdefault("texts", []).append(text_item)
    if is_suppress_raw_table_markdown_enabled():
        _suppress_raw_table_texts(docling_document_json, normalized)
elif _exp_on:
    # Experimental path: re-enable reverted _table_facts.py
    from app._table_facts import synthesize_table_facts
    synthesize_table_facts(docling_document_json, active_pass=pass_name)
# else: today's production behavior — nothing inserted into texts[].
```

(Adjust variable names like `pass_name`, `docling_document_json` to match actual locals in `main.py`.)

- [ ] **Step 7: Insert field-provenance enrichment call at `main.py:794`**

Find where `context._chunk_to_self_refs` is set (around line 794, after the trace-map building). Just after, add:

```python
# Enrich field provenance with cell_refs (spec 2026-05-11 §11.6 channel A).
from app.services.table_normalization._provenance_bridge import cell_refs_for_text_idx
import re
_TEXTS_REF_RE = re.compile(r"^#/texts/(\d+)$")

def _enrich_field_provenance_with_cell_refs(
    rows: list,
    chunk_to_self_refs: dict[int, list[str]],
) -> list:
    out = []
    for r in rows:
        ci = getattr(r, "chunk_index", None)
        if ci is None:
            out.append(r); continue
        cell_refs: list[str] = []
        for ref in (chunk_to_self_refs.get(int(ci), []) or []):
            m = _TEXTS_REF_RE.match(ref)
            if not m:
                continue
            crefs = cell_refs_for_text_idx(int(m.group(1)))
            if crefs:
                cell_refs = crefs
                break
        if cell_refs:
            r = r.model_copy(update={"cell_refs": cell_refs})
        out.append(r)
    return out

# Apply enrichment to the response's field_provenance list before serialization.
# Find where field_provenance is assembled (search main.py for "field_provenance");
# wrap with _enrich_field_provenance_with_cell_refs(..., chunk_to_self_refs).
```

This must be placed at the actual field-provenance assembly site. Locate it via:

```bash
grep -n "field_provenance" docker/docling-graph/app/main.py
```

**If multiple sites surface:** wrap the response-serialization site (the latest point in the request lifecycle that touches `field_provenance` before returning). Wrapping earlier risks the rows being rebuilt downstream without `cell_refs`.

- [ ] **Step 8: Rebuild docling-graph image (required — COPY semantics)**

```bash
docker compose build docling-graph
docker compose up -d docling-graph
```

Per `feedback_container_rebuild`: docling-graph uses COPY in Dockerfile, so rebuild is required for code changes (bind-mounts don't apply).

- [ ] **Step 9: Run existing graph-extraction tests**

```bash
.venv/bin/pytest tests/integration/ -k "graph or extraction" -v
```

Expected: existing tests still pass (with master switches off).

- [ ] **Step 10: Commit**

```bash
git add app/services/table_normalization/_text_item.py docker/docling-graph/app/main.py docker/docling-graph/Dockerfile tests/unit/test_text_item_from_chunk.py
git commit -m "feat(table-norm): graph-side integration (main.py) + _text_item

- _text_item_from_chunk converts GraphTableChunk → docling TextItem dict;
  records (text_idx, cell_refs) in the provenance bridge for downstream
  field-provenance enrichment.
- main.py:564 flag-matrix integration block per §9.1 (4 combinations).
- main.py:794 field-provenance enrichment call adds cell_refs to
  ExtractionFieldProvenance via two-hop lookup.
- Dockerfile COPY for the new module (required since docling-graph
  uses COPY semantics, not bind-mounts).

With master switches off (default): today's production behavior preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## End of Chunk 2

Chunk 2 completes the database migration and the cross-pipeline integration:
- Postgres schema (Task 10) + ArcadeDB schema (Task 11)
- Pipeline hooks module (Task 12)
- HybridChunker primary path (Task 13)
- Legacy `structure_aware_chunk` path (Task 14)
- Suppression helper (Task 15)
- `ExtractionFieldProvenance` schema additions (Task 16)
- Graph-side integration (Task 17)

After Chunk 2, all code paths are wired but all switches default off. Chunk 3 covers retrieval surfacing, merge gates, and the rollout dance.

---

## Chunk 3: Phase 4 (Retrieval) + Phase 5 (Gates) + Phase 6 (Rollout)

## Phase 4 — Retrieval surfacing

### Task 18: `/v1/retrieval/query` table_chunk block

**Files:**
- Modify: `app/api/v1/retrieval.py` (or wherever the retrieval endpoint lives)
- Modify: The response schema for `/v1/retrieval/query`
- Create: `tests/integration/test_retrieval_table_chunk_surfacing.py`

- [ ] **Step 1: Locate the retrieval endpoint**

```bash
grep -rn "v1/retrieval/query\|@router.post.*retrieval" app/api/ | head -5
```

Read the endpoint and the response model.

- [ ] **Step 2: Add the optional `table_chunk` block to the response schema**

Find the response Pydantic model for query results (e.g., `RetrievalResultItem`). Add:

```python
class TableChunkBlock(BaseModel):
    kind: str
    table_ref: str
    table_caption: Optional[str] = None
    entity_display_name: Optional[str] = None
    section: Optional[str] = None
    cell_refs: list[str] = Field(default_factory=list)
    row_labels: list[str] = Field(default_factory=list)


class RetrievalResultItem(BaseModel):
    # ... existing fields ...
    table_chunk: Optional[TableChunkBlock] = None
```

- [ ] **Step 3: Populate `table_chunk` from `TextChunk.chunk_metadata` in the endpoint**

In the retrieval endpoint, where each result is constructed, add:

```python
table_chunk_block = None
if row.chunk_metadata:
    cm = row.chunk_metadata
    table_chunk_block = TableChunkBlock(
        kind=cm.get("chunk_kind", "unknown"),
        table_ref=cm.get("table_ref", ""),
        table_caption=cm.get("table_caption"),
        entity_display_name=cm.get("entity_display_name"),
        section=cm.get("section"),
        cell_refs=cm.get("cell_refs", []),
        row_labels=cm.get("row_labels", []),
    )

result = RetrievalResultItem(
    # ... existing fields ...
    table_chunk=table_chunk_block,
)
```

- [ ] **Step 4: Write the integration test**

```python
# tests/integration/test_retrieval_table_chunk_surfacing.py
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.retrieval import TextChunk
from app.models.ingest import Document
from app.db.session import get_db_sync


def test_table_chunk_block_present_when_chunk_metadata_set():
    """Retrieval endpoint returns table_chunk block when chunk_metadata is non-null."""
    client = TestClient(app)

    with get_db_sync() as db:
        doc = Document(id=uuid.uuid4(), document_metadata={"title": "test"})
        db.add(doc)
        db.flush()
        chunk = TextChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            chunk_text="S-75M2 max range 56000",
            modality="table",
            chunk_metadata={
                "chunk_kind": "table_entity_column",
                "table_ref": "#/tables/3",
                "entity_display_name": "S-75M2",
                "cell_refs": ["#/tables/3/data/table_cells/42"],
                "row_labels": ["Max Range"],
            },
        )
        db.add(chunk)
        db.commit()

    # Query the endpoint with a matching query
    response = client.post("/v1/retrieval/query", json={
        "query": "S-75M2 max range",
        "top_k": 5,
    })
    assert response.status_code == 200
    results = response.json().get("results", [])
    assert results, "no results returned"
    # At least one result should carry the table_chunk block
    table_results = [r for r in results if r.get("table_chunk")]
    assert table_results, "no result carries table_chunk block"
    tc = table_results[0]["table_chunk"]
    assert tc["kind"] == "table_entity_column"
    assert tc["entity_display_name"] == "S-75M2"
    assert "#/tables/3/data/table_cells/42" in tc["cell_refs"]


def test_table_chunk_omitted_when_chunk_metadata_null():
    """Prose chunks (chunk_metadata IS NULL) have no table_chunk block."""
    client = TestClient(app)
    # Similar setup but without chunk_metadata
    with get_db_sync() as db:
        doc = Document(id=uuid.uuid4(), document_metadata={})
        db.add(doc)
        db.flush()
        chunk = TextChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            chunk_text="plain prose chunk",
            modality="text",
            chunk_metadata=None,
        )
        db.add(chunk)
        db.commit()

    response = client.post("/v1/retrieval/query", json={
        "query": "plain prose",
        "top_k": 5,
    })
    assert response.status_code == 200
    results = response.json().get("results", [])
    for r in results:
        if r.get("text") == "plain prose chunk":
            assert r.get("table_chunk") is None
            return
    pytest.fail("our seeded chunk didn't appear in results")
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/integration/test_retrieval_table_chunk_surfacing.py -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add app/api/v1/retrieval.py tests/integration/test_retrieval_table_chunk_surfacing.py
git commit -m "feat(retrieval): surface table_chunk block on /v1/retrieval/query

Additive optional field on the response item. Populated from
TextChunk.chunk_metadata JSONB; omitted when null. Backwards-compatible
with existing consumers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5 — Merge gates + drift guards

### Task 19: Master kill-switch byte-equality test

**Files:**
- Create: `tests/integration/test_master_kill_switch_byte_equality.py`

This is the MERGE GATE for Phase 1 — proves all code lands without changing production behavior when both flags are off.

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_master_kill_switch_byte_equality.py
"""Phase 1 merge gate (spec §15.1).

With both *_NORMALIZATION_ENABLED=false (default), the new code must
produce byte-identical chunks vs the §19 baseline fixtures.
"""
import json
import os
import pytest
from pathlib import Path


BASELINE_DIR = Path("tests/fixtures/sa2")


@pytest.fixture
def baseline_meta():
    meta_path = BASELINE_DIR / "baseline.meta.json"
    if not meta_path.exists():
        pytest.skip(f"baseline not captured at {meta_path}; run Task 0a first")
    return json.loads(meta_path.read_text())


def test_master_kill_switch_doc_json_texts_byte_identical(baseline_meta, monkeypatch):
    """doc_json['texts'] after sanitization matches §19 baseline."""
    # Ensure flags off
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("EMBEDDING_TABLE_NORMALIZATION_ENABLED", "false")
    monkeypatch.setenv("DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS", "false")

    for docid_file in BASELINE_DIR.glob("*_texts_today.json"):
        docid = docid_file.stem.replace("_texts_today", "")
        expected = json.loads(docid_file.read_text())

        # Reproduce: ingest this doc up through sanitization on the new code
        # The actual mechanism depends on the user's ingest flow. Two options:
        #   (a) Run the full ingest command from baseline_meta["ingest_command"]
        #       on the new code, capture texts via the CAPTURE_BASELINE_TEXTS
        #       hook (temporarily re-enabled for the test).
        #   (b) Call the sanitizer + texts assembler as pure functions on a
        #       known input doc, compare outputs.
        # Option (b) is preferred for test isolation. If the sanitizer is
        # tightly coupled to FastAPI request lifecycle, use option (a).
        #
        # Implementation: see actual code in docker/docling-graph/app/main.py
        # for the sanitizer call site; expose a wrapper if needed for testing.

        actual = _reproduce_texts_for_doc(docid)
        assert actual == expected, f"doc {docid}: texts diverged from baseline"


def _reproduce_texts_for_doc(docid: str) -> list:
    """Wrapper to reproduce `doc_json['texts']` after sanitization for a doc.

    Implementer: wire to the actual sanitizer entry point. If extract-pass
    must run end-to-end, capture via test fixture instead.
    """
    raise NotImplementedError("wire to the actual sanitizer pipeline")


def test_postgres_chunk_metadata_is_null_with_flags_off(baseline_meta, monkeypatch, db):
    """All TextChunk rows have chunk_metadata IS NULL when flags are off."""
    from sqlalchemy import text
    null_count = db.execute(text(
        "SELECT COUNT(*) FROM retrieval.text_chunks WHERE chunk_metadata IS NULL"
    )).scalar()
    total = db.execute(text("SELECT COUNT(*) FROM retrieval.text_chunks")).scalar()
    assert null_count == total, "some chunks have non-NULL chunk_metadata with flags off"


def test_arcadedb_chunk_kind_absent_or_null_with_flags_off(baseline_meta, monkeypatch):
    """All ArcadeDB TextChunk vertices have chunk_kind == None when flags off."""
    from app.db.session import get_graph_store
    gs = get_graph_store()
    result = gs.execute_sync("SELECT chunk_kind FROM TextChunk LIMIT 100")
    for row in result:
        ck = row.get("chunk_kind")
        assert ck is None or ck == "", f"unexpected chunk_kind={ck!r}"
```

- [ ] **Step 2: Decide on the reproduction mechanism**

Decide whether to:
- (a) Run the full ingest command from baseline + capture via temp hook
- (b) Refactor the sanitizer into a testable pure function and call directly

**Prefer option (b)** for CI runnability — this is the Phase 1 merge gate; running the full ingest stack in CI is fragile. Option (a) is fine for local development if option (b) requires substantial refactoring. Document the choice in the test file's docstring.

- [ ] **Step 3: Implement `_reproduce_texts_for_doc`**

Wire to the actual sanitizer pipeline per the choice above.

- [ ] **Step 4: Run the test**

```bash
.venv/bin/pytest tests/integration/test_master_kill_switch_byte_equality.py -v
```

Expected: all 3 PASS, OR test surfaces a real divergence to fix before merging.

If divergence found: trace which file's change introduced it. Most likely culprits:
- New `metadata` field on `StructuredChunk` — verify it's `None` by default and isn't serialized into `chunk_text`.
- New `chunk_metadata` upsert behavior — verify the `set_` clause doesn't fire when flag is off.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_master_kill_switch_byte_equality.py
git commit -m "test(table-norm): master kill-switch byte-equality merge gate

With both *_NORMALIZATION_ENABLED=false: doc_json['texts'] is byte-
identical to §19 baseline; chunk_metadata IS NULL in Postgres; ArcadeDB
chunk_kind property is None. This is the Phase 1 merge gate (§15.1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 20: Disallowed-combination fallback test

**Files:**
- Create: `tests/integration/test_disallowed_combination_fallback.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_disallowed_combination_fallback.py
"""When both DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED and
DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS are true, the integration
code at main.py:564 must log an ERROR and fall back to today's
behavior (neither path fires).
"""
import logging
import pytest


def test_both_flags_true_falls_back_to_today(caplog, monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "true")
    monkeypatch.setenv("DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS", "true")

    # Trigger the integration code (mock or real extract-pass)
    # Assert: ERROR log emitted; doc_json['texts'] unchanged
    with caplog.at_level(logging.ERROR):
        result = _trigger_integration_block_on_test_doc()

    error_logs = [r for r in caplog.records if "Both" in r.getMessage() and "true" in r.getMessage()]
    assert error_logs, "no ERROR log for disallowed combination"
    # texts unchanged from baseline
    assert result == _expected_baseline_texts()
```

- [ ] **Step 2: Implement test helpers**

Same approach as Task 19 — either real extract-pass or unit-test wrapper around the integration block.

- [ ] **Step 3: Run + commit**

```bash
.venv/bin/pytest tests/integration/test_disallowed_combination_fallback.py -v
git add tests/integration/test_disallowed_combination_fallback.py
git commit -m "test(table-norm): disallowed-combination fallback test

Both *_ENABLED + experimental flag → ERROR log + fallback to today's
raw-blob behavior. Per §9.1 flag matrix row 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 21: Experimental-path drift guard

**Files:**
- Create: `tests/integration/test_legacy_table_facts_drift.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_legacy_table_facts_drift.py
"""Drift guard: ensure _table_facts.py's synthesize_table_facts still
produces non-empty output on SA-2 fixtures. Count-based assertion
(±10% tolerance) tolerates whitespace drift but catches catastrophic
rot (returns zero items, or 10× too many).
"""
import json
import pytest
from pathlib import Path


@pytest.fixture
def sa2_doc():
    return json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def test_synthesize_table_facts_still_emits(sa2_doc):
    from docker.docling_graph.app._table_facts import synthesize_table_facts

    doc = {"tables": [sa2_doc], "texts": []}
    pre_count = len(doc["texts"])
    synthesize_table_facts(doc, active_pass="missile_propulsion")
    post_count = len(doc["texts"])

    new_items = post_count - pre_count
    EXPECTED = 12  # Captured empirically once after fixture lands; update on intentional changes
    assert new_items > 0, "synthesize_table_facts emitted zero items"
    assert abs(new_items - EXPECTED) / EXPECTED <= 0.10, \
        f"emitted {new_items} items, expected ~{EXPECTED} ± 10%"
```

- [ ] **Step 2: Capture the EXPECTED count once on the new code**

Run the synthesizer once on the fixture, observe the actual count, **replace the `EXPECTED = 12` placeholder in the test with the observed value**, then commit. Future drift fails the test.

- [ ] **Step 3: Run + commit**

```bash
.venv/bin/pytest tests/integration/test_legacy_table_facts_drift.py -v
git add tests/integration/test_legacy_table_facts_drift.py
git commit -m "test(table-norm): experimental _table_facts.py drift guard

Count-based assertion (±10% tolerance) on SA-2 fixture catches
catastrophic rot in the experimental path that we preserve for A/B.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 22: Graph-side provenance cell_refs test

**Files:**
- Create: `tests/integration/test_graph_provenance_cell_refs.py`

This is the MERGE GATE for the cell-ref provenance feature (Channel A from §11.6).

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_graph_provenance_cell_refs.py
"""§15.2 provenance gate.

With graph-side normalization on, extract an entity from a SA-2 fixture.
Assert ExtractionFieldProvenance.cell_refs is populated with at least
one #/tables/N/data/table_cells/M ref.
"""
import json
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def enable_normalization(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "true")
    monkeypatch.setenv("DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS", "false")


def test_field_provenance_carries_cell_refs():
    """Extract Missile.max_range_m on S-75M2; assert cell_refs populated."""
    # Use the smallest SA-2 fixture that produces an extraction
    # Run extract-pass; inspect response.field_provenance
    response = _run_extract_pass_on_sa2_fixture()
    field_prov = response["field_provenance"]

    # Find S-75M2.max_range_m row
    target = [
        fp for fp in field_prov
        if fp.get("field_name") == "max_range_m"
        # ... add entity-identity filter as appropriate ...
    ]
    assert target, "no max_range_m provenance row found"

    # At least one must carry cell_refs pointing to the SA-2 table
    with_refs = [fp for fp in target if fp.get("cell_refs")]
    assert with_refs, f"max_range_m has no cell_refs; rows: {target}"

    fp = with_refs[0]
    for ref in fp["cell_refs"]:
        assert "/data/table_cells/" in ref, f"unexpected cell_ref shape: {ref}"
```

- [ ] **Step 2: Implement `_run_extract_pass_on_sa2_fixture`**

Use the FastAPI TestClient against the docling-graph service, or a direct call to `run_extraction_pass`.

- [ ] **Step 3: Run + commit**

```bash
.venv/bin/pytest tests/integration/test_graph_provenance_cell_refs.py -v
git add tests/integration/test_graph_provenance_cell_refs.py
git commit -m "test(table-norm): provenance gate — cell_refs on ExtractionFieldProvenance

§15.2 merge gate. With graph normalization on, extracted SA-2 fields
carry cell-level provenance through the channel-A two-hop lookup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 23: End-to-end SA-2 pipeline test

**Files:**
- Create: `tests/integration/test_sa2_table_pipeline_e2e.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_sa2_table_pipeline_e2e.py
"""End-to-end: ingest SA-2 sample → extract → score → cell_refs traceable.

Smaller scope than the §19 baseline comparison (which runs the full corpus);
this is a single-doc sanity check that the whole pipeline wires together.
"""
import pytest


@pytest.fixture(autouse=True)
def enable_both(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_TABLE_NORMALIZATION_ENABLED", "true")


def test_sa2_sample_ingest_extract_retrieve():
    # 1. Ingest the SA-2 sample doc
    # 2. Verify TextChunk rows include rows with chunk_kind = "table_entity_*"
    # 3. Verify extraction produces a Missile entity for S-75M2
    # 4. Verify the missile's max_range_m has cell_refs
    # 5. Verify /v1/retrieval/query for "S-75M2 max range" returns
    #    top-3 with at least one table_chunk-shaped result
    pass  # Wire to actual ingest/extract/retrieve harness
```

- [ ] **Step 2: Implement against the actual stack**

Use the existing integration test patterns in `tests/integration/`. This test exercises every component end-to-end.

- [ ] **Step 3: Run + commit**

```bash
.venv/bin/pytest tests/integration/test_sa2_table_pipeline_e2e.py -v
git add tests/integration/test_sa2_table_pipeline_e2e.py
git commit -m "test(table-norm): e2e sanity check — SA-2 ingest → extract → retrieve

Single-doc end-to-end test; complements the §19 baseline-comparison
flow (Task 21 rollout step) which runs the full corpus.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6 — Rollout

### Task 24: Delete deprecated files + update VERIFICATION_CHECKLIST.md

**Files:**
- Delete: `docker/docling-graph/app/_table_pivot.py`
- Delete: `docker/docling-graph/tests/test_table_pivot.py`
- Modify: `VERIFICATION_CHECKLIST.md`

- [ ] **Step 1: Re-verify deletion safety**

```bash
grep -rn "_table_pivot" --include="*.py" . 2>/dev/null | grep -v ".venv"
```

Expected: only `_table_pivot.py` and `test_table_pivot.py` themselves appear (no other consumers).

- [ ] **Step 2: Delete the two files**

```bash
git rm docker/docling-graph/app/_table_pivot.py
git rm docker/docling-graph/tests/test_table_pivot.py
```

- [ ] **Step 3: Update VERIFICATION_CHECKLIST.md**

Append the rows per §14:

```markdown
## Table-aware chunking (spec 2026-05-11)

- [ ] Step 0a baseline fixtures committed under tests/fixtures/sa2/ (main_sha recorded in baseline.meta.json)
- [ ] Step 0b spike completed: provenance-flow tests pass (with fixes from §20 applied if needed)
- [ ] Phase 1 merge: master kill-switch byte-equality test passes against §19 baseline (both HybridChunker and legacy paths)
- [ ] Phase 2 flip: SA-2 missile_propulsion ✓ exact ≥ today-baseline ✓ exact
- [ ] Phase 2 flip: missile_propulsion wrong count ≤ today-baseline wrong + 1
- [ ] Phase 2 flip: at most ONE non-propulsion pass regresses by exactly 1; all others ≥ today-baseline ✓ exact
- [ ] Phase 2 flip: corpus-wide ✓ exact sum ≥ today-baseline sum − 1
- [ ] Phase 2 flip: variance-mode (strict / median) decision recorded in baseline.meta.json
- [ ] Phase 2 flip: legacy-path smoke test passes (ingest one doc forcing legacy chunker; verify text_chunks rows carry non-NULL chunk_metadata for tables)
- [ ] .env + .env.example contain all 8 new variables
```

- [ ] **Step 4: Commit**

```bash
git add VERIFICATION_CHECKLIST.md
git commit -m "chore(table-norm): delete _table_pivot.py + update verification checklist

Already-deprecated _table_pivot.py removed; sole consumer was its own
test file (grep verified). VERIFICATION_CHECKLIST.md adds 10 rows
covering Step 0a baseline, Step 0b spike, Phase 1 merge gate, and
Phase 2 flip gates.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 25: Phase 1 merge

- [ ] **Step 1: Run the full test suite**

```bash
.venv/bin/pytest tests/ -v --tb=short
```

Expected: all tests pass. Address any failures before merge.

- [ ] **Step 2: Run the master kill-switch byte-equality merge gate explicitly**

```bash
.venv/bin/pytest tests/integration/test_master_kill_switch_byte_equality.py -v
```

Expected: all 3 PASS. This is the merge-blocking check.

- [ ] **Step 3: Tick the Phase 1 checklist items in VERIFICATION_CHECKLIST.md**

Mark the Phase 1 rows complete in the checklist (commit the update).

- [ ] **Step 4: Merge `feat/table-aware-chunking` to `main`**

(Subject to user PR-flow preferences — `git merge`, GitHub PR, etc.)

After merge: production behavior is byte-identical to pre-merge. Master switches default false. No user-facing changes.

### Task 26: Phase 2 flip — operational, not committed

Phase 2 is an operational step run by the user. The plan documents the procedure; the implementer doesn't auto-flip.

**Procedure (executed by user):**

1. Set master switches to true:
   ```bash
   export DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true
   export EMBEDDING_TABLE_NORMALIZATION_ENABLED=true
   # (Update .env if persistent)
   ```

2. Run `./manage.sh --blow-away` to nuke existing chunks + ArcadeDB state.

3. Re-ingest the SA-2 corpus using the same command from `baseline.meta.json`.

4. Run N=3 extraction passes per the comparison-mode recorded in baseline.

5. Compute new per-pass `{exact, wrong, null}` counts. Compare against `tests/fixtures/sa2/<docid>_extraction_counts_today.json`.

6. Verify against §15.2 flip gates:
   - missile_propulsion ✓ exact ≥ today-baseline ✓ exact
   - missile_propulsion wrong count ≤ today-baseline wrong + 1
   - At most one other pass regresses by exactly 1 ✓ exact
   - Corpus-wide ✓ exact sum ≥ today-baseline sum − 1

7. Run the legacy-path smoke test: ingest one doc forcing legacy chunker (e.g., a doc whose docling_document.json has `enrichments.version = None`); verify `text_chunks` rows for table elements have non-NULL `chunk_metadata`.

8. Run the retrieval check: `/v1/retrieval/query` for "S-75M2 max range" — top-3 contains a chunk with `chunk_kind in {table_entity_column, table_entity_section}`.

9. Tick the Phase 2 checklist items in `VERIFICATION_CHECKLIST.md`.

**If any gate fails:** flip flags back to false, re-blow-away + re-ingest, confirm rollback by re-running `test_master_kill_switch_byte_equality.py`. File a follow-up issue with the captured comparison data.

---

## End of Chunk 3

Plan covers Phase 4 (retrieval surfacing), Phase 5 (5 merge-gate tests), Phase 6 (rollout — deletion, merge, operational flip). Total: 26 tasks across 3 chunks.

---

## Summary

**Tasks:** 26 total
**Branches:** Single branch `feat/table-aware-chunking`
**Merge model:** Phase 1 (code lands, no behavior change) → Phase 2 (operational flag flip, regression-gated)
**Risk floor:** Master kill-switch byte-equality test (Task 19) blocks any Phase 1 merge that would change today's behavior on flag-off

**Critical path:**
- Step 0a (baseline capture) MUST complete before any other task
- Step 0b (spike) MUST complete before Task 17 (provenance enrichment)
- All Phase 1 tasks must pass before Task 25 (merge)
- Phase 2 flip (Task 26) is user-driven, gated on §15.2 conditions

**Rollback path:** flip master switches false → `--blow-away` + re-ingest → re-run byte-equality test

